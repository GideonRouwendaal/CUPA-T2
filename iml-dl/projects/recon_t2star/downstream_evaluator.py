import logging
import os.path
import numpy as np
import torch.nn
import wandb
import merlinth
from time import time
from dl_utils import *
from core.DownstreamEvaluator import DownstreamEvaluator
from data.t2star_loader import RawMotionBootstrapSamples
from projects.recon_t2star.utils import *
from mr_utils.parameter_fitting import T2StarFit


class PDownstreamEvaluator(DownstreamEvaluator):
    """Downstream Tasks"""

    def __init__(self,
                 name,
                 model,
                 hyper_setup,
                 device,
                 test_data_dict,
                 checkpoint_path,
                 task="recon",
                 include_brainmask=False,
                 save_predictions=False,
                 continuous_excl_rate=True,
                 add_mean_fs_img=False):
        super(PDownstreamEvaluator, self).__init__(
            name, model, hyper_setup, device, test_data_dict, checkpoint_path
        )

        self.name = name
        self.task = task
        self.include_brainmask = include_brainmask
        self.save_predictions = save_predictions
        self.continuous_excl_rate = continuous_excl_rate
        self.add_mean_fs = add_mean_fs_img
        self.uncertainty_estimation = True if "unc" in task else False
            

    def start_task(self, global_model):
        """Function to perform analysis after training is finished."""

        if "recon" in self.task:
            self.test_reconstruction(global_model)
        else:
            logging.info("[DownstreamEvaluator::ERROR]: This task is not "
                         "implemented.")

    def test_reconstruction(self, global_model):
        """Validation of reconstruction downstream task."""

        logging.info("################ Reconstruction test #################")
        self.model.load_state_dict(global_model)
        self.model.eval()
        task_name = ""
        for tag in ["LowExclRate", "MediumExclRate", "HighExclRate"]:
            if tag in self.name:
                task_name = tag

        keys = ["SSIM_magn_pred", "SSIM_magn_zf", "PSNR_magn_pred",
                "PSNR_magn_zf", "MSE_magn_pred", "MSE_magn_zf",
                "T2s_MAE_CSF_pred", "T2s_MAE_CSF_zf", "SSIM_phase_pred",
                "SSIM_phase_zf", "PSNR_phase_pred", "PSNR_phase_zf",
                "MSE_phase_pred", "MSE_phase_zf"]
        metrics = {k: [] for k in keys}

        for dataset_key in self.test_data_dict.keys():
            logging.info('DATASET: {}'.format(dataset_key))
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {k: [] for k in metrics.keys()}

            for idx, data in enumerate(dataset):
                with torch.no_grad():
                    # Input
                    (img_cc_zf, kspace_zf, mask, sens_maps,
                     img_cc_fs, brain_mask, A, _, _) = process_input_data(
                        self.device, data, add_mean_fs=self.add_mean_fs
                    )

                    # Forward Pass
                    prediction = forward_pass(self.model, self.hyper_setup,
                                              img_cc_zf, kspace_zf, mask,
                                              sens_maps,
                                              self.continuous_excl_rate, uncertainty=self.uncertainty_estimation)

                    uncertainty_map = None
                    if self.uncertainty_estimation:
                        prediction, uncertainty_map = prediction
                        uncertainty_map = detach_torch(uncertainty_map)

                    calc_t2star = T2StarFit(dim=4)
                    t2star_zf = detach_torch(calc_t2star(img_cc_zf,
                                                         mask=brain_mask))
                    t2star_pred = detach_torch(calc_t2star(prediction,
                                                           mask=brain_mask))
                    t2star_fs = detach_torch(calc_t2star(img_cc_fs,
                                                         mask=brain_mask))

                    for i in range(len(prediction)):
                        count = str(idx * len(prediction) + i)

                        img_metrics = [k for k in test_metrics.keys()
                                       if "T2s" not in k]
                        metrics_pred = calculate_img_metrics(
                            target=img_cc_fs[i],
                            data=prediction[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in img_metrics
                                                if "pred" in k],
                            include_brainmask=self.include_brainmask)

                        metrics_zf = calculate_img_metrics(
                            target=img_cc_fs[i],
                            data=img_cc_zf[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in img_metrics
                                                if "zf" in k],
                            include_brainmask=self.include_brainmask)

                        test_metrics = update_metrics_dict(metrics_pred,
                                                           test_metrics)
                        test_metrics = update_metrics_dict(metrics_zf,
                                                           test_metrics)

                        t2s_metrics = [k for k in test_metrics.keys()
                                       if "T2s" in k]
                        metrics_zf = calculate_t2star_metrics(
                            target=t2star_fs[i],
                            data=t2star_zf[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in t2s_metrics
                                                if "zf" in k],
                            include_brainmask=self.include_brainmask)
                        metrics_pred = calculate_t2star_metrics(
                            target=t2star_fs[i],
                            data=t2star_pred[i],
                            bm=brain_mask[i][None].repeat(12, 1, 1),
                            metrics_to_be_calc=[k for k in t2s_metrics
                                                if "pred" in k],
                            include_brainmask=self.include_brainmask)

                        test_metrics = update_metrics_dict(metrics_zf,
                                                           test_metrics)
                        test_metrics = update_metrics_dict(metrics_pred,
                                                           test_metrics)


                        if idx % 2 == 0 and i % 10 == 0:
                            log_images_to_wandb(
                                prepare_for_logging(prediction[i]),
                                prepare_for_logging(img_cc_fs[i]),
                                prepare_for_logging(img_cc_zf[i]),
                                uncertainty_map_example=prepare_for_logging(uncertainty_map[i]) if uncertainty_map is not None else None,
                                wandb_log_name="Recon_Examples_{}/{}"
                                               "/".format(task_name,
                                                          dataset_key),
                                count=count,
                                data_types=["magn", "phase"]
                            )

                            plot2wandb(
                                maps=[t2star_zf[i],
                                t2star_pred[i]],
                                ref_map=t2star_fs[i],
                                titles=["Zero-filled",
                                        "Predicted",
                                        "Fully-sampled"],
                                wandb_log_name=
                                "Recon_Quantitative_Examples_{}/{}/{}".format(
                                    task_name, dataset_key, str(count)
                                )
                            )

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(
                    metric,
                    np.nanmean(test_metrics[metric]),
                    np.nanstd(test_metrics[metric]))
                )
                metrics[metric].append(test_metrics[metric])

    @staticmethod
    def _calculate_t2star_map(img, bm):
        """Calculate T2* maps from complex-valued T2*-weighted images."""

        FitError = T2starFit(detach_torch(img[:, None]),
                             detach_torch(bm))
        t2star, _ = FitError.t2star_linregr()

        return t2star
