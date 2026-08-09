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
        """Validation of reconstruction downstream task model and gt."""

        logging.info("################ Generating T2* maps #################")
        self.model.load_state_dict(global_model)
        self.model.eval()

        task_name = ""
        for tag in ["LowExclRate", "MediumExclRate", "HighExclRate"]:
            if tag in self.name:
                task_name = tag
        for dataset_key in self.test_data_dict.keys():
            logging.info('DATASET: {}'.format(dataset_key))
            dataset = self.test_data_dict[dataset_key]
            output_dir = f"/home/iml/gideon.rouwendaal/PHIMO2/data/T2_param_data/{dataset_key}/"
            os.makedirs(output_dir, exist_ok=True)
            output_dir = os.path.join(output_dir, tag)
            os.makedirs(output_dir, exist_ok=True)
            output_dir_gt = f"{output_dir}/gt/"
            os.makedirs(output_dir_gt, exist_ok=True)
            output_dir_wo_unc = f"{output_dir}/wo_unc/"
            os.makedirs(output_dir_wo_unc, exist_ok=True)
            output_dir_zf = f"{output_dir}/zf/"
            os.makedirs(output_dir_zf, exist_ok=True)

            for _, data in enumerate(dataset):
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

                    t2star_fs = detach_torch(calc_t2star(img_cc_fs,
                                                         mask=brain_mask))


                    zf, pred = detach_torch(img_cc_zf), detach_torch(prediction)

                    # Save t2star_fs
                    for i in range(len(t2star_fs)):
                        data_location = data[3][i]
                        subject_id = data_location.split("/")[-1].split("_")[0]
                        slice = data[4][i].item()
                        final_output_dir_gt = os.path.join(output_dir_gt, subject_id)
                        final_output_dir_wo_unc = os.path.join(output_dir_wo_unc, subject_id)
                        final_output_dir_zf = os.path.join(output_dir_zf, subject_id)
                        os.makedirs(final_output_dir_gt, exist_ok=True)
                        os.makedirs(final_output_dir_wo_unc, exist_ok=True)
                        os.makedirs(final_output_dir_zf, exist_ok=True)

                        np.save(os.path.join(final_output_dir_gt, f"t2star_slice_{slice}.npy"), t2star_fs[i])
                        np.save(os.path.join(final_output_dir_wo_unc, f"t2star_slice_{slice}.npy"), zf[i])
                        np.save(os.path.join(final_output_dir_zf, f"t2star_slice_{slice}.npy"), pred[i])

    @staticmethod
    def _calculate_t2star_map(img, bm):
        """Calculate T2* maps from complex-valued T2*-weighted images."""

        FitError = T2starFit(detach_torch(img[:, None]),
                             detach_torch(bm))
        t2star, _ = FitError.t2star_linregr()

        return t2star
