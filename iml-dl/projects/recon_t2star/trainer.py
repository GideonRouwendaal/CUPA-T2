import os.path
import torch
from core.Trainer import Trainer
from time import time
import wandb
import logging
from torchinfo import summary
import matplotlib.pyplot as plt
import inspect
from projects.recon_t2star.utils import *


class PTrainer(Trainer):
    def __init__(self, training_params, model, hyper_setup, data, device,
                 log_wandb=True):
        super(PTrainer, self).__init__(
            training_params, model, hyper_setup, data, device, log_wandb)

        self.loss_domain = training_params.get('loss_domain', 'I')
        self.mask_image = training_params['loss']['params'].get('mask_image',
                                                                False)
        self.continuous_excl_rate = training_params.get('continuous_excl_rate',
                                                        True)
        self.add_mean_fs = training_params.get('add_mean_fs_img', False)

        if self.mask_image:
            print("[Trainer::init]:: Image masking for criterion_rec is"
                  "enabled. Using the twice eroded brainmask without CSF.")

        for s in self.train_ds:
            input_size = [
                s[0].numpy().shape, s[1].numpy().shape,
                s[1].numpy().shape, s[1].numpy().shape
            ]
            break

        if self.hyper_setup:
            print("Training in HyperNetwork Setting:")
            print("HyperNetwork:")
            print("Input size: ", self.model.input_dim)
            print(self.model.get_hyper_network())
            print("Number of model parameters: ",
                  sum(p.numel()
                      for p in self.model.get_hyper_network().parameters())
                  )
            if self.continuous_excl_rate:
                print("Only last convolutional layer and DC weights are "
                      "predicted by HyperNetwork.")
            print("###############################################")
            print("Main Network:")
            print("Input size: ", input_size)
            print("Number of main model parameters: ",
                  sum(p.numel()
                      for p in self.model.get_main_network().parameters())
                  )
        else:
            dtypes = [torch.complex64, torch.complex64, torch.complex64,
                      torch.complex64]
            print(f'Input size of summary is: {input_size}')
            summary(self.model, input_size, dtypes=dtypes)

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        """
        Trains the local client with the option to initialize the model and
        optimizer states.

        Parameters
        ----------
        model_state : dict, optional
            Weights of the global model. If provided, the local model is
            initialized with these weights.
        opt_state : dict, optional
            State of the optimizer. If provided, the local optimizer is
            initialized with this state.
        start_epoch : int, optional
            Starting epoch for training.

        Returns
        -------
        dict
            The state dictionary of the trained local model.
        """

        if model_state is not None:
            self.model.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        self.early_stop = False

        self.model.train()

        print(self.model)

        for epoch in range(self.training_params['nr_epochs']):
            print('Epoch: ', epoch)
            if start_epoch > epoch:
                continue
            if self.early_stop is True:
                logging.info(
                    "[Trainer::test]: ################ Finished  training "
                    "(early stopping) ################"
                )
                break

            self._log_model_weights(epoch)

            start_time = time()
            batch_loss = {"combined": 0}
            count_images = 0

            for data in self.train_ds:
                (img_cc_zf, kspace_zf, mask, sens_maps,
                 img_cc_fs, brain_mask, A, _, _) = process_input_data(
                    self.device, data, add_mean_fs=self.add_mean_fs
                )
                count_images += img_cc_zf.shape[0]

                # Forward Pass
                self.optimizer.zero_grad()
                prediction = forward_pass(self.model, self.hyper_setup,
                                          img_cc_zf, kspace_zf,
                                          mask, sens_maps,
                                          self.continuous_excl_rate)

                # Reconstruction Loss
                bm = (brain_mask[:, None].repeat(1, 12, 1, 1)
                      if self.mask_image else None)

                if self.loss_domain == 'I':
                    losses = list(
                        self.criterion_rec(img_cc_fs, prediction,
                                           output_components=True, mask=bm)
                    )
                elif self.loss_domain == 'k':
                    kspace_pred = A(prediction, torch.ones_like(bm),
                                    sens_maps)
                    kspace_fs = A(img_cc_fs, torch.ones_like(bm),
                                  sens_maps)
                    losses = list(
                        self.criterion_rec(kspace_fs, kspace_pred,
                                           output_components=True, mask=bm)
                    )
                else:
                    logging.info(
                        "[Trainer::train]: This loss domain is not "
                        "implemented."
                    )
                losses[0] *= self.lambda_rec

                # Physics Loss
                if self.criterion_physics is not None:
                    num_args = len(
                        inspect.signature(self.criterion_physics).parameters
                    )
                    if num_args == 2:
                        loss_physics = self.criterion_physics(
                            prediction,
                            mask=brain_mask[:, None].repeat(1, 12, 1, 1)
                        )
                    if num_args == 3:
                        loss_physics = self.criterion_physics(
                            img_cc_fs,
                            prediction,
                            mask=brain_mask[:, None].repeat(1, 12, 1, 1)
                        )
                    else:
                        raise ValueError("[Trainer::train]: Unexpected number "
                                         "of arguments for criterion_physics.")
                    losses[0] += self.lambda_physics * loss_physics
                    if self.lambda_physics > 0:
                        losses.append(loss_physics)

                # Regularisation
                if self.hyper_setup:
                    if self.criterion_reg is not None:
                        loss_reg = self.criterion_reg(self.model, self.device)
                        losses[0] += self.lambda_reg * loss_reg
                        if self.lambda_reg > 0:
                            losses.append(loss_reg)

                # Backward Pass on the combined loss
                losses[0].backward()
                self.optimizer.step()
                batch_loss = self._update_batch_loss(
                    batch_loss, losses, img_cc_zf.size(0)
                )

            self._track_epoch_loss(epoch, batch_loss, start_time, count_images)

            # Save latest model
            torch.save({'model_weights': self.model.state_dict(),
                        'optimizer_weights': self.optimizer.state_dict(),
                        'epoch': epoch}, self.client_path + '/latest_model.pt')

            # Run validation
            self.test(self.model.state_dict(), self.val_ds, 'Val',
                      self.optimizer.state_dict(), epoch)

        return self.best_weights, self.best_opt_weights

    def test(self, model_weights, test_data, task='Val', opt_weights=None,
             epoch=0):
        """
        Tests the local client.

        Parameters
        ----------
        model_weights : dict
            Weights of the global model.
        test_data : DataLoader
            Test data for evaluation.
        task : str, optional
            Task identifier (default is 'Val').
        opt_weights : dict, optional
            Optimal weights (default is None).
        epoch : int, optional
            Current epoch number (default is 0).
        """

        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval()
        metrics = {'Loss_rec': 0}
        if self.criterion_physics is not None:
            metrics['Loss_physics'] = 0
        test_total = 0

        val_image_available = False
        with torch.no_grad():
            for data in test_data:
                # Input
                (img_cc_zf, kspace_zf, mask, sens_maps,
                 img_cc_fs, brain_mask, A,
                 filename, slice_num) = process_input_data(
                    self.device, data, add_mean_fs=self.add_mean_fs
                )

                test_total += img_cc_zf.shape[0]

                # Forward Pass
                prediction = forward_pass(self.test_model,
                                          self.hyper_setup,
                                          img_cc_zf,
                                          kspace_zf, mask, sens_maps,
                                          self.continuous_excl_rate)

                prediction_track = prediction.clone().detach()
                gt_track = img_cc_fs.clone().detach()
                zf_track = img_cc_zf.clone().detach()
                mask_track = mask.clone().detach()

                # Reconstruction Loss
                mask = (brain_mask[:, None].repeat(1, 12, 1, 1)
                        if self.mask_image else None)
                if self.loss_domain == 'I':
                    loss_ = self.criterion_rec(img_cc_fs, prediction,
                                               output_components=False,
                                               mask=mask)
                elif self.loss_domain == 'k':
                    kspace_pred = A(prediction, torch.ones_like(mask),
                                    sens_maps)
                    kspace_fs = A(img_cc_fs, torch.ones_like(mask),
                                  sens_maps)
                    loss_ = self.criterion_rec(kspace_fs, kspace_pred,
                                               output_components=False,
                                               mask=mask)
                else:
                    logging.info(
                        "[Trainer::test]: This loss domain is not "
                        "implemented."
                    )
                metrics['Loss_rec'] += loss_.item() * img_cc_zf.size(0)

                if self.criterion_physics is not None:
                    num_args = len(
                        inspect.signature(self.criterion_physics).parameters
                    )
                    if num_args == 2:
                        loss_physics = self.criterion_physics(
                            prediction,
                            mask=brain_mask[:, None].repeat(1, 12, 1, 1)
                        )
                    if num_args == 3:
                        loss_physics = self.criterion_physics(
                            img_cc_fs,
                            prediction,
                            mask=brain_mask[:, None].repeat(1, 12, 1, 1)
                        )
                    else:
                        raise ValueError("[Trainer::train]: Unexpected number "
                                         "of arguments for criterion_physics.")
                    metrics['Loss_physics'] += (loss_physics.item()
                                                * img_cc_zf.size(0)
                                                )

                if task == 'Val':
                    search_string = ('SQ-struct-40_nr_09082023_1643103_4_2_'
                                     'wip_t2s_air_sg_fV4.mat_15')
                    tmp = self._find_validation_image(
                        search_string, filename, slice_num, prediction_track,
                        gt_track, zf_track, mask_track
                    )
                    if tmp[4]:
                        val_image_available = True
                        gt_ = tmp[0]
                        prediction_ = tmp[1]
                        zf_ = tmp[2]
                        mask_ = tmp[3]

            if task == 'Val':
                if not val_image_available:
                    print('[Trainer - test] ERROR: No validation image can be '
                          'tracked, since the required filename is '
                          'not in the validation set.\nUsing the last '
                          'available example instead')
                    gt_ = gt_track[0]
                    prediction_ = prediction_track[0]
                    zf_ = zf_track[0]
                    mask_ = mask_track[0]

                log_images_to_wandb(
                    prepare_for_logging(prediction_),
                    prepare_for_logging(gt_),
                    prepare_for_logging(zf_),
                    mask_example=prepare_for_logging(mask_),
                    wandb_log_name="{}/Example_".format(task),
                    count=""
                )

            for metric_key in metrics.keys():
                metric_name = task + '/' + str(metric_key)
                metric_score = metrics[metric_key] / test_total
                wandb.log({metric_name: metric_score, '_step_': epoch})
            wandb.log({
                'lr': self.optimizer.param_groups[0]['lr'], '_step_': epoch}
            )

            if task == 'Val':
                epoch_val_loss = metrics['Loss_rec'] / test_total
                if self.criterion_physics is not None:
                    epoch_val_loss += (self.lambda_physics
                                       * metrics['Loss_physics']
                                       / test_total)
                print(
                    'Epoch: {} \tValidation Loss: {:.6f} , computed for {} '
                    'samples'.format(epoch, epoch_val_loss, test_total)
                )
                if epoch_val_loss < self.min_val_loss:
                    self.min_val_loss = epoch_val_loss
                    torch.save({'model_weights': model_weights,
                                'optimizer_weights': opt_weights,
                                'epoch': epoch},
                               self.client_path + '/best_model.pt')
                    self.best_weights = model_weights
                    self.best_opt_weights = opt_weights
                self.early_stop = self.early_stopping(epoch_val_loss)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch_val_loss)

    def _log_model_weights(self, epoch):
        """Log some of the model weights to wandb every 10 epochs."""

        if epoch % 100 == 0:
            parameters_DC = []
            parameters_denoiser_weights = {}
            all_parameters = []

            # Extract weights from HyperNetwork or MainNetwork
            if self.hyper_setup:
                exclusion_rates = [0.05, 0.3, 0.5]
                for e in exclusion_rates:
                    excl_rate = torch.tensor([e], device=self.device).to(
                        torch.float32)
                    all_parameters.append(
                        self.model.predict_parameters(hyper_input=excl_rate)
                    )
            else:
                all_parameters.append(self.model.state_dict())
                exclusion_rates = ["NA"]

            # Reformat weights for plotting
            for e, parameters in zip(exclusion_rates, all_parameters):
                parameters_DC.append([detach_torch(parameters[k])
                                      for k in parameters.keys() if
                                      k.startswith("DC")])
                nr_iter = len([p for p in parameters.keys() if "denoiser" in p
                               and "ops.8.weight" in p])
                for i in range(0, nr_iter):
                    name = f'denoiser.{i}.ops.8.weight'
                    if name not in parameters.keys():
                        continue
                    if name not in parameters_denoiser_weights.keys():
                        parameters_denoiser_weights[name] = []
                    parameters_denoiser_weights[name].append(
                        detach_torch(parameters[name][0, 0:4].flatten())
                    )

            fig = plt.figure(figsize=(10, 3))
            plt.imshow(np.array(parameters_DC))
            plt.colorbar()
            plt.ylabel("Exclusion Rate")
            plt.xlabel("Iterations")
            plt.yticks(list(range(len(exclusion_rates))), exclusion_rates)
            wandb.log({"Train/Weights/Data_Consistency": fig, '_step_': epoch})

            for name in parameters_denoiser_weights.keys():
                fig = plt.figure(figsize=(15, 2))
                plt.imshow(np.array(parameters_denoiser_weights[name]))
                plt.colorbar()
                plt.ylabel("Exclusion Rate")
                plt.xlabel("Convolutional weights (partial)")
                plt.yticks(list(range(len(exclusion_rates))), exclusion_rates)
                wandb.log({"Train/Weights/"+name: fig, '_step_': epoch})

            del parameters_DC, parameters_denoiser_weights, all_parameters
            torch.cuda.empty_cache()


    def _update_batch_loss(self, batch_loss, losses, batch_size):
        """Update batch losses with current losses"""

        batch_loss["combined"] += losses[0].item() * batch_size
        for i in range(1, len(losses)):
            if f"component_{i}" not in batch_loss.keys():
                batch_loss[f"component_{i}"] = 0
            batch_loss[f"component_{i}"] += losses[i].item() * batch_size

        return batch_loss

    def _track_epoch_loss(self, epoch, batch_loss, start_time, count_images):
        """Track and log epoch loss."""

        loss_components = batch_loss.keys()
        epoch_loss = {}
        for component in loss_components:
            epoch_loss[component] = (batch_loss[component] / count_images
                                     if count_images > 0
                                     else batch_loss[component])

        end_time = time()
        loss_msg = (
            'Epoch: {} \tTraining Loss: {:.6f} , computed in {} '
            'seconds for {} samples').format(
            epoch, epoch_loss["combined"], end_time - start_time,
            count_images
        )
        print(loss_msg)

        for component in loss_components:
            name = f'Train/Loss_{component.replace("component_", "Comp")}'
            wandb.log({name: epoch_loss[component], '_step_': epoch})


    def _find_validation_image(self, search_string, filename, slice_num,
                               prediction_track, gt_track, zf_track,
                               mask_track):
        """Search for a specific validation image and retrieve the data."""

        # Search for a specific validation image
        search = [os.path.basename(f) + '_' + str(s.numpy()) for f, s in
                  zip(filename, slice_num)]

        # Check if the search string is in the list
        try:
            ind = search.index(search_string)
            val_image_available = True
        except ValueError:
            ind = -1
            val_image_available = False

        # Retrieve data for the found image or provide default values
        gt_ = gt_track[ind] if val_image_available else torch.tensor([])
        prediction_ = prediction_track[
            ind] if val_image_available else torch.tensor([])
        zf_ = zf_track[ind] if val_image_available else torch.tensor([])
        mask_ = mask_track[ind] if val_image_available else torch.tensor([])

        return gt_, prediction_, zf_, mask_, val_image_available