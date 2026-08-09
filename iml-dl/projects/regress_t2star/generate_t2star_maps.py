"""Stage 1 of the CUPA pipeline: export everything the regression stage consumes.

Run through ``core/Main.py`` with a config from ``configs/generate_t2_star/``.
For every slice this evaluator runs the trained reconstruction network twice --
once with the uncertainty head enabled (MC dropout) and once without -- fits T2*
and S0 to both, and writes the results as one ``.npy`` per slice.

Layout produced under ``<CUPA_ROOT>/data/T2_param_data``::

    <unc>/<experiment>/acc_rate_<R>/<split>/
        gt/  gt_s0/  gt_magnitude/          fully-sampled reference
        zf/  zf_t2star/  zf_s0/             zero-filled input
        unc/  no_unc/                       reconstructions (unc/ has the
                                            uncertainty map concatenated on)
        recon_{t2star,s0}_{unc,no_unc}_signal/
        raw_mc_predictions/                 (S,E,H,W) MC samples -> covariance
        brain_masks/  gray_matter_masks/  white_matter_masks/  csf_masks/
        kspace/  kspace_mask/  sens_maps/

    <subject>/slice_<n>.npy in each, except sens_maps/ (see _save_slice).

`generate_sample_matrices.py` reads gt/, gt_s0/, raw_mc_predictions/ and
brain_masks/ from here; `data/t2star_regress_loader.py` reads the rest.
"""

import logging
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from core.DownstreamEvaluator import DownstreamEvaluator
from dl_utils.config_utils import set_seed
from mr_utils.parameter_fitting import T2StarFit
from projects.recon_t2star.utils import (detach_torch, forward_pass,
                                         process_input_data)

# ── Path anchors ──────────────────────────────────────────────────────────────
# Derived from this file's location so the project can be moved or checked out
# anywhere. PROJECT_DIR/../.. is the iml-dl repo root; its parent holds ./data.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
T2_PARAM_DATA = CUPA_ROOT / "data" / "T2_param_data"
MR_DATA_FOLDER = CUPA_ROOT / "data" / "mr_data" / "data_folder"
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42

# Downstream-task names this evaluator knows how to route, e.g.
# "T2StarReconstruction-With_Unc_Acc_2".
_ACC_RATE_RE = re.compile(r"(?:^|[_\-])acc[_\-]?(\d+)(?:[_\-]|$)", re.IGNORECASE)


def parse_task_name(name):
    """Split a downstream-task name into (uncertainty flavour, experiment, acc rate).

    ``"T2StarReconstruction-With_Unc_Acc_2"`` -> ``("With_Unc", "2")``.
    These three fields become the output directory, so an unrecognised name is a
    hard error rather than a silent fallback: writing a whole dataset into the
    wrong folder is much more expensive to notice than a crash here.
    """
    match = _ACC_RATE_RE.search(name)
    if match is None:
        raise ValueError(
            f"Cannot read an acceleration rate from downstream-task name {name!r}. "
            "Expected something like 'T2StarReconstruction-With_Unc_Acc_2'."
        )
    acc_rate = match.group(1)

    if "With_Unc" in name:
        unc = "With_Unc"
    elif "Without_Unc" in name:
        unc = "Without_Unc"
    else:
        raise ValueError(
            f"Downstream-task name {name!r} contains neither 'With_Unc' nor "
            "'Without_Unc'; cannot decide where to write."
        )

    experiment = name.split("_")[-1] if "exp" in name.lower() else ""
    return unc, experiment, acc_rate


def load_mask_from_nii(filename, binary=True):
    """Load a segmentation mask from NIfTI, cropped and flipped to match the T2* grid."""
    mask = nib.load(filename).get_fdata()[10:-10][::-1, ::-1, :]
    if binary:
        mask = np.where(mask < 0.5, 0, 1)
    return mask


def _fit_t2star(image, brain_mask):
    """Regression-mode T2*/S0 fit. Returns (t2star, s0), both still on device."""
    fit = T2StarFit(dim=4)
    fit.mode = "regression"
    return fit(image, mask=brain_mask)


class PDownstreamEvaluator(DownstreamEvaluator):
    """Writes per-slice reconstructions, parameter fits and masks to disk."""

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
        self.uncertainty_estimation = "unc" in task

    def start_task(self, global_model):
        """Entry point called by the framework once training has finished."""
        if "recon" in self.task:
            self.test_reconstruction(global_model)
        else:
            logging.info("[DownstreamEvaluator::ERROR]: This task is not "
                         "implemented.")

    def test_reconstruction(self, global_model):
        """Reconstruct every slice of every dataset and export the tensors."""
        set_seed(SEED)

        logging.info("################ Generating T2* maps #################")
        self.model.load_state_dict(global_model)
        self.model.eval()

        unc, experiment, acc_rate = parse_task_name(self.name)

        for dataset_key, dataset in self.test_data_dict.items():
            output_dir = Path(
                f"{T2_PARAM_DATA}/{unc}/{experiment}/acc_rate_{acc_rate}/{dataset_key}"
            )
            logging.info("DATASET: %s -> %s", dataset_key, output_dir)

            segmentation_cache = {}
            for data in dataset:
                with torch.no_grad():
                    self._process_batch(data, output_dir, dataset_key,
                                        segmentation_cache)

            logging.info("Finished generating T2* maps for %s.", dataset_key)

    def _process_batch(self, data, output_dir, dataset_key, segmentation_cache):
        """Reconstruct one batch and write every slice in it."""
        (img_cc_zf, kspace_zf, mask, sens_maps,
         img_cc_fs, brain_mask, _A, _, _) = process_input_data(
            self.device, data, add_mean_fs=self.add_mean_fs
        )

        # Two forward passes through the same weights: the uncertainty head is
        # what makes the second one stochastic (MC dropout), so both are needed.
        no_unc_prediction = forward_pass(
            self.model, self.hyper_setup, img_cc_zf, kspace_zf, mask, sens_maps,
            self.continuous_excl_rate, uncertainty=False
        )
        unc_prediction, uncertainty_map, raw_mc_preds = forward_pass(
            self.model, self.hyper_setup, img_cc_zf, kspace_zf, mask, sens_maps,
            self.continuous_excl_rate, uncertainty=True
        )

        t2star_fs, s0_fs = _fit_t2star(img_cc_fs, brain_mask)
        t2star_zf, s0_zf = _fit_t2star(img_cc_zf, brain_mask)
        t2star_unc, s0_unc = _fit_t2star(unc_prediction, brain_mask)
        t2star_no_unc, s0_no_unc = _fit_t2star(no_unc_prediction, brain_mask)

        # The uncertainty map rides along with its reconstruction as extra
        # channels, which is the layout t2star_regress_loader expects in unc/.
        unc_prediction = np.concatenate(
            (detach_torch(unc_prediction), detach_torch(uncertainty_map)), axis=1
        )

        # One entry per output subdirectory; each value is indexed by slice below.
        arrays = {
            "gt": detach_torch(t2star_fs),
            "gt_s0": detach_torch(s0_fs),
            "gt_magnitude": detach_torch(img_cc_fs),
            "zf": detach_torch(img_cc_zf),
            "zf_t2star": detach_torch(t2star_zf),
            "zf_s0": detach_torch(s0_zf),
            "unc": unc_prediction,
            "no_unc": detach_torch(no_unc_prediction),
            "recon_t2star_unc_signal": detach_torch(t2star_unc),
            "recon_t2star_no_unc_signal": detach_torch(t2star_no_unc),
            "recon_s0_unc_signal": detach_torch(s0_unc),
            "recon_s0_no_unc_signal": detach_torch(s0_no_unc),
            "raw_mc_predictions": detach_torch(raw_mc_preds),
            "brain_masks": detach_torch(brain_mask),
            "kspace": detach_torch(kspace_zf),
            "kspace_mask": detach_torch(mask),
            "sens_maps": detach_torch(sens_maps),
        }

        for i in range(len(t2star_fs)):
            data_location = data[3][i]
            subject_id = data_location.split("/")[-1].split("_")[0]
            slice_idx = data[4][i].item()

            for name, array in arrays.items():
                # raw_mc_predictions is (samples, batch, ...); the rest are batch-first.
                sample = array[:, i, ...] if name == "raw_mc_predictions" else array[i]
                self._save_slice(output_dir, name, subject_id, slice_idx, sample)

            gray_matter, white_matter = self._get_segmentation_slices(
                subject_id, slice_idx, segmentation_cache, dataset_key
            )
            bm = arrays["brain_masks"][i]
            csf = bm * (bm - (gray_matter + white_matter))
            for name, array in (("gray_matter_masks", gray_matter),
                                ("white_matter_masks", white_matter),
                                ("csf_masks", csf)):
                self._save_slice(output_dir, name, subject_id, slice_idx, array)

            logging.debug("Saved subject %s, slice %s", subject_id, slice_idx)

    @staticmethod
    def _save_slice(output_dir, name, subject_id, slice_idx, array):
        """Write one slice to ``<output_dir>/<name>/<subject>/slice_<n>.npy``.

        sens_maps is the exception: it is written flat, without the subject
        level. Nothing downstream reads it, so the original layout is kept.
        """
        target = output_dir / name
        if name != "sens_maps":
            target = target / subject_id
        os.makedirs(target, exist_ok=True)
        np.save(target / f"slice_{slice_idx}.npy", array)

    @staticmethod
    def _get_segmentation_slices(subject_id, slice_idx, cache, dataset_key):
        """Gray/white matter masks for one slice, caching the volume per subject."""
        if subject_id not in cache:
            filepath = f"{MR_DATA_FOLDER}/{dataset_key}/{subject_id}/"
            cache[subject_id] = (
                load_mask_from_nii(filepath + "seg_gm_reg-to-t2s.nii"),
                load_mask_from_nii(filepath + "seg_wm_reg-to-t2s.nii"),
            )

        gray_matter, white_matter = cache[subject_id]
        slice_idx = int(slice_idx)
        return gray_matter[:, :, slice_idx], white_matter[:, :, slice_idx]
