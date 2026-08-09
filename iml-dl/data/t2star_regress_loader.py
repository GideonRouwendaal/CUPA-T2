"""
T2* Regression Data Loader with GPU Batch Sampling.

For MC training modes, returns raw ingredients (mean_ri, L_or_std) for GPU
sampling in the trainer. CV computation is integrated into _load_data() for
faster initialization.

Normalization
-------------
Inputs are normalised by the first echo: every per-echo vector is divided by its
S₁ entry, so the network sees signal decay relative to the first echo rather than
absolute magnitude. Targets are scaled by t2_max. This is the only normalization
this project uses.
"""

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Guard against division by zero in magnitude/ratio computations.
NORM_EPS = 1e-9


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class UncMode:
    """Uncertainty mode string constants."""
    NONE             = "none"
    UNCORRELATED     = "uncorrelated"
    UNCORRELATED_CAT = "uncorrelated_concat"
    CHOLESKY         = "cholesky"
    CHOLESKY_CAT     = "cholesky_concat"
    LOW_RANK         = "low_rank"
    LOW_RANK_CAT     = "low_rank_concat"

    MC_MODES = {
        UNCORRELATED, UNCORRELATED_CAT,
        CHOLESKY, CHOLESKY_CAT,
        LOW_RANK, LOW_RANK_CAT,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StaticPixelDataset(Dataset):
    """Training / validation dataset with GPU-based batch sampling for MC modes."""

    def __init__(
        self,
        data_path_input: str,
        te_values: list = None,
        min_t2: float = 0.0,
        max_t2: float = 100.0,
        mode: str = "train",
        uncertainty_mode: str = UncMode.NONE,
        lowrank_rank: int = None,
        return_slices: bool = False,
        **kwargs,
    ):
        if te_values is None:
            te_values = list(range(5, 65, 5))

        self.te_values  = torch.tensor(te_values, dtype=torch.float32)
        self.num_echoes = len(te_values)
        self.t2_min     = min_t2
        self.t2_max     = max_t2
        self.mode       = mode

        self.input_dir  = data_path_input
        self.gt_dir     = os.path.join(data_path_input, "gt") + "/"

        self.uncertainty_mode = uncertainty_mode
        self.lowrank_rank     = lowrank_rank
        self.return_slices    = return_slices

        self.use_mc = uncertainty_mode in UncMode.MC_MODES

        # Optional features from kwargs
        self.mc_dir        = kwargs.pop("mc_dir", None)
        self.mixup_beta    = kwargs.pop("mixup_beta", None)
        self.mc_dtype      = kwargs.pop("mc_dtype", "float16")
        self.acc_factor    = kwargs.pop("acc_factor", None)
        self.include_csf   = kwargs.pop("include_csf", False)
        self.use_log_input = kwargs.pop("use_log_input", False)

        self.use_uncertainty_encoder  = kwargs.pop("use_uncertainty_encoder", False)
        self.heteroscedastic          = kwargs.pop("heteroscedastic", False)
        self.use_adaptive_loss        = kwargs.pop("use_adaptive_loss", False)
        self.use_adaptive_noise       = kwargs.pop("use_adaptive_noise", False)
        self.use_consistency_loss     = kwargs.pop("use_consistency_loss", False)
        self.use_cv_correlation_loss  = kwargs.pop("use_cv_correlation_loss", False)
        self.use_rms_correlation_loss = kwargs.pop("use_rms_correlation_loss", False)
        self.use_phys_correlation_loss = kwargs.pop("use_physics_correlation_loss", False)

        self.use_log_transform = False
        self.log_epsilon       = 1e-6
        self.log_min           = np.log(10.0)
        self.log_max           = np.log(200.0)

        self.add_positional_encoding = False

        self.noise_schedule       = kwargs.pop("noise_schedule", "constant")
        self.noise_scale          = kwargs.pop("noise_scale", 1.0)
        self.current_noise_scale  = 0.0

        self.adapt_q_lo                  = kwargs.pop("adapt_q_lo", 0.50)
        self.adapt_q_hi                  = kwargs.pop("adapt_q_hi", 0.75)
        self.adapt_target_lo             = kwargs.pop("adapt_target_lo", 0.10)
        self.adapt_target_hi             = kwargs.pop("adapt_target_hi", 0.15)
        self.adapt_scale_clip            = kwargs.pop("adapt_scale_clip", (0.0, 2.0))
        self.adapt_eff_cap               = kwargs.pop("adapt_eff_cap", None)
        self.save_noise_calibration_path = kwargs.pop("save_noise_calibration_path", None)

        self.voxel_noise_scales = None
        self.voxel_cvs          = None

        self._init_storage()
        self._load_data()
        self._post_load_setup()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_storage(self):
        """Allocate list buffers that _load_data will populate."""
        if self.return_slices:
            self.slice_list = []
            return

        self.all_gt_t2_pixels    = []
        self.all_input_pixels_t2 = []   # raw mean_mag (N, E) — always stored
        self.all_tissue_labels   = []
        self.all_raw_mc_samples  = []
        self.all_t2_std_mc       = []

        needs_cv = (
            (self.use_adaptive_loss or self.use_adaptive_noise or self.use_cv_correlation_loss)
            and not self.return_slices
        )
        if needs_cv:
            self.all_cvs = []

        if self.use_rms_correlation_loss and not self.return_slices:
            self.all_rms = []
        if self.use_phys_correlation_loss and not self.return_slices:
            self.all_phys = []

        if self.use_mc:
            self.all_means         = []
            self.all_corr_matrices = []
            needs_std_buffer = (
                "concat" in self.uncertainty_mode.lower() or self.mode != "train"
            )
            if needs_std_buffer:
                self.all_stds = []

        if self.use_phys_correlation_loss:
            te_np = self.te_values.numpy().astype(np.float64)
            _A = np.stack([te_np, np.ones_like(te_np)], axis=1)
            self._feat_J_slope = np.linalg.lstsq(_A, np.eye(len(te_np)), rcond=None)[0][0]
            self._feat_A_ols   = _A

    def _post_load_setup(self):
        """Post-loading: adaptive noise calibration (slice mode needs nothing)."""
        if self.return_slices:
            return
        if self.mode != "train":
            return
        if self.use_adaptive_noise:
            print("Computing adaptive noise scales from loaded CVs...")
            self.compute_adaptive_noise_scales()
        elif self.use_adaptive_loss and not hasattr(self, "voxel_cvs"):
            if hasattr(self, "all_cvs") and len(self.all_cvs) > 0:
                self.voxel_cvs = torch.from_numpy(
                    np.concatenate(self.all_cvs)
                ).float()

    # ------------------------------------------------------------------
    # Epoch / noise schedule
    # ------------------------------------------------------------------

    def set_epoch(self, epoch: int, max_epochs: int):
        if self.noise_schedule == "curriculum":
            warmup_epochs = int(0.2 * max_epochs)
            self.current_noise_scale = min(1.0, epoch / warmup_epochs) * self.noise_scale
        elif self.noise_schedule == "reduced":
            self.current_noise_scale = self.noise_scale
        else:
            self.current_noise_scale = 1.0

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch}: noise scale = {self.current_noise_scale:.2f}")

    # ------------------------------------------------------------------
    # CV / RMS helpers
    # ------------------------------------------------------------------

    def _compute_cv_from_data(self, mean_mag, std_mag):
        return (std_mag / (mean_mag + 1e-9)).mean(axis=-1)

    def _compute_cv_from_covariance(self, mean_complex, L_matrices, is_lowrank=False):
        cov_diag      = np.sum(L_matrices ** 2, axis=-1)
        std_real      = np.sqrt(cov_diag[..., : self.num_echoes])
        std_imag      = np.sqrt(cov_diag[..., self.num_echoes :])
        std_magnitude = np.sqrt(std_real ** 2 + std_imag ** 2) / np.sqrt(2)
        mean_real     = mean_complex[..., : self.num_echoes]
        mean_imag     = mean_complex[..., self.num_echoes :]
        mean_magnitude = np.sqrt(mean_real ** 2 + mean_imag ** 2)
        return (std_magnitude / (mean_magnitude + 1e-9)).mean(axis=-1)

    def _compute_rms_from_data(self, raw_mc_samples):
        std_real = np.std(raw_mc_samples.real, axis=0)
        std_imag = np.std(raw_mc_samples.imag, axis=0)
        return np.sqrt(np.sum(std_real ** 2 + std_imag ** 2, axis=0) / (2.0 * self.num_echoes))

    def _compute_rms_from_covariance(self, L_matrices):
        D     = L_matrices.shape[-2]
        trace = np.sum(L_matrices ** 2, axis=(-2, -1))
        return np.sqrt(trace / float(D))

    @staticmethod
    def _compute_mc_t2_std(raw_mc, valid_mask, te_values):
        S, E, H, W = raw_mc.shape
        A      = np.stack([te_values, np.ones_like(te_values)], axis=1)
        J      = np.linalg.lstsq(A, np.eye(E), rcond=None)[0]
        J_slope = J[0]
        mc_valid = raw_mc[:, :, valid_mask]
        log_mag  = np.log(np.abs(mc_valid).clip(1e-9))
        slopes   = np.einsum('e,sen->sn', J_slope, log_mag)
        t2_samp  = np.clip(-1.0 / (slopes + 1e-9), 0.0, 400.0)
        return t2_samp.std(axis=0).astype(np.float32)

    @staticmethod
    # ------------------------------------------------------------------
    # Adaptive noise
    # ------------------------------------------------------------------

    def _compute_voxel_cvs(self):
        if hasattr(self, "all_cvs") and len(self.all_cvs) > 0:
            self.voxel_cvs = torch.from_numpy(np.concatenate(self.all_cvs)).float()
            return
        if "concat" in self.uncertainty_mode.lower() and hasattr(self, "all_stds"):
            mean_mag = self.all_input_pixels_t2[:, : self.num_echoes]
            std_mag  = self.all_input_pixels_t2[:, self.num_echoes :]
            self.voxel_cvs = (std_mag / (mean_mag + 1e-9)).mean(dim=1)
            return
        if self.uncertainty_mode == UncMode.NONE:
            print("Warning: cannot compute CV for 'none' mode without stored stds.")
            return
        cvs = []
        for idx in tqdm(range(len(self.all_corr_matrices)), desc="Computing CVs"):
            L    = self.all_corr_matrices[idx]
            mean = self.all_means[idx]
            cov_diag      = torch.sum(L ** 2, dim=-1)
            std_real      = torch.sqrt(cov_diag[: self.num_echoes])
            std_imag      = torch.sqrt(cov_diag[self.num_echoes :])
            std_magnitude = torch.sqrt(std_real ** 2 + std_imag ** 2) / np.sqrt(2)
            mean_real     = mean[: self.num_echoes]
            mean_imag     = mean[self.num_echoes :]
            mean_magnitude = torch.sqrt(mean_real ** 2 + mean_imag ** 2)
            cvs.append((std_magnitude / (mean_magnitude + 1e-9)).mean().item())
        self.voxel_cvs = torch.tensor(cvs, dtype=torch.float32)
        self._log_stats("CV", self.voxel_cvs)

    def compute_adaptive_noise_scales(self):
        if not hasattr(self, "voxel_cvs") or self.voxel_cvs is None:
            self._compute_voxel_cvs()
        if self.voxel_cvs is None:
            print("Warning: cannot compute adaptive noise scales without CVs.")
            return

        cvs  = self.voxel_cvs.float()
        eps  = 1e-9
        q_lo, q_hi = float(self.adapt_q_lo), float(self.adapt_q_hi)
        t_lo, t_hi = float(self.adapt_target_lo), float(self.adapt_target_hi)

        cv_lo = torch.quantile(cvs, q_lo)
        cv_hi = torch.quantile(cvs, q_hi)
        s_min = t_lo / (cv_lo.item() + eps)
        s_max = t_hi / (cv_hi.item() + eps)

        t      = torch.clamp((cvs - cv_lo) / (cv_hi - cv_lo + eps), 0.0, 1.0)
        scales = torch.clamp(
            s_min + t * (s_max - s_min),
            float(self.adapt_scale_clip[0]), float(self.adapt_scale_clip[1]),
        )
        if self.adapt_eff_cap is not None:
            eff    = torch.clamp(cvs * scales, 0.0, float(self.adapt_eff_cap))
            scales = eff / (cvs + eps)

        self.voxel_noise_scales = scales
        eff = cvs * scales
        print("Adaptive noise calibration:")
        print(f"  q_lo={q_lo:.2f} cv_lo={cv_lo:.4f} -> s_min={s_min:.4f} (target~{t_lo})")
        print(f"  q_hi={q_hi:.2f} cv_hi={cv_hi:.4f} -> s_max={s_max:.4f} (target~{t_hi})")
        print(f"  scale  mean={scales.mean():.4f} median={scales.median():.4f} "
              f"min={scales.min():.4f} max={scales.max():.4f}")
        print(f"  eff    mean={eff.mean():.4f} median={eff.median():.4f} "
              f"p95={torch.quantile(eff,0.95):.4f} p99={torch.quantile(eff,0.99):.4f}")

        if self.save_noise_calibration_path is not None:
            calib = {
                "q_lo": q_lo, "q_hi": q_hi,
                "cv_lo": cv_lo.item(), "cv_hi": cv_hi.item(),
                "target_lo": t_lo, "target_hi": t_hi, "s_min": s_min, "s_max": s_max,
                "scale_clip": list(self.adapt_scale_clip), "eff_cap": self.adapt_eff_cap,
                "scale_stats": {"mean": scales.mean().item(), "median": scales.median().item(),
                                "min": scales.min().item(), "max": scales.max().item()},
                "effective_stats": {"mean": eff.mean().item(), "median": eff.median().item(),
                                    "p95": torch.quantile(eff, 0.95).item(),
                                    "p99": torch.quantile(eff, 0.99).item()},
            }
            with open(self.save_noise_calibration_path, "w") as f:
                json.dump(calib, f, indent=2)
            print(f"Saved calibration to: {self.save_noise_calibration_path}")

    def _get_effective_noise_scale(self, voxel_idx: int):
        if self.use_adaptive_noise and self.voxel_noise_scales is not None:
            return self.current_noise_scale * self.voxel_noise_scales[voxel_idx].item()
        return self.current_noise_scale

    # ------------------------------------------------------------------
    # Target normalisation
    # ------------------------------------------------------------------

    def _normalize_parameter_maps(self, gt_map: np.ndarray) -> np.ndarray:
        """Normalise the GT T2* map by t2_max (or a log-scale transform)."""
        if self.mode == "test":
            return gt_map
        if self.use_log_transform:
            gt_log = np.log(np.clip(gt_map, 10.0, 200.0))
            return (gt_log - self.log_min) / (self.log_max - self.log_min)
        return gt_map / self.t2_max

    def inverse_transform_t2(self, normalized_pred: np.ndarray) -> np.ndarray:
        """Invert target normalisation to recover T2* in ms."""
        if self.use_log_transform:
            return np.exp(normalized_pred * (self.log_max - self.log_min) + self.log_min)
        return normalized_pred * self.t2_max

    # ------------------------------------------------------------------
    # MC helpers
    # ------------------------------------------------------------------

    def _convert_mc_to_magnitude_mean_std(self, mc_complex):
        mean_complex   = np.mean(mc_complex, axis=0)
        std_real       = np.std(mc_complex.real, axis=0)
        std_imag       = np.std(mc_complex.imag, axis=0)
        mean_magnitude = np.abs(mean_complex)
        std_magnitude  = np.sqrt(std_real ** 2 + std_imag ** 2) / np.sqrt(2)
        return mean_magnitude, std_magnitude

    def _flat_valid(self, arr, valid_mask, n_cols):
        flat = arr.transpose(1, 2, 0).reshape(-1, n_cols)
        return flat[valid_mask] if self.mode != "test" else flat

    # ------------------------------------------------------------------
    # Data loading: per-mode handlers
    # ------------------------------------------------------------------

    def _load_slice_data(self, subject: str, slice_file: str) -> dict:
        def mask_dir(name):
            return self.gt_dir.replace("gt", name)

        brain_mask = np.load(os.path.join(mask_dir("brain_masks"),        subject, slice_file))
        csf_mask   = np.load(os.path.join(mask_dir("csf_masks"),          subject, slice_file))
        gm_mask    = np.load(os.path.join(mask_dir("gray_matter_masks"),  subject, slice_file))
        wm_mask    = np.load(os.path.join(mask_dir("white_matter_masks"), subject, slice_file))
        gt_t2_map  = np.load(os.path.join(self.gt_dir, subject, slice_file)).astype(np.float64)

        valid_mask = brain_mask & (gt_t2_map != 0)
        if not self.include_csf:
            valid_mask = valid_mask & (csf_mask != 1)

        data = {
            "brain_mask": brain_mask, "csf_mask": csf_mask,
            "gm_mask": gm_mask, "wm_mask": wm_mask,
            "valid_mask": valid_mask, "gt_t2_map": gt_t2_map,
            "subject": subject, "slice_file": slice_file,
        }
        data["raw_mc_samples"] = np.load(
            os.path.join(self.gt_dir.replace("gt", "raw_mc_predictions"), subject, slice_file)
        )
        if self.return_slices:
            data["recon_t2"] = np.load(
                os.path.join(
                    self.gt_dir.replace("gt", "recon_t2star_unc_signal"), subject, slice_file,
                )
            )
        return data

    def _accumulate_non_mc(self, data, valid_mask, compute_cvs, compute_rms, compute_phys):
        raw_mc    = data["raw_mc_samples"]
        mean, std = self._convert_mc_to_magnitude_mean_std(raw_mc)
        mean = self._flat_valid(mean, valid_mask, self.num_echoes)
        std  = self._flat_valid(std,  valid_mask, self.num_echoes)

        raw_mc_flat = raw_mc.transpose(2, 3, 0, 1).reshape(-1, raw_mc.shape[0], raw_mc.shape[1])
        self.all_raw_mc_samples.append(
            raw_mc_flat[valid_mask] if self.mode != "test" else raw_mc_flat
        )

        # Concat modes feed mean magnitude followed by its standard deviation.
        if "concat" in self.uncertainty_mode.lower():
            self.all_input_pixels_t2.append(np.concatenate([mean, std], axis=1))
        else:
            self.all_input_pixels_t2.append(mean)

        if compute_cvs:
            self.all_cvs.append(self._compute_cv_from_data(mean, std))
        if compute_rms:
            rms_map = self._compute_rms_from_data(raw_mc).reshape(-1)
            if self.mode != "test":
                rms_map = rms_map[valid_mask]
            self.all_rms.append(rms_map)
        if compute_phys:
            # L, mu = self._load_cholesky(data["subject"], data["slice_file"])
            # L, mu = L[valid_mask], mu[valid_mask]
            # E = self.num_echoes
            # mR = mu[:, :E]
            # mI = mu[:, E:]
            # # One batched matmul: (N,2E,2E) @ (N,2E,2E).T  — replaces the old
            # # O(N·S·(2E)²) centering + einsum.  NumPy dispatches this to BLAS DGEMM.
            # cov_ri = L @ L.transpose(0, 2, 1)      # (N, 2E, 2E)
            # cov_RR = cov_ri[:, :E, :E]             # (N, E, E)
            # cov_II = cov_ri[:, E:, E:]             # (N, E, E)
            # cov_RI = cov_ri[:, :E, E:]             # (N, E, E)  (Σ_IR = Σ_RI.T)
            # mM = np.sqrt(mR ** 2 + mI ** 2 + NORM_EPS)  # (N, E)  |E[z_e]|
            # mM2  = mM ** 2 + NORM_EPS                          # (N, E)
            # denom = mM2[:, :, None] * mM2[:, None, :]          # (N, E, E)
            # # All four cross-term types — vectorised over N
            # cov_log_cx = (
            #     mR[:, :, None] * mR[:, None, :] * cov_RR                   # R_e R_f
            #     + mI[:, :, None] * mI[:, None, :] * cov_II                   # I_e I_f
            #     + mR[:, :, None] * mI[:, None, :] * cov_RI                   # R_e I_f
            #     + mI[:, :, None] * mR[:, None, :] * cov_RI.transpose(0, 2, 1)# I_e R_f
            # ) / denom

            # JsigJ_cx = np.einsum('i,nij,j->n', self._feat_J_slope, cov_log_cx, self._feat_J_slope)
            # log_mM    = np.log(mM.clip(NORM_EPS))              # (N, E)
            # slope_ols = np.einsum('e,ne->n', self._feat_J_slope, log_mM)
            # slope_safe = np.clip(np.abs(slope_ols), 1e-4, None)
            # dt2_ds   = 1.0 / (slope_safe ** 2 + NORM_EPS)
            # propagated_t2_var_complex = ((dt2_ds ** 2) * JsigJ_cx).astype(np.float32)
            # self.all_phys.append(propagated_t2_var_complex)
            mc_mag  = np.abs(data["raw_mc_samples"])
            S, E, H, W = mc_mag.shape
            N          = H * W
            log_mc  = np.log(mc_mag.clip(NORM_EPS).reshape(N * S, E))
            slopes  = np.einsum('e,ne->n', self._feat_J_slope, log_mc)
            t2_samp = np.clip(-1.0 / (slopes + NORM_EPS), 0.0, 400.0).reshape(N, S)
            self.all_phys.append(t2_samp.std(axis=1).astype(np.float32))

    def _accumulate_mc_cholesky(self, subject, slice_file, data, valid_mask,
                                 compute_cvs, compute_rms, compute_phys):
        means_path    = self.gt_dir.replace("gt", "cholesky_means")
        matrices_path = self.gt_dir.replace("gt", "cholesky_matrices")
        means    = np.load(os.path.join(means_path,    subject, slice_file))
        matrices = np.load(os.path.join(matrices_path, subject, slice_file))
        D = self.num_echoes * 2
        means    = means.reshape(-1, D)[valid_mask]
        matrices = matrices.reshape(-1, D, D)[valid_mask]
        if compute_cvs:
            self.all_cvs.append(self._compute_cv_from_covariance(means, matrices))
        if compute_rms:
            self.all_rms.append(self._compute_rms_from_covariance(matrices))
        if compute_phys:                                        # ← ADD THIS BLOCK
            # E  = self.num_echoes
            # mR = means[:, :E]
            # mI = means[:, E:]
            # cov_ri = matrices @ matrices.transpose(0, 2, 1)    # (N, 2E, 2E)
            # cov_RR = cov_ri[:, :E, :E]
            # cov_II = cov_ri[:, E:, E:]
            # cov_RI = cov_ri[:, :E, E:]
            # mM     = np.sqrt(mR**2 + mI**2 + NORM_EPS)
            # mM2    = mM**2 + NORM_EPS
            # denom  = mM2[:, :, None] * mM2[:, None, :]
            # cov_log_cx = (
            #     mR[:, :, None] * mR[:, None, :] * cov_RR
            #     + mI[:, :, None] * mI[:, None, :] * cov_II
            #     + mR[:, :, None] * mI[:, None, :] * cov_RI
            #     + mI[:, :, None] * mR[:, None, :] * cov_RI.transpose(0, 2, 1)
            # ) / denom
            # log_mM     = np.log(mM.clip(NORM_EPS))
            # slope_ols  = np.einsum('e,ne->n', self._feat_J_slope, log_mM)
            # slope_safe = np.clip(np.abs(slope_ols), 1e-4, None)
            # dt2_ds     = 1.0 / (slope_safe**2 + NORM_EPS)
            # JsigJ_cx   = np.einsum('i,nij,j->n', self._feat_J_slope, cov_log_cx, self._feat_J_slope)
            # self.all_phys.append(((dt2_ds**2) * JsigJ_cx).astype(np.float32))
            mc_mag  = np.abs(data["raw_mc_samples"])
            S, E, H, W = mc_mag.shape
            N          = H * W
            log_mc  = np.log(mc_mag.clip(NORM_EPS).reshape(N * S, E))
            slopes  = np.einsum('e,ne->n', self._feat_J_slope, log_mc)
            t2_samp = np.clip(-1.0 / (slopes + NORM_EPS), 0.0, 400.0).reshape(N, S)
            self.all_phys.append(t2_samp.std(axis=1).astype(np.float32))
            
        self.all_means.append(means)
        self.all_corr_matrices.append(matrices)

    def _accumulate_mc_lowrank(self, subject, slice_file, data, valid_mask,
                                compute_cvs, compute_rms, compute_phys):
        rank          = self.lowrank_rank
        means_path    = self.gt_dir.replace("gt", f"lowrank_means/rank_{rank}")
        matrices_path = self.gt_dir.replace("gt", f"lowrank_U_matrices/rank_{rank}")
        means    = np.load(os.path.join(means_path,    subject, slice_file))
        matrices = np.load(os.path.join(matrices_path, subject, slice_file))
        D = self.num_echoes * 2
        means    = means.reshape(-1, D)[valid_mask]
        matrices = matrices.reshape(-1, D, rank)[valid_mask]
        if compute_cvs:
            self.all_cvs.append(self._compute_cv_from_covariance(means, matrices, is_lowrank=True))
        if compute_rms:
            self.all_rms.append(self._compute_rms_from_covariance(matrices))
        self.all_means.append(means)
        self.all_corr_matrices.append(matrices)

    def _accumulate_mc_uncorrelated(self, data, valid_mask, compute_cvs, compute_rms, compute_phys):
        raw_mc = data["raw_mc_samples"]
        D      = self.num_echoes * 2
        mean_real = np.mean(raw_mc.real, axis=0)
        mean_imag = np.mean(raw_mc.imag, axis=0)
        std_real  = np.std(raw_mc.real,  axis=0)
        std_imag  = np.std(raw_mc.imag,  axis=0)
        # Flatten each (E,H,W) block to (N,E) first, then join real and imag along the
        # feature axis. Concatenating the (E,H,W) arrays directly would join them along
        # W, which still yields N rows of width 2E but interleaves neighbouring voxels.
        def _pack(real, imag):
            return np.concatenate([real.transpose(1, 2, 0).reshape(-1, self.num_echoes),
                                   imag.transpose(1, 2, 0).reshape(-1, self.num_echoes)],
                                  axis=1)

        mean = _pack(mean_real, mean_imag)
        std  = _pack(std_real,  std_imag)
        mean, std = mean[valid_mask], std[valid_mask]
        if compute_cvs:
            sr, si = std[:, :self.num_echoes], std[:, self.num_echoes:]
            mr, mi = mean[:, :self.num_echoes], mean[:, self.num_echoes:]
            std_mag  = np.sqrt(sr**2 + si**2) / np.sqrt(2)
            mean_mag = np.sqrt(mr**2 + mi**2)
            self.all_cvs.append((std_mag / (mean_mag + 1e-9)).mean(axis=-1))
        if compute_rms:
            self.all_rms.append(self._compute_rms_from_data(raw_mc))
        self.all_means.append(mean)
        self.all_corr_matrices.append(std)

    def _maybe_accumulate_concat_std(self, data, valid_mask):
        """Concat modes need the per-echo std alongside the mean magnitude."""
        if "concat" not in self.uncertainty_mode.lower():
            return
        _, std_mag = self._convert_mc_to_magnitude_mean_std(data["raw_mc_samples"])
        self.all_stds.append(self._flat_valid(std_mag, valid_mask, self.num_echoes))

    # ------------------------------------------------------------------
    # Tissue label helpers
    # ------------------------------------------------------------------

    def _make_tissue_labels(self, data, valid_mask):
        wm_v  = data["wm_mask"].astype(bool).flatten()[valid_mask]
        gm_v  = data["gm_mask"].astype(bool).flatten()[valid_mask]
        csf_v = data["csf_mask"].astype(bool).flatten()[valid_mask]
        tissue    = np.full(wm_v.shape[0], 3, dtype=np.int64)
        tissue[csf_v] = 2
        tissue[wm_v]  = 0
        tissue[gm_v]  = 1
        return tissue

    # ------------------------------------------------------------------
    # Main loading loop
    # ------------------------------------------------------------------

    def _load_data(self):
        subjects = sorted(
            d for d in os.listdir(self.gt_dir)
            if os.path.isdir(os.path.join(self.gt_dir, d))
        )
        compute_cvs = (
            (self.use_adaptive_loss or self.use_adaptive_noise or self.use_cv_correlation_loss)
            and not self.return_slices
        )
        compute_rms = self.use_rms_correlation_loss and not self.return_slices
        compute_phys = self.use_phys_correlation_loss and not self.return_slices

        for subject in tqdm(subjects, desc="Loading data"):
            if self.return_slices:
                for sf in sorted(os.listdir(os.path.join(self.gt_dir, subject))):
                    self.slice_list.append((subject, sf))
                continue

            for slice_file in sorted(os.listdir(os.path.join(self.gt_dir, subject))):
                data = self._load_slice_data(subject, slice_file)
                if data["valid_mask"].sum() == 0:
                    continue

                valid_mask = data["valid_mask"].flatten()

                gt_t2 = self._normalize_parameter_maps(data["gt_t2_map"])
                if self.mode != "test":
                    gt_t2 = gt_t2.flatten()[valid_mask]
                self.all_gt_t2_pixels.append(gt_t2)

                self.all_t2_std_mc.append(
                    self._compute_mc_t2_std(
                        data["raw_mc_samples"], data["valid_mask"], self.te_values.numpy()
                    )
                )
                self.all_tissue_labels.append(self._make_tissue_labels(data, valid_mask))

                # Route to per-mode accumulator (populates all_input_pixels_t2 with mean_mag)
                if not self.use_mc or self.mode != "train":
                    self._accumulate_non_mc(data, valid_mask, compute_cvs, compute_rms, compute_phys)
                else:
                    mode_lower = self.uncertainty_mode.lower()
                    if mode_lower in (UncMode.CHOLESKY, UncMode.CHOLESKY_CAT):
                        self._accumulate_mc_cholesky(
                            subject, slice_file, data, valid_mask, compute_cvs, compute_rms, compute_phys
                        )
                    elif mode_lower in (UncMode.LOW_RANK, UncMode.LOW_RANK_CAT):
                        self._accumulate_mc_lowrank(
                            subject, slice_file, data, valid_mask, compute_cvs, compute_rms, compute_phys
                        )
                    elif mode_lower in (UncMode.UNCORRELATED, UncMode.UNCORRELATED_CAT):
                        self._accumulate_mc_uncorrelated(
                            data, valid_mask, compute_cvs, compute_rms, compute_phys
                        )
                    self._maybe_accumulate_concat_std(data, valid_mask)

        self._finalize_data()

    def _finalize_data(self):
        """Convert the list buffers filled by _load_data into tensors."""
        if self.return_slices:
            return

        self.all_gt_t2_pixels  = torch.from_numpy(np.concatenate(self.all_gt_t2_pixels)).float()
        self.all_tissue_labels = torch.from_numpy(np.concatenate(self.all_tissue_labels)).long()
        self.all_t2_std_mc     = torch.from_numpy(np.concatenate(self.all_t2_std_mc)).float()

        # MC training keeps the raw ingredients instead; the trainer samples on GPU.
        if self.mode == "test" or not (self.use_mc and self.mode == "train"):
            self.all_input_pixels_t2 = torch.from_numpy(
                np.concatenate(self.all_input_pixels_t2)
            ).float()
            if len(self.all_raw_mc_samples) > 0:
                self.all_raw_mc_samples = torch.from_numpy(
                    np.concatenate(self.all_raw_mc_samples)
                )

        if self.use_mc and self.mode == "train":
            self.all_means         = torch.from_numpy(np.concatenate(self.all_means)).float()
            self.all_corr_matrices = torch.from_numpy(np.concatenate(self.all_corr_matrices)).float()

        if hasattr(self, "all_stds") and len(self.all_stds) > 0:
            self.all_stds = torch.from_numpy(np.concatenate(self.all_stds)).float()

        if hasattr(self, "all_cvs") and len(self.all_cvs) > 0:
            self.voxel_cvs = torch.from_numpy(np.concatenate(self.all_cvs)).float()
            self._log_stats("CV (from loading)", self.voxel_cvs)
        if hasattr(self, "all_rms") and len(self.all_rms) > 0:
            self.voxel_rms = torch.from_numpy(np.concatenate(self.all_rms)).float()
            self._log_stats("RMS (from loading)", self.voxel_rms)
        if hasattr(self, "all_phys") and len(self.all_phys) > 0:
            self.voxel_phys = torch.from_numpy(np.concatenate(self.all_phys)).float()
            self._log_stats("Physical Variance (from loading)", self.voxel_phys)

    # ------------------------------------------------------------------
    # Norm stats fitting / loading
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Handcrafted feature computation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    @staticmethod
    def _log_stats(name, t):
        print(f"{name} statistics: mean={t.mean():.4f} median={t.median():.4f} "
              f"min={t.min():.4f} max={t.max():.4f}")

    def __len__(self):
        if self.return_slices:
            return len(self.slice_list)
        return len(self.all_gt_t2_pixels)

    def __getitem__(self, idx):
        return self._get_slice(idx) if self.return_slices else self._get_pixel(idx)

    # ------------------------------------------------------------------
    # Pixel-mode item
    # ------------------------------------------------------------------

    def _get_pixel(self, idx):
        sample = {
            "gt_pixel":    self.all_gt_t2_pixels[idx].unsqueeze(0),
            "valid_mask":  torch.tensor([True], dtype=torch.bool),
            "tissue_type": self.all_tissue_labels[idx].unsqueeze(0),
            "t2_std_mc":   self.all_t2_std_mc[idx].unsqueeze(0),
        }
        if hasattr(self, "voxel_cvs") and self.voxel_cvs is not None:
            sample["cv"] = self.voxel_cvs[idx].unsqueeze(0)
        if hasattr(self, "voxel_rms") and self.voxel_rms is not None:
            sample["rms"] = self.voxel_rms[idx].unsqueeze(0)
        if hasattr(self, "voxel_phys") and self.voxel_phys is not None:
            sample["t2_var"] = self.voxel_phys[idx].unsqueeze(0)

        noise_scale = (
            self.voxel_noise_scales[idx].item()
            if (self.use_adaptive_noise and self.voxel_noise_scales is not None)
            else 1.0
        )
        sample["voxel_noise_scale"]   = torch.tensor([noise_scale],                     dtype=torch.float32)
        sample["current_noise_scale"] = torch.tensor([float(self.current_noise_scale)],  dtype=torch.float32)

        if self.use_mc and self.mode == "train":
            return self._get_pixel_mc(idx, sample)
        return self._get_pixel_precomputed(idx, sample)

    def _get_pixel_mc(self, idx, sample):
        """MC training path — return raw ingredients for GPU sampling in trainer."""
        sample["mean_ri"]  = self.all_means[idx].unsqueeze(0)
        sample["L_or_std"] = self.all_corr_matrices[idx].unsqueeze(0)
        sample["unc_mode"] = self.uncertainty_mode
        sample["std_mag"]  = (
            self.all_stds[idx].unsqueeze(0).to(torch.float32)
            if (isinstance(getattr(self, "all_stds", None), torch.Tensor))
            else "None"
        )
        sample["need_pair"]      = bool(self.use_consistency_loss and self.mode == "train")
        sample["input_pixel_t2"] = "None"
        sample["L_matrix"]       = "None"
        return sample

    def _get_pixel_precomputed(self, idx, sample):
        """Val / test / non-MC train path: normalise by the first echo on the fly."""
        feat = self.all_input_pixels_t2[idx].unsqueeze(0)
        nf   = feat[:, 0:1] + NORM_EPS
        feat = (
            torch.log(feat + NORM_EPS) - torch.log(feat[:, 0:1] + NORM_EPS)
            if self.use_log_input
            else feat / nf
        )
        sample["input_pixel_t2"] = feat
        sample["raw_mc_samples"] = (
            self.all_raw_mc_samples[idx]
            if isinstance(self.all_raw_mc_samples, torch.Tensor)
            else "None"
        )
        sample["L_matrix"] = (
            self.all_corr_matrices[idx].unsqueeze(0)
            if self.use_uncertainty_encoder
            else "None"
        )
        # Raw (un-normalised) magnitudes, used by the physics loss.
        sample["raw_mag"] = self.all_input_pixels_t2[idx]
        return sample

    # ------------------------------------------------------------------
    # Slice-mode item
    # ------------------------------------------------------------------
    def _get_slice(self, idx):
        subject, slice_file = self.slice_list[idx]
        data = self._load_slice_data(subject, slice_file)

        if np.sum(data["brain_mask"]) == 0:
            print(f"Warning: empty brain mask for {subject}/{slice_file}. Skipping.")
            return None

        mean_mag, std_mag = self._convert_mc_to_magnitude_mean_std(data["raw_mc_samples"])
        mean_mag = mean_mag.transpose(1, 2, 0)   # (H, W, E)
        std_mag  = std_mag.transpose(1, 2, 0)    # (H, W, E)
        slice_cv  = self._compute_cv_from_data(mean_mag, std_mag)
        slice_rms = self._compute_rms_from_data(data["raw_mc_samples"])

        # Normalise each per-echo vector by its first echo.
        nf       = mean_mag[:, :, 0:1] + 1e-9
        input_t2 = (
            np.concatenate([mean_mag / nf, std_mag / nf], axis=-1)
            if (self.uncertainty_mode.endswith("_concat") or self.uncertainty_mode == "concat")
            else mean_mag / nf
        )
        gt_norm = self._normalize_parameter_maps(data["gt_t2_map"])

        return {
            "input_pixels_t2":     torch.from_numpy(input_t2).float(),
            "gt_map":              torch.from_numpy(gt_norm).float(),
            "brain_mask":          torch.from_numpy(data["brain_mask"]).bool(),
            "wm_mask":             torch.from_numpy(data["wm_mask"]).bool(),
            "gm_mask":             torch.from_numpy(data["gm_mask"]).bool(),
            "csf_mask":            torch.from_numpy(data["csf_mask"]).bool(),
            "signal_measurements": torch.from_numpy(mean_mag).float(),
            "recon_signal_t2":     torch.from_numpy(data["recon_t2"]).float(),
            "subject":             subject,
            "slice_num":           slice_file.replace("slice_", "").replace(".npy", ""),
            "cv":                  torch.from_numpy(slice_cv).float(),
            "rms":                 torch.from_numpy(slice_rms).float(),
            "valid_mask":          torch.from_numpy(data["valid_mask"]).bool(),
        }
    