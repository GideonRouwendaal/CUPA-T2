import glob
import json
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from scipy.stats import norm, pearsonr, probplot, spearmanr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.t2star_regress_loader import StaticPixelDataset
from model_zoo.fcn_regress import FCN, HeteroscedasticMLP

# ── Path anchors ──────────────────────────────────────────────────────────────
# Everything is derived from this file's location so the project can be moved or
# checked out anywhere. PROJECT_DIR/../.. is the iml-dl repo root; its parent is
# the CUPA root that holds ./data.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
RESULTS_REGRESSION = REPO_ROOT / "results" / "regression"
COLOR_MAP_DIR = PROJECT_DIR / "color_map"
# ──────────────────────────────────────────────────────────────────────────────

# Slices cached in full (prediction, uncertainty, masks) so the qualitative
# figure can be rebuilt without re-running evaluation. Mid-brain slices, where
# all three tissue classes are well represented.
FIGURE_SLICES = (10, 13, 15, 16, 17, 18)


def _compute_gaussian_nll(y_true, y_pred, sigma, eps=1e-9):
    """Compute Gaussian negative log-likelihood."""
    sigma = np.maximum(sigma, eps)
    squared_error = (y_true - y_pred) ** 2
    nll = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(sigma ** 2) + 0.5 * (squared_error / (sigma ** 2))
    return float(np.mean(nll))


def _compute_empirical_coverage(errors, uncertainties, sigma_levels=[1, 2, 3]):
    coverage = {}
    for k in sigma_levels:
        within_interval = errors <= (k * uncertainties)
        coverage[f'{k}sigma'] = float(np.mean(within_interval))

    # add ECE over the provided sigma_levels
    theoretical = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    devs = [abs(coverage[f"{k}sigma"] - theoretical[k]) for k in sigma_levels]
    coverage["ece"] = float(np.mean(devs))

    return coverage


def _compute_informativeness_spearman(errors, uncertainties):
    """Compute Spearman correlation between absolute error and uncertainty."""
    try:
        rho, p_val = spearmanr(errors, uncertainties)
        return float(rho), float(p_val)
    except Exception:
        return np.nan, np.nan

def _compute_aurc(errors, uncertainties):
    """AURC with coverage increasing from 1/n .. 1 (lower is better)."""
    errors = np.asarray(errors, dtype=np.float64)
    uncertainties = np.asarray(uncertainties, dtype=np.float64)

    valid = np.isfinite(errors) & np.isfinite(uncertainties)
    errors = errors[valid]
    uncertainties = uncertainties[valid]
    n = errors.size
    if n == 0:
        return np.nan, np.array([]), np.array([])

    # Keep low-uncertainty first (ascending)
    order = np.argsort(uncertainties)
    e = errors[order]

    # risk(k) = mean error of retained k least-uncertain points
    cumsum = np.cumsum(e)
    ks = np.arange(1, n + 1)
    risks = cumsum / ks
    coverage = ks / n  # increasing

    aurc = float(np.trapz(risks, coverage))
    return aurc, coverage, risks



def _compute_cross_stage_consistency(rms_uncertainty, aleatoric_uncertainty):
    """Compute Pearson correlation between RMS and aleatoric uncertainty."""
    try:
        r, p_val = pearsonr(rms_uncertainty, aleatoric_uncertainty)
        return float(r), float(p_val)
    except Exception:
        return np.nan, np.nan

class RegressionEvaluator:
    """Clean evaluator for regression models (T2*, S0, or both) with RMS-based input uncertainty analysis."""

    def __init__(self, config, device, recon_model=None):
        self.config = config
        self.device = device

        # Extract configuration
        self.regression_params = config['regression_params']
        self.train_params = self.regression_params['regression_train_params']
        self.model_params = self.regression_params['regression_model_params']

        self.model_path_override = self.regression_params.get("model_path_override", None)
        self.output_dir_override = self.regression_params.get("output_dir_override", None)


        # Parameter ranges
        self.max_t2, self.min_t2 = config.get('max_t2', 200.0), config.get('min_t2', 0.0)
        self.max_s0, self.min_s0 = self.regression_params.get('s0_max', 2.5), self.regression_params.get('s0_min', 0.0)

        # Model settings
        self.uncertainty_mode = self.regression_params.get('uncertainty_mode', 'none')
        self.heteroscedastic = self.regression_params.get('heteroscedastic', False)
        self.use_mc_dropout = self.regression_params.get('use_mc_dropout', False)
        self.mc_dropout_samples = self.regression_params.get('mc_dropout_samples', 100)

        # Other settings
        self.acc_rate = str(self.regression_params.get('acc_rate', 2))
        self.use_rms_correlation_loss = self.train_params.get('use_rms_correlation_loss', False)
        self.rms_correlation_weight = self.train_params.get('rms_correlation_weight', 0.1)

        # Plotting
        self.to_plot = False

        # SSIM mode: "bbox" (recommended) or "full"
        self.ssim_mode = self.regression_params.get('ssim_mode', 'full')  # 'bbox' or 'full'

        # Calibration settings
        self.calibration_bins = int(self.regression_params.get('calibration_bins', 10))
        self.calibration_min_points_per_bin = int(self.regression_params.get('calibration_min_points_per_bin', 200))

        # Initialize
        self.model = None
        self.test_loader = None
        self._init_statistics()
        self._setup_colormap()

    # ==================== UTILS ====================

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray):
        """Return bbox slices (yslice, xslice) around True region. If empty, return None."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return slice(y0, y1 + 1), slice(x0, x1 + 1)

    @staticmethod
    def _safe_ssim_win_size(h: int, w: int, preferred: int = 11) -> int:
        """Pick an odd win_size <= min(h,w)."""
        m = min(h, w)
        if m < 3:
            return 3
        win = min(preferred, m)
        if win % 2 == 0:
            win -= 1
        return max(3, win)

    @staticmethod
    def _pearson_spearman(x, y):
        try:
            rp, pp = pearsonr(x, y)
        except Exception:
            rp, pp = np.nan, np.nan
        try:
            rs, ps = spearmanr(x, y)
        except Exception:
            rs, ps = np.nan, np.nan
        return rp, pp, rs, ps





    def _tissue_masks_1d(self, data, base_mask=None):
        """
        Returns list of (name, mask_1d) for: Overall, WM, GM, CSF (if labels exist).
        base_mask can further constrain (e.g., finite x/y).
        """
        n = len(data.get('errors', []))
        if n == 0:
            return [('Overall', np.array([], dtype=bool))]

        if base_mask is None:
            base_mask = np.ones(n, dtype=bool)

        out = [('Overall', base_mask.copy())]

        labels = data.get('tissue_labels', None)
        if labels is not None and len(labels) == n:
            out.append(('WM', base_mask & (labels == 0)))
            out.append(('GM', base_mask & (labels == 1)))
            out.append(('CSF', base_mask & (labels == 2)))
        return out

    # ==================== INIT ====================

    def _init_statistics(self):
        """Initialize data collection structures."""
        self.pred_t2_list, self.gt_t2_list, self.recon_t2_list = [], [], []

        # Pixel-level data for uncertainty analysis
        self.pixel_level_data = {
            'predictions': [],
            'ground_truths': [],

            # store abs error (MAE per-pixel) as before
            'errors': [],

            # ---- RMS instead of CV ----
            'input_rms': [],           # raw RMS map values
            'input_rms_norm': [],      # RMS normalized by TE0 magnitude (dimensionless)

            # Predicted uncertainties
            'aleatoric': [],
            'epistemic': [],
            'total_uncertainty': [],

            # Tissue labels: 0=WM,1=GM,2=CSF; -1=other brain pixels
            'tissue_labels': [],
        }
        self.per_slice_records = []

    def _setup_colormap(self):
        """Setup T2* colormap."""
        csv_path = str(COLOR_MAP_DIR / 'navia.csv')
        ori = np.loadtxt(csv_path)
        ori[0, :] = 0.0
        self.t2_cmap = ListedColormap(ori, name="t2star")

    # ==================== SETUP ====================

    def setup_test_dataset(self):
        """Setup test dataset."""
        dataset_params = {
            'te_values': self.regression_params.get('te_values', list(range(5, 65, 5))),
            'min_t2': self.min_t2, 'max_t2': self.max_t2,
            'min_s0': self.min_s0, 'max_s0': self.max_s0,
            'uncertainty_mode': self.uncertainty_mode,
            'lowrank_rank': self.regression_params.get('lowrank_rank', None),
            'return_slices': True
        }

        gt_path = self.regression_params['regression_gt_test_location']

        test_dataset = StaticPixelDataset(
            data_path_input=gt_path,
            mode='test',
            **dataset_params
        )
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        logging.info(f'Test dataset: {len(test_dataset)} slices')

    def load_model(self):
        """Load trained model with correct configuration."""
        model_path = self.model_path_override if self.model_path_override is not None else self._build_model_path()
        model_path = str(model_path)

        model_params = self._get_model_params()
        model_class = HeteroscedasticMLP if self.heteroscedastic else FCN
        logging.info(f'Loading checkpoint {model_path} into '
                     f'{model_class.__name__}({model_params})')

        self.model = model_class(**model_params).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        # Set mode
        if self.use_mc_dropout:
            self.model.train()
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        else:
            self.model.eval()

        logging.info(f'Loaded model: {model_path}')

    def _get_model_params(self):
        """Model kwargs, with the same runtime adjustments the trainer applied."""
        params = self.model_params.copy()

        if self.uncertainty_mode.endswith("_concat") or self.uncertainty_mode == "concat":
            params['in_ch'] *= 2
        if self.use_mc_dropout and params.get('dropout_rate', 0) <= 0:
            params['dropout_rate'] = 0.2

        return params

    def _build_model_path(self):
        """Rebuild the directory the trainer wrote to, then locate the checkpoint.

        This must stay in step with `RegressionTrainer.setup_output_directory`;
        the literal `T2_only` / `no_physics` / `no_pe` segments are frozen there
        for the same reason.
        """
        base = self.regression_params.get('regress_model_dir') or str(RESULTS_REGRESSION)
        base = str(base).rstrip('/') + '/'

        test_loc = self.regression_params["regression_gt_test_location"]
        m = re.search(r"(Exp\d+)", str(test_loc))
        exp = m.group(1) if m else ""

        path = os.path.join(base, f"{exp}/acc_rate_{self.acc_rate}/{self.uncertainty_mode}/")
        if "low_rank" in self.uncertainty_mode.lower():
            path = os.path.join(path, f"rank_{self.regression_params.get('lowrank_rank')}/")

        path = os.path.join(path, 'T2_only/')
        path = os.path.join(path, 'no_physics/')
        path = os.path.join(path, 'heteroscedastic/' if self.heteroscedastic else 'homoscedastic/')
        path = os.path.join(path, 'mc_dropout/' if self.use_mc_dropout else 'no_dropout/')
        path = os.path.join(path, 'no_pe/')
        path = os.path.join(path, f'{self.train_params.get("loss", "L2")}/')

        if self.use_rms_correlation_loss:
            path = os.path.join(path, 'rms_corr_loss/')
            path = os.path.join(path, f'rms_corr_weight_{self.rms_correlation_weight}/')

        return self._resolve_checkpoint(path)

    @staticmethod
    def _resolve_checkpoint(run_root):
        """Find best_regression_model.pth under a hyperparameter directory.

        The trainer writes each run into its own '<run_tag>_<hash>/' sub-directory,
        so the checkpoint normally sits one level below the path built from the
        hyperparameters. Older runs wrote it directly into that directory, which is
        still honoured. When several runs share the same hyperparameter directory the
        choice is ambiguous, so list them and let the caller pass model_path_override.
        """
        direct = os.path.join(run_root, 'best_regression_model.pth')
        if os.path.isfile(direct):
            return direct

        nested = sorted(glob.glob(os.path.join(run_root, '*', 'best_regression_model.pth')))
        if len(nested) == 1:
            return nested[0]
        if len(nested) > 1:
            listed = "\n  ".join(nested)
            raise FileNotFoundError(
                f"{len(nested)} checkpoints share the hyperparameter directory\n"
                f"  {run_root}\n"
                f"Pass model_path_override (or --checkpoint) to pick one:\n  {listed}"
            )
        return direct
    def setup_output_directory(self, base=None):
        """Create the output directory, mirroring the checkpoint's directory layout."""
        if base is None:
            base = RESULTS_REGRESSION / "test"
        if self.output_dir_override is not None:
            out = str(self.output_dir_override)
        else:
            # Mirror the checkpoint's own hyperparameter sub-tree under `base`, so
            # sweeps and nested run directories keep their structure in the results.
            ckpt = Path(self.model_path_override) if self.model_path_override else Path(self._build_model_path())
            out = str(Path(base) / self._checkpoint_relpath(ckpt.parent.resolve()))

        print(f'Creating output directory: {out}')
        os.makedirs(out, exist_ok=True)
        os.makedirs(os.path.join(out, 'uncertainty_analysis'), exist_ok=True)
        logging.info(f'Output directory: {out}')
        return out

    def _checkpoint_relpath(self, ckpt_dir):
        """Path of a checkpoint directory relative to whichever results root holds it.

        Checkpoints normally live under <repo>/results/regression, but runs trained
        with a custom regress_model_dir do not; fall back to that root, and finally to
        the bare directory name so evaluation never fails on path bookkeeping alone.
        """
        roots = [RESULTS_REGRESSION]
        configured = self.regression_params.get('regress_model_dir')
        if configured:
            roots.append(Path(configured))
        for root in roots:
            try:
                return ckpt_dir.relative_to(Path(root).resolve())
            except ValueError:
                continue
        return Path(ckpt_dir.name)

    def _safe_subject_str(self, x):
        # Handles bytes / numpy / torch-ish containers
        try:
            if isinstance(x, (bytes, bytearray)):
                return x.decode("utf-8")
        except Exception:
            pass
        return str(x)

    def _safe_int(self, x, default=-1):
        try:
            # torch tensor -> python int
            if hasattr(x, "item"):
                return int(x.item())
            return int(x)
        except Exception:
            return default

    def _run_metadata(self):
        """Metadata repeated per row so you can merge across runs later."""
        test_loc = str(self.regression_params.get("regression_gt_test_location", ""))
        m = re.search(r"(Exp\d+)", test_loc)

        return {
            "exp": m.group(1) if m else "Unknown",
            "acc_rate": str(self.acc_rate),
            "uncertainty_mode": str(self.uncertainty_mode),
            "heteroscedastic": bool(self.heteroscedastic),
            "use_mc_dropout": bool(self.use_mc_dropout),
            "mc_dropout_samples": int(self.mc_dropout_samples) if self.use_mc_dropout else 0,
            "checkpoint_path": str(self.model_path_override) if self.model_path_override else str(self._build_model_path()),
        }
    def _append_per_slice_records(self, metrics, data, predictions):
        """
        Save rows per (subject, slice, tissue, method).
        tissue ∈ {overall, wm, gm, csf}
        method ∈ {pred, recon}
        """
        subj = self._safe_subject_str(data.get("subject", "Unknown"))
        slc  = self._safe_int(data.get("slice_num", -1))

        gt = predictions.get("gt_t2", None)
        if gt is None:
            return

        brain = data["brain_mask"].astype(bool)
        base  = brain & (gt != 0)

        tissue_masks = {
            "overall": base,
            "wm":  base & data["wm_mask"].astype(bool),
            "gm":  base & data["gm_mask"].astype(bool),
            "csf": base & data["csf_mask"].astype(bool),
        }

        PERF_KEYS = ["nrmse", "mse", "mae", "ssim", "psnr"]

        # ---- EXPANDED probabilistic keys (all coverage variants + ECE) ----
        PROB_KEYS = [
            # Global NLL
            "nll",
            # Total-uncertainty coverage (used as legacy "coverage_Xsigma")
            "coverage_1sigma", "coverage_2sigma", "coverage_3sigma",
            # Aleatoric-uncertainty coverage (NEW)
            "coverage_alea_1sigma", "coverage_alea_2sigma", "coverage_alea_3sigma",
            # Total-uncertainty coverage (explicit, NEW)
            "coverage_total_1sigma", "coverage_total_2sigma", "coverage_total_3sigma",
            # ECE variants (NEW)
            "coverage_ece_alea_123",   # mean|empirical − nominal| over 1/2/3σ (aleatoric σ)
            "coverage_ece_total_123",  # same but for total σ
            # Informativeness
            "spearman_err_unc", "spearman_err_unc_pval",
            # Selective prediction
            "aurc",
            # Cross-stage consistency
            "cross_stage_consistency", "cross_stage_consistency_pval",
        ]

        meta = self._run_metadata()

        def k_pred(t):  return "t2"          if t == "overall" else f"{t}_t2"
        def k_recon(t): return "recon_t2"    if t == "overall" else f"{t}_recon_t2"
        def k_prob(t):  return "t2_probabilistic" if t == "overall" else f"{t}_t2_probabilistic"

        for tissue_name, m in tissue_masks.items():
            n_pix = int(np.sum(m))
            if n_pix <= 0:
                continue

            # --- pred row ---
            row = {
                **meta,
                "subject":   subj,
                "slice_num": slc,
                "tissue":    tissue_name,
                "method":    "pred",
                "n_pixels":  n_pix,
            }

            pm = metrics.get(k_pred(tissue_name), {})
            for kk in PERF_KEYS:
                row[kk] = float(pm.get(kk, np.nan))

            pb = metrics.get(k_prob(tissue_name), {})
            for kk in PROB_KEYS:
                row[kk] = float(pb.get(kk, np.nan))

            self.per_slice_records.append(row)

            # --- recon row ---
            row_b = {
                **meta,
                "subject":   subj,
                "slice_num": slc,
                "tissue":    tissue_name,
                "method":    "recon",
                "n_pixels":  n_pix,
            }

            bm = metrics.get(k_recon(tissue_name), {})
            for kk in PERF_KEYS:
                row_b[kk] = float(bm.get(kk, np.nan))

            # recon has no probabilistic metrics
            for kk in PROB_KEYS:
                row_b[kk] = np.nan

            self.per_slice_records.append(row_b)

    def _save_per_slice_records(self, output_path):
        if not getattr(self, "per_slice_records", None):
            print("No per-slice records collected; skipping per_slice_metrics export.")
            return

        df = pd.DataFrame(self.per_slice_records)
        df.sort_values(["subject", "slice_num", "tissue", "method"], inplace=True)

        csv_path = os.path.join(output_path, "per_slice_tissue_metrics.csv")
        df.to_csv(csv_path, index=False)

        # Parquet is really handy later for stats/merging (optional)
        try:
            pq_path = os.path.join(output_path, "per_slice_tissue_metrics.parquet")
            df.to_parquet(pq_path, index=False)
        except Exception as e:
            print(f"Parquet export skipped ({e}). CSV is saved.")

        print(f"✓ Saved per-slice metrics: {csv_path}")

    # ==================== EVALUATION ====================

    def evaluate(self):
        """Main evaluation loop."""
        output_path = self.setup_output_directory()
        self.setup_test_dataset()
        self.load_model()

        metrics_by_type = self._init_metrics_dict()

        for batch in tqdm(self.test_loader, desc='Evaluating'):
            if not self._is_valid_slice(batch):
                continue

            results = self._process_batch(batch, output_path)
            if results:
                self._accumulate_metrics(results, metrics_by_type)

        self._save_results(output_path, metrics_by_type)

        if self.heteroscedastic or self.use_mc_dropout:
            self._perform_uncertainty_analysis(output_path)

        return metrics_by_type

    def _init_metrics_dict(self):
        def base():
            return {'nrmse': [], 'mse': [], 'mae': [], 'ssim': [], 'psnr': []}

        def prob_base():
            return {
                'nll': [],  # Gaussian negative log-likelihood
                'coverage_1sigma': [],  # Empirical coverage at 1σ
                'coverage_2sigma': [],  # Empirical coverage at 2σ
                'coverage_3sigma': [],  # Empirical coverage at 3σ
                # new keys
                'coverage_alea_1sigma': [],
                'coverage_alea_2sigma': [],
                'coverage_alea_3sigma': [],
                'coverage_total_1sigma': [],
                'coverage_total_2sigma': [],
                'coverage_total_3sigma': [],
                'coverage_ece_alea_123': [],
                'coverage_ece_total_123': [],
                'spearman_err_unc': [],  # Spearman(|error|, uncertainty)
                'spearman_err_unc_pval': [],  # p-value
                'aurc': [],  # Area under risk-coverage curve
                'cross_stage_consistency': [],  # Pearson(RMS, aleatoric)
                'cross_stage_consistency_pval': []  # p-value
            }

        metrics = {
            't2': base(),
            'recon_t2': base(),
            't2_probabilistic': prob_base(),
        }

        for tissue in ['wm', 'gm', 'csf']:
            for param in ['t2', 'recon_t2']:
                metrics[f'{tissue}_{param}'] = base()
            metrics[f'{tissue}_t2_probabilistic'] = prob_base()

        return metrics


    def _compute_probabilistic_metrics(self, predictions, data, mask):
        """Compute comprehensive probabilistic metrics for a slice."""
        if not np.any(mask):
            return {
                'nll': np.nan, 'coverage_1sigma': np.nan, 'coverage_2sigma': np.nan,
                'coverage_3sigma': np.nan, 'spearman_err_unc': np.nan,
                'spearman_err_unc_pval': np.nan, 'aurc': np.nan,
                'cross_stage_consistency': np.nan, 'cross_stage_consistency_pval': np.nan,
                'coverage_ece_alea_123': np.nan, 'coverage_ece_total_123': np.nan
            }
        
        pred_t2 = predictions['pred_t2'][mask].flatten()
        gt_t2 = predictions['gt_t2'][mask].flatten()
        errors = np.abs(pred_t2 - gt_t2)
        
        prob_metrics = {}
        
        # 1. Gaussian NLL
        if 'uncertainty_total' in predictions:
            total_unc = predictions['uncertainty_total'][mask].flatten()
            prob_metrics['nll'] = _compute_gaussian_nll(gt_t2, pred_t2, total_unc)
        else:
            prob_metrics['nll'] = np.nan
        
        # 2. Empirical Coverage
        # Coverage with aleatoric σ
        # 2. Empirical Coverage (choose one to report under the existing keys)
        if 'uncertainty_total' in predictions:
            sigma = predictions['uncertainty_total'][mask].flatten()
            cov = _compute_empirical_coverage(errors, sigma, [1,2,3])
            prob_metrics['coverage_total_1sigma'] = cov['1sigma']
            prob_metrics['coverage_total_2sigma'] = cov['2sigma']
            prob_metrics['coverage_total_3sigma'] = cov['3sigma']
            prob_metrics['coverage_ece_total_123'] = cov['ece']
        else:
            prob_metrics['coverage_total_1sigma'] = np.nan
            prob_metrics['coverage_total_2sigma'] = np.nan
            prob_metrics['coverage_total_3sigma'] = np.nan
            prob_metrics['coverage_ece_total_123'] = np.nan
        
        if 'uncertainty_aleatoric' in predictions:
            sigma = predictions['uncertainty_aleatoric'][mask].flatten()
            cov = _compute_empirical_coverage(errors, sigma, [1,2,3])
            prob_metrics['coverage_alea_1sigma'] = cov['1sigma']
            prob_metrics['coverage_alea_2sigma'] = cov['2sigma']
            prob_metrics['coverage_alea_3sigma'] = cov['3sigma']
            prob_metrics['coverage_ece_alea_123'] = cov['ece']
        else:
            prob_metrics['coverage_alea_1sigma'] = np.nan
            prob_metrics['coverage_alea_2sigma'] = np.nan
            prob_metrics['coverage_alea_3sigma'] = np.nan
            prob_metrics['coverage_ece_alea_123'] = np.nan
        
        # 3. Informativeness (Spearman)
        if 'uncertainty_total' in predictions:
            total_unc = predictions['uncertainty_total'][mask].flatten()
            rho, p_val = _compute_informativeness_spearman(errors, total_unc)
            prob_metrics['spearman_err_unc'] = rho
            prob_metrics['spearman_err_unc_pval'] = p_val
        else:
            prob_metrics['spearman_err_unc'] = np.nan
            prob_metrics['spearman_err_unc_pval'] = np.nan
        
        # 4. AURC
        if 'uncertainty_total' in predictions:
            total_unc = predictions['uncertainty_total'][mask].flatten()
            aurc, _, _ = _compute_aurc(errors, total_unc)
            prob_metrics['aurc'] = aurc
        else:
            prob_metrics['aurc'] = np.nan
        
        # 5. Cross-stage Consistency
        if 'uncertainty_aleatoric' in predictions and 'rms_map' in data:
            rms = data['rms_map'][mask].flatten()
            aleatoric = predictions['uncertainty_aleatoric'][mask].flatten()
            
            valid = np.isfinite(rms) & np.isfinite(aleatoric) & (rms > 0) & (aleatoric > 0)
            if np.sum(valid) > 10:
                r, p_val = _compute_cross_stage_consistency(rms[valid], aleatoric[valid])
                prob_metrics['cross_stage_consistency'] = r
                prob_metrics['cross_stage_consistency_pval'] = p_val
            else:
                prob_metrics['cross_stage_consistency'] = np.nan
                prob_metrics['cross_stage_consistency_pval'] = np.nan
        else:
            prob_metrics['cross_stage_consistency'] = np.nan
            prob_metrics['cross_stage_consistency_pval'] = np.nan
        
        return prob_metrics


    import os, json
    import numpy as np

    def _save_single_slice_cache(self, predictions, data, batch, output_path):
        """
        Save one slice worth of:
        - maps for paper/debug plots (T2*, error, σ_total/alea/epi, RMS, TE background)
        - pixel-level arrays needed for AURC re-plotting offline (errors, σ, RMS, tissue labels, etc.)
        Output:
        output_path/cache_slices/<sub>_slice_<k>.npz
        output_path/cache_slices/<sub>_slice_<k>.json
        """
        cache_dir = os.path.join(output_path, "cache_slices")
        os.makedirs(cache_dir, exist_ok=True)

        subj = self._safe_subject_str(data.get("subject", "Unknown"))
        slc  = self._safe_int(data.get("slice_num", -1))
        tag = f"{subj}_slice_{slc:03d}"

        # ---------- Core masks ----------
        brain = data["brain_mask"].astype(bool)
        wm    = data.get("wm_mask", np.zeros_like(brain, bool)).astype(bool)
        gm    = data.get("gm_mask", np.zeros_like(brain, bool)).astype(bool)
        csf   = data.get("csf_mask", np.zeros_like(brain, bool)).astype(bool)

        # Use same validity you used for pixel_level_data (important!)
        gt = predictions.get("gt_t2", None)
        pred = predictions.get("pred_t2", None)
        if gt is None or pred is None:
            print(f"[WARN] {tag}: missing gt_t2 or pred_t2; not caching.")
            return

        valid = brain & (gt != 0)

        # ---------- Tissue labels map (aligned with your conventions) ----------
        tissue_map = np.full_like(brain, fill_value=-1, dtype=np.int8)
        tissue_map[wm]  = 0
        tissue_map[gm]  = 1
        tissue_map[csf] = 2

        # ---------- Uncertainty maps ----------
        sigma_total = predictions.get("uncertainty_total", None)
        sigma_alea  = predictions.get("uncertainty_aleatoric", None)
        sigma_epi   = predictions.get("uncertainty_epistemic", None)

        # ---------- RMS (raw + optional TE0-normalized) ----------
        rms_raw = None
        rms_t = batch.get("rms", None)
        if rms_t is not None:
            rms_raw = rms_t.squeeze().cpu().numpy().astype(np.float32)

        rms_norm = rms_raw.copy() if rms_raw is not None else None

        # ---------- Helpful background TE image for later plotting ----------
        # Save one TE magnitude (you can change index); also save TE list if present.
        te_bg_index = int(self.regression_params.get("cache_te_bg_index", 3))
        bg = None
        sig = data.get("signal_measurements", None)
        if isinstance(sig, np.ndarray) and sig.ndim == 3 and sig.shape[-1] > te_bg_index:
            bg = sig[..., te_bg_index].astype(np.float32)

        # ---------- Pixel-level vectors for offline AURC ----------
        # (These let you recompute AURC for this slice without eval.)
        pred_v = pred[valid].astype(np.float32).ravel()
        gt_v   = gt[valid].astype(np.float32).ravel()
        err_v  = np.abs(gt_v - pred_v).astype(np.float32)

        # selection scores
        def vec_from_map(m):
            if m is None:
                return np.full(err_v.shape, np.nan, dtype=np.float32)
            return m[valid].astype(np.float32).ravel()

        sig_total_v = vec_from_map(sigma_total)
        sig_alea_v  = vec_from_map(sigma_alea)
        sig_epi_v   = vec_from_map(sigma_epi)

        rms_v = (rms_norm[valid].astype(np.float32).ravel()
                if rms_norm is not None else np.full(err_v.shape, np.nan, dtype=np.float32))

        tissue_v = tissue_map[valid].astype(np.int8).ravel()

        # ---------- Save NPZ ----------
        npz_path = os.path.join(cache_dir, f"{tag}.npz")
        np.savez_compressed(
            npz_path,

            # maps for plotting
            gt_t2=gt.astype(np.float32),
            pred_t2=pred.astype(np.float32),
            recon_t2=np.asarray(predictions.get("recon_t2", np.zeros_like(gt)), np.float32),
            err_abs=np.abs(gt - pred).astype(np.float32),

            brain_mask=brain.astype(np.uint8),
            wm_mask=wm.astype(np.uint8),
            gm_mask=gm.astype(np.uint8),
            csf_mask=csf.astype(np.uint8),
            valid_mask=valid.astype(np.uint8),
            tissue_map=tissue_map,

            sigma_total=(sigma_total.astype(np.float32) if sigma_total is not None else np.array([])),
            sigma_aleatoric=(sigma_alea.astype(np.float32) if sigma_alea is not None else np.array([])),
            sigma_epistemic=(sigma_epi.astype(np.float32) if sigma_epi is not None else np.array([])),

            rms_raw=(rms_raw.astype(np.float32) if rms_raw is not None else np.array([])),
            rms_norm=(rms_norm.astype(np.float32) if rms_norm is not None else np.array([])),

            te_bg_index=np.array([te_bg_index], dtype=np.int32),
            te_bg=(bg if bg is not None else np.array([])),

            # vectors for offline AURC / correlations
            pred_vec=pred_v,
            gt_vec=gt_v,
            err_abs_vec=err_v,
            sigma_total_vec=sig_total_v,
            sigma_alea_vec=sig_alea_v,
            sigma_epi_vec=sig_epi_v,
            rms_vec=rms_v,
            tissue_labels_vec=tissue_v,
        )

        # ---------- Save JSON metadata ----------
        meta = {
            "subject": subj,
            "slice_num": int(slc),
            "tag": tag,
            "n_valid_pixels": int(err_v.size),
            "has_sigma_total": bool(sigma_total is not None),
            "has_sigma_alea": bool(sigma_alea is not None),
            "has_sigma_epi": bool(sigma_epi is not None),
            "has_rms": bool(rms_raw is not None),
            "te_bg_index": int(te_bg_index),
            "checkpoint_path": str(self.model_path_override) if self.model_path_override else str(self._build_model_path()),
            "uncertainty_mode": str(self.uncertainty_mode),
            "heteroscedastic": bool(self.heteroscedastic),
            "use_mc_dropout": bool(self.use_mc_dropout),
            "mc_dropout_samples": int(self.mc_dropout_samples) if self.use_mc_dropout else 0,
        }
        json_path = os.path.join(cache_dir, f"{tag}.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Cached single slice: {npz_path}")


    def _is_valid_slice(self, batch):
        brain_mask = batch['brain_mask'].squeeze().cpu().numpy().astype(bool)
        gt_t2 = batch['gt_map'].squeeze().cpu().numpy()
        # gt_s0 = batch['s0_map'].squeeze().cpu().numpy()
        # return np.any(brain_mask & (gt_t2 != 0) & (gt_s0 != 0))
        return np.any(brain_mask & (gt_t2 != 0))

    def _process_batch(self, batch, output_path):
        data = self._extract_batch_data(batch)

        input_pixels = self._prepare_input(data)
        outputs, uncertainties = self._forward_pass(input_pixels)

        predictions = self._process_predictions(outputs, data, uncertainties)
        self._collect_pixel_level_data(predictions, data, batch)

        metrics = self._compute_metrics(predictions, data)
        self._append_per_slice_records(metrics, data, predictions)

        if self.to_plot:
            dbg_dir = os.path.join(output_path, "debug_slice_maps")
            os.makedirs(dbg_dir, exist_ok=True)
            self._plot_slice_debug_maps(predictions, data, batch, dbg_dir, vmax_t2=self.max_t2)

        if self._safe_int(data["slice_num"], default=-1) in FIGURE_SLICES:
            self._save_single_slice_cache(predictions, data, batch, output_path)

        return metrics

    def _extract_batch_data(self, batch):
        """Extract batch data (include signal_measurements for RMS normalization)."""
        sig = batch.get('signal_measurements', None)
        sig_np = sig.squeeze().cpu().numpy() if sig is not None else None

        # Extract RMS map (needed for cross-stage consistency in per-slice probabilistic metrics)
        rms_t = batch.get('rms', None)
        rms_map = rms_t.squeeze().cpu().numpy() if rms_t is not None else None

        return {
            'input_t2': batch['input_pixels_t2'].squeeze(),
            'gt_t2': batch['gt_map'].squeeze().cpu().numpy(),
            'recon_t2': batch['recon_signal_t2'].squeeze().cpu().numpy(),

            'brain_mask': batch['brain_mask'].squeeze().cpu().numpy().astype(bool),
            'wm_mask': batch['wm_mask'].squeeze().cpu().numpy().astype(bool),
            'gm_mask': batch['gm_mask'].squeeze().cpu().numpy().astype(bool),
            'csf_mask': batch['csf_mask'].squeeze().cpu().numpy().astype(bool),

            'signal_measurements': sig_np,   # (H,W,TE) magnitude mean (UN-normalized)
            'rms_map': rms_map,              # ✅ ADD THIS (your _compute_probabilistic_metrics expects it)

            'subject': batch['subject'][0],
            'slice_num': batch['slice_num'][0],
        }

    def _prepare_input(self, data):
        return data['input_t2'].to(self.device).float()

    def _forward_pass(self, input_pixels):
        """Predict T2* for one slice, plus whatever uncertainty the arm supports.

        The four arms split the total uncertainty differently:

            aleatoric   from the heteroscedastic head, sqrt(E[exp(logvar)])
            epistemic   spread across MC-dropout draws, std(mu)
            total       both in quadrature, or whichever single one exists

        Uncertainties come out in milliseconds: the network predicts on the
        normalised T2*/max_t2 scale, so every term is scaled back by max_t2.
        """
        uncertainties = {}

        with torch.no_grad():
            if self.use_mc_dropout and self.heteroscedastic:      # CUPA, Het, LR
                self.model.train()
                draws = [self.model(input_pixels) for _ in range(self.mc_dropout_samples)]
                self.model.eval()

                mc_means = torch.stack([d[0] for d in draws])
                mc_logvars = torch.stack([d[1] for d in draws])

                uncertainties['aleatoric'] = torch.sqrt(
                    torch.exp(mc_logvars).mean(dim=0)) * self.max_t2
                uncertainties['epistemic'] = mc_means.std(dim=0) * self.max_t2
                uncertainties['total'] = torch.sqrt(uncertainties['aleatoric'] ** 2
                                                    + uncertainties['epistemic'] ** 2)
                outputs = mc_means.mean(dim=0)

            elif self.use_mc_dropout:                             # epistemic only
                self.model.train()
                mc_preds = torch.stack(
                    [self.model(input_pixels) for _ in range(self.mc_dropout_samples)])
                self.model.eval()

                uncertainties['epistemic'] = mc_preds.std(dim=0) * self.max_t2
                uncertainties['total'] = uncertainties['epistemic']
                outputs = mc_preds.mean(dim=0)

            elif self.heteroscedastic:                            # aleatoric only
                mean_pred, logvar_pred = self.model(input_pixels)
                logvar_pred = torch.clamp(logvar_pred, min=-10.0, max=10.0)

                uncertainties['aleatoric'] = torch.sqrt(torch.exp(logvar_pred)) * self.max_t2
                uncertainties['total'] = uncertainties['aleatoric']
                outputs = mean_pred

            else:                                                 # PUQ: point estimate
                outputs = self.model(input_pixels)

        return outputs, uncertainties


    def _plot_calibration_mae_rmse(self, data, output_dir, n_bins=10):
        if 'total_uncertainty' not in data or len(data['total_uncertainty']) == 0:
            print("Skipping MAE/RMSE calibration (no uncertainty data)")
            return

        y = data['ground_truths'].astype(np.float64)
        mu = data['predictions'].astype(np.float64)
        s = data['total_uncertainty'].astype(np.float64)
        labels = data.get('tissue_labels', np.full_like(y, -1))

        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(s) & (s > 0)
        y, mu, s, labels = y[valid], mu[valid], s[valid], labels[valid]
        e = y - mu
        ae = np.abs(e)

        groups = [("overall", labels >= 0), ("wm", labels == 0), ("gm", labels == 1), ("csf", labels == 2)]

        fig, axes = plt.subplots(len(groups), 2, figsize=(10, 3.2 * len(groups)))

        for gi, (name, m) in enumerate(groups):
            ax_mae = axes[gi, 0]
            ax_rmse = axes[gi, 1]

            if not np.any(m):
                ax_mae.axis("off"); ax_rmse.axis("off")
                continue

            ss = s[m]
            aee = ae[m]
            ee = e[m]

            edges = np.percentile(ss, np.linspace(0, 100, n_bins + 1))

            xs, mae_bin, mae_std, rmse_bin = [], [], [], []
            for bi in range(n_bins):
                mm = (ss >= edges[bi]) & (ss < edges[bi + 1])
                if bi == n_bins - 1:
                    mm |= (ss == edges[bi + 1])
                if not np.any(mm):
                    continue

                xs.append(float(np.mean(ss[mm])))
                mae_bin.append(float(np.mean(aee[mm])))
                mae_std.append(float(np.std(aee[mm])))
                rmse_bin.append(float(np.sqrt(np.mean(ee[mm] ** 2))))

            xs = np.array(xs)
            mae_bin = np.array(mae_bin)
            mae_std = np.array(mae_std)
            rmse_bin = np.array(rmse_bin)

            # MAE calibration: E|Z|=sqrt(2/pi)=0.798... so ideal MAE ≈ 0.798*σ
            ax_mae.errorbar(xs, mae_bin, yerr=mae_std, fmt="o-", capsize=4, linewidth=2, label="Actual MAE (mean±std)")
            mx = max(xs.max(), mae_bin.max()) if len(xs) else 1.0
            ax_mae.plot([0, mx], [0, 0.7978845608 * mx], "k--", linewidth=2, label="Ideal: MAE=0.798·σ")
            ax_mae.set_title(f"{name.upper()}: MAE calibration")
            ax_mae.set_xlabel("Predicted σ [ms]")
            ax_mae.set_ylabel("Actual |error| (MAE) [ms]")
            ax_mae.grid(alpha=0.3)
            ax_mae.legend(fontsize=9)

            # RMSE calibration: ideal RMSE ≈ σ
            ax_rmse.plot(xs, rmse_bin, "o-", linewidth=2, label="Actual RMSE")
            mx2 = max(xs.max(), rmse_bin.max()) if len(xs) else 1.0
            ax_rmse.plot([0, mx2], [0, mx2], "k--", linewidth=2, label="Ideal: RMSE=σ")
            ax_rmse.set_title(f"{name.upper()}: RMSE calibration")
            ax_rmse.set_xlabel("Predicted σ [ms]")
            ax_rmse.set_ylabel("Actual RMSE [ms]")
            ax_rmse.grid(alpha=0.3)
            ax_rmse.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "calibration_mae_rmse_overall_and_tissue.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Saved: calibration_mae_rmse_overall_and_tissue.png")



    def _collect_pixel_level_data(self, predictions, data, batch):
        # valid_mask = data['brain_mask'] & (predictions['gt_t2'] != 0) & (predictions['gt_s0'] != 0)
        valid_mask = data['brain_mask'] & (predictions['gt_t2'] != 0)
        if not np.any(valid_mask):
            return

        pred_vec = predictions['pred_t2'][valid_mask].astype(np.float64).ravel()
        gt_vec   = predictions['gt_t2'][valid_mask].astype(np.float64).ravel()
        e        = (gt_vec - pred_vec)  # IMPORTANT: y - mu convention
        ae       = np.abs(e)

        n = pred_vec.size

        self.pixel_level_data['predictions'].extend(pred_vec.tolist())
        self.pixel_level_data['ground_truths'].extend(gt_vec.tolist())
        self.pixel_level_data['errors'].extend(ae.tolist())  # keep your old key used elsewhere
        # optional if you want signed later:
        if 'errors_signed' in self.pixel_level_data:
            self.pixel_level_data['errors_signed'].extend(e.tolist())

        # --- RMS (raw + normalized) ---
        rms_t = batch.get('rms', None)
        if rms_t is not None:
            rms_map = rms_t.squeeze().cpu().numpy().astype(np.float64)
            rms_vec = rms_map[valid_mask].ravel()
        else:
            rms_vec = np.full(n, np.nan, dtype=np.float64)

        self.pixel_level_data['input_rms'].extend(rms_vec.tolist())

        self.pixel_level_data['input_rms_norm'].extend(rms_vec.tolist())

        # --- uncertainties (store aligned; NaNs if missing) ---
        def _vec_or_nan(key):
            if key in predictions:
                m = predictions[key][valid_mask].astype(np.float64).ravel()
                return m
            return np.full(n, np.nan, dtype=np.float64)

        self.pixel_level_data['aleatoric'].extend(_vec_or_nan('uncertainty_aleatoric').tolist())
        self.pixel_level_data['epistemic'].extend(_vec_or_nan('uncertainty_epistemic').tolist())
        self.pixel_level_data['total_uncertainty'].extend(_vec_or_nan('uncertainty_total').tolist())

        # --- tissue labels aligned ---
        tissue_map = np.full_like(data['brain_mask'], fill_value=-1, dtype=int)
        tissue_map[data['wm_mask']] = 0
        tissue_map[data['gm_mask']] = 1
        tissue_map[data['csf_mask']] = 2
        tvec = tissue_map[valid_mask].ravel()
        self.pixel_level_data['tissue_labels'].extend(tvec.tolist())



    def _mask_from_tissue(self, labels: np.ndarray, tissue: str):
        if tissue == "overall":
            return labels >= 0
        mapping = {"wm": 0, "gm": 1, "csf": 2}
        return labels == mapping[tissue]


    @staticmethod
    def _gaussian_nll(y, mu, sigma, eps=1e-9):
        # average NLL; includes constant 0.5*log(2π)
        s = np.maximum(sigma, eps)
        z2 = ((y - mu) / s) ** 2
        return 0.5 * np.log(2 * np.pi * (s ** 2)) + 0.5 * z2


    @staticmethod
    def _gaussian_crps(y, mu, sigma, eps=1e-9):
        # CRPS for Gaussian: σ[ z(2Φ(z)-1) + 2φ(z) - 1/√π ], z=(y-μ)/σ
        s = np.maximum(sigma, eps)
        z = (y - mu) / s
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
        return s * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


    def _proper_scoring_and_residuals(self, data, output_dir):
        """
        Computes:
        - Gaussian NLL, Gaussian CRPS
        - standardized residuals z=(y-mu)/σ : mean/var + hist + QQ
        - ratio test E[e^2/σ^2]
        - corr(|e|, σ)
        Saved overall + per-tissue.
        """
        labels = data.get("tissue_labels", None)
        if labels is None or len(labels) == 0:
            print("Skipping proper scoring rules (no tissue labels)")
            return

        y = data["ground_truths"].astype(np.float64)
        mu = data["predictions"].astype(np.float64)
        sigma = data.get("total_uncertainty", None)
        if sigma is None or len(sigma) == 0:
            print("Skipping proper scoring rules (no total_uncertainty)")
            return
        sigma = sigma.astype(np.float64)

        # valid: finite + sigma>0
        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
        y, mu, sigma, labels = y[valid], mu[valid], sigma[valid], labels[valid]

        import pandas as pd
        rows = []
        for tissue in ["overall", "wm", "gm", "csf"]:
            m = self._mask_from_tissue(labels, tissue)
            if not np.any(m):
                continue

            yy, mm, ss = y[m], mu[m], sigma[m]
            e = yy - mm
            z = e / (ss + 1e-12)

            nll = self._gaussian_nll(yy, mm, ss).mean()
            crps = self._gaussian_crps(yy, mm, ss).mean()

            ratio = np.mean((e ** 2) / (ss ** 2 + 1e-12))
            corr_abs = np.corrcoef(np.abs(e), ss)[0, 1] if len(e) > 2 else np.nan

            rows.append({
                "Category": tissue.upper(),
                "N": int(len(yy)),
                "NLL": float(nll),
                "CRPS": float(crps),
                "z_mean": float(np.mean(z)),
                "z_var": float(np.var(z)),
                "E[e^2/s^2]": float(ratio),
                "corr(|e|,s)": float(corr_abs),
                "sharpness_mean_sigma": float(np.mean(ss)),
            })

            # plots for standardized residuals
            self._plot_standardized_residuals(z, output_dir, tag=tissue)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "proper_scoring_and_residuals.csv"), index=False)
        with open(os.path.join(output_dir, "proper_scoring_and_residuals.md"), "w") as f:
            f.write(df.to_markdown(index=False))

        print("✓ Saved: proper_scoring_and_residuals.csv/md")


    def _plot_standardized_residuals(self, z, output_dir, tag="overall"):
        z = z[np.isfinite(z)]
        if len(z) < 100:
            return

        # Histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(z, bins=80)
        ax.set_title(f"{tag.upper()}: Standardized residuals z=(y-μ)/σ\nmean={z.mean():.3f}, var={z.var():.3f}")
        ax.set_xlabel("z")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"z_hist_{tag}.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # Q-Q
        fig, ax = plt.subplots(figsize=(6, 6))
        probplot(z, dist="norm", plot=ax)
        ax.set_title(f"{tag.upper()}: Q–Q plot of z vs N(0,1)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"z_qq_{tag}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def _process_predictions(self, outputs, data, uncertainties):
        """Reshape network output back to the slice grid and undo the T2* scaling."""
        predictions = {
            'gt_t2': data['gt_t2'],
            'recon_t2': data['recon_t2'],
        }

        pred = outputs.squeeze().cpu().numpy().reshape(data['gt_t2'].shape)
        if getattr(self.test_dataset, "use_log_transform", False):
            pred = self.test_dataset.inverse_transform_t2(pred)
            predictions['pred_t2'] = np.clip(pred, self.min_t2, self.max_t2)
        else:
            predictions['pred_t2'] = np.clip(pred * self.max_t2, self.min_t2, self.max_t2)

        for key, val in uncertainties.items():
            predictions[f'uncertainty_{key}'] = \
                val.squeeze().cpu().numpy().reshape(data['gt_t2'].shape)

        return predictions

    # ==================== METRICS ====================


    def _compute_tissue_metrics(self, predictions, data):
        metrics = {}
        for tissue, tmask in [('wm', data['wm_mask']), ('gm', data['gm_mask']), ('csf', data['csf_mask'])]:
            combined = data['brain_mask'] & tmask & (predictions['gt_t2'] != 0)
            if np.any(combined):
                # Standard metrics
                metrics[f'{tissue}_t2'] = self._compute_slice_metrics(predictions['pred_t2'], predictions['gt_t2'], combined)
                metrics[f'{tissue}_recon_t2'] = self._compute_slice_metrics(predictions['recon_t2'], predictions['gt_t2'], combined)
                
                # NEW: Probabilistic metrics
                metrics[f'{tissue}_t2_probabilistic'] = self._compute_probabilistic_metrics(predictions, data, combined)
        
        return metrics


    def _compute_metrics(self, predictions, data):
        metrics = {}
        # valid_mask = data['brain_mask'] & (predictions['gt_t2'] != 0) & (predictions['gt_s0'] != 0)
        valid_mask = data['brain_mask'] & (predictions['gt_t2'] != 0)

        if 'pred_t2' in predictions:
            metrics['t2'] = self._compute_slice_metrics(predictions['pred_t2'], predictions['gt_t2'], valid_mask)
            metrics['recon_t2'] = self._compute_slice_metrics(predictions['recon_t2'], predictions['gt_t2'], valid_mask)
            metrics['t2_probabilistic'] = self._compute_probabilistic_metrics(predictions, data, valid_mask)


        metrics.update(self._compute_tissue_metrics(predictions, data))
        return metrics

    def _compute_slice_metrics(self, pred, gt, mask):
        """Compute metrics for a slice.

        - NRMSE: sqrt( mean((pred-gt)^2) / mean(gt^2) ) over masked pixels
        - SSIM/PSNR: computed on bbox crop (default) or full image, using data_range=max-min
        """
        if np.sum(mask) == 0:
            return {'nrmse': np.inf, 'mse': np.inf, 'mae': np.inf, 'ssim': 0.0, 'psnr': 0.0}

        pred_m, gt_m = pred[mask], gt[mask]
        mse = float(np.mean((pred_m - gt_m) ** 2))
        norms = float(np.mean(gt_m ** 2))
        nrmse = float(np.sqrt(mse / (norms + 1e-12)))
        mae = float(np.mean(np.abs(pred_m - gt_m)))

        # For SSIM/PSNR: zero-out background, then crop bbox if requested
        pred0 = pred * mask
        gt0 = gt * mask

        if self.ssim_mode == 'full':
            gt_r, pred_r = gt0, pred0
        else:
            bb = self._bbox_from_mask(mask)
            if bb is None:
                return {'nrmse': nrmse, 'mse': mse, 'mae': mae, 'ssim': 0.0, 'psnr': 0.0}
            ys, xs = bb
            gt_r, pred_r = gt0[ys, xs], pred0[ys, xs]

        dr = float(gt_r.max() - gt_r.min())
        if dr < 1e-12:
            ssim_val = 0.0
            psnr_val = np.inf if mse == 0 else 0.0
        else:
            win = self._safe_ssim_win_size(gt_r.shape[0], gt_r.shape[1], preferred=11)
            ssim_val = float(ssim(gt_r, pred_r, data_range=dr, win_size=win))
            psnr_val = float(10 * np.log10((dr ** 2) / (mse + 1e-12)))

        return {'nrmse': nrmse, 'mse': mse, 'mae': mae, 'ssim': ssim_val, 'psnr': psnr_val}

    def _accumulate_metrics(self, results, metrics_by_type):
        for key, value in results.items():
            if isinstance(value, dict):
                for metric_name, metric_val in value.items():
                    metrics_by_type[key][metric_name].append(metric_val)


    def _calibration_by_bins_and_ence(self, data, output_dir, n_bins=10):
        sigma = data.get("total_uncertainty", None)
        if sigma is None or len(sigma) == 0:
            print("Skipping ENCE (no total_uncertainty)")
            return

        y = data["ground_truths"].astype(np.float64)
        mu = data["predictions"].astype(np.float64)
        s = sigma.astype(np.float64)
        labels = data.get("tissue_labels", np.full_like(y, -1))

        valid = np.isfinite(y) & np.isfinite(mu) & np.isfinite(s) & (s > 0)
        y, mu, s, labels = y[valid], mu[valid], s[valid], labels[valid]

        rows = []

        for tissue in ["overall", "wm", "gm", "csf"]:
            m = self._mask_from_tissue(labels, tissue)
            if not np.any(m):
                continue

            ss = s[m]
            e = (y[m] - mu[m])

            # bin by sigma
            edges = np.percentile(ss, np.linspace(0, 100, n_bins + 1))
            ence_terms = []
            for bi in range(n_bins):
                mm = (ss >= edges[bi]) & (ss < edges[bi + 1])
                if bi == n_bins - 1:
                    mm |= (ss == edges[bi + 1])
                if not np.any(mm):
                    continue

                mean_sigma = float(np.mean(ss[mm]))
                rmse = float(np.sqrt(np.mean(e[mm] ** 2)))
                rel_gap = abs(rmse - mean_sigma) / (mean_sigma + 1e-12)
                ence_terms.append(rel_gap)

                rows.append({
                    "Category": tissue.upper(),
                    "Bin": bi + 1,
                    "Sigma_range": f"{edges[bi]:.3f}-{edges[bi+1]:.3f}",
                    "Mean_sigma": mean_sigma,
                    "RMSE": rmse,
                    "Rel_gap": float(rel_gap),
                    "N": int(np.sum(mm)),
                })

            ENCE = float(np.mean(ence_terms)) if len(ence_terms) else np.nan

            # plot RMSE(bin) vs mean sigma(bin)
            trows = [r for r in rows if r["Category"] == tissue.upper()]
            if len(trows) >= 3:
                xs = np.array([r["Mean_sigma"] for r in trows])
                ys = np.array([r["RMSE"] for r in trows])

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(xs, ys, "o-")
                mx = max(xs.max(), ys.max())
                ax.plot([0, mx], [0, mx], "k--", linewidth=2)
                ax.set_xlabel("Mean predicted σ (bin) [ms]")
                ax.set_ylabel("Empirical RMSE (bin) [ms]")
                ax.set_title(f"{tissue.upper()}: Bin calibration (ENCE={ENCE:.3f})")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"bin_calibration_{tissue}.png"), dpi=300, bbox_inches="tight")
                plt.close()

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, "bin_calibration_and_ence.csv"), index=False)
        with open(os.path.join(output_dir, "bin_calibration_and_ence.md"), "w") as f:
            f.write(df.to_markdown(index=False))

        print("✓ Saved: bin_calibration_and_ence.csv/md + per-tissue bin_calibration_*.png")


    def _plot_slice_debug_maps(self, predictions, data, batch, out_dir, vmax_t2=200.0):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ---------------------------
        # Settings you asked for
        # ---------------------------
        te_bg_index = 3          # "fourth TE" (0-based)
        transpose_plot = True    # transpose brains for plotting
        origin = "lower"         # consistent orientation

        # error scaling: robust clipping to avoid CSF dominating the colormap
        err_p_hi = 99.0          # try 99.0 or 99.5
        err_use_log = False      # set True if still too dominated / you want more detail

        overlay_alpha = 0.55
        overlay_p_hi = 99.0

        # ---------------------------
        # Helpers
        # ---------------------------
        def _as_nan_masked(img2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
            out = img2d.astype(np.float32, copy=True)
            out[~mask2d] = np.nan
            return out

        def _robust_vmin_vmax(arr2d: np.ndarray, mask2d: np.ndarray, p_lo=2, p_hi=98):
            vals = arr2d[mask2d].reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return 0.0, 1.0
            return float(np.percentile(vals, p_lo)), float(np.percentile(vals, p_hi))

        def _overlay_panel(ax, bg_img_masked, heat, title, mask2d, vmin_h=0.0, vmax_h=None, cmap=None):
            ax.imshow(bg_img_masked, cmap="gray", vmin=vmin_bg, vmax=vmax_bg,
                    interpolation="nearest", origin=origin)

            if heat is None:
                ax.set_title(title); ax.axis("off"); return

            heat = heat.astype(np.float32, copy=False)
            heat_m = _as_nan_masked(heat, mask2d)

            if vmax_h is None:
                hv = heat[mask2d]
                hv = hv[np.isfinite(hv)]
                vmax_h = float(np.percentile(hv, overlay_p_hi)) if hv.size else 1.0

            im = ax.imshow(heat_m, alpha=overlay_alpha, vmin=vmin_h, vmax=vmax_h,
                        cmap=cmap, interpolation="nearest", origin=origin)
            ax.set_title(title)
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        def _show_map(ax, img, title, mask2d, vmin=None, vmax=None, cmap=None):
            if img is None:
                ax.axis("off"); return
            img_m = _as_nan_masked(img, mask2d)
            im = ax.imshow(img_m, vmin=vmin, vmax=vmax, cmap=cmap,
                        interpolation="nearest", origin=origin)
            ax.set_title(title)
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        def _maybe_T(x):
            return x.T if (transpose_plot and x is not None) else x

        # ---------------------------
        # Core maps
        # ---------------------------
        pred = predictions.get("pred_t2", None)
        gt   = predictions.get("gt_t2", None)
        if pred is None or gt is None:
            return

        mask = data["brain_mask"].astype(bool)
        err  = np.abs(gt - pred)

        tot  = predictions.get("uncertainty_total", None)
        alea = predictions.get("uncertainty_aleatoric", None)
        epi  = predictions.get("uncertainty_epistemic", None)

        rms_t = batch.get("rms", None)
        rms_norm = (rms_t.squeeze().cpu().numpy().astype(np.float64)
                    if rms_t is not None else None)

        # ---------------------------
        # Background TE image: take TE index 3 (4th echo)
        # ---------------------------
        sig = data.get("signal_measurements", None)
        if sig is None or not isinstance(sig, np.ndarray) or sig.ndim != 3 or sig.shape[-1] <= te_bg_index:
            # fallback: use GT as background if no signal
            bg = gt.astype(np.float32)
            bg_title = "GT T2* (bg)"
            bg_disp = bg
        else:
            bg = sig[..., te_bg_index].astype(np.float32)
            bg_title = f"Signal magnitude @ TE index {te_bg_index}"
            bg_disp = np.log1p(np.maximum(bg, 0.0))  # match your overlay style

        # ---------------------------
        # Transpose for plotting (ALL maps + mask + bg)
        # ---------------------------
        mask = _maybe_T(mask)
        gt   = _maybe_T(gt)
        pred = _maybe_T(pred)
        err  = _maybe_T(err)
        bg_disp = _maybe_T(bg_disp)

        tot  = _maybe_T(tot)  if tot  is not None else None
        alea = _maybe_T(alea) if alea is not None else None
        epi  = _maybe_T(epi)  if epi  is not None else None
        rms_norm = _maybe_T(rms_norm) if rms_norm is not None else None

        # ---------------------------
        # Background scaling
        # ---------------------------
        vmin_bg, vmax_bg = _robust_vmin_vmax(bg_disp, mask, p_lo=2, p_hi=98)
        bg_masked = _as_nan_masked(bg_disp, mask)

        # ---------------------------
        # Error scaling to mitigate CSF dominance
        #   - compute vmax using a scale-mask that can exclude CSF if available
        # ---------------------------
        scale_mask = mask.copy()

        # If you have a CSF mask available in `data`, exclude it for scaling (optional but great)
        csf_mask = data.get("csf_mask", None)  # <-- if you store it
        if csf_mask is not None:
            csf_mask = csf_mask.astype(bool)
            csf_mask = _maybe_T(csf_mask)
            scale_mask = scale_mask & (~csf_mask)

        err_for_scale = err[scale_mask]
        err_for_scale = err_for_scale[np.isfinite(err_for_scale)]
        if err_for_scale.size > 0:
            err_vmax = float(np.percentile(err_for_scale, err_p_hi))
        else:
            err_vmax = float(np.nanmax(err)) if np.isfinite(err).any() else 1.0

        if err_use_log:
            err_disp = np.log1p(np.maximum(err, 0.0))
            err_title = "|Error| [ms] (log1p)"
            # scale log-space robustly
            vals = err_disp[scale_mask]
            vals = vals[np.isfinite(vals)]
            err_vmax_disp = float(np.percentile(vals, err_p_hi)) if vals.size else 1.0
            err_vmin_disp = 0.0
        else:
            err_disp = np.clip(err, 0.0, err_vmax)
            err_title = f"|Error| [ms] (clipped @ p{err_p_hi:.1f})"
            err_vmin_disp, err_vmax_disp = 0.0, err_vmax

        # ---------------------------
        # Plot: 2 rows × 4 cols (your desired order)
        # ---------------------------
        fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
        ax = axes.ravel()

        # Top row: TE image, GT, Pred, Error
        _show_map(ax[0], bg_disp, bg_title, mask, vmin=vmin_bg, vmax=vmax_bg, cmap="gray")
        _show_map(ax[1], gt,     "GT T2* [ms]",   mask, vmin=0, vmax=vmax_t2, cmap=self.t2_cmap)
        _show_map(ax[2], pred,   "Pred T2* [ms]", mask, vmin=0, vmax=vmax_t2, cmap=self.t2_cmap)
        _show_map(ax[3], err_disp, err_title,     mask, vmin=err_vmin_disp, vmax=err_vmax_disp, cmap=None)

        # Bottom row: overlays (same as before, now transposed + with TE4 bg)
        _overlay_panel(ax[4], bg_masked, tot,      "Total σ overlay",     mask)
        _overlay_panel(ax[5], bg_masked, alea,     "Aleatoric σ overlay", mask)
        _overlay_panel(ax[6], bg_masked, epi,      "Epistemic σ overlay", mask)

        rms_title = "RMS"
        _overlay_panel(ax[7], bg_masked, rms_norm, f"{rms_title} overlay", mask)

        fn = f"{data['subject']}_slice_{int(data['slice_num']):03d}_debug_maps.png"
        plt.savefig(os.path.join(out_dir, fn), dpi=300, bbox_inches="tight")
        plt.close(fig)



    def _save_pixel_level_cache(self, data, unc_dir):
        meta = self._run_metadata()
        meta["ssim_mode"] = str(self.ssim_mode)

        out = os.path.join(unc_dir, "pixel_level_cache.npz")
        np.savez_compressed(
            out,
            **{k: np.asarray(v) for k, v in data.items()},
            meta_json=json.dumps(meta),
        )
        print(f"✓ Saved pixel-level cache: {out}")


    # ==================== UNCERTAINTY ANALYSIS (RMS) ====================

    def _perform_uncertainty_analysis(self, output_path):
        print("\n" + "=" * 80)
        print("PERFORMING UNCERTAINTY ANALYSIS (RMS)")
        print("=" * 80)

        unc_dir = os.path.join(output_path, 'uncertainty_analysis')
        os.makedirs(unc_dir, exist_ok=True)

        data = {k: np.array(v) for k, v in self.pixel_level_data.items() if len(v) > 0}

        self._save_pixel_level_cache(data, unc_dir)

        # (1) Quartiles: overall + tissue rows
        self._analyze_uncertainty_sources_rms(data, unc_dir)

        # (2) RMS vs uncertainty: overall + tissue rows
        self._plot_rms_vs_uncertainty(data, unc_dir)

        # (3) Calibration: keep MAE plot, add RMSE plot (overall + tissue rows)
        self._plot_calibration(data, unc_dir)

        # (4) Decomposition: already tissue-centric, keep it
        self._plot_uncertainty_decomposition(data, unc_dir)

        # (5) Coverage: overall + tissue rows
        self._plot_coverage(data, unc_dir)

        # (6) Stats table: overall + tissue
        self._save_uncertainty_stats_rms(data, unc_dir)

        self._plot_calibration_mae_rmse(data, unc_dir, n_bins=10)
        self._proper_scoring_and_residuals(data, unc_dir)
        self._calibration_by_bins_and_ence(data, unc_dir, n_bins=10)
        self._plot_aurc_overall(data, unc_dir)
        self._plot_aurc_tissues_one_plot(data, unc_dir)
        self._plot_aurc_rank_by_rms(data, unc_dir)
        self._plot_aurc_rank_by_components_per_tissue(data, unc_dir)
        
        self._plot_cross_stage_consistency(data, unc_dir)

        print(f"✓ Uncertainty analysis complete! Saved to {unc_dir}\n")

    def _analyze_uncertainty_sources_rms(self, data, output_dir):
        """Quartile analysis: RMS (normalized) vs predicted uncertainty, and Pred T2* vs uncertainty.
        Now: creates a 4x2 figure: rows=Overall/WM/GM/CSF, cols=(RMS quartiles, T2* quartiles).
        """
        import pandas as pd

        if 'total_uncertainty' not in data:
            logging.warning("No uncertainty data available for source analysis")
            return

        unc = data['total_uncertainty'].astype(np.float64)
        pred = data.get('predictions', np.array([])).astype(np.float64)
        rms = data.get('input_rms_norm', np.array([])).astype(np.float64)

        n = len(unc)
        if n == 0:
            return

        # base finite mask
        base = np.isfinite(unc)
        if pred.size == n:
            base = base & np.isfinite(pred)
        if rms.size == n:
            base = base & np.isfinite(rms) & (rms >= 0)

        tissue_masks = self._tissue_masks_1d(data, base_mask=base)

        def quartile_rows(x, y, source_name):
            rows = []
            if x.size == 0:
                return rows
            # robust percentiles
            qs = np.array([0, 25, 50, 75, 100], dtype=float)
            bins = np.percentile(x, qs)
            for i in range(len(bins) - 1):
                m = (x >= bins[i]) & (x < bins[i + 1])
                if i == len(bins) - 2:
                    m |= (x == bins[i + 1])
                if np.any(m):
                    rows.append({
                        'Source': source_name,
                        'Quartile': f'Q{i+1}',
                        'Range': f'{bins[i]:.5f}-{bins[i+1]:.5f}' if source_name.startswith('Input') else f'{bins[i]:.1f}-{bins[i+1]:.1f} ms',
                        'Mean_Value': float(x[m].mean()),
                        'Mean_Uncertainty': float(y[m].mean()),
                        'Std_Uncertainty': float(y[m].std()),
                        'N_pixels': int(np.sum(m)),
                    })
            return rows

        all_rows = []

        # Build CSV/MD with tissue column too
        for tname, tm in tissue_masks:
            if np.sum(tm) < 10:
                continue
            if rms.size == n:
                all_rows += [{'Tissue': tname, **r} for r in quartile_rows(rms[tm], unc[tm],
                            'Input RMS')]
            if pred.size == n:
                all_rows += [{'Tissue': tname, **r} for r in quartile_rows(pred[tm], unc[tm], 'Predicted T2*')]

        if len(all_rows) == 0:
            return

        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(output_dir, 'uncertainty_source_analysis_rms.csv'), index=False)
        with open(os.path.join(output_dir, 'uncertainty_source_analysis_rms.md'), 'w') as f:
            f.write("# Uncertainty Source Analysis (RMS)\n\n")
            f.write(df.to_markdown(index=False))
        print("✓ Saved: uncertainty_source_analysis_rms.csv/md")

        # ---- Plot 4x2 ----
        rows = len(tissue_masks)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        if rows == 1:
            axes = np.array([axes])

        for ri, (tname, tm) in enumerate(tissue_masks):
            # RMS quartiles panel
            ax = axes[ri, 0]
            if rms.size == n and np.sum(tm) >= 10:
                x = rms[tm]
                y = unc[tm]
                # guard empty/constant
                if x.size > 0:
                    bins = np.percentile(x, [0, 25, 50, 75, 100])
                    means, stds, labels = [], [], []
                    for i in range(4):
                        m = (x >= bins[i]) & (x < bins[i+1])
                        if i == 3:
                            m |= (x == bins[i+1])
                        if np.any(m):
                            means.append(y[m].mean())
                            stds.append(y[m].std())
                            labels.append(f"Q{i+1}\n{bins[i]:.5f}-{bins[i+1]:.5f}")
                    if len(means) > 0:
                        xi = np.arange(len(means))
                        ax.bar(xi, means, yerr=stds, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)
                        ax.set_xticks(xi)
                        ax.set_xticklabels(labels, fontsize=8)
                        ax.set_ylabel("Mean σ [ms]", fontsize=11)
                        ax.set_title(f"{tname}: σ vs RMS quartiles", fontsize=12, fontweight='bold')
                        ax.grid(axis='y', alpha=0.25)
                        rp, _, rs, _ = self._pearson_spearman(x, y)
                        ax.text(0.98, 0.95, f"Pearson r={rp:.3f}\nSpearman ρ={rs:.3f}",
                                transform=ax.transAxes, ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
                    else:
                        ax.axis('off')
                else:
                    ax.axis('off')
            else:
                ax.axis('off')

            # T2* quartiles panel
            ax = axes[ri, 1]
            if pred.size == n and np.sum(tm) >= 10:
                x = pred[tm]
                y = unc[tm]
                if x.size > 0:
                    bins = np.percentile(x, [0, 25, 50, 75, 100])
                    means, stds, labels = [], [], []
                    for i in range(4):
                        m = (x >= bins[i]) & (x < bins[i+1])
                        if i == 3:
                            m |= (x == bins[i+1])
                        if np.any(m):
                            means.append(y[m].mean())
                            stds.append(y[m].std())
                            labels.append(f"Q{i+1}\n{bins[i]:.1f}-{bins[i+1]:.1f}")
                    if len(means) > 0:
                        xi = np.arange(len(means))
                        ax.bar(xi, means, yerr=stds, capsize=4, alpha=0.85, edgecolor='black', linewidth=0.8)
                        ax.set_xticks(xi)
                        ax.set_xticklabels(labels, fontsize=8)
                        ax.set_ylabel("Mean σ [ms]", fontsize=11)
                        ax.set_title(f"{tname}: σ vs Pred T2* quartiles", fontsize=12, fontweight='bold')
                        ax.grid(axis='y', alpha=0.25)
                        rp, _, rs, _ = self._pearson_spearman(x, y)
                        ax.text(0.98, 0.95, f"Pearson r={rp:.3f}\nSpearman ρ={rs:.3f}",
                                transform=ax.transAxes, ha='right', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
                    else:
                        ax.axis('off')
                else:
                    ax.axis('off')
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_source_quartiles_rms.png'), dpi=250, bbox_inches='tight')
        plt.close()
        print("✓ Saved: uncertainty_source_quartiles_rms.png")

    def _plot_rms_vs_uncertainty(self, data, output_dir):
        """RMS vs predicted uncertainty (aleatoric/epistemic/total) as:
        rows = Overall/WM/GM/CSF, cols = Aleatoric/Epistemic/Total.
        """
        rms = data.get('input_rms_norm', np.array([])).astype(np.float64)
        if len(rms) == 0:
            print("Skipping RMS vs uncertainty plot (no RMS data available)")
            return
        else:
            rms = np.clip(rms, 0, np.percentile(rms, 99.5))  # robust clipping for better visualization

        # which uncertainty types exist
        unc_types = []
        if 'aleatoric' in data and len(data['aleatoric']) > 0:
            unc_types.append(('Aleatoric', data['aleatoric'].astype(np.float64)))
        if 'epistemic' in data and len(data['epistemic']) > 0:
            unc_types.append(('Epistemic', data['epistemic'].astype(np.float64)))
        if 'total_uncertainty' in data and len(data['total_uncertainty']) > 0:
            unc_types.append(('Total', data['total_uncertainty'].astype(np.float64)))

        if len(unc_types) == 0:
            print("Skipping RMS vs uncertainty plot (no uncertainty arrays)")
            return

        n = len(rms)
        base = np.isfinite(rms) & (rms >= 0)
        # constrain also by finiteness of at least total_uncertainty if present
        if 'total_uncertainty' in data and len(data['total_uncertainty']) == n:
            base = base & np.isfinite(data['total_uncertainty'])

        tissue_masks = self._tissue_masks_1d(data, base_mask=base)

        rows = len(tissue_masks)
        cols = 3  # keep 3 columns layout like before
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4.8 * rows))

        if rows == 1:
            axes = np.array([axes])

        for ri, (tname, tm) in enumerate(tissue_masks):
            for ci in range(cols):
                ax = axes[ri, ci]
                if ci >= len(unc_types):
                    ax.axis('off')
                    continue

                name, unc = unc_types[ci]
                m = tm & np.isfinite(unc)
                x = rms[m]
                y = unc[m]

                if x.size < 200:
                    ax.text(0.5, 0.5, f"{tname}\nNot enough points", ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    continue

                hb = ax.hexbin(x, y, gridsize=60, cmap='Blues', mincnt=1)
                plt.colorbar(hb, ax=ax, label='Pixel Count')

                rp, _, rs, _ = self._pearson_spearman(x, y)

                # linear fit (guard constant x)
                try:
                    if np.std(x) > 1e-12:
                        z = np.polyfit(x, y, 1)
                        pfit = np.poly1d(z)
                        xr = np.linspace(np.min(x), np.max(x), 200)
                        ax.plot(xr, pfit(xr), 'r--', linewidth=2.5,
                                label=f'Pearson r={rp:.3f}\nSpearman ρ={rs:.3f}\nSlope={z[0]:.2f}')
                        ax.legend(fontsize=9, loc='upper left')
                    else:
                        ax.text(0.05, 0.95, f"Pearson r={rp:.3f}\nSpearman ρ={rs:.3f}",
                                transform=ax.transAxes, ha='left', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
                except Exception:
                    pass

                ax.set_xlabel('Input RMS',
                              fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted σ [ms]', fontsize=12, fontweight='bold')
                ax.set_title(f'{tname}: {name}', fontsize=13, fontweight='bold')
                ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rms_vs_uncertainty.png'), dpi=250, bbox_inches='tight')
        plt.close()
        print("✓ Saved: rms_vs_uncertainty.png")

    def _calibration_binned(self, sig, err_abs, mask, n_bins=10, min_points=200):
        """Return binned arrays: mean_sigma, mae_mean, mae_std, rmse."""
        sig = sig[mask]
        err = err_abs[mask]
        if sig.size < max(min_points, 10):
            return None

        # percentiles on sigma inside this mask
        edges = np.percentile(sig, np.linspace(0, 100, n_bins + 1))
        # handle duplicates
        edges = np.unique(edges)
        if edges.size < 3:
            return None

        mean_sigma = []
        mae_mean, mae_std = [], []
        rmse = []

        for i in range(edges.size - 1):
            lo, hi = edges[i], edges[i + 1]
            m = (sig >= lo) & (sig < hi)
            if i == edges.size - 2:
                m |= (sig == hi)
            if np.sum(m) < min_points:
                continue

            s_bin = sig[m]
            e_bin = err[m]

            mean_sigma.append(float(np.mean(s_bin)))
            mae_mean.append(float(np.mean(e_bin)))
            mae_std.append(float(np.std(e_bin)))
            rmse.append(float(np.sqrt(np.mean(e_bin ** 2))))

        if len(mean_sigma) < 2:
            return None

        return (np.array(mean_sigma), np.array(mae_mean), np.array(mae_std), np.array(rmse))

    def _plot_calibration(self, data, output_dir):
        """
        FIXED:
        - Keep MAE calibration plot (now with correct "Perfect" reference line: E|e| = sqrt(2/pi)*sigma).
        - Add RMSE calibration plot (perfect line y=x).
        - For BOTH: also generate tissue-specific rows (Overall/WM/GM/CSF).
        Output: calibration_curve.png (4 rows x 2 cols).
        """
        if 'total_uncertainty' not in data or len(data['total_uncertainty']) == 0:
            print("Skipping calibration plot (no uncertainty data)")
            return
        if 'errors' not in data or len(data['errors']) == 0:
            print("Skipping calibration plot (no error data)")
            return

        sig_all = data['total_uncertainty'].astype(np.float64)
        err_abs_all = data['errors'].astype(np.float64)

        n = len(sig_all)
        base = np.isfinite(sig_all) & np.isfinite(err_abs_all) & (sig_all >= 0) & (err_abs_all >= 0)
        tissue_masks = self._tissue_masks_1d(data, base_mask=base)

        rows = len(tissue_masks)
        fig, axes = plt.subplots(rows, 2, figsize=(13, 4.6 * rows))
        if rows == 1:
            axes = np.array([axes])

        # Perfect MAE line for Gaussian residuals:
        # E|e| = sigma * sqrt(2/pi)
        perfect_mae_factor = float(np.sqrt(2.0 / np.pi))

        for ri, (tname, tm) in enumerate(tissue_masks):
            b = self._calibration_binned(
                sig_all, err_abs_all, tm,
                n_bins=self.calibration_bins,
                min_points=self.calibration_min_points_per_bin
            )
            # Left: MAE
            ax_mae = axes[ri, 0]
            ax_rmse = axes[ri, 1]

            if b is None:
                ax_mae.text(0.5, 0.5, f"{tname}\nNot enough points", ha='center', va='center', fontsize=12)
                ax_rmse.text(0.5, 0.5, f"{tname}\nNot enough points", ha='center', va='center', fontsize=12)
                ax_mae.axis('off')
                ax_rmse.axis('off')
                continue

            mean_sigma, mae_mean, mae_std, rmse = b

            # MAE calibration with std(|e|) error bars
            ax_mae.errorbar(mean_sigma, mae_mean, yerr=mae_std, fmt='o-', linewidth=2, capsize=4,
                            label='Actual MAE (mean±std)')
            maxv = float(max(np.max(mean_sigma), np.max(mae_mean)) * 1.05)
            ax_mae.plot([0, maxv], [0, perfect_mae_factor * maxv], 'k--', linewidth=2,
                        label=f'Perfect: y={perfect_mae_factor:.3f}·σ')
            ax_mae.set_xlabel('Predicted uncertainty σ [ms]', fontsize=12)
            ax_mae.set_ylabel('Actual |error| (MAE) [ms]', fontsize=12)
            ax_mae.set_title(f'{tname}: MAE calibration', fontsize=13, fontweight='bold')
            ax_mae.grid(alpha=0.25)
            ax_mae.legend(fontsize=10)
            ax_mae.set_aspect('equal', adjustable='box')

            # RMSE calibration (no error bars by default: much clearer)
            ax_rmse.plot(mean_sigma, rmse, 'o-', linewidth=2, label='Actual RMSE')
            maxv2 = float(max(np.max(mean_sigma), np.max(rmse)) * 1.05)
            ax_rmse.plot([0, maxv2], [0, maxv2], 'k--', linewidth=2, label='Perfect: y=σ')
            ax_rmse.set_xlabel('Predicted uncertainty σ [ms]', fontsize=12)
            ax_rmse.set_ylabel('Actual RMSE [ms]', fontsize=12)
            ax_rmse.set_title(f'{tname}: RMSE calibration', fontsize=13, fontweight='bold')
            ax_rmse.grid(alpha=0.25)
            ax_rmse.legend(fontsize=10)
            ax_rmse.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=250, bbox_inches='tight')
        plt.close()
        print("✓ Saved: calibration_curve.png")

    def _plot_aurc_overall(self, data, output_dir, fname="aurc_overall.png"):
        """Plot overall risk–coverage curve ranked by total_uncertainty."""
        if "errors" not in data or "total_uncertainty" not in data:
            print("Skipping overall AURC plot (missing arrays)")
            return

        errors = np.asarray(data["errors"], np.float64)               # |e|
        score  = np.asarray(data["total_uncertainty"], np.float64)    # σ_total

        valid = np.isfinite(errors) & np.isfinite(score) & (score >= 0) & (errors >= 0)
        if np.sum(valid) < 100:
            print("Skipping overall AURC plot (not enough valid points)")
            return

        aurc, coverage, risks = _compute_aurc(errors[valid], score[valid])

        plt.figure(figsize=(9, 6))
        plt.plot(coverage, risks, linewidth=2.5)
        plt.xlabel("Coverage (fraction retained)", fontsize=12)
        plt.ylabel("Risk (mean absolute error) [ms]", fontsize=12)
        plt.title(f"Risk–Coverage (rank by σ_total)  |  AURC = {aurc:.4f}", fontsize=14)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"✓ Saved: {fname}")


    def _plot_aurc_tissues_one_plot(self, data, output_dir, fname="aurc_tissues.png"):
        """Plot WM/GM/CSF risk–coverage curves (rank by σ_total) in one figure."""
        if "errors" not in data or "total_uncertainty" not in data or "tissue_labels" not in data:
            print("Skipping tissue AURC plot (missing arrays)")
            return

        errors = np.asarray(data["errors"], np.float64)               # |e|
        score  = np.asarray(data["total_uncertainty"], np.float64)    # σ_total
        labels = np.asarray(data["tissue_labels"])

        base = np.isfinite(errors) & np.isfinite(score) & (score >= 0) & (errors >= 0) & (labels >= 0)
        if np.sum(base) < 500:
            print("Skipping tissue AURC plot (not enough valid tissue points)")
            return

        tissues = [("WM", 0), ("GM", 1), ("CSF", 2)]

        plt.figure(figsize=(9.5, 6.5))

        for name, tid in tissues:
            m = base & (labels == tid)
            if np.sum(m) < 200:
                continue
            aurc, cov, risk = _compute_aurc(errors[m], score[m])
            plt.plot(cov, risk, linewidth=2.5, label=f"{name} (AURC={aurc:.4f}, N={int(np.sum(m))})")

        plt.xlabel("Coverage (fraction retained)", fontsize=12)
        plt.ylabel("Risk (mean absolute error) [ms]", fontsize=12)
        plt.title("Risk–Coverage per tissue (rank by σ_total)", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"✓ Saved: {fname}")


    def _plot_aurc_rank_by_rms(self, data, output_dir, fname="aurc_rank_by_rms.png"):
        """Plot overall risk–coverage curve ranked by RMS (instead of predicted σ)."""
        if "errors" not in data or "input_rms_norm" not in data:
            print("Skipping RMS-ranked AURC plot (missing arrays)")
            return

        errors = np.asarray(data["errors"], np.float64)           # |e|
        rms    = np.asarray(data["input_rms_norm"], np.float64)   # RMS score (lower should mean more confident ideally)

        valid = np.isfinite(errors) & np.isfinite(rms) & (rms >= 0) & (errors >= 0)
        if np.sum(valid) < 100:
            print("Skipping RMS-ranked AURC plot (not enough valid points)")
            return

        aurc, cov, risk = _compute_aurc(errors[valid], rms[valid])

        plt.figure(figsize=(9, 6))
        plt.plot(cov, risk, linewidth=2.5)
        plt.xlabel("Coverage (fraction retained)", fontsize=12)
        plt.ylabel("Risk (mean absolute error) [ms]", fontsize=12)
        plt.title(f"Risk–Coverage (rank by RMS)  |  AURC = {aurc:.4f}", fontsize=14)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"✓ Saved: {fname}")




    def _plot_aurc_rank_by_components_per_tissue(
        self,
        data,
        output_dir,
        fname="aurc_rank_by_components_per_tissue.png",
        min_points=200,
    ):
        """
        3-row plot (WM/GM/CSF). In each row: risk–coverage curves when ranking by
        Total σ, Aleatoric σ, Epistemic σ. Legend shows AURC per component.
        """
        if "errors" not in data or "tissue_labels" not in data:
            print("Skipping per-tissue component AURC plot (missing errors/tissue_labels).")
            return

        errors = np.asarray(data["errors"], np.float64)  # |e|
        labels = np.asarray(data["tissue_labels"])

        # Build candidates (only those that exist and align in length)
        candidates = []
        if "total_uncertainty" in data and len(data["total_uncertainty"]) == len(errors):
            candidates.append(("Total σ", np.asarray(data["total_uncertainty"], np.float64)))
        if "aleatoric" in data and len(data["aleatoric"]) == len(errors):
            candidates.append(("Aleatoric σ", np.asarray(data["aleatoric"], np.float64)))
        if "epistemic" in data and len(data["epistemic"]) == len(errors):
            candidates.append(("Epistemic σ", np.asarray(data["epistemic"], np.float64)))

        if len(candidates) == 0:
            print("Skipping per-tissue component AURC plot (no uncertainty arrays).")
            return

        tissues = [("WM", 0), ("GM", 1), ("CSF", 2)]

        fig, axes = plt.subplots(len(tissues), 1, figsize=(10, 4.2 * len(tissues)), sharex=True)
        if len(tissues) == 1:
            axes = [axes]

        for ax, (tname, tid) in zip(axes, tissues):
            tissue_mask = (labels == tid)

            # Base validity for this tissue
            base = tissue_mask & np.isfinite(errors) & (errors >= 0)

            if np.sum(base) < min_points:
                ax.text(0.5, 0.5, f"{tname}: not enough points (N={int(np.sum(base))})",
                        ha="center", va="center", fontsize=12)
                ax.axis("off")
                continue

            for label, score in candidates:
                valid = base & np.isfinite(score) & (score >= 0)
                if np.sum(valid) < min_points:
                    continue

                aurc, cov, risk = _compute_aurc(errors[valid], score[valid])
                ax.plot(cov, risk, linewidth=2.5, label=f"{label} (AURC={aurc:.4f})")

            ax.set_title(f"Risk–Coverage ({tname}): ranking by uncertainty component", fontsize=13)
            ax.set_ylabel("Risk (MAE) [ms]", fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)

        axes[-1].set_xlabel("Coverage (fraction retained)", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close()
        print(f"✓ Saved: {fname}")



    def _plot_cross_stage_consistency(self, data, output_dir):
        """Scatter plot of RMS vs aleatoric uncertainty."""
        rms = np.array(data['input_rms_norm'])
        aleatoric = np.array(data['aleatoric'])
        
        valid = np.isfinite(rms) & np.isfinite(aleatoric)
        rms, aleatoric = rms[valid], aleatoric[valid]
        
        r, p_val = pearsonr(rms, aleatoric)
        
        rms = np.clip(rms, 0, np.percentile(rms, 99.5))
        plt.figure(figsize=(10, 10))
        plt.hexbin(rms, aleatoric, gridsize=50, cmap='Blues', mincnt=1)
        plt.colorbar(label='Count')
        plt.xlabel('Reconstruction RMS Uncertainty', fontsize=12)
        plt.ylabel('Predicted Aleatoric Uncertainty', fontsize=12)
        plt.title(f'Cross-Stage Consistency (r = {r:.4f}, p = {p_val:.2e})', fontsize=14)
        
        # Add diagonal reference line
        max_val = max(rms.max(), aleatoric.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect correlation')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'cross_stage_consistency.png'), dpi=300)
        plt.close()

    def _plot_uncertainty_decomposition(self, data, output_dir):
        """Kept (already tissue-specific)."""
        if 'tissue_labels' not in data:
            print("Skipping uncertainty decomposition (no tissue labels)")
            return

        tissues = ['WM', 'GM', 'CSF']
        rows = []

        for ti, name in enumerate(tissues):
            m = data['tissue_labels'] == ti
            if not np.any(m):
                continue

            row = {'Tissue': name, 'MAE': float(data['errors'][m].mean())}

            if 'aleatoric' in data and len(data['aleatoric']) > 0:
                row['Aleatoric'] = float(data['aleatoric'][m].mean())
            if 'epistemic' in data and len(data['epistemic']) > 0:
                row['Epistemic'] = float(data['epistemic'][m].mean())
            if 'total_uncertainty' in data and len(data['total_uncertainty']) > 0:
                row['Total'] = float(data['total_uncertainty'][m].mean())

            rms = data.get('input_rms_norm', None)
            if rms is not None and len(rms) == len(data['tissue_labels']):
                row['Mean_RMS'] = float(rms[m].mean())

            rows.append(row)

        if not rows:
            return

        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'tissue_uncertainty_summary.csv'), index=False)

        fig, ax = plt.subplots(figsize=(10, 6))

        if 'Aleatoric' in df.columns and 'Epistemic' in df.columns:
            df.plot(x='Tissue', y=['Aleatoric', 'Epistemic'], kind='bar', stacked=True, ax=ax, width=0.6)
            ax.set_ylabel('Uncertainty [ms]')
            ax.set_title('Uncertainty decomposition by tissue')
        elif 'Total' in df.columns:
            df.plot(x='Tissue', y='Total', kind='bar', ax=ax, width=0.6)
            ax.set_ylabel('Uncertainty [ms]')
            ax.set_title('Total uncertainty by tissue')

        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_decomposition.png'), dpi=250, bbox_inches='tight')
        plt.close()
        print("✓ Saved: uncertainty_decomposition.png")
        print("✓ Saved: tissue_uncertainty_summary.csv")

    def _plot_coverage(self, data, output_dir):
        """
        Coverage plot as 4 rows: Overall/WM/GM/CSF.
        Uses abs error, so theoretical Gaussian coverage is still [0.6827, 0.9545, 0.9973].
        """
        if 'total_uncertainty' not in data or len(data['total_uncertainty']) == 0:
            print("Skipping coverage plot")
            return

        sig = data['total_uncertainty'].astype(np.float64)
        err = data['errors'].astype(np.float64)

        n = len(sig)
        base = np.isfinite(sig) & np.isfinite(err) & (sig >= 0) & (err >= 0)
        tissue_masks = self._tissue_masks_1d(data, base_mask=base)

        sigmas = np.array([1, 2, 3], dtype=float)
        theoretical = np.array([0.6827, 0.9545, 0.9973], dtype=float)

        rows = len(tissue_masks)
        fig, axes = plt.subplots(rows, 1, figsize=(8.5, 3.8 * rows))
        if rows == 1:
            axes = np.array([axes])

        for ri, (tname, tm) in enumerate(tissue_masks):
            ax = axes[ri]
            if np.sum(tm) < 200:
                ax.text(0.5, 0.5, f"{tname}\nNot enough points", ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue

            actual = []
            for k in sigmas:
                actual.append(float((err[tm] <= k * sig[tm]).mean()))
            actual = np.array(actual)

            ax.plot(sigmas, actual, 'o-', linewidth=2, markersize=9, label='Actual')
            ax.plot(sigmas, theoretical, 's--', linewidth=2, markersize=7, label='Theoretical (Gaussian)')

            ax.set_xlabel('k (interval = ±kσ)', fontsize=12)
            ax.set_ylabel('Coverage', fontsize=12)
            ax.set_title(f'{tname}: Prediction interval coverage', fontsize=13, fontweight='bold')
            ax.set_xticks(sigmas)
            ax.set_ylim([0, 1.05])
            ax.grid(alpha=0.25)
            ax.legend(fontsize=10)

            for k, cov in zip(sigmas, actual):
                ax.text(k, cov + 0.02, f'{cov * 100:.1f}%', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coverage_analysis.png'), dpi=250, bbox_inches='tight')
        plt.close()
        print("✓ Saved: coverage_analysis.png")

    def _save_uncertainty_stats_rms(self, data, output_dir):
        import pandas as pd

        rows = []
        rms = data.get('input_rms_norm', np.array([])).astype(np.float64)
        labels = data.get('tissue_labels', np.array([]))

        # Overall
        row = {'Category': 'Overall', 'N_pixels': int(len(data.get('errors', [])))}
        if len(rms) > 0:
            row['Mean_RMS'] = float(np.mean(rms))
            row['Std_RMS'] = float(np.std(rms))

        if 'total_uncertainty' in data and len(rms) > 0 and len(data['total_uncertainty']) == len(rms):
            rp, pp, rs, ps = self._pearson_spearman(rms, data['total_uncertainty'])
            row['RMS_vs_Total_Pearson_r'] = float(rp)
            row['RMS_vs_Total_Pearson_p'] = float(pp)
            row['RMS_vs_Total_Spearman_rho'] = float(rs)
            row['RMS_vs_Total_Spearman_p'] = float(ps)

        row['Mean_Error_MAE'] = float(np.mean(data['errors']))
        row['Mean_Error_RMSE'] = float(np.sqrt(np.mean(np.array(data['errors'], dtype=np.float64) ** 2)))
        rows.append(row)

        # Per tissue
        if len(labels) > 0 and len(rms) == len(labels):
            for ti, name in enumerate(['WM', 'GM', 'CSF']):
                m = labels == ti
                if not np.any(m):
                    continue

                rr = {'Category': name, 'N_pixels': int(np.sum(m))}
                rr['Mean_RMS'] = float(np.mean(rms[m]))
                rr['Std_RMS'] = float(np.std(rms[m]))
                rr['Mean_Error_MAE'] = float(np.mean(data['errors'][m]))
                rr['Mean_Error_RMSE'] = float(np.sqrt(np.mean((data['errors'][m].astype(np.float64) ** 2))))

                if 'total_uncertainty' in data and len(data['total_uncertainty']) == len(rms):
                    rp, pp, rs, ps = self._pearson_spearman(rms[m], data['total_uncertainty'][m])
                    rr['RMS_vs_Total_Pearson_r'] = float(rp)
                    rr['RMS_vs_Total_Pearson_p'] = float(pp)
                    rr['RMS_vs_Total_Spearman_rho'] = float(rs)
                    rr['RMS_vs_Total_Spearman_p'] = float(ps)

                rows.append(rr)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'uncertainty_statistics_rms.csv'), index=False)
        with open(os.path.join(output_dir, 'uncertainty_statistics_rms.md'), 'w') as f:
            f.write("# Uncertainty Statistics (RMS)\n\n")
            f.write(df.to_markdown(index=False))

        print("✓ Saved: uncertainty_statistics_rms.csv/md")

    # ==================== SAVE RESULTS ====================

    def _save_results(self, output_path, metrics_by_type):
        summary = self._generate_summary(metrics_by_type)
        with open(os.path.join(output_path, 'test_summary.txt'), 'w') as f:
            f.write(summary)

        metrics_json = {
            k: {f'{m}_mean': float(np.mean(v)) for m, v in vals.items() if len(v) > 0}
            for k, vals in metrics_by_type.items()
        }
        with open(os.path.join(output_path, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=2)

        self._save_per_slice_records(output_path)


        logging.info(f'Results saved to: {output_path}')

    # Coverage a correctly calibrated Gaussian should achieve at 1/2/3 sigma.
    NOMINAL_COVERAGE = {1: 0.6827, 2: 0.9545, 3: 0.9973}

    def _generate_summary(self, metrics):
        """Human-readable report: accuracy, calibration, informativeness, per tissue."""
        lines = ['=' * 80, 'TEST SUMMARY', '=' * 80, '']
        if metrics['t2']['nrmse']:
            lines += self._summary_overall(metrics)
        lines += self._summary_tissues(metrics)
        lines.append('\n' + '=' * 80)
        return '\n'.join(lines)

    def _summary_overall(self, metrics):
        """Whole-brain accuracy, then the four uncertainty-quality sections."""
        prob = metrics['t2_probabilistic']

        lines = ['\n=== OVERALL T2* METRICS ===', "\nStandard Performance Metrics:"]
        for m in ['nrmse', 'mae', 'ssim', 'psnr']:
            vals = metrics['t2'][m]
            lines.append(f'  {m.upper()}: {np.mean(vals):.6f} ± {np.std(vals):.6f}')

        lines += self._summary_section("PROBABILISTIC PREDICTIVE QUALITY")
        if prob['nll']:
            lines.append(f"  Gaussian NLL: {np.mean(prob['nll']):.6f} ± {np.std(prob['nll']):.6f}")

        lines += self._summary_section("UNCERTAINTY CALIBRATION (Empirical Coverage)")
        lines.append("Expected coverage for well-calibrated uncertainty:")
        lines.append("  1σ: 68.27%  |  2σ: 95.45%  |  3σ: 99.73%")
        lines += self._summary_coverage(prob, "aleatoric σ", "coverage_alea")
        lines += self._summary_coverage(prob, "total σ", "coverage_total")

        lines += self._summary_section("INFORMATIVENESS OF UNCERTAINTY ESTIMATES")
        if prob['spearman_err_unc']:
            lines.append(f"  Spearman ρ(|error|, uncertainty): "
                         f"{np.mean(prob['spearman_err_unc']):.4f} ± "
                         f"{np.std(prob['spearman_err_unc']):.4f}")
            lines.append("  (Higher positive correlation = better informativeness)")

        lines += self._summary_section("SELECTIVE PREDICTION PERFORMANCE")
        if prob['aurc']:
            lines.append(f"  AURC (Area Under Risk-Coverage): "
                         f"{np.mean(prob['aurc']):.6f} ± {np.std(prob['aurc']):.6f}")
            lines.append("  (Lower AURC = better selective prediction)")

        lines += self._summary_section("CROSS-STAGE CONSISTENCY")
        if prob['cross_stage_consistency']:
            valid = [x for x in prob['cross_stage_consistency'] if not np.isnan(x)]
            if len(valid) > 0:
                lines.append(f"  Pearson r(RMS, aleatoric): "
                             f"{np.mean(valid):.4f} ± {np.std(valid):.4f}")
                lines.append("  (Higher correlation = better cross-stage consistency)")

        lines.append("\nBaseline Performance:")
        for m in ['nrmse', 'mae', 'ssim', 'psnr']:
            vals = metrics['recon_t2'][m]
            lines.append(f'  {m.upper()}: {np.mean(vals):.6f} ± {np.std(vals):.6f}')

        pred_nrmse = np.mean(metrics['t2']['nrmse'])
        base_nrmse = np.mean(metrics['recon_t2']['nrmse'])
        improvement = ((base_nrmse - pred_nrmse) / (base_nrmse + 1e-12)) * 100
        lines.append(f"\nNRMSE Improvement: {improvement:.2f}%")
        return lines

    @staticmethod
    def _summary_section(title):
        return ["\n" + "=" * 60, title, "=" * 60]

    def _summary_coverage(self, prob_metrics, tag, key_prefix):
        """Empirical coverage at 1/2/3σ, plus the mean gap to nominal (coverage ECE)."""
        lines = [f"\nActual coverage ({tag}):"]
        means = {}

        for k in (1, 2, 3):
            vals = np.asarray(prob_metrics.get(f"{key_prefix}_{k}sigma", []) or [],
                              dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                lines.append(f"  {k}σ: n/a")
                continue
            means[k] = float(np.mean(vals))
            lines.append(f"  {k}σ: {means[k] * 100:.2f}% ± {float(np.std(vals)) * 100:.2f}%")

        if means:
            ece = float(np.mean([abs(m - self.NOMINAL_COVERAGE[k]) for k, m in means.items()]))
            lines.append(f"  Coverage ECE (mean |empirical - nominal| over 1/2/3σ): {ece:.4f}")
        else:
            lines.append("  Coverage ECE: n/a")
        return lines

    def _summary_tissues(self, metrics):
        """Same numbers again, restricted to WM, GM and CSF."""
        lines = ['\n' + '=' * 80, '=== TISSUE-SPECIFIC METRICS ===', '=' * 80]

        for tissue, name in (('wm', 'White Matter'), ('gm', 'Gray Matter'), ('csf', 'CSF')):
            pred_nrmse = metrics[f'{tissue}_t2']['nrmse']
            if not pred_nrmse:
                continue
            base_nrmse = metrics[f'{tissue}_recon_t2']['nrmse']
            pred, base = np.mean(pred_nrmse), np.mean(base_nrmse)

            lines.append(f'\n{name}:')
            lines.append(f'  Prediction NRMSE: {pred:.6f} ± {np.std(pred_nrmse):.6f}')
            lines.append(f'  Baseline NRMSE:   {base:.6f} ± {np.std(base_nrmse):.6f}')
            lines.append(f'  Improvement:      {((base - pred) / (base + 1e-12)) * 100:.2f}%')

            prob = metrics.get(f'{tissue}_t2_probabilistic')
            if prob is None:
                continue
            for key, label, fmt in (('nll', 'Gaussian NLL:    ', '.6f'),
                                    ('spearman_err_unc', 'Spearman ρ:      ', '.4f'),
                                    ('aurc', 'AURC:            ', '.6f')):
                if prob[key]:
                    lines.append(f'  {label} {np.mean(prob[key]):{fmt}}')
        return lines


def _load_config_any(path: str):
    import json
    p = Path(path)
    if p.suffix.lower() in [".yaml", ".yml"]:
        import yaml
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported config format: {p.suffix}")

def _device_from_arg(s: str):
    if s.startswith("cuda") and torch.cuda.is_available():
        return torch.device(s)
    return torch.device("cpu")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Run config yaml/json used for training this checkpoint")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to best_regression_model.pth for THIS run")
    ap.add_argument("--out_dir", default=None, type=str, help="Override output directory")
    ap.add_argument("--device", default="cuda:0", type=str, help="cuda:0 / cpu")
    ap.add_argument("--num_workers", default=None, type=int, help="Override DataLoader workers")
    ap.add_argument("--to_plot", action="store_true")
    args = ap.parse_args()

    cfg = _load_config_any(args.config)
    dev = _device_from_arg(args.device)

    ev = RegressionEvaluator(cfg, dev)
    ev.model_path_override = args.checkpoint
    if args.out_dir is not None:
        ev.output_dir_override = args.out_dir
    if args.num_workers is not None:
        ev.eval_num_workers = int(args.num_workers)
    ev.to_plot = bool(args.to_plot)

    ev.evaluate()