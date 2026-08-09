#!/usr/bin/env python3
"""
evaluate_explicit_paths.py

Evaluates pre-specified model directories for each acceleration rate and model type,
then generates:
  1. LaTeX tables for overall T2* NRMSE/SSIM and tissue-specific NRMSE
  2. Wilcoxon signed-rank test tables comparing CUPA models vs baselines,
     at significance thresholds p=0.05 and p=0.001

Model types:
  - puq       : none_concat, homoscedastic, no dropout
  - hetero    : none_concat, heteroscedastic, mc_dropout
  - cupa_ch   : cholesky_concat, heteroscedastic, mc_dropout
  - cupa_lr   : low_rank_concat, heteroscedastic, mc_dropout

Wilcoxon comparisons (per acceleration rate):
  - CUPA-CH vs PUQ
  - CUPA-CH vs Hetero
  - CUPA-LR vs PUQ
  - CUPA-LR vs Hetero

Metrics tested:
  Performance : NRMSE (overall, WM, GM, CSF), SSIM
  Uncertainty : AURC, Spearman ρ(|e|,σ), Pearson r(RMS,σ_alea), ECE_alea
               — each evaluated overall AND per tissue (WM, GM, CSF)

Usage
-----
  python evaluate_explicit_paths.py \\
      --evaluator_py /path/to/regression_evaluator.py \\
      --test_data_base /path/to/data \\
      --out_root /path/to/evaluations \\
      --device cuda:0 \\
      [--dry_run]   # print paths only, skip evaluation

  After evaluation, re-run with --skip_eval to regenerate tables only:
      python evaluate_explicit_paths.py --skip_eval --out_root /path/to/evaluations
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Path anchors (derived from this file's location, so the tree can be moved)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
RESULTS_REGRESSION = REPO_ROOT / "results" / "regression"

# ---------------------------------------------------------------------------
# Default model paths
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, str] = {
    # ACC 2
    "acc2_puq": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc2_PUQ_MODEL>"
    ),
    "acc2_hetero": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc2_HET_MODEL>"
    ),
    "acc2_cupa_ch": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc2_CUPA_CH_MODEL>"
    ),
    "acc2_cupa_lr": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc2_CUPA_LR_MODEL>"
    ),
    # ACC 3
    "acc3_puq": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc3_PUQ_MODEL>"
    ),
    "acc3_hetero": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc3_HET_MODEL>"
    ),
    "acc3_cupa_ch": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc3_CUPA_CH_MODEL>"
    ),
    "acc3_cupa_lr": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc3_CUPA_LR_MODEL>"
    ),
    # ACC 4
    "acc4_puq": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_PUQ_MODEL>"
    ),
    "acc4_hetero": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_HET_MODEL>"
    ),
    "acc4_cupa_ch": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_MODEL>"
    ),
    "acc4_cupa_lr": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_LR_MODEL>"
    ),
    "acc4_cupa_ch_no_rms": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_NO_RMS_MODEL>"
    ),
    "acc4_cupa_ch_no_sampling": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_NO_SAMPLING_MODEL>"
    ),
    "acc4_cupa_ch_no_covar": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_NO_COVAR_MODEL>"
    ),
    "acc4_cupa_ch_high_lambda": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_HIGH_LAMBDA_MODEL>"
    ),
    "acc4_cupa_ch_low_lambda": (
        f"{RESULTS_REGRESSION}/<PATH_TO_Acc4_CUPA_CH_LOW_LAMBDA_MODEL>"
    ),
}

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    acc_rate: int
    model_type: str
    run_dir: Path

    @property
    def uncertainty_mode(self) -> str:
        return {
            "puq":                    "none_concat",
            "hetero":                 "none_concat",
            "cupa_ch":                "cholesky_concat",
            "cupa_ch_no_sampling":    "cholesky_concat",
            "cupa_ch_no_rms":         "cholesky_concat",
            "cupa_ch_no_covar":       "cholesky_concat",
            "cupa_ch_high_lambda":    "cholesky_concat",
            "cupa_ch_low_lambda":     "cholesky_concat",
            "cupa_lr":                "low_rank_concat",
        }[self.model_type]

    @property
    def heteroscedastic(self) -> bool:
        return self.model_type != "puq"

    @property
    def use_mc_dropout(self) -> bool:
        return self.model_type != "puq"

    @property
    def lowrank_rank(self) -> Optional[int]:
        if self.model_type != "cupa_lr":
            return None
        m = re.search(r"rank_(\d+)", str(self.run_dir))
        return int(m.group(1)) if m else 10

    @property
    def rms_weight(self) -> Optional[float]:
        m = re.search(r"rms_corr_weight_([\d]+p?[\d]*)", str(self.run_dir))
        if m:
            val = m.group(1).replace("p", ".")
            try:
                w = float(val)
                return w if w > 0 else None
            except ValueError:
                return None
        return None

    @property
    def display_name(self) -> str:
        names = {
            "puq":                 "PUQ",
            "hetero":              "Hetero",
            "cupa_ch":             r"CUPA-T2* (CH)",
            "cupa_lr":             r"CUPA-T2* (LR)",
            "cupa_ch_no_sampling": r"CUPA-CH (No Samp.)",
            "cupa_ch_no_rms":      r"CUPA-CH (No RMS)",
            "cupa_ch_no_covar":    r"CUPA-CH (No CoVar)",
            "cupa_ch_high_lambda": r"CUPA-CH (High λ)",
            "cupa_ch_low_lambda":  r"CUPA-CH (Low λ)",
        }
        return names.get(self.model_type, self.model_type)


# ---------------------------------------------------------------------------
# Checkpoint finding
# ---------------------------------------------------------------------------

CKPT_CANDIDATES = [
    "best_regression_model.pth",
    "best_model.pth",
    "checkpoint_best.pth",
    "model_best.pth",
]


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for name in CKPT_CANDIDATES:
        p = run_dir / name
        if p.exists():
            return p
    for p in run_dir.glob("*.pth"):
        if "best" in p.name.lower():
            return p
    return None


# ---------------------------------------------------------------------------
# Build test config
# ---------------------------------------------------------------------------

def build_test_config(spec: ModelSpec, test_data_base: str) -> Dict[str, Any]:
    return {
        "task": "test_regression",
        "include_uncertainty": True,
        "max_t2": 200.0,
        "min_t2": 0.0,
        "s0_max": 2.5,
        "s0_min": 0.0,
        "regression_params": {
            "regression_gt_test_location": f"{test_data_base}/acc_rate_{spec.acc_rate}/test/",
            "acc_rate": spec.acc_rate,
            "regress_model_dir": str(RESULTS_REGRESSION) + "/",
            "uncertainty_mode": spec.uncertainty_mode,
            "lowrank_rank": spec.lowrank_rank,
            "heteroscedastic": spec.heteroscedastic,
            "use_mc_dropout": spec.use_mc_dropout,
            "mc_dropout_samples": 100,
            "s0_max": 2.5,
            "s0_min": 0.0,
            "te_values": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            "regression_model_params": {
                "in_ch": 12,
                "hidden_ch": 64,
                "out_ch": 1,
                "num_layers": 5,
                "activation": "None",
                "dropout_rate": 0.2 if spec.use_mc_dropout else 0.0,
            },
            "regression_train_params": {
                "loss": "L2",
                "use_rms_correlation_loss": spec.rms_weight is not None,
                "rms_correlation_weight": spec.rms_weight if spec.rms_weight is not None else 0.0,
                "use_cv_correlation_loss": False,
                "cv_correlation_weight": 0.1,
            },
        },
    }


# ---------------------------------------------------------------------------
# Evaluator helpers
# ---------------------------------------------------------------------------

def import_evaluator(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_regression_evaluator", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize_metrics(metrics_by_type: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    def add(prefix, d, keys):
        for k in keys:
            vals = [v for v in d.get(k, []) if v is not None and not math.isnan(v)]
            if not vals:
                out[f"{prefix}_{k}_mean"] = float("nan")
                out[f"{prefix}_{k}_std"]  = float("nan")
            else:
                mu = sum(vals) / len(vals)
                out[f"{prefix}_{k}_mean"] = mu
                out[f"{prefix}_{k}_std"]  = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))

    add("t2",       metrics_by_type.get("t2", {}),       ["nrmse", "mae", "ssim", "psnr"])
    add("recon_t2", metrics_by_type.get("recon_t2", {}), ["nrmse", "mae", "ssim", "psnr"])
    add("t2p",      metrics_by_type.get("t2_probabilistic", {}), [
        "nll", "coverage_1sigma", "coverage_2sigma", "coverage_3sigma",
        "coverage_alea_1sigma", "coverage_alea_2sigma", "coverage_alea_3sigma",
        "coverage_ece_alea_123", "coverage_ece_total_123",
        "spearman_err_unc", "aurc", "cross_stage_consistency",
    ])
    for tissue in ["wm", "gm", "csf"]:
        add(f"{tissue}_t2",       metrics_by_type.get(f"{tissue}_t2", {}),       ["nrmse"])
        add(f"{tissue}_recon_t2", metrics_by_type.get(f"{tissue}_recon_t2", {}), ["nrmse"])
        add(f"{tissue}_t2p",      metrics_by_type.get(f"{tissue}_t2_probabilistic", {}), [
            "nll", "spearman_err_unc", "aurc",
            "coverage_ece_alea_123", "cross_stage_consistency",
        ])
    return out


# ---------------------------------------------------------------------------
# LaTeX table generation (performance)
# ---------------------------------------------------------------------------

_MODEL_TYPES = [
    "puq", "hetero", "cupa_ch", "cupa_lr",
    "cupa_ch_no_sampling", "cupa_ch_no_rms", "cupa_ch_no_covar",
    "cupa_ch_high_lambda", "cupa_ch_low_lambda",
]

_DISPLAY_NAMES = {
    "puq":                 "PUQ Baseline",
    "hetero":              "Hetero Baseline",
    "cupa_ch":             r"CUPA-$T_2^*$ (Chol)",
    "cupa_lr":             r"CUPA-$T_2^*$ (LR)",
    "cupa_ch_no_sampling": r"CUPA-CH (No Samp.)",
    "cupa_ch_no_rms":      r"CUPA-CH (No RMS)",
    "cupa_ch_no_covar":    r"CUPA-CH (No CoVar)",
    "cupa_ch_high_lambda": r"CUPA-CH (High $\lambda$)",
    "cupa_ch_low_lambda":  r"CUPA-CH (Low $\lambda$)",
}


def _fmt(val: Optional[float], digits: int = 3) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "--"
    return f"{val:.{digits}f}"


def _best_mask(values: List[Optional[float]], lower_is_better: bool) -> List[bool]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return [False] * len(values)
    best = min(valid) if lower_is_better else max(valid)
    return [
        (v is not None and not math.isnan(v) and math.isclose(v, best, rel_tol=1e-6))
        for v in values
    ]


def _bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def generate_overall_latex(
    results: Dict[str, Dict[str, Any]],
    acc_rates: List[int],
    exp_name: str = "",
) -> str:
    n_acc      = len(acc_rates)
    n_metric   = 2
    n_data_cols = n_acc * n_metric
    col_groups  = "|".join(["cc"] * n_acc)
    col_spec    = f"|l|{col_groups}|"

    acc_header_cells = " & ".join(
        rf"\multicolumn{{{n_metric}}}{{c|}}{{{r}$\times$}}" for r in acc_rates
    )
    sub_cells = " & ".join([r"NRMSE $\downarrow$ & SSIM $\uparrow$"] * n_acc)

    nrmse_vals = [
        [results.get(f"acc{acc}_{mt}", {}).get("t2_nrmse_mean") for mt in _MODEL_TYPES]
        for acc in acc_rates
    ]
    ssim_vals = [
        [results.get(f"acc{acc}_{mt}", {}).get("t2_ssim_mean") for mt in _MODEL_TYPES]
        for acc in acc_rates
    ]
    nrmse_bold = [_best_mask(nrmse_vals[i], lower_is_better=True)  for i in range(n_acc)]
    ssim_bold  = [_best_mask(ssim_vals[i],  lower_is_better=False) for i in range(n_acc)]

    body_lines = []
    for mi, mtype in enumerate(_MODEL_TYPES):
        cells = [_DISPLAY_NAMES[mtype]]
        for ai, acc in enumerate(acc_rates):
            nv = nrmse_vals[ai][mi]
            sv = ssim_vals[ai][mi]
            nv_str = _fmt(nv, 3)
            sv_str = _fmt(sv, 3)
            if nrmse_bold[ai][mi] and nv_str != "--":
                nv_str = _bold(nv_str)
            if ssim_bold[ai][mi] and sv_str != "--":
                sv_str = _bold(sv_str)
            cells += [nv_str, sv_str]
        body_lines.append(" & ".join(cells) + r" \\")

    rows_str = "\n\\hline\n".join(body_lines)

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{exp_name}. General test metrics (NRMSE lower is better, SSIM higher is better).}}" "\n"
        r"\label{tab:overall_metrics}" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\renewcommand{\arraystretch}{1.15}" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\hline" "\n"
        rf"\multicolumn{{1}}{{|c|}}{{Method}} & {acc_header_cells} \\" "\n"
        rf"\cline{{2-{1 + n_data_cols}}}" "\n"
        rf"\multicolumn{{1}}{{|c|}}{{}} & {sub_cells} \\" "\n"
        r"\hline" "\n"
        + rows_str + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def generate_tissue_latex(
    results: Dict[str, Dict[str, Any]],
    acc_rates: List[int],
    exp_name: str = "",
) -> str:
    n_acc      = len(acc_rates)
    n_tissue   = 3
    n_data_cols = n_acc * n_tissue
    col_groups  = "|".join(["ccc"] * n_acc)
    col_spec    = f"|l|{col_groups}|"

    acc_header_cells = " & ".join(
        rf"\multicolumn{{{n_tissue}}}{{c|}}{{{r}$\times$}}" for r in acc_rates
    )
    sub_cells = " & ".join([r"WM $\downarrow$ & GM $\downarrow$ & CSF $\downarrow$"] * n_acc)

    wm_vals  = [[results.get(f"acc{acc}_{mt}", {}).get("wm_t2_nrmse_mean")  for mt in _MODEL_TYPES] for acc in acc_rates]
    gm_vals  = [[results.get(f"acc{acc}_{mt}", {}).get("gm_t2_nrmse_mean")  for mt in _MODEL_TYPES] for acc in acc_rates]
    csf_vals = [[results.get(f"acc{acc}_{mt}", {}).get("csf_t2_nrmse_mean") for mt in _MODEL_TYPES] for acc in acc_rates]

    wm_bold  = [_best_mask(wm_vals[i],  True) for i in range(n_acc)]
    gm_bold  = [_best_mask(gm_vals[i],  True) for i in range(n_acc)]
    csf_bold = [_best_mask(csf_vals[i], True) for i in range(n_acc)]

    body_lines = []
    for mi, mtype in enumerate(_MODEL_TYPES):
        cells = [_DISPLAY_NAMES[mtype]]
        for ai, acc in enumerate(acc_rates):
            wv  = _fmt(wm_vals[ai][mi],  3)
            gv  = _fmt(gm_vals[ai][mi],  3)
            cv  = _fmt(csf_vals[ai][mi], 3)
            if wm_bold[ai][mi]  and wv != "--": wv = _bold(wv)
            if gm_bold[ai][mi]  and gv != "--": gv = _bold(gv)
            if csf_bold[ai][mi] and cv != "--": cv = _bold(cv)
            cells += [wv, gv, cv]
        body_lines.append(" & ".join(cells) + r" \\")

    rows_str = "\n\\hline\n".join(body_lines)

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{exp_name}. Tissue-specific T2* NRMSE (lower is better).}}" "\n"
        r"\label{tab:tissue_nrmse}" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\renewcommand{\arraystretch}{1.15}" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\hline" "\n"
        rf"\multicolumn{{1}}{{|c|}}{{Method}} & {acc_header_cells} \\" "\n"
        rf"\cline{{2-{1 + n_data_cols}}}" "\n"
        rf"\multicolumn{{1}}{{|c|}}{{}} & {sub_cells} \\" "\n"
        r"\hline" "\n"
        + rows_str + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ===========================================================================
# WILCOXON ANALYSIS
# ===========================================================================

# The four pairwise comparisons we always test
_WILCOXON_COMPARISONS = [
    ("cupa_ch", "puq"),
    ("cupa_ch", "hetero"),
    ("cupa_lr", "puq"),
    ("cupa_lr", "hetero"),
]

# Performance metrics: (metric_label, csv_column, lower_is_better, tissue_filter)
# tissue_filter: "overall" uses all tissue rows; "wm"/"gm"/"csf" filters to that tissue.
_PERF_METRICS = [
    ("NRMSE",      "nrmse", True,  "overall"),
    ("NRMSE-WM",   "nrmse", True,  "wm"),
    ("NRMSE-GM",   "nrmse", True,  "gm"),
    ("NRMSE-CSF",  "nrmse", True,  "csf"),
    ("SSIM",       "ssim",  False, "overall"),
]

# Uncertainty metrics: evaluated overall AND per tissue.
# Base definitions (tissue will be expanded below).
_UNC_METRIC_BASES = [
    ("AURC",  "aurc",                    True),   # lower is better
    ("Spear", "spearman_err_unc",        False),  # higher is better
    ("Pears", "cross_stage_consistency", False),  # higher is better
    ("ECE",   "coverage_ece_alea_123",   True),   # lower is better
]

# Expanded uncertainty metrics: overall + per tissue
_UNC_METRICS = []
for _base_label, _col, _lib in _UNC_METRIC_BASES:
    _UNC_METRICS.append((_base_label,           _col, _lib, "overall"))
    _UNC_METRICS.append((f"{_base_label}-WM",   _col, _lib, "wm"))
    _UNC_METRICS.append((f"{_base_label}-GM",   _col, _lib, "gm"))
    _UNC_METRICS.append((f"{_base_label}-CSF",  _col, _lib, "csf"))

# All uncertainty metric labels (for PUQ exclusion logic)
_UNC_METRIC_LABELS = {label for label, _, _, _ in _UNC_METRICS}


def _load_slice_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load per_slice_tissue_metrics.csv; return None if not found."""
    if not csv_path.exists():
        print(f"  [WARN] CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df


def _get_paired_series(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col: str,
    tissue: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract paired (subject, slice_num) values for `col` from both DataFrames.

    - tissue="overall": uses all tissue rows (wm + gm + csf combined).
    - tissue="wm"/"gm"/"csf": filters to that specific tissue only.

    Filters to method='pred', then inner-joins on (subject, slice_num[, tissue]).
    Returns two aligned numpy arrays.
    """
    def _extract(df: pd.DataFrame) -> pd.DataFrame:
        mask = df["method"] == "pred"
        if tissue != "overall":
            mask = mask & (df["tissue"] == tissue)
        # Keep tissue in the join key so pairs remain tissue-specific when
        # tissue=="overall" (avoids accidentally pairing wm-a with gm-b).
        sub = df.loc[mask, ["subject", "slice_num", "tissue", col]].copy()
        sub = sub.dropna(subset=[col])
        sub = sub.rename(columns={col: "value"})
        return sub

    sa = _extract(df_a)
    sb = _extract(df_b)

    # Join on (subject, slice_num, tissue) to guarantee proper pairing
    merged = sa.merge(sb, on=["subject", "slice_num", "tissue"], suffixes=("_a", "_b"))
    return merged["value_a"].values, merged["value_b"].values


def _run_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    lower_is_better: bool,
) -> Dict[str, Any]:
    """
    Run two-sided Wilcoxon signed-rank test on paired (x, y).
    x = model A (e.g. CUPA-CH), y = model B (e.g. PUQ).
    Returns dict with stat, p_value, n, direction (which is better).
    """
    if len(x) < 10 or len(y) < 10:
        return {"stat": np.nan, "p_value": np.nan, "n": 0, "direction": "?"}

    diff = x - y
    n_nonzero = int(np.sum(diff != 0))

    if n_nonzero < 5:
        return {"stat": np.nan, "p_value": np.nan, "n": len(x), "direction": "?"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p_val = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

    med_a = float(np.median(x))
    med_b = float(np.median(y))

    if lower_is_better:
        direction = "A" if med_a < med_b else "B"
    else:
        direction = "A" if med_a > med_b else "B"

    return {
        "stat":       float(stat),
        "p_value":    float(p_val),
        "n":          len(x),
        "n_nonzero":  n_nonzero,
        "median_a":   med_a,
        "median_b":   med_b,
        "direction":  direction,   # "A" = CUPA model is better, "B" = baseline is better
    }


def _fmt_wilcoxon_cell(
    result: Dict[str, Any],
    model_a_short: str,
    model_b_short: str,
    alpha: float,
) -> str:
    """
    Format a single Wilcoxon result for a LaTeX table cell.
    """
    if result is None or math.isnan(result.get("p_value", float("nan"))):
        return "--"

    p    = result["p_value"]
    dir_ = result["direction"]
    winner = model_a_short if dir_ == "A" else model_b_short
    arrow  = r"$\downarrow$" if dir_ == "A" else r"$\uparrow$"

    dir_label = rf"[{arrow}\texttt{{{winner}}}]"
    p_str = f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"
    cell  = rf"$p$={p_str} {dir_label}"

    if p < alpha:
        cell = rf"\textbf{{{cell}}}"

    return cell


def _build_wilcoxon_latex_table(
    wilcoxon_rows: List[Dict],
    acc_rate: int,
    alpha: float,
    exp_name: str = "",
    group: str = "perf",  # "perf" or "unc"
) -> str:
    """
    Build a LaTeX table for one acceleration rate, one alpha, one metric group.
    Columns: Metric | CH vs PUQ | CH vs Hetero | LR vs PUQ | LR vs Hetero
    Rows: one per metric (uncertainty rows now include tissue-specific variants).
    """
    comp_labels = [
        ("cupa_ch", "puq",    r"CH vs PUQ"),
        ("cupa_ch", "hetero", r"CH vs Hetero"),
        ("cupa_lr", "puq",    r"LR vs PUQ"),
        ("cupa_lr", "hetero", r"LR vs Hetero"),
    ]

    alpha_str   = "0.001" if alpha < 0.005 else "0.05"
    group_label = "Performance" if group == "perf" else "Uncertainty"
    col_spec    = r"|l|c|c|c|c|"
    header      = r"Metric & CH vs PUQ & CH vs Hetero & LR vs PUQ & LR vs Hetero \\"

    body_lines = []
    for row in wilcoxon_rows:
        if row.get("group") != group:
            continue
        cells = [row["metric_label"]]
        for a_type, b_type, _ in comp_labels:
            key = (a_type, b_type, row["metric_label"])
            res = row.get("result", {}).get(key, None)

            if b_type == "puq" and row["metric_label"] in _UNC_METRIC_LABELS:
                cells.append("--")
                continue
            if res is None:
                cells.append("--")
                continue

            a_short = "CH" if "ch" in a_type else "LR"
            b_short = "PUQ" if b_type == "puq" else "Het"
            cells.append(_fmt_wilcoxon_cell(res, a_short, b_short, alpha))

        body_lines.append(" & ".join(cells) + r" \\")

    if not body_lines:
        return f"% No data for {group} metrics at acc{acc_rate}\n"

    rows_str = "\n\\hline\n".join(body_lines)

    caption = (
        rf"{exp_name}. Wilcoxon signed-rank test ({group_label} metrics), "
        rf"$R={acc_rate}\times$, $\alpha={alpha_str}$. "
        r"Bold = significant. Arrow shows which model is better "
        r"(CH/LR = CUPA model, PUQ/Het = baseline)."
    )
    label = f"tab:wilcoxon_{group}_acc{acc_rate}_a{alpha_str.replace('.', '')}"

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        rf"\label{{{label}}}" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{4pt}" "\n"
        r"\renewcommand{\arraystretch}{1.15}" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\hline" "\n"
        + header + "\n"
        r"\hline" "\n"
        + rows_str + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def run_wilcoxon_analysis(
    eval_root: Path,
    acc_rates: List[int],
    out_dir: Path,
    exp_name: str = "",
) -> None:
    """
    Main Wilcoxon analysis function.
    Reads per_slice_tissue_metrics.csv for each (acc_rate, model_type),
    runs paired Wilcoxon tests for all comparison pairs and metrics
    (including tissue-specific uncertainty metrics),
    writes LaTeX tables and a summary CSV.

    Directory structure assumed:
        eval_root / acc{R} / {model_type} / per_slice_tissue_metrics.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    all_alphas   = [0.05, 0.001]
    all_metrics  = _PERF_METRICS + _UNC_METRICS

    csv_rows: List[Dict] = []

    for acc in acc_rates:
        print(f"\n--- Wilcoxon analysis for acc={acc} ---")

        # Load all four model CSVs
        dfs: Dict[str, Optional[pd.DataFrame]] = {}
        for mtype in ["puq", "hetero", "cupa_ch", "cupa_lr"]:
            csv_path = eval_root / f"acc{acc}" / mtype / "per_slice_tissue_metrics.csv"
            dfs[mtype] = _load_slice_csv(csv_path)

        # ----------------------------------------------------------------
        # Compute Wilcoxon for every (comparison, metric) combination
        # ----------------------------------------------------------------
        wilcoxon_results: Dict[Tuple, Dict] = {}

        for a_type, b_type in _WILCOXON_COMPARISONS:
            df_a = dfs.get(a_type)
            df_b = dfs.get(b_type)

            if df_a is None or df_b is None:
                print(f"  Skipping {a_type} vs {b_type}: CSV missing")
                continue

            for metric_label, col, lower_is_better, tissue in all_metrics:

                # Skip uncertainty metrics for PUQ (no uncertainty estimates)
                if b_type == "puq" and metric_label in _UNC_METRIC_LABELS:
                    continue

                # Check column exists in both DataFrames
                if col not in df_a.columns or col not in df_b.columns:
                    print(f"  [WARN] Column '{col}' not found for {a_type} or {b_type}")
                    continue

                x, y = _get_paired_series(df_a, df_b, col, tissue)

                if len(x) < 10:
                    print(f"  [WARN] Not enough paired samples for "
                          f"{a_type} vs {b_type}, {metric_label} (tissue={tissue}): n={len(x)}")
                    result = {"stat": np.nan, "p_value": np.nan, "n": len(x), "direction": "?"}
                else:
                    result = _run_wilcoxon(x, y, lower_is_better)
                    print(f"  {a_type:12s} vs {b_type:7s} | {metric_label:12s} "
                          f"| tissue={tissue:7s} | n={result['n']:4d} "
                          f"| p={result['p_value']:.4f} | dir={result['direction']}")

                wilcoxon_results[(a_type, b_type, metric_label)] = result

                csv_rows.append({
                    "acc_rate":      acc,
                    "model_a":       a_type,
                    "model_b":       b_type,
                    "metric":        metric_label,
                    "tissue":        tissue,
                    "csv_column":    col,
                    "n_pairs":       result.get("n", 0),
                    "n_nonzero":     result.get("n_nonzero", np.nan),
                    "median_a":      result.get("median_a", np.nan),
                    "median_b":      result.get("median_b", np.nan),
                    "direction":     result.get("direction", "?"),
                    "statistic":     result.get("stat", np.nan),
                    "p_value":       result.get("p_value", np.nan),
                    "sig_0.05":      result.get("p_value", 1.0) < 0.05,
                    "sig_0.001":     result.get("p_value", 1.0) < 0.001,
                })

        # ----------------------------------------------------------------
        # Organise rows for table builders
        # ----------------------------------------------------------------
        table_rows = []
        for metric_label, col, lower_is_better, tissue in all_metrics:
            is_unc  = metric_label in _UNC_METRIC_LABELS
            group   = "unc" if is_unc else "perf"
            row = {
                "metric_label": metric_label,
                "tissue":       tissue,
                "group":        group,
                "result":       {},
            }
            for a_type, b_type in _WILCOXON_COMPARISONS:
                key = (a_type, b_type, metric_label)
                if key in wilcoxon_results:
                    row["result"][key] = wilcoxon_results[key]
            table_rows.append(row)

        # ----------------------------------------------------------------
        # Generate and save individual LaTeX tables
        # ----------------------------------------------------------------
        for alpha in all_alphas:
            alpha_str = "0p001" if alpha < 0.005 else "0p05"
            for group in ["perf", "unc"]:
                tex = _build_wilcoxon_latex_table(
                    table_rows, acc_rate=acc, alpha=alpha,
                    exp_name=exp_name, group=group,
                )
                fname = f"wilcoxon_{group}_acc{acc}_alpha{alpha_str}.tex"
                fpath = out_dir / fname
                fpath.write_text(tex)
                print(f"  ✓ Saved: {fpath}")

        # ----------------------------------------------------------------
        # Combined table (perf + unc with tissue sections)
        # ----------------------------------------------------------------
        combined = _build_combined_wilcoxon_latex(
            table_rows, acc_rate=acc, exp_name=exp_name
        )
        cpath = out_dir / f"wilcoxon_combined_acc{acc}.tex"
        cpath.write_text(combined)
        print(f"  ✓ Saved combined: {cpath}")

    # ----------------------------------------------------------------
    # Save master CSV
    # ----------------------------------------------------------------
    if csv_rows:
        df_csv  = pd.DataFrame(csv_rows)
        csv_out = out_dir / "wilcoxon_all_results.csv"
        df_csv.to_csv(csv_out, index=False)
        print(f"\n✓ Wilcoxon CSV -> {csv_out}")


def _build_combined_wilcoxon_latex(
    wilcoxon_rows: List[Dict],
    acc_rate: int,
    exp_name: str = "",
) -> str:
    """
    Single combined table per acceleration rate.

    Structure:
      ── Performance ──────────────────────────────────
        NRMSE | NRMSE-WM | NRMSE-GM | NRMSE-CSF | SSIM
      ── Uncertainty (Overall) ────────────────────────
        AURC | Spear | Pears | ECE
      ── Uncertainty (WM) ─────────────────────────────
        AURC-WM | Spear-WM | Pears-WM | ECE-WM
      ── Uncertainty (GM) ─────────────────────────────
        AURC-GM | Spear-GM | Pears-GM | ECE-GM
      ── Uncertainty (CSF) ────────────────────────────
        AURC-CSF | Spear-CSF | Pears-CSF | ECE-CSF

    Significance markers: * p<0.05, ** p<0.001 (two-sided).
    """
    comp_labels_short = [
        ("cupa_ch", "puq",    "CH vs PUQ"),
        ("cupa_ch", "hetero", "CH vs Het."),
        ("cupa_lr", "puq",    "LR vs PUQ"),
        ("cupa_lr", "hetero", "LR vs Het."),
    ]

    col_spec   = r"|l|c|c|c|c|"
    header_row = r"Metric & CH vs PUQ & CH vs Het. & LR vs PUQ & LR vs Het. \\"

    def fmt_cell(res, a_short, b_short):
        if res is None or math.isnan(res.get("p_value", float("nan"))):
            return "--"
        p     = res["p_value"]
        direc = res["direction"]
        winner = a_short if direc == "A" else b_short

        if p < 0.001:
            sig_marker = r"$^{**}$"
        elif p < 0.05:
            sig_marker = r"$^{*}$"
        else:
            sig_marker = ""

        p_str     = f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"
        dir_label = rf"\textsf{{{winner}}}"
        cell      = rf"$p$={p_str}{sig_marker} [{dir_label}]"

        if p < 0.05:
            cell = rf"\textbf{{{cell}}}"
        return cell

    def make_metric_row(metric_label: str) -> str:
        """Return a full LaTeX row string for one metric."""
        row_d = next((r for r in wilcoxon_rows if r["metric_label"] == metric_label), None)
        cells = [metric_label]
        for a_type, b_type, _ in comp_labels_short:
            if b_type == "puq" and metric_label in _UNC_METRIC_LABELS:
                cells.append("--")
                continue
            res     = row_d["result"].get((a_type, b_type, metric_label)) if row_d else None
            a_short = "CH" if "ch" in a_type else "LR"
            b_short = "PUQ" if b_type == "puq" else "Het"
            cells.append(fmt_cell(res, a_short, b_short))
        return " & ".join(cells) + r" \\"

    def section_header(title: str) -> str:
        return rf"\multicolumn{{5}}{{|l|}}{{\textit{{{title}}}}} \\"

    body_lines = []

    # ── Performance section ──────────────────────────────────────────────
    body_lines.append(section_header("Performance"))
    body_lines.append(r"\hline")
    for metric_label, _, _, _ in _PERF_METRICS:
        body_lines.append(make_metric_row(metric_label))
        body_lines.append(r"\hline")

    # ── Uncertainty sections (overall + per tissue) ──────────────────────
    for tissue_label, tissue_key in [
        ("Overall", "overall"),
        ("WM",      "wm"),
        ("GM",      "gm"),
        ("CSF",     "csf"),
    ]:
        body_lines.append(section_header(f"Uncertainty Quality — {tissue_label}"))
        body_lines.append(r"\hline")

        # Select the uncertainty metrics that belong to this tissue
        if tissue_key == "overall":
            metrics_for_tissue = [
                (label, col, lib, t)
                for label, col, lib, t in _UNC_METRICS
                if t == "overall"
            ]
        else:
            suffix = f"-{tissue_key.upper()}"
            metrics_for_tissue = [
                (label, col, lib, t)
                for label, col, lib, t in _UNC_METRICS
                if t == tissue_key
            ]

        for metric_label, _, _, _ in metrics_for_tissue:
            body_lines.append(make_metric_row(metric_label))
            body_lines.append(r"\hline")

    rows_str = "\n".join(body_lines)

    caption = (
        rf"{exp_name}. Wilcoxon signed-rank test, $R={acc_rate}\times$. "
        r"$^*p<0.05$, $^{{**}}p<0.001$ (two-sided). "
        r"Bold = significant at $p<0.05$. "
        r"Bracket shows which model has better median "
        r"(CH/LR = CUPA; PUQ/Het = baseline). "
        r"Uncertainty metrics are shown overall and per tissue (WM/GM/CSF). "
        r"-- = not applicable (PUQ has no uncertainty estimates)."
    )
    label = f"tab:wilcoxon_combined_acc{acc_rate}"

    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        rf"\label{{{label}}}" "\n"
        r"\scriptsize" "\n"
        r"\setlength{\tabcolsep}{3pt}" "\n"
        r"\renewcommand{\arraystretch}{1.2}" "\n"
        rf"\begin{{tabular}}{{{col_spec}}}" "\n"
        r"\hline" "\n"
        + header_row + "\n"
        r"\hline" "\n"
        + rows_str + "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--evaluator_py",   type=Path, default=None,
                    help="Path to regression_evaluator.py")
    ap.add_argument("--test_data_base", type=str,
                    default=str(PHIMO_ROOT / "data" / "T2_param_data" / "With_Unc"),
                    help="Base path for test data")
    ap.add_argument("--out_root",       type=Path,
                    default=PHIMO_ROOT / "evaluations",
                    help="Root directory for evaluation outputs")
    ap.add_argument("--results_csv",    type=Path, default=None)
    ap.add_argument("--device",         type=str,  default="cuda:0")
    ap.add_argument("--num_workers",    type=int,  default=4)
    ap.add_argument("--to_plot",        action="store_true")
    ap.add_argument("--dry_run",        action="store_true",
                    help="Print paths/checkpoints only; skip evaluation")
    ap.add_argument("--skip_eval",      action="store_true",
                    help="Skip evaluation; regenerate tables and run Wilcoxon only")
    ap.add_argument("--exp_name",       type=str,  default="")

    # Per-model path overrides
    for key in DEFAULTS:
        ap.add_argument(f"--{key}", type=str, default=None)

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Resolve model paths -> ModelSpec list
    # ------------------------------------------------------------------
    specs: List[ModelSpec] = []
    for key, default_path in DEFAULTS.items():
        raw = getattr(args, key, None)
        path_str = raw if raw is not None else default_path
        if not path_str:
            print(f"  [SKIP] {key}: no path")
            continue

        acc_str  = key.split("_")[0]           # "acc2"
        mtype    = key[len(acc_str) + 1:]      # everything after "acc2_"
        acc_rate = int(acc_str.replace("acc", ""))

        specs.append(ModelSpec(key=key, acc_rate=acc_rate, model_type=mtype,
                               run_dir=Path(path_str)))

    print(f"\nResolved {len(specs)} model paths.\n")

    # ------------------------------------------------------------------
    # Find checkpoints
    # ------------------------------------------------------------------
    valid_specs: List[Tuple[ModelSpec, Path]] = []
    for spec in specs:
        ckpt   = find_checkpoint(spec.run_dir)
        status = f"  {spec.key:30s}  acc={spec.acc_rate}  {spec.model_type:20s}"
        if ckpt:
            print(f"{status}  ckpt={ckpt.name}")
            valid_specs.append((spec, ckpt))
        else:
            print(f"{status}  [WARN] No checkpoint — will be skipped")

    if args.dry_run:
        print("\nDRY RUN — exiting.")
        return

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    out_root   = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or (out_root / "results.csv")

    all_results: Dict[str, Dict[str, Any]] = {}
    rows_out:    List[Dict[str, Any]]       = []
    failed:      List[str]                  = []

    if not args.skip_eval:
        if args.evaluator_py is None:
            print("ERROR: --evaluator_py is required unless --skip_eval is set.")
            return

        ev_mod = import_evaluator(args.evaluator_py)
        RegressionEvaluator = getattr(ev_mod, "RegressionEvaluator")
        _device_from_arg    = getattr(ev_mod, "_device_from_arg")
        dev = _device_from_arg(args.device)

        for spec, ckpt in valid_specs:
            out_dir    = out_root / f"acc{spec.acc_rate}" / spec.model_type
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_file = out_dir / "eval_summary.json"

            if cache_file.exists():
                print(f"\n[{spec.key}] Loading cached results from {cache_file}")
                with cache_file.open() as f:
                    row = json.load(f)
                all_results[spec.key] = row
                rows_out.append(row)
                continue

            print(f"\n[{spec.key}] EVALUATING  (ckpt={ckpt.name})")

            try:
                cfg = build_test_config(spec, test_data_base=args.test_data_base)
                ev  = RegressionEvaluator(cfg, dev)
                ev.model_path_override = str(ckpt)
                ev.output_dir_override = str(out_dir)
                ev.to_plot             = bool(args.to_plot)

                metrics = ev.evaluate()
                summ    = summarize_metrics(metrics)

                row: Dict[str, Any] = {
                    "key":              spec.key,
                    "acc_rate":         spec.acc_rate,
                    "model_type":       spec.model_type,
                    "display_name":     spec.display_name,
                    "uncertainty_mode": spec.uncertainty_mode,
                    "heteroscedastic":  spec.heteroscedastic,
                    "use_mc_dropout":   spec.use_mc_dropout,
                    "lowrank_rank":     spec.lowrank_rank,
                    "rms_weight":       spec.rms_weight,
                    "run_dir":          str(spec.run_dir),
                    "checkpoint_path":  str(ckpt),
                    "eval_out_dir":     str(out_dir),
                }
                row.update(summ)

                all_results[spec.key] = row
                rows_out.append(row)

                with cache_file.open("w") as f:
                    json.dump(row, f, indent=2)
                print(f"  ✓ Done -> {cache_file}")

            except Exception:
                failed.append(spec.key)
                print(f"  FAILED: {spec.key}")
                traceback.print_exc()

    else:
        # Load cached results
        print("\n--skip_eval: loading cached eval_summary.json files...")
        for spec, ckpt in valid_specs:
            out_dir    = out_root / f"acc{spec.acc_rate}" / spec.model_type
            cache_file = out_dir / "eval_summary.json"
            if cache_file.exists():
                with cache_file.open() as f:
                    row = json.load(f)
                all_results[spec.key] = row
                rows_out.append(row)
                print(f"  Loaded: {spec.key}")
            else:
                print(f"  [WARN] No cache for {spec.key}: {cache_file}")

    # ------------------------------------------------------------------
    # Write master CSV
    # ------------------------------------------------------------------
    if rows_out:
        base_cols   = ["key", "acc_rate", "model_type", "display_name",
                        "uncertainty_mode", "heteroscedastic", "use_mc_dropout",
                        "lowrank_rank", "rms_weight",
                        "run_dir", "checkpoint_path", "eval_out_dir"]
        metric_cols = sorted(c for c in rows_out[0] if c not in base_cols)
        fieldnames  = base_cols + metric_cols

        with results_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"\n✓ Results CSV -> {results_csv.resolve()}")

    # ------------------------------------------------------------------
    # Performance LaTeX tables
    # ------------------------------------------------------------------
    evaluated_accs = sorted({spec.acc_rate for spec, _ in valid_specs
                              if spec.key in all_results})
    if not evaluated_accs:
        evaluated_accs = [2, 3, 4]

    latex_overall = generate_overall_latex(all_results, evaluated_accs, exp_name=args.exp_name)
    latex_tissue  = generate_tissue_latex(all_results, evaluated_accs, exp_name=args.exp_name)

    overall_tex = out_root / "table_overall.tex"
    tissue_tex  = out_root / "table_tissue_nrmse.tex"
    overall_tex.write_text(latex_overall)
    tissue_tex.write_text(latex_tissue)

    print(f"\n✓ LaTeX overall table -> {overall_tex.resolve()}")
    print(f"✓ LaTeX tissue table  -> {tissue_tex.resolve()}")
    print("\n--- TABLE: Overall T2* metrics ---")
    print(latex_overall)
    print("\n--- TABLE: Tissue-specific NRMSE ---")
    print(latex_tissue)

    # ------------------------------------------------------------------
    # Wilcoxon analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING WILCOXON SIGNED-RANK ANALYSIS")
    print("=" * 70)

    wilcoxon_dir = out_root / "wilcoxon_tables"
    run_wilcoxon_analysis(
        eval_root  = out_root,
        acc_rates  = evaluated_accs,
        out_dir    = wilcoxon_dir,
        exp_name   = args.exp_name,
    )

    # Print combined tables to stdout
    for acc in evaluated_accs:
        combined_tex = wilcoxon_dir / f"wilcoxon_combined_acc{acc}.tex"
        if combined_tex.exists():
            print(f"\n--- WILCOXON TABLE: acc={acc} ---")
            print(combined_tex.read_text())

    if failed:
        print("\nFailed evaluations:")
        for k in failed:
            print("  -", k)

    print("\nAll done.")


if __name__ == "__main__":
    main()