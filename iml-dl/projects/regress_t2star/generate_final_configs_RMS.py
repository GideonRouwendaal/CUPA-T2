#!/usr/bin/env python3
"""
sweep_rms_correlation.py

Stage-2 sweep: fix augmentation (from top-K combined configs) and sweep
rms_correlation_weight. Reads the CSV produced by scan_tissue_results.py,
takes the top-K configs, and crosses them with RMS_WEIGHTS.

Strategy
--------
  Top-K configs: 3  (by mean WM+GM NRMSE from combined sweep)
  RMS weights:   [0.02, 0.05, 0.08, 0.1, 0.12, 0.25, 0.5, 0.75, 1.0]
  Total runs:    3 × 9 = 27

Usage
-----
python sweep_rms_correlation.py \
    --csv /path/to/tissue_sweep_results.csv \
    --top_k 3 \
    --acc_rates 4 \
    --out_root /path/to/output/rms_sweep
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# =============================================================================
# Paths
# =============================================================================
# =============================================================================
# Path anchors — derived from this file's location so the tree can be moved.
# PROJECT_ROOT/../.. is the iml-dl repo root; its parent is the CUPA_ROOT root.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PROJECT_ROOT.parents[1]
CUPA_ROOT    = REPO_ROOT.parent
DATA_ROOT    = CUPA_ROOT / "data" / "T2_param_data" / "With_Unc"


# =============================================================================
# RMS weight grid — coarse + fine combined into one pass
# =============================================================================
RMS_WEIGHTS = [0.01, 0.025, 0.05, 0.75, 0.1, 0.12, 0.2, 0.3, 0.5, 1.0]


# =============================================================================
# Base config — same as combined sweep (n_epochs=200)
# =============================================================================
def get_base_config(exp_name: str, acc_rate: int) -> dict:
    return {
        "max_t2": 200.0,
        "min_t2": 0.0,
        "regression_params": {
            "s0_max": 2.5,
            "s0_min": 0.0,
            "acc_rate": str(acc_rate),
            "param_to_predict": "t2star",
            "te_values": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],

            "regression_gt_train_location": str(DATA_ROOT / exp_name / f"acc_rate_{acc_rate}" / "train") + "/",
            "regression_gt_val_location":   str(DATA_ROOT / exp_name / f"acc_rate_{acc_rate}" / "val")   + "/",

            "heteroscedastic": True,
            "use_mc_dropout":  True,
            "multitask":       False,
            "add_positional_encoding": False,

            "uncertainty_mode": "cholesky_concat",
            "lowrank_rank":     None,
            "run_tag":          None,

            "regression_train_params": {
                "loss":                    "L2",
                "n_epochs":                200,
                "early_stopping_patience": 40,

                "batch_size":    1024,
                "lr":            1e-4,
                "weight_decay":  0.0,
                "grad_max_norm": 0,

                "include_physics_loss": False,
                "noise_schedule":       "constant",
                "noise_scale":          1.0,

                # RMS — swept here
                "use_rms_correlation_loss": True,
                "rms_correlation_weight":   0.0,  # filled per config
                "use_cv_correlation_loss":  False,
                "cv_correlation_weight":    0.1,
                "use_consistency_loss":     False,
                "consistency_weight":       0.1,

                "uncertainty_schedule": {},
            },

            "regression_model_params": {
                "in_ch":        12,
                "out_ch":       1,
                "hidden_ch":    64,
                "num_layers":   5,
                "dropout_rate": 0.0,
                "activation":   "None",
            },
        },
    }


def deep_copy(cfg: dict) -> dict:
    return yaml.safe_load(yaml.safe_dump(cfg))


# =============================================================================
# Baseline configs
# =============================================================================
def generate_baselines(exp_name: str, acc_rate: int) -> List[Tuple[str, dict]]:
    baselines = []

    # --- Baseline 1: none_concat, no heteroscedastic, no mc_dropout ---
    cfg1 = deep_copy(get_base_config(exp_name, acc_rate))
    rp1  = cfg1["regression_params"]
    rp1["uncertainty_mode"] = "none_concat"
    rp1["heteroscedastic"]  = False
    rp1["use_mc_dropout"]   = False
    tag1 = f"baseline__none_concat__acc{acc_rate}"
    rp1["run_tag"]    = tag1
    rp1["sweep_type"] = "baseline"
    rp1["regression_train_params"]["uncertainty_schedule"] = {}
    baselines.append((tag1, cfg1))

    # --- Baseline 2: none_concat + heteroscedastic + mc_dropout ---
    cfg2 = deep_copy(get_base_config(exp_name, acc_rate))
    rp2  = cfg2["regression_params"]
    rp2["uncertainty_mode"] = "none_concat"
    rp2["heteroscedastic"]  = True
    rp2["use_mc_dropout"]   = True
    tag2 = f"baseline__none_concat_hetero_mc__acc{acc_rate}"
    rp2["run_tag"]    = tag2
    rp2["sweep_type"] = "baseline"
    rp2["regression_train_params"]["uncertainty_schedule"] = {}
    baselines.append((tag2, cfg2))

    return baselines


# =============================================================================
# Schedule builder — identical to combined sweep
# =============================================================================
def build_combined_schedule(
    q_lo: float, q_hi: float,
    wm_alpha_min: float, wm_alpha_max: float, wm_gamma: float,
    gm_alpha_min: float, gm_alpha_max: float, gm_gamma: float,
    csf_alpha_min: float, csf_alpha_max: float, csf_gamma: float,
) -> dict:
    return {
        "mode":  "tissue_quantile",
        "q_lo":  float(q_lo),
        "q_hi":  float(q_hi),
        "alpha_min": float(min(wm_alpha_min, gm_alpha_min, csf_alpha_min)),
        "alpha_max": float(max(wm_alpha_max, gm_alpha_max, csf_alpha_max)),
        "gamma":     1.0,
        "beta":  0.0,
        "w_min": 1.0,
        "compute_quantiles_at_start": True,
        "max_quantile_samples":       5_000_000,
        "use_fixed_quantiles":        False,
        "fallback_mode":              "none",
        "tissue_schedule": {
            "wm":  {"alpha_min": float(wm_alpha_min),  "alpha_max": float(wm_alpha_max),  "gamma": float(wm_gamma)},
            "gm":  {"alpha_min": float(gm_alpha_min),  "alpha_max": float(gm_alpha_max),  "gamma": float(gm_gamma)},
            "csf": {"alpha_min": float(csf_alpha_min), "alpha_max": float(csf_alpha_max), "gamma": float(csf_gamma)},
        },
    }


# =============================================================================
# Parse tissue params from yaml_name — same regex as scan_tissue_results.py
# =============================================================================
def parse_yaml_name(yaml_name: str) -> Optional[Dict]:
    """
    Parse tissue quantile / alpha / gamma params from a yaml_name string.
    Returns a dict with keys: q_lo, q_hi, wm, gm, csf  (each tissue has
    alpha_min, alpha_max, gamma).  Returns None if the pattern is not found.
    """
    q_m = re.search(r"__q([\dp]+)-([\dp]+)__", yaml_name)
    if not q_m:
        return None
    q_lo = float(q_m.group(1).replace("p", "."))
    q_hi = float(q_m.group(2).replace("p", "."))

    tissue_pattern = re.compile(r"(wm|gm|csf)([\dp]+)-([\dp]+)g([\dp]+)")
    tissue_params = {}
    for m in tissue_pattern.finditer(yaml_name):
        tissue = m.group(1)
        tissue_params[tissue] = {
            "alpha_min": float(m.group(2).replace("p", ".")),
            "alpha_max": float(m.group(3).replace("p", ".")),
            "gamma":     float(m.group(4).replace("p", ".")),
        }

    if not all(t in tissue_params for t in ["wm", "gm", "csf"]):
        return None

    return {"q_lo": q_lo, "q_hi": q_hi, **tissue_params}


def parse_rank_from_yaml_name(yaml_name: str) -> Optional[int]:
    """
    Extract the lowrank rank from a yaml_name string.
    Matches the pattern __lr__rN__ (e.g. __lr__r6__ -> 6).
    Returns None if the pattern is absent (i.e. not a low-rank run).
    """
    m = re.search(r"__lr__r(\d+)__", yaml_name)
    return int(m.group(1)) if m else None


# =============================================================================
# Load top-K from CSV — also extracts rank from yaml_name
# =============================================================================
def load_top_k_from_csv(csv_path: Path, top_k: int, acc_rate: int) -> List[Dict]:
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yaml_name = row.get("yaml_name", "")
            if f"acc{acc_rate}" not in yaml_name and f"acc_{acc_rate}" not in yaml_name:
                if re.search(r"acc\d+", yaml_name):
                    continue

            nrmse_wm = float(row["nrmse_wm"]) if row.get("nrmse_wm") else None
            nrmse_gm = float(row["nrmse_gm"]) if row.get("nrmse_gm") else None
            if nrmse_wm is None or nrmse_gm is None:
                continue

            rank = parse_rank_from_yaml_name(yaml_name)

            records.append({
                **row,
                "mean_nrmse": 0.5 * (nrmse_wm + nrmse_gm),
                "rank":       rank,
            })

    records.sort(key=lambda r: r["mean_nrmse"])
    top = records[:top_k]

    print(f"  Loaded {len(records)} valid records; taking top {len(top)}")
    for i, r in enumerate(top, 1):
        rank_str = f"rank={r['rank']}" if r["rank"] is not None else "rank=N/A"
        print(f"    #{i}  mean_nrmse={r['mean_nrmse']:.6f}  {rank_str}  {r['yaml_name']}")

    return top


# =============================================================================
# Generate configs: top-K × RMS_WEIGHTS
# Propagates rank into the generated config and run_tag.
# =============================================================================
def generate_rms_configs(
    top_records: List[Dict],
    exp_name:    str,
    acc_rate:    int,
    rms_weights: List[float],
) -> List[Tuple[str, dict]]:
    cfgs = []
    for src_idx, r in enumerate(top_records, start=1):
        yaml_name = r.get("yaml_name", "")
        params = parse_yaml_name(yaml_name)
        if params is None:
            print(f"  [WARN] Could not parse tissue params from: {yaml_name} — skipping")
            continue

        rank = r.get("rank")  # int | None

        for rms_w in rms_weights:
            cfg = deep_copy(get_base_config(exp_name, acc_rate))
            rp  = cfg["regression_params"]
            tp  = rp["regression_train_params"]

            # propagate rank into config (leave as None if not a low-rank run)
            if rank is not None:
                rp["lowrank_rank"] = rank

            # set RMS weight
            tp["use_rms_correlation_loss"] = True
            tp["rms_correlation_weight"]   = float(rms_w)

            # rebuild the same augmentation schedule
            sched = build_combined_schedule(
                q_lo=params["q_lo"], q_hi=params["q_hi"],
                wm_alpha_min=params["wm"]["alpha_min"], wm_alpha_max=params["wm"]["alpha_max"], wm_gamma=params["wm"]["gamma"],
                gm_alpha_min=params["gm"]["alpha_min"], gm_alpha_max=params["gm"]["alpha_max"], gm_gamma=params["gm"]["gamma"],
                csf_alpha_min=params["csf"]["alpha_min"], csf_alpha_max=params["csf"]["alpha_max"], csf_gamma=params["csf"]["gamma"],
            )
            tp["uncertainty_schedule"] = sched

            def fp(v): return str(v).replace(".", "p")

            # include rank in the tag so it's traceable
            rank_tag = f"__r{rank}" if rank is not None else ""
            tag = (
                f"rms_sweep__acc{acc_rate}"
                f"__src{src_idx:02d}"
                f"{rank_tag}"
                f"__rmsw{fp(rms_w)}"
                f"__q{fp(params['q_lo'])}-{fp(params['q_hi'])}"
                f"__wm{fp(params['wm']['alpha_min'])}-{fp(params['wm']['alpha_max'])}g{fp(params['wm']['gamma'])}"
                f"__gm{fp(params['gm']['alpha_min'])}-{fp(params['gm']['alpha_max'])}g{fp(params['gm']['gamma'])}"
                f"__csf{fp(params['csf']['alpha_min'])}-{fp(params['csf']['alpha_max'])}g{fp(params['csf']['gamma'])}"
            )
            rp["run_tag"]          = tag
            rp["sweep_type"]       = "rms_correlation"
            rp["sweep_src_idx"]    = src_idx
            rp["sweep_rms_weight"] = float(rms_w)
            rp["sweep_src_rank"]   = rank  # bookkeeping

            cfgs.append((tag, cfg))

    return cfgs


# =============================================================================
# Save helpers — identical structure to other sweep scripts
# =============================================================================
def save_configs(
    cfgs:         List[Tuple[str, dict]],
    out_dir:      Path,
    n_batches:    int,
    batch_prefix: str = "batch_rms",
):
    all_dir = out_dir / "all_configs"
    all_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in cfgs:
        with open(all_dir / f"{name}.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    eff   = out_dir / "efficient_running"
    eff.mkdir(parents=True, exist_ok=True)
    total = len(cfgs)
    base  = total // n_batches
    rem   = total % n_batches
    sizes = [base + (1 if i < rem else 0) for i in range(n_batches)]
    s = 0
    for bi, sz in enumerate(sizes, start=1):
        if sz == 0:
            continue
        bdir = eff / f"{batch_prefix}_{bi:02d}"
        bdir.mkdir(parents=True, exist_ok=True)
        for name, cfg in cfgs[s: s + sz]:
            with open(bdir / f"{name}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        s += sz

    print(f"  Wrote {total} configs in {n_batches} batches -> {out_dir}")


def write_sbatch(
    out_dir:     Path,
    sbatch_root: Path,
    exp_name:    str,
    acc_rate:    int,
    subfolder:   str = "rms_sweep",
):
    eff = out_dir / "efficient_running"
    if not eff.exists():
        return

    batch_dirs = sorted([p for p in eff.iterdir() if p.is_dir()])
    dst = sbatch_root / subfolder / f"acc_rate_{acc_rate}"
    dst.mkdir(parents=True, exist_ok=True)

    for bdir in batch_dirs:
        try:
            batch_num = int(bdir.name.split("_")[-1])
        except Exception:
            continue

        config_abs = str(bdir.resolve())
        job_name   = f"{exp_name}_rms_{batch_num:02d}_acc{acc_rate}"

        script = f"""#!/bin/bash


source $(conda info --base)/etc/profile.d/conda.sh
conda activate cupa_T2_star

cd {PROJECT_ROOT}

export PYTHONPATH="{REPO_ROOT}"

python {PROJECT_ROOT}/run_group.py \\
    --config_path {config_abs}
"""
        script_path = dst / f"batch_{batch_num:02d}_acc{acc_rate}.sbatch"
        script_path.write_text(script)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       type=str, required=True,
                    help="CSV from scan_tissue_results.py (combined sweep)")
    ap.add_argument("--top_k",     type=int, default=3)
    ap.add_argument("--exp",       type=str, default="")
    ap.add_argument("--acc_rates", type=str, default="2")
    ap.add_argument("--n_batches", type=int, default=13,
                    help="27 runs / 3 batches = 9 runs each — fits in one GPU job")
    ap.add_argument("--out_root",  type=str,
                    default=str(PROJECT_ROOT / "configs" / "sweeps_rms"))
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    exp_name  = args.exp
    acc_rates = [int(x) for x in args.acc_rates.split(",") if x.strip()]
    ts        = time.strftime("%Y%m%d_%H%M%S")

    base_out    = Path(args.out_root) / exp_name / f"rms_sweep_{ts}"
    sbatch_root = base_out / "sbatch_scripts"
    base_out.mkdir(parents=True, exist_ok=True)

    summary = []

    for acc in acc_rates:
        print(f"\n{'='*60}")
        print(f"  RMS sweep | acc={acc} | exp={exp_name} | top_k={args.top_k}")
        print(f"  Weights: {RMS_WEIGHTS}")
        print(f"  Total runs: {args.top_k} × {len(RMS_WEIGHTS)} = {args.top_k * len(RMS_WEIGHTS)}")
        print(f"{'='*60}")

        top_records = load_top_k_from_csv(csv_path, args.top_k, acc)
        cfgs = generate_rms_configs(top_records, exp_name, acc, RMS_WEIGHTS)

        out_dir = base_out / "rms_sweep" / f"acc_rate_{acc}"
        save_configs(cfgs, out_dir, n_batches=args.n_batches)
        write_sbatch(out_dir, sbatch_root, exp_name, acc)

        # ── Baselines ────────────────────────────────────────────────────
        print(f"\n  Generating baselines for acc={acc}...")
        baseline_cfgs = generate_baselines(exp_name, acc)

        baseline_dir = base_out / "baselines" / f"acc_rate_{acc}"
        save_configs(baseline_cfgs, baseline_dir, n_batches=1, batch_prefix="batch_baseline")
        write_sbatch(baseline_dir, sbatch_root, exp_name, acc, subfolder="baselines")

        print(f"\n  Baselines generated:")
        for name, _ in baseline_cfgs:
            print(f"    - {name}.yaml")

        summary.append({
            "acc_rate":     acc,
            "n_configs":    len(cfgs),
            "n_baselines":  len(baseline_cfgs),
            "final_dir":    str(out_dir),
            "baseline_dir": str(baseline_dir),
        })

    summary_path = base_out / "rms_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("RMS SWEEP SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        print(f"  acc={s['acc_rate']} | n={s['n_configs']} | {s['final_dir']}")
    print(f"\n  Summary: {summary_path}")
    print(f"\nSubmit all:")
    for acc in acc_rates:
        print(f"  for f in {sbatch_root}/rms_sweep/acc_rate_{acc}/*.sbatch; do sbatch $f; done")


if __name__ == "__main__":
    main()