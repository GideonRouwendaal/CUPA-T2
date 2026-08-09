#!/usr/bin/env python3
"""
scan_and_evaluate.py

Scans all .log files in a directory (recursively), parses all runs, then:

  - Selects the top-1 BASELINE (PUQ) run:
      Uncertainty Mode = none_concat, Heteroscedastic = False, MC Dropout = False
  - Selects the top-1 BASELINE (Hetero) run:
      Uncertainty Mode = none_concat, Heteroscedastic = True,  MC Dropout = True
  - Selects the top-1 SWEEP run (all non-none_concat uncertainty modes):
      Best val_nrmse overall

  Top-1 is determined by lowest Best Validation NRMSE.

  For each selected run, prints a full config summary including rms_weight,
  heteroscedastic, mc_dropout, output_dir, uncertainty schedule, and tissue NRMSEs.

  Optionally evaluates the selected runs using a RegressionEvaluator loaded from
  a user-supplied evaluator .py file (same interface as the first script).

Usage
-----
  # Dry-run: just print selected runs
  python scan_and_evaluate.py --log_dir /path/to/logs --dry_run

  # Full evaluation
  python scan_and_evaluate.py \
      --log_dir /path/to/logs \
      --evaluator_py /path/to/regression_evaluator.py \
      --test_data_base /path/to/data \
      --exp_root /path/to/results/regression \
      --out_root /path/to/evaluations \
      --device cuda:0
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path anchors (derived from this file's location, so the tree can be moved)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
RESULTS_REGRESSION = REPO_ROOT / "results" / "regression"
DEFAULT_TEST_DATA_BASE = CUPA_ROOT / "data" / "T2_param_data" / "With_Unc"

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
RUN_START_RE = re.compile(r"\[\d+/\d+\] Running:\s*(\S+\.yaml)")

_RE_VAL_NRMSE   = re.compile(r"Best Validation NRMSE:\s*([\d.eE+\-]+)")
_RE_WM          = re.compile(r"Best Checkpoint T2\* \(WM\):\s*([\d.eE+\-]+)")
_RE_GM          = re.compile(r"Best Checkpoint T2\* \(GM\):\s*([\d.eE+\-]+)")
_RE_CSF         = re.compile(r"Best Checkpoint T2\* \(CSF\):\s*([\d.eE+\-]+)")
_RE_MODE        = re.compile(r"Uncertainty Mode:\s*([A-Za-z0-9_\-]+)")
_RE_HETERO      = re.compile(r"Heteroscedastic:\s*(True|False)", re.IGNORECASE)
_RE_MCDROP      = re.compile(r"MC Dropout:\s*(True|False)", re.IGNORECASE)
_RE_RMS_WEIGHT  = re.compile(r"Using RMS correlation loss with weight\s+([\d.eE+\-]+|None)")
_RE_OUT_DIR     = re.compile(r"Output directory:\s*(\S+)")

# Schedule block (everything after "Uncertainty Schedule Config:" until blank line or next section)
_RE_SCHED_START = re.compile(r"Uncertainty Schedule Config:")


def _f(m: Optional[re.Match]) -> Optional[float]:
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _b(m: Optional[re.Match]) -> bool:
    if not m:
        return False
    return m.group(1).lower() == "true"


# ---------------------------------------------------------------------------
# Per-run record
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    # Identity
    yaml_name: str
    log_file: str
    log_path: str
    run_index: int          # 1-based within the batch log

    # Key metrics
    val_nrmse: Optional[float]
    wm_nrmse: Optional[float]
    gm_nrmse: Optional[float]
    csf_nrmse: Optional[float]

    # Model settings (parsed from log lines, NOT from path)
    uncertainty_mode: Optional[str]
    heteroscedastic: bool
    use_mc_dropout: bool
    rms_weight: Optional[float]   # None means weight=None (not used)

    # Output directory (for checkpoint finding)
    output_dir: Optional[str]

    # Raw schedule config text block
    schedule_config_raw: str = ""

    # Parsed schedule key-values (flat)
    schedule_kv: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_scalar(v: str) -> Any:
    s = v.strip()
    if s.lower() in {"none", "null"}:
        return None
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_schedule_block(chunk: str) -> Tuple[str, Dict[str, Any]]:
    """Extract uncertainty schedule config as raw text and flat key-value dict."""
    m = _RE_SCHED_START.search(chunk)
    if not m:
        return "", {}

    lines = chunk[m.end():].splitlines()
    raw_lines: List[str] = []
    kv: Dict[str, Any] = {}
    kv_re = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)\s*:\s*(.*)\s*$")

    # We stop at a blank line that is followed by a non-indented section header,
    # or at a line that clearly belongs to summary output (e.g. "Best WM NRMSE")
    stop_re = re.compile(r"^(Best |Training |Validation |Average |Early |===|Training completed|\u2705)")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            raw_lines.append(line)
            continue
        if stop_re.match(stripped):
            break
        raw_lines.append(line)
        mm = kv_re.match(line)
        if mm:
            indent = len(mm.group(1))
            k = mm.group(2).strip()
            v = _parse_scalar(mm.group(3))
            # Only keep top-level (non-indented) and one-level-indented keys
            if indent <= 2:
                kv[k] = v

    return "\n".join(raw_lines).strip(), kv


def parse_run_chunk(yaml_name: str, chunk: str, log_file: str,
                    log_path: str, run_index: int) -> Optional[RunRecord]:
    """Parse one run chunk from a batch log."""

    # Must have a Training Summary block to be considered complete
    if "Training Summary" not in chunk:
        return None

    # --- Metrics ---
    val   = _f(_RE_VAL_NRMSE.search(chunk))
    wm    = _f(_RE_WM.search(chunk))
    gm    = _f(_RE_GM.search(chunk))
    csf   = _f(_RE_CSF.search(chunk))

    # Need at least val_nrmse to rank
    if val is None:
        return None

    # --- Mode & settings (from model initialization log lines) ---
    mode_m = _RE_MODE.search(chunk)
    uncertainty_mode = mode_m.group(1) if mode_m else None

    hetero  = _b(_RE_HETERO.search(chunk))
    mcdrop  = _b(_RE_MCDROP.search(chunk))

    rms_m = _RE_RMS_WEIGHT.search(chunk)
    if rms_m:
        val_str = rms_m.group(1)
        rms_weight = None if val_str.lower() == "none" else float(val_str)
    else:
        rms_weight = None

    # --- Output directory ---
    out_m = _RE_OUT_DIR.search(chunk)
    output_dir = out_m.group(1).rstrip("/") if out_m else None

    # --- Schedule block ---
    sched_raw, sched_kv = _parse_schedule_block(chunk)

    return RunRecord(
        yaml_name=yaml_name,
        log_file=log_file,
        log_path=log_path,
        run_index=run_index,
        val_nrmse=val,
        wm_nrmse=wm,
        gm_nrmse=gm,
        csf_nrmse=csf,
        uncertainty_mode=uncertainty_mode,
        heteroscedastic=hetero,
        use_mc_dropout=mcdrop,
        rms_weight=rms_weight,
        output_dir=output_dir,
        schedule_config_raw=sched_raw,
        schedule_kv=sched_kv,
    )


def parse_log(path: Path) -> List[RunRecord]:
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return []

    matches = list(RUN_START_RE.finditer(text))
    if not matches:
        # Single run, no batch header
        chunks = [("unknown.yaml", text)]
    else:
        chunks = []
        for i, m in enumerate(matches):
            yaml_name = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append((yaml_name, text[start:end]))

    records = []
    for idx, (yaml_name, chunk) in enumerate(chunks, start=1):
        r = parse_run_chunk(yaml_name, chunk, path.name, str(path), idx)
        if r is not None:
            records.append(r)
    return records


def scan_logs(log_dir: Path) -> List[RunRecord]:
    log_files = sorted(log_dir.rglob("*.log"))
    if not log_files:
        print(f"[ERROR] No .log files found under {log_dir}")
        return []

    print(f"Found {len(log_files)} .log files under {log_dir}\n")
    all_records: List[RunRecord] = []
    for p in log_files:
        recs = parse_log(p)
        all_records.extend(recs)
        print(f"  {p.name}: {len(recs)} completed runs parsed")

    print(f"\nTotal completed runs: {len(all_records)}\n")
    return all_records


# ---------------------------------------------------------------------------
# Baseline classification
# ---------------------------------------------------------------------------

def is_puq_baseline(r: RunRecord) -> bool:
    """PUQ: none_concat, NOT heteroscedastic, NOT mc_dropout."""
    return (
        r.uncertainty_mode == "none_concat"
        and not r.heteroscedastic
        and not r.use_mc_dropout
    )


def is_hetero_baseline(r: RunRecord) -> bool:
    """Hetero: none_concat, heteroscedastic=True, mc_dropout=True."""
    return (
        r.uncertainty_mode == "none_concat"
        and r.heteroscedastic
        and r.use_mc_dropout
    )


def is_sweep_run(r: RunRecord) -> bool:
    """Any non-none_concat run."""
    return r.uncertainty_mode != "none_concat"


# ---------------------------------------------------------------------------
# Selection: top-1 by val_nrmse (lowest)
# ---------------------------------------------------------------------------

def _safe_val(r: RunRecord) -> float:
    v = r.val_nrmse
    return v if (v is not None and not math.isnan(v)) else float("inf")


def top1(records: List[RunRecord]) -> Optional[RunRecord]:
    if not records:
        return None
    return min(records, key=_safe_val)


# ---------------------------------------------------------------------------
# Pretty-print config
# ---------------------------------------------------------------------------

def print_run_config(label: str, r: RunRecord) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  yaml_name        : {r.yaml_name}")
    print(f"  log_file         : {r.log_file}")
    print(f"  run_index        : {r.run_index}")
    print()
    print(f"  uncertainty_mode : {r.uncertainty_mode}")
    print(f"  heteroscedastic  : {r.heteroscedastic}")
    print(f"  use_mc_dropout   : {r.use_mc_dropout}")
    print(f"  rms_weight       : {r.rms_weight}")
    print()
    print(f"  val_nrmse        : {r.val_nrmse}")
    print(f"  wm_nrmse         : {r.wm_nrmse}")
    print(f"  gm_nrmse         : {r.gm_nrmse}")
    print(f"  csf_nrmse        : {r.csf_nrmse}")
    mean_wm_gm = (
        (r.wm_nrmse + r.gm_nrmse) / 2
        if r.wm_nrmse is not None and r.gm_nrmse is not None
        else None
    )
    print(f"  mean_wm_gm_nrmse : {mean_wm_gm:.6f}" if mean_wm_gm is not None else "  mean_wm_gm_nrmse : N/A")
    print()
    print(f"  output_dir       : {r.output_dir}")
    print()

    if r.schedule_config_raw:
        print("  Uncertainty Schedule Config:")
        for line in r.schedule_config_raw.splitlines():
            print(f"    {line}")
    else:
        print("  Uncertainty Schedule Config: (none)")

    print(sep)


# ---------------------------------------------------------------------------
# Checkpoint finding (mirrors first script)
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
# Build test config from RunRecord (adapted from first script)
# ---------------------------------------------------------------------------

def build_test_config(r, test_data_base, acc_rate_override=None):
    # Infer acc_rate from output_dir path
    acc_rate_int = acc_rate_override  # use explicit if provided
    if acc_rate_int is None:
        acc_rate_int = 2  # fallback
        if r.output_dir:
            m = re.search(r"acc_rate_(\d+)", r.output_dir)
            if m:
                acc_rate_int = int(m.group(1))
    lowrank_rank = None
    if r.uncertainty_mode and "low_rank" in r.uncertainty_mode.lower():
        lowrank_rank = 10

    cfg = {
        "task": "test_regression",
        "include_uncertainty": True,
        "max_t2": 200.0,
        "min_t2": 0.0,
        "s0_max": 2.5,
        "s0_min": 0.0,
        "regression_params": {
            "regression_gt_test_location": f"{test_data_base}/acc_rate_{acc_rate_int}/test/",
            "acc_rate": acc_rate_int,
            "regress_model_dir": str(RESULTS_REGRESSION) + "/",
            "uncertainty_mode": r.uncertainty_mode,
            "lowrank_rank": lowrank_rank,
            "heteroscedastic": r.heteroscedastic,
            "use_mc_dropout": r.use_mc_dropout,
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
                "dropout_rate": 0.2 if r.use_mc_dropout else 0.0,
            },
            "regression_train_params": {
                "loss": "L2",
                "use_rms_correlation_loss": r.rms_weight is not None and r.rms_weight > 0,
                "rms_correlation_weight": r.rms_weight if r.rms_weight is not None else 0.0,
                "use_cv_correlation_loss": False,
                "cv_correlation_weight": 0.1,
                "use_consistency_loss": False,
                "consistency_weight": 0.1,
                "uncertainty_schedule": {},
            },
        },
    }
    return cfg


# ---------------------------------------------------------------------------
# Evaluator helpers (from first script)
# ---------------------------------------------------------------------------

def import_evaluator(py_path: Path):
    spec = importlib.util.spec_from_file_location("user_regression_evaluator", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import evaluator from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize_metrics(metrics_by_type: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    import math
    out: Dict[str, float] = {}

    def add(prefix, d, keys):
        for k in keys:
            vals = [v for v in d.get(k, []) if v is not None and not math.isnan(v)]
            if not vals:
                out[f"{prefix}_{k}_mean"] = float("nan")
                out[f"{prefix}_{k}_std"] = float("nan")
            else:
                mu = sum(vals) / len(vals)
                out[f"{prefix}_{k}_mean"] = mu
                out[f"{prefix}_{k}_std"] = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))

    add("t2", metrics_by_type.get("t2", {}), ["nrmse", "mae", "ssim", "psnr"])
    add("recon_t2", metrics_by_type.get("recon_t2", {}), ["nrmse", "mae", "ssim", "psnr"])
    add("t2p", metrics_by_type.get("t2_probabilistic", {}), [
        "nll", "coverage_1sigma", "coverage_2sigma", "coverage_3sigma",
        "spearman_err_unc", "aurc", "cross_stage_consistency",
    ])
    for tissue in ["wm", "gm", "csf"]:
        add(f"{tissue}_t2", metrics_by_type.get(f"{tissue}_t2", {}), ["nrmse"])
        add(f"{tissue}_recon_t2", metrics_by_type.get(f"{tissue}_recon_t2", {}), ["nrmse"])
        add(f"{tissue}_t2p", metrics_by_type.get(f"{tissue}_t2_probabilistic", {}), [
            "nll", "spearman_err_unc", "aurc", "cross_stage_consistency",
        ])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log_dir",       type=Path, required=True,
                    help="Directory containing .log files (searched recursively)")
    ap.add_argument("--evaluator_py",  type=Path, default=None,
                    help="Path to regression_evaluator.py (omit for dry-run only)")
    ap.add_argument("--test_data_base", type=str,
                    default=str(DEFAULT_TEST_DATA_BASE),
                    help="Base path for test data (appended with /acc_rate_X/test/)")
    ap.add_argument("--exp_root",      type=Path, default=None,
                    help="Experiment root for computing relative output paths")
    ap.add_argument("--out_root",      type=Path, default=None,
                    help="Where to write evaluation outputs")
    ap.add_argument("--results_csv",   type=Path, default=None,
                    help="Where to write evaluation summary CSV")
    ap.add_argument("--device",        type=str,  default="cuda:0")
    ap.add_argument("--num_workers",   type=int,  default=4)
    ap.add_argument("--to_plot",       action="store_true")
    ap.add_argument("--dry_run",       action="store_true",
                    help="Print selected runs and configs only, do not evaluate")
    ap.add_argument("--acc_rate", type=int, default=None,
                    help="Acceleration rate (overrides auto-detection from output_dir)")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # 1) Scan logs
    # ------------------------------------------------------------------
    all_records = scan_logs(args.log_dir)
    if not all_records:
        raise SystemExit("No completed runs found.")

    # ------------------------------------------------------------------
    # 2) Split into categories
    # ------------------------------------------------------------------
    puq_runs    = [r for r in all_records if is_puq_baseline(r)]
    hetero_runs = [r for r in all_records if is_hetero_baseline(r)]
    sweep_runs  = [r for r in all_records if is_sweep_run(r)]

    print(f"Baseline (PUQ)   runs found : {len(puq_runs)}")
    print(f"Baseline (Hetero) runs found: {len(hetero_runs)}")
    print(f"Sweep runs found            : {len(sweep_runs)}")

    # ------------------------------------------------------------------
    # 3) Select top-1 from each category
    # ------------------------------------------------------------------
    best_puq    = top1(puq_runs)
    best_hetero = top1(hetero_runs)
    best_sweep  = top1(sweep_runs)

    # ------------------------------------------------------------------
    # 4) Print full configs
    # ------------------------------------------------------------------
    if best_puq:
        print_run_config("TOP-1  BASELINE (PUQ)  —  none_concat / homoscedastic / no dropout", best_puq)
    else:
        print("\n[WARNING] No PUQ baseline runs found.")

    if best_hetero:
        print_run_config("TOP-1  BASELINE (Hetero)  —  none_concat / heteroscedastic / mc_dropout", best_hetero)
    else:
        print("\n[WARNING] No Hetero baseline runs found.")

    if best_sweep:
        print_run_config("TOP-1  SWEEP RUN  —  best val_nrmse across all non-none_concat modes", best_sweep)
    else:
        print("\n[WARNING] No sweep runs found.")

    # ------------------------------------------------------------------
    # 5) Collect selected runs (skip Nones)
    # ------------------------------------------------------------------
    selected: List[Tuple[str, RunRecord]] = []
    if best_puq:
        selected.append(("BASELINE_PUQ", best_puq))
    if best_hetero:
        selected.append(("BASELINE_Hetero", best_hetero))
    if best_sweep:
        selected.append(("SWEEP_TOP1", best_sweep))

    if args.dry_run or args.evaluator_py is None:
        print("\nDRY RUN — not evaluating. Pass --evaluator_py to enable evaluation.")
        return

    # ------------------------------------------------------------------
    # 6) Evaluate selected runs
    # ------------------------------------------------------------------
    ev_mod = import_evaluator(args.evaluator_py)
    RegressionEvaluator = getattr(ev_mod, "RegressionEvaluator")
    _device_from_arg    = getattr(ev_mod, "_device_from_arg")
    dev = _device_from_arg(args.device)

    out_root = args.out_root or (args.log_dir / "_evaluations")
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.results_csv or (out_root / "evaluated_runs.csv")

    rows_out: List[Dict[str, Any]] = []
    missing_ckpt: List[str] = []
    failed: List[str] = []

    for label, r in selected:
        if not r.output_dir:
            print(f"\n[{label}] SKIP — no output_dir found in log.")
            missing_ckpt.append(f"{label}: no output_dir")
            continue

        run_dir = Path(r.output_dir)
        ckpt = find_checkpoint(run_dir)
        if ckpt is None:
            print(f"\n[{label}] SKIP — no checkpoint in {run_dir}")
            missing_ckpt.append(str(run_dir))
            continue

        # Output sub-directory
        if args.exp_root and args.exp_root.exists():
            try:
                rel = run_dir.relative_to(args.exp_root)
                out_dir = out_root / rel
            except ValueError:
                out_dir = out_root / label
        else:
            out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{label}] EVALUATING")
        print(f"  uncertainty_mode : {r.uncertainty_mode}")
        print(f"  heteroscedastic  : {r.heteroscedastic}")
        print(f"  use_mc_dropout   : {r.use_mc_dropout}")
        print(f"  rms_weight       : {r.rms_weight}")
        print(f"  val_nrmse        : {r.val_nrmse}")
        print(f"  ckpt             : {ckpt}")
        print(f"  out_dir          : {out_dir}")

        try:
            cfg = build_test_config(r, test_data_base=args.test_data_base, acc_rate_override=args.acc_rate)
            ev = RegressionEvaluator(cfg, dev)
            ev.model_path_override  = str(ckpt)
            ev.output_dir_override  = str(out_dir)
            ev.to_plot              = bool(args.to_plot)

            metrics = ev.evaluate()
            summ    = summarize_metrics(metrics)

            row: Dict[str, Any] = {
                "label":            label,
                "yaml_name":        r.yaml_name,
                "log_file":         r.log_file,
                "output_dir":       r.output_dir,
                "checkpoint_path":  str(ckpt),
                "eval_out_dir":     str(out_dir),
                "uncertainty_mode": r.uncertainty_mode,
                "heteroscedastic":  r.heteroscedastic,
                "use_mc_dropout":   r.use_mc_dropout,
                "rms_weight":       r.rms_weight,
                "rank_val_nrmse":   r.val_nrmse,
                "rank_wm_nrmse":    r.wm_nrmse,
                "rank_gm_nrmse":    r.gm_nrmse,
                "rank_csf_nrmse":   r.csf_nrmse,
            }
            row.update(summ)
            rows_out.append(row)

            with (out_dir / "eval_summary.json").open("w") as f:
                json.dump(row, f, indent=2)

            print(f"  ✓ Done. Summary written to {out_dir / 'eval_summary.json'}")

        except Exception:
            failed.append(f"{label}: {r.output_dir}")
            print(f"  FAILED: {r.output_dir}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 7) Write master CSV
    # ------------------------------------------------------------------
    if rows_out:
        base_cols = [
            "label", "yaml_name", "log_file",
            "uncertainty_mode", "heteroscedastic", "use_mc_dropout", "rms_weight",
            "rank_val_nrmse", "rank_wm_nrmse", "rank_gm_nrmse", "rank_csf_nrmse",
            "output_dir", "checkpoint_path", "eval_out_dir",
        ]
        metric_cols = sorted(c for c in rows_out[0] if c not in base_cols)
        fieldnames = base_cols + metric_cols

        with results_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows_out:
                w.writerow(row)

        print(f"\n✓ Evaluation summary CSV -> {results_csv.resolve()}")
    else:
        print("\nNo successful evaluations to write.")

    if missing_ckpt:
        print("\nMissing checkpoints / output dirs:")
        for d in missing_ckpt:
            print("  -", d)
    if failed:
        print("\nFailed evaluations:")
        for d in failed:
            print("  -", d)


if __name__ == "__main__":
    main()