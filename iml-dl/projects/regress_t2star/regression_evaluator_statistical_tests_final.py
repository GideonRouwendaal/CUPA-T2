#!/usr/bin/env python3
"""
scan_and_test.py

Combines the run-selection logic of scan_and_evaluate.py with the
statistical-testing logic of the Wilcoxon script.

Steps
-----
1. Scan .log files under --log_dir and identify:
     - TOP-1 BASELINE (PUQ)  : none_concat, homoscedastic, no dropout
     - TOP-1 BASELINE (Hetero): none_concat, heteroscedastic, mc_dropout
     - TOP-1 SWEEP RUN       : best val_nrmse among all non-none_concat modes

2. For each selected run, locate its per_slice_tissue_metrics.csv:
     - First looks inside the run's training output_dir.
     - Then looks inside --eval_root/<relative-or-flat path>/.
     - Also accepts explicit overrides via --puq_eval_dir / --hetero_eval_dir / --sweep_eval_dir.

3. Run paired Wilcoxon tests (sweep vs PUQ, sweep vs Hetero where available)
   for every requested (metric, tissue) combination, with Holm–Bonferroni correction.

4. Write full results CSV and compact summary CSVs (WM-best / GM-best / Overall-best).

Usage examples
--------------
# Minimal — eval CSVs are inside the training output_dir
python scan_and_test.py \\
    --log_dir /path/to/logs \\
    --out_csv results/wilcoxon.csv

# Eval CSVs are stored under a separate eval_root tree
python scan_and_test.py \\
    --log_dir /path/to/logs \\
    --eval_root /path/to/evaluations/Final/Acc4 \\
    --out_csv results/wilcoxon.csv

# Override eval dirs explicitly (e.g. the directory that contains the CSV)
python scan_and_test.py \\
    --log_dir /path/to/logs \\
    --puq_eval_dir  /path/to/evaluations/Final/Acc4/BASELINE_PUQ \\
    --sweep_eval_dir /path/to/evaluations/Final/Acc4/SWEEP_TOP1 \\
    --out_csv results/wilcoxon.csv
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

# ============================================================
# ── SECTION 1: Log parsing (from scan_and_evaluate.py) ──────
# ============================================================

RUN_START_RE    = re.compile(r"\[\d+/\d+\] Running:\s*(\S+\.yaml)")
_RE_VAL_NRMSE   = re.compile(r"Best Validation NRMSE:\s*([\d.eE+\-]+)")
_RE_WM          = re.compile(r"Best Checkpoint T2\* \(WM\):\s*([\d.eE+\-]+)")
_RE_GM          = re.compile(r"Best Checkpoint T2\* \(GM\):\s*([\d.eE+\-]+)")
_RE_CSF         = re.compile(r"Best Checkpoint T2\* \(CSF\):\s*([\d.eE+\-]+)")
_RE_MODE        = re.compile(r"Uncertainty Mode:\s*([A-Za-z0-9_\-]+)")
_RE_HETERO      = re.compile(r"Heteroscedastic:\s*(True|False)", re.IGNORECASE)
_RE_MCDROP      = re.compile(r"MC Dropout:\s*(True|False)", re.IGNORECASE)
_RE_RMS_WEIGHT  = re.compile(r"Using RMS correlation loss with weight\s+([\d.eE+\-]+|None)")
_RE_OUT_DIR     = re.compile(r"Output directory:\s*(\S+)")
_RE_SCHED_START = re.compile(r"Uncertainty Schedule Config:")


def _f(m: Optional[re.Match]) -> Optional[float]:
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _b(m: Optional[re.Match]) -> bool:
    return bool(m) and m.group(1).lower() == "true"


@dataclass
class RunRecord:
    yaml_name: str
    log_file: str
    log_path: str
    run_index: int
    val_nrmse: Optional[float]
    wm_nrmse: Optional[float]
    gm_nrmse: Optional[float]
    csf_nrmse: Optional[float]
    uncertainty_mode: Optional[str]
    heteroscedastic: bool
    use_mc_dropout: bool
    rms_weight: Optional[float]
    output_dir: Optional[str]


def parse_run_chunk(yaml_name, chunk, log_file, log_path, run_index) -> Optional[RunRecord]:
    if "Training Summary" not in chunk:
        return None
    val = _f(_RE_VAL_NRMSE.search(chunk))
    if val is None:
        return None
    wm  = _f(_RE_WM.search(chunk))
    gm  = _f(_RE_GM.search(chunk))
    csf = _f(_RE_CSF.search(chunk))
    mode_m = _RE_MODE.search(chunk)
    uncertainty_mode = mode_m.group(1) if mode_m else None
    hetero = _b(_RE_HETERO.search(chunk))
    mcdrop = _b(_RE_MCDROP.search(chunk))
    rms_m = _RE_RMS_WEIGHT.search(chunk)
    if rms_m:
        vs = rms_m.group(1)
        rms_weight = None if vs.lower() == "none" else float(vs)
    else:
        rms_weight = None
    out_m = _RE_OUT_DIR.search(chunk)
    output_dir = out_m.group(1).rstrip("/") if out_m else None
    return RunRecord(
        yaml_name=yaml_name, log_file=log_file, log_path=log_path,
        run_index=run_index, val_nrmse=val, wm_nrmse=wm, gm_nrmse=gm,
        csf_nrmse=csf, uncertainty_mode=uncertainty_mode,
        heteroscedastic=hetero, use_mc_dropout=mcdrop,
        rms_weight=rms_weight, output_dir=output_dir,
    )


def parse_log(path: Path) -> List[RunRecord]:
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return []
    matches = list(RUN_START_RE.finditer(text))
    chunks = []
    if not matches:
        chunks = [("unknown.yaml", text)]
    else:
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append((m.group(1), text[start:end]))
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


def _safe_val(r: RunRecord) -> float:
    v = r.val_nrmse
    return v if (v is not None and not math.isnan(v)) else float("inf")


def top1(records: List[RunRecord]) -> Optional[RunRecord]:
    return min(records, key=_safe_val) if records else None


def is_puq_baseline(r: RunRecord) -> bool:
    return r.uncertainty_mode == "none_concat" and not r.heteroscedastic and not r.use_mc_dropout


def is_hetero_baseline(r: RunRecord) -> bool:
    return r.uncertainty_mode == "none_concat" and r.heteroscedastic and r.use_mc_dropout


def is_sweep_run(r: RunRecord) -> bool:
    return r.uncertainty_mode != "none_concat"


def print_selected_runs(best_puq, best_hetero, best_sweep):
    def _show(label, r):
        if r is None:
            print(f"  [{label}] NOT FOUND")
            return
        print(f"  [{label}]  mode={r.uncertainty_mode}  hetero={r.heteroscedastic}"
              f"  mc_dropout={r.use_mc_dropout}  val_nrmse={r.val_nrmse:.6f}"
              f"  WM={r.wm_nrmse}  GM={r.gm_nrmse}  CSF={r.csf_nrmse}")
        print(f"     output_dir: {r.output_dir}")

    print("\n" + "=" * 70)
    print("SELECTED RUNS")
    print("=" * 70)
    _show("BASELINE_PUQ  ", best_puq)
    _show("BASELINE_HETERO", best_hetero)
    _show("SWEEP_TOP1    ", best_sweep)
    print("=" * 70 + "\n")


# ============================================================
# ── SECTION 2: CSV lookup ────────────────────────────────────
# ============================================================

METRICS_FILENAME = "per_slice_tissue_metrics.csv"


def find_metrics_csv(
    run_dir: Optional[str],
    eval_root: Optional[Path] = None,
    override_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Search order:
      1. explicit override_dir (if given)
      2. run_dir (training output dir) and its children
      3. eval_root searched by flat label name matching run_dir's last component
      4. eval_root searched globally
    """
    fname = METRICS_FILENAME

    # 1. Explicit override
    if override_dir is not None:
        direct = override_dir / fname
        if direct.exists():
            print(f"    CSV found (override): {direct}")
            return direct
        hits = list(override_dir.glob(f"**/{fname}"))
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"    CSV found (override glob): {hits[0]}")
            return hits[0]
        print(f"    [WARN] override_dir supplied but CSV not found in {override_dir}")

    # 2. Training output_dir
    if run_dir:
        rd = Path(run_dir)
        direct = rd / fname
        if direct.exists():
            print(f"    CSV found (output_dir): {direct}")
            return direct
        hits = list(rd.glob(f"**/{fname}"))
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"    CSV found (output_dir glob): {hits[0]}")
            return hits[0]

    # 3. eval_root by run_dir's anchor folder name
    if eval_root is not None and run_dir:
        anchor = Path(run_dir).name
        hits = list(eval_root.glob(f"**/{anchor}/**/{fname}"))
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"    CSV found (eval_root anchor): {hits[0]}")
            return hits[0]

    # 4. eval_root global search
    if eval_root is not None:
        hits = list(eval_root.glob(f"**/{fname}"))
        if hits:
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print(f"    CSV found (eval_root global): {hits[0]}")
            return hits[0]

    print(f"    [WARN] CSV not found for run_dir={run_dir}")
    return None


# ============================================================
# ── SECTION 3: Statistical tests (from Wilcoxon script) ─────
# ============================================================

def load_metrics(csv_path: Path, split: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "tissue" in df.columns:
        df["tissue"] = df["tissue"].astype(str).str.lower()
    if "split" in df.columns:
        df = df[df["split"].astype(str) == split]
    if "method" in df.columns:
        df = df[df["method"].astype(str) == method]
    return df


def aggregate_units(
    df: pd.DataFrame,
    metric: str,
    tissue: str,
    agg: str,
    unit: str,
    require_pixels: bool = True,
) -> pd.Series:
    tissue = tissue.lower()
    d = df[df["tissue"] == tissue].copy()
    if require_pixels and "n_pixels" in d.columns:
        d = d[d["n_pixels"].fillna(0) > 0]
    d = d.dropna(subset=[metric])

    slice_col = "slice" if "slice" in d.columns else ("slice_num" if "slice_num" in d.columns else None)

    if unit == "slice":
        if slice_col is None:
            raise ValueError("Slice-wise unit requested but no slice column found.")
        if agg == "mean":
            return d.groupby(["subject", slice_col])[metric].mean()
        return d.groupby(["subject", slice_col])[metric].median()
    elif unit == "subject":
        if slice_col is not None:
            d = d.groupby(["subject", slice_col], as_index=False)[metric].mean()
        if agg == "mean":
            return d.groupby("subject")[metric].mean()
        return d.groupby("subject")[metric].median()
    else:
        raise ValueError(f"unit must be 'subject' or 'slice', got: {unit}")


def wilcoxon_with_rbc(diffs: np.ndarray) -> Tuple[float, float, float, int]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    res = wilcoxon(diffs, zero_method="pratt", alternative="two-sided", method="auto")
    p = float(res.pvalue) if res.pvalue is not None else float("nan")
    nz = diffs != 0
    diffs_nz = diffs[nz]
    n = int(diffs_nz.size)
    if n == 0:
        return p, 0.0, 0.0, 0
    ranks = rankdata(np.abs(diffs_nz), method="average")
    wplus = float(ranks[diffs_nz > 0].sum())
    wminus = float(ranks[diffs_nz < 0].sum())
    total = n * (n + 1) / 2.0
    rbc = (wplus - wminus) / total
    return p, wplus, float(rbc), n


def holm_adjust(pvals: List[float]) -> List[float]:
    clean = [1.0 if (p is None or math.isnan(float(p))) else float(p) for p in pvals]
    m = len(clean)
    order = np.argsort(clean)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        p = clean[idx]
        a = min(1.0, max((m - k) * p, prev))
        adj[idx] = a
        prev = a
    return adj.tolist()


def lower_is_better(metric: str) -> bool:
    return metric.lower() in {"nrmse", "mae", "mse", "rmse", "nll", "gnll", "gaussian_nll"}


def run_tests(
    comparisons: List[Tuple[str, pd.DataFrame, str, pd.DataFrame]],
    # list of (baseline_label, baseline_df, model_label, model_df)
    metrics: List[str],
    tissues: List[str],
    agg: str,
    unit: str,
    min_subjects: int,
) -> pd.DataFrame:
    rows = []
    pvals = []
    agg_func = np.nanmean if agg == "mean" else np.nanmedian

    for base_label, base_df, model_label, model_df in comparisons:
        base_pred  = base_df[base_df["method"] == "pred"].copy() if "method" in base_df.columns else base_df.copy()
        model_pred = model_df[model_df["method"] == "pred"].copy() if "method" in model_df.columns else model_df.copy()

        for metric in metrics:
            for tissue in tissues:
                try:
                    s_base = aggregate_units(base_pred, metric=metric, tissue=tissue, agg=agg, unit=unit)
                    s_mod  = aggregate_units(model_pred, metric=metric, tissue=tissue, agg=agg, unit=unit)
                except Exception as e:
                    print(f"  SKIP ({base_label} vs {model_label}, {metric}/{tissue}): {e}")
                    continue

                common_idx = s_base.index.intersection(s_mod.index)
                if len(common_idx) < min_subjects:
                    print(f"  SKIP ({base_label} vs {model_label}, {metric}/{tissue}): "
                          f"only {len(common_idx)} paired units (need {min_subjects})")
                    continue

                try:
                    common_idx = common_idx.sort_values()
                except Exception:
                    common_idx = sorted(common_idx)

                x = s_base.loc[common_idx].to_numpy(dtype=float)
                y = s_mod.loc[common_idx].to_numpy(dtype=float)

                diffs = (x - y) if lower_is_better(metric) else (y - x)
                p, wplus, rbc, n_nonzero = wilcoxon_with_rbc(diffs)

                if isinstance(common_idx, pd.MultiIndex):
                    n_subj = int(pd.Index(common_idx.get_level_values(0)).nunique())
                else:
                    n_subj = int(pd.Index(common_idx).nunique())

                rows.append(dict(
                    baseline_label=base_label,
                    model_label=model_label,
                    metric=metric,
                    tissue=tissue.lower(),
                    agg=agg,
                    unit=unit,
                    n_units=len(common_idx),
                    n_subjects_unique=n_subj,
                    n_nonzero=n_nonzero,
                    baseline_agg=float(agg_func(x)),
                    model_agg=float(agg_func(y)),
                    median_model_minus_baseline=float(np.nanmedian(y - x)),
                    median_improvement=float(np.nanmedian(diffs)),
                    mean_improvement=float(np.nanmean(diffs)),
                    wplus=wplus,
                    rbc=rbc,
                    p_value=p,
                ))
                pvals.append(p)

    if not rows:
        return pd.DataFrame()

    padj = holm_adjust([float(r["p_value"]) for r in rows])
    for r, a in zip(rows, padj):
        r["p_holm"] = a

    out = pd.DataFrame(rows)
    out = out.sort_values(["baseline_label", "metric", "tissue", "p_holm"]).reset_index(drop=True)
    return out


# ============================================================
# ── SECTION 4: Summary tables ────────────────────────────────
# ============================================================

def build_summary_df(out: pd.DataFrame, criterion_tissue: str, p_col: str = "p_value") -> pd.DataFrame:
    """One row per (baseline_label, model_label) with WM/GM/CSF/SSIM columns."""
    rows_out = []
    for base_label in out["baseline_label"].unique():
        df_base = out[out["baseline_label"] == base_label]
        for model_label in df_base["model_label"].unique():
            df_pair = df_base[df_base["model_label"] == model_label]

            def get(metric, tissue):
                r = df_pair[
                    (df_pair["metric"].str.lower() == metric.lower()) &
                    (df_pair["tissue"].str.lower() == tissue.lower())
                ]
                if r.empty:
                    return float("nan"), float("nan"), float("nan")
                row = r.iloc[0]
                return float(row["baseline_agg"]), float(row["model_agg"]), float(row[p_col])

            b_wm,  m_wm,  p_wm  = get("nrmse", "wm")
            b_gm,  m_gm,  p_gm  = get("nrmse", "gm")
            b_csf, m_csf, p_csf = get("nrmse", "csf")
            b_ssim, m_ssim, p_ssim = get("ssim", "overall")

            rows_out.append({
                "baseline_label": base_label,
                "model_label": model_label,
                "baseline_wm_nrmse": b_wm, "model_wm_nrmse": m_wm, "p_wm": p_wm,
                "baseline_gm_nrmse": b_gm, "model_gm_nrmse": m_gm, "p_gm": p_gm,
                "baseline_csf_nrmse": b_csf, "model_csf_nrmse": m_csf, "p_csf": p_csf,
                "baseline_ssim": b_ssim, "model_ssim": m_ssim, "p_ssim": p_ssim,
            })

    return pd.DataFrame(rows_out)


# ============================================================
# ── MAIN ─────────────────────────────────────────────────────
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Log scanning ──
    ap.add_argument("--log_dir", type=Path, required=True,
                    help="Directory containing .log files (searched recursively)")

    # ── CSV location ──
    ap.add_argument("--eval_root", type=Path, default=None,
                    help="Root directory where evaluation outputs were written "
                         "(e.g. evaluations/Final/Acc4). "
                         "CSVs are searched here when not found in training output_dir.")
    ap.add_argument("--puq_eval_dir",   type=Path, default=None,
                    help="Explicit directory containing the PUQ baseline CSV (overrides auto-search)")
    ap.add_argument("--hetero_eval_dir", type=Path, default=None,
                    help="Explicit directory containing the Hetero baseline CSV (overrides auto-search)")
    ap.add_argument("--sweep_eval_dir",  type=Path, default=None,
                    help="Explicit directory containing the Sweep top-1 CSV (overrides auto-search)")

    # ── Test parameters ──
    ap.add_argument("--split",       type=str,  default="test")
    ap.add_argument("--method",      type=str,  default="pred", help="Value in the 'method' column to keep")
    ap.add_argument("--agg",         type=str,  default="median", choices=["mean", "median"])
    ap.add_argument("--metric",      action="append", default=None,
                    help="Metric column(s) to test (default: nrmse, ssim). Repeat to add more.")
    ap.add_argument("--tissue",      action="append", default=None,
                    help="Tissue value(s) to test (default: overall, wm, gm, csf). Repeat to add more.")
    ap.add_argument("--unit",        type=str,  default="subject", choices=["subject", "slice"],
                    help="Aggregate per subject (default) or per subject-slice pair.")
    ap.add_argument("--min_subjects", type=int, default=5,
                    help="Minimum paired units required to run a test.")

    # ── Output ──
    ap.add_argument("--out_csv",     type=Path, default=Path("wilcoxon_results.csv"))
    ap.add_argument("--summary_dir", type=Path, default=None,
                    help="Where to write summary CSVs (default: same folder as --out_csv).")
    ap.add_argument("--summary_p_col", type=str, default="p_value",
                    choices=["p_value", "p_holm"],
                    help="Which p-value to use in the summary table.")

    args = ap.parse_args()

    metrics = args.metric  or ["nrmse", "ssim"]
    tissues = args.tissue  or ["overall", "wm", "gm", "csf"]

    # ------------------------------------------------------------------
    # 1) Scan logs and select runs
    # ------------------------------------------------------------------
    all_records = scan_logs(args.log_dir)
    if not all_records:
        raise SystemExit("No completed runs found.")

    puq_runs    = [r for r in all_records if is_puq_baseline(r)]
    hetero_runs = [r for r in all_records if is_hetero_baseline(r)]
    sweep_runs  = [r for r in all_records if is_sweep_run(r)]

    print(f"Baseline (PUQ)    runs found: {len(puq_runs)}")
    print(f"Baseline (Hetero) runs found: {len(hetero_runs)}")
    print(f"Sweep runs found            : {len(sweep_runs)}")

    best_puq    = top1(puq_runs)
    best_hetero = top1(hetero_runs)
    best_sweep  = top1(sweep_runs)

    print_selected_runs(best_puq, best_hetero, best_sweep)

    if best_sweep is None:
        raise SystemExit("No sweep runs found – cannot compare.")

    # ------------------------------------------------------------------
    # 2) Locate per_slice_tissue_metrics.csv for each selected run
    # ------------------------------------------------------------------
    print("Locating per_slice_tissue_metrics.csv files …")

    puq_csv    = None
    hetero_csv = None
    sweep_csv  = None

    if best_puq:
        print(f"  PUQ baseline (output_dir: {best_puq.output_dir})")
        puq_csv = find_metrics_csv(best_puq.output_dir, args.eval_root, args.puq_eval_dir)

    if best_hetero:
        print(f"  Hetero baseline (output_dir: {best_hetero.output_dir})")
        hetero_csv = find_metrics_csv(best_hetero.output_dir, args.eval_root, args.hetero_eval_dir)

    print(f"  Sweep top-1 (output_dir: {best_sweep.output_dir})")
    sweep_csv = find_metrics_csv(best_sweep.output_dir, args.eval_root, args.sweep_eval_dir)

    if sweep_csv is None:
        raise SystemExit("Could not find the CSV for the sweep top-1 run. "
                         "Pass --sweep_eval_dir or --eval_root to help locate it.")

    if puq_csv is None and hetero_csv is None:
        raise SystemExit("Could not find CSVs for any baseline. "
                         "Pass --puq_eval_dir / --hetero_eval_dir or --eval_root.")

    print()

    # ------------------------------------------------------------------
    # 3) Load DataFrames
    # ------------------------------------------------------------------
    def load(csv_path, label):
        df = load_metrics(csv_path, split=args.split, method=args.method)
        print(f"  Loaded {label}: {len(df)} rows  (tissues: {sorted(df['tissue'].unique()) if 'tissue' in df.columns else 'n/a'})")
        return df

    print("Loading CSVs …")
    sweep_df  = load(sweep_csv,  "SWEEP_TOP1")
    puq_df    = load(puq_csv,    "BASELINE_PUQ")    if puq_csv    else None
    hetero_df = load(hetero_csv, "BASELINE_HETERO") if hetero_csv else None
    print()

    # ------------------------------------------------------------------
    # 4) Build comparison pairs: (baseline_label, baseline_df, model_label, model_df)
    # ------------------------------------------------------------------
    comparisons = []
    if puq_df is not None:
        comparisons.append(("BASELINE_PUQ", puq_df, "SWEEP_TOP1", sweep_df))
    if hetero_df is not None:
        comparisons.append(("BASELINE_HETERO", hetero_df, "SWEEP_TOP1", sweep_df))

    # ------------------------------------------------------------------
    # 5) Run Wilcoxon tests
    # ------------------------------------------------------------------
    print(f"Running Wilcoxon tests  (unit={args.unit}, agg={args.agg}) …")
    out = run_tests(
        comparisons=comparisons,
        metrics=metrics,
        tissues=tissues,
        agg=args.agg,
        unit=args.unit,
        min_subjects=args.min_subjects,
    )

    if out.empty:
        raise SystemExit("No tests were run. Check that the CSVs contain the requested metrics/tissues.")

    # ------------------------------------------------------------------
    # 6) Print and write results
    # ------------------------------------------------------------------
    show_cols = [
        "baseline_label", "model_label",
        "metric", "tissue", "n_units", "n_subjects_unique", "n_nonzero",
        "baseline_agg", "model_agg",
        "median_improvement", "rbc", "p_value", "p_holm",
    ]
    available_show = [c for c in show_cols if c in out.columns]
    print("\n" + out[available_show].to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n✓ Full results   → {args.out_csv.resolve()}")

    # ------------------------------------------------------------------
    # 7) Summary tables
    # ------------------------------------------------------------------
    summary_dir = args.summary_dir or args.out_csv.parent
    summary_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_csv.stem
    pcol = args.summary_p_col

    for crit_tissue, suffix in [("wm", "WM_best"), ("gm", "GM_best"), ("overall", "overall_best")]:
        s = build_summary_df(out, criterion_tissue=crit_tissue, p_col=pcol)
        if not s.empty:
            p = summary_dir / f"{stem}_summary_{suffix}.csv"
            s.to_csv(p, index=False)
            print(f"✓ Summary ({suffix}) → {p.resolve()}")

    # ------------------------------------------------------------------
    # 8) Print a compact human-readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPACT SUMMARY  (median performance ± improvement, p-value)")
    print("=" * 70)
    for base_label in out["baseline_label"].unique():
        for model_label in out["model_label"].unique():
            sub = out[(out["baseline_label"] == base_label) & (out["model_label"] == model_label)]
            if sub.empty:
                continue
            print(f"\n  {base_label}  vs  {model_label}")
            for _, row in sub.iterrows():
                sig = "*" if row["p_holm"] < 0.05 else " "
                direction = "↑better" if row["median_improvement"] > 0 else ("↓worse" if row["median_improvement"] < 0 else "~equal")
                print(f"  {sig} {row['metric']:8s} {row['tissue']:8s} | "
                      f"baseline={row['baseline_agg']:.4f}  model={row['model_agg']:.4f}  "
                      f"Δmedian={row['median_improvement']:+.4f} {direction}  "
                      f"RBC={row['rbc']:+.3f}  p={row['p_value']:.4f}  p_holm={row['p_holm']:.4f}")
    print()


if __name__ == "__main__":
    main()