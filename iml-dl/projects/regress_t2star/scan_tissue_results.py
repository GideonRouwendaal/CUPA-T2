#!/usr/bin/env python3
"""
scan_tissue_results.py

Scans all .log files in a directory (recursively). Each log file may contain
multiple sequential runs (e.g. [1/12] Running: ..., [2/12] Running: ...).
Each run is split out and parsed independently.

For each run + tissue, takes the LAST occurrence of:
  "New best {TISSUE} NRMSE = X.XXXXXX"
which is the best value that run achieved.

Usage
-----
python scan_tissue_results.py --log_dir /path/to/logs --top_k 10
"""

import argparse
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional


# Marker that starts a new run inside a batch log
RUN_START_RE = re.compile(r"\[\d+/\d+\] Running:\s*(\S+\.yaml)")


def split_into_runs(text: str) -> List[tuple]:
    """
    Split a log file into (yaml_name, run_text) chunks.
    Each chunk starts at a '[N/M] Running: foo.yaml' line.
    """
    matches = list(RUN_START_RE.finditer(text))
    if not matches:
        # single run, no batch header
        return [("unknown.yaml", text)]

    chunks = []
    for i, m in enumerate(matches):
        yaml_name = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunks.append((yaml_name, text[start:end]))
    return chunks


def parse_run(yaml_name: str, chunk: str) -> Optional[Dict]:
    """Parse a single run's text chunk."""

    result = {"yaml_name": yaml_name}

    # ------------------------------------------------------------------
    # Focus tissue: from yaml name first, then output directory line
    # ------------------------------------------------------------------
    m = re.search(r"tissue_iso_(wm|gm|csf)__", yaml_name, re.IGNORECASE)
    if m:
        result["focus_tissue"] = m.group(1).lower()
    else:
        m = re.search(r"tissue_iso_(wm|gm|csf)__", chunk, re.IGNORECASE)
        result["focus_tissue"] = m.group(1).lower() if m else None

    # ------------------------------------------------------------------
    # Run tag from output directory line
    # ------------------------------------------------------------------
    m = re.search(r"Output directory:.*/(tissue_iso_(?:wm|gm|csf)__[^\s/]+)", chunk)
    result["run_tag"] = m.group(1) if m else yaml_name.replace(".yaml", "")

    # ------------------------------------------------------------------
    # Best NRMSE per tissue = LAST "New best X NRMSE = Y" in this chunk
    # ------------------------------------------------------------------
    for tissue, label in [("wm", "WM"), ("gm", "GM"), ("csf", "CSF")]:
        matches = re.findall(rf"New best {label} NRMSE\s*=\s*([\d.]+)", chunk)
        result[f"nrmse_{tissue}"] = float(matches[-1]) if matches else None

    # skip runs with no NRMSE at all (crashed before first val)
    if all(result.get(f"nrmse_{t}") is None for t in ["wm", "gm", "csf"]):
        return None

    # ------------------------------------------------------------------
    # Schedule config: q_lo, q_hi
    # ------------------------------------------------------------------
    m = re.search(r"^q_lo:\s*([\d.]+)", chunk, re.MULTILINE)
    result["q_lo"] = float(m.group(1)) if m else None

    m = re.search(r"^q_hi:\s*([\d.]+)", chunk, re.MULTILINE)
    result["q_hi"] = float(m.group(1)) if m else None

    # ------------------------------------------------------------------
    # alpha_min / alpha_max / u_lo_val / u_hi_val for focus tissue block
    # ------------------------------------------------------------------
    focus = result.get("focus_tissue")
    if focus:
        block_m = re.search(
            rf"^\s*{focus}:\s*\n((?:[ \t]+\S.*\n)*)",
            chunk, re.MULTILINE | re.IGNORECASE
        )
        block = block_m.group(1) if block_m else ""
        for key in ["alpha_min", "alpha_max", "u_lo_val", "u_hi_val"]:
            m = re.search(rf"{key}:\s*([\d.e+\-]+)", block)
            if not m:
                m = re.search(rf"^{key}:\s*([\d.e+\-]+)", chunk, re.MULTILINE)
            result[key] = float(m.group(1)) if m else None
    else:
        for key in ["alpha_min", "alpha_max", "u_lo_val", "u_hi_val"]:
            result[key] = None

    return result


def parse_log(path: Path) -> List[Dict]:
    """Parse all runs inside a single .log file."""
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return []

    runs = split_into_runs(text)
    records = []
    for yaml_name, chunk in runs:
        r = parse_run(yaml_name, chunk)
        if r is not None:
            r["log_file"] = path.name
            r["log_path"] = str(path)
            records.append(r)
    return records


def scan_logs(log_dir: Path) -> List[Dict]:
    log_files = sorted(log_dir.rglob("*.log"))
    if not log_files:
        print(f"[ERROR] No .log files found under {log_dir}")
        return []

    print(f"Found {len(log_files)} .log files\n")
    all_records = []
    total_runs = 0
    skipped = 0
    for p in log_files:
        records = parse_log(p)
        # count how many run chunks existed vs parsed
        try:
            text = p.read_text(errors="replace")
            n_chunks = max(len(list(RUN_START_RE.finditer(text))), 1)
        except Exception:
            n_chunks = 0
        total_runs += n_chunks
        skipped += n_chunks - len(records)
        all_records.extend(records)

    print(f"Total run chunks found: {total_runs}")
    print(f"Successfully parsed:    {len(all_records)}")
    print(f"Skipped (no NRMSE):     {skipped}\n")
    return all_records


def print_top_k(records: List[Dict], focus_tissue: str, top_k: int):
    metric_key = f"nrmse_{focus_tissue}"
    filtered = [
        r for r in records
        if r.get("focus_tissue") == focus_tissue and r.get(metric_key) is not None
    ]

    if not filtered:
        print(f"  [No completed runs for focus_tissue={focus_tissue.upper()}]\n")
        return

    ranked = sorted(filtered, key=lambda r: r[metric_key])
    tissue_up = focus_tissue.upper()
    k = min(top_k, len(ranked))

    print(f"{'='*95}")
    print(f"  TOP {k} for {tissue_up}  |  total completed: {len(filtered)}")
    print(f"{'='*95}")
    print(f"  {'#':>3}  {'NRMSE_'+tissue_up:<12}  {'q_lo':<5}  {'q_hi':<5}  "
          f"{'amin':<5}  {'amax':<5}  {'u_lo_val':<11}  {'u_hi_val':<11}  yaml_name")
    print(f"  {'-'*3}  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  "
          f"{'-'*11}  {'-'*11}  {'-'*50}")

    for rank, r in enumerate(ranked[:k], start=1):
        score = r[metric_key]
        u_lo  = r.get("u_lo_val")
        u_hi  = r.get("u_hi_val")
        print(f"  {rank:>3}  {score:<12.6f}  "
              f"{str(r.get('q_lo','?')):<5}  {str(r.get('q_hi','?')):<5}  "
              f"{str(r.get('alpha_min','?')):<5}  {str(r.get('alpha_max','?')):<5}  "
              f"{(f'{u_lo:.5g}' if u_lo is not None else '?'):<11}  "
              f"{(f'{u_hi:.5g}' if u_hi is not None else '?'):<11}  "
              f"{r.get('yaml_name','?')}")
    print()


def save_csv(records: List[Dict], out_path: Path):
    if not records:
        return
    fields = [
        "focus_tissue", "nrmse_wm", "nrmse_gm", "nrmse_csf",
        "q_lo", "q_hi", "alpha_min", "alpha_max",
        "u_lo_val", "u_hi_val", "run_tag", "yaml_name", "log_file", "log_path",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Full CSV saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True,
                    help="Directory containing .log files (searched recursively)")
    ap.add_argument("--top_k",  type=int, default=10)
    ap.add_argument("--out_csv", type=str, default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    records = scan_logs(log_dir)
    if not records:
        return

    for tissue in ["wm", "gm", "csf"]:
        print_top_k(records, tissue, args.top_k)

    out_csv = Path(args.out_csv) if args.out_csv else log_dir / "tissue_sweep_results.csv"
    save_csv(records, out_csv)


if __name__ == "__main__":
    main()