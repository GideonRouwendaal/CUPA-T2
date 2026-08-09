#!/usr/bin/env python3
"""
Regenerate the uncertainty analysis (plots + CSV/MD tables) from a stored
`pixel_level_cache.npz`, without needing the model checkpoint.

Every routine behind `RegressionEvaluator._perform_uncertainty_analysis` is a
pure function of the cached pixel-level arrays, so an evaluation that has
already been run once can have its uncertainty outputs rebuilt at any time --
useful when the checkpoints are gone but the evaluation folder survived.

Usage:
    # rebuild in place (overwrites the existing uncertainty_analysis/ contents)
    python regenerate_uncertainty_analysis.py \\
        --eval_dir evaluations/Final/Acc4/SWEEP_TOP1

    # rebuild into a separate directory instead
    python regenerate_uncertainty_analysis.py \\
        --eval_dir evaluations/Final/Acc4/SWEEP_TOP1 \\
        --out_dir  /tmp/unc_rebuild

The cache is written by the evaluator at
    <eval_dir>/uncertainty_analysis/pixel_level_cache.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from regression_evaluator import RegressionEvaluator

PROJECT_DIR = Path(__file__).resolve().parent

# Settings the analysis routines read off `self`; these mirror the evaluator's
# own defaults (regression_evaluator.py, __init__).
_ANALYSIS_DEFAULTS = {
    "normalize_rms_by_te0": False,
    "rms_te0_floor_percentile": 1.0,
    "ssim_mode": "full",
    "calibration_bins": 10,
    "calibration_min_points_per_bin": 200,
}

_ARRAY_KEYS = (
    "predictions", "ground_truths", "errors",
    "input_rms", "input_rms_norm",
    "aleatoric", "epistemic", "total_uncertainty",
    "tissue_labels",
)


def load_cache(cache_path: Path):
    """Return (data dict, meta dict) from a pixel_level_cache.npz."""
    z = np.load(cache_path, allow_pickle=False)
    data = {k: z[k] for k in _ARRAY_KEYS if k in z.files}
    meta = {}
    if "meta_json" in z.files:
        try:
            meta = json.loads(str(z["meta_json"]))
        except (ValueError, TypeError):
            meta = {}
    return data, meta


def build_stub_evaluator(overrides: dict, meta: dict) -> RegressionEvaluator:
    """An evaluator shell carrying only what the analysis routines need.

    __init__ is bypassed on purpose: it would build a model and a dataloader,
    which is exactly what this script exists to avoid. Run metadata is replayed
    from the cache instead of recomputed, so the rebuilt cache is identical.
    """
    ev = object.__new__(RegressionEvaluator)
    settings = dict(_ANALYSIS_DEFAULTS)
    # the cache records the settings the original run actually used
    for key in _ANALYSIS_DEFAULTS:
        if key in meta:
            settings[key] = meta[key]
    settings.update(overrides)
    for key, value in settings.items():
        setattr(ev, key, value)

    ev._run_metadata = lambda: dict(meta)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", type=Path, required=True,
                    help="Evaluation folder containing uncertainty_analysis/pixel_level_cache.npz")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to write the rebuilt analysis (default: rebuild in place)")
    ap.add_argument("--calibration_bins", type=int, default=None)
    ap.add_argument("--calibration_min_points_per_bin", type=int, default=None)
    args = ap.parse_args()

    cache_path = args.eval_dir / "uncertainty_analysis" / "pixel_level_cache.npz"
    if not cache_path.is_file():
        raise SystemExit(f"No pixel-level cache found at {cache_path}")

    data, meta = load_cache(cache_path)
    n = len(next(iter(data.values()))) if data else 0
    print(f"Loaded {n} cached voxels from {cache_path}")
    if meta:
        print(f"Cache meta: {meta}")

    overrides = {k: v for k, v in {
        "calibration_bins": args.calibration_bins,
        "calibration_min_points_per_bin": args.calibration_min_points_per_bin,
    }.items() if v is not None}

    ev = build_stub_evaluator(overrides, meta)
    ev.pixel_level_data = data

    out_root = args.out_dir if args.out_dir is not None else args.eval_dir
    out_root.mkdir(parents=True, exist_ok=True)
    # _perform_uncertainty_analysis appends 'uncertainty_analysis' itself and
    # re-saves the cache, so the rebuilt folder stays self-contained.
    ev._perform_uncertainty_analysis(str(out_root))


if __name__ == "__main__":
    main()
