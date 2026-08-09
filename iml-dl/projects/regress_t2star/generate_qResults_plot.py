"""
plot_comparison.py
------------------
Generate a multi-row / multi-column comparison figure for T2* regression models.

Rows:
  0  T2* map
  1  Error map (|pred - GT|)
  2  Predictive uncertainty (total)
  3  Aleatoric uncertainty
  4  Epistemic uncertainty

Columns:
  0  Reference  (GT T2* only; remaining rows show text labels)
  1  PUQ        (T2* + error only; remaining rows show text labels)
  2  Heteroscedastic
  3  CUPA-T2* (LR)
  4  CUPA-T2* (Chol)

Usage
-----
python plot_comparison.py \
    --config         path/to/base_config.yaml \
    --puq_ckpt       path/to/puq/best_regression_model.pth \
    --hetero_ckpt    path/to/hetero/best_regression_model.pth \
    --cupa_lr_ckpt   path/to/cupa_lr/best_regression_model.pth \
    --cupa_chol_ckpt path/to/cupa_chol/best_regression_model.pth \
    --acc_rate  2 \
    --subject   sub-01 \
    --slice_num 20 \
    --out       comparison.png

The four columns default to the trained arms (none_concat / none_concat+MC /
low_rank_concat / cholesky_concat). Override them only for other model variants:

    --model_overrides '{"cupa_lr": {"lowrank_rank": 6}}'

Each key in --model_overrides maps to a model column name (puq / hetero / cupa_lr / cupa_chol).
The dict values are merged into config["regression_params"], so any regression_params key can
be overridden (e.g. heteroscedastic, use_mc_dropout, uncertainty_mode, lowrank_rank, ...).
Keys not present in --model_overrides keep their value from the base config.
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Helpers to interface with your existing codebase
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    import json
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported config format: {p.suffix}")


def _build_colormap(csv_path: str = None) -> ListedColormap:
    """Load the navia T2* colormap, or fall back to a red-yellow LUT."""
    if csv_path and os.path.isfile(csv_path):
        try:
            ori = np.loadtxt(csv_path)
            ori[0, :] = 0.0
            return ListedColormap(ori, name="t2star")
        except Exception:
            pass
    # Fallback: simple dark-red → yellow
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "t2star_fallback",
        [(0, 0, 0), (0.5, 0, 0), (1, 0.6, 0)],
        N=256,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper around your RegressionEvaluator to run inference on ONE slice
# ─────────────────────────────────────────────────────────────────────────────

def _apply_overrides(config: dict, acc_rate: str, regression_overrides: dict) -> dict:
    """
    Return a deep-copy of *config* with:
      - regression_params["acc_rate"] set to acc_rate
      - every key/value in regression_overrides merged into regression_params

    regression_overrides examples:
      {"heteroscedastic": False, "use_mc_dropout": False, "uncertainty_mode": "none"}
      {"heteroscedastic": True, "use_mc_dropout": True, "uncertainty_mode": "low_rank"}
    """
    import copy
    cfg = copy.deepcopy(config)
    rp = cfg.setdefault("regression_params", {})
    rp["acc_rate"] = int(acc_rate)
    for key, val in regression_overrides.items():
        rp[key] = val
    return cfg


def run_model_on_slice(
    config: dict,
    checkpoint_path: str,
    subject: str,
    slice_num: int,
    acc_rate: str,
    device: torch.device,
    regression_overrides: Optional[dict] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Returns a dict with keys:
        pred_t2, gt_t2, recon_t2,
        uncertainty_total, uncertainty_aleatoric, uncertainty_epistemic  (may be absent)
        nrmse, ssim  (scalars)
    or None if the slice is not found.

    regression_overrides: dict of regression_params keys to patch before building the
        evaluator, e.g. {"heteroscedastic": True, "use_mc_dropout": False,
        "uncertainty_mode": "none"}.  This is how each column gets the right model
        architecture without needing a separate config file per model.
    """
    try:
        # ── Import your evaluator ──────────────────────────────────────────
        # Adjust the import below to match where RegressionEvaluator lives in
        # your project, e.g.:
        #   from projects.recon_t2star_regress.evaluate import RegressionEvaluator
        from regression_evaluator import RegressionEvaluator  # <── ADJUST
    except ImportError as e:
        raise ImportError(
            f"Cannot import RegressionEvaluator: {e}\n"
            "Please adjust the import inside run_model_on_slice() to match your project."
        )

    cfg = _apply_overrides(config, acc_rate, regression_overrides or {})

    ev = RegressionEvaluator(cfg, device)
    ev.model_path_override = checkpoint_path
    ev.to_plot = False

    ev.setup_test_dataset()
    ev.load_model()

    for batch in ev.test_loader:
        # Match subject + slice
        batch_subject = batch["subject"][0]
        if isinstance(batch_subject, (bytes, bytearray)):
            batch_subject = batch_subject.decode()
        else:
            batch_subject = str(batch_subject)

        batch_slice = int(batch["slice_num"][0])

        if batch_subject != subject or batch_slice != slice_num:
            continue

        # Found it – run inference
        if not ev._is_valid_slice(batch):
            return None

        data = ev._extract_batch_data(batch)
        input_pixels = ev._prepare_input(data)
        outputs, uncertainties = ev._forward_pass(input_pixels)
        predictions = ev._process_predictions(outputs, data, uncertainties)
        metrics = ev._compute_metrics(predictions, data)

        result = {
            "pred_t2":  predictions["pred_t2"],
            "gt_t2":    predictions["gt_t2"],
            "recon_t2": predictions["recon_t2"],
            "brain_mask": data["brain_mask"].astype(bool),
        }
        for k in ("uncertainty_total", "uncertainty_aleatoric", "uncertainty_epistemic"):
            if k in predictions:
                result[k] = predictions[k]

        # Pull NRMSE / SSIM from per-slice metrics
        t2_metrics = metrics.get("t2", {})
        result["nrmse"] = float(t2_metrics.get("nrmse", np.nan))
        result["ssim"]  = float(t2_metrics.get("ssim",  np.nan))

        return result

    warnings.warn(f"Subject '{subject}' slice {slice_num} not found in test dataset.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

NROWS = 5
NCOLS = 5

ROW_LABELS = [
    "T2* map",
    "Error map",
    "Predictive\nUncertainty",
    "Aleatoric\nUncertainty",
    "Epistemic\nUncertainty",
]

COL_LABELS = [
    "Reference",
    "PUQ",
    "Heteroscedastic",
    "CUPA-T2*\n(LR)",
    "CUPA-T2*\n(Chol)",
]

# Which (row, col) cells are *active* (contain an image)
ACTIVE = {
    (0, 0),  # Reference – T2* only
    (0, 1), (1, 1),          # PUQ – T2* + error
    (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),   # Heteroscedastic – all
    (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),   # CUPA LR – all
    (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),   # CUPA Chol – all
}

# Text labels for inactive cells (row, col) → label string
INACTIVE_LABELS = {
    (1, 0): "Error",
    (2, 0): "",
    (3, 0): "",
    (4, 0): "",
    (2, 1): "Predictive\nUncertainty",
    (3, 1): "Aleatoric\nUncertainty",
    (4, 1): "Epistemic\nUncertainty",
}


def _as_masked(img: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~mask, img)


def _make_figure(
    results: Dict[str, Optional[dict]],   # key = col name
    t2_cmap: ListedColormap,
    max_t2: float = 200.0,
    transpose: bool = True,
    figsize: Tuple[float, float] = (18, 16),
    dpi: int = 200,
) -> plt.Figure:
    """Build and return the comparison figure."""

    col_keys = ["reference", "puq", "hetero", "cupa_lr", "cupa_chol"]

    fig = plt.figure(figsize=figsize, facecolor="black")
    gs = gridspec.GridSpec(
        NROWS, NCOLS,
        figure=fig,
        hspace=0.04,
        wspace=0.04,
        left=0.10, right=0.98,
        top=0.94, bottom=0.02,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(NCOLS)] for r in range(NROWS)]

    # ── Pre-compute shared error / uncertainty colour limits ────────────────
    # Gather all error maps across model columns (cols 1-4)
    all_errors, all_total_unc, all_alea_unc, all_epi_unc = [], [], [], []
    for ck in ["puq", "hetero", "cupa_lr", "cupa_chol"]:
        res = results.get(ck)
        if res is None:
            continue
        mask = res["brain_mask"]  # use raw mask here, no transpose needed (just indexing)
        if "pred_t2" in res and "gt_t2" in res:
            err = np.abs(res["gt_t2"] - res["pred_t2"])
            all_errors.append(err[mask])
        for key, lst in [
            ("uncertainty_total", all_total_unc),
            ("uncertainty_aleatoric", all_alea_unc),
            ("uncertainty_epistemic", all_epi_unc),
        ]:
            if key in res:
                lst.append(res[key][mask])

    def _shared_range(arrays, p_lo=2, p_hi=98):
        if not arrays:
            return 0.0, 1.0
        vals = np.concatenate(arrays)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        return float(np.percentile(vals, p_lo)), float(np.percentile(vals, p_hi))

    err_vmin, err_vmax     = _shared_range(all_errors,    p_lo=0, p_hi=99)
    tot_vmin, tot_vmax     = _shared_range(all_total_unc, p_lo=0, p_hi=99)
    alea_vmin, alea_vmax   = _shared_range(all_alea_unc,  p_lo=0, p_hi=99)
    epi_vmin, epi_vmax     = _shared_range(all_epi_unc,   p_lo=0, p_hi=99)

    unc_ranges = {
        1: (err_vmin,  err_vmax),
        2: (tot_vmin,  tot_vmax),
        3: (alea_vmin, alea_vmax),
        4: (epi_vmin,  epi_vmax),
    }

    # ── Helper: maybe transpose ─────────────────────────────────────────────
    def T(x):
        return x.T if (transpose and x is not None) else x

    # ── Draw each cell ──────────────────────────────────────────────────────
    for c, ck in enumerate(col_keys):
        res = results.get(ck)

        for r in range(NROWS):
            ax = axes[r][c]
            ax.set_facecolor("black")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if (r, c) not in ACTIVE:
                # Show text label or leave blank
                label = INACTIVE_LABELS.get((r, c), "")
                if label:
                    ax.text(
                        0.5, 0.5, label,
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold",
                        style="italic",
                    )
                continue

            if res is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=10)
                continue

            mask = T(res["brain_mask"])
            gt   = T(res.get("gt_t2"))
            pred = T(res.get("pred_t2"))

            # ── Row 0: T2* map ────────────────────────────────────────────
            if r == 0:
                if c == 0:
                    # Reference: GT
                    img = _as_masked(gt, mask)
                    ax.imshow(img, cmap=t2_cmap, vmin=0, vmax=max_t2,
                              interpolation="nearest", origin="lower")
                else:
                    # Predicted T2*
                    img = _as_masked(pred, mask)
                    im = ax.imshow(img, cmap=t2_cmap, vmin=0, vmax=max_t2,
                                   interpolation="nearest", origin="lower")
                    # Overlay NRMSE / SSIM
                    nrmse = res.get("nrmse", np.nan)
                    ssim  = res.get("ssim",  np.nan)
                    label_str = f"{nrmse:.5f} / {ssim:.4f}"
                    # Inside the panel: above the axes it would sit on the column title.
                    ax.text(
                        0.5, 0.02, label_str,
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        color="yellow", fontsize=8, fontweight="bold",
                    )

            # ── Row 1: Error map ──────────────────────────────────────────
            elif r == 1:
                err = np.abs(gt - pred)
                img = _as_masked(err, mask)
                ax.imshow(img, cmap="hot", vmin=err_vmin, vmax=err_vmax,
                          interpolation="nearest", origin="lower")

            # ── Row 2: Total uncertainty ──────────────────────────────────
            elif r == 2:
                unc = res.get("uncertainty_total")
                if unc is not None:
                    img = _as_masked(T(unc), mask)
                    ax.imshow(img, cmap="inferno", vmin=tot_vmin, vmax=tot_vmax,
                              interpolation="nearest", origin="lower")
                else:
                    ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=14)

            # ── Row 3: Aleatoric ──────────────────────────────────────────
            elif r == 3:
                unc = res.get("uncertainty_aleatoric")
                if unc is not None:
                    img = _as_masked(T(unc), mask)
                    ax.imshow(img, cmap="inferno", vmin=alea_vmin, vmax=alea_vmax,
                              interpolation="nearest", origin="lower")
                else:
                    ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=14)

            # ── Row 4: Epistemic ──────────────────────────────────────────
            elif r == 4:
                unc = res.get("uncertainty_epistemic")
                if unc is not None:
                    img = _as_masked(T(unc), mask)
                    ax.imshow(img, cmap="inferno", vmin=epi_vmin, vmax=epi_vmax,
                              interpolation="nearest", origin="lower")
                else:
                    ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                            ha="center", va="center", color="gray", fontsize=14)

    # ── Column headers ───────────────────────────────────────────────────────
    for c, label in enumerate(COL_LABELS):
        axes[0][c].set_title(label, color="white", fontsize=11, fontweight="bold", pad=4)

    # ── Row labels (left margin) ─────────────────────────────────────────────
    for r, label in enumerate(ROW_LABELS):
        axes[r][0].set_ylabel(
            label, color="white", fontsize=10, fontweight="bold",
            labelpad=6, rotation=90, va="center",
        )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate T2* model comparison figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",         required=True,  help="Base training config YAML/JSON")
    p.add_argument("--puq_ckpt",       required=True,  help="PUQ checkpoint (.pth)")
    p.add_argument("--hetero_ckpt",    required=True,  help="Heteroscedastic checkpoint")
    p.add_argument("--cupa_lr_ckpt",   required=True,  help="CUPA-T2* (LR) checkpoint")
    p.add_argument("--cupa_chol_ckpt", required=True,  help="CUPA-T2* (Chol) checkpoint")
    p.add_argument(
        "--model_overrides",
        default=None,
        help=(
            "JSON string mapping model keys (puq / hetero / cupa_lr / cupa_chol) to dicts "
            "of regression_params overrides.  Example:\n"
            '\'{"puq": {"heteroscedastic": false, "use_mc_dropout": false, "uncertainty_mode": "none"}, '
            '"hetero": {"heteroscedastic": true, "use_mc_dropout": false}, '
            '"cupa_lr": {"heteroscedastic": true, "use_mc_dropout": true, "uncertainty_mode": "low_rank"}, '
            '"cupa_chol": {"heteroscedastic": true, "use_mc_dropout": true, "uncertainty_mode": "cholesky"}}\''
        ),
    )
    p.add_argument("--acc_rate",       default="2",    help="Acceleration rate (default: 2)")
    p.add_argument("--subject",        required=True,  help="Subject identifier string")
    p.add_argument("--slice_num",      type=int, required=True, help="Slice index")
    p.add_argument("--device",         default="cuda:0")
    p.add_argument("--max_t2",         type=float, default=200.0, help="T2* colormap max [ms]")
    p.add_argument("--no_transpose",   action="store_true", help="Disable image transpose")
    p.add_argument("--cmap_csv",       default=None,
                   help="Path to navia.csv colormap. Falls back to the path in RegressionEvaluator.")
    p.add_argument("--out",            default="comparison.png", help="Output figure path")
    p.add_argument("--dpi",            type=int, default=200)
    return p.parse_args()


def main():
    import json
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    config = _load_config(args.config)

    # ── Parse --model_overrides ──────────────────────────────────────────────
    # Default: sensible per-column settings that match the four model types.
    # These match the four trained arms: every one takes mean magnitude concatenated
    # with the standard deviation, so the modes must carry the "_concat" suffix — it
    # is what doubles in_ch from 12 to 24, and a bare mode silently fails to load.
    default_overrides = {
        "puq":       {"heteroscedastic": False, "use_mc_dropout": False, "uncertainty_mode": "none_concat"},
        "hetero":    {"heteroscedastic": True,  "use_mc_dropout": True,  "uncertainty_mode": "none_concat"},
        "cupa_lr":   {"heteroscedastic": True,  "use_mc_dropout": True,  "uncertainty_mode": "low_rank_concat"},
        "cupa_chol": {"heteroscedastic": True,  "use_mc_dropout": True,  "uncertainty_mode": "cholesky_concat"},
    }
    if args.model_overrides:
        user_overrides = json.loads(args.model_overrides)
        # Merge user values on top of defaults (key by key so partial specs work)
        for model_key, patches in user_overrides.items():
            default_overrides.setdefault(model_key, {}).update(patches)
    model_overrides = default_overrides

    print("\nEffective regression_params overrides per model:")
    for k, v in model_overrides.items():
        print(f"  {k:12s}: {v}")

    # ── Colormap ─────────────────────────────────────────────────────────────
    cmap_csv = args.cmap_csv or str(
        Path(__file__).resolve().parent / "color_map" / "navia.csv"
    )
    t2_cmap = _build_colormap(cmap_csv)

    # ── Run inference for each model ─────────────────────────────────────────
    model_specs = [
        ("puq",       args.puq_ckpt),
        ("hetero",    args.hetero_ckpt),
        ("cupa_lr",   args.cupa_lr_ckpt),
        ("cupa_chol", args.cupa_chol_ckpt),
    ]

    results: Dict[str, Optional[dict]] = {"reference": None}
    gt_cache = None  # reuse GT from the first successful model run
    failed: List[Tuple[str, str]] = []

    for key, ckpt in model_specs:
        overrides = model_overrides.get(key, {})
        print(f"\nRunning inference: {key}")
        print(f"  checkpoint : {ckpt}")
        print(f"  overrides  : {overrides}")
        try:
            res = run_model_on_slice(
                config=config,
                checkpoint_path=ckpt,
                subject=args.subject,
                slice_num=args.slice_num,
                acc_rate=args.acc_rate,
                device=device,
                regression_overrides=overrides,
            )
            results[key] = res
            if res is None:
                # _run_one returns None when the requested slice is absent from the
                # test set or fails the validity check. That is not an exception, so
                # without this it would reach the figure as a blank panel.
                failed.append((key, f"no usable slice for {args.subject} "
                                    f"slice {args.slice_num}"))
            elif gt_cache is None:
                gt_cache = {
                    "gt_t2":      res["gt_t2"],
                    "brain_mask": res["brain_mask"],
                }
        except Exception as e:
            warnings.warn(f"Model {key} failed: {e}")
            failed.append((key, str(e)))
            results[key] = None

    # A silently missing panel looks like a finished figure, so refuse to continue.
    if failed:
        detail = "\n".join(f"  {key}: {msg}" for key, msg in failed)
        raise SystemExit(
            f"{len(failed)} of {len(model_specs)} models could not be run:\n{detail}\n"
            "The figure would be missing those panels. Check that each checkpoint "
            "matches its column's uncertainty_mode (pass --model_overrides to adjust), "
            "and that --subject / --slice_num exist in the test split."
        )

    results["reference"] = gt_cache

    # ── Build figure ─────────────────────────────────────────────────────────
    print("\nBuilding figure …")
    fig = _make_figure(
        results=results,
        t2_cmap=t2_cmap,
        max_t2=args.max_t2,
        transpose=not args.no_transpose,
        dpi=args.dpi,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"\n✓ Saved to: {out_path}")


if __name__ == "__main__":
    main()