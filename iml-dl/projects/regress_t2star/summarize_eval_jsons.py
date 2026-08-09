#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Nominal Gaussian coverage
NOM_C1 = 0.682
NOM_C2 = 0.954
NOM_C3 = 0.997

# ============================
# Helpers
# ============================
def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _fmt_pm(mean: Optional[float], std: Optional[float], nd: int) -> str:
    """Compact LaTeX formatting: 0.234$\\pm$0.053"""
    if mean is None or std is None:
        return "NA"
    return f"{mean:.{nd}f}$\\pm${std:.{nd}f}"


def _fmt_mean(mean: Optional[float], nd: int) -> str:
    if mean is None:
        return "NA"
    return f"{mean:.{nd}f}"


def _bold(s: str) -> str:
    if s == "NA":
        return s
    return f"\\textbf{{{s}}}"


def _min_idx(vals: List[Optional[float]]) -> Optional[int]:
    best_i = None
    best_v = None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if best_v is None or v < best_v:
            best_v = v
            best_i = i
    return best_i


def _max_idx(vals: List[Optional[float]]) -> Optional[int]:
    best_i = None
    best_v = None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v = v
            best_i = i
    return best_i


# ============================
# Path parsing
# ============================
_RE_PFLOAT = re.compile(r"^-?\d+p\d+$")
_RE_RMS_TOKEN = re.compile(r"(?:^|_)rms(?P<val>\d+p\d+)(?:_|$)")
_RE_HETERO_TOKEN = re.compile(r"(?:^|_)hetero([01])(?:_|$)")


def _decode_pfloat(s: str) -> float:
    s = s.strip()
    if not _RE_PFLOAT.match(s):
        raise ValueError(f"Not a p-float: {s}")
    return float(s.replace("p", "."))


def infer_acc_and_mode_from_eval_path(eval_json: Path) -> Tuple[str, str]:
    """
    Given .../_evaluations/acc_rate_4/cholesky_concat/.../eval_summary.json
    returns ("4","cholesky_concat")
    """
    parts = eval_json.parts
    acc = "unknown"
    mode = "unknown"
    for i, p in enumerate(parts):
        if p.startswith("acc_rate_"):
            acc = p.split("acc_rate_", 1)[1]
            if i + 1 < len(parts):
                mode = parts[i + 1]
            break
    return acc, mode


def infer_hetero_from_path(eval_json: Path) -> Optional[bool]:
    parts = set(eval_json.parts)
    if "heteroscedastic" in parts:
        return True
    if "homoscedastic" in parts:
        return False
    run_name = eval_json.parent.name
    m = _RE_HETERO_TOKEN.search(run_name)
    if m:
        return bool(int(m.group(1)))
    return None


def infer_rms_corr_weight_from_path(eval_json: Path) -> float:
    # 1) folder rms_corr_weight_X
    for p in eval_json.parts:
        if p.startswith("rms_corr_weight_"):
            tail = p.split("rms_corr_weight_", 1)[1]
            try:
                return float(tail)
            except Exception:
                pass
    # 2) token _rms0p05_
    run_name = eval_json.parent.name
    m = _RE_RMS_TOKEN.search(run_name)
    if m:
        try:
            return _decode_pfloat(m.group("val"))
        except Exception:
            pass
    return 0.0


# ============================
# Data model
# ============================
@dataclass(frozen=True)
class EvalRun:
    acc_rate: str
    mode: str
    heteroscedastic: Optional[bool]
    rms_corr_weight: float
    eval_json: Path
    data: Dict[str, Any]

    def mean(self, key: str) -> Optional[float]:
        return _safe_float(self.data.get(key))

    def std(self, key: str) -> Optional[float]:
        return _safe_float(self.data.get(key))


# ============================
# Selection logic
# ============================
def pick_best(
    runs: List[EvalRun],
    *,
    mode_pred,
    hetero_pref: Optional[bool],
    score_key: str,
) -> Optional[EvalRun]:
    cand = [r for r in runs if mode_pred(r.mode)]
    if not cand:
        return None

    if hetero_pref is not None:
        pref = [r for r in cand if r.heteroscedastic is hetero_pref]
        if pref:
            cand = pref

    best = None
    best_v = None
    for r in cand:
        v = r.mean(score_key)
        if v is None:
            continue
        if best_v is None or v < best_v:
            best = r
            best_v = v
    return best


def method_display(method_key: str, picked_by_acc: Dict[str, Optional[EvalRun]], accs: List[str]) -> str:
    """
    Build row label. For CUPA rows, include w per acc if it varies.
    """
    if method_key == "PUQ":
        return "PUQ Baseline"
    if method_key == "HETERO":
        return "Hetero Baseline"

    if method_key in {"CHOL", "LR"}:
        base = "CUPA-$T_2^*$ (Chol)" if method_key == "CHOL" else "CUPA-$T_2^*$ (LR)"
        ws: List[Optional[float]] = []
        for a in accs:
            r = picked_by_acc.get(a)
            ws.append(None if r is None else float(r.rms_corr_weight))
        # If all None or all equal, show single w; else show list in acc order
        clean = [w for w in ws if w is not None]
        if len(clean) == 0:
            return base
        all_equal = all(abs(clean[i] - clean[0]) < 1e-12 for i in range(len(clean)))
        if all_equal:
            return f"{base} ($w$={clean[0]:.2f})"
        # show as [4x:0.05,6x:0.10,...]
        parts = []
        for a, w in zip(accs, ws):
            if w is None:
                continue
            parts.append(f"{a}$\\times$:{w:.2f}")
        w_str = ", ".join(parts)
        return f"{base} ($w$={w_str})"

    return method_key


# ============================
# LaTeX writers
# ============================
def write_tex(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s)


def latex_table_general_nrmse_ssim(
    caption: str,
    label: str,
    accs: List[str],
    rows: List[Tuple[str, Dict[str, Tuple[str, str]]]],
) -> str:
    # rows: (method_display, {acc: (nrmse_cell, ssim_cell)})
    colspec = "|l|" + "cc|" * len(accs)
    last_col = 1 + 2 * len(accs)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\hline")

    hdr1 = ["\\multicolumn{1}{|c|}{Method}"]
    for a in accs:
        hdr1.append(f"\\multicolumn{{2}}{{c|}}{{{a}$\\times$}}")
    out.append(" & ".join(hdr1) + " \\\\")
    out.append(f"\\cline{{2-{last_col}}}")

    hdr2 = ["\\multicolumn{1}{|c|}{}"]
    for _ in accs:
        hdr2 += ["NRMSE $\\downarrow$", "SSIM $\\uparrow$"]
    out.append(" & ".join(hdr2) + " \\\\")
    out.append("\\hline")

    for method_disp, cellmap in rows:
        row_cells = [method_disp]
        for a in accs:
            nrmse_cell, ssim_cell = cellmap.get(a, ("NA", "NA"))
            row_cells += [nrmse_cell, ssim_cell]
        out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\hline")

    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_table_general_mae_psnr(
    caption: str,
    label: str,
    accs: List[str],
    rows: List[Tuple[str, Dict[str, Tuple[str, str]]]],
) -> str:
    colspec = "|l|" + "cc|" * len(accs)
    last_col = 1 + 2 * len(accs)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\hline")

    hdr1 = ["\\multicolumn{1}{|c|}{Method}"]
    for a in accs:
        hdr1.append(f"\\multicolumn{{2}}{{c|}}{{{a}$\\times$}}")
    out.append(" & ".join(hdr1) + " \\\\")
    out.append(f"\\cline{{2-{last_col}}}")

    hdr2 = ["\\multicolumn{1}{|c|}{}"]
    for _ in accs:
        hdr2 += ["MAE $\\downarrow$", "PSNR $\\uparrow$"]
    out.append(" & ".join(hdr2) + " \\\\")
    out.append("\\hline")

    for method_disp, cellmap in rows:
        row_cells = [method_disp]
        for a in accs:
            mae_cell, psnr_cell = cellmap.get(a, ("NA", "NA"))
            row_cells += [mae_cell, psnr_cell]
        out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\hline")

    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_table_tissue_nrmse(
    caption: str,
    label: str,
    accs: List[str],
    rows: List[Tuple[str, Dict[str, Tuple[str, str, str]]]],
) -> str:
    colspec = "|l|" + "ccc|" * len(accs)
    last_col = 1 + 3 * len(accs)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{4pt}")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\hline")

    hdr1 = ["\\multicolumn{1}{|c|}{Method}"]
    for a in accs:
        hdr1.append(f"\\multicolumn{{3}}{{c|}}{{{a}$\\times$}}")
    out.append(" & ".join(hdr1) + " \\\\")
    out.append(f"\\cline{{2-{last_col}}}")

    hdr2 = ["\\multicolumn{1}{|c|}{}"]
    for _ in accs:
        hdr2 += ["WM $\\downarrow$", "GM $\\downarrow$", "CSF $\\downarrow$"]
    out.append(" & ".join(hdr2) + " \\\\")
    out.append("\\hline")

    for method_disp, cellmap in rows:
        row_cells = [method_disp]
        for a in accs:
            wm_cell, gm_cell, csf_cell = cellmap.get(a, ("NA", "NA", "NA"))
            row_cells += [wm_cell, gm_cell, csf_cell]
        out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\hline")

    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_table_uncertainty(
    caption: str,
    label: str,
    accs: List[str],
    rows: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]],
) -> str:
    # NLL, Cov@1σ, Spearman, AURC
    colspec = "|l|" + "cccc|" * len(accs)
    last_col = 1 + 4 * len(accs)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\hline")

    hdr1 = ["\\multicolumn{1}{|c|}{Method}"]
    for a in accs:
        hdr1.append(f"\\multicolumn{{4}}{{c|}}{{{a}$\\times$}}")
    out.append(" & ".join(hdr1) + " \\\\")
    out.append(f"\\cline{{2-{last_col}}}")

    hdr2 = ["\\multicolumn{1}{|c|}{}"]
    for _ in accs:
        hdr2 += ["NLL $\\downarrow$", "Cov@1$\\sigma$", "Spearman $\\uparrow$", "AURC $\\downarrow$"]
    out.append(" & ".join(hdr2) + " \\\\")
    out.append("\\hline")

    for method_disp, cellmap in rows:
        row_cells = [method_disp]
        for a in accs:
            nll, c1, sp, aurc = cellmap.get(a, ("NA", "NA", "NA", "NA"))
            row_cells += [nll, c1, sp, aurc]
        out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\hline")

    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


def latex_table_uncertainty_calib_consistency(
    caption: str,
    label: str,
    accs: List[str],
    rows: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]],
) -> str:
    # Cov@1σ, Cov@2σ, Cov@3σ, Cross-stage
    colspec = "|l|" + "cccc|" * len(accs)
    last_col = 1 + 4 * len(accs)

    out = []
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\scriptsize")
    out.append("\\setlength{\\tabcolsep}{3pt}")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    out.append(f"\\begin{{tabular}}{{{colspec}}}")
    out.append("\\hline")

    hdr1 = ["\\multicolumn{1}{|c|}{Method}"]
    for a in accs:
        hdr1.append(f"\\multicolumn{{4}}{{c|}}{{{a}$\\times$}}")
    out.append(" & ".join(hdr1) + " \\\\")
    out.append(f"\\cline{{2-{last_col}}}")

    hdr2 = ["\\multicolumn{1}{|c|}{}"]
    for _ in accs:
        hdr2 += ["Cov@1$\\sigma$", "Cov@2$\\sigma$", "Cov@3$\\sigma$", "Cross-stage $\\uparrow$"]
    out.append(" & ".join(hdr2) + " \\\\")
    out.append("\\hline")

    for method_disp, cellmap in rows:
        row_cells = [method_disp]
        for a in accs:
            c1, c2, c3, cs = cellmap.get(a, ("NA", "NA", "NA", "NA"))
            row_cells += [c1, c2, c3, cs]
        out.append(" & ".join(row_cells) + " \\\\")
        out.append("\\hline")

    out.append("\\end{tabular}")
    out.append("\\end{table}")
    return "\n".join(out)


# ============================
# Main
# ============================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=Path, required=True, help=".../_evaluations")
    ap.add_argument("--out_dir", type=Path, default=None, help="Default: <eval_root>/_tables_grouped")
    ap.add_argument("--acc_rates", type=str, default=None, help="Comma-separated list like 4,6,8 (default: all found)")
    ap.add_argument("--mean_only", action="store_true", help="Use mean-only cells (no ± std)")
    args = ap.parse_args()

    eval_root: Path = args.eval_root
    if not eval_root.exists():
        raise SystemExit(f"ERROR: eval_root does not exist: {eval_root}")

    out_dir = args.out_dir if args.out_dir is not None else (eval_root / "_tables_grouped")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_jsons = [
        p for p in eval_root.rglob("eval_summary.json")
        if "_tables" not in p.parts and "_tables_grouped" not in p.parts
    ]
    if not eval_jsons:
        raise SystemExit(f"ERROR: No eval_summary.json found under {eval_root}")

    runs: List[EvalRun] = []
    for p in eval_jsons:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue

        acc, mode = infer_acc_and_mode_from_eval_path(p)
        hetero = infer_hetero_from_path(p)
        rms_w = infer_rms_corr_weight_from_path(p)

        runs.append(EvalRun(
            acc_rate=acc,
            mode=mode,
            heteroscedastic=hetero,
            rms_corr_weight=float(rms_w),
            eval_json=p,
            data=data,
        ))

    all_accs = sorted({r.acc_rate for r in runs if r.acc_rate != "unknown"}, key=lambda x: int(x))
    if args.acc_rates is not None:
        wanted = [a.strip() for a in args.acc_rates.split(",") if a.strip()]
        accs = [a for a in all_accs if a in wanted]
    else:
        accs = all_accs
    if not accs:
        raise SystemExit("ERROR: No acc rates selected/found.")

    # Selection variants (best-per-acc determined by one score)
    selection_variants = [
        ("by_t2", "t2_nrmse_mean", "T$_2^*$ NRMSE"),
        ("by_wm", "wm_t2_nrmse_mean", "WM NRMSE"),
        ("by_gm", "gm_t2_nrmse_mean", "GM NRMSE"),
        ("by_csf", "csf_t2_nrmse_mean", "CSF NRMSE"),
    ]

    def cell_pm(run: Optional[EvalRun], mean_key: str, std_key: str, nd: int) -> Tuple[str, Optional[float]]:
        if run is None:
            return "NA", None
        m = run.mean(mean_key)
        s = run.std(std_key)
        if args.mean_only:
            return _fmt_mean(m, nd), m
        return _fmt_pm(m, s, nd), m

    method_keys = ["PUQ", "HETERO", "CHOL", "LR"]

    for tag, score_key, score_name in selection_variants:
        # Pick best run per acc for each method family
        picked: Dict[str, Dict[str, Optional[EvalRun]]] = {mk: {} for mk in method_keys}
        for acc in accs:
            acc_runs = [r for r in runs if r.acc_rate == acc]

            picked["PUQ"][acc] = pick_best(
                acc_runs, mode_pred=lambda m: m == "none_concat", hetero_pref=False, score_key=score_key
            )
            picked["HETERO"][acc] = pick_best(
                acc_runs, mode_pred=lambda m: m == "none_concat", hetero_pref=True, score_key=score_key
            )
            picked["CHOL"][acc] = pick_best(
                acc_runs, mode_pred=lambda m: m == "cholesky_concat", hetero_pref=True, score_key=score_key
            )
            picked["LR"][acc] = pick_best(
                acc_runs, mode_pred=lambda m: m.startswith("low_rank"), hetero_pref=True, score_key=score_key
            )

        # -------------------------
        # General compact: (NRMSE, SSIM)
        # -------------------------
        gen_rows: List[Tuple[str, Dict[str, Tuple[str, str]]]] = []
        nrmse_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        ssim_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}

        for mk in method_keys:
            disp = method_display(mk, picked[mk], accs)
            cellmap: Dict[str, Tuple[str, str]] = {}
            for acc in accs:
                r = picked[mk].get(acc)
                nrmse_cell, nrmse_mean = cell_pm(r, "t2_nrmse_mean", "t2_nrmse_std", 3)
                ssim_cell, ssim_mean = cell_pm(r, "t2_ssim_mean", "t2_ssim_std", 3)
                cellmap[acc] = (nrmse_cell, ssim_cell)
                nrmse_means_by_acc[acc].append(nrmse_mean)
                ssim_means_by_acc[acc].append(ssim_mean)
            gen_rows.append((disp, cellmap))

        # bold per acc
        for acc in accs:
            bi_nrmse = _min_idx(nrmse_means_by_acc[acc])
            bi_ssim = _max_idx(ssim_means_by_acc[acc])
            for mi in range(len(gen_rows)):
                disp, cellmap = gen_rows[mi]
                nrmse_cell, ssim_cell = cellmap[acc]
                if bi_nrmse is not None and mi == bi_nrmse:
                    nrmse_cell = _bold(nrmse_cell)
                if bi_ssim is not None and mi == bi_ssim:
                    ssim_cell = _bold(ssim_cell)
                cellmap[acc] = (nrmse_cell, ssim_cell)
                gen_rows[mi] = (disp, cellmap)

        tex_general = latex_table_general_nrmse_ssim(
            caption=f"General test metrics under different acceleration rates (best by {score_name} per acceleration).",
            label=f"tab:general_grouped_compact_{tag}",
            accs=accs,
            rows=gen_rows,
        )
        write_tex(out_dir / f"general_grouped_compact_{tag}.tex", tex_general)

        # -------------------------
        # General: (MAE, PSNR)
        # -------------------------
        gen2_rows: List[Tuple[str, Dict[str, Tuple[str, str]]]] = []
        mae_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        psnr_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}

        for mk in method_keys:
            disp = method_display(mk, picked[mk], accs)
            cellmap: Dict[str, Tuple[str, str]] = {}
            for acc in accs:
                r = picked[mk].get(acc)
                mae_cell, mae_mean = cell_pm(r, "t2_mae_mean", "t2_mae_std", 2)
                psnr_cell, psnr_mean = cell_pm(r, "t2_psnr_mean", "t2_psnr_std", 2)
                cellmap[acc] = (mae_cell, psnr_cell)
                mae_means_by_acc[acc].append(mae_mean)
                psnr_means_by_acc[acc].append(psnr_mean)
            gen2_rows.append((disp, cellmap))

        for acc in accs:
            bi_mae = _min_idx(mae_means_by_acc[acc])
            bi_psnr = _max_idx(psnr_means_by_acc[acc])
            for mi in range(len(gen2_rows)):
                disp, cellmap = gen2_rows[mi]
                mae_cell, psnr_cell = cellmap[acc]
                if bi_mae is not None and mi == bi_mae:
                    mae_cell = _bold(mae_cell)
                if bi_psnr is not None and mi == bi_psnr:
                    psnr_cell = _bold(psnr_cell)
                cellmap[acc] = (mae_cell, psnr_cell)
                gen2_rows[mi] = (disp, cellmap)

        tex_general2 = latex_table_general_mae_psnr(
            caption=f"General test metrics (MAE/PSNR) under different acceleration rates (best by {score_name} per acceleration).",
            label=f"tab:general_grouped_mae_psnr_{tag}",
            accs=accs,
            rows=gen2_rows,
        )
        write_tex(out_dir / f"general_grouped_mae_psnr_{tag}.tex", tex_general2)

        # -------------------------
        # Tissue: NRMSE (WM/GM/CSF)
        # -------------------------
        tis_rows: List[Tuple[str, Dict[str, Tuple[str, str, str]]]] = []
        wm_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        gm_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        cs_means_by_acc: Dict[str, List[Optional[float]]] = {a: [] for a in accs}

        for mk in method_keys:
            disp = method_display(mk, picked[mk], accs)
            cellmap: Dict[str, Tuple[str, str, str]] = {}
            for acc in accs:
                r = picked[mk].get(acc)
                wm_cell, wm_mean = cell_pm(r, "wm_t2_nrmse_mean", "wm_t2_nrmse_std", 3)
                gm_cell, gm_mean = cell_pm(r, "gm_t2_nrmse_mean", "gm_t2_nrmse_std", 3)
                cs_cell, cs_mean = cell_pm(r, "csf_t2_nrmse_mean", "csf_t2_nrmse_std", 3)
                cellmap[acc] = (wm_cell, gm_cell, cs_cell)
                wm_means_by_acc[acc].append(wm_mean)
                gm_means_by_acc[acc].append(gm_mean)
                cs_means_by_acc[acc].append(cs_mean)
            tis_rows.append((disp, cellmap))

        for acc in accs:
            bi_wm = _min_idx(wm_means_by_acc[acc])
            bi_gm = _min_idx(gm_means_by_acc[acc])
            bi_cs = _min_idx(cs_means_by_acc[acc])
            for mi in range(len(tis_rows)):
                disp, cellmap = tis_rows[mi]
                wm_cell, gm_cell, cs_cell = cellmap[acc]
                if bi_wm is not None and mi == bi_wm:
                    wm_cell = _bold(wm_cell)
                if bi_gm is not None and mi == bi_gm:
                    gm_cell = _bold(gm_cell)
                if bi_cs is not None and mi == bi_cs:
                    cs_cell = _bold(cs_cell)
                cellmap[acc] = (wm_cell, gm_cell, cs_cell)
                tis_rows[mi] = (disp, cellmap)

        tex_tis = latex_table_tissue_nrmse(
            caption=f"Tissue-specific NRMSE under different acceleration rates (best by {score_name} per acceleration).",
            label=f"tab:tissue_nrmse_grouped_{tag}",
            accs=accs,
            rows=tis_rows,
        )
        write_tex(out_dir / f"tissue_nrmse_grouped_{tag}.tex", tex_tis)

        # -------------------------
        # Uncertainty #1: (NLL, Cov@1σ, Spearman, AURC)
        # Bold: NLL min, Cov@1σ closest to nominal, Spearman max, AURC min
        # -------------------------
        unc_rows: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]] = []
        nll_means: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        c1_gap: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        sp_means: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        aurc_means: Dict[str, List[Optional[float]]] = {a: [] for a in accs}

        for mk in method_keys:
            disp = method_display(mk, picked[mk], accs)
            cellmap: Dict[str, Tuple[str, str, str, str]] = {}
            for acc in accs:
                r = picked[mk].get(acc)

                nll_cell, nll_mean = cell_pm(r, "t2p_nll_mean", "t2p_nll_std", 3)
                c1_cell, c1_mean = cell_pm(r, "t2p_coverage_1sigma_mean", "t2p_coverage_1sigma_std", 3)
                sp_cell, sp_mean = cell_pm(r, "t2p_spearman_err_unc_mean", "t2p_spearman_err_unc_std", 3)
                aurc_cell, aurc_mean = cell_pm(r, "t2p_aurc_mean", "t2p_aurc_std", 2)

                cellmap[acc] = (nll_cell, c1_cell, sp_cell, aurc_cell)

                nll_means[acc].append(nll_mean)
                sp_means[acc].append(sp_mean)
                aurc_means[acc].append(aurc_mean)
                c1_gap[acc].append(None if c1_mean is None else abs(c1_mean - NOM_C1))

            unc_rows.append((disp, cellmap))

        for acc in accs:
            bi_nll = _min_idx(nll_means[acc])
            bi_c1 = _min_idx(c1_gap[acc])  # closest to nominal
            bi_sp = _max_idx(sp_means[acc])
            bi_au = _min_idx(aurc_means[acc])

            for mi in range(len(unc_rows)):
                disp, cellmap = unc_rows[mi]
                nll_cell, c1_cell, sp_cell, au_cell = cellmap[acc]
                if bi_nll is not None and mi == bi_nll:
                    nll_cell = _bold(nll_cell)
                if bi_c1 is not None and mi == bi_c1:
                    c1_cell = _bold(c1_cell)
                if bi_sp is not None and mi == bi_sp:
                    sp_cell = _bold(sp_cell)
                if bi_au is not None and mi == bi_au:
                    au_cell = _bold(au_cell)
                cellmap[acc] = (nll_cell, c1_cell, sp_cell, au_cell)
                unc_rows[mi] = (disp, cellmap)

        tex_unc = latex_table_uncertainty(
            caption=f"Uncertainty metrics under different acceleration rates (best by {score_name} per acceleration).",
            label=f"tab:uncertainty_grouped_{tag}",
            accs=accs,
            rows=unc_rows,
        )
        write_tex(out_dir / f"uncertainty_grouped_{tag}.tex", tex_unc)

        # -------------------------
        # Uncertainty #2: calibration + cross-stage consistency
        # Bold: Cov@1/2/3σ closest to nominal, Cross-stage max
        # -------------------------
        unc2_rows: List[Tuple[str, Dict[str, Tuple[str, str, str, str]]]] = []
        c1_gap2: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        c2_gap2: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        c3_gap2: Dict[str, List[Optional[float]]] = {a: [] for a in accs}
        cs_means2: Dict[str, List[Optional[float]]] = {a: [] for a in accs}

        for mk in method_keys:
            disp = method_display(mk, picked[mk], accs)
            cellmap: Dict[str, Tuple[str, str, str, str]] = {}
            for acc in accs:
                r = picked[mk].get(acc)

                c1_cell, c1_mean = cell_pm(r, "t2p_coverage_1sigma_mean", "t2p_coverage_1sigma_std", 3)
                c2_cell, c2_mean = cell_pm(r, "t2p_coverage_2sigma_mean", "t2p_coverage_2sigma_std", 3)
                c3_cell, c3_mean = cell_pm(r, "t2p_coverage_3sigma_mean", "t2p_coverage_3sigma_std", 3)
                cs_cell, cs_mean = cell_pm(r, "t2p_cross_stage_consistency_mean", "t2p_cross_stage_consistency_std", 3)

                cellmap[acc] = (c1_cell, c2_cell, c3_cell, cs_cell)

                c1_gap2[acc].append(None if c1_mean is None else abs(c1_mean - NOM_C1))
                c2_gap2[acc].append(None if c2_mean is None else abs(c2_mean - NOM_C2))
                c3_gap2[acc].append(None if c3_mean is None else abs(c3_mean - NOM_C3))
                cs_means2[acc].append(cs_mean)

            unc2_rows.append((disp, cellmap))

        for acc in accs:
            bi_c1 = _min_idx(c1_gap2[acc])
            bi_c2 = _min_idx(c2_gap2[acc])
            bi_c3 = _min_idx(c3_gap2[acc])
            bi_cs = _max_idx(cs_means2[acc])

            for mi in range(len(unc2_rows)):
                disp, cellmap = unc2_rows[mi]
                c1_cell, c2_cell, c3_cell, cs_cell = cellmap[acc]
                if bi_c1 is not None and mi == bi_c1:
                    c1_cell = _bold(c1_cell)
                if bi_c2 is not None and mi == bi_c2:
                    c2_cell = _bold(c2_cell)
                if bi_c3 is not None and mi == bi_c3:
                    c3_cell = _bold(c3_cell)
                if bi_cs is not None and mi == bi_cs:
                    cs_cell = _bold(cs_cell)
                cellmap[acc] = (c1_cell, c2_cell, c3_cell, cs_cell)
                unc2_rows[mi] = (disp, cellmap)

        tex_unc2 = latex_table_uncertainty_calib_consistency(
            caption=f"Uncertainty calibration and cross-stage consistency under different acceleration rates (best by {score_name} per acceleration).",
            label=f"tab:uncertainty_calib_consistency_grouped_{tag}",
            accs=accs,
            rows=unc2_rows,
        )
        write_tex(out_dir / f"uncertainty_calib_consistency_grouped_{tag}.tex", tex_unc2)

        print(f"[{tag}] wrote grouped tables for accs={accs}")

    print(f"\nDone. Grouped tables in: {out_dir}")


if __name__ == "__main__":
    main()
