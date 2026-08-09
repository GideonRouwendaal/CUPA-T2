"""
Optimized Regression Trainer with GPU Batch Sampling
Key improvements:
- GPU-based batch sampling for MC modes (faster than CPU sampling)
- All normalization happens after sampling on GPU
- Preserves all existing features: adaptive noise, CV correlation, consistency loss
- Compatible with "Without_Unc" datasets
"""

import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from data.t2star_regress_loader import StaticPixelDataset
from model_zoo.fcn_regress import (FCN,
                                    FCN_Multitask, FCN_Multitask_Heteroscedastic,
                                    MultitaskLoss, NoiseAdaptiveLoss, FCN_with_UncertaintyEncoder, HeteroscedasticMLP,
                                    heteroscedastic_loss)
import re
from collections import defaultdict

import yaml
import json
import hashlib

# ── Path anchors ──────────────────────────────────────────────────────────────
# Derived from this file's location so the project can live anywhere.
# PROJECT_DIR/../.. is the iml-dl repo root; its parent is the CUPA_ROOT root.
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
CUPA_ROOT = REPO_ROOT.parent
RESULTS_REGRESSION = CUPA_ROOT / "results" / "regression"
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================================
# GPU Sampling Helper Functions
# ============================================================================

def _u_hat_linear(u, u_low, u_high):
    u_hat = (u - float(u_low)) / (float(u_high) - float(u_low) + 1e-12)
    return torch.clamp(u_hat, 0.0, 1.0)


def _sanitize_for_path(s: str, maxlen: int = 120) -> str:
    s = str(s)
    # keep it filesystem-safe
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "=", "+", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:maxlen].rstrip("_")


def _stable_hash_dict(d: dict, digest_bytes: int = 8) -> str:
    # deterministic small hash for config identity
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_bytes).hexdigest()


def _make_unique_dir(path: str) -> str:
    """
    If path exists, create path__001, path__002, ...
    Returns the created directory path.
    """
    base = path.rstrip("/")

    if not os.path.exists(base):
        os.makedirs(base, exist_ok=False)
        return base + "/"

    for i in range(1, 10000):
        cand = f"{base}__{i:03d}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand + "/"

    raise RuntimeError(f"Could not create a unique directory for: {base}")



_TISSUE_NAME = {0: "wm", 1: "gm", 2: "csf"}


def _subsample(x: torch.Tensor, k: int):
    """Subsample up to k elements from 1D tensor x."""
    x = x.view(-1)
    n = x.numel()
    if n == 0:
        return x
    if (k is None) or (n <= k):
        return x
    idx = torch.randint(0, n, (k,), device=x.device)
    return x[idx]

def _append_tissue_buffers(buffers, tissue_ids, values, valid_mask=None, max_per_batch=4096):
    """
    buffers: dict like buffers['wm']['eff'] = [tensor, tensor, ...]
    tissue_ids: [B] int
    values: dict of name->tensor, each [B]
    valid_mask: optional [B] bool
    """
    tissue_ids = tissue_ids.view(-1)
    if valid_mask is not None:
        valid_mask = valid_mask.view(-1).bool()
    else:
        valid_mask = torch.ones_like(tissue_ids, dtype=torch.bool)

    for tid, tname in _TISSUE_NAME.items():
        m = (tissue_ids == tid) & valid_mask
        if not m.any():
            continue
        for key, v in values.items():
            vv = v.view(-1)[m]
            vv = _subsample(vv, max_per_batch)
            if vv.numel() > 0:
                buffers[tname][key].append(vv.detach().cpu())

def _finalize_tissue_buffers(buffers, prefix="train/tissue"):
    """
    Returns flat dict of scalars suitable for wandb.log
    Example keys:
      train/tissue/wm/eff/p50
      train/tissue/wm/alpha_eff/p99
    """
    out = {}
    qs = [0.50, 0.90, 0.99]
    for tname, d in buffers.items():
        for key, chunks in d.items():
            if len(chunks) == 0:
                continue
            arr = torch.cat(chunks).float()
            out[f"{prefix}/{tname}/{key}/mean"] = float(arr.mean().item())
            for q in qs:
                out[f"{prefix}/{tname}/{key}/p{int(q*100)}"] = float(torch.quantile(arr, q).item())
            out[f"{prefix}/{tname}/{key}/max"] = float(arr.max().item())
            out[f"{prefix}/{tname}/{key}/n"] = int(arr.numel())
    return out


def _apply_eff_cap(alpha: torch.Tensor, u: torch.Tensor, eff_cap: float, hinge_u0: None):
    """
    alpha: [B,1]
    u:     [B]
    eff_cap: scalar cap on eff = alpha*u
    hinge_u0: if set, apply cap only when u > hinge_u0
    """
    eps = 1e-12
    u_safe = u.clamp_min(eps)                      # [B]
    alpha_cap = (eff_cap / u_safe).view(-1, 1)     # [B,1]
    alpha_eff = torch.minimum(alpha, alpha_cap)    # [B,1]

    if hinge_u0 is not None:
        m = (u > hinge_u0).view(-1, 1)
        alpha_eff = torch.where(m, alpha_eff, alpha)

    return alpha_eff


def _trace_from_factor(A: torch.Tensor) -> torch.Tensor:
    """
    Compute trace(Sigma) given a factor A where Sigma ≈ A A^T.

    A shapes:
      - [B, D, D]  (Cholesky or full factor)
      - [B, D, R]  (Low-rank factor)
      - [B, D]     (Uncorrelated std vector)
    Returns:
      trace: [B]
    """
    if A.dim() == 2:
        return (A ** 2).sum(dim=-1)
    return (A ** 2).sum(dim=(-2, -1))


def _rms_from_factor(A: torch.Tensor, D: int) -> torch.Tensor:
    """RMS-std = sqrt(trace(Sigma)/D)."""
    tr = _trace_from_factor(A)
    return torch.sqrt(tr / float(D) + 1e-12)


def _u_hat_from_quantiles(u: torch.Tensor, valid_mask: torch.Tensor, q_lo: float = 0.50, q_hi: float = 0.90) -> torch.Tensor:
    """
    Map uncertainty score u -> u_hat in [0,1] via quantiles on valid entries.
    """
    if valid_mask.dim() > 1:
        valid_mask = valid_mask.view(-1)
    valid = u[valid_mask]
    if valid.numel() == 0:
        return torch.zeros_like(u)

    q_low = torch.quantile(valid, q_lo)
    q_high = torch.quantile(valid, q_hi)
    u_hat = (u - q_low) / (q_high - q_low + 1e-12)
    return torch.clamp(u_hat, 0.0, 1.0)

def _compute_u_hat_linear(u, u_lo, u_hi):
    u_hat = (u - float(u_lo)) / (float(u_hi) - float(u_lo) + 1e-12)
    return torch.clamp(u_hat, 0.0, 1.0)


def _alpha_from_u_hat(u_hat, alpha_min, alpha_max, gamma=1.0):
    return float(alpha_min) + (float(alpha_max) - float(alpha_min)) * (u_hat ** float(gamma))


def _u_hat_from_fixed(u: torch.Tensor, q_low: float, q_high: float) -> torch.Tensor:
    """Map u -> u_hat in [0,1] using precomputed quantiles."""
    u_hat = (u - float(q_low)) / (float(q_high) - float(q_low) + 1e-12)
    return torch.clamp(u_hat, 0.0, 1.0)

def _weights_and_alpha(
    u_hat: torch.Tensor,
    beta: float,
    w_min: float,
    alpha_min: float,
    alpha_max: float,
    gamma: float,
    ):
    """
    Loss weight:  w = clip(exp(-beta * u_hat), w_min, 1)
    Sample alpha: a = alpha_min + (alpha_max-alpha_min) * u_hat^gamma
    """
    if beta == 0.0:
        w = torch.ones_like(u_hat)
    else:
        w = torch.exp(-beta * u_hat)
        w = torch.clamp(w, min=w_min, max=1.0)
    a = alpha_min + (alpha_max - alpha_min) * (u_hat ** gamma)
    return w, a


def _sample_from_factor(mean_ri: torch.Tensor, A: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Sample: x = mean_ri + alpha * (A @ eps)   (or elementwise if A is std vector)
    mean_ri: [B, 24]
    A:
      - [B, 24, K] or
      - [B, 24]
    alpha: [B,1]
    """
    if A.dim() == 2:
        z = torch.randn_like(mean_ri)
        return mean_ri + alpha * (A * z)

    B = mean_ri.shape[0]
    K = A.shape[-1]
    z = torch.randn(B, K, device=mean_ri.device, dtype=mean_ri.dtype)
    eps = torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)  # [B, 24]
    return mean_ri + alpha * eps


def _ri_to_mag(sampled_ri: torch.Tensor) -> torch.Tensor:
    """[B,24] -> [B,12] magnitude"""
    B, D = sampled_ri.shape
    E = D // 2
    real = sampled_ri[:, :E]
    imag = sampled_ri[:, E:]
    return torch.sqrt(real * real + imag * imag + 1e-12)


def _apply_eff_cap_per_tissue(alpha, u, tissue, caps, hinge_u0=None):
    """
    alpha:  [B,1]
    u:      [B]
    tissue: [B] with 0=WM,1=GM,2=CSF
    caps: dict like {"wm":0.02,"gm":0.02,"csf":0.012}
    """
    # compute capped alpha for each tissue type, starting from current alpha
    alpha_out = alpha

    if caps is None:
        return alpha_out

    # helper to cap only where mask is true
    def cap_where(mask, cap_val):
        if cap_val is None:
            return alpha_out
        alpha_cap = _apply_eff_cap(alpha_out, u, float(cap_val),
                                   None if hinge_u0 is None else float(hinge_u0))
        return torch.where(mask.view(-1,1), alpha_cap, alpha_out)

    alpha_out = cap_where(tissue == 0, caps.get("wm", None))
    alpha_out = cap_where(tissue == 1, caps.get("gm", None))
    alpha_out = cap_where(tissue == 2, caps.get("csf", None))
    return alpha_out



def build_features_from_batch(batch, device, schedule_cfg=None, deterministic=False):
    if schedule_cfg is None:
        schedule_cfg = {}

    beta      = float(schedule_cfg.get("beta", 2.0))
    w_min     = float(schedule_cfg.get("w_min", 0.15))
    q_lo      = float(schedule_cfg.get("q_lo", 0.50))
    q_hi      = float(schedule_cfg.get("q_hi", 0.90))
    alpha_min = float(schedule_cfg.get("alpha_min", 0.05))
    alpha_max = float(schedule_cfg.get("alpha_max", 0.50))
    gamma     = float(schedule_cfg.get("gamma", 1.0))

    eff_cap  = schedule_cfg.get("eff_cap", None)
    hinge_u0 = schedule_cfg.get("hinge_u0", None)

    mode = str(schedule_cfg.get("mode", "quantile")).lower()


    mean_ri = batch["mean_ri"].to(device, dtype=torch.float32)
    A       = batch["L_or_std"].to(device, dtype=torch.float32)
    std_mag = batch.get("std_mag", None)
    if std_mag is not None and type(std_mag) == torch.Tensor:
        std_mag = std_mag.to(device, dtype=torch.float32)

    # squeeze singleton dataset dim
    if mean_ri.dim() == 3 and mean_ri.shape[1] == 1:
        mean_ri = mean_ri.squeeze(1)
    if A.dim() >= 3 and A.shape[1] == 1:
        A = A.squeeze(1)
    if type(std_mag) == torch.Tensor:
        if std_mag is not None and std_mag.dim() == 3 and std_mag.shape[1] == 1:
            std_mag = std_mag.squeeze(1)

    valid_mask = batch.get("valid_mask", None)
    if valid_mask is None:
        valid_mask = torch.ones(mean_ri.shape[0], device=device, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device).view(-1)

    # uncertainty score u = RMS-std from factor
    D = mean_ri.shape[-1]  # 24
    u = _rms_from_factor(A, D)  # [B]
    mag_mean = _ri_to_mag(mean_ri)              # [B,12]
    nf0 = mag_mean[:, 0:1].clamp_min(1e-9)     # [B,1]
    # u = u
    # u = u / nf0.squeeze(1)                        # [B] dimensionless uncertainty score (RMS-std relative to TE1 magnitude)
    u = u
    # if mode == "linear_u":
        
    #     u_lo = float(schedule_cfg["u_lo"])
    #     u_hi = float(schedule_cfg["u_hi"])
    #     u_hat = _u_hat_linear(u, u_lo, u_hi)

    # else:
    #     if bool(schedule_cfg.get("use_fixed_quantiles", False)):
    #         q_low  = float(schedule_cfg["fixed_q50"])
    #         q_high = float(schedule_cfg["fixed_q90"])
    #         u_hat = _u_hat_from_fixed(u, q_low=q_low, q_high=q_high)
    #     else:
    #         u_hat = _u_hat_from_quantiles(u, valid_mask, q_lo=q_lo, q_hi=q_hi)

    # compute GLOBAL u_hat (for logging + optional global weights)
    # NOTE: tissue_schedule will compute its own u_hat_t for alpha later.
    if bool(schedule_cfg.get("use_fixed_quantiles", False)):
        q_low  = float(schedule_cfg["fixed_q50"])
        q_high = float(schedule_cfg["fixed_q90"])
        u_hat = _u_hat_from_fixed(u, q_low=q_low, q_high=q_high)
    else:
        u_hat = _u_hat_from_quantiles(u, valid_mask, q_lo=q_lo, q_hi=q_hi)

    if deterministic:
        # no curriculum in validation
        w = torch.ones_like(u_hat)
        alpha = torch.zeros_like(u_hat).view(-1, 1)

        # deterministic feature = magnitude(mean_ri)
        mag_mean = _ri_to_mag(mean_ri)              # [B,12]
        nf = mag_mean[:, 0:1].clamp_min(1e-9)
        feat = mag_mean / nf

        if std_mag is not None and type(std_mag) == torch.Tensor:
            feat = torch.cat([feat, std_mag / nf], dim=1)

        aux = {
            "u": u.detach(),
            "u_hat": u_hat.detach(),
            "loss_weight": w.detach(),
            "alpha": alpha.view(-1).detach(),
        }
        return feat, None, aux

    # inside build_features_from_batch, after u computed:
    tissue = batch.get("tissue_type", None)
    tissue_schedule = schedule_cfg.get("tissue_schedule", None)

    using_tissue_schedule = (tissue is not None) and (tissue_schedule is not None)


    if (tissue is not None) and (tissue_schedule is not None):
        tissue = tissue.to(device).view(-1)

        # start with zeros; fill per tissue
        alpha = torch.zeros_like(u).view(-1, 1)
        w = torch.ones_like(u_hat) if beta == 0.0 else torch.clamp(torch.exp(-beta * u_hat), min=w_min, max=1.0)


        # per tissue loop
        for tid, tname in [(0, "wm"), (1, "gm"), (2, "csf")]:
            cfg_t = tissue_schedule.get(tname, None)
            if cfg_t is None:
                continue

            m = (tissue == tid) & valid_mask
            if not m.any():
                continue

            mode_t = str(schedule_cfg.get("mode", "quantile")).lower()
            if mode_t == "linear_u":
                # manually specified absolute u thresholds
                u_hat_t = _compute_u_hat_linear(u[m], cfg_t["u_low"], cfg_t["u_hi"])
            elif mode_t == "tissue_quantile":
                # precomputed tissue-specific quantile anchors (written by _compute_tissue_quantiles_from_train)
                u_lo_val = float(cfg_t.get("u_lo_val", 0.0))
                u_hi_val = float(cfg_t.get("u_hi_val", 1.0))
                u_hat_t = _compute_u_hat_linear(u[m], u_lo_val, u_hi_val)
            else:
                # fallback: use global u_hat
                u_hat_t = u_hat[m]

            alpha_t = _alpha_from_u_hat(
                u_hat_t,
                cfg_t.get("alpha_min", 0.0),
                cfg_t.get("alpha_max", 0.0),
                cfg_t.get("gamma", schedule_cfg.get("gamma", 1.0)),
            ).view(-1, 1)

            # tissue-specific eff cap (+ hinge)
            eff_cap_t  = cfg_t.get("eff_cap", None)
            hinge_u0_t = cfg_t.get("hinge_u0", None)
            if eff_cap_t is not None:
                alpha_t = _apply_eff_cap(alpha_t, u[m], float(eff_cap_t),
                                        None if hinge_u0_t is None else float(hinge_u0_t))

            alpha[m] = alpha_t

        # optional: overwrite global alpha/w with tissue-specific
        # (u_hat in aux can remain global; or you can store u_hat_t per tissue if you want)
    else:
        
        # ALWAYS compute weights + alpha
        w, alpha = _weights_and_alpha(
            u_hat, beta=beta, w_min=w_min,
            alpha_min=alpha_min, alpha_max=alpha_max, gamma=gamma
        )
        alpha = alpha.view(-1, 1)  # [B,1]

    if not using_tissue_schedule:
        if bool(schedule_cfg.get("use_tissue_aware_eff_cap", False)) and ("tissue_type" in batch):
            caps = schedule_cfg.get("eff_cap_by_tissue", None)
            tissue2 = batch["tissue_type"].to(device).view(-1)
            alpha = _apply_eff_cap_per_tissue(alpha, u, tissue2, caps, hinge_u0=hinge_u0)

        use_tissue_cap = bool(schedule_cfg.get("use_tissue_aware_eff_cap", False))
        eff_cap_csf = schedule_cfg.get("eff_cap_csf", None)

        if eff_cap is not None:
            alpha = _apply_eff_cap(alpha, u, float(eff_cap), None if hinge_u0 is None else float(hinge_u0))

            if use_tissue_cap and (eff_cap_csf is not None) and ("tissue_type" in batch):
                tissue2 = batch["tissue_type"].to(device).view(-1)
                alpha_csf = _apply_eff_cap(alpha, u, float(eff_cap_csf), None if hinge_u0 is None else float(hinge_u0))
                alpha = torch.where((tissue2 == 2).view(-1, 1), alpha_csf, alpha)

    # if bool(schedule_cfg.get("use_tissue_aware_eff_cap", False)) and ("tissue_type" in batch):
    #     caps = schedule_cfg.get("eff_cap_by_tissue", None)
    #     tissue = batch["tissue_type"].to(device).view(-1)
    #     alpha = _apply_eff_cap_per_tissue(alpha, u, tissue, caps, hinge_u0=hinge_u0)


    # use_tissue_cap = bool(schedule_cfg.get("use_tissue_aware_eff_cap", False))
    # eff_cap_csf = schedule_cfg.get("eff_cap_csf", None)

    # if eff_cap is not None:
    #     alpha = _apply_eff_cap(alpha, u, float(eff_cap), None if hinge_u0 is None else float(hinge_u0))

    #     if use_tissue_cap and (eff_cap_csf is not None) and ("tissue_type" in batch):
    #         tissue = batch["tissue_type"].to(device).view(-1)
    #         # compute the CSF-capped alpha
    #         alpha_csf = _apply_eff_cap(alpha, u, float(eff_cap_csf), None if hinge_u0 is None else float(hinge_u0))
    #         # overwrite only CSF
    #         alpha = torch.where((tissue == 2).view(-1,1), alpha_csf, alpha)


    fallback_mode = str(schedule_cfg.get("fallback_mode", "none")).lower()
    vm = valid_mask.view(-1, 1)  # [B,1]

    # --- CSF-only mask ---
    tissue = batch.get("tissue_type", None)
    if tissue is not None:
        tissue = tissue.to(device).view(-1)
        csf_mask = (tissue == 2).view(-1, 1) & vm     # [B,1]
    else:
        # If tissue labels not available, fall back to "all valid"
        csf_mask = vm

    if fallback_mode == "hard":
        u_cut = float(schedule_cfg["fallback_u_cut"])
        m = (u > u_cut).view(-1, 1) & csf_mask
        alpha = torch.where(m, torch.zeros_like(alpha), alpha)

    elif fallback_mode == "taper":
        u_start = float(schedule_cfg["fallback_u_start"])
        u_end   = float(schedule_cfg["fallback_u_end"])
        t = (u - u_start) / (u_end - u_start + 1e-12)      # [B]
        t = torch.clamp(t, 0.0, 1.0).view(-1, 1)           # [B,1]
        # only taper CSF
        alpha = torch.where(csf_mask, alpha * (1.0 - t), alpha)

    elif fallback_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown fallback_mode: {fallback_mode}")


    # sample and normalize
    sampled_ri = _sample_from_factor(mean_ri, A, alpha)
    mag = _ri_to_mag(sampled_ri)
    nf = mag[:, 0:1].clamp_min(1e-9)
    feat_mag = mag / nf0


    # deterministic counterpart (for delta logging)
    with torch.no_grad():
        mag_mean = _ri_to_mag(mean_ri)               # [B,12]
        nf0 = mag_mean[:, 0:1].clamp_min(1e-9)
        feat_det_mag = mag_mean / nf0                # [B,12]

        delta = (feat_mag - feat_det_mag).abs()
        delta_mean_abs = float(delta.mean().item())
        delta_p95_abs  = float(torch.quantile(delta.flatten(), 0.95).item())

        u_mean        = float(u.mean().item())
        u_hat_mean    = float(u_hat.mean().item())
        alpha_mean    = float(alpha.mean().item())
        alpha_max_val = float(alpha.max().item())
        w_mean        = float(w.mean().item())
        uhat_frac0    = float((u_hat <= 1e-6).float().mean().item())
        uhat_frac1    = float((u_hat >= 1.0 - 1e-6).float().mean().item())

    # now build the final feature vector
    feat = feat_mag
    if std_mag is not None and type(std_mag) == torch.Tensor:
        feat = torch.cat([feat_mag, std_mag / nf0], dim=1)

    # second draw (optional)
    feat_2 = None
    need_pair = batch.get("need_pair", False)
    if isinstance(need_pair, torch.Tensor):
        need_pair = bool(need_pair.any().item())
    if need_pair:
        sampled_ri_2 = _sample_from_factor(mean_ri, A, alpha)
        mag2 = _ri_to_mag(sampled_ri_2)
        nf2 = mag2[:, 0:1].clamp_min(1e-9)
        feat_2 = mag2 / nf2
        if std_mag is not None:
            feat_2 = torch.cat([feat_2, std_mag / nf2], dim=1)

    # IMPORTANT: aux must include these tensors (used downstream)
    aux = {
        "u": u.detach(),                      # [B]
        "u_hat": u_hat.detach(),              # [B]
        "loss_weight": w.detach(),            # [B]
        "alpha": alpha.detach(),              # [B,1]

        # optional scalar diagnostics
        "delta_mean_abs": delta_mean_abs,
        "delta_p95_abs":  delta_p95_abs,
        "u_mean":         u_mean,
        "u_hat_mean":     u_hat_mean,
        "alpha_mean":     alpha_mean,
        "alpha_max":      alpha_max_val,
        "w_mean":         w_mean,
        "uhat_frac0":     uhat_frac0,
        "uhat_frac1":     uhat_frac1,
    }

    delta_per = (feat_mag - feat_det_mag).abs().mean(dim=1)   # [B]
    aux["delta_per"] = delta_per.detach()
    aux["nf0"] = nf0.squeeze(1).detach()                      # [B]
    aux["alpha_eff"] = alpha.squeeze(1).detach()              # [B]
    aux["eff"] = (alpha.squeeze(1) * u).detach()              # [B]

    return feat, feat_2, aux



def _compute_rms_correlation_loss(self, pred_log_var, batch, eps=1e-8):
    # batch["u_rms"] should be [B,1] or [B]
    u = batch["u_rms"].to(self.device).view(-1)
    log_u = torch.log(u + eps)

    log_sigma = 0.5 * pred_log_var
    log_sigma = log_sigma.view(-1)

    x = log_sigma - log_sigma.mean()
    y = log_u - log_u.mean()
    corr = (x*y).mean() / (x.std(unbiased=False)*y.std(unbiased=False) + eps)
    corr = corr.clamp(-1.0, 1.0)
    return 1.0 - corr


def _is_mc_ingredient_batch(batch) -> bool:
    """Check if batch contains MC ingredients (for GPU sampling)."""
    return ("mean_ri" in batch) and (batch["mean_ri"] is not None)


# ============================================================================
# Regression Trainer
# ============================================================================

class RegressionTrainer:
    """Handles training of regression models for T2* and S0 prediction."""
    
    def __init__(self, config, device, recon_model=None):
        self.config = config
        self.device = device
        self.recon_model = recon_model
        
        # Extract configuration
        self.regression_params = config['regression_params']
        self.train_params = self.regression_params['regression_train_params']
        self.model_params = self.regression_params['regression_model_params']
        
        # Parameter ranges
        self.max_t2 = config.get('max_t2', 200.0)
        self.min_t2 = config.get('min_t2', 0.0)
        self.max_s0 = self.regression_params.get('s0_max', 2.5)
        self.min_s0 = self.regression_params.get('s0_min', 0.0)
        
        self.acc_rate = str(self.regression_params.get('acc_rate', 'high'))

        # Uncertainty configuration
        self.uncertainty_mode = self.regression_params.get('uncertainty_mode', 'none')  
        self.lowrank_rank = self.regression_params.get('lowrank_rank', None)
        self.use_uncertainty_encoder = self.model_params.get('use_uncertainty_encoder', False)

        # New RMS-quantile uncertainty schedule (replaces old adaptive noise/loss)
        self.uncertainty_schedule_cfg = self.train_params.get("uncertainty_schedule", {
            "beta": 2.0,
            "w_min": 0.15,
            "q_lo": 0.50,
            "q_hi": 0.90,
            "alpha_min": 0.05,
            "alpha_max": 0.50,
            "gamma": 1.0,
            "compute_quantiles_at_start": True,
            "max_quantile_samples": 500000,
            "use_fixed_quantiles": False,
            "fixed_q50": None,
            "fixed_q90": None,
        })

        # Force-disable legacy adaptive strategies (overridden by uncertainty_schedule_cfg)
        self.use_adaptive_noise = False
        self.use_adaptive_loss = False

        # Uncertainty model settings
        self.heteroscedastic = self.regression_params.get('heteroscedastic', False)
        self.use_cv_correlation_loss = self.train_params.get('use_cv_correlation_loss', False)
        self.cv_correlation_weight = self.train_params.get('cv_correlation_weight', 0.1)
        if self.use_cv_correlation_loss:
            logging.info(f'Using CV correlation loss with weight {self.cv_correlation_weight}')
        self.use_mc_dropout = self.regression_params.get('use_mc_dropout', False)
        
        # Loss
        self.loss = self.train_params.get('loss', 'L1')
        self.include_physics_loss = self.train_params.get('include_physics_loss', False)
        self.weighted_physics_loss = self.train_params.get('weighted_physics_loss', False)
        self.noise_schedule = self.train_params.get('noise_schedule', 'constant')
        self.noise_scale = self.train_params.get('noise_scale', 1.0)
        self.use_consistency_loss = self.train_params.get('use_consistency_loss', False)
        self.consistency_weight = self.train_params.get('consistency_weight', 0.1)

        self.use_rms_correlation_loss = self.train_params.get('use_rms_correlation_loss', False)
        self.rms_correlation_weight = self.train_params.get('rms_correlation_weight', 0.1)
        if self.use_rms_correlation_loss:
            logging.info(f'Using RMS correlation loss with weight {self.rms_correlation_weight}')
        
        # Adaptive Loss
        self.use_adaptive_loss = False
        self.adaptive_loss_max_discount = self.train_params.get('adaptive_loss_max_discount', 0.5)
        self.adaptive_loss_mode = self.train_params.get('adaptive_loss_mode', 'auto')

        # Adaptive Noise (disabled; replaced by uncertainty_schedule)
        self.use_adaptive_noise = False

        # Model configuration
        self.activation = self.model_params.get('activation', "None")
        self.add_positional_encoding = self.regression_params.get('add_positional_encoding', False)
        self.pe_dim = self.regression_params.get('pe_dim', 32)
        self.param_to_predict = self.regression_params.get('param_to_predict', 't2star')
        
        self.model = None
        self.optimizer = None

        # Multitask learning (optional)
        self.multitask = self.regression_params.get('multitask', False)
        self.num_tissue_classes = self.regression_params.get('num_tissue_classes', 3)
        self.alpha_tissue = self.train_params.get('alpha_tissue', 0.3)
        
        self.tissue_class_weights = self.train_params.get('tissue_class_weights', None)
        if self.tissue_class_weights:
            self.tissue_class_weights = torch.tensor(self.tissue_class_weights).to(device)
        self.criterion = None


    def _evaluate_best_checkpoint(self, output_path: str):
        ckpt = os.path.join(output_path, "best_regression_model.pth")
        if not os.path.exists(ckpt):
            logging.warning(f"No checkpoint found at {ckpt} — cannot evaluate best model.")
            return None

        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # epoch arg is not used for anything important in _validate_epoch right now
        best_val_metrics = self._validate_epoch(epoch=-1)

        # Print tissue-specific performance (T2*)
        lines = []
        lines.append("=== BEST CHECKPOINT: Tissue-specific T2* performance ===")
        for t in ["wm", "gm", "csf"]:
            k_nrmse = f"val_nrmse_t2_{t}"
            k_mae   = f"val_mae_t2_{t}"
            k_n     = f"val_n_{t}"
            if k_nrmse in best_val_metrics:
                lines.append(
                    f"{t.upper():>3}: "
                    f"NRMSE={best_val_metrics.get(k_nrmse):.6f} | "
                    f"MAE={best_val_metrics.get(k_mae, float('nan')):.6f} | "
                    f"N={best_val_metrics.get(k_n, 0)}"
                )

        msg = "\n".join(lines)
        logging.info(msg)
        print(msg)

        return best_val_metrics

    def setup_output_directory(self, base_path=None):
        """Create output directory based on configuration.

        Defaults to <repo>/results/regression; regress_model_dir overrides it, and
        the evaluator reads the same key so training and evaluation stay in step.
        """
        if base_path is None:
            base_path = self.regression_params.get("regress_model_dir") or str(RESULTS_REGRESSION)
        path = str(base_path).rstrip("/") + "/"

        loc = self.regression_params["regression_gt_train_location"]

        m = re.search(r"(Exp\d+)", loc)   # finds Exp6, Exp12, etc.
        if m:
            path += f"{m.group(1)}/"


        # Acceleration rate
        path += f'acc_rate_{self.acc_rate}/'
        
        # Uncertainty mode
        path += f'{self.uncertainty_mode}/'

        if "lowrank" in self.uncertainty_mode.lower() or "low_rank" in self.uncertainty_mode.lower():
            rank = self.lowrank_rank
            path += f'rank_{rank}/'
        
        # Output channels
        out_ch = self.model_params['out_ch']
        path += 'T2_and_S0/' if self.param_to_predict == 'both' else ('T2_only/' if out_ch == 1 and self.param_to_predict == 't2star' else 'S0_only/')
        
        # Physics loss
        if self.include_physics_loss:
            path += 'physics_loss/'
            path += 'weighted/' if self.weighted_physics_loss else 'unweighted/'
        else:
            path += 'no_physics/'
        
        # Model architecture
        path += 'heteroscedastic/' if self.heteroscedastic else 'homoscedastic/'
        path += 'mc_dropout/' if self.use_mc_dropout else 'no_dropout/'
        path += 'with_pe/' if self.add_positional_encoding else 'no_pe/'
        path += 'multitask/' if self.multitask else ''
        path += 'uncertainty_encoder/' if self.use_uncertainty_encoder else ''
        
        path += self.loss + '/'
        path += self.noise_schedule + '/' if self.noise_schedule != 'constant' else ''
        if self.noise_schedule != 'constant':
            path += f'noise_scale_{self.noise_scale}/'

        # Adaptive strategies
        path += 'adaptive_loss/' if self.use_adaptive_loss else ''
        path += 'adaptive_noise/' if self.use_adaptive_noise else ''
        path += 'consistency_loss/' if self.use_consistency_loss else ''

        if self.use_cv_correlation_loss:
            path += 'cv_corr_loss/'
            path += f'cv_corr_weight_{self.cv_correlation_weight}/'

        if self.use_rms_correlation_loss:
            path += 'rms_corr_loss/'
            path += f'rms_corr_weight_{self.rms_correlation_weight}/'


        # -----------------------------
        # NEW: unique run subdir to avoid overwriting sweep runs
        # -----------------------------
        sched = self.train_params.get("uncertainty_schedule", {})
        run_tag = self.regression_params.get("run_tag", None)

        # hash only the knobs that actually change run identity
        id_dict = {
            "acc_rate": self.acc_rate,
            "uncertainty_mode": self.uncertainty_mode,
            "lowrank_rank": self.lowrank_rank,
            "heteroscedastic": self.heteroscedastic,
            "use_mc_dropout": self.use_mc_dropout,
            "dropout_rate": self.model_params.get("dropout_rate", None),
            "loss": self.loss,
            "use_rms_correlation_loss": self.use_rms_correlation_loss,
            "rms_correlation_weight": self.rms_correlation_weight,
            "use_cv_correlation_loss": self.use_cv_correlation_loss,
            "cv_correlation_weight": self.cv_correlation_weight,
            "use_consistency_loss": self.use_consistency_loss,
            "consistency_weight": self.consistency_weight,
            "uncertainty_schedule": sched,
        }
        run_hash = _stable_hash_dict(id_dict, digest_bytes=8)

        if run_tag is None:
            run_tag = f"run_{run_hash}"
        else:
            run_tag = _sanitize_for_path(run_tag)

        path = os.path.join(path, f"{run_tag}_{run_hash}")

        # Make it unique on disk even if something collides
        path = _make_unique_dir(path)

        logging.info(f'Output directory: {path}')
        return path

        os.makedirs(path, exist_ok=True)
        
        logging.info(f'Output directory: {path}')
        
        return path
    
    def setup_datasets(self):
        """Setup training and validation datasets."""
        # Common dataset parameters
        dataset_params = {
            'te_values': self.regression_params.get('te_values', list(range(5, 65, 5))),
            'min_t2': self.min_t2,
            'max_t2': self.max_t2,
            'min_s0': self.min_s0,
            'max_s0': self.max_s0,
            'uncertainty_mode': self.uncertainty_mode,
            'lowrank_rank': self.lowrank_rank,
            'noise_schedule': self.noise_schedule,
            'use_adaptive_loss': self.use_adaptive_loss,
            'use_adaptive_noise': self.use_adaptive_noise,
            'noise_scale': self.noise_scale,
            'heteroscedastic': self.heteroscedastic,
            'use_cv_correlation_loss': self.use_cv_correlation_loss,
            'use_consistency_loss': self.use_consistency_loss,
            'use_rms_correlation_loss': self.use_rms_correlation_loss,
            'rms_correlation_weight': self.rms_correlation_weight,
            
            # Adaptive noise calibration parameters
            'adapt_q_lo': self.regression_params.get('adapt_q_lo', 0.50),
            'adapt_q_hi': self.regression_params.get('adapt_q_hi', 0.75),
            'adapt_target_lo': self.regression_params.get('adapt_target_lo', 0.10),
            'adapt_target_hi': self.regression_params.get('adapt_target_hi', 0.15),
            'adapt_scale_clip': tuple(self.regression_params.get('adapt_scale_clip', [0.0, 2.0])),
            'adapt_eff_cap': self.regression_params.get('adapt_eff_cap', None),
            'save_noise_calibration_path': self.regression_params.get('save_noise_calibration_path', None),
        }
        
        logging.info(f'Using uncertainty mode: {self.uncertainty_mode}')
        logging.info(f'Number of hidden layers: {self.model_params["num_layers"]}')
        logging.info(f'LR: {self.train_params["lr"]}')
        if self.lowrank_rank:
            logging.info(f'Using lowrank rank: {self.lowrank_rank}')
        if self.noise_schedule != 'constant':
            logging.info(f'Using noise schedule: {self.noise_schedule}')
        if self.use_adaptive_loss:
            logging.info(f'Using adaptive loss weighting (max_discount={self.adaptive_loss_max_discount})')

        # Training dataset
        train_dataset = StaticPixelDataset(
            data_path_input=self.regression_params['regression_gt_train_location'],
            mode='train',
            **dataset_params
        )

        # Validation dataset
        val_dataset = StaticPixelDataset(
            data_path_input=self.regression_params['regression_gt_val_location'],
            mode='val',
            **dataset_params
        )

        # Create dataloaders
        batch_size = self.train_params['batch_size']
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )

        logging.info(f'Training: {len(train_dataset)} pixels')
        logging.info(f'Validation: {len(val_dataset)} pixels')
    
    def _initialize_loss_function(self):
        if self.use_adaptive_loss and not self.heteroscedastic and not self.multitask:
            self.criterion = NoiseAdaptiveLoss(
                base_loss=self.loss,
                max_discount=self.adaptive_loss_max_discount,
                mode=self.adaptive_loss_mode
            )
            logging.info(f'Using NoiseAdaptiveLoss (max_discount={self.adaptive_loss_max_discount})')
        
        elif self.heteroscedastic:
            self.criterion = heteroscedastic_loss
        elif self.loss.upper() == 'L1':
            print("Using L1 Loss")
            self.criterion = nn.L1Loss()
        elif self.loss.upper() == 'L2':
            print("Using L2 Loss")
            self.criterion = nn.MSELoss()
        else:
            self.criterion = None




    def _compute_global_quantiles_from_train(self):
        """Compute global q_lo/q_hi of RMS uncertainty on TRAIN loader once and freeze.

        This avoids per-batch quantile drift when the dataset grows.
        Uses reservoir sampling to limit memory.
        """
        sched = self.uncertainty_schedule_cfg
        if not bool(sched.get("compute_quantiles_at_start", True)):
            return

        # If already provided, do nothing
        if bool(sched.get("use_fixed_quantiles", False)) and (sched.get("fixed_q50") is not None) and (sched.get("fixed_q90") is not None):
            return

        q_lo = float(sched.get("q_lo", 0.50))
        q_hi = float(sched.get("q_hi", 0.90))
        max_keep = int(sched.get("max_quantile_samples", 500000))

        reservoir = None
        n_seen = 0

        def reservoir_update(arr_np: np.ndarray):
            nonlocal reservoir, n_seen
            arr_np = np.asarray(arr_np, dtype=np.float32).ravel()
            if arr_np.size == 0:
                return

            # If reservoir not initialized, start empty
            if reservoir is None:
                reservoir = np.empty((0,), dtype=np.float32)

            # 1) Fill phase: append until reservoir reaches max_keep
            if reservoir.size < max_keep:
                need = max_keep - reservoir.size
                take = min(need, arr_np.size)
                if take > 0:
                    reservoir = np.concatenate([reservoir, arr_np[:take]])
                    n_seen += take
                    arr_np = arr_np[take:]
                if arr_np.size == 0:
                    return

            # 2) Replacement phase: reservoir sampling
            for v in arr_np:
                n_seen += 1
                j = np.random.randint(0, n_seen)  # [0, n_seen-1]
                if j < max_keep:
                    reservoir[j] = v

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Prepass: global RMS quantiles", leave=False):
                if not _is_mc_ingredient_batch(batch):
                    continue
                mean_ri = batch["mean_ri"].to(self.device, dtype=torch.float32)
                A = batch["L_or_std"].to(self.device, dtype=torch.float32)

                if mean_ri.dim() == 3 and mean_ri.shape[1] == 1:
                    mean_ri = mean_ri.squeeze(1)
                if A.dim() >= 3 and A.shape[1] == 1:
                    A = A.squeeze(1)

                valid_mask = batch.get("valid_mask", None)
                if valid_mask is None:
                    valid_mask = torch.ones(mean_ri.shape[0], device=self.device, dtype=torch.bool)
                else:
                    valid_mask = valid_mask.to(self.device).view(-1)

                mag0 = _ri_to_mag(mean_ri)
                nf0  = mag0[:, 0:1].clamp_min(1e-9)          # [B,1]
                D    = mean_ri.shape[-1]
                u_raw  = _rms_from_factor(A, D)              # [B]
                u_norm = u_raw / nf0.squeeze(1)              # [B]
                u_norm = u_raw

                u_norm = u_norm[valid_mask]
                reservoir_update(u_norm.detach().cpu().numpy())

                # D = mean_ri.shape[-1]  # 24
                # u = _rms_from_factor(A, D)
                # # nf0 = mag_mean[:, 0].clamp_min(1e-9)
                # u_valid = u[valid_mask]
                # reservoir_update(u_valid.detach().cpu().numpy())

        if reservoir is None or reservoir.size == 0:
            logging.warning("Global quantile prepass found no valid voxels; keeping per-batch quantiles.")
            return
        else:
            logging.info(f"Global quantile prepass saw {n_seen} valid voxels; reservoir size {reservoir.size}.")
            q10 = float(np.quantile(reservoir, 0.10))
            q50 = float(np.quantile(reservoir, 0.50))
            q90 = float(np.quantile(reservoir, 0.90))
            q99 = float(np.quantile(reservoir, 0.99))
            logging.info(f"RMS quantiles: q10={q10:.6g}, q50={q50:.6g}, q90={q90:.6g}, q99={q99:.6g}")


        q50 = float(np.quantile(reservoir, q_lo))
        q90 = float(np.quantile(reservoir, q_hi))
        if q90 <= q50:
            logging.warning(f"Global quantiles degenerate: q{int(q_lo*100)}={q50:.6g}, q{int(q_hi*100)}={q90:.6g}. Keeping per-batch.")
            return

        sched["use_fixed_quantiles"] = True
        sched["fixed_q50"] = q50
        sched["fixed_q90"] = q90

        # ensure wandb config sees frozen values
        self.config["regression_params"]["regression_train_params"]["uncertainty_schedule"] = sched

        logging.info(f"Frozen global RMS quantiles: q{int(q_lo*100)}={q50:.6g}, q{int(q_hi*100)}={q90:.6g} (kept {min(max_keep, n_seen)} samples)")


    def _compute_rms_correlation_loss(self, pred_log_var, batch, eps=1e-8):
        u = batch["rms"].to(self.device).view(-1)    # <-- you already set this in _prepare_batch
        log_u = torch.log(u + eps)

        log_sigma = 0.5 * pred_log_var
        log_sigma = log_sigma.view(-1)

        x = log_sigma - log_sigma.mean()
        y = log_u - log_u.mean()
        corr = (x*y).mean() / (x.std(unbiased=False)*y.std(unbiased=False) + eps)
        corr = corr.clamp(-1.0, 1.0)
        return 1.0 - corr


    def _alpha_from_u_norm(self,
                        u_norm: torch.Tensor,
                        q_lo_val: float,
                        q_hi_val: float,
                        alpha_min: float,
                        alpha_max: float,
                        gamma: float) -> torch.Tensor:
        # Normalize to [0,1] between q_lo and q_hi
        denom = max(q_hi_val - q_lo_val, 1e-12)
        t = (u_norm - q_lo_val) / denom
        t = torch.clamp(t, 0.0, 1.0)

        # Shape
        if gamma != 1.0:
            t = t ** gamma

        # Map to [alpha_min, alpha_max]
        return alpha_min + (alpha_max - alpha_min) * t

    def _compute_tissue_quantiles_from_train(self):
        """Compute tissue-specific quantiles of u_norm on TRAIN and print/log them.
        Also computes/prints per-tissue alpha distribution using the SAME global anchors
        that will be used for sampling (unless you choose tissue anchors).
        """
        sched = self.uncertainty_schedule_cfg
        if not bool(sched.get("compute_quantiles_at_start", True)):
            return

        q_lo = float(sched.get("q_lo", 0.50))
        q_hi = float(sched.get("q_hi", 0.90))
        max_keep = int(sched.get("max_quantile_samples", 500000))

        alpha_min = float(sched.get("alpha_min", 0.05))
        alpha_max = float(sched.get("alpha_max", 0.5))
        gamma     = float(sched.get("gamma", 1.0))

        # one reservoir per tissue (u_norm)
        reservoirs = {"wm": None, "gm": None, "csf": None}
        n_seen = {"wm": 0, "gm": 0, "csf": 0}

        # global reservoir for anchors (u_norm)
        global_res = None
        global_seen = 0

        # alpha reservoirs per tissue
        alpha_reservoirs = {"wm": None, "gm": None, "csf": None}
        alpha_seen = {"wm": 0, "gm": 0, "csf": 0}

        def reservoir_update_generic(res_dict, seen_dict, name, arr_np):
            arr_np = np.asarray(arr_np, dtype=np.float32).ravel()
            if arr_np.size == 0:
                return
            if res_dict[name] is None:
                res_dict[name] = np.empty((0,), dtype=np.float32)

            res = res_dict[name]
            # fill
            if res.size < max_keep:
                need = max_keep - res.size
                take = min(need, arr_np.size)
                if take > 0:
                    res = np.concatenate([res, arr_np[:take]])
                    seen_dict[name] += take
                    arr_np = arr_np[take:]
                res_dict[name] = res
                if arr_np.size == 0:
                    return

            # replace (reservoir sampling)
            res = res_dict[name]
            for v in arr_np:
                seen_dict[name] += 1
                j = np.random.randint(0, seen_dict[name])
                if j < max_keep:
                    res[j] = v
            res_dict[name] = res

        def reservoir_update_global(arr_np):
            nonlocal global_res, global_seen
            arr_np = np.asarray(arr_np, dtype=np.float32).ravel()
            if arr_np.size == 0:
                return
            if global_res is None:
                global_res = np.empty((0,), dtype=np.float32)

            res = global_res
            # fill
            if res.size < max_keep:
                need = max_keep - res.size
                take = min(need, arr_np.size)
                if take > 0:
                    res = np.concatenate([res, arr_np[:take]])
                    global_seen += take
                    arr_np = arr_np[take:]
                global_res = res
                if arr_np.size == 0:
                    return

            # replace
            res = global_res
            for v in arr_np:
                global_seen += 1
                j = np.random.randint(0, global_seen)
                if j < max_keep:
                    res[j] = v
            global_res = res

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Prepass: tissue RMS quantiles", leave=False):
                if not _is_mc_ingredient_batch(batch):
                    continue
                if "tissue_type" not in batch:
                    continue

                mean_ri = batch["mean_ri"].to(self.device, dtype=torch.float32)
                A       = batch["L_or_std"].to(self.device, dtype=torch.float32)
                tissue  = batch["tissue_type"].to(self.device).view(-1)

                if mean_ri.dim() == 3 and mean_ri.shape[1] == 1:
                    mean_ri = mean_ri.squeeze(1)
                if A.dim() >= 3 and A.shape[1] == 1:
                    A = A.squeeze(1)

                valid_mask = batch.get("valid_mask", None)
                if valid_mask is None:
                    valid_mask = torch.ones(mean_ri.shape[0], device=self.device, dtype=torch.bool)
                else:
                    valid_mask = valid_mask.to(self.device).view(-1)

                # u_norm = RMS(A)/|TE1|
                mag0 = _ri_to_mag(mean_ri)
                nf0  = mag0[:, 0:1].clamp_min(1e-9)   # [B,1]
                D    = mean_ri.shape[-1]              # 24
                u_raw  = _rms_from_factor(A, D)       # [B]
                u_norm = u_raw / nf0.squeeze(1)       # [B]
                u_norm = u_raw

                u_norm = u_norm[valid_mask]
                tissue_v = tissue[valid_mask]

                # update global anchors reservoir from ALL valid voxels
                reservoir_update_global(u_norm.detach().cpu().numpy())

                # per tissue u_norm reservoir
                for tid, tname in [(0, "wm"), (1, "gm"), (2, "csf")]:
                    m = (tissue_v == tid)
                    if not m.any():
                        continue
                    reservoir_update_generic(reservoirs, n_seen, tname, u_norm[m].detach().cpu().numpy())

        # ----- compute global anchor values (the ones used to map u_norm -> alpha) -----
        if global_res is None or global_res.size == 0:
            logging.warning("[global quantiles] no voxels found; cannot compute anchors.")
            return

        q_lo_val = float(np.quantile(global_res, q_lo))
        q_hi_val = float(np.quantile(global_res, q_hi))
        logging.info(f"[alpha anchors] global: q_lo={q_lo:.3g} -> {q_lo_val:.6g}, q_hi={q_hi:.3g} -> {q_hi_val:.6g}")

        # store anchors so other code can reuse EXACT same values
        sched.setdefault("quantiles_used", {})
        sched["quantiles_used"]["q_lo"] = q_lo
        sched["quantiles_used"]["q_hi"] = q_hi
        sched["quantiles_used"]["q_lo_val"] = q_lo_val
        sched["quantiles_used"]["q_hi_val"] = q_hi_val
        sched["quantiles_used"]["global_seen"] = int(global_seen)
        sched["quantiles_used"]["global_kept"] = int(global_res.size)

        # ----- now compute alpha distributions PER TISSUE using those anchors -----
        tissue_quantiles = {}

        for tname in ["wm", "gm", "csf"]:
            res = reservoirs[tname]
            if res is None or res.size == 0:
                logging.warning(f"[tissue quantiles] {tname}: no voxels found")
                continue

            q10 = float(np.quantile(res, 0.10))
            q50 = float(np.quantile(res, 0.50))
            q90 = float(np.quantile(res, 0.90))
            q95 = float(np.quantile(res, 0.95))
            q99 = float(np.quantile(res, 0.99))

            logging.info(
                f"[tissue u_norm] {tname}: n_seen={n_seen[tname]} kept={res.size} "
                f"q10={q10:.6g} q50={q50:.6g} q90={q90:.6g} q95={q95:.6g} q99={q99:.6g}"
            )

            tissue_quantiles[tname] = {
                "n_seen": int(n_seen[tname]),
                "kept": int(res.size),
                "q10": q10, "q50": q50, "q90": q90, "q95": q95, "q99": q99,
                f"q{int(q_lo*100)}": float(np.quantile(res, q_lo)),
                f"q{int(q_hi*100)}": float(np.quantile(res, q_hi)),
                "u_lo_val": float(np.quantile(res, q_lo)),
                "u_hi_val": float(np.quantile(res, q_hi)),
            }

            # alpha distribution for this tissue (compute from its u_norm reservoir)
            u_t = torch.from_numpy(res).to(self.device)
            a_t = self._alpha_from_u_norm(u_t, q_lo_val, q_hi_val, alpha_min, alpha_max, gamma)
            a_np = a_t.detach().cpu().numpy()
            alpha_reservoirs[tname] = a_np
            alpha_seen[tname] = int(n_seen[tname])  # same count conceptually

            a10 = float(np.quantile(a_np, 0.10))
            a50 = float(np.quantile(a_np, 0.50))
            a90 = float(np.quantile(a_np, 0.90))
            a95 = float(np.quantile(a_np, 0.95))
            a99 = float(np.quantile(a_np, 0.99))

            logging.info(
                f"[tissue alpha] {tname}: kept={a_np.size} "
                f"q10={a10:.6g} q50={a50:.6g} q90={a90:.6g} q95={a95:.6g} q99={a99:.6g} "
                f"min={a_np.min():.6g} max={a_np.max():.6g}"
            )

         # store diagnostic quantiles
        sched["tissue_quantiles"] = tissue_quantiles

        # --- Step 3: push precomputed u_lo_val/u_hi_val INTO tissue_schedule ---
        # so that build_features_from_batch can read them at runtime
        ts = sched.get("tissue_schedule", None)
        if ts is not None:
            for tname in ["wm", "gm", "csf"]:
                if tname in ts and tname in tissue_quantiles:
                    ts[tname]["u_lo_val"] = tissue_quantiles[tname]["u_lo_val"]
                    ts[tname]["u_hi_val"] = tissue_quantiles[tname]["u_hi_val"]
                    logging.info(
                        f"[tissue_schedule] {tname}: u_lo_val={ts[tname]['u_lo_val']:.6g}, "
                        f"u_hi_val={ts[tname]['u_hi_val']:.6g}"
                    )
            sched["tissue_schedule"] = ts

        self.config["regression_params"]["regression_train_params"]["uncertainty_schedule"] = sched
    def initialize_model(self):
        """Initialize regression model and optimizer."""
        model_params = self.model_params.copy()
        
        # Adjust input channels based on configuration
        if self.uncertainty_mode.endswith("_concat") or self.uncertainty_mode == "concat": 
            model_params['in_ch'] *= 2
        
        if self.add_positional_encoding:
            model_params['in_ch'] += self.pe_dim
        
        if self.use_mc_dropout:
            model_params['dropout_rate'] = self.model_params.get('dropout_rate', 0.2)
            if model_params['dropout_rate'] <= 0.0:
                model_params['dropout_rate'] = 0.2
                logging.warning('MC Dropout enabled but dropout_rate <= 0. Setting to 0.2.')
        
        # Multitask model
        if self.multitask:
            model_params['num_tissue_classes'] = self.num_tissue_classes
            
            if self.heteroscedastic:
                self.model = FCN_Multitask_Heteroscedastic(**model_params).to(self.device)
            else:
                self.model = FCN_Multitask(**model_params).to(self.device)
            
            logging.info(f'Multitask model initialized with {self.num_tissue_classes} tissue classes')
        else:
            # Standard model
            if self.heteroscedastic:
                self.model = HeteroscedasticMLP(**model_params).to(self.device)
            elif self.use_uncertainty_encoder:
                self.model = FCN_with_UncertaintyEncoder(**model_params).to(self.device)
            else:
                self.model = FCN(**model_params).to(self.device)

        logging.info(f'Model initialized:')
        logging.info(f'  Input channels: {model_params["in_ch"]}')
        logging.info(f'  Output channels: {model_params["out_ch"]}')
        logging.info(f'  Uncertainty mode: {self.uncertainty_mode}')
        logging.info(f'  Heteroscedastic: {self.heteroscedastic}')
        logging.info(f'  MC Dropout: {self.use_mc_dropout}')
        logging.info(f'  Use Uncertainty Encoder: {self.use_uncertainty_encoder}')
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_params['lr'],
            weight_decay=self.train_params.get('weight_decay', 0.0)
        )
    
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=4,          # pick 2-4; use 3 for now
            threshold=1e-4,
            min_lr=1e-6,
            verbose=True,        # IMPORTANT: prints when LR changes
        )

    def train(self):
        """Main training loop."""
        output_path = self.setup_output_directory()
        self.setup_datasets()
        # Freeze global RMS quantiles (epoch 0 prepass) to stabilize u_hat mapping
        self._compute_global_quantiles_from_train()
        self._compute_tissue_quantiles_from_train()
        if self.uncertainty_schedule_cfg.get("use_fixed_quantiles", False):
            print(
                f"[u_norm quantiles@start] q50={self.uncertainty_schedule_cfg['fixed_q50']:.6g} "
                f"q90={self.uncertainty_schedule_cfg['fixed_q90']:.6g} (q_lo={self.uncertainty_schedule_cfg.get('q_lo',0.5)}, q_hi={self.uncertainty_schedule_cfg.get('q_hi',0.9)})"
            )
        self.initialize_model()
        self._initialize_loss_function()

        if self.use_adaptive_loss:
            logging.info("Using adaptive loss weighting during training.")
            self.criterion.set_cv_thresholds(self.train_loader.dataset)
        
        # Initialize wandb
        loc = self.regression_params["regression_gt_train_location"]
        m = re.search(r"(Exp\d+)", loc)   # finds Exp6, Exp12, etc.
        if m:
            exp_name = f"{m.group(1)}_"
        else:
            exp_name = "full_dataset_"
        
        # project_name = f"Debug_{exp_name}_{self.acc_rate}t2_{'and_s0' if self.model_params['out_ch'] == 2 else ('T2_only' if self.param_to_predict == 't2star' else 'S0_only')}_regression"
        project_name = f"Final_{exp_name}_{self.acc_rate}_regression"
        # if self.include_physics_loss:
        #     project_name += "_physics"
        # if self.weighted_physics_loss:
        #     project_name += "_weighted"
        # if self.heteroscedastic:
        #     project_name += "_hetero"
        # if self.use_mc_dropout:
        #     project_name += "_mcdrop"
        # if self.multitask:
        #     project_name += "_multitask"
        # if self.use_cv_correlation_loss:
        #     project_name += "_cv_corr"
        # if self.use_rms_correlation_loss:
        #     project_name += "_rms_corr"

        # rank = self.lowrank_rank if self.lowrank_rank else ''
        # Uncertainty naming
        unc_config = f"q_lo:{self.uncertainty_schedule_cfg.get('q_lo', 0.50)}_q_hi:{self.uncertainty_schedule_cfg.get('q_hi', 0.90)}"
        unc_config += f"_beta:{self.uncertainty_schedule_cfg.get('beta', 2.0)}"
        unc_config += f"_wmin:{self.uncertainty_schedule_cfg.get('w_min', 0.15)}"
        unc_config += f"_alpha_min:{self.uncertainty_schedule_cfg.get('alpha_min', 0.05)}"
        unc_config += f"_alpha_max:{self.uncertainty_schedule_cfg.get('alpha_max', 0.50)}"
        unc_config += f"_gamma:{self.uncertainty_schedule_cfg.get('gamma', 1.0)}"
        rank = self.lowrank_rank if self.lowrank_rank else ''
        name = f"{self.uncertainty_mode}_{rank}_{'rms_corr' if self.use_rms_correlation_loss else ''}_{self.acc_rate}_{unc_config}"
        if self.noise_schedule != 'constant':
            name += f'_noise_scale_{self.noise_scale}'
        name += "_adaptive_loss" if self.use_adaptive_loss else ""
        name += "_adaptive_noise" if self.use_adaptive_noise else ""
        name += "_consistency_loss" if self.use_consistency_loss else ""
        name += f"_{int(time.time())}"
        wandb.init(project=project_name, 
                   name=name,
                   config=self.config
                   )
        
        best_score = np.inf
        best_metrics = {}
        early_stopping_counter = 0
        patience = self.train_params.get('early_stopping_patience', 20)

        epoch_times = []
        
        best_tissue_scores  = {"wm": np.inf, "gm": np.inf, "csf": np.inf}
        best_tissue_metrics = {"wm": {},     "gm": {},      "csf": {}}

        for epoch in range(self.train_params['n_epochs']):
            start_time = time.time()
            self.model.train()
            
            # Update dataset epoch (for noise scheduling)
            self.train_loader.dataset.set_epoch(epoch, self.train_params['n_epochs'])
            
            # Get detailed training metrics
            train_loss, train_loss_components, train_corr_metrics, train_aux = self._train_epoch(epoch)
            
            logging.info(f'Epoch {epoch}, Train Loss: {train_loss:.6f}')
            if self.use_cv_correlation_loss:
                logging.info(f'  Hetero Loss: {train_loss_components["hetero_loss"]:.6f}')
                logging.info(f'  CV Corr Loss: {train_loss_components["cv_correlation_loss"]:.6f}')
                if 'train_cv_corr_pearson' in train_corr_metrics:
                    logging.info(f'  CV Correlation (Pearson): {train_corr_metrics["train_cv_corr_pearson"]:.4f}')
            
            epoch_times.append(time.time() - start_time)
            
            # Validation
            self.model.eval()
            val_metrics = self._validate_epoch(epoch)
            monitor_key = "val_nrmse_t2_wm_gm_mean"
            monitor = val_metrics.get(monitor_key, None)
            if monitor is None:
                monitor_key = "val_nrmse_t2_wm_gm_mean"  # fallback
                monitor = val_metrics.get(monitor_key, None)
                if monitor is None:
                    monitor = val_metrics["val_nrmse"]

            self.scheduler.step(float(monitor))
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"[LR] epoch={epoch} monitor({monitor_key})={monitor:.6f} lr={lr:.3e}")

            # Prepare wandb logging dict
            wandb_log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_time': epoch_times[-1],
                **val_metrics
            }

            wandb_log.update(train_aux)
            
            # Add training loss components
            for key, value in train_loss_components.items():
                wandb_log[f'train_{key}'] = value
            
            # Add training correlation metrics
            wandb_log.update(train_corr_metrics)
            
            # Log to wandb
            wandb.log(wandb_log)
            for _t in ["wm", "gm", "csf"]:
                _k = f"val_nrmse_t2_{_t}"
                if _k in val_metrics and val_metrics[_k] is not None:
                    if val_metrics[_k] < best_tissue_scores[_t]:
                        best_tissue_scores[_t] = val_metrics[_k]
                        best_tissue_metrics[_t] = val_metrics.copy()
                        best_tissue_metrics[_t]["epoch"] = epoch
                        logging.info(
                            f'Epoch {epoch}: New best {_t.upper()} NRMSE = {best_tissue_scores[_t]:.6f}'
                        )

            # --- global best (tissue-mean) ---
            if monitor < best_score:
                best_score = monitor
                best_metrics = val_metrics.copy()
                best_metrics['train_loss'] = train_loss
                best_metrics.update(train_corr_metrics)
                self._save_checkpoint(output_path)
                logging.info(f'Epoch {epoch}: New best model (val_nrmse: {best_score:.6f})')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logging.info(f'Early stopping at epoch {epoch}')
                    break
        train_time_avg, train_time_std = np.mean(epoch_times), np.std(epoch_times)
        logging.info(f'Average epoch training time: {train_time_avg:.2f} ± {train_time_std:.2f} seconds')
        best_ckpt_metrics = self._evaluate_best_checkpoint(output_path)
        if best_ckpt_metrics is not None:
            # keep your original "best_metrics" but enrich it with best-checkpoint tissue readout
            best_metrics.update({f"bestckpt/{k}": v for k, v in best_ckpt_metrics.items()})

        self._save_training_summary(output_path, best_metrics, train_time_avg=train_time_avg, train_time_std=train_time_std)
        # log best-per-tissue to wandb summary and save
        for _t in ["wm", "gm", "csf"]:
            if best_tissue_metrics[_t]:
                wandb.run.summary[f"best_nrmse_t2_{_t}"]       = best_tissue_scores[_t]
                wandb.run.summary[f"best_nrmse_t2_{_t}_epoch"] = best_tissue_metrics[_t].get("epoch", -1)
                best_metrics[f"best_tissue/{_t}/nrmse"] = best_tissue_scores[_t]
                best_metrics[f"best_tissue/{_t}/epoch"] = best_tissue_metrics[_t].get("epoch", -1)
                logging.info(
                    f"Best {_t.upper()} NRMSE = {best_tissue_scores[_t]:.6f} "
                    f"(epoch {best_tissue_metrics[_t].get('epoch', -1)})"
                )
        wandb.finish()
        return best_metrics

    def _train_epoch(self, epoch):
        """Train for one epoch with detailed loss tracking."""
        total_loss = 0.0
        loss_components = {
            'hetero_loss': 0.0,
            'cv_correlation_loss': 0.0,
            'consistency_loss': 0.0,
        }
        n_batches = 0
        
        # inside _train_epoch, near the top:
        tissue_buffers = {
            "wm": defaultdict(list),
            "gm": defaultdict(list),
            "csf": defaultdict(list),
        }

        # For correlation computation
        all_pred_stds = []
        all_input_cvs = []
        all_input_rms = []

        
        
        # For logging purposes
        train_aux = {}
        alpha_means, alpha_maxs = [], []
        w_means = []
        uhat_means = []
        delta_means, delta_p95s = [], []
        uhat0_fracs, uhat1_fracs = [], []
        u_means = []


        for batch in tqdm(self.train_loader, desc=f'Training Epoch {epoch}'):
            input_pixels, gt_pixels, s0_pixels = self._prepare_batch(batch)
            if "alpha_sampling" in batch and "u_rms" in batch and "tissue_type" in batch:
                a = batch["alpha_sampling"].to(self.device).view(-1)
                u = batch["u_rms"].to(self.device).view(-1)          # this is u_norm in your code
                eff = (a.view(-1) * u).detach()
                t = batch["tissue_type"].to(self.device).view(-1)

                # for tid, name in [(0,"wm"), (1,"gm"), (2,"csf")]:
                #     m = (t == tid)
                #     if m.any():
                #         train_aux[f"train/{name}/alpha_frac_pos"] = float((a[m] > 0).float().mean().item())
                #         train_aux[f"train/{name}/u_p90"] = float(torch.quantile(u[m], 0.90).item())
                #         train_aux[f"train/{name}/eff_p90"] = float(torch.quantile(eff[m], 0.90).item())
                #         print(f"  {name.upper()}: alpha>0 fraction={train_aux[f'train/{name}/alpha_frac_pos']:.3f}, u_p90={train_aux[f'train/{name}/u_p90']:.6g}, eff_p90={train_aux[f'train/{name}/eff_p90']:.6g}")
            # Only if MC batch (i.e., you have these diagnostics)
            if ("alpha_eff" in batch) and ("eff" in batch) and ("tissue_type" in batch):
                tissue = batch["tissue_type"].to(self.device).view(-1)

                # note: you stored u_rms as [B,1] in _prepare_batch -> squeeze it
                u_rms     = batch["u_rms"].to(self.device).view(-1)
                alpha_eff = batch["alpha_eff"].to(self.device).view(-1)
                eff       = batch["eff"].to(self.device).view(-1)

                valid_mask = batch.get("valid_mask", None)
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device).view(-1).bool()

            #     _append_tissue_buffers(
            #         tissue_buffers,
            #         tissue_ids=tissue,
            #         values={"u_rms": u_rms, "alpha_eff": alpha_eff, "eff": eff},
            #         valid_mask=valid_mask,
            #         max_per_batch=2048,   # tweak: 1024-8192 depending on speed/mem
            #     )

            # if (n_batches % 20 == 0) and ("eff" in batch):
            #     tissue = batch["tissue_type"].to(self.device).view(-1)
            #     u = batch["u_rms"].view(-1)
            #     eff = batch["eff"].view(-1)
            #     aeff = batch["alpha_eff"].view(-1)

            #     # optional
            #     delta = batch.get("delta_per", None)
            #     nf0 = batch.get("nf0", None)

            #     def _tissue_stats(mask):
            #         out = {}
            #         uu = u[mask]
            #         ee = eff[mask]
            #         aa = aeff[mask]
            #         if uu.numel() == 0:
            #             return None

            #         out["u_mean"] = uu.mean().item()
            #         out["u_p90"]  = torch.quantile(uu, 0.90).item()
            #         out["u_p99"]  = torch.quantile(uu, 0.99).item()

            #         out["eff_mean"] = ee.mean().item()
            #         out["eff_p90"]  = torch.quantile(ee, 0.90).item()
            #         out["eff_p99"]  = torch.quantile(ee, 0.99).item()

            #         out["alpha_eff_mean"] = aa.mean().item()
            #         out["alpha_eff_p05"]  = torch.quantile(aa, 0.05).item()

            #         if delta is not None:
            #             dd = delta[mask]
            #             out["delta_mean"] = dd.mean().item()
            #             out["delta_p95"]  = torch.quantile(dd, 0.95).item()

            #         if nf0 is not None:
            #             nn = nf0[mask]
            #             out["nf0_p10"] = torch.quantile(nn, 0.10).item()
            #             out["nf0_p50"] = torch.quantile(nn, 0.50).item()

            #         return out

            #     for name, m in [("wm", tissue==0), ("gm", tissue==1), ("csf", tissue==2)]:
            #         st = _tissue_stats(m)
            #         if st is None:
            #             continue
            #         # push into wandb_log later; easiest is store in a dict you merge
            #         for k,v in st.items():
            #             train_aux[f"train/{name}/{k}"] = v

            # ---- schedule / sampling diagnostics ----
            if "alpha_sampling" in batch:
                # these are [B,1] tensors you already set
                alpha_means.append(batch["alpha_sampling"].mean().item())
                alpha_maxs.append(batch["alpha_sampling"].max().item())

            if "loss_weight" in batch:
                w_means.append(batch["loss_weight"].mean().item())

            if "u_hat" in batch:
                uhat_means.append(batch["u_hat"].mean().item())
                # clamp rates computed directly from u_hat (works even if you didn't store uhat_frac0/1)
                uhat0_fracs.append((batch["u_hat"] <= 1e-6).float().mean().item())
                uhat1_fracs.append((batch["u_hat"] >= 1.0 - 1e-6).float().mean().item())

            if "u_rms" in batch:
                u_means.append(batch["u_rms"].mean().item())
                all_input_rms.append(batch["u_rms"].view(-1))
            # delta diagnostics (these are scalar tensors if you stored them in _prepare_batch)
            if "delta_mean_abs" in batch:
                delta_means.append(batch["delta_mean_abs"].item())

            if "delta_p95_abs" in batch:
                delta_p95s.append(batch["delta_p95_abs"].item())

            # If you stored explicit clamp fractions in aux -> batch:
            if "uhat_frac0" in batch:
                uhat0_fracs.append(batch["uhat_frac0"].item())
            if "uhat_frac1" in batch:
                uhat1_fracs.append(batch["uhat_frac1"].item())
            self.optimizer.zero_grad()
            outputs = self.model(input_pixels) if not isinstance(input_pixels, list) else self.model(*input_pixels)
            if not self.heteroscedastic:
                outputs = outputs.squeeze(-1)
            
            loss, loss_dict = self._compute_loss(outputs, gt_pixels, s0_pixels, batch, epoch)
            
            # Consistency loss
            if self.use_consistency_loss and "input_pixel_t2_2" in batch:
                input_pixels_2 = batch["input_pixel_t2_2"].to(self.device)
                outputs_2 = self.model(input_pixels_2) if not isinstance(input_pixels_2, list) else self.model(*input_pixels_2)
                if not self.heteroscedastic:
                    outputs_2 = outputs_2.squeeze(-1)
                    consistency_loss = self._compute_consistency_loss(outputs, outputs_2)
                else:
                    mean, log_var = outputs
                    mean_2, log_var_2 = outputs_2
                    consistency_loss = self._compute_consistency_loss(mean, mean_2)
                    consistency_loss += self._compute_consistency_loss(log_var, log_var_2)
                    consistency_loss /= 2.0
                loss += self.consistency_weight * consistency_loss
                loss_dict['consistency_loss'] = consistency_loss.item()
            
            loss.backward()
            
            # Gradient clipping
            if self.train_params.get('grad_max_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_params['grad_max_norm'],
                    norm_type=2
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_components.keys():
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
            n_batches += 1
            
            # Collect data for correlation computation (every 10 batches)
            if self.use_cv_correlation_loss and self.heteroscedastic and n_batches % 10 == 0:
                with torch.no_grad():
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        mu, log_sigma_sq = outputs
                        pred_std = torch.exp(0.5 * log_sigma_sq)
                        input_cv = batch['cv'].to(self.device)
                        
                        # Shape matching and flatten
                        if pred_std.dim() > input_cv.dim():
                            pred_std = pred_std.squeeze(-1)
                        elif input_cv.dim() > pred_std.dim():
                            input_cv = input_cv.squeeze(-1)
                        
                        # Flatten immediately
                        all_pred_stds.append(pred_std.flatten().cpu())
                        all_input_cvs.append(input_cv.flatten().cpu())
                    elif self.use_rms_correlation_loss and self.heteroscedastic and n_batches % 10 == 0:
                        with torch.no_grad():
                            if isinstance(outputs, tuple) and len(outputs) == 2:
                                _, pred_log_var = outputs
                                input_rms = batch['rms'].squeeze(-1).cpu()  # CPU immediately!
                                
                                pred_log_var_cpu = pred_log_var.squeeze(-1).cpu()
                                
                                all_pred_stds.append(pred_log_var_cpu.flatten())
                                all_input_rms.append(input_rms.flatten())
        
        # Compute average losses
        avg_total_loss = total_loss / n_batches
        for key in loss_components:
            loss_components[key] /= n_batches

        # Compute correlation if we have CV correlation loss
        correlation_metrics = {}
        if self.use_cv_correlation_loss and len(all_pred_stds) > 0:
            all_pred_stds = torch.cat(all_pred_stds).numpy()
            all_input_cvs = torch.cat(all_input_cvs).numpy()
            
            from scipy.stats import pearsonr, spearmanr
            
            r_pearson, p_pearson = pearsonr(all_input_cvs, all_pred_stds)
            r_spearman, p_spearman = spearmanr(all_input_cvs, all_pred_stds)
            
            correlation_metrics = {
                'train_cv_corr_pearson': r_pearson,
                'train_cv_corr_spearman': r_spearman,
                'train_pred_std_mean': all_pred_stds.mean(),
                'train_pred_std_std': all_pred_stds.std(),
                'train_input_cv_mean': all_input_cvs.mean(),
            }
        if self.use_rms_correlation_loss and len(all_pred_stds) > 0:
            all_pred_stds = torch.cat(all_pred_stds).numpy()
            all_input_rms = torch.cat(all_input_rms).numpy()

            from scipy.stats import pearsonr, spearmanr
            r_pearson, _ = pearsonr(all_input_rms, all_pred_stds)
            r_spearman, _ = spearmanr(all_input_rms, all_pred_stds)

            correlation_metrics = {
                "train_rms_corr_pearson": r_pearson,
                "train_rms_corr_spearman": r_spearman,
                "train_pred_std_mean": all_pred_stds.mean(),
                "train_pred_std_std": all_pred_stds.std(),
                "train_input_rms_mean": all_input_rms.mean(),
            }
        


        def _mean(xs):
            return float(np.mean(xs)) if len(xs) else None

        # only add keys that exist (avoids cluttering wandb with NaNs)
        if len(alpha_means): train_aux["train/alpha_mean"] = _mean(alpha_means)
        if len(alpha_maxs):  train_aux["train/alpha_max"]  = _mean(alpha_maxs)
        if len(w_means):     train_aux["train/w_mean"]     = _mean(w_means)
        if len(uhat_means):  train_aux["train/u_hat_mean"] = _mean(uhat_means)
        if len(u_means):     train_aux["train/u_rms_mean"] = _mean(u_means)

        if len(uhat0_fracs): train_aux["train/u_hat_frac0"] = _mean(uhat0_fracs)
        if len(uhat1_fracs): train_aux["train/u_hat_frac1"] = _mean(uhat1_fracs)

        if len(delta_means): train_aux["train/delta_mean_abs"] = _mean(delta_means)
        if len(delta_p95s):  train_aux["train/delta_p95_abs"]  = _mean(delta_p95s)

        tissue_stats = _finalize_tissue_buffers(tissue_buffers, prefix="train/tissue")
        train_aux.update(tissue_stats)

        return avg_total_loss, loss_components, correlation_metrics, train_aux

    def _validate_epoch(self, epoch):
        metrics = {
            "total_loss": 0.0,
            "t2_loss": 0.0,
            "s0_loss": 0.0,
            "mae_t2": 0.0,
            "mae_s0": 0.0,
        }
        n = 0
        sse_t2 = 0.0
        sse_s0 = 0.0
        den_t2 = 0.0
        den_s0 = 0.0
        out_ch = self.model_params["out_ch"]

        # Tissue accumulators for T2 (val)
        sse_t2_tissue = {"wm": 0.0, "gm": 0.0, "csf": 0.0}
        den_t2_tissue = {"wm": 0.0, "gm": 0.0, "csf": 0.0}
        mae_t2_tissue = {"wm": 0.0, "gm": 0.0, "csf": 0.0}
        n_tissue      = {"wm": 0,   "gm": 0,   "csf": 0}


        with torch.no_grad():
            for batch in self.val_loader:
                input_pixels, gt_pixels, s0_pixels = self._prepare_batch(batch, deterministic=True)

                outputs = self.model(input_pixels) if not isinstance(input_pixels, list) else self.model(*input_pixels)
                if not self.heteroscedastic:
                    outputs = outputs.squeeze(-1)

                loss, loss_dict = self._compute_loss(outputs, gt_pixels, s0_pixels, batch, epoch)

                # Extract predictions
                if self.multitask:
                    pred_t2, pred_s0, _ = self._extract_predictions(outputs, out_ch)
                else:
                    pred_t2, pred_s0 = self._extract_predictions(outputs, out_ch)

                tissue = batch.get("tissue_type", None)
                if tissue is not None:
                    tissue = tissue.to(self.device).view(-1)  # [B]

                    # Make sure pred/gt are [B]
                    pred_v = pred_t2.view(-1)
                    gt_v   = gt_pixels.view(-1)

                    # masks
                    wm_m  = (tissue == 0)
                    gm_m  = (tissue == 1)
                    csf_m = (tissue == 2)

                    for name, msk in [("wm", wm_m), ("gm", gm_m), ("csf", csf_m)]:
                        if msk.any():
                            pv = pred_v[msk]
                            gv = gt_v[msk]
                            sse_t2_tissue[name] += torch.sum((pv - gv) ** 2).item()
                            den_t2_tissue[name] += torch.sum((gv) ** 2).item()
                            mae_t2_tissue[name] += torch.sum(torch.abs(pv - gv)).item()
                            n_tissue[name]      += int(msk.sum().item())


                bs = gt_pixels.shape[0]
                n += bs

                sse_t2 += torch.sum((pred_t2 - gt_pixels) ** 2).item()
                den_t2 += torch.sum(gt_pixels ** 2).item()

                # if out_ch == 2 and pred_s0 is not None:
                #     sse_s0 += torch.sum((pred_s0 - s0_pixels) ** 2).item()
                #     den_s0 += torch.sum(s0_pixels ** 2).item()

                metrics["total_loss"] += loss.item() * bs
                for k, v in loss_dict.items():
                    if k in metrics:
                        metrics[k] += v * bs

                metrics["mae_t2"] += torch.sum(torch.abs(pred_t2 - gt_pixels)).item()
                # if out_ch == 2 and pred_s0 is not None:
                #     metrics["mae_s0"] += torch.sum(torch.abs(pred_s0 - s0_pixels)).item()

        # Finalize averages
        metrics["total_loss"] /= max(n, 1)
        if "t2_loss" in metrics: 
            metrics["t2_loss"] /= max(n, 1)
        if "s0_loss" in metrics: 
            metrics["s0_loss"] /= max(n, 1)

        metrics["mae_t2"] = metrics["mae_t2"] / max(n, 1)
        if out_ch == 2:
            metrics["mae_s0"] = metrics["mae_s0"] / max(n, 1)

        eps = 1e-8
        metrics["val_nmse_t2"] = sse_t2 / (den_t2 + eps)
        metrics["val_nrmse_t2"] = (metrics["val_nmse_t2"]) ** 0.5

        if out_ch == 2:
            metrics["val_nmse_s0"] = sse_s0 / (den_s0 + eps)
            metrics["val_nrmse_s0"] = (metrics["val_nmse_s0"]) ** 0.5

        metrics["val_nrmse"] = metrics["val_nrmse_t2"] if out_ch == 1 else 0.5 * (
            metrics["val_nrmse_t2"] + metrics["val_nrmse_s0"]
        )
        eps = 1e-8
        for name in ["wm", "gm", "csf"]:
            if n_tissue[name] > 0 and den_t2_tissue[name] > 0:
                nmse = sse_t2_tissue[name] / (den_t2_tissue[name] + eps)
                metrics[f"val_nmse_t2_{name}"] = nmse
                metrics[f"val_nrmse_t2_{name}"] = float(np.sqrt(nmse))
                metrics[f"val_mae_t2_{name}"] = mae_t2_tissue[name] / (n_tissue[name] + eps)
                metrics[f"val_n_{name}"] = n_tissue[name]

        # Balanced (equal-weight) tissue mean for early stopping / model selection
        if all(f"val_nrmse_t2_{t}" in metrics for t in ["wm", "gm", "csf"]):
            metrics["val_nrmse_t2_tissue_mean"] = (
                metrics["val_nrmse_t2_wm"] + metrics["val_nrmse_t2_gm"] + metrics["val_nrmse_t2_csf"]
            ) / 3.0
        else:
            metrics["val_nrmse_t2_tissue_mean"] = None


        # Balanced (equal-weight) tissue mean for early stopping / model selection
        if all(f"val_nrmse_t2_{t}" in metrics for t in ["wm", "gm"]):
            metrics["val_nrmse_t2_wm_gm_mean"] = (
                metrics["val_nrmse_t2_wm"] + metrics["val_nrmse_t2_gm"] 
            ) / 2.0
        else:
            metrics["val_nrmse_t2_wm_gm_mean"] = None


        if all(f"val_nrmse_t2_{t}" in metrics for t in ["wm", "gm", "csf"]):
            metrics["val_nrmse_t2_weighted"] = (
                0.4 * metrics["val_nrmse_t2_wm"]
            + 0.4 * metrics["val_nrmse_t2_gm"]
            + 0.2 * metrics["val_nrmse_t2_csf"]
            )

        return metrics

    def _prepare_batch(self, batch, deterministic=False):
        """
        Prepare batch data - supports MC-ingredient batches (GPU sampling) + normal batches.
        
        Returns:
            input_pixels: Features ready for model
            gt_pixels: Ground truth T2*
            s0_pixels: Ground truth S0
        """
        gt_pixels = batch['gt_pixel'].to(self.device)
        # s0_pixels = batch['s0_pixel'].to(self.device)

        input_rms = batch.get('rms', batch.get('u_rms', None))
        if input_rms is not None:
            batch['u_rms'] = input_rms.to(self.device)

        # MC-ingredient batch (GPU sampling)
        if _is_mc_ingredient_batch(batch):
            # Build sampled + normalized features on GPU
            feat_t2, feat_t2_2, aux = build_features_from_batch(batch, self.device, self.uncertainty_schedule_cfg, deterministic=deterministic)
            
            for k in ["delta_mean_abs", "delta_p95_abs", "u_mean", "u_hat_mean", "alpha_mean", "alpha_max", "w_mean", "uhat_frac0", "uhat_frac1"]:
                if k in aux:
                    batch[k] = torch.tensor(aux[k], device=self.device, dtype=torch.float32)
            # store weights/alpha for weighted loss + logging
            batch["loss_weight"] = aux["loss_weight"].to(self.device).view(-1, 1)
            batch["alpha_sampling"] = aux["alpha"].to(self.device).view(-1, 1)
            batch["u_hat"] = aux["u_hat"].to(self.device).view(-1, 1)
            batch["u_rms"] = aux["u"].to(self.device).view(-1, 1)
            batch["rms"] = batch["u_rms"]
            feat_s0 = feat_t2  # Same features for S0
            feat_s0_2 = feat_t2_2

            # per-sample debug tensors (optional)
            if "delta_per" in aux:
                batch["delta_per"] = aux["delta_per"].to(self.device)
            if "nf0" in aux:
                batch["nf0"] = aux["nf0"].to(self.device)
            if "alpha_eff" in aux:
                batch["alpha_eff"] = aux["alpha_eff"].to(self.device)
            if "eff" in aux:
                batch["eff"] = aux["eff"].to(self.device)


            # Select which input to feed
            if self.param_to_predict == "s0":
                input_pixels = feat_s0
            else:
                input_pixels = feat_t2

            # Dual-output case: concatenate t2 and s0 inputs
            if self.model_params["out_ch"] == 2:
                input_pixels = torch.cat([feat_t2, feat_s0], dim=-1).float()

            # Positional encoding
            if self.add_positional_encoding:
                pos_enc = batch['positional_encoding'].to(self.device)
                if pos_enc.dim() == 3:
                    pos_enc = pos_enc.squeeze(1)
                input_pixels = torch.cat([input_pixels, pos_enc], dim=-1).float()

            # Uncertainty encoder
            if self.use_uncertainty_encoder:
                L = batch["L_or_std"].to(self.device)
                if L.dim() == 4 and L.shape[1] == 1:
                    L = L.squeeze(1)
                input_pixels = [input_pixels.float(), L.float()]

            # Store second sample for consistency loss
            if self.use_consistency_loss and feat_t2_2 is not None:
                batch["input_pixel_t2_2"] = feat_t2_2

            return input_pixels, gt_pixels, None

        # Normal batch (pre-computed features)
        if self.param_to_predict == 's0':
            input_pixels = batch['input_pixel_s0'].to(self.device)
        else:
            input_pixels = batch['input_pixel_t2'].to(self.device)

        if self.model_params['out_ch'] == 2:
            input_pixels_t2 = batch['input_pixel_t2'].to(self.device)
            input_pixels_s0 = batch['input_pixel_s0'].to(self.device)
            input_pixels = torch.cat((input_pixels_t2, input_pixels_s0), dim=-1).float()

        if self.add_positional_encoding:
            pos_enc = batch['positional_encoding'].to(self.device)
            input_pixels = torch.cat((input_pixels, pos_enc), dim=-1).float()

        if self.use_uncertainty_encoder:
            L_matrix = batch['L_matrix'].to(self.device)
            input_pixels = [input_pixels.float(), L_matrix.float()]
        

        return input_pixels, gt_pixels, None
    
    def _extract_predictions(self, outputs, out_ch):
        """Extract mean predictions from model outputs."""
        # Multitask outputs
        if self.multitask:
            if self.heteroscedastic:
                if out_ch == 2:
                    pred_t2 = outputs[0][0]
                    pred_s0 = outputs[1][0]
                    tissue_logits = outputs[2]
                else:
                    pred_t2 = outputs[0][0]
                    pred_s0 = None
                    tissue_logits = outputs[1]
            else:
                if out_ch == 2:
                    pred_t2 = outputs[0]
                    pred_s0 = outputs[1]
                    tissue_logits = outputs[2]
                else:
                    pred_t2 = outputs[0]
                    pred_s0 = None
                    tissue_logits = outputs[1]
            
            return pred_t2, pred_s0, tissue_logits
        
        # Standard outputs
        if self.heteroscedastic:
            if out_ch == 2:
                pred_t2 = outputs[0][0]
                pred_s0 = outputs[1][0]
            else:
                pred_t2, log_var_t2 = outputs[0].squeeze(-1), outputs[1].squeeze(-1)
                pred_s0 = None
        else:
            if out_ch == 2:
                pred_t2, pred_s0 = outputs
            else:
                pred_t2 = outputs
                pred_s0 = None
        
        return pred_t2, pred_s0

    def _compute_consistency_loss(self, outputs_1, outputs_2):
        """Compute consistency loss between predictions from two samples."""
        if self.heteroscedastic:
            if isinstance(outputs_1, tuple):
                mu_1, _ = outputs_1
                mu_2, _ = outputs_2
            else:
                mu_1 = outputs_1[0] if isinstance(outputs_1, list) else outputs_1
                mu_2 = outputs_2[0] if isinstance(outputs_2, list) else outputs_2
        else:
            mu_1 = outputs_1
            mu_2 = outputs_2
        
        return nn.functional.mse_loss(mu_1, mu_2)

    def _compute_cv_correlation_loss(self, pred_log_var, batch, eps=1e-8):
        """
        Compute loss that encourages predicted uncertainty to correlate with input CV.
        Uses log-space correlation for better numerical stability.
        """
        input_cv = batch["cv"].to(self.device)
        log_sigma = 0.5 * pred_log_var
        log_cv = torch.log(input_cv + eps)

        # Match shapes
        if log_sigma.dim() > log_cv.dim():
            log_sigma = log_sigma.squeeze(-1)
        elif log_cv.dim() > log_sigma.dim():
            log_cv = log_cv.squeeze(-1)

        # Pearson correlation (needs centering)
        x = log_sigma - log_sigma.mean()
        y = log_cv - log_cv.mean()

        x_std = x.std(unbiased=False)
        y_std = y.std(unbiased=False)

        corr = (x * y).mean() / (x_std * y_std + eps)
        corr = corr.clamp(-1.0, 1.0)

        return 1.0 - corr

    def _compute_loss(self, outputs, gt_pixels, s0_pixels, batch, epoch, alpha=0.5):
        """Compute loss. Supports per-sample loss weights via batch["loss_weight"] (NEW)."""
        out_ch = self.model_params['out_ch']
        loss_dict = {}

        loss_weight = batch.get("loss_weight", None)
        if loss_weight is not None:
            loss_weight = loss_weight.to(self.device)
            if loss_weight.dim() > 1:
                loss_weight = loss_weight.view(-1)

        if not self.heteroscedastic:
            # Ensure both have same shape
            if outputs.dim() == 1 and gt_pixels.dim() == 2:
                gt_pixels = gt_pixels.squeeze(-1)
            elif outputs.dim() == 2 and gt_pixels.dim() == 1:
                outputs = outputs.squeeze(-1)

        # Multitask learning path
        if self.multitask:
            multitask_loss_fn = MultitaskLoss(
                alpha_regression=alpha,
                alpha_s0=(1-alpha),
                alpha_tissue=self.alpha_tissue,
                class_weights=self.tissue_class_weights
            )
            
            tissue_labels = batch['tissue_type'].to(self.device)
            
            total_loss, loss_dict = multitask_loss_fn(
                outputs, gt_pixels, s0_pixels, tissue_labels, 
                heteroscedastic=self.heteroscedastic
            )
            
            return total_loss, loss_dict
        
        # Heteroscedastic loss
        if self.heteroscedastic:
            if out_ch == 2:
                t2_output, s0_output = outputs[0], outputs[1]
                t2_preds, t2_log_sigma_sq = t2_output
                s0_preds, s0_log_sigma_sq = s0_output
                
                if t2_output.dim() == 1 and gt_pixels.dim() == 2:
                    gt_pixels = gt_pixels.squeeze(-1)
                elif t2_output.dim() == 2 and gt_pixels.dim() == 1:
                    t2_output = t2_output.squeeze(-1)
                    s0_output = s0_output.squeeze(-1)
                    t2_log_sigma_sq = t2_log_sigma_sq.squeeze(-1)
                    s0_log_sigma_sq = s0_log_sigma_sq.squeeze(-1)

                t2_loss = self.criterion(t2_preds, t2_log_sigma_sq, gt_pixels)
                s0_loss = self.criterion(s0_preds, s0_log_sigma_sq, s0_pixels)
                
                loss_dict['t2_loss'] = t2_loss.item()
                loss_dict['s0_loss'] = s0_loss.item()
                
                total_loss = alpha * t2_loss + (1 - alpha) * s0_loss
                
                return total_loss, loss_dict
            else:
                mu, log_sigma_sq = outputs
                true_pixels = gt_pixels if self.param_to_predict == 't2star' else s0_pixels
                
                if mu.dim() == 2 and true_pixels.dim() == 3:
                    true_pixels = true_pixels.squeeze(-1)
                elif mu.dim() == 3 and true_pixels.dim() == 2:
                    mu = mu.squeeze(-1)
                    log_sigma_sq = log_sigma_sq.squeeze(-1)
                
                hetero_loss = self.criterion(mu, log_sigma_sq, true_pixels)

                if self.use_cv_correlation_loss:
                    cv_loss = self._compute_cv_correlation_loss(log_sigma_sq, batch)
                    loss = hetero_loss + self.cv_correlation_weight * cv_loss
                    loss_dict['hetero_loss'] = hetero_loss.item()
                    loss_dict['cv_correlation_loss'] = cv_loss.item()
                elif self.use_rms_correlation_loss:
                    # Implement RMS correlation loss if needed (not shown here for brevity)
                    rms_loss = self._compute_rms_correlation_loss(log_sigma_sq, batch)
                    loss = hetero_loss + self.rms_correlation_weight * rms_loss
                    loss_dict['hetero_loss'] = hetero_loss.item()
                    loss_dict['rms_correlation_loss'] = rms_loss.item()
                else:
                    loss = hetero_loss

                loss_dict['loss'] = loss.item()
                return loss, loss_dict
        
        # Standard loss (non-heteroscedastic)
        if out_ch == 2:
            # NEW: per-sample weighted loss (from RMS schedule)
            if loss_weight is not None:
                t2_preds, s0_preds = outputs
                # flatten to [B]
                t2_preds_ = t2_preds.view(t2_preds.shape[0], -1).mean(dim=1)
                gt_ = gt_pixels.view(gt_pixels.shape[0], -1).mean(dim=1)
                s0_preds_ = s0_preds.view(s0_preds.shape[0], -1).mean(dim=1)
                s0_ = s0_pixels.view(s0_pixels.shape[0], -1).mean(dim=1)

                if self.loss.upper() == 'L1':
                    per_t2 = (t2_preds_ - gt_).abs()
                    per_s0 = (s0_preds_ - s0_).abs()
                elif self.loss.upper() == 'HUBER':
                    per_t2 = torch.nn.functional.smooth_l1_loss(t2_preds_, gt_, reduction='none')
                    per_s0 = torch.nn.functional.smooth_l1_loss(s0_preds_, s0_, reduction='none')
                else:  # default L2
                    per_t2 = (t2_preds_ - gt_).pow(2)
                    per_s0 = (s0_preds_ - s0_).pow(2)

                t2_loss = (per_t2 * loss_weight).sum() / (loss_weight.sum() + 1e-12)
                s0_loss = (per_s0 * loss_weight).sum() / (loss_weight.sum() + 1e-12)

                loss_dict['t2_loss'] = float(t2_loss.detach().cpu())
                loss_dict['s0_loss'] = float(s0_loss.detach().cpu())

                total_loss = alpha * t2_loss + (1 - alpha) * s0_loss
                return total_loss, loss_dict

            t2_preds, s0_preds = outputs
            
            if self.criterion is not None and self.use_adaptive_loss:
                t2_loss = self.criterion(t2_preds, gt_pixels, batch)
                s0_loss = self.criterion(s0_preds, s0_pixels, batch)
            elif self.loss.upper() in ['L1', 'L2', 'HUBER']:
                t2_loss = self.criterion(t2_preds, gt_pixels)
                s0_loss = self.criterion(s0_preds, s0_pixels)
            else:
                if self.loss.upper() == 'L1':
                    t2_loss = nn.functional.l1_loss(t2_preds, gt_pixels)
                    s0_loss = nn.functional.l1_loss(s0_preds, s0_pixels)
                elif self.loss.upper() == 'L2':
                    t2_loss = nn.functional.mse_loss(t2_preds, gt_pixels)
                    s0_loss = nn.functional.mse_loss(s0_preds, s0_pixels)
                elif self.loss.upper() == 'HUBER':
                    t2_loss = nn.functional.smooth_l1_loss(t2_preds, gt_pixels)
                    s0_loss = nn.functional.smooth_l1_loss(s0_preds, s0_pixels)
                else:
                    raise ValueError(f'Unsupported loss type: {self.loss}')
            
            loss_dict['t2_loss'] = t2_loss.item()
            loss_dict['s0_loss'] = s0_loss.item()
            
            total_loss = alpha * t2_loss + (1 - alpha) * s0_loss
            
            return total_loss, loss_dict
        else:
            # Single output
            # NEW: per-sample weighted loss (from RMS schedule)
            if loss_weight is not None:
                true_pixels = gt_pixels if self.param_to_predict == 't2star' else s0_pixels
                pred_ = outputs.view(outputs.shape[0], -1).mean(dim=1)
                true_ = true_pixels.view(true_pixels.shape[0], -1).mean(dim=1)

                if self.loss.upper() == 'L1':
                    per = (pred_ - true_).abs()
                elif self.loss.upper() == 'HUBER':
                    per = torch.nn.functional.smooth_l1_loss(pred_, true_, reduction='none')
                else:
                    per = (pred_ - true_).pow(2)

                loss = (per * loss_weight).sum() / (loss_weight.sum() + 1e-12)
                loss_dict['loss'] = float(loss.detach().cpu())
                return loss, loss_dict

            true_pixels = gt_pixels if self.param_to_predict == 't2star' else s0_pixels
            
            if self.criterion and self.use_adaptive_loss:
                loss = self.criterion(outputs, true_pixels, batch)
            elif self.loss.upper() in ['L1', 'L2', 'HUBER']:
                loss = self.criterion(outputs, true_pixels)
            else:
                if self.loss.upper() == 'L1':
                    loss = nn.functional.l1_loss(outputs, true_pixels)
                elif self.loss.upper() == 'L2':
                    loss = nn.functional.mse_loss(outputs, true_pixels)
                elif self.loss.upper() == 'HUBER':
                    loss = nn.functional.smooth_l1_loss(outputs, true_pixels)
                else:
                    raise ValueError(f'Unsupported loss type: {self.loss}')
            
            loss_dict['loss'] = loss.item()
            return loss, loss_dict
    
    def _save_checkpoint(self, output_path):
        """Save model checkpoint."""
        torch.save(
            self.model.state_dict(),
            os.path.join(output_path, 'best_regression_model.pth')
        )
    
    def _save_training_summary(self, output_path, metrics, train_time_avg=None, train_time_std=None):
        """Save training summary."""
        summary = f"""Training Summary
        ================
        Uncertainty Mode: {self.uncertainty_mode}
        Best Validation NRMSE: {metrics.get('val_nrmse', 0):.6f}
        Best Validation NRMSE (T2): {metrics.get('val_nrmse_t2', 0):.6f}
        Training Loss: {metrics.get('train_loss', 0):.6f}
        Validation Loss (objective): {metrics.get('total_loss', 0):.6f}
        """
        if 'val_nrmse_s0' in metrics:
            summary += f"Best Validation NRMSE (S0): {metrics.get('val_nrmse_s0', 0):.6f}\n"
        
        if 'bestckpt/val_nrmse_t2_wm' in metrics:
            summary += f"Best Checkpoint T2* (WM): {metrics.get('bestckpt/val_nrmse_t2_wm', 0):.6f}\n"
        if 'bestckpt/val_nrmse_t2_gm' in metrics:
            summary += f"Best Checkpoint T2* (GM): {metrics.get('bestckpt/val_nrmse_t2_gm', 0):.6f}\n"
        if 'bestckpt/val_nrmse_t2_csf' in metrics:
            summary += f"Best Checkpoint T2* (CSF): {metrics.get('bestckpt/val_nrmse_t2_csf', 0):.6f}\n"

        if train_time_avg is not None and train_time_std is not None:
            summary += f"Average Epoch Training Time: {train_time_avg:.2f} ± {train_time_std:.2f} seconds\n"

        # Print the uncertainty schedule configuration
        if self.uncertainty_schedule_cfg is not None and str(self.uncertainty_mode).lower() not in [None, "none", "none_concat"]:
            summary += "Uncertainty Schedule Config:\n"
            summary += yaml.safe_dump(self.uncertainty_schedule_cfg, sort_keys=False)
            summary += "\n"

        with open(os.path.join(output_path, 'training_summary.txt'), 'w') as f:
            f.write(summary)
        
        logging.info(summary)