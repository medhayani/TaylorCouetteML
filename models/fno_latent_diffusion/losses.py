"""All NEPTUNE losses.

L_surr = L_diff + lambda_s L_switch + lambda_p L_PINO + lambda_sigma L_spec
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def diffusion_loss(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(eps_pred, eps_true)


def switch_loss(
    prob_logits: torch.Tensor,
    switch_label: torch.Tensor,
    center_pred: torch.Tensor,
    center_true: torch.Tensor,
    width_pred: Optional[torch.Tensor] = None,
    width_true: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(prob_logits, switch_label)
    huber_c = F.huber_loss(center_pred, center_true, delta=huber_delta)
    if width_pred is None or width_true is None:
        return bce + huber_c
    huber_w = F.huber_loss(width_pred, width_true, delta=huber_delta)
    return bce + huber_c + huber_w


def pino_loss(
    ta_pred: torch.Tensor,
    s: torch.Tensor,
    kappa_pred: torch.Tensor,
    near_min_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize | d^2 Ta / ds^2 - kappa | near the minimum."""
    # second derivative via central differences on the resampled grid
    if ta_pred.dim() != 2:
        raise ValueError("ta_pred expected shape (B, T)")
    d1 = (ta_pred[:, 2:] - ta_pred[:, :-2]) / 2.0
    # need a uniform spacing assumption; s is normalized in [0,1]
    d2 = ta_pred[:, 2:] - 2.0 * ta_pred[:, 1:-1] + ta_pred[:, :-2]
    target = kappa_pred.unsqueeze(-1).expand_as(d2)
    err2 = (d2 - target) ** 2
    if near_min_mask is not None:
        m = near_min_mask[:, 1:-1]
        err2 = err2 * m
        denom = m.sum().clamp(min=1.0)
        return err2.sum() / denom
    return err2.mean()


def spectral_loss(ta_pred: torch.Tensor, k_max: int = 16) -> torch.Tensor:
    """Penalize spectral energy beyond k_max modes (anti-aliasing)."""
    Xf = torch.fft.rfft(ta_pred, dim=-1)             # (B, T//2+1)
    if k_max >= Xf.size(-1):
        return ta_pred.new_zeros(())
    high = Xf[..., k_max:]
    return (high.abs() ** 2).mean()
