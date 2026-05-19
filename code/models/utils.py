"""Kink detection and kink-weighted MSE loss for marginal-stability curves.

Mode-crossings in the Floquet eigenvalue problem produce cusps (angular
points) in Ta(k). Standard MSE underweights these regions because they
occupy a small fraction of points. Up-weighting gives the network the
right gradient signal to capture them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def kink_weights(ta_norm: torch.Tensor, alpha: float = 5.0,
                 threshold_z: float = 2.0) -> torch.Tensor:
    """Per-point loss weights from second-difference magnitude.

    Args:
        ta_norm:     (B, T) normalised curve in [0, 1].
        alpha:       peak multiplier at strong kinks (default 5).
        threshold_z: z-score (in standard deviations) above which a point
                      is considered "kink-like" (default 2.0).
    Returns:
        w: (B, T) weights, mean ~ 1, peaks ~ alpha at detected kinks.
    """
    # central second difference
    d2_inner = ta_norm[:, 2:] - 2 * ta_norm[:, 1:-1] + ta_norm[:, :-2]
    d2 = torch.cat([d2_inner[:, :1], d2_inner, d2_inner[:, -1:]], dim=-1)
    std = d2.abs().std(dim=-1, keepdim=True) + 1e-6
    z = d2.abs() / std
    return 1.0 + (alpha - 1.0) * torch.sigmoid(2.0 * (z - threshold_z))


def kink_weighted_mse(pred: torch.Tensor, true: torch.Tensor,
                      alpha: float = 5.0, threshold_z: float = 2.0) -> torch.Tensor:
    """MSE with per-point kink weights derived from the *true* curve."""
    w = kink_weights(true.detach(), alpha=alpha, threshold_z=threshold_z)
    return ((pred - true) ** 2 * w).mean()


def detect_kinks_np(ta_norm: np.ndarray, threshold_z: float = 2.5) -> np.ndarray:
    """Numpy variant: indices where |d2 Ta| exceeds threshold_z standard deviations."""
    ta_norm = np.asarray(ta_norm, dtype=np.float64)
    d2 = np.zeros_like(ta_norm)
    d2[1:-1] = ta_norm[2:] - 2 * ta_norm[1:-1] + ta_norm[:-2]
    std = float(np.std(d2)) + 1e-9
    z = np.abs(d2) / std
    return np.where(z > threshold_z)[0]


def selective_smooth(pred: np.ndarray, true_norm: np.ndarray,
                     window: int = 7, polyorder: int = 2,
                     edge: int = 3, threshold_z: float = 2.5,
                     transition: int = 2) -> np.ndarray:
    """Smooth a prediction everywhere EXCEPT in kink zones of the true curve.

    Pipeline:
        1. Detect kinks in `true_norm` (indices where |d2 Ta| > threshold_z * std).
        2. Apply Savitzky-Golay smoothing to the entire `pred`.
        3. Build a binary mask = 1 in kink zones (+/- `edge` grid points),
           0 elsewhere. Soften it with a small box filter (`transition` half-width)
           so the blend has no visible seam.
        4. Return  mask * pred_raw + (1 - mask) * pred_smooth.

    Args:
        pred:          (T,) prediction in any units (physical or normalised).
        true_norm:     (T,) reference curve used for kink detection (normalised).
        window:        Savitzky-Golay window length (odd).
        polyorder:     Savitzky-Golay polynomial order.
        edge:          half-width of the "preserve" zone around each kink.
        threshold_z:   z-score threshold passed to detect_kinks_np.
        transition:    half-width of the box filter used to soften mask edges.
    Returns:
        out: (T,) selectively smoothed prediction with cusps preserved.
    """
    from scipy.signal import savgol_filter

    pred = np.asarray(pred, dtype=float)
    T = len(pred)
    if T < window:
        return pred

    pred_smooth = savgol_filter(pred, window_length=window, polyorder=polyorder,
                                  mode="nearest")
    kinks = detect_kinks_np(true_norm, threshold_z=threshold_z)
    if len(kinks) == 0:
        return pred_smooth

    mask = np.zeros(T, dtype=float)
    for ki in kinks:
        lo, hi = max(0, ki - edge), min(T, ki + edge + 1)
        mask[lo:hi] = 1.0
    if transition > 0:
        kernel = np.ones(2 * transition + 1) / (2 * transition + 1)
        mask = np.convolve(mask, kernel, mode="same")
        mask = np.clip(mask, 0.0, 1.0)

    return mask * pred + (1.0 - mask) * pred_smooth
