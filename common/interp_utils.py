"""1D interpolation + finite difference helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def unique_sorted_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    xu, idx = np.unique(np.round(x, 10), return_index=True)
    yu = y[idx]
    return xu.astype(np.float32), yu.astype(np.float32)


def interp1_safe(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    xo, yo = unique_sorted_xy(x_old, y_old)
    x_new = np.asarray(x_new, dtype=np.float64).reshape(-1)
    if len(xo) == 0:
        return np.full_like(x_new, np.nan, dtype=np.float32)
    return np.interp(x_new, xo, yo).astype(np.float32)


def finite_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(y) < 2:
        return np.zeros_like(y)
    return np.gradient(y, x, edge_order=1).astype(np.float32)
