"""PyTorch dataset of the RL windows.

    HydraWindowsDataset : the RL windows (obs_seq, static_vec, y_true, y_pred)
                          used by the SARL and MARL refiners

It reads directly the window files of data/rl_windows/,
so you do not need to re-run the early data pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---- the 23 static branch descriptors, in the order used everywhere in the repository ----
STATIC_FEATURES = [
    "log10E",
    "branch_order_norm",
    "width_k",
    "width_asymmetry",
    "rise_asymmetry",
    "slope_left_local",
    "slope_right_local",
    "global_slope",
    "curvature_at_min",
    "roughness_rmse",
    "normalized_arc_length",
    "has_switch_left",
    "has_switch_right",
    "mean_abs_curvature",
    "amplitude",
    "n_branches",
    "is_first_branch",
    "is_last_branch",
    "left_width",
    "right_width",
    "left_rise",
    "right_rise",
    "mean_abs_slope",
]


def _safe_pivot_curves(
    df: pd.DataFrame,
    target_col: str = "Ta_norm_resampled",
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, int]], np.ndarray]:
    """Group profile-level rows by (E, branch_local_id) and pivot Ta into (N, T) array.

    Returns:
        ta_arr   : (N, T) target curves
        s_arr    : (N, T) abscissa values
        keys     : list of (E, branch_local_id) tuples in row order
        splits   : (N,) array of split labels (str)
    """
    ta_rows: List[np.ndarray] = []
    s_rows: List[np.ndarray] = []
    keys: List[Tuple[float, int]] = []
    splits: List[str] = []
    for (E_val, b_id), g in df.groupby(["E", "branch_local_id"], sort=False):
        g = g.sort_values("s")
        ta_rows.append(g[target_col].to_numpy(dtype=np.float32))
        s_rows.append(g["s"].to_numpy(dtype=np.float32))
        keys.append((float(E_val), int(b_id)))
        splits.append(str(g["split"].iloc[0]))
    ta_arr = np.stack(ta_rows, axis=0)
    s_arr = np.stack(s_rows, axis=0)
    return ta_arr, s_arr, keys, np.asarray(splits)


def _build_static_context(
    df: pd.DataFrame,
    keys: List[Tuple[float, int]],
    static_features: List[str] = STATIC_FEATURES,
) -> np.ndarray:
    """Pull the static features for each (E, branch_local_id) key (one row per branch)."""
    branch_first = (
        df.sort_values(["E", "branch_local_id", "s"])
          .groupby(["E", "branch_local_id"], sort=False)
          .first()
          .reset_index()
    )
    lookup = {(float(r.E), int(r.branch_local_id)): r for _, r in branch_first.iterrows()}
    ctx_rows = []
    for (E_val, b_id) in keys:
        row = lookup[(E_val, b_id)]
        feat = []
        for col in static_features:
            v = row.get(col, np.nan) if isinstance(row, pd.Series) else getattr(row, col, np.nan)
            try:
                feat.append(float(v))
            except (TypeError, ValueError):
                feat.append(0.0)
        ctx_rows.append(feat)
    arr = np.asarray(ctx_rows, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class HydraWindowsDataset(Dataset):
    """Dataset over the .npz windows produced by Step F (one window per branch)."""

    def __init__(self, npz_path: Path, normalize_static: bool = True,
                 stat_path: Optional[Path] = None):
        d = np.load(npz_path, allow_pickle=False)

        def clean(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float32)
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        self.obs_seq = clean(d["obs_seq"])              # (N, T, F)
        self.static_vec = clean(d["static_vec"])        # (N, S)
        self.y_true = clean(d["y_true"])                # (N, T)
        self.y_pred = clean(d["y_pred"])                # (N, T)
        self.local_grid = clean(d["local_grid"])        # (N, T)
        self.center_pred = clean(d["center_pred"])
        self.center_true = clean(d["center_true"])
        self.window_half_width = clean(d["window_half_width"])
        self.E = clean(d["E"])
        self.log10E = clean(d["log10E"])
        self.branch_local_id = d["branch_local_id"].astype(np.int64)
        self.window_id = d["window_id"].astype(np.int64)
        if normalize_static:
            if stat_path and Path(stat_path).exists():
                stat = json.loads(Path(stat_path).read_text(encoding="utf-8"))
                mean = np.asarray(stat["mean"], dtype=np.float32)
                std = np.asarray(stat["std"], dtype=np.float32) + 1e-6
            else:
                mean = np.nanmean(self.static_vec, axis=0)
                std = np.nanstd(self.static_vec, axis=0) + 1e-6
                mean = np.nan_to_num(mean, nan=0.0)
                std = np.nan_to_num(std, nan=1.0)
            self.static_mean, self.static_std = mean, std
            self.static_vec = (self.static_vec - mean) / std
            self.static_vec = clean(self.static_vec)
            self.static_vec = np.clip(self.static_vec, -5.0, 5.0)

    def __len__(self) -> int:
        return self.obs_seq.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "obs_seq":     torch.from_numpy(self.obs_seq[i]),
            "static_vec":  torch.from_numpy(self.static_vec[i]),
            "y_true":      torch.from_numpy(self.y_true[i]),
            "y_pred":      torch.from_numpy(self.y_pred[i]),
            "local_grid":  torch.from_numpy(self.local_grid[i]),
            "center_pred": torch.tensor(self.center_pred[i]),
            "center_true": torch.tensor(self.center_true[i]),
            "E":           torch.tensor(self.E[i]),
            "log10E":      torch.tensor(self.log10E[i]),
            "branch_local_id": torch.tensor(self.branch_local_id[i]),
            "window_id":   torch.tensor(self.window_id[i]),
        }
