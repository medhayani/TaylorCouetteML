"""Build canonical curve table (split, E, log10E, branch, point_id, s, k, Ta_true, Ta_d5).

Faithful re-port of code4/06_Canonical/build_canonical_curve_table.py — but using
the shared common helpers instead of re-defining them inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from common import (
    choose_first,
    coerce_numeric,
    normalize_case_columns,
    require_columns,
)


def load_truth_long(profile_csv: Path) -> pd.DataFrame:
    df = normalize_case_columns(pd.read_csv(profile_csv))
    s_col = choose_first(df.columns, ["s", "s_resampled", "point_idx", "idx"])
    k_col = choose_first(df.columns, ["k_resampled", "k", "k_norm_resampled", "k_norm"])
    y_col = choose_first(df.columns, ["Ta_norm_resampled", "Ta_resampled", "Ta", "y_true"])
    if s_col is None or y_col is None:
        raise ValueError("profile_csv requires columns like 's' and 'Ta_resampled'.")
    df["s"] = pd.to_numeric(df[s_col], errors="coerce")
    df["k"] = pd.to_numeric(df[k_col] if k_col else df[s_col], errors="coerce")
    df["Ta_true"] = pd.to_numeric(df[y_col], errors="coerce")
    require_columns(df, ["split", "E", "branch_local_id", "s", "k", "Ta_true"], "profile_csv")
    df = coerce_numeric(df, ["E", "log10E", "branch_local_id", "s", "k", "Ta_true"])
    return df[["split", "E", "log10E", "branch_local_id", "s", "k", "Ta_true"]].copy()


def load_pred_long(pred_csv: Path) -> pd.DataFrame:
    df = normalize_case_columns(pd.read_csv(pred_csv))
    s_col = choose_first(df.columns, ["s", "s_resampled", "point_idx", "idx"])
    k_col = choose_first(df.columns, ["k_resampled", "k", "k_norm_resampled", "k_norm"])
    y_col = choose_first(df.columns, ["y_pred", "pred", "Ta_pred", "Ta_norm_pred"])
    if s_col is None or y_col is None:
        raise ValueError("pred_csv requires columns like 's' and 'y_pred'.")
    df["s"] = pd.to_numeric(df[s_col], errors="coerce")
    df["k"] = pd.to_numeric(df[k_col] if k_col else df[s_col], errors="coerce")
    df["Ta_d5"] = pd.to_numeric(df[y_col], errors="coerce")
    require_columns(df, ["split", "E", "branch_local_id", "s", "k", "Ta_d5"], "pred_csv")
    df = coerce_numeric(df, ["E", "log10E", "branch_local_id", "s", "k", "Ta_d5"])
    return df[["split", "E", "log10E", "branch_local_id", "s", "k", "Ta_d5"]].copy()


def build_canonical_table(profile_csv: Path, pred_csv: Path) -> pd.DataFrame:
    truth = load_truth_long(profile_csv)
    pred = load_pred_long(pred_csv)
    truth["_s_key"] = truth["s"].round(8)
    pred["_s_key"] = pred["s"].round(8)
    merged = truth.merge(
        pred[["split", "E", "branch_local_id", "_s_key", "Ta_d5"]],
        on=["split", "E", "branch_local_id", "_s_key"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Canonical merge returned zero rows.")
    merged = merged.sort_values(["split", "E", "branch_local_id", "s"]).reset_index(drop=True)
    merged["point_id"] = (
        merged.groupby(["split", "E", "branch_local_id"], sort=False)
        .cumcount()
        .astype(int)
    )
    keep = ["split", "E", "log10E", "branch_local_id", "point_id", "s", "k", "Ta_true", "Ta_d5"]
    return merged[keep].copy()
