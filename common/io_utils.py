"""DataFrame helpers reused across data-pipeline + canonical export steps."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def choose_first(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None


def maybe_add_log10E(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "log10E" not in out.columns:
        ecol = choose_first(out.columns, ["E", "E_value", "energy", "e"])
        if ecol is not None:
            vals = pd.to_numeric(out[ecol], errors="coerce")
            mask = vals > 0
            out.loc[mask, "log10E"] = np.log10(vals[mask])
            if "E" not in out.columns:
                out["E"] = vals
    return out


def normalize_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = maybe_add_log10E(df)
    if "split" not in out.columns:
        alt = choose_first(out.columns, ["set", "subset"])
        if alt is not None:
            out = out.rename(columns={alt: "split"})
    if "branch_local_id" not in out.columns:
        alt = choose_first(out.columns, ["branch_id", "b", "branch", "branch_index"])
        if alt is not None:
            out = out.rename(columns={alt: "branch_local_id"})
    if "E" not in out.columns and "log10E" in out.columns:
        out["E"] = 10.0 ** pd.to_numeric(out["log10E"], errors="coerce")
    return out


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def safe_np(x, nan: float = 0.0, pos: float = 0.0, neg: float = 0.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(arr, nan=nan, posinf=pos, neginf=neg).astype(np.float32)
