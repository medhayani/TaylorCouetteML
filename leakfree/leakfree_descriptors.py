"""Leak-free branch descriptors.

For every elasticity E of the base, the branch structure (number of
branches and their k-supports), the shape descriptors and the four
anchors (k_left, k_right, Ta_min, Ta_max) are NOT computed from the
marginal curve of E itself.  They are interpolated in log10(E) between
the two nearest elasticities of the TRAINING set, E itself being
excluded when it belongs to the training set (leave-self-out).

Nothing from the target curve therefore enters the inputs of the model,
nor the normalisation of its targets.  A margin is applied to the
interpolated Ta_min (downwards) and Ta_max (upwards) so that the true
curve, resampled on the interpolated support, stays inside the
normalised range [0, 1] that the sigmoid head of the model can reach.

Branch matching between the two neighbours:
  * same number of branches       -> branch i with branch i (order along k),
                                      every numeric column interpolated
                                      linearly in log10(E) (Ta_min, Ta_max
                                      in log10);
  * different number of branches  -> the whole structure of the nearest
                                      neighbour (in log10 E) is taken.
Switch flags (has_switch_left/right) are always those of the nearest
neighbour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FLAG_COLS = ["has_switch_left", "has_switch_right"]
LOG_COLS = ["Ta_min", "Ta_max"]
KEEP_COLS = ["E", "branch_local_id"]


def _nearest_two(E: float, train_E: np.ndarray):
    """Two nearest training elasticities bracketing E (E itself excluded)."""
    cand = train_E[~np.isclose(train_E, E, rtol=1e-9)]
    below = cand[cand < E]
    above = cand[cand > E]
    lo = below.max() if below.size else None
    hi = above.min() if above.size else None
    if lo is None and hi is None:
        raise ValueError("no training elasticity available")
    if lo is None:                      # E below the training range
        two = np.sort(above)[:2]
        return float(two[0]), float(two[-1])
    if hi is None:                      # E above the training range
        two = np.sort(below)[-2:]
        return float(two[0]), float(two[-1])
    return float(lo), float(hi)


def interpolate_descriptors(desc: pd.DataFrame, train_E, margin: float = 0.05) -> pd.DataFrame:
    """Return a descriptor table with the same columns as `desc`, one row per
    predicted branch of every E, built only from training elasticities."""
    desc = desc.copy()
    train_E = np.asarray(sorted(set(np.round(np.asarray(train_E, dtype=float), 10))))
    num_cols = [c for c in desc.columns if c not in KEEP_COLS + FLAG_COLS
                and pd.api.types.is_numeric_dtype(desc[c])]
    groups = {round(float(E), 10): g.sort_values("branch_local_id").reset_index(drop=True)
              for E, g in desc.groupby("E")}
    out = []
    for E in sorted(groups):
        lo, hi = _nearest_two(E, train_E)
        g_lo, g_hi = groups[round(lo, 10)], groups[round(hi, 10)]
        lE, llo, lhi = np.log10(E), np.log10(lo), np.log10(hi)
        w = 0.0 if np.isclose(lhi, llo) else float(np.clip((lE - llo) / (lhi - llo), -0.5, 1.5))
        nearest = g_lo if abs(lE - llo) <= abs(lE - lhi) else g_hi
        if len(g_lo) == len(g_hi):
            new = g_lo.copy()
            for c in num_cols:
                a, b = g_lo[c].to_numpy(float), g_hi[c].to_numpy(float)
                if c in LOG_COLS:
                    a, b = np.log10(np.clip(a, 1e-9, None)), np.log10(np.clip(b, 1e-9, None))
                    new[c] = 10 ** ((1 - w) * a + w * b)
                else:
                    new[c] = (1 - w) * a + w * b
            for c in FLAG_COLS:
                if c in new.columns:
                    new[c] = nearest[c].to_numpy()
        else:
            new = nearest.copy()
        new["E"] = E
        new["branch_local_id"] = np.arange(len(new))
        # margins on the Taylor-number anchors
        new["Ta_min"] = new["Ta_min"] * (1.0 - margin)
        new["Ta_max"] = new["Ta_max"] * (1.0 + margin)
        if "amplitude" in new.columns:
            new["amplitude"] = new["Ta_max"] - new["Ta_min"]
        if "width_k" in new.columns:
            new["width_k"] = new["k_right"] - new["k_left"]
        new["source_lo"] = lo
        new["source_hi"] = hi
        new["same_branch_count"] = int(len(g_lo) == len(g_hi))
        out.append(new)
    return pd.concat(out, ignore_index=True)


def interpolate_curves(raw: pd.DataFrame, desc_lf: pd.DataFrame, n: int = 101) -> dict:
    """Neighbour-interpolated marginal curve for every predicted branch.

    For each row of `desc_lf` (a predicted branch of E with its interpolated
    support [k_left, k_right] and anchors, and the two training neighbours
    source_lo, source_hi that produced it), the raw marginal curves of the two
    neighbours are read at the k-grid of the branch and interpolated linearly
    in log10(E) (log10 Ta).  The result is normalised with the anchors of the
    branch, exactly as the target is.  Nothing from the curve of E is used.

    Returns {(E, branch_local_id): array(n)} of normalised context curves.
    """
    raw = raw.copy(); raw.columns = ["Ta", "k", "E"]
    curves = {round(float(E), 10): g.sort_values("k") for E, g in raw.groupby("E")}
    s = np.linspace(0.0, 1.0, n); out = {}
    for _, r in desc_lf.iterrows():
        E = float(r["E"]); lo = float(r["source_lo"]); hi = float(r["source_hi"])
        k_grid = r["k_left"] + s * (r["k_right"] - r["k_left"])
        lE, llo, lhi = np.log10(E), np.log10(lo), np.log10(hi)
        w = 0.0 if np.isclose(lhi, llo) else float(np.clip((lE - llo) / (lhi - llo), -0.5, 1.5))
        vals = []
        for nb in (lo, hi):
            g = curves[round(nb, 10)]
            vals.append(np.interp(k_grid, g["k"].to_numpy(), np.log10(np.clip(g["Ta"].to_numpy(), 1e-9, None)),
                                  left=np.nan, right=np.nan))
        a, b = vals
        c = np.where(np.isnan(a), b, np.where(np.isnan(b), a, (1 - w) * a + w * b))
        if np.isnan(c).any():                       # outside both neighbours: nearest valid value
            ok = ~np.isnan(c)
            c = np.interp(s, s[ok], c[ok]) if ok.any() else np.full(n, np.log10(max(r["Ta_min"], 1e-9)))
        amp = max(float(r["Ta_max"] - r["Ta_min"]), 1e-12)
        out[(round(E, 8), int(r["branch_local_id"]))] = ((10 ** c - float(r["Ta_min"])) / amp).astype(np.float32)
    return out


def check_against_truth(desc_true: pd.DataFrame, desc_lf: pd.DataFrame) -> dict:
    """Diagnostics: how far the interpolated anchors are from the true ones
    (dominant branch only), and how often the branch count is mismatched."""
    t = desc_true.loc[desc_true.groupby("E").Ta_min.idxmin()].set_index("E")
    l = desc_lf.loc[desc_lf.groupby("E").Ta_min.idxmin()].set_index("E")
    j = t[["Ta_min", "k_min"]].join(l[["Ta_min", "k_min", "same_branch_count"]], rsuffix="_lf", how="inner")
    err = 100 * (j.Ta_min_lf / (1 - 0.0) / j.Ta_min - 1)   # includes the margin
    nb_t = desc_true.groupby("E").size(); nb_l = desc_lf.groupby("E").size()
    return {
        "n_E": int(len(j)),
        "Ta_min_ratio_mean_pct": float(err.mean()), "Ta_min_ratio_max_pct": float(err.abs().max()),
        "k_min_MAE": float((j.k_min_lf - j.k_min).abs().mean()),
        "branch_count_equal_frac": float((nb_t.reindex(nb_l.index) == nb_l).mean()),
        "same_neighbour_count_frac": float(desc_lf.groupby("E").same_branch_count.first().mean()),
    }
