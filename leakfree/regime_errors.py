"""Errors on the critical point, band by band in elasticity, on the 64 held-out
elasticities only.  Feeds the validation figure of the paper.

Bands as in the paper: E < 6.6e-3, 6.6e-3 <= E < 0.58, E >= 0.58.
For each model: mean relative error on Ta_c and on k_c (per cent), and the mean
absolute error on k_c.  Writes results/regime_errors.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
B1, B2 = 6.6e-3, 0.58
BANDS = [("low", 0.0, B1), ("intermediate", B1, B2), ("high", B2, np.inf)]


def load() -> dict[str, pd.DataFrame]:
    """One frame per model: E, Ta_c_true, k_c_true, Ta_c_pred, k_c_pred, test rows only."""
    out = {}
    for name, f in (("CNP zero-shot", "cnp/per_E.csv"), ("CNP conditioned", "cnp_nbr/per_E.csv")):
        d = pd.read_csv(RES / f)
        out[name] = d[d.split == "test"][["E", "Ta_c_true", "k_c_true", "Ta_c_pred", "k_c_pred"]]
    out["DIST"] = pd.read_csv(RES / "dist/per_E_dist.csv")[["E", "Ta_c_true", "k_c_true", "Ta_c_pred", "k_c_pred"]]
    d = pd.read_csv(RES / "rl/per_E_rl.csv")
    out["Median"] = d.rename(columns={"Ta_c_base": "Ta_c_pred", "k_c_base": "k_c_pred"})[
        ["E", "Ta_c_true", "k_c_true", "Ta_c_pred", "k_c_pred"]]
    out["SARL"] = d.rename(columns={"Ta_c_rl": "Ta_c_pred", "k_c_rl": "k_c_pred"})[
        ["E", "Ta_c_true", "k_c_true", "Ta_c_pred", "k_c_pred"]]
    d = pd.read_csv(RES / "marl/per_E_marl.csv")
    out["MARL"] = d.rename(columns={"Ta_c_marl": "Ta_c_pred", "k_c_marl": "k_c_pred"})[
        ["E", "Ta_c_true", "k_c_true", "Ta_c_pred", "k_c_pred"]]
    return out


def main() -> None:
    frames = load()
    table = {}
    for name, d in frames.items():
        rows = {}
        for band, lo, hi in BANDS + [("all", 0.0, np.inf)]:
            g = d[(d.E >= lo) & (d.E < hi)]
            rows[band] = dict(
                n=int(len(g)),
                Ta_c_mape=float(100 * np.mean(np.abs(g.Ta_c_pred / g.Ta_c_true - 1))),
                k_c_mape=float(100 * np.mean(np.abs(g.k_c_pred / g.k_c_true - 1))),
                k_c_mae=float(np.mean(np.abs(g.k_c_pred - g.k_c_true))),
            )
        table[name] = rows
    (RES / "regime_errors.json").write_text(json.dumps(table, indent=1), encoding="utf-8")

    print(f"{'model':18s} {'band':13s} {'n':>3s} {'Ta_c %':>8s} {'k_c %':>8s} {'|dk_c|':>8s}")
    for name, rows in table.items():
        for band, r in rows.items():
            print(f"{name:18s} {band:13s} {r['n']:3d} {r['Ta_c_mape']:8.2f} {r['k_c_mape']:8.2f} {r['k_c_mae']:8.2f}")
        print()


if __name__ == "__main__":
    main()
