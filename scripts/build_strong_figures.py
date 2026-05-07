"""Build the strong-retraining figures once Phases A/B/C are .done.

Reuses the pipeline from TaylorCouetteML/scripts/make_final_seven_models.py
but substitutes the new strong-retraining checkpoints. Saves into
strong_retraining/figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "figure.dpi": 130,
})


def smooth(y, window=7, polyorder=2):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="nearest")


def main():
    runs_strong = ROOT.parent / "data" / "runs_strong"
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    # Look for the strong canonical predictions.
    pt = runs_strong / "ptssst_long" / "canonical_pt_ssst.csv"
    fno = runs_strong / "fnoldm_ensemble" / "canonical_fnoldm.csv"

    available = []
    if pt.exists():
        available.append(("Sparse-MoE-T (PyTorch, strong)", pt, "Ta_ssst_pro", "#9467bd"))
    if fno.exists():
        available.append(("FNO + Latent Diffusion (M=5, strong)", fno, "Ta_neptune", "#ff7f0e"))

    if not available:
        print("No strong predictions found yet. Run phases A/B first.")
        sys.exit(0)

    canon = pd.read_csv(ROOT.parent / "code4" / "06_Canonical__Build_Canonical_Base"
                          / "02_outputs" / "canonical_curve_table_norm_fixed.csv")
    rows = []
    for name, csv, col, _color in available:
        df = pd.read_csv(csv)
        if "Ta_true" not in df.columns:
            df = df.merge(canon[["split", "E", "branch_local_id", "point_id", "Ta_true"]],
                          on=["split", "E", "branch_local_id", "point_id"], how="left")
        for split, g in df.groupby("split"):
            err = g["Ta_true"].to_numpy() - g[col].to_numpy()
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((g["Ta_true"] - g["Ta_true"].mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            rows.append({"split": split, "model": name,
                          "mae": mae, "rmse": rmse, "r2": r2})
    bench = pd.DataFrame(rows).sort_values(["split", "mae"])
    bench.to_csv(out / "benchmark_strong.csv", index=False)
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
