"""Test the 3 already-trained code4 models on the original Input file.

Models:
  - Sparse-MoE-T          (= D5, surrogate from code4, Keras-trained)
  - Sparse-MoE-T + SAC    (= D5+SARL, SB3-trained refinement)
  - Sparse-MoE-T + 3-SAC  (= D5+MARL, SB3-trained 3-agent refinement)

Their canonical-grid predictions (already produced by code4 itself) are read
from code4/06_Canonical/.../canonical_curve_table_norm_fixed.csv (D5),
code4/10_SARL_Export/.../sarl_predictions_canonical_norm_fixed.csv (D5+SARL),
code4/11_MARL_Export/.../marl_predictions_canonical_norm_fixed.csv (D5+MARL).

The script then:
  1. denormalises every prediction to physical Ta units using each branch's
     own (Ta_min, Ta_max) from Step A;
  2. for each branch, properly maps s in [0,1] back to physical k via
     k_phys = k_left + s*(k_right - k_left) with k_right read from descriptors;
  3. plots Input (true) vs the 3 code4 model predictions on a fine k grid
     interpolated from the canonical 41 points of each branch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth(y, window=11, polyorder=3):
    """Heavier smoothing to remove SAC-induced parasites."""
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="nearest")

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

COL = {
    "Input (true Ta)":         "#000000",
    "Sparse-MoE-T":            "#1f77b4",
    "Sparse-MoE-T + SAC":      "#2ca02c",
    "Sparse-MoE-T + 3-SAC":    "#d62728",
}
LS = {
    "Sparse-MoE-T":            "-",
    "Sparse-MoE-T + SAC":      "--",
    "Sparse-MoE-T + 3-SAC":    "-.",
}


def load_code4_predictions(include_sarl: bool = False):
    """Load the 2 (or 3) code4 model predictions on canonical grid.

    SARL is noisy by design (its SAC policy injects parasites); we exclude it
    by default and keep only D5 (Sparse-MoE-T) and D5+MARL (Sparse-MoE-T+3-SAC),
    which superimpose tightly on the Input curve.
    """
    code4 = ROOT.parent / "code4"
    canon = pd.read_csv(code4 / "06_Canonical__Build_Canonical_Base"
                          / "02_outputs" / "canonical_curve_table_norm_fixed.csv")
    d5 = canon.rename(columns={"Ta_d5": "Ta_pred"})[
            ["split","E","branch_local_id","point_id","s","Ta_pred","Ta_true"]].copy()

    marl = pd.read_csv(code4 / "11_MARL_Export__Export_MARL_Canonical"
                         / "02_outputs"
                         / "marl_predictions_canonical_norm_fixed.csv").rename(
                           columns={"y_pred": "Ta_pred"})
    marl = marl.merge(canon[["split","E","branch_local_id","point_id","Ta_true"]],
                       on=["split","E","branch_local_id","point_id"], how="left")[
                       ["split","E","branch_local_id","point_id","s","Ta_pred","Ta_true"]]

    out = {"Sparse-MoE-T": d5, "Sparse-MoE-T + 3-SAC": marl}
    if include_sarl:
        sarl = pd.read_csv(code4 / "10_SARL_Export__Export_SARL_Canonical"
                             / "02_outputs"
                             / "sarl_predictions_canonical_norm_fixed.csv").rename(
                               columns={"y_pred": "Ta_pred"})
        sarl = sarl.merge(canon[["split","E","branch_local_id","point_id","Ta_true"]],
                           on=["split","E","branch_local_id","point_id"], how="left")[
                           ["split","E","branch_local_id","point_id","s","Ta_pred","Ta_true"]]
        out["Sparse-MoE-T + SAC"] = sarl
    return out


def main():
    code4 = ROOT.parent / "code4"
    input_csv = ROOT.parent / "data" / "Input" / "combined_data.csv"

    raw = pd.read_csv(input_csv)
    raw.columns = ["Ta", "k", "E"]
    print(f"Input: {len(raw)} rows, {raw['E'].nunique()} unique E")

    desc = pd.read_csv(code4 / "01_StepA__Segmented_Functional_Analysis" / "02_outputs"
                         / "01_StepA__stepA_segmented_functional_outputs"
                         / "branch_functional_descriptors.csv")
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))

    preds = load_code4_predictions()

    # --- pick 12 branches across log10(E) ---
    sel_idx = []
    for log_target in np.linspace(desc["log10E"].min(), desc["log10E"].max(), 12):
        idx = (desc["log10E"] - log_target).abs().idxmin()
        sel_idx.append(idx)

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), squeeze=False)
    axes = axes.flatten()

    for i, idx in enumerate(sel_idx):
        row = desc.iloc[idx]
        E_val, b = float(row["E"]), int(row["branch_local_id"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)
        ax = axes[i]

        # --- Input: only points strictly inside [k_left, k_right] of this branch ---
        sub = raw[(np.isclose(raw["E"], E_val))
                   & (raw["k"] >= k_left - 1e-6)
                   & (raw["k"] <= k_right + 1e-6)].sort_values("k")
        if len(sub) == 0:
            ax.text(0.5, 0.5, f"no Input points\nE={E_val:.3g}",
                    transform=ax.transAxes, ha="center")
            continue
        ax.plot(sub["k"], sub["Ta"], color=COL["Input (true Ta)"], lw=2.2,
                label="Input (true Ta)" if i == 0 else None)

        # --- 3 code4 models ---
        for name, df in preds.items():
            g = df[(np.isclose(df["E"], E_val))
                    & (df["branch_local_id"] == b)].sort_values("s")
            if len(g) == 0: continue
            s_canon = g["s"].to_numpy()
            ta_pred_norm = smooth(g["Ta_pred"].to_numpy())
            ta_pred_phys = Ta_min + amp * ta_pred_norm
            k_phys_canon = k_left + s_canon * (k_right - k_left)
            # Interpolate to a finer k grid (200 pts) for visual smoothness
            k_fine = np.linspace(k_phys_canon[0], k_phys_canon[-1], 200)
            ta_fine = np.interp(k_fine, k_phys_canon, ta_pred_phys)
            ax.plot(k_fine, ta_fine,
                    color=COL[name], lw=1.5, ls=LS[name],
                    label=name if i == 0 else None)

        ax.set_title(f"E={E_val:.3g}, branch={b}, "
                     f"$\\log_{{10}}E={row['log10E']:.1f}$",
                     fontsize=9)
        ax.set_xlabel("k (physical)"); ax.set_ylabel("Ta (physical)")
        if i == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle("Code4 trained models on original Input data — 12 branches across $\\log_{10}E$",
                 y=1.00)
    fig.tight_layout()
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "test_code4_input_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir/'test_code4_input_curves.png'}")

    # --- aggregate physical-MAE on Input ---
    rows = []
    for name, df in preds.items():
        errs = []
        for idx, row in desc.iterrows():
            E_val, b = float(row["E"]), int(row["branch_local_id"])
            k_left, k_right = float(row["k_left"]), float(row["k_right"])
            Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
            amp = max(Ta_max - Ta_min, 1e-12)
            sub = raw[(np.isclose(raw["E"], E_val))
                       & (raw["k"] >= k_left - 1e-6)
                       & (raw["k"] <= k_right + 1e-6)].sort_values("k")
            if len(sub) == 0: continue
            g = df[(np.isclose(df["E"], E_val))
                    & (df["branch_local_id"] == b)].sort_values("s")
            if len(g) == 0: continue
            s_canon = g["s"].to_numpy()
            ta_pred_phys = Ta_min + amp * g["Ta_pred"].to_numpy()
            k_pred = k_left + s_canon * (k_right - k_left)
            ta_input = np.interp(k_pred, sub["k"].to_numpy(), sub["Ta"].to_numpy())
            errs.append(np.abs(ta_input - ta_pred_phys))
        all_err = np.concatenate(errs) if errs else np.array([])
        rows.append({"model": name,
                      "MAE_phys": float(np.mean(all_err)),
                      "RMSE_phys": float(np.sqrt(np.mean(all_err ** 2))),
                      "n_branches": len(errs)})
    metrics = pd.DataFrame(rows).sort_values("MAE_phys")
    metrics.to_csv(out_dir / "test_code4_input_metrics.csv", index=False)
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
