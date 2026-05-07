"""Test the strong-retrained models on the original Input file.

Input format:
    Value, Time, E   (= Ta, k, E)  in physical units
    122820 rows, 420 unique E values, 720 underlying branches.

Output:
    - figures/test_on_input_curves.png   (12 branches, Ta(k) physical, true vs all 7 models)
    - figures/test_on_input_metrics.csv  (per-model MAE in physical Ta units)
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
    "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "figure.dpi": 130,
})


def smooth(y, window=7, polyorder=2):
    y = np.asarray(y, dtype=float)
    if len(y) < window:
        return y
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="nearest")


COL = {
    "Sparse-MoE-T + SAC":      "#2ca02c",
    "Sparse-MoE-T + 3-SAC":    "#d62728",
    "FNO + Latent Diffusion":  "#ff7f0e",
    "Sparse-MoE-T (PyTorch)":  "#9467bd",
    "SAC (Conv1D+AttnPool)":   "#17becf",
    "3-SAC + Cross-Attn":      "#bcbd22",
    "Diffusion-Policy MARL":   "#e377c2",
    "true":                    "#000000",
    "sigma":                   "#ffbb78",
}
LS = {
    "Sparse-MoE-T + SAC":      ":",
    "Sparse-MoE-T + 3-SAC":    ":",
    "FNO + Latent Diffusion":  "-",
    "Sparse-MoE-T (PyTorch)":  "--",
    "SAC (Conv1D+AttnPool)":   "-.",
    "3-SAC + Cross-Attn":      "-.",
    "Diffusion-Policy MARL":   "--",
}
MODELS = list(LS.keys())


def main():
    code4 = ROOT.parent / "code4"
    runs_strong = ROOT.parent / "data" / "runs_strong"
    input_csv = ROOT.parent / "data" / "Input" / "combined_data.csv"

    # ---- 1) Load original Input ----
    raw = pd.read_csv(input_csv)
    raw.columns = ["Ta", "k", "E"]
    print(f"Original input: {len(raw)} rows, {raw['E'].nunique()} unique E")

    # ---- 2) Branch descriptors (Step A output) ----
    desc = pd.read_csv(code4 / "01_StepA__Segmented_Functional_Analysis" / "02_outputs"
                         / "01_StepA__stepA_segmented_functional_outputs"
                         / "branch_functional_descriptors.csv")
    print(f"Branches: {len(desc)} (= 720 expected)")

    # ---- 3) Load strong canonical predictions ----
    canon = pd.read_csv(code4 / "06_Canonical__Build_Canonical_Base"
                          / "02_outputs" / "canonical_curve_table_norm_fixed.csv")
    sarl_legacy = pd.read_csv(code4 / "10_SARL_Export__Export_SARL_Canonical"
                                / "02_outputs"
                                / "sarl_predictions_canonical_norm_fixed.csv").rename(
                                  columns={"y_pred":"Ta_pred"})
    marl_legacy = pd.read_csv(code4 / "11_MARL_Export__Export_MARL_Canonical"
                                / "02_outputs"
                                / "marl_predictions_canonical_norm_fixed.csv").rename(
                                  columns={"y_pred":"Ta_pred"})
    fno_strong = pd.read_csv(runs_strong / "inference_fnoldm" / "canonical_neptune_hydra.csv").rename(
                              columns={"Ta_neptune":"Ta_pred"})
    pt_strong = pd.read_csv(runs_strong / "ptssst_long" / "canonical_ssst_pro.csv").rename(
                             columns={"Ta_ssst_pro":"Ta_pred"})
    sarl_strong  = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_sarl_strong.csv")
    marl_strong  = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_marl_strong.csv")
    hydra_strong = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_hydra_strong.csv")

    pred_dfs = {
        "Sparse-MoE-T + SAC":     sarl_legacy,
        "Sparse-MoE-T + 3-SAC":   marl_legacy,
        "FNO + Latent Diffusion": fno_strong,
        "Sparse-MoE-T (PyTorch)": pt_strong,
        "SAC (Conv1D+AttnPool)":  sarl_strong,
        "3-SAC + Cross-Attn":     marl_strong,
        "Diffusion-Policy MARL":  hydra_strong,
    }

    # ---- 4) Pick 12 representative branches across log10(E) ----
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
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

        # --- Original Input data restricted to this branch ---
        sub = raw[(np.isclose(raw["E"], E_val)) &
                  (raw["k"] >= k_left - 0.05) & (raw["k"] <= k_right + 0.05)].sort_values("k")
        if len(sub) == 0:
            ax.text(0.5, 0.5, f"no data\nE={E_val:.3g}", transform=ax.transAxes,
                    ha="center")
            continue
        ax.plot(sub["k"], sub["Ta"], color="black", lw=2.0, label="Input (true Ta)")

        # --- Each model: load canonical prediction, denormalize, plot in physical units ---
        for name, df in pred_dfs.items():
            g = df[(np.isclose(df["E"], E_val)) &
                   (df["branch_local_id"] == b)].sort_values("s")
            if len(g) == 0:
                continue
            s_canon = g["s"].to_numpy()
            ta_pred_norm = smooth(g["Ta_pred"].to_numpy())
            # denormalize: Ta = Ta_min + amp * Ta_norm
            ta_pred_phys = Ta_min + amp * ta_pred_norm
            # rebuild physical k from s
            k_phys = k_left + s_canon * (k_right - k_left)
            ax.plot(k_phys, ta_pred_phys,
                    color=COL[name], lw=1.2, ls=LS[name],
                    label=name if i == 0 else None)

        ax.set_title(f"E={E_val:.3g}, branch={b}, "
                     f"$\\log_{{10}}E={row['log10E']:.1f}$",
                     fontsize=9)
        ax.set_xlabel("k (physical)"); ax.set_ylabel("Ta (physical)")
        if i == 0:
            ax.legend(fontsize=6, loc="best", ncol=2)

    fig.suptitle(r"Strong-retrained models on original Input data — 12 branches across $\log_{10}E$",
                 y=1.00)
    fig.tight_layout()

    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "test_on_input_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir/'test_on_input_curves.png'}")

    # Also copy to article figures
    target = ROOT.parent / "TaylorCouetteML" / "docs" / "figures" / "test_on_input_curves.png"
    target.write_bytes((out_dir / "test_on_input_curves.png").read_bytes())
    print(f"Copied to article figures -> {target}")

    # ---- 5) Aggregate physical-MAE across the entire Input ----
    rows = []
    for name, df in pred_dfs.items():
        errs = []
        for idx, row in desc.iterrows():
            E_val, b = float(row["E"]), int(row["branch_local_id"])
            k_left, k_right = float(row["k_left"]), float(row["k_right"])
            Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
            amp = max(Ta_max - Ta_min, 1e-12)
            sub = raw[(np.isclose(raw["E"], E_val)) &
                      (raw["k"] >= k_left - 0.05) & (raw["k"] <= k_right + 0.05)].sort_values("k")
            if len(sub) == 0: continue
            g = df[(np.isclose(df["E"], E_val)) &
                   (df["branch_local_id"] == b)].sort_values("s")
            if len(g) == 0: continue
            s_canon = g["s"].to_numpy()
            ta_pred_phys = Ta_min + amp * g["Ta_pred"].to_numpy()
            k_pred = k_left + s_canon * (k_right - k_left)
            ta_input = np.interp(k_pred, sub["k"].to_numpy(), sub["Ta"].to_numpy())
            errs.append(np.abs(ta_input - ta_pred_phys))
        all_err = np.concatenate(errs) if errs else np.array([])
        rows.append({"model": name,
                      "MAE_phys (Ta units)": float(np.mean(all_err)),
                      "RMSE_phys": float(np.sqrt(np.mean(all_err**2))),
                      "n_branches": len(errs)})
    metrics = pd.DataFrame(rows).sort_values("MAE_phys (Ta units)")
    metrics.to_csv(out_dir / "test_on_input_metrics.csv", index=False)
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
