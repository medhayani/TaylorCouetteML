"""Final comparison figure with the seven *strong* models.

Models:
    Sparse-MoE-T + SAC          (legacy reference, code4)
    Sparse-MoE-T + 3-SAC        (legacy reference, code4)
    FNO + Latent Diffusion      (this work, M=5 ensemble, 200 epochs)
    Sparse-MoE-T (PyTorch)      (this work, 300 epochs)
    SAC (Conv1D+AttnPool)       (this work, refines PT-SSST)
    3-SAC + Cross-Attn          (this work, refines PT-SSST)
    Diffusion-Policy MARL       (this work, refines PT-SSST)
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


def per_split_metrics(df, true_col, pred_col, name):
    rows = []
    for split, g in df.groupby("split"):
        err = g[true_col].to_numpy() - g[pred_col].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((g[true_col] - g[true_col].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({"split": split, "model": name,
                      "mae": mae, "rmse": rmse, "r2": r2,
                      "n_rows": len(g),
                      "n_branches": g[["E","branch_local_id"]]
                                       .drop_duplicates().shape[0]})
    return pd.DataFrame(rows)


def load_all():
    code4 = ROOT.parent / "code4"
    runs_strong = ROOT.parent / "data" / "runs_strong"

    # Legacy (code4)
    canon = pd.read_csv(code4 / "06_Canonical__Build_Canonical_Base"
                         / "02_outputs" / "canonical_curve_table_norm_fixed.csv")
    sarl_legacy = pd.read_csv(code4 / "10_SARL_Export__Export_SARL_Canonical"
                                / "02_outputs"
                                / "sarl_predictions_canonical_norm_fixed.csv")
    sarl_legacy = sarl_legacy.rename(columns={"y_pred": "Ta_pred"}).merge(
        canon[["split","E","branch_local_id","point_id","Ta_true"]],
        on=["split","E","branch_local_id","point_id"], how="left")
    sarl_legacy["model"] = "Sparse-MoE-T + SAC"

    marl_legacy = pd.read_csv(code4 / "11_MARL_Export__Export_MARL_Canonical"
                                / "02_outputs"
                                / "marl_predictions_canonical_norm_fixed.csv")
    marl_legacy = marl_legacy.rename(columns={"y_pred": "Ta_pred"}).merge(
        canon[["split","E","branch_local_id","point_id","Ta_true"]],
        on=["split","E","branch_local_id","point_id"], how="left")
    marl_legacy["model"] = "Sparse-MoE-T + 3-SAC"

    # Strong models
    fno_strong = pd.read_csv(runs_strong / "inference_fnoldm" / "canonical_neptune_hydra.csv")
    fno_strong = fno_strong.rename(columns={"Ta_neptune":"Ta_pred"})
    fno_strong["model"] = "FNO + Latent Diffusion"

    pt_strong = pd.read_csv(runs_strong / "ptssst_long" / "canonical_ssst_pro.csv")
    pt_strong = pt_strong.rename(columns={"Ta_ssst_pro":"Ta_pred"})
    pt_strong["model"] = "Sparse-MoE-T (PyTorch)"

    sarl_strong = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_sarl_strong.csv")
    sarl_strong["model"] = "SAC (Conv1D+AttnPool)"

    marl_strong = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_marl_strong.csv")
    marl_strong["model"] = "3-SAC + Cross-Attn"

    hydra_strong = pd.read_csv(runs_strong / "inference_refiners_strong" / "canonical_hydra_strong.csv")
    hydra_strong["model"] = "Diffusion-Policy MARL"

    # Add k_phys
    profile_csv = (code4 / "04_StepC__Build_Modeling_Datasets" / "02_outputs"
                    / "04_StepC__stepC_modeling_datasets" / "model_profile_level_dataset.csv")
    prof = pd.read_csv(profile_csv)
    bf = (prof.sort_values(["E","branch_local_id","s"])
              .groupby(["E","branch_local_id"], sort=False).first().reset_index()
              [["E","branch_local_id","k_left","width_k"]])
    for d in [sarl_legacy, marl_legacy, fno_strong, pt_strong,
              sarl_strong, marl_strong, hydra_strong]:
        m = d.merge(bf, on=["E","branch_local_id"], how="left")
        d["k_phys"] = (m["k_left"] + m["s"]*m["width_k"]).to_numpy()

    bench = pd.concat([
        per_split_metrics(sarl_legacy,  "Ta_true", "Ta_pred", "Sparse-MoE-T + SAC"),
        per_split_metrics(marl_legacy,  "Ta_true", "Ta_pred", "Sparse-MoE-T + 3-SAC"),
        per_split_metrics(fno_strong,   "Ta_true", "Ta_pred", "FNO + Latent Diffusion"),
        per_split_metrics(pt_strong,    "Ta_true", "Ta_pred", "Sparse-MoE-T (PyTorch)"),
        per_split_metrics(sarl_strong,  "Ta_true", "Ta_pred", "SAC (Conv1D+AttnPool)"),
        per_split_metrics(marl_strong,  "Ta_true", "Ta_pred", "3-SAC + Cross-Attn"),
        per_split_metrics(hydra_strong, "Ta_true", "Ta_pred", "Diffusion-Policy MARL"),
    ], ignore_index=True)
    return {"Sparse-MoE-T + SAC": sarl_legacy,
             "Sparse-MoE-T + 3-SAC": marl_legacy,
             "FNO + Latent Diffusion": fno_strong,
             "Sparse-MoE-T (PyTorch)": pt_strong,
             "SAC (Conv1D+AttnPool)": sarl_strong,
             "3-SAC + Cross-Attn": marl_strong,
             "Diffusion-Policy MARL": hydra_strong,
             "bench": bench}


def fig_metrics_bar(bench, out):
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    width = 0.11; x = np.arange(len(splits))
    for ax, met, lbl in zip(axes,
                              ["mae", "rmse", "r2"],
                              ["MAE_norm", "RMSE_norm", r"$R^2$"]):
        for i, mdl in enumerate(MODELS):
            vals = []
            for sp in splits:
                row = bench[(bench["split"]==sp) & (bench["model"]==mdl)]
                vals.append(float(row[met].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i-3)*width, vals, width=width, label=mdl,
                   color=COL[mdl])
        ax.set_xticks(x); ax.set_xticklabels(splits)
        ax.set_title(lbl); ax.legend(fontsize=7, ncol=2)
        if met != "r2": ax.set_ylabel("error")
    fig.suptitle("Performance by split — strong-retrained pipeline", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_curves(data, out, n=12, seed=0):
    keys = list(data["FNO + Latent Diffusion"][data["FNO + Latent Diffusion"]["split"]=="test"]
                  .groupby(["E","branch_local_id"], sort=False).groups.keys())
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(keys), size=min(n,len(keys)), replace=False)
    sel = sorted(sel, key=lambda i: keys[i][0])

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 2.9*nrows), squeeze=False)
    axes = axes.flatten()

    for i_p, idx in enumerate(sel):
        E_val, b = keys[idx]
        ax = axes[i_p]
        np_df = data["FNO + Latent Diffusion"]
        gn = np_df[(np_df["E"]==E_val) & (np_df["branch_local_id"]==b)].sort_values("k_phys")
        kf = gn["k_phys"].to_numpy()
        ta_n = smooth(gn["Ta_pred"].to_numpy())
        sig = smooth(gn["sigma_ep"].to_numpy())
        ax.fill_between(kf, ta_n-sig, ta_n+sig, alpha=0.15, color=COL["sigma"])
        ta_true = gn["Ta_true"].to_numpy()
        ax.plot(kf, ta_true, color=COL["true"], lw=2.2,
                label=r"$T_a^{\rm true}$" if i_p==0 else None)
        for name in MODELS:
            d = data[name]
            g = d[(d["E"]==E_val) & (d["branch_local_id"]==b)].sort_values("k_phys")
            if len(g)==0: continue
            ax.plot(g["k_phys"].to_numpy(), smooth(g["Ta_pred"].to_numpy()),
                    color=COL[name], lw=1.3, ls=LS[name],
                    label=name if i_p==0 else None)
        ax.set_title(f"$E={E_val:.2g}$, b={int(b)}, "
                     f"$\\log_{{10}}E={np.log10(max(E_val,1e-30)):.1f}$",
                     fontsize=9)
        ax.set_xlabel("$k$"); ax.set_ylabel(r"$T_a$")
        if i_p==0: ax.legend(fontsize=6, loc="best", ncol=2)
    for j in range(len(sel), len(axes)): axes[j].axis("off")
    fig.suptitle(r"Marginal stability $T_a(k)$ — strong-retrained pipeline (seven models)", y=1.00)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_mae_vs_logE(data, out):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    markers = {"Sparse-MoE-T + SAC":"^", "Sparse-MoE-T + 3-SAC":"v",
                "FNO + Latent Diffusion":"s",
                "Sparse-MoE-T (PyTorch)":"x", "SAC (Conv1D+AttnPool)":"o",
                "3-SAC + Cross-Attn":"D", "Diffusion-Policy MARL":"*"}
    for name in MODELS:
        d = data[name]
        test = d[d["split"]=="test"].copy()
        g = (test.assign(err=lambda x: (x["Ta_true"]-x["Ta_pred"]).abs())
                  .groupby(["E","log10E","branch_local_id"])["err"].mean()
                  .reset_index())
        ax.scatter(g["log10E"], g["err"], s=22, alpha=0.7,
                   color=COL[name], marker=markers[name], label=name)
    ax.set_xlabel(r"$\log_{10} E$"); ax.set_ylabel("MAE per branch (test)")
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2)
    ax.set_title("Per-branch MAE vs elasticity number — strong-retrained pipeline")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_uncertainty(data, out):
    test = data["FNO + Latent Diffusion"][data["FNO + Latent Diffusion"]["split"]=="test"].copy()
    err = (test["Ta_true"]-test["Ta_pred"]).abs()
    sig = test["sigma_ep"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sig, err, s=4, alpha=0.10, color=COL["FNO + Latent Diffusion"])
    bins = np.linspace(0, sig.quantile(0.99), 12)
    digit = np.digitize(sig, bins); means = []
    for i in range(1, len(bins)):
        m = digit==i
        means.append((bins[i-1], err[m].mean() if m.any() else np.nan))
    bx, by = zip(*means)
    ax.plot(bx, by, "ko-", lw=1.5, label="binned mean |error|")
    lim = max(sig.max(), err.max())*1.05
    ax.plot([0,lim], [0,lim], "k--", alpha=0.4, label="ideal calibration")
    ax.set_xlabel(r"FNO+LDM $\sigma_{\rm ep}$ per point")
    ax.set_ylabel("|error| per point")
    ax.set_title("FNO+LDM (strong, M=5) predictive uncertainty calibration on test split")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    data = load_all()
    bench = data["bench"]
    out_strong = ROOT / "figures"
    out_strong.mkdir(parents=True, exist_ok=True)
    fig_metrics_bar(bench, out_strong / "fig_metrics_bar.png")
    fig_curves(data, out_strong / "fig_curves_vs_k.png")
    fig_mae_vs_logE(data, out_strong / "fig_mae_vs_logE.png")
    fig_uncertainty(data, out_strong / "fig_uncertainty.png")

    out_article = ROOT.parent / "TaylorCouetteML" / "docs" / "figures"
    for f in ["fig_metrics_bar.png", "fig_curves_vs_k.png",
              "fig_mae_vs_logE.png", "fig_uncertainty.png"]:
        (out_article / f).write_bytes((out_strong / f).read_bytes())

    bench.sort_values(["split","mae"]).to_csv(out_strong / "benchmark_strong.csv",
                                                index=False)
    print()
    print(bench.sort_values(["split","mae"]).to_string(index=False))


if __name__ == "__main__":
    main()
