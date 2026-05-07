"""High-quality figures for the code7 PRO article (EN + FR).

Produces 4 figures + a benchmark table CSV:
    fig_metrics_bar.png       — MAE/RMSE/R^2 bars by split (NEPTUNE_PRO + SSST_PRO)
    fig_curves_examples.png   — 12 representative test branches (true, NEPTUNE_PRO, SSST_PRO, sigma)
    fig_mae_vs_logE.png       — per-branch MAE vs log10(E)
    fig_uncertainty.png       — |error| vs sigma_ep calibration plot
    benchmark_pro.csv         — combined metrics table
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import get_logger


mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "figure.dpi": 130,
})

COLORS = {
    "NEPTUNE_PRO": "#ff7f0e",
    "SSST_PRO":    "#1f77b4",
    "true":        "#000000",
    "sigma":       "#ffbb78",
}


def fig_metrics_bar(df_combined: pd.DataFrame, out: Path):
    splits = ["train", "val", "test"]
    models = ["NEPTUNE_PRO", "SSST_PRO"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    width = 0.35; x = np.arange(len(splits))
    for ax, met, lbl in zip(axes,
                              ["mae", "rmse", "r2"],
                              ["MAE_norm", "RMSE_norm", r"$R^2$"]):
        for i, mdl in enumerate(models):
            vals = []
            for sp in splits:
                row = df_combined[(df_combined["split"] == sp) & (df_combined["model"] == mdl)]
                vals.append(float(row[met].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=mdl,
                   color=COLORS.get(mdl, "gray"))
        ax.set_xticks(x); ax.set_xticklabels(splits)
        ax.set_title(lbl); ax.legend(fontsize=8)
        if met != "r2": ax.set_ylabel("error")
    fig.suptitle("Code7 PRO — performance by split", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_curves_examples(df_n: pd.DataFrame, df_s: pd.DataFrame, out: Path,
                          n: int = 12, seed: int = 0):
    test_n = df_n[df_n["split"] == "test"]
    keys = list(test_n.groupby(["E", "branch_local_id"], sort=False).groups.keys())
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    sel = sorted(sel, key=lambda i: keys[i][0])

    nrows = 4; ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.4 * nrows),
                              squeeze=False)
    axes = axes.flatten()

    for k, idx in enumerate(sel):
        E_val, b = keys[idx]
        gn = test_n[(test_n["E"] == E_val) & (test_n["branch_local_id"] == b)].sort_values("s")
        gs = df_s[(df_s["E"] == E_val) & (df_s["branch_local_id"] == b)].sort_values("s")
        s = gn["s"].to_numpy()
        ta_t = gn["Ta_true"].to_numpy()
        ta_n = gn["Ta_neptune_pro"].to_numpy(); sig = gn["sigma_ep"].to_numpy()
        ax = axes[k]
        ax.fill_between(s, ta_n - sig, ta_n + sig, alpha=0.20, color=COLORS["sigma"])
        ax.plot(s, ta_t, color=COLORS["true"], lw=1.8, label=r"$T_a^{\rm true}$")
        ax.plot(s, ta_n, color=COLORS["NEPTUNE_PRO"], lw=1.5, label="NEPTUNE_PRO")
        if len(gs) > 0:
            ax.plot(s, gs["Ta_ssst_pro"].to_numpy(), color=COLORS["SSST_PRO"],
                    lw=1.2, ls="--", label="SSST_PRO")
        ax.set_title(f"E={E_val:.2g}  b={int(b)}  log10E={np.log10(max(E_val,1e-30)):.1f}",
                     fontsize=9)
        ax.set_xlabel("s", fontsize=9); ax.set_ylabel(r"$T_a$_norm", fontsize=9)
        if k == 0: ax.legend(fontsize=7)
    for j in range(len(sel), len(axes)): axes[j].axis("off")
    fig.suptitle(r"Example test branches: $T_a$_true vs PRO models", y=1.00)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_mae_vs_logE(df_n: pd.DataFrame, df_s: pd.DataFrame, out: Path):
    test_n = df_n[df_n["split"] == "test"].copy()
    test_s = df_s[df_s["split"] == "test"].copy()
    g_n = (test_n.assign(err=lambda x: (x["Ta_true"] - x["Ta_neptune_pro"]).abs())
                  .groupby(["E", "log10E", "branch_local_id"])["err"].mean().reset_index())
    g_s = (test_s.assign(err=lambda x: (x["Ta_true"] - x["Ta_ssst_pro"]).abs())
                  .groupby(["E", "log10E", "branch_local_id"])["err"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.scatter(g_n["log10E"], g_n["err"], s=24, alpha=0.85,
               color=COLORS["NEPTUNE_PRO"], label="NEPTUNE_PRO")
    ax.scatter(g_s["log10E"], g_s["err"], s=22, alpha=0.7,
               color=COLORS["SSST_PRO"], marker="x", label="SSST_PRO")
    ax.set_xlabel(r"$\log_{10} E$"); ax.set_ylabel("MAE per branch (test)")
    ax.set_yscale("log")
    ax.set_title("Per-branch MAE vs elasticity number — code7 PRO models")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_uncertainty(df_n: pd.DataFrame, out: Path):
    test = df_n[df_n["split"] == "test"].copy()
    err = (test["Ta_true"] - test["Ta_neptune_pro"]).abs()
    sig = test["sigma_ep"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(sig, err, s=4, alpha=0.10, color=COLORS["NEPTUNE_PRO"])
    bins = np.linspace(0, sig.quantile(0.99), 12)
    digit = np.digitize(sig, bins); means = []
    for i in range(1, len(bins)):
        m = digit == i
        means.append((bins[i - 1], err[m].mean() if m.any() else np.nan))
    bx, by = zip(*means)
    ax.plot(bx, by, "ko-", lw=1.5, label="binned mean |error|")
    lim = max(sig.max(), err.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, label="ideal calibration")
    ax.set_xlabel(r"NEPTUNE_PRO $\sigma_{\rm ep}$ per point")
    ax.set_ylabel("|error| per point")
    ax.set_title("Predictive uncertainty calibration on test split")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    log = get_logger("pro_figs")
    inf_dir = ROOT.parent / "data" / "runs_pro" / "inference_pro"
    df_n = pd.read_csv(inf_dir / "canonical_neptune_pro.csv")
    df_s = pd.read_csv(inf_dir / "canonical_ssst_pro.csv")
    m_n = pd.read_csv(inf_dir / "metrics_neptune_pro.csv")
    m_s = pd.read_csv(inf_dir / "metrics_ssst_pro.csv")
    m_n["model"] = "NEPTUNE_PRO"; m_s["model"] = "SSST_PRO"
    df_combined = pd.concat([m_n, m_s], ignore_index=True)
    df_combined = df_combined.sort_values(["split", "model"]).reset_index(drop=True)

    out_dir = ROOT.parent / "data" / "runs_pro" / "figures_pro"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_metrics_bar(df_combined, out_dir / "fig_metrics_bar.png");          log.info("  fig_metrics_bar.png")
    fig_curves_examples(df_n, df_s, out_dir / "fig_curves_examples.png");   log.info("  fig_curves_examples.png")
    fig_mae_vs_logE(df_n, df_s, out_dir / "fig_mae_vs_logE.png");           log.info("  fig_mae_vs_logE.png")
    fig_uncertainty(df_n, out_dir / "fig_uncertainty.png");                 log.info("  fig_uncertainty.png")

    df_combined.to_csv(out_dir / "benchmark_pro.csv", index=False)
    log.info(f"saved benchmark_pro.csv -> {out_dir/'benchmark_pro.csv'}")
    print()
    print(df_combined.to_string(index=False))


if __name__ == "__main__":
    main()
