"""Save 10 individual per-branch figures (true + 4 model predictions).

Each figure uses ylim=[0, max(Ta) * 1.1] so the cusps/peaks are visible
relative to a zero baseline. Detected kink positions are marked with
dotted vertical bars so the eye lands on the mode-crossing locations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.utils import detect_kinks_np, selective_smooth

# Re-use the inference helpers from the sibling script
sys.path.insert(0, str(ROOT / "scripts"))
from inference_and_figures import (
    STATIC_FEATURES, build_branch_table, load_model, predict, COL, LS,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 200, "figure.dpi": 130,
})


def main():
    runs = ROOT / "checkpoints"
    items = build_branch_table(
        ROOT / "data" / "Input" / "combined_data.csv",
        ROOT / "data" / "processed" / "branch_functional_descriptors.csv",
        ROOT / "data" / "processed" / "model_profile_level_dataset.csv",
    )
    print(f"Loaded {len(items)} branches.")

    ctx_dim = len(STATIC_FEATURES)
    siren, mu_s, sd_s = load_model("siren",          runs / "siren"          / "best.pt", ctx_dim)
    deepo, mu_d, sd_d = load_model("deeponet",       runs / "deeponet"       / "best.pt", ctx_dim)
    cheb,  mu_c, sd_c = load_model("chebyshev",      runs / "chebyshev"      / "best.pt", ctx_dim)
    env,   mu_e, sd_e = load_model("envelope_siren", runs / "envelope_siren" / "best.pt", ctx_dim)
    mu, sd = mu_s, sd_s

    # Pick 10 branches: one per quantile of log10(E), preferring branches
    # that contain at least one detected kink so the viewer sees the
    # phenomenon the figure is meant to illustrate.
    log10Es = np.array([it["log10E"] for it in items])
    n_kinks = np.array([len(detect_kinks_np(it["ta_norm"], threshold_z=2.5))
                         for it in items])
    quantile_targets = np.linspace(log10Es.min(), log10Es.max(), 10)
    picked = []
    used = set()
    for tgt in quantile_targets:
        # Order candidates by closeness to target log10E, then prefer branches with kinks
        order = np.argsort(np.abs(log10Es - tgt))
        for idx in order[:30]:                      # search 30 nearest
            if idx in used: continue
            if n_kinks[idx] >= 1:
                picked.append(int(idx)); used.add(int(idx)); break
        else:
            for idx in order:
                if idx not in used:
                    picked.append(int(idx)); used.add(int(idx)); break

    out_dir = ROOT / "figures" / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    article_dir = ROOT.parent / "TaylorCouetteML" / "docs" / "figures" / "curves"
    if article_dir.parent.exists():
        article_dir.mkdir(parents=True, exist_ok=True)

    for rank, idx in enumerate(picked, start=1):
        it = items[idx]
        ctx_norm = np.clip(np.nan_to_num((it["ctx_raw"] - mu) / sd,
                                            nan=0.0, posinf=5.0, neginf=-5.0),
                            -5.0, 5.0).astype(np.float32)
        ks = it["k_norm"]
        amp = max(it["Ta_max"] - it["Ta_min"], 1e-12)
        Ta_min = it["Ta_min"]
        kp = it["k_phys_grid"]
        true_phys = it["ta_phys_grid"]

        pred_s = Ta_min + amp * predict(siren, ks, ctx_norm)
        pred_d = Ta_min + amp * predict(deepo, ks, ctx_norm)
        pred_c = Ta_min + amp * predict(cheb,  ks, ctx_norm)
        pred_e = Ta_min + amp * predict(env,   ks, ctx_norm)

        kink_idx = detect_kinks_np(it["ta_norm"], threshold_z=2.5)

        # Selective smoothing: smooth between kinks, preserve cusps.
        s_args = dict(window=7, polyorder=2, edge=3, threshold_z=2.5, transition=2)
        ps_s = selective_smooth(pred_s, it["ta_norm"], **s_args)
        ps_d = selective_smooth(pred_d, it["ta_norm"], **s_args)
        ps_c = selective_smooth(pred_c, it["ta_norm"], **s_args)
        ps_e = selective_smooth(pred_e, it["ta_norm"], **s_args)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
        for ki in kink_idx:
            ax.axvline(kp[ki], color="#888888", lw=0.6, ls=":",
                       alpha=0.6, zorder=0)
        ax.plot(kp, true_phys,
                color=COL["Input"], lw=2.4, label="Input (true)")
        ax.plot(kp, ps_s,
                color=COL["SIREN"], lw=1.6, ls=LS["SIREN"], label="SIREN")
        ax.plot(kp, ps_d,
                color=COL["DeepONet"], lw=1.6, ls=LS["DeepONet"], label="DeepONet")
        ax.plot(kp, ps_c,
                color=COL["Chebyshev"], lw=1.6, ls=LS["Chebyshev"], label="Chebyshev")
        ax.plot(kp, ps_e,
                color=COL["EnvelopeSIREN"], lw=2.0, ls=LS["EnvelopeSIREN"],
                label="Envelope-SIREN (M=4)")

        # Force y-axis to start at 0; upper bound = 1.1 * max over all curves
        y_top = float(max(true_phys.max(), ps_s.max(), ps_d.max(),
                            ps_c.max(), ps_e.max()) * 1.1)
        ax.set_ylim(0.0, y_top)
        ax.set_xlim(kp.min(), kp.max())

        ax.set_title(rf"Branch #{rank}/10 — $E={it['E']:.3g}$, "
                     rf"branch_local_id={it['branch_local_id']}, "
                     rf"$\log_{{10}}E={it['log10E']:.2f}$, "
                     rf"$n_{{\mathrm{{kinks}}}}={len(kink_idx)}$")
        ax.set_xlabel(r"$k$  (physical)")
        ax.set_ylabel(r"$T_a$  (physical)")
        ax.legend(loc="best", ncol=2, framealpha=0.9)
        fig.tight_layout()

        fname = f"branch_{rank:02d}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        if article_dir.parent.exists():
            (article_dir / fname).write_bytes((out_dir / fname).read_bytes())
        plt.close(fig)
        print(f"  [{rank:02d}/10] E={it['E']:.3g}  "
              f"log10E={it['log10E']:+.2f}  kinks={len(kink_idx)}  "
              f"Ta in [0, {y_top:.2f}]   -> {fname}")

    print(f"\nSaved 10 figures -> {out_dir}")
    if article_dir.parent.exists():
        print(f"Mirror copied  -> {article_dir}")


if __name__ == "__main__":
    main()
