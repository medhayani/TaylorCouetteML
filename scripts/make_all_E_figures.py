"""Render one figure per unique E value.

For every E in the original Input, we plot all branches at that E together
(usually 1-2) overlaid with the 4 model predictions and detected kink
markers. y-axis always starts at 0 (top = 1.1 * local max) so cusps and
peaks remain visible regardless of the absolute scale of Ta.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.utils import detect_kinks_np, selective_smooth

from inference_and_figures import (
    STATIC_FEATURES, build_branch_table, load_model, predict, COL, LS,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 160, "figure.dpi": 100,
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
    siren, mu_s, _ = load_model("siren",          runs / "siren"          / "best.pt", ctx_dim)
    deepo, _,    _ = load_model("deeponet",       runs / "deeponet"       / "best.pt", ctx_dim)
    cheb,  _,    _ = load_model("chebyshev",      runs / "chebyshev"      / "best.pt", ctx_dim)
    env,   _,    _ = load_model("envelope_siren", runs / "envelope_siren" / "best.pt", ctx_dim)
    mu = mu_s
    sd = np.asarray(torch.load(runs / "siren" / "best.pt",
                                  map_location="cpu", weights_only=False)["ctx_std"],
                     dtype=np.float32)

    # Group items by E value
    by_E = defaultdict(list)
    for it in items:
        by_E[float(it["E"])].append(it)
    E_values = sorted(by_E.keys())
    print(f"{len(E_values)} unique E values, {len(items)} total branches.")

    out_dir = ROOT / "figures" / "curves_all_E"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rank, E_val in enumerate(sorted(E_values, key=lambda e: np.log10(max(e, 1e-30))), start=1):
        branches = sorted(by_E[E_val], key=lambda it: it["k_left"])
        log10E = np.log10(max(E_val, 1e-30))

        # Compute predictions for every branch at this E
        all_top = 0.0
        per_branch_preds = []
        s_args = dict(window=7, polyorder=2, edge=3, threshold_z=2.5, transition=2)
        for it in branches:
            ctx_norm = np.clip(np.nan_to_num((it["ctx_raw"] - mu) / sd,
                                                nan=0.0, posinf=5.0, neginf=-5.0),
                                -5.0, 5.0).astype(np.float32)
            amp = max(it["Ta_max"] - it["Ta_min"], 1e-12)
            Ta_min = it["Ta_min"]
            ks = it["k_norm"]
            pred_s = Ta_min + amp * predict(siren, ks, ctx_norm)
            pred_d = Ta_min + amp * predict(deepo, ks, ctx_norm)
            pred_c = Ta_min + amp * predict(cheb,  ks, ctx_norm)
            pred_e = Ta_min + amp * predict(env,   ks, ctx_norm)
            # Selective smoothing: smooth between kinks, preserve cusps.
            ps_s = selective_smooth(pred_s, it["ta_norm"], **s_args)
            ps_d = selective_smooth(pred_d, it["ta_norm"], **s_args)
            ps_c = selective_smooth(pred_c, it["ta_norm"], **s_args)
            ps_e = selective_smooth(pred_e, it["ta_norm"], **s_args)
            kink_idx = detect_kinks_np(it["ta_norm"], threshold_z=2.5)
            per_branch_preds.append({
                "kp": it["k_phys_grid"], "true": it["ta_phys_grid"],
                "siren": ps_s, "deepo": ps_d, "cheb": ps_c, "env": ps_e,
                "kinks": kink_idx, "b": it["branch_local_id"],
            })
            local_top = float(max(it["ta_phys_grid"].max(), ps_s.max(),
                                    ps_d.max(), ps_c.max(), ps_e.max()))
            all_top = max(all_top, local_top)

        fig, ax = plt.subplots(1, 1, figsize=(9, 5.0))
        # Plot kink bars first (lowest zorder), then curves
        for p in per_branch_preds:
            for ki in p["kinks"]:
                ax.axvline(p["kp"][ki], color="#888888", lw=0.5, ls=":",
                           alpha=0.55, zorder=0)
        # Show legend only for first branch (consistent labels)
        for j, p in enumerate(per_branch_preds):
            lbl = lambda name: name if j == 0 else None
            ax.plot(p["kp"], p["true"],   color=COL["Input"],         lw=2.2,
                     label=lbl("Input (true)"))
            ax.plot(p["kp"], p["siren"],  color=COL["SIREN"],         lw=1.4,
                     ls=LS["SIREN"], label=lbl("SIREN"))
            ax.plot(p["kp"], p["deepo"],  color=COL["DeepONet"],      lw=1.4,
                     ls=LS["DeepONet"], label=lbl("DeepONet"))
            ax.plot(p["kp"], p["cheb"],   color=COL["Chebyshev"],     lw=1.4,
                     ls=LS["Chebyshev"], label=lbl("Chebyshev"))
            ax.plot(p["kp"], p["env"],    color=COL["EnvelopeSIREN"], lw=1.8,
                     ls=LS["EnvelopeSIREN"],
                     label=lbl("Envelope-SIREN (M=4)"))

        ax.set_ylim(0.0, all_top * 1.1)
        kp_min = min(p["kp"].min() for p in per_branch_preds)
        kp_max = max(p["kp"].max() for p in per_branch_preds)
        ax.set_xlim(kp_min, kp_max)
        n_kinks_total = sum(len(p["kinks"]) for p in per_branch_preds)
        ax.set_title(rf"$E={E_val:.4g}$, $\log_{{10}}E={log10E:+.2f}$, "
                     rf"branches={len(branches)}, "
                     rf"$n_{{\mathrm{{kinks}}}}={n_kinks_total}$",
                     fontsize=10)
        ax.set_xlabel(r"$k$  (physical)")
        ax.set_ylabel(r"$T_a$  (physical)")
        ax.legend(loc="best", ncol=2, framealpha=0.9)
        fig.tight_layout()

        fname = f"E_{rank:03d}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        if rank % 25 == 0 or rank == 1 or rank == len(E_values):
            print(f"  [{rank:3d}/{len(E_values)}] E={E_val:.4g}  "
                   f"log10E={log10E:+.2f}  branches={len(branches)}  -> {fname}")

    print(f"\nSaved {len(E_values)} figures -> {out_dir}")


if __name__ == "__main__":
    main()
