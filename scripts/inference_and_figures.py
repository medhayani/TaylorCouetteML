"""Test the 3 trained precision-curve models on the original Input.

Outputs:
  - figures/predictions_vs_input.png      12 branches across log10E,
                                           true Ta(k) vs SIREN/DeepONet/Cheb.
  - figures/coeff_diagnostic.png           Chebyshev coefficient magnitudes
                                           for the same 12 branches (log10).
  - figures/metrics.csv                    Per-model physical-MAE and RMSE
                                           on every branch in the Input.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.utils import detect_kinks_np, selective_smooth

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


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]

COL = {
    "SIREN":         "#2ca02c",
    "DeepONet":      "#1f77b4",
    "Chebyshev":     "#d62728",
    "EnvelopeSIREN": "#ff7f0e",
    "Input":         "#000000",
}
LS = {
    "SIREN":         "-",
    "DeepONet":      "--",
    "Chebyshev":     ":",
    "EnvelopeSIREN": "-",
}


def build_branch_table(input_csv: Path, desc_csv: Path, profile_csv: Path,
                        n_resampled: int = 101):
    raw = pd.read_csv(input_csv); raw.columns = ["Ta", "k", "E"]
    desc = pd.read_csv(desc_csv)
    prof = pd.read_csv(profile_csv).replace([np.inf, -np.inf], np.nan)
    prof_first = (prof.sort_values(["E", "branch_local_id", "s"])
                       .groupby(["E", "branch_local_id"], sort=False)
                       .first().reset_index())
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))

    items = []
    for _, row in desc.iterrows():
        E_val, b = float(row["E"]), int(row["branch_local_id"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)

        sub = raw[(np.isclose(raw["E"], E_val))
                   & (raw["k"] >= k_left - 1e-6)
                   & (raw["k"] <= k_right + 1e-6)].sort_values("k")
        if len(sub) < 5: continue

        s_grid = np.linspace(0.0, 1.0, n_resampled)
        k_grid = k_left + s_grid * (k_right - k_left)
        ta_resamp = np.interp(k_grid, sub["k"].to_numpy(), sub["Ta"].to_numpy())
        ta_norm = (ta_resamp - Ta_min) / amp
        k_norm = 2.0 * s_grid - 1.0

        feat_row = prof_first[(np.isclose(prof_first["E"], E_val))
                                 & (prof_first["branch_local_id"] == b)]
        if len(feat_row) == 0: continue
        feats = []
        for c in STATIC_FEATURES:
            try: feats.append(float(feat_row[c].iloc[0]))
            except Exception: feats.append(0.0)
        ctx = np.asarray(feats, dtype=np.float32)
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        items.append({"E": E_val, "branch_local_id": b,
                       "k_left": k_left, "k_right": k_right,
                       "Ta_min": Ta_min, "Ta_max": Ta_max,
                       "k_norm": k_norm.astype(np.float32),
                       "k_phys_grid": k_grid.astype(np.float32),
                       "ta_norm": ta_norm.astype(np.float32),
                       "ta_phys_grid": ta_resamp.astype(np.float32),
                       "ctx_raw": ctx,
                       "log10E": np.log10(max(E_val, 1e-30))})
    return items


def load_model(name: str, ckpt_path: Path, ctx_dim: int):
    if name == "siren":
        m = SIRENRegressor(ctx_dim=ctx_dim)
    elif name == "deeponet":
        m = DeepONet(ctx_dim=ctx_dim)
    elif name == "chebyshev":
        m = ChebyshevSpectralRegressor(ctx_dim=ctx_dim, n_modes=16)
    elif name == "envelope_siren":
        m = MultiModeEnvelopeSIREN(ctx_dim=ctx_dim, n_modes=4)
    else:
        raise ValueError(name)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, np.asarray(ck["ctx_mean"], dtype=np.float32), \
              np.asarray(ck["ctx_std"], dtype=np.float32)


@torch.no_grad()
def predict(model, k_norm_np, ctx_norm_np):
    k = torch.from_numpy(k_norm_np[None, :])
    c = torch.from_numpy(ctx_norm_np[None, :])
    pred = model(k, c).cpu().numpy()[0]
    return pred


def main():
    runs = ROOT / "checkpoints"
    items = build_branch_table(
        ROOT / "data" / "Input" / "combined_data.csv",
        ROOT / "data" / "processed" / "branch_functional_descriptors.csv",
        ROOT / "data" / "processed" / "model_profile_level_dataset.csv",
    )
    print(f"Loaded {len(items)} branches from Input.")

    ctx_dim = len(STATIC_FEATURES)
    siren, mu_s, sd_s = load_model("siren",         runs / "siren"          / "best.pt", ctx_dim)
    deepo, mu_d, sd_d = load_model("deeponet",      runs / "deeponet"       / "best.pt", ctx_dim)
    cheb,  mu_c, sd_c = load_model("chebyshev",     runs / "chebyshev"      / "best.pt", ctx_dim)
    env,   mu_e, sd_e = load_model("envelope_siren", runs / "envelope_siren" / "best.pt", ctx_dim)

    # Use train_ds normalisation (saved in checkpoint). Same for all here.
    mu, sd = mu_s, sd_s

    # ---- Pick 12 representative branches across log10E ----
    log10Es = np.array([it["log10E"] for it in items])
    sel = []
    for tgt in np.linspace(log10Es.min(), log10Es.max(), 12):
        sel.append(int(np.argmin(np.abs(log10Es - tgt))))

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), squeeze=False)
    axes = axes.flatten()
    for i, idx in enumerate(sel):
        it = items[idx]
        ctx_norm = np.clip(np.nan_to_num((it["ctx_raw"] - mu) / sd,
                                            nan=0.0, posinf=5.0, neginf=-5.0),
                            -5.0, 5.0).astype(np.float32)
        ks = it["k_norm"]
        true_n = it["ta_norm"]
        pred_s = predict(siren, ks, ctx_norm)
        pred_d = predict(deepo, ks, ctx_norm)
        pred_c = predict(cheb,  ks, ctx_norm)
        pred_e = predict(env,   ks, ctx_norm)

        amp = max(it["Ta_max"] - it["Ta_min"], 1e-12)
        Ta_min = it["Ta_min"]
        kp = it["k_phys_grid"]
        true_phys = it["ta_phys_grid"]

        # Detect kink positions in the TRUE normalised curve.
        kink_idx = detect_kinks_np(true_n, threshold_z=2.5)

        # Selective smoothing: smooth between kinks, preserve cusps.
        s_args = dict(window=7, polyorder=2, edge=3, threshold_z=2.5, transition=2)
        ps_s = selective_smooth(Ta_min + amp * pred_s, true_n, **s_args)
        ps_d = selective_smooth(Ta_min + amp * pred_d, true_n, **s_args)
        ps_c = selective_smooth(Ta_min + amp * pred_c, true_n, **s_args)
        ps_e = selective_smooth(Ta_min + amp * pred_e, true_n, **s_args)

        ax = axes[i]
        ax.plot(kp, true_phys, color=COL["Input"], lw=2.0, label="Input (true)")
        for ki in kink_idx:
            ax.axvline(kp[ki], color="#888888", lw=0.4, ls=":", alpha=0.55, zorder=0)
        ax.plot(kp, ps_s, color=COL["SIREN"], lw=1.4, ls=LS["SIREN"], label="SIREN")
        ax.plot(kp, ps_d, color=COL["DeepONet"], lw=1.4, ls=LS["DeepONet"], label="DeepONet")
        ax.plot(kp, ps_c, color=COL["Chebyshev"], lw=1.4, ls=LS["Chebyshev"], label="Chebyshev")
        ax.plot(kp, ps_e, color=COL["EnvelopeSIREN"], lw=1.6, ls=LS["EnvelopeSIREN"],
                 label="Envelope-SIREN (M=4)")
        ax.set_title(rf"$E={it['E']:.3g}$, branch={it['branch_local_id']}, "
                     rf"$\log_{{10}}E={it['log10E']:.1f}$, "
                     rf"$n_{{\mathrm{{kinks}}}}={len(kink_idx)}$",
                     fontsize=9)
        ax.set_xlabel("k (physical)"); ax.set_ylabel("Ta (physical)")
        if i == 0:
            ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle("Precision curve models — 12 representative branches "
                  r"(SIREN, DeepONet, Chebyshev, Envelope-SIREN) vs Input. "
                  "Dotted vertical bars = detected mode-crossing kinks",
                  y=1.00)
    fig.tight_layout()
    out_dir = ROOT / "figures"; out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "predictions_vs_input.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_dir/'predictions_vs_input.png'}")

    # Copy to article figures folder
    target = ROOT.parent / "TaylorCouetteML" / "docs" / "figures" / "predictions_vs_input.png"
    if target.parent.exists():
        target.write_bytes((out_dir / "predictions_vs_input.png").read_bytes())
        print(f"Copied to article figures -> {target}")

    # ---- Chebyshev coefficient diagnostic ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    for i, idx in enumerate(sel[:6]):
        it = items[idx]
        ctx_norm = np.clip(np.nan_to_num((it["ctx_raw"] - mu) / sd,
                                            nan=0.0, posinf=5.0, neginf=-5.0),
                            -5.0, 5.0).astype(np.float32)
        with torch.no_grad():
            c = cheb.coefficients(torch.from_numpy(ctx_norm[None, :])).cpu().numpy()[0]
        ax2.semilogy(np.abs(c) + 1e-12, marker="o", lw=1.0,
                     label=fr"$\log_{{10}}E={it['log10E']:.1f}$")
    ax2.set_xlabel("Chebyshev mode n")
    ax2.set_ylabel(r"$|c_n|$")
    ax2.set_title("Chebyshev coefficient decay (selected branches)")
    ax2.legend(fontsize=7, loc="best", ncol=2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "coeff_diagnostic.png", bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved -> {out_dir/'coeff_diagnostic.png'}")

    # ---- Aggregate physical MAE / RMSE on all branches, plus kink-zone MAE ----
    rows = []
    for name, model, mu_m, sd_m in [
        ("SIREN",         siren, mu_s, sd_s),
        ("DeepONet",      deepo, mu_d, sd_d),
        ("Chebyshev",     cheb,  mu_c, sd_c),
        ("EnvelopeSIREN", env,   mu_e, sd_e),
    ]:
        all_err = []
        kink_err = []
        for it in items:
            ctx_norm = np.clip(np.nan_to_num((it["ctx_raw"] - mu_m) / sd_m,
                                                nan=0.0, posinf=5.0, neginf=-5.0),
                                -5.0, 5.0).astype(np.float32)
            pred = predict(model, it["k_norm"], ctx_norm)
            amp = max(it["Ta_max"] - it["Ta_min"], 1e-12)
            ta_pred_phys = it["Ta_min"] + amp * pred
            err = np.abs(it["ta_phys_grid"] - ta_pred_phys)
            all_err.append(err)
            # Kink-zone error: ±2 grid points around each detected kink
            kidx = detect_kinks_np(it["ta_norm"], threshold_z=2.5)
            mask = np.zeros_like(err, dtype=bool)
            for ki in kidx:
                lo, hi = max(0, ki - 2), min(len(err), ki + 3)
                mask[lo:hi] = True
            if mask.any():
                kink_err.append(err[mask])
        flat = np.concatenate(all_err)
        kflat = np.concatenate(kink_err) if kink_err else np.array([np.nan])
        rows.append({"model": name,
                     "MAE_phys": float(np.mean(flat)),
                     "RMSE_phys": float(np.sqrt(np.mean(flat ** 2))),
                     "MaxErr_phys": float(np.max(flat)),
                     "MAE_kink_zone": float(np.mean(kflat)),
                     "n_branches": len(items)})
    metrics = pd.DataFrame(rows).sort_values("MAE_phys")
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
