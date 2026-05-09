"""Predict Ta(k) for several E values using all trained models on the same panel.

Models overlaid (one panel per (E, branch)):
  - SIREN, DeepONet, Chebyshev, Envelope-SIREN  (precision family)
  - SSST  (Sparse-MoE Transformer)
  - NEPTUNE  (FNO + Latent Diffusion, best ensemble member)

Defaults to 8 representative E values across log10E in [-4, 1]; override with
--Es "1e-4,5e-3,0.1,3,10".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.dataset import NeptuneProfileDataset, STATIC_FEATURES
from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.sparse_moe_transformer.ssst_model import SSSTProSurrogate
from models.fno_latent_diffusion.trainer import NeptuneProSurrogate


# ---- 23 ctx features used by precision models (matches train_precision_curves.py) ----
PRECISION_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def load_precision(name, ckpt_path, ctx_dim, device):
    cls = {
        "SIREN": SIRENRegressor,
        "DeepONet": DeepONet,
        "Chebyshev": ChebyshevSpectralRegressor,
        "EnvelopeSIREN": (lambda **kw: MultiModeEnvelopeSIREN(ctx_dim=kw["ctx_dim"], n_modes=4)),
    }[name]
    if name == "EnvelopeSIREN":
        m = cls(ctx_dim=ctx_dim).to(device)
    else:
        m = cls(ctx_dim=ctx_dim).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Es", default="", help='Comma-separated E values (e.g. "1e-4,1e-3,0.1,1,10")')
    ap.add_argument("--n_pred", type=int, default=101,
                    help="number of (k, Ta) points predicted per branch")
    ap.add_argument("--out_png", default=str(ROOT / "figures" / "predict_all_models.png"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- Load branch descriptors (gives Ta_min, Ta_max, k_left, k_right per branch) ----
    desc_path = ROOT / "data" / "processed" / "branch_functional_descriptors.csv"
    desc = pd.read_csv(desc_path).copy()
    desc["log10E"] = np.log10(np.clip(desc["E"].astype(float), 1e-30, None))

    # ---- Pick E values ----
    all_Es = sorted(desc["E"].unique())
    if args.Es.strip():
        Es = [float(x) for x in args.Es.split(",") if x.strip()]
    else:
        log_picks = np.linspace(np.log10(all_Es).min(), np.log10(all_Es).max(), 8)
        Es = [all_Es[np.argmin(np.abs(np.log10(all_Es) - lp))] for lp in log_picks]
    print(f"E values: {Es}")

    # ---- Load NEPTUNE-style profile dataset for SSST/NEPTUNE inputs ----
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    # ---- Load all the heavy models ----
    runs = ROOT / "data" / "runs"
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))

    # SSST
    cfg_ssst = full_cfg["ssst_pro"]
    cfg_ssst["ctx_dim"] = train_ds.in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    ck_ssst = torch.load(runs / "ssst" / "best.pt", map_location=device, weights_only=False)
    ssst.load_state_dict(ck_ssst["state_dict"] if "state_dict" in ck_ssst else ck_ssst)
    ssst.eval()
    print("SSST loaded")

    # NEPTUNE (best member)
    cfg_n = full_cfg["neptune_pro"]
    cfg_n["context"]["in_dim"] = train_ds.in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    ck_n = torch.load(runs / "neptune" / "best_member" / "best.pt",
                       map_location=device, weights_only=False)
    neptune.load_state_dict(ck_n["state_dict"] if "state_dict" in ck_n else ck_n)
    neptune.eval()
    print("NEPTUNE loaded")

    # Precision models — built from raw 23 ctx features (NOT NeptuneProfileDataset's ctx)
    prec_models = {}
    prec_ck = {}
    for name in ["SIREN", "DeepONet", "Chebyshev", "EnvelopeSIREN"]:
        ckpt_dir_name = name.lower().replace("envelopesiren", "envelope_siren")
        ckpt_path = runs / "precision" / ckpt_dir_name / "best.pt"
        m, ck = load_precision(name, ckpt_path, ctx_dim=23, device=device)
        prec_models[name] = m
        prec_ck[name] = ck
        print(f"{name} loaded")

    # ---- Predict for each E ----
    figs_dir = Path(args.out_png).parent
    figs_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    for E in Es:
        sub = desc[np.isclose(desc["E"], E)]
        if len(sub) == 0:
            print(f"  E={E:g} not in dataset, skip")
            continue
        # take the first branch for this E
        row = sub.sort_values("branch_local_id").iloc[0]
        selections.append((E, int(row["branch_local_id"]), row))
    print(f"Selected {len(selections)} (E, branch) pairs")

    n = len(selections)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.0 * rows), dpi=110)
    axes = np.atleast_1d(axes).ravel()

    colors = {"SIREN": "tab:blue", "DeepONet": "tab:green",
                "Chebyshev": "tab:purple", "EnvelopeSIREN": "tab:orange",
                "SSST": "tab:red", "NEPTUNE": "tab:brown"}

    for ax, (E, b_id, row) in zip(axes, selections):
        Ta_min = float(row["Ta_min"]); Ta_max = float(row["Ta_max"])
        k_left = float(row["k_left"]); k_right = float(row["k_right"])
        amp = max(Ta_max - Ta_min, 1e-12)

        # Ground truth (from raw Input combined_data.csv if available)
        try:
            inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                                names=["Ta", "k", "E"], header=None, skiprows=1)
            mask = (np.isclose(inp["E"], E) & (inp["k"] >= k_left - 1e-6)
                     & (inp["k"] <= k_right + 1e-6))
            sub = inp[mask].sort_values("k")
            if len(sub) >= 5:
                ax.plot(sub["k"], sub["Ta"], "k.", ms=3, label="data")
        except Exception:
            pass

        # Common s grid for predictions
        s_grid = np.linspace(0.0, 1.0, args.n_pred, dtype=np.float32)
        k_grid = k_left + s_grid * (k_right - k_left)
        k_norm = 2.0 * s_grid - 1.0

        # ---- Precision models ----
        ctx_prec = []
        for col in PRECISION_FEATURES:
            v = row.get(col, np.nan)
            try:
                ctx_prec.append(float(v))
            except (TypeError, ValueError):
                ctx_prec.append(0.0)
        ctx_prec = torch.tensor(ctx_prec, dtype=torch.float32,
                                  device=device).unsqueeze(0)
        k_t = torch.tensor(k_norm, dtype=torch.float32,
                             device=device).unsqueeze(0)
        with torch.no_grad():
            for name, m in prec_models.items():
                ta_norm = m(k_t, ctx_prec).squeeze(0).cpu().numpy()
                Ta = Ta_min + ta_norm * amp
                ax.plot(k_grid, Ta, "-", lw=1.4, color=colors[name],
                          label=name if ax is axes[0] else None)

        # ---- SSST + NEPTUNE need the NeptuneProfileDataset's normalized ctx ----
        if (E, b_id) in keys_to_idx:
            i = keys_to_idx[(E, b_id)]
            ctx_n = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
            s_n = torch.from_numpy(s_grid).unsqueeze(0).to(device)
            with torch.no_grad():
                ta_ssst = ssst.predict(ctx_n, s_n).squeeze(0).cpu().numpy()
                Ta_ssst = Ta_min + ta_ssst * amp
                ax.plot(k_grid, Ta_ssst, "-", lw=1.4, color=colors["SSST"],
                          label="SSST" if ax is axes[0] else None)
                ta_nept = neptune.sample(ctx_n, s_n,
                                            num_steps=cfg_n["diffusion"]["inference"]["num_steps"])
                ta_nept = ta_nept.squeeze(0).cpu().numpy()
                Ta_nept = Ta_min + ta_nept * amp
                ax.plot(k_grid, Ta_nept, "--", lw=1.2, color=colors["NEPTUNE"],
                          label="NEPTUNE" if ax is axes[0] else None)

        ax.set_title(f"E = {E:.4g}   branch = {b_id}", fontsize=10)
        ax.set_xlabel("k"); ax.set_ylabel("Ta")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(selections):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                     bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("Ta(k) predictions — all 6 models, multiple E values\n"
                  "(500-epoch checkpoints, Kaggle P100 GPU)",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {args.out_png}")


if __name__ == "__main__":
    main()
