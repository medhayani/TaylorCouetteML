"""Generate one large prediction figure PER E value.

Each figure shows Ta(k) on one panel with all 6 trained models overlaid:
SIREN, DeepONet, Chebyshev, Envelope-SIREN, SSST, NEPTUNE.

Output: figures/per_E/E_<value>.png   (one PNG per E in --Es)
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

from data_pipeline.dataset import NeptuneProfileDataset
from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.sparse_moe_transformer.ssst_model import SSSTProSurrogate
from models.fno_latent_diffusion.trainer import NeptuneProSurrogate


PRECISION_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Es", default="0.0001,0.001,0.01,0.1,1,10",
                    help="Comma-separated E values")
    ap.add_argument("--n_pred", type=int, default=201)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = ROOT / "data" / "runs"
    out_dir = ROOT / "figures" / "per_E"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(np.clip(desc["E"].astype(float), 1e-30, None))

    # Precision-model normalization stats (saved at training time)
    prec_norm = np.load(runs / "precision" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6

    # match user-provided E to nearest available
    all_Es = sorted(desc["E"].unique())
    requested = [float(x) for x in args.Es.split(",") if x.strip()]
    Es = [all_Es[np.argmin(np.abs(np.array(all_Es) - r))] for r in requested]
    print(f"Resolved E: {Es}")

    # input curve data
    inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                       names=["Ta", "k", "E"], header=None, skiprows=1)

    # ---- profile dataset for SSST/NEPTUNE ----
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    # ---- models ----
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))

    cfg_ssst = full_cfg["ssst_pro"]; cfg_ssst["ctx_dim"] = train_ds.in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    ck = torch.load(runs / "ssst" / "best.pt", map_location=device, weights_only=False)
    ssst.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); ssst.eval()

    cfg_n = full_cfg["neptune_pro"]; cfg_n["context"]["in_dim"] = train_ds.in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    ck = torch.load(runs / "neptune" / "best_member" / "best.pt",
                    map_location=device, weights_only=False)
    neptune.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck); neptune.eval()

    siren = SIRENRegressor(ctx_dim=23).to(device)
    siren.load_state_dict(torch.load(runs / "precision" / "siren" / "best.pt",
                                          map_location=device, weights_only=False)["state_dict"])
    siren.eval()
    deeponet = DeepONet(ctx_dim=23).to(device)
    deeponet.load_state_dict(torch.load(runs / "precision" / "deeponet" / "best.pt",
                                              map_location=device, weights_only=False)["state_dict"])
    deeponet.eval()
    cheby = ChebyshevSpectralRegressor(ctx_dim=23).to(device)
    cheby.load_state_dict(torch.load(runs / "precision" / "chebyshev" / "best.pt",
                                           map_location=device, weights_only=False)["state_dict"])
    cheby.eval()
    env = MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=4).to(device)
    env.load_state_dict(torch.load(runs / "precision" / "envelope_siren" / "best.pt",
                                          map_location=device, weights_only=False)["state_dict"])
    env.eval()
    print("All models loaded")

    colors = {
        "SIREN":         ("tab:blue",   "-",  1.6),
        "DeepONet":      ("tab:green",  "-",  1.6),
        "Chebyshev":     ("tab:purple", "-",  1.6),
        "EnvelopeSIREN": ("tab:orange", "-",  1.6),
        "SSST":          ("tab:red",    "-",  1.6),
        "NEPTUNE":       ("tab:brown",  "--", 1.4),
    }

    for E in Es:
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        if len(sub_desc) == 0:
            print(f"  E={E:g} not in dataset; skip")
            continue
        row = sub_desc.iloc[0]
        b_id = int(row["branch_local_id"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        amp = max(Ta_max - Ta_min, 1e-12)

        s_grid = np.linspace(0.0, 1.0, args.n_pred, dtype=np.float32)
        k_grid = k_left + s_grid * (k_right - k_left)
        k_norm = 2.0 * s_grid - 1.0

        fig, ax = plt.subplots(figsize=(10, 7), dpi=130)

        # ground truth
        mask = (np.isclose(inp["E"], E) & (inp["k"] >= k_left - 1e-6)
                 & (inp["k"] <= k_right + 1e-6))
        gt = inp[mask].sort_values("k")
        if len(gt) >= 5:
            ax.plot(gt["k"], gt["Ta"], "k.", ms=6, label="data (ground truth)", zorder=10)

        # precision models
        ctx_p = []
        for col in PRECISION_FEATURES:
            v = row.get(col, np.nan)
            try: ctx_p.append(float(v))
            except: ctx_p.append(0.0)
        ctx_p_arr = np.asarray(ctx_p, dtype=np.float32)
        ctx_p_arr = (ctx_p_arr - prec_ctx_mean) / prec_ctx_std
        ctx_p_arr = np.clip(np.nan_to_num(ctx_p_arr, nan=0.0,
                                              posinf=0.0, neginf=0.0), -5.0, 5.0)
        ctx_p = torch.from_numpy(ctx_p_arr).unsqueeze(0).to(device)
        k_t = torch.tensor(k_norm, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            for name, m in [("SIREN", siren), ("DeepONet", deeponet),
                              ("Chebyshev", cheby), ("EnvelopeSIREN", env)]:
                ta = m(k_t, ctx_p).squeeze(0).cpu().numpy()
                Ta = Ta_min + ta * amp
                c, ls, lw = colors[name]
                ax.plot(k_grid, Ta, ls=ls, lw=lw, color=c, label=name)

        # SSST + NEPTUNE
        if (E, b_id) in keys_to_idx:
            i = keys_to_idx[(E, b_id)]
            ctx_n = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
            s_n = torch.from_numpy(s_grid).unsqueeze(0).to(device)
            with torch.no_grad():
                ta_ssst = ssst.predict(ctx_n, s_n).squeeze(0).cpu().numpy()
                Ta_ssst = Ta_min + ta_ssst * amp
                c, ls, lw = colors["SSST"]
                ax.plot(k_grid, Ta_ssst, ls=ls, lw=lw, color=c, label="SSST")
                ta_n = neptune.sample(ctx_n, s_n,
                                        num_steps=cfg_n["diffusion"]["inference"]["num_steps"])
                ta_n = ta_n.squeeze(0).cpu().numpy()
                Ta_n = Ta_min + ta_n * amp
                c, ls, lw = colors["NEPTUNE"]
                ax.plot(k_grid, Ta_n, ls=ls, lw=lw, color=c, label="NEPTUNE")

        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel("Ta", fontsize=12)
        ax.set_title(f"Ta(k) at E = {E:.4g}   (branch {b_id})\n"
                       f"k in [{k_left:.2f}, {k_right:.2f}]   "
                       f"Ta in [{Ta_min:.1f}, {Ta_max:.1f}]",
                       fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)
        fig.tight_layout()
        out_png = out_dir / f"E_{E:g}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_png.name}")


if __name__ == "__main__":
    main()
