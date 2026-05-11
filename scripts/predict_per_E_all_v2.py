"""Per-E predictions using ALL V2 models (1000 epochs each).

14 models in total:
- 12 precision V2  (3 seeds x 4 archis from data/runs/precision_v2/)
- 1 SSST V2        (data/runs/ssst_v2/best.pt)
- 1 NEPTUNE V2     (data/runs/neptune_v2/member_00/best.pt)

Each model predicts Ta on every branch of the requested E values. Per-branch
Savitzky-Golay smoothing is applied so the cusps at branch boundaries stay
sharp. The median across the 14 models is drawn in bold black.

Output: figures/per_E_all_v2/E_<idx>_<E>.png  (default 100 figures, k in [0, 20])
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
from scipy.signal import savgol_filter

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


def smooth(y, window=31, poly=3):
    n = len(y)
    if n < window: return y
    if window % 2 == 0: window += 1
    return savgol_filter(y, window_length=window, polyorder=poly, mode="nearest")


def load_v2_precision(runs_v2, ctx_dim, device):
    seed_dirs = sorted([d.name for d in runs_v2.iterdir()
                          if d.is_dir() and d.name.startswith("seed_")])
    seed_models = {}
    for sd in seed_dirs:
        sdir = runs_v2 / sd
        ms = {}
        ms["SIREN"] = SIRENRegressor(ctx_dim=ctx_dim, hidden=384, depth=8).to(device)
        ms["SIREN"].load_state_dict(torch.load(sdir / "siren" / "best.pt",
                                                    map_location=device, weights_only=False)["state_dict"])
        ms["SIREN"].eval()
        ms["DeepONet"] = DeepONet(ctx_dim=ctx_dim,
                                       branch_layers=(384, 384, 384, 384),
                                       trunk_layers=(384, 384, 384, 384),
                                       latent_dim=192, fourier_bands=24).to(device)
        ms["DeepONet"].load_state_dict(torch.load(sdir / "deeponet" / "best.pt",
                                                       map_location=device, weights_only=False)["state_dict"])
        ms["DeepONet"].eval()
        ms["Chebyshev"] = ChebyshevSpectralRegressor(ctx_dim=ctx_dim, n_modes=32).to(device)
        ms["Chebyshev"].load_state_dict(torch.load(sdir / "chebyshev" / "best.pt",
                                                        map_location=device, weights_only=False)["state_dict"])
        ms["Chebyshev"].eval()
        ms["EnvelopeSIREN"] = MultiModeEnvelopeSIREN(ctx_dim=ctx_dim, n_modes=8,
                                                              hidden=320, depth=7).to(device)
        ms["EnvelopeSIREN"].load_state_dict(torch.load(sdir / "envelope_siren" / "best.pt",
                                                                 map_location=device, weights_only=False)["state_dict"])
        ms["EnvelopeSIREN"].eval()
        seed_models[sd] = ms
    return seed_models, seed_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=100)
    ap.add_argument("--n_pred", type=int, default=151)
    ap.add_argument("--smooth_window", type=int, default=31)
    ap.add_argument("--neptune_steps", type=int, default=200)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = ROOT / "data" / "runs"
    out_dir = ROOT / "figures" / "per_E_all_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(np.clip(desc["E"].astype(float), 1e-30, None))

    all_Es = sorted(desc["E"].unique())
    log_min = np.log10(max(min(all_Es), 1e-4))
    log_max = np.log10(min(max(all_Es), 10.0))
    log_picks = np.linspace(log_min, log_max, args.n_figs)
    Es, seen = [], set()
    for lp in log_picks:
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - lp)))]
        if nearest not in seen:
            Es.append(nearest); seen.add(nearest)
    print(f"Producing {len(Es)} figures")

    inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                       names=["Ta", "k", "E"], header=None, skiprows=1)

    # Precision V2 ensemble (12 models)
    prec_norm = np.load(runs / "precision_v2" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6
    seed_models, seed_dirs = load_v2_precision(runs / "precision_v2", ctx_dim=23,
                                                  device=device)
    print(f"Loaded precision V2: {seed_dirs}")

    # SSST V2 + NEPTUNE V2 (need profile dataset for normalized ctx)
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))

    cfg_ssst = full_cfg["ssst_pro"]; cfg_ssst["ctx_dim"] = train_ds.in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    state_s = torch.load(runs / "ssst_v2" / "best.pt", map_location=device, weights_only=False)
    ssst.load_state_dict(state_s["state_dict"] if "state_dict" in state_s else state_s)
    ssst.eval()
    print("Loaded SSST V2")

    cfg_n = full_cfg["neptune_pro"]; cfg_n["context"]["in_dim"] = train_ds.in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    state_n = torch.load(runs / "neptune_v2" / "member_00" / "best.pt",
                            map_location=device, weights_only=False)
    neptune.load_state_dict(state_n["state_dict"] if "state_dict" in state_n else state_n)
    neptune.eval()
    print("Loaded NEPTUNE V2")

    light_colors = {
        "SIREN": "tab:blue", "DeepONet": "tab:green",
        "Chebyshev": "tab:purple", "EnvelopeSIREN": "tab:orange",
        "SSST": "tab:red", "NEPTUNE": "tab:brown",
    }

    for fig_idx, E in enumerate(Es):
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        if len(sub_desc) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        gt = inp[np.isclose(inp["E"], E)].sort_values("k")
        if len(gt) >= 5:
            ax.plot(gt["k"], gt["Ta"], "k.", ms=5, label="data", zorder=20)

        for _, row in sub_desc.iterrows():
            b_id = int(row["branch_local_id"])
            Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
            k_left, k_right = float(row["k_left"]), float(row["k_right"])
            amp = max(Ta_max - Ta_min, 1e-12)
            s_grid = np.linspace(0.0, 1.0, args.n_pred, dtype=np.float32)
            k_grid = k_left + s_grid * (k_right - k_left)
            k_norm = 2.0 * s_grid - 1.0

            # Precision ctx (normalized with precision stats)
            ctx_p = []
            for col in PRECISION_FEATURES:
                v = row.get(col, np.nan)
                try: ctx_p.append(float(v))
                except (TypeError, ValueError): ctx_p.append(0.0)
            ctx_p_arr = np.asarray(ctx_p, dtype=np.float32)
            ctx_p_arr = (ctx_p_arr - prec_ctx_mean) / prec_ctx_std
            ctx_p_arr = np.clip(np.nan_to_num(ctx_p_arr, nan=0.0,
                                                  posinf=0.0, neginf=0.0), -5.0, 5.0)
            ctx_t = torch.from_numpy(ctx_p_arr).unsqueeze(0).to(device)
            k_t = torch.tensor(k_norm, dtype=torch.float32, device=device).unsqueeze(0)

            branch_curves = []
            with torch.no_grad():
                # 12 precision V2 predictions
                for sd, models in seed_models.items():
                    for name, m in models.items():
                        ta_n = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                        Ta = Ta_min + ta_n * amp
                        Ta_s = smooth(Ta, window=args.smooth_window)
                        branch_curves.append(Ta_s)
                        ax.plot(k_grid, Ta_s, "-", lw=0.7, color=light_colors[name],
                                  alpha=0.20, zorder=2)
                # SSST V2 + NEPTUNE V2
                if (E, b_id) in keys_to_idx:
                    i = keys_to_idx[(E, b_id)]
                    ctx_n_t = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
                    s_n_t = torch.from_numpy(s_grid).unsqueeze(0).to(device)
                    ta_ssst = ssst.predict(ctx_n_t, s_n_t).squeeze(0).cpu().numpy()
                    Ta_ssst = Ta_min + ta_ssst * amp
                    Ta_ssst_s = smooth(Ta_ssst, window=args.smooth_window)
                    branch_curves.append(Ta_ssst_s)
                    ax.plot(k_grid, Ta_ssst_s, "-", lw=0.8, color=light_colors["SSST"],
                              alpha=0.35, zorder=2)
                    ta_nept = neptune.sample(ctx_n_t, s_n_t, num_steps=args.neptune_steps)
                    ta_nept = ta_nept.squeeze(0).cpu().numpy()
                    Ta_nept = Ta_min + ta_nept * amp
                    Ta_nept_s = smooth(Ta_nept, window=args.smooth_window)
                    branch_curves.append(Ta_nept_s)
                    ax.plot(k_grid, Ta_nept_s, "--", lw=0.8, color=light_colors["NEPTUNE"],
                              alpha=0.30, zorder=2)

            if branch_curves:
                stack = np.stack(branch_curves, axis=0)
                med = np.median(stack, axis=0)
                ax.plot(k_grid, med, "-", lw=2.6, color="black", zorder=15)

        n_models = 12 + (2 if (E, int(sub_desc.iloc[0]["branch_local_id"])) in keys_to_idx else 0)
        ax.plot([], [], "-", lw=2.6, color="black",
                  label=f"ENSEMBLE median ({n_models} models, 1000 ep)")
        for n, c in light_colors.items():
            ax.plot([], [], "-", lw=1.0, color=c, alpha=0.6, label=n)

        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("k", fontsize=12); ax.set_ylabel("Ta", fontsize=12)
        ax.set_title(f"Ta(k) at E = {E:.5g}   ({len(sub_desc)} branch{'es' if len(sub_desc) > 1 else ''})\n"
                       f"ALL V2 models @ 1000 epochs: 12 precision + SSST + NEPTUNE = {n_models}",
                       fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        fig.tight_layout()

        E_str = f"{E:.5g}".replace("+", "").replace(".", "p")
        out_png = out_dir / f"E_{fig_idx:03d}_{E_str}.png"
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        if (fig_idx + 1) % 10 == 0:
            print(f"  ... {fig_idx + 1}/{len(Es)} done")

    print(f"Saved {len(Es)} figures under {out_dir}")


if __name__ == "__main__":
    main()
