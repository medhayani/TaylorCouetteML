"""Per-E predictions using the V2 ensemble (3 seeds x 4 architectures = 12 models).

Loads every checkpoint under data/runs/precision_v2/seed_<s>/<model>/best.pt,
computes a smoothed prediction per branch for each, and overlays the median
of the 12 predictions in bold. Individual models stay light.

Output: figures/per_E_v2/E_<idx>_<E>.png  (default 100 figures)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN


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


def load_seed_models(runs_v2: Path, seed_dir_name: str, ctx_dim: int, device):
    seed_dir = runs_v2 / seed_dir_name
    out = {}
    # SIREN bigger
    siren = SIRENRegressor(ctx_dim=ctx_dim, hidden=384, depth=8).to(device)
    siren.load_state_dict(torch.load(seed_dir / "siren" / "best.pt",
                                          map_location=device, weights_only=False)["state_dict"])
    siren.eval(); out["SIREN"] = siren
    # DeepONet bigger
    deeponet = DeepONet(ctx_dim=ctx_dim,
                          branch_layers=(384, 384, 384, 384),
                          trunk_layers=(384, 384, 384, 384),
                          latent_dim=192, fourier_bands=24).to(device)
    deeponet.load_state_dict(torch.load(seed_dir / "deeponet" / "best.pt",
                                              map_location=device, weights_only=False)["state_dict"])
    deeponet.eval(); out["DeepONet"] = deeponet
    # Chebyshev
    cheb = ChebyshevSpectralRegressor(ctx_dim=ctx_dim, n_modes=32).to(device)
    cheb.load_state_dict(torch.load(seed_dir / "chebyshev" / "best.pt",
                                          map_location=device, weights_only=False)["state_dict"])
    cheb.eval(); out["Chebyshev"] = cheb
    # Envelope-SIREN
    env = MultiModeEnvelopeSIREN(ctx_dim=ctx_dim, n_modes=8, hidden=320, depth=7).to(device)
    env.load_state_dict(torch.load(seed_dir / "envelope_siren" / "best.pt",
                                          map_location=device, weights_only=False)["state_dict"])
    env.eval(); out["EnvelopeSIREN"] = env
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=100)
    ap.add_argument("--n_pred", type=int, default=151)
    ap.add_argument("--smooth_window", type=int, default=31)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_v2 = ROOT / "data" / "runs" / "precision_v2"
    out_dir = ROOT / "figures" / "per_E_v2"
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

    prec_norm = np.load(runs_v2 / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6

    # discover available seed directories
    seed_dirs = sorted([d.name for d in runs_v2.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    print(f"Found seeds: {seed_dirs}")

    ctx_dim = 23
    seed_models = {sd: load_seed_models(runs_v2, sd, ctx_dim, device) for sd in seed_dirs}

    light_colors = {
        "SIREN": "tab:blue", "DeepONet": "tab:green",
        "Chebyshev": "tab:purple", "EnvelopeSIREN": "tab:orange",
    }

    for fig_idx, E in enumerate(Es):
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        if len(sub_desc) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        gt = inp[np.isclose(inp["E"], E)].sort_values("k")
        if len(gt) >= 5:
            ax.plot(gt["k"], gt["Ta"], "k.", ms=5, label="data", zorder=20)

        all_curves = []  # (k, Ta) tuples to compute median later
        for _, row in sub_desc.iterrows():
            Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
            k_left, k_right = float(row["k_left"]), float(row["k_right"])
            amp = max(Ta_max - Ta_min, 1e-12)
            s_grid = np.linspace(0.0, 1.0, args.n_pred, dtype=np.float32)
            k_grid = k_left + s_grid * (k_right - k_left)
            k_norm = 2.0 * s_grid - 1.0

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
                for sd, models in seed_models.items():
                    for name, m in models.items():
                        ta_n = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                        Ta = Ta_min + ta_n * amp
                        Ta_s = smooth(Ta, window=args.smooth_window)
                        branch_curves.append(Ta_s)
                        c = light_colors[name]
                        ax.plot(k_grid, Ta_s, "-", lw=0.8, color=c, alpha=0.25, zorder=2)

            if branch_curves:
                stack = np.stack(branch_curves, axis=0)
                med = np.median(stack, axis=0)
                ax.plot(k_grid, med, "-", lw=2.6, color="black", zorder=15)
                all_curves.append((k_grid, med))

        # Manually add legend handles
        ax.plot([], [], "-", lw=2.6, color="black",
                  label=f"ENSEMBLE median ({len(seed_dirs)} seeds x 4 archis = {len(seed_dirs) * 4} models)")
        for n, c in light_colors.items():
            ax.plot([], [], "-", lw=1.0, color=c, alpha=0.6, label=n)

        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("k", fontsize=12); ax.set_ylabel("Ta", fontsize=12)
        ax.set_title(f"Ta(k) at E = {E:.5g}   ({len(sub_desc)} branch{'es' if len(sub_desc) > 1 else ''})\n"
                       f"V2 ensemble: {len(seed_dirs)} seeds x 4 archis, 1000 epochs",
                       fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.85)
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
