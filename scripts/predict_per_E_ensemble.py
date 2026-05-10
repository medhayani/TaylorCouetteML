"""Per-E predictions with stronger smoothing and an ensemble-average overlay.

Same per-branch smoothing as predict_per_E_smooth.py, but:
- Stronger Savitzky-Golay (window=31, polyorder=3) on each branch.
- NEPTUNE uses 200 diffusion steps (cleaner than the 4-step default).
- An ENSEMBLE mean of the 5 deterministic models (SIREN, DeepONet, Chebyshev,
  Envelope-SIREN, SSST) is overlaid in bold black, so you see one
  "consensus" curve overlapping the data.
- Individual model curves are drawn light (alpha=0.4) so they don't compete.

Default: 100 figures, k in [0, 20].
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


def smooth_branch(y, window=31, poly=3):
    n = len(y)
    if n < window:
        if n < 5:
            return y
        w = max(5, n - 1 if (n - 1) % 2 else n - 2)
        return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1),
                              mode="nearest")
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=100)
    ap.add_argument("--n_pred", type=int, default=151)
    ap.add_argument("--smooth_window", type=int, default=31)
    ap.add_argument("--smooth_poly", type=int, default=3)
    ap.add_argument("--neptune_steps", type=int, default=200)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = ROOT / "data" / "runs"
    out_dir = ROOT / "figures" / "per_E_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(np.clip(desc["E"].astype(float), 1e-30, None))

    # ---- pick E values uniformly in log10(E) ----
    all_Es = sorted(desc["E"].unique())
    log_min = np.log10(max(min(all_Es), 1e-4))
    log_max = np.log10(min(max(all_Es), 10.0))
    log_picks = np.linspace(log_min, log_max, args.n_figs)
    Es, seen = [], set()
    for lp in log_picks:
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - lp)))]
        if nearest not in seen:
            Es.append(nearest); seen.add(nearest)
    print(f"Producing {len(Es)} figures from E={Es[0]:.4g} to E={Es[-1]:.4g}")

    inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                       names=["Ta", "k", "E"], header=None, skiprows=1)

    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

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

    prec_norm = np.load(runs / "precision" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6
    print("All models loaded")

    light_colors = {
        "SIREN":         ("tab:blue",   0.35),
        "DeepONet":      ("tab:green",  0.35),
        "Chebyshev":     ("tab:purple", 0.35),
        "EnvelopeSIREN": ("tab:orange", 0.35),
        "SSST":          ("tab:red",    0.35),
        "NEPTUNE":       ("tab:brown",  0.30),
    }
    # Models contributing to the ensemble (NEPTUNE excluded — too noisy at inference).
    ENSEMBLE_NAMES = ["SIREN", "DeepONet", "Chebyshev", "EnvelopeSIREN", "SSST"]

    for fig_idx, E in enumerate(Es):
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        if len(sub_desc) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        gt = inp[np.isclose(inp["E"], E)].sort_values("k")
        if len(gt) >= 5:
            ax.plot(gt["k"], gt["Ta"], "k.", ms=5, label="data", zorder=20)

        per_model_pieces = {n: [] for n in
                              ["SIREN", "DeepONet", "Chebyshev", "EnvelopeSIREN",
                               "SSST", "NEPTUNE"]}

        for _, row in sub_desc.iterrows():
            b_id = int(row["branch_local_id"])
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
            with torch.no_grad():
                for name, m in [("SIREN", siren), ("DeepONet", deeponet),
                                  ("Chebyshev", cheby), ("EnvelopeSIREN", env)]:
                    ta_n = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                    Ta = Ta_min + ta_n * amp
                    Ta_s = smooth_branch(Ta, window=args.smooth_window, poly=args.smooth_poly)
                    per_model_pieces[name].append((k_grid, Ta_s))

            if (E, b_id) in keys_to_idx:
                i = keys_to_idx[(E, b_id)]
                ctx_n = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
                s_n = torch.from_numpy(s_grid).unsqueeze(0).to(device)
                with torch.no_grad():
                    ta_ssst = ssst.predict(ctx_n, s_n).squeeze(0).cpu().numpy()
                    Ta_ssst = Ta_min + ta_ssst * amp
                    per_model_pieces["SSST"].append(
                        (k_grid, smooth_branch(Ta_ssst, window=args.smooth_window,
                                                  poly=args.smooth_poly)))
                    ta_nept = neptune.sample(ctx_n, s_n, num_steps=args.neptune_steps)
                    ta_nept = ta_nept.squeeze(0).cpu().numpy()
                    Ta_nept = Ta_min + ta_nept * amp
                    per_model_pieces["NEPTUNE"].append(
                        (k_grid, smooth_branch(Ta_nept, window=args.smooth_window,
                                                   poly=args.smooth_poly)))

        # Plot individual models lightly
        for name, pieces in per_model_pieces.items():
            c, a = light_colors[name]
            first = True
            for kk, yy in pieces:
                ax.plot(kk, yy, "-", lw=1.0, color=c, alpha=a,
                          label=name if first else None, zorder=2)
                first = False

        # Compute ensemble mean per branch and overlay it bold
        # (use the same k_grid since all models share it within a given branch)
        first_ens = True
        for branch_idx in range(len(sub_desc)):
            ens_curves = []
            for name in ENSEMBLE_NAMES:
                if branch_idx < len(per_model_pieces[name]):
                    ens_curves.append(per_model_pieces[name][branch_idx][1])
            if not ens_curves:
                continue
            kk = per_model_pieces[ENSEMBLE_NAMES[0]][branch_idx][0]
            ens_mean = np.median(np.stack(ens_curves, axis=0), axis=0)
            ax.plot(kk, ens_mean, "-", lw=2.6, color="black",
                      label="ENSEMBLE median (5 models)" if first_ens else None,
                      zorder=15)
            first_ens = False

        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel("Ta", fontsize=12)
        ax.set_title(f"Ta(k) at E = {E:.5g}   ({len(sub_desc)} branch{'es' if len(sub_desc) > 1 else ''})\n"
                       f"Per-branch SG smoothing w={args.smooth_window}, "
                       f"NEPTUNE {args.neptune_steps} diff. steps,  ENSEMBLE = median of 5 models",
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
