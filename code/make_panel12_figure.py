"""Build a single 4-row x 3-column panel figure showing Ta(k) at 12 E values.

No per-subplot title. One global legend at the bottom of the figure.
Each subplot has a small annotation E=... inside its top-right corner.

This is the main qualitative-result figure of the article.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import savgol_filter

ROOT = Path("C:/Users/hayan/Desktop/Code_Final_IA7/_TCML_check")
sys.path.insert(0, str(ROOT))

from data_pipeline.dataset import NeptuneProfileDataset, HydraWindowsDataset
from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.sparse_moe_transformer.ssst_model import SSSTProSurrogate
from models.fno_latent_diffusion.trainer import NeptuneProSurrogate
from models.cson.model import CSON
from models.cnp_tc.model import CNP_TC
from models.star.model import STAR
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro
from models.marl_3sac.marl_model import MARLProSystem


PRECISION_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def reconstruct_marl_correction(actions, T):
    a_loc, a_sh, a_geo = actions
    B = a_loc.size(0)
    s = torch.linspace(0, 1, T, device=a_loc.device).unsqueeze(0).expand(B, T)
    dc = a_loc[:, 0:1] * 0.5 + 0.5
    da = a_loc[:, 1:2] * 0.5
    bump = da * torch.exp(-((s - dc) ** 2) / (2 * 0.05 ** 2))
    bias, scale = a_sh[:, 0:1], a_sh[:, 1:2]
    sh = bias + scale * bump
    sl, sr, w, asy = a_geo[:, 0:1], a_geo[:, 1:2], a_geo[:, 2:3], a_geo[:, 3:4]
    geo = (sl * (s - dc).clamp(max=0) + sr * (s - dc).clamp(min=0)
           + w * (s - dc).abs() + asy * (s - 0.5))
    return sh + 0.1 * geo


def load_rl_v2_and_windows(device, root):
    win_root = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/"
                       "07_StepF__Export_RL_Windows_Pro/02_outputs/"
                       "switch_rl_dataset_pro_v3_norm_fixed"))
    parts = []
    for split in ("train", "val", "test"):
        parts.append(HydraWindowsDataset(win_root / f"rl_switch_windows_{split}.npz"))
    obs = np.concatenate([p.obs_seq for p in parts])
    sv = np.concatenate([p.static_vec for p in parts])
    yp = np.concatenate([p.y_pred for p in parts])
    lg = np.concatenate([p.local_grid for p in parts])
    centres = np.concatenate([p.center_pred for p in parts])
    halves = np.concatenate([p.window_half_width for p in parts])
    Es = np.concatenate([p.E for p in parts])
    bids = np.concatenate([p.branch_local_id for p in parts])
    rl_idx = {(round(float(e), 7), int(b)): i for i, (e, b) in enumerate(zip(Es, bids))}

    import json as _json
    # SARL_v2 5 seeds with val_mae weights
    sarl_dir = root / "data" / "runs" / "sarl_v2"
    sarl_seeds = sorted([d for d in sarl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    sarl_models = []; sarl_val = []
    for sd in sarl_seeds:
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        cfg = ck["cfg"]; oshape = ck["obs_seq_shape"]
        ext = SARLProFeatureExtractor(oshape[2], oshape[1], ck["static_dim"],
                                          cfg["feature_extractor"]).to(device)
        sac = SACPro(feature_dim=ext.out_dim, action_dim=ck["action_dim"],
                       actor_layers=cfg["actor_layers"],
                       critic_layers=cfg["critic_layers"]).to(device)
        ext.load_state_dict(ck["extractor"])
        sac.load_state_dict(ck["sac"]); ext.eval(); sac.eval()
        sarl_models.append((ext, sac))
        h = _json.loads((sd / "history.json").read_text(encoding="utf-8"))
        sarl_val.append(min(e["val_mae"] for e in h))

    arrays = {"obs_seq": obs.astype(np.float32),
                "static_vec": sv.astype(np.float32),
                "y_pred": yp.astype(np.float32),
                "local_grid": lg.astype(np.float32),
                "center_pred": centres.astype(np.float32),
                "window_half_width": halves.astype(np.float32),
                "E": Es, "branch_local_id": bids,
                "sarl_val_mae": np.array(sarl_val, dtype=np.float32)}
    return sarl_models, rl_idx, arrays


def make_key(E, b):
    return (round(float(E), 7), int(b))


def rl_refine(sarl_models, rl_idx, rl_arr, E, b_id, Ta_min, Ta_max,
                k_left, k_right, device):
    key = make_key(E, b_id)
    if key not in rl_idx: return None
    i = rl_idx[key]
    amp = max(Ta_max - Ta_min, 1e-12)
    obs_t = torch.from_numpy(rl_arr["obs_seq"][i:i+1]).to(device)
    sv_t = torch.from_numpy(rl_arr["static_vec"][i:i+1]).to(device)
    y_pred = torch.from_numpy(rl_arr["y_pred"][i:i+1]).to(device)
    samples = []
    with torch.no_grad():
        for (ext, sac) in sarl_models:
            feat = ext(obs_t, sv_t)
            mu = sac.actor.mean(sac.actor.body(feat))
            corr = torch.tanh(mu)
            samples.append((y_pred + corr).squeeze(0).cpu().numpy())
    sarl_arr = np.stack(samples, axis=0)
    vmae = rl_arr["sarl_val_mae"]
    w = 1.0 / (vmae + 1e-6); w = w / w.sum()
    y_rl_norm = np.clip(np.einsum("i...,i->...", sarl_arr, w), -0.2, 1.2)
    Ta_rl = Ta_min + y_rl_norm * amp
    local_grid = rl_arr["local_grid"][i]
    centre = float(rl_arr["center_pred"][i])
    half = float(rl_arr["window_half_width"][i])
    s_abs = centre + half * local_grid
    k_rl = k_left + s_abs * (k_right - k_left)
    return k_rl, Ta_rl


def smooth(y, window=9):
    n = len(y)
    if n < window: return y
    if window % 2 == 0: window += 1
    return savgol_filter(y, window_length=window, polyorder=3, mode="nearest")


def load_supervised(runs, device, train_in_dim):
    prec_models = {}
    for sd in sorted([d for d in (runs / "precision_v2").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        ms = {}
        m1 = SIRENRegressor(ctx_dim=23, hidden=384, depth=8).to(device)
        m1.load_state_dict(torch.load(sd / "siren" / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m1.eval(); ms["SIREN"] = m1
        m2 = DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4,
                          latent_dim=192, fourier_bands=24).to(device)
        m2.load_state_dict(torch.load(sd / "deeponet" / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m2.eval(); ms["DeepONet"] = m2
        m3 = ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32).to(device)
        m3.load_state_dict(torch.load(sd / "chebyshev" / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m3.eval(); ms["Chebyshev"] = m3
        m4 = MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7).to(device)
        m4.load_state_dict(torch.load(sd / "envelope_siren" / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m4.eval(); ms["EnvelopeSIREN"] = m4
        prec_models[sd.name] = ms

    cson_models = []
    for sd in sorted([d for d in (runs / "cson_v1").iterdir() if d.is_dir() and d.name.startswith("seed_")]):
        m = CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5, num_heads=8,
                    dropout=0.05, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m.eval(); cson_models.append(m)
    cnp_models = []
    for sd in sorted([d for d in (runs / "cnp_v1").iterdir() if d.is_dir() and d.name.startswith("seed_")]):
        m = CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6,
                      n_dec_layers=3, n_heads=8, dropout=0.05, n_E_freq=32, n_k_freq=16,
                      max_ctx_points=32).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m.eval(); cnp_models.append(m)
    star_models = []
    for sd in sorted([d for d in (runs / "star_v1").iterdir() if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12,
                    dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m.eval(); star_models.append(m)
    distil_models = []
    for sd in sorted([d for d in (runs / "distil_v1").iterdir() if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12,
                    dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device, weights_only=False)["state_dict"])
        m.eval(); distil_models.append(m)
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg_ssst = full_cfg["ssst_pro"]; cfg_ssst["ctx_dim"] = train_in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    ssst.load_state_dict(torch.load(runs / "ssst_v2" / "best.pt", map_location=device, weights_only=False)["state_dict"])
    ssst.eval()
    cfg_n = full_cfg["neptune_pro"]; cfg_n["context"]["in_dim"] = train_in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    neptune.load_state_dict(torch.load(runs / "neptune_v2" / "member_00" / "best.pt", map_location=device, weights_only=False)["state_dict"])
    neptune.eval()
    return prec_models, cson_models, cnp_models, star_models, distil_models, ssst, neptune


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                     default=Path("C:/Users/hayan/Desktop/Code_Final_IA7/Livraison/v2/Article/figures"))
    ap.add_argument("--n_pred", type=int, default=151)
    ap.add_argument("--neptune_steps", type=int, default=60)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    runs = ROOT / "data" / "runs"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                        names=["Ta", "k", "E"], header=None, skiprows=1)

    # Pick 12 E values log-spread, prefer branches with switch windows for RL visibility.
    all_Es = sorted(desc["E"].unique())
    log_picks = np.linspace(-4, 1, 12)
    Es, seen = [], set()
    for lp in log_picks:
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - lp)))]
        if nearest not in seen:
            Es.append(nearest); seen.add(nearest)
    print("E values:", [f"{e:.4g}" for e in Es])

    prec_norm = np.load(runs / "precision_v2" / "test_branches.npz", allow_pickle=False)
    prec_ctx_mean = prec_norm["ctx_mean"].astype(np.float32)
    prec_ctx_std = prec_norm["ctx_std"].astype(np.float32) + 1e-6
    anc_arr = np.stack([
        desc["k_left"].values, desc["k_right"].values,
        np.log10(desc["Ta_min"].clip(lower=1e-6).values),
        np.log10(desc["Ta_max"].clip(lower=1e-6).values),
    ], axis=1).astype(np.float32)
    anc_mean = anc_arr.mean(axis=0); anc_std = anc_arr.std(axis=0) + 1e-6

    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    full_ds = NeptuneProfileDataset(profile_csv,
                                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    keys_to_idx = {k: i for i, k in enumerate(full_ds.keys)}

    print("Loading supervised surrogates ...")
    (prec_models, cson_models, cnp_models, star_models, distil_models,
       ssst, neptune) = load_supervised(runs, device, train_ds.in_dim)
    print("Loading RL refiner (weighted-SARL, base D5) ...")
    sarl_models, rl_idx, rl_arr = load_rl_v2_and_windows(device, ROOT)

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(4, 3, figsize=(11, 12), dpi=130, sharex=True)
    axes_flat = axes.ravel()

    for ax, E in zip(axes_flat, Es):
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        gt = inp[np.isclose(inp["E"], E)].sort_values("k")
        ax.plot(gt["k"], gt["Ta"], "k.", ms=3.5, zorder=20)

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
            ctx_p_arr = np.clip(np.nan_to_num(ctx_p_arr, nan=0.0, posinf=0.0,
                                                  neginf=0.0), -5.0, 5.0)
            ctx_t = torch.from_numpy(ctx_p_arr).unsqueeze(0).to(device)
            k_t = torch.tensor(k_norm, dtype=torch.float32, device=device).unsqueeze(0)
            anc_raw = np.array([k_left, k_right, np.log10(max(Ta_min, 1e-6)),
                                  np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
            anc_norm = np.clip(np.nan_to_num((anc_raw - anc_mean) / anc_std,
                                                  nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)
            anc_t = torch.from_numpy(anc_norm).unsqueeze(0).to(device)
            logE_t = torch.tensor([np.log10(max(E, 1e-30))], dtype=torch.float32, device=device)

            preds = []
            with torch.no_grad():
                for _, ms in prec_models.items():
                    for _, m in ms.items():
                        ta = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                        preds.append(Ta_min + ta * amp)
                for m in cson_models:
                    preds.append(Ta_min + m(k_t, ctx_t).squeeze(0).cpu().numpy() * amp)
                empty_ck = torch.zeros(1, 32, device=device)
                empty_cta = torch.zeros(1, 32, device=device)
                empty_mask = torch.zeros(1, 32, device=device)
                for m in cnp_models:
                    ta = m(k_t, ctx_t, anc_t, logE_t, empty_ck, empty_cta, empty_mask).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta * amp)
                for m in star_models:
                    ta = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta * amp)
                for m in distil_models:
                    ta = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta * amp)
                if (E, b_id) in keys_to_idx:
                    i = keys_to_idx[(E, b_id)]
                    ctx_n_t = torch.from_numpy(full_ds.ctx[i]).unsqueeze(0).to(device)
                    s_n_t = torch.from_numpy(s_grid).unsqueeze(0).to(device)
                    ta_s = ssst.predict(ctx_n_t, s_n_t).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta_s * amp)
                    ta_n = neptune.sample(ctx_n_t, s_n_t, num_steps=args.neptune_steps).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta_n * amp)
            stack = np.stack(preds, axis=0)
            Ta_med = smooth(np.median(stack, axis=0))
            ax.plot(k_grid, Ta_med, "-", lw=1.0, color="tab:blue", zorder=12)
            if len(gt) >= 5:
                Ta_true_grid = np.interp(k_grid, gt["k"].values, gt["Ta"].values)
                err = np.abs(stack - Ta_true_grid[None, :])
                best_idx = np.argmin(err, axis=0)
                Ta_orc = stack[best_idx, np.arange(len(Ta_true_grid))]
                ax.plot(k_grid, Ta_orc, "--", lw=1.0, color="tab:green", zorder=11)
            # RL refined + augmented (full range): MED-NN + Tukey blend with RL window
            rl = rl_refine(sarl_models, rl_idx, rl_arr, E, b_id,
                              Ta_min, Ta_max, k_left, k_right, device)
            if rl is not None:
                k_rl, Ta_rl = rl
                Ta_rl_aug = Ta_med.copy()
                k_rl_min, k_rl_max = float(np.min(k_rl)), float(np.max(k_rl))
                in_win = (k_grid >= k_rl_min) & (k_grid <= k_rl_max)
                if in_win.any():
                    Ta_rl_on_grid = np.interp(k_grid[in_win], k_rl, Ta_rl)
                    span = max(k_rl_max - k_rl_min, 1e-6)
                    s_local = (k_grid[in_win] - k_rl_min) / span
                    edge = 0.20
                    alpha = np.ones_like(s_local)
                    lm = s_local < edge; rm = s_local > 1.0 - edge
                    alpha[lm] = 0.5 * (1 - np.cos(np.pi * s_local[lm] / edge))
                    alpha[rm] = 0.5 * (1 - np.cos(np.pi * (1.0 - s_local[rm]) / edge))
                    Ta_rl_aug[in_win] = alpha * Ta_rl_on_grid + (1.0 - alpha) * Ta_med[in_win]
                ax.plot(k_grid, Ta_rl_aug, "-", lw=1.2, color="tab:red", zorder=15)

        ax.set_xlim(args.xmin, args.xmax)
        ax.text(0.97, 0.95, f"E = {E:.4g}",
                  transform=ax.transAxes, fontsize=9, ha="right", va="top",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="gray", alpha=0.85))

    for row_axes in axes:
        row_axes[0].set_ylabel(r"$T_a$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$k$")

    # Single shared legend at the bottom
    handles = [
        plt.Line2D([], [], color="black", linestyle="None", marker=".",
                     markersize=6, label="Floquet data"),
        plt.Line2D([], [], color="tab:blue", lw=1.6,
                     label="Ensemble median (34 surrogates)"),
        plt.Line2D([], [], color="tab:green", lw=1.6, linestyle="--",
                     label="Pointwise oracle (upper bound)"),
        plt.Line2D([], [], color="tab:red", lw=1.6,
                     label="RL refiner (weighted ensemble)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
                 frameon=True, fontsize=10,
                 bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = args.out_dir / "panel12_Ta_k_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
