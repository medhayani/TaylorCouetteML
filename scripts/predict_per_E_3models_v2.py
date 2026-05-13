"""Per-E superposition: data + CNP-v2 + DISTIL-STAR-v1 + RL-v2.

For each selected E in combined_data.csv:
  - black points = data
  - blue curve  = CNP-v2 zero-shot ensemble mean (5 seeds)
  - green curve = DISTIL-STAR-v1 ensemble mean (5 seeds)
  - red curve   = RL-v2 refined (windows-only): the windows' base y_pred + delta
                  from (SARL_v2 + MARL_v2)/2 averaged over 5 seeds each.

Output: docs/figures/3models_per_E/E_<idx>_<E>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.cnp_tc.model import CNP_TC
from models.star.model import STAR
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro
from models.marl_3sac.marl_model import MARLProSystem
from data_pipeline.dataset import HydraWindowsDataset


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 150, "figure.dpi": 110,
})


def reconstruct_marl_correction(actions, T, device):
    a_loc, a_sh, a_geo = actions
    B = a_loc.size(0)
    s = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)
    dc = a_loc[:, 0:1] * 0.5 + 0.5
    da = a_loc[:, 1:2] * 0.5
    bump = da * torch.exp(-((s - dc) ** 2) / (2 * 0.05 ** 2))
    bias, scale = a_sh[:, 0:1], a_sh[:, 1:2]
    sh = bias + scale * bump
    sl, sr, w, asy = a_geo[:, 0:1], a_geo[:, 1:2], a_geo[:, 2:3], a_geo[:, 3:4]
    geo = (sl * (s - dc).clamp(max=0) + sr * (s - dc).clamp(min=0)
           + w * (s - dc).abs() + asy * (s - 0.5))
    return sh + 0.1 * geo


def build_per_branch_inputs(desc_csv: Path, n_resampled: int = 151):
    desc = pd.read_csv(desc_csv)
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    items = []
    for _, row in desc.iterrows():
        E_val = float(row["E"]); b = int(row["branch_local_id"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)
        s_grid = np.linspace(0.0, 1.0, n_resampled, dtype=np.float32)
        k_grid = k_left + s_grid * (k_right - k_left)
        k_norm = 2.0 * s_grid - 1.0
        ctx_raw = []
        for f in STATIC_FEATURES:
            v = row.get(f, np.nan)
            try: ctx_raw.append(float(v))
            except (TypeError, ValueError): ctx_raw.append(0.0)
        ctx_raw = np.nan_to_num(np.asarray(ctx_raw, dtype=np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        anc_raw = np.asarray([k_left, k_right,
                                np.log10(max(Ta_min, 1e-6)),
                                np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
        items.append({
            "E": E_val, "branch_local_id": b,
            "Ta_min": Ta_min, "Ta_max": Ta_max,
            "k_left": k_left, "k_right": k_right,
            "amp": amp,
            "k_grid": k_grid, "k_norm": k_norm.astype(np.float32),
            "ctx_raw": ctx_raw, "anc_raw": anc_raw,
            "log10E": float(np.log10(max(E_val, 1e-30))),
        })
    return items


def normalize(arr, mean, std):
    out = np.clip(np.nan_to_num((arr - mean) / std, nan=0.0,
                                   posinf=0.0, neginf=0.0), -5.0, 5.0)
    return out.astype(np.float32)


def load_cnp_seeds(seed_dir: Path, ctx_dim: int, device, cfg_args: dict):
    seeds = sorted([d for d in seed_dir.iterdir() if d.is_dir()
                       and d.name.startswith("seed_")])
    models = []
    for sd in seeds:
        m = CNP_TC(ctx_dim=ctx_dim, **cfg_args).to(device)
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)
    return models


def load_star_seeds(seed_dir: Path, ctx_dim: int, device, cfg_args: dict):
    seeds = sorted([d for d in seed_dir.iterdir() if d.is_dir()
                       and d.name.startswith("seed_")])
    models = []
    for sd in seeds:
        m = STAR(ctx_dim=ctx_dim, **cfg_args).to(device)
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)
    return models


def load_sarl_seeds(seed_dir: Path, device):
    seeds = sorted([d for d in seed_dir.iterdir() if d.is_dir()
                       and d.name.startswith("seed_")])
    out = []
    for sd in seeds:
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        cfg = ck["cfg"]; obs = ck["obs_seq_shape"]
        extractor = SARLProFeatureExtractor(obs[2], obs[1], ck["static_dim"],
                                              cfg["feature_extractor"]).to(device)
        sac = SACPro(feature_dim=extractor.out_dim, action_dim=ck["action_dim"],
                       actor_layers=cfg["actor_layers"],
                       critic_layers=cfg["critic_layers"]).to(device)
        extractor.load_state_dict(ck["extractor"])
        sac.load_state_dict(ck["sac"])
        extractor.eval(); sac.eval()
        out.append((extractor, sac))
    return out


def load_marl_seeds(seed_dir: Path, obs_shape, static_dim, marl_cfg, device):
    seeds = sorted([d for d in seed_dir.iterdir() if d.is_dir()
                       and d.name.startswith("seed_")])
    out = []
    for sd in seeds:
        marl = MARLProSystem(obs_seq_dim=obs_shape[2], obs_seq_T=obs_shape[1],
                              static_dim=static_dim, cfg=marl_cfg).to(device)
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        marl.load_state_dict(ck["state_dict"]); marl.eval()
        out.append(marl)
    return out


@torch.no_grad()
def predict_cnp_curve(models, item, ctx_mean, ctx_std, anc_mean, anc_std, device):
    ctx_n = normalize(item["ctx_raw"], ctx_mean, ctx_std)
    anc_n = normalize(item["anc_raw"], anc_mean, anc_std)
    k_t = torch.from_numpy(item["k_norm"]).unsqueeze(0).to(device)
    ctx_t = torch.from_numpy(ctx_n).unsqueeze(0).to(device)
    anc_t = torch.from_numpy(anc_n).unsqueeze(0).to(device)
    logE_t = torch.tensor([item["log10E"]], dtype=torch.float32, device=device)
    T = k_t.size(1)
    M = models[0].max_ctx_points
    ck = torch.zeros(1, M, device=device)
    cta = torch.zeros(1, M, device=device)
    cm = torch.zeros(1, M, device=device)  # all-zero mask: zero-shot mode
    preds = []
    for m in models:
        p = m(k_t, ctx_t, anc_t, logE_t, ck, cta, cm).squeeze(0).cpu().numpy()
        preds.append(p)
    return np.stack(preds, axis=0)  # (n_seeds, T)


@torch.no_grad()
def predict_star_curve(models, item, ctx_mean, ctx_std, anc_mean, anc_std, device):
    ctx_n = normalize(item["ctx_raw"], ctx_mean, ctx_std)
    anc_n = normalize(item["anc_raw"], anc_mean, anc_std)
    k_t = torch.from_numpy(item["k_norm"]).unsqueeze(0).to(device)
    ctx_t = torch.from_numpy(ctx_n).unsqueeze(0).to(device)
    anc_t = torch.from_numpy(anc_n).unsqueeze(0).to(device)
    logE_t = torch.tensor([item["log10E"]], dtype=torch.float32, device=device)
    preds = []
    for m in models:
        p = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
        preds.append(p)
    return np.stack(preds, axis=0)


@torch.no_grad()
def predict_rl_for_windows(sarl_models, marl_models, ds, device):
    obs_t = torch.from_numpy(ds.obs_seq).to(device)
    sv_t = torch.from_numpy(ds.static_vec).to(device)
    y_pred = torch.from_numpy(ds.y_pred).to(device)
    T = ds.y_true.shape[1]

    sarl_corr = []
    for (extractor, sac) in sarl_models:
        feat = extractor(obs_t, sv_t)
        mu = sac.actor.mean(sac.actor.body(feat))
        sarl_corr.append(torch.tanh(mu).cpu().numpy())
    sarl_mean = np.mean(np.stack(sarl_corr, axis=0), axis=0)

    marl_corr = []
    for marl in marl_models:
        actions = marl.sample_actions(obs_t, sv_t)
        delta = reconstruct_marl_correction(actions, T, device)
        marl_corr.append(delta.cpu().numpy())
    marl_mean = np.mean(np.stack(marl_corr, axis=0), axis=0)

    y_pred_np = y_pred.cpu().numpy()
    y_corr_sarl = y_pred_np + sarl_mean
    y_corr_marl = y_pred_np + marl_mean
    y_corr_rl = 0.5 * (y_corr_sarl + y_corr_marl)
    return y_pred_np, y_corr_rl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=50)
    ap.add_argument("--out_dir", type=Path,
                     default=Path("C:/Users/hayan/Desktop/Code_Final_IA7/_TCML_check/figures/per_E_50_3models_v4"))
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    desc_csv = ROOT / "data" / "processed" / "branch_functional_descriptors.csv"
    input_csv = ROOT / "data" / "Input" / "combined_data.csv"
    items = build_per_branch_inputs(desc_csv)
    print(f"branches: {len(items)}", flush=True)

    raw = pd.read_csv(input_csv); raw.columns = ["Ta", "k", "E"]

    # Normalisations from training
    cnp_norm = np.load(ROOT / "data" / "runs" / "cnp_v2" / "test_branches.npz",
                         allow_pickle=False)
    star_norm = np.load(ROOT / "data" / "runs" / "distil_v1" / "test_branches.npz",
                          allow_pickle=False)

    # CNP model args (from cnp_v2 kernel)
    cnp_kwargs = dict(n_cheb=64, n_cos=32, d_model=256,
                       n_enc_layers=6, n_dec_layers=3, n_heads=8,
                       dropout=0.05, n_E_freq=32, n_k_freq=16,
                       max_ctx_points=32)
    # STAR args (from distil_v1 kernel)
    star_kwargs = dict(n_cheb=64, n_cos=32, d_model=384,
                        n_layers=8, n_heads=12, dropout=0.05,
                        n_E_freq=32, n_k_freq=16)

    print("Loading CNP-v2 seeds ...", flush=True)
    cnp_models = load_cnp_seeds(ROOT / "data" / "runs" / "cnp_v2",
                                  ctx_dim=len(STATIC_FEATURES),
                                  device=device, cfg_args=cnp_kwargs)
    print(f"  -> {len(cnp_models)} seeds", flush=True)

    print("Loading DISTIL-STAR-v1 seeds ...", flush=True)
    star_models = load_star_seeds(ROOT / "data" / "runs" / "distil_v1",
                                     ctx_dim=len(STATIC_FEATURES),
                                     device=device, cfg_args=star_kwargs)
    print(f"  -> {len(star_models)} seeds", flush=True)

    print("Loading SARL-v2 seeds ...", flush=True)
    sarl_models = load_sarl_seeds(ROOT / "data" / "runs" / "sarl_v2", device)
    print(f"  -> {len(sarl_models)} seeds", flush=True)

    # Load RL windows (test split)
    win_path = ROOT / "data" / "rl_windows" / "rl_switch_windows_test.npz"
    print(f"Loading windows: {win_path.name}", flush=True)
    win_ds = HydraWindowsDataset(win_path)
    print(f"  -> {len(win_ds)} windows", flush=True)

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml")
                                  .read_text(encoding="utf-8"))
    marl_cfg = full_cfg["marl_pro"]
    marl_cfg["agent_features"].setdefault("seq_out",
                                              marl_cfg["agent_features"]["seq_hidden"])
    print("Loading MARL-v2 seeds ...", flush=True)
    marl_models = load_marl_seeds(ROOT / "data" / "runs" / "marl_v2",
                                     obs_shape=win_ds.obs_seq.shape,
                                     static_dim=win_ds.static_vec.shape[1],
                                     marl_cfg=marl_cfg, device=device)
    print(f"  -> {len(marl_models)} seeds", flush=True)

    # Pre-compute RL refined per window
    print("Running RL inference on all test windows ...", flush=True)
    y_pred_win, y_rl_win = predict_rl_for_windows(sarl_models, marl_models,
                                                       win_ds, device)
    # Build a lookup (rounded_E, branch_local_id) -> list of windows.
    # Float32 storage in the windows file vs float64 in the descriptor CSV
    # gives slightly different bit-patterns, so we round both sides.
    def make_key(E_val, b):
        return (round(float(E_val), 7), int(b))

    win_lookup = {}
    for w in range(len(win_ds)):
        key = make_key(win_ds.E[w], win_ds.branch_local_id[w])
        win_lookup.setdefault(key, []).append(w)

    # Pick E values spread on the log scale
    all_Es = sorted(np.unique([it["E"] for it in items]).tolist())
    log_min = np.log10(max(min(all_Es), 1e-4))
    log_max = np.log10(min(max(all_Es), 10.0))
    log_picks = np.linspace(log_min, log_max, args.n_figs)
    Es = []
    seen = set()
    for lp in log_picks:
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - lp)))]
        if nearest not in seen:
            Es.append(nearest); seen.add(nearest)
    print(f"Producing {len(Es)} figures", flush=True)

    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)

    for fig_idx, E in enumerate(Es):
        branches = [it for it in items if np.isclose(it["E"], E)]
        if not branches: continue
        branches.sort(key=lambda x: x["branch_local_id"])

        fig, ax = plt.subplots(figsize=(11, 7), dpi=120)
        gt = raw[np.isclose(raw["E"], E)].sort_values("k")
        if len(gt):
            ax.plot(gt["k"], gt["Ta"], "k.", ms=5,
                      label="data", zorder=20)

        for it in branches:
            cnp_preds = predict_cnp_curve(
                cnp_models, it, cnp_norm["ctx_mean"], cnp_norm["ctx_std"] + 1e-6,
                cnp_norm["anc_mean"], cnp_norm["anc_std"] + 1e-6, device)
            star_preds = predict_star_curve(
                star_models, it, star_norm["ctx_mean"], star_norm["ctx_std"] + 1e-6,
                star_norm["anc_mean"], star_norm["anc_std"] + 1e-6, device)
            cnp_mean = cnp_preds.mean(axis=0)
            star_mean = star_preds.mean(axis=0)
            Ta_cnp = it["Ta_min"] + cnp_mean * it["amp"]
            Ta_star = it["Ta_min"] + star_mean * it["amp"]
            ax.plot(it["k_grid"], Ta_cnp, "-", lw=1.7,
                      color="#1f77b4", alpha=0.9,
                      label="CNP-v2 (5-seed mean)" if it is branches[0] else None)
            ax.plot(it["k_grid"], Ta_star, "--", lw=1.7,
                      color="#2ca02c", alpha=0.9,
                      label="DISTIL-STAR-v1 (5-seed mean)" if it is branches[0] else None)

            key = make_key(it["E"], it["branch_local_id"])
            if key in win_lookup:
                for w in win_lookup[key]:
                    lg = win_ds.local_grid[w]
                    half = float(win_ds.window_half_width[w])
                    centre = float(win_ds.center_pred[w])
                    # Window schema: absolute s = center_pred + half * local_grid,
                    # with local_grid in [-1, 1] and s in branch-normalised [0, 1].
                    s_branch = centre + half * lg
                    k_win = it["k_left"] + s_branch * (it["k_right"] - it["k_left"])
                    Ta_win = it["Ta_min"] + y_rl_win[w] * it["amp"]
                    ax.plot(k_win, Ta_win, "-", lw=2.2,
                              color="#d62728", alpha=0.95,
                              label="RL-v2 refined (SARL+MARL)" if (it is branches[0] and w == win_lookup[key][0]) else None)

        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("k", fontsize=12)
        ax.set_ylabel(r"$T_a$", fontsize=12)
        ax.set_title(rf"$T_a(k)$ at $E={E:.5g}$  ({len(branches)} branch"
                       f"{'es' if len(branches) > 1 else ''})",
                       fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
        fig.tight_layout()

        E_str = f"{E:.5g}".replace("+", "").replace(".", "p")
        out_png = out_dir / f"E_{fig_idx:03d}_{E_str}.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        if (fig_idx + 1) % 4 == 0 or fig_idx == len(Es) - 1:
            print(f"  {fig_idx + 1}/{len(Es)} done", flush=True)

    print(f"Saved {len(Es)} figures under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
