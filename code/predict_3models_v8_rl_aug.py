"""3-model superposition v5: MED-NN + ORC-NN + RL-v2 (new SARL_v2 + MARL_v2 ensemble).

Identical pipeline to predict_3models_clean.py *except* the third model
(RL refiner) now uses the V2 ensembles trained on Kaggle yesterday:
    data/runs/sarl_v2/seed_42..46/best.pt        (5 seeds, 1500 ep)
    data/runs/marl_v2/seed_42..46/best.pt        (5 seeds, 1500 ep)

The refined window prediction is the median of  SARL_v2(5 seeds) + MARL_v2(5 seeds)
= 10 refiner samples per window.

Model 1 (MED-NN) = median of 34 supervised surrogates (precision_v2 12 + cson_v1 5
                   + cnp_v1 5 + star_v1 7 + distil_v1 3 + ssst_v2 1 + neptune_v2 1).
Model 2 (ORC-NN) = oracle best per-point selection across the 34 surrogates
                   (upper bound, needs the data at inference time).
Model 3 (RL-v2)  = median of 10 refiner samples (SARL_v2 + MARL_v2) applied to
                   the precomputed window base y_pred (only on branches that have
                   an associated switch window).
"""
from __future__ import annotations

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


def load_rl_v2_and_windows(device, root, agg: str = "weighted_mean_sarl"):
    """Load the V2 SARL+MARL ensembles (5 seeds each) + concatenated windows.

    agg : one of {"median_all", "median_sarl", "weighted_mean_sarl",
                    "weighted_mean_all", "mean_sarl"}.
    """
    win_root = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/"
                       "07_StepF__Export_RL_Windows_Pro/02_outputs/"
                       "switch_rl_dataset_pro_v3_norm_fixed"))
    ds_test = HydraWindowsDataset(win_root / "rl_switch_windows_test.npz")
    ds_val = HydraWindowsDataset(win_root / "rl_switch_windows_val.npz")
    ds_train = HydraWindowsDataset(win_root / "rl_switch_windows_train.npz")
    obs = np.concatenate([ds_train.obs_seq, ds_val.obs_seq, ds_test.obs_seq])
    sv = np.concatenate([ds_train.static_vec, ds_val.static_vec, ds_test.static_vec])
    yp = np.concatenate([ds_train.y_pred, ds_val.y_pred, ds_test.y_pred])
    yt = np.concatenate([ds_train.y_true, ds_val.y_true, ds_test.y_true])
    lg = np.concatenate([ds_train.local_grid, ds_val.local_grid, ds_test.local_grid])
    centres = np.concatenate([ds_train.center_pred, ds_val.center_pred, ds_test.center_pred])
    halves = np.concatenate([ds_train.window_half_width, ds_val.window_half_width,
                                ds_test.window_half_width])
    Es = np.concatenate([ds_train.E, ds_val.E, ds_test.E])
    bids = np.concatenate([ds_train.branch_local_id, ds_val.branch_local_id,
                                ds_test.branch_local_id])
    print(f"RL total windows: {len(obs)}")
    rl_idx = {(round(float(e), 7), int(b)): i for i, (e, b) in enumerate(zip(Es, bids))}

    # SARL_v2 (5 seeds)
    import json as _json
    sarl_dir = root / "data" / "runs" / "sarl_v2"
    sarl_seeds = sorted([d for d in sarl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    sarl_models = []
    sarl_val = []
    for sd in sarl_seeds:
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        cfg = ck["cfg"]; oshape = ck["obs_seq_shape"]
        ext = SARLProFeatureExtractor(oshape[2], oshape[1], ck["static_dim"],
                                          cfg["feature_extractor"]).to(device)
        sac = SACPro(feature_dim=ext.out_dim, action_dim=ck["action_dim"],
                       actor_layers=cfg["actor_layers"],
                       critic_layers=cfg["critic_layers"]).to(device)
        ext.load_state_dict(ck["extractor"])
        sac.load_state_dict(ck["sac"])
        ext.eval(); sac.eval()
        sarl_models.append((ext, sac))
        try:
            h = _json.loads((sd / "history.json").read_text(encoding="utf-8"))
            sarl_val.append(min(e["val_mae"] for e in h))
        except Exception:
            sarl_val.append(1.0)
    print(f"Loaded SARL_v2: {len(sarl_models)} seeds, val_mae = {sarl_val}")

    # MARL_v2 (5 seeds)
    full_cfg = yaml.safe_load((root / "configs" / "sizes.yaml")
                                  .read_text(encoding="utf-8"))
    marl_cfg = full_cfg["marl_pro"]
    marl_cfg["agent_features"].setdefault("seq_out",
                                              marl_cfg["agent_features"]["seq_hidden"])
    marl_dir = root / "data" / "runs" / "marl_v2"
    marl_seeds = sorted([d for d in marl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    marl_models = []
    marl_val = []
    for sd in marl_seeds:
        m = MARLProSystem(obs_seq_dim=obs.shape[2], obs_seq_T=obs.shape[1],
                              static_dim=sv.shape[1], cfg=marl_cfg).to(device)
        st = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(st["state_dict"]); m.eval()
        marl_models.append(m)
        try:
            h = _json.loads((sd / "history.json").read_text(encoding="utf-8"))
            marl_val.append(min(e["val_mae"] for e in h))
        except Exception:
            marl_val.append(1.0)
    print(f"Loaded MARL_v2: {len(marl_models)} seeds, val_mae = {marl_val}")

    arrays = {"obs_seq": obs.astype(np.float32),
                "static_vec": sv.astype(np.float32),
                "y_pred": yp.astype(np.float32),
                "y_true": yt.astype(np.float32),
                "local_grid": lg.astype(np.float32),
                "center_pred": centres.astype(np.float32),
                "window_half_width": halves.astype(np.float32),
                "E": Es, "branch_local_id": bids,
                "sarl_val_mae": np.array(sarl_val, dtype=np.float32),
                "marl_val_mae": np.array(marl_val, dtype=np.float32),
                "agg": agg}
    return sarl_models, marl_models, rl_idx, arrays


def rl_v2_refine(sarl_models, marl_models, rl_idx, rl_arr,
                  E, b_id, Ta_min, Ta_max, k_left, k_right, device):
    key = None
    for (e_k, b_k), i_k in rl_idx.items():
        if abs(e_k - E) / max(E, 1e-30) < 1e-4 and b_k == b_id:
            key = i_k; break
    if key is None: return None
    amp = max(Ta_max - Ta_min, 1e-12)
    obs_t = torch.from_numpy(rl_arr["obs_seq"][key:key+1]).to(device)
    sv_t = torch.from_numpy(rl_arr["static_vec"][key:key+1]).to(device)
    y_pred = torch.from_numpy(rl_arr["y_pred"][key:key+1]).to(device)
    T = y_pred.shape[1]
    sarl_samples = []
    marl_samples = []
    with torch.no_grad():
        for (ext, sac) in sarl_models:
            feat = ext(obs_t, sv_t)
            mu = sac.actor.mean(sac.actor.body(feat))
            corr = torch.tanh(mu)
            sarl_samples.append((y_pred + corr).squeeze(0).cpu().numpy())
        for marl in marl_models:
            actions = marl.sample_actions(obs_t, sv_t)
            delta = reconstruct_marl_correction(actions, T)
            marl_samples.append((y_pred + delta).squeeze(0).cpu().numpy())
    sarl_arr = np.stack(sarl_samples, axis=0)
    marl_arr = np.stack(marl_samples, axis=0)
    agg = rl_arr.get("agg", "weighted_mean_sarl")
    if agg == "median_sarl":
        y_rl_norm = np.median(sarl_arr, axis=0)
    elif agg == "median_all":
        y_rl_norm = np.median(np.concatenate([sarl_arr, marl_arr], axis=0), axis=0)
    elif agg == "mean_sarl":
        y_rl_norm = np.mean(sarl_arr, axis=0)
    elif agg == "weighted_mean_all":
        vmae = np.concatenate([rl_arr["sarl_val_mae"], rl_arr["marl_val_mae"]])
        w = 1.0 / (vmae + 1e-6); w = w / w.sum()
        y_rl_norm = np.einsum("i...,i->...", np.concatenate([sarl_arr, marl_arr], axis=0), w)
    else:  # weighted_mean_sarl (best per eval_rl_aggregations.py)
        vmae = rl_arr["sarl_val_mae"]
        w = 1.0 / (vmae + 1e-6); w = w / w.sum()
        y_rl_norm = np.einsum("i...,i->...", sarl_arr, w)
    y_rl_norm = np.clip(y_rl_norm, -0.2, 1.2)
    Ta_rl = Ta_min + y_rl_norm * amp
    # Correct mapping (per rl_windows_schema_pro_v3.json):
    # absolute s = center_pred + window_half_width * local_grid, local_grid in [-1, 1].
    local_grid = rl_arr["local_grid"][key]
    centre = float(rl_arr["center_pred"][key])
    half = float(rl_arr["window_half_width"][key])
    s_abs = centre + half * local_grid
    k_phys = k_left + s_abs * (k_right - k_left)
    return k_phys, Ta_rl


def smooth(y, window=9):
    n = len(y)
    if n < window: return y
    if window % 2 == 0: window += 1
    return savgol_filter(y, window_length=window, polyorder=3, mode="nearest")


def load_supervised_surrogates(runs, device, train_in_dim):
    prec_models = {}
    for sd in sorted([d for d in (runs / "precision_v2").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        ms = {}
        m1 = SIRENRegressor(ctx_dim=23, hidden=384, depth=8).to(device)
        m1.load_state_dict(torch.load(sd / "siren" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m1.eval(); ms["SIREN"] = m1
        m2 = DeepONet(ctx_dim=23, branch_layers=(384,)*4, trunk_layers=(384,)*4,
                          latent_dim=192, fourier_bands=24).to(device)
        m2.load_state_dict(torch.load(sd / "deeponet" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m2.eval(); ms["DeepONet"] = m2
        m3 = ChebyshevSpectralRegressor(ctx_dim=23, n_modes=32).to(device)
        m3.load_state_dict(torch.load(sd / "chebyshev" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m3.eval(); ms["Chebyshev"] = m3
        m4 = MultiModeEnvelopeSIREN(ctx_dim=23, n_modes=8, hidden=320, depth=7).to(device)
        m4.load_state_dict(torch.load(sd / "envelope_siren" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m4.eval(); ms["EnvelopeSIREN"] = m4
        prec_models[sd.name] = ms

    cson_models = []
    for sd in sorted([d for d in (runs / "cson_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = CSON(ctx_dim=23, n_modes=48, d_model=320, num_layers=5,
                    num_heads=8, dropout=0.05, use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); cson_models.append(m)

    cnp_models = []
    for sd in sorted([d for d in (runs / "cnp_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = CNP_TC(ctx_dim=23, n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6,
                      n_dec_layers=3, n_heads=8, dropout=0.05,
                      n_E_freq=32, n_k_freq=16, max_ctx_points=32).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); cnp_models.append(m)

    star_models = []
    for sd in sorted([d for d in (runs / "star_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                    n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                    use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); star_models.append(m)

    distil_models = []
    for sd in sorted([d for d in (runs / "distil_v1").iterdir()
                          if d.is_dir() and d.name.startswith("seed_")]):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8,
                    n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16,
                    use_spectral_norm=True).to(device)
        m.load_state_dict(torch.load(sd / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
        m.eval(); distil_models.append(m)

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg_ssst = full_cfg["ssst_pro"]; cfg_ssst["ctx_dim"] = train_in_dim
    ssst = SSSTProSurrogate(cfg_ssst).to(device)
    ssst.load_state_dict(torch.load(runs / "ssst_v2" / "best.pt", map_location=device,
                                          weights_only=False)["state_dict"])
    ssst.eval()
    cfg_n = full_cfg["neptune_pro"]; cfg_n["context"]["in_dim"] = train_in_dim
    neptune = NeptuneProSurrogate(cfg_n).to(device)
    neptune.load_state_dict(torch.load(runs / "neptune_v2" / "member_00" / "best.pt",
                                            map_location=device, weights_only=False)["state_dict"])
    neptune.eval()
    return prec_models, cson_models, cnp_models, star_models, distil_models, ssst, neptune


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_figs", type=int, default=50)
    ap.add_argument("--n_pred", type=int, default=151)
    ap.add_argument("--neptune_steps", type=int, default=120)
    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=20.0)
    ap.add_argument("--out_subdir", type=str, default="per_E_50_3models_v5")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    runs = ROOT / "data" / "runs"
    out_dir = ROOT / "figures" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    desc = pd.read_csv(ROOT / "data" / "processed" / "branch_functional_descriptors.csv").copy()
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    inp = pd.read_csv(ROOT / "data" / "Input" / "combined_data.csv",
                        names=["Ta", "k", "E"], header=None, skiprows=1)

    all_Es = sorted(desc["E"].unique())
    log_picks = np.linspace(np.log10(max(min(all_Es), 1e-4)),
                                  np.log10(min(max(all_Es), 10.0)),
                                  args.n_figs)
    Es, seen = [], set()
    for lp in log_picks:
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - lp)))]
        if nearest not in seen:
            Es.append(nearest); seen.add(nearest)
    print(f"Producing {len(Es)} figures")

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

    print("Loading supervised surrogates...")
    prec_models, cson_models, cnp_models, star_models, distil_models, ssst, neptune = \
        load_supervised_surrogates(runs, device, train_ds.in_dim)
    print(f"  precision: {len(prec_models)} seeds x 4 = {4*len(prec_models)} models")
    print(f"  CSON: {len(cson_models)}, CNP: {len(cnp_models)}, "
            f"STAR: {len(star_models)}, DISTIL: {len(distil_models)}")

    print("Loading RL_v2 refiners (SARL_v2 + MARL_v2 ensembles)...")
    sarl_models, marl_models, rl_idx, rl_arr = load_rl_v2_and_windows(device, ROOT)
    print(f"RL refiner samples per window: {len(sarl_models) + len(marl_models)}")

    for fig_idx, E in enumerate(Es):
        sub_desc = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        if len(sub_desc) == 0: continue
        fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
        gt = inp[np.isclose(inp["E"], E)].sort_values("k")
        if len(gt) >= 5:
            ax.plot(gt["k"], gt["Ta"], "k.", ms=6, label="data", zorder=20)

        branch_boundaries = []
        for _, row_b in sub_desc.iterrows():
            branch_boundaries.append(float(row_b["k_left"]))
            branch_boundaries.append(float(row_b["k_right"]))
        for kb in sorted(set(branch_boundaries)):
            ax.axvline(kb, color="gray", lw=0.8, ls=":", alpha=0.6, zorder=1)

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
            anc_raw = np.array([k_left, k_right,
                                  np.log10(max(Ta_min, 1e-6)),
                                  np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
            anc_norm = np.clip(np.nan_to_num((anc_raw - anc_mean) / anc_std,
                                                  nan=0.0, posinf=0.0, neginf=0.0),
                                  -5.0, 5.0)
            anc_t = torch.from_numpy(anc_norm).unsqueeze(0).to(device)
            logE_t = torch.tensor([np.log10(max(E, 1e-30))], dtype=torch.float32,
                                      device=device)

            preds = []
            with torch.no_grad():
                for seed, ms in prec_models.items():
                    for name, m in ms.items():
                        ta = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                        preds.append(Ta_min + ta * amp)
                for m in cson_models:
                    ta = m(k_t, ctx_t).squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta * amp)
                empty_ck = torch.zeros(1, 32, device=device)
                empty_cta = torch.zeros(1, 32, device=device)
                empty_mask = torch.zeros(1, 32, device=device)
                for m in cnp_models:
                    ta = m(k_t, ctx_t, anc_t, logE_t,
                              empty_ck, empty_cta, empty_mask).squeeze(0).cpu().numpy()
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
                    ta_n = neptune.sample(ctx_n_t, s_n_t, num_steps=args.neptune_steps)
                    ta_n = ta_n.squeeze(0).cpu().numpy()
                    preds.append(Ta_min + ta_n * amp)

            stack = np.stack(preds, axis=0)
            # Model 1: MED-NN
            Ta_med = smooth(np.median(stack, axis=0))
            ax.plot(k_grid, Ta_med, ".", ms=4, color="tab:blue",
                      label="_nolegend_", zorder=15)
            # Model 2: ORC-NN
            if len(gt) >= 5:
                Ta_true_grid = np.interp(k_grid, gt["k"].values, gt["Ta"].values)
                err = np.abs(stack - Ta_true_grid[None, :])
                best_idx = np.argmin(err, axis=0)
                Ta_orc = stack[best_idx, np.arange(len(Ta_true_grid))]
                ax.plot(k_grid, Ta_orc, ".", ms=4, color="lime",
                          label="_nolegend_", zorder=16)
            # Model 3: RL-AUG -- MED-NN partout, RL substitue dans la fenetre du switch
            # avec une transition Tukey (cosinus) sur 20% des cotes de la fenetre
            # pour eviter un raccord brusque.
            Ta_rl_aug = Ta_med.copy()
            rl = rl_v2_refine(sarl_models, marl_models, rl_idx, rl_arr,
                                  E, b_id, Ta_min, Ta_max, k_left, k_right, device)
            if rl is not None:
                k_rl, Ta_rl = rl
                # Interpole le RL sur k_grid ; en-dehors de la fenetre,
                # on garde MED-NN. On combine avec un blend cosinus dans
                # une bande d'overlap.
                k_rl_min, k_rl_max = float(np.min(k_rl)), float(np.max(k_rl))
                in_win = (k_grid >= k_rl_min) & (k_grid <= k_rl_max)
                if in_win.any():
                    Ta_rl_on_grid = np.interp(k_grid[in_win], k_rl, Ta_rl)
                    # alpha (poids du RL) : 1 au coeur, 0 aux bords sur 20% chacun
                    span = max(k_rl_max - k_rl_min, 1e-6)
                    s_local = (k_grid[in_win] - k_rl_min) / span         # in [0, 1]
                    edge = 0.20
                    alpha = np.ones_like(s_local)
                    left_mask = s_local < edge
                    right_mask = s_local > 1.0 - edge
                    alpha[left_mask] = 0.5 * (1 - np.cos(np.pi * s_local[left_mask] / edge))
                    alpha[right_mask] = 0.5 * (1 - np.cos(np.pi * (1.0 - s_local[right_mask]) / edge))
                    Ta_rl_aug[in_win] = alpha * Ta_rl_on_grid + (1.0 - alpha) * Ta_med[in_win]
            ax.plot(k_grid, Ta_rl_aug, ".", ms=4, color="red",
                      label="_nolegend_", zorder=17)

        n_sup = 12 + 5 + 5 + 7 + 3 + 2
        agg_label = rl_arr.get("agg", "weighted_mean_sarl")
        if agg_label == "weighted_mean_sarl":
            rl_legend = "RL-aug  (MED-NN + weighted-SARL inside switch window)"
        elif agg_label == "median_sarl":
            rl_legend = "RL-aug  (MED-NN + median-SARL inside switch window)"
        else:
            rl_legend = f"RL-aug  ({agg_label})"
        ax.plot([], [], ".", ms=8, color="tab:blue",
                  label=f"MED-NN  (median of {n_sup} supervised neural surrogates)")
        ax.plot([], [], ".", ms=8, color="lime",
                  label=f"ORC-NN  (oracle best per point, upper bound)")
        ax.plot([], [], ".", ms=8, color="red", label=rl_legend)
        ax.plot([], [], "k:", lw=0.8, alpha=0.7,
                  label="branch boundaries (cusps)")

        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("k", fontsize=12); ax.set_ylabel("Ta", fontsize=12)
        ax.set_title(
            f"Marginal stability Ta(k) at E = {E:.5g}   "
            f"({len(sub_desc)} branch{'es' if len(sub_desc) > 1 else ''})\n"
            f"3 surrogate types: supervised median  vs  supervised oracle  vs  RL-v2 ensemble refiner",
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
