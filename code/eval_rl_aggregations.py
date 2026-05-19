"""Benchmark several RL-v2 ensemble aggregations on the test windows.

For each aggregation strategy we compute:
    - mean test MAE  (lower is better)
    - mean test RMSE
    - gain vs the D5 baseline

Strategies:
  median_all      = current default (median of SARL_v2 5 seeds + MARL_v2 5 seeds)
  median_sarl     = SARL-only median (5 seeds)
  median_marl     = MARL-only median (5 seeds)
  weighted_all    = weighted median (weight = 1 / val_mae) over all 10
  weighted_sarl   = weighted median over SARL only
  best_sarl       = single best SARL seed by val_mae
  trim_mean_all   = trimmed mean (drop the 2 worst contributions per point)
  weighted_mean_all = weighted MEAN (weight = 1 / val_mae) over all 10
  weighted_mean_sarl = weighted MEAN over SARL only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path("C:/Users/hayan/Desktop/Code_Final_IA7/_TCML_check")
sys.path.insert(0, str(ROOT))

from data_pipeline.dataset import HydraWindowsDataset
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro
from models.marl_3sac.marl_model import MARLProSystem


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    test_npz = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/"
                       "07_StepF__Export_RL_Windows_Pro/02_outputs/"
                       "switch_rl_dataset_pro_v3_norm_fixed/rl_switch_windows_test.npz"))
    ds = HydraWindowsDataset(test_npz)
    print(f"test windows: {len(ds)}, T={ds.y_true.shape[1]}")

    obs_t = torch.from_numpy(ds.obs_seq).to(device)
    sv_t = torch.from_numpy(ds.static_vec).to(device)
    y_pred = torch.from_numpy(ds.y_pred).to(device)
    y_true = torch.from_numpy(ds.y_true).to(device).cpu().numpy()
    T = ds.y_true.shape[1]

    # Load SARL_v2 seeds with their val_mae
    sarl_dir = ROOT / "data" / "runs" / "sarl_v2"
    sarl_seeds = sorted([d for d in sarl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    sarl_corr = []; sarl_val = []
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
        with torch.no_grad():
            feat = ext(obs_t, sv_t)
            mu = sac.actor.mean(sac.actor.body(feat))
            corr = torch.tanh(mu)
            sarl_corr.append((y_pred + corr).cpu().numpy())
        h = json.loads((sd / "history.json").read_text(encoding="utf-8"))
        sarl_val.append(min(e["val_mae"] for e in h))
    sarl_corr = np.stack(sarl_corr, axis=0)
    print(f"Loaded SARL_v2: {len(sarl_corr)} seeds, val_mae = {sarl_val}")

    # Load MARL_v2 seeds
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    marl_cfg = full_cfg["marl_pro"]
    marl_cfg["agent_features"].setdefault("seq_out",
                                              marl_cfg["agent_features"]["seq_hidden"])
    marl_dir = ROOT / "data" / "runs" / "marl_v2"
    marl_seeds = sorted([d for d in marl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    marl_corr = []; marl_val = []
    for sd in marl_seeds:
        m = MARLProSystem(obs_seq_dim=ds.obs_seq.shape[2],
                              obs_seq_T=ds.obs_seq.shape[1],
                              static_dim=ds.static_vec.shape[1],
                              cfg=marl_cfg).to(device)
        st = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(st["state_dict"]); m.eval()
        with torch.no_grad():
            actions = m.sample_actions(obs_t, sv_t)
            delta = reconstruct_marl_correction(actions, T)
            marl_corr.append((y_pred + delta).cpu().numpy())
        h = json.loads((sd / "history.json").read_text(encoding="utf-8"))
        marl_val.append(min(e["val_mae"] for e in h))
    marl_corr = np.stack(marl_corr, axis=0)
    print(f"Loaded MARL_v2: {len(marl_corr)} seeds, val_mae = {marl_val}")

    def metrics(name, y_pred_arr):
        err = y_pred_arr - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        return name, mae, rmse, r2

    rows = []
    # Baseline (D5 y_pred)
    rows.append(metrics("D5_baseline       ", y_pred.cpu().numpy()))

    # Strategies
    rows.append(metrics("median_all_10     ", np.median(np.concatenate([sarl_corr, marl_corr], axis=0), axis=0)))
    rows.append(metrics("median_sarl_5     ", np.median(sarl_corr, axis=0)))
    rows.append(metrics("median_marl_5     ", np.median(marl_corr, axis=0)))

    all_corr = np.concatenate([sarl_corr, marl_corr], axis=0)
    all_val = np.array(sarl_val + marl_val)
    weights_all = 1.0 / (all_val + 1e-6)
    weights_all = weights_all / weights_all.sum()
    rows.append(metrics("weighted_mean_all ", np.einsum("i...,i->...", all_corr, weights_all)))

    sarl_w = 1.0 / (np.array(sarl_val) + 1e-6)
    sarl_w = sarl_w / sarl_w.sum()
    rows.append(metrics("weighted_mean_sarl", np.einsum("i...,i->...", sarl_corr, sarl_w)))

    rows.append(metrics("mean_all_10       ", np.mean(all_corr, axis=0)))
    rows.append(metrics("mean_sarl_5       ", np.mean(sarl_corr, axis=0)))
    rows.append(metrics("mean_marl_5       ", np.mean(marl_corr, axis=0)))

    best_sarl_i = int(np.argmin(sarl_val))
    rows.append(metrics(f"best_sarl_seed_{42+best_sarl_i}  ", sarl_corr[best_sarl_i]))

    # Trimmed mean (drop the 2 worst per point)
    sorted_all = np.sort(all_corr, axis=0)
    rows.append(metrics("trim_mean_all_8   ", sorted_all[1:-1].mean(axis=0)))

    print("\n  strategy                MAE       RMSE      R^2")
    print("  " + "-" * 56)
    for name, mae, rmse, r2 in rows:
        print(f"  {name}  {mae:.5f}   {rmse:.5f}   {r2:.5f}")


if __name__ == "__main__":
    main()
