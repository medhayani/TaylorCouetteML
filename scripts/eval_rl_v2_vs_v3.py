"""Benchmark rl_v2 (D5 base) vs rl_v3 (MED-NN base) on the test windows.

Each version is evaluated on its OWN test split (v2: rl_switch_windows_test.npz
with y_pred = D5  ;  v3: rl_switch_windows_v4_test.npz with y_pred = MED-NN).
We report MAE / RMSE / R^2 against y_true for several aggregations,
plus the gain over each version's baseline.
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


def metrics(name, y_pred, y_true):
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return name, mae, rmse, r2


def eval_version(version: str, sarl_dir: Path, marl_dir: Path,
                  test_npz: Path, device, marl_cfg):
    print(f"\n========== {version} ==========")
    print(f"windows: {test_npz}")
    ds = HydraWindowsDataset(test_npz)
    obs_t = torch.from_numpy(ds.obs_seq).to(device)
    sv_t = torch.from_numpy(ds.static_vec).to(device)
    y_pred = torch.from_numpy(ds.y_pred).to(device)
    y_true_np = ds.y_true
    T = ds.y_true.shape[1]
    print(f"N windows: {len(ds)}, T={T}")

    # Load SARL seeds
    sarl_seeds = sorted([d for d in sarl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    sarl_arr = []; sarl_val = []
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
            sarl_arr.append((y_pred + corr).cpu().numpy())
        h = json.loads((sd / "history.json").read_text(encoding="utf-8"))
        sarl_val.append(min(e["val_mae"] for e in h))
    sarl_arr = np.stack(sarl_arr, axis=0)

    # Load MARL seeds
    marl_seeds = sorted([d for d in marl_dir.iterdir() if d.is_dir()
                            and d.name.startswith("seed_")])
    marl_arr = []; marl_val = []
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
            marl_arr.append((y_pred + delta).cpu().numpy())
        h = json.loads((sd / "history.json").read_text(encoding="utf-8"))
        marl_val.append(min(e["val_mae"] for e in h))
    marl_arr = np.stack(marl_arr, axis=0)
    print(f"SARL val_mae: {sarl_val}")
    print(f"MARL val_mae: {marl_val}")

    rows = []
    rows.append(metrics(f"{version}_baseline    ", y_pred.cpu().numpy(), y_true_np))

    all_arr = np.concatenate([sarl_arr, marl_arr], axis=0)
    all_val = np.array(sarl_val + marl_val)
    rows.append(metrics(f"{version}_median_all   ", np.median(all_arr, axis=0), y_true_np))
    rows.append(metrics(f"{version}_median_sarl  ", np.median(sarl_arr, axis=0), y_true_np))
    rows.append(metrics(f"{version}_median_marl  ", np.median(marl_arr, axis=0), y_true_np))

    w_all = 1 / (all_val + 1e-6); w_all = w_all / w_all.sum()
    rows.append(metrics(f"{version}_weighted_all ", np.einsum("i...,i->...", all_arr, w_all), y_true_np))
    w_sarl = 1 / (np.array(sarl_val) + 1e-6); w_sarl = w_sarl / w_sarl.sum()
    rows.append(metrics(f"{version}_weighted_sarl", np.einsum("i...,i->...", sarl_arr, w_sarl), y_true_np))

    rows.append(metrics(f"{version}_mean_sarl    ", np.mean(sarl_arr, axis=0), y_true_np))
    best_sarl_i = int(np.argmin(sarl_val))
    rows.append(metrics(f"{version}_best_sarl    ", sarl_arr[best_sarl_i], y_true_np))

    sorted_all = np.sort(all_arr, axis=0)
    rows.append(metrics(f"{version}_trim_mean    ", sorted_all[1:-1].mean(axis=0), y_true_np))
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    runs = ROOT / "data" / "runs"

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    marl_cfg = full_cfg["marl_pro"]
    marl_cfg["agent_features"].setdefault("seq_out",
                                              marl_cfg["agent_features"]["seq_hidden"])

    rows = []
    # v2 : D5 base
    test_v2 = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/"
                       "07_StepF__Export_RL_Windows_Pro/02_outputs/"
                       "switch_rl_dataset_pro_v3_norm_fixed/rl_switch_windows_test.npz"))
    rows += eval_version("v2", runs / "sarl_v2", runs / "marl_v2", test_v2, device, marl_cfg)
    # v3 : MED-NN base
    test_v3 = Path("C:/Users/hayan/Desktop/Code_Final_IA7/Livraison/v2/Input/"
                     "rl_windows_v4/rl_switch_windows_v4_test.npz")
    rows += eval_version("v3", runs / "sarl_v3", runs / "marl_v3", test_v3, device, marl_cfg)

    print("\n\n  strategy                    MAE       RMSE      R^2")
    print("  " + "-" * 60)
    for name, mae, rmse, r2 in rows:
        print(f"  {name}    {mae:.5f}   {rmse:.5f}   {r2:.5f}")


if __name__ == "__main__":
    main()
