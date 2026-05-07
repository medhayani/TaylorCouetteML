"""Inference of the 3 strong refiners (SARL, MARL, HYDRA) on canonical 41-pt grid.

Uses the strong PT-SSST canonical predictions as the surrogate baseline,
applies each refiner's correction in the 49-pt window, and projects back.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CODE7 = ROOT.parent / "code7" / "src"
sys.path.insert(0, str(CODE7))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import HydraWindowsDataset
from sarl_pro.feature_extractor import SARLProFeatureExtractor
from sarl_pro.sac_pro import SACPro
from marl_pro.marl_model import MARLProSystem
from hydra_marl_pro.policies.diffusion_policy import DiffusionPolicy
from hydra_marl_pro.communication.gat import GATCommunication
from hydra_marl_pro.agents.validator import ValidatorAgent


def reconstruct_correction_marl(actions, T):
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


def project_to_canonical(refined_rows, baseline_csv, baseline_col):
    """Project refined predictions to the 41-pt canonical grid.

    Default per point = baseline (PT-SSST_strong). For points inside a window,
    use the refined value.
    """
    canon = pd.read_csv(baseline_csv).sort_values(
        ["split","E","branch_local_id","s"]).reset_index(drop=True)
    out = canon.copy()
    out["Ta_pred"] = canon[baseline_col].copy()
    by_branch = {}
    for r in refined_rows:
        key = (r["split"], float(r["E"]), int(r["branch_local_id"]))
        by_branch.setdefault(key, []).append((r["local_grid"], r["y_refined"]))
    n_replaced = 0
    for (split, E, b), entries in by_branch.items():
        mask = ((out["split"]==split) & (out["E"]==E)
                 & (out["branch_local_id"]==int(b)))
        if not mask.any():
            continue
        sub = out[mask].sort_values("s")
        s_canon = sub["s"].to_numpy(np.float32)
        merged = sub[baseline_col].to_numpy(np.float32).copy()
        for grid, y_ref in entries:
            for j, s_pt in enumerate(s_canon):
                idx = int(np.argmin(np.abs(grid - s_pt)))
                if abs(grid[idx] - s_pt) < 0.05:
                    merged[j] = float(y_ref[idx]); n_replaced += 1
        out.loc[sub.index, "Ta_pred"] = merged
    return out, n_replaced


def infer_sarl(ckpt, windows_dir, device, log):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; obs = ck["obs_seq_shape"]
    ext = SARLProFeatureExtractor(obs[2], obs[1], ck["static_dim"],
                                    cfg["feature_extractor"]).to(device)
    ext.load_state_dict(ck["extractor"]); ext.eval()
    sac = SACPro(ext.out_dim, ck["action_dim"], cfg["actor_layers"],
                  cfg["critic_layers"]).to(device)
    sac.load_state_dict(ck["sac"]); sac.eval()
    out = []
    for split in ["train","val","test"]:
        npz = windows_dir / f"rl_switch_windows_{split}.npz"
        if not npz.exists(): continue
        ds = HydraWindowsDataset(npz)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        for b in loader:
            b = {k:v.to(device) for k,v in b.items()}
            with torch.no_grad():
                feat = ext(b["obs_seq"], b["static_vec"])
                mu = sac.actor.mean(sac.actor.body(feat))
                refined = b["y_pred"] + torch.tanh(mu)
            for j in range(refined.size(0)):
                out.append({"split":split, "E":float(b["E"][j]),
                             "branch_local_id":int(b["branch_local_id"][j]),
                             "local_grid":b["local_grid"][j].cpu().numpy(),
                             "y_refined":refined[j].cpu().numpy()})
    return out


def infer_marl(ckpt, windows_dir, device, log):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg["agent_features"].setdefault("seq_out", cfg["agent_features"]["seq_hidden"])
    obs = ck["obs_seq_shape"]
    marl = MARLProSystem(obs[2], obs[1], ck["static_dim"], cfg).to(device)
    marl.load_state_dict(ck["state_dict"]); marl.eval()
    out = []
    for split in ["train","val","test"]:
        npz = windows_dir / f"rl_switch_windows_{split}.npz"
        if not npz.exists(): continue
        ds = HydraWindowsDataset(npz)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        T = ds.y_true.shape[1]
        for b in loader:
            b = {k:v.to(device) for k,v in b.items()}
            with torch.no_grad():
                actions = marl.sample_actions(b["obs_seq"], b["static_vec"])
                refined = b["y_pred"] + reconstruct_correction_marl(actions, T)
            for j in range(refined.size(0)):
                out.append({"split":split, "E":float(b["E"][j]),
                             "branch_local_id":int(b["branch_local_id"][j]),
                             "local_grid":b["local_grid"][j].cpu().numpy(),
                             "y_refined":refined[j].cpu().numpy()})
    return out


def infer_hydra(ckpt, windows_dir, device, log):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dim = ck["state_dim"]; a_dims = ck["action_dims"]
    policies = [DiffusionPolicy(action_dim=a_dims[i], state_dim=state_dim,
                                  msg_dim=state_dim, num_timesteps=200).to(device)
                for i in range(6)]
    for i, sd in enumerate(ck["policies"]):
        policies[i].load_state_dict(sd); policies[i].eval()
    gat = GATCommunication(dim=state_dim, num_rounds=4, num_heads=1).to(device)
    gat.load_state_dict(ck["gat"]); gat.eval()
    val = ValidatorAgent(state_dim, a_dims[6], 256).to(device)
    val.load_state_dict(ck["validator"]); val.eval()
    out = []
    for split in ["train","val","test"]:
        npz = windows_dir / f"rl_switch_windows_{split}.npz"
        if not npz.exists(): continue
        ds = HydraWindowsDataset(npz)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        for b in loader:
            b = {k:v.to(device) for k,v in b.items()}
            B = b["y_pred"].size(0)
            state = torch.cat([b["obs_seq"].reshape(B,-1), b["static_vec"],
                                b["y_pred"]], dim=-1)
            with torch.no_grad():
                h = state.unsqueeze(1).expand(-1,7,-1).contiguous()
                h = gat(h)
                smooth_field = policies[3].sample(state, h[:,3,:], num_steps=20)
                gates = val(state)
                refined = b["y_pred"] + gates[:,3:4] * smooth_field
            for j in range(refined.size(0)):
                out.append({"split":split, "E":float(b["E"][j]),
                             "branch_local_id":int(b["branch_local_id"][j]),
                             "local_grid":b["local_grid"][j].cpu().numpy(),
                             "y_refined":refined[j].cpu().numpy()})
    return out


def per_split_metrics(df, true_col, pred_col):
    rows = []
    for split, g in df.groupby("split"):
        err = g[true_col].to_numpy() - g[pred_col].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((g[true_col] - g[true_col].mean())**2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({"split":split, "mae":mae, "rmse":rmse, "r2":r2})
    return pd.DataFrame(rows)


def main():
    log = get_logger("refiners_strong_inf")
    set_all_seeds(42, deterministic=False)
    device = torch.device("cpu")
    runs_strong = ROOT.parent / "data" / "runs_strong"
    out_dir = runs_strong / "inference_refiners_strong"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = (ROOT.parent / "code4" / "07_StepF__Export_RL_Windows_Pro"
                    / "02_outputs" / "switch_rl_dataset_pro_v3_norm_fixed")
    canon_pt = runs_strong / "ptssst_long" / "canonical_ssst_pro.csv"
    if not canon_pt.exists():
        log.error(f"missing {canon_pt}; run Phase A first"); sys.exit(2)

    # Add Ta_true to canon_pt for projection use
    df_pt = pd.read_csv(canon_pt)
    log.info(f"PT-SSST canonical: {len(df_pt)} rows, columns: {list(df_pt.columns)[:8]}")

    for name, ckpt, infer_fn in [
        ("sarl_strong",  runs_strong / "refiners" / "sarl"  / "best.pt",          infer_sarl),
        ("marl_strong",  runs_strong / "refiners" / "marl"  / "best.pt",          infer_marl),
        ("hydra_strong", runs_strong / "refiners" / "hydra" / "mappo_phase3.pt", infer_hydra),
    ]:
        log.info(f"=== {name} ===")
        refined = infer_fn(ckpt, windows_dir, device, log)
        df, n_rep = project_to_canonical(refined, canon_pt, "Ta_ssst_pro")
        df.to_csv(out_dir / f"canonical_{name}.csv", index=False)
        log.info(f"  refined {len(refined)} windows, replaced {n_rep} pts")
        m = per_split_metrics(df, "Ta_true", "Ta_pred")
        m.to_csv(out_dir / f"metrics_{name}.csv", index=False)
        for _, r in m.iterrows():
            log.info(f"  {r['split']:<6} MAE={r['mae']:.5f}  RMSE={r['rmse']:.5f}  R2={r['r2']:.5f}")


if __name__ == "__main__":
    main()
