"""MARL refiner inference + figure on test windows.

Loads data/runs/marl/best.pt and the test split rl_switch_windows_test.npz,
computes the refined prediction y_corr = y_pred + correction, and shows the
improvement vs the base y_pred.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.dataset import HydraWindowsDataset
from models.marl_3sac.marl_model import MARLProSystem


def reconstruct_correction(actions, T):
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
    win_test = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/07_StepF__Export_RL_Windows_Pro")
                / "02_outputs" / "switch_rl_dataset_pro_v3_norm_fixed"
                / "rl_switch_windows_test.npz")
    ckpt_path = ROOT / "data" / "runs" / "marl" / "best.pt"
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    test_ds = HydraWindowsDataset(win_test)
    print(f"test windows: {len(test_ds)}, T = {test_ds.y_true.shape[1]}")

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["marl_pro"]
    cfg["agent_features"].setdefault("seq_out", cfg["agent_features"]["seq_hidden"])

    marl = MARLProSystem(obs_seq_dim=test_ds.obs_seq.shape[2],
                          obs_seq_T=test_ds.obs_seq.shape[1],
                          static_dim=test_ds.static_vec.shape[1],
                          cfg=cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    marl.load_state_dict(state["state_dict"])
    marl.eval()

    T = test_ds.y_true.shape[1]
    obs = torch.from_numpy(test_ds.obs_seq).to(device)
    sv = torch.from_numpy(test_ds.static_vec).to(device)
    y_pred = torch.from_numpy(test_ds.y_pred).to(device)
    y_true = torch.from_numpy(test_ds.y_true).to(device)

    with torch.no_grad():
        actions = marl.sample_actions(obs, sv)
        delta = reconstruct_correction(actions, T)
        y_corr = y_pred + delta

    # Metrics
    mae_base = (y_pred - y_true).abs().mean().item()
    mae_corr = (y_corr - y_true).abs().mean().item()
    rmse_base = ((y_pred - y_true) ** 2).mean().sqrt().item()
    rmse_corr = ((y_corr - y_true) ** 2).mean().sqrt().item()
    print(f"BASELINE  MAE = {mae_base:.5f}  RMSE = {rmse_base:.5f}")
    print(f"REFINED   MAE = {mae_corr:.5f}  RMSE = {rmse_corr:.5f}")
    improvement = (mae_base - mae_corr) / mae_base * 100
    print(f"Improvement: {improvement:+.1f}% vs baseline")

    # 12-panel figure across log10E
    Es = test_ds.E
    log10E = np.log10(np.clip(Es, 1e-30, None))
    order = np.argsort(log10E)
    pick = order[np.linspace(0, len(order) - 1, 12).astype(int)]

    yp = y_pred.cpu().numpy()
    yt = y_true.cpu().numpy()
    yc = y_corr.cpu().numpy()
    grid = test_ds.local_grid

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), dpi=110)
    for ax, idx in zip(axes.ravel(), pick):
        ax.plot(grid[idx], yt[idx], "k.", ms=3, label="truth")
        ax.plot(grid[idx], yp[idx], "--", lw=1.2, color="gray", label="base y_pred")
        ax.plot(grid[idx], yc[idx], "-", lw=1.6, color="tab:red",
                  label="MARL refined")
        e_val = float(Es[idx])
        mae_b = np.abs(yp[idx] - yt[idx]).mean()
        mae_c = np.abs(yc[idx] - yt[idx]).mean()
        ax.set_title(f"E={e_val:.4g}  base={mae_b:.4f}  refined={mae_c:.4f}",
                       fontsize=9)
        ax.set_xlabel("k local"); ax.set_ylabel("Ta")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"MARL (3-SAC + Cross-Attn) 500 epochs — test ({len(test_ds)} windows)\n"
                  f"BASELINE MAE={mae_base:.4f} -> REFINED MAE={mae_corr:.4f} ({improvement:+.1f}%)",
                  fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "marl_predictions_test.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


if __name__ == "__main__":
    main()
