"""SARL refiner inference + figure on test windows."""
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
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro


def main():
    win_test = (Path("C:/Users/hayan/Desktop/Code_Final_IA7/code4/07_StepF__Export_RL_Windows_Pro")
                / "02_outputs" / "switch_rl_dataset_pro_v3_norm_fixed"
                / "rl_switch_windows_test.npz")
    ckpt_path = ROOT / "data" / "runs" / "sarl" / "best.pt"
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = HydraWindowsDataset(win_test)
    print(f"test windows: {len(test_ds)}, T = {test_ds.y_true.shape[1]}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["cfg"]; obs = state["obs_seq_shape"]
    extractor = SARLProFeatureExtractor(obs[2], obs[1], state["static_dim"],
                                          cfg["feature_extractor"]).to(device)
    sac = SACPro(feature_dim=extractor.out_dim, action_dim=state["action_dim"],
                   actor_layers=cfg["actor_layers"],
                   critic_layers=cfg["critic_layers"]).to(device)
    extractor.load_state_dict(state["extractor"])
    sac.load_state_dict(state["sac"])
    extractor.eval(); sac.eval()

    obs_t = torch.from_numpy(test_ds.obs_seq).to(device)
    sv_t = torch.from_numpy(test_ds.static_vec).to(device)
    y_pred = torch.from_numpy(test_ds.y_pred).to(device)
    y_true = torch.from_numpy(test_ds.y_true).to(device)

    with torch.no_grad():
        feat = extractor(obs_t, sv_t)
        mu = sac.actor.mean(sac.actor.body(feat))
        correction = torch.tanh(mu)
        y_corr = y_pred + correction

    mae_base = (y_pred - y_true).abs().mean().item()
    mae_corr = (y_corr - y_true).abs().mean().item()
    rmse_base = ((y_pred - y_true) ** 2).mean().sqrt().item()
    rmse_corr = ((y_corr - y_true) ** 2).mean().sqrt().item()
    print(f"BASELINE  MAE = {mae_base:.5f}  RMSE = {rmse_base:.5f}")
    print(f"REFINED   MAE = {mae_corr:.5f}  RMSE = {rmse_corr:.5f}")
    improvement = (mae_base - mae_corr) / mae_base * 100
    print(f"Improvement: {improvement:+.1f}% vs baseline")

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
        ax.plot(grid[idx], yc[idx], "-", lw=1.6, color="tab:blue",
                  label="SARL refined")
        e_val = float(Es[idx])
        mae_b = np.abs(yp[idx] - yt[idx]).mean()
        mae_c = np.abs(yc[idx] - yt[idx]).mean()
        ax.set_title(f"E={e_val:.4g}  base={mae_b:.4f}  refined={mae_c:.4f}",
                       fontsize=9)
        ax.set_xlabel("k local"); ax.set_ylabel("Ta")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"SARL (Conv1D + AttnPool SAC) 500 epochs — test ({len(test_ds)} windows)\n"
                  f"BASELINE MAE={mae_base:.4f} -> REFINED MAE={mae_corr:.4f} ({improvement:+.1f}%)",
                  fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "sarl_predictions_test.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


if __name__ == "__main__":
    main()
