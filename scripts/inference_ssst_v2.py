"""SSST inference + figure on the held-out test split.

Loads the 500-epoch checkpoint at data/runs/ssst/best.pt, runs the model on
the test branches from data/processed/model_profile_level_dataset.csv, and
saves a 12-panel figure plus a CSV of per-branch MAE.
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

from data_pipeline.dataset import NeptuneProfileDataset
from models.sparse_moe_transformer.ssst_model import SSSTProSurrogate


def main():
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"
    ckpt_path = ROOT / "data" / "runs" / "ssst" / "best.pt"
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["ssst_pro"]
    cfg.setdefault("ctx_dim", 24)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    test_ds = NeptuneProfileDataset(profile_csv, split="test",
                                       ctx_mean=train_ds.ctx_mean,
                                       ctx_std=train_ds.ctx_std)
    cfg["ctx_dim"] = train_ds.in_dim
    print(f"test branches: {len(test_ds)}")

    model = SSSTProSurrogate(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    preds = []
    truths = []
    s_vals = []
    Es = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            ctx = item["ctx"].unsqueeze(0).to(device)
            s = item["s"].unsqueeze(0).to(device)
            ta_pred = model.predict(ctx, s)[0].cpu().numpy()
            preds.append(ta_pred)
            truths.append(item["ta_true"].numpy())
            s_vals.append(item["s"].numpy())
            Es.append(float(item["E"]))

    preds = np.stack(preds)
    truths = np.stack(truths)
    s_vals = np.stack(s_vals)
    Es = np.asarray(Es)

    abs_err = np.abs(preds - truths)
    print(f"MAE_norm (mean over test): {abs_err.mean():.5f}")
    print(f"RMSE_norm: {np.sqrt(((preds - truths)**2).mean()):.5f}")
    per_branch_mae = abs_err.mean(axis=1)

    # 12-panel figure: pick branches across log10(E)
    log10E = np.log10(np.clip(Es, 1e-30, None))
    order = np.argsort(log10E)
    pick_idx = order[np.linspace(0, len(order) - 1, 12).astype(int)]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), dpi=110)
    for ax, idx in zip(axes.ravel(), pick_idx):
        ax.plot(s_vals[idx], truths[idx], "k.", ms=3, label="truth")
        ax.plot(s_vals[idx], preds[idx], "-", lw=1.6, color="tab:orange",
                  label="SSST pred")
        ax.set_title(f"E = {Es[idx]:.4g}   MAE = {per_branch_mae[idx]:.4f}")
        ax.set_xlabel("s"); ax.set_ylabel("Ta_norm")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(loc="upper right")
    fig.suptitle(f"SSST 500 epochs — test set ({len(test_ds)} branches), "
                  f"overall MAE_norm = {abs_err.mean():.4f}", fontsize=12)
    fig.tight_layout()
    out_png = out_dir / "ssst_predictions_test.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


if __name__ == "__main__":
    main()
