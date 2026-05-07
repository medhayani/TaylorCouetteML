"""Train MARL_PRO via joint behavior cloning with cross-agent attention.

Three SAC agents (Localizer / Shape / Geometry) share an attention layer
and are jointly trained to reproduce the optimal correction split into:
    Localizer : (delta_center, delta_amplitude)
    Shape     : (bias, scale)
    Geometry  : (left_slope, right_slope, width, asymmetry)
The combined correction is reconstructed and trained against y_true - y_pred.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import HydraWindowsDataset
from marl_pro.marl_model import MARLProSystem


def reconstruct_correction(actions, T: int) -> torch.Tensor:
    """Combine the 3 agents' actions into a length-T correction.

    Localizer: (dc, da) -> Gaussian bump at center, sigma fixed
    Shape:     (b, sc)  -> bias + multiplicative scale on Localizer bump
    Geometry:  (sl, sr, w, asym) -> additive triangular asymmetry
    """
    a_loc, a_sh, a_geo = actions
    B = a_loc.size(0)
    s = torch.linspace(0, 1, T, device=a_loc.device).unsqueeze(0).expand(B, T)
    dc = a_loc[:, 0:1] * 0.5 + 0.5         # center in [0,1]
    da = a_loc[:, 1:2] * 0.5               # amplitude in [-0.5, 0.5]
    bump = da * torch.exp(-((s - dc) ** 2) / (2 * 0.05 ** 2))
    bias, scale = a_sh[:, 0:1], a_sh[:, 1:2]
    sh = bias + scale * bump
    sl, sr, w, asy = a_geo[:, 0:1], a_geo[:, 1:2], a_geo[:, 2:3], a_geo[:, 3:4]
    geo = (sl * (s - dc).clamp(max=0) + sr * (s - dc).clamp(min=0)
           + w * (s - dc).abs() + asy * (s - 0.5))
    return sh + 0.1 * geo                    # (B, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_train", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("../data/runs_pro/marl_pro"), type=Path)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = get_logger("marl_pro.train")
    set_all_seeds(args.seed, deterministic=False)
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["marl_pro"]
    cfg["agent_features"].setdefault("seq_out", cfg["agent_features"]["seq_hidden"])
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = HydraWindowsDataset(args.windows_train.resolve())
    log.info(f"train windows: N={len(train_ds)}, "
             f"obs_seq={tuple(train_ds.obs_seq.shape)}")

    marl = MARLProSystem(obs_seq_dim=train_ds.obs_seq.shape[2],
                          obs_seq_T=train_ds.obs_seq.shape[1],
                          static_dim=train_ds.static_vec.shape[1],
                          cfg=cfg).to(device)
    log.info(f"params={sum(p.numel() for p in marl.parameters())/1e6:.2f} M  "
             f"epochs={args.epochs}  batch={args.batch}")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    optim = torch.optim.AdamW(marl.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []; t0 = time.time(); best_mae = float("inf")
    T = train_ds.y_true.shape[1]

    for ep in range(1, args.epochs + 1):
        ag, an = 0.0, 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            actions = marl.sample_actions(b["obs_seq"], b["static_vec"])
            delta_pred = reconstruct_correction(actions, T)
            target = (b["y_true"] - b["y_pred"]).clamp(-2.0, 2.0)
            loss = F.mse_loss(delta_pred, target)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(marl.parameters(), 1.0); optim.step()
            ag += float(loss); an += 1
        mae = float((delta_pred - target).abs().mean().detach())
        log.info(f"ep {ep:3d}/{args.epochs}  loss={ag/max(an,1):.5f}  "
                 f"residual_mae={mae:.5f}")
        history.append({"epoch": ep, "loss": ag / max(an, 1), "residual_mae": mae})
        if mae < best_mae:
            best_mae = mae
            torch.save({"state_dict": marl.state_dict(), "cfg": cfg,
                         "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                         "static_dim": train_ds.static_vec.shape[1]},
                        out / "best.pt")
    torch.save({"state_dict": marl.state_dict(), "cfg": cfg,
                 "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                 "static_dim": train_ds.static_vec.shape[1]}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info(f"MARL_PRO done {(time.time()-t0)/60:.1f} min  best_mae={best_mae:.5f}")


if __name__ == "__main__":
    main()
