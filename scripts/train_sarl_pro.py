"""Train SARL_PRO via behavior cloning of the optimal correction.

Pure-PyTorch supervised regression — actor predicts delta = y_true - y_pred,
critic regresses the negative reward. No SB3, no replay buffer, fast on CPU.
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
sys.path.insert(0, str(ROOT))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import HydraWindowsDataset
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_train", required=True, type=Path)
    ap.add_argument("--windows_val", default=None, type=Path)
    ap.add_argument("--out_dir", default=Path("../data/runs_pro/sarl_pro"), type=Path)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = get_logger("sarl_pro.train")
    set_all_seeds(args.seed, deterministic=False)
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["sarl_pro"]
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = HydraWindowsDataset(args.windows_train.resolve())
    log.info(f"train windows: N={len(train_ds)}, "
             f"obs_seq={tuple(train_ds.obs_seq.shape)}")

    extractor = SARLProFeatureExtractor(
        obs_seq_dim=train_ds.obs_seq.shape[2],
        obs_seq_T=train_ds.obs_seq.shape[1],
        static_dim=train_ds.static_vec.shape[1],
        cfg=cfg["feature_extractor"],
    ).to(device)
    action_dim = train_ds.y_true.shape[1]                  # full window correction
    sac = SACPro(feature_dim=extractor.out_dim, action_dim=action_dim,
                  actor_layers=cfg["actor_layers"],
                  critic_layers=cfg["critic_layers"]).to(device)
    log.info(f"params: extractor={sum(p.numel() for p in extractor.parameters())/1e6:.2f}M  "
             f"sac={sum(p.numel() for p in sac.parameters())/1e6:.2f}M")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    optim_e = torch.optim.AdamW(extractor.parameters(), lr=args.lr)
    optim_a = torch.optim.AdamW(sac.actor.parameters(), lr=args.lr)
    optim_c = torch.optim.AdamW(sac.critic.parameters(), lr=args.lr)

    history = []; t0 = time.time(); best_mae = float("inf")
    for ep in range(1, args.epochs + 1):
        ag, an = 0.0, 0
        bc_l, c_l = 0.0, 0.0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            feat = extractor(b["obs_seq"], b["static_vec"])
            target_action = (b["y_true"] - b["y_pred"]).clamp(-2.0, 2.0)
            # Behavior-cloning loss on actor: deterministic mean
            mu = sac.actor.mean(sac.actor.body(feat))
            actor_bc = F.mse_loss(torch.tanh(mu), target_action)
            # Critic: regress the negative MSE between (y_pred + a) and y_true
            with torch.no_grad():
                a = torch.tanh(mu)
                reward = -((b["y_pred"] + a - b["y_true"]) ** 2).mean(dim=-1)
            q1, q2 = sac.critic(feat, target_action.detach())
            crit = F.mse_loss(q1, reward) + F.mse_loss(q2, reward)
            loss = actor_bc + 0.1 * crit
            optim_e.zero_grad(); optim_a.zero_grad(); optim_c.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor.parameters(), 1.0)
            optim_e.step(); optim_a.step(); optim_c.step()
            ag += float(loss); an += 1
            bc_l += float(actor_bc); c_l += float(crit)
        mae = float(((target_action - torch.tanh(mu)).abs()).mean().detach())
        log.info(f"ep {ep:3d}/{args.epochs}  loss={ag/max(an,1):.5f}  "
                 f"bc={bc_l/max(an,1):.5f}  crit={c_l/max(an,1):.5f}  "
                 f"residual_mae={mae:.5f}")
        history.append({"epoch": ep, "loss": ag / max(an, 1),
                         "bc_loss": bc_l / max(an, 1), "crit_loss": c_l / max(an, 1),
                         "residual_mae": mae})
        if mae < best_mae:
            best_mae = mae
            torch.save({"extractor": extractor.state_dict(),
                         "sac": sac.state_dict(),
                         "cfg": cfg, "action_dim": action_dim,
                         "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                         "static_dim": train_ds.static_vec.shape[1]},
                        out / "best.pt")
    torch.save({"extractor": extractor.state_dict(), "sac": sac.state_dict(),
                 "cfg": cfg, "action_dim": action_dim,
                 "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                 "static_dim": train_ds.static_vec.shape[1]}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info(f"SARL_PRO done {(time.time()-t0)/60:.1f} min  best_mae={best_mae:.5f}")


if __name__ == "__main__":
    main()
