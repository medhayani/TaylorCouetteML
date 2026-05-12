"""Train SARL_PRO v2 — much longer training, multi-seed ensemble.

Improvements vs train_sarl_pro.py:
- 1500 epochs (vs 80)
- 5 seeds for ensemble (vs 1)
- Cosine annealing with warm restarts
- Validation-set monitoring (save best on val MAE)
- Per-seed checkpoint
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

from data_pipeline.dataset import HydraWindowsDataset
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro


def train_one_seed(seed, args, cfg, device, train_ds, val_ds, out_seed):
    torch.manual_seed(seed); np.random.seed(seed)
    extractor = SARLProFeatureExtractor(
        obs_seq_dim=train_ds.obs_seq.shape[2],
        obs_seq_T=train_ds.obs_seq.shape[1],
        static_dim=train_ds.static_vec.shape[1],
        cfg=cfg["feature_extractor"],
    ).to(device)
    action_dim = train_ds.y_true.shape[1]
    sac = SACPro(feature_dim=extractor.out_dim, action_dim=action_dim,
                  actor_layers=cfg["actor_layers"],
                  critic_layers=cfg["critic_layers"]).to(device)

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch)
    params = list(extractor.parameters()) + list(sac.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=max(args.epochs // 3, 50), T_mult=2)

    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        extractor.train(); sac.train()
        ag, an, bc_l, c_l = 0.0, 0, 0.0, 0.0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            feat = extractor(b["obs_seq"], b["static_vec"])
            target_action = (b["y_true"] - b["y_pred"]).clamp(-2.0, 2.0)
            mu = sac.actor.mean(sac.actor.body(feat))
            actor_bc = F.mse_loss(torch.tanh(mu), target_action)
            with torch.no_grad():
                a = torch.tanh(mu)
                reward = -((b["y_pred"] + a - b["y_true"]) ** 2).mean(dim=-1)
            q1, q2 = sac.critic(feat, target_action.detach())
            crit = F.mse_loss(q1, reward) + F.mse_loss(q2, reward)
            loss = actor_bc + 0.1 * crit
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step(); sched.step(ep + an / max(len(tl), 1))
            ag += float(loss); bc_l += float(actor_bc); c_l += float(crit); an += 1

        # Validation
        extractor.eval(); sac.eval()
        v_mae, vn = 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = {k: v.to(device) for k, v in b.items()}
                feat = extractor(b["obs_seq"], b["static_vec"])
                mu = sac.actor.mean(sac.actor.body(feat))
                a = torch.tanh(mu)
                y_corr = b["y_pred"] + a
                v_mae += float((y_corr - b["y_true"]).abs().sum())
                vn += b["y_true"].numel()
        v_mae /= max(vn, 1)
        history.append({"epoch": ep, "loss": ag / max(an, 1),
                          "bc_loss": bc_l / max(an, 1),
                          "crit_loss": c_l / max(an, 1),
                          "val_mae": v_mae})
        if v_mae < best_val:
            best_val = v_mae
            torch.save({"extractor": extractor.state_dict(), "sac": sac.state_dict(),
                          "cfg": cfg, "action_dim": action_dim,
                          "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                          "static_dim": train_ds.static_vec.shape[1]},
                         out_seed / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(f"  seed {seed}  ep {ep:4d}/{args.epochs}  loss={ag/max(an,1):.5f}  "
                    f"val_mae={v_mae:.5f}", flush=True)

    torch.save({"extractor": extractor.state_dict(), "sac": sac.state_dict(),
                  "cfg": cfg, "action_dim": action_dim,
                  "obs_seq_shape": tuple(train_ds.obs_seq.shape),
                  "static_dim": train_ds.static_vec.shape[1]}, out_seed / "last.pt")
    (out_seed / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"  seed {seed} done in {(time.time()-t0)/60:.1f} min  best_val_mae={best_val:.5f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_train", required=True, type=Path)
    ap.add_argument("--windows_val", required=True, type=Path)
    ap.add_argument("--out_dir", type=Path)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_base", type=int, default=42)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = ROOT / "data" / "runs" / "sarl_v2"
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["sarl_pro"]
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    train_ds = HydraWindowsDataset(args.windows_train.resolve())
    val_ds = HydraWindowsDataset(args.windows_val.resolve())
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}", flush=True)

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n===== SARL_v2 SEED {seed} ({s_idx+1}/{args.n_seeds}) =====", flush=True)
        out_seed = out / f"seed_{seed}"; out_seed.mkdir(parents=True, exist_ok=True)
        train_one_seed(seed, args, cfg, device, train_ds, val_ds, out_seed)
    print("ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
