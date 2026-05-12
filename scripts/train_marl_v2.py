"""Train MARL_PRO v2 — much longer training, multi-seed ensemble.

Improvements vs train_marl_pro.py:
- 1500 epochs (vs 80)
- 5 seeds for ensemble
- Cosine annealing with warm restarts
- Validation-set monitoring
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
from models.marl_3sac.marl_model import MARLProSystem


def reconstruct_correction(actions, T: int) -> torch.Tensor:
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


def train_one_seed(seed, args, cfg, device, train_ds, val_ds, out_seed):
    torch.manual_seed(seed); np.random.seed(seed)
    marl = MARLProSystem(obs_seq_dim=train_ds.obs_seq.shape[2],
                            obs_seq_T=train_ds.obs_seq.shape[1],
                            static_dim=train_ds.static_vec.shape[1],
                            cfg=cfg).to(device)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch)
    optim = torch.optim.AdamW(marl.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=max(args.epochs // 3, 50), T_mult=2)

    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        marl.train()
        ag, an = 0.0, 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            target = (b["y_true"] - b["y_pred"]).clamp(-2.0, 2.0)
            T = target.shape[1]
            actions = marl.sample_actions(b["obs_seq"], b["static_vec"])
            delta = reconstruct_correction(actions, T)
            bc_loss = F.mse_loss(delta, target)
            l2_act = sum((a ** 2).mean() for a in actions) * 1e-4
            loss = bc_loss + l2_act
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(marl.parameters(), 1.0)
            optim.step(); sched.step(ep + an / max(len(tl), 1))
            ag += float(loss); an += 1

        # Validation
        marl.eval()
        v_mae, vn = 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = {k: v.to(device) for k, v in b.items()}
                T = b["y_true"].shape[1]
                actions = marl.sample_actions(b["obs_seq"], b["static_vec"])
                delta = reconstruct_correction(actions, T)
                y_corr = b["y_pred"] + delta
                v_mae += float((y_corr - b["y_true"]).abs().sum())
                vn += b["y_true"].numel()
        v_mae /= max(vn, 1)
        history.append({"epoch": ep, "loss": ag / max(an, 1), "val_mae": v_mae})
        if v_mae < best_val:
            best_val = v_mae
            torch.save({"state_dict": marl.state_dict()}, out_seed / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == args.epochs:
            print(f"  seed {seed}  ep {ep:4d}/{args.epochs}  loss={ag/max(an,1):.5f}  "
                    f"val_mae={v_mae:.5f}", flush=True)

    torch.save({"state_dict": marl.state_dict()}, out_seed / "last.pt")
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
        args.out_dir = ROOT / "data" / "runs" / "marl_v2"
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["marl_pro"]
    cfg["agent_features"].setdefault("seq_out", cfg["agent_features"]["seq_hidden"])
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    train_ds = HydraWindowsDataset(args.windows_train.resolve())
    val_ds = HydraWindowsDataset(args.windows_val.resolve())
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}", flush=True)

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n===== MARL_v2 SEED {seed} ({s_idx+1}/{args.n_seeds}) =====", flush=True)
        out_seed = out / f"seed_{seed}"; out_seed.mkdir(parents=True, exist_ok=True)
        train_one_seed(seed, args, cfg, device, train_ds, val_ds, out_seed)
    print("ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
