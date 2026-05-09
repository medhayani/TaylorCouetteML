"""Train NEPTUNE_PRO ensemble (default: M=3 members, 100 epochs each).

Smaller defaults than configs/sizes.yaml so it actually finishes on CPU.
Bump --epochs / --ensemble for the full run.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import NeptuneProfileDataset
from models.fno_latent_diffusion.trainer import NeptuneProSurrogate


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def evaluate(model, loader, device):
    model.eval()
    s, n = 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            s += float(model.compute_loss(b)["loss"]) * b["ta_true"].size(0)
            n += b["ta_true"].size(0)
    return s / max(n, 1)


def build_cfg(in_dim: int, full_cfg: dict) -> dict:
    n = full_cfg["neptune_pro"]
    n["context"]["in_dim"] = in_dim
    return n


def train_member(member_id, profile_csv, out_dir, full_cfg, epochs, batch, lr,
                  weight_decay, seed, device, log):
    set_all_seeds(seed, deterministic=False)
    train_ds = NeptuneProfileDataset(profile_csv, split="train")
    val_ds = NeptuneProfileDataset(profile_csv, split="val",
                                    ctx_mean=train_ds.ctx_mean,
                                    ctx_std=train_ds.ctx_std)
    cfg = build_cfg(train_ds.in_dim, full_cfg)
    model = NeptuneProSurrogate(cfg).to(device)
    log.info(f"[m{member_id}] train={len(train_ds)} val={len(val_ds)} "
             f"params={sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    tl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                     collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch, collate_fn=collate)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    md = out_dir / f"member_{member_id:02d}"
    md.mkdir(parents=True, exist_ok=True)
    history, best_val = [], float("inf")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        agg = {"loss": 0.0, "loss_diff": 0.0, "loss_switch": 0.0,
               "loss_pino": 0.0, "loss_spectral": 0.0}; n = 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            losses = model.compute_loss(b)
            optim.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
            for k in agg: agg[k] += float(losses[k])
            n += 1
        sched.step()
        tr = {k: agg[k] / max(n, 1) for k in agg}
        vloss = evaluate(model, vl, device)
        log.info(f"[m{member_id}] ep {ep:3d}/{epochs}  loss={tr['loss']:.4f} "
                 f"diff={tr['loss_diff']:.4f}  switch={tr['loss_switch']:.4f} "
                 f"val={vloss:.4f}")
        history.append({"epoch": ep, "train": tr, "val_loss": vloss,
                         "lr": optim.param_groups[0]["lr"]})
        if vloss < best_val:
            best_val = vloss
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                         "ctx_mean": train_ds.ctx_mean.tolist(),
                         "ctx_std": train_ds.ctx_std.tolist()}, md / "best.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "ctx_mean": train_ds.ctx_mean.tolist(),
                 "ctx_std": train_ds.ctx_std.tolist()}, md / "last.pt")
    (md / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info(f"[m{member_id}] done {(time.time()-t0)/60:.1f} min  best_val={best_val:.4f}")
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_csv", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("../data/runs_pro/neptune_pro"), type=Path)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight_decay", type=float, default=1.0e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = get_logger("neptune_pro.train")
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    log.info(f"out_dir={out}  ensemble={args.ensemble}  epochs={args.epochs}  batch={args.batch}")

    summary = []
    for m in range(args.ensemble):
        bv = train_member(m, args.profile_csv.resolve(), out, full_cfg,
                            args.epochs, args.batch, args.lr, args.weight_decay,
                            args.seed + m, torch.device(args.device), log)
        summary.append({"member": m, "best_val": bv})
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("NEPTUNE_PRO ALL DONE")


if __name__ == "__main__":
    main()
