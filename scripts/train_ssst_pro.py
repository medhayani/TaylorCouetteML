"""Train SSST_PRO (PyTorch port of code4 SSST, scaled up).

Supervised regression on Ta_norm_resampled with switch + grad + curv + spectral
auxiliary losses.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import set_all_seeds, get_logger
from data_pipeline.dataset import NeptuneProfileDataset
from ssst_pro.ssst_model import SSSTProSurrogate


def collate(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def evaluate(model, loader, device):
    model.eval()
    s, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            losses = model.compute_loss(b)
            pred = model.predict(b["ctx"], b["s"])
            mae += float((b["ta_true"] - pred).abs().mean()) * b["ta_true"].size(0)
            s += float(losses["loss"]) * b["ta_true"].size(0)
            n += b["ta_true"].size(0)
    return s / max(n, 1), mae / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile_csv", required=True, type=Path)
    ap.add_argument("--out_dir", default=Path("../data/runs_pro/ssst_pro"), type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    log = get_logger("ssst_pro.train")
    set_all_seeds(args.seed, deterministic=False)
    full_cfg = yaml.safe_load((ROOT / "configs" / "sizes.yaml").read_text(encoding="utf-8"))
    cfg = full_cfg["ssst_pro"]; cfg.setdefault("ctx_dim", 24)
    out = args.out_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = NeptuneProfileDataset(args.profile_csv.resolve(), split="train")
    val_ds = NeptuneProfileDataset(args.profile_csv.resolve(), split="val",
                                    ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    cfg["ctx_dim"] = train_ds.in_dim
    model = SSSTProSurrogate(cfg).to(device)
    log.info(f"train={len(train_ds)}  val={len(val_ds)}  "
             f"params={sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    log.info(f"epochs={args.epochs}  batch={args.batch}  out_dir={out}")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                     collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    history, best = [], (float("inf"), float("inf"))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        agg = {"loss": 0.0, "loss_data": 0.0, "loss_grad": 0.0, "loss_curv": 0.0,
               "loss_switch": 0.0, "loss_spectral": 0.0}; n = 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            losses = model.compute_loss(b)
            optim.zero_grad(); losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
            for k in agg: agg[k] += float(losses[k])
            n += 1
        sched.step()
        tr = {k: agg[k] / max(n, 1) for k in agg}
        vloss, vmae = evaluate(model, vl, device)
        log.info(f"ep {ep:3d}/{args.epochs}  loss={tr['loss']:.4f}  "
                 f"data={tr['loss_data']:.4f}  switch={tr['loss_switch']:.4f}  "
                 f"val_loss={vloss:.4f}  val_mae={vmae:.4f}")
        history.append({"epoch": ep, "train": tr, "val_loss": vloss, "val_mae": vmae,
                         "lr": optim.param_groups[0]["lr"]})
        if vmae < best[1]:
            best = (vloss, vmae)
            torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                         "ctx_mean": train_ds.ctx_mean.tolist(),
                         "ctx_std": train_ds.ctx_std.tolist()}, out / "best.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "ctx_mean": train_ds.ctx_mean.tolist(),
                 "ctx_std": train_ds.ctx_std.tolist()}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info(f"DONE {(time.time()-t0)/60:.1f} min  best_val_mae={best[1]:.4f}")


if __name__ == "__main__":
    main()
