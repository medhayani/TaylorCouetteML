"""Train CNP-TC, leak-free AND conditioned on the training neighbours.

Same as train_cnp_leakfree.py, plus: the marginal curve interpolated from the
two nearest TRAINING elasticities (leakfree_descriptors.interpolate_curves) is
given to the model as its context points.  The model therefore learns the
correction to plain interpolation, and never sees the target curve of an
elasticity in its inputs.

Context: n_ctx points evenly spaced along the branch, (k_norm, ta_ctx), with
a small noise on ta_ctx during training.  Targets: all resampled points of the
true curve.  With probability p_zero_shot the context is dropped (the model
keeps a zero-shot mode).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "leakfree"))

from models.cnp_tc.model import CNP_TC
from leakfree_descriptors import interpolate_descriptors, interpolate_curves, check_against_truth
from train_cnp_leakfree import STATIC_FEATURES, build_dataset, split_by_E, smoothness_penalty, masked_mse


class CNPDS(Dataset):
    def __init__(self, items, ctx_curves, ctx_mean=None, ctx_std=None, anc_mean=None, anc_std=None, ctx_noise: float = 0.0):
        self.k = np.stack([it["k_norm"] for it in items]); self.y = np.stack([it["ta_norm"] for it in items])
        self.c = np.stack([ctx_curves[(round(it["E"], 8), it["branch_local_id"])] for it in items]).astype(np.float32)
        self.logE = np.array([it["log10E"] for it in items], dtype=np.float32)
        ctx_arr = np.stack([it["ctx"] for it in items]).astype(np.float32)
        anc_arr = np.stack([it["anchors_raw"] for it in items]).astype(np.float32)
        if ctx_mean is None: ctx_mean = ctx_arr.mean(axis=0); ctx_std = ctx_arr.std(axis=0) + 1e-6
        if anc_mean is None: anc_mean = anc_arr.mean(axis=0); anc_std = anc_arr.std(axis=0) + 1e-6
        self.ctx_mean, self.ctx_std, self.anc_mean, self.anc_std = ctx_mean, ctx_std, anc_mean, anc_std
        self.ctx_norm = np.clip(np.nan_to_num((ctx_arr - ctx_mean) / ctx_std, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        self.anc_norm = np.clip(np.nan_to_num((anc_arr - anc_mean) / anc_std, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        self.ctx_noise = float(ctx_noise)
    def __len__(self): return len(self.k)
    def __getitem__(self, i):
        ctx = self.ctx_norm[i]
        if self.ctx_noise > 0: ctx = ctx + np.random.randn(*ctx.shape).astype(np.float32) * self.ctx_noise
        return {"k_full": torch.from_numpy(self.k[i]), "y_full": torch.from_numpy(self.y[i]), "c_full": torch.from_numpy(self.c[i]),
                "ctx": torch.from_numpy(ctx), "anchors": torch.from_numpy(self.anc_norm[i]),
                "log10E": torch.tensor(self.logE[i], dtype=torch.float32)}


def make_collate(n_ctx: int, max_ctx_points: int, p_zero_shot: float, ta_noise: float):
    rng = np.random.default_rng()
    def collate(batch):
        B = len(batch); T = batch[0]["k_full"].shape[0]
        idx = np.unique(np.round(np.linspace(0, T - 1, n_ctx)).astype(int))
        ctx_k = torch.zeros(B, max_ctx_points); ctx_ta = torch.zeros(B, max_ctx_points); ctx_mask = torch.zeros(B, max_ctx_points)
        for i, item in enumerate(batch):
            if rng.random() < p_zero_shot: continue
            m = len(idx)
            ctx_k[i, :m] = item["k_full"][idx]
            ctx_ta[i, :m] = item["c_full"][idx] + torch.randn(m) * ta_noise
            ctx_mask[i, :m] = 1.0
        return {"ctx": torch.stack([b["ctx"] for b in batch]), "anchors": torch.stack([b["anchors"] for b in batch]),
                "log10E": torch.stack([b["log10E"] for b in batch]),
                "ctx_k": ctx_k, "ctx_ta": ctx_ta, "ctx_mask": ctx_mask,
                "tgt_k": torch.stack([b["k_full"] for b in batch]), "tgt_y": torch.stack([b["y_full"] for b in batch]),
                "tgt_mask": torch.ones(B, T)}
    return collate


def train_one_seed(model, train_ds, val_ds, out_dir, epochs, lr, batch, device, lam_spec, lam_smooth, collate):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=batch, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(epochs // 3, 50), T_mult=2)
    history = []; best_val = float("inf"); t0 = time.time()
    def run(b, train):
        ctx = b["ctx"].to(device); anc = b["anchors"].to(device); logE = b["log10E"].to(device)
        ck = b["ctx_k"].to(device); cta = b["ctx_ta"].to(device); cm = b["ctx_mask"].to(device)
        tk = b["tgt_k"].to(device); ty = b["tgt_y"].to(device); tm = b["tgt_mask"].to(device)
        pred = model(tk, ctx, anc, logE, ck, cta, cm)
        L_data = masked_mse(pred, ty, tm)
        if not train: return L_data, pred, ty, tm
        L_spec = model.spectral_penalty(ctx, anc, logE, ck, cta, cm, tk); L_sm = smoothness_penalty(pred)
        return L_data, L_spec, L_sm
    for ep in range(1, epochs + 1):
        model.train(); agg = np.zeros(3); n = 0
        for b in tl:
            L_data, L_spec, L_sm = run(b, True)
            loss = L_data + lam_spec * L_spec + lam_smooth * L_sm
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(ep + n / max(len(tl), 1))
            agg += [float(L_data), float(L_spec), float(L_sm)]; n += 1
        model.eval(); v_mse = v_mae = 0.0; vn = 0
        with torch.no_grad():
            for b in vl:
                L, pred, ty, tm = run(b, False)
                v_mse += float((((pred - ty) ** 2) * tm).sum()); v_mae += float(((pred - ty).abs() * tm).sum()); vn += float(tm.sum())
        v_mse /= max(vn, 1); v_mae /= max(vn, 1); agg /= max(n, 1)
        history.append({"epoch": ep, "train_mse": agg[0], "train_spec": agg[1], "train_smooth": agg[2], "val_mse": v_mse, "val_mae": v_mae})
        if v_mse < best_val:
            best_val = v_mse; torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"    ep {ep:4d}/{epochs}  data={agg[0]:.5f}  spec={agg[1]:.4f}  smooth={agg[2]:.5f}  val_mse={v_mse:.5f}  val_mae={v_mae:.5f}", flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"state_dict": model.state_dict()}, out_dir / "last.pt")
    print(f"  seed done in {(time.time()-t0)/60:.1f} min  best_val_mse={best_val:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1200); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_base", type=int, default=42); ap.add_argument("--ctx_noise", type=float, default=0.03)
    ap.add_argument("--ta_noise", type=float, default=0.01, help="noise on the context curve values during training")
    ap.add_argument("--n_ctx", type=int, default=32, help="context points taken along the neighbour-interpolated curve")
    ap.add_argument("--max_ctx_points", type=int, default=32); ap.add_argument("--p_zero_shot", type=float, default=0.10)
    ap.add_argument("--n_cheb", type=int, default=64); ap.add_argument("--n_cos", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--n_enc_layers", type=int, default=6)
    ap.add_argument("--n_dec_layers", type=int, default=3); ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_E_freq", type=int, default=32); ap.add_argument("--n_k_freq", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05); ap.add_argument("--lam_spec", type=float, default=5e-5)
    ap.add_argument("--lam_smooth", type=float, default=5e-4); ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--split_csv", default=str(ROOT / "data" / "split_by_E.csv"))
    ap.add_argument("--input_csv", default=str(ROOT / "data" / "combined_data.csv"))
    ap.add_argument("--desc_csv", default=str(ROOT / "data" / "branch_functional_descriptors.csv"))
    ap.add_argument("--out_root", default=str(ROOT / "results" / "runs" / "cnp_nbr"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True); print(f"args: {vars(args)}", flush=True)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv); raw.columns = ["Ta", "k", "E"]
    desc_true = pd.read_csv(args.desc_csv); split_df = pd.read_csv(args.split_csv)
    train_E = split_df.loc[split_df.split == "train", "E"].to_numpy(float)
    desc_lf = interpolate_descriptors(desc_true, train_E, margin=args.margin)
    ctx_curves = interpolate_curves(raw, desc_lf, n=101)
    desc_lf.to_csv(out_root / "descriptors_leakfree.csv", index=False); split_df.to_csv(out_root / "split_by_E.csv", index=False)
    keys = sorted(ctx_curves); np.savez(out_root / "context_curves.npz", E=np.array([k[0] for k in keys]),
                                        branch_local_id=np.array([k[1] for k in keys]), ta=np.stack([ctx_curves[k] for k in keys]))
    diag = check_against_truth(desc_true, desc_lf); print("leak-free descriptors:", json.dumps(diag), flush=True)
    (out_root / "leakfree_diag.json").write_text(json.dumps({"args": vars(args), "diag": diag, "conditioned": True}, indent=2))

    items = build_dataset(raw, desc_lf); tr, va, te = split_by_E(items, split_df)
    print(f"Built dataset: {len(items)} branches; split by E: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    collate = make_collate(args.n_ctx, args.max_ctx_points, args.p_zero_shot, args.ta_noise)
    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n========== SEED {seed} ({s_idx + 1}/{args.n_seeds}) ==========", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        train_ds = CNPDS(tr, ctx_curves, ctx_noise=args.ctx_noise)
        val_ds = CNPDS(va, ctx_curves, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std, anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)
        seed_root = out_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
        if s_idx == 0:
            np.savez(out_root / "test_branches.npz", E=np.array([it["E"] for it in te]), branch_local_id=np.array([it["branch_local_id"] for it in te]),
                     ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std, anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)
        model = CNP_TC(ctx_dim=len(STATIC_FEATURES), n_cheb=args.n_cheb, n_cos=args.n_cos, d_model=args.d_model,
                       n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers, n_heads=args.n_heads, dropout=args.dropout,
                       n_E_freq=args.n_E_freq, n_k_freq=args.n_k_freq, max_ctx_points=args.max_ctx_points).to(device)
        print(f"CNP-TC parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
        train_one_seed(model, train_ds, val_ds, seed_root, args.epochs, args.lr, args.batch, device, args.lam_spec, args.lam_smooth, collate)
    print("\n========== ALL SEEDS DONE ==========", flush=True)


if __name__ == "__main__":
    main()
