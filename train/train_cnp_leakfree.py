"""Train CNP-TC with a LEAK-FREE protocol.

Differences with the CNP trainer of the first submission (everything else is identical):
  1. Branch structure, descriptors and anchors of every elasticity are
     interpolated from the two nearest TRAINING elasticities
     (leakfree_descriptors.py); the target curve of E is never used to
     build its own inputs, nor to normalise its own targets.
  2. The train / validation / test split is BY ELASTICITY, read from a
     file (--split_csv, columns E, split), and identical for all seeds.
  3. The interpolated descriptor table and the split are saved with the
     run, so that the evaluation uses exactly the same inputs.
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
from leakfree_descriptors import interpolate_descriptors, check_against_truth

STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def build_dataset(raw: pd.DataFrame, desc: pd.DataFrame, n_resampled: int = 101):
    desc = desc.copy()
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    items = []
    for _, row in desc.iterrows():
        E_val, b = float(row["E"]), int(row["branch_local_id"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)
        sub = raw[(np.isclose(raw["E"], E_val))
                   & (raw["k"] >= k_left - 1e-6)
                   & (raw["k"] <= k_right + 1e-6)].sort_values("k")
        if len(sub) < 5: continue
        k_phys = sub["k"].to_numpy(); ta_phys = sub["Ta"].to_numpy()
        s_grid = np.linspace(0.0, 1.0, n_resampled)
        k_grid = k_left + s_grid * (k_right - k_left)
        ta_resamp = np.interp(k_grid, k_phys, ta_phys)
        ta_norm = np.clip((ta_resamp - Ta_min) / amp, 0.0, 1.0)
        k_norm = 2.0 * s_grid - 1.0
        ctx = []
        for f in STATIC_FEATURES:
            v = row.get(f, np.nan)
            try: ctx.append(float(v))
            except (TypeError, ValueError): ctx.append(0.0)
        ctx = np.nan_to_num(np.asarray(ctx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        anchors_raw = np.asarray([k_left, k_right, np.log10(max(Ta_min, 1e-6)),
                                  np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
        items.append({"E": E_val, "branch_local_id": b,
                      "k_norm": k_norm.astype(np.float32), "ta_norm": ta_norm.astype(np.float32),
                      "ctx": ctx, "anchors_raw": anchors_raw,
                      "log10E": float(np.log10(max(E_val, 1e-30))),
                      "Ta_min": Ta_min, "Ta_max": Ta_max, "k_left": k_left, "k_right": k_right})
    return items


def split_by_E(items, split_df: pd.DataFrame):
    lab = {round(float(r.E), 8): str(r.split) for r in split_df.itertuples()}
    tr = [it for it in items if lab.get(round(it["E"], 8)) == "train"]
    va = [it for it in items if lab.get(round(it["E"], 8)) == "val"]
    te = [it for it in items if lab.get(round(it["E"], 8)) == "test"]
    return tr, va, te


class CNPDS(Dataset):
    def __init__(self, items, ctx_mean=None, ctx_std=None, anc_mean=None, anc_std=None, ctx_noise: float = 0.0):
        self.k = np.stack([it["k_norm"] for it in items]); self.y = np.stack([it["ta_norm"] for it in items])
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
        return {"k_full": torch.from_numpy(self.k[i]), "y_full": torch.from_numpy(self.y[i]),
                "ctx": torch.from_numpy(ctx), "anchors": torch.from_numpy(self.anc_norm[i]),
                "log10E": torch.tensor(self.logE[i], dtype=torch.float32)}


def make_cnp_collate(max_ctx_points: int, p_zero_shot: float = 0.30):
    rng = np.random.default_rng()
    def collate(batch):
        B = len(batch); T_full = batch[0]["k_full"].shape[0]
        ctx_k = torch.zeros(B, max_ctx_points); ctx_ta = torch.zeros(B, max_ctx_points); ctx_mask = torch.zeros(B, max_ctx_points)
        tgt_k, tgt_y, per_n_tgt, max_tgt = [], [], [], 0
        for i, item in enumerate(batch):
            k_full = item["k_full"].numpy(); y_full = item["y_full"].numpy(); n = T_full
            n_ctx = 0 if rng.random() < p_zero_shot else int(rng.integers(1, max_ctx_points + 1))
            n_ctx = min(n_ctx, n - 8)
            all_idx = np.arange(n); rng.shuffle(all_idx)
            ctx_idx = all_idx[:n_ctx]; tgt_idx = all_idx[n_ctx:]; ctx_idx.sort(); tgt_idx.sort()
            ctx_k[i, :n_ctx] = torch.from_numpy(k_full[ctx_idx]) if n_ctx > 0 else 0.0
            ctx_ta[i, :n_ctx] = torch.from_numpy(y_full[ctx_idx]) if n_ctx > 0 else 0.0
            ctx_mask[i, :n_ctx] = 1.0
            tgt_k.append(torch.from_numpy(k_full[tgt_idx])); tgt_y.append(torch.from_numpy(y_full[tgt_idx]))
            per_n_tgt.append(len(tgt_idx)); max_tgt = max(max_tgt, len(tgt_idx))
        tgt_k_pad = torch.zeros(B, max_tgt); tgt_y_pad = torch.zeros(B, max_tgt); tgt_mask = torch.zeros(B, max_tgt)
        for i in range(B):
            n_t = per_n_tgt[i]; tgt_k_pad[i, :n_t] = tgt_k[i]; tgt_y_pad[i, :n_t] = tgt_y[i]; tgt_mask[i, :n_t] = 1.0
        return {"ctx": torch.stack([b["ctx"] for b in batch]), "anchors": torch.stack([b["anchors"] for b in batch]),
                "log10E": torch.stack([b["log10E"] for b in batch]),
                "ctx_k": ctx_k, "ctx_ta": ctx_ta, "ctx_mask": ctx_mask,
                "tgt_k": tgt_k_pad, "tgt_y": tgt_y_pad, "tgt_mask": tgt_mask}
    return collate


def smoothness_penalty(pred):
    d2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    return (d2 ** 2).mean()


def masked_mse(pred, true, mask):
    return ((pred - true) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)


def train_one_seed(model, train_ds, val_ds, out_dir, epochs, lr, batch, device,
                   lam_spec, lam_smooth, max_ctx_points, p_zero_shot):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    collate = make_cnp_collate(max_ctx_points, p_zero_shot)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=batch, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(epochs // 3, 50), T_mult=2)
    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); agg_data = agg_spec = agg_sm = 0.0; n = 0
        for b in tl:
            ctx = b["ctx"].to(device); anc = b["anchors"].to(device); logE = b["log10E"].to(device)
            ck = b["ctx_k"].to(device); cta = b["ctx_ta"].to(device); cm = b["ctx_mask"].to(device)
            tk = b["tgt_k"].to(device); ty = b["tgt_y"].to(device); tm = b["tgt_mask"].to(device)
            pred = model(tk, ctx, anc, logE, ck, cta, cm)
            L_data = masked_mse(pred, ty, tm)
            L_spec = model.spectral_penalty(ctx, anc, logE, ck, cta, cm, tk)
            L_sm = smoothness_penalty(pred)
            loss = L_data + lam_spec * L_spec + lam_smooth * L_sm
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(ep + n / max(len(tl), 1))
            agg_data += float(L_data); agg_spec += float(L_spec); agg_sm += float(L_sm); n += 1
        model.eval(); v_mse = v_mae = 0.0; vn = 0
        with torch.no_grad():
            for b in vl:
                ctx = b["ctx"].to(device); anc = b["anchors"].to(device); logE = b["log10E"].to(device)
                ck = b["ctx_k"].to(device); cta = b["ctx_ta"].to(device); cm = b["ctx_mask"].to(device)
                tk = b["tgt_k"].to(device); ty = b["tgt_y"].to(device); tm = b["tgt_mask"].to(device)
                pred = model(tk, ctx, anc, logE, ck, cta, cm)
                v_mse += float((((pred - ty) ** 2) * tm).sum()); v_mae += float(((pred - ty).abs() * tm).sum()); vn += float(tm.sum())
        v_mse /= max(vn, 1); v_mae /= max(vn, 1)
        history.append({"epoch": ep, "train_mse": agg_data / max(n, 1), "train_spec": agg_spec / max(n, 1),
                        "train_smooth": agg_sm / max(n, 1), "val_mse": v_mse, "val_mae": v_mae})
        if v_mse < best_val:
            best_val = v_mse; torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"    ep {ep:4d}/{epochs}  data={agg_data/max(n,1):.5f}  spec={agg_spec/max(n,1):.4f}  "
                  f"smooth={agg_sm/max(n,1):.5f}  val_mse={v_mse:.5f}  val_mae={v_mae:.5f}", flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"state_dict": model.state_dict()}, out_dir / "last.pt")
    print(f"  seed done in {(time.time()-t0)/60:.1f} min  best_val_mse={best_val:.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1200); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_base", type=int, default=42); ap.add_argument("--ctx_noise", type=float, default=0.03)
    ap.add_argument("--max_ctx_points", type=int, default=32); ap.add_argument("--p_zero_shot", type=float, default=0.30)
    ap.add_argument("--n_cheb", type=int, default=64); ap.add_argument("--n_cos", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--n_enc_layers", type=int, default=6)
    ap.add_argument("--n_dec_layers", type=int, default=3); ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_E_freq", type=int, default=32); ap.add_argument("--n_k_freq", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05); ap.add_argument("--lam_spec", type=float, default=5e-5)
    ap.add_argument("--lam_smooth", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=0.05, help="margin on the interpolated Ta_min / Ta_max")
    ap.add_argument("--split_csv", default=str(ROOT / "data" / "split_by_E.csv"))
    ap.add_argument("--input_csv", default=str(ROOT / "data" / "combined_data.csv"))
    ap.add_argument("--desc_csv", default=str(ROOT / "data" / "branch_functional_descriptors.csv"))
    ap.add_argument("--out_root", default=str(ROOT / "results" / "runs" / "cnp_leakfree"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True); print(f"args: {vars(args)}", flush=True)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv); raw.columns = ["Ta", "k", "E"]
    desc_true = pd.read_csv(args.desc_csv)
    split_df = pd.read_csv(args.split_csv)
    train_E = split_df.loc[split_df.split == "train", "E"].to_numpy(float)
    desc_lf = interpolate_descriptors(desc_true, train_E, margin=args.margin)
    desc_lf.to_csv(out_root / "descriptors_leakfree.csv", index=False)
    split_df.to_csv(out_root / "split_by_E.csv", index=False)
    diag = check_against_truth(desc_true, desc_lf)
    print("leak-free descriptors:", json.dumps(diag), flush=True)
    (out_root / "leakfree_diag.json").write_text(json.dumps({"args": vars(args), "diag": diag}, indent=2))

    items = build_dataset(raw, desc_lf)
    tr, va, te = split_by_E(items, split_df)
    print(f"Built dataset: {len(items)} branches; split by E: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    ctx_dim = len(STATIC_FEATURES)
    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n========== SEED {seed} ({s_idx + 1}/{args.n_seeds}) ==========", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        train_ds = CNPDS(tr, ctx_noise=args.ctx_noise)
        val_ds = CNPDS(va, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std, anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)
        seed_root = out_root / f"seed_{seed}"; seed_root.mkdir(parents=True, exist_ok=True)
        if s_idx == 0:
            np.savez(out_root / "test_branches.npz",
                     E=np.array([it["E"] for it in te]), branch_local_id=np.array([it["branch_local_id"] for it in te]),
                     ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std, anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)
        model = CNP_TC(ctx_dim=ctx_dim, n_cheb=args.n_cheb, n_cos=args.n_cos, d_model=args.d_model,
                       n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers, n_heads=args.n_heads,
                       dropout=args.dropout, n_E_freq=args.n_E_freq, n_k_freq=args.n_k_freq,
                       max_ctx_points=args.max_ctx_points).to(device)
        print(f"CNP-TC parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
        train_one_seed(model, train_ds, val_ds, seed_root, epochs=args.epochs, lr=args.lr, batch=args.batch,
                       device=device, lam_spec=args.lam_spec, lam_smooth=args.lam_smooth,
                       max_ctx_points=args.max_ctx_points, p_zero_shot=args.p_zero_shot)
    print("\n========== ALL SEEDS DONE ==========", flush=True)


if __name__ == "__main__":
    main()
