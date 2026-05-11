"""Train STAR (Spectral Transformer with Anchored Residuals).

Smart input preparation (the difference vs prior architectures):
- log10E projected onto 32 log-spaced sin/cos frequencies before tokenisation
- k_norm expanded to 16 Fourier features before the query
- Branch endpoints (k_left, k_right, Ta_min, Ta_max) injected as anchor tokens
- All 23 functional descriptors kept as separate tokens (attention finds couplings)

Ensemble training: `n_seeds` independent runs; at inference, take the
median across seeds.

Loss = kink-weighted MSE
     + lam_spec   * spectral_penalty (Cheb + cos coefficient decay)
     + lam_smooth * smoothness_penalty (second-difference of prediction)

Output layout:
    out_root/seed_<s>/best.pt          per-seed checkpoint
    out_root/seed_<s>/history.json     training curves
    out_root/test_branches.npz         test split + ctx/anchor stats
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.star.model import STAR
from models.utils import kink_weighted_mse


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def build_dataset(input_csv, desc_csv, n_resampled: int = 101):
    raw = pd.read_csv(input_csv); raw.columns = ["Ta", "k", "E"]
    desc = pd.read_csv(desc_csv)
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
        ta_norm = (ta_resamp - Ta_min) / amp
        ta_norm = np.clip(ta_norm, 0.0, 1.0)
        k_norm = 2.0 * s_grid - 1.0

        ctx = []
        for f in STATIC_FEATURES:
            v = row.get(f, np.nan)
            try: ctx.append(float(v))
            except (TypeError, ValueError): ctx.append(0.0)
        ctx = np.nan_to_num(np.asarray(ctx, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        anchors_raw = np.asarray([k_left, k_right,
                                    np.log10(max(Ta_min, 1e-6)),
                                    np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
        items.append({
            "E": E_val, "branch_local_id": b,
            "k_norm": k_norm.astype(np.float32),
            "ta_norm": ta_norm.astype(np.float32),
            "ctx": ctx, "anchors_raw": anchors_raw,
            "log10E": float(np.log10(max(E_val, 1e-30))),
            "Ta_min": Ta_min, "Ta_max": Ta_max,
            "k_left": k_left, "k_right": k_right,
        })
    return items


def stratified_split(items, seed=42):
    rng = np.random.default_rng(seed)
    log10E = np.array([it["log10E"] for it in items])
    bins = np.linspace(log10E.min(), log10E.max(), 11)
    bin_idx = np.digitize(log10E, bins) - 1
    tr, va, te = [], [], []
    for b in range(len(bins)):
        idx = np.where(bin_idx == b)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr, nva = int(0.7 * n), int(0.85 * n)
        tr.extend([items[i] for i in idx[:ntr]])
        va.extend([items[i] for i in idx[ntr:nva]])
        te.extend([items[i] for i in idx[nva:]])
    return tr, va, te


class CurveDS(Dataset):
    def __init__(self, items, ctx_mean=None, ctx_std=None,
                  anc_mean=None, anc_std=None, ctx_noise: float = 0.0):
        self.k = np.stack([it["k_norm"] for it in items])
        self.y = np.stack([it["ta_norm"] for it in items])
        self.logE = np.array([it["log10E"] for it in items], dtype=np.float32)
        ctx_arr = np.stack([it["ctx"] for it in items]).astype(np.float32)
        anc_arr = np.stack([it["anchors_raw"] for it in items]).astype(np.float32)
        if ctx_mean is None:
            ctx_mean = ctx_arr.mean(axis=0)
            ctx_std = ctx_arr.std(axis=0) + 1e-6
        if anc_mean is None:
            anc_mean = anc_arr.mean(axis=0)
            anc_std = anc_arr.std(axis=0) + 1e-6
        self.ctx_mean, self.ctx_std = ctx_mean, ctx_std
        self.anc_mean, self.anc_std = anc_mean, anc_std
        self.ctx_norm = np.clip(np.nan_to_num(
            (ctx_arr - ctx_mean) / ctx_std,
            nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        self.anc_norm = np.clip(np.nan_to_num(
            (anc_arr - anc_mean) / anc_std,
            nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        self.ctx_noise = float(ctx_noise)

    def __len__(self): return len(self.k)

    def __getitem__(self, i):
        ctx = self.ctx_norm[i]
        if self.ctx_noise > 0:
            ctx = ctx + np.random.randn(*ctx.shape).astype(np.float32) * self.ctx_noise
        return {
            "k_norm": torch.from_numpy(self.k[i]),
            "ta_norm": torch.from_numpy(self.y[i]),
            "ctx":     torch.from_numpy(ctx),
            "anchors": torch.from_numpy(self.anc_norm[i]),
            "log10E":  torch.tensor(self.logE[i], dtype=torch.float32),
        }


def smoothness_penalty(pred: torch.Tensor) -> torch.Tensor:
    d2 = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    return (d2 ** 2).mean()


def train_one_seed(model, train_ds, val_ds, out_dir, epochs, lr, batch, device,
                   lam_spec: float, lam_smooth: float):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(epochs // 3, 50), T_mult=2)

    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); agg_data = 0.0; agg_spec = 0.0; agg_sm = 0.0; n = 0
        for b in tl:
            k = b["k_norm"].to(device); y = b["ta_norm"].to(device)
            ctx = b["ctx"].to(device); anc = b["anchors"].to(device)
            logE = b["log10E"].to(device)
            pred = model(k, ctx, anc, logE)
            L_data = kink_weighted_mse(pred, y)
            L_spec = model.spectral_penalty(ctx, anc, logE)
            L_sm = smoothness_penalty(pred)
            loss = L_data + lam_spec * L_spec + lam_smooth * L_sm
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(ep + n / max(len(tl), 1))
            agg_data += float(L_data); agg_spec += float(L_spec)
            agg_sm += float(L_sm); n += 1
        # validation
        model.eval(); v_mse, v_mae, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                k = b["k_norm"].to(device); y = b["ta_norm"].to(device)
                ctx = b["ctx"].to(device); anc = b["anchors"].to(device)
                logE = b["log10E"].to(device)
                pred = model(k, ctx, anc, logE)
                v_mse += float(F.mse_loss(pred, y, reduction="sum"))
                v_mae += float((pred - y).abs().sum())
                vn += y.numel()
        v_mse /= max(vn, 1); v_mae /= max(vn, 1)
        history.append({
            "epoch": ep,
            "train_kw_mse": agg_data / max(n, 1),
            "train_spec": agg_spec / max(n, 1),
            "train_smooth": agg_sm / max(n, 1),
            "val_mse": v_mse, "val_mae": v_mae,
        })
        if v_mse < best_val:
            best_val = v_mse
            torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"    ep {ep:4d}/{epochs}  data={agg_data/max(n,1):.5f}  "
                    f"spec={agg_spec/max(n,1):.4f}  smooth={agg_sm/max(n,1):.5f}  "
                    f"val_mse={v_mse:.5f}  val_mae={v_mae:.5f}", flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"state_dict": model.state_dict()}, out_dir / "last.pt")
    print(f"  seed done in {(time.time()-t0)/60:.1f} min  best_val_mse={best_val:.5f}")
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_seeds", type=int, default=7)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--ctx_noise", type=float, default=0.03)
    ap.add_argument("--n_cheb", type=int, default=64)
    ap.add_argument("--n_cos", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--n_E_freq", type=int, default=32)
    ap.add_argument("--n_k_freq", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lam_spec", type=float, default=5e-5)
    ap.add_argument("--lam_smooth", type=float, default=5e-4)
    ap.add_argument("--out_root", default=str(ROOT / "data" / "runs" / "star_v1"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)
    print(f"args: {vars(args)}", flush=True)

    input_csv = ROOT / "data" / "Input" / "combined_data.csv"
    desc_csv = ROOT / "data" / "processed" / "branch_functional_descriptors.csv"

    items = build_dataset(input_csv, desc_csv)
    print(f"Built dataset: {len(items)} branches", flush=True)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    ctx_dim = len(STATIC_FEATURES)

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n========== SEED {seed} ({s_idx + 1}/{args.n_seeds}) ==========",
              flush=True)
        torch.manual_seed(seed); np.random.seed(seed)

        tr, va, te = stratified_split(items, seed=seed)
        print(f"Splits: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

        train_ds = CurveDS(tr, ctx_noise=args.ctx_noise)
        val_ds = CurveDS(va, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std,
                            anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)

        seed_root = out_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        if s_idx == 0:
            np.savez(out_root / "test_branches.npz",
                       E=np.array([it["E"] for it in te]),
                       branch_local_id=np.array([it["branch_local_id"] for it in te]),
                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std,
                       anc_mean=train_ds.anc_mean, anc_std=train_ds.anc_std)

        model = STAR(
            ctx_dim=ctx_dim, n_cheb=args.n_cheb, n_cos=args.n_cos,
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
            dropout=args.dropout, n_E_freq=args.n_E_freq, n_k_freq=args.n_k_freq,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"STAR parameters: {n_params/1e6:.2f}M", flush=True)

        train_one_seed(
            model=model, train_ds=train_ds, val_ds=val_ds, out_dir=seed_root,
            epochs=args.epochs, lr=args.lr, batch=args.batch, device=device,
            lam_spec=args.lam_spec, lam_smooth=args.lam_smooth,
        )

    print("\n========== ALL SEEDS DONE ==========", flush=True)


if __name__ == "__main__":
    main()
