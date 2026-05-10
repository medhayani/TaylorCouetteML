"""Enhanced precision-curve training: bigger models, longer schedule, ensemble of seeds.

Architecture changes vs train_precision_curves.py:
- SIREN:           hidden 256 -> 384, depth 6 -> 8
- DeepONet:        trunk hidden +50% (where the model exposes hidden)
- Chebyshev:       n_modes 16 -> 32, hidden +50%
- Envelope-SIREN:  n_modes 4 -> 8, hidden 192 -> 320, depth 5 -> 7

Training changes:
- 1000 epochs (vs 500) with cosine + 1 warm restart
- Light context-noise augmentation (sigma=0.02 std on normalised ctx) during training
- Trains 3 seeds (configurable) per architecture for ensemble robustness
  -> 4 architectures x 3 seeds = 12 final models

Output layout:
  out_root/seed_<s>/<model_name>/best.pt
  out_root/test_branches.npz (shared across seeds)
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

from models.siren.model import SIRENRegressor
from models.deeponet.model import DeepONet
from models.chebyshev_spectral.model import ChebyshevSpectralRegressor
from models.envelope_siren.model import MultiModeEnvelopeSIREN
from models.utils import kink_weighted_mse


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def build_dataset(input_csv, desc_csv, profile_csv, n_resampled: int = 101):
    raw = pd.read_csv(input_csv); raw.columns = ["Ta", "k", "E"]
    desc = pd.read_csv(desc_csv)
    prof = pd.read_csv(profile_csv).replace([np.inf, -np.inf], np.nan)
    prof_first = (prof.sort_values(["E", "branch_local_id", "s"])
                       .groupby(["E", "branch_local_id"], sort=False)
                       .first().reset_index())
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
        k_norm = 2.0 * s_grid - 1.0

        ctx = []
        for f in STATIC_FEATURES:
            v = row.get(f, np.nan)
            try: ctx.append(float(v))
            except (TypeError, ValueError): ctx.append(0.0)
        ctx = np.asarray(ctx, dtype=np.float32)
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        items.append({
            "E": E_val, "branch_local_id": b,
            "k_norm": k_norm.astype(np.float32),
            "ta_norm": ta_norm.astype(np.float32),
            "ctx": ctx, "Ta_min": Ta_min, "Ta_max": Ta_max,
            "k_left": k_left, "k_right": k_right,
        })
    return items


def stratified_split(items, seed=42):
    rng = np.random.default_rng(seed)
    log10E = np.array([np.log10(max(it["E"], 1e-30)) for it in items])
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
    def __init__(self, items, ctx_mean=None, ctx_std=None, ctx_noise: float = 0.0):
        self.k = np.stack([it["k_norm"] for it in items])
        self.y = np.stack([it["ta_norm"] for it in items])
        ctx_arr = np.stack([it["ctx"] for it in items]).astype(np.float32)
        if ctx_mean is None:
            ctx_mean = ctx_arr.mean(axis=0)
            ctx_std = ctx_arr.std(axis=0) + 1e-6
        self.ctx_mean, self.ctx_std = ctx_mean, ctx_std
        self.ctx_norm = ((ctx_arr - ctx_mean) / ctx_std).astype(np.float32)
        self.ctx_norm = np.clip(np.nan_to_num(self.ctx_norm, nan=0.0,
                                                  posinf=0.0, neginf=0.0), -5.0, 5.0)
        self.ctx_noise = float(ctx_noise)

    def __len__(self): return len(self.k)
    def __getitem__(self, i):
        ctx = self.ctx_norm[i]
        if self.ctx_noise > 0:
            ctx = ctx + np.random.randn(*ctx.shape).astype(np.float32) * self.ctx_noise
        return {"k_norm": torch.from_numpy(self.k[i]),
                "ta_norm": torch.from_numpy(self.y[i]),
                "ctx":     torch.from_numpy(ctx)}


def train_one(model, name, train_ds, val_ds, out_dir, epochs, lr, batch, device,
                use_warm_restart=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    if use_warm_restart:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(epochs // 3, 50), T_mult=2)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []; best_val = float("inf"); t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); agg, n = 0.0, 0
        for b in tl:
            k = b["k_norm"].to(device); y = b["ta_norm"].to(device)
            ctx = b["ctx"].to(device)
            pred = model(k, ctx)
            loss = kink_weighted_mse(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(ep + n / max(len(tl), 1))
            agg += float(loss); n += 1
        # validation
        model.eval(); v_mse, v_mae, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                k = b["k_norm"].to(device); y = b["ta_norm"].to(device)
                ctx = b["ctx"].to(device)
                pred = model(k, ctx)
                v_mse += float(F.mse_loss(pred, y, reduction="sum"))
                v_mae += float((pred - y).abs().sum())
                vn += y.numel()
        v_mse /= max(vn, 1); v_mae /= max(vn, 1)
        history.append({"epoch": ep, "train_kw_mse": agg / max(n, 1),
                          "val_mse": v_mse, "val_mae": v_mae})
        if v_mse < best_val:
            best_val = v_mse
            torch.save({"state_dict": model.state_dict()},
                        out_dir / "best.pt")
        if ep == 1 or ep % 50 == 0 or ep == epochs:
            print(f"    ep {ep:4d}/{epochs}  train_kw_mse={agg/max(n,1):.5f}  "
                    f"val_mse={v_mse:.5f}  val_mae={v_mae:.5f}", flush=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"  [{name}] done {(time.time()-t0)/60:.1f} min  best_val_mse={best_val:.5f}")
    return best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed_base", type=int, default=42)
    ap.add_argument("--ctx_noise", type=float, default=0.02)
    ap.add_argument("--out_root", default=str(ROOT / "data" / "runs" / "precision_v2"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    input_csv = ROOT / "data" / "Input" / "combined_data.csv"
    desc_csv = ROOT / "data" / "processed" / "branch_functional_descriptors.csv"
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"

    items = build_dataset(input_csv, desc_csv, profile_csv)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    for s_idx in range(args.n_seeds):
        seed = args.seed_base + s_idx
        print(f"\n========== SEED {seed} ({s_idx + 1}/{args.n_seeds}) ==========")
        torch.manual_seed(seed); np.random.seed(seed)

        tr, va, te = stratified_split(items, seed=seed)
        print(f"Splits: train={len(tr)} val={len(va)} test={len(te)}")

        train_ds = CurveDS(tr, ctx_noise=args.ctx_noise)
        val_ds   = CurveDS(va, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)

        seed_root = out_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        # Save norm stats only from the first seed
        if s_idx == 0:
            np.savez(out_root / "test_branches.npz",
                       E=np.array([it["E"] for it in te]),
                       branch_local_id=np.array([it["branch_local_id"] for it in te]),
                       ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)

        ctx_dim = len(STATIC_FEATURES)

        # ---- SIREN bigger ----
        print(">>> SIREN (hidden=384, depth=8)")
        try:
            siren = SIRENRegressor(ctx_dim=ctx_dim, hidden=384, depth=8).to(device)
        except TypeError:
            siren = SIRENRegressor(ctx_dim=ctx_dim).to(device)
        train_one(siren, "SIREN", train_ds, val_ds, seed_root / "siren",
                    epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)

        # ---- DeepONet bigger (branch+trunk 384x4, latent 192, 24 fourier bands) ----
        print(">>> DeepONet (384x4, latent=192, fourier=24)")
        deeponet = DeepONet(ctx_dim=ctx_dim,
                              branch_layers=(384, 384, 384, 384),
                              trunk_layers=(384, 384, 384, 384),
                              latent_dim=192, fourier_bands=24).to(device)
        train_one(deeponet, "DeepONet", train_ds, val_ds, seed_root / "deeponet",
                    epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)

        # ---- Chebyshev (n_modes 16 -> 32) ----
        print(">>> Chebyshev (n_modes=32)")
        cheb = ChebyshevSpectralRegressor(ctx_dim=ctx_dim, n_modes=32).to(device)
        train_one(cheb, "Chebyshev", train_ds, val_ds, seed_root / "chebyshev",
                    epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)

        # ---- Envelope-SIREN (M=8, hidden=320, depth=7) ----
        print(">>> Envelope-SIREN (M=8, hidden=320, depth=7)")
        env = MultiModeEnvelopeSIREN(ctx_dim=ctx_dim, n_modes=8,
                                            hidden=320, depth=7).to(device)
        train_one(env, "EnvelopeSIREN", train_ds, val_ds, seed_root / "envelope_siren",
                    epochs=args.epochs, lr=args.lr, batch=args.batch, device=device)

    print("\n========== ALL SEEDS DONE ==========")


if __name__ == "__main__":
    main()
