"""Train the three precision-curve models on the original Input data.

Pipeline:
  1. Read combined_data.csv (raw Ta(k,E)) + Step A branch descriptors
  2. For each branch: extract context xi (24 features), normalise k_norm in
     [-1, 1] and Ta_norm in [0, 1] using the branch's own (Ta_min, Ta_max)
  3. Stratified split by log10(E) into 70/15/15
  4. Train SIREN, DeepONet, Chebyshev sequentially
  5. Save checkpoints to data/runs_precision/<model>/best.pt

Each model is trained with AdamW + cosine annealing, 200 epochs, batch=8.
Test-time MAE on Ta_norm should be < 0.02 for at least one of the three.
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


# ---- 24 context features (identical to the legacy pipeline) ----
STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def build_dataset(input_csv: Path, desc_csv: Path, profile_csv: Path,
                    n_resampled: int = 101):
    """Build per-branch (k_norm, Ta_norm, ctx) triples from the raw Input."""
    raw = pd.read_csv(input_csv); raw.columns = ["Ta", "k", "E"]
    desc = pd.read_csv(desc_csv)
    prof = pd.read_csv(profile_csv).replace([np.inf, -np.inf], np.nan)
    prof_first = (prof.sort_values(["E", "branch_local_id", "s"])
                       .groupby(["E", "branch_local_id"], sort=False)
                       .first().reset_index())

    # Add log10E + n_branches if missing
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))

    items = []
    for _, row in desc.iterrows():
        E_val, b = float(row["E"]), int(row["branch_local_id"])
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)

        # Pick raw points strictly inside [k_left, k_right]
        sub = raw[(np.isclose(raw["E"], E_val))
                   & (raw["k"] >= k_left - 1e-6)
                   & (raw["k"] <= k_right + 1e-6)].sort_values("k")
        if len(sub) < 5: continue

        k_phys = sub["k"].to_numpy()
        ta_phys = sub["Ta"].to_numpy()

        # Resample on a uniform grid of n_resampled points
        s_grid = np.linspace(0.0, 1.0, n_resampled)
        k_grid = k_left + s_grid * (k_right - k_left)
        ta_resamp = np.interp(k_grid, k_phys, ta_phys)
        ta_norm = (ta_resamp - Ta_min) / amp        # in [0, 1]
        k_norm = 2.0 * s_grid - 1.0                  # in [-1, 1]

        # Look up the 23 features for this branch in profile_first
        feat_row = prof_first[(np.isclose(prof_first["E"], E_val))
                                 & (prof_first["branch_local_id"] == b)]
        if len(feat_row) == 0: continue
        feats = []
        for c in STATIC_FEATURES:
            try: feats.append(float(feat_row[c].iloc[0]))
            except Exception: feats.append(0.0)
        ctx = np.asarray(feats, dtype=np.float32)
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)

        items.append({"E": E_val, "branch_local_id": b,
                       "k_left": k_left, "k_right": k_right,
                       "Ta_min": Ta_min, "Ta_max": Ta_max,
                       "k_norm": k_norm.astype(np.float32),
                       "ta_norm": ta_norm.astype(np.float32),
                       "ctx": ctx,
                       "log10E": np.log10(max(E_val, 1e-30))})
    print(f"Built {len(items)} branches from Input.")
    return items


def stratified_split(items, train_frac=0.70, val_frac=0.15, seed=42):
    """Stratified split by log10(E)."""
    rng = np.random.default_rng(seed)
    log10Es = np.array([it["log10E"] for it in items])
    order = np.argsort(log10Es)
    bins = np.array_split(order, 8)              # 8 quantile-ish bins
    train, val, test = [], [], []
    for bin_idx in bins:
        bin_idx = list(bin_idx)
        rng.shuffle(bin_idx)
        n = len(bin_idx)
        n_tr = int(round(train_frac * n))
        n_va = int(round(val_frac * n))
        train += [items[i] for i in bin_idx[:n_tr]]
        val   += [items[i] for i in bin_idx[n_tr:n_tr + n_va]]
        test  += [items[i] for i in bin_idx[n_tr + n_va:]]
    return train, val, test


class CurveDS(Dataset):
    def __init__(self, items, ctx_mean=None, ctx_std=None):
        self.items = items
        ctx_arr = np.stack([it["ctx"] for it in items], axis=0)
        if ctx_mean is None:
            ctx_mean = ctx_arr.mean(axis=0)
            ctx_std = ctx_arr.std(axis=0) + 1e-6
        self.ctx_mean, self.ctx_std = ctx_mean, ctx_std
        self.ctx_norm = ((ctx_arr - ctx_mean) / ctx_std).astype(np.float32)
        self.ctx_norm = np.clip(np.nan_to_num(self.ctx_norm, nan=0.0,
                                                posinf=5.0, neginf=-5.0), -5.0, 5.0)

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return {"k_norm":  torch.from_numpy(it["k_norm"]),
                 "ta_norm": torch.from_numpy(it["ta_norm"]),
                 "ctx":     torch.from_numpy(self.ctx_norm[i])}


def collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def train_one(model, name, train_ds, val_ds, out_dir, epochs=200, lr=2e-4,
              weight_decay=1e-4, batch_size=8, kink_alpha=5.0, kink_z=2.0,
              device=torch.device("cpu")):
    """Train with kink-weighted MSE: points where |d2 Ta_true| exceeds
    kink_z standard deviations get weight kink_alpha (else weight 1).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                     collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [{name}] params={n_params:.2f}M  train={len(train_ds)}  val={len(val_ds)}")

    history = []; best_val = float("inf")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        agg, n = 0.0, 0
        for b in tl:
            b = {k: v.to(device) for k, v in b.items()}
            pred = model(b["k_norm"], b["ctx"])
            loss = kink_weighted_mse(pred, b["ta_norm"],
                                       alpha=kink_alpha, threshold_z=kink_z)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
            agg += float(loss); n += 1
        sched.step()
        # val (plain MSE/MAE so the metric stays comparable)
        model.eval(); va, vmae, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                b = {k: v.to(device) for k, v in b.items()}
                pred = model(b["k_norm"], b["ctx"])
                va += float(F.mse_loss(pred, b["ta_norm"])) * b["ta_norm"].size(0)
                vmae += float((pred - b["ta_norm"]).abs().mean()) * b["ta_norm"].size(0)
                vn += b["ta_norm"].size(0)
        v = va / max(vn, 1); vmae = vmae / max(vn, 1)
        history.append({"epoch": ep, "train_kw_mse": agg / max(n, 1),
                         "val_mse": v, "val_mae": vmae})
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"    ep {ep:3d}/{epochs}  train_kw_mse={agg/max(n,1):.5f}  "
                   f"val_mse={v:.5f}  val_mae={vmae:.5f}")
        if v < best_val:
            best_val = v
            torch.save({"state_dict": model.state_dict(),
                         "ctx_mean": train_ds.ctx_mean.tolist(),
                         "ctx_std":  train_ds.ctx_std.tolist()},
                        out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")
    print(f"  [{name}] done {(time.time()-t0)/60:.1f} min  best_val_mse={best_val:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default=str(ROOT / "checkpoints"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    input_csv = ROOT / "data" / "Input" / "combined_data.csv"
    desc_csv = ROOT / "data" / "processed" / "branch_functional_descriptors.csv"
    profile_csv = ROOT / "data" / "processed" / "model_profile_level_dataset.csv"

    items = build_dataset(input_csv, desc_csv, profile_csv)
    tr, va, te = stratified_split(items, seed=args.seed)
    print(f"Splits: train={len(tr)} val={len(va)} test={len(te)}")

    train_ds = CurveDS(tr)
    val_ds   = CurveDS(va, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)
    test_ds  = CurveDS(te, ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    # Save the test split index for inference reproducibility
    np.savez(out_root / "test_branches.npz",
              E=np.array([it["E"] for it in te]),
              branch_local_id=np.array([it["branch_local_id"] for it in te]),
              ctx_mean=train_ds.ctx_mean, ctx_std=train_ds.ctx_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[precision] device = {device}")

    print("\n>>> Training SIREN (kink-weighted)")
    siren = SIRENRegressor(ctx_dim=len(STATIC_FEATURES)).to(device)
    train_one(siren, "SIREN", train_ds, val_ds, out_root / "siren",
                epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=device)

    print("\n>>> Training DeepONet (kink-weighted)")
    deeponet = DeepONet(ctx_dim=len(STATIC_FEATURES)).to(device)
    train_one(deeponet, "DeepONet", train_ds, val_ds, out_root / "deeponet",
                epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=device)

    print("\n>>> Training Chebyshev (kink-weighted)")
    cheb = ChebyshevSpectralRegressor(ctx_dim=len(STATIC_FEATURES), n_modes=16).to(device)
    train_one(cheb, "Chebyshev", train_ds, val_ds, out_root / "chebyshev",
                epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=device)

    print("\n>>> Training Multi-Mode Envelope SIREN (M=4 modes, kink-weighted)")
    env = MultiModeEnvelopeSIREN(ctx_dim=len(STATIC_FEATURES), n_modes=4).to(device)
    train_one(env, "EnvelopeSIREN", train_ds, val_ds, out_root / "envelope_siren",
                epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=device)

    print("\n>>> ALL FOUR TRAINED.")


if __name__ == "__main__":
    main()
