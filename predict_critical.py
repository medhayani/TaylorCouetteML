"""Sweep the elasticity number E and predict the critical Taylor number
T_{a,c}(E) = min_k T_a(k, E) and the critical wavenumber
k_c(E) = arg min_k T_a(k, E) on each branch.

Usage examples
--------------
    # Default sweep: 80 log-spaced E values in [1e-4, 10]
    python predict_critical.py

    # Custom range and density (here: 200 log-spaced points)
    python predict_critical.py --E_min 1e-4 --E_max 10 --n_E 200

    # Or specify the E step on the log10 axis explicitly (here: 0.05 dex)
    python predict_critical.py --E_step 0.05

    # Save plot and CSV
    python predict_critical.py --save critical_curves.png --csv critical_curves.csv

    # Choose the surrogate
    python predict_critical.py --model dist     # default, fastest
    python predict_critical.py --model cnp      # CNP zero-shot

Output
------
The script produces a two-panel figure:
    top:    T_{a,c}(E)  vs  log10 E       (critical Taylor number)
    bottom: k_c(E)      vs  log10 E       (critical wavenumber)
and a CSV with columns  E, branch_local_id, log10E, Ta_c, k_c.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from models.cnp_tc.model import CNP_TC
from models.star.model import STAR


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def load_models(model_kind: str, device):
    """Return a list of (model, ctx_mean, ctx_std, anc_mean, anc_std)
    tuples for each available seed."""
    if model_kind == "dist":
        ckpt_dir = ROOT / "models_trained" / "dist"
        args = dict(n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12,
                     dropout=0.05, n_E_freq=32, n_k_freq=16)
        cls = STAR
        kwargs_for_class = args
    elif model_kind == "cnp":
        ckpt_dir = ROOT / "models_trained" / "cnp"
        args = dict(n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6,
                     n_dec_layers=3, n_heads=8, dropout=0.05,
                     n_E_freq=32, n_k_freq=16, max_ctx_points=32)
        cls = CNP_TC
        kwargs_for_class = args
    else:
        raise ValueError(f"Unknown model {model_kind}")

    norm = np.load(ckpt_dir / "test_branches.npz", allow_pickle=False)
    ctx_mean = norm["ctx_mean"].astype(np.float32)
    ctx_std = norm["ctx_std"].astype(np.float32) + 1e-6
    anc_mean = norm["anc_mean"].astype(np.float32)
    anc_std = norm["anc_std"].astype(np.float32) + 1e-6
    seed_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    models = []
    for sd in seed_dirs:
        m = cls(ctx_dim=23, **kwargs_for_class).to(device)
        ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)
    print(f"Loaded {len(models)} {model_kind.upper()} seeds")
    return models, ctx_mean, ctx_std, anc_mean, anc_std


def predict_branch(models, model_kind, row, n_pred, ctx_mean, ctx_std, anc_mean, anc_std, device):
    """Run the ensemble forward pass on one branch row. Return (k_grid, Ta_phys)."""
    k_left = float(row["k_left"]); k_right = float(row["k_right"])
    Ta_min = float(row["Ta_min"]); Ta_max = float(row["Ta_max"])
    amp = max(Ta_max - Ta_min, 1e-12)
    s_grid = np.linspace(0.0, 1.0, n_pred, dtype=np.float32)
    k_grid = k_left + s_grid * (k_right - k_left)
    k_norm = 2.0 * s_grid - 1.0
    ctx = []
    for f in STATIC_FEATURES:
        v = row.get(f, np.nan)
        try: ctx.append(float(v))
        except (TypeError, ValueError): ctx.append(0.0)
    ctx = np.nan_to_num(np.asarray(ctx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    anc = np.asarray([k_left, k_right, np.log10(max(Ta_min, 1e-6)), np.log10(max(Ta_max, 1e-6))],
                       dtype=np.float32)
    ctx_n = np.clip(np.nan_to_num((ctx - ctx_mean) / ctx_std, nan=0.0, posinf=0.0, neginf=0.0),
                      -5.0, 5.0).astype(np.float32)
    anc_n = np.clip(np.nan_to_num((anc - anc_mean) / anc_std, nan=0.0, posinf=0.0, neginf=0.0),
                      -5.0, 5.0).astype(np.float32)
    k_t = torch.from_numpy(k_norm).unsqueeze(0).to(device)
    ctx_t = torch.from_numpy(ctx_n).unsqueeze(0).to(device)
    anc_t = torch.from_numpy(anc_n).unsqueeze(0).to(device)
    logE_t = torch.tensor([np.log10(max(float(row["E"]), 1e-30))],
                              dtype=torch.float32, device=device)
    if model_kind == "cnp":
        ck = torch.zeros(1, 32, device=device)
        cta = torch.zeros(1, 32, device=device)
        cm = torch.zeros(1, 32, device=device)
    seeds = []
    with torch.no_grad():
        for m in models:
            if model_kind == "cnp":
                p = m(k_t, ctx_t, anc_t, logE_t, ck, cta, cm).squeeze(0).cpu().numpy()
            else:
                p = m(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
            seeds.append(p)
    y_norm = np.mean(np.stack(seeds, axis=0), axis=0)
    Ta = Ta_min + y_norm * amp
    return k_grid, Ta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["dist", "cnp"], default="dist",
                    help="Surrogate to use (default: dist)")
    ap.add_argument("--E_min", type=float, default=1e-4,
                    help="Smallest E in the sweep (default 1e-4)")
    ap.add_argument("--E_max", type=float, default=10.0,
                    help="Largest E in the sweep (default 10)")
    ap.add_argument("--n_E", type=int, default=80,
                    help="Number of log-spaced E points (ignored if --E_step is set)")
    ap.add_argument("--E_step", type=float, default=None,
                    help="Step in log10(E). If given, overrides --n_E.")
    ap.add_argument("--n_pred", type=int, default=251,
                    help="Number of k points per branch (resolution of Tac, kc detection)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save the Tac(E)+kc(E) plot to this PNG")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Save the Tac, kc values to this CSV")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build the sweep
    if args.E_step is not None:
        log_grid = np.arange(np.log10(args.E_min), np.log10(args.E_max) + 1e-9, args.E_step)
        print(f"E sweep with step {args.E_step} dex -> {len(log_grid)} points")
    else:
        log_grid = np.linspace(np.log10(args.E_min), np.log10(args.E_max), args.n_E)
        print(f"E sweep with {args.n_E} log-spaced points")

    # Snap to the closest E values that actually exist in the dataset
    desc = pd.read_csv(ROOT / "data" / "branch_functional_descriptors.csv")
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    all_Es = np.array(sorted(desc["E"].unique()))
    sel_E = []
    for lp in log_grid:
        nearest = float(all_Es[int(np.argmin(np.abs(np.log10(all_Es) - lp)))])
        if nearest not in sel_E:
            sel_E.append(nearest)
    print(f"After snapping to available branches: {len(sel_E)} unique E values")

    # Load model
    models, ctx_mean, ctx_std, anc_mean, anc_std = load_models(args.model, device)

    # Sweep
    rows = []
    for i, E in enumerate(sel_E):
        sub = desc[np.isclose(desc["E"], E)].sort_values("branch_local_id")
        for _, row in sub.iterrows():
            k_grid, Ta = predict_branch(models, args.model, row, args.n_pred,
                                              ctx_mean, ctx_std, anc_mean, anc_std, device)
            j = int(np.argmin(Ta))
            Ta_c = float(Ta[j]); k_c = float(k_grid[j])
            rows.append({"E": E, "branch_local_id": int(row["branch_local_id"]),
                          "log10E": float(np.log10(max(E, 1e-30))),
                          "Ta_c": Ta_c, "k_c": k_c})
        if (i + 1) % 10 == 0 or i == len(sel_E) - 1:
            print(f"  {i+1}/{len(sel_E)} E values done")
    df = pd.DataFrame(rows)

    if args.csv is not None:
        df.to_csv(args.csv, index=False)
        print(f"Saved CSV -> {args.csv}")

    # Take the global minimum across branches at each E
    df_min = df.loc[df.groupby("E")["Ta_c"].idxmin()].sort_values("E")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), dpi=120, sharex=True)
    ax = axes[0]
    ax.scatter(df["log10E"], df["Ta_c"], s=15, c="lightgray", label="per-branch")
    ax.plot(df_min["log10E"], df_min["Ta_c"], "-o", ms=4, color="tab:red",
              label="global min across branches")
    ax.set_ylabel(r"$T_{a,c}$ (critical Taylor)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(df["log10E"], df["k_c"], s=15, c="lightgray", label="per-branch")
    ax.plot(df_min["log10E"], df_min["k_c"], "-o", ms=4, color="tab:blue",
              label="argmin branch")
    ax.set_xlabel(r"$\log_{10} E$")
    ax.set_ylabel(r"$k_c$ (critical wavenumber)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"Critical curves predicted by {args.model.upper()} ensemble "
                   f"over {len(sel_E)} values of E", fontsize=11)
    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=140, bbox_inches="tight")
        print(f"Saved figure -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
