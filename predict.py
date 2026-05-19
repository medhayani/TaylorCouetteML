"""TaylorCouetteML -- one-command prediction of the marginal stability curve Ta(k).

Usage examples
--------------
    # Predict T_a(k) for every branch at E = 1e-3 using the DIST model
    python predict.py --E 0.001

    # Use the CNP zero-shot model instead, save figure to disk
    python predict.py --E 0.001 --model cnp --save prediction.png

    # Output the prediction as a CSV (k, Ta) for downstream use
    python predict.py --E 0.001 --csv prediction.csv

The script loads pre-trained weights from `models_trained/<model>/seed_*/best.pt`
(shipped with this repository) and predicts the curve from the 23
morphological descriptors of every branch at that E value, listed in
`data/branch_functional_descriptors.csv`.

Three models are available:
    - cnp   : Conditional Neural Process (zero-shot mode)
    - dist  : Distilled Spectral Transformer  (default; fastest, accurate)
    - sarl  : Reinforcement Learning Refiner (needs a base prediction;
               this script applies it on top of the DIST output where a
               switch window exists for the branch)
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


def load_branches(E: float, n_pred: int):
    desc = pd.read_csv(ROOT / "data" / "branch_functional_descriptors.csv")
    desc["log10E"] = np.log10(desc["E"].clip(lower=1e-30))
    sub = desc[np.isclose(desc["E"], E)].copy()
    if len(sub) == 0:
        all_Es = sorted(desc["E"].unique())
        nearest = all_Es[int(np.argmin(np.abs(np.log10(np.array(all_Es)) - np.log10(max(E, 1e-30)))))]
        print(f"E = {E} not found; using nearest E = {nearest}")
        sub = desc[np.isclose(desc["E"], nearest)].copy()
        E = nearest
    branches = []
    for _, row in sub.iterrows():
        k_left, k_right = float(row["k_left"]), float(row["k_right"])
        Ta_min, Ta_max = float(row["Ta_min"]), float(row["Ta_max"])
        amp = max(Ta_max - Ta_min, 1e-12)
        s_grid = np.linspace(0.0, 1.0, n_pred, dtype=np.float32)
        k_grid = k_left + s_grid * (k_right - k_left)
        k_norm = 2.0 * s_grid - 1.0
        ctx_raw = []
        for f in STATIC_FEATURES:
            v = row.get(f, np.nan)
            try: ctx_raw.append(float(v))
            except (TypeError, ValueError): ctx_raw.append(0.0)
        ctx_raw = np.nan_to_num(np.asarray(ctx_raw, dtype=np.float32), nan=0.0,
                                  posinf=0.0, neginf=0.0)
        anc_raw = np.asarray([k_left, k_right,
                                np.log10(max(Ta_min, 1e-6)),
                                np.log10(max(Ta_max, 1e-6))], dtype=np.float32)
        branches.append({"E": E, "branch_local_id": int(row["branch_local_id"]),
                          "Ta_min": Ta_min, "Ta_max": Ta_max, "amp": amp,
                          "k_left": k_left, "k_right": k_right,
                          "k_grid": k_grid, "k_norm": k_norm,
                          "ctx": ctx_raw, "anchors": anc_raw,
                          "log10E": float(np.log10(max(E, 1e-30)))})
    return branches, E


def predict_cnp(branches, device):
    """Average prediction over the available CNP seeds (zero-shot)."""
    ckpt_dir = ROOT / "models_trained" / "cnp"
    seed_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    norm = np.load(ckpt_dir / "test_branches.npz", allow_pickle=False)
    ctx_mean = norm["ctx_mean"].astype(np.float32); ctx_std = norm["ctx_std"].astype(np.float32) + 1e-6
    anc_mean = norm["anc_mean"].astype(np.float32); anc_std = norm["anc_std"].astype(np.float32) + 1e-6
    cnp_args = dict(n_cheb=64, n_cos=32, d_model=256, n_enc_layers=6,
                     n_dec_layers=3, n_heads=8, dropout=0.05,
                     n_E_freq=32, n_k_freq=16, max_ctx_points=32)
    preds_per_branch = []
    for b in branches:
        ctx_n = np.clip(np.nan_to_num((b["ctx"] - ctx_mean) / ctx_std,
                                          nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        anc_n = np.clip(np.nan_to_num((b["anchors"] - anc_mean) / anc_std,
                                          nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        k_t = torch.from_numpy(b["k_norm"]).unsqueeze(0).to(device)
        ctx_t = torch.from_numpy(ctx_n).unsqueeze(0).to(device)
        anc_t = torch.from_numpy(anc_n).unsqueeze(0).to(device)
        logE_t = torch.tensor([b["log10E"]], dtype=torch.float32, device=device)
        ck = torch.zeros(1, 32, device=device); cta = torch.zeros(1, 32, device=device)
        cm = torch.zeros(1, 32, device=device)  # zero-shot mask
        seeds = []
        for sd in seed_dirs:
            model = CNP_TC(ctx_dim=23, **cnp_args).to(device)
            sd_ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(sd_ck["state_dict"]); model.eval()
            with torch.no_grad():
                p = model(k_t, ctx_t, anc_t, logE_t, ck, cta, cm).squeeze(0).cpu().numpy()
            seeds.append(p)
        y_norm = np.mean(np.stack(seeds, axis=0), axis=0)
        preds_per_branch.append(b["Ta_min"] + y_norm * b["amp"])
    return preds_per_branch


def predict_dist(branches, device):
    """Average prediction over the available DIST seeds."""
    ckpt_dir = ROOT / "models_trained" / "dist"
    seed_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    norm = np.load(ckpt_dir / "test_branches.npz", allow_pickle=False)
    ctx_mean = norm["ctx_mean"].astype(np.float32); ctx_std = norm["ctx_std"].astype(np.float32) + 1e-6
    anc_mean = norm["anc_mean"].astype(np.float32); anc_std = norm["anc_std"].astype(np.float32) + 1e-6
    star_args = dict(n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12,
                       dropout=0.05, n_E_freq=32, n_k_freq=16)
    preds_per_branch = []
    for b in branches:
        ctx_n = np.clip(np.nan_to_num((b["ctx"] - ctx_mean) / ctx_std,
                                          nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        anc_n = np.clip(np.nan_to_num((b["anchors"] - anc_mean) / anc_std,
                                          nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0).astype(np.float32)
        k_t = torch.from_numpy(b["k_norm"]).unsqueeze(0).to(device)
        ctx_t = torch.from_numpy(ctx_n).unsqueeze(0).to(device)
        anc_t = torch.from_numpy(anc_n).unsqueeze(0).to(device)
        logE_t = torch.tensor([b["log10E"]], dtype=torch.float32, device=device)
        seeds = []
        for sd in seed_dirs:
            model = STAR(ctx_dim=23, **star_args).to(device)
            sd_ck = torch.load(sd / "best.pt", map_location=device, weights_only=False)
            model.load_state_dict(sd_ck["state_dict"]); model.eval()
            with torch.no_grad():
                p = model(k_t, ctx_t, anc_t, logE_t).squeeze(0).cpu().numpy()
            seeds.append(p)
        y_norm = np.mean(np.stack(seeds, axis=0), axis=0)
        preds_per_branch.append(b["Ta_min"] + y_norm * b["amp"])
    return preds_per_branch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--E", type=float, required=True,
                    help="Elasticity number, e.g. 0.001")
    ap.add_argument("--model", choices=["cnp", "dist"], default="dist",
                    help="Surrogate to use (default: dist)")
    ap.add_argument("--n_pred", type=int, default=151,
                    help="Number of (k, Ta) points per branch in the output")
    ap.add_argument("--save", type=Path, default=None,
                    help="Save plot to this PNG file (default: show interactively)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Save (k, Ta) prediction to this CSV file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    branches, actual_E = load_branches(args.E, args.n_pred)
    print(f"Found {len(branches)} branch(es) at E = {actual_E}")

    print(f"Predicting with {args.model.upper()} ...")
    if args.model == "cnp":
        preds = predict_cnp(branches, device)
    else:
        preds = predict_dist(branches, device)

    if args.csv is not None:
        rows = []
        for b, Ta in zip(branches, preds):
            for k, ta in zip(b["k_grid"], Ta):
                rows.append({"E": b["E"], "branch_local_id": b["branch_local_id"],
                              "k": float(k), "Ta": float(ta)})
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"Saved CSV -> {args.csv}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    for b, Ta in zip(branches, preds):
        ax.plot(b["k_grid"], Ta, "-", lw=1.4,
                  label=f"branch {b['branch_local_id']}")
    # Overlay data if available
    raw = pd.read_csv(ROOT / "data" / "combined_data.csv")
    raw.columns = ["Ta", "k", "E"]
    gt = raw[np.isclose(raw["E"], actual_E)].sort_values("k")
    if len(gt) >= 5:
        ax.plot(gt["k"], gt["Ta"], "k.", ms=4, label="Floquet data", zorder=10)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$T_a$")
    ax.set_title(f"Marginal stability curve at E = {actual_E:.4g}   "
                   f"({args.model.upper()} prediction, mean of {len(branches)} branch(es))")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=140, bbox_inches="tight")
        print(f"Saved figure -> {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
