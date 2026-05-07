"""End-to-end NEPTUNE_PRO ensemble inference on the canonical 41-pt grid.

Adapted from code6's full_inference_code6.py for the bigger NEPTUNE_PRO model.
Sampling is done on the 101-pt training grid then interpolated to canonical.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common import set_all_seeds, get_logger
from neptune_pro.trainer import NeptuneProSurrogate


STATIC_FEATURES = [
    "log10E", "branch_order_norm", "width_k", "width_asymmetry",
    "rise_asymmetry", "slope_left_local", "slope_right_local", "global_slope",
    "curvature_at_min", "roughness_rmse", "normalized_arc_length",
    "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude",
    "n_branches", "is_first_branch", "is_last_branch",
    "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    ap.add_argument("--canonical_csv", required=True, type=Path)
    ap.add_argument("--profile_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--num_steps", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log = get_logger("full_inference_pro")
    set_all_seeds(args.seed, deterministic=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- load M models ----
    log.info(f"loading {len(args.checkpoints)} NEPTUNE_PRO checkpoints")
    first = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    ctx_mean = np.asarray(first["ctx_mean"], dtype=np.float32)
    ctx_std = np.asarray(first["ctx_std"], dtype=np.float32)
    models = []
    for cp in args.checkpoints:
        c = torch.load(cp, map_location="cpu", weights_only=False)
        m = NeptuneProSurrogate(c["cfg"]).to(device); m.load_state_dict(c["state_dict"])
        m.eval(); models.append(m)
        log.info(f"  {cp.parent.name}: params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    # ---- build context lookup from profile CSV ----
    prof = pd.read_csv(args.profile_csv)
    prof = prof.replace([np.inf, -np.inf], np.nan)
    branch_first = (prof.sort_values(["E", "branch_local_id", "s"])
                       .groupby(["E", "branch_local_id"], sort=False)
                       .first().reset_index())
    ctx_lookup = {}
    for _, r in branch_first.iterrows():
        feats = []
        for c in STATIC_FEATURES:
            try: feats.append(float(r.get(c, 0.0)))
            except Exception: feats.append(0.0)
        ctx_lookup[(float(r["E"]), int(r["branch_local_id"]))] = np.asarray(feats, dtype=np.float32)

    canon = pd.read_csv(args.canonical_csv)
    canon = canon.sort_values(["split", "E", "branch_local_id", "s"]).reset_index(drop=True)
    s_train = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    BATCH = 8

    rows_out: List[dict] = []
    grouped = list(canon.groupby(["split", "E", "branch_local_id"], sort=False))
    log.info(f"running ensemble M={len(models)} on {len(grouped)} curves...")
    bk, bc, bs = [], [], []
    t0 = time.time()

    def _flush(keys, ctx_list, canon_s_list):
        if not keys: return
        ctx_t = torch.from_numpy(np.stack(ctx_list, axis=0)).to(device)
        ctx_t = (ctx_t - torch.from_numpy(ctx_mean).to(device)) / torch.from_numpy(ctx_std).to(device)
        ctx_t = torch.nan_to_num(ctx_t, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        s_t = torch.from_numpy(np.tile(s_train, (len(keys), 1))).to(device)
        member_preds = []
        with torch.no_grad():
            for m in models:
                member_preds.append(m.sample(ctx_t, s_t, num_steps=args.num_steps).cpu().numpy())
        stack = np.stack(member_preds, axis=0)
        mean = stack.mean(axis=0)
        sigma_ep = stack.std(axis=0) if stack.shape[0] > 1 else np.zeros_like(mean)
        for j, (split, E, b) in enumerate(keys):
            ta_canon = np.interp(canon_s_list[j], s_train, mean[j])
            sig_canon = np.interp(canon_s_list[j], s_train, sigma_ep[j])
            rows_out.append({"split": split, "E": E, "branch_local_id": b,
                              "Ta_neptune_pro": ta_canon, "sigma_ep": sig_canon,
                              "s_grid": canon_s_list[j]})

    for k, ((split, E, b), g) in enumerate(grouped):
        ctx_v = ctx_lookup.get((float(E), int(b)))
        if ctx_v is None: continue
        s_canon = g["s"].to_numpy(dtype=np.float32)
        bk.append((str(split), float(E), int(b))); bc.append(ctx_v); bs.append(s_canon)
        if len(bk) >= BATCH:
            _flush(bk, bc, bs); bk, bc, bs = [], [], []
            if (k + 1) % 80 == 0:
                log.info(f"  ... {k+1}/{len(grouped)} ({(time.time()-t0)/60:.1f} min)")
    _flush(bk, bc, bs)
    log.info(f"inference done in {(time.time()-t0)/60:.1f} min")

    # ---- long-form CSV ----
    long_rows = []
    for entry in rows_out:
        for j, sj in enumerate(entry["s_grid"]):
            long_rows.append({"split": entry["split"], "E": entry["E"],
                               "log10E": float(np.log10(max(entry["E"], 1e-30))),
                               "branch_local_id": int(entry["branch_local_id"]),
                               "point_id": j, "s": float(sj), "k": float(sj),
                               "Ta_neptune_pro": float(entry["Ta_neptune_pro"][j]),
                               "sigma_ep": float(entry["sigma_ep"][j])})
    df = pd.DataFrame(long_rows)
    df = df.merge(canon[["split", "E", "branch_local_id", "point_id", "Ta_true"]],
                  on=["split", "E", "branch_local_id", "point_id"], how="inner")
    out_csv = args.out_dir / "canonical_neptune_pro.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"saved -> {out_csv}  ({len(df)} rows)")

    # ---- per-split metrics ----
    rows = []
    for split, g in df.groupby("split"):
        err = g["Ta_true"].to_numpy() - g["Ta_neptune_pro"].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((g["Ta_true"] - g["Ta_true"].mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({"split": split, "model": "NEPTUNE_PRO M=3 ensemble",
                      "mae": mae, "rmse": rmse, "r2": r2,
                      "n_rows": len(g),
                      "n_branches": g[["E", "branch_local_id"]].drop_duplicates().shape[0]})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.out_dir / "metrics_neptune_pro.csv", index=False)
    print()
    print("=" * 80)
    print(f"{'split':<8} {'model':<28} {'MAE':>10} {'RMSE':>10} {'R^2':>9} {'rows':>7} {'curves':>7}")
    print("-" * 80)
    for _, r in metrics_df.sort_values(["split", "mae"]).iterrows():
        print(f"{r['split']:<8} {r['model']:<28} {r['mae']:>10.5f} "
              f"{r['rmse']:>10.5f} {r['r2']:>9.5f} "
              f"{int(r['n_rows']):>7} {int(r['n_branches']):>7}")
    print("=" * 80)


if __name__ == "__main__":
    main()
