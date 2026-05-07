"""SSST_PRO inference on canonical grid."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import set_all_seeds, get_logger
from ssst_pro.ssst_model import SSSTProSurrogate

STATIC_FEATURES = ["log10E","branch_order_norm","width_k","width_asymmetry",
    "rise_asymmetry","slope_left_local","slope_right_local","global_slope",
    "curvature_at_min","roughness_rmse","normalized_arc_length",
    "has_switch_left","has_switch_right","mean_abs_curvature","amplitude",
    "n_branches","is_first_branch","is_last_branch",
    "left_width","right_width","left_rise","right_rise","mean_abs_slope"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--canonical_csv", required=True, type=Path)
    ap.add_argument("--profile_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    log = get_logger("ssst_pro_inf")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(42, deterministic=False)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SSSTProSurrogate(ck["cfg"]); model.load_state_dict(ck["state_dict"]); model.eval()
    ctx_mean = np.asarray(ck["ctx_mean"], dtype=np.float32)
    ctx_std = np.asarray(ck["ctx_std"], dtype=np.float32)
    log.info(f"loaded {args.checkpoint}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    prof = pd.read_csv(args.profile_csv).replace([np.inf, -np.inf], np.nan)
    branch_first = (prof.sort_values(["E","branch_local_id","s"])
                       .groupby(["E","branch_local_id"], sort=False).first().reset_index())
    ctx_lookup = {}
    for _, r in branch_first.iterrows():
        feats = []
        for c in STATIC_FEATURES:
            try: feats.append(float(r.get(c, 0.0)))
            except: feats.append(0.0)
        ctx_lookup[(float(r["E"]), int(r["branch_local_id"]))] = np.asarray(feats, dtype=np.float32)

    canon = pd.read_csv(args.canonical_csv).sort_values(["split","E","branch_local_id","s"]).reset_index(drop=True)
    s_train = np.linspace(0.0, 1.0, 101, dtype=np.float32)

    rows_out = []
    grouped = list(canon.groupby(["split","E","branch_local_id"], sort=False))
    BATCH = 16; bk = []; bc = []; bs = []

    def _flush(keys, ctx_list, canon_s_list):
        if not keys: return
        ctx_t = torch.from_numpy(np.stack(ctx_list, axis=0))
        ctx_t = (ctx_t - torch.from_numpy(ctx_mean)) / torch.from_numpy(ctx_std)
        ctx_t = torch.nan_to_num(ctx_t, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        s_t = torch.from_numpy(np.tile(s_train, (len(keys), 1)))
        with torch.no_grad():
            preds = model.predict(ctx_t, s_t).numpy()      # (B, 101)
        for j, (split, E, b) in enumerate(keys):
            ta_canon = np.interp(canon_s_list[j], s_train, preds[j])
            rows_out.append({"split": split, "E": E, "branch_local_id": b,
                              "Ta_ssst_pro": ta_canon, "s_grid": canon_s_list[j]})

    t0 = time.time()
    for k, ((split, E, b), g) in enumerate(grouped):
        ctx_v = ctx_lookup.get((float(E), int(b)))
        if ctx_v is None: continue
        bk.append((str(split), float(E), int(b))); bc.append(ctx_v); bs.append(g["s"].to_numpy(dtype=np.float32))
        if len(bk) >= BATCH:
            _flush(bk, bc, bs); bk, bc, bs = [], [], []
    _flush(bk, bc, bs)
    log.info(f"inference done in {time.time()-t0:.1f}s")

    long_rows = []
    for entry in rows_out:
        for j, sj in enumerate(entry["s_grid"]):
            long_rows.append({"split": entry["split"], "E": entry["E"],
                               "log10E": float(np.log10(max(entry["E"], 1e-30))),
                               "branch_local_id": int(entry["branch_local_id"]),
                               "point_id": j, "s": float(sj),
                               "Ta_ssst_pro": float(entry["Ta_ssst_pro"][j])})
    df = pd.DataFrame(long_rows)
    df = df.merge(canon[["split","E","branch_local_id","point_id","Ta_true"]],
                  on=["split","E","branch_local_id","point_id"], how="inner")
    out = args.out_dir / "canonical_ssst_pro.csv"
    df.to_csv(out, index=False)
    log.info(f"saved -> {out}")

    rows = []
    for split, g in df.groupby("split"):
        err = g["Ta_true"].to_numpy() - g["Ta_ssst_pro"].to_numpy()
        mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((g["Ta_true"] - g["Ta_true"].mean())**2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        rows.append({"split": split, "model": "SSST_PRO",
                      "mae": mae, "rmse": rmse, "r2": r2,
                      "n_rows": len(g),
                      "n_branches": g[["E","branch_local_id"]].drop_duplicates().shape[0]})
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics_ssst_pro.csv", index=False)
    print("\n" + "="*70)
    for r in rows:
        print(f"{r['split']:<8} SSST_PRO  MAE={r['mae']:.5f}  RMSE={r['rmse']:.5f}  R2={r['r2']:.5f}")
    print("="*70)


if __name__ == "__main__":
    main()
