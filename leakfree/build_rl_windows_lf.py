"""Leak-free RL switch windows (mirror of stepF_export_switch_rl_windows_pro_v2).

One window per predicted branch, centred on the curvature peak of the BASE
prediction (the leak-free ensemble median), half-width 0.16 in s, 49 points.
  obs_seq   (N, 49, 9): s_rel, y_pred, dy, d2y, |dy|, |d2y|, switch_prob (=0,
             no switch head), switch_focus, curv_focus
  static_vec (N, 23):   13 leak-free descriptors + 10 engineered scalars
  y_true    (N, 49):    true marginal curve on the window grid, normalised
             with the LEAK-FREE anchors (as everywhere else)
  y_pred    (N, 49):    base prediction (ensemble median) on the window grid
Nothing from the target curve of E enters the inputs; y_true is the training
target only.  Split by elasticity from split_by_E.csv.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

RAW_STATIC = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry",
              "slope_left_local", "slope_right_local", "global_slope", "curvature_at_min",
              "roughness_rmse", "normalized_arc_length", "has_switch_left", "has_switch_right"]
EPS = 1e-12


def d1(y, x): return np.gradient(y, x).astype(np.float32)
def d2(y, x): return np.gradient(np.gradient(y, x), x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", required=True); ap.add_argument("--median", required=True)
    ap.add_argument("--raw", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--half_width", type=float, default=0.16); ap.add_argument("--points", type=int, default=49)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    desc = pd.read_csv(a.desc); desc["log10E"] = np.log10(desc.E.clip(lower=1e-30))
    n_b = desc.groupby("E").branch_local_id.transform("count")
    desc["n_branches"] = n_b; desc["branch_order_norm"] = np.where(n_b > 1, desc.branch_local_id / (n_b - 1).clip(lower=1), 0.0)
    med = np.load(a.median); mkey = {(round(float(e), 6), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
    s_full = np.linspace(0.0, 1.0, med["ta_norm_median"].shape[1], dtype=np.float32)
    raw = pd.read_csv(a.raw); raw.columns = ["Ta", "k", "E"]
    curves = {round(float(E), 8): g.sort_values("k") for E, g in raw.groupby("E")}
    split = pd.read_csv(a.split); lab = {round(float(r.E), 8): str(r.split) for r in split.itertuples()}
    grid = np.linspace(-1.0, 1.0, a.points, dtype=np.float32)
    payload = {k: [] for k in ["train", "val", "test"]}
    wid = 0
    for _, r in desc.iterrows():
        E = float(r.E); b = int(r.branch_local_id); key = (round(E, 8), b)
        sp = lab.get(round(E, 8));
        if sp not in payload or key not in mkey: continue
        y_pred = med["ta_norm_median"][mkey[key]].astype(np.float32)
        g = curves[round(E, 8)]; k_grid = r.k_left + s_full * (r.k_right - r.k_left)
        ta = np.interp(k_grid, g.k.values, g.Ta.values); amp = max(r.Ta_max - r.Ta_min, 1e-12)
        y_true = np.clip((ta - r.Ta_min) / amp, 0.0, 1.0).astype(np.float32)
        center_pred = float(np.clip(s_full[int(np.argmax(np.abs(d2(y_pred, s_full))))], 0, 1))     # curvature peak of the base
        center_true = float(s_full[int(np.argmax(np.abs(d2(y_true, s_full))))])                     # diagnostic only
        s_c = np.clip(center_pred + a.half_width * grid, 0.0, 1.0)
        yt_w = np.interp(s_c, s_full, y_true).astype(np.float32); yp_w = np.interp(s_c, s_full, y_pred).astype(np.float32)
        dy_w = np.interp(s_c, s_full, d1(y_pred, s_full)); d2y_w = np.interp(s_c, s_full, d2(y_pred, s_full))
        sw_prob = np.zeros_like(grid); sw_focus = np.exp(-0.5 * (grid / 0.30) ** 2).astype(np.float32)
        curv = np.abs(d2y_w); curv_focus = (curv / curv.max()).astype(np.float32) if curv.max() > 1e-8 else np.zeros_like(grid)
        obs = np.stack([grid, yp_w, dy_w, d2y_w, np.abs(dy_w), np.abs(d2y_w), sw_prob, sw_focus, curv_focus], -1)
        obs = np.clip(np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0).astype(np.float32)
        base_mae = float(np.mean(np.abs(yt_w - yp_w))); base_rmse = float(np.sqrt(np.mean((yt_w - yp_w) ** 2)))
        raw_static = np.array([float(r.get(c, 0.0)) if pd.notna(r.get(c, np.nan)) else 0.0 for c in RAW_STATIC], np.float32)
        # Leak-free: Step F put six scalars computed on the TRUE curve into the
        # static vector (true centre, has_true_center, centre error, relative
        # centre error, base MAE, base RMSE).  They are neutralised here, in
        # every split, so that the refiner never sees them.  The true values are
        # still saved below as separate diagnostic arrays, unused by the trainers.
        eng = np.array([center_pred, center_pred, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, float(curv.max()), a.half_width], np.float32)
        payload[sp].append(dict(obs_seq=obs, static_vec=np.concatenate([raw_static, eng]), y_true=yt_w, y_pred=yp_w, local_grid=grid,
                                center_pred=center_pred, center_true=center_true, window_half_width=a.half_width,
                                baseline_mae=base_mae, baseline_rmse=base_rmse, E=E, log10E=np.log10(E), branch_local_id=b, window_id=wid))
        wid += 1
    for sp, rows in payload.items():
        if not rows: continue
        np.savez(out / f"rl_switch_windows_lf_{sp}.npz",
                 **{k: np.stack([w[k] for w in rows]).astype(np.float32) for k in ["obs_seq", "static_vec", "y_true", "y_pred", "local_grid"]},
                 **{k: np.array([w[k] for w in rows], np.float32) for k in ["center_pred", "center_true", "window_half_width", "baseline_mae", "baseline_rmse", "E", "log10E"]},
                 branch_local_id=np.array([w["branch_local_id"] for w in rows], np.int32), window_id=np.array([w["window_id"] for w in rows], np.int32))
        print(f"{sp:5s}: {len(rows)} windows  base MAE {np.mean([w['baseline_mae'] for w in rows]):.4f}")
    (out / "manifest.json").write_text(json.dumps({"desc": a.desc, "median": a.median, "half_width": a.half_width, "points": a.points,
                                                   "static_features": RAW_STATIC + ["center_pred", "center_true", "has_true_center", "center_error_s",
                                                   "center_error_rel", "baseline_mae", "baseline_rmse", "switch_prob_peak", "curv_peak_abs", "window_half_width"]}, indent=2))


if __name__ == "__main__":
    main()
