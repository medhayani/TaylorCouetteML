"""Interpolation baselines on the 64 held-out elasticities, without any network.

Same data budget as the models: everything is fitted on the 292 training
elasticities of data/split_by_E.csv and evaluated on the 64 test ones.

  PCHIP-1D        PCHIP of Ta_c(E) and k_c(E) -> critical point only
  PCHIP-2D        PCHIP in log10 E, at fixed k, of the full marginal curves
  Branch-interp   leak-free branch structure and anchors (as for the models),
                  normalised branch shapes interpolated between the two
                  neighbouring training elasticities -> no learning at all

Writes results/baselines/metrics.json and per_E.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "baselines"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "leakfree"))
from leakfree_descriptors import interpolate_descriptors  # noqa: E402

R6 = lambda e: round(float(e), 6)                                            # noqa: E731
S = np.linspace(0.0, 1.0, 101)
kk = np.round(np.arange(1.0, 50.0001, 0.1), 4)

raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
curves = {R6(E): g.sort_values("k") for E, g in raw.groupby("E")}
Es = np.array(sorted(curves)); lx = np.log10(Es)
M = np.full((len(Es), len(kk)), np.nan)
for i, E in enumerate(Es):
    g = curves[E]; M[i] = interp1d(g.k.values, np.log10(g.Ta.values), bounds_error=False)(kk)
Tc = np.array([10 ** np.nanmin(M[i]) for i in range(len(Es))])
kc = np.array([kk[np.nanargmin(M[i])] for i in range(len(Es))])
split = pd.read_csv(REPO / "data/split_by_E.csv"); lab = {R6(e): s for e, s in zip(split.E, split.split)}
is_tr = np.array([lab.get(E) == "train" for E in Es]); is_te = np.array([lab.get(E) == "test" for E in Es])
desc_true = pd.read_csv(REPO / "data/branch_functional_descriptors.csv")
true_groups = {R6(E): g.sort_values("branch_local_id").reset_index(drop=True) for E, g in desc_true.groupby("E")}
desc_lf = interpolate_descriptors(desc_true, Es[is_tr], margin=0.05)


def assemble(pieces):
    out = np.full(len(kk), np.nan)
    for k_arr, T_arr in pieces:
        m = (kk >= k_arr.min() - 1e-9) & (kk <= k_arr.max() + 1e-9)
        v = np.interp(kk[m], k_arr, T_arr)
        out[m] = np.where(np.isnan(out[m]), v, np.minimum(out[m], v))
    ok = ~np.isnan(out)
    if ok.sum() >= 2:
        i0, i1 = np.where(ok)[0][[0, -1]]; seg = np.arange(i0, i1 + 1)
        out[seg] = np.interp(kk[seg], kk[ok], out[ok])
    return out


def pchip2d():
    tgt = np.where(is_te)[0]; kp = np.where(is_tr)[0]
    P = np.full((len(tgt), len(kk)), np.nan)
    for j in range(len(kk)):
        col = M[:, j]; ok = is_tr & ~np.isnan(col)
        if ok.sum() >= 4:
            P[:, j] = PchipInterpolator(lx[ok], col[ok], extrapolate=True)(lx[tgt])
    out = {}
    for a, i in enumerate(tgt):
        lo = kp[kp < i]; hi = kp[kp > i]
        nb = [x for x in [lo[-1] if len(lo) else None, hi[0] if len(hi) else None] if x is not None]
        sup = np.all([~np.isnan(M[n]) for n in nb], axis=0)
        out[Es[i]] = np.where(sup, 10 ** P[a], np.nan)
    return out


def shape(Enb, b):
    r = true_groups[R6(Enb)].iloc[b]; g = curves[R6(Enb)]
    ta = np.interp(r.k_left + S * (r.k_right - r.k_left), g.k.values, g.Ta.values)
    return (ta - r.Ta_min) / max(r.Ta_max - r.Ta_min, 1e-12)


def branch_interp(E, margin=0.05):
    pieces = []
    for _, r in desc_lf[np.isclose(desc_lf.E, E)].sort_values("branch_local_id").iterrows():
        lo, hi, b = float(r.source_lo), float(r.source_hi), int(r.branch_local_id)
        lE, llo, lhi = np.log10(E), np.log10(lo), np.log10(hi)
        w = 0.0 if np.isclose(lhi, llo) else float(np.clip((lE - llo) / (lhi - llo), -0.5, 1.5))
        if int(r.same_branch_count) == 1:
            y = (1 - w) * shape(lo, b) + w * shape(hi, b)
        else:
            y = shape(lo if abs(lE - llo) <= abs(lE - lhi) else hi, b)
        tmin, tmax = r.Ta_min / (1 - margin), r.Ta_max / (1 + margin)
        pieces.append((r.k_left + S * (r.k_right - r.k_left), tmin + y * (tmax - tmin)))
    return assemble(pieces)


c2d = pchip2d()
kc1d = dict(zip(Es[is_te], PchipInterpolator(lx[is_tr], kc[is_tr])(lx[is_te])))
tc1d = dict(zip(Es[is_te], 10 ** PchipInterpolator(lx[is_tr], np.log10(Tc[is_tr]))(lx[is_te])))
rows = []
for E in Es[is_te]:
    i = int(np.where(Es == E)[0][0]); truth = 10 ** M[i]
    for name, curve in [("PCHIP-2D", c2d[E]), ("Branch-interp", branch_interp(E))]:
        ok = ~np.isnan(curve) & ~np.isnan(truth)
        rows.append(dict(method=name, E=float(E), Ta_c_true=Tc[i], k_c_true=kc[i],
                         Ta_c_pred=float(np.nanmin(curve)), k_c_pred=float(kk[np.nanargmin(curve)]),
                         curve_MAPE=float(100 * np.mean(np.abs(curve[ok] / truth[ok] - 1))),
                         covered_frac=float(ok.sum() / (~np.isnan(truth)).sum())))
    rows.append(dict(method="PCHIP-1D", E=float(E), Ta_c_true=Tc[i], k_c_true=kc[i],
                     Ta_c_pred=float(tc1d[E]), k_c_pred=float(kc1d[E]), curve_MAPE=np.nan, covered_frac=np.nan))
d = pd.DataFrame(rows); d["err_Ta_c_pct"] = 100 * (d.Ta_c_pred / d.Ta_c_true - 1).abs()
d["err_k_c"] = (d.k_c_pred - d.k_c_true).abs()
d.to_csv(OUT / "per_E.csv", index=False)
res = {m: dict(n_test_E=int((d.method == m).sum()),
               MAPE_Ta=float(d[d.method == m].err_Ta_c_pct.mean()),
               MAE_k=float(d[d.method == m].err_k_c.mean()),
               curve_MAPE=float(np.nanmean(d[d.method == m].curve_MAPE)),
               curve_coverage=float(np.nanmean(d[d.method == m].covered_frac)))
       for m in ["PCHIP-1D", "PCHIP-2D", "Branch-interp"]}
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
for m, v in res.items():
    print(f"{m:14s} Ta_c {v['MAPE_Ta']:.2f} %   k_c {v['MAE_k']:.2f}   curve {v['curve_MAPE']:.2f} % "
          f"(on {100 * v['curve_coverage']:.0f} % of the Floquet grid)")
