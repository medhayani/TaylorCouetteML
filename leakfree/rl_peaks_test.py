"""Does the leak-free SARL refiner restore the peaks between modes?

On the 64 test elasticities, the marginal curve is rebuilt twice from the
leak-free ensemble median: as is (base) and with the SARL correction blended
in its window (Tukey taper, seeds weighted by 1/validation MAE, as in the
article).  Peaks of Ta(k) are detected and matched to the Floquet peaks with
the same rules as modes_test.py (prominence >= 2 % of Ta_c, |dk| <= 1).
Diagnostic of the window placement: distance between the window centre
(curvature peak of the base) and the true curvature peak of the branch.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]            # repository root
OUT = REPO / "results/rl"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code")); sys.path.insert(0, str(REPO))
from data_pipeline.dataset import HydraWindowsDataset
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro

kk = np.round(np.arange(1.0, 50.0001, 0.1), 4)
R6 = lambda e: round(float(e), 6)
TAPER = 0.20

# ---------------------------------------------------------------- SARL correction on the test windows
ds = HydraWindowsDataset(REPO / "data/rl_windows/rl_switch_windows_lf_test.npz")
obs, sv = torch.from_numpy(ds.obs_seq), torch.from_numpy(ds.static_vec)
corrs, vmae = [], []
for ck in sorted((REPO / "models_trained/sarl").glob("seed_*/best.pt")):
    st = torch.load(ck, map_location="cpu", weights_only=False); cfg = st["cfg"]
    ext = SARLProFeatureExtractor(obs.shape[2], obs.shape[1], st["static_dim"], cfg["feature_extractor"])
    sac = SACPro(feature_dim=ext.out_dim, action_dim=st["action_dim"], actor_layers=cfg["actor_layers"], critic_layers=cfg["critic_layers"])
    ext.load_state_dict(st["extractor"]); sac.load_state_dict(st["sac"]); ext.eval(); sac.eval()
    with torch.no_grad():
        corrs.append(torch.tanh(sac.actor.mean(sac.actor.body(ext(obs, sv)))).numpy())
    vmae.append(min(r["val_mae"] for r in json.load(open(ck.parent / "history.json")) if r.get("val_mae") is not None))
w_seed = 1.0 / (np.array(vmae) + 1e-6); w_seed /= w_seed.sum()
corr = np.einsum("i...,i->...", np.stack(corrs), w_seed)

# ---------------------------------------------------------------- curves
desc = pd.read_csv(REPO / "data/branch_functional_descriptors_leakfree_all.csv")
med = np.load(REPO / "data/ensemble_median_targets_lf.npz")
mkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
S = np.linspace(0, 1, med["ta_norm_median"].shape[1])
wkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(ds.E, ds.branch_local_id))}
raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
split = pd.read_csv(REPO / "data/split_by_E.csv"); test_E = sorted(R6(e) for e in split.loc[split.split == "test", "E"])


def assemble(pieces):
    out = np.full(len(kk), np.nan)
    for k_arr, T_arr in pieces:
        m = (kk >= k_arr.min() - 1e-9) & (kk <= k_arr.max() + 1e-9)
        v = np.interp(kk[m], k_arr, T_arr)
        cur = out[m]; out[m] = np.where(np.isnan(cur), v, np.minimum(cur, v))
    ok = ~np.isnan(out)
    if ok.sum() >= 2:
        i0, i1 = np.where(ok)[0][[0, -1]]; seg = np.arange(i0, i1 + 1)
        out[seg] = np.interp(kk[seg], kk[ok], out[ok])
    return out


def peaks(curve, thr):
    ok = ~np.isnan(curve)
    if ok.sum() < 5: return np.array([]), np.array([])
    i0, i1 = np.where(ok)[0][[0, -1]]; seg = curve[i0:i1 + 1]
    p, _ = find_peaks(seg, prominence=thr)
    return kk[i0:i1 + 1][p], seg[p]


def match(kt, Ht, kp, Hp, tol):
    used, dk, dh = set(), [], []
    for a in np.argsort(kt):
        cand = [(abs(kp[b] - kt[a]), b) for b in range(len(kp)) if b not in used and abs(kp[b] - kt[a]) <= tol]
        if cand:
            d, b = min(cand); used.add(b); dk.append(d); dh.append(100 * abs(Hp[b] / Ht[a] - 1))
    return len(used), dk, dh


store = {}
for E in test_E:
    g = desc[np.isclose(desc.E, E)].sort_values("branch_local_id")
    pb, pr = [], []
    for _, r in g.iterrows():
        key = (E, int(r.branch_local_id)); base = med["ta_norm_median"][mkey[key]].copy(); ref = base.copy()
        if key in wkey:
            i = wkey[key]; c0 = float(ds.center_pred[i]); h = float(ds.window_half_width[i])
            s_w = np.clip(c0 + h * ds.local_grid[i], 0, 1)
            cf = np.interp(S, s_w, corr[i], left=0.0, right=0.0)
            d = np.abs(S - c0) / max(h, 1e-9)
            w = np.where(d <= 1 - TAPER, 1.0, np.where(d >= 1, 0.0, 0.5 * (1 + np.cos(np.pi * (d - (1 - TAPER)) / TAPER))))
            ref = np.clip(base + w * cf, 0.0, 1.0)
        k_b = r.k_left + S * (r.k_right - r.k_left); amp = r.Ta_max - r.Ta_min
        pb.append((k_b, r.Ta_min + base * amp)); pr.append((k_b, r.Ta_min + ref * amp))
    gt = raw[np.isclose(raw.E, E)].sort_values("k")
    truth = np.interp(kk, gt.k.values, gt.Ta.values, left=np.nan, right=np.nan)
    cv = {"truth": truth, "Median-29": assemble(pb), "Median-29 + SARL": assemble(pr)}
    common = np.all([~np.isnan(v) for v in cv.values()], axis=0)
    store[E] = {n: np.where(common, v, np.nan) for n, v in cv.items()}

res = {"seed_weights": w_seed.tolist()}
for rel in [0.01, 0.02, 0.05]:
    for tol in [0.5, 1.0]:
        tab = {}
        for n in ["Median-29", "Median-29 + SARL"]:
            nt = npred = hits = 0; dks, dhs = [], []
            for E, cvs in store.items():
                thr = rel * np.nanmin(cvs["truth"])
                kt, Ht = peaks(cvs["truth"], thr); kp, Hp = peaks(cvs[n], thr)
                hh, dk, dh = match(kt, Ht, kp, Hp, tol); nt += len(kt); npred += len(kp); hits += hh; dks += dk; dhs += dh
            rec, pre = hits / max(nt, 1), hits / max(npred, 1)
            tab[n] = dict(true_peaks=nt, pred_peaks=npred, hits=hits, recall=rec, precision=pre, F1=2 * rec * pre / max(rec + pre, 1e-12),
                          mean_dk=float(np.mean(dks)) if dks else None, mean_height_err_pct=float(np.mean(dhs)) if dhs else None)
        res[f"rel{rel}_tol{tol}"] = tab

# ---------------------------------------------------------------- window placement
dc = np.abs(ds.center_pred - ds.center_true)
res["window_centre"] = dict(n=int(len(dc)), median_abs_ds=float(np.median(dc)), frac_within_half_width=float(np.mean(dc <= 0.16)),
                            frac_within_quarter_half_width=float(np.mean(dc <= 0.04)))
json.dump(res, open(OUT / "metrics_peaks_rl.json", "w"), indent=1)

t = res["rel0.02_tol1.0"]
print(f"64 test E -- peaks of Ta(k) (prominence >= 2% of Ta_c, |dk| <= 1); true peaks: {t['Median-29']['true_peaks']}")
for n, r in t.items():
    print("  %-17s pred %3d  hits %3d  recall %4.0f%%  precision %4.0f%%  F1 %.2f  |dk| %.2f  height err %.1f%%" % (
        n, r["pred_peaks"], r["hits"], 100 * r["recall"], 100 * r["precision"], r["F1"], r["mean_dk"] or 0, r["mean_height_err_pct"] or 0))
for key in sorted(k for k in res if k.startswith("rel")):
    print("   ", key, "  ".join(f"{n} F1 {res[key][n]['F1']:.2f}" for n in res[key]))
wc = res["window_centre"]
print("window centre vs true curvature peak: median |ds| %.3f ; within the half-width 0.16: %.0f%% ; within 0.04: %.0f%%" % (
    wc["median_abs_ds"], 100 * wc["frac_within_half_width"], 100 * wc["frac_within_quarter_half_width"]))
