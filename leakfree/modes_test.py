"""Point 6 -- do the methods reproduce the MODE STRUCTURE, not only the threshold?

Test 1 (64 test elasticities, random split): peaks of the marginal curve Ta(k)
between successive branches (the signature of the mode exchanges along k),
and identification of the dominant mode (true branch carrying k_c).
Test 2 (resonance block E in [0.08, 0.16] withheld): mode exchanges along E.

Methods, all leak-free, same training elasticities:
  PCHIP-1D      PCHIP of Ta_c(E), k_c(E)              -> critical point only, no curve
  PCHIP-2D      PCHIP in log10 E at fixed k of the full marginal curves
  Branch-interp branch-aware interpolation, no learning: interpolated branch supports
                and anchors, normalised branch shapes interpolated between neighbours
  Median-29     median of the 29 leak-free supervised networks
  CNP-cond      CNP conditioned on the neighbour-interpolated curve
  CNP-block     zero-shot leak-free CNP trained without the resonance block (test 2)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]            # repository root
OUT = REPO / "results/modes"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code")); sys.path.insert(0, str(REPO))
from models.cnp_tc.model import CNP_TC

kk = np.round(np.arange(1.0, 50.0001, 0.1), 4)
R6 = lambda e: round(float(e), 6)
S = np.linspace(0, 1, 101)
FEATS = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry", "slope_left_local",
         "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse", "normalized_arc_length",
         "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch",
         "is_last_branch", "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]

# ------------------------------------------------------------------ truth
raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
curves = {R6(E): g.sort_values("k") for E, g in raw.groupby("E")}
Es = np.array(sorted(curves)); lx = np.log10(Es)
M = np.full((len(Es), len(kk)), np.nan)
for i, E in enumerate(Es):
    g = curves[E]; M[i] = interp1d(g.k.values, np.log10(g.Ta.values), bounds_error=False)(kk)
Tc = np.array([10 ** np.nanmin(M[i]) for i in range(len(Es))]); kc = np.array([kk[np.nanargmin(M[i])] for i in range(len(Es))])
ix = {E: i for i, E in enumerate(Es)}
desc_true = pd.read_csv(REPO / "data/branch_functional_descriptors.csv")
true_groups = {R6(E): g.sort_values("branch_local_id").reset_index(drop=True) for E, g in desc_true.groupby("E")}


def branch_of(E, k):
    g = true_groups[R6(E)]
    d = np.maximum(np.maximum(g.k_left.values - k, 0), k - g.k_right.values)
    c = np.abs(k - 0.5 * (g.k_left.values + g.k_right.values))
    return int(np.lexsort((c, d))[0])


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


def pchip2d(keep, target):
    tgt = np.where(target)[0]; kp = np.where(keep)[0]
    P = np.full((len(tgt), len(kk)), np.nan)
    for j in range(len(kk)):
        col = M[:, j]; ok = keep & ~np.isnan(col)
        if ok.sum() >= 4: P[:, j] = PchipInterpolator(lx[ok], col[ok], extrapolate=True)(lx[tgt])
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


def branch_interp(E, gl, margin=0.05):
    pieces = []
    for _, r in gl.iterrows():
        lo, hi, b = float(r.source_lo), float(r.source_hi), int(r.branch_local_id)
        lE, llo, lhi = np.log10(E), np.log10(lo), np.log10(hi)
        w = 0.0 if np.isclose(lhi, llo) else float(np.clip((lE - llo) / (lhi - llo), -0.5, 1.5))
        if int(r.same_branch_count) == 1: y = (1 - w) * shape(lo, b) + w * shape(hi, b)
        else: y = shape(lo if abs(lE - llo) <= abs(lE - lhi) else hi, b)
        tmin, tmax = r.Ta_min / (1 - margin), r.Ta_max / (1 + margin)
        pieces.append((r.k_left + S * (r.k_right - r.k_left), tmin + y * (tmax - tmin)))
    return assemble(pieces)


med = np.load(REPO / "data/ensemble_median_targets_lf.npz")
mkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
def median_curve(E, gl):
    return assemble([(r.k_left + S * (r.k_right - r.k_left), r.Ta_min + med["ta_norm_median"][mkey[(R6(E), int(r.branch_local_id))]] * (r.Ta_max - r.Ta_min))
                     for _, r in gl.iterrows()])


run = REPO / "models_trained/cnp_nbr"
targs = json.load(open(run / "leakfree_diag.json"))["args"]
HP = dict(ctx_dim=23, **{k: targs[k] for k in ["n_cheb", "n_cos", "d_model", "n_enc_layers", "n_dec_layers", "n_heads", "dropout", "n_E_freq", "n_k_freq", "max_ctx_points"]})
cnps = []
for sd in sorted(run.glob("seed_*/best.pt")):
    m = CNP_TC(**HP); m.load_state_dict(torch.load(sd, map_location="cpu")["state_dict"]); m.eval(); cnps.append(m)
st = np.load(run / "test_branches.npz"); cm, cs, am, asd = st["ctx_mean"], st["ctx_std"], st["anc_mean"], st["anc_std"]
zc = np.load(run / "context_curves.npz"); ctxc = {(round(float(e), 8), int(b)): t for e, b, t in zip(zc["E"], zc["branch_local_id"], zc["ta"])}
cidx = np.unique(np.round(np.linspace(0, 100, int(targs.get("n_ctx", 32)))).astype(int))
k_norm = torch.tensor(2 * S - 1, dtype=torch.float32).unsqueeze(0)
def cnp_curve(E, gl):
    pieces = []
    for _, r in gl.iterrows():
        ctx = np.array([float(r.get(f)) if (f in r.index and pd.notna(r.get(f))) else 0.0 for f in FEATS], np.float32)
        ctx = np.clip(np.nan_to_num((ctx - cm) / cs), -5, 5).astype(np.float32)
        anc = np.array([r.k_left, r.k_right, np.log10(max(r.Ta_min, 1e-6)), np.log10(max(r.Ta_max, 1e-6))], np.float32)
        anc = np.clip(np.nan_to_num((anc - am) / asd), -5, 5).astype(np.float32)
        P = HP["max_ctx_points"]; ck = torch.zeros(1, P); cta = torch.zeros(1, P); cmk = torch.zeros(1, P)
        c = ctxc[(round(E, 8), int(r.branch_local_id))]
        ck[0, :len(cidx)] = k_norm[0, cidx]; cta[0, :len(cidx)] = torch.from_numpy(c[cidx]); cmk[0, :len(cidx)] = 1.0
        with torch.no_grad():
            y = np.mean([mm(k_norm, torch.from_numpy(ctx).unsqueeze(0), torch.from_numpy(anc).unsqueeze(0),
                            torch.tensor([np.log10(E)], dtype=torch.float32), ck, cta, cmk).squeeze(0).numpy() for mm in cnps], 0)
        pieces.append((r.k_left + S * (r.k_right - r.k_left), r.Ta_min + y * (r.Ta_max - r.Ta_min)))
    return assemble(pieces)


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


# ------------------------------------------------------------------ test 1
split = pd.read_csv(REPO / "data/split_by_E.csv"); lab = {R6(e): s for e, s in zip(split.E, split.split)}
is_tr = np.array([lab.get(E) == "train" for E in Es]); is_te = np.array([lab.get(E) == "test" for E in Es])
dlf = pd.read_csv(REPO / "data/branch_functional_descriptors_leakfree_all.csv")
g_lf = {R6(E): g.sort_values("branch_local_id") for E, g in dlf.groupby("E")}
dcn = pd.read_csv(run / "descriptors_leakfree.csv"); dcn["log10E"] = np.log10(dcn.E.clip(lower=1e-30))
g_cn = {R6(E): g.sort_values("branch_local_id") for E, g in dcn.groupby("E")}
c2d = pchip2d(is_tr, is_te)
kc1d = dict(zip(Es[is_te], PchipInterpolator(lx[is_tr], kc[is_tr])(lx[is_te])))
METHODS = ["PCHIP-2D", "Branch-interp", "Median-29", "CNP-cond"]
store = {}
for E in Es[is_te]:
    i = ix[E]
    cv = {"truth": 10 ** M[i], "PCHIP-2D": c2d[E], "Branch-interp": branch_interp(E, g_lf[E]),
          "Median-29": median_curve(E, g_lf[E]), "CNP-cond": cnp_curve(E, g_cn[E])}
    common = np.all([~np.isnan(v) for v in cv.values()], axis=0)
    store[E] = {n: np.where(common, v, np.nan) for n, v in cv.items()} | {"_full": cv}
res = {}
for rel in [0.01, 0.02, 0.05]:
    for tol in [0.5, 1.0]:
        tab = {}
        for n in METHODS:
            nt = npred = hits = exact = 0; dks, dhs = [], []
            for E, cvs in store.items():
                thr = rel * Tc[ix[E]]
                kt, Ht = peaks(cvs["truth"], thr); kp, Hp = peaks(cvs[n], thr)
                h, dk, dh = match(kt, Ht, kp, Hp, tol)
                nt += len(kt); npred += len(kp); hits += h; dks += dk; dhs += dh; exact += int(len(kt) == len(kp))
            rec = hits / max(nt, 1); pre = hits / max(npred, 1)
            tab[n] = dict(true_peaks=nt, pred_peaks=npred, hits=hits, recall=rec, precision=pre,
                          F1=2 * rec * pre / max(rec + pre, 1e-12), mean_dk=float(np.mean(dks)) if dks else None,
                          mean_height_err_pct=float(np.mean(dhs)) if dhs else None, count_exact_frac=exact / len(store))
        res[f"rel{rel}_tol{tol}"] = tab
dom = {}
for n in ["PCHIP-1D"] + METHODS:
    ok = []; errk = []
    for E, cvs in store.items():
        bt = branch_of(E, kc[ix[E]])
        if n == "PCHIP-1D": kpred = float(kc1d[E])
        else:
            f = cvs["_full"][n]; kpred = kk[np.nanargmin(f)]
        ok.append(branch_of(E, kpred) == bt); errk.append(abs(kpred - kc[ix[E]]))
    dom[n] = dict(dominant_mode_accuracy=float(np.mean(ok)), MAE_k_c=float(np.mean(errk)))
res["dominant_mode"] = dom

# ------------------------------------------------------------------ test 2: resonance block
blk = (Es >= 0.08) & (Es <= 0.16)
spb = pd.read_csv(REPO / "data/split_block.csv"); labb = {R6(e): s for e, s in zip(spb.E, spb.split)}
trb = np.array([labb.get(E) == "train" for E in Es])
c2db = pchip2d(trb, blk)
runb = REPO / "models_trained/cnp_block"
dbl = pd.read_csv(runb / "descriptors_leakfree.csv"); g_b = {R6(E): g.sort_values("branch_local_id") for E, g in dbl.groupby("E")}
peb = pd.read_csv(REPO / "results/cnp_block/per_E.csv"); cnpb = {R6(e): k for e, k in zip(peb.E, peb.k_c_pred)}
Eb = Es[blk]; kct = kc[blk]
series = {"truth": kct, "PCHIP-1D": PchipInterpolator(lx[trb], kc[trb])(lx[blk]),
          "PCHIP-2D": np.array([kk[np.nanargmin(c2db[E])] for E in Eb]),
          "Branch-interp": np.array([kk[np.nanargmin(branch_interp(E, g_b[E]))] for E in Eb]),
          "CNP-block": np.array([cnpb[R6(E)] for E in Eb])}
jt = set(np.where(np.abs(np.diff(kct)) > 1.0)[0])
blockres = {}
for n, s_ in series.items():
    jp = set(np.where(np.abs(np.diff(s_)) > 1.0)[0])
    hit = sum(1 for j in jt if any(abs(j - q) <= 1 for q in jp))
    blockres[n] = dict(exchanges_true=len(jt), exchanges_pred=len(jp), exchanges_found_pm1step=hit,
                       dominant_mode_accuracy=float(np.mean([branch_of(E, k) == branch_of(E, t) for E, k, t in zip(Eb, s_, kct)])),
                       MAE_k_c=float(np.mean(np.abs(s_ - kct))))
res["block"] = blockres
json.dump(res, open(OUT / "metrics_modes.json", "w"), indent=1)

# ------------------------------------------------------------------ print
t = res["rel0.02_tol1.0"]
print(f"TEST 1 -- 64 test E, peaks of Ta(k) (prominence >= 2% of Ta_c, match within dk <= 1)")
print(f"  true peaks in total: {t['CNP-cond']['true_peaks']}")
print("  %-14s %6s %6s %7s %8s %7s %10s %9s" % ("method", "pred", "hits", "recall", "precis.", "|dk|", "height err", "count ok"))
for n in METHODS:
    r = t[n]; print("  %-14s %6d %6d %6.0f%% %7.0f%% %7.2f %9.1f%% %8.0f%%" % (n, r["pred_peaks"], r["hits"], 100 * r["recall"], 100 * r["precision"], r["mean_dk"] or 0, r["mean_height_err_pct"] or 0, 100 * r["count_exact_frac"]))
print("  F1 by threshold / tolerance:")
for key in sorted(k for k in res if k.startswith("rel")):
    print("   ", key, "  ".join(f"{n} {res[key][n]['F1']:.2f}" for n in METHODS))
print("  dominant mode (true branch carrying k_c):")
for n, r in dom.items(): print("   %-14s accuracy %5.1f%%   MAE k_c %.2f" % (n, 100 * r["dominant_mode_accuracy"], r["MAE_k_c"]))
print(f"\nTEST 2 -- resonance block withheld ({blk.sum()} E): exchanges of the dominant mode along E")
for n, r in blockres.items():
    print("   %-14s exchanges %2d (true %d, found +-1 step %d)  dominant-mode accuracy %5.1f%%  MAE k_c %.2f" % (n, r["exchanges_pred"], r["exchanges_true"], r["exchanges_found_pm1step"], 100 * r["dominant_mode_accuracy"], r["MAE_k_c"]))

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
npk = {E: len(peaks(store[E]["truth"], 0.02 * Tc[ix[E]])[0]) for E in store}
cand = sorted([E for E in store if npk[E] >= 1])
picks = [cand[int(round(q))] for q in np.linspace(0, len(cand) - 1, min(6, len(cand)))]
col = {"truth": "k", "PCHIP-2D": "tab:green", "Branch-interp": "tab:orange", "Median-29": "tab:blue", "CNP-cond": "tab:red"}
fig, axs = plt.subplots(2, 3, figsize=(15, 8)); axs = axs.ravel()
for ax, E in zip(axs, picks):
    thr = 0.02 * Tc[ix[E]]
    for n in ["truth"] + METHODS:
        c = store[E][n]; ax.plot(kk, c, color=col[n], lw=2.2 if n == "truth" else 1.2, ls="-" if n in ("truth", "CNP-cond") else "--", label=n)
        kp_, hp_ = peaks(c, thr); ax.plot(kp_, hp_, "v" if n == "truth" else "^", color=col[n], ms=9 if n == "truth" else 6)
    ok = ~np.isnan(store[E]["truth"]); ax.set_xlim(kk[ok][0], kk[ok][-1])
    ax.set_title(f"E = {E:.4g} (test): {npk[E]} peak(s)"); ax.set_xlabel("k"); ax.set_ylabel("Ta")
axs[0].legend(fontsize=8); fig.suptitle("Peaks of the marginal curve between modes (markers: detected peaks)"); fig.tight_layout()
fig.savefig(OUT / "fig_modes_curves.png", dpi=140)
fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(Eb, kct, "k.-", lw=1.5, label="Floquet")
mk = {"PCHIP-1D": ("x", "tab:green"), "PCHIP-2D": ("s", "tab:olive"), "Branch-interp": ("D", "tab:orange"), "CNP-block": ("o", "tab:red")}
for n, (m_, c_) in mk.items(): ax[0].plot(Eb, series[n], m_, color=c_, mfc="none", ms=6, label=n)
ax[0].set_ylabel("$k_c$"); ax[0].legend(fontsize=8, ncol=3); ax[0].set_title("Resonance band withheld: dominant mode along E")
ax[1].plot(Eb, Tc[blk], "k.-", lw=1.5); ax[1].plot(Eb, 10 ** PchipInterpolator(lx[trb], np.log10(Tc[trb]))(lx[blk]), "x", color="tab:green")
ax[1].plot(Eb, np.array([np.nanmin(c2db[E]) for E in Eb]), "s", color="tab:olive", mfc="none")
ax[1].plot(Eb, np.array([np.nanmin(branch_interp(E, g_b[E])) for E in Eb]), "D", color="tab:orange", mfc="none")
ax[1].plot(peb.set_index(peb.E.round(6)).loc[[R6(E) for E in Eb], "E"], peb.set_index(peb.E.round(6)).loc[[R6(E) for E in Eb], "Ta_c_pred"], "o", color="tab:red", mfc="none")
ax[1].set_ylabel("$Ta_c$"); ax[1].set_xlabel("E"); fig.tight_layout(); fig.savefig(OUT / "fig_modes_block.png", dpi=140)
print("\nfigures:", OUT / "fig_modes_curves.png", OUT / "fig_modes_block.png")
