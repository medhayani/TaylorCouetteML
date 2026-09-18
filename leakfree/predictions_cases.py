"""Prediction cases of the leak-free retrained models on unseen test elasticities.

Every model receives, for each branch, the 23 descriptors and the 4 anchors
interpolated from the two neighbouring training elasticities, and log E.
  CNP zero-shot     runs/cnp_leakfree   (5 seeds, mean)
  CNP conditionne   runs/cnp_nbr        (5 seeds, mean; context = neighbour curve)
  DIST              runs/distil_lf      (5 seeds, median; STAR student)
  RL                median of the 29 teachers + SARL correction (5 seeds,
                    weights 1/validation MAE, Tukey blend)
Figures: marginal curves for 9 test elasticities evenly spread in rank,
critical point Ta_c(E), k_c(E) on the 64 test elasticities, zooms on the peaks
between modes.  Per-E errors are computed as in eval_cnp_leakfree.py /
eval_teachers_lf.py (branch points concatenated, minimum, curve error on the
Floquet grid inside the predicted support).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.signal import find_peaks, savgol_filter

SMOOTH_WIN, SMOOTH_ORDER = 9, 3          # light filter inside one branch (101 points)


def smooth_branch(y):
    """Remove the small oscillations of the network inside one branch.
    Applied per branch, so the cusps between modes are untouched."""
    return np.clip(savgol_filter(y, SMOOTH_WIN, SMOOTH_ORDER, mode="interp"), 0.0, 1.0)

REPO = Path(__file__).resolve().parents[1]            # repository root
OUT = REPO / "results/predictions"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code")); sys.path.insert(0, str(REPO))
from models.cnp_tc.model import CNP_TC
from models.star.model import STAR
from data_pipeline.dataset import HydraWindowsDataset
from models.sac_pro.feature_extractor import SARLProFeatureExtractor
from models.sac_pro.sac_pro import SACPro
torch.set_num_threads(12)

FEATS = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry", "slope_left_local",
         "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse", "normalized_arc_length",
         "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch",
         "is_last_branch", "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]
R6 = lambda e: round(float(e), 6)
S = np.linspace(0, 1, 101)
kk = np.round(np.arange(1.0, 50.0001, 0.1), 4)
MODELS = ["CNP zero-shot", "CNP conditionné", "DIST", "RL (médiane + SARL)"]
COL = {"CNP zero-shot": "#5B9BD5", "CNP conditionné": "#1F4E79", "DIST": "#B03A2E", "RL (médiane + SARL)": "#2E7D5B"}
LS = {"CNP zero-shot": (0, (4, 2)), "CNP conditionné": "-", "DIST": "-", "RL (médiane + SARL)": (0, (1, 1.2))}

raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
curves = {R6(E): g.sort_values("k") for E, g in raw.groupby("E")}
split = pd.read_csv(REPO / "data/split_by_E.csv"); test_E = sorted(R6(e) for e in split.loc[split.split == "test", "E"])
desc_lf = pd.read_csv(REPO / "data/branch_functional_descriptors_leakfree_all.csv"); desc_lf["log10E"] = np.log10(desc_lf.E)


def test_rows(desc):
    d = desc[desc.E.map(lambda e: R6(e) in set(test_E))].sort_values(["E", "branch_local_id"]).reset_index(drop=True)
    if "log10E" not in d: d["log10E"] = np.log10(d.E)
    return d


def ctx_matrix(rows):
    return np.array([[float(r[f]) if (f in rows.columns and pd.notna(r[f])) else 0.0 for f in FEATS] for _, r in rows.iterrows()], np.float32)


def anc_matrix(rows):
    return np.stack([rows.k_left, rows.k_right, np.log10(rows.Ta_min.clip(lower=1e-6)), np.log10(rows.Ta_max.clip(lower=1e-6))], 1).astype(np.float32)


# ---------------------------------------------------------------- CNP (zero-shot and conditioned)
def cnp_predict(run):
    targs = json.load(open(run / "leakfree_diag.json"))["args"]
    HP = dict(ctx_dim=23, **{k: targs[k] for k in ["n_cheb", "n_cos", "d_model", "n_enc_layers", "n_dec_layers", "n_heads", "dropout", "n_E_freq", "n_k_freq", "max_ctx_points"]})
    models = []
    for sd in sorted(run.glob("seed_*/best.pt")):
        m = CNP_TC(**HP); m.load_state_dict(torch.load(sd, map_location="cpu")["state_dict"]); m.eval(); models.append(m)
    st = np.load(run / "test_branches.npz")
    desc = pd.read_csv(run / "descriptors_leakfree.csv"); rows = test_rows(desc)
    ctx = np.clip(np.nan_to_num((ctx_matrix(rows) - st["ctx_mean"]) / st["ctx_std"]), -5, 5).astype(np.float32)
    anc = np.clip(np.nan_to_num((anc_matrix(rows) - st["anc_mean"]) / st["anc_std"]), -5, 5).astype(np.float32)
    logE = np.log10(rows.E.values).astype(np.float32); N, P = len(rows), HP["max_ctx_points"]
    ck = np.zeros((N, P), np.float32); cta = np.zeros((N, P), np.float32); cm = np.zeros((N, P), np.float32)
    if (run / "context_curves.npz").exists():
        z = np.load(run / "context_curves.npz"); cc = {(round(float(e), 8), int(b)): t for e, b, t in zip(z["E"], z["branch_local_id"], z["ta"])}
        idx = np.unique(np.round(np.linspace(0, 100, int(targs.get("n_ctx", 32)))).astype(int))
        for i, r in rows.iterrows():
            ck[i, :len(idx)] = (2 * S - 1)[idx]; cta[i, :len(idx)] = cc[(round(float(r.E), 8), int(r.branch_local_id))][idx]; cm[i, :len(idx)] = 1.0
    kq = torch.tensor(np.tile(2 * S - 1, (N, 1)), dtype=torch.float32)
    with torch.no_grad():
        y = np.mean([m(kq, torch.from_numpy(ctx), torch.from_numpy(anc), torch.from_numpy(logE), torch.from_numpy(ck),
                       torch.from_numpy(cta), torch.from_numpy(cm)).numpy() for m in models], 0)
    return rows, y


# ---------------------------------------------------------------- DIST (STAR student)
def dist_predict():
    run = REPO / "models_trained/dist"
    st = np.load(run / "test_branches.npz"); rows = test_rows(desc_lf)
    ctx = np.clip(np.nan_to_num((ctx_matrix(rows) - st["ctx_mean"]) / (st["ctx_std"] + 1e-6)), -5, 5).astype(np.float32)
    anc = np.clip(np.nan_to_num((anc_matrix(rows) - st["anc_mean"]) / st["anc_std"]), -5, 5).astype(np.float32)
    lt = torch.tensor(np.log10(rows.E.values), dtype=torch.float32); ys = []
    for sd in sorted(run.glob("seed_*/best.pt")):
        m = STAR(ctx_dim=23, n_cheb=64, n_cos=32, d_model=384, n_layers=8, n_heads=12, dropout=0.05, n_E_freq=32, n_k_freq=16, use_spectral_norm=True)
        m.load_state_dict(torch.load(sd, map_location="cpu", weights_only=False)["state_dict"]); m.eval()
        with torch.no_grad():
            ys.append(np.concatenate([m(torch.tensor(np.tile(2 * S - 1, (1, 1)), dtype=torch.float32), torch.from_numpy(ctx[i:i + 1]),
                                        torch.from_numpy(anc[i:i + 1]), lt[i:i + 1]).numpy() for i in range(len(rows))], 0))
    return rows, np.median(np.stack(ys), 0)


# ---------------------------------------------------------------- RL: median of the teachers + SARL
def rl_predict(taper=0.20):
    ds = HydraWindowsDataset(REPO / "data/rl_windows/rl_switch_windows_lf_test.npz")
    obs, sv = torch.from_numpy(ds.obs_seq), torch.from_numpy(ds.static_vec); corrs, vmae = [], []
    for ck in sorted((REPO / "models_trained/sarl").glob("seed_*/best.pt")):
        st = torch.load(ck, map_location="cpu", weights_only=False); cfg = st["cfg"]
        ext = SARLProFeatureExtractor(obs.shape[2], obs.shape[1], st["static_dim"], cfg["feature_extractor"])
        sac = SACPro(feature_dim=ext.out_dim, action_dim=st["action_dim"], actor_layers=cfg["actor_layers"], critic_layers=cfg["critic_layers"])
        ext.load_state_dict(st["extractor"]); sac.load_state_dict(st["sac"]); ext.eval(); sac.eval()
        with torch.no_grad():
            corrs.append(torch.tanh(sac.actor.mean(sac.actor.body(ext(obs, sv)))).numpy())
        vmae.append(min(r["val_mae"] for r in json.load(open(ck.parent / "history.json")) if r.get("val_mae") is not None))
    w = 1.0 / (np.array(vmae) + 1e-6); w /= w.sum(); corr = np.einsum("i...,i->...", np.stack(corrs), w)
    med = np.load(REPO / "data/ensemble_median_targets_lf.npz")
    mkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(med["E"], med["branch_local_id"]))}
    wkey = {(R6(e), int(b)): i for i, (e, b) in enumerate(zip(ds.E, ds.branch_local_id))}
    rows = test_rows(desc_lf); y = np.zeros((len(rows), 101), np.float32)
    for i, r in rows.iterrows():
        key = (R6(r.E), int(r.branch_local_id)); base = med["ta_norm_median"][mkey[key]].copy()
        if key in wkey:
            j = wkey[key]; c0 = float(ds.center_pred[j]); h = float(ds.window_half_width[j])
            cf = np.interp(S, np.clip(c0 + h * ds.local_grid[j], 0, 1), corr[j], left=0.0, right=0.0)
            d = np.abs(S - c0) / max(h, 1e-9)
            wt = np.where(d <= 1 - taper, 1.0, np.where(d >= 1, 0.0, 0.5 * (1 + np.cos(np.pi * (d - (1 - taper)) / taper))))
            base = np.clip(base + wt * cf, 0.0, 1.0)
        y[i] = base
    return rows, y


# ---------------------------------------------------------------- reconstruction and errors
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


pred = {"CNP zero-shot": cnp_predict(REPO / "models_trained/cnp"),
        "CNP conditionné": cnp_predict(REPO / "models_trained/cnp_nbr"),
        "DIST": dist_predict(), "RL (médiane + SARL)": rl_predict()}
curve_on_grid, branches, per_E = {n: {} for n in MODELS}, {n: {} for n in MODELS}, []
for n in MODELS:
    rows, y = pred[n]
    for E, g in rows.groupby("E"):
        E = R6(E)
        seg = [(r.k_left + S * (r.k_right - r.k_left), r.Ta_min + y[i] * (r.Ta_max - r.Ta_min),
                r.Ta_min + smooth_branch(y[i]) * (r.Ta_max - r.Ta_min)) for i, r in g.iterrows()]
        branches[n][E] = seg                                  # one entry per predicted mode
        pieces = [(k, ta) for k, ta, _ in seg]
        curve_on_grid[n][E] = assemble(pieces)
        gt = curves[E]; kt, Tt = gt.k.values, gt.Ta.values
        row = dict(model=n, E=E, n_branches_pred=len(g), Ta_c_true=Tt.min(), k_c_true=kt[np.argmin(Tt)])
        for tag, col in [("", 1), ("_smooth", 2)]:
            kp = np.concatenate([s[0] for s in seg]); Tp = np.concatenate([s[col] for s in seg])
            o = np.argsort(kp); kp, Tp = kp[o], Tp[o]
            Ti = np.interp(kt, kp, Tp, left=np.nan, right=np.nan); v = ~np.isnan(Ti)
            row.update({f"Ta_c_pred{tag}": Tp.min(), f"k_c_pred{tag}": kp[np.argmin(Tp)],
                        f"err_Ta_c_pct{tag}": 100 * abs(Tp.min() / Tt.min() - 1),
                        f"err_k_c{tag}": abs(kp[np.argmin(Tp)] - kt[np.argmin(Tt)]),
                        f"err_curve_pct{tag}": 100 * np.mean(np.abs(Ti[v] / Tt[v] - 1))})
        per_E.append(row)
per_E = pd.DataFrame(per_E); per_E.to_csv(OUT / "per_E_predictions.csv", index=False)
summary = per_E.groupby("model")[["err_Ta_c_pct", "err_k_c", "err_curve_pct",
                                 "err_Ta_c_pct_smooth", "err_k_c_smooth", "err_curve_pct_smooth"]].mean().loc[MODELS]
print(summary.round(3).to_string())
print("(les colonnes _smooth : memes predictions, filtrees a l'interieur de chaque branche)")
np.savez(OUT / "predicted_curves.npz", k=kk, E=np.array(test_E), **{n.split(" ")[0] + ("_cond" if "cond" in n else "") : np.stack([curve_on_grid[n][E] for E in test_E]) for n in MODELS})

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
fr = lambda x, p=1: f"{x:.{p}f}".replace(".", ",")
nb_true = pd.read_csv(REPO / "data/branch_functional_descriptors.csv").groupby("E").size()
nb_true = {R6(e): int(c) for e, c in nb_true.items()}


def truth_on_grid(E):
    g = curves[E]; return np.interp(kk, g.k.values, g.Ta.values, left=np.nan, right=np.nan)


def true_peaks(E):
    t = truth_on_grid(E); ok = ~np.isnan(t); i0, i1 = np.where(ok)[0][[0, -1]]
    p, _ = find_peaks(t[i0:i1 + 1], prominence=0.02 * np.nanmin(t)); return kk[i0:i1 + 1][p], t[i0:i1 + 1][p]


def panel(ax, E, zoom=None, legend=False, errors=True):
    t = truth_on_grid(E); ok = ~np.isnan(t)
    ax.plot(kk[ok], t[ok], color="k", lw=2.4, label="Floquet (vérité)", zorder=5)
    kp_, hp_ = true_peaks(E); ax.plot(kp_, hp_, "v", color="k", ms=7, zorder=6, label="pics entre modes (Floquet)")
    ax.plot(kk[np.nanargmin(t)], np.nanmin(t), "*", color="k", ms=13, zorder=7)
    for n in MODELS:
        seg = branches[n][E]
        for j, (kb, _, tb) in enumerate(seg):                 # one line per predicted mode
            ax.plot(kb, tb, color=COL[n], ls=LS[n], lw=1.6, zorder=8,
                    label=n if j == 0 else None)
        kall = np.concatenate([s[0] for s in seg]); tall = np.concatenate([s[2] for s in seg])
        i = int(np.argmin(tall)); ax.plot(kall[i], tall[i], "o", color=COL[n], ms=5, mfc="white", mew=1.5, zorder=9)
    for kb, _, _ in branches[MODELS[1]][E][1:]:               # junctions between predicted modes
        ax.axvline(kb[0], color="0.8", lw=0.8, ls=":", zorder=1)
    if zoom:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    else:
        ax.set_xlim(kk[ok][0], kk[ok][-1])
    nbp = int((per_E[(per_E.model == "DIST") & (per_E.E == E)].n_branches_pred).iloc[0])
    ax.set_title(f"E = {E:g}   ({nb_true.get(E, '?')} branche(s), {len(kp_)} pic(s))", fontsize=10)
    ax.set_xlabel("k"); ax.set_ylabel("Ta")
    if errors:
        txt = "erreur sur Ta$_c$ / courbe :\n" + "\n".join(
            f"{n.split(' (')[0]} : {fr(per_E[(per_E.model == n) & (per_E.E == E)].err_Ta_c_pct_smooth.iloc[0])} % / {fr(per_E[(per_E.model == n) & (per_E.E == E)].err_curve_pct_smooth.iloc[0])} %" for n in MODELS)
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9), zorder=10)
    if legend: ax.legend(fontsize=7, loc="upper left")


pdf = PdfPages(OUT / "predictions_modeles_sans_fuite.pdf")
# A: nine test elasticities evenly spread in rank (no selection)
picks = [test_E[int(round(q))] for q in np.linspace(0, len(test_E) - 1, 9)]
fig, axs = plt.subplots(3, 3, figsize=(16, 13)); axs = axs.ravel()
for i, (ax, E) in enumerate(zip(axs, picks)): panel(ax, E, legend=(i == 0))
fig.suptitle("Courbes marginales prédites sur 9 élasticités de test jamais vues (réparties régulièrement, sans sélection)\n"
             "Modèles réentraînés sans fuite ; chaque mode est tracé comme une branche à part, filtrée à l'intérieur de la branche.\n"
             "★ : (k$_c$, Ta$_c$) Floquet ;  ○ : minimum prédit ;  pointillés verticaux : jonctions entre modes prédits", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "fig_cas_courbes.png", dpi=130); pdf.savefig(fig); plt.close(fig)

# B: zooms on the peaks between modes (test elasticities with at least two peaks)
multi = [E for E in test_E if len(true_peaks(E)[0]) >= 2]
zpicks = [multi[int(round(q))] for q in np.linspace(0, len(multi) - 1, min(6, len(multi)))]
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5)); axs = axs.ravel()
for i, (ax, E) in enumerate(zip(axs, zpicks)):
    t = truth_on_grid(E); kp_, hp_ = true_peaks(E); tc = np.nanmin(t)
    x0, x1 = max(1.0, min(kp_.min(), kk[np.nanargmin(t)]) - 2.0), min(kk[~np.isnan(t)][-1], max(kp_.max(), kk[np.nanargmin(t)]) + 2.0)
    panel(ax, E, zoom=((x0, x1), (0.97 * tc, max(hp_.max(), tc) * 1.06 + 0.02 * tc)), legend=(i == 0))
for ax in axs[len(zpicks):]: ax.axis("off")
fig.suptitle(f"Zoom sur les pics entre modes : {len(zpicks)} élasticités de test à deux pics ou plus ({len(multi)} au total), réparties régulièrement", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / "fig_zoom_pics.png", dpi=130); pdf.savefig(fig); plt.close(fig)

# C: critical point along E
allE = np.array(sorted(curves)); tc_all = np.array([curves[E].Ta.min() for E in allE]); kc_all = np.array([curves[E].k.values[np.argmin(curves[E].Ta.values)] for E in allE])
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
axs[0].plot(allE, tc_all, color="0.6", lw=1.2, label="Floquet, 420 élasticités"); axs[1].plot(allE, kc_all, color="0.6", lw=1.2)
te = np.array(test_E)
axs[0].plot(te, [curves[E].Ta.min() for E in te], "o", color="k", ms=4, label="Floquet, 64 élasticités de test")
axs[1].plot(te, [curves[E].k.values[np.argmin(curves[E].Ta.values)] for E in te], "o", color="k", ms=4)
mk = {"CNP zero-shot": "^", "CNP conditionné": "s", "DIST": "D", "RL (médiane + SARL)": "v"}
for n in MODELS:
    d = per_E[per_E.model == n].sort_values("E")
    axs[0].plot(d.E, d.Ta_c_pred, mk[n], color=COL[n], mfc="none", ms=6, label=n)
    axs[1].plot(d.E, d.k_c_pred, mk[n], color=COL[n], mfc="none", ms=6)
    axs[2].plot(d.E, d.err_Ta_c_pct, mk[n] + "-", color=COL[n], mfc="none", ms=5, lw=0.7, label=f"{n} (moyenne {fr(d.err_Ta_c_pct.mean(), 2)} %)")
axs[0].set_ylabel("Ta$_c$"); axs[1].set_ylabel("k$_c$"); axs[2].set_ylabel("erreur sur Ta$_c$ (%)"); axs[2].set_xlabel("E")
for ax in axs: ax.set_xscale("log"); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8); axs[2].legend(fontsize=8)
axs[0].set_title("Point critique prédit sur les 64 élasticités de test")
fig.tight_layout(); fig.savefig(OUT / "fig_seuil_critique.png", dpi=130); pdf.savefig(fig); plt.close(fig)
pdf.close()
print("picks A:", picks); print("picks B:", zpicks)
print("written to", OUT)
