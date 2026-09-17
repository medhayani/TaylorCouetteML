"""Descriptor analysis for the leak-free CNP models.

1. Information content: which of the 23 descriptors vary, which are constant
   in the tables actually read by the curve models.
2. Redundancy: Spearman correlation among the informative descriptors
   (training branches).
3. Importance for prediction: permutation importance on the 64 test
   elasticities.  One input (or one group of inputs) is shuffled across the
   test branches; the degradation of the predicted marginal curve, of Ta_c and
   of k_c is recorded.  Anchors are shuffled as NETWORK INPUTS only; the
   denormalisation keeps the branch's own interpolated anchors.
Models: leak-free zero-shot CNP (runs/cnp_leakfree) and neighbour-conditioned
CNP (runs/cnp_nbr).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[1]            # repository root
OUT = REPO / "results/features"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code")); sys.path.insert(0, str(REPO))
from models.cnp_tc.model import CNP_TC
torch.set_num_threads(12)

FEATS = ["log10E", "branch_order_norm", "width_k", "width_asymmetry", "rise_asymmetry", "slope_left_local",
         "slope_right_local", "global_slope", "curvature_at_min", "roughness_rmse", "normalized_arc_length",
         "has_switch_left", "has_switch_right", "mean_abs_curvature", "amplitude", "n_branches", "is_first_branch",
         "is_last_branch", "left_width", "right_width", "left_rise", "right_rise", "mean_abs_slope"]
GROUPS = {
    "elasticite (log E)":        ["log10E", "__logE__"],
    "etendue et position":       ["width_k", "left_width", "right_width", "width_asymmetry"],
    "hauteurs":                  ["amplitude", "left_rise", "right_rise", "rise_asymmetry"],
    "pentes":                    ["slope_left_local", "slope_right_local", "global_slope", "mean_abs_slope"],
    "courbure":                  ["curvature_at_min", "mean_abs_curvature"],
    "complexite de forme":       ["roughness_rmse", "normalized_arc_length"],
    "structure des modes (6)":   ["branch_order_norm", "n_branches", "is_first_branch", "is_last_branch", "has_switch_left", "has_switch_right"],
    "ancrages (k_L,k_R,Ta_min,Ta_max)": ["__a0__", "__a1__", "__a2__", "__a3__"],
}
R6 = lambda e: round(float(e), 6)
S = np.linspace(0, 1, 101)

# ---------------------------------------------------------------- truth
raw = pd.read_csv(REPO / "data/combined_data.csv"); raw.columns = ["Ta", "k", "E"]
curves = {R6(E): g.sort_values("k") for E, g in raw.groupby("E")}
split = pd.read_csv(REPO / "data/split_by_E.csv"); lab = {R6(e): s for e, s in zip(split.E, split.split)}

# ---------------------------------------------------------------- 1. information content
desc_true = pd.read_csv(REPO / "data/branch_functional_descriptors.csv")
info = {}
for f in FEATS:
    if f == "log10E": info[f] = "varie (calcule par les scripts)"; continue
    if f not in desc_true.columns: info[f] = "absent du fichier : 0 pour les modeles"; continue
    v = desc_true[f]
    info[f] = "entierement NaN : 0 pour les modeles" if v.isna().all() else ("constant" if v.nunique() <= 1 else f"varie ({v.nunique()} valeurs)")
informative = [f for f in FEATS if info[f].startswith("varie")]

# ---------------------------------------------------------------- 2. redundancy (training branches)
tr_mask = desc_true.E.map(lambda e: lab.get(R6(e)) == "train")
dtr = desc_true[tr_mask].copy(); dtr["log10E"] = np.log10(dtr.E)
corr = dtr[informative].corr(method="spearman")
pairs = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(informative) for b in informative[i + 1:] if abs(corr.loc[a, b]) >= 0.9]
pairs.sort(key=lambda t: -abs(t[2]))


# ---------------------------------------------------------------- 3. permutation importance
def load_run(run):
    targs = json.load(open(run / "leakfree_diag.json"))["args"]
    HP = dict(ctx_dim=23, **{k: targs[k] for k in ["n_cheb", "n_cos", "d_model", "n_enc_layers", "n_dec_layers", "n_heads", "dropout", "n_E_freq", "n_k_freq", "max_ctx_points"]})
    models = []
    for sd in sorted(run.glob("seed_*/best.pt")):
        m = CNP_TC(**HP); m.load_state_dict(torch.load(sd, map_location="cpu")["state_dict"]); m.eval(); models.append(m)
    st = np.load(run / "test_branches.npz")
    desc = pd.read_csv(run / "descriptors_leakfree.csv"); desc["log10E"] = np.log10(desc.E.clip(lower=1e-30))
    rows = desc[desc.E.map(lambda e: lab.get(R6(e)) == "test")].sort_values(["E", "branch_local_id"]).reset_index(drop=True)
    ctx = np.array([[float(r[f]) if (f in rows.columns and pd.notna(r[f])) else 0.0 for f in FEATS] for _, r in rows.iterrows()], np.float32)
    ctx = np.clip(np.nan_to_num((ctx - st["ctx_mean"]) / st["ctx_std"]), -5, 5).astype(np.float32)
    anc = np.stack([rows.k_left, rows.k_right, np.log10(rows.Ta_min.clip(lower=1e-6)), np.log10(rows.Ta_max.clip(lower=1e-6))], 1).astype(np.float32)
    anc = np.clip(np.nan_to_num((anc - st["anc_mean"]) / st["anc_std"]), -5, 5).astype(np.float32)
    logE = np.log10(rows.E.values).astype(np.float32)
    P = HP["max_ctx_points"]; N = len(rows)
    ck = np.zeros((N, P), np.float32); cta = np.zeros((N, P), np.float32); cm = np.zeros((N, P), np.float32)
    if (run / "context_curves.npz").exists():
        z = np.load(run / "context_curves.npz"); cc = {(round(float(e), 8), int(b)): t for e, b, t in zip(z["E"], z["branch_local_id"], z["ta"])}
        idx = np.unique(np.round(np.linspace(0, 100, int(targs.get("n_ctx", 32)))).astype(int))
        for i, r in rows.iterrows():
            ck[i, :len(idx)] = (2 * S - 1)[idx]; cta[i, :len(idx)] = cc[(round(float(r.E), 8), int(r.branch_local_id))][idx]; cm[i, :len(idx)] = 1.0
    return dict(models=models, rows=rows, ctx=ctx, anc=anc, logE=logE, ck=ck, cta=cta, cm=cm, conditioned=bool(cm.any()))


def predict(R, ctx, anc, logE, cta):
    N = len(R["rows"]); kq = torch.tensor(np.tile(2 * S - 1, (N, 1)), dtype=torch.float32)
    with torch.no_grad():
        ys = [m(kq, torch.from_numpy(ctx), torch.from_numpy(anc), torch.from_numpy(logE),
                torch.from_numpy(R["ck"]), torch.from_numpy(cta), torch.from_numpy(R["cm"])).numpy() for m in R["models"]]
    return np.mean(ys, 0)


def metrics(R, y):
    rows = R["rows"]; tc, kc, cv = [], [], []
    for E, g in rows.groupby("E"):
        kp, Tp = [], []
        for i in g.index:
            r = rows.loc[i]; kp.append(r.k_left + S * (r.k_right - r.k_left)); Tp.append(r.Ta_min + y[i] * (r.Ta_max - r.Ta_min))
        kp, Tp = np.concatenate(kp), np.concatenate(Tp); o = np.argsort(kp); kp, Tp = kp[o], Tp[o]
        gt = curves[R6(E)]; kt, Tt = gt.k.values, gt.Ta.values
        Ti = np.interp(kt, kp, Tp, left=np.nan, right=np.nan); v = ~np.isnan(Ti)
        tc.append(abs(Tp.min() / Tt.min() - 1) * 100); kc.append(abs(kp[np.argmin(Tp)] - kt[np.argmin(Tt)])); cv.append(100 * np.mean(np.abs(Ti[v] / Tt[v] - 1)))
    return np.array([np.mean(tc), np.mean(kc), np.mean(cv)])


def importance(R, cols, reps, seed0):
    base = metrics(R, predict(R, R["ctx"], R["anc"], R["logE"], R["cta"])) if "base" not in R else R["base"]
    R["base"] = base
    ci = [FEATS.index(c) for c in cols if c in FEATS]; ai = [int(c[3]) for c in cols if c.startswith("__a")]
    if ci and all(np.allclose(R["ctx"][:, j], R["ctx"][0, j]) for j in ci) and not ai and "__logE__" not in cols and "__ctx__" not in cols:
        return np.zeros(3), np.zeros(3), True
    out = []
    N = len(R["rows"])
    for rep in range(reps):
        perm = np.random.default_rng(seed0 + rep).permutation(N)
        ctx, anc, logE, cta = R["ctx"].copy(), R["anc"].copy(), R["logE"].copy(), R["cta"].copy()
        for j in ci: ctx[:, j] = R["ctx"][perm, j]
        for j in ai: anc[:, j] = R["anc"][perm, j]
        if "__logE__" in cols: logE = R["logE"][perm]
        if "__ctx__" in cols: cta = R["cta"][perm]
        out.append(metrics(R, predict(R, ctx, anc, logE, cta)) - base)
    out = np.array(out)
    return out.mean(0), out.std(0), False


res = {"information": info, "informative": informative, "redundant_pairs_abs_rho_ge_0.9": pairs, "models": {}}
FIG_ONLY = "--figures-only" in sys.argv   # redraw the figures from feature_importance.json
if FIG_ONLY:
    res["models"] = json.load(open(OUT / "feature_importance.json"))["models"]
else:
    for name, run in [("CNP zero-shot", REPO / "models_trained/cnp"), ("CNP conditionne", REPO / "models_trained/cnp_nbr")]:
        R = load_run(run); print(f"\n=== {name}: {len(R['models'])} seeds, {len(R['rows'])} test branches, conditioned={R['conditioned']}", flush=True)
        base = metrics(R, predict(R, R["ctx"], R["anc"], R["logE"], R["cta"])); R["base"] = base
        print("  base: Ta_c %.2f %%  k_c %.2f  curve %.2f %%" % tuple(base), flush=True)
        M = {"base": base.tolist(), "groups": {}, "single": {}}
        groups = dict(GROUPS)
        if R["conditioned"]: groups["contexte (courbe des voisins)"] = ["__ctx__"]
        # all shape descriptors at once: redundant descriptors cannot stand in for each other
        groups["tous les descripteurs geometriques (16)"] = [f for f in informative if f != "log10E"]
        for gi, (g, cols) in enumerate(groups.items()):
            m, s, const = importance(R, cols, 5, 100 + 10 * gi)
            M["groups"][g] = dict(dTa=float(m[0]), dk=float(m[1]), dcurve=float(m[2]), sd_curve=float(s[2]), constant=const)
            print("  group %-34s dTa %+6.2f  dk %+5.2f  dcurve %+6.2f %s" % (g, m[0], m[1], m[2], "(constant: no effect)" if const else ""), flush=True)
        singles = informative + ["__a0__", "__a1__", "__a2__", "__a3__"]
        for si, f in enumerate(singles):
            cols = [f, "__logE__"] if f == "log10E" else [f]
            m, s, const = importance(R, cols, 3, 1000 + 10 * si)
            M["single"][f] = dict(dTa=float(m[0]), dk=float(m[1]), dcurve=float(m[2]), sd_curve=float(s[2]))
        res["models"][name] = M
        json.dump(res, open(OUT / "feature_importance.json", "w"), indent=1)

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
AN = {"__a0__": "ancrage $k_L$", "__a1__": "ancrage $k_R$", "__a2__": r"ancrage $\log Ta_{\min}$", "__a3__": r"ancrage $\log Ta_{\max}$"}
GL = {"elasticite (log E)": "élasticité ($\log E$)", "ancrages (k_L,k_R,Ta_min,Ta_max)": "ancrages ($k_L,k_R,Ta_{\min},Ta_{\max}$)",
      "contexte (courbe des voisins)": "contexte (courbe des voisins)",
      "tous les descripteurs geometriques (16)": "les 16 descripteurs de forme ensemble",
      "etendue et position": "  étendue et position (4)", "hauteurs": "  hauteurs (4)", "pentes": "  pentes (4)",
      "courbure": "  courbure (2)", "complexite de forme": "  complexité de forme (2)",
      "structure des modes (6)": "organisation des modes (6)"}
ORDER = ["elasticite (log E)", "ancrages (k_L,k_R,Ta_min,Ta_max)", "contexte (courbe des voisins)",
         "tous les descripteurs geometriques (16)", "etendue et position", "hauteurs", "pentes", "courbure",
         "complexite de forme", "structure des modes (6)"]
TITLE = {"CNP zero-shot": "CNP zero-shot", "CNP conditionne": "CNP conditionné"}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, (name, M) in zip(axs, res["models"].items()):
    gs = [g for g in ORDER if g in M["groups"]]
    vals = [M["groups"][g]["dcurve"] for g in gs]; err = [M["groups"][g]["sd_curve"] for g in gs]
    y = np.arange(len(gs))
    ax.barh(y, vals, xerr=err, color=["#9E9E9E" if M["groups"][g]["constant"] else ("#B7791F" if g.startswith("anc") or g.startswith("cont") else "#1F4E79") for g in gs])
    xmax = max(vals) * 1.32
    for yi, g, v in zip(y, gs, vals):
        ax.text(max(v, 0) + xmax * 0.012 + M["groups"][g]["sd_curve"], yi, "sans information" if M["groups"][g]["constant"] else f"{v:+.2f}".replace(".", ","), va="center", fontsize=8)
    ax.set_xlim(min(0, min(vals)) - xmax * 0.02, xmax)
    ax.set_yticks(y); ax.set_yticklabels([GL[g] for g in gs], fontsize=9); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{TITLE[name]} (erreur de base sur la courbe : {M['base'][2]:.2f} %)".replace(".", ","), fontsize=10)
    ax.set_xlabel("hausse de l'erreur sur la courbe marginale (points de %)")
fig.suptitle("Importance des groupes d'entrées par permutation, 64 élasticités de test"); fig.tight_layout()
fig.savefig(OUT / "fig_importance_groupes.png", dpi=150)
fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, (name, M) in zip(axs, res["models"].items()):
    items = sorted(M["single"].items(), key=lambda t: -t[1]["dcurve"])
    lbl = [AN.get(k, k) for k, _ in items]; vals = [v["dcurve"] for _, v in items]; err = [v["sd_curve"] for _, v in items]
    y = np.arange(len(items)); ax.barh(y, vals, xerr=err, color=["#B7791F" if k.startswith("__a") else "#1F4E79" for k, _ in items])
    ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=8); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{TITLE[name]} (erreur de base sur la courbe : {M['base'][2]:.2f} %)".replace(".", ","), fontsize=10)
    ax.set_xlabel("hausse de l'erreur sur la courbe (points de %)")
fig.suptitle("Importance de chaque entrée informative (les 6 descripteurs sans information sont omis)"); fig.tight_layout()
fig.savefig(OUT / "fig_importance_descripteurs.png", dpi=150)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(informative))); ax.set_xticklabels(informative, rotation=90, fontsize=7)
ax.set_yticks(range(len(informative))); ax.set_yticklabels(informative, fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman")
ax.set_title("Corrélation de Spearman entre les 17 descripteurs informatifs\n(branches d'entraînement)"); fig.tight_layout()
fig.savefig(OUT / "fig_correlation_informatifs.png", dpi=150)
print("\nredundant pairs |rho| >= 0.9:"); [print("  %-22s %-22s %+.2f" % p) for p in pairs]
print("figures written to", OUT)
