"""TaylorCouetteML -- critical curve Ta_c(E), k_c(E) over a range of elasticities.

Every elasticity is predicted leak-free, as in predict.py: its branch structure,
descriptors and anchors are interpolated from the two nearest TRAINING
elasticities.  With --test-only the sweep is restricted to the 64 held-out
elasticities, which reproduces results/predictions/per_E_predictions.csv.

Examples
--------
    python predict_critical.py --n 40                       # 40 elasticities in [1e-4, 10]
    python predict_critical.py --test-only --model dist     # the 64 held-out elasticities
    python predict_critical.py --n 60 --save critical.png --csv critical.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from leakfree.lf_predict import MODELS, Database, critical, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cnp_nbr", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=40, help="number of log-spaced elasticities")
    ap.add_argument("--E-min", type=float, default=1e-4); ap.add_argument("--E-max", type=float, default=10.0)
    ap.add_argument("--test-only", action="store_true", help="sweep the 64 held-out elasticities instead")
    ap.add_argument("--csv", type=Path); ap.add_argument("--save", type=Path)
    a = ap.parse_args()

    db = Database()
    if a.test_only:
        Es = sorted(e for e, s in db.split.items() if s == "test")
    else:
        Es = np.logspace(np.log10(a.E_min), np.log10(a.E_max), a.n)
    rows = []
    for E in Es:
        k, ta, br, _ = predict(db, E, a.model)
        Ta_c, k_c = critical(k, ta)
        truth = db.truth(E)
        r = dict(E=float(E), n_branches=len(br), Ta_c_pred=Ta_c, k_c_pred=k_c)
        if truth is not None:
            kt, tat = truth; it = int(np.argmin(tat))
            r.update(Ta_c_true=float(tat[it]), k_c_true=float(kt[it]),
                     err_Ta_c_pct=100 * abs(Ta_c / tat[it] - 1), err_k_c=abs(k_c - kt[it]))
        rows.append(r)
        print(f"E = {E:10.5g}   Ta_c = {Ta_c:9.3f}   k_c = {k_c:6.2f}"
              + (f"   error {r['err_Ta_c_pct']:5.2f} % / {r['err_k_c']:4.2f}" if truth is not None else ""))
    d = pd.DataFrame(rows)
    if "err_Ta_c_pct" in d:
        print(f"\nmean over {int(d.err_Ta_c_pct.notna().sum())} elasticities of the database: "
              f"{d.err_Ta_c_pct.mean():.2f} % on Ta_c, {d.err_k_c.mean():.2f} on k_c")
    if a.csv:
        d.to_csv(a.csv, index=False); print("values written to", a.csv)
    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        allE = np.array(sorted(db.curves));
        axs[0].plot(allE, [db.curves[e].Ta.min() for e in allE], color="0.6", lw=1.2, label="Floquet")
        axs[1].plot(allE, [db.curves[e].k.values[np.argmin(db.curves[e].Ta.values)] for e in allE], color="0.6", lw=1.2)
        axs[0].plot(d.E, d.Ta_c_pred, "o", color="#B7791F", mfc="none", ms=6, label=f"prediction ({a.model})")
        axs[1].plot(d.E, d.k_c_pred, "o", color="#B7791F", mfc="none", ms=6)
        axs[0].set_ylabel("Ta$_c$"); axs[1].set_ylabel("k$_c$"); axs[1].set_xlabel("E")
        for ax in axs: ax.set_xscale("log"); ax.grid(alpha=0.3)
        axs[0].legend(); fig.tight_layout(); fig.savefig(a.save, dpi=140)
        print("figure written to", a.save)


if __name__ == "__main__":
    main()
