"""TaylorCouetteML -- marginal stability curve Ta(k) at a given elasticity.

The prediction is leak-free: the branch structure, the 23 descriptors and the 4
anchors of the requested elasticity are interpolated from its two nearest
TRAINING elasticities, never read on its own marginal curve (see
leakfree/lf_predict.py and the README).

Examples
--------
    python predict.py --E 0.001                        # conditioned CNP (default)
    python predict.py --E 0.001 --model dist           # distilled transformer
    python predict.py --E 0.0043 --model cnp           # elasticity outside the database
    python predict.py --E 0.06 --save curve.png --csv curve.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from leakfree.lf_predict import MODELS, Database, critical, predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--E", type=float, required=True, help="elasticity number, 1e-4 <= E <= 10")
    ap.add_argument("--model", default="cnp_nbr", choices=list(MODELS),
                    help="cnp | cnp_nbr (default) | dist | sarl")
    ap.add_argument("--margin", type=float, default=0.05, help="margin on the interpolated Ta anchors")
    ap.add_argument("--csv", type=Path, help="write the predicted curve as (k, Ta)")
    ap.add_argument("--save", type=Path, help="write the figure")
    ap.add_argument("--no-truth", action="store_true", help="do not draw the Floquet curve even when it exists")
    a = ap.parse_args()

    db = Database()
    k, ta, rows, _ = predict(db, a.E, a.model, margin=a.margin)
    Ta_c, k_c = critical(k, ta)
    split = db.split.get(round(float(a.E), 6), "outside the database")
    print(f"E = {a.E:g}   model = {a.model}   ({split})")
    print(f"  {len(rows)} predicted branch(es), neighbours E = {rows.source_lo.iloc[0]:g} and {rows.source_hi.iloc[0]:g}")
    print(f"  Ta_c = {Ta_c:.3f}   k_c = {k_c:.2f}")
    truth = None if a.no_truth else db.truth(a.E)
    if truth is not None:
        kt, tat = truth
        it = int(np.argmin(tat))
        print(f"  Floquet: Ta_c = {tat[it]:.3f}  k_c = {kt[it]:.2f}"
              f"   |   error {100 * abs(Ta_c / tat[it] - 1):.2f} % on Ta_c, {abs(k_c - kt[it]):.2f} on k_c")

    if a.csv:
        np.savetxt(a.csv, np.c_[k, ta], delimiter=",", header="k,Ta", comments="", fmt="%.6g")
        print("  curve written to", a.csv)
    if a.save:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        if truth is not None:
            ax.plot(truth[0], truth[1], color="k", lw=2.2, label="Floquet")
        ax.plot(k, ta, color="#B7791F", lw=1.6, label=f"prediction ({a.model})")
        ax.plot(k_c, Ta_c, "o", color="#B7791F", mfc="white", mew=1.6, ms=7)
        for _, r in rows.iterrows():
            ax.axvline(r.k_left, color="0.85", lw=0.8, zorder=0)
        ax.axvline(rows.k_right.iloc[-1], color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("k"); ax.set_ylabel("Ta"); ax.set_title(f"E = {a.E:g}")
        ax.legend(); fig.tight_layout(); fig.savefig(a.save, dpi=140)
        print("  figure written to", a.save)


if __name__ == "__main__":
    main()
