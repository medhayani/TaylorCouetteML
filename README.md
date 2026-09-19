# TaylorCouetteML

**Author** Mohamed Hayani Choujaa
&middot; Laboratoire de Mecanique, Faculte des Sciences Ain Chock, Universite Hassan II de Casablanca, Maroc

Three neural-network surrogates for the linear-stability marginal
curve $`T_a(k,E)`$ of an oscillatory Taylor-Couette flow in an
upper-convected Maxwell (UCM) fluid with co-oscillating cylinders
(the two cylinders oscillate in phase about a zero mean).

> **State of this repository (September 2026).** Everything published
> here follows the **leak-free protocol** described below: predicting
> at an elasticity never uses the marginal curve of that elasticity.
> The weights, figures and numbers shipped here are those of the
> retrained models. The pipeline and the weights of the first
> submission remain in the history of the repository (commit
> `3e8cdd2` and earlier). This README reports what the code does and
> what was measured, nothing else.

---

## The Floquet data

**Wall condition.** Both cylinders oscillate in phase about a zero mean,

```math
\Omega_1(t)=\Omega_2(t)=\Omega_0\cos(\omega t),
```

the co-oscillating cell of Hayani Choujaa et al., Phys. Fluids 33,
074105 (2021) and J. Non-Newtonian Fluid Mech. 325, 105202 (2024).

**Parameters.** $`\gamma=\dfrac{\omega d^{2}}{\nu}=5`$,
$`\varepsilon=\dfrac{d}{R_1}=0.14`$, UCM fluid without solvent, 420 elasticities
in $`E\in[10^{-4},10]`$.

**Data.** `data/combined_data.csv` is the concatenation, in absolute
value, of the raw solver output, one file per elasticity
(`E_<value>.csv`, columns $`\mathrm{Ta}`$ and $`k`$), computed by the
spectral-Floquet solver of the references above. This file has not
changed since the first commit of the repository: same git blob in
every commit.

---

## The leak-free protocol

A surrogate is only useful if it predicts an elasticity it has never
seen, from quantities available *before* the Floquet computation. The
rule applied everywhere in this repository is therefore:

> nothing of the marginal curve of the predicted elasticity enters the
> inputs, the normalisation, or the choice of a checkpoint.

**What a branch is.** A branch lives on $`[k_L,k_R]`$ and is described in a
canonical coordinate, on 101 points:

```math
s=\frac{k-k_L}{k_R-k_L}\in[0,1],\qquad
y(s)=\frac{\mathrm{Ta}\bigl(k(s)\bigr)-\mathrm{Ta}_{\min}}{\mathrm{Ta}_{\max}-\mathrm{Ta}_{\min}}\in[0,1].
```

The networks learn $`y(s)`$ and nothing else. Four anchors,
$`\mathbf{a}=(k_L,k_R,\log_{10}\mathrm{Ta}_{\min},\log_{10}\mathrm{Ta}_{\max})`$,
put the branch back in the physical plane:

```math
\mathrm{Ta}(k)=\mathrm{Ta}_{\min}+\hat y(s)\,(\mathrm{Ta}_{\max}-\mathrm{Ta}_{\min}),
\qquad k=k_L+s\,(k_R-k_L).
```

**How the inputs are built** (`leakfree/leakfree_descriptors.py`).
For an elasticity $`E^{*}`$:

1. its two nearest **training** elasticities are taken,
   $`E_-<E^{*}<E_+`$, with $`E^{*}`$ itself excluded when it belongs to the
   training set, and the weight

   ```math
w=\frac{\log_{10}E^{*}-\log_{10}E_-}{\log_{10}E_+-\log_{10}E_-};
```

2. if both neighbours have the same number of branches, branches are
   matched by rank along $`k`$ and every quantity is interpolated,
   $`q^{*}=(1-w)\,q_-+w\,q_+`$, with $`\mathrm{Ta}_{\min}`$ and
   $`\mathrm{Ta}_{\max}`$ in logarithm; otherwise the whole structure of the
   nearer neighbour is used;
3. the interpolated anchors are widened by 5 %,
   $`\mathrm{Ta}_{\min}\leftarrow0.95\,\mathrm{Ta}_{\min}`$ and
   $`\mathrm{Ta}_{\max}\leftarrow1.05\,\mathrm{Ta}_{\max}`$, so that the true
   curve stays inside $`[0,1]`$, the range the sigmoid head can reach;
4. for the conditioned CNP, the curves of the two neighbours are
   interpolated on the predicted support,
   $`\log\mathrm{Ta}^{\text{ctx}}(k)=(1-w)\log\mathrm{Ta}_-(k)+w\log\mathrm{Ta}_+(k)`$,
   and 32 points of it are supplied as context.

**What this changed with respect to the first submission.** Five
points were found in the code of the submitted version and corrected
here:

| # | Point | Correction |
|---|-------|------------|
| 1 | descriptors and anchors were read on the curve of the predicted elasticity; on the dominant branch $`\mathrm{Ta}_{\min}`$ *is* the threshold being predicted | interpolated from the neighbours (above) |
| 2 | the CNP script drew its own split, by branch and with a different seed per model, so every branch trained at least one of the five averaged models | one split by elasticity, 292 / 64 / 64, shared by every model and every seed (`data/split_by_E.csv`) |
| 3 | six of the 23 scalars of the RL window were computed on the true curve (true centre of the exchange, its presence, absolute and relative distance to the predicted centre, mean and rms error of the base) | neutralised in every split |
| 4 | the centre of the RL window came in part from a detector trained on the truth | curvature peak of the base prediction |
| 5 | the target of the refiners was clamped to $`[-2,2]`$ while the action is bounded to $`[-1,1]`$ | clamped to $`[-1,1]`$ |

**Cost of the correction.** With the leak, the published CNP reached
1.91 % on Ta_c; without it, the same architecture reaches 1.53 %
zero-shot and 1.29 % conditioned on its neighbours -- but the error is
now an honest out-of-sample error.

---

## What is in this repository

```
data/                      Floquet database and the tables the models read
  combined_data.csv                      420 elasticities, 720 branches
  branch_functional_descriptors.csv      true branch structure (k supports, anchors, descriptors)
  branch_functional_descriptors_leakfree_all.csv    the same, interpolated from the neighbours
  combined_data_trainval.csv             curves of the training and validation elasticities only
  branch_functional_descriptors_leakfree_trainval.csv
  split_by_E.csv                         292 / 64 / 64 split by elasticity
  split_block.csv                        variant: resonance band E in [0.08, 0.16] withheld
  ensemble_median_targets_lf.npz         median of the 29 leak-free teachers, per branch
  rl_windows/rl_switch_windows_lf_*.npz  corrected RL windows (49 points, 9 channels)
models_trained/            leak-free weights, 5 seeds each, with their normalisation statistics
  cnp/  cnp_nbr/  cnp_block/  dist/  sarl/
  marl/                                  three seeds, weights in half precision: a float32
                                         checkpoint weighs 111 MB, above the 100 MB limit of
                                         GitHub. The evaluation casts them back and returns
                                         the same numbers to the seventh decimal.
code/                      architectures (models/) and window dataset (data_pipeline/)
train/                     training scripts: teachers, CNP, DIST, SARL and MARL
leakfree/                  leak-free inputs, prediction, evaluation and analysis
kaggle_notebooks/          one notebook per training, ready to push to Kaggle
results/                   every metric quoted below, as produced by the scripts
figures/                   the figures produced from these results
floquet_check/             MATLAB check of the wall condition
predict.py                 marginal curve at one elasticity
predict_critical.py        Ta_c(E), k_c(E) over a range of elasticities
```

The 29 teachers (12 precision, 5 CSON, 5 CNP, 7 STAR) are not shipped,
only the median of their predictions
(`data/ensemble_median_targets_lf.npz`), which is what DIST and the RL
refiner consume. The notebooks retrain them from scratch.

---

## Quickstart

```bash
pip install torch numpy pandas scipy matplotlib
python predict.py --E 0.001                       # conditioned CNP (default)
python predict.py --E 0.0043 --model dist         # any elasticity, inside or outside the database
python predict.py --E 0.06 --save curve.png --csv curve.csv
python predict_critical.py --n 40 --save critical.png
python predict_critical.py --test-only --model dist
```

`--model` is one of `cnp` (zero-shot), `cnp_nbr` (conditioned on the
neighbours, default), `dist` (distilled transformer) or `sarl` (median
of the teachers refined by the RL agent; available for the
elasticities of the database, for which the median is shipped).

Every prediction prints the neighbours it used, so that the leak-free
chain stays visible:

```
E = 0.0055   model = cnp_nbr   (test)
  2 predicted branch(es), neighbours E = 0.0054 and 0.0056
  Ta_c = 154.308   k_c = 4.70
  Floquet: Ta_c = 154.270  k_c = 4.70   |   error 0.02 % on Ta_c, 0.00 on k_c
```

---

## What the models do on the 64 held-out elasticities

Everything below is produced by the scripts of `leakfree/` and stored
in `results/`. The split is by elasticity, so none of these 64
elasticities was seen during training, in any form.

| Method | Ta_c (mean rel. error) | k_c (mean abs. error) | marginal curve | source |
|---|---|---|---|---|
| CNP zero-shot | 1.53 % | 1.18 | 2.33 % | `results/cnp/metrics.json` |
| **CNP conditioned on the neighbours** | **1.29 %** | **0.71** | **1.38 %** | `results/cnp_nbr/metrics.json` |
| DIST (distilled student, 5 seeds) | 1.56 % | 1.16 | 2.53 % | `results/dist/metrics.json` |
| Median of the 29 teachers | 1.55 % | 1.03 | 1.52 % | `results/teachers/metrics.json` |
| Median + SARL refiner (5 seeds) | 1.58 % | 1.03 | 1.51 % | `results/rl/metrics.json` |
| Median + MARL refiner (3 seeds) | 1.58 % | 1.03 | 1.54 % | `results/marl/metrics.json` |
| Best single teacher (Envelope-SIREN) | 1.08 % | 0.67 | 1.84 % | `results/teachers/metrics.json` |
| PCHIP interpolation of the critical values | 0.17 % | 0.30 | -- | `results/baselines/metrics.json` |
| PCHIP interpolation of the curves (2-D) | 0.20 % | 0.03 | 0.37 % | `results/baselines/metrics.json` |
| Branch interpolation, no network | 0.33 % | 0.28 | 0.45 % | `results/baselines/metrics.json` |

Read this table before using a network: **on this database, plain
interpolation between neighbouring elasticities is more accurate than
every surrogate**, on the threshold and on the curve alike. The
networks are a compact, differentiable, millisecond-fast
representation of the map, not a more accurate one. The branch
interpolation shows how much of their accuracy comes from the
leak-free branch structure alone.

The curve error of the networks and of the branch interpolation is
measured on their own support, which covers 84 % of the Floquet grid
(see *Known limits*); the 2-D interpolation covers it entirely.

**Mode structure** (`results/modes/metrics_modes.json`; peaks of the
marginal curve with a prominence of at least 2 % of Ta_c, matched
within |dk| <= 1):

| Method | peaks found | predicted peaks that are real | dominant mode |
|---|---|---|---|
| PCHIP-2D | 91 % | 90 % | 100 % |
| Branch interpolation | 85 % | 96 % | 98 % |
| CNP conditioned | 75 % | 89 % | 98 % |
| Median of the 29 teachers | 71 % | 89 % | 97 % |

**Resonance band withheld** (`data/split_block.csv`: the whole band
E in [0.08, 0.16], 29 elasticities, removed from training, then
predicted). No method crosses it: the error on Ta_c rises to 5.0 % for
the conditioned CNP and 3.5 % for PCHIP, and the two exchanges of
dominant mode inside the band are missed by all of them. This is the
honest limit of the surrogate: it interpolates the map, it does not
extrapolate a bifurcation it has never seen.

**The 23 descriptors of a branch.** Here $`w=k_R-k_L`$,
$`A=\mathrm{Ta}_{\max}-\mathrm{Ta}_{\min}`$, $`\mathrm{Ta}_L`$ and $`\mathrm{Ta}_R`$
are the values at the two ends, and $`k_{\min}`$ the position of the
minimum. The six marked with a dagger are constant over the whole
database in the pipeline used here.

| Descriptor | Definition | What it says about the mode |
|---|---|---|
| $`\log_{10}E`$ | log-elasticity | elasticity |
| branch order † | rank of the branch along $`k`$, in $`[0,1]`$ | place in the succession of modes |
| number of branches † | branches at this elasticity | modes in competition |
| first, last branch † | 1 for the first, for the last branch | long-, short-wavelength mode |
| width | $`w=k_R-k_L`$ | extent of the mode in $`k`$ |
| amplitude | $`A=\mathrm{Ta}_{\max}-\mathrm{Ta}_{\min}`$ | height of the branch |
| left, right width | $`k_{\min}-k_L`$, $`k_R-k_{\min}`$ | position of the minimum |
| left, right rise | $`\mathrm{Ta}_L-\mathrm{Ta}_{\min}`$, $`\mathrm{Ta}_R-\mathrm{Ta}_{\min}`$ | height of the bounding peaks |
| width asymmetry | $`\dfrac{(k_R-k_{\min})-(k_{\min}-k_L)}{w}`$ | asymmetry in $`k`$ |
| rise asymmetry | $`\dfrac{(\mathrm{Ta}_R-\mathrm{Ta}_{\min})-(\mathrm{Ta}_L-\mathrm{Ta}_{\min})}{A}`$ | asymmetry in height |
| local slopes | least squares on the four points on each side of the minimum | steepness at the minimum |
| global slope | least squares over the whole branch | trend of the branch |
| mean abs. slope | mean of $`\left\lvert\dfrac{\mathrm{d}\mathrm{Ta}}{\mathrm{d}k}\right\rvert`$ | mean variation |
| mean abs. curvature | mean of $`\left\lvert\dfrac{\mathrm{d}^{2}\mathrm{Ta}}{\mathrm{d}k^{2}}\right\rvert`$ | mean curvature |
| curvature at minimum | $`\dfrac{\mathrm{d}^{2}\mathrm{Ta}}{\mathrm{d}k^{2}}`$ at $`k_{\min}`$ | selectivity of the mode |
| roughness | $`\sqrt{\dfrac{1}{n}\sum_{i=1}^{n}\bigl(\mathrm{Ta}_i-c_i\bigr)^{2}}`$, $`c`$ the chord joining the two ends | departure from a straight branch |
| arc length | $`\dfrac{L}{w}`$, $`L`$ the length of $`\mathrm{Ta}(k)`$ | complexity of the shape |
| switch left, right † | 1 if a mode exchange bounds the branch | presence of an edge peak |

**Permutation importance**, input by input, on the 64 held-out
elasticities: increase of the mean relative error on the marginal curve,
in points of per cent, when the input is shuffled between branches (mean
of five draws). Reference error without shuffling: 2.33 % for the
zero-shot model, 1.38 % for the conditioned one.

| Shuffled input | CNP zero-shot | CNP conditioned |
|---|---|---|
| $`\log_{10}E`$ | +19.9 | +13.3 |
| anchors $`(k_L,k_R,\mathrm{Ta}_{\min},\mathrm{Ta}_{\max})`$ | +17.9 | +2.4 |
| neighbour-interpolated curve | -- | +20.9 |
| the sixteen shape descriptors, together | +0.45 | +0.10 |
| the six mode-organisation descriptors | 0 (constant) | 0 (constant) |

**What the 23 descriptors actually bring**
(`results/features/feature_importance.json`: permutation importance on
the same 64 elasticities). Six of them are constant: the branch rank,
the number of branches, the first/last-branch indicators (absent from
the descriptor file) and the two mode-exchange flags (columns present
but empty -- nothing in the pipeline ever computed them). Of the
seventeen that vary, only log10 E weighs: shuffling all sixteen shape
descriptors together costs 0.45 point of curve error for the zero-shot
CNP and 0.10 for the conditioned one, against 19.9 and 13.3 points for
log10 E. Leak-free, those descriptors are themselves interpolated from
the neighbours, so they add little to what log10 E and the anchors
already carry.

---

## Known limits

1. **Branches stop at k = 20.** For the 142 elasticities whose Floquet
   curve is computed up to k = 50, the branch table only covers
   k <= 20, so no model predicts beyond that. The critical point is
   never beyond k = 20, so Ta_c and k_c are unaffected, but 15 of the
   peaks of the test set lie outside the modelled range and none of
   them can be predicted.
2. **Exchanges inside a branch.** Of the 79 peaks at k <= 20 of the
   test set, 54 lie at a branch boundary and are recovered (51 to 53
   depending on the model); the 25 that lie inside a single branch,
   where the segmentation marked no exchange, are mostly missed (9 to
   12).
3. **Flat minima.** Where the valley of the marginal curve is flat
   (E >= 3.7, Floquet k_c = 18.6), the models place the minimum
   between 14 and 17 while keeping Ta_c within 1 to 2.5 %.
4. **RL refiner.** Leak-free, it changes nothing: window error
   0.0278 -> 0.0286, peaks recovered 71 % -> 72 %. With the leak it
   was given the true position of the exchange and the error of the
   base; without them it has nothing left to correct with.
5. **MARL.** The multi-agent variant first diverged: its policy is
   stochastic and the sampled actions saturated the tanh, where the
   gradient vanishes. The trainer now builds the correction from the mean
   action, penalises the actions ten times more and decays the learning
   rate without restarts. It then converges -- validation error 0.0215
   for the base against 0.0183 -- but on the test set it changes nothing
   either, exactly like SARL.

The natural next step, for whoever continues this work: detect the
exchanges that the segmentation misses, extend the branch table to
k = 50, compute the six empty descriptors from that structure, and
retrain. Points 1 and 2 are the ones that decide whether the peaks are
reproduced.

---

## Retraining on Kaggle

`kaggle_notebooks/` holds one folder per training, each with a
`run.ipynb` and a `kernel-metadata.json`. They expect **this
repository**, its `data/` folder included, as their only dataset.

```bash
# 1. publish the repository as a Kaggle dataset named taylorcouetteml
kaggle datasets create -p . -r zip
# 2. put your username in kernel-metadata.json, then
kaggle kernels push   -p kaggle_notebooks/cnp_nbr_lf
kaggle kernels status YOUR-KAGGLE-USERNAME/tcml-cnp-nbr-lf
kaggle kernels output YOUR-KAGGLE-USERNAME/tcml-cnp-nbr-lf -p runs/cnp_nbr
```

Order of the pipeline, if everything is retrained from scratch:

1. `precision_lf`, `cson_lf`, `star_lf`, `cnp_lf` -- the 29 teachers;
2. `leakfree/precompute_ensemble_median_lf.py` -- their median;
3. `leakfree/build_rl_windows_lf.py` -- the RL windows;
4. `distil_lf` (student of the median) and `rl_lf` (SARL and MARL);
5. `cnp_nbr_lf`, plus the two `*_block` notebooks for the resonance test.

Notes: enable *Internet* in the notebook settings (the notebooks
install a torch build that supports the whole GPU fleet), and count
about 1 h per CNP seed and 7.5 h for the five DIST seeds. The RL
training needs a GPU: our own leak-free run fell back to CPU and hit
the 12 h limit of the platform with MARL unfinished.

---

## Reproducing the evaluation locally

```bash
python leakfree/baselines.py                       # interpolation baselines
python leakfree/eval_cnp_leakfree.py  --run models_trained/cnp_nbr --repo . \
       --critical_csv data/combined_data.csv --out results/cnp_nbr
python leakfree/eval_teachers_lf.py   --repo . --runs models_trained \
       --desc data/branch_functional_descriptors_leakfree_all.csv \
       --split data/split_by_E.csv --families dist --out results/dist
python leakfree/eval_rl_lf.py         --repo . --rl_run models_trained/sarl \
       --windows data/rl_windows/rl_switch_windows_lf_test.npz \
       --median data/ensemble_median_targets_lf.npz \
       --desc data/branch_functional_descriptors_leakfree_all.csv \
       --split data/split_by_E.csv --out results/rl --agg weighted
python leakfree/modes_test.py                      # peaks and mode exchanges
python leakfree/rl_peaks_test.py                   # peaks, with and without the RL refiner
python leakfree/feature_importance.py              # what the 23 descriptors bring
python leakfree/predictions_cases.py               # the figures of the test elasticities
```

In those figures every predicted mode is drawn as its own branch, and the
comparison stops at k = 20, the range the branch table covers. Inside a
branch the prediction is lightly filtered (Savitzky-Golay, 9 points of 101)
with a weight that falls to zero at both ends, so the peaks where one mode
gives way to the next keep their predicted value exactly. The filter is for
reading only and moves the errors by less than 0.12 point of per cent; both
values are printed by the script and kept in
`results/predictions/per_E_predictions.csv`.

```bash
```

Each script writes its metrics under `results/` and its figures under
the folder named in its header; the values quoted above are exactly
those files.

---

## Contact

Mohamed Hayani Choujaa &mdash; <hayani.med11@gmail.com>
&middot; Laboratoire de Mecanique, Faculte des Sciences Ain Chock,
Universite Hassan II de Casablanca, Maroc.
