# TaylorCouetteML

**Author** Mohamed Hayani Choujaa
&middot; Laboratoire de Mecanique, Faculte des Sciences Ain Chock, Universite Hassan II de Casablanca, Maroc

Three neural-network surrogates for the linear-stability marginal
curve $T_a(k,E)$ of an oscillatory Taylor-Couette flow in an
upper-convected Maxwell (UCM) fluid with counter-oscillating
cylinders.

---

## What is in this repository

```
TaylorCouetteML/
|-- predict.py                           <- ONE-COMMAND PREDICTION  Ta(k)
|-- predict_critical.py                  <- ONE-COMMAND CRITICAL CURVES  Ta_c(E), k_c(E)
|
|-- kaggle_notebooks/                    <- ready-to-push Kaggle kernels
|   |-- cnp_v2/                              (CNP training)
|   |-- distil_v1/                           (DIST training)
|   |-- rl_v2/                               (SARL + MARL ensemble training)
|   `-- ... (12 more for the supervised baselines)
|
|-- code/                                <- everything you need to run
|   |-- predict_3models_v8_rl_aug.py        (main figure renderer)
|   |-- make_panel12_figure.py              (12-panel summary figure)
|   |-- eval_rl_aggregations.py             (benchmark 9 RL aggregations)
|   |-- eval_rl_v2_vs_v3.py                 (D5-base vs MED-NN-base study)
|   |-- build_rl_windows_v4_mednn_base.py   (rebuild switch windows)
|   |-- models/                             (architecture code: CNP, STAR,
|   |                                        SARL, MARL, SIREN, DeepONet, ...)
|   |-- data_pipeline/                      (HydraWindowsDataset, ...)
|   |-- configs/sizes.yaml                  (SAC/MARL hyperparameters)
|
|-- models_trained/                      <- shipped trained weights
|   |-- cnp/seed_42..46/best.pt             (CNP, 5 seeds, ~31 MB each)
|   |-- dist/seed_42..46/best.pt            (DIST, 5 seeds, ~55 MB each)
|   `-- sarl/seed_42..46/best.pt            (SARL refiner, 5 seeds, ~20 MB each)
|
|-- data/                                <- input data shipped with the repo
|   |-- combined_data.csv                   (~145k Floquet triplets (T_a,k,E))
|   |-- branch_functional_descriptors.csv   (23 descriptors / 720 branches)
|   |-- model_profile_level_dataset.csv     (normalized 720x101 dataset)
|   |-- ensemble_median_targets.npz         (29-surrogate median targets)
|   |-- rl_windows/                         (switch-window RL dataset, D5 base)
|
`-- figures/                             <- 47 prediction figures Ta(k)
                                            across log-spread E in [1e-4, 10]
```

Total repository size: ~460 MB (code + data + 15 trained checkpoints).
The 29 supervised surrogates of the ensemble baseline and the MARL
refiner weights (~5 GB combined) are **not** stored in git -- they
live on Kaggle and can be fetched on demand (see below).

---

## The three architectures

| Name     | What it does                                         | Inference cost |
|----------|------------------------------------------------------|----------------|
| **CNP**  | Conditional Neural Process. Zero-shot from the 23 descriptors, OR few-shot if you provide a few exact $(k_c,T_{a,c})$ points as context. | 1 forward pass |
| **DIST** | Distilled Spectral Transformer. Single student model that reproduces the median of 29 supervised surrogates. | 1 forward pass (~30x faster than the explicit median) |
| **RL-REF** | Reinforcement-learning refiner: 5 SARL seeds + 5 MARL seeds, weighted-mean aggregation over the SARL ensemble. Specialises in the cusp/switch zone. | ~6 forward passes |

Final performance on the test split (107 branches, 101 points each):

| Model                          | MAE_norm | RMSE_norm | R^2    |
|--------------------------------|----------|-----------|--------|
| Ensemble median of 29          | 0.009    | 0.016     | 0.996  |
| Pointwise oracle (upper bound) | 0.004    | 0.010     | 0.999  |
| CNP zero-shot                  | 0.011    | 0.018     | 0.994  |
| CNP few-shot (N_c = 4)         | 0.004    | 0.010     | 0.999  |
| DIST                           | 0.010    | 0.017     | 0.995  |
| RL-REF (refiner ensemble)      | 0.008    | 0.015     | 0.996  |

---

## Quickstart -- one-command prediction

The script `predict.py` (in the repository root) loads the shipped
trained weights and produces a prediction of the marginal stability
curve $T_a(k)$ at any value of the elasticity number $E$. Install
the dependencies once, then a single command outputs both a CSV and
a plot.

### Install

```bash
pip install "numpy<2" torch==2.2.2 matplotlib pandas pyyaml scipy
```

(`numpy<2` is required because torch 2.2.2 was built against
NumPy~1.x.)

### Predict

```bash
# Predict at E = 0.001 with the distilled transformer (default, fastest):
python predict.py --E 0.001

# Same E but with the Conditional Neural Process (zero-shot):
python predict.py --E 0.001 --model cnp

# Save the prediction figure to disk:
python predict.py --E 0.001 --save prediction.png

# Also save the (k, Ta) prediction as a CSV file:
python predict.py --E 0.001 --csv prediction.csv
```

The script automatically falls back to the closest available $E$
value if the exact one is not in the dataset, overlays the Floquet
ground-truth points when present, and runs on CPU (no GPU needed).
Inference time on a typical laptop: <5 seconds.

### Critical-curve sweep over E

To trace the critical Taylor number $T_{a,c}(E) = \min_k T_a(k,E)$
and the critical wavenumber $k_c(E) = \arg\min_k T_a(k,E)$ over a
range of elasticity numbers, use `predict_critical.py`:

```bash
# Default: 80 log-spaced E values in [1e-4, 10] with the DIST model
python predict_critical.py

# Custom range, choose the number of E points
python predict_critical.py --E_min 1e-4 --E_max 10 --n_E 200

# Or specify the step in log10(E) directly (here: one E per 0.05 dex)
python predict_critical.py --E_step 0.05

# Save the figure (Tac(E) and kc(E) panels) and the values
python predict_critical.py --E_step 0.1 --save critical.png --csv critical.csv

# Try the CNP zero-shot model instead of the distilled transformer
python predict_critical.py --model cnp --n_E 60
```

Output:
- A two-panel figure with $T_{a,c}(E)$ on top and $k_c(E)$ on
  bottom (one point per branch, plus the global minimum across
  branches in colour).
- An optional CSV with columns `E, branch_local_id, log10E, Ta_c, k_c`.

Runtime: ~1-2 minutes on CPU for 80 E values (DIST), ~3-4 min for CNP.

---

## Ready-to-use Kaggle notebooks

A copy of every Kaggle notebook used to train the models is shipped
in `kaggle_notebooks/` (15 kernels, ~140 KB total). Each subfolder
contains a `kernel-metadata.json` descriptor and a `run.ipynb`
notebook (install torch, copy code, launch the training script, list
outputs).

| Folder | Kernel title | Trains |
|--------|--------------|--------|
| `cnp_v2/`       | TCML CNP v2 zero-shot dominant     | CNP zero-shot dominant, 5 seeds              |
| `distil_v1/`    | TCML DISTIL STAR v1                | DIST (distillation of 29-surrogate median), 5 seeds |
| `rl_v2/`        | TCML RL v2 SARL + MARL 1500ep      | SARL+MARL ensemble used in production        |
| `cnp_v1/`       | TCML CNP v1 1200ep 5seeds          | CNP-TC, 5 seeds                              |
| `cson_v1/`      | TCML CSON v1 1000ep 5seeds         | CSON, 5 seeds                                |
| `star_v1/`      | TCML STAR v1 1500ep 7seeds         | STAR, 7 seeds                                |
| `precision_v2/` | TCML Precision v2 ensemble         | 12 supervised surrogates (SIREN/DeepONet/Cheb/Envelope-SIREN x 3 seeds) |
| `ssst_v2/`      | TCML SSST v2 1000ep                | Sparse-MoE Transformer                       |
| `neptune_v2/`   | TCML NEPTUNE v2 1000ep M1          | FNO + Latent Diffusion                       |
| `hydra/`        | TCML Hydra prod                    | Diffusion-Policy MARL baseline               |
| `sarl/`,`marl/` | original single-seed refiners      | reference, superseded by `rl_v2/`            |

### Push one of these notebooks to your Kaggle account

```bash
pip install kaggle
# Place your kaggle.json API token in ~/.kaggle/kaggle.json
cd kaggle_notebooks/cnp_v2/
# Edit kernel-metadata.json: replace "hayanichoujaamohamed/..." with your-user/...
kaggle kernels push -p .
kaggle kernels status   <your-user>/tcml-cnp-v2
kaggle kernels output   <your-user>/tcml-cnp-v2 -p ./runs/cnp_v2
```

Each notebook expects two input datasets:
`hayanichoujaamohamed/tcml-aux` (~38 MB of input data and RL windows)
and `hayanichoujaamohamed/tcml-code` (a copy of this repository,
packaged as a Kaggle dataset). They can be added directly from my
public account or forked.

---

## How to use this repository on Kaggle (detail)

All training was done on Kaggle (GPU NVIDIA Tesla P100, 16 GB),
and all checkpoints used by the inference scripts here live on
Kaggle datasets and kernel outputs.

### Public Kaggle assets

| Asset | Kaggle path | Content |
|-------|-------------|---------|
| Auxiliary data | `hayanichoujaamohamed/tcml-aux` | `combined_data.csv`, descriptors, ensemble median, RL windows (D5 and MED-NN bases) |
| Source code | `hayanichoujaamohamed/tcml-code` | this repository, packaged as a dataset for Kaggle notebooks |
| CNP weights | kernel `tcml-cnp-v2-zero-shot-dominant` | 5 seeds best.pt, ~31 MB each |
| DIST weights | kernel `tcml-distil-star-v1` | 5 seeds best.pt, ~55 MB each |
| RL-REF weights | kernel `tcml-rl-v2-sarl-marl-1500ep-5seeds` | 5 SARL + 5 MARL seeds, total ~630 MB |
| 29 supervised surrogates | kernels `tcml-precision-v2-ensemble`, `tcml-cson-v1-1000ep-5seeds`, `tcml-cnp-v1-1200ep-5seeds`, `tcml-star-v1-1500ep-7seeds`, `tcml-distil-star-v1`, `tcml-ssst-v2-1000ep`, `tcml-neptune-v2-1000ep-m1` | base models for the ensemble median |

### Re-train a model on Kaggle

1. Add `hayanichoujaamohamed/tcml-aux` and `hayanichoujaamohamed/tcml-code`
   to your Kaggle notebook as input datasets.
2. In the first cell, copy the code into `/kaggle/working/TaylorCouetteML/`:
   ```python
   from pathlib import Path
   import shutil
   INPUT = Path('/kaggle/input')
   AUX_DIR  = list(INPUT.rglob('combined_data.csv'))[0].parent
   CODE_DIR = list(INPUT.rglob('scripts'))[0].parent
   REPO = Path('/kaggle/working/TaylorCouetteML')
   if REPO.exists(): shutil.rmtree(REPO)
   shutil.copytree(CODE_DIR, REPO)
   ```
3. Copy the input files where each script expects them
   (`data/Input/combined_data.csv`, `data/rl_windows/...`, ...).
4. Launch a training script. For instance, CNP (1200 epochs x 5 seeds, ~1.5 h on P100):
   ```bash
   python scripts/train_cnp_pro.py --epochs 1200 --n_seeds 5 \
       --out_root /kaggle/working/runs/cnp_v2
   ```
   The kernel `tcml-rl-v2-sarl-marl-1500ep-5seeds` does the same
   for SARL + MARL refiners using
   `data/rl_windows/rl_switch_windows_*.npz` from the `tcml-aux`
   dataset.

### Download the trained weights from a Kaggle kernel

```bash
pip install kaggle
kaggle kernels output hayanichoujaamohamed/tcml-cnp-v2-zero-shot-dominant      -p ./runs/cnp_v2
kaggle kernels output hayanichoujaamohamed/tcml-distil-star-v1                 -p ./runs/distil_v1
kaggle kernels output hayanichoujaamohamed/tcml-rl-v2-sarl-marl-1500ep-5seeds  -p ./runs/rl_v2
```

The output of each kernel contains `runs/<model>/seed_<n>/best.pt`
files ready to load. Place them under `data/runs/<model>/seed_<n>/`
in your local working copy before running the inference scripts.

---

## How to reproduce the figures locally

### Setup

```bash
pip install "numpy<2" torch==2.2.2 matplotlib pandas pyyaml scipy
```

`numpy<2` is required: torch 2.2.2 was built against NumPy 1.x and
will raise `RuntimeError: Numpy is not available` with NumPy >= 2.0.

### Layout expected by the scripts

The inference scripts use relative paths assuming this layout:

```
<root>/
|-- code/         (scripts + models + data_pipeline + configs)
`-- data/
    |-- Input/combined_data.csv                 (input data)
    |-- processed/                              (functional descriptors, ...)
    |-- runs/cnp_v2/seed_*/best.pt              (CNP weights)
    |-- runs/distil_v1/seed_*/best.pt           (DIST weights)
    |-- runs/sarl_v2/seed_*/best.pt             (SARL weights)
    |-- runs/marl_v2/seed_*/best.pt             (MARL weights)
    |-- runs/precision_v2/seed_*/{siren,deeponet,chebyshev,envelope_siren}/best.pt
    |-- runs/cson_v1/seed_*/best.pt
    |-- runs/cnp_v1/seed_*/best.pt
    |-- runs/star_v1/seed_*/best.pt
    |-- runs/ssst_v2/best.pt
    `-- runs/neptune_v2/member_00/best.pt
```

Download all `runs/` checkpoints via the Kaggle commands above
before running the inference. The input CSV and descriptors
shipped in `data/` of this repository are enough for the data
side.

### Generate the figures

```bash
# Main 12-panel figure (Ta(k) at 12 log-spread E values, one shared legend)
python code/make_panel12_figure.py \
    --out_dir figures/

# 47-figure full sweep, one PNG per E value, with RL-aug full coverage
python code/predict_3models_v8_rl_aug.py \
    --n_figs 50 \
    --out_subdir figures/per_E_50_3models_v8
```

Execution time on CPU (8-core AMD Ryzen): ~3 minutes for the
47-figure sweep, ~30 seconds for the 12-panel figure.

### Reproduce the RL aggregation benchmark

```bash
python code/eval_rl_aggregations.py
python code/eval_rl_v2_vs_v3.py     # only if you also downloaded sarl_v3 / marl_v3 kernel outputs
```

The first script reproduces the choice of *weighted-mean SARL* as
the production aggregation. The second documents the empirical
verification that D5 (specialist) beats MED-NN (generalist) as the
base prediction for the refiner.

---

## Contact

Mohamed Hayani Choujaa &mdash; <hayani.med11@gmail.com>
&middot; Laboratoire de Mecanique, Faculte des Sciences Ain Chock,
Universite Hassan II de Casablanca, Maroc.
