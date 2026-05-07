# Taylor-Couette ML — neural surrogates for linear stability

Machine learning surrogates for the linear stability of an oscillatory
Taylor-Couette flow of an Upper-Convected Maxwell (UCM) fluid with
counter-oscillating cylinders. Extends Hayani Choujaa et al.,
*Phys. Fluids* **33**, 074105 (2021).

The common task: predict the marginal stability curve $T_a(k, E)$ for
720 branches parameterised by $E \in [10^{-4}, 10]$ and 23 functional
descriptors.

## Repository layout

```
Work/
├── README.md                this file
├── requirements.txt
├── .gitignore
│
├── data/                    place combined_data.csv here
├── data_pipeline/           datasets, RL windows, normalisations
├── common/                  logging, seeds, IO, interpolation utils
├── configs/                 YAML configs (sizes.yaml etc.)
│
├── models/                  9 architectures
│   ├── utils.py                       kink detection + selective smoothing (shared)
│   ├── sparse_moe_transformer/        Switch-Segment Sparse Transformer (49 M params)
│   ├── fno_latent_diffusion/          Fourier Neural Operator + Latent Diffusion (DDIM)
│   ├── sac_pro/                       SAC single-agent refiner (Conv1D + AttnPool)
│   ├── marl_3sac/                     3-SAC MARL refiner with Cross-Attention
│   ├── diffusion_policy_marl/         7-agent Diffusion-Policy MARL + GAT communication
│   ├── siren/                         SIREN with FiLM conditioning
│   ├── deeponet/                      DeepONet with Fourier features on the trunk
│   ├── chebyshev_spectral/            Transformer → 16 Chebyshev coefficients
│   └── envelope_siren/                Multi-Mode Envelope SIREN  ★ best
│
├── scripts/                 training, inference, figure generation
└── checkpoints/             pretrained weights (4 precision models)
```

## Installation

```bash
python -m pip install -r requirements.txt
# (optional) virtual environment
#   python -m venv .venv && source .venv/Scripts/activate
```

Dependencies: Python ≥ 3.10, PyTorch 2.4, numpy, pandas, scipy,
matplotlib, PyYAML, h5py.

## Data

Place the source file `combined_data.csv` (Floquet eigenvalue ground
truth) under `data/Input/`:

```
data/
├── Input/
│   └── combined_data.csv      columns: Value (= Ta), Time (= k), E
└── processed/
    ├── branch_functional_descriptors.csv     720 branches × 23 features
    └── model_profile_level_dataset.csv       per-(E, branch, s) feature lookup
```

Both processed CSVs are bundled in `data/processed/` so the precision
training and inference scripts run out of the box.

## How to train each model

Each model has a dedicated script in `scripts/`. All scripts accept the
same standard arguments: `--epochs`, `--batch`, `--lr`, `--seed`.

| Model | Script | Config | CPU time |
|---|---|---|---|
| Sparse-MoE Transformer | `train_ssst_pro.py` | `configs/sizes.yaml` | 4–6 h |
| FNO + Latent Diffusion | `train_neptune_pro.py` | `configs/sizes.yaml` | 3–5 h |
| SAC single-agent | `train_sarl_pro.py` | `configs/sizes.yaml` | 1–2 h |
| 3-SAC + Cross-Attention | `train_marl_pro.py` | `configs/sizes.yaml` | 2–3 h |
| Diffusion-Policy MARL | `train_hydra_pro.py` | `configs/sizes.yaml` | 4–7 h |
| SIREN + FiLM | `train_precision_curves.py` | (built-in) | 12 min |
| DeepONet | `train_precision_curves.py` | (built-in) | 5 min |
| Chebyshev Spectral | `train_precision_curves.py` | (built-in) | 17 min |
| Multi-Mode Envelope SIREN ★ | `train_precision_curves.py` | (built-in) | 49 min |

### Precision models (the latest 4)

A single script trains the four sequentially:

```bash
python scripts/train_precision_curves.py --epochs 200 --batch 8 --lr 2e-4
```

Output: `checkpoints/{siren,deeponet,chebyshev,envelope_siren}/best.pt`.

To retrain only one model, comment out the other `train_one(...)` calls
in the `main()` function of `train_precision_curves.py`.

### Heavy-capacity models

Prerequisite: canonical datasets (output of `data_pipeline/canonical.py` +
the canonical RL windows pipeline).

```bash
# Long Sparse-MoE-T training (300 epochs)
python scripts/phase_A_long_ptssst.py
# FNO + Latent Diffusion ensemble (M = 5)
python scripts/phase_B_fnoldm_ensemble.py
# Refiners: SAC, 3-SAC, Diffusion-Policy MARL
python scripts/phase_C_refiner_cycle.py
```

Phase A produces `canonical_ssst_pro.csv`, phase B produces
`canonical_neptune_hydra.csv`, and phase C produces the
`canonical_{sarl,marl,hydra}_strong.csv` refiner predictions.

## Inference and figures

Every script expects checkpoints under `checkpoints/<model>/best.pt`.

```bash
# Evaluate the 4 precision models on the 720 input branches (panel + metrics)
python scripts/inference_and_figures.py
# 10 individual branch figures (one per representative branch)
python scripts/make_curve_gallery.py
# 420 figures, one per unique E value
python scripts/make_all_E_figures.py
# Test the heavy-capacity models on the original physical Input
python scripts/test_on_original_input.py
```

## Final performance (MAE in physical $T_a$ units)

Evaluation on the 720 branches of `combined_data.csv`:

| Model | MAE_phys | RMSE_phys | Kink-zone MAE | MaxErr |
|---|---|---|---|---|
| **Multi-Mode Envelope SIREN** ★ | **0.549** | 1.10 | **0.785** | 26.7 |
| SIREN + FiLM | 0.897 | 1.55 | 1.30 | 30.1 |
| DeepONet | 1.34 | 2.52 | 2.10 | 48.0 |
| Sparse-MoE-T + 3-SAC (legacy reference) | 1.31 | — | — | — |
| Chebyshev Spectral | 2.53 | 7.27 | 3.33 | 182.9 |

The **Multi-Mode Envelope SIREN** beats the legacy reference by a factor
of ×2.4 thanks to its physics-aligned inductive bias:
$T_a(k, E) = \min_{m=1\ldots M} T_m(k, E)$ over $M = 4$ Floquet modes,
which produces cusps by construction wherever the argmin index switches.

## Reference

```bibtex
@article{HayaniChoujaa2021,
  author  = {M. Hayani Choujaa and others},
  title   = {Linear stability of an oscillatory Taylor-Couette flow of
             an Upper-Convected Maxwell fluid with counter-oscillating
             cylinders},
  journal = {Physics of Fluids},
  volume  = {33},
  number  = {7},
  pages   = {074105},
  year    = {2021},
}
```
