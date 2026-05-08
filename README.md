# Taylor-Couette ML — neural surrogates for linear stability

Machine learning surrogates for the linear stability of an oscillatory
Taylor-Couette flow of an Upper-Convected Maxwell (UCM) fluid with
counter-oscillating cylinders. Extends Hayani Choujaa et al.,
*Phys. Fluids* **33**, 074105 (2021).

The common task: predict the marginal stability curve $T_a(k, E)$ for
720 branches parameterised by $E \in [10^{-4}, 10]$ and 23 functional
descriptors. Nine architectures are provided, ranging from sparse
mixture-of-experts transformers to a custom multi-mode envelope SIREN
that encodes the Floquet mode-crossing structure.

> **TL;DR.** Clone, install the requirements, and run inference on the
> bundled checkpoints — you should see `MAE_phys = 0.549` for
> Envelope-SIREN, `0.897` for SIREN, `1.34` for DeepONet, `2.53` for
> Chebyshev. Training every model from scratch is also documented below.

---

## Repository layout

```
Work/
├── README.md                this file
├── requirements.txt
├── .gitignore
│
├── data/                    bundled inputs
│   ├── Input/                  combined_data.csv (122 820 rows)
│   └── processed/              branch descriptors + profile features (720 branches)
├── data_pipeline/           datasets, RL windows, normalisations
├── common/                  logging, seeds, IO, interpolation utils
├── configs/                 YAML configs (sizes.yaml etc.)
│
├── models/                  9 architectures
│   ├── utils.py                       kink detection + selective smoothing (shared)
│   ├── sparse_moe_transformer/        Switch-Segment Sparse Transformer (49 M params)
│   ├── fno_latent_diffusion/          Fourier Neural Operator + Latent Diffusion (DDIM)
│   ├── sac_pro/                       single-agent SAC refiner (Conv1D + AttnPool)
│   ├── marl_3sac/                     3-SAC MARL refiner with Cross-Attention
│   ├── diffusion_policy_marl/         7-agent Diffusion-Policy MARL + GAT
│   ├── siren/                         SIREN with FiLM conditioning
│   ├── deeponet/                      DeepONet with Fourier features on the trunk
│   ├── chebyshev_spectral/            Transformer → 16 Chebyshev coefficients
│   └── envelope_siren/                Multi-Mode Envelope SIREN  ★ best
│
├── scripts/                 training, inference, figure generation
├── checkpoints/             pretrained weights (4 precision models)
└── docs/                    presentation PDFs (English + French)
```

---

## 1. Environment setup

Tested on Python **3.10 / 3.11**, PyTorch **2.4.x**, CPU-only and
CUDA 12. CPU-only training works for every model except the
heavy-capacity ones (Sparse-MoE-T, FNO+LDM, MARL) where a GPU is
strongly recommended.

### 1a. Clone the repo

```bash
git clone https://github.com/medhayani/TaylorCouetteML.git
cd TaylorCouetteML
```

### 1b. Install dependencies

#### 🪟 Windows (PowerShell)

```powershell
# Create a virtual environment
python -m venv .venv
# Activate it
.\.venv\Scripts\Activate.ps1
# If activation fails: PowerShell may block scripts. Run once as admin:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# CPU-only PyTorch is what requirements.txt installs by default.
# For NVIDIA GPU (CUDA 12.1):
#   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
```

#### 🐧 Ubuntu / Linux (bash)

```bash
# Create a virtual environment
python3 -m venv .venv
# Activate it
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# For NVIDIA GPU (CUDA 12.1):
#   python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
```

### 1c. Verify the installation

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from models.envelope_siren.model import MultiModeEnvelopeSIREN; print('imports OK')"
```

You should see something like `torch 2.4.1 cuda False` (CPU) and
`imports OK`. If imports fail, make sure you launch Python from the
repo root so that `models/`, `common/`, and `data_pipeline/` are
visible on the `PYTHONPATH`.

### 1d. Data sanity check

```bash
python -c "
import pandas as pd
raw = pd.read_csv('data/Input/combined_data.csv')
desc = pd.read_csv('data/processed/branch_functional_descriptors.csv')
print('Input rows:', len(raw), 'unique E:', raw.iloc[:,2].nunique())
print('Branches  :', len(desc))
"
```

Expected: `Input rows: 122820 unique E: 420` and `Branches: 720`.

---

## 2. Run inference on the pretrained checkpoints (no training needed)

Four precision-model checkpoints are committed in `checkpoints/`.
Reproduce the published metrics in seconds:

```bash
# 12-panel figure + CSV table (predictions vs Input)
python scripts/inference_and_figures.py
```

Expected end-of-run output:

```
        model  MAE_phys  RMSE_phys  MaxErr_phys  MAE_kink_zone  n_branches
EnvelopeSIREN  0.549484   1.095493    26.678650       0.784591         720
        SIREN  0.896536   1.547207    30.102631       1.302510         720
     DeepONet  1.340526   2.520379    47.971466       2.103407         720
    Chebyshev  2.534420   7.270223   182.947021       3.330832         720
```

Two additional figure scripts:

```bash
# 10 individual branch figures (one per representative branch across log10 E)
python scripts/make_curve_gallery.py

# 420 figures, one per unique E value
python scripts/make_all_E_figures.py
```

All outputs are written under `figures/` (the folder is created on the fly).

### Loading a single checkpoint in your own code

```python
import torch
from models.envelope_siren.model import MultiModeEnvelopeSIREN

m = MultiModeEnvelopeSIREN(ctx_dim=24, n_modes=4)
ck = torch.load("checkpoints/envelope_siren/best.pt", weights_only=False, map_location="cpu")
m.load_state_dict(ck["state_dict"])
m.eval()

# ck also contains "ctx_mean" and "ctx_std" — apply the same standardisation
# to your context vectors before calling m(k_norm, ctx).
```

---

## 3. Train each model from scratch

Every script accepts the standard arguments
`--epochs`, `--batch`, `--lr`, `--seed`. Default values are tuned for CPU.

| # | Model | Script | CPU time | GPU time | Output |
|---|---|---|---|---|---|
| 1 | Sparse-MoE Transformer | `train_ssst_pro.py` | 4–6 h | ~30 min | `checkpoints/ssst_pro/best.pt` |
| 2 | FNO + Latent Diffusion | `train_neptune_pro.py` | 3–5 h | ~25 min | `checkpoints/neptune_pro/best.pt` |
| 3 | SAC single-agent | `train_sarl_pro.py` | 1–2 h | ~10 min | `checkpoints/sarl_pro/best.pt` |
| 4 | 3-SAC + Cross-Attn | `train_marl_pro.py` | 2–3 h | ~15 min | `checkpoints/marl_pro/best.pt` |
| 5 | Diffusion-Policy MARL | `train_hydra_pro.py` | 4–7 h | ~40 min | `checkpoints/hydra_pro/best.pt` |
| 6 | SIREN + FiLM | `train_precision_curves.py` | 12 min | < 5 min | `checkpoints/siren/best.pt` |
| 7 | DeepONet | `train_precision_curves.py` | 5 min | < 2 min | `checkpoints/deeponet/best.pt` |
| 8 | Chebyshev Spectral | `train_precision_curves.py` | 17 min | < 5 min | `checkpoints/chebyshev/best.pt` |
| 9 | **Envelope-SIREN** ★ | `train_precision_curves.py` | 49 min | ~12 min | `checkpoints/envelope_siren/best.pt` |

### 3a. Train all 4 precision models (one command)

These are the models that produced the published numbers above.

```bash
python scripts/train_precision_curves.py --epochs 200 --batch 8 --lr 2e-4
```

Total runtime: ~80 min on CPU (i7), ~25 min on a single GPU.

You will see per-epoch logs like:

```
>>> Training SIREN (kink-weighted)
  [SIREN] params=0.83M  train=504  val=112
    ep   1/200  train_kw_mse=0.06865  val_mse=0.06245  val_mae=0.18402
    ep  50/200  train_kw_mse=0.00102  val_mse=0.00170  val_mae=0.01672
    ep 200/200  train_kw_mse=0.00006  val_mse=0.00115  val_mae=0.00982
  [SIREN] done 12.0 min  best_val_mse=0.00115
```

To retrain only one of the four models, comment out the other
`train_one(...)` calls in `main()` of `train_precision_curves.py` —
each block is independent.

### 3b. Train the heavy-capacity models (recommended GPU)

These need a few extra files produced by `data_pipeline/canonical.py`
(canonical 41-pt grid) and the RL windows export (49-pt windows).
The convenient three-phase pipeline that builds everything in order:

```bash
# Phase A — Sparse-MoE-T (300 epochs)
python scripts/phase_A_long_ptssst.py
# Phase B — FNO + Latent Diffusion (ensemble M = 5)
python scripts/phase_B_fnoldm_ensemble.py
# Phase C — SAC, 3-SAC, Diffusion-Policy MARL refiners
python scripts/phase_C_refiner_cycle.py
```

Each phase writes its checkpoints and metrics under `runs/<phase>/...`
and a canonical-grid CSV that the next phase consumes.

You can also train any single model directly:

```bash
python scripts/train_ssst_pro.py     --epochs 100 --batch 16
python scripts/train_neptune_pro.py  --epochs 100 --batch 16
python scripts/train_sarl_pro.py     --epochs 50  --batch 16
python scripts/train_marl_pro.py     --epochs 50  --batch 16
python scripts/train_hydra_pro.py    --epochs 50  --batch 8
```

### 3c. GPU vs CPU

By default, every script uses CPU (`device = torch.device("cpu")`).
To use a GPU, edit the top of the relevant script or set the
environment variable before launching:

```bash
# Linux/macOS
export CUDA_VISIBLE_DEVICES=0
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"
```

`scripts/train_precision_curves.py` already auto-detects CUDA if
available — change line `device = torch.device("cpu")` to
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
to enable.

### 3d. Reproducibility

Every script seeds NumPy and PyTorch via `--seed` (default 42). The
exact data splits are deterministic given the seed. Bit-exact
reproducibility on CUDA additionally requires
`torch.use_deterministic_algorithms(True)` and the matching cuDNN
backend flags — see `common/seeds.py` for the helper.

---

## 4. Final benchmark

Evaluation on the 720 branches of `combined_data.csv`, MAE in
physical $T_a$ units:

| Model | MAE_phys | RMSE_phys | Kink-zone MAE | MaxErr |
|---|---|---|---|---|
| **Multi-Mode Envelope SIREN** ★ | **0.549** | 1.10 | **0.785** | 26.7 |
| SIREN + FiLM | 0.897 | 1.55 | 1.30 | 30.1 |
| DeepONet | 1.34 | 2.52 | 2.10 | 48.0 |
| Sparse-MoE-T + 3-SAC (legacy reference) | 1.31 | — | — | — |
| Chebyshev Spectral | 2.53 | 7.27 | 3.33 | 182.9 |

The Multi-Mode Envelope SIREN beats the legacy reference by a factor
of ×2.4 thanks to its physics-aligned inductive bias:
$T_a(k, E) = \min_{m=1\ldots M} T_m(k, E)$ over $M = 4$ Floquet modes,
which produces cusps by construction wherever the argmin index switches.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: models` | running from a sub-folder | `cd` to the repo root first |
| `CUDA out of memory` | batch too large | lower `--batch`, e.g. `--batch 4` |
| `enable_nested_tensor` warning at startup | PyTorch 2.4 cosmetic | harmless, ignore |
| `LF will be replaced by CRLF` (Windows git) | line-ending normalisation | harmless, ignore |
| `pdflatex not found` (when reading `docs/`) | no need to compile, PDFs are bundled | open them directly |
| Import succeeds but `forward` mismatches | wrong `ctx_dim` | always pass `ctx_dim=24` for precision models |
| Checkpoint loading raises `unexpected key` | older / mismatched architecture | use the matching model class (see `models/<name>/model.py`) |

---

## 6. Citation

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

---

## 7. Documentation

Full mathematical write-up of the nine architectures, the Floquet
formulation, the kink-weighted loss, and the selective smoothing is
available in `docs/`:

- 🇬🇧 `docs/presentation_en.pdf`
- 🇫🇷 `docs/presentation_fr.pdf`

License: research code, all rights reserved by the author. Open an
issue if you would like a permissive license added.
