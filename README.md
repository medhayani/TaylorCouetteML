# Taylor-Couette ML — modèles de stabilité linéaire

Surrogates ML pour la stabilité linéaire d'un écoulement de Taylor-Couette
oscillatoire d'un fluide UCM avec cylindres contre-oscillants. Étend
M. Hayani Choujaa et al., *Phys. Fluids* **33**, 074105 (2021).

Cible commune : prédire la courbe de stabilité marginale $T_a(k, E)$
pour 720 branches paramétrées par $E \in [10^{-4}, 10]$ et 23 descripteurs
fonctionnels.

## Structure du dépôt

```
Work/
├── README.md                this file
├── requirements.txt
├── .gitignore
│
├── data/                    placer ici combined_data.csv
├── data_pipeline/           datasets, fenêtres, normalisations
├── common/                  logging, seeds, IO, interp utils
├── configs/                 fichiers YAML (sizes.yaml etc.)
│
├── models/                  9 architectures
│   ├── utils.py                       kink detection + selective smoothing (partagé)
│   ├── sparse_moe_transformer/        Switch-Segment Sparse Transformer (49 M params)
│   ├── fno_latent_diffusion/          FNO 1D + Latent Diffusion (DDIM)
│   ├── sac_pro/                       SAC mono-agent (Conv1D + AttnPool)
│   ├── marl_3sac/                     3-SAC + Cross-Attention
│   ├── diffusion_policy_marl/         7-agent Diffusion Policy MARL (HYDRA)
│   ├── siren/                         SIREN + FiLM
│   ├── deeponet/                      DeepONet + Fourier features
│   ├── chebyshev_spectral/            Transformer → 16 coefficients de Chebyshev
│   └── envelope_siren/                Multi-Mode Envelope SIREN  ★ meilleur résultat
│
├── scripts/                 entraînement, inférence, génération de figures
└── checkpoints/             poids pré-entraînés (4 modèles précision)
```

## Installation

```bash
python -m pip install -r requirements.txt
# Optionnel : créer un venv avant
#   python -m venv .venv && source .venv/Scripts/activate
```

Dépendances : Python ≥ 3.10, PyTorch 2.4, numpy, pandas, scipy, matplotlib,
PyYAML, h5py.

## Données

Placer le fichier source `combined_data.csv` dans `data/Input/` :

```
data/
└── Input/
    └── combined_data.csv      colonnes : Value (= Ta), Time (= k), E
```

Les descripteurs de branches (sortie StepA du pipeline original code4)
sont attendus dans `code4/01_StepA__Segmented_Functional_Analysis/02_outputs/`.
Voir `data_pipeline/dataset.py` pour le format exact.

## Comment entraîner chaque modèle

Chaque modèle a son script dédié dans `scripts/`. Tous prennent les mêmes
arguments standards : `--epochs`, `--batch`, `--lr`, `--seed`.

| # | Modèle | Script | Config | Temps CPU |
|---|---|---|---|---|
| 1 | Sparse-MoE Transformer | `train_ssst_pro.py` | `configs/sizes.yaml` | 4-6 h |
| 2 | FNO + Latent Diffusion | `train_neptune_pro.py` | `configs/sizes.yaml` | 3-5 h |
| 3 | SAC mono-agent | `train_sarl_pro.py` | `configs/sizes.yaml` | 1-2 h |
| 4 | 3-SAC + Cross-Attn | `train_marl_pro.py` | `configs/sizes.yaml` | 2-3 h |
| 5 | Diffusion-Policy MARL | `train_hydra_pro.py` | `configs/sizes.yaml` | 4-7 h |
| 6 | SIREN + FiLM | `train_precision_curves.py` | (intégrée) | 12 min |
| 7 | DeepONet | `train_precision_curves.py` | (intégrée) | 5 min |
| 8 | Chebyshev Spectral | `train_precision_curves.py` | (intégrée) | 17 min |
| 9 | Envelope-SIREN ★ | `train_precision_curves.py` | (intégrée) | 49 min |

### Modèles « précision » (les 4 derniers)

Un seul script entraîne les 4 séquentiellement :

```bash
python scripts/train_precision_curves.py --epochs 200 --batch 8 --lr 2e-4
```

Sortie : `data/runs_precision/{siren,deeponet,chebyshev,envelope_siren}/best.pt`.

Pour ne ré-entraîner qu'un seul modèle, commentez les autres `train_one(...)`
appels dans le `main()` de `train_precision_curves.py`.

### Modèles « PRO » (code7)

Pré-requis : datasets canoniques (sortie de `data_pipeline/canonical.py` +
`07_StepF__Export_RL_Windows_Pro` du pipeline original).

```bash
# Lancer un entraînement long Sparse-MoE-T (300 époques)
python scripts/phase_A_long_ptssst.py
# Ensemble FNO + LDM
python scripts/phase_B_fnoldm_ensemble.py
# Refiners SAC, 3-SAC, Diffusion-MARL
python scripts/phase_C_refiner_cycle.py
```

Ces trois phases reproduisent le pipeline `strong_retraining/` ; la phase A
sort `canonical_ssst_pro.csv`, la B `canonical_neptune_hydra.csv`, et la C
les `canonical_{sarl,marl,hydra}_strong.csv`.

## Inférence et figures

Tous les scripts attendent les checkpoints dans `checkpoints/<model>/best.pt`
ou `data/runs_precision/<model>/best.pt`.

```bash
# Évaluer les 4 modèles précision sur les 720 branches du fichier Input
python scripts/inference_and_figures.py
# 10 figures individuelles (une par branche représentative)
python scripts/make_curve_gallery.py
# 420 figures, une par valeur unique de E
python scripts/make_all_E_figures.py
# Tester les modèles strong (code7) sur le fichier Input physique
python scripts/test_on_original_input.py
```

## Performances finales (MAE en unités physiques de $T_a$)

Évaluation sur les 720 branches de `combined_data.csv` :

| Modèle | MAE_phys | RMSE_phys | Kink-zone MAE | MaxErr |
|---|---|---|---|---|
| **Envelope-SIREN (M=4)** ★ | **0.549** | 1.10 | **0.785** | 26.7 |
| SIREN + FiLM | 0.897 | 1.55 | 1.30 | 30.1 |
| DeepONet | 1.34 | 2.52 | 2.10 | 48.0 |
| D5 + MARL legacy (référence) | 1.31 | — | — | — |
| Chebyshev Spectral | 2.53 | 7.27 | 3.33 | 182.9 |

L'**Envelope-SIREN** bat la baseline legacy d'un facteur ×2.4 grâce à son
biais inductif qui colle à la physique : $T_a(k,E)=\min_m T_m(k,E)$ sur
$M=4$ modes Floquet, donnant des cusps par construction là où l'argmin
change.

## Référence

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
