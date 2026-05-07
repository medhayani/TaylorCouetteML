# models/

Neuf architectures, par ordre de capacité et complexité. Chaque sous-dossier
est une unité autonome contenant le code de l'architecture.

## Modèles « production » entraînés sur les 720 branches

| Folder | Architecture | Idée clé |
|---|---|---|
| `sparse_moe_transformer/` | Switch-Segment Sparse Transformer (49M) | Routage top-k entre experts par segment |
| `fno_latent_diffusion/` | FNO 1D + Latent Diffusion (DDIM) | Spectral operator + score matching |
| `sac_pro/` | SAC mono-agent + Conv1D + AttnPool | Refiner RL sur fenêtres 49-pts |
| `marl_3sac/` | 3-SAC + Cross-Attention | Trois agents (loc/shape/geo) + cross-attn |
| `diffusion_policy_marl/` | 7-agent Diffusion Policy + GAT + validateur | MAPPO/COMA/IQL/MADT/PBT |

## Modèles « précision » — 4 architectures dédiées au fit serré

| Folder | Architecture | Params | MAE_phys | Kink-zone MAE |
|---|---|---|---|---|
| `siren/` | SIREN + FiLM | 0.83M | 0.897 | 1.30 |
| `deeponet/` | DeepONet + Fourier features | 0.48M | 1.34 | 2.10 |
| `chebyshev_spectral/` | Transformer → 16 coeffs Chebyshev | 3.23M | 2.53 | 3.33 |
| `envelope_siren/` ★ | $T_a=\min_m T_m$, M=4 SIRENs | 1.77M | **0.549** | **0.785** |

`utils.py` (à la racine de `models/`) fournit les helpers partagés :
- `detect_kinks_np(ta_norm)` : détecte les cusps via $|d^2 T_a|$ z-score
- `kink_weighted_mse(pred, true)` : MSE pondérée ×5 aux kinks
- `selective_smooth(pred, true_norm)` : Savitzky-Golay sauf zones de kink
