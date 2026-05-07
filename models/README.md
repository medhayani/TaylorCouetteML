# models/

Nine architectures, ordered by increasing capacity and specialisation.
Each subfolder is a self-contained unit holding the architecture source.

## Heavy-capacity models trained on the 720 branches

| Folder | Architecture | Key idea |
|---|---|---|
| `sparse_moe_transformer/` | Switch-Segment Sparse Transformer (49 M) | top-k routing across experts per segment |
| `fno_latent_diffusion/` | FNO 1D + Latent Diffusion (DDIM) | spectral operator + denoising score matching |
| `sac_pro/` | Single-agent SAC + Conv1D + AttnPool | RL refiner over 49-pt windows |
| `marl_3sac/` | 3-SAC + Cross-Attention | three agents (location / shape / geometry) |
| `diffusion_policy_marl/` | 7-agent Diffusion Policy + GAT + validator | MAPPO / COMA / IQL / MADT / PBT |

## Precision models — 4 architectures dedicated to tight curve fitting

| Folder | Architecture | Params | MAE_phys | Kink-zone MAE |
|---|---|---|---|---|
| `siren/` | SIREN + FiLM | 0.83 M | 0.897 | 1.30 |
| `deeponet/` | DeepONet + Fourier features | 0.48 M | 1.34 | 2.10 |
| `chebyshev_spectral/` | Transformer → 16 Chebyshev coefficients | 3.23 M | 2.53 | 3.33 |
| `envelope_siren/` ★ | $T_a = \min_m T_m$, M = 4 SIRENs | 1.77 M | **0.549** | **0.785** |

`utils.py` (at the root of `models/`) provides shared helpers:
- `detect_kinks_np(ta_norm)` — locates cusps via the z-score of $|d^2 T_a|$
- `kink_weighted_mse(pred, true)` — MSE with ×5 weight on kink samples
- `selective_smooth(pred, true_norm)` — Savitzky-Golay everywhere except the kink zones
