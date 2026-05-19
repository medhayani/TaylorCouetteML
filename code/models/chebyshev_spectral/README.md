# Chebyshev Spectral Regressor

Decomposes each branch $T_a(k)$ on the truncated Chebyshev basis of the
first kind on $[-1, 1]$:

$$T_a(k_\text{norm}) = \sum_{n=0}^{16} c_n(\xi)\, T_n(k_\text{norm})$$

A Transformer encodes the context $\xi$ and predicts the 17 coefficients $c_n$.

```
ChebyshevSpectralRegressor(ctx_dim=24, n_modes=16,
                           d_model=256, num_layers=4, num_heads=8)
  elem_proj: Linear(1, 256)               # lift each ctx scalar
  encoder:   TransformerEncoder × 4       # CLS + 24 ctx tokens
  head:      LayerNorm → Linear → GELU → Linear → 17 coefficients
  reconstr.: einsum("btn, bn -> bt", T_n(k_norm), c)
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE |
|---|---|---|---|
| 3.23 M | 0.022 | 2.53 | 3.33 |

Note: a truncated polynomial basis suffers from Gibbs ringing near
cusps. Avoid for curves with strong mode-crossing kinks.

**Train**: `python scripts/train_precision_curves.py`.
**Load**: `from models.chebyshev_spectral.model import ChebyshevSpectralRegressor`.
