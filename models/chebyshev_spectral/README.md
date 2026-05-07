# Chebyshev Spectral Regressor

Décomposition de chaque courbe $T_a(k)$ sur la base des polynômes de
Chebyshev de première espèce $T_n$ tronquée à $N=16$ :

$$T_a(k_\text{norm}) = \sum_{n=0}^{16} c_n(\xi) \, T_n(k_\text{norm})$$

Un Transformer encode le contexte $\xi$ et produit les 17 coefficients $c_n$.

```
ChebyshevSpectralRegressor(ctx_dim=24, n_modes=16,
                           d_model=256, num_layers=4, num_heads=8)
  elem_proj: Linear(1, 256)               # lift each ctx scalar
  encoder:   TransformerEncoder × 4       # CLS + 24 ctx tokens
  head:      LayerNorm→Linear→GELU→Linear → 17 coefficients
  reconstr.: einsum("btn,bn->bt", T_n(k_norm), c)
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE |
|---|---|---|---|
| 3.23 M | 0.022 | 2.53 | 3.33 |

Note : la base polynomiale tronquée souffre de Gibbs ringing aux kinks.
À éviter pour des courbes avec cusps marqués.

**Entraîner** : `python scripts/train_precision_curves.py`.
**Charger** : `from models.chebyshev_spectral.model import ChebyshevSpectralRegressor`.
