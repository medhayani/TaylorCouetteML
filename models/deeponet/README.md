# DeepONet

Lu, Jin, Pang, Zhang, Karniadakis. *Learning nonlinear operators via DeepONet*. Nature Machine Intelligence 3, 218 (2021).

The branch network encodes the context $\xi$ to $b \in \mathbb{R}^p$;
the trunk network encodes the spatial coordinate $k$ to $t(k) \in \mathbb{R}^p$
through Fourier features (K = 16 frequency bands).
Output: $T_a(k) = \langle b, t(k)\rangle + b_0$.

```
DeepONet(ctx_dim=24, latent_dim=128, fourier_bands=16, omega_max=16)
  branch:    MLP(24, 256, 256, 256, 256, 128)
  trunk:     MLP(33, 256, 256, 256, 256, 128)     ← input = [k, sin(2π w_i k), cos(2π w_i k)]
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE |
|---|---|---|---|
| 0.48 M | 0.022 | 1.34 | 2.10 |

**Train**: `python scripts/train_precision_curves.py`.
**Load**: `from models.deeponet.model import DeepONet`.
