# SIREN + FiLM

Sitzmann, Martel, Bergman, Lindell, Wetzstein. *Implicit Neural Representations with Periodic Activation Functions*. NeurIPS 2020.

MLP with sinusoidal activations ($\omega_0 = 30$) and FiLM conditioning
on the 24-dim context vector $\xi$.

```
SIRENRegressor(ctx_dim=24, hidden=256, depth=6)
  ctx_enc:   Linear(24, 128) → GELU → Linear → GELU → Linear     (128-dim latent)
  first:     SineLayer(1, 256, omega_0 = 30)
  body:      6 × [FiLM(128, 256) → SineLayer(256, 256, omega_0 = 30)]
  out:       Linear(256, 1)
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE |
|---|---|---|---|
| 0.83 M | 0.0098 | 0.897 | 1.30 |

**Train**: `python scripts/train_precision_curves.py` (trains all 4 precision models).
**Load**: `from models.siren.model import SIRENRegressor`.
