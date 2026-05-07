# DeepONet

Lu, Jin, Pang, Zhang, Karniadakis. *Learning nonlinear operators via DeepONet*. Nature Machine Intelligence 3, 218 (2021).

Branche encode le contexte $\xi$ vers $b \in \mathbb R^p$ ; trunk encode la
coordonnée spatiale $k$ vers $t \in \mathbb R^p$ via Fourier features
(K=16 bandes). Sortie : $T_a(k) = \langle b, t(k)\rangle + b_0$.

```
DeepONet(ctx_dim=24, latent_dim=128, fourier_bands=16, omega_max=16)
  branch:    MLP(24, 256, 256, 256, 256, 128)
  trunk:     MLP(33, 256, 256, 256, 256, 128)     ← input = [k, sin(2πw_i k), cos(2πw_i k)]
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE |
|---|---|---|---|
| 0.48 M | 0.022 | 1.34 | 2.10 |

**Entraîner** : `python scripts/train_precision_curves.py`.
**Charger** : `from models.deeponet.model import DeepONet`.
