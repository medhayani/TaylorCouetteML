# Multi-Mode Envelope SIREN ★

Custom architecture designed for the Floquet stability problem, where
the marginal curve $T_a(k, E)$ is the lower envelope of several
mode-specific threshold curves:

$$T_a(k, E) = \min_{m=1\ldots M}\, T_m(k, E)$$

The hard min naturally produces cusps wherever the argmin index changes,
which is exactly how a mode crossing manifests in a Floquet eigenvalue
problem.

```
MultiModeEnvelopeSIREN(ctx_dim=24, n_modes=4, hidden=192, depth=5)
  ctx_enc:   Linear(24, 128) → GELU → Linear → GELU → Linear     (128 latent)
  modes:     4 × [SineLayer + 5 × (FiLM + SineLayer) + Linear]
  offsets:   Parameter(4)                                          (symmetry breaking)
  forward:   stack 4 modes, hard-min, return env
```

| Params | val_MAE | MAE_phys (Input) | Kink-zone MAE | Gain vs legacy |
|---|---|---|---|---|
| 1.77 M | 0.011 | **0.549** | **0.785** | ×2.4 |

Best global / kink-zone trade-off observed across the 720 branches.

**Train**: `python scripts/train_precision_curves.py` (kink-weighted MSE, ×5).
**Load**:
```python
from models.envelope_siren.model import MultiModeEnvelopeSIREN
m = MultiModeEnvelopeSIREN(ctx_dim=24, n_modes=4)
ck = torch.load("checkpoints/envelope_siren/best.pt", weights_only=False)
m.load_state_dict(ck["state_dict"]); m.eval()
```
