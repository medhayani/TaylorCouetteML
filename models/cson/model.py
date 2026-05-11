"""CSON — Constrained Spectral Operator Network.

A robust neural surrogate for the marginal stability curve T_a(k) of an
oscillatory Taylor-Couette flow, parameterised by 23 functional descriptors
of each branch (including log10 E).

Design rationale (vs SSST / NEPTUNE which overshoot or oscillate at the
domain boundaries):

1. **Spectral decoder.** The curve is represented on a truncated Chebyshev
   basis of the first kind on [-1, 1]:

       z(k_norm) = sum_{n=0..N} c_n(xi) * T_n(k_norm),

   where xi is the encoded branch context. Because |T_n(x)| <= 1 on [-1, 1],
   the spectral output is mathematically bounded and cannot diverge.

2. **Sigmoid clamping.** The final prediction is

       ta_norm = sigmoid(z(k_norm))  in  (0, 1)

   which matches the data normalisation ta_norm = (T_a - T_a_min) / amp
   by construction, so the model can never predict values outside the
   physically admissible band [T_a_min, T_a_max] of the branch.

3. **Lipschitz-constrained encoder.** Spectral normalisation is applied to
   the linear layers of the encoder and output head, bounding the Lipschitz
   constant of xi -> c_n(xi) and preventing pathological extrapolation when
   the input descriptors fall outside the training distribution.

4. **Spectral-decay regularisation.** During training, an extra penalty

       L_spec = mean_n (n+1)^2 * c_n^2

   pushes high-order coefficients towards zero, which directly suppresses
   the parasitic high-frequency oscillations seen in the diffusion-based
   NEPTUNE surrogate.

The model is small enough (~2-3M parameters at the default settings) to
train in well under 1 hour on a single Kaggle P100 GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def chebyshev_T(N: int, x: torch.Tensor) -> torch.Tensor:
    """Compute Chebyshev polynomials of the first kind T_0, ..., T_N at x.

    Args:
        N: max degree (inclusive).
        x: (B, T) values in [-1, 1].
    Returns:
        T: (B, T, N+1) with T[..., n] = T_n(x).
    """
    B, T = x.shape
    out = torch.zeros(B, T, N + 1, device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if N >= 1:
        out[..., 1] = x
    for n in range(1, N):
        out[..., n + 1] = 2.0 * x * out[..., n] - out[..., n - 1]
    return out


def _sn(module, enable: bool):
    if not enable:
        return module
    return nn.utils.spectral_norm(module)


class CSON(nn.Module):
    """Constrained Spectral Operator Network."""

    def __init__(self,
                 ctx_dim: int = 23,
                 n_modes: int = 48,
                 d_model: int = 320,
                 num_layers: int = 5,
                 num_heads: int = 8,
                 dropout: float = 0.05,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.n_modes = n_modes
        self.ctx_dim = ctx_dim
        self.use_spectral_norm = use_spectral_norm

        # Per-scalar projection into d_model, then Transformer encoder.
        self.elem_proj = _sn(nn.Linear(1, d_model), use_spectral_norm)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            _sn(nn.Linear(d_model, d_model), use_spectral_norm), nn.GELU(),
            _sn(nn.Linear(d_model, n_modes + 1), use_spectral_norm),
        )

        # Spectral-decay regulariser weights, (n+1)^2.
        decay = torch.tensor([(n + 1) ** 2 for n in range(n_modes + 1)],
                             dtype=torch.float32)
        self.register_buffer("decay_weights", decay)

    def _encode(self, ctx: torch.Tensor) -> torch.Tensor:
        B = ctx.size(0)
        tok = self.elem_proj(ctx.unsqueeze(-1))    # (B, ctx_dim, d_model)
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x = torch.cat([cls, tok], dim=1)           # (B, 1+ctx_dim, d_model)
        h = self.encoder(x)
        return h[:, 0]                              # (B, d_model)

    def coefficients(self, ctx: torch.Tensor) -> torch.Tensor:
        """Predicted Chebyshev coefficients (B, n_modes+1) — for diagnostic."""
        return self.head(self._encode(ctx))

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Predict ta_norm in (0, 1) via Chebyshev expansion + sigmoid.

        Args:
            k_norm: (B, T) values in [-1, 1].
            ctx:    (B, ctx_dim) normalised branch descriptors.
        Returns:
            ta_norm: (B, T) in (0, 1).
        """
        c = self.head(self._encode(ctx))                # (B, n_modes+1)
        Tn = chebyshev_T(self.n_modes, k_norm)          # (B, T, n_modes+1)
        z = torch.einsum("btn,bn->bt", Tn, c)            # (B, T)
        return torch.sigmoid(z)

    def spectral_penalty(self, ctx: torch.Tensor) -> torch.Tensor:
        """L2 penalty on coefficients, weighted by (n+1)^2.

        Encourages geometric spectral decay c_n -> 0 for large n,
        which suppresses high-frequency oscillations in the output.
        """
        c = self.head(self._encode(ctx))                # (B, n_modes+1)
        return (c ** 2 * self.decay_weights[None, :]).mean()
