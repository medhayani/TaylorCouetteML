"""STAR — Spectral Transformer with Anchored Residuals.

A single-architecture, robust neural surrogate for the marginal stability
curve T_a(k) of an oscillatory Taylor-Couette flow, designed to outperform
the previous 7-architecture ensemble by *preparing inputs in a way that
makes the prediction task fundamentally easier*.

The three input-preparation ideas at the core of STAR are:

1. **Sinusoidal positional embedding of log10(E).** Instead of feeding
   log10E as a raw scalar (forcing the encoder to learn its own basis to
   separate four decades), we project log10E onto 32 log-spaced sine /
   cosine frequencies. Adjacent decades of E become orthogonal directions
   in the embedding space.

2. **Fourier features of k_norm.** k_norm in [-1, 1] is expanded into
   16 sinusoidal features, exposing high-frequency structure to the
   network at the input level instead of asking the model to synthesise
   it through layers.

3. **Anchor tokens.** Four physical boundary values of each branch
   (k_left, k_right, T_a_min, T_a_max) are injected as their own
   tokens. The model can attend to these anchors when shaping the
   curve, which encourages it to respect the branch endpoints.

The architecture is a cross-attention Transformer between a *query
token* (containing the Fourier-encoded k_norm) and a *context bag* of
tokens (descriptors, E-embedding, anchors). The decoder is dual: a
Chebyshev expansion handles the smooth backbone while a cosine
expansion captures sharper cusps. A sigmoid clamps the output to (0, 1)
so the prediction can never violate the branch bounds.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def chebyshev_T(N: int, x: torch.Tensor) -> torch.Tensor:
    """Compute Chebyshev polynomials of the first kind T_0..T_N at x.

    Args:
        N: max degree (inclusive).
        x: (B, T) values in [-1, 1].
    Returns:
        (B, T, N+1) tensor with T[..., n] = T_n(x).
    """
    B, T = x.shape
    out = torch.zeros(B, T, N + 1, device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if N >= 1:
        out[..., 1] = x
    for n in range(1, N):
        out[..., n + 1] = 2.0 * x * out[..., n] - out[..., n - 1]
    return out


def cosine_basis(M: int, x: torch.Tensor) -> torch.Tensor:
    """Half-cosine basis cos(m * pi * (x + 1) / 2) for m = 1..M.

    Captures sharp transitions / cusps that the Chebyshev backbone
    smooths out.

    Args:
        M: number of cosine modes.
        x: (B, T) values in [-1, 1].
    Returns:
        (B, T, M) tensor.
    """
    B, T = x.shape
    m = torch.arange(1, M + 1, device=x.device, dtype=x.dtype)
    phase = (x.unsqueeze(-1) + 1.0) * 0.5 * math.pi
    return torch.cos(phase * m)


def _sn(module, enable: bool):
    return nn.utils.spectral_norm(module) if enable else module


class STAR(nn.Module):
    """Spectral Transformer with Anchored Residuals."""

    def __init__(self,
                 ctx_dim: int = 23,
                 n_cheb: int = 64,
                 n_cos: int = 32,
                 d_model: int = 384,
                 n_layers: int = 8,
                 n_heads: int = 12,
                 dropout: float = 0.05,
                 n_E_freq: int = 32,
                 n_k_freq: int = 16,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.n_cheb = n_cheb
        self.n_cos = n_cos
        self.n_E_freq = n_E_freq
        self.n_k_freq = n_k_freq

        # ---- Frequency banks (log-spaced for E, geometric for k) ----
        # E spans about 4 decades; use frequencies log-spaced.
        log_freqs = torch.logspace(0, math.log10(2.0 * math.pi * 16.0),
                                       steps=n_E_freq, base=10.0)
        self.register_buffer("E_freqs", log_freqs)
        k_freqs = torch.tensor([(2.0 ** i) * math.pi for i in range(n_k_freq)],
                                 dtype=torch.float32)
        self.register_buffer("k_freqs", k_freqs)

        # ---- Per-input embeddings into d_model ----
        # 23 descriptors each become an independent token.
        self.desc_proj = _sn(nn.Linear(1, d_model), use_spectral_norm)
        # Each descriptor has its own learned position embedding (helps
        # the attention layer distinguish them).
        self.desc_pos = nn.Parameter(torch.randn(ctx_dim, d_model) * 0.02)

        # E embedding: 32 sin + 32 cos = 64 -> project to d_model.
        self.E_embed_proj = _sn(nn.Linear(2 * n_E_freq, d_model), use_spectral_norm)
        # Anchor tokens (4 scalars, each becomes a token).
        self.anchor_proj = _sn(nn.Linear(1, d_model), use_spectral_norm)
        self.anchor_pos = nn.Parameter(torch.randn(4, d_model) * 0.02)

        # Query token: built from Fourier features of k_norm
        # (sin + cos of 16 freqs = 32) projected to d_model.
        self.k_query_proj = _sn(nn.Linear(2 * n_k_freq + 1, d_model), use_spectral_norm)

        # ---- Transformer cross-attention encoder ----
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Spectral output head ----
        self.head_cheb = nn.Sequential(
            nn.LayerNorm(d_model),
            _sn(nn.Linear(d_model, d_model), use_spectral_norm), nn.GELU(),
            _sn(nn.Linear(d_model, n_cheb + 1), use_spectral_norm),
        )
        self.head_cos = nn.Sequential(
            nn.LayerNorm(d_model),
            _sn(nn.Linear(d_model, d_model // 2), use_spectral_norm), nn.GELU(),
            _sn(nn.Linear(d_model // 2, n_cos), use_spectral_norm),
        )

        # Decay weights for spectral penalty.
        cheb_decay = torch.tensor([(n + 1) ** 2 for n in range(n_cheb + 1)],
                                     dtype=torch.float32)
        cos_decay = torch.tensor([(m + 1) ** 2 for m in range(n_cos)],
                                    dtype=torch.float32)
        self.register_buffer("cheb_decay", cheb_decay)
        self.register_buffer("cos_decay", cos_decay)

    def _E_embed(self, log10E: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional embedding of log10E.

        Args:
            log10E: (B,) scalar value per sample.
        Returns:
            (B, d_model) embedded vector.
        """
        phase = log10E.unsqueeze(-1) * self.E_freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.E_embed_proj(emb)

    def _k_query(self, k_norm: torch.Tensor) -> torch.Tensor:
        """Build the per-point query embedding from Fourier features of k.

        Args:
            k_norm: (B, T) values in [-1, 1].
        Returns:
            (B, T, d_model) query embeddings.
        """
        phase = k_norm.unsqueeze(-1) * self.k_freqs
        feats = torch.cat([k_norm.unsqueeze(-1),
                            torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.k_query_proj(feats)

    def _build_context(self, ctx: torch.Tensor, anchors: torch.Tensor,
                        log10E: torch.Tensor) -> torch.Tensor:
        """Build the full context token sequence.

        Tokens (per sample): [CLS, 23 descriptors, 1 E_embed, 4 anchors] = 29 tokens.
        """
        B = ctx.size(0)
        # Descriptors -> 23 tokens
        desc_tok = self.desc_proj(ctx.unsqueeze(-1)) + self.desc_pos.unsqueeze(0)
        # Anchors -> 4 tokens
        anc_tok = self.anchor_proj(anchors.unsqueeze(-1)) + self.anchor_pos.unsqueeze(0)
        # E -> 1 token
        E_tok = self._E_embed(log10E).unsqueeze(1)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, desc_tok, E_tok, anc_tok], dim=1)

    def forward(self, k_norm: torch.Tensor, ctx: torch.Tensor,
                anchors: torch.Tensor, log10E: torch.Tensor) -> torch.Tensor:
        """Predict ta_norm in (0, 1).

        Args:
            k_norm:  (B, T) values in [-1, 1].
            ctx:     (B, 23) normalised descriptors.
            anchors: (B, 4) normalised [k_left, k_right, Ta_min, Ta_max].
            log10E:  (B,) log10(E) for sinusoidal embedding.
        Returns:
            ta_norm: (B, T) in (0, 1).
        """
        ctx_tokens = self._build_context(ctx, anchors, log10E)
        h = self.encoder(ctx_tokens)
        z_ctx = h[:, 0]                                  # (B, d_model)

        c_cheb = self.head_cheb(z_ctx)                    # (B, n_cheb+1)
        c_cos = self.head_cos(z_ctx)                      # (B, n_cos)

        Tn = chebyshev_T(self.n_cheb, k_norm)              # (B, T, n_cheb+1)
        Cm = cosine_basis(self.n_cos, k_norm)              # (B, T, n_cos)
        z = torch.einsum("btn,bn->bt", Tn, c_cheb) + torch.einsum("btm,bm->bt", Cm, c_cos)
        return torch.sigmoid(z)

    def coefficients(self, ctx: torch.Tensor, anchors: torch.Tensor,
                      log10E: torch.Tensor):
        ctx_tokens = self._build_context(ctx, anchors, log10E)
        h = self.encoder(ctx_tokens)
        z_ctx = h[:, 0]
        return self.head_cheb(z_ctx), self.head_cos(z_ctx)

    def spectral_penalty(self, ctx: torch.Tensor, anchors: torch.Tensor,
                          log10E: torch.Tensor) -> torch.Tensor:
        c_cheb, c_cos = self.coefficients(ctx, anchors, log10E)
        L = (c_cheb ** 2 * self.cheb_decay[None, :]).mean()
        L = L + (c_cos ** 2 * self.cos_decay[None, :]).mean()
        return L
