"""CNP-TC - Conditional Neural Process for Taylor-Couette.

This architecture learns the *structure* of T_a(k) explicitly, rather than
treating each (k, Ta) point as an independent regression target.

Training procedure (the key novelty):
    For each branch, we randomly partition its 101 resampled points into
        - N_ctx context points  (k_c, Ta_c)        with N_ctx in {0, ..., 30}
        - the rest as target points (k_t, Ta_t)
    The model must predict ta_norm(k_t) given the descriptors AND the
    observed context. Because N_ctx is itself a random variable, the model
    learns *both* zero-shot prediction (N_ctx = 0, descriptors only) and
    few-shot interpolation (N_ctx > 0).

Inference modes:
    - Zero-shot (N_ctx = 0): pure descriptor-based prediction, drop-in
      replacement for the current models.
    - Few-shot (N_ctx > 0): user computes a handful of T_a values with the
      exact solver, feeds them as context, model fills in the rest much
      more accurately.

Smart input preparation:
    - log10E -> sinusoidal embedding at 32 log-spaced frequencies
    - k_norm -> 16 Fourier features (sin/cos at geometric frequencies)
    - 23 descriptors -> 23 distinct tokens with learned position embedding
    - Each context point (k_c, ta_c) -> 1 token built from both Fourier
      features of k_c and ta_c directly.

Output:
    sigmoid(Chebyshev(64) + Cosine(32))  in (0, 1).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def chebyshev_T(N: int, x: torch.Tensor) -> torch.Tensor:
    """Compute T_0..T_N at x in [-1, 1]. x has shape (B, T)."""
    B, T = x.shape
    out = torch.zeros(B, T, N + 1, device=x.device, dtype=x.dtype)
    out[..., 0] = 1.0
    if N >= 1: out[..., 1] = x
    for n in range(1, N):
        out[..., n + 1] = 2.0 * x * out[..., n] - out[..., n - 1]
    return out


def cosine_basis(M: int, x: torch.Tensor) -> torch.Tensor:
    """cos(m * pi * (x + 1) / 2) for m = 1..M. x in [-1, 1], shape (B, T)."""
    m = torch.arange(1, M + 1, device=x.device, dtype=x.dtype)
    phase = (x.unsqueeze(-1) + 1.0) * 0.5 * math.pi
    return torch.cos(phase * m)


def _sn(module, enable: bool):
    return nn.utils.spectral_norm(module) if enable else module


class CNP_TC(nn.Module):
    """Conditional Neural Process for Taylor-Couette stability curves."""

    def __init__(self,
                 ctx_dim: int = 23,
                 n_cheb: int = 64,
                 n_cos: int = 32,
                 d_model: int = 256,
                 n_enc_layers: int = 6,
                 n_dec_layers: int = 3,
                 n_heads: int = 8,
                 dropout: float = 0.05,
                 n_E_freq: int = 32,
                 n_k_freq: int = 16,
                 max_ctx_points: int = 32,
                 use_spectral_norm: bool = True):
        super().__init__()
        self.n_cheb = n_cheb
        self.n_cos = n_cos
        self.n_E_freq = n_E_freq
        self.n_k_freq = n_k_freq
        self.max_ctx_points = max_ctx_points

        # Frequency banks
        E_freqs = torch.logspace(0, math.log10(2.0 * math.pi * 16.0),
                                    steps=n_E_freq, base=10.0)
        k_freqs = torch.tensor([(2.0 ** i) * math.pi for i in range(n_k_freq)],
                                  dtype=torch.float32)
        self.register_buffer("E_freqs", E_freqs)
        self.register_buffer("k_freqs", k_freqs)

        # ---- Encoder side: descriptor tokens, E embed, anchor, context points ----
        # 23 descriptors -> 23 tokens
        self.desc_proj = _sn(nn.Linear(1, d_model), use_spectral_norm)
        self.desc_pos = nn.Parameter(torch.randn(ctx_dim, d_model) * 0.02)
        # E -> 1 token from sinusoidal embedding
        self.E_proj = _sn(nn.Linear(2 * n_E_freq, d_model), use_spectral_norm)
        # 4 anchors (k_left, k_right, log10 Ta_min, log10 Ta_max)
        self.anchor_proj = _sn(nn.Linear(1, d_model), use_spectral_norm)
        self.anchor_pos = nn.Parameter(torch.randn(4, d_model) * 0.02)
        # Each context point: features = Fourier(k_c)[2*n_k_freq+1] + ta_c[1] = 2*n_k_freq+2
        self.ctx_pt_proj = _sn(nn.Linear(2 * n_k_freq + 2, d_model), use_spectral_norm)
        # Distinct token-type embeddings so transformer can tell them apart
        self.type_embed = nn.Parameter(torch.randn(4, d_model) * 0.02)
        # types: 0 = descriptor, 1 = E, 2 = anchor, 3 = context point
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Transformer encoder over the bag of context tokens ----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ---- Query side: build query embeddings, then cross-attend to encoder output ----
        self.q_proj = _sn(nn.Linear(2 * n_k_freq + 1, d_model), use_spectral_norm)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # ---- Output: pool encoder output to single d_model, predict spectral coefs ----
        self.pool_norm = nn.LayerNorm(d_model)
        self.head_cheb = nn.Sequential(
            _sn(nn.Linear(d_model, d_model), use_spectral_norm), nn.GELU(),
            _sn(nn.Linear(d_model, n_cheb + 1), use_spectral_norm),
        )
        self.head_cos = nn.Sequential(
            _sn(nn.Linear(d_model, d_model // 2), use_spectral_norm), nn.GELU(),
            _sn(nn.Linear(d_model // 2, n_cos), use_spectral_norm),
        )

        # Spectral decay
        cheb_decay = torch.tensor([(n + 1) ** 2 for n in range(n_cheb + 1)],
                                     dtype=torch.float32)
        cos_decay = torch.tensor([(m + 1) ** 2 for m in range(n_cos)],
                                    dtype=torch.float32)
        self.register_buffer("cheb_decay", cheb_decay)
        self.register_buffer("cos_decay", cos_decay)

    def _E_embed(self, log10E: torch.Tensor) -> torch.Tensor:
        phase = log10E.unsqueeze(-1) * self.E_freqs.unsqueeze(0)
        return self.E_proj(torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1))

    def _k_features(self, k_norm: torch.Tensor) -> torch.Tensor:
        """(B, T) k_norm -> (B, T, 2 * n_k_freq + 1) Fourier features."""
        phase = k_norm.unsqueeze(-1) * self.k_freqs
        return torch.cat([k_norm.unsqueeze(-1),
                            torch.sin(phase), torch.cos(phase)], dim=-1)

    def _encode_context(self, ctx: torch.Tensor, anchors: torch.Tensor,
                          log10E: torch.Tensor,
                          ctx_k: torch.Tensor, ctx_ta: torch.Tensor,
                          ctx_mask: torch.Tensor) -> tuple:
        """Build the encoder token sequence and run it through the encoder.

        Args:
            ctx:      (B, 23) descriptors
            anchors:  (B, 4)  anchor scalars
            log10E:   (B,)    log10E
            ctx_k:    (B, N)  context k values in [-1, 1] (padded)
            ctx_ta:   (B, N)  context ta_norm values (padded)
            ctx_mask: (B, N)  1 if context point is real, 0 if padding
        Returns:
            (encoder_output, encoder_pad_mask, pooled_output)
        """
        B = ctx.size(0)
        # Descriptor tokens
        desc_tok = self.desc_proj(ctx.unsqueeze(-1)) + self.desc_pos.unsqueeze(0)
        desc_tok = desc_tok + self.type_embed[0]
        # E token
        E_tok = self._E_embed(log10E).unsqueeze(1) + self.type_embed[1]
        # Anchor tokens
        anc_tok = (self.anchor_proj(anchors.unsqueeze(-1))
                     + self.anchor_pos.unsqueeze(0)
                     + self.type_embed[2])
        # Context point tokens
        k_feat = self._k_features(ctx_k)                              # (B, N, 2*nk+1)
        pt_feat = torch.cat([k_feat, ctx_ta.unsqueeze(-1)], dim=-1)    # (B, N, 2*nk+2)
        pt_tok = self.ctx_pt_proj(pt_feat) + self.type_embed[3]        # (B, N, d_model)
        # CLS
        cls = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls, desc_tok, E_tok, anc_tok, pt_tok], dim=1)
        # Padding mask (True where padded)
        n_fixed = 1 + ctx.size(1) + 1 + 4
        ctx_pad = (ctx_mask < 0.5)                                    # (B, N)
        pad_mask = torch.cat([torch.zeros(B, n_fixed, dtype=torch.bool, device=tokens.device),
                                ctx_pad], dim=1)
        h = self.encoder(tokens, src_key_padding_mask=pad_mask)
        return h, pad_mask, self.pool_norm(h[:, 0])

    def forward(self, k_query: torch.Tensor, ctx: torch.Tensor,
                anchors: torch.Tensor, log10E: torch.Tensor,
                ctx_k: torch.Tensor, ctx_ta: torch.Tensor,
                ctx_mask: torch.Tensor) -> torch.Tensor:
        """Predict ta_norm(k_query) given descriptors + optional context points."""
        h_enc, pad_mask, pooled = self._encode_context(
            ctx, anchors, log10E, ctx_k, ctx_ta, ctx_mask)
        # Query
        q_feat = self._k_features(k_query)
        q_tok = self.q_proj(q_feat)                                   # (B, T, d_model)
        # Cross-attend queries to encoder output
        h_dec = self.decoder(tgt=q_tok, memory=h_enc, memory_key_padding_mask=pad_mask)
        # Combine pooled context with each query token
        z = h_dec + pooled.unsqueeze(1)
        c_cheb = self.head_cheb(z)                                    # (B, T, n_cheb+1)
        c_cos = self.head_cos(z)                                      # (B, T, n_cos)

        Tn = chebyshev_T(self.n_cheb, k_query)                         # (B, T, n_cheb+1)
        Cm = cosine_basis(self.n_cos, k_query)                         # (B, T, n_cos)
        z_main = (Tn * c_cheb).sum(dim=-1) + (Cm * c_cos).sum(dim=-1)
        return torch.sigmoid(z_main)

    def spectral_penalty(self, ctx, anchors, log10E,
                          ctx_k, ctx_ta, ctx_mask, k_query):
        h_enc, pad_mask, pooled = self._encode_context(
            ctx, anchors, log10E, ctx_k, ctx_ta, ctx_mask)
        q_tok = self.q_proj(self._k_features(k_query))
        h_dec = self.decoder(tgt=q_tok, memory=h_enc, memory_key_padding_mask=pad_mask)
        z = h_dec + pooled.unsqueeze(1)
        c_cheb = self.head_cheb(z); c_cos = self.head_cos(z)
        L = (c_cheb ** 2 * self.cheb_decay[None, None, :]).mean()
        L = L + (c_cos ** 2 * self.cos_decay[None, None, :]).mean()
        return L
