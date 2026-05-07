"""Fourier Neural Operator core blocks.

v_{l+1}(s) = sigma( W v_l(s) + F^{-1} [ R_theta * F[v_l] ](s) ).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv1D(nn.Module):
    """Truncated spectral convolution: F -> multiply by R_theta (k <= k_max) -> F^{-1}."""

    def __init__(self, in_channels: int, out_channels: int, num_modes: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, num_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                # (B, C, T//2+1)
        out_ft = torch.zeros(
            B, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device
        )
        m = min(self.num_modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bcm,com->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=T, dim=-1)


class FNOBlock(nn.Module):
    """One FNO block with residual + spectral conv + 1x1 conv + GELU."""

    def __init__(self, channels: int, num_modes: int = 16, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1D(channels, channels, num_modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral(x) + self.pointwise(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.drop(h)
        return x + h
