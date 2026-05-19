"""Deep ensemble + Bayesian last layer for calibrated epistemic uncertainty.

p(T_a | E, c) = (1/M) sum_m  int  p_{theta_m}(T_a | E, c, W) q_{phi_m}(W) dW
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class BayesianLinear(nn.Module):
    """Mean-field Gaussian variational linear layer.

    q(W) = N(mu_W, diag(sigma_W^2)),  KL towards N(0, I).
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        self.mu_w = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.log_sigma_w = nn.Parameter(torch.full((out_features, in_features), -4.0))
        self.mu_b = nn.Parameter(torch.zeros(out_features))
        self.log_sigma_b = nn.Parameter(torch.full((out_features,), -4.0))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            eps_w = torch.randn_like(self.mu_w)
            eps_b = torch.randn_like(self.mu_b)
            W = self.mu_w + torch.exp(self.log_sigma_w) * eps_w
            b = self.mu_b + torch.exp(self.log_sigma_b) * eps_b
        else:
            W = self.mu_w
            b = self.mu_b
        return torch.nn.functional.linear(x, W, b)

    def kl_to_prior(self) -> torch.Tensor:
        var_w = torch.exp(2.0 * self.log_sigma_w)
        var_b = torch.exp(2.0 * self.log_sigma_b)
        prior_var = self.prior_sigma ** 2
        kl_w = 0.5 * ((self.mu_w ** 2 + var_w) / prior_var - 1.0
                      - 2.0 * self.log_sigma_w + math.log(prior_var)).sum()
        kl_b = 0.5 * ((self.mu_b ** 2 + var_b) / prior_var - 1.0
                      - 2.0 * self.log_sigma_b + math.log(prior_var)).sum()
        return kl_w + kl_b


class DeepEnsemble(nn.Module):
    """Holds M independent surrogates; aggregates predictions.

    Members are full surrogate modules with identical interfaces.
    """

    def __init__(self, members: List[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def __len__(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def predict(self, *args, num_samples_per_member: int = 1, **kwargs):
        """Returns (mean, sigma_ep, sigma_al) over members.

        Each member must implement .sample(*args, **kwargs) -> (B, T) tensor.
        """
        per_member_means = []
        per_member_vars = []
        for m in self.members:
            samples = []
            for _ in range(num_samples_per_member):
                samples.append(m.sample(*args, **kwargs))
            stack = torch.stack(samples, dim=0)              # (K, B, T)
            per_member_means.append(stack.mean(dim=0))       # (B, T)
            per_member_vars.append(stack.var(dim=0))         # (B, T)
        means = torch.stack(per_member_means, dim=0)         # (M, B, T)
        var_within = torch.stack(per_member_vars, dim=0).mean(dim=0)
        mean = means.mean(dim=0)
        var_between = means.var(dim=0)
        return mean, var_between.sqrt(), var_within.sqrt()
