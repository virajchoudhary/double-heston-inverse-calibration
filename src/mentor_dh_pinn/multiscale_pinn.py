"""Smooth maturity experts extending an existing regular pricing PINN.

Zero output initialization preserves the parent price function. The parent is
frozen, but coordinate derivatives through it remain enabled for the PDE.
The Black variance ansatz enforces payoff and call bounds, not convexity.
"""
import math

import torch
from torch import nn

from .regular_pinn_torch import TorchRegularVariancePINN


class TorchMultiscaleVariancePINN(TorchRegularVariancePINN):
    def __init__(self, base, expert_width=32, gated=True, residual_limit=0.05):
        nn.Module.__init__(self)
        if base.factors != 2 or expert_width < 1 or residual_limit <= 0:
            raise ValueError("Two factors, positive width and residual limit required")
        self.base = base
        self.base.requires_grad_(False)
        for key in ("factors", "width", "depth", "tau_min", "tau_max",
                    "x_half_width", "correction_limit"):
            setattr(self, key, getattr(base, key))
        self.gated = gated
        self.expert_width = expert_width
        self.residual_limit = residual_limit
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(26, expert_width), nn.Tanh(),
                          nn.Linear(expert_width, expert_width), nn.Tanh(),
                          nn.Linear(expert_width, 1)) for _ in range(3)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        self.double()

    def maturity_weights(self, tau):
        if not self.gated:
            return torch.ones_like(tau)[..., None].expand(*tau.shape, 3) / 3
        log_days = torch.log(365 * tau)
        first = torch.sigmoid((log_days - math.log(30)) / 0.5)
        second = torch.sigmoid((log_days - math.log(90)) / 0.5)
        return torch.stack([1-first, first-second, second], dim=-1)

    def correction(self, coords, structural):
        features = self.base.features(coords, structural)
        decay = torch.exp(-structural[..., :, 0] * coords[..., -1:])
        features = torch.cat([features, decay], dim=-1)
        experts = torch.cat([expert(features) for expert in self.experts], dim=-1)
        delta = self.residual_limit * (
            self.maturity_weights(coords[..., -1]) * torch.tanh(experts)).sum(-1)
        return self.base.correction(coords, structural) + delta
