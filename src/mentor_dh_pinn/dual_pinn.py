"""Two specialist inverse networks for the ten Double Heston parameters.

The split is measured, not assumed. Relative price sensitivity
``|d log C / d log p|`` was evaluated per parameter across maturity slices from
7 days to 2 years. It separates cleanly, but *not* along the slow/fast axis:

    short end (7-30d)   v0_slow 0.70, v0_fast 0.53, and BOTH factors'
                        sigma and rho -- every one of them peaks at 7 days
    long end (180-730d) theta_fast 0.13, theta_slow 0.09, kappa_slow,
                        kappa_fast -- every one of them peaks at 180-730 days

So the natural decomposition is **state and shape** against **level and speed**.
Both factors' vol-of-vol and correlation are short-dated objects; both factors'
long-run level and mean-reversion speed are long-dated objects.

Constraint coupling forces awareness. The Feller condition ties
``sigma_i`` to ``kappa_i theta_i``, so a short-end network cannot emit a valid
``sigma`` without knowing the long end. The short network therefore emits the
dimensionless Feller ratio ``eta = sigma / sqrt(2 kappa theta)``, which carries
no constraint of its own, and ``sigma`` is formed at combination time. With
``cross_conditioning`` off this is the independent design; with it on each
network also sees the other's current estimate and the pair is iterated to a
fixed point.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .model_v4 import PARAM_BOX

SHORT_DAYS = (30.0, 60.0, 90.0)
LONG_DAYS = (180.0, 365.0)
N_STRIKES = 9
SHORT_QUOTES = len(SHORT_DAYS) * N_STRIKES
LONG_QUOTES = len(LONG_DAYS) * N_STRIKES
CANONICAL = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
             "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast")
V0_BOX = (0.01, 0.60)
ETA_MAX = 0.995
RHO_RADIUS = 0.97


def _logbox(u, lo, hi):
    return lo * torch.exp(torch.sigmoid(u) * math.log(hi / lo))


def _mlp(inp, out, width, depth):
    layers, prev = [], inp
    for _ in range(depth):
        layers += [nn.Linear(prev, width), nn.SiLU()]
        prev = width
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


def encode_surface(prices: torch.Tensor) -> torch.Tensor:
    """Scale-free surface features: log price, plus per-expiry level and slope."""
    x = torch.log(torch.clamp(prices, min=1e-10))
    return x


class ShortEndNet(nn.Module):
    """Sees short expiries; emits state and shape: v0_slow, v0_fast, eta_s, eta_f, rho_s, rho_f."""

    def __init__(self, width=256, depth=5, cross=True):
        super().__init__()
        self.cross = cross
        extra = 4 if cross else 0        # theta_s, theta_f, kappa_s, kappa_f
        self.net = _mlp(SHORT_QUOTES + 1 + extra, 6, width, depth)
        self.double()

    def forward(self, short_prices, noise_level, long_params=None):
        f = [encode_surface(short_prices), noise_level.reshape(-1, 1)]
        if self.cross:
            if long_params is None:
                long_params = torch.zeros(short_prices.shape[0], 4, dtype=torch.float64)
            f.append(long_params)
        u = self.net(torch.cat(f, dim=1))
        v0_slow = _logbox(u[:, 0], *V0_BOX)
        v0_fast = _logbox(u[:, 1], *V0_BOX)
        eta_slow = ETA_MAX * torch.sigmoid(u[:, 2])
        eta_fast = ETA_MAX * torch.sigmoid(u[:, 3])
        rb, fb = PARAM_BOX["rho_slow"], PARAM_BOX["rho_fast"]
        rho_slow = 0.5 * (rb[0] + rb[1]) + 0.5 * (rb[1] - rb[0]) * torch.tanh(u[:, 4])
        rho_fast = 0.5 * (fb[0] + fb[1]) + 0.5 * (fb[1] - fb[0]) * torch.tanh(u[:, 5])
        radius = torch.sqrt(rho_slow ** 2 + rho_fast ** 2) + 1e-12
        scale = torch.clamp(RHO_RADIUS / radius, max=1.0)
        return {"v0_slow": v0_slow, "v0_fast": v0_fast,
                "eta_slow": eta_slow, "eta_fast": eta_fast,
                "rho_slow": rho_slow * scale, "rho_fast": rho_fast * scale}


class LongEndNet(nn.Module):
    """Sees long expiries; emits level and speed: theta_slow, theta_fast, kappa_slow, kappa_fast."""

    def __init__(self, width=256, depth=5, cross=True):
        super().__init__()
        self.cross = cross
        extra = 6 if cross else 0        # v0_s, v0_f, eta_s, eta_f, rho_s, rho_f
        self.net = _mlp(LONG_QUOTES + 1 + extra, 4, width, depth)
        self.double()

    def forward(self, long_prices, noise_level, short_params=None):
        f = [encode_surface(long_prices), noise_level.reshape(-1, 1)]
        if self.cross:
            if short_params is None:
                short_params = torch.zeros(long_prices.shape[0], 6, dtype=torch.float64)
            f.append(short_params)
        u = self.net(torch.cat(f, dim=1))
        theta_slow = _logbox(u[:, 0], *PARAM_BOX["theta_slow"])
        theta_fast = _logbox(u[:, 1], *PARAM_BOX["theta_fast"])
        kappa_slow = _logbox(u[:, 2], *PARAM_BOX["kappa_slow"])
        # ordering by construction: kappa_fast strictly above kappa_slow
        kappa_fast = kappa_slow + (PARAM_BOX["kappa_fast"][1] - kappa_slow) * torch.sigmoid(u[:, 3])
        return {"theta_slow": theta_slow, "theta_fast": theta_fast,
                "kappa_slow": kappa_slow, "kappa_fast": kappa_fast}


def combine(short: dict, long: dict) -> dict:
    """Form the canonical ten-vector. Feller holds because sigma = eta sqrt(2 kappa theta)."""
    out = dict(long)
    out["v0_slow"], out["v0_fast"] = short["v0_slow"], short["v0_fast"]
    out["rho_slow"], out["rho_fast"] = short["rho_slow"], short["rho_fast"]
    out["sigma_slow"] = short["eta_slow"] * torch.sqrt(2.0 * long["kappa_slow"] * long["theta_slow"])
    out["sigma_fast"] = short["eta_fast"] * torch.sqrt(2.0 * long["kappa_fast"] * long["theta_fast"])
    return out


def stack_short(d: dict) -> torch.Tensor:
    return torch.stack([d["v0_slow"], d["v0_fast"], d["eta_slow"], d["eta_fast"],
                        d["rho_slow"], d["rho_fast"]], dim=1)


def stack_long(d: dict) -> torch.Tensor:
    return torch.stack([d["theta_slow"], d["theta_fast"],
                        d["kappa_slow"], d["kappa_fast"]], dim=1)


def stack_canonical(d: dict) -> torch.Tensor:
    return torch.stack([d[k] for k in CANONICAL], dim=1)


class DualPINN(nn.Module):
    """The pair, run either independently (one pass) or to a fixed point."""

    def __init__(self, width=256, depth=5, cross=True, sweeps=3):
        super().__init__()
        self.cross, self.sweeps = cross, sweeps
        self.short = ShortEndNet(width, depth, cross)
        self.long = LongEndNet(width, depth, cross)
        self.double()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, short_prices, long_prices, noise_level):
        s = self.short(short_prices, noise_level, None)
        l = self.long(long_prices, noise_level, stack_short(s) if self.cross else None)
        if self.cross:
            for _ in range(self.sweeps - 1):
                s = self.short(short_prices, noise_level, stack_long(l))
                l = self.long(long_prices, noise_level, stack_short(s))
        return combine(s, l), s, l
