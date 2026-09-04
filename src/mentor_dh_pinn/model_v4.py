"""Parameter-conditioned hard-constrained Double Heston PINN.

The V3 model is tied to one canonical parameter vector, so it cannot be used to
*calibrate*: the inverse problem needs a pricer that is differentiable in the
eight structural parameters as well as in the state.  This model conditions on
them, which turns it into a fast differentiable surrogate for the production
Fourier engine and makes bounded multi-start inverse calibration affordable.

The ansatz is V3's, unchanged.  ``vbar`` remains the exact Double Heston expected
integrated variance, so ``g = 0`` still reproduces the model exactly at zero
vol-of-vol, and the static no-arbitrage bounds, the terminal payoff and both
stock boundaries still hold by construction for every parameter vector.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .model_v3 import (G_LIMIT, Z_BOUND, black76_normalised,
                       mean_reversion_ratio)

PARAM_NAMES = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow",
               "kappa_fast", "theta_fast", "sigma_fast", "rho_fast")

# Box of the sealed 10,000-vector panel, padded outward so calibration never
# runs into an edge the network was not trained on.
PARAM_BOX = {
    "kappa_slow": (0.15, 2.80), "theta_slow": (0.014, 0.24),
    "sigma_slow": (0.045, 0.95), "rho_slow": (-0.90, 0.62),
    "kappa_fast": (1.30, 11.0), "theta_fast": (0.010, 0.19),
    "sigma_fast": (0.065, 1.60), "rho_fast": (-0.90, 0.72),
}
FEATURE_COUNT = 20


def _unit_log(value, lo, hi):
    return 2.0 * (torch.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)) - 1.0


def _unit(value, lo, hi):
    return 2.0 * (value - lo) / (hi - lo) - 1.0


class ConditionedDoubleHestonPINN(nn.Module):
    """``(x, v_slow, v_fast, tau, 8 structural parameters) -> implied total variance``."""

    def __init__(self, *, hidden_layers: int = 6, hidden_width: int = 192,
                 tau_low: float = 7.0 / 365.0, tau_high: float = 92.0 / 365.0,
                 variance_low: float = 5.0e-5, variance_high: float = 1.5,
                 x_half_width: float = 6.0):
        super().__init__()
        self.tau_low, self.tau_high = tau_low, tau_high
        self.variance_low, self.variance_high = variance_low, variance_high
        self.x_half_width = x_half_width
        layers: list[nn.Module] = []
        previous = FEATURE_COUNT
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, hidden_width), nn.Tanh()))
            previous = hidden_width
        head = nn.Linear(previous, 1)
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        layers.append(head)
        self.network = nn.Sequential(*layers)
        self.double()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def vbar(self, v_slow, v_fast, tau, p):
        slow = p["theta_slow"] + (v_slow - p["theta_slow"]) * mean_reversion_ratio(p["kappa_slow"] * tau)
        fast = p["theta_fast"] + (v_fast - p["theta_fast"]) * mean_reversion_ratio(p["kappa_fast"] * tau)
        return slow + fast

    def features(self, x, v_slow, v_fast, tau, p):
        vb = self.vbar(v_slow, v_fast, tau, p)
        z = x / torch.sqrt(tau * vb + x * x / (Z_BOUND ** 2) + 1e-14)
        root_tau = torch.sqrt(tau)
        # Dimensionless groups that actually shape the smile: mean reversion over
        # the horizon, vol-of-vol over the horizon, and the correlation-weighted
        # skew driver, one pair per factor.
        nu_s = p["sigma_slow"] * root_tau / torch.sqrt(vb + 1e-12)
        nu_f = p["sigma_fast"] * root_tau / torch.sqrt(vb + 1e-12)
        return torch.stack([
            x / self.x_half_width,
            z / Z_BOUND,
            torch.tanh(z / 1.5),
            _unit_log(tau, self.tau_low, self.tau_high),
            root_tau / math.sqrt(self.tau_high),
            _unit_log(torch.clamp(v_slow, min=1e-8), self.variance_low, self.variance_high),
            _unit_log(torch.clamp(v_fast, min=1e-8), self.variance_low, self.variance_high),
            _unit_log(p["kappa_slow"], *PARAM_BOX["kappa_slow"]),
            _unit_log(p["theta_slow"], *PARAM_BOX["theta_slow"]),
            _unit_log(p["sigma_slow"], *PARAM_BOX["sigma_slow"]),
            _unit(p["rho_slow"], *PARAM_BOX["rho_slow"]),
            _unit_log(p["kappa_fast"], *PARAM_BOX["kappa_fast"]),
            _unit_log(p["theta_fast"], *PARAM_BOX["theta_fast"]),
            _unit_log(p["sigma_fast"], *PARAM_BOX["sigma_fast"]),
            _unit(p["rho_fast"], *PARAM_BOX["rho_fast"]),
            torch.tanh(p["kappa_slow"] * tau),
            torch.tanh(p["kappa_fast"] * tau),
            torch.tanh(nu_s + nu_f),
            p["rho_slow"] * torch.tanh(nu_s) + p["rho_fast"] * torch.tanh(nu_f),
            torch.tanh(torch.log((v_slow + 1e-12) / (v_fast + 1e-12))),
        ], dim=-1)

    def log_total_variance(self, x, v_slow, v_fast, tau, p):
        vb = self.vbar(v_slow, v_fast, tau, p)
        g = G_LIMIT * torch.tanh(self.network(self.features(x, v_slow, v_fast, tau, p))[..., 0])
        return torch.log(tau) + torch.log(torch.clamp(vb, min=1e-12)) + 2.0 * g

    def price(self, spot, v_slow, v_fast, tau, strike, rate, carry, p) -> torch.Tensor:
        positive = spot > 0
        safe = torch.where(positive, spot, torch.ones_like(spot))
        x = torch.log(safe / strike) + (rate - carry) * tau
        ell = self.log_total_variance(x, v_slow, v_fast, tau, p)
        value = strike * torch.exp(-rate * tau) * black76_normalised(x, torch.exp(ell))
        return torch.where(positive, value, torch.zeros_like(value))


def broadcast_params(vector, n: int, dtype=torch.float64) -> dict:
    """Expand a canonical ten-vector (or 8 structural values) to a per-row dict."""
    v = np.asarray(vector, dtype=float)
    idx = {"kappa_slow": 0, "theta_slow": 1, "sigma_slow": 2, "rho_slow": 3,
           "kappa_fast": 5, "theta_fast": 6, "sigma_fast": 7, "rho_fast": 8}
    return {k: torch.full((n,), float(v[i]), dtype=dtype) for k, i in idx.items()}


def params_from_columns(cols: dict, dtype=torch.float64) -> dict:
    return {k: torch.tensor(np.asarray(cols[k], dtype=float), dtype=dtype) for k in PARAM_NAMES}
