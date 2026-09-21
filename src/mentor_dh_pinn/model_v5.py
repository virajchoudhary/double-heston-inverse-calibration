"""V5 conditioned Double Heston PINN: richer conditioning plus free structural regularisers.

Two changes from V4.

**Features.** V4 collapsed the two factors into combined drivers (one vol-of-vol
group, one skew group). The smile actually responds to each factor separately --
the slow factor sets the level and the long end, the fast factor the short-dated
curvature -- so the drivers are now carried per factor, together with the
variance fraction that says how much of the current total each contributes.

**Regularisation.** Enforcing the PDE already requires ``C_S``, ``C_SS``,
``C_v_slow`` and ``C_v_fast`` by autograd, so four structural properties of a
call price can be penalised at no extra cost:

    C_S in [0, e^{-q tau}]        delta bounds
    C_SS >= 0                     convexity in spot, i.e. no butterfly arbitrage
    C_v_slow >= 0, C_v_fast >= 0  a call cannot fall when variance rises

These are true of the model, not merely desirable, so penalising their violation
is a physics regulariser rather than a bias.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model_v3 import G_LIMIT, Z_BOUND, black76_normalised, mean_reversion_ratio
from .model_v4 import PARAM_BOX, PARAM_NAMES

FEATURE_COUNT = 27


def _unit_log(v, lo, hi):
    return 2.0 * (torch.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) - 1.0


def _unit(v, lo, hi):
    return 2.0 * (v - lo) / (hi - lo) - 1.0


class ConditionedDoubleHestonPINNV5(nn.Module):
    def __init__(self, *, hidden_layers: int = 6, hidden_width: int = 256,
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
        vb_safe = torch.clamp(vb, min=1e-12)
        z = x / torch.sqrt(tau * vb + x * x / (Z_BOUND ** 2) + 1e-14)
        rt = torch.sqrt(tau)
        nu_s = torch.tanh(p["sigma_slow"] * rt / torch.sqrt(vb_safe))
        nu_f = torch.tanh(p["sigma_fast"] * rt / torch.sqrt(vb_safe))
        eta_s = p["sigma_slow"] / torch.sqrt(2.0 * p["kappa_slow"] * p["theta_slow"])
        eta_f = p["sigma_fast"] / torch.sqrt(2.0 * p["kappa_fast"] * p["theta_fast"])
        frac = v_slow / (v_slow + v_fast + 1e-14)
        zt = torch.tanh(z / 1.5)
        return torch.stack([
            x / self.x_half_width, z / Z_BOUND, zt, torch.tanh(z / 4.0), zt * zt,
            _unit_log(tau, self.tau_low, self.tau_high), rt / math.sqrt(self.tau_high),
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
            torch.tanh(p["kappa_slow"] * tau), torch.tanh(p["kappa_fast"] * tau),
            nu_s, nu_f,                                   # per-factor vol-of-vol over the horizon
            p["rho_slow"] * nu_s, p["rho_fast"] * nu_f,   # per-factor skew drivers
            zt * (p["rho_slow"] * nu_s + p["rho_fast"] * nu_f),   # skew x moneyness
            2.0 * frac - 1.0,                             # which factor holds the variance now
            torch.tanh(torch.log(eta_s)), torch.tanh(torch.log(eta_f)),
        ], dim=-1)

    def log_total_variance(self, x, v_slow, v_fast, tau, p):
        vb = torch.clamp(self.vbar(v_slow, v_fast, tau, p), min=1e-12)
        g = G_LIMIT * torch.tanh(self.network(self.features(x, v_slow, v_fast, tau, p))[..., 0])
        return torch.log(tau) + torch.log(vb) + 2.0 * g

    def price(self, spot, v_slow, v_fast, tau, strike, rate, carry, p) -> torch.Tensor:
        positive = spot > 0
        safe = torch.where(positive, spot, torch.ones_like(spot))
        x = torch.log(safe / strike) + (rate - carry) * tau
        ell = self.log_total_variance(x, v_slow, v_fast, tau, p)
        value = strike * torch.exp(-rate * tau) * black76_normalised(x, torch.exp(ell))
        return torch.where(positive, value, torch.zeros_like(value))


def _d(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]


def residual_and_structure(model, spot, v_slow, v_fast, tau, strike, rate, carry, p):
    """Canonical PDE residual plus the four structural quantities, one autograd pass.

    The residual reproduces ``src/model3_pde/operator.double_heston_pde_residual``
    term for term; it is recomputed here only so the intermediate derivatives can
    be reused for the structural penalties instead of being discarded.
    """
    prices = model.price(spot, v_slow, v_fast, tau, strike, rate, carry, p)
    c_s = _d(prices, spot)
    c_ss = _d(c_s, spot)
    c_tau = _d(prices, tau)
    c_vs = _d(prices, v_slow); c_vss = _d(c_vs, v_slow); c_svs = _d(c_s, v_slow)
    c_vf = _d(prices, v_fast); c_vff = _d(c_vf, v_fast); c_svf = _d(c_s, v_fast)
    generator = ((rate - carry) * spot * c_s
                 + 0.5 * (v_slow + v_fast) * spot.square() * c_ss
                 + p["kappa_slow"] * (p["theta_slow"] - v_slow) * c_vs
                 + p["kappa_fast"] * (p["theta_fast"] - v_fast) * c_vf
                 + p["rho_slow"] * p["sigma_slow"] * v_slow * spot * c_svs
                 + p["rho_fast"] * p["sigma_fast"] * v_fast * spot * c_svf
                 + 0.5 * p["sigma_slow"].square() * v_slow * c_vss
                 + 0.5 * p["sigma_fast"].square() * v_fast * c_vff)
    residual = c_tau - (generator - rate * prices)
    return residual, prices, {"c_s": c_s, "c_ss": c_ss, "c_vs": c_vs, "c_vf": c_vf}


def structural_penalty(deriv, tau, carry) -> torch.Tensor:
    """Delta bounds, convexity in spot, and non-negative vega on both factors."""
    relu = torch.nn.functional.relu
    delta_cap = torch.exp(-carry * tau)
    return (relu(-deriv["c_s"]).square()
            + relu(deriv["c_s"] - delta_cap).square()
            + relu(-deriv["c_ss"]).square()
            + relu(-deriv["c_vs"]).square()
            + relu(-deriv["c_vf"]).square()).mean()
