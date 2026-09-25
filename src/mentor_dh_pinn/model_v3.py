"""Hard-constrained Double Heston forward PINN.

The V1/V2 network emits an unconstrained raw price.  Nothing in that design
forces ``C >= 0``, and measurement showed 13.0% of points over the full box
returning a negative price.  A calibration objective built on such a pricer can
be pulled toward parameter regions where prices are negative, so the inverse
stage cannot safely stand on it.

This model predicts **implied total variance** instead and recovers the price
analytically:

    x       = log(F/K),  F = S exp((r-q) tau)
    vbar(t) = sum over factors of  theta_i + (v_i - theta_i) (1 - e^{-kappa_i t}) / (kappa_i t)
    log w   = log(tau) + log(vbar) + 2 g(x, v_slow, v_fast, tau)
    C       = K exp(-r tau) * [ e^x N(d1) - N(d2) ],   d1,2 = x/sqrt(w) +/- sqrt(w)/2

``vbar`` is the exact Double Heston expected integrated variance per unit time,
so ``g = 0`` reproduces the model exactly in the zero-vol-of-vol limit and the
network only learns the smile correction on top of a physically exact backbone.

Four properties then hold by construction rather than by penalty:

* ``tau -> 0`` gives ``w -> 0`` and Black-76 collapses onto ``max(S-K, 0)``;
* ``S -> 0`` gives ``C -> 0``;  ``S -> inf`` gives ``C -> S e^{-q tau} - K e^{-r tau}``;
* ``max(S e^{-q tau} - K e^{-r tau}, 0) <= C <= S e^{-q tau}`` pointwise;
* ``C >= 0`` always.

The canonical PDE is still enforced on the resulting raw price field, so this
remains physics-informed; the terminal and boundary losses become diagnostics
rather than objectives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

FEATURE_NAMES = ("x_scaled", "z_bounded", "z_fine", "log_tau", "sqrt_tau",
                 "log_v_slow", "log_v_fast", "v_slow_vs_theta", "v_fast_vs_theta", "factor_ratio")
FEATURE_COUNT = len(FEATURE_NAMES)
G_LIMIT = 1.5     # implied volatility stays within e^{+/-1.5} of the vbar backbone
Z_BOUND = 8.0     # bound on the standardised-moneyness feature, by construction
SQRT_2 = math.sqrt(2.0)


def mean_reversion_ratio(u: torch.Tensor, switch: float = 0.05) -> torch.Tensor:
    """``(1 - e^{-u}) / u`` with a Taylor branch near zero.

    The closed form cancels catastrophically for small ``u`` -- numerator and
    denominator are both O(u) and their difference is O(u^2) -- and the PDE needs
    second derivatives of this quantity.
    """
    safe = torch.where(u < switch, torch.ones_like(u), u)
    exact = (1.0 - torch.exp(-safe)) / safe
    series = 1.0 - u / 2.0 + u * u / 6.0 - u ** 3 / 24.0 + u ** 4 / 120.0
    return torch.where(u < switch, series, exact)


def expected_integrated_variance(v_slow, v_fast, tau, params) -> torch.Tensor:
    """Exact Double Heston ``E[int_0^tau (v_s + v_f) ds] / tau``."""
    ks, ts = params["kappa_slow"], params["theta_slow"]
    kf, tf = params["kappa_fast"], params["theta_fast"]
    slow = ts + (v_slow - ts) * mean_reversion_ratio(ks * tau)
    fast = tf + (v_fast - tf) * mean_reversion_ratio(kf * tau)
    return slow + fast


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / SQRT_2))


def black76_normalised(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``c = e^x N(d1) - N(d2)``; the undiscounted forward call over strike."""
    root = torch.sqrt(torch.clamp(w, min=1e-300))
    d1 = x / root + 0.5 * root
    value = torch.exp(x) * _normal_cdf(d1) - _normal_cdf(d1 - root)
    # Deep out of the money both terms are tiny and nearly equal, so float64
    # cancellation can leave a value of order -1e-18. The analytic lower bound is
    # zero, so clamping there removes rounding noise without changing the model.
    return torch.clamp(value, min=0.0)


class HardConstrainedDoubleHestonPINN(nn.Module):
    """Predicts implied total variance; the price is analytic in it."""

    def __init__(self, params: dict, domain, *, hidden_layers: int = 5, hidden_width: int = 128):
        super().__init__()
        self.params = {k: float(v) for k, v in params.items()}
        self.register_buffer("x_half_width", torch.tensor(6.0, dtype=torch.float64))
        self.log_tau_lo = math.log(domain.maturity_low)
        self.log_tau_hi = math.log(domain.maturity_high)
        self.log_vs_lo, self.log_vs_hi = math.log(domain.v_slow_low), math.log(domain.v_slow_high)
        self.log_vf_lo, self.log_vf_hi = math.log(domain.v_fast_low), math.log(domain.v_fast_high)
        layers: list[nn.Module] = []
        previous = FEATURE_COUNT
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, hidden_width), nn.Tanh()))
            previous = hidden_width
        head = nn.Linear(previous, 1)
        # Start at g = 0 exactly: the zero-vol-of-vol solution, where the PDE
        # residual vanishes identically. Training begins on the solution manifold.
        nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        layers.append(head)
        self.network = nn.Sequential(*layers)
        self.double()

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _unit(self, value, lo, hi):
        return 2.0 * (value - lo) / (hi - lo) - 1.0

    def features(self, x, v_slow, v_fast, tau, vbar):
        # Standardised moneyness bounded by construction: writing it as
        # x / sqrt(T vbar + x^2/Z^2) keeps |z| < Z without a clip, which a hard
        # clip cannot do without sending the feature's derivative to exactly zero
        # while its chain factor diverges.
        z = x / torch.sqrt(tau * vbar + x * x / (Z_BOUND ** 2) + 1e-14)
        ts, tf = self.params["theta_slow"], self.params["theta_fast"]
        return torch.stack([
            x / self.x_half_width,
            z / Z_BOUND,
            torch.tanh(z / 1.5),
            self._unit(torch.log(tau), self.log_tau_lo, self.log_tau_hi),
            torch.sqrt(tau / math.exp(self.log_tau_hi)),
            self._unit(torch.log(v_slow), self.log_vs_lo, self.log_vs_hi),
            self._unit(torch.log(v_fast), self.log_vf_lo, self.log_vf_hi),
            torch.tanh(torch.log(v_slow / ts)),
            torch.tanh(torch.log(v_fast / tf)),
            torch.tanh(torch.log(v_slow / v_fast)),
        ], dim=-1)

    def log_total_variance(self, x, v_slow, v_fast, tau):
        vbar = expected_integrated_variance(v_slow, v_fast, tau, self.params)
        g = G_LIMIT * torch.tanh(self.network(self.features(x, v_slow, v_fast, tau, vbar))[..., 0])
        return torch.log(tau) + torch.log(vbar) + 2.0 * g, vbar, g

    def price(self, spot, v_slow, v_fast, tau, strike, rate, carry) -> torch.Tensor:
        """Full call price.  ``spot = 0`` returns exactly zero."""
        positive = spot > 0
        safe_spot = torch.where(positive, spot, torch.ones_like(spot))
        x = torch.log(safe_spot / strike) + (rate - carry) * tau
        ell, _, _ = self.log_total_variance(x, v_slow, v_fast, tau)
        c = black76_normalised(x, torch.exp(ell))
        value = strike * torch.exp(-rate * tau) * c
        return torch.where(positive, value, torch.zeros_like(value))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Accepts the V1/V2 feature matrix so the two models are interchangeable."""
        s, vs, vf, tau, k, r, q = (features[:, i] for i in range(7))
        return self.price(s, vs, vf, tau, k, r, q)

    def predict_price(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward(features).reshape(-1)
