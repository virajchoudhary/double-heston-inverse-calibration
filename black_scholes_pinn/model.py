"""Neural price surface and Black--Scholes physics for the standalone PINN.

The network never calls the analytical Black--Scholes formula.  It learns a
normalized call price c = C / K from market observations, boundary/terminal
conditions, and the PDE

    c_tau - 0.5 sigma^2 c_xx
          - (r-q-0.5 sigma^2) c_x + r c = 0,

where x = log(S/K) and tau is time to maturity.  Volatility is an inverse
parameter learned jointly with the neural-network weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Domain:
    """Training domain expressed in log-moneyness and years."""

    x_min: float = math.log(0.55)
    x_max: float = math.log(1.65)
    tau_max: float = 2.0

    def validate(self) -> None:
        if not self.x_min < self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.tau_max <= 0.0:
            raise ValueError("tau_max must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LossWeights:
    """Weights for the genuine inverse-PINN objective."""

    pde: float = 1.0
    market: float = 100.0
    boundary: float = 10.0
    terminal: float = 10.0

    def validate(self) -> None:
        if min(asdict(self).values()) < 0.0:
            raise ValueError("loss weights must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class BlackScholesPINN(nn.Module):
    """Smooth MLP price field with a jointly calibrated constant volatility.

    Boundary and terminal conditions are enforced explicitly by loss terms. This
    avoids carrying the payoff kink into positive maturities, where the true
    parabolic PDE solution is smooth.
    """

    def __init__(
        self,
        *,
        domain: Domain,
        rate: float,
        dividend: float,
        hidden_width: int = 64,
        hidden_layers: int = 4,
        sigma_initial: float = 0.35,
        sigma_min: float = 0.03,
        sigma_max: float = 0.80,
    ) -> None:
        super().__init__()
        domain.validate()
        if hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("hidden_width and hidden_layers must be positive")
        if not sigma_min < sigma_initial < sigma_max:
            raise ValueError("sigma_initial must lie strictly inside its bounds")

        self.domain = domain
        self.rate = float(rate)
        self.dividend = float(dividend)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        layers: list[nn.Module] = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.Tanh()])
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_network()

        unit_initial = (sigma_initial - sigma_min) / (sigma_max - sigma_min)
        raw_initial = math.log(unit_initial / (1.0 - unit_initial))
        self.raw_sigma = nn.Parameter(torch.tensor(raw_initial))

    def _initialize_network(self) -> None:
        linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        for layer in linear_layers:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        # A modest positive initial price stabilizes optimization while remaining
        # independent of the analytical solution.
        nn.init.constant_(linear_layers[-1].bias, -2.5)

    @property
    def sigma(self) -> torch.Tensor:
        span = self.sigma_max - self.sigma_min
        return self.sigma_min + span * torch.sigmoid(self.raw_sigma)

    def normalized_inputs(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x_scaled = 2.0 * (x - self.domain.x_min) / (self.domain.x_max - self.domain.x_min) - 1.0
        tau_scaled = 2.0 * tau / self.domain.tau_max - 1.0
        return torch.cat((x_scaled, tau_scaled), dim=1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Return c(x,tau)=C/K without using the analytical BS formula."""
        if x.ndim != 2 or tau.ndim != 2 or x.shape != tau.shape or x.shape[1] != 1:
            raise ValueError("x and tau must both have shape (n, 1)")

        return F.softplus(self.network(self.normalized_inputs(x, tau)))

    def pde_residual(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Evaluate the Black--Scholes PDE residual by automatic differentiation."""
        if not x.requires_grad or not tau.requires_grad:
            raise ValueError("collocation x and tau must require gradients")
        price = self(x, tau)
        ones = torch.ones_like(price)
        c_tau = torch.autograd.grad(
            price, tau, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        c_x = torch.autograd.grad(
            price, x, grad_outputs=ones, create_graph=True, retain_graph=True
        )[0]
        c_xx = torch.autograd.grad(
            c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True, retain_graph=True
        )[0]
        sigma_squared = self.sigma.square()
        return (
            c_tau
            - 0.5 * sigma_squared * c_xx
            - (self.rate - self.dividend - 0.5 * sigma_squared) * c_x
            + self.rate * price
        )

    def condition_errors(
        self,
        terminal_x: torch.Tensor,
        boundary_tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return terminal and two-sided boundary MSE diagnostics."""
        zeros = torch.zeros_like(terminal_x)
        terminal_target = F.relu(torch.exp(terminal_x) - 1.0)
        terminal_loss = F.mse_loss(self(terminal_x, zeros), terminal_target)

        low_x = torch.full_like(boundary_tau, self.domain.x_min)
        high_x = torch.full_like(boundary_tau, self.domain.x_max)
        low_target = torch.zeros_like(boundary_tau)
        high_target = (
            torch.exp(high_x - self.dividend * boundary_tau)
            - torch.exp(-self.rate * boundary_tau)
        )
        boundary_loss = 0.5 * (
            F.mse_loss(self(low_x, boundary_tau), low_target)
            + F.mse_loss(self(high_x, boundary_tau), high_target)
        )
        return terminal_loss, boundary_loss

    def metadata(self) -> dict[str, object]:
        linear_layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        return {
            "architecture": "smooth Black-Scholes inverse PINN",
            "inputs": ["log_moneyness", "time_to_maturity"],
            "output": "normalized_call_price_C_over_K",
            "hidden_width": linear_layers[0].out_features,
            "hidden_layers": len(linear_layers) - 1,
            "activation": "tanh",
            "positive_price_transform": "softplus",
            "condition_enforcement": "explicit terminal and boundary losses",
            "rate": self.rate,
            "dividend": self.dividend,
            "sigma_bounds": [self.sigma_min, self.sigma_max],
            "domain": self.domain.to_dict(),
            "analytical_price_used_in_network": False,
        }
