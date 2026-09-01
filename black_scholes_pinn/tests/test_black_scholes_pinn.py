from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIRECTORY))

from market import (
    black_scholes_call,
    dense_evaluation_grid,
    generate_synthetic_market,
    market_to_normalized,
)
from model import BlackScholesPINN, Domain


class BlackScholesPINNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = BlackScholesPINN(
            domain=Domain(), rate=0.03, dividend=0.01, hidden_width=16, hidden_layers=2
        ).double()

    def test_terminal_and_boundary_losses_are_finite_and_differentiable(self) -> None:
        domain = self.model.domain
        x = torch.linspace(domain.x_min, domain.x_max, 101, dtype=torch.float64).reshape(-1, 1)
        tau = torch.linspace(0.0, domain.tau_max, 101, dtype=torch.float64).reshape(-1, 1)
        terminal, boundary = self.model.condition_errors(x, tau)
        condition_loss = terminal + boundary
        self.assertTrue(torch.isfinite(condition_loss))
        self.assertGreaterEqual(float(condition_loss.detach()), 0.0)
        condition_loss.backward()
        first_weight = next(self.model.network.parameters())
        self.assertIsNotNone(first_weight.grad)

    def test_pde_residual_has_second_order_autodiff_and_sigma_gradient(self) -> None:
        x = torch.tensor([[-0.2], [0.1]], dtype=torch.float64, requires_grad=True)
        tau = torch.tensor([[0.4], [1.2]], dtype=torch.float64, requires_grad=True)
        loss = self.model.pde_residual(x, tau).square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.raw_sigma.grad)
        self.assertTrue(torch.isfinite(self.model.raw_sigma.grad))
        self.assertGreater(abs(float(self.model.raw_sigma.grad)), 0.0)

    def test_sigma_is_bounded(self) -> None:
        with torch.no_grad():
            self.model.raw_sigma.fill_(100.0)
        self.assertLessEqual(float(self.model.sigma.detach()), self.model.sigma_max)
        with torch.no_grad():
            self.model.raw_sigma.fill_(-100.0)
        self.assertGreaterEqual(float(self.model.sigma.detach()), self.model.sigma_min)

    def test_reference_formula_known_value(self) -> None:
        value = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
        self.assertAlmostEqual(float(value), 10.450583572185565, places=10)

    def test_synthetic_market_normalization(self) -> None:
        data = generate_synthetic_market(
            true_sigma=0.2, rate=0.03, dividend=0.01, seed=4
        )
        x, tau, normalized = market_to_normalized(data)
        self.assertEqual(len(data), 180)
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(tau > 0.0))
        self.assertTrue(np.all(normalized >= 0.0))

    def test_dense_evaluation_contracts_are_out_of_sample(self) -> None:
        market = generate_synthetic_market(
            true_sigma=0.2, rate=0.03, dividend=0.01, seed=4
        )
        evaluation = dense_evaluation_grid(
            rate=0.03, dividend=0.01, true_sigma=0.2
        )
        trained_contracts = set(zip(market["strike"].round(10), market["tau"].round(10)))
        scored_contracts = set(
            zip(evaluation["strike"].round(10), evaluation["tau"].round(10))
        )
        self.assertTrue(trained_contracts.isdisjoint(scored_contracts))


if __name__ == "__main__":
    unittest.main()
