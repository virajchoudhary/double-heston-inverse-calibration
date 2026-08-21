from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dheston.pricing.heston import FourierConfig, price_double_heston_numpy, price_standard_heston_numpy


def test_put_call_parity_holds_reasonably() -> None:
    config = FourierConfig(integration_steps=384, u_max=140.0)
    params = np.asarray([0.05, 2.5, 0.06, 0.45, -0.6, 0.03, 0.9, 0.04, 0.30, -0.35], dtype=np.float64)
    strikes = np.asarray([90.0, 100.0, 110.0], dtype=np.float64)
    spot = np.asarray([100.0, 100.0, 100.0], dtype=np.float64)
    tau = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
    rates = np.asarray([0.03, 0.03, 0.03], dtype=np.float64)
    dividends = np.asarray([0.00, 0.00, 0.00], dtype=np.float64)
    calls = price_double_heston_numpy(spot, strikes, tau, rates, dividends, np.ones_like(strikes), params, config)
    puts = price_double_heston_numpy(spot, strikes, tau, rates, dividends, np.zeros_like(strikes), params, config)
    lhs = calls - puts
    rhs = spot - strikes * np.exp(-rates * tau)
    assert np.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_double_heston_reduces_toward_standard_heston() -> None:
    config = FourierConfig(integration_steps=384, u_max=140.0)
    standard = np.asarray([0.05, 2.5, 0.06, 0.45, -0.6], dtype=np.float64)
    double = np.asarray([0.05, 2.5, 0.06, 0.45, -0.6, 1e-6, 0.2, 1e-6, 0.05, -0.1], dtype=np.float64)
    strikes = np.asarray([95.0, 100.0, 105.0], dtype=np.float64)
    spot = np.asarray([100.0, 100.0, 100.0], dtype=np.float64)
    tau = np.asarray([0.75, 0.75, 0.75], dtype=np.float64)
    rates = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
    dividends = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    standard_prices = price_standard_heston_numpy(spot, strikes, tau, rates, dividends, np.ones_like(strikes), standard, config)
    double_prices = price_double_heston_numpy(spot, strikes, tau, rates, dividends, np.ones_like(strikes), double, config)
    assert np.allclose(standard_prices, double_prices, atol=2e-1, rtol=2e-1)
