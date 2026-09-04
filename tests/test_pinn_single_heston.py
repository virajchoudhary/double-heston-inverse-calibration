#!/usr/bin/env python3
"""Deterministic checks for the single-Heston PINN: physics, domain and ansatz."""

import math
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

import mlx.core as mx

import pinn_heston_core as C

BOX = C.Box()
TRADED = ((3.0, 0.09, 0.60, -0.70, 0.09),
          (9.0, 0.16, 1.60, -0.20, 0.05),
          (0.8, 0.30, 0.35, +0.30, 0.20),
          (12.0, 0.05, 1.00, -0.90, 0.35))


# ---------------------------------------------------------------- analytics

def test_lewis_reference_matches_repository_engine():
    """The Lewis contour sign is the one that reproduces the repository engine.

    A flipped sign still reproduces Black-Scholes exactly, because on the
    Im(u) = -1/2 contour the Black-Scholes characteristic function is real and
    even, so this check -- not a Black-Scholes check -- is what pins it down.
    """
    for params in TRADED:
        for maturity in (14 / 365, 30 / 365, 60 / 365, 90 / 365):
            x = np.array([-0.30, -0.10, 0.0, 0.10, 0.30])
            lewis = C.normalised_forward_call_batch(x, np.full(x.size, maturity), *params)
            repo = np.array([
                C.SH.heston_call_prices(math.exp(xi), np.array([1.0]), maturity, 0.0, 0.0, params)[0]
                for xi in x])
            assert np.max(np.abs(lewis - repo)) < 1e-8


def test_negative_correlation_lifts_low_strike_volatility():
    """rho < 0 must make low strikes (large x = log(F/K)) more expensive in vol."""
    maturity, v0 = 0.25, 0.09
    x = np.array([-0.30, 0.30])
    for rho, expect_up in ((-0.8, True), (0.8, False)):
        price = C.normalised_forward_call_batch(x, np.full(2, maturity), 3.0, 0.09, 0.6, rho, v0)
        vol = np.sqrt(C.implied_total_variance(price, x) / maturity)
        assert bool(vol[1] > vol[0]) == expect_up


def test_black76_inversion_round_trip():
    """Round-trip only where the price is actually informative about variance.

    Beyond about six standard deviations the exact price equals its own
    no-arbitrage bound to float64 precision, so total variance is not recoverable
    there by construction -- not a defect of the inversion.
    """
    grid_x, grid_w = np.meshgrid(np.linspace(-1.5, 1.5, 25), np.linspace(0.002, 0.4, 25))
    x, w = grid_x.ravel(), grid_w.ravel()
    informative = np.abs(x) / np.sqrt(w) <= 6.0
    x, w = x[informative], w[informative]
    recovered = C.implied_total_variance(C.black76_normalised(x, w), x)
    assert np.isfinite(recovered).all()
    assert np.max(np.abs(recovered - w) / w) < 1e-6


def test_expected_integrated_variance_matches_zero_vol_of_vol_price():
    """With sigma -> 0 the exact price is Black-76 at total variance T * vbar."""
    for kappa, theta, v0 in ((3.0, 0.09, 0.04), (0.5, 0.20, 0.35), (12.0, 0.05, 0.15)):
        for maturity in (7 / 365, 30 / 365, 90 / 365):
            x = np.array([-0.2, 0.0, 0.25])
            # sigma is 1e-3, not 0: the characteristic function divides by
            # sigma^2, so an exact zero is not evaluable and 1e-6 is already
            # dominated by that cancellation.
            exact = C.normalised_forward_call_batch(
                x, np.full(3, maturity), kappa, theta, 1e-3, 0.0, v0)
            vbar = C.expected_integrated_variance_rate(v0, maturity, kappa, theta)
            assert np.max(np.abs(exact - C.black76_normalised(x, maturity * vbar))) < 1e-7


# ------------------------------------------------------------------ physics

def _log_total_variance_exact(x, v, tau, structural):
    price = C.normalised_forward_call_batch(x, tau, *structural, v)
    return np.log(C.implied_total_variance(price, x))


def test_pde_residual_vanishes_on_the_exact_heston_solution():
    """The whole physics claim: plug in exact Heston, the residual must vanish.

    Derivatives come from fourth-order central differences on the exact price,
    so the tolerance is finite-difference truncation error, not model error.
    """
    cases = ((3.0, 0.09, 0.60, -0.70, 0.09, 45 / 365, -0.05),
             (9.0, 0.16, 1.60, -0.20, 0.05, 30 / 365, +0.08),
             (0.8, 0.30, 0.35, +0.30, 0.20, 60 / 365, -0.12),
             (12.0, 0.05, 1.00, -0.90, 0.35, 21 / 365, +0.00))
    for kappa, theta, sigma, rho, v, tau, x0 in cases:
        structural = (kappa, theta, sigma, rho)
        hx, hv, ht = 2e-3, v * 1.5e-3, tau * 1.5e-3
        f = lambda X, V, T: _log_total_variance_exact(
            np.array([X]), np.array([V]), np.array([T]), structural)[0]
        l = f(x0, v, tau)
        lx = (-f(x0 + 2 * hx, v, tau) + 8 * f(x0 + hx, v, tau)
              - 8 * f(x0 - hx, v, tau) + f(x0 - 2 * hx, v, tau)) / (12 * hx)
        lv = (-f(x0, v + 2 * hv, tau) + 8 * f(x0, v + hv, tau)
              - 8 * f(x0, v - hv, tau) + f(x0, v - 2 * hv, tau)) / (12 * hv)
        lt = (-f(x0, v, tau + 2 * ht) + 8 * f(x0, v, tau + ht)
              - 8 * f(x0, v, tau - ht) + f(x0, v, tau - 2 * ht)) / (12 * ht)
        lxx = (-f(x0 + 2 * hx, v, tau) + 16 * f(x0 + hx, v, tau) - 30 * l
               + 16 * f(x0 - hx, v, tau) - f(x0 - 2 * hx, v, tau)) / (12 * hx * hx)
        lvv = (-f(x0, v + 2 * hv, tau) + 16 * f(x0, v + hv, tau) - 30 * l
               + 16 * f(x0, v - hv, tau) - f(x0, v - 2 * hv, tau)) / (12 * hv * hv)
        lxv = (f(x0 + hx, v + hv, tau) - f(x0 + hx, v - hv, tau)
               - f(x0 - hx, v + hv, tau) + f(x0 - hx, v - hv, tau)) / (4 * hx * hv)
        w = math.exp(l)
        a = 0.5 * x0 * x0 - 0.125 * w * w - 0.5 * w
        dx = 2.0 + 2.0 * (0.5 * w - x0) * lx + a * lx * lx + w * (lxx + lx * lx) - w * lx
        cr = (0.5 * w - x0) * lv + a * lx * lv + w * (lxv + lx * lv)
        dv = a * lv * lv + w * (lvv + lv * lv)
        residual = (w * lt - 0.5 * v * dx - rho * sigma * v * cr
                    - 0.5 * sigma * sigma * v * dv - kappa * (theta - v) * w * lv)
        scale = v + float(C.expected_integrated_variance_rate(v, tau, kappa, theta))
        assert abs(residual / scale) < 2e-3


def test_untrained_network_sits_on_the_zero_vol_of_vol_solution():
    """g is initialised to exactly zero, so sigma -> 0 must give a zero residual."""
    model = C.ImpliedVarianceNet(width=32, depth=2)
    mx.eval(model.parameters())
    norm = C.Normaliser.from_box(BOX)
    n = 256
    rng = np.random.default_rng(0)
    f = lambda a: mx.array(np.asarray(a, dtype=np.float32))
    tau = rng.uniform(BOX.maturity_low, BOX.maturity_high, n)
    v = np.exp(rng.uniform(math.log(0.01), math.log(0.5), n))
    residual, _ = C.heston_residual(
        model, f(rng.uniform(-2, 2, n)), f(v), f(tau),
        f(rng.uniform(0.5, 8.0, n)), f(np.exp(rng.uniform(math.log(0.02), math.log(0.5), n))),
        f(np.full(n, 1e-5)), f(rng.uniform(-0.9, 0.5, n)), norm)
    mx.eval(residual)
    assert float(mx.max(mx.abs(residual))) < 1e-3


def test_derivatives_are_finite_across_the_whole_box():
    """The saturating-feature failure mode: tanh' underflows while its chain factor diverges."""
    model = C.ImpliedVarianceNet(width=48, depth=3)
    mx.eval(model.parameters())
    norm = C.Normaliser.from_box(BOX)
    points = C.sample_collocation(1500, {"S": 1000.0}, {"S": (20.0, 900.0)}, {"S": 1.0},
                                  BOX, seed=7)
    f = lambda a: mx.array(np.asarray(a, dtype=np.float32))
    residual, diag = C.heston_residual(
        model, f(points["x"]), f(points["variance"]), f(points["maturity"]),
        f(points["kappa"]), f(points["theta"]), f(points["sigma"]), f(points["rho"]), norm)
    mx.eval(residual, diag["w"])
    assert np.isfinite(np.asarray(residual, dtype=np.float64)).all()


# ------------------------------------------------------------------- ansatz

def test_ansatz_enforces_terminal_and_no_arbitrage_bounds_exactly():
    """Black-76 of a positive total variance can never leave the arbitrage band."""
    x = np.linspace(-3.5, 3.5, 60)
    for w in (1e-8, 1e-3, 0.05, 0.9):
        price = C.black76_normalised(x, np.full(x.size, w))
        assert (price >= np.maximum(np.exp(x) - 1.0, 0.0) - 1e-12).all()
        assert (price <= np.exp(x) + 1e-12).all()
    payoff = C.black76_normalised(x, np.full(x.size, 1e-14))
    assert np.max(np.abs(payoff - np.maximum(np.exp(x) - 1.0, 0.0))) < 1e-6


def test_parameter_transform_round_trips_inside_the_box():
    transform = C.ParameterTransform(BOX)
    for params in TRADED:
        recovered = transform.decode(transform.encode(params))
        assert np.max(np.abs(recovered - np.array(params)) / np.abs(params)) < 1e-6


# -------------------------------------------------------------------- domain

def test_collocation_respects_the_specified_physical_domain():
    """Spot in [0, 1.5 x ten-year max]; maturity on the NSE three-month cycle."""
    ceilings = {"NHPC": 153.0, "TATAPOWER": 840.0, "TORNTPOWER": 3000.0}
    strikes = {"NHPC": (2.5, 102.0), "TATAPOWER": (19.0, 560.0), "TORNTPOWER": (125.0, 2000.0)}
    n = 18000
    points = C.sample_collocation(n, ceilings, strikes, {k: 1.0 for k in ceilings}, BOX, seed=11)
    assert 14000 <= n <= 20000
    assert points["x"].size == n
    assert (points["spot"] >= 0.0).all()
    assert (points["spot"] <= points["spot_ceiling"] + 1e-9).all()
    assert (points["maturity_days"] >= BOX.maturity_low_days - 1e-9).all()
    assert (points["maturity_days"] <= BOX.maturity_high_days + 1e-9).all()
    for slice_days in C.MONTH_SLICE_DAYS:
        assert np.isclose(points["maturity_days"], slice_days, atol=1e-6).sum() > 100
    assert (np.abs(points["x"]) <= BOX.x_half_width + 1e-9).all()
    assert points["spot"].max() > 0.9 * max(ceilings.values())


def test_variance_axis_comes_from_inverse_black_scholes_range():
    """v is (inverse-BSM implied volatility)^2; the box must contain the market's."""
    assert BOX.variance_low <= 0.013     # observed minimum of market IV^2
    assert BOX.variance_high >= 1.10     # observed maximum of market IV^2


def test_parameter_screen_keeps_implied_volatility_economic():
    rng = np.random.default_rng(3)
    drawn = C.draw_parameters(rng, 200, BOX)
    for tau in (30 / 365, 90 / 365):
        vol, _, _ = C.implied_vol_batch(
            np.zeros(200), np.full(200, tau), drawn["kappa"], drawn["theta"],
            drawn["sigma"], drawn["rho"], drawn["theta"])
        assert np.isfinite(vol).all()
        assert (vol >= C.ATM_VOL_LOW).all() and (vol <= C.ATM_VOL_HIGH).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
