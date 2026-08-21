"""Node C PDE/physics audit tests (overnight 2026-08-22).

Focused evidence tests for the canonical Double Heston physics contract:

1. the differentiable Gauss-Laguerre pricer satisfies the canonical pricing
   PDE (derived in
   .ai-research/overnight/2026-08-22/node-C/derivations/) to near machine
   precision, and the check is sensitive to deliberately corrupted
   coefficients;
2. Archive-2's ``pde_residual_loss`` variance-state derivatives are
   structurally zero (autograd views created after the forward pass are not
   graph nodes), so its residual omits all variance dynamics;
3. Archive-2's hard parameter transform emits vectors invalid under the
   canonical structural contract;
4. limiting cases: exact factor additivity, two-half-factor reduction to
   single-factor Heston (cross-checked against the independent COS pricer),
   deterministic-variance Black-Scholes limit, put-call parity.

Run: /usr/bin/python3 -m pytest tests/test_node_c_pde_physics_audit.py -q
(repo targets Python >= 3.10; under 3.9 the single_heston/torch tests rely on
the same typing shim used by the probe -- see conftest shim below).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch

if sys.version_info < (3, 10):
    # src/double_heston.py uses typing.TypeAlias and PEP 604 alias VALUES at
    # module scope (Python >= 3.10). Under 3.9 load it with those three
    # typing-only lines neutralised; all pricing mathematics is unmodified and
    # each replacement is asserted to match exactly once.
    import importlib.util

    _source = (SRC_ROOT / "double_heston.py").read_text()
    for _old, _new in (
        ("from typing import TypeAlias", "TypeAlias = object"),
        ("ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]", "ParameterInput = None"),
        ("ComplexResult: TypeAlias = complex | np.ndarray", "ComplexResult = None"),
    ):
        assert _source.count(_old) == 1, _old
        _source = _source.replace(_old, _new)
    import src as _src_pkg

    _mod = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("src.double_heston", SRC_ROOT / "double_heston.py")
    )
    _mod.__dict__["__package__"] = "src"
    sys.modules["src.double_heston"] = _mod
    exec(compile(_source, "double_heston.py", "exec"), _mod.__dict__)
    double_heston = _mod
else:
    from src import double_heston

from src.constraints import validate_parameters
from src.torch_double_heston import price_double_heston_call_tensor

# Canonical order; both vectors satisfy every canonical structural constraint
# (per-factor Feller, kappa ordering, correlation disk) and the Archive-2 box.
VEC_A = [0.60, 0.040, 0.180, -0.55, 0.040, 2.80, 0.030, 0.350, -0.35, 0.020]
VEC_B = [0.30, 0.060, 0.160, -0.30, 0.050, 6.00, 0.060, 0.700, -0.60, 0.030]
POINTS = [(100.0, 100.0, 0.25), (100.0, 100.0, 1.0), (100.0, 90.0, 0.5), (100.0, 110.0, 0.5)]


def canonical_residual(vec, S, K, tau, r=0.05, q=0.0, mix_factor=1.0):
    """Assemble the derived PDE residual via autograd on the canonical torch
    pricer. The v0 entries are genuine leaves stacked into the parameter
    vector, so autograd reaches them. ``mix_factor`` scales both rho
    cross-term coefficients (1.0 = canonical PDE, 0.0 = rho terms dropped)."""
    v0_s = torch.tensor(vec[4], dtype=torch.float64, requires_grad=True)
    v0_f = torch.tensor(vec[9], dtype=torch.float64, requires_grad=True)
    constants = [torch.tensor(v, dtype=torch.float64) for v in (vec[:4] + vec[5:9])]
    params = torch.stack([*constants[:4], v0_s, *constants[4:], v0_f])
    spot = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    maturity = torch.tensor(tau, dtype=torch.float64, requires_grad=True)

    price = price_double_heston_call_tensor(
        spot, torch.tensor(K), maturity,
        torch.tensor(r, dtype=torch.float64), torch.tensor(q, dtype=torch.float64), params,
    )

    def g(out, inp):
        return torch.autograd.grad(out, inp, create_graph=True, retain_graph=True)[0]

    d_tau = g(price, maturity)
    delta = g(price, spot)
    gamma = g(delta, spot)
    d_vs, d_vf = g(price, v0_s), g(price, v0_f)
    cross_s, cross_f = g(delta, v0_s), g(delta, v0_f)
    d2_vs, d2_vf = g(d_vs, v0_s), g(d_vf, v0_f)

    k_s, th_s, sg_s, r_s = vec[0:4]
    k_f, th_f, sg_f, r_f = vec[5:9]
    diffusion = 0.5 * (v0_s + v0_f) * spot.square() * gamma
    drift = (r - q) * spot * delta - r * price
    f_slow = k_s * (th_s - v0_s) * d_vs + mix_factor * r_s * sg_s * v0_s * spot * cross_s + 0.5 * sg_s ** 2 * v0_s * d2_vs
    f_fast = k_f * (th_f - v0_f) * d_vf + mix_factor * r_f * sg_f * v0_f * spot * cross_f + 0.5 * sg_f ** 2 * v0_f * d2_vf
    residual = d_tau - (diffusion + drift + f_slow + f_fast)
    price_value = float(price.detach())
    return float(residual.detach()) / max(abs(price_value), 1.0), price_value


@pytest.mark.parametrize("vec_name,vec", [("A", VEC_A), ("B", VEC_B)])
@pytest.mark.parametrize("S,K,tau", POINTS)
def test_canonical_pricer_satisfies_canonical_pde(vec_name, vec, S, K, tau):
    rel, _ = canonical_residual(vec, S, K, tau)
    # Observed magnitudes <= 1.4e-15; tolerance leaves a 100x headroom while
    # staying ~7 orders below the smallest perturbed residual (6e-3).
    assert abs(rel) < 1e-12


@pytest.mark.parametrize("vec_name,vec", [("A", VEC_A), ("B", VEC_B)])
@pytest.mark.parametrize("S,K,tau", POINTS)
def test_residual_detects_wrong_correlation_coefficient(vec_name, vec, S, K, tau):
    rel_correct, _ = canonical_residual(vec, S, K, tau)
    rel_no_rho, _ = canonical_residual(vec, S, K, tau, mix_factor=0.0)
    assert abs(rel_no_rho) > 1e-3
    assert abs(rel_no_rho) > 1e6 * abs(rel_correct)


@pytest.mark.parametrize("vec_name,vec", [("A", VEC_A), ("B", VEC_B)])
def test_half_mix_residual_is_exactly_half(vec_name, vec):
    """Halving the cross coefficient must halve the perturbed residual:
    internal consistency of the derivative assembly."""
    for S, K, tau in POINTS:
        rel_full, _ = canonical_residual(vec, S, K, tau, mix_factor=1.0)
        assert abs(rel_full) < 1e-12  # canonical operator cancels
        rel_no, _ = canonical_residual(vec, S, K, tau, mix_factor=0.0)
        rel_half, _ = canonical_residual(vec, S, K, tau, mix_factor=0.5)
        assert rel_half == pytest.approx(0.5 * rel_no, rel=1e-6, abs=1e-15)


def canonical_to_archive2(vec):
    k_s, th_s, sg_s, r_s, v_s, k_f, th_f, sg_f, r_f, v_f = vec
    return [v_f, k_f, th_f, sg_f, r_f, v_s, k_s, th_s, sg_s, r_s]


def _archive2_batch():
    return {
        "mask": torch.ones(1, 1, dtype=torch.bool),
        "spot": torch.tensor([100.0], dtype=torch.float64),
        "strike": torch.tensor([[100.0]], dtype=torch.float64),
        "tau": torch.tensor([[0.5]], dtype=torch.float64),
        "rate": torch.tensor([0.05], dtype=torch.float64),
        "dividend": torch.tensor([0.0], dtype=torch.float64),
        "is_call": torch.ones(1, 1, dtype=torch.float64),
        "market_price": torch.zeros(1, 1, dtype=torch.float64),
    }


@pytest.mark.parametrize("vec", [VEC_A, VEC_B])
def test_archive2_variance_derivatives_are_structurally_zero(vec):
    """Reproduce losses.py's exact construction: chosen_params[:, 0] / [:, 5]
    views created AFTER pricing are not graph nodes, so _safe_grad returns
    zeros while the true dV/dv0 is large."""
    from dheston.models.losses import _safe_grad
    from dheston.pricing.heston import FourierConfig, price_double_heston_torch

    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
    batch = _archive2_batch()
    surface_index = torch.tensor([0])
    point_index = torch.tensor([0])
    chosen_params = params[surface_index]
    spot = batch["spot"][surface_index].detach().clone().requires_grad_(True)
    tau = batch["tau"][surface_index, point_index].detach().clone().requires_grad_(True)
    prices = price_double_heston_torch(
        spot, batch["strike"][surface_index, point_index], tau,
        batch["rate"][surface_index], batch["dividend"][surface_index],
        batch["is_call"][surface_index, point_index], chosen_params,
        FourierConfig(),
    )
    v01, v02 = chosen_params[:, 0], chosen_params[:, 5]
    for broken in (_safe_grad(prices, v01), _safe_grad(prices, v02),
                   _safe_grad(_safe_grad(prices, spot), v01),
                   _safe_grad(_safe_grad(prices, v01), v01)):
        assert float(broken.abs()) == 0.0
    true_grad = torch.autograd.grad(prices, params, create_graph=True)[0]
    assert abs(float(true_grad[0, 0])) > 1.0  # genuine dV/dv0_fast exists


def test_archive2_emits_canonical_invalid_parameters():
    from dheston.calibration.transforms import constrain_parameter_tensor

    raw = torch.full((1, 10), -25.0, dtype=torch.float64)
    raw[0, 3] = 25.0  # sigma1 -> box upper bound 1.5
    raw[0, 8] = 25.0  # sigma2 -> box upper bound 1.5
    emitted = constrain_parameter_tensor(raw)[0].detach().numpy().tolist()
    canonical = [
        emitted[6], emitted[7], emitted[8], emitted[9], emitted[5],
        emitted[1], emitted[2], emitted[3], emitted[4], emitted[0],
    ]
    violations = validate_parameters(canonical)["violations"]
    assert any("Feller" in v for v in violations)
    assert any("rho_slow^2 + rho_fast^2" in v for v in violations)


def test_exact_factor_additivity_identity():
    """exponent(theta, v0) == 2 * exponent(theta/2, v0/2): the algebraic
    signature of independent additive-variance factors. Requires the stricter
    half-factor Feller gate kappa*theta > sigma^2."""
    u_grid = np.linspace(0.1, 60.0, 41)
    for k, th, sg, rho, v0 in ([1.5, 0.04, 0.20, -0.5, 0.04], [0.9, 0.06, 0.16, -0.35, 0.05]):
        full = double_heston.heston_log_characteristic_exponent(u_grid, 0.5, k, th, sg, rho, v0)
        half = double_heston.heston_log_characteristic_exponent(u_grid, 0.5, k, th / 2, sg, rho, v0 / 2)
        assert np.max(np.abs(full - 2.0 * half)) < 1e-13


def test_two_half_factors_reduce_to_single_heston():
    """Canonical Double Heston with two identical half-size factors must equal
    single-factor Heston; cross-checked with the independent COS pricer."""
    from dheston.pricing.heston import FourierConfig, price_standard_heston_numpy

    k, th, sg, rho, v0 = 1.5, 0.04, 0.20, -0.5, 0.04
    half_vector = [k, th / 2, sg, rho, v0 / 2, k, th / 2, sg, rho, v0 / 2]
    strikes = np.asarray([90.0, 95.0, 100.0, 105.0, 110.0])
    dual = [
        double_heston.price_double_heston_call(100.0, K, 0.75, 0.05, 0.0, half_vector, enforce_ordering=False)
        for K in strikes
    ]
    single = price_standard_heston_numpy(
        np.full_like(strikes, 100.0), strikes, np.full_like(strikes, 0.75),
        np.full_like(strikes, 0.05), np.zeros_like(strikes), np.ones_like(strikes),
        np.asarray([v0, k, th, sg, rho]), FourierConfig(integration_steps=384, u_max=140.0),
    )
    assert np.max(np.abs(np.asarray(dual) - single)) < 1e-8


def test_deterministic_variance_black_scholes_limit():
    """sigma -> small, rho -> small: variance paths become deterministic and
    prices converge to Black-Scholes with integrated total variance (observed
    relative errors <= 4e-4; tolerance allows 5e-3 headroom)."""
    vec = [0.8, 0.035, 0.02, -0.02, 0.030, 3.0, 0.025, 0.02, -0.02, 0.020]
    for K in (90.0, 100.0, 110.0):
        for tau in (0.25, 1.0):
            total_var = sum(
                th * tau + (v0 - th) * (1 - math.exp(-kap * tau)) / kap
                for kap, th, v0 in ((0.8, 0.035, 0.030), (3.0, 0.025, 0.020))
            )
            price = double_heston.price_double_heston_call(100.0, K, tau, 0.05, 0.0, vec)
            vol = math.sqrt(total_var / tau)
            d1 = (math.log(100.0 / K) + (0.05 + 0.5 * vol ** 2) * tau) / (vol * math.sqrt(tau))
            d2 = d1 - vol * math.sqrt(tau)
            bs = 100.0 * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * math.exp(-0.05 * tau) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
            assert abs(price - bs) / max(bs, 1e-8) < 5e-3


def test_put_call_parity_canonical():
    for K in (95.0, 100.0, 105.0):
        call = double_heston.price_double_heston_call(100.0, K, 0.5, 0.05, 0.01, VEC_A)
        put = double_heston.price_double_heston_put(100.0, K, 0.5, 0.05, 0.01, VEC_A)
        parity = 100.0 * math.exp(-0.01 * 0.5) - K * math.exp(-0.05 * 0.5)
        assert call - put == pytest.approx(parity, abs=1e-10)
