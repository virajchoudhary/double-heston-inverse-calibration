"""Node C focused PDE-physics probe.

Evidence generator for the overnight 2026-08-22 Node C audit.
Runs with /usr/bin/python3 (3.9, numpy 1.26, torch 2.8 CPU). No training,
no package changes. Produces printed tables used in FINDINGS.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Python 3.9 compatibility loader for src.double_heston (repo targets 3.10+).
# Only the three typing-only alias statements are neutralised; every line of
# pricing mathematics executes unmodified. Asserted to be exact replacements.
# ---------------------------------------------------------------------------

import importlib.util

def _load_double_heston_py39():
    module_source = (SRC_ROOT / "double_heston.py").read_text()
    replacements = [
        ("from typing import TypeAlias", "TypeAlias = object"),
        (
            "ParameterInput: TypeAlias = Sequence[float] | Mapping[str, float]",
            "ParameterInput = None",
        ),
        (
            "ComplexResult: TypeAlias = complex | np.ndarray",
            "ComplexResult = None",
        ),
    ]
    for old, new in replacements:
        assert module_source.count(old) == 1, old
        module_source = module_source.replace(old, new)
    import src  # ensure package exists before attaching the submodule
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("src.double_heston", SRC_ROOT / "double_heston.py")
    )
    module.__dict__["__package__"] = "src"
    sys.modules["src.double_heston"] = module
    code = compile(module_source, str(SRC_ROOT / "double_heston.py"), "exec")
    exec(code, module.__dict__)
    return module


_double_heston = _load_double_heston_py39()
heston_log_characteristic_exponent = _double_heston.heston_log_characteristic_exponent
price_double_heston_call = _double_heston.price_double_heston_call
price_double_heston_put = _double_heston.price_double_heston_put

from src.constraints import validate_parameters
from src.torch_double_heston import price_double_heston_call_tensor

RESULTS: dict[str, object] = {}


def log(section: str, payload: object) -> None:
    RESULTS[section] = payload
    print(f"\n=== {section} ===")
    print(payload)


# Canonical order: [kappa_s, theta_s, sigma_s, rho_s, v0_s, kappa_f, theta_f, sigma_f, rho_f, v0_f]
# Both vectors satisfy: per-factor Feller > 0, kappa_slow < kappa_fast,
# rho_slow^2 + rho_fast^2 < 1, and dheston box bounds (for cross-stack tests).
VEC_A = [0.60, 0.040, 0.180, -0.55, 0.040, 2.80, 0.030, 0.350, -0.35, 0.020]
VEC_B = [0.30, 0.060, 0.160, -0.30, 0.050, 6.00, 0.060, 0.700, -0.60, 0.030]
for name, vec in (("VEC_A", VEC_A), ("VEC_B", VEC_B)):
    diagnostics = validate_parameters(vec)
    assert not diagnostics["violations"], (name, diagnostics["violations"])


# ---------------------------------------------------------------------------
# 1. Canonical PDE residual evaluated by autograd on the differentiable
#    Gauss-Laguerre pricer. By time homogeneity, price(S, v0_s, v0_f, tau)
#    equals U(S, v_s=v0_s, v_f=v0_f, tau), so autograd w.r.t. (spot, tau,
#    v0 entries) evaluates the derived forward-tau PDE operator.
# ---------------------------------------------------------------------------

def canonical_pde_terms(vec: list[float], S: float, K: float, tau: float, r: float, q: float):
    # v0 entries as genuine leaf tensors; the parameter vector is assembled by
    # torch.stack so autograd reaches them (in contrast to differentiating w.r.t.
    # post-hoc column views, which autograd treats as unused -- see section 2b).
    v0_s_leaf = torch.tensor(vec[4], dtype=torch.float64, requires_grad=True)
    v0_f_leaf = torch.tensor(vec[9], dtype=torch.float64, requires_grad=True)
    params = torch.stack([
        torch.tensor(vec[0], dtype=torch.float64), torch.tensor(vec[1], dtype=torch.float64),
        torch.tensor(vec[2], dtype=torch.float64), torch.tensor(vec[3], dtype=torch.float64),
        v0_s_leaf,
        torch.tensor(vec[5], dtype=torch.float64), torch.tensor(vec[6], dtype=torch.float64),
        torch.tensor(vec[7], dtype=torch.float64), torch.tensor(vec[8], dtype=torch.float64),
        v0_f_leaf,
    ])
    spot = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    maturity = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(r, dtype=torch.float64)
    dividend = torch.tensor(q, dtype=torch.float64)

    price = price_double_heston_call_tensor(spot, torch.tensor(K), maturity, rate, dividend, params)

    def g(out, inp):
        return torch.autograd.grad(out, inp, create_graph=True, retain_graph=True)[0]

    d_tau = g(price, maturity)
    delta = g(price, spot)
    gamma = g(delta, spot)
    v0_s, v0_f = v0_s_leaf, v0_f_leaf
    d_vs = g(price, v0_s)
    d_vf = g(price, v0_f)
    cross_s = g(delta, v0_s)
    cross_f = g(delta, v0_f)
    d2_vs = g(d_vs, v0_s)
    d2_vf = g(d_vf, v0_f)

    (k_s, th_s, sg_s, r_s, _), (k_f, th_f, sg_f, r_f, _) = vec[0:5], vec[5:10]
    diffusion = 0.5 * (v0_s + v0_f) * spot.square() * gamma
    drift = (rate - dividend) * spot * delta - rate * price
    factor_slow = (k_s * (th_s - v0_s) * d_vs + r_s * sg_s * v0_s * spot * cross_s + 0.5 * sg_s ** 2 * v0_s * d2_vs)
    factor_fast = (k_f * (th_f - v0_f) * d_vf + r_f * sg_f * v0_f * spot * cross_f + 0.5 * sg_f ** 2 * v0_f * d2_vf)
    residual = d_tau - (diffusion + drift + factor_slow + factor_fast)

    # Perturbed variants: deliberately wrong coefficients.
    residual_no_rho = d_tau - (diffusion + drift
                               + k_s * (th_s - v0_s) * d_vs + 0.5 * sg_s ** 2 * v0_s * d2_vs
                               + k_f * (th_f - v0_f) * d_vf + 0.5 * sg_f ** 2 * v0_f * d2_vf)
    residual_half_mix = d_tau - (diffusion + drift
                                 + k_s * (th_s - v0_s) * d_vs + 0.5 * r_s * sg_s * v0_s * spot * cross_s + 0.5 * sg_s ** 2 * v0_s * d2_vs
                                 + k_f * (th_f - v0_f) * d_vf + 0.5 * r_f * sg_f * v0_f * spot * cross_f + 0.5 * sg_f ** 2 * v0_f * d2_vf)

    price_v = float(price.detach())
    scale = max(abs(price_v), 1.0)
    return {
        "price": price_v,
        "residual": float(residual.detach()),
        "rel_residual": float(residual.detach()) / scale,
        "rel_no_rho": float(residual_no_rho.detach()) / scale,
        "rel_half_mix": float(residual_half_mix.detach()) / scale,
    }


points = [
    (100.0, 100.0, 0.25, 0.05, 0.0),
    (100.0, 100.0, 1.00, 0.05, 0.0),
    (100.0, 90.0, 0.50, 0.05, 0.0),
    (100.0, 110.0, 0.50, 0.05, 0.0),
]
table = {}
for vec_name, vec in (("VEC_A", VEC_A), ("VEC_B", VEC_B)):
    rows = []
    for S, K, tau, r, q in points:
        rows.append({"S,K,tau": f"{S},{K},{tau}", **canonical_pde_terms(vec, S, K, tau, r, q)})
    table[vec_name] = rows
log("canonical_gl_pde_residual", table)


# ---------------------------------------------------------------------------
# 2. Archive-2 COS pricer: same residual construction via dheston's own
#    pde_residual_loss on synthetic batches, for the same physical models.
# ---------------------------------------------------------------------------

from dheston.models.losses import pde_residual_loss
from dheston.pricing.heston import FourierConfig

def canonical_to_archive2(vec: list[float]) -> list[float]:
    """Map canonical [k_s,th_s,sg_s,rho_s,v0_s, k_f,th_f,sg_f,rho_f,v0_f]
    to dheston [v01,k1,th1,sg1,rho1, v02,k2,th2,sg2,rho2] with factor1 = fast."""
    k_s, th_s, sg_s, r_s, v_s, k_f, th_f, sg_f, r_f, v_f = vec
    return [v_f, k_f, th_f, sg_f, r_f, v_s, k_s, th_s, sg_s, r_s]


def archive2_residual(vec: list[float], integration_steps: int = 256) -> float:
    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64)
    strikes = torch.tensor([[90.0, 100.0, 110.0, 100.0, 105.0, 95.0]], dtype=torch.float64)
    batch = {
        "mask": torch.ones(1, 6, dtype=torch.bool),
        "spot": torch.tensor([100.0], dtype=torch.float64),
        "strike": strikes,
        "tau": torch.tensor([[0.25, 0.5, 1.0, 0.5, 0.75, 0.35]], dtype=torch.float64),
        "rate": torch.tensor([0.05], dtype=torch.float64),
        "dividend": torch.tensor([0.0], dtype=torch.float64),
        "is_call": torch.ones(1, 6, dtype=torch.float64),
        "market_price": torch.zeros(1, 6, dtype=torch.float64),  # unused by residual
    }
    config = FourierConfig(integration_steps=integration_steps)
    return float(pde_residual_loss(params, batch, config, max_points=6))


archive2_table = {
    "VEC_A_residual_loss": archive2_residual(VEC_A),
    "VEC_B_residual_loss": archive2_residual(VEC_B),
    "VEC_A_384_steps": archive2_residual(VEC_A, integration_steps=384),
    "note": "mean squared RELATIVE residual (losses.py scale = max(|V|,1)); compare with canonical rel_residual^2",
}
log("archive2_cos_pde_residual", archive2_table)


def _cos_price(steps: int) -> float:
    from dheston.pricing.heston import price_double_heston_torch as _p
    params = torch.tensor([canonical_to_archive2(VEC_A)], dtype=torch.float64)
    return float(_p(torch.tensor([100.0]), torch.tensor([100.0]), torch.tensor([0.5]),
                    torch.tensor([0.05]), torch.tensor([0.0]), torch.tensor([1.0]),
                    params, FourierConfig(integration_steps=steps)).detach())


log("archive2_integration_steps_sensitivity", {
    "price_steps_64": _cos_price(64),
    "price_steps_256": _cos_price(256),
    "price_steps_1024": _cos_price(1024),
})


# ---------------------------------------------------------------------------
# 2b. Defect proof: losses.py differentiates w.r.t. chosen_params[:, 0] /
#     chosen_params[:, 5] -- column views created AFTER the forward pass.
#     autograd matches graph nodes, not tensor semantics: such fresh views are
#     NOT ancestors of the pricer output, so _safe_grad's allow_unused path
#     silently returns ZERO for every variance-state derivative. Replicate the
#     exact losses.py construction and instrument it.
# ---------------------------------------------------------------------------

from dheston.models.losses import _safe_grad, predict_surface_prices  # noqa: E402
from dheston.pricing.heston import price_double_heston_torch  # noqa: E402


def archive2_derivative_instrumentation(vec: list[float]) -> dict[str, object]:
    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
    batch = {
        "mask": torch.ones(1, 1, dtype=torch.bool),
        "spot": torch.tensor([100.0], dtype=torch.float64),
        "strike": torch.tensor([[100.0]], dtype=torch.float64),
        "tau": torch.tensor([[0.5]], dtype=torch.float64),
        "rate": torch.tensor([0.05], dtype=torch.float64),
        "dividend": torch.tensor([0.0], dtype=torch.float64),
        "is_call": torch.ones(1, 1, dtype=torch.float64),
        "market_price": torch.zeros(1, 1, dtype=torch.float64),
    }
    config = FourierConfig(integration_steps=256)
    surface_index = torch.tensor([0])
    point_index = torch.tensor([0])
    chosen_params = params[surface_index]                      # losses.py:96
    spot = batch["spot"][surface_index].detach().clone().requires_grad_(True)   # losses.py:97
    tau = batch["tau"][surface_index, point_index].detach().clone().requires_grad_(True)  # losses.py:99
    prices = price_double_heston_torch(spot, batch["strike"][surface_index, point_index],
                                       tau, batch["rate"][surface_index], batch["dividend"][surface_index],
                                       batch["is_call"][surface_index, point_index], chosen_params, config)
    d_tau = _safe_grad(prices, tau)
    delta = _safe_grad(prices, spot)
    v01 = chosen_params[:, 0]                                  # losses.py:110 (fresh view, AFTER forward)
    v02 = chosen_params[:, 5]                                  # losses.py:115
    d_v01 = _safe_grad(prices, v01)
    d_v02 = _safe_grad(prices, v02)
    cross_sv01 = _safe_grad(delta, v01)
    d2_v01 = _safe_grad(d_v01, v01)
    # Reference: gradient w.r.t. the ACTUAL ancestor (the parameters tensor).
    d_params = torch.autograd.grad(prices, params, create_graph=True)[0]
    return {
        "price": float(prices.detach()),
        "d_tau_nonzero": bool(float(d_tau.abs()) > 0),
        "delta": float(delta.detach()),
        "d_v01_value": float(d_v01.detach()),
        "d_v02_value": float(d_v02.detach()),
        "cross_sv01_value": float(cross_sv01.detach()),
        "d2_v01_value": float(d2_v01.detach()),
        "true_dV_dv01_via_params_grad": float(d_params[0, 0].detach()),
        "v_derivatives_all_zero": bool(
            float(d_v01.abs()) == 0.0 and float(d_v02.abs()) == 0.0
            and float(cross_sv01.abs()) == 0.0 and float(d2_v01.abs()) == 0.0
        ),
    }


log("archive2_derivative_instrumentation", {
    "VEC_A": archive2_derivative_instrumentation(VEC_A),
    "VEC_B": archive2_derivative_instrumentation(VEC_B),
})


# 2c. Consequence: the effective residual actually penalised by losses.py is
#     V_tau - [ 0.5(v1+v2)S^2 V_SS + (r-q)S V_S - rV ]  (all v-dynamics terms
#     zero). Compare the full-terms residual against this effective residual
#     built from the SAME derivatives: they are identical iff v-terms vanish.

def archive2_effective_residual(vec: list[float]) -> dict[str, float]:
    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
    batch = {
        "mask": torch.ones(1, 1, dtype=torch.bool),
        "spot": torch.tensor([100.0], dtype=torch.float64),
        "strike": torch.tensor([[100.0]], dtype=torch.float64),
        "tau": torch.tensor([[0.5]], dtype=torch.float64),
        "rate": torch.tensor([0.05], dtype=torch.float64),
        "dividend": torch.tensor([0.0], dtype=torch.float64),
        "is_call": torch.ones(1, 1, dtype=torch.float64),
        "market_price": torch.zeros(1, 1, dtype=torch.float64),
    }
    config = FourierConfig(integration_steps=256)
    surface_index = torch.tensor([0])
    point_index = torch.tensor([0])
    chosen_params = params[surface_index]
    spot = batch["spot"][surface_index].detach().clone().requires_grad_(True)
    tau = batch["tau"][surface_index, point_index].detach().clone().requires_grad_(True)
    prices = price_double_heston_torch(spot, batch["strike"][surface_index, point_index],
                                       tau, batch["rate"][surface_index], batch["dividend"][surface_index],
                                       batch["is_call"][surface_index, point_index], chosen_params, config)
    d_tau = _safe_grad(prices, tau)
    delta = _safe_grad(prices, spot)
    gamma = _safe_grad(delta, spot)
    v01, v02 = chosen_params[:, 0], chosen_params[:, 5]
    k1, th1, sg1, r1 = chosen_params[:, 1], chosen_params[:, 2], chosen_params[:, 3], chosen_params[:, 4]
    k2, th2, sg2, r2 = chosen_params[:, 6], chosen_params[:, 7], chosen_params[:, 8], chosen_params[:, 9]
    d_v01, d_v02 = _safe_grad(prices, v01), _safe_grad(prices, v02)
    cross1, cross2 = _safe_grad(delta, v01), _safe_grad(delta, v02)
    d2_1, d2_2 = _safe_grad(d_v01, v01), _safe_grad(d_v02, v02)
    diffusion = 0.5 * (v01 + v02) * spot.square() * gamma
    drift = (batch["rate"][surface_index] - batch["dividend"][surface_index]) * spot * delta - batch["rate"][surface_index] * prices
    f1 = k1 * (th1 - v01) * d_v01 + r1 * sg1 * v01 * spot * cross1 + 0.5 * sg1.square() * v01 * d2_1
    f2 = k2 * (th2 - v02) * d_v02 + r2 * sg2 * v02 * spot * cross2 + 0.5 * sg2.square() * v02 * d2_2
    residual = d_tau - (diffusion + drift + f1 + f2)
    return {
        "residual": float(residual.detach()),
        "residual_without_v_terms": float((d_tau - (diffusion + drift)).detach()),
        "price": float(prices.detach()),
    }


log("archive2_effective_residual", {
    "VEC_A": archive2_effective_residual(VEC_A),
    "VEC_B": archive2_effective_residual(VEC_B),
})


# ---------------------------------------------------------------------------
# 3. Archive-2 constraint gaps: the sigmoid box map can emit vectors that are
#    invalid under the canonical structural contract.
# ---------------------------------------------------------------------------

from dheston.calibration.transforms import constrain_parameter_tensor

raw_extreme = torch.full((1, 10), -25.0, dtype=torch.float64)
raw_extreme[0, 3] = 25.0   # sigma1 -> upper bound 1.5
raw_extreme[0, 8] = 25.0   # sigma2 -> upper bound 1.5
archive2_vec = constrain_parameter_tensor(raw_extreme)[0].detach().numpy().tolist()
# Correct canonical mapping. dheston order is [v01,k1,th1,sg1,rho1, v02,k2,th2,sg2,rho2]
# with factor1 = FAST (kappa2 <= kappa1) and factor2 = SLOW; canonical order is
# [kappa_s,theta_s,sigma_s,rho_s,v0_s, kappa_f,theta_f,sigma_f,rho_f,v0_f].
canonical_full = [
    archive2_vec[6],  # kappa_slow = kappa2
    archive2_vec[7],  # theta_slow = theta2
    archive2_vec[8],  # sigma_slow = sigma2
    archive2_vec[9],  # rho_slow  = rho2
    archive2_vec[5],  # v0_slow   = v02
    archive2_vec[1],  # kappa_fast = kappa1
    archive2_vec[2],  # theta_fast = theta1
    archive2_vec[3],  # sigma_fast = sigma1
    archive2_vec[4],  # rho_fast  = rho1
    archive2_vec[0],  # v0_fast   = v01
]
gap = {
    "archive2_emitted": dict(zip(["v01","k1","th1","sg1","rho1","v02","k2","th2","sg2","rho2"], archive2_vec)),
    "canonical_mapped": dict(zip(
        ["kappa_slow","theta_slow","sigma_slow","rho_slow","v0_slow","kappa_fast","theta_fast","sigma_fast","rho_fast","v0_fast"],
        canonical_full)),
    "canonical_violations": validate_parameters(canonical_full)["violations"],
    "feller_gap_slow_mapped": 2 * canonical_full[0] * canonical_full[1] - canonical_full[2] ** 2,
    "feller_gap_fast_mapped": 2 * canonical_full[5] * canonical_full[6] - canonical_full[7] ** 2,
    "correlation_disk": canonical_full[3] ** 2 + canonical_full[8] ** 2,
}
log("archive2_constraint_gap", gap)


# ---------------------------------------------------------------------------
# 4. Exact affine additivity identity (one-factor reduction at CF level):
#    exponent(u,T,k,theta,sigma,rho,v0) == 2 * exponent(u,T,k,theta/2,sigma,rho,v0/2)
# ---------------------------------------------------------------------------

u_grid = np.linspace(0.1, 60.0, 41)
# The identity needs the HALF factor to satisfy the repo's strict Feller gate:
# 2*kappa*(theta/2) - sigma^2 > 0  <=>  kappa*theta > sigma^2 (stronger than the
# full-factor Feller 2*kappa*theta > sigma^2). Choose vectors valid for both.
IDENT_VECTORS = [
    [1.50, 0.040, 0.200, -0.50, 0.040],  # kappa*theta - sigma^2 = 0.06 - 0.04 = 0.02 > 0
    [0.90, 0.060, 0.160, -0.35, 0.050],  # 0.054 - 0.0256 = 0.0284 > 0
]
ident_rows = []
for vec in IDENT_VECTORS:
    k_s, th_s, sg_s, r_s, v_s = vec
    full = heston_log_characteristic_exponent(u_grid, 0.5, k_s, th_s, sg_s, r_s, v_s)
    half = heston_log_characteristic_exponent(u_grid, 0.5, k_s, th_s / 2, sg_s, r_s, v_s / 2)
    err = np.max(np.abs(full - 2.0 * half))
    ident_rows.append({"vec": vec, "max_abs_identity_error": float(err)})
log("factor_additivity_identity", ident_rows)


# ---------------------------------------------------------------------------
# 5. Price-level reduction: canonical Double Heston with two identical
#    half-size factors == standard single-factor Heston priced by the
#    independent dheston COS pricer.
# ---------------------------------------------------------------------------

from dheston.pricing.heston import price_standard_heston_numpy

k, th, sg, rho, v0 = 1.5, 0.04, 0.20, -0.5, 0.04
assert 2 * k * th - sg ** 2 > 0 and k * th - sg ** 2 > 0  # Feller on full and half factor
assert 2 * rho ** 2 < 1  # disk with the rho duplicated
double_vec = [k, th / 2, sg, rho, v0 / 2, k, th / 2, sg, rho, v0 / 2]
strikes = np.asarray([90.0, 95.0, 100.0, 105.0, 110.0])
tau = 0.75
r, q = 0.05, 0.0
dual_prices = [
    price_double_heston_call(100.0, K, tau, r, q, double_vec, enforce_ordering=False) for K in strikes
]
single_std = [v0, k, th, sg, rho]
config = FourierConfig(integration_steps=384, u_max=140.0)
single_prices = price_standard_heston_numpy(
    np.full_like(strikes, 100.0), strikes, np.full_like(strikes, tau),
    np.full_like(strikes, r), np.full_like(strikes, q), np.ones_like(strikes),
    np.asarray(single_std), config,
)
reduction = {
    "strikes": strikes.tolist(),
    "canonical_double_half_factors": [float(p) for p in dual_prices],
    "dheston_single_heston": [float(p) for p in single_prices],
    "max_abs_diff": float(np.max(np.abs(np.asarray(dual_prices) - np.asarray(single_prices)))),
}
log("one_factor_price_reduction", reduction)


# ---------------------------------------------------------------------------
# 6. Black-Scholes deterministic-variance limit (sigma -> small, rho -> small):
#    variance paths become deterministic; price -> BS with integrated variance.
# ---------------------------------------------------------------------------

def bs_call(S, K, tau, r, q, total_var):
    if total_var <= 0:
        return max(S * math.exp(-q * tau) - K * math.exp(-r * tau), 0.0)
    vol = math.sqrt(total_var / tau)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol ** 2) * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    return S * math.exp(-q * tau) * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * math.exp(-r * tau) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))


sg_small, rho_small = 0.02, -0.02
vec_bs = [0.8, 0.035, sg_small, rho_small, 0.030, 3.0, 0.025, sg_small, rho_small, 0.020]
bs_rows = []
for K in (90.0, 100.0, 110.0):
    for tau in (0.25, 1.0):
        total_var = sum(
            th * tau + (v0 - th) * (1 - math.exp(-kap * tau)) / kap
            for kap, th, v0 in ((0.8, 0.035, 0.030), (3.0, 0.025, 0.020))
        )
        dh = price_double_heston_call(100.0, K, tau, 0.05, 0.0, vec_bs)
        ref = bs_call(100.0, K, tau, 0.05, 0.0, total_var)
        bs_rows.append({"K": K, "tau": tau, "canonical": float(dh), "bs_limit": ref,
                        "abs_err": float(dh) - ref, "rel_err": (float(dh) - ref) / max(ref, 1e-8)})
log("black_scholes_limit", bs_rows)


# ---------------------------------------------------------------------------
# 7. Put-call parity on the canonical pricer (public API).
# ---------------------------------------------------------------------------

parity_rows = []
for K in (95.0, 100.0, 105.0):
    call = price_double_heston_call(100.0, K, 0.5, 0.05, 0.01, VEC_A)
    put = price_double_heston_put(100.0, K, 0.5, 0.05, 0.01, VEC_A)
    lhs = call - put
    rhs = 100.0 * math.exp(-0.01 * 0.5) - K * math.exp(-0.05 * 0.5)
    parity_rows.append({"K": K, "call-put": float(lhs), "S*e^-qT-K*e^-rT": rhs,
                        "abs_err": float(lhs) - rhs})
log("put_call_parity", parity_rows)


out_path = Path(__file__).with_name("probe_results.json")
out_path.write_text(json.dumps(RESULTS, indent=2, default=str))
print(f"\nSaved -> {out_path}")
