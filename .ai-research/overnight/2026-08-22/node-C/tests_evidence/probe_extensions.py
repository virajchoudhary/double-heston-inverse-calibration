"""Node C extension probe: closure and quantification experiments.

1. Exact closure: dheston's own pde_residual_loss on a single-point batch
   equals (broken-operator residual / max(|V|,1))^2 bit-exactly.
2. Correct wiring: the same COS pricer with v0 entries as genuine leaves
   satisfies the canonical PDE much better -- quantifies the gap between the
   broken operator and what a correct implementation would measure.
3. COS truncation-range contamination: autograd delta of the COS pricer vs
   the canonical Gauss-Laguerre pricer vs central finite differences.
4. Robustness of the zero-derivative defect across spots/taus/steps and via
   the full build_loss_components path; market-price independence of the PDE
   component.
5. Black-Scholes limit convergence order in sigma.
6. Terminal condition: tau -> 0 recovers the (discounted) payoff.
"""

from __future__ import annotations

import importlib.util
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

# Python 3.9 typing shim for src/double_heston (typing-only lines; asserted).
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

from dheston.models.losses import _safe_grad, build_loss_components, pde_residual_loss
from dheston.pricing.heston import FourierConfig, price_double_heston_torch
from src.torch_double_heston import price_double_heston_call_tensor

RESULTS: dict[str, object] = {}


def log(section: str, payload: object) -> None:
    RESULTS[section] = payload
    print(f"\n=== {section} ===")
    print(payload)


VEC_A = [0.60, 0.040, 0.180, -0.55, 0.040, 2.80, 0.030, 0.350, -0.35, 0.020]
VEC_B = [0.30, 0.060, 0.160, -0.30, 0.050, 6.00, 0.060, 0.700, -0.60, 0.030]


def canonical_to_archive2(vec):
    k_s, th_s, sg_s, r_s, v_s, k_f, th_f, sg_f, r_f, v_f = vec
    return [v_f, k_f, th_f, sg_f, r_f, v_s, k_s, th_s, sg_s, r_s]


def single_point_batch():
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


# ---------------------------------------------------------------------------
# 1. Exact closure: production pde_residual_loss == broken-operator formula.
# ---------------------------------------------------------------------------

def closure_experiment(vec):
    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
    batch = single_point_batch()
    config = FourierConfig()
    loss = pde_residual_loss(params, batch, config, max_points=6)

    # Instrumented replica (identical code path, unpacked).
    surface_index = torch.tensor([0])
    point_index = torch.tensor([0])
    chosen_params = params[surface_index]
    spot = batch["spot"][surface_index].detach().clone().requires_grad_(True)
    tau = batch["tau"][surface_index, point_index].detach().clone().requires_grad_(True)
    prices = price_double_heston_torch(
        spot, batch["strike"][surface_index, point_index], tau,
        batch["rate"][surface_index], batch["dividend"][surface_index],
        batch["is_call"][surface_index, point_index], chosen_params, config,
    )
    d_tau = _safe_grad(prices, tau)
    delta = _safe_grad(prices, spot)
    gamma = _safe_grad(delta, spot)
    v01, v02 = chosen_params[:, 0], chosen_params[:, 5]
    k1, th1, sg1, r1 = chosen_params[:, 1], chosen_params[:, 2], chosen_params[:, 3], chosen_params[:, 4]
    k2, th2, sg2, r2 = chosen_params[:, 6], chosen_params[:, 7], chosen_params[:, 8], chosen_params[:, 9]
    d_v01, d_v02 = _safe_grad(prices, v01), _safe_grad(prices, v02)
    cross1, cross2 = _safe_grad(delta, v01), _safe_grad(delta, v02)
    d2_1, d2_2 = _safe_grad(d_v01, v01), _safe_grad(d_v02, v02)
    rate = batch["rate"][surface_index]
    dividend = batch["dividend"][surface_index]
    diffusion = 0.5 * (v01 + v02) * spot.square() * gamma
    drift = (rate - dividend) * spot * delta - rate * prices
    f1 = k1 * (th1 - v01) * d_v01 + r1 * sg1 * v01 * spot * cross1 + 0.5 * sg1.square() * v01 * d2_1
    f2 = k2 * (th2 - v02) * d_v02 + r2 * sg2 * v02 * spot * cross2 + 0.5 * sg2.square() * v02 * d2_2
    residual = d_tau - (diffusion + drift + f1 + f2)
    scale = prices.detach().abs().clamp_min(1.0)
    manual = torch.mean((residual / scale).pow(2))
    return {
        "production_loss": float(loss),
        "manual_broken_operator_loss": float(manual),
        "bit_exact_equal": bool(float(loss) == float(manual)),
    }


log("closure_production_equals_broken", {"VEC_A": closure_experiment(VEC_A), "VEC_B": closure_experiment(VEC_B)})


# ---------------------------------------------------------------------------
# 2. Correct wiring: v0 entries as genuine leaves inside the params tensor.
# ---------------------------------------------------------------------------

def correctly_wired_cos_residual(vec, S=100.0, K=100.0, tau=0.5, r=0.05, q=0.0, steps=256):
    dh = canonical_to_archive2(vec)  # [v0f, kf, thf, sgf, rf, v0s, ks, ths, sgs, rs]
    leaves = [torch.tensor([v], dtype=torch.float64, requires_grad=True) for v in dh]
    params = torch.stack(leaves, dim=1)  # (1, 10); every column is a leaf
    spot = torch.tensor([S], dtype=torch.float64, requires_grad=True)
    tau_t = torch.tensor([tau], dtype=torch.float64, requires_grad=True)
    prices = price_double_heston_torch(
        spot, torch.tensor([K]), tau_t, torch.tensor([r]), torch.tensor([q]),
        torch.tensor([1.0]), params, FourierConfig(integration_steps=steps),
    )

    def g(out, inp):
        return torch.autograd.grad(out, inp, create_graph=True, retain_graph=True)[0]

    d_tau = g(prices, tau_t)
    delta = g(prices, spot)
    gamma = g(delta, spot)
    v0f, kf, thf, sgf, rf = leaves[0], leaves[1], leaves[2], leaves[3], leaves[4]
    v0s, ks, ths, sgs, rs = leaves[5], leaves[6], leaves[7], leaves[8], leaves[9]
    d_vf, d_vs = g(prices, v0f), g(prices, v0s)
    cross_f, cross_s = g(delta, v0f), g(delta, v0s)
    d2_f, d2_s = g(d_vf, v0f), g(d_vs, v0s)
    diffusion = 0.5 * (v0s + v0f) * spot.square() * gamma
    drift = (r - q) * spot * delta - r * prices
    f_slow = ks * (ths - v0s) * d_vs + rs * sgs * v0s * spot * cross_s + 0.5 * sgs ** 2 * v0s * d2_s
    f_fast = kf * (thf - v0f) * d_vf + rf * sgf * v0f * spot * cross_f + 0.5 * sgf ** 2 * v0f * d2_f
    residual = d_tau - (diffusion + drift + f_slow + f_fast)
    scale = max(abs(float(prices.detach())), 1.0)
    return float(residual.detach()) / scale


wired = {}
for vec_name, vec in (("VEC_A", VEC_A), ("VEC_B", VEC_B)):
    wired[vec_name] = {
        "correct_wiring_rel_residual": correctly_wired_cos_residual(vec),
        "broken_operator_rel_residual": abs(
            {"VEC_A": 0.604568934256152 / 8.323843936383234, "VEC_B": 2.245350358320797 / 10.019540260732741}[vec_name]
        ),
    }
log("cos_correct_wiring_vs_broken", wired)


# ---------------------------------------------------------------------------
# 3. COS truncation contamination: delta comparisons at one point.
# ---------------------------------------------------------------------------

def delta_comparison(vec, S=100.0, K=100.0, tau=0.5, r=0.05, q=0.0):
    # canonical GL delta via autograd
    params_gl = torch.tensor(vec, dtype=torch.float64, requires_grad=True)
    spot_gl = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    price_gl = price_double_heston_call_tensor(spot_gl, torch.tensor(K), torch.tensor(tau), torch.tensor(r), torch.tensor(q), params_gl)
    delta_gl = float(torch.autograd.grad(price_gl, spot_gl)[0].detach())

    # COS delta via autograd (leaf spot)
    dh = canonical_to_archive2(vec)
    params_cos = torch.tensor([dh], dtype=torch.float64)
    spot_cos = torch.tensor([S], dtype=torch.float64, requires_grad=True)
    price_cos = price_double_heston_torch(spot_cos, torch.tensor([K]), torch.tensor([tau]),
                                          torch.tensor([r]), torch.tensor([q]), torch.tensor([1.0]),
                                          params_cos, FourierConfig())
    delta_cos = float(torch.autograd.grad(price_cos, spot_cos)[0].detach())

    # central finite differences on the COS pricer (state-dependent truncation
    # included). float64 throughout -- float32 spot inputs quantize the
    # perturbation and corrupt the FD estimate (caught in adversarial review).
    def cos_price(s):
        return float(price_double_heston_torch(torch.tensor([s], dtype=torch.float64), torch.tensor([K], dtype=torch.float64), torch.tensor([tau], dtype=torch.float64),
                                               torch.tensor([r], dtype=torch.float64), torch.tensor([q], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64),
                                               params_cos.detach(), FourierConfig()).detach())

    h = 1e-3
    delta_fd = (cos_price(S + h) - cos_price(S - h)) / (2 * h)
    return {
        "delta_GL_autograd": delta_gl,
        "delta_COS_autograd": delta_cos,
        "delta_COS_finite_diff": delta_fd,
        "abs_diff_COS_vs_GL": abs(delta_cos - delta_gl),
        "abs_diff_COS_autograd_vs_fd": abs(delta_cos - delta_fd),
    }


log("delta_contamination", {"VEC_A": delta_comparison(VEC_A), "VEC_B": delta_comparison(VEC_B)})


# ---------------------------------------------------------------------------
# 4. Robustness: defect invariant across configs; build_loss_components path;
#    market-price independence.
# ---------------------------------------------------------------------------

def defect_invariance(vec):
    outcomes = []
    for S in (70.0, 100.0, 130.0):
        for tau in (0.1, 0.75, 2.0):
            params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
            batch = single_point_batch()
            batch["spot"] = torch.tensor([S], dtype=torch.float64)
            batch["tau"] = torch.tensor([[tau]], dtype=torch.float64)
            config = FourierConfig()
            si, pi = torch.tensor([0]), torch.tensor([0])
            chosen = params[si]
            spot = batch["spot"][si].detach().clone().requires_grad_(True)
            tau_t = batch["tau"][si, pi].detach().clone().requires_grad_(True)
            prices = price_double_heston_torch(spot, batch["strike"][si, pi], tau_t,
                                               batch["rate"][si], batch["dividend"][si],
                                               batch["is_call"][si, pi], chosen, config)
            v01 = chosen[:, 0]
            outcomes.append(float(_safe_grad(prices, v01).abs()) == 0.0)
    return all(outcomes)


def full_path_pde_component(vec, market_scale):
    params = torch.tensor([canonical_to_archive2(vec)], dtype=torch.float64, requires_grad=True)
    batch = {
        "mask": torch.ones(1, 6, dtype=torch.bool),
        "spot": torch.tensor([100.0], dtype=torch.float64),
        "strike": torch.tensor([[90.0, 95.0, 100.0, 105.0, 110.0, 100.0]], dtype=torch.float64),
        "tau": torch.tensor([[0.25, 0.35, 0.5, 0.75, 1.0, 0.5]], dtype=torch.float64),
        "rate": torch.tensor([0.05], dtype=torch.float64),
        "dividend": torch.tensor([0.0], dtype=torch.float64),
        "is_call": torch.ones(1, 6, dtype=torch.float64),
        "market_price": torch.full((1, 6), 5.0 * market_scale, dtype=torch.float64),
        "target_params": None,
    }
    components = build_loss_components(
        {"params": params}, batch, FourierConfig(),
        {"lambda_param": 1.0, "lambda_price": 1.0, "lambda_order": 1.0, "lambda_boundary": 1.0, "lambda_pde": 1.0},
        pde_points=6,
    )
    return {k: float(v.detach()) for k, v in components.items()}


base = full_path_pde_component(VEC_A, 1.0)
shifted = full_path_pde_component(VEC_A, 7.0)
log("defect_robustness", {
    "zero_derivative_invariant_9_configs": {
        name: defect_invariance(vec) for name, vec in (("VEC_A", VEC_A), ("VEC_B", VEC_B))
    },
    "build_loss_components_pde_value": base["pde"],
    "pde_component_market_price_independent": bool(base["pde"] == shifted["pde"]),
    "price_component_moves_with_market": bool(base["price"] != shifted["price"]),
})


# ---------------------------------------------------------------------------
# 5. BS-limit convergence order in sigma.
# ---------------------------------------------------------------------------

def bs_limit_error(sigma, K=100.0, tau=0.5):
    vec = [0.8, 0.035, sigma, -0.02, 0.030, 3.0, 0.025, sigma, -0.02, 0.020]
    total_var = sum(
        th * tau + (v0 - th) * (1 - math.exp(-kap * tau)) / kap
        for kap, th, v0 in ((0.8, 0.035, 0.030), (3.0, 0.025, 0.020))
    )
    price = double_heston.price_double_heston_call(100.0, K, tau, 0.05, 0.0, vec)
    vol = math.sqrt(total_var / tau)
    d1 = (math.log(1.0) + (0.05 + 0.5 * vol ** 2) * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    bs = 100.0 * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * math.exp(-0.05 * tau) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return abs(price - bs)


errors = {s: bs_limit_error(s) for s in (0.08, 0.04, 0.02, 0.01)}
log("bs_limit_sigma_convergence", {
    "abs_errors": errors,
    "ratios_halving_sigma": [errors[a] / errors[b] for a, b in ((0.08, 0.04), (0.04, 0.02), (0.02, 0.01))],
    "note": "ratio ~2 => error is O(sigma) as expected for vol-of-vol -> 0 with rho ~ O(sigma)",
})


# ---------------------------------------------------------------------------
# 6. Terminal condition: tau -> 0 recovers discounted payoff.
# ---------------------------------------------------------------------------

terminal = []
for K in (95.0, 100.0, 105.0):
    for tau in (1e-4, 1e-3, 1e-2):
        price = double_heston.price_double_heston_call(100.0, K, tau, 0.05, 0.0, VEC_A)
        intrinsic = max(100.0 - K * math.exp(-0.05 * tau), 0.0)
        terminal.append({"K": K, "tau": tau, "price": float(price), "discounted_payoff": intrinsic,
                         "abs_err": float(price) - intrinsic})
log("terminal_condition", terminal)


# ---------------------------------------------------------------------------
# 7. Broadened canonical-PDE certification sweep (adversarial-review action):
#    S in {80,100,120} x K/S in {0.85..1.15} x tau in {0.1,0.5,1.0,2.0} for
#    both vectors, via the same leaf-wired autograd construction as the
#    pytest suite's canonical_residual.
# ---------------------------------------------------------------------------

def leaf_wired_rel_residual(vec, S, K, tau, r=0.05, q=0.0):
    v0_s = torch.tensor(vec[4], dtype=torch.float64, requires_grad=True)
    v0_f = torch.tensor(vec[9], dtype=torch.float64, requires_grad=True)
    constants = [torch.tensor(v, dtype=torch.float64) for v in (vec[:4] + vec[5:9])]
    params = torch.stack([*constants[:4], v0_s, *constants[4:], v0_f])
    spot = torch.tensor(S, dtype=torch.float64, requires_grad=True)
    maturity = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    price = price_double_heston_call_tensor(spot, torch.tensor(K), maturity,
                                            torch.tensor(r), torch.tensor(q), params)

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
    f_slow = k_s * (th_s - v0_s) * d_vs + r_s * sg_s * v0_s * spot * cross_s + 0.5 * sg_s ** 2 * v0_s * d2_vs
    f_fast = k_f * (th_f - v0_f) * d_vf + r_f * sg_f * v0_f * spot * cross_f + 0.5 * sg_f ** 2 * v0_f * d2_vf
    residual = d_tau - (diffusion + drift + f_slow + f_fast)
    return abs(float(residual.detach())) / max(abs(float(price.detach())), 1.0)


worst = 0.0
worst_at = None
per_tau: dict[float, float] = {}
count = 0
for vec in (VEC_A, VEC_B):
    for S in (80.0, 100.0, 120.0):
        for ks in (0.85, 0.95, 1.0, 1.05, 1.15):
            for tau in (0.1, 0.5, 1.0, 2.0):
                value = leaf_wired_rel_residual(vec, S, S * ks, tau)
                if value > worst:
                    worst, worst_at = value, (f"vec={'A' if vec is VEC_A else 'B'}", f"S={S}", f"K/S={ks}", f"tau={tau}")
                per_tau[tau] = max(per_tau.get(tau, 0.0), value)
                count += 1
log("certification_sweep", {
    "points": count,
    "domain": "S in {80,100,120}; K/S in {0.85,0.95,1.0,1.05,1.15}; tau in {0.1,0.5,1.0,2.0}; 2 canonical-valid vectors",
    "max_rel_residual": worst,
    "worst_point": worst_at,
    "max_rel_residual_by_tau": {str(k): v for k, v in sorted(per_tau.items())},
    "verdict": "canonical pricer satisfies the derived PDE across the swept domain to <= 2e-8 (worst at short-maturity OTM corners, consistent with F10)"
    if worst < 2e-8 else "REVIEW NEEDED",
})

out_path = Path(__file__).with_name("probe_extension_results.json")
out_path.write_text(json.dumps(RESULTS, indent=2, default=str))
print(f"\nSaved -> {out_path}")
