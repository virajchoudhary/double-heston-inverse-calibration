"""Node A diagnostic: cross-stack pricer agreement on the constraint intersection.

Compares three Double Heston pricers on identical canonical-order parameters:
  1. frozen production engine (src/double_heston.py, Gauss-Laguerre);
  2. Stack A differentiable torch mirror (src/torch_double_heston.py, Gauss-Laguerre);
  3. Stack B archive-2 COS pricer (src/dheston/pricing/heston.py).

Parameters are chosen inside the intersection of both stacks' constraint sets
(canonical structural validity AND archive-2 box bounds with negative rho).
CPU-only, seeded, small. Diagnostic evidence only — no training, no dataset writes.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from src.constants import CALL_OPTION, PARAMETER_NAMES
from src.double_heston import price_double_heston_option
from src.torch_double_heston import price_double_heston_option_tensor
from dheston.pricing.heston import FourierConfig, price_double_heston_torch as price_b_torch

torch.manual_seed(42)
np.random.default_rng(42)

# Canonical order: [kappa_s, theta_s, sigma_s, rho_s, v0_s, kappa_f, theta_f, sigma_f, rho_f, v0_f]
# Satisfies Stack A structural validity (Feller margins, disk, kappa ordering) AND
# Stack B boxes (negative rho, kappa1=fast in (0.3,8) >= kappa2=slow in (0.1,6)).
CANONICAL_PARAMS = {
    "kappa_slow": 0.8, "theta_slow": 0.04, "sigma_slow": 0.2, "rho_slow": -0.4, "v0_slow": 0.03,
    "kappa_fast": 3.0, "theta_fast": 0.05, "sigma_fast": 0.3, "rho_fast": -0.6, "v0_fast": 0.04,
}
assert 2 * 0.8 * 0.04 > 0.2**2 and 2 * 3.0 * 0.05 > 0.3**2, "Feller violated"
assert (-0.4) ** 2 + (-0.6) ** 2 < 1, "correlation disk violated"

canonical_vector = np.array([CANONICAL_PARAMS[name] for name in PARAMETER_NAMES], dtype=np.float64)
# Stack B order: [v01,kappa1,theta1,sigma1,rho1, v02,kappa2,theta2,sigma2,rho2], factor1=fast
b_order_vector = np.array([
    CANONICAL_PARAMS["v0_fast"], CANONICAL_PARAMS["kappa_fast"], CANONICAL_PARAMS["theta_fast"],
    CANONICAL_PARAMS["sigma_fast"], CANONICAL_PARAMS["rho_fast"],
    CANONICAL_PARAMS["v0_slow"], CANONICAL_PARAMS["kappa_slow"], CANONICAL_PARAMS["theta_slow"],
    CANONICAL_PARAMS["sigma_slow"], CANONICAL_PARAMS["rho_slow"],
], dtype=np.float64)

SPOT = 100.0
RATE = 0.05
DIV = 0.0
MATURITIES_Y = [7 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365]
LOG_M = [-0.30, -0.10, 0.0, 0.10, 0.30]

config_b = FourierConfig()

rows = []
max_abs_rel_a, max_abs_rel_b = 0.0, 0.0
for tau in MATURITIES_Y:
    for lm in LOG_M:
        strike = SPOT * float(np.exp(lm))
        ref = price_double_heston_option(SPOT, strike, tau, RATE, DIV, CALL_OPTION, canonical_vector)

        p_t = torch.tensor(SPOT, dtype=torch.float64)
        k_t = torch.tensor(strike, dtype=torch.float64)
        tau_t = torch.tensor(tau, dtype=torch.float64)
        r_t = torch.tensor(RATE, dtype=torch.float64)
        q_t = torch.tensor(DIV, dtype=torch.float64)
        params_t = torch.tensor(canonical_vector, dtype=torch.float64)
        stack_a = price_double_heston_option_tensor(
            p_t, k_t, tau_t, r_t, q_t, CALL_OPTION, params_t
        ).item()

        spot_b = torch.tensor([SPOT], dtype=torch.float64)
        strike_b = torch.tensor([strike], dtype=torch.float64)
        tau_b = torch.tensor([tau], dtype=torch.float64)
        rate_b = torch.tensor([RATE], dtype=torch.float64)
        div_b = torch.tensor([DIV], dtype=torch.float64)
        is_call_b = torch.tensor([1.0], dtype=torch.float64)
        params_b = torch.tensor(b_order_vector, dtype=torch.float64)
        stack_b = price_b_torch(
            spot_b, strike_b, tau_b, rate_b, div_b, is_call_b, params_b, config_b
        ).item()

        rel_a = (stack_a - ref) / ref
        rel_b = (stack_b - ref) / ref
        max_abs_rel_a = max(max_abs_rel_a, abs(rel_a))
        max_abs_rel_b = max(max_abs_rel_b, abs(rel_b))
        rows.append((tau, lm, ref, stack_a, rel_a, stack_b, rel_b))

print(f"{'tau':>7} {'logm':>6} {'prod':>10} {'A(GLQ)':>10} {'relA':>11} {'B(COS)':>10} {'relB':>11}")
for tau, lm, ref, a, rel_a, b, rel_b in rows:
    print(f"{tau:7.4f} {lm:6.2f} {ref:10.5f} {a:10.5f} {rel_a:11.3e} {b:10.5f} {rel_b:11.3e}")

print(f"\nmax |rel| Stack A torch GLQ vs frozen production: {max_abs_rel_a:.3e}")
print(f"max |rel| Stack B COS vs frozen production:        {max_abs_rel_b:.3e}")

# Gradients flow check for both differentiable pricers (architecture requirement).
params_t.requires_grad_(True)
price_double_heston_option_tensor(
    p_t, k_t, tau_t, r_t, q_t, CALL_OPTION, params_t
).backward()
print(f"\nStack A grad wrt canonical params (x1e6): {np.array2string(params_t.grad.numpy() * 1e6, precision=2)}")

params_b.requires_grad_(True)
price_b_torch(spot_b, strike_b, tau_b, rate_b, div_b, is_call_b, params_b, config_b).backward()
print(f"Stack B grad wrt B-order params (x1e6):   {np.array2string(params_b.grad.numpy() * 1e6, precision=2)}")
