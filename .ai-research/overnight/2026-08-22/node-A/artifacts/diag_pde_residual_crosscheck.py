"""Node A diagnostic: PDE residual cross-check across pricers and differentiation methods.

Motivation: Node A's provisional conclusion (F2 #3, F14) is that archive-2's PDE-residual
loss is noise-dominated when differentiated through the COS pricer. Treat as provisional
until independently verified (Node C absent at time of writing). This diagnostic separates
three possible failure modes:

  (1) GLQ autograd residual ~0, COS autograd residual large  -> COS/autograd noise (confirms)
  (2) GLQ autograd large, production FD small                -> autograd machinery issue
  (3) both autograd and FD large                             -> residual formula suspect

Residual formula mirrors dheston/models/losses.py:78-134 exactly (canonical indices):
  residual = V_tau - [ 0.5(v_s+v_f)S^2 V_SS + (r-q)S V_S - rV
                       + sum_i ( k_i(t_i-v_i) V_vi + rho_i s_i v_i S V_Svi + 0.5 s_i^2 v_i V_vivi ) ]
normalized by max(|V|,1), squared, averaged over points (the smoke-run reported 8.91).

CPU-only, seeded, no training. Diagnostic evidence only.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from src.constants import CALL_OPTION, PARAMETER_NAMES
from src.double_heston import price_double_heston_option as price_prod
from src.torch_double_heston import price_double_heston_option_tensor as price_glq
from dheston.pricing.heston import FourierConfig, price_double_heston_torch as price_cos

torch.manual_seed(42)

CANONICAL = {
    "kappa_slow": 0.8, "theta_slow": 0.04, "sigma_slow": 0.2, "rho_slow": -0.4, "v0_slow": 0.03,
    "kappa_fast": 3.0, "theta_fast": 0.05, "sigma_fast": 0.3, "rho_fast": -0.6, "v0_fast": 0.04,
}
vec = np.array([CANONICAL[n] for n in PARAMETER_NAMES], dtype=np.float64)
b_vec = np.array([
    CANONICAL["v0_fast"], CANONICAL["kappa_fast"], CANONICAL["theta_fast"], CANONICAL["sigma_fast"], CANONICAL["rho_fast"],
    CANONICAL["v0_slow"], CANONICAL["kappa_slow"], CANONICAL["theta_slow"], CANONICAL["sigma_slow"], CANONICAL["rho_slow"],
])
SPOT, RATE, DIV = 100.0, 0.05, 0.0
POINTS = [(90.0, 30 / 365), (100.0, 30 / 365), (110.0, 30 / 365), (90.0, 180 / 365), (100.0, 180 / 365), (110.0, 180 / 365)]


def residual_from_derivs(V, dtau, delta, gamma, dv_s, dv_f, cross_s, cross_f, d2v_s, d2v_f, S, p):
    diffusion = 0.5 * (p["v0_slow"] + p["v0_fast"]) * S**2 * gamma
    drift = (RATE - DIV) * S * delta - RATE * V
    f_slow = p["kappa_slow"] * (p["theta_slow"] - p["v0_slow"]) * dv_s + p["rho_slow"] * p["sigma_slow"] * p["v0_slow"] * S * cross_s + 0.5 * p["sigma_slow"] ** 2 * p["v0_slow"] * d2v_s
    f_fast = p["kappa_fast"] * (p["theta_fast"] - p["v0_fast"]) * dv_f + p["rho_fast"] * p["sigma_fast"] * p["v0_fast"] * S * cross_f + 0.5 * p["sigma_fast"] ** 2 * p["v0_fast"] * d2v_f
    return dtau - (diffusion + drift + f_slow + f_fast)


def glq_autograd_residual(K, tau):
    S = torch.tensor(SPOT, dtype=torch.float64, requires_grad=True)
    T = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    P = torch.tensor(vec, dtype=torch.float64, requires_grad=True)
    Vt = price_glq(S, torch.tensor(K, dtype=torch.float64), T, torch.tensor(RATE, dtype=torch.float64), torch.tensor(DIV, dtype=torch.float64), CALL_OPTION, P)
    V = Vt.detach().item()

    def grad(out, inp):
        return torch.autograd.grad(out, inp, create_graph=True, retain_graph=True)[0]

    dtau = grad(Vt, T)
    delta = grad(Vt, S)
    gamma = grad(delta, S)
    dv = grad(Vt, P)
    dv_s, dv_f = dv[4], dv[9]
    cross_s, cross_f = grad(delta, P)[4], grad(delta, P)[9]
    d2v_s, d2v_f = grad(dv_s, P)[4], grad(dv_f, P)[9]
    r = residual_from_derivs(V, dtau.item(), delta.item(), gamma.item(), dv_s.item(), dv_f.item(),
                             cross_s.item(), cross_f.item(), d2v_s.item(), d2v_f.item(), SPOT, CANONICAL)
    return V, r


def cos_autograd_residual(K, tau):
    cfg = FourierConfig()
    S = torch.tensor([SPOT], dtype=torch.float64).detach().clone().requires_grad_(True)
    T = torch.tensor([tau], dtype=torch.float64).detach().clone().requires_grad_(True)
    P = torch.tensor(b_vec, dtype=torch.float64, requires_grad=True)
    Vt = price_cos(S, torch.tensor([K], dtype=torch.float64), T, torch.tensor([RATE], dtype=torch.float64),
                   torch.tensor([DIV], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64), P, cfg)
    V = Vt.detach().item()

    def grad(out, inp):
        g = torch.autograd.grad(out, inp, create_graph=True, retain_graph=True, allow_unused=True)[0]
        return torch.zeros_like(inp) if g is None else g

    dtau = grad(Vt, T)
    delta = grad(Vt, S)
    gamma = grad(delta, S)
    dv = grad(Vt, P)
    dv_s, dv_f = dv[5], dv[0]  # B order: v02=slow idx5, v01=fast idx0
    cross = grad(delta, P)
    cross_s, cross_f = cross[5], cross[0]
    d2v_s, d2v_f = grad(dv_s, P)[5], grad(dv_f, P)[0]
    r = residual_from_derivs(V, dtau.item(), delta.item(), gamma.item(), dv_s.item(), dv_f.item(),
                             cross_s.item(), cross_f.item(), d2v_s.item(), d2v_f.item(), SPOT, CANONICAL)
    return V, r


def production_fd_residual(K, tau, h=1e-4):
    def pr(spot, tau_, v0_s=None, v0_f=None):
        p = vec.copy()
        if v0_s is not None:
            p[4] = v0_s
        if v0_f is not None:
            p[9] = v0_f
        return price_prod(spot, K, tau_, RATE, DIV, CALL_OPTION, p)

    V = pr(SPOT, tau)
    dtau = (pr(SPOT, tau + h) - pr(SPOT, tau - h)) / (2 * h)
    delta = (pr(SPOT + h, tau) - pr(SPOT - h, tau)) / (2 * h)
    gamma = (pr(SPOT + h, tau) - 2 * V + pr(SPOT - h, tau)) / h**2
    vs, vf = CANONICAL["v0_slow"], CANONICAL["v0_fast"]
    dv_s = (pr(SPOT, tau, v0_s=vs + h) - pr(SPOT, tau, v0_s=vs - h)) / (2 * h)
    dv_f = (pr(SPOT, tau, v0_f=vf + h) - pr(SPOT, tau, v0_f=vf - h)) / (2 * h)
    d2v_s = (pr(SPOT, tau, v0_s=vs + h) - 2 * V + pr(SPOT, tau, v0_s=vs - h)) / h**2
    d2v_f = (pr(SPOT, tau, v0_f=vf + h) - 2 * V + pr(SPOT, tau, v0_f=vf - h)) / h**2
    dpm_s = (pr(SPOT + h, tau, v0_s=vs + h) - pr(SPOT + h, tau, v0_s=vs - h)) / (2 * h)
    dmm_s = (pr(SPOT - h, tau, v0_s=vs + h) - pr(SPOT - h, tau, v0_s=vs - h)) / (2 * h)
    cross_s = (dpm_s - dmm_s) / (2 * h)
    dpm_f = (pr(SPOT + h, tau, v0_f=vf + h) - pr(SPOT + h, tau, v0_f=vf - h)) / (2 * h)
    dmm_f = (pr(SPOT - h, tau, v0_f=vf + h) - pr(SPOT - h, tau, v0_f=vf - h)) / (2 * h)
    cross_f = (dpm_f - dmm_f) / (2 * h)
    r = residual_from_derivs(V, dtau, delta, gamma, dv_s, dv_f, cross_s, cross_f, d2v_s, d2v_f, SPOT, CANONICAL)
    return V, r


print(f"{'K':>6} {'tau':>7} | {'V':>9} | {'res GLQ-auto':>13} {'res COS-auto':>13} {'res prod-FD':>13} | normalized^2 (GLQ/COS/FD)")
sq_glq, sq_cos, sq_fd = [], [], []
for K, tau in POINTS:
    V1, r1 = glq_autograd_residual(K, tau)
    V2, r2 = cos_autograd_residual(K, tau)
    V3, r3 = production_fd_residual(K, tau)
    s = max(abs(V1), 1.0)
    sq_glq.append((r1 / s) ** 2)
    sq_cos.append((r2 / s) ** 2)
    sq_fd.append((r3 / s) ** 2)
    print(f"{K:6.0f} {tau:7.4f} | {V1:9.4f} | {r1:13.3e} {r2:13.3e} {r3:13.3e} | {sq_glq[-1]:.3e} / {sq_cos[-1]:.3e} / {sq_fd[-1]:.3e}")

print(f"\nmean normalized^2 residual  GLQ-autograd: {np.mean(sq_glq):.4e}")
print(f"mean normalized^2 residual  COS-autograd: {np.mean(sq_cos):.4e}   (smoke-run analogue: 8.91)")
print(f"mean normalized^2 residual  production-FD: {np.mean(sq_fd):.4e}")
