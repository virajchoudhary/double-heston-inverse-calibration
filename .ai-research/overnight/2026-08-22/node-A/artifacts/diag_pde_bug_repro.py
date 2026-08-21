"""Node A artifact: minimal reproduction of the archive-2 PDE-residual autograd bug.

CLAIM (F19): dheston pde_residual_loss silently zeroes ALL variance-factor derivative
terms because it differentiates wrt slice views (chosen_params[:, 0]) that are not on
the executed autograd graph; _safe_grad converts the resulting None into zeros. The
implemented residual is therefore d_tau - (diffusion + drift), a WRONG PDE missing both
factor terms. Correctly computed (differentiate wrt the leaf, then slice), the residual
is machine-zero for an accurate pricer and thus non-discriminating for parameters.

Run directly; CPU; seeded; no training. Evidence for the morning report / Node C.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from dheston.models.losses import pde_residual_loss, _safe_grad
from dheston.pricing.heston import FourierConfig, price_double_heston_torch

torch.manual_seed(42)
cfg = FourierConfig()
MID = [0.305, 4.15, 0.305, 0.775, -0.5, 0.305, 3.05, 0.305, 0.775, -0.5]  # B-order, factor1=fast
S0, RATE, DIV, TAU = 344.35, 0.06, 0.0, 0.03561644
STRIKES = [315.0, 327.5, 345.0]


def part1_repo_loss_is_large_in_both_dtypes() -> None:
    print("== Part 1: repo pde_residual_loss on a minimal synthetic put batch ==")
    for dtype in (torch.float32, torch.float64):
        batch = {
            "spot": torch.tensor([S0], dtype=dtype),
            "strike": torch.tensor([STRIKES], dtype=dtype),
            "tau": torch.tensor([[TAU] * 3], dtype=dtype),
            "rate": torch.tensor([RATE], dtype=dtype),
            "dividend": torch.tensor([DIV], dtype=dtype),
            "is_call": torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype),
            "market_price": torch.tensor([[8.0, 12.0, 20.0]], dtype=dtype),
            "mask": torch.tensor([[True, True, True]]),
            "target_params": None,
        }
        params = torch.tensor(MID, dtype=dtype).unsqueeze(0).requires_grad_(True)
        loss = pde_residual_loss(params, batch, cfg, max_points=3)
        print(f"  dtype={str(dtype):16s} loss={loss.item():.6e}")


def part2_v0_derivatives_are_zero() -> None:
    print("== Part 2: inside the repo mechanics, v0-derivative terms are exactly zero ==")
    dt = torch.float64
    parameters = torch.tensor(MID, dtype=dt).unsqueeze(0).requires_grad_(True)
    surface_index = torch.tensor([0, 0, 0])
    spot = torch.full((3,), S0, dtype=dt).requires_grad_(True)
    tau = torch.full((3,), TAU, dtype=dt).requires_grad_(True)
    chosen = parameters[surface_index]
    v01 = chosen[:, 0]
    prices = price_double_heston_torch(
        spot, torch.tensor(STRIKES, dtype=dt), tau,
        torch.full((3,), RATE, dtype=dt), torch.full((3,), DIV, dtype=dt),
        torch.zeros(3, dtype=dt), chosen, cfg,
    )
    d_v01 = _safe_grad(prices, v01)
    print(f"  v01.requires_grad: {v01.requires_grad}  (guard passes)")
    g_view = torch.autograd.grad(prices, v01, grad_outputs=torch.ones_like(prices), allow_unused=True, retain_graph=True)[0]
    g_leaf = torch.autograd.grad(prices, parameters, grad_outputs=torch.ones_like(prices), allow_unused=True, retain_graph=True)[0]
    print(f"  grad(prices, slice-view v01): {'None -> _safe_grad returns zeros' if g_view is None else g_view.tolist()}")
    print(f"  grad(prices, leaf)[:, 0]    : {[f'{v:.4f}' for v in g_leaf[:, 0].tolist()]}  (healthy)")
    print(f"  d_v01 via _safe_grad        : {d_v01.tolist()}  (silently zero)")
    d_tau = _safe_grad(prices, tau)
    delta = _safe_grad(prices, spot)
    gamma = _safe_grad(delta, spot)
    k1, t1, s1, r1 = MID[1], MID[2], MID[3], MID[4]
    k2, t2, s2, r2 = MID[6], MID[7], MID[8], MID[9]
    residual = d_tau - (0.5 * (MID[0] + MID[5]) * S0**2 * gamma + (RATE - DIV) * S0 * delta - RATE * prices)
    print(f"  implemented residual (factors dropped): {[f'{v:.3f}' for v in residual.tolist()]}")
    # full true factor terms via per-point leaf differentiation
    chosen_leaf = torch.tensor(MID, dtype=dt).repeat(3, 1).requires_grad_(True)
    spot_l = torch.full((3,), S0, dtype=dt).requires_grad_(True)
    tau_l = torch.full((3,), TAU, dtype=dt).requires_grad_(True)
    prices_l = price_double_heston_torch(spot_l, torch.tensor(STRIKES, dtype=dt), tau_l,
                                         torch.full((3,), RATE, dtype=dt), torch.full((3,), DIV, dtype=dt),
                                         torch.zeros(3, dtype=dt), chosen_leaf, cfg)
    delta_l = torch.autograd.grad(prices_l, spot_l, grad_outputs=torch.ones_like(prices_l), create_graph=True, retain_graph=True)[0]
    dv_full = torch.autograd.grad(prices_l, chosen_leaf, grad_outputs=torch.ones_like(prices_l), create_graph=True, retain_graph=True)[0]
    cross_full = torch.autograd.grad(delta_l, chosen_leaf, grad_outputs=torch.ones_like(delta_l), create_graph=True, retain_graph=True)[0]
    d2v1 = torch.autograd.grad(dv_full[:, 0], chosen_leaf, grad_outputs=torch.ones_like(dv_full[:, 0]), retain_graph=True)[0]
    d2v2 = torch.autograd.grad(dv_full[:, 5], chosen_leaf, grad_outputs=torch.ones_like(dv_full[:, 5]), retain_graph=True)[0]
    f1 = k1 * (t1 - MID[0]) * dv_full[:, 0] + r1 * s1 * MID[0] * S0 * cross_full[:, 0] + 0.5 * s1**2 * MID[0] * d2v1[:, 0]
    f2 = k2 * (t2 - MID[5]) * dv_full[:, 5] + r2 * s2 * MID[5] * S0 * cross_full[:, 5] + 0.5 * s2**2 * MID[5] * d2v2[:, 5]
    print(f"  true (dropped) f1+f2 per point        : {[f'{v:.3f}' for v in (f1 + f2).tolist()]}")
    print(f"  residual - (f1+f2) per point          : {[f'{v:.3f}' for v in (residual - (f1 + f2)).tolist()]}  (~0 => residual == dropped terms)")


def part3_correct_residual_is_machine_zero() -> None:
    print("== Part 3: correctly-wired residual (differentiate the LEAF) is machine-zero ==")
    dt = torch.float64
    spot = torch.tensor([S0], dtype=dt).requires_grad_(True)
    tau = torch.tensor([TAU], dtype=dt).requires_grad_(True)
    params = torch.tensor(MID, dtype=dt).requires_grad_(True)
    V = price_double_heston_torch(spot, torch.tensor([STRIKES[0]], dtype=dt), tau,
                                  torch.tensor([RATE], dtype=dt), torch.tensor([DIV], dtype=dt),
                                  torch.tensor([0.0], dtype=dt), params, cfg)

    def grad(o, i):
        return torch.autograd.grad(o, i, create_graph=True, retain_graph=True)[0]

    dtau, delta = grad(V, tau), grad(V, spot)
    gamma = grad(delta, spot)
    dv = grad(V, params)
    cross = grad(delta, params)
    d2v1 = grad(dv[0], params)[0]
    d2v2 = grad(dv[5], params)[5]
    k1, t1, s1, r1 = MID[1], MID[2], MID[3], MID[4]
    k2, t2, s2, r2 = MID[6], MID[7], MID[8], MID[9]
    diffusion = 0.5 * (MID[0] + MID[5]) * S0**2 * gamma
    drift = (RATE - DIV) * S0 * delta - RATE * V
    f1 = k1 * (t1 - MID[0]) * dv[0] + r1 * s1 * MID[0] * S0 * cross[0] + 0.5 * s1**2 * MID[0] * d2v1
    f2 = k2 * (t2 - MID[5]) * dv[5] + r2 * s2 * MID[5] * S0 * cross[5] + 0.5 * s2**2 * MID[5] * d2v2
    res = dtau - (diffusion + drift + f1 + f2)
    print(f"  residual at K={STRIKES[0]}: {res.item():.3e}  (vs implemented-path ~9.6 at same point)")


if __name__ == "__main__":
    part1_repo_loss_is_large_in_both_dtypes()
    part2_v0_derivatives_are_zero()
    part3_correct_residual_is_machine_zero()
    print("\nCONCLUSION: implemented PDE residual = d_tau - (diffusion + drift); factor terms")
    print("silently zeroed by the autograd slice-view bug (None -> zeros in _safe_grad).")
    print("Correctly computed, an accurate pricer satisfies the PDE to machine precision,")
    print("so the term carries no parameter-discriminating signal for the inverse network.")
