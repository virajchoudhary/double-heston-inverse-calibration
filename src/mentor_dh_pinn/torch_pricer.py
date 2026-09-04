"""Exact Double Heston Fourier pricer, in PyTorch, differentiable in the ten parameters.

This is a line-for-line port of ``src.double_heston.price_double_heston_call`` -- the same
Little-Heston-Trap exponent, the same P1/P2 decomposition, the same Gauss-Laguerre rule and
node count. It is *not* a surrogate and involves no learned component. It exists so that the
exact engine can be the teacher during training and the Jacobian source inside the unrolled
refinement, which the previous design could not do: it had to train against a learned V5
surrogate because the production engine was NumPy-only.

Conventions, verified against the NumPy engine in `tests/test_torch_pricer.py`:

    CF of log(S_T):  exp( i u [log S + (r - q) tau] + psi_slow(u) + psi_fast(u) )

    psi(u) = (kappa theta / sigma^2) [ (b - d) tau - 2 log((1 - g e^{-d tau})/(1 - g)) ]
             + v0 (b - d)/sigma^2 * (1 - e^{-d tau})/(1 - g e^{-d tau})

    b = kappa - rho sigma i u,   d = sqrt(b^2 + sigma^2 (u^2 + i u)),  Re(d) >= 0,
    g = (b - d)/(b + d)

    price = S e^{-q tau} P1 - K e^{-r tau} P2

Everything is float64/complex128. float32 is not offered: the characteristic function
cancels catastrophically at long maturity and the P1/P2 difference loses the leading digits.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_NODES = 64
_CACHE: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def laguerre_rule(node_count: int = DEFAULT_NODES, device="cpu"):
    """Gauss-Laguerre nodes and weights, identical to the NumPy engine's."""
    key = (int(node_count), str(device))
    if key not in _CACHE:
        n, w = np.polynomial.laguerre.laggauss(int(node_count))
        _CACHE[key] = (torch.tensor(n, dtype=torch.float64, device=device),
                       torch.tensor(w, dtype=torch.float64, device=device))
    return _CACHE[key]


def _heston_log_cf(u, tau, kappa, theta, sigma, rho, v0):
    """Little-Heston-Trap log-characteristic exponent of one CIR factor.

    All arguments broadcast. ``u`` is complex; the parameters are real tensors carrying
    gradients. No in-place branch selection, so autograd sees a clean graph.
    """
    iu = 1j * u
    b = kappa - rho * sigma * iu
    disc = b * b + (sigma * sigma) * (u * u + iu)
    d = torch.sqrt(disc)
    d = torch.where(d.real < 0.0, -d, d)              # principal branch, Re(d) >= 0
    denom = b + d
    g = (b - d) / denom
    exp_mdt = torch.exp(-d * tau)
    one = torch.ones((), dtype=g.dtype, device=g.device)
    # log((1 - g e^{-d tau})/(1 - g)) written as a difference of log1p for accuracy at g -> 0
    log_ratio = torch.log1p(-g * exp_mdt) - torch.log1p(-g)
    s2 = sigma * sigma
    c_term = (kappa * theta / s2) * ((b - d) * tau - 2.0 * log_ratio)
    d_term = ((b - d) / s2) * ((one - exp_mdt) / (one - g * exp_mdt))
    return c_term + d_term * v0


def _dh_cf(u, log_spot, tau, rate, carry, p):
    """Double Heston characteristic function; factors combine additively in log space."""
    slow = _heston_log_cf(u, tau, p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4])
    fast = _heston_log_cf(u, tau, p[..., 5], p[..., 6], p[..., 7], p[..., 8], p[..., 9])
    return torch.exp(1j * u * (log_spot + (rate - carry) * tau) + slow + fast)


def price_call(params, spot, strike, tau, rate, carry, *, node_count: int = DEFAULT_NODES):
    """European call prices, differentiable in ``params``.

    params : (..., 10) canonical order kappa_s, theta_s, sigma_s, rho_s, v0_s, then fast.
    spot, strike, tau, rate, carry : broadcastable to the quote shape (..., N).
    returns : (..., N) real prices.

    The caller is responsible for structural validity; this function does not validate,
    so that it can be called inside an optimiser without raising. Non-finite outputs are
    returned as NaN and must be handled explicitly by the caller -- never silently zeroed.
    """
    nodes, weights = laguerre_rule(node_count, params.device)
    u = nodes.to(torch.complex128)                                    # (J,)
    sp, st = torch.as_tensor(spot), torch.as_tensor(strike)
    tt, rr, cc = torch.as_tensor(tau), torch.as_tensor(rate), torch.as_tensor(carry)

    # quote axis gets a trailing node axis; parameters get both
    log_spot = torch.log(sp).unsqueeze(-1)                            # (..., N, 1)
    t_, r_, c_ = tt.unsqueeze(-1), rr.unsqueeze(-1), cc.unsqueeze(-1)
    p_nj = params.unsqueeze(-2).unsqueeze(-2)                         # (..., 1, 1, 10)
    p_n = params.unsqueeze(-2)                                        # (..., 1, 10)

    phi_u = _dh_cf(u, log_spot, t_, r_, c_, p_nj)                     # (..., N, J)
    phi_s = _dh_cf(u - 1j, log_spot, t_, r_, c_, p_nj)                # (..., N, J)
    phi_mi = _dh_cf(torch.tensor(-1j, dtype=torch.complex128, device=params.device),
                    torch.log(sp), tt, rr, cc, p_n)                   # (..., N)

    osc = torch.exp(-1j * u * torch.log(st).unsqueeze(-1))            # (..., N, J)
    inv_iu = 1.0 / (1j * u)
    comp = torch.exp(nodes) * weights                                 # (J,) Laguerre e^{+u} w
    p1 = 0.5 + (comp * (osc * phi_s * inv_iu / phi_mi.unsqueeze(-1)).real).sum(-1) / np.pi
    p2 = 0.5 + (comp * (osc * phi_u * inv_iu).real).sum(-1) / np.pi
    return sp * torch.exp(-cc * tt) * p1 - st * torch.exp(-rr * tt) * p2


def price_call_single(params, spot, strike, tau, rate, carry, *, node_count: int = DEFAULT_NODES):
    """Single-factor Heston, same convention and quadrature. params: (..., 5)."""
    nodes, weights = laguerre_rule(node_count, params.device)
    u = nodes.to(torch.complex128)
    sp, st = torch.as_tensor(spot), torch.as_tensor(strike)
    tt, rr, cc = torch.as_tensor(tau), torch.as_tensor(rate), torch.as_tensor(carry)
    log_spot = torch.log(sp).unsqueeze(-1)
    t_, r_, c_ = tt.unsqueeze(-1), rr.unsqueeze(-1), cc.unsqueeze(-1)
    p_nj = params.unsqueeze(-2).unsqueeze(-2)
    p_n = params.unsqueeze(-2)

    def cf(uu, ls, ta, ra, ca, pp):
        e = _heston_log_cf(uu, ta, pp[..., 0], pp[..., 1], pp[..., 2], pp[..., 3], pp[..., 4])
        return torch.exp(1j * uu * (ls + (ra - ca) * ta) + e)

    phi_u = cf(u, log_spot, t_, r_, c_, p_nj)
    phi_s = cf(u - 1j, log_spot, t_, r_, c_, p_nj)
    phi_mi = cf(torch.tensor(-1j, dtype=torch.complex128, device=params.device),
                torch.log(sp), tt, rr, cc, p_n)
    osc = torch.exp(-1j * u * torch.log(st).unsqueeze(-1))
    inv_iu = 1.0 / (1j * u)
    comp = torch.exp(nodes) * weights
    p1 = 0.5 + (comp * (osc * phi_s * inv_iu / phi_mi.unsqueeze(-1)).real).sum(-1) / np.pi
    p2 = 0.5 + (comp * (osc * phi_u * inv_iu).real).sum(-1) / np.pi
    return sp * torch.exp(-cc * tt) * p1 - st * torch.exp(-rr * tt) * p2
