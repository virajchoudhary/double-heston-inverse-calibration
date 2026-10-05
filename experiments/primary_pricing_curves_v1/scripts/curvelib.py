"""Pricing layer for the PRIMARY price-curve figures: C vs S and C vs tau.

Nothing here refits, rescales or offsets anything. All model prices come from the frozen
artifacts already validated in this repository:

  * controlled benchmark  : r = q = 0, K = 100, so F = S and C(S,tau) = S * c(log(S/K), tau)
  * Double Heston (exact) : frozen published two-factor parameters, v0 set by the frozen
                            FIXED_TOTAL_TWIST state (v_fast + v_slow = 0.04 -> 20% instantaneous)
  * Single Heston         : frozen strongest fit (global + 12 starts) for the same state
  * Black-Scholes         : two baselines, both kept
                              BS20     constant sigma = 20% (matches the instantaneous vol)
                              BS_TERM  frozen calibrated term structure, one sigma per maturity knot
  * Double-Heston PINN    : dh_pinn_v5 FINAL (two-seed mean), target = exact Double Heston
  * Black-Scholes PINN    : branch black-scholes-with-pinn; it owns r = 0.03, q = 0.01 and its own
                            calibrated sigma, so it is NEVER placed on the r = q = 0 benchmark.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EXP = OUT.parent
sys.path.insert(0, str(EXP / 'financial_curve_figures_v2' / 'scripts'))
sys.path.insert(0, str(EXP / 'clear_separation_figures_v1' / 'vendor_bs_pinn'))
import common as CC
from model import BlackScholesPINN, Domain

K          = CC.K              # 100
SIGMA_CONST = 0.20             # sqrt(v_fast + v_slow) of the frozen FIXED_TOTAL_TWIST state
TAU_DAYS    = [7., 30., 90., 180., 365., 730.]     # predeclared maturity family
S_LO, S_HI, S_N = 0.60, 1.40, 601                  # S / K sweep, >= 500 points, one shared grid
PINN_X_HALF = CC.PINN_X_HALF                       # 0.36  ->  S/K in [0.6977, 1.4333]

MODELS = ['BS20', 'BS_TERM', 'SH', 'DH', 'PINN']
STYLE = {
    'BS20':    dict(color='#7a7a7a', ls='-',  lw=1.8, label='Black-Scholes  (constant $\\sigma$=20%)'),
    'BS_TERM': dict(color='#4d4d4d', ls=':',  lw=1.8, label='Black-Scholes  (calibrated term $\\sigma(\\tau)$)'),
    'SH':      dict(color='#d9822b', ls='-',  lw=1.8, label='Single Heston  (exact)'),
    'DH':      dict(color='#1f3f8a', ls='-',  lw=2.3, label='Double Heston  (exact)'),
    'PINN':    dict(color='#16a085', ls='--', lw=2.0, label='Double Heston PINN'),
    'PAYOFF':  dict(color='#b03060', ls='-.', lw=1.5, label='payoff  $\\max(S-K,0)$'),
}
MAT_CMAP = ['#00d26a', '#00c2a8', '#00a9d6', '#3f7fd6', '#7a5bd6', '#b34bd0']   # neon-green family


def S_grid():
    return np.linspace(S_LO * K, S_HI * K, S_N)


# ---------------------------------------------------------------- analytic Black-Scholes, r = q = 0
def bs_const(S, tau, sigma=SIGMA_CONST, strike=K):
    """Closed-form European call, r = q = 0. Written straight from the Black-Scholes equation."""
    S   = np.atleast_1d(np.asarray(S, float))
    tau = np.broadcast_to(np.atleast_1d(np.asarray(tau, float)), S.shape)
    out = np.maximum(S - strike, 0.0).astype(float)
    m   = tau > 0
    if m.any():
        v  = sigma * np.sqrt(tau[m])
        d1 = (np.log(S[m] / strike) + 0.5 * v ** 2) / v
        out[m] = S[m] * ndtr(d1) - strike * ndtr(d1 - v)
    return out


def price(model, S, tau_years, scenario=CC.PRIMARY):
    """C in currency units on the controlled benchmark. PINN is NaN outside its trained domain."""
    S   = np.atleast_1d(np.asarray(S, float))
    tau = np.broadcast_to(np.atleast_1d(np.asarray(tau_years, float)), S.shape).copy()
    if model == 'BS20':
        return bs_const(S, tau)
    if model == 'PINN':
        ok = (np.abs(np.log(S / K)) <= PINN_X_HALF + 1e-12) & \
             (tau >= CC.PINN_TAU_MIN_DAYS / 365. - 1e-9) & (tau <= CC.PINN_TAU_MAX_DAYS / 365. + 1e-9)
        out = np.full(S.shape, np.nan)
        if ok.any():
            out[ok] = CC.price_S('PINN', scenario, S[ok], tau[ok])
        return out
    return CC.price_S({'BS_TERM': 'BS'}.get(model, model), scenario, S, tau)


def payoff(S):
    return np.maximum(np.asarray(S, float) - K, 0.0)


# ---------------------------------------------------------------- financial validation
def validate(S, C, tau_years, label, tol=1e-7):
    """Delta >= 0, Gamma >= 0, C >= max(S-K,0), C <= S  (all exact for r = q = 0)."""
    m = np.isfinite(C)
    Sm, Cm = np.asarray(S)[m], np.asarray(C)[m]
    if len(Sm) < 3:
        return {'case': label, 'points': int(len(Sm))}
    d = np.gradient(Cm, Sm); g = np.gradient(d, Sm)
    lower = np.maximum(Sm - K, 0.0)
    return {
        'case': label, 'tau_days': float(np.atleast_1d(tau_years)[0] * 365.), 'points': int(len(Sm)),
        'delta_min': float(d.min()), 'delta_max': float(d.max()),
        'delta_violations': int((d < -tol).sum()),
        'gamma_min': float(g.min()), 'gamma_max': float(g.max()),
        'gamma_violations': int((g < -tol).sum()),
        'gamma_violations_1pct_of_peak': int((g < -0.01 * g.max()).sum()),
        'worst_gamma_as_pct_of_peak': float(100 * g.min() / g.max()) if g.max() > 0 else float('nan'),
        'min_time_value': float((Cm - lower).min()),
        'below_intrinsic': int((Cm < lower - 1e-6).sum()),
        'above_spot': int((Cm > Sm + 1e-6).sum()),
    }


# ---------------------------------------------------------------- Black-Scholes PINN (own convention)
_BSP = None
def bs_pinn():
    global _BSP
    if _BSP is None:
        V = EXP / 'clear_separation_figures_v1' / 'vendor_bs_pinn'
        cfg = json.loads((V / 'run_config.json').read_text()); c, m = cfg['config'], cfg['model']
        net = BlackScholesPINN(domain=Domain(x_min=m['domain']['x_min'], x_max=m['domain']['x_max'],
                                             tau_max=m['domain']['tau_max']),
                               rate=c['rate'], dividend=c['dividend'], hidden_width=c['hidden_width'],
                               hidden_layers=c['hidden_layers'], sigma_initial=c['sigma_initial'])
        st = torch.load(V / 'black_scholes_pinn.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(st['model_state_dict'] if isinstance(st, dict) and 'model_state_dict' in st else st)
        net.double().eval()
        for p in net.parameters(): p.requires_grad_(False)
        _BSP = net
    return _BSP


def bs_pinn_meta():
    n = bs_pinn()
    return {'sigma': float(n.sigma), 'rate': n.rate, 'dividend': n.dividend,
            'x_min': n.domain.x_min, 'x_max': n.domain.x_max, 'tau_max': n.domain.tau_max}


def bs_pinn_price(S, tau, strike=K):
    S = np.atleast_1d(np.asarray(S, float))
    t = np.broadcast_to(np.atleast_1d(np.asarray(tau, float)), S.shape)
    x = torch.tensor(np.log(S / strike).reshape(-1, 1)); tt = torch.tensor(np.asarray(t, float).reshape(-1, 1))
    with torch.no_grad():
        return bs_pinn()(x, tt).numpy().ravel() * strike


def bs_analytic_rq(S, tau, strike=K):
    """Analytic Black-Scholes at the BS-PINN's OWN r, q, sigma -- its correct target."""
    n = bs_pinn(); sig, r, q = float(n.sigma), n.rate, n.dividend
    S = np.atleast_1d(np.asarray(S, float))
    tau = np.broadcast_to(np.atleast_1d(np.asarray(tau, float)), S.shape)
    F = S * np.exp((r - q) * tau); D = np.exp(-r * tau)
    v = sig * np.sqrt(tau); x = np.log(F / strike)
    return D * (F * ndtr(x / v + v / 2) - strike * ndtr(x / v - v / 2))
