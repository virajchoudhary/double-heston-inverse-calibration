"""Shared pricing layer for the v2 financial curve figures.

Convention, verified in code (not assumed):
  * The repository's exact pricers return the FORWARD-normalised call c = C / F, undiscounted.
  * The controlled benchmark is generated with r = q = 0 (literature_exact.batch_teacher passes
    zero rate and zero dividend), so F = S exp((r-q)tau) = S. Every figure therefore plots
    C(S) = S * c(log(S/K), tau) and the horizontal axis is genuinely the underlying price S.
  * The PINN's own head returns C / K; c = (C/K) * exp(-x) with x = log(F/K), so C = (C/K) * K.
Nothing is rescaled, shifted or offset anywhere in this module.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                    # double-heston-v2-controlled
EXP = ROOT / 'experiments'
V4 = EXP / 'nifty_multifactor_v4'
V5 = EXP / 'dh_pinn_v5'
sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V4))
sys.path.insert(0, str(ROOT))

from engine import exact as exact_price, black                     # btc_multifactor_v1 engine (forward-normalised)
from literature_exact import bs_predict                            # frozen v4 Black-Scholes term predictor
from evaluate import load_stage                                    # v5 checkpoint loader
from train import structural, tensor                               # v5 helpers

C4 = json.loads((V4 / 'config.json').read_text())
DH_PUB = np.array(C4['published_double_slow_first'])               # slow factor first
KAPPA_SLOW, KAPPA_FAST = float(DH_PUB[0]), float(DH_PUB[5])
K = 100.0                                                          # normalised strike for the controlled benchmark
RATE = 0.0                                                         # r = q = 0 in the controlled benchmark, so F = S

# ------------------------------------------------------------------ predeclared controlled scenario
# nifty_multifactor_v4 config, representative_states_fast_slow['FIXED_TOTAL_TWIST'] = [[v_fast, v_slow], ...]
TWIST = C4['representative_states_fast_slow']['FIXED_TOTAL_TWIST']
SCENARIOS = {
    'FAST_HEAVY': {'case': 'FIXED_TOTAL_TWIST_representative_0', 'v_fast': TWIST[0][0], 'v_slow': TWIST[0][1]},
    'SLOW_HEAVY': {'case': 'FIXED_TOTAL_TWIST_representative_1', 'v_fast': TWIST[1][0], 'v_slow': TWIST[1][1]},
}
PRIMARY = 'FAST_HEAVY'                                             # predeclared primary panel scenario
REPRESENTATIVE_TAU_DAYS = 90.0                                     # predeclared, not chosen by advantage
PANEL_TAU_DAYS = [30., 90., 180., 365., 730.]                      # predeclared maturities
PINN_TAU_MIN_DAYS, PINN_TAU_MAX_DAYS = 7., 730.                    # trained domain
PINN_X_HALF = 0.36                                                 # |log(F/K)| <= 0.36  ->  S/K in [0.698, 1.433]

STYLE = {
    'BS':   dict(color='#7a7a7a', ls='-',  lw=1.9, label='Black-Scholes'),
    'SH':   dict(color='#d9822b', ls='-',  lw=1.9, label='Single Heston'),
    'DH':   dict(color='#1f3f8a', ls='-',  lw=2.3, label='Double Heston (exact)'),
    'PINN': dict(color='#16a085', ls='--', lw=2.1, label='Double Heston PINN (improved)'),
}
ORDER = ['BS', 'SH', 'DH', 'PINN']
_NETS = None


def nets():
    global _NETS
    if _NETS is None:
        _NETS = load_stage('FINAL', 'final')
    return _NETS


def dh_params(scenario):
    s = SCENARIOS[scenario]
    p = DH_PUB.copy()
    p[4] = s['v_slow']                                             # slow factor v0
    p[9] = s['v_fast']                                             # fast factor v0
    return p


def baseline(scenario):
    """Frozen optimised Single-Heston fit and the selected non-leaking Black-Scholes fit."""
    d = json.loads((V4 / 'artifacts' / 'baselines' / f"{SCENARIOS[scenario]['case']}.json").read_text())
    return np.array(d['SH']['best']['params']), d[d['BS_selected']], d['BS_selected'], d


def _c_forward(model, scenario, x, tau):
    """Forward-normalised call price c = C/F for the requested model."""
    x = np.atleast_1d(np.asarray(x, float)); tau = np.atleast_1d(np.asarray(tau, float))
    x, tau = np.broadcast_arrays(x, tau)
    if model == 'DH':
        return exact_price(dh_params(scenario), x, tau, feller=False)
    if model == 'SH':
        return exact_price(baseline(scenario)[0], x, tau, feller=False)
    if model == 'BS':
        return bs_predict(baseline(scenario)[1], x, tau)
    if model == 'PINN':
        p = dh_params(scenario)
        co = tensor(np.column_stack([x, np.full(len(x), p[4]), np.full(len(x), p[9]), tau]))
        st = structural(p[None]).expand(len(x), 2, 4)
        with torch.no_grad():
            over_k = np.mean([n.price(co, st).numpy() for n in nets()], 0)                # C / K
        return over_k * np.exp(-x)                                                        # C / F
    raise KeyError(model)


def price_S(model, scenario, S, tau):
    """European call price C in currency units, as a function of the underlying price S.

    r = q = 0 in this benchmark so F = S; x = log(F/K) is recomputed from S for every point.
    """
    S = np.atleast_1d(np.asarray(S, float)); tau = np.atleast_1d(np.asarray(tau, float))
    S, tau = np.broadcast_arrays(S, tau)
    F = S * np.exp(RATE * tau)                                                            # explicit, = S here
    x = np.log(F / K)
    return _c_forward(model, scenario, x, tau) * F


def payoff(S):
    return np.maximum(np.asarray(S, float) - K, 0.0)


def shape_checks(S, C, tol=1e-7):
    """Finite-difference delta and gamma violations plus standard European-call bounds."""
    d = np.gradient(C, S); g = np.gradient(d, S)
    lower = np.maximum(S - K, 0.0)
    return {
        'points': int(len(S)),
        'delta_violations': int((d < -tol).sum()), 'delta_violation_pct': float(100 * (d < -tol).mean()),
        'delta_min': float(d.min()), 'delta_max': float(d.max()),
        'gamma_violations': int((g < -tol).sum()), 'gamma_violation_pct': float(100 * (g < -tol).mean()),
        'gamma_min': float(g.min()),
        'below_intrinsic': int((C < lower - 1e-6).sum()), 'above_spot': int((C > S + 1e-6).sum()),
    }


def in_pinn_domain(S=None, tau_days=None):
    ok = np.ones(1, bool)
    if S is not None:
        ok = np.abs(np.log(np.asarray(S, float) / K)) <= PINN_X_HALF + 1e-12
    if tau_days is not None:
        t = (np.asarray(tau_days, float) >= PINN_TAU_MIN_DAYS - 1e-9) & (np.asarray(tau_days, float) <= PINN_TAU_MAX_DAYS + 1e-9)
        ok = ok & t if ok.shape == t.shape else t
    return ok
