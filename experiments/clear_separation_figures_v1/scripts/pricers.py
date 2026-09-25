"""Pricing layer for the separation figures. Reuses frozen artifacts; nothing is refitted.

Conventions, verified in code:
  * Controlled benchmark: r = q = 0, so F = S. Prices come from the frozen exact pricers and the
    frozen strongest baselines (Single Heston: global + 12 starts; Black-Scholes: BS_TERM, one
    volatility per maturity knot, family selected on an inner calibration holdout).
  * Double-Heston PINN: dh_pinn_v5 FINAL (gated, two-seed mean). Target = exact Double Heston.
  * Black-Scholes PINN: branch black-scholes-with-pinn, high_accuracy_run checkpoint. It carries its
    OWN convention (r = 0.03, q = 0.01, calibrated sigma) and outputs C/K. Target = analytic
    Black-Scholes at those parameters. It is never placed on the controlled benchmark, whose rate,
    dividend and volatility differ.
"""
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EXP = OUT.parent
sys.path.insert(0, str(EXP / 'financial_curve_figures_v2' / 'scripts'))
sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
sys.path.insert(0, str(OUT / 'vendor_bs_pinn'))
import common as CC                                         # controlled pricing layer (frozen)
from engine import iv as inv_iv, black
from model import BlackScholesPINN, Domain                  # vendored from the BS-PINN branch

K = CC.K                                                    # 100
STYLE = {
    'BS':      dict(color='#7a7a7a', ls='-',  lw=1.9, label='Black-Scholes (exact)'),
    'BS_PINN': dict(color='#c0392b', ls='--', lw=1.9, label='Black-Scholes PINN'),
    'SH':      dict(color='#d9822b', ls='-',  lw=1.9, label='Single Heston'),
    'DH':      dict(color='#1f3f8a', ls='-',  lw=2.3, label='Double Heston (exact)'),
    'PINN':    dict(color='#16a085', ls='--', lw=2.1, label='Double Heston PINN'),
    'MARKET':  dict(color='k', ls='none', marker='o', ms=4.5, label='market'),
}
CONTROLLED_MODELS = ['BS', 'SH', 'DH', 'PINN']              # BS-PINN excluded: different convention


# ------------------------------------------------------------------ controlled benchmark
def price(model, S, tau_years, scenario=CC.PRIMARY):
    """European call price in currency units on the controlled benchmark (r = q = 0, F = S)."""
    return CC.price_S(model, scenario, S, tau_years)


def time_value(model, S, tau_years, scenario=CC.PRIMARY):
    S = np.atleast_1d(np.asarray(S, float))
    return price(model, S, tau_years, scenario) - np.maximum(S - K, 0.0)


def implied_vol(model, S, tau_years, scenario=CC.PRIMARY):
    """Black-76 implied volatility of the model price (r = q = 0 so F = S)."""
    S = np.atleast_1d(np.asarray(S, float)); tau = np.broadcast_to(np.atleast_1d(np.asarray(tau_years, float)), S.shape)
    c = price(model, S, tau, scenario) / S                  # forward-normalised
    return inv_iv(c, np.log(S / K), tau)


def atm_iv(model, tau_years, scenario=CC.PRIMARY):
    tau = np.atleast_1d(np.asarray(tau_years, float))
    return implied_vol(model, np.full_like(tau, K), tau, scenario)


# ------------------------------------------------------------------ Black-Scholes PINN (own convention)
_BS_PINN = None


def bs_pinn():
    global _BS_PINN
    if _BS_PINN is None:
        cfg = json.loads((OUT / 'vendor_bs_pinn' / 'run_config.json').read_text())
        c, m = cfg['config'], cfg['model']
        net = BlackScholesPINN(domain=Domain(x_min=m['domain']['x_min'], x_max=m['domain']['x_max'],
                                             tau_max=m['domain']['tau_max']),
                               rate=c['rate'], dividend=c['dividend'],
                               hidden_width=c['hidden_width'], hidden_layers=c['hidden_layers'],
                               sigma_initial=c['sigma_initial'])
        state = torch.load(OUT / 'vendor_bs_pinn' / 'black_scholes_pinn.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state)
        net.double().eval()
        for p in net.parameters(): p.requires_grad_(False)
        _BS_PINN = net
    return _BS_PINN


def bs_pinn_meta():
    net = bs_pinn(); cfg = json.loads((OUT / 'vendor_bs_pinn' / 'run_config.json').read_text())
    return {'sigma': float(net.sigma), 'rate': net.rate, 'dividend': net.dividend,
            'x_min': net.domain.x_min, 'x_max': net.domain.x_max, 'tau_max': net.domain.tau_max,
            'hidden_width': cfg['config']['hidden_width'], 'hidden_layers': cfg['config']['hidden_layers'],
            'params': int(sum(p.numel() for p in net.parameters()))}


def bs_pinn_price(S, tau_years, strike=K):
    """C in currency units from the BS-PINN (its own r, q, sigma). Output is C/K, so C = (C/K)*K."""
    S = np.atleast_1d(np.asarray(S, float)); tau = np.broadcast_to(np.atleast_1d(np.asarray(tau_years, float)), S.shape)
    x = torch.tensor(np.log(S / strike).reshape(-1, 1)); t = torch.tensor(np.asarray(tau, float).reshape(-1, 1))
    with torch.no_grad():
        return bs_pinn()(x, t).numpy().ravel() * strike


def bs_analytic(S, tau_years, strike=K, sigma=None, rate=None, div=None):
    """Analytic Black-Scholes with dividends, the BS-PINN's own target."""
    net = bs_pinn()
    sigma = float(net.sigma) if sigma is None else sigma
    rate = net.rate if rate is None else rate
    div = net.dividend if div is None else div
    S = np.atleast_1d(np.asarray(S, float)); tau = np.broadcast_to(np.atleast_1d(np.asarray(tau_years, float)), S.shape)
    F = S * np.exp((rate - div) * tau); D = np.exp(-rate * tau)
    x = np.log(F / strike)
    return D * F * _n(x / (sigma * np.sqrt(tau)) + sigma * np.sqrt(tau) / 2) - D * strike * _n(x / (sigma * np.sqrt(tau)) - sigma * np.sqrt(tau) / 2)


def _n(z):
    from scipy.special import ndtr
    return ndtr(z)


def european_lower_bound(S, tau_years, strike=K, rate=None, div=None):
    net = bs_pinn()
    rate = net.rate if rate is None else rate
    div = net.dividend if div is None else div
    S = np.atleast_1d(np.asarray(S, float)); tau = np.broadcast_to(np.atleast_1d(np.asarray(tau_years, float)), S.shape)
    return np.maximum(S * np.exp(-div * tau) - strike * np.exp(-rate * tau), 0.0)


def shape_checks(S, C, strike=K, tol=1e-7):
    d = np.gradient(C, S); g = np.gradient(d, S)
    return {'points': int(len(S)), 'delta_violations': int((d < -tol).sum()), 'gamma_violations': int((g < -tol).sum()),
            'delta_min': float(d.min()), 'delta_max': float(d.max()), 'gamma_min': float(g.min()),
            'below_intrinsic': int((C < np.maximum(S - strike, 0) - 1e-6).sum()),
            'above_spot': int((C > S + 1e-6).sum()),
            'negative_time_value': int(((C - np.maximum(S - strike, 0)) < -1e-6).sum())}
