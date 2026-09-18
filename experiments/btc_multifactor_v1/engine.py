"""Exact pricing and calibration for the BTC multifactor experiment.

Reuses the audited separable four-shock characteristic exponent
(src.double_heston_reference._factor_exponent), the same exponent used by the
nifty_multifactor literature_exact experiments. The only differences from that
adapter are an optional (not mandatory) Feller condition and market-scale bounds.
"""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution, least_squares, minimize_scalar
from scipy.special import ndtr
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.double_heston_reference import _factor_exponent, _quad_diagnostic
from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance


def admissible(p, feller):
    a = np.asarray(p, float).reshape(-1, 5)
    if len(a) not in (1, 2) or not np.isfinite(a).all(): raise ValueError('bad parameter shape or value')
    if np.any(a[:, [0, 1, 2, 4]] <= 0) or np.any(abs(a[:, 3]) >= 1): raise ValueError('economic bound failure')
    if feller and np.any(2 * a[:, 0] * a[:, 1] <= a[:, 2] ** 2): raise ValueError('Feller failure')
    if len(a) == 2 and a[0, 0] >= a[1, 0]: raise ValueError('slow-first order required')


@lru_cache(maxsize=4)
def rule(n):
    u, w = np.polynomial.laguerre.laggauss(n)
    return u, w * np.exp(u) / (np.pi * 1j * u)


class Grid:
    """Forward-normalised call prices c = C/F for fixed quote geometry; x = log(F/K)."""

    def __init__(self, x, tau, nodes=128, feller=True):
        self.x, self.tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
        assert self.x.ndim == 1 and np.isfinite(self.x).all() and (self.tau > 0).all()
        self.times, self.ix = np.unique(self.tau, return_inverse=True)
        self.u, self.w = rule(nodes)
        self.osc = np.exp(1j * self.x[:, None] * self.u) * self.w
        self.k = np.exp(-self.x); self.feller = feller

    def __call__(self, p):
        admissible(p, self.feller)
        f = np.asarray(p).reshape(-1, 5); t = self.times[:, None]
        cs = np.exp(sum(_factor_exponent(self.u - 1j, t, *a) for a in f))
        cu = np.exp(sum(_factor_exponent(self.u, t, *a) for a in f))
        result = (.5 + np.real(self.osc * cs[self.ix]).sum(1)) - self.k * (.5 + np.real(self.osc * cu[self.ix]).sum(1))
        if not np.isfinite(result).all(): raise FloatingPointError('non-finite Fourier price')
        return result


def adaptive(p, x, t, feller, eps=1e-10):
    admissible(p, feller)
    factors = np.asarray(p).reshape(-1, 5)
    def cf(u): return np.exp(sum(_factor_exponent(u, t, *a) for a in factors))
    one, d1 = _quad_diagnostic(lambda u: float(np.real(np.exp(1j * u * x) * cf(u - 1j) / (1j * u))), epsabs=eps, epsrel=eps, limit=500)
    two, d2 = _quad_diagnostic(lambda u: float(np.real(np.exp(1j * u * x) * cf(u) / (1j * u))), epsabs=eps, epsrel=eps, limit=500)
    value = .5 + one / np.pi - np.exp(-x) * (.5 + two / np.pi)
    if not (d1.get('reliable') and d2.get('reliable') and np.isfinite(value)): raise FloatingPointError('unreliable adaptive reference')
    return float(value)


def exact(p, x, tau, feller, tol=1e-7, audit=None):
    """Evaluation pricing: 128 and 96 nodes must agree; otherwise independent adaptive quadrature."""
    x, tau = np.asarray(x, float), np.asarray(tau, float)
    a = Grid(x, tau, 128, feller)(p); b = Grid(x, tau, 96, feller)(p); lower = np.maximum(1 - np.exp(-x), 0)
    bad = (abs(a - b) > tol) | (a < lower - 1e-9) | (a > 1 + 1e-9)
    for j in np.flatnonzero(bad):
        a[j] = adaptive(p, float(x[j]), float(tau[j]), feller)
        if audit is not None: audit.append({'x': float(x[j]), 'tau': float(tau[j])})
    return a


def black(x, tau, sigma):
    root = np.asarray(sigma) * np.sqrt(tau); d1 = x / root + root / 2
    return ndtr(d1) - np.exp(-x) * ndtr(d1 - root)


def vega(x, tau, sigma):
    root = np.asarray(sigma) * np.sqrt(tau); d1 = x / root + root / 2
    return np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi) * np.sqrt(tau)


def iv(price, x, tau):
    return np.sqrt(invert_total_variance(np.asarray(price, float) * np.exp(x), np.asarray(x, float)) / tau)


# ------------------------------------------------------------------ parameter maps
def box(kind, variant, b):
    L = np.log
    s = [b['eta']] if variant == 'feller_strict' else [[L(b['sigma'][0]), L(b['sigma'][1])]]
    if kind == 'SH':
        rows = [[L(b['kappa_single'][0]), L(b['kappa_single'][1])], [L(b['theta'][0]), L(b['theta'][1])], *s, b['rho'], [L(b['v0'][0]), L(b['v0'][1])]]
    else:
        rows = [[L(b['kappa_slow'][0]), L(b['kappa_slow'][1])], [L(b['kappa_gap'][0]), L(b['kappa_gap'][1])],
                [L(b['theta'][0]), L(b['theta'][1])], *s, b['rho'], [L(b['v0'][0]), L(b['v0'][1])],
                [L(b['theta'][0]), L(b['theta'][1])], *s, b['rho'], [L(b['v0'][0]), L(b['v0'][1])]]
    return np.array(rows, float).T


def decode(kind, variant, z):
    z = np.asarray(z, float)
    def factor(k, lt, s, r, lv):
        t = np.exp(lt); sig = s * np.sqrt(2 * k * t) if variant == 'feller_strict' else np.exp(s)
        return [k, t, sig, r, np.exp(lv)]
    if kind == 'SH': return np.array(factor(np.exp(z[0]), *z[1:5]))
    ks = np.exp(z[0]); kf = ks + np.exp(z[1])
    return np.array(factor(ks, *z[2:6]) + factor(kf, *z[6:10]))


# ------------------------------------------------------------------ calibration
def fit_heston(kind, variant, x, tau, y, w, cfg):
    """Global differential evolution, then 12 bounded Levenberg-Marquardt-style starts. All starts retained."""
    oc, lo_hi = cfg['optimizer'], box(kind, variant, cfg['bounds'])
    lo, hi = lo_hi; grid = Grid(x, tau, cfg['numerics']['nodes'], variant == 'feller_strict')
    def resid(z):
        try: return (grid(decode(kind, variant, z)) - y) / w
        except (ValueError, FloatingPointError): return np.full(len(y), 1e3)
    glob = differential_evolution(lambda z: float(np.mean(resid(z) ** 2)), list(zip(lo, hi)), seed=oc['seed'], maxiter=oc['de_maxiter'],
                                  popsize=oc['de_popsize_multiplier'], polish=False, workers=1, updating='deferred')
    starts = [glob.x, *list(lo + (hi - lo) * qmc.LatinHypercube(len(lo), seed=oc['seed'] + 1).random(oc['multistarts'] - 1))]
    records = []
    for j, z in enumerate(starts):
        try:
            r = least_squares(resid, np.clip(z, lo, hi), bounds=(lo, hi), max_nfev=oc['max_nfev'], xtol=1e-10, ftol=1e-10, gtol=1e-10, x_scale='jac')
            records.append({'start': j, 'success': bool(r.success), 'status': int(r.status), 'nfev': int(r.nfev), 'objective': float(np.mean(r.fun ** 2)),
                            'z': r.x.tolist(), 'params': decode(kind, variant, r.x).tolist(),
                            'near_bound': bool(np.any(np.minimum(r.x - lo, hi - r.x) / (hi - lo) < .005))})
        except (ValueError, FloatingPointError) as e:
            records.append({'start': j, 'success': False, 'objective': None, 'message': str(e)})
    finite = [r for r in records if r['objective'] is not None and r['objective'] < 1e5]
    if not finite: raise RuntimeError('all starts failed')
    best = min(finite, key=lambda r: r['objective'])
    near = sum(r['success'] and r['objective'] <= best['objective'] + max(1e-12, .01 * best['objective']) for r in finite)
    return {'kind': kind, 'variant': variant, 'best': best, 'starts': records, 'converged_near_best': int(near),
            'global': {'success': bool(glob.success), 'message': str(glob.message), 'nfev': int(glob.nfev)}, 'global_optimum_proven': False}


def fit_bs_term(x, tau, y, w, b):
    knots = np.unique(tau); vols = []
    for t in knots:
        m = tau == t
        r = minimize_scalar(lambda s: float(np.mean(((black(x[m], tau[m], s) - y[m]) / w[m]) ** 2)), bounds=tuple(b['bs_vol']), method='bounded', options={'xatol': 1e-10})
        vols.append(float(r.x))
    return {'knots': knots.tolist(), 'volatility': vols}


def bs_predict(fit, x, tau):
    t = np.asarray(fit['knots']); v = np.asarray(fit['volatility'])
    wv = np.interp(tau, t, t * v * v)
    wv = np.where(tau < t[0], tau * v[0] ** 2, np.where(tau > t[-1], tau * v[-1] ** 2, wv))
    return black(x, tau, np.sqrt(wv / tau))
