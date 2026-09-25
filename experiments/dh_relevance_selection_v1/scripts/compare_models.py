"""Model comparison on the surface selected ex ante by SELECTION_RULE.md.

Two settings, both reported:
  (a) MATCHED CAPACITY - every model carries the published parameters with one level scale fitted
      on the surface. This is the only setting the PINN can enter, since it is trained on that
      parameter family, so it is the like-for-like comparison.
  (b) STRONGEST FAIR CALIBRATION - Black-Scholes gets one volatility per expiry, Single Heston all
      five parameters and Double Heston all ten, each fitted with the same global-plus-multistart
      optimiser on the same quotes. This asks how well each structure can represent the surface at
      all. The PINN cannot enter, because free parameters leave its trained family.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent; OUT = HERE.parent; EXP = OUT.parent
sys.path.insert(0, str(EXP / 'dh_pinn_v5')); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
from track_b import prior_c
from evaluate import load_stage, locked
from engine import iv as inv_iv, fit_heston, exact
from amend01 import fit_bs_expiry, predict_bs_expiry

CFG = json.loads((EXP / 'btc_multifactor_v1' / 'config.json').read_text())
SEL = json.loads((OUT / 'selection.json').read_text())
SYM = SEL['selected']
NETS = {'PINN_NEW': load_stage('FINAL', 'final'), 'PINN_OLD': locked()}
BUCK = [(7, 30, '<=30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 9999, '>365d')]
MON = [(-.36, -.15, 'x<-0.15'), (-.15, -.05, '-0.15..-0.05'), (-.05, .05, 'ATM'), (.05, .15, '0.05..0.15'), (.15, .36, 'x>0.15')]


def bucket_rmse(e, v, bins):
    return {lab: float(np.sqrt(np.mean(e[(v >= a) & (v < b)] ** 2))) for a, b, lab in bins if ((v >= a) & (v < b)).any()}


def matched_capacity(q):
    x, t, y, w = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w'])
    out = {}
    for name, kind, net in [('BS', 'BS', None), ('SH', 'SH', None), ('DH', 'DH', None),
                            ('DH_PINN_OLD', 'DH_PINN', 'PINN_OLD'), ('DH_PINN_NEW', 'DH_PINN', 'PINN_NEW')]:
        r = minimize_scalar(lambda s: float(np.mean(((prior_c(kind, x, t, s, NETS.get(net)) - y) / w) ** 2)),
                            bounds=(0.7, 3.2), method='bounded', options={'xatol': 1e-6})
        s = float(r.x); c = prior_c(kind, x, t, s, NETS.get(net)); mi = inv_iv(c, x, t)
        ok = np.isfinite(mi) & np.isfinite(q.market_iv)
        e = 100 * (mi[ok] - q.market_iv.to_numpy()[ok])
        out[name] = {'scale': s, 'scale_at_bound': bool(s <= .7005 or s >= 3.1995),
                     'IV_RMSE': float(np.sqrt(np.mean(e ** 2))), 'IV_bias': float(np.mean(e)),
                     'price_RMSE': float(np.sqrt(np.mean((c - y) ** 2))),
                     'by_maturity': bucket_rmse(e, q.days.to_numpy()[ok], BUCK),
                     'by_moneyness': bucket_rmse(e, x[ok], MON)}
        print(f'  matched {name:12} scale {s:.3f}{" (bound)" if out[name]["scale_at_bound"] else "":8} IV RMSE {out[name]["IV_RMSE"]:6.3f}', flush=True)
    return out


def strongest(q):
    x, t, y, w, e_ = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w', 'expiry'])
    out = {}
    f = fit_bs_expiry(e_, x, t, y, w, CFG['bounds']); c = predict_bs_expiry(f, e_, x, t)
    out['BS_per_expiry'] = _score(q, c, x, t, y, {'expiries': len(f['expiries'])})
    for kind in ['SH', 'DH']:
        fit = fit_heston(kind, 'feller_free', x, t, y, w, CFG)
        p = np.array(fit['best']['params']); c = exact(p, x, t, feller=False)
        out[f'{kind}_calibrated'] = _score(q, c, x, t, y, {'params': p.tolist(), 'near_best_starts': fit['converged_near_best']})
    for k, v in out.items(): print(f'  strongest {k:16} IV RMSE {v["IV_RMSE"]:6.3f}', flush=True)
    return out


def _score(q, c, x, t, y, extra):
    mi = inv_iv(c, x, t); ok = np.isfinite(mi) & np.isfinite(q.market_iv)
    e = 100 * (mi[ok] - q.market_iv.to_numpy()[ok])
    return {**extra, 'IV_RMSE': float(np.sqrt(np.mean(e ** 2))), 'IV_bias': float(np.mean(e)),
            'price_RMSE': float(np.sqrt(np.mean((c - y) ** 2))),
            'by_maturity': bucket_rmse(e, q.days.to_numpy()[ok], BUCK),
            'by_moneyness': bucket_rmse(e, x[ok], MON)}


def heldout(q):
    """Same strongest-fair calibration, but fitted on a checkerboard half and scored on the other half.
    Removes the parameter-count objection: every model is judged on quotes it never saw."""
    q = q.copy()
    order = q.groupby('expiry').tau.median().sort_values().index.tolist()
    q['expiry_rank'] = q.expiry.map({e: i for i, e in enumerate(order)})
    q = q.sort_values(['expiry_rank', 'strike']).reset_index(drop=True)
    q['strike_rank'] = q.groupby('expiry_rank').cumcount()
    cal = ((q.expiry_rank + q.strike_rank) % 2 == 0).to_numpy()
    x, t, y, w, ex = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w', 'expiry'])
    out = {'calibration_quotes': int(cal.sum()), 'heldout_quotes': int((~cal).sum())}
    f = fit_bs_expiry(ex[cal], x[cal], t[cal], y[cal], w[cal], CFG['bounds'])
    out['BS_per_expiry'] = _score(q[~cal], predict_bs_expiry(f, ex, x, t)[~cal], x[~cal], t[~cal], y[~cal], {})
    for kind in ['SH', 'DH']:
        fit = fit_heston(kind, 'feller_free', x[cal], t[cal], y[cal], w[cal], CFG)
        p = np.array(fit['best']['params'])
        out[f'{kind}_calibrated'] = _score(q[~cal], exact(p, x[~cal], t[~cal], feller=False), x[~cal], t[~cal], y[~cal],
                                           {'params': p.tolist(), 'near_best_starts': fit['converged_near_best']})
    for k, v in out.items():
        if isinstance(v, dict): print(f'  held-out  {k:16} IV RMSE {v["IV_RMSE"]:6.3f}', flush=True)
    return out


if __name__ == '__main__':
    q = pd.read_csv(OUT / 'surfaces' / f'{SYM}.csv')
    print(f'{SYM}: {len(q)} quotes, {q.expiry.nunique()} expiries')
    res = {'symbol': SYM, 'selection': {k: SEL[k] for k in ['rule_sha256', 'selected', 'selected_score', 'selected_error_known_before_selection']},
           'quotes': int(len(q)), 'expiries': int(q.expiry.nunique()),
           'matched_capacity': matched_capacity(q), 'strongest_fair_in_sample': strongest(q), 'strongest_fair_heldout': heldout(q)}
    (OUT / 'model_comparison.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
    print('\nwritten model_comparison.json')
