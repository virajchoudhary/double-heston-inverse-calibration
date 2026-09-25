"""Residual-complexity diagnostic: how large, how smooth and how learnable is the market
residual left by each fixed-parameter prior? Calibration quotes only."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from track_b import prior_c, fit_scale, OUT
sys.path.insert(0, str(HERE.parent / 'btc_multifactor_v1'))
from engine import iv as invert_iv


def run():
    q = pd.read_csv(OUT / 'spx_split.csv'); cal = q[q.set != 'heldout']
    res = {}
    for kind in ['BS', 'SH', 'DH']:
        s = fit_scale(kind, cal)
        g = cal.copy(); g['model_iv'] = invert_iv(prior_c(kind, g.x.to_numpy(), g.tau.to_numpy(), s), g.x.to_numpy(), g.tau.to_numpy())
        g['r'] = 100 * (g.market_iv - g.model_iv)                     # residual in vol points
        g = g[np.isfinite(g.r)]
        # smoothness: mean |second difference| of the residual along strikes within an expiry, and along expiries at similar moneyness
        sk = [np.abs(np.diff(v.sort_values('x').r.to_numpy(), 2)).mean() for _, v in g.groupby('expiry') if len(v) > 3]
        g['mbin'] = pd.cut(g.x, np.linspace(-.36, .36, 9))
        mt = [np.abs(np.diff(v.sort_values('tau').r.to_numpy(), 2)).mean() for _, v in g.groupby('mbin', observed=True) if len(v) > 3]
        res[kind] = {'scale': s, 'residual_RMSE_volpts': float(np.sqrt(np.mean(g.r ** 2))), 'residual_bias': float(g.r.mean()),
                     'residual_std': float(g.r.std()), 'residual_P95_abs': float(np.quantile(abs(g.r), .95)),
                     'roughness_across_strikes': float(np.mean(sk)), 'roughness_across_maturities': float(np.mean(mt)),
                     'by_maturity': {lab: float(np.sqrt(np.mean(g.r[m] ** 2))) for a, b, lab in
                                     [(7, 30, '7-30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 999, '>365d')]
                                     for m in [(g.days >= a) & (g.days < b)] if m.any()}}
        print(kind, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[kind].items() if not isinstance(v, dict)}, flush=True)
    (OUT / 'residual_complexity.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
    return res


if __name__ == '__main__':
    run()
