"""Fixed-parameter SPX benchmark, rerun with the improved PINN alongside the locked one."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; EXP = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
from track_b import prior_c, fit_scale, DH, SH, BSVOL
from evaluate import load_stage, locked
from engine import iv as invert_iv

BUCKETS = [(7, 30, '<=30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 9999, '>365d')]
MON = [(-1, -.15, 'x<-0.15'), (-.15, -.05, '-0.15..-0.05'), (-.05, .05, 'ATM'), (.05, .15, '0.05..0.15'), (.15, 1, 'x>0.15')]


def run(surface=None, out=None):
    q = pd.read_csv(surface or EXP / 'hardcoded_v1' / 'artifacts' / 'spx_surface.csv')
    nets = {'DH_PINN_LOCKED': locked()}
    if (HERE / 'artifacts' / 'final' / 'FINAL').exists(): nets['DH_PINN_V5'] = load_stage('FINAL', 'final')
    res = {'quotes': len(q), 'nothing_fitted': {}, 'one_scale': {}, 'buckets_one_scale': {}, 'moneyness_one_scale': {}}
    models = [('BS', 'BS'), ('SH', 'SH'), ('DH', 'DH')] + [(k, 'DH_PINN') for k in nets]
    x, t = q.x.to_numpy(), q.tau.to_numpy()
    for name, kind in models:
        net = nets.get(name)
        for tag, s in [('nothing_fitted', 1.), ('one_scale', None)]:
            s = fit_scale(kind, q, net) if s is None else s
            c = prior_c(kind, x, t, s, net); mi = invert_iv(c, x, t); ok = np.isfinite(mi) & np.isfinite(q.market_iv)
            e = 100 * (mi[ok] - q.market_iv.to_numpy()[ok])
            res[tag][name] = {'scale': s, 'IV_RMSE': float(np.sqrt(np.mean(e ** 2))), 'IV_bias': float(np.mean(e)),
                              'price_RMSE': float(np.sqrt(np.mean((c - q.c.to_numpy()) ** 2))), 'n': int(ok.sum())}
            if tag == 'one_scale':
                days = q.days.to_numpy()[ok]
                res['buckets_one_scale'].setdefault(name, {}); res['moneyness_one_scale'].setdefault(name, {})
                for a, b, lab in BUCKETS:
                    m = (days >= a) & (days < b)
                    if m.any(): res['buckets_one_scale'][name][lab] = {'IV_RMSE': float(np.sqrt(np.mean(e[m] ** 2))), 'n': int(m.sum())}
                xs = x[ok]
                for a, b, lab in MON:
                    m = (xs >= a) & (xs < b)
                    if m.any(): res['moneyness_one_scale'][name][lab] = {'IV_RMSE': float(np.sqrt(np.mean(e[m] ** 2))), 'n': int(m.sum())}
        print(name, 'fixed %.3f | scaled %.3f' % (res['nothing_fitted'][name]['IV_RMSE'], res['one_scale'][name]['IV_RMSE']), flush=True)
    if out: Path(out).write_text(json.dumps(res, indent=2, default=float) + '\n')
    return res


if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser(); a.add_argument('--out', default='artifacts/spx_benchmark.json'); run(out=a.parse_args().out)
