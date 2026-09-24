"""Buckets, paired cell statistics and no-arbitrage checks for the Track B hybrids."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent; OUT = HERE / 'artifacts' / 'track_b'
sys.path.insert(0, str(HERE.parent / 'btc_multifactor_v1'))
from engine import black
BUCK = [(7, 30, '<=30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 9999, '>365d')]
MON = [(-1, -.15, 'x<-0.15'), (-.15, -.05, '-0.15..-0.05'), (-.05, .05, 'ATM'), (.05, .15, '0.05..0.15'), (.15, 1, 'x>0.15')]


def cells(d):
    mb = np.digitize(d['x'], [-.15, -.05, .05, .15]); tb = np.digitize(d['days'], [30, 90, 365])
    return np.array([f'{a}_{b}' for a, b in zip(tb, mb)])


def boot(diff, group, seed=20260920, n=5000):
    rng = np.random.default_rng(seed); ids = np.unique(group); means = np.array([diff[group == g].mean() for g in ids])
    draws = rng.choice(means, size=(n, len(means))).mean(1)
    return np.quantile(draws, [.025, .975]).tolist(), means


def rmse(e): return float(np.sqrt(np.nanmean(e ** 2)))


def run():
    priors = [p.stem.split('heldout_')[1] for p in OUT.glob('heldout_*.npz')]
    D = {k: dict(np.load(OUT / f'heldout_{k}.npz')) for k in priors}
    res = {'priors': priors, 'overall': {}, 'by_maturity': {}, 'by_moneyness': {}, 'pairwise': {}, 'cell_wins': {}, 'no_arbitrage': {}}
    for k, d in D.items():
        res['overall'][k] = {'prior_IV_RMSE': rmse(d['err_prior']), 'hybrid_IV_RMSE': rmse(d['err_corr']),
                             'hybrid_IV_bias': float(np.nanmean(d['err_corr'])), 'hybrid_IV_P95': float(np.nanquantile(abs(d['err_corr']), .95))}
        res['by_maturity'][k] = {lab: {'prior': rmse(d['err_prior'][m]), 'hybrid': rmse(d['err_corr'][m]), 'n': int(m.sum())}
                                 for a, b, lab in BUCK for m in [(d['days'] >= a) & (d['days'] < b)] if m.any()}
        res['by_moneyness'][k] = {lab: {'prior': rmse(d['err_prior'][m]), 'hybrid': rmse(d['err_corr'][m]), 'n': int(m.sum())}
                                  for a, b, lab in MON for m in [(d['x'] >= a) & (d['x'] < b)] if m.any()}
    base = 'DH' if 'DH' in D else priors[0]
    g = cells(D[base])
    for other in [p for p in priors if p != base]:
        a, b = D[other], D[base]
        for tag, ka, kb in [('prior', 'err_prior', 'err_prior'), ('hybrid', 'err_corr', 'err_corr')]:
            diff = np.abs(a[ka]) - np.abs(b[kb]); ok = np.isfinite(diff)
            ci, means = boot(diff[ok], g[ok])
            p = float(stats.wilcoxon(means, alternative='greater').pvalue) if len(means) > 1 and np.any(means != 0) else float('nan')
            res['pairwise'][f'{other}_minus_{base}_{tag}'] = {'mean_abs_err_diff': float(np.nanmean(diff)), 'cells': int(len(means)),
                                                              'cells_favouring_' + base: f'{int((means > 0).sum())}/{len(means)}',
                                                              'wilcoxon_p': p, 'cluster_bootstrap95': ci,
                                                              f'{base}_beats_{other}': bool(p < .05 and ci[0] > 0)}
    # no-arbitrage on the hybrid surfaces: convexity and monotonicity in strike within each expiry band, and total variance in maturity
    q = pd.read_csv(OUT / 'spx_split.csv'); ho = q[q.set == 'heldout'].reset_index(drop=True)
    checks = {k: pd.DataFrame({'expiry': ho.expiry, 'x': d['x'], 'tau': ho.tau, 'iv': d['corr_iv']}).dropna() for k, d in D.items()}
    checks['MARKET_QUOTES_CONTROL'] = pd.DataFrame({'expiry': ho.expiry, 'x': ho.x, 'tau': ho.tau, 'iv': ho.market_iv}).dropna()
    for k, f in checks.items():
        f = f.copy(); f['K_over_F'] = np.exp(-f.x); f['C'] = black(f.x.to_numpy(), f.tau.to_numpy(), f.iv.to_numpy())
        mono = conv = tot = 0; n_m = n_c = n_t = 0
        for _, v in f.groupby('expiry'):
            v = v.sort_values('K_over_F')
            if len(v) < 3: continue
            dC = np.diff(v.C.to_numpy()); mono += int((dC > 1e-9).sum()); n_m += len(dC)          # C must fall as K rises
            d2 = np.diff(v.C.to_numpy(), 2); conv += int((d2 < -1e-9).sum()); n_c += len(d2)
        f['w'] = f.iv ** 2 * f.tau; f['mbin'] = pd.cut(f.x, np.linspace(-.36, .36, 9))
        for _, v in f.groupby('mbin', observed=True):
            v = v.sort_values('tau')
            if len(v) < 3: continue
            dw = np.diff(v.w.to_numpy()); tot += int((dw < -1e-9).sum()); n_t += len(dw)
        res['no_arbitrage'][k] = {'strike_monotonicity_violations': f'{mono}/{n_m}', 'strike_convexity_violations': f'{conv}/{n_c}',
                                  'calendar_total_variance_violations': f'{tot}/{n_t}'}
    (OUT / 'analysis.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
    print(json.dumps({k: res[k] for k in ['overall', 'pairwise', 'no_arbitrage']}, indent=1, default=float))
    return res


if __name__ == '__main__':
    run()
