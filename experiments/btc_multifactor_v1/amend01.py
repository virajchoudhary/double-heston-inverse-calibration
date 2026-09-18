"""Amendment 01 (see AMENDMENT_01_BS_EXPIRY.md): Black-Scholes grouped by expiry.
Commands: freeze -> fit -> evaluate -> secondary.  Frozen v1 files are verified, never modified.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run
from run import OUT, read, save, sha, stamp, verify, all_dates, metrics, cluster_boot
from engine import black, bs_predict, iv

A = OUT / 'amend01'
FILES = ['AMENDMENT_01_BS_EXPIRY.md', 'amend01.py', 'test_amend01.py']


def fit_bs_expiry(expiry, x, tau, y, w, b):
    knots, vols = [], []
    for e in sorted(set(expiry), key=lambda e: np.median(tau[expiry == e])):
        m = expiry == e
        r = minimize_scalar(lambda s: float(np.mean(((black(x[m], tau[m], s) - y[m]) / w[m]) ** 2)), bounds=tuple(b['bs_vol']), method='bounded', options={'xatol': 1e-10})
        knots.append(float(np.median(tau[m]))); vols.append(float(r.x))
    return {'expiries': sorted(set(expiry), key=lambda e: np.median(tau[expiry == e])), 'knots': knots, 'volatility': vols}


def predict_bs_expiry(fit, expiry, x, tau):
    own = dict(zip(fit['expiries'], fit['volatility']))
    interp = bs_predict(fit, x, tau)                       # total-variance interpolation between expiry knots, flat outside
    flat = black(x, tau, np.array([own.get(e, np.nan) for e in expiry]))
    return np.where(np.isin(expiry, list(own)), flat, interp)


def freeze():
    verify(); assert not (A / 'manifest.json').exists() and not (A / 'fits').exists()
    save(A / 'manifest.json', {'frozen_utc': stamp(), 'sha256': {f: sha(HERE / f) for f in FILES}, 'v1_manifest_sha256': sha(OUT / 'manifest.json'),
                               'v1_results_sha256': sha(OUT / 'results.json'), 'corrected_bs_fitted_before_freeze': False})
    print('AMENDMENT FROZEN')


def check():
    verify(); m = read(A / 'manifest.json')
    for f, h in m['sha256'].items(): assert sha(HERE / f) == h, f'amendment source changed: {f}'
    assert sha(OUT / 'results.json') == m['v1_results_sha256'], 'v1 results must stay untouched'
    return run.read(HERE / 'config.json')


def fit():
    c = check(); usable = {a['date'] for a in read(OUT / 'data_audit.json')['dates'] if a['usable']}
    for stage in ['validation', 'test']:
        for r in all_dates(stage):
            if r['date'] not in usable: continue
            q = pd.read_csv(OUT / 'surfaces' / f"{r['date']}.csv")
            for d in ['B', 'A']:
                out = A / 'fits' / d / r['date'] / 'BS_EXPIRY.json'
                if out.exists(): continue
                cal = ~q[f'heldout_{d}'].to_numpy(); e, x, t, y, w = (q[k].to_numpy() for k in ['expiry', 'x', 'tau', 'c', 'vega_w'])
                f = fit_bs_expiry(e[cal], x[cal], t[cal], y[cal], w[cal], c['bounds'])
                save(out, {'date': r['date'], 'design': d, 'model': 'BS_EXPIRY', 'fit': f, 'pred': predict_bs_expiry(f, e, x, t).tolist(), 'utc': stamp()})
    print('fits done')


def preds(date, design, sel):
    """Held-out predictions per model: BS_EXPIRY from the amendment, SH/DH (selected variants) and flawed BS_TERM from v1."""
    p = {'BS_EXPIRY': read(A / 'fits' / design / date / 'BS_EXPIRY.json')['pred']}
    for m, v in [('BS_TERM', None), ('SH', sel['SH']), ('DH', sel['DH'])]:
        j = read(OUT / 'fits' / design / date / f"{m}{'_' + v if v else ''}.json")
        if 'error' not in j: p[m] = j['pred']
    return p


def scores(stage, target='c'):
    sel = read(OUT / 'variant_selection.json'); rows = []
    for r in all_dates(stage):
        f = OUT / 'surfaces' / f"{r['date']}.csv"
        if not f.exists(): continue
        q = pd.read_csv(f)
        if target != 'c':
            q['c'] = q[target]; q['market_iv'] = iv(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
        for d in ['B', 'A']:
            held = q[f'heldout_{d}'].to_numpy()
            for m, p in preds(r['date'], d, sel).items():
                rows.append({**r, 'design': d, 'model': m, **metrics(q, p, held), **{f'insample_{k}': v for k, v in metrics(q, p, ~held).items()}})
    return pd.DataFrame(rows)


def cells(s, bs='BS_EXPIRY'):
    res = []; ok = s[s.IV_RMSE_vol_points.notna()]
    for regime in ['shock', 'calm']:
        for design in ['B', 'A']:
            g = ok[ok.regime.eq(regime) & ok.design.eq(design)]
            piv = g.pivot_table(index=['date', 'episode'] if regime == 'shock' else ['date'], columns='model', values='IV_RMSE_vol_points')[[bs, 'SH', 'DH']].dropna().reset_index()
            unit = piv.episode.to_numpy() if regime == 'shock' else np.arange(len(piv))
            cell = {'regime': regime, 'design': design, 'dates': len(piv), 'units': int(len(np.unique(unit))), 'median_IV_RMSE': {m: float(piv[m].median()) for m in [bs, 'SH', 'DH']}}
            for comp in ['SH', bs]:
                dlt = (piv[comp] - piv.DH).to_numpy(); ci, means = cluster_boot(dlt, unit, 20260920)
                p = float(stats.wilcoxon(means, alternative='greater').pvalue) if len(means) > 1 and np.any(means != 0) else float('nan')
                cell[f'{comp}_minus_DH'] = {'mean': float(dlt.mean()), 'median': float(np.median(dlt)), 'DH_wins_dates': f'{int((dlt > 0).sum())}/{len(dlt)}',
                                            'unit_means_positive': f'{int((means > 0).sum())}/{len(means)}', 'wilcoxon_one_sided_p': p, 'cluster_bootstrap95': ci, 'DH_beats': bool(p < .05 and ci[0] > 0)}
            res.append(cell)
    return res


def evaluate():
    check(); assert not (A / 'results.json').exists()
    s = scores('test'); s.to_csv(A / 'test_scores.csv', index=False); c = cells(s)
    prim = next(x for x in c if x['regime'] == 'shock' and x['design'] == 'B')
    ins = s.groupby('model').insample_IV_RMSE_vol_points.median().to_dict()
    res = {'utc': stamp(), 'amendment': 'AMENDMENT_01_BS_EXPIRY.md', 'cells': c, 'median_insample_IV_RMSE': ins,
           'primary': {'DH_beats_SH': prim['SH_minus_DH']['DH_beats'], 'DH_beats_BS_EXPIRY': prim['BS_EXPIRY_minus_DH']['DH_beats']}}
    save(A / 'results.json', res); print(json.dumps(res, indent=1, default=float))


def secondary():
    """Predeclared secondaries: mark-price target, maturity/moneyness buckets, price RMSE. Fits unchanged (targets only)."""
    check(); B = OUT / 'secondary'; assert not (B / 'secondary.json').exists(); B.mkdir(exist_ok=True); out = {'utc': stamp()}
    mk = scores('test', 'c_mark'); mk.to_csv(B / 'test_scores_mark_target.csv', index=False)
    out['mark_target_cells'] = cells(mk)
    s = pd.read_csv(A / 'test_scores.csv')
    out['price_RMSE_USD_median'] = s.groupby(['regime', 'design', 'model']).price_RMSE_USD.median().unstack().round(1).to_dict('index')
    out['price_RMSE_USD_median'] = {f'{k[0]}/{k[1]}': v for k, v in out['price_RMSE_USD_median'].items()}
    sel = read(OUT / 'variant_selection.json'); rows = []
    for r in all_dates('test'):
        f = OUT / 'surfaces' / f"{r['date']}.csv"
        if not f.exists(): continue
        q = pd.read_csv(f)
        for d in ['B', 'A']:
            held = q[f'heldout_{d}'].to_numpy()
            for m, p in preds(r['date'], d, sel).items():
                if m == 'BS_TERM': continue
                mdl = iv(np.asarray(p), q.x.to_numpy(), q.tau.to_numpy())
                rows.append(pd.DataFrame({'regime': r['regime'], 'design': d, 'model': m, 'days': q.days, 'x': q.x, 'err': 100 * (mdl - q.market_iv)})[held])
    e = pd.concat(rows); e = e[np.isfinite(e.err)]
    e['maturity'] = pd.cut(e.days, [0, 7, 30, 90, 400], labels=['<=7d', '7-30d', '30-90d', '>90d'])
    e['moneyness'] = pd.cut(e.x, [-1.01, -.15, -.05, .05, .15, 1.01], labels=['x<-.15 (OTM call wing)', '-.15..-.05', 'ATM +-.05', '.05...15', 'x>.15 (OTM put wing)'])
    for b in ['maturity', 'moneyness']:
        t = e.groupby(['regime', 'design', b, 'model'], observed=True).err.agg(lambda z: float(np.sqrt(np.mean(z ** 2)))).unstack().round(2)
        n = e[e.model.eq('DH')].groupby(['regime', 'design', b], observed=True).size()
        t['quotes'] = n; t.to_csv(B / f'bucket_{b}.csv'); out[f'bucket_{b}_pooled_IV_RMSE'] = {' / '.join(map(str, k)): v for k, v in t.to_dict('index').items()}
    save(B / 'secondary.json', out); print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    cmd = argparse.ArgumentParser(); cmd.add_argument('command', choices=['freeze', 'fit', 'evaluate', 'secondary']); a = cmd.parse_args()
    globals()[a.command]()
