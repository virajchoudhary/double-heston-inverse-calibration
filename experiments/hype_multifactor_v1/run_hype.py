"""HYPE multifactor replication. Commands, in order:
freeze -> validate -> select-dates -> fetch -> fit -> evaluate -> secondary
All rules live in config.json and were frozen before any HYPE option surface was fetched.
Engine, BS_EXPIRY and metrics are imported unchanged from btc_multifactor_v1 (hashed in the manifest).
"""
import argparse
import datetime as dt
import hashlib
import json
import statistics
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; BTC = HERE.parent / 'btc_multifactor_v1'; ROOT = HERE.parents[1]; OUT = HERE / 'artifacts'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(BTC)); sys.path.insert(0, str(ROOT))
from engine import exact, fit_heston, iv
from amend01 import fit_bs_expiry, predict_bs_expiry
from run import cluster_boot
from scipy import stats
from run import metrics

SOURCES = ['config.json', 'PROTOCOL.md', 'data_hype.py', 'run_hype.py', 'test_hype.py']
BTC_SOURCES = ['engine.py', 'data.py', 'run.py', 'amend01.py']
REPO = ['src/double_heston_reference.py', 'src/mentor_dh_pinn/regular_pinn_data.py']
MODELS = {'BS_EXPIRY': None, 'SH': 'feller_free', 'DH': 'feller_free'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2, default=float) + '\n')
def read(p): return json.loads(Path(p).read_text())
def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def freeze():
    assert not (OUT / 'manifest.json').exists() and not (OUT / 'surfaces').exists() and not (OUT / 'raw').exists()
    c = read(HERE / 'config.json')
    m = {'frozen_utc': stamp(), 'config': c, 'source_sha256': {f: sha(HERE / f) for f in SOURCES}, 'btc_sha256': {f: sha(BTC / f) for f in BTC_SOURCES},
         'repo_sha256': {f: sha(ROOT / f) for f in REPO}, 'spot_sha256': sha(HERE / c['date_rule']['spot_file']),
         'hype_option_surfaces_fetched_before_freeze': False, 'models_fitted_before_freeze': False}
    save(OUT / 'manifest.json', m); (OUT / 'manifest.sha256').write_text(sha(OUT / 'manifest.json') + '\n'); print('FROZEN', sha(OUT / 'manifest.json'))


def verify():
    m = read(OUT / 'manifest.json'); assert sha(OUT / 'manifest.json') == (OUT / 'manifest.sha256').read_text().strip()
    for f, h in m['source_sha256'].items(): assert sha(HERE / f) == h, f'frozen source changed: {f}'
    for f, h in m['btc_sha256'].items(): assert sha(BTC / f) == h, f'frozen btc source changed: {f}'
    for f, h in m['repo_sha256'].items(): assert sha(ROOT / f) == h, f'frozen repo source changed: {f}'
    assert sha(HERE / m['config']['date_rule']['spot_file']) == m['spot_sha256']
    return m['config']


def validate():
    verify(); r = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(HERE / 'test_hype.py')], cwd=ROOT, capture_output=True, text=True)
    (OUT / 'tests.txt').write_text(r.stdout + r.stderr); save(OUT / 'tests.json', {'passed': r.returncode == 0, 'utc': stamp()})
    print(r.stdout[-600:]); assert r.returncode == 0, 'tests failed: no data fetch or fitting permitted'


def rv_ratio(rule):
    rows = read(HERE / rule['spot_file'])['rows']; days = [r['date'] for r in rows]; px = np.array([r['close'] for r in rows])
    ret = np.diff(np.log(px)); W, L = rule['rv_window_days'], rule['lookback_days']
    rv = [None] * len(px)
    for i in range(W, len(px)): rv[i] = float(np.std(ret[i - W:i], ddof=1) * np.sqrt(365))
    ratio = [None] * len(px)
    for i in range(W + L, len(px)): ratio[i] = rv[i] / statistics.median(rv[i - L:i])
    return days, rv, ratio


def select_dates():
    c = verify(); rule = c['date_rule']; assert read(OUT / 'tests.json')['passed'] and not (OUT / 'dates.json').exists()
    days, rv, ratio = rv_ratio(rule); pos = {d: i for i, d in enumerate(days)}
    onsets, last = [], -10 ** 9
    for i, d in enumerate(days):
        if ratio[i] is not None and '2026-06-01' <= d <= rule['last_date'] and ratio[i] >= rule['shock_ratio']:
            if i - last > rule['dedupe_days']: onsets.append(i)
            last = i
    lo, hi = rule['shock_window_days']; out = []
    d, end = dt.date.fromisoformat(rule['first_date']), dt.date.fromisoformat(rule['last_date'])
    while d <= end:
        ep = [e for e, i in enumerate(onsets) if lo <= (d - dt.date.fromisoformat(days[i])).days <= hi]
        iso = d.isocalendar(); prev = pos.get((d - dt.timedelta(days=1)).isoformat())
        out.append({'date': d.isoformat(), 'week': f'{iso[0]}-W{iso[1]:02d}', 'regime': 'shock' if ep else 'other', 'episode': ep[0] if ep else None,
                    'rv_ratio_prev_day': None if prev is None or ratio[prev] is None else round(ratio[prev], 4), 'stage': 'test'})
        d += dt.timedelta(days=1)
    save(OUT / 'dates.json', {'utc': stamp(), 'rule': rule, 'onsets': [days[i] for i in onsets], 'dates': out})
    print(f"{len(out)} dates; onsets {[days[i] for i in onsets]}; shock-window dates {sum(r['regime'] == 'shock' for r in out)}")


def all_dates():
    return read(OUT / 'dates.json')['dates']


def fetch():
    from data_hype import fetch as get, clean
    c = verify(); assert (OUT / 'dates.json').exists(); audits = []; (OUT / 'raw').mkdir(exist_ok=True); (OUT / 'surfaces').mkdir(exist_ok=True)
    for r in all_dates():
        path = OUT / 'surfaces' / f"{r['date']}.csv"
        try:
            trades, h = get(r['date'], c['window_utc_hours'], c['sources']['trades'], c['sources']['instrument_prefix'], OUT / 'raw')
            q, audit = clean(trades, r['date'], c); audit['raw_sha256'] = h
            if audit['usable']: q.to_csv(path, index=False)
        except Exception as e:
            audit = {'date': r['date'], 'usable': False, 'reason': f'fetch/clean error: {e!r}'}
        audits.append({**r, **audit}); print(r['date'], r['regime'], 'usable' if audit['usable'] else audit.get('reason'), audit.get('quotes'), audit.get('expiries'), flush=True)
    save(OUT / 'data_audit.json', {'utc': stamp(), 'dates': audits})


def fit_task(job):
    date, design, model = job; c = read(HERE / 'config.json'); variant = MODELS[model]
    q = pd.read_csv(OUT / 'surfaces' / f'{date}.csv'); cal = ~q[f'heldout_{design}'].to_numpy()
    e, x, t, y, w = (q[k].to_numpy() for k in ['expiry', 'x', 'tau', 'c', 'vega_w'])
    out = OUT / 'fits' / design / date / f'{model}.json'
    if out.exists(): return str(out)
    try:
        if model == 'BS_EXPIRY':
            fit = fit_bs_expiry(e[cal], x[cal], t[cal], y[cal], w[cal], c['bounds']); pred = predict_bs_expiry(fit, e, x, t); extra = {'fallbacks': 0}
        else:
            fit = fit_heston(model, variant, x[cal], t[cal], y[cal], w[cal], c); audit = []
            pred = exact(fit['best']['params'], x, t, False, c['numerics']['evaluation_disagreement_tolerance'], audit); extra = {'fallbacks': len(audit)}
        save(out, {'date': date, 'design': design, 'model': model, 'variant': variant, 'fit': fit, 'pred': pred.tolist(), **extra, 'utc': stamp()})
    except Exception as ex:
        save(out, {'date': date, 'design': design, 'model': model, 'variant': variant, 'error': repr(ex), 'utc': stamp()})
    return str(out)


def fit(workers):
    verify(); usable = {a['date'] for a in read(OUT / 'data_audit.json')['dates'] if a['usable']}
    jobs = sorted({(r['date'], d, m) for r in all_dates() if r['date'] in usable for d in ['B', 'A'] for m in MODELS}, key=lambda j: j[2] != 'DH')
    print(f'{len(jobs)} fit jobs', flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for k, _ in enumerate(pool.map(fit_task, jobs), 1):
            if k % 10 == 0 or k == len(jobs): print(f'{k}/{len(jobs)} done', flush=True)


def scores(target='c'):
    rows = []
    for r in all_dates():
        f = OUT / 'surfaces' / f"{r['date']}.csv"
        if not f.exists(): continue
        q = pd.read_csv(f)
        if target != 'c': q['c'] = q[target]; q['market_iv'] = iv(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
        for d in ['B', 'A']:
            held = q[f'heldout_{d}'].to_numpy()
            if not held.any(): continue
            for m in MODELS:
                j = read(OUT / 'fits' / d / r['date'] / f'{m}.json')
                if 'error' in j: rows.append({**r, 'design': d, 'model': m, 'error': j['error']}); continue
                rows.append({**r, 'design': d, 'model': m, **metrics(q, j['pred'], held), **{f'insample_{k}': v for k, v in metrics(q, j['pred'], ~held).items()},
                             'near_best_starts': j['fit'].get('converged_near_best'), 'fallbacks': j.get('fallbacks')})
    s = pd.DataFrame(rows)
    return s[s.IV_RMSE_vol_points.notna()] if 'IV_RMSE_vol_points' in s else s


def cells(s):
    res = []; M = ['BS_EXPIRY', 'SH', 'DH']
    for group in ['all', 'shock', 'other']:
        for design in ['B', 'A']:
            g = s if group == 'all' else s[s.regime.eq(group)]; g = g[g.design.eq(design)]
            unit_col = 'episode' if group == 'shock' else 'week'
            piv = g.pivot_table(index=['date', unit_col], columns='model', values='IV_RMSE_vol_points')
            if piv.empty or not set(M) <= set(piv.columns): continue
            piv = piv[M].dropna().reset_index(); unit = piv[unit_col].astype(str).to_numpy()
            cell = {'group': group, 'design': design, 'unit': unit_col, 'dates': len(piv), 'units': int(len(np.unique(unit))), 'median_IV_RMSE': {m: float(piv[m].median()) for m in M}, 'mean_IV_RMSE': {m: float(piv[m].mean()) for m in M}}
            for comp in ['SH', 'BS_EXPIRY']:
                dlt = (piv[comp] - piv.DH).to_numpy(); ci, means = cluster_boot(dlt, unit, 20260920)
                p = float(stats.wilcoxon(means, alternative='greater').pvalue) if len(means) > 1 and np.any(means != 0) else float('nan')
                cell[f'{comp}_minus_DH'] = {'mean': float(dlt.mean()), 'median': float(np.median(dlt)), 'DH_wins_dates': f'{int((dlt > 0).sum())}/{len(dlt)}',
                                            'unit_means_positive': f'{int((means > 0).sum())}/{len(means)}', 'wilcoxon_one_sided_p': p, 'cluster_bootstrap95': ci, 'DH_beats': bool(p < .05 and ci[0] > 0)}
            res.append(cell)
    return res


def evaluate():
    verify(); assert not (OUT / 'results.json').exists()
    s = scores(); s.to_csv(OUT / 'test_scores.csv', index=False); c = cells(s)
    prim = next(x for x in c if x['group'] == 'all' and x['design'] == 'B')
    res = {'utc': stamp(), 'primary_endpoint': read(HERE / 'config.json')['primary_endpoint'], 'cells': c,
           'median_insample_IV_RMSE': s.groupby('model').insample_IV_RMSE_vol_points.median().to_dict(),
           'primary': {'DH_beats_SH': prim['SH_minus_DH']['DH_beats'], 'DH_beats_BS_EXPIRY': prim['BS_EXPIRY_minus_DH']['DH_beats']}}
    save(OUT / 'results.json', res); print(json.dumps(res, indent=1, default=float))


def secondary():
    verify(); assert (OUT / 'results.json').exists() and not (OUT / 'secondary' / 'secondary.json').exists(); B = OUT / 'secondary'; B.mkdir(exist_ok=True)
    mk = scores('c_mark'); mk.to_csv(B / 'test_scores_mark_target.csv', index=False); out = {'utc': stamp(), 'mark_target_cells': cells(mk)}
    s = pd.read_csv(OUT / 'test_scores.csv')
    out['price_RMSE_USD_median'] = {f'{k[0]}/{k[1]}': v for k, v in s.groupby(['regime', 'design', 'model']).price_RMSE_USD.median().unstack().round(4).to_dict('index').items()}
    rows = []
    for r in all_dates():
        f = OUT / 'surfaces' / f"{r['date']}.csv"
        if not f.exists(): continue
        q = pd.read_csv(f)
        for d in ['B', 'A']:
            held = q[f'heldout_{d}'].to_numpy()
            for m in MODELS:
                j = read(OUT / 'fits' / d / r['date'] / f'{m}.json')
                if 'error' in j: continue
                mdl = iv(np.asarray(j['pred']), q.x.to_numpy(), q.tau.to_numpy())
                rows.append(pd.DataFrame({'regime': r['regime'], 'design': d, 'model': m, 'days': q.days, 'x': q.x, 'err': 100 * (mdl - q.market_iv)})[held])
    e = pd.concat(rows); e = e[np.isfinite(e.err)]
    e['maturity'] = pd.cut(e.days, [0, 7, 30, 90, 400], labels=['<=7d', '7-30d', '30-90d', '>90d'])
    e['moneyness'] = pd.cut(e.x, [-1.01, -.15, -.05, .05, .15, 1.01], labels=['x<-.15 (OTM call wing)', '-.15..-.05', 'ATM +-.05', '.05...15', 'x>.15 (OTM put wing)'])
    for b in ['maturity', 'moneyness']:
        t = e.groupby(['regime', 'design', b, 'model'], observed=True).err.agg(lambda z: float(np.sqrt(np.mean(z ** 2)))).unstack().round(2)
        t['quotes'] = e[e.model.eq('DH')].groupby(['regime', 'design', b], observed=True).size(); t.to_csv(B / f'bucket_{b}.csv')
    save(B / 'secondary.json', out); print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['freeze', 'validate', 'select-dates', 'fetch', 'fit', 'evaluate', 'secondary'])
    ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    {'freeze': freeze, 'validate': validate, 'select-dates': select_dates, 'fetch': fetch, 'evaluate': evaluate, 'secondary': secondary,
     'fit': lambda: fit(a.workers)}[a.command]()
