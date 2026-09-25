"""ETH multifactor replication. Commands, in order:
freeze -> validate -> select-dates -> fetch -> fit -> evaluate -> secondary
All rules live in config.json and were frozen before any ETH option surface was fetched.
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
from amend01 import fit_bs_expiry, predict_bs_expiry, cells as btc_cells
from run import metrics

SOURCES = ['config.json', 'PROTOCOL.md', 'run_eth.py']
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
         'repo_sha256': {f: sha(ROOT / f) for f in REPO}, 'dvol_sha256': sha(HERE / c['date_rule']['dvol_file']),
         'eth_option_surfaces_fetched_before_freeze': False, 'models_fitted_before_freeze': False}
    save(OUT / 'manifest.json', m); (OUT / 'manifest.sha256').write_text(sha(OUT / 'manifest.json') + '\n'); print('FROZEN', sha(OUT / 'manifest.json'))


def verify():
    m = read(OUT / 'manifest.json'); assert sha(OUT / 'manifest.json') == (OUT / 'manifest.sha256').read_text().strip()
    for f, h in m['source_sha256'].items(): assert sha(HERE / f) == h, f'frozen source changed: {f}'
    for f, h in m['btc_sha256'].items(): assert sha(BTC / f) == h, f'frozen btc source changed: {f}'
    for f, h in m['repo_sha256'].items(): assert sha(ROOT / f) == h, f'frozen repo source changed: {f}'
    assert sha(HERE / m['config']['date_rule']['dvol_file']) == m['dvol_sha256']
    return m['config']


def validate():
    verify(); r = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(BTC / 'test_btc.py'), str(BTC / 'test_amend01.py')], cwd=ROOT, capture_output=True, text=True)
    (OUT / 'tests.txt').write_text(r.stdout + r.stderr); save(OUT / 'tests.json', {'passed': r.returncode == 0, 'utc': stamp()})
    print(r.stdout[-600:]); assert r.returncode == 0, 'tests failed: no data fetch or fitting permitted'


def select_dates():
    """BTC date rule verbatim on ETH DVOL; every date is a test date."""
    c = verify(); rule = c['date_rule']; assert read(OUT / 'tests.json')['passed'] and not (OUT / 'dates.json').exists()
    rows = read(HERE / rule['dvol_file'])['rows']; v = [r['dvol_close'] for r in rows]; days = [r['date'] for r in rows]; L = rule['lookback_days']
    ratio = [None] * L + [v[i] / statistics.median(v[i - L:i]) for i in range(L, len(v))]
    onsets, last = [], -10 ** 9
    for i in range(L, len(v)):
        if ratio[i] >= rule['shock_ratio']:
            if i - last > rule['dedupe_days']: onsets.append(i)
            last = i
    today = dt.date(2026, 9, 18); shock = []
    for e, i in enumerate(onsets):
        for off in rule['days_after_onset']:
            d = dt.date.fromisoformat(days[i]) + dt.timedelta(days=off)
            if d < today: shock.append({'date': d.isoformat(), 'episode': e, 'onset': days[i], 'offset_days': off, 'dvol_ratio_at_onset': round(ratio[i], 4), 'stage': 'test'})
    near = {dt.date.fromisoformat(days[i]) + dt.timedelta(days=k) for i in onsets for k in range(-rule['calm_exclusion_days_from_shock'], rule['calm_exclusion_days_from_shock'] + 1)}
    pool = [days[i] for i in range(L, len(v)) if abs(ratio[i] - 1) <= rule['calm_band'] and max(v[i - 20:i + 1]) / min(v[i - 20:i + 1]) <= rule['calm_max_range_20d']
            and dt.date.fromisoformat(days[i]) not in near and dt.date.fromisoformat(days[i]) + dt.timedelta(days=1) < today]
    rng = np.random.default_rng(rule['seed']); pick = sorted(rng.choice(len(pool), size=min(rule['calm_sample'], len(pool)), replace=False))
    calm = [{'date': (dt.date.fromisoformat(pool[j]) + dt.timedelta(days=1)).isoformat(), 'calm_reference_day': pool[j], 'stage': 'test'} for j in pick]
    save(OUT / 'dates.json', {'utc': stamp(), 'rule': rule, 'episodes': len(onsets), 'onsets': [days[i] for i in onsets], 'calm_pool_size': len(pool), 'shock': shock, 'calm': calm})
    print(f'{len(onsets)} shock episodes -> {len(shock)} shock dates; calm pool {len(pool)} -> {len(calm)} calm dates')


def all_dates():
    d = read(OUT / 'dates.json'); return [{**s, 'regime': 'shock'} for s in d['shock']] + [{**s, 'regime': 'calm'} for s in d['calm']]


def fetch():
    from data import fetch as get, clean
    c = verify(); assert (OUT / 'dates.json').exists(); audits = []; (OUT / 'raw').mkdir(exist_ok=True); (OUT / 'surfaces').mkdir(exist_ok=True)
    for r in all_dates():
        path = OUT / 'surfaces' / f"{r['date']}.csv"
        try:
            trades, h = get(r['date'], c['window_utc_hours'], c['sources']['trades'], OUT / 'raw')
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


def evaluate():
    verify(); assert not (OUT / 'results.json').exists()
    s = scores(); s.to_csv(OUT / 'test_scores.csv', index=False); c = btc_cells(s)
    prim = next(x for x in c if x['regime'] == 'shock' and x['design'] == 'B')
    res = {'utc': stamp(), 'primary_endpoint': read(HERE / 'config.json')['primary_endpoint'], 'cells': c,
           'median_insample_IV_RMSE': s.groupby('model').insample_IV_RMSE_vol_points.median().to_dict(),
           'primary': {'DH_beats_SH': prim['SH_minus_DH']['DH_beats'], 'DH_beats_BS_EXPIRY': prim['BS_EXPIRY_minus_DH']['DH_beats']}}
    save(OUT / 'results.json', res); print(json.dumps(res, indent=1, default=float))


def secondary():
    verify(); assert (OUT / 'results.json').exists() and not (OUT / 'secondary' / 'secondary.json').exists(); B = OUT / 'secondary'; B.mkdir(exist_ok=True)
    mk = scores('c_mark'); mk.to_csv(B / 'test_scores_mark_target.csv', index=False); out = {'utc': stamp(), 'mark_target_cells': btc_cells(mk)}
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
