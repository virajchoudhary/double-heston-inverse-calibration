"""BTC multifactor experiment. Commands, in order:
freeze -> validate -> select-dates -> fetch -> fit --stage validation -> select-variant
       -> fit --stage test -> evaluate -> report
All rules live in config.json and were frozen before any option data was fetched.
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
from scipy import stats

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; OUT = HERE / 'artifacts'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from engine import exact, fit_heston, fit_bs_term, bs_predict, iv

SOURCES = ['engine.py', 'data.py', 'run.py', 'test_btc.py', 'config.json', 'PROTOCOL.md']
REPO = ['src/double_heston_reference.py', 'src/mentor_dh_pinn/regular_pinn_data.py']
MODELS = {'BS_TERM': [None], 'SH': ['feller_strict', 'feller_free'], 'DH': ['feller_strict', 'feller_free']}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2, default=float) + '\n')
def read(p): return json.loads(Path(p).read_text())
def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def freeze():
    assert not (OUT / 'manifest.json').exists() and not (OUT / 'surfaces').exists()
    c = read(HERE / 'config.json')
    m = {'frozen_utc': stamp(), 'config': c, 'source_sha256': {f: sha(HERE / f) for f in SOURCES}, 'repo_sha256': {f: sha(ROOT / f) for f in REPO},
         'dvol_sha256': sha(HERE / c['date_rule']['dvol_file']), 'option_data_fetched_before_freeze': False, 'models_fitted_before_freeze': False}
    save(OUT / 'manifest.json', m); (OUT / 'manifest.sha256').write_text(sha(OUT / 'manifest.json') + '\n'); print('FROZEN', sha(OUT / 'manifest.json'))


def verify():
    m = read(OUT / 'manifest.json'); assert sha(OUT / 'manifest.json') == (OUT / 'manifest.sha256').read_text().strip()
    for f, h in m['source_sha256'].items(): assert sha(HERE / f) == h, f'frozen source changed: {f}'
    for f, h in m['repo_sha256'].items(): assert sha(ROOT / f) == h, f'frozen repo source changed: {f}'
    assert sha(HERE / m['config']['date_rule']['dvol_file']) == m['dvol_sha256']
    return m['config']


def validate():
    verify(); r = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(HERE / 'test_btc.py')], cwd=ROOT, capture_output=True, text=True)
    (OUT / 'tests.txt').write_text(r.stdout + r.stderr); save(OUT / 'tests.json', {'passed': r.returncode == 0, 'utc': stamp()})
    print(r.stdout[-600:]); assert r.returncode == 0, 'tests failed: no data fetch or fitting permitted'


def select_dates():
    c = verify(); rule = c['date_rule']; assert not (OUT / 'dates.json').exists()
    rows = read(HERE / rule['dvol_file'])['rows']; v = [r['dvol_close'] for r in rows]; days = [r['date'] for r in rows]; L = rule['lookback_days']
    ratio = [None] * L + [v[i] / statistics.median(v[i - L:i]) for i in range(L, len(v))]
    onsets, last = [], -10 ** 9
    for i in range(L, len(v)):
        if ratio[i] >= rule['shock_ratio']:
            if i - last > rule['dedupe_days']: onsets.append(i)
            last = i
    today = dt.date.today()
    shock = []
    for e, i in enumerate(onsets):
        for off in rule['days_after_onset']:
            d = dt.date.fromisoformat(days[i]) + dt.timedelta(days=off)
            if d < today: shock.append({'date': d.isoformat(), 'episode': e, 'onset': days[i], 'offset_days': off, 'dvol_ratio_at_onset': round(ratio[i], 4)})
    near = {dt.date.fromisoformat(days[i]) + dt.timedelta(days=k) for i in onsets for k in range(-rule['calm_exclusion_days_from_shock'], rule['calm_exclusion_days_from_shock'] + 1)}
    calm_pool = [days[i] for i in range(L, len(v)) if abs(ratio[i] - 1) <= rule['calm_band'] and max(v[i - 20:i + 1]) / min(v[i - 20:i + 1]) <= rule['calm_max_range_20d']
                 and dt.date.fromisoformat(days[i]) not in near and dt.date.fromisoformat(days[i]) + dt.timedelta(days=1) < today]
    rng = np.random.default_rng(rule['seed']); pick = sorted(rng.choice(len(calm_pool), size=min(rule['calm_sample'], len(calm_pool)), replace=False))
    calm = [{'date': (dt.date.fromisoformat(calm_pool[j]) + dt.timedelta(days=1)).isoformat(), 'calm_reference_day': calm_pool[j]} for j in pick]
    nval_e = int(len(onsets) * rule['validation_fraction']); nval_c = int(len(calm) * rule['validation_fraction'])
    for s in shock: s['stage'] = 'validation' if s['episode'] < nval_e else 'test'
    for j, s in enumerate(calm): s['stage'] = 'validation' if j < nval_c else 'test'
    save(OUT / 'dates.json', {'utc': stamp(), 'rule': rule, 'episodes': len(onsets), 'onsets': [days[i] for i in onsets], 'shock': shock, 'calm': calm})
    print(f'{len(onsets)} shock episodes -> {len(shock)} shock dates; {len(calm)} calm dates; validation episodes {nval_e}, calm validation {nval_c}')


def all_dates(stage=None):
    d = read(OUT / 'dates.json'); rows = [{**s, 'regime': 'shock'} for s in d['shock']] + [{**s, 'regime': 'calm'} for s in d['calm']]
    return [r for r in rows if stage is None or r['stage'] == stage]


def fetch():
    from data import fetch as get, clean
    c = verify(); audits = []; (OUT / 'raw').mkdir(exist_ok=True); (OUT / 'surfaces').mkdir(exist_ok=True)
    for r in all_dates():
        path = OUT / 'surfaces' / f"{r['date']}.csv"
        try:
            trades, h = get(r['date'], c['window_utc_hours'], c['sources']['trades'], OUT / 'raw')
            q, audit = clean(trades, r['date'], c); audit['raw_sha256'] = h
            if audit['usable']: q.to_csv(path, index=False)
        except Exception as e:
            audit = {'date': r['date'], 'usable': False, 'reason': f'fetch/clean error: {e}'}
        audits.append({**r, **audit}); print(r['date'], r['regime'], r['stage'], 'usable' if audit['usable'] else audit.get('reason'), audit.get('quotes'), audit.get('expiries'), flush=True)
    save(OUT / 'data_audit.json', {'utc': stamp(), 'dates': audits})


def fit_task(job):
    date, design, model, variant = job; c = read(HERE / 'config.json')
    q = pd.read_csv(OUT / 'surfaces' / f'{date}.csv'); held = q[f'heldout_{design}'].to_numpy(); cal = ~held
    x, t, y, w = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w'])
    out = OUT / 'fits' / design / date / f"{model}{'_' + variant if variant else ''}.json"
    if out.exists(): return str(out)
    try:
        if model == 'BS_TERM':
            fit = fit_bs_term(x[cal], t[cal], y[cal], w[cal], c['bounds']); pred = bs_predict(fit, x, t); extra = {'fallbacks': 0}
        else:
            fit = fit_heston(model, variant, x[cal], t[cal], y[cal], w[cal], c); audit = []
            pred = exact(fit['best']['params'], x, t, variant == 'feller_strict', c['numerics']['evaluation_disagreement_tolerance'], audit); extra = {'fallbacks': len(audit)}
        save(out, {'date': date, 'design': design, 'model': model, 'variant': variant, 'fit': fit, 'pred': pred.tolist(), **extra, 'utc': stamp()})
    except Exception as e:
        save(out, {'date': date, 'design': design, 'model': model, 'variant': variant, 'error': repr(e), 'utc': stamp()})
    return str(out)


def fit(stage, workers):
    verify(); usable = {a['date'] for a in read(OUT / 'data_audit.json')['dates'] if a['usable']}
    if stage == 'test':
        sel = read(OUT / 'variant_selection.json'); models = {'BS_TERM': [None], 'SH': [sel['SH']], 'DH': [sel['DH']]}
    else:
        assert not (OUT / 'variant_selection.json').exists(); models = MODELS
    jobs = [(r['date'], d, m, v) for r in all_dates(stage) if r['date'] in usable for d in ['B', 'A'] for m, vs in models.items() for v in vs]
    jobs.sort(key=lambda j: j[2] != 'DH')   # longest jobs first
    print(f'{len(jobs)} fit jobs for stage {stage}', flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for k, p in enumerate(pool.map(fit_task, jobs), 1):
            if k % 10 == 0 or k == len(jobs): print(f'{k}/{len(jobs)} done', flush=True)


def metrics(q, pred, mask):
    y, x, t, F = (q[k].to_numpy()[mask] for k in ['c', 'x', 'tau', 'forward']); p = np.asarray(pred)[mask]
    a, b = q.market_iv.to_numpy()[mask], iv(p, x, t); ok = np.isfinite(a) & np.isfinite(b)
    e = p - y
    return {'quotes': int(mask.sum()), 'IV_valid': int(ok.sum()), 'IV_RMSE_vol_points': float(100 * np.sqrt(np.mean((a[ok] - b[ok]) ** 2))) if ok.any() else np.nan,
            'price_RMSE_USD': float(np.sqrt(np.mean((e * F) ** 2))), 'price_MAE_USD': float(np.mean(abs(e * F))), 'fwd_price_RMSE': float(np.sqrt(np.mean(e ** 2)))}


def score(stage, variants=None):
    rows = []
    for r in all_dates(stage):
        f = OUT / 'surfaces' / f"{r['date']}.csv"
        if not f.exists(): continue
        q = pd.read_csv(f)
        for design in ['B', 'A']:
            held = q[f'heldout_{design}'].to_numpy()
            for m, vs in (variants or MODELS).items():
                for v in vs:
                    p = OUT / 'fits' / design / r['date'] / f"{m}{'_' + v if v else ''}.json"
                    if not p.exists(): continue
                    j = read(p)
                    if 'error' in j: rows.append({**r, 'design': design, 'model': m, 'variant': v, 'error': j['error']}); continue
                    rows.append({**r, 'design': design, 'model': m, 'variant': v, **metrics(q, j['pred'], held),
                                 **{f'insample_{k}': val for k, val in metrics(q, j['pred'], ~held).items()},
                                 'near_best_starts': j['fit'].get('converged_near_best'), 'fallbacks': j.get('fallbacks')})
    return pd.DataFrame(rows)


def select_variant():
    verify(); assert not (OUT / 'variant_selection.json').exists()
    s = score('validation'); s.to_csv(OUT / 'validation_scores.csv', index=False); out = {'utc': stamp(), 'rule': read(HERE / 'config.json')['variant_selection']}
    for m in ['SH', 'DH']:
        g = s[s.model.eq(m)]; med = g.groupby('variant').IV_RMSE_vol_points.median(); out[m] = med.idxmin(); out[f'{m}_validation_median_IV_RMSE'] = med.to_dict()
    save(OUT / 'variant_selection.json', out); print(out)


def cluster_boot(d, groups, seed, n=5000):
    rng = np.random.default_rng(seed); ids = np.unique(groups); means = np.array([d[groups == g].mean() for g in ids])
    return np.quantile(rng.choice(means, size=(n, len(means))).mean(1), [.025, .975]).tolist(), means


def evaluate():
    c = verify(); sel = read(OUT / 'variant_selection.json'); assert not (OUT / 'results.json').exists()
    s = score('test', {'BS_TERM': [None], 'SH': [sel['SH']], 'DH': [sel['DH']]}); s.to_csv(OUT / 'test_scores.csv', index=False)
    res = {'utc': stamp(), 'variant_selection': sel, 'primary_endpoint': c['primary_endpoint'], 'cells': []}
    ok = s[s.IV_RMSE_vol_points.notna()] if 'IV_RMSE_vol_points' in s else s
    for regime in ['shock', 'calm']:
        for design in ['B', 'A']:
            g = ok[ok.regime.eq(regime) & ok.design.eq(design)]
            piv = g.pivot_table(index=['date', 'episode'] if regime == 'shock' else ['date'], columns='model', values='IV_RMSE_vol_points').dropna().reset_index()
            if piv.empty: continue
            unit = piv.episode.to_numpy() if regime == 'shock' else np.arange(len(piv))
            cell = {'regime': regime, 'design': design, 'dates': len(piv), 'units': int(len(np.unique(unit))),
                    'median_IV_RMSE': {m: float(piv[m].median()) for m in ['BS_TERM', 'SH', 'DH'] if m in piv}}
            for comp in ['SH', 'BS_TERM']:
                d = (piv[comp] - piv.DH).to_numpy(); ci, means = cluster_boot(d, unit, 20260920)
                p = float(stats.wilcoxon(means, alternative='greater').pvalue) if len(means) > 1 and np.any(means != 0) else float('nan')
                cell[f'{comp}_minus_DH'] = {'mean': float(d.mean()), 'median': float(np.median(d)), 'DH_wins_dates': f'{int((d > 0).sum())}/{len(d)}',
                                            'unit_means_positive': f'{int((means > 0).sum())}/{len(means)}', 'wilcoxon_one_sided_p': p, 'cluster_bootstrap95': ci,
                                            'DH_beats': bool(p < .05 and ci[0] > 0)}
            res['cells'].append(cell)
    prim = next((x for x in res['cells'] if x['regime'] == 'shock' and x['design'] == 'B'), None)
    res['primary'] = {'DH_beats_SH': prim and prim['SH_minus_DH']['DH_beats'], 'DH_beats_BS_TERM': prim and prim['BS_TERM_minus_DH']['DH_beats']}
    save(OUT / 'results.json', res); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['freeze', 'validate', 'select-dates', 'fetch', 'fit', 'select-variant', 'evaluate'])
    ap.add_argument('--stage', choices=['validation', 'test'], default='validation'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    {'freeze': freeze, 'validate': validate, 'select-dates': select_dates, 'fetch': fetch, 'select-variant': select_variant, 'evaluate': evaluate,
     'fit': lambda: fit(a.stage, a.workers)}[a.command]()
