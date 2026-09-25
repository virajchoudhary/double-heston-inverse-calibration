"""Hard-coded (published) parameters vs real markets, with the locked DH-PINN.
Commands: freeze -> fetch-spx -> evaluate.  See PROTOCOL.md; nothing is fitted in the primary analysis.
"""
import argparse, datetime as dt, gzip, hashlib, json, re, sys, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; ROOT = EXP.parents[0]; OUT = HERE / 'artifacts'
sys.path.insert(0, str(EXP / 'btc_multifactor_v1')); sys.path.insert(0, str(ROOT))
from engine import exact, black, iv, vega
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN

V4 = EXP / 'nifty_multifactor_v4'
C = json.loads((HERE / 'config.json').read_text())
DH = np.array(C['parameters']['DH_PUB']); SH = np.array(C['parameters']['SH_PUB']); BSVOL = float(np.sqrt(DH[4] + DH[9]))
MODELS = ['BS_FIXED', 'SH_PUB', 'DH_PUB', 'DH_PINN']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, o): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(o, indent=2, default=float) + '\n')
def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def nets():
    out = []
    for seed in [17, 43]:
        n = TorchRegularVariancePINN(factors=2, width=256, depth=5, tau_min=7 / 365, tau_max=2., x_half_width=.36)
        n.load_state_dict(torch.load(V4 / 'artifacts' / 'candidates' / 'C3' / f'pinn_s{seed}' / 'weights.pt')); n.eval(); n.requires_grad_(False); out.append(n)
    return out


NETS = None


def price(model, x, tau, scale=1.):
    """Forward-normalised call price c = C/(D*F) for hard-coded parameters, optionally level-scaled."""
    global NETS
    x, tau = np.broadcast_arrays(np.asarray(x, float), np.asarray(tau, float))
    if model == 'BS_FIXED': return black(x, tau, BSVOL * np.sqrt(scale))
    p = (SH if model == 'SH_PUB' else DH).copy()
    idx = ([1, 2, 4], []) if model == 'SH_PUB' else ([1, 6, 4, 9], [2, 7])
    p[idx[0]] *= scale                                  # theta and v0 scale linearly
    for j in ([2] if model == 'SH_PUB' else idx[1]): p[j] *= np.sqrt(scale)   # sigma scales as sqrt
    if model != 'DH_PINN': return exact(p, x, tau, feller=False)
    if NETS is None: NETS = nets()
    co = torch.tensor(np.column_stack([x, np.full(len(x), p[4]), np.full(len(x), p[9]), tau]), dtype=torch.float64)
    st = torch.tensor(np.stack([p[0:4], p[5:9]])[None], dtype=torch.float64).expand(len(x), 2, 4)
    return np.mean([n.price(co, st).numpy() for n in NETS], 0) * np.exp(-x)


def freeze():
    assert not (OUT / 'manifest.json').exists()
    src = {f: sha(HERE / f) for f in ['config.json', 'PROTOCOL.md', 'run_hardcoded.py']}
    w = {f'pinn_s{s}': sha(V4 / 'artifacts' / 'candidates' / 'C3' / f'pinn_s{s}' / 'weights.pt') for s in [17, 43]}
    save(OUT / 'manifest.json', {'frozen_utc': stamp(), 'config': C, 'source_sha256': src, 'pinn_weights_sha256': w,
                                 'engine_sha256': sha(EXP / 'btc_multifactor_v1' / 'engine.py'), 'spx_fetched_before_freeze': False})
    (OUT / 'manifest.sha256').write_text(sha(OUT / 'manifest.json') + '\n'); print('FROZEN', sha(OUT / 'manifest.json'))


def verify():
    m = json.loads((OUT / 'manifest.json').read_text()); assert sha(OUT / 'manifest.json') == (OUT / 'manifest.sha256').read_text().strip()
    for f, h in m['source_sha256'].items(): assert sha(HERE / f) == h, f'frozen source changed: {f}'
    return m


def fetch_spx():
    verify(); raw = OUT / 'spx_chain.json.gz'
    if not raw.exists():
        u = 'https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json'
        raw.write_bytes(gzip.compress(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120).read()))
    d = json.loads(gzip.decompress(raw.read_bytes())); ts = d['timestamp']; spot = float(d['data']['close'] or d['data']['current_price'])
    f = C['spx_filters']; rows = []
    for o in d['data']['options']:
        mm = re.match(r'^SPX[W]?(\d{6})([CP])(\d{8})$', o['option'])
        if not mm: continue
        ymd, cp, k = mm.groups(); K = int(k) / 1000.
        exp = dt.datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]), 20, tzinfo=dt.timezone.utc)   # 16:00 ET settlement
        tau = (exp - dt.datetime.now(dt.timezone.utc)).total_seconds() / (365 * 86400)
        b, a = float(o['bid']), float(o['ask'])
        if tau <= 0 or not (b > 0 and a > b and a <= 3 * b): continue
        rows.append({'expiry': exp.date().isoformat(), 'cp': cp, 'strike': K, 'tau': tau, 'days': tau * 365, 'mid': (a + b) / 2, 'bid': b, 'ask': a, 'cboe_iv': float(o['iv'])})
    q = pd.DataFrame(rows); audit = {'timestamp': ts, 'spot': spot, 'quotes_after_spread_filter': len(q), 'raw_sha256': sha(raw)}
    # forward and discount per expiry from put-call parity: C - P = D*F - D*K
    keep, fwd = [], {}
    for e, g in q.groupby('expiry'):
        w = g.pivot_table(index='strike', columns='cp', values='mid').dropna()
        if len(w) < 4: continue
        y = (w.C - w.P).to_numpy(); Ks = w.index.to_numpy(); A = np.column_stack([np.ones(len(Ks)), Ks])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        D, F = -coef[1], -coef[0] / coef[1]
        if r2 < .999 or not (0 < D <= 1.05) or not (.5 * spot < F < 2 * spot): continue
        fwd[e] = {'discount': float(D), 'forward': float(F), 'r2': float(r2), 'pairs': int(len(w))}; keep.append(e)
    q = q[q.expiry.isin(keep)].copy(); q['forward'] = q.expiry.map(lambda e: fwd[e]['forward']); q['discount'] = q.expiry.map(lambda e: fwd[e]['discount'])
    q['k'] = q.strike / q.forward; q['x'] = -np.log(q.k)
    q = q[np.where(q.cp.eq('C'), q.k >= 1, q.k < 1)]                                  # OTM only
    q['c'] = np.where(q.cp.eq('C'), q.mid / (q.discount * q.forward), q.mid / (q.discount * q.forward) + 1 - q.k)
    q['market_iv'] = iv(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
    q = q[(q.days >= f['min_days']) & (q.days <= f['max_days']) & (q.x.abs() <= f['max_abs_log_moneyness']) & q.market_iv.between(*f['iv_range'])]
    n = q.groupby('expiry').size(); q = q[q.expiry.isin(n[n >= f['min_quotes_per_expiry']].index)].copy()
    q['vega_w'] = np.maximum(vega(q.x.to_numpy(), q.tau.to_numpy(), q.market_iv.to_numpy()), 1e-4)
    q['date'] = ts[:10]; q.to_csv(OUT / 'spx_surface.csv', index=False)
    audit.update({'expiries_kept': len(keep), 'quotes_scored': len(q), 'forwards': fwd, 'atm_iv_by_expiry':
                  {e: float(g.loc[g.x.abs().idxmin(), 'market_iv']) for e, g in q.groupby('expiry')}})
    save(OUT / 'spx_audit.json', audit); print(f"SPX {ts}: {len(q)} quotes, {q.expiry.nunique()} expiries, spot {spot:.0f}")


def score(q, scale=1.):
    out = {}
    x, t, y, w = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w'])
    for m in MODELS:
        p = price(m, x, t, scale); mi = iv(p, x, t); ok = np.isfinite(mi) & np.isfinite(q.market_iv.to_numpy())
        e = 100 * (mi[ok] - q.market_iv.to_numpy()[ok])
        out[m] = {'IV_RMSE': float(np.sqrt(np.mean(e ** 2))), 'IV_bias': float(np.mean(e)), 'fwd_price_RMSE': float(np.sqrt(np.mean((p - y) ** 2))), 'scored': int(ok.sum())}
    return out


def best_scale(q, model):
    x, t, y, w = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w'])
    lo, hi = C['secondary_scale_analysis']['bounds'] if isinstance(C['secondary_scale_analysis']['bounds'], list) else (0.7, 3.2)
    r = minimize_scalar(lambda s: float(np.mean(((price(model, x, t, s) - y) / w) ** 2)), bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
    return float(r.x)


def surfaces():
    yield 'SPX ' + json.loads((OUT / 'spx_audit.json').read_text())['timestamp'][:10] + ' (calm, VIX 14.2)', pd.read_csv(OUT / 'spx_surface.csv')
    f = C['comparison_domain']
    for asset, exp in [('BTC', 'btc_multifactor_v1'), ('ETH', 'eth_multifactor_v1')]:
        d = json.loads((EXP / exp / 'artifacts' / 'dates.json').read_text())
        dates = [s['date'] for s in d['shock'] if (EXP / exp / 'artifacts' / 'surfaces' / f"{s['date']}.csv").exists()]
        q = pd.concat([pd.read_csv(EXP / exp / 'artifacts' / 'surfaces' / f'{x}.csv') for x in dates])
        q = q[(q.x.abs() <= .36) & (q.days >= 7) & (q.days <= 730)]
        yield f'{asset} shock dates ({len(dates)} days)', q


def evaluate():
    verify(); assert not (OUT / 'results.json').exists()
    res = {'utc': stamp(), 'primary_nothing_fitted': {}, 'secondary_one_scale_fitted': {}, 'buckets': {}}
    for name, q in surfaces():
        res['primary_nothing_fitted'][name] = {'quotes': len(q), 'median_market_iv': float(q.market_iv.median()), **score(q)}
        sc = {m: best_scale(q, m) for m in MODELS}
        res['secondary_one_scale_fitted'][name] = {'scale': sc, **{m: score(q, sc[m])[m] for m in MODELS}}
        b = q.copy(); b['bucket'] = pd.cut(b.days, [0, 30, 90, 365, 730], labels=['<=30d', '30-90d', '90-365d', '>365d'])
        res['buckets'][name] = {str(k): {m: score(g, sc[m])[m]['IV_RMSE'] for m in MODELS} | {'quotes': len(g)} for k, g in b.groupby('bucket', observed=True)}
    save(OUT / 'results.json', res); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('command', choices=['freeze', 'fetch-spx', 'evaluate']); c = a.parse_args().command
    {'freeze': freeze, 'fetch-spx': fetch_spx, 'evaluate': evaluate}[c]()
