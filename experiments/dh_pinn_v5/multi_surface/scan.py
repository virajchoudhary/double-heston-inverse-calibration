"""Fetch today's chains, build surfaces, score them by the model-free rule in RULE.md."""
import gzip, json, re, sys, datetime as dt, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent; V5 = HERE.parent
sys.path.insert(0, str(V5.parent / 'btc_multifactor_v1'))
from engine import iv as inv_iv, vega
SYMS = ['_SPX', '_NDX', '_RUT', '_DJX', 'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'XLE', 'JPM', 'AAPL', 'MSFT', 'NVDA', 'TSLA']
RAW = HERE / 'raw'; RAW.mkdir(exist_ok=True)


def chain(sym):
    f = RAW / f'{sym}.json.gz'
    if not f.exists():
        u = f'https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json'
        f.write_bytes(gzip.compress(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120).read()))
    return json.loads(gzip.decompress(f.read_bytes()))


def surface(sym):
    d = chain(sym); spot = float(d['data']['close'] or d['data']['current_price']); rows = []
    root = sym.lstrip('_')
    for o in d['data']['options']:
        m = re.match(rf'^{root}[A-Z]?(\d{{6}})([CP])(\d{{8}})$', o['option'])
        if not m: continue
        ymd, cp, k = m.groups(); K = int(k) / 1000.
        exp = dt.datetime(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]), 20, tzinfo=dt.timezone.utc)
        tau = (exp - dt.datetime.now(dt.timezone.utc)).total_seconds() / (365 * 86400)
        b, a = float(o['bid']), float(o['ask'])
        if tau <= 0 or not (b > 0 and a > b and a <= 3 * b): continue
        rows.append({'expiry': exp.date().isoformat(), 'cp': cp, 'strike': K, 'tau': tau, 'days': tau * 365, 'mid': (a + b) / 2})
    q = pd.DataFrame(rows)
    if q.empty: return None
    keep, fwd = [], {}
    for e, g in q.groupby('expiry'):
        w = g.pivot_table(index='strike', columns='cp', values='mid').dropna()
        if len(w) < 4: continue
        y = (w.C - w.P).to_numpy(); Ks = w.index.to_numpy(); A = np.column_stack([np.ones(len(Ks)), Ks])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        D, F = -coef[1], -coef[0] / coef[1]
        if r2 < .999 or not (0 < D <= 1.05) or not (.5 * spot < F < 2 * spot): continue
        fwd[e] = {'discount': float(D), 'forward': float(F)}; keep.append(e)
    q = q[q.expiry.isin(keep)].copy()
    if q.empty: return None
    q['forward'] = q.expiry.map(lambda e: fwd[e]['forward']); q['discount'] = q.expiry.map(lambda e: fwd[e]['discount'])
    q['k'] = q.strike / q.forward; q['x'] = -np.log(q.k)
    q = q[np.where(q.cp.eq('C'), q.k >= 1, q.k < 1)]
    q['c'] = np.where(q.cp.eq('C'), q.mid / (q.discount * q.forward), q.mid / (q.discount * q.forward) + 1 - q.k)
    q['market_iv'] = inv_iv(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
    q = q[(q.days >= 7) & (q.days <= 730) & (q.x.abs() <= .36) & q.market_iv.between(.02, 3.)]
    n = q.groupby('expiry').size(); q = q[q.expiry.isin(n[n >= 6].index)].copy()
    if q.expiry.nunique() < 6: return None
    q['vega_w'] = np.maximum(vega(q.x.to_numpy(), q.tau.to_numpy(), q.market_iv.to_numpy()), 1e-4)
    q['symbol'] = sym; q['spot'] = spot
    return q


def atm_curve(q):
    rows = []
    for e, v in q.groupby('expiry'):
        v = v.sort_values('x')
        if len(v) < 3 or not (v.x.min() < 0 < v.x.max()): continue
        rows.append({'tau': float(v.tau.median()), 'iv': float(np.interp(0., v.x.to_numpy(), v.market_iv.to_numpy()))})
    a = pd.DataFrame(rows).sort_values('tau'); a['w'] = a.iv ** 2 * a.tau
    return a


def score(a):
    """Relative misfit of the best SINGLE-timescale total-variance curve (market data only)."""
    tau, w = a.tau.to_numpy(), a.w.to_numpy()
    def resid(p):
        th, v0, kap = np.exp(p)
        return th * tau + (v0 - th) * (1 - np.exp(-kap * tau)) / kap - w
    best = None
    for k0 in [.3, 1., 3., 10.]:
        r = least_squares(resid, np.log([max(w[-1] / tau[-1], 1e-4), max(w[0] / tau[0], 1e-4), k0]), max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    return float(np.sqrt(np.mean(best.fun ** 2)) / np.mean(w)), np.exp(best.x).tolist()


if __name__ == '__main__':
    out = {}
    (HERE / 'surfaces').mkdir(exist_ok=True)
    for s in SYMS:
        try:
            q = surface(s)
            if q is None: print(f'{s:6} unusable'); continue
            q.to_csv(HERE / 'surfaces' / f'{s}.csv', index=False)
            a = atm_curve(q); sc, p = score(a)
            out[s] = {'quotes': len(q), 'expiries': int(q.expiry.nunique()), 'spot': float(q.spot.iloc[0]),
                      'atm_iv_short': float(a.iv.iloc[0]), 'atm_iv_long': float(a.iv.iloc[-1]),
                      'two_timescale_score': sc, 'single_timescale_fit': {'theta': p[0], 'v0': p[1], 'kappa': p[2]},
                      'atm_curve': a.to_dict('records')}
            print(f"{s:6} {out[s]['quotes']:6} quotes {out[s]['expiries']:3} expiries  atm {a.iv.iloc[0]*100:5.1f}% -> {a.iv.iloc[-1]*100:5.1f}%  score {sc:.4f}  kappa {p[2]:.2f}", flush=True)
        except Exception as e:
            print(f'{s:6} error {e!r}')
    (HERE / 'scan.json').write_text(json.dumps(out, indent=2, default=float) + '\n')
