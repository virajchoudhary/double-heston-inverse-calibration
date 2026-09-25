"""Deribit HYPE_USDC option surfaces (same code as XRP) from the public historical trade archive.

Same pipeline as btc_multifactor_v1/data.py, adapted to linear USDC-quoted options:
strikes use 'd' as the decimal point, the price is in USDC (undiscounted Black-76),
and the forward is solved in price space rather than as K/F.
"""
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

BTC = Path(__file__).resolve().parents[1] / 'btc_multifactor_v1'; sys.path.insert(0, str(BTC))
from data import expiry_of, forward_price, _get
from engine import iv as implied_vol, vega

UTC = dt.timezone.utc


def fetch(date, hours, url, prefix, cache):
    """Every option trade with the prefix in [date hours[0]:00, hours[1]:00) UTC, cached gzip; returns trades and file sha256."""
    path = Path(cache) / f'{date}.json.gz'
    if not path.exists():
        d = dt.date.fromisoformat(date); base = dt.datetime(d.year, d.month, d.day, tzinfo=UTC)
        s = int((base + dt.timedelta(hours=hours[0])).timestamp() * 1000); e = int((base + dt.timedelta(hours=hours[1])).timestamp() * 1000)
        seen, trades = set(), []
        while True:
            r = _get(f'{url}&start_timestamp={s}&end_timestamp={e}&count=1000&sorting=asc')
            new = [t for t in r['trades'] if t['trade_id'] not in seen]
            for t in new: seen.add(t['trade_id']); trades.append(t)
            if not r['has_more'] or not r['trades']: break
            last = r['trades'][-1]['timestamp']; s = last if new else last + 1
            time.sleep(.05)
        path.write_bytes(gzip.compress(json.dumps([t for t in trades if t['instrument_name'].startswith(prefix)]).encode()))
    trades = json.loads(gzip.decompress(path.read_bytes()))
    return trades, hashlib.sha256(path.read_bytes()).hexdigest()


def solve_f(price, strike, sigma, tau, call, index):
    """Forward F such that F * forward_price(K/F) equals the USDC trade price."""
    if not (price > 0) or not (sigma > 0) or not (tau > 0): return np.nan
    f = lambda lf: np.exp(lf) * forward_price(strike / np.exp(lf), sigma, tau, call) - price
    try: return float(np.exp(brentq(f, np.log(index) - 1, np.log(index) + 1, xtol=1e-14, maxiter=200)))
    except ValueError: return np.nan


def parse(name):
    under, code, strike, cp = name.split('-')
    return code, float(strike.replace('d', '.')), cp


def clean(trades, date, cfg):
    f = cfg['filters']; rows = []
    for t in trades:
        code, strike, cp = parse(t['instrument_name'])
        ts = dt.datetime.fromtimestamp(t['timestamp'] / 1000, UTC); exp = expiry_of(code)
        tau = (exp - ts).total_seconds() / (365 * 86400)
        if tau <= 0: continue
        rows.append({'instrument': t['instrument_name'], 'expiry': exp.date().isoformat(), 'strike': strike, 'cp': cp, 'ts': t['timestamp'], 'tau': tau,
                     'price_usdc': float(t['price']), 'mark_usdc': float(t['mark_price']), 'trade_iv': float(t['iv']) / 100, 'index': float(t['index_price'])})
    a = pd.DataFrame(rows)
    audit = {'date': date, 'trades': len(trades), 'instruments_traded': int(a.instrument.nunique()) if len(a) else 0}
    if a.empty: return a, {**audit, 'usable': False, 'reason': 'no trades'}
    a['f_solved'] = [solve_f(p, k, s, t, c == 'C', i) for p, k, s, t, c, i in zip(a.price_usdc, a.strike, a.trade_iv, a.tau, a.cp, a['index'])]
    a['ratio'] = a.f_solved / a['index']
    ratio = a.groupby('expiry').ratio.median(); audit['forward_ratio_by_expiry'] = {e: float(r) for e, r in ratio.items()}
    q = a.sort_values('ts').groupby('instrument').tail(1).copy()
    q['forward'] = q.expiry.map(ratio) * q['index']; q['k'] = q.strike / q.forward; q['x'] = -np.log(q.k)
    q['otm'] = np.where(q.cp.eq('C'), q.k >= 1, q.k < 1)
    q['c'] = np.where(q.cp.eq('C'), q.price_usdc / q.forward, q.price_usdc / q.forward + 1 - q.k)
    q['c_mark'] = np.where(q.cp.eq('C'), q.mark_usdc / q.forward, q.mark_usdc / q.forward + 1 - q.k)
    q['market_iv'] = implied_vol(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
    q['days'] = q.tau * 365
    keep = (q.days >= f['min_days']) & (q.days <= f['max_days']) & (q.otm if f['otm_only'] else True) & (q.price_usdc >= f['min_price_usdc']) \
        & q.market_iv.between(*f['iv_range']) & (q.x.abs() <= f['max_abs_log_moneyness']) & np.isfinite(q.forward)
    audit['dropped_by_filter'] = int((~keep).sum()); q = q[keep].copy()
    counts = q.groupby('expiry').size(); good = counts[counts >= f['min_quotes_per_expiry']].index
    audit['dropped_thin_expiries'] = sorted(set(counts.index) - set(good)); q = q[q.expiry.isin(good)].copy()
    audit['expiries'] = int(q.expiry.nunique()); audit['quotes'] = len(q)
    audit['usable'] = bool(q.expiry.nunique() >= f['min_expiries_per_date'])
    if not audit['usable']: audit['reason'] = 'fewer than the minimum number of expiries after filters'
    q['vega_w'] = np.maximum(vega(q.x.to_numpy(), q.tau.to_numpy(), q.market_iv.to_numpy()), 1e-4)
    q = q.sort_values(['expiry', 'strike']); q['heldout_A'] = q.groupby('expiry').cumcount() % 2 == 1
    order = sorted(q.expiry.unique()); rank = {e: i for i, e in enumerate(order)}
    q['heldout_B'] = q.expiry.map(lambda e: rank[e] % 2 == 1 and rank[e] != len(order) - 1)
    q['date'] = date
    return q.reset_index(drop=True), audit
