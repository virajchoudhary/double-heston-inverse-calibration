"""Deribit BTC option surfaces from the public historical trade archive.

Per date: fetch every option trade in the fixed UTC window, recover each expiry's
forward from the exchange's own (price, implied vol) pairs, keep the last trade per
instrument, convert OTM puts to forward-normalised calls by parity and apply the
predeclared filters. No quote is invented, interpolated or dropped by model error.
"""
import datetime as dt
import gzip
import hashlib
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from engine import iv as implied_vol, vega

UTC = dt.timezone.utc
MONTHS = {m: i for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}


def expiry_of(code):
    return dt.datetime(2000 + int(code[-2:]), MONTHS[code[-5:-2]], int(code[:-5]), 8, tzinfo=UTC)


def _get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60) as r:
                return json.load(r)['result']
        except Exception:
            if i == tries - 1: raise
            time.sleep(2 * (i + 1))


def fetch(date, hours, url, cache):
    """Every option trade in [date hours[0]:00, hours[1]:00) UTC, cached gzip; returns trades and file sha256."""
    path = Path(cache) / f'{date}.json.gz'
    if not path.exists():
        d = dt.date.fromisoformat(date)
        s = int(dt.datetime(d.year, d.month, d.day, hours[0], tzinfo=UTC).timestamp() * 1000)
        e = int(dt.datetime(d.year, d.month, d.day, hours[1], tzinfo=UTC).timestamp() * 1000)
        seen, trades = set(), []
        while True:
            r = _get(f'{url}&start_timestamp={s}&end_timestamp={e}&count=1000&sorting=asc')
            new = [t for t in r['trades'] if t['trade_id'] not in seen]
            for t in new: seen.add(t['trade_id']); trades.append(t)
            if not r['has_more'] or not r['trades']: break
            last = r['trades'][-1]['timestamp']; s = last if new else last + 1
            time.sleep(.05)
        path.write_bytes(gzip.compress(json.dumps(trades).encode()))
    trades = json.loads(gzip.decompress(path.read_bytes()))
    return trades, hashlib.sha256(path.read_bytes()).hexdigest()


def forward_price(k, sigma, tau, call):
    root = sigma * np.sqrt(tau); d1 = -np.log(k) / root + root / 2; d2 = d1 - root
    return ndtr(d1) - k * ndtr(d2) if call else k * ndtr(-d2) - ndtr(-d1)


def solve_k(price, sigma, tau, call):
    """K/F consistent with the exchange convention price = Black76/F (no discount)."""
    if not (0 < price < 1) or not (sigma > 0) or not (tau > 0): return np.nan
    f = lambda lk: forward_price(np.exp(lk), sigma, tau, call) - price
    try: return float(np.exp(brentq(f, np.log(1e-4), np.log(1e4), xtol=1e-14, maxiter=200)))
    except ValueError: return np.nan


def clean(trades, date, cfg):
    f = cfg['filters']; rows = []
    for t in trades:
        cur, code, strike, cp = t['instrument_name'].split('-')
        ts = dt.datetime.fromtimestamp(t['timestamp'] / 1000, UTC); exp = expiry_of(code)
        tau = (exp - ts).total_seconds() / (365 * 86400)
        if tau <= 0: continue
        rows.append({'instrument': t['instrument_name'], 'expiry': exp.date().isoformat(), 'strike': float(strike), 'cp': cp, 'ts': t['timestamp'],
                     'tau': tau, 'price_btc': float(t['price']), 'mark_btc': float(t['mark_price']), 'trade_iv': float(t['iv']) / 100, 'index': float(t['index_price'])})
    a = pd.DataFrame(rows)
    audit = {'date': date, 'trades': len(trades), 'instruments_traded': int(a.instrument.nunique()) if len(a) else 0}
    if a.empty: return a, {**audit, 'usable': False, 'reason': 'no trades'}
    a['k_solved'] = [solve_k(p, s, t, c == 'C') for p, s, t, c in zip(a.price_btc, a.trade_iv, a.tau, a.cp)]
    a['ratio'] = a.strike / a.k_solved / a['index']
    ratio = a.groupby('expiry').ratio.median(); audit['forward_ratio_by_expiry'] = {e: float(r) for e, r in ratio.items()}
    q = a.sort_values('ts').groupby('instrument').tail(1).copy()
    q['forward'] = q.expiry.map(ratio) * q['index']; q['k'] = q.strike / q.forward; q['x'] = -np.log(q.k)
    q['otm'] = np.where(q.cp.eq('C'), q.k >= 1, q.k < 1)
    q['c'] = np.where(q.cp.eq('C'), q.price_btc, q.price_btc + 1 - q.k)
    q['c_mark'] = np.where(q.cp.eq('C'), q.mark_btc, q.mark_btc + 1 - q.k)
    q['market_iv'] = implied_vol(q.c.to_numpy(), q.x.to_numpy(), q.tau.to_numpy())
    q['days'] = q.tau * 365
    keep = (q.days >= f['min_days']) & (q.days <= f['max_days']) & (q.otm if f['otm_only'] else True) & (q.price_btc >= f['min_price_btc']) \
        & q.market_iv.between(*f['iv_range']) & (q.x.abs() <= f['max_abs_log_moneyness']) & np.isfinite(q.forward)
    audit['dropped_by_filter'] = int((~keep).sum()); q = q[keep].copy()
    counts = q.groupby('expiry').size(); good = counts[counts >= f['min_quotes_per_expiry']].index
    audit['dropped_thin_expiries'] = sorted(set(counts.index) - set(good)); q = q[q.expiry.isin(good)].copy()
    audit['expiries'] = int(q.expiry.nunique()); audit['quotes'] = len(q)
    audit['usable'] = bool(q.expiry.nunique() >= f['min_expiries_per_date'])
    if not audit['usable']: audit['reason'] = 'fewer than the minimum number of expiries after filters'
    q['vega_w'] = np.maximum(vega(q.x.to_numpy(), q.tau.to_numpy(), q.market_iv.to_numpy()), 1e-4)
    # Design A: odd-ranked strikes within each expiry are held out.
    q = q.sort_values(['expiry', 'strike']); q['heldout_A'] = q.groupby('expiry').cumcount() % 2 == 1
    # Design B: odd-ranked expiries (by maturity) are held out, never the longest.
    order = sorted(q.expiry.unique()); rank = {e: i for i, e in enumerate(order)}
    q['heldout_B'] = q.expiry.map(lambda e: rank[e] % 2 == 1 and rank[e] != len(order) - 1)
    q['date'] = date
    return q.reset_index(drop=True), audit
