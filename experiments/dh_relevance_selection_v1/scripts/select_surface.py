"""Ex-ante DH-relevance selection. Market data only: no pricing model is evaluated here.

Implements SELECTION_RULE.md exactly. Writes candidates.csv and DATA_SELECTION_REPORT.md.
No Black-Scholes, Single Heston, Double Heston or PINN error appears anywhere in this file.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
EXP = OUT.parent
sys.path.insert(0, str(EXP / 'dh_pinn_v5' / 'multi_surface'))
from scan import surface, atm_curve, chain                       # frozen surface builder (market data only)
import time


def polite(sym, tries=5):
    """Fetch with backoff; chains are cached on disk by scan.chain()."""
    for i in range(tries):
        try:
            chain(sym); return True
        except Exception as e:
            if '429' not in repr(e) and i == tries - 1: raise
            time.sleep(4 * (i + 1))
    return False

SURF = OUT / 'surfaces'; SURF.mkdir(exist_ok=True)
ERROR_KNOWN = {'_SPX', '_NDX', '_RUT', '_DJX', 'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'XLE', 'JPM', 'AAPL', 'MSFT', 'NVDA', 'TSLA'}
CANDIDATES = sorted(ERROR_KNOWN | {
    'SMH', 'XLF', 'XLU', 'XLV', 'XLI', 'XLP', 'XLK', 'XLY', 'XLRE', 'XBI', 'EEM', 'EFA', 'FXI', 'EWZ', 'EWJ',
    'SLV', 'USO', 'UNG', 'HYG', 'LQD', 'IEF', 'VXX', 'AMZN', 'GOOGL', 'META', 'AMD', 'INTC', 'BA', 'XOM', 'CVX',
    'WMT', 'KO', 'PFE', 'T', 'V', 'MA', 'UNH', 'HD', 'CAT', 'GE', 'F', 'BAC', 'C', 'WFC', 'GS', 'MS'})
VOL_LO, VOL_HI = np.sqrt(0.0255 * 0.7), np.sqrt(0.0255 * 3.2)     # PINN level domain, from market ATM vol


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def skew_by_expiry(q, half=0.2):
    rows = []
    for e, v in q.groupby('expiry'):
        v = v.sort_values('x')
        if len(v) < 5 or v.x.min() > -half or v.x.max() < half: continue
        iv = np.interp([half, -half, 0.], v.x.to_numpy(), v.market_iv.to_numpy())
        rows.append({'tau': float(v.tau.median()), 'sk_val': float(iv[0] - iv[1]), 'atm': float(iv[2])})
    return pd.DataFrame(rows)


def components(q):
    a = atm_curve(q)
    tau, w, iv = a.tau.to_numpy(), a.w.to_numpy(), a.iv.to_numpy()

    def resid(p):
        th, v0, kap = np.exp(p)
        return th * tau + (v0 - th) * (1 - np.exp(-kap * tau)) / kap - w
    best = None
    for k0 in [.3, 1., 3., 10.]:
        r = least_squares(resid, np.log([max(w[-1] / tau[-1], 1e-4), max(w[0] / tau[0], 1e-4), k0]), max_nfev=2000)
        if best is None or r.cost < best.cost: best = r
    S1 = float(np.sqrt(np.mean(best.fun ** 2)) / np.mean(w))
    lt = np.log(tau); S2 = float(np.sqrt(np.mean(np.diff(iv, 2) ** 2)) / np.mean(iv)) if len(iv) > 3 else np.nan
    sk = skew_by_expiry(q)
    S3 = float(np.mean(np.abs(sk.sk_val)) / np.mean(sk.atm)) if len(sk) else np.nan
    S4 = float((sk.sk_val.max() - sk.sk_val.min()) / np.mean(sk.atm)) if len(sk) > 1 else np.nan
    short = a[a.tau * 365 < 60].iv.median(); long = a[a.tau * 365 > 180].iv.median()
    S5 = float(abs(short - long) / np.mean(iv)) if np.isfinite(short) and np.isfinite(long) else np.nan
    return {'S1_one_timescale_misfit': S1, 'S2_term_curvature': S2, 'S3_skew_magnitude': S3,
            'S4_skew_variation': S4, 'S5_short_long_separation': S5,
            'atm_iv_short': float(a.iv.iloc[0]), 'atm_iv_long': float(a.iv.iloc[-1]),
            'expiries': int(q.expiry.nunique()), 'quotes': int(len(q)),
            'expiries_under_90d': int((a.tau * 365 < 90).sum()), 'expiries_over_180d': int((a.tau * 365 > 180).sum()),
            'single_timescale_kappa': float(np.exp(best.x[2]))}


def domain_ok(r):
    reasons = []
    if r['expiries'] < 6: reasons.append('fewer than 6 expiries')
    if r['quotes'] < 150: reasons.append('fewer than 150 quotes')
    if not (VOL_LO <= r['atm_iv_short'] <= VOL_HI): reasons.append(f"short ATM IV {r['atm_iv_short']*100:.1f}% outside [{VOL_LO*100:.1f}, {VOL_HI*100:.1f}]%")
    if r['expiries_under_90d'] < 3: reasons.append('fewer than 3 expiries under 90d')
    if r['expiries_over_180d'] < 2: reasons.append('fewer than 2 expiries over 180d')
    return (len(reasons) == 0), '; '.join(reasons)


def main():
    rows = []
    for sym in CANDIDATES:
        try:
            polite(sym); time.sleep(1.2)
            q = surface(sym)
            if q is None:
                rows.append({'symbol': sym, 'usable': False, 'domain_pass': False, 'reject_reason': 'no usable surface',
                             'error_known': sym in ERROR_KNOWN}); print(f'{sym:7} unusable', flush=True); continue
            q.to_csv(SURF / f'{sym}.csv', index=False)
            c = components(q); ok, why = domain_ok(c)
            rows.append({'symbol': sym, 'usable': True, 'domain_pass': ok, 'reject_reason': why,
                         'error_known': sym in ERROR_KNOWN, **c})
            print(f"{sym:7} {'PASS' if ok else 'fail'} S1 {c['S1_one_timescale_misfit']:.4f} S2 {c['S2_term_curvature']:.4f} "
                  f"S3 {c['S3_skew_magnitude']:.3f} S4 {c['S4_skew_variation']:.3f} S5 {c['S5_short_long_separation']:.3f}"
                  f"{'' if ok else '  <- ' + why}", flush=True)
        except Exception as e:
            rows.append({'symbol': sym, 'usable': False, 'domain_pass': False, 'reject_reason': f'error {e!r}', 'error_known': sym in ERROR_KNOWN})
            print(f'{sym:7} error {e!r}', flush=True)
    d = pd.DataFrame(rows)
    elig = d[d.domain_pass & d[['S1_one_timescale_misfit', 'S2_term_curvature', 'S3_skew_magnitude', 'S4_skew_variation', 'S5_short_long_separation']].notna().all(axis=1)].copy()
    W = {'S1_one_timescale_misfit': .40, 'S2_term_curvature': .20, 'S3_skew_magnitude': .15, 'S4_skew_variation': .15, 'S5_short_long_separation': .10}
    for k in W:
        z = (elig[k] - elig[k].mean()) / elig[k].std(ddof=0)
        elig['z_' + k] = z
    elig['DH_SCORE'] = sum(w * elig['z_' + k] for k, w in W.items())
    elig = elig.sort_values('DH_SCORE', ascending=False)
    d = d.merge(elig[['symbol', 'DH_SCORE'] + ['z_' + k for k in W]], on='symbol', how='left')
    d.to_csv(OUT / 'candidates.csv', index=False)
    pick = elig.iloc[0]['symbol']
    sel = {'rule_sha256': sha(OUT / 'SELECTION_RULE.md'), 'weights': W, 'candidates': int(len(d)),
           'usable': int(d.usable.sum()), 'domain_pass': int(d.domain_pass.sum()), 'scored': int(len(elig)),
           'selected': pick, 'selected_score': float(elig.iloc[0]['DH_SCORE']),
           'selected_error_known_before_selection': bool(elig.iloc[0]['error_known']),
           'ranking': [{'symbol': r.symbol, 'DH_SCORE': float(r.DH_SCORE), 'error_known': bool(r.error_known)} for r in elig.itertuples()]}
    (OUT / 'selection.json').write_text(json.dumps(sel, indent=2, default=float) + '\n')

    lines = ['# Data selection report (ex ante)', '',
             f'Rule: `SELECTION_RULE.md`, SHA-256 `{sel["rule_sha256"]}`.', '',
             'No pricing-model error of any kind appears in this report. Errors were computed only after this',
             'file was written and hashed.', '',
             f'Candidates examined: **{len(d)}**.  Usable surfaces: **{int(d.usable.sum())}**.  '
             f'Passed the domain filter: **{int(d.domain_pass.sum())}**.  Scored: **{len(elig)}**.', '',
             '## Selected surface', '',
             f'**{pick}**, DH-relevance score **{sel["selected_score"]:.3f}**.  '
             f'Errors for this surface were {"ALREADY KNOWN" if sel["selected_error_known_before_selection"] else "NOT known"} before selection.', '',
             '## Ranking of surfaces that passed the domain filter', '',
             '| rank | symbol | error status | S1 one-timescale misfit | S2 curvature | S3 skew | S4 skew variation | S5 short-vs-long | DH score |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for i, r in enumerate(elig.itertuples(), 1):
        lines.append(f'| {i} | {r.symbol} | {"error known" if r.error_known else "error blind"} | '
                     f'{r.S1_one_timescale_misfit:.4f} | {r.S2_term_curvature:.4f} | {r.S3_skew_magnitude:.3f} | '
                     f'{r.S4_skew_variation:.3f} | {r.S5_short_long_separation:.3f} | **{r.DH_SCORE:.3f}** |')
    rej = d[~d.domain_pass]
    lines += ['', '## Rejected before scoring', '', '| symbol | reason |', '|---|---|']
    for r in rej.itertuples(): lines.append(f'| {r.symbol} | {r.reject_reason} |')
    lines += ['', '## Notes', '',
              '- The domain filter uses market quotes only: short-end at-the-money implied volatility must lie in',
              f'  [{VOL_LO*100:.1f}%, {VOL_HI*100:.1f}%], the level range the PINN was trained on, so no surface is selected that would force',
              '  the network to extrapolate.',
              '- Components are standardised as z-scores across the surfaces that passed the filter, then combined',
              '  with the fixed weights 0.40 / 0.20 / 0.15 / 0.15 / 0.10.',
              '- Sixteen symbols had model errors computed earlier in this project and are labelled `error known`;',
              '  the rest are labelled `error blind`. The identical score is applied to both groups.']
    (OUT / 'DATA_SELECTION_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\nSELECTED', pick, 'score %.3f' % sel['selected_score'], '| error_known:', sel['selected_error_known_before_selection'])
    print('top 5:', [(r.symbol, round(r.DH_SCORE, 3)) for r in elig.head(5).itertuples()])


if __name__ == '__main__':
    main()
