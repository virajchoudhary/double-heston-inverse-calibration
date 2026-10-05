"""Find where the three models separate MOST in PRICE space on held-out expiries."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact
from amend01 import predict_bs_expiry

sc = pd.read_csv(HERE / 'artifacts' / 'test_scores.csv')
dates = sorted(sc[(sc.design == 'B') & (sc.stage == 'test')].date.unique())
rows = []
for date in dates:
    f = HERE / 'artifacts' / 'surfaces' / f'{date}.csv'
    fb = HERE / 'artifacts' / 'fits' / 'B' / date
    fa = HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / date / 'BS_EXPIRY.json'
    if not (f.exists() and fb.exists() and fa.exists()): continue
    q = pd.read_csv(f)
    dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
    sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
    bs = json.load(open(fa))['fit']
    for e, sub in q.groupby('expiry'):
        held = bool(sub.heldout_B.all())          # entire expiry held out
        if len(sub) < 6: continue
        # evaluate on a common S/K window where options actually trade
        lo, hi = max(0.70, float(np.exp(sub.x.min()))), min(1.45, float(np.exp(sub.x.max())))
        if hi - lo < 0.25: continue
        ms = np.linspace(lo, hi, 300); xs = np.log(ms)
        tau0 = float(sub.tau.median()); taus = np.full_like(xs, tau0)
        C = {'DH': exact(dh, xs, taus, False) * ms, 'SH': exact(sh, xs, taus, False) * ms,
             'BS': predict_bs_expiry(bs, np.array([e] * len(xs)), xs, taus) * ms}
        yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
        gbs = float(np.abs(C['BS'] - C['DH']).max()); gsh = float(np.abs(C['SH'] - C['DH']).max())
        rows.append({'date': date, 'expiry': e, 'days': float(sub.days.median()), 'n': len(sub),
                     'held_out_expiry': held, 'y_range': yr,
                     'gap_BS': gbs, 'gap_SH': gsh,
                     'BS_pct_of_range': 100 * gbs / yr, 'SH_pct_of_range': 100 * gsh / yr,
                     'min_sep_pct': 100 * min(gbs, gsh) / yr})
d = pd.DataFrame(rows)
d.to_csv(HERE / 'figures_dh_advantage' / 'separation_sweep.csv', index=False)
print('panels scanned:', len(d), '| held-out expiries:', int(d.held_out_expiry.sum()))
print()
print('=== HELD-OUT expiries, ranked by SMALLEST separation being largest ===')
print('(min_sep_pct = both BS and SH are at least this far from DH, as % of the panel y-range)')
h = d[d.held_out_expiry]
print(h.nlargest(12, 'min_sep_pct')[['date','expiry','days','n','BS_pct_of_range','SH_pct_of_range','min_sep_pct']].round(2).to_string(index=False))
print()
print('=== compare: CALIBRATED (not held out) expiries ===')
c = d[~d.held_out_expiry]
print('median min_sep_pct  held-out: %.2f%%   calibrated: %.2f%%' % (h.min_sep_pct.median(), c.min_sep_pct.median()))
print('max    min_sep_pct  held-out: %.2f%%   calibrated: %.2f%%' % (h.min_sep_pct.max(), c.min_sep_pct.max()))
