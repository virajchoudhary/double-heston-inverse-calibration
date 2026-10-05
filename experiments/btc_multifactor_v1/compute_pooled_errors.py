"""Per-quote implied-volatility errors on the design-B held-out quotes, all 50 test dates."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, iv as inv_iv
from amend01 import predict_bs_expiry

sc = pd.read_csv(HERE / 'artifacts' / 'test_scores.csv')
dates = sorted(sc[(sc.design == 'B') & (sc.stage == 'test')].date.unique())
rows = []
for date in dates:
    f = HERE / 'artifacts' / 'surfaces' / f'{date}.csv'
    fb = HERE / 'artifacts' / 'fits' / 'B' / date
    fa = HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / date / 'BS_EXPIRY.json'
    if not (f.exists() and fb.exists() and fa.exists()):
        continue
    q = pd.read_csv(f)
    q = q[q.heldout_B == True]                      # predeclared held-out set
    if not len(q):
        continue
    dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
    sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
    bs = json.load(open(fa))['fit']
    x, t, e = q.x.to_numpy(), q.tau.to_numpy(), q.expiry.to_numpy()
    E = {}
    for k, c in [('DH', exact(dh, x, t, False)), ('SH', exact(sh, x, t, False)),
                 ('BS', predict_bs_expiry(bs, e, x, t))]:
        E[k] = 100 * (inv_iv(c, x, t) - q.market_iv.to_numpy())
    # ONE common mask: a quote is kept only if all three models invert to a finite IV,
    # so the three error series are paired quote-by-quote.
    ok = np.isfinite(E['DH']) & np.isfinite(E['SH']) & np.isfinite(E['BS'])
    for k in ['DH', 'SH', 'BS']:
        rows.append(pd.DataFrame({'date': date, 'model': k, 'err': E[k][ok],
                                  'abs_err': np.abs(E[k][ok]), 'moneyness': np.exp(x)[ok],
                                  'days': q.days.to_numpy()[ok],
                                  'qid': np.arange(ok.sum())}))
d = pd.concat(rows, ignore_index=True)
d.to_csv(HERE / 'figures_dh_advantage' / 'pooled_heldout_errors.csv', index=False)
print('dates:', d.date.nunique(), '| quotes per model:', len(d) // 3)
print()
print(d.groupby('model').abs_err.describe()[['count', 'mean', '50%', '75%', 'max']].round(3).to_string())
