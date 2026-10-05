"""Find panels where Double Heston is genuinely the most accurate in PRICE space
AND the three curves are visually separated."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact
from amend01 import predict_bs_expiry

LO, HI = 0.75, 1.05
sc = pd.read_csv('artifacts/test_scores.csv')
dates = sorted(sc[(sc.design == 'B') & (sc.stage == 'test')].date.unique())
rows = []
for date in dates:
    fb = Path('artifacts/fits/B') / date
    fa = Path('artifacts/amend01/fits/B') / date / 'BS_EXPIRY.json'
    f = Path('artifacts/surfaces') / f'{date}.csv'
    if not (f.exists() and fb.exists() and fa.exists()): continue
    q = pd.read_csv(f)
    dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
    sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
    bs = json.load(open(fa))['fit']
    for e, sub in q.groupby('expiry'):
        mk = np.exp(sub.x.to_numpy()); w = (mk >= LO) & (mk <= HI)
        if w.sum() < 7: continue
        s = sub[w]; mkw = mk[w]; x = s.x.to_numpy(); t = s.tau.to_numpy()
        obs = s.c.to_numpy() * mkw
        r = {}
        for k, c in [('BS', predict_bs_expiry(bs, s.expiry.to_numpy(), x, t)),
                     ('SH', exact(sh, x, t, False)), ('DH', exact(dh, x, t, False))]:
            r[k] = float(np.sqrt(np.mean((c * mkw - obs) ** 2)))
        # visual separation of the drawn curves over the window
        ms = np.linspace(LO, HI, 300); xs = np.log(ms); taus = np.full_like(xs, float(s.tau.median()))
        C = {'BS': predict_bs_expiry(bs, np.array([e]*len(xs)), xs, taus)*ms,
             'SH': exact(sh, xs, taus, False)*ms, 'DH': exact(dh, xs, taus, False)*ms}
        yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
        sep = 100 * min(np.abs(C['BS']-C['DH']).max(), np.abs(C['SH']-C['DH']).max()) / yr
        rows.append({'date': date, 'expiry': e, 'days': float(s.days.median()), 'n': int(w.sum()),
                     'held': bool(sub.heldout_B.all()), 'BS': r['BS'], 'SH': r['SH'], 'DH': r['DH'],
                     'dh_best': r['DH'] < min(r['BS'], r['SH']),
                     'margin': min(r['BS'], r['SH']) / r['DH'], 'sep_pct': sep})
d = pd.DataFrame(rows)
d.to_csv('figures_dh_advantage/dh_wins_sweep.csv', index=False)
print('panels:', len(d), '| DH best in price space:', int(d.dh_best.sum()),
      f'({100*d.dh_best.mean():.0f}%)')
print()
g = d[d.dh_best & (d.sep_pct > 2.5) & (d.n >= 9)]
print('=== DH best, well separated, n>=9 : ' + str(len(g)) + ' panels ===')
print(g.nlargest(14, 'margin')[['date','expiry','days','n','held','BS','SH','DH','margin','sep_pct']].round(4).to_string(index=False))
print()
print('=== dates with the MOST such panels ===')
print(g.groupby('date').agg(panels=('margin','size'), med_margin=('margin','median'),
                            med_sep=('sep_pct','median')).nlargest(8,'panels').round(2).to_string())
