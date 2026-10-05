"""BTC: the full range of Double-Heston outcomes across the 50 test dates.

Best / typical / worst, side by side, so the best case cannot be mistaken for the
typical one. Black-Scholes is the corrected AMENDMENT_01 per-expiry fit.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, iv as inv_iv, bs_predict
from amend01 import predict_bs_expiry

OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COL = {'BS': '#8a8a8a', 'SH': '#d9822b', 'DH': '#1f3f8a'}
LAB = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston'}

CASES = [('2025-01-15', 'BEST for Double Heston', 'largest advantage of 50 test dates'),
         ('2025-09-02', 'TYPICAL', 'median advantage'),
         ('2024-10-20', 'WORST for Double Heston', 'the one where Single Heston wins by most')]


def errs(date):
    q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{date}.csv')
    fb = HERE / 'artifacts' / 'fits' / 'B' / date
    dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
    sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
    bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / date / 'BS_EXPIRY.json'))['fit']
    x, t, e = q.x.to_numpy(), q.tau.to_numpy(), q.expiry.to_numpy()
    out = {}
    for k, c in [('DH', exact(dh, x, t, False)), ('SH', exact(sh, x, t, False)),
                 ('BS', predict_bs_expiry(bs, e, x, t))]:
        mi = inv_iv(c, x, t)
        d = 100 * (mi - q.market_iv.to_numpy())
        out[k] = d
    return q, out


fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.8), sharey=True)
summary = []
for ax, (date, title, sub) in zip(axs, CASES):
    q, E = errs(date)
    mn = np.exp(q.x.to_numpy())
    for k in ['BS', 'SH', 'DH']:
        d = E[k]; ok = np.isfinite(d)
        ax.scatter(mn[ok], d[ok], s=26, color=COL[k], alpha=.8,
                   label=f'{LAB[k]}   RMSE {np.sqrt(np.mean(d[ok]**2)):.2f}')
        summary.append({'date': date, 'case': title, 'model': k,
                        'IV_RMSE_volpts': float(np.sqrt(np.mean(d[ok] ** 2))), 'n': int(ok.sum())})
    ax.axhline(0, color='k', lw=1.2)
    ax.set_title(f'{title}\n{date}   ·   {sub}   ·   {len(q)} quotes', fontsize=10)
    ax.set_xlabel('$S/K$   (forward / strike)')
    ax.legend(frameon=True, framealpha=.92, loc='lower left', fontsize=8, markerscale=1.2)
axs[0].set_ylabel('model $-$ market implied volatility  (vol points)')
axs[0].set_ylim(-22, 14)
fig.suptitle('BITCOIN: the full range of outcomes across 50 held-out test dates\n'
             'Double Heston beats Single Heston on 42 of 50 dates. Both ends of the range are shown.',
             y=1.0, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .87])
fig.savefig(OUT / 'BTC5_range_of_outcomes.png'); plt.close(fig)
print(' BTC5_range_of_outcomes')
s = pd.DataFrame(summary); s.to_csv(OUT / 'BTC5_range_summary.csv', index=False)
print()
print(s.pivot_table(index=['case', 'date'], columns='model', values='IV_RMSE_volpts').round(3).to_string())
