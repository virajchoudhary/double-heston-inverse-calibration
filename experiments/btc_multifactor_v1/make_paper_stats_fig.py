"""Compact two-panel statistical figure for the paper: ECDF + forest plot.
Both panels use the 2,223 design-B held-out quotes over 50 BTC test dates."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures_dh_advantage'
plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COL = {'BS': '#8a8a8a', 'SH': '#e07b1a', 'DH': '#12408f'}
NAME = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston'}

d = pd.read_csv(OUT / 'pooled_heldout_errors.csv')
w = d.pivot_table(index=['date', 'qid'], columns='model', values='err').reset_index()
for m in ['BS', 'SH', 'DH']:
    w['a' + m] = w[m].abs()

fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.4))

# ---- ECDF
for m in ['BS', 'SH', 'DH']:
    y = np.sort(d[d.model == m].abs_err.to_numpy())
    a.step(y, np.arange(1, len(y) + 1) / len(y), where='post', color=COL[m], lw=2.2,
           label=f'{NAME[m]}  (median {np.median(y):.2f})')
a.axhline(.5, color='#999', ls=':', lw=1)
a.set_xscale('log'); a.set_xlim(.02, 40); a.set_ylim(0, 1.02)
a.set_xlabel('absolute implied-volatility error (vol points, log scale)')
a.set_ylabel('proportion of quotes at or below')
a.set_title('(a) Error distribution over 2,223 held-out quotes')
a.legend(frameon=False, loc='lower right')

# ---- forest
g = w.groupby('date').apply(lambda t: pd.Series({
    'shd': (t.aSH - t.aDH).mean(),
    'se': (t.aSH - t.aDH).std(ddof=1) / np.sqrt(len(t))}), include_groups=False).reset_index()
g = g.sort_values('shd').reset_index(drop=True)
b.errorbar(g.shd, g.index, xerr=1.96 * g.se, fmt='none', ecolor='#c8c8c8', elinewidth=1, zorder=1)
b.scatter(g.shd, g.index, s=22, c=np.where(g.shd > 0, COL['DH'], COL['SH']), zorder=3)
b.axvline(0, color='k', lw=1.3)
rng = np.random.default_rng(20260920)
bs = [np.mean(rng.choice(g.shd.to_numpy(), len(g), replace=True)) for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); mu = g.shd.mean()
b.axvspan(lo, hi, color=COL['DH'], alpha=.13)
b.axvline(mu, color=COL['DH'], lw=2)
b.text(mu + .14, 3, f'mean {mu:+.2f}\n95% CI [{lo:.2f}, {hi:.2f}]', fontsize=9,
       color=COL['DH'], fontweight='bold')
b.set_xlabel('per-date mean |error| advantage of Double Heston over Single Heston')
b.set_ylabel('test date, ranked')
b.set_title(f'(b) Consistency across {len(g)} test dates')

fig.tight_layout()
fig.savefig(OUT / 'fig3_statistical_evidence.png'); plt.close(fig)
print(' fig3_statistical_evidence written')
print(f'   DH better on {(g.shd>0).sum()}/{len(g)} dates; overall {mu:+.3f} [{lo:.3f}, {hi:.3f}]')
print('   medians:', {m: round(float(np.median(d[d.model==m].abs_err)),3) for m in ['BS','SH','DH']})
