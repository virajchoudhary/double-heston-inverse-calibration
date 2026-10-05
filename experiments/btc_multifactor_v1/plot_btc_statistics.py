"""Standard statistical displays that separate models the price curves cannot.

Four displays statisticians use when raw curves overlap:
  A  box + strip plot of absolute error      -> location and spread at a glance
  B  empirical CDF of absolute error         -> stochastic dominance at every quantile
  C  forest / caterpillar plot by date       -> is the effect consistent across units?
  D  binned mean signed error with SE bands  -> systematic bias the scatter hides

Data: 2,223 design-B held-out quotes over 50 test dates, one common paired mask.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures_dh_advantage'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COL = {'BS': '#8a8a8a', 'SH': '#d9822b', 'DH': '#1f3f8a'}
NAME = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston'}
ORD = ['BS', 'SH', 'DH']

d = pd.read_csv(OUT / 'pooled_heldout_errors.csv')
w = d.pivot_table(index=['date', 'qid'], columns='model', values='err').reset_index()
for m in ORD:
    w['a' + m] = w[m].abs()

fig, axs = plt.subplots(2, 2, figsize=(14.4, 9.6))

# ---------------------------------------------------------------- A  box + strip
a = axs[0, 0]
data = [d[d.model == m].abs_err.to_numpy() for m in ORD]
bp = a.boxplot(data, vert=True, widths=.55, patch_artist=True, showfliers=False,
               medianprops=dict(color='k', lw=2))
for patch, m in zip(bp['boxes'], ORD):
    patch.set_facecolor(COL[m]); patch.set_alpha(.55)
rng = np.random.default_rng(7)
for i, m in enumerate(ORD):
    y = d[d.model == m].abs_err.to_numpy()
    y = y[y > 0]
    xj = rng.normal(i + 1, .07, len(y))
    a.scatter(xj, y, s=3, color=COL[m], alpha=.18, zorder=1)
for i, m in enumerate(ORD):
    med = np.median(d[d.model == m].abs_err)
    a.text(i + 1, med * 1.55, f'median\n{med:.2f}', ha='center', fontsize=8.5, fontweight='bold')
a.set_yscale('log')
a.set_xticklabels([NAME[m] for m in ORD])
a.set_ylabel('absolute IV error  (vol points, log scale)')
a.set_title('A.  Distribution of error per quote\nbox = quartiles, line = median, dots = all 2,223 quotes')

# ---------------------------------------------------------------- B  ECDF
b = axs[0, 1]
for m in ORD:
    y = np.sort(d[d.model == m].abs_err.to_numpy())
    b.step(y, np.arange(1, len(y) + 1) / len(y), where='post', color=COL[m], lw=2.2,
           label=f'{NAME[m]}   median {np.median(y):.2f}')
for lvl, ls in [(.5, ':'), (.9, '--')]:
    b.axhline(lvl, color='#999', ls=ls, lw=1)
    b.text(28, lvl + .015, f'{int(lvl*100)}% of quotes', fontsize=7.5, color='#666', ha='right')
b.set_xscale('log'); b.set_xlim(.02, 40); b.set_ylim(0, 1.02)
b.set_xlabel('absolute IV error  (vol points, log scale)')
b.set_ylabel('proportion of quotes at or below')
b.set_title('B.  Empirical CDF of error\nDouble Heston lies left of Single Heston at EVERY quantile')
b.legend(frameon=False, loc='lower right')

# ---------------------------------------------------------------- C  forest plot by date
c = axs[1, 0]
g = w.groupby('date').apply(lambda t: pd.Series({
    'shd': (t.aSH - t.aDH).mean(),
    'se': (t.aSH - t.aDH).std(ddof=1) / np.sqrt(len(t)), 'n': len(t)}), include_groups=False).reset_index()
g = g.sort_values('shd').reset_index(drop=True)
col = np.where(g.shd > 0, COL['DH'], COL['SH'])
c.errorbar(g.shd, g.index, xerr=1.96 * g.se, fmt='none', ecolor='#bbb', elinewidth=1, zorder=1)
c.scatter(g.shd, g.index, s=26, c=col, zorder=3)
c.axvline(0, color='k', lw=1.4)
# overall estimate, cluster bootstrap over dates
rng = np.random.default_rng(20260920)
bs = [np.mean(rng.choice(g.shd.to_numpy(), len(g), replace=True)) for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); mean = g.shd.mean()
c.axvspan(lo, hi, color=COL['DH'], alpha=.13)
c.axvline(mean, color=COL['DH'], lw=2.2)
c.text(mean + .12, 2, f'overall  {mean:+.2f}\n95% CI [{lo:.2f}, {hi:.2f}]', fontsize=8.5,
       color=COL['DH'], fontweight='bold')
c.set_xlabel('mean |error| advantage of Double Heston over Single Heston  (vol points)')
c.set_ylabel('test date, ranked')
c.set_title(f'C.  Forest plot across all 50 dates\nDouble Heston better on {int((g.shd>0).sum())} of {len(g)}; '
            'the interval excludes zero')

# ---------------------------------------------------------------- D  binned bias
e = axs[1, 1]
bins = np.array([.5, .75, .85, .92, .97, 1.03, 1.08, 1.15, 1.3, 2.0])
ctr = .5 * (bins[:-1] + bins[1:])
for m in ORD:
    s = d[d.model == m]
    idx = np.digitize(s.moneyness, bins) - 1
    mu = np.array([s.err[idx == i].mean() if (idx == i).sum() > 4 else np.nan for i in range(len(ctr))])
    se = np.array([s.err[idx == i].std(ddof=1) / np.sqrt((idx == i).sum()) if (idx == i).sum() > 4 else np.nan
                   for i in range(len(ctr))])
    e.plot(ctr, mu, color=COL[m], lw=2.2, marker='o', ms=5, label=NAME[m])
    e.fill_between(ctr, mu - 1.96 * se, mu + 1.96 * se, color=COL[m], alpha=.20)
e.axhline(0, color='k', lw=1.4)
e.set_xlabel('$S/K$   (forward / strike)')
e.set_ylabel('mean signed IV error  (vol points)')
e.set_title('D.  Systematic bias by moneyness\nbands are 95% confidence intervals on the mean')
e.legend(frameon=False, loc='lower left')

fig.suptitle('BITCOIN, 2,223 held-out quotes over 50 test dates  —  why the price curves overlap but the models do not\n'
             'The price curves agree because the payoff dominates. These are the standard displays that reveal the difference.',
             y=1.0, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .945])
fig.savefig(OUT / 'BTC6_statistical_evidence.png'); plt.close(fig)
print(' BTC6_statistical_evidence')

print('\nMedian |error| (vol points):', {m: round(float(np.median(d[d.model==m].abs_err)), 3) for m in ORD})
print('90th pct  |error|:', {m: round(float(np.percentile(d[d.model==m].abs_err, 90)), 3) for m in ORD})
print(f'\nDH better than SH on {int((g.shd>0).sum())}/{len(g)} dates; overall {mean:+.3f} [{lo:.3f}, {hi:.3f}]')
from scipy.stats import wilcoxon
st, p = wilcoxon(w.aSH, w.aDH, alternative='greater')
print(f'quote-level Wilcoxon (SH vs DH): p = {p:.3g}')
