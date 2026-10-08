"""Generate the five PNG files the capstone-4 manuscript expects, with exactly those names.

  fig_error_ecdf_panel.png   ECDF of quote-level |IV error|, 50 dates / 2,223 held-out quotes
  fig_error_date_panel.png   ranked date-level mean |error| advantage with uncertainty bars
  fig_range_best.png         15 Jan 2025, 97 quotes   (largest DH advantage)
  fig_range_typical.png       2 Sep 2025, 133 quotes  (median advantage)
  fig_range_worst.png        20 Oct 2024, 60 quotes   (largest SH advantage)

All values come from the frozen artifacts. Black-Scholes is the corrected
AMENDMENT_01 per-expiry fit.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, iv as inv_iv
from amend01 import predict_bs_expiry

OUT = HERE / 'capstone4_figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 11.5,
                     'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
COL = {'BS': '#8a8a8a', 'SH': '#e07b1a', 'DH': '#12408f'}
NAME = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston'}


def finish(fig, name):
    p = OUT / name
    fig.savefig(p); plt.close(fig)
    im = Image.open(p).convert('RGB')          # flatten alpha for pdfLaTeX
    if max(im.size) > 2000:
        r = 2000 / max(im.size); im = im.resize((round(im.size[0]*r), round(im.size[1]*r)), Image.LANCZOS)
    im.save(p, 'PNG', optimize=True)
    print(f'  {name:30s} {im.size[0]}x{im.size[1]}')


# ---------------------------------------------------------------- panels (a) and (b)
d = pd.read_csv(HERE / 'figures_dh_advantage' / 'pooled_heldout_errors.csv')
w = d.pivot_table(index=['date', 'qid'], columns='model', values='err').reset_index()
for m in ['BS', 'SH', 'DH']:
    w['a' + m] = w[m].abs()

fig, a = plt.subplots(figsize=(8.4, 5.6))
for m in ['BS', 'SH', 'DH']:
    y = np.sort(d[d.model == m].abs_err.to_numpy())
    a.step(y, np.arange(1, len(y)+1)/len(y), where='post', color=COL[m], lw=2.6,
           label=f'{NAME[m]}  (median {np.median(y):.2f})')
a.axhline(.5, color='#999', ls=':', lw=1.1)
a.set_xscale('log'); a.set_xlim(.02, 40); a.set_ylim(0, 1.02)
a.set_xlabel('absolute implied-volatility error (vol points, log scale)')
a.set_ylabel('proportion of quotes at or below')
a.set_title('Quote-level error distribution, 50 dates / 2,223 held-out quotes')
a.legend(frameon=False, loc='lower right')
fig.tight_layout(); finish(fig, 'fig_error_ecdf_panel.png')

g = w.groupby('date').apply(lambda t: pd.Series({
    'shd': (t.aSH - t.aDH).mean(),
    'se': (t.aSH - t.aDH).std(ddof=1)/np.sqrt(len(t))}), include_groups=False).reset_index()
g = g.sort_values('shd').reset_index(drop=True)
fig, b = plt.subplots(figsize=(8.4, 5.6))
b.errorbar(g.shd, g.index, xerr=1.96*g.se, fmt='none', ecolor='#c8c8c8', elinewidth=1.1, zorder=1)
b.scatter(g.shd, g.index, s=30, c=np.where(g.shd > 0, COL['DH'], COL['SH']), zorder=3)
b.axvline(0, color='k', lw=1.4)
rng = np.random.default_rng(20260920)
bs = [np.mean(rng.choice(g.shd.to_numpy(), len(g), replace=True)) for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); mu = g.shd.mean()
b.axvspan(lo, hi, color=COL['DH'], alpha=.13); b.axvline(mu, color=COL['DH'], lw=2.2)
b.text(mu + .15, 3, f'mean {mu:+.2f}\n95% CI [{lo:.2f}, {hi:.2f}]', fontsize=10.5,
       color=COL['DH'], fontweight='bold')
b.set_xlabel('per-date mean |error| advantage of Double Heston over Single Heston')
b.set_ylabel('test date, ranked')
b.set_title(f'Date-level advantage across {len(g)} test dates')
fig.tight_layout(); finish(fig, 'fig_error_date_panel.png')
print(f'   (DH ahead on {(g.shd>0).sum()}/{len(g)} dates; mean {mu:+.3f} [{lo:.3f}, {hi:.3f}])')


# ---------------------------------------------------------------- the three single-date panels
def errs(date):
    q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{date}.csv')
    fb = HERE / 'artifacts' / 'fits' / 'B' / date
    dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
    sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
    bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / date / 'BS_EXPIRY.json'))['fit']
    x, t, e = q.x.to_numpy(), q.tau.to_numpy(), q.expiry.to_numpy()
    E = {}
    for k, c in [('DH', exact(dh, x, t, False)), ('SH', exact(sh, x, t, False)),
                 ('BS', predict_bs_expiry(bs, e, x, t))]:
        E[k] = 100 * (inv_iv(c, x, t) - q.market_iv.to_numpy())
    return q, E

CASES = [('2025-01-15', 'fig_range_best.png',    'Largest Double-Heston advantage'),
         ('2025-09-02', 'fig_range_typical.png', 'Median advantage'),
         ('2024-10-20', 'fig_range_worst.png',   'Largest Single-Heston advantage')]
for date, fname, title in CASES:
    q, E = errs(date)
    mn = np.exp(q.x.to_numpy())
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    n = 0
    for k in ['BS', 'SH', 'DH']:
        v = E[k]; ok = np.isfinite(v); n = max(n, int(ok.sum()))
        ax.scatter(mn[ok], v[ok], s=42, color=COL[k], alpha=.82,
                   label=f'{NAME[k]}   RMSE {np.sqrt(np.mean(v[ok]**2)):.2f}')
    ax.axhline(0, color='k', lw=1.4)
    ax.set_xlabel('$S/K$   (forward / strike)')
    ax.set_ylabel('model $-$ market implied volatility  (vol points)')
    ax.set_title(f'{title}\n{date}   ·   {n} quotes')
    ax.legend(frameon=True, framealpha=.94, loc='lower left')
    fig.tight_layout(); finish(fig, fname)
