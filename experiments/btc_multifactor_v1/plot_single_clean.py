"""The 20-day panel, clean: curves and text only. No shading, no bars.
Two variants: colour-coded curves, and a fully monochrome version.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact
from amend01 import predict_bs_expiry

DATE, EXPIRY, LO, HI = '2024-07-27', '2024-08-16', 0.71, 0.88
OUT = HERE / 'figures_dh_advantage'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .3, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']
sub = q[q.expiry == EXPIRY]
days = float(sub.days.median()); tau0 = days / 365.


def cur(xs, taus, expiry):
    return {'BS': predict_bs_expiry(bs, expiry, xs, taus),
            'SH': exact(sh, xs, taus, False), 'DH': exact(dh, xs, taus, False)}


ms = np.linspace(LO, HI, 600); xs = np.log(ms); taus = np.full_like(xs, tau0)
C = {k: v * ms for k, v in cur(xs, taus, np.array([EXPIRY] * len(xs))).items()}
mk = np.exp(sub.x.to_numpy()); inw = (mk >= LO) & (mk <= HI)
s2 = sub[inw]; obs = s2.c.to_numpy() * mk[inw]
r = {k: float(np.sqrt(np.mean((v * mk[inw] - obs) ** 2)))
     for k, v in cur(s2.x.to_numpy(), s2.tau.to_numpy(), s2.expiry.to_numpy()).items()}
fac = min(r['BS'], r['SH']) / r['DH']
print('price RMSE x1e3  BS %.2f  SH %.2f  DH %.2f   -> DH %.1fx better'
      % (r['BS'] * 1e3, r['SH'] * 1e3, r['DH'] * 1e3, fac))

STYLES = {
    'colour': {'BS': dict(color='#9a9a9a', ls='-', lw=3.0), 'SH': dict(color='#e07b1a', ls='--', lw=3.0),
               'DH': dict(color='#12408f', ls='-', lw=4.2)},
    'mono':   {'BS': dict(color='#666666', ls=':',  lw=2.8), 'SH': dict(color='#222222', ls='--', lw=2.6),
               'DH': dict(color='black',   ls='-',  lw=4.0)},
}
NAME = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston'}

for tag, ST in STYLES.items():
    accent = '#12408f' if tag == 'colour' else 'black'
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    for k in ['BS', 'SH', 'DH']:
        ax.plot(ms, C[k], **ST[k], label=NAME[k], solid_capstyle='round',
                zorder=4 if k != 'DH' else 5)
    ax.scatter(mk[inw], obs, s=165, facecolor='white', edgecolor='black', linewidth=2.6,
               zorder=9, label='market quotes')
    ax.set_title(f'{days:.0f} days to expiry', fontsize=16)
    ax.set_xlabel('$S/K$   (underlying / strike)')
    ax.set_ylabel('call price / strike     $C/K$')
    ax.set_xlim(LO, HI)
    bar = "\u2500" * 30
    head = "\n".join([
        f"Double Heston is {fac:.1f}x more accurate",
        bar,
        "price error vs market  (RMSE x 1000)",
        f"   Black-Scholes     {r['BS']*1e3:5.2f}",
        f"   Single Heston      {r['SH']*1e3:5.2f}",
        f"   Double Heston    {r['DH']*1e3:5.2f}",
    ])
    ax.text(.035, .975, head, transform=ax.transAxes, va='top', ha='left', fontsize=13,
            color=accent, linespacing=1.55, family='DejaVu Sans Mono',
            bbox=dict(boxstyle='round,pad=0.6', fc='white', ec=accent, lw=2, alpha=.97))
    ax.legend(frameon=False, loc='lower right', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f'PANEL20d_{tag}.png'); plt.close(fig)
    print(' PANEL20d_' + tag)
