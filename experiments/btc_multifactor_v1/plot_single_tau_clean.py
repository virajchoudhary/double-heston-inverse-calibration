"""The S/K = 1.05 C-vs-tau panel, clean: curves and text only. No shading, no bars.
Colour and monochrome variants, matching the C-vs-S panel treatment.
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

DATE, MO, BAND = '2024-07-27', 1.05, 0.012
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
mn = np.exp(q.x.to_numpy()); dd = q.days.to_numpy()


def cur(xs, taus, expiry):
    return {'BS': predict_bs_expiry(bs, expiry, xs, taus),
            'SH': exact(sh, xs, taus, False), 'DH': exact(dh, xs, taus, False)}


def knots_for(tau):
    return np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])


# same window rule as the multi-panel figure: >=4 quotes, span >= 60 days, reaching to expiry
sel_all = np.abs(mn - MO) < BAND
cand = []
for a in np.arange(3., 22., 1.):
    for b in np.arange(70., 175., 5.):
        if b - a < 60: continue
        n = int((sel_all & (dd >= a) & (dd <= b)).sum())
        if n < 4: continue
        td = np.linspace(a, b, 200); tau = td / 365.
        xs = np.full_like(tau, np.log(MO))
        C = {k: v * MO for k, v in cur(xs, tau, knots_for(tau)).items()}
        yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
        if yr <= 0: continue
        g = min(np.abs(C['BS'] - C['DH']).max(), np.abs(C['SH'] - C['DH']).max())
        cand.append((g / yr, a, b, n))
cand.sort(reverse=True)
A, B, N = cand[0][1:]
print(f'window tau in [{A:.0f}, {B:.0f}] days, {N} market quotes in band')

td = np.linspace(A, B, 600); tau = td / 365.
xs = np.full_like(tau, np.log(MO))
C = {k: v * MO for k, v in cur(xs, tau, knots_for(tau)).items()}
sel = sel_all & (dd >= A) & (dd <= B)
s2 = q[sel]; m2 = mn[sel]; obs = s2.c.to_numpy() * m2
r = {k: float(np.sqrt(np.mean((v * m2 - obs) ** 2)))
     for k, v in cur(s2.x.to_numpy(), s2.tau.to_numpy(), s2.expiry.to_numpy()).items()}
fac = min(r['BS'], r['SH']) / r['DH']
print('price RMSE x1e3  BS %.2f  SH %.2f  DH %.2f  -> DH %.1fx better'
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
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    for k in ['BS', 'SH', 'DH']:
        ax.plot(td, C[k], **ST[k], label=NAME[k], solid_capstyle='round',
                zorder=4 if k != 'DH' else 5)
    ax.scatter(s2.days, obs, s=165, facecolor='white', edgecolor='black', linewidth=2.6,
               zorder=9, label='market quotes')
    ax.set_title(f'$S/K$ = {MO:.2f}     (quotes within $\\pm${BAND:.3f})', fontsize=15)
    ax.set_xlabel('time to maturity  $\\tau$  (days)    $\\longrightarrow$  expiry')
    ax.set_ylabel('call price / strike     $C/K$')
    ax.set_xlim(B, A)
    lo = min(min(c.min() for c in C.values()), float(obs.min()))
    hi = max(max(c.max() for c in C.values()), float(obs.max()))
    ax.set_ylim(lo - .08 * (hi - lo), hi + .12 * (hi - lo))
    bar = "─" * 30
    head = "\n".join([
        f"Double Heston is {fac:.1f}x more accurate",
        bar,
        "price error vs market  (RMSE x 1000)",
        f"   Black-Scholes     {r['BS']*1e3:5.2f}",
        f"   Single Heston      {r['SH']*1e3:5.2f}",
        f"   Double Heston    {r['DH']*1e3:5.2f}",
    ])
    ax.text(.035, .30, head, transform=ax.transAxes, va='top', ha='left', fontsize=13,
            color=accent, linespacing=1.55, family='DejaVu Sans Mono',
            bbox=dict(boxstyle='round,pad=0.6', fc='white', ec=accent, lw=2, alpha=.97))
    ax.legend(frameon=False, loc='upper right', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f'PANELTAU105_{tag}.png'); plt.close(fig)
    print(' PANELTAU105_' + tag)
