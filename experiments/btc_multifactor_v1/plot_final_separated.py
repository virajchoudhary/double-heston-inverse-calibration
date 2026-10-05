"""C vs S and C vs tau for Bitcoin, with the three models VISIBLY SEPARATED.

The separation was always there; it was hidden by the axis. Deep in-the-money calls are
almost pure intrinsic value (C ~ S-K), so including them stretches the y-axis by about 4x
and compresses every model difference into a line width. Restricting the plot to the
moneyness region where options carry real time value, S/K in [0.75, 1.05], leaves the
curves untouched and lets the real gap show.

Nothing is offset, rescaled or smoothed. Same calibrated models, same market quotes.
Black-Scholes is the corrected AMENDMENT_01 per-expiry fit.
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

DATE = '2024-01-09'
LO, HI = 0.75, 1.05
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11.5, 'axes.labelsize': 10.5, 'legend.fontsize': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .28, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
ST = {'BS': dict(color='#8a8a8a', ls='-',  lw=2.6, label='Black-Scholes'),
      'SH': dict(color='#e07b1a', ls='--', lw=2.6, label='Single Heston'),
      'DH': dict(color='#1f3f8a', ls='-',  lw=3.2, label='Double Heston')}

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']


def curves(xs, taus, expiry):
    return {'BS': predict_bs_expiry(bs, expiry, xs, taus),
            'SH': exact(sh, xs, taus, False),
            'DH': exact(dh, xs, taus, False)}


# ----------------------------------------------------------------- C vs S
info = q.groupby('expiry').agg(n=('c', 'size'), days=('days', 'median'), held=('heldout_B', 'all'))
exps = [e for e in info.sort_values('days').index
        if info.loc[e, 'n'] >= 9 and np.exp(q[q.expiry == e].x).min() < LO + .04][:4]
fig, axs = plt.subplots(1, len(exps), figsize=(5.4 * len(exps), 5.6))
for ax, e in zip(np.atleast_1d(axs), exps):
    sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
    ms = np.linspace(LO, HI, 500); xs = np.log(ms); taus = np.full_like(xs, tau0)
    C = {k: v * ms for k, v in curves(xs, taus, np.array([e] * len(xs))).items()}
    for k in ['BS', 'SH', 'DH']:
        ax.plot(ms, C[k], **ST[k])
    mk = np.exp(sub.x.to_numpy()); inw = (mk >= LO) & (mk <= HI)
    ax.scatter(mk[inw], (sub.c.to_numpy() * mk)[inw], s=52, facecolor='white',
               edgecolor='k', linewidth=1.5, zorder=7, label='market quotes')
    gb = np.abs(C['BS'] - C['DH']).max(); gs = np.abs(C['SH'] - C['DH']).max()
    yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
    ax.set_title(f'{info.loc[e,"days"]:.0f} days to expiry'
                 + ('   ·   HELD OUT' if info.loc[e, 'held'] else ''), fontsize=11.5)
    ax.text(.03, .97, f'gap to Double Heston\nBS {100*gb/yr:.1f}%   SH {100*gs/yr:.1f}%  of range',
            transform=ax.transAxes, va='top', fontsize=9, color='#8a2020', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#ccc', alpha=.93))
    ax.set_xlabel('$S/K$   (underlying / strike)')
    ax.set_xlim(LO, HI)
np.atleast_1d(axs)[0].set_ylabel('call price / strike     $C/K$')
np.atleast_1d(axs)[0].legend(frameon=True, framealpha=.95, loc='lower right', fontsize=9.5)
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs UNDERLYING PRICE,  three calibrated models\n'
             f'plotted over $S/K \\in [{LO}, {HI}]$, the region where options carry time value',
             y=.99, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(OUT / 'FINAL_C_vs_S.png'); plt.close(fig)
print(' FINAL_C_vs_S')

# ----------------------------------------------------------------- C vs tau
mons = [0.80, 0.90, 1.00]
td = np.linspace(5., 175., 500); tau = td / 365.
ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
fig, axs = plt.subplots(1, 3, figsize=(16.2, 5.6))
mn = np.exp(q.x.to_numpy())
for ax, mo in zip(axs, mons):
    xs = np.full_like(tau, np.log(mo))
    C = {k: v * mo for k, v in curves(xs, tau, ee).items()}
    for k in ['BS', 'SH', 'DH']:
        ax.plot(td, C[k], **ST[k])
    sel = (np.abs(mn - mo) < .025) & (q.days.to_numpy() <= 175)
    if sel.sum():
        ax.scatter(q.days[sel], q.c[sel] * mn[sel], s=52, facecolor='white', edgecolor='k',
                   linewidth=1.5, zorder=7, label='market quotes')
    gb = np.abs(C['BS'] - C['DH']).max(); gs = np.abs(C['SH'] - C['DH']).max()
    yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
    ax.text(.03, .97, f'gap to Double Heston\nBS {100*gb/yr:.1f}%   SH {100*gs/yr:.1f}%  of range',
            transform=ax.transAxes, va='top', fontsize=9, color='#8a2020', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#ccc', alpha=.93))
    ax.set_title(f'$S/K$ = {mo:.2f}', fontsize=11.5)
    ax.set_xlabel('time to maturity  $\\tau$  (days)    $\\longrightarrow$  expiry')
    ax.set_xlim(td.max(), 0.)
axs[0].set_ylabel('call price / strike     $C/K$')
axs[0].legend(frameon=True, framealpha=.95, loc='lower left', fontsize=9.5)
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs TIME TO MATURITY,  three calibrated models\n'
             'maturity decreases left to right, so each curve decays toward its payoff at expiry',
             y=.99, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(OUT / 'FINAL_C_vs_tau.png'); plt.close(fig)
print(' FINAL_C_vs_tau')
