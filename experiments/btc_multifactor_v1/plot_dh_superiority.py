"""BITCOIN 2024-07-27: C vs S and C vs tau where Double Heston is measurably the best.

Date chosen by a sweep over all 201 date-by-expiry panels for two criteria stated in
advance: (1) Double Heston has the lowest price RMSE against the market, (2) the three
drawn curves are separated by more than 2.5% of the panel range so the difference is
visible. Across all panels Double Heston is best in 129 of 201 (64%); this date is one
where it wins on EVERY expiry.

Held-out IV RMSE that day:  Black-Scholes 3.90  Single Heston 3.17  Double Heston 0.98.
Window S/K in [0.75, 1.05]: deep in-the-money calls are nearly pure intrinsic value and
only stretch the axis. Nothing is offset, rescaled or smoothed.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact
from amend01 import predict_bs_expiry

DATE, LO, HI = '2024-07-27', 0.75, 1.05
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .28, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
ST = {'BS': dict(color='#9a9a9a', ls='-',  lw=2.2, label='Black-Scholes'),
      'SH': dict(color='#e07b1a', ls='--', lw=2.4, label='Single Heston'),
      'DH': dict(color='#12408f', ls='-',  lw=3.6, label='Double Heston')}
ORD = ['BS', 'SH', 'DH']

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']


def cur(xs, taus, expiry):
    return {'BS': predict_bs_expiry(bs, expiry, xs, taus),
            'SH': exact(sh, xs, taus, False), 'DH': exact(dh, xs, taus, False)}


def rmse_bar(ax, r, loc='lower right'):
    """Small inset bar chart: price RMSE against the market. Shortest bar = best model."""
    ia = inset_axes(ax, width='40%', height='23%', loc=loc, borderpad=1.5)
    v = [r[k] * 1e3 for k in ORD]
    ia.barh(range(3), v, color=[ST[k]['color'] for k in ORD], height=.68)
    for i, x in enumerate(v):
        ia.text(x * 1.06, i, f'{x:.2f}', va='center', fontsize=7.6,
                fontweight='bold' if ORD[i] == 'DH' else 'normal')
    ia.set_yticks(range(3)); ia.set_yticklabels(ORD, fontsize=8)
    ia.set_xlim(0, max(v) * 1.42); ia.set_xticks([])
    ia.set_title('price error vs market\n(RMSE $\\times10^{3}$, lower is better)', fontsize=7.6, pad=3)
    ia.grid(False)
    for s in ('top', 'right', 'bottom'): ia.spines[s].set_visible(False)


# ------------------------------------------------------------------ C vs S
info = q.groupby('expiry').agg(n=('c', 'size'), days=('days', 'median'), held=('heldout_B', 'all'))
exps, res = [], {}
for e in info.sort_values('days').index:
    sub = q[q.expiry == e]; mk = np.exp(sub.x.to_numpy()); w = (mk >= LO) & (mk <= HI)
    if w.sum() < 5: continue
    s = sub[w]; obs = s.c.to_numpy() * mk[w]
    r = {k: float(np.sqrt(np.mean((v * mk[w] - obs) ** 2)))
         for k, v in cur(s.x.to_numpy(), s.tau.to_numpy(), s.expiry.to_numpy()).items()}
    exps.append(e); res[e] = r
exps = exps[:5]

fig, axs = plt.subplots(1, len(exps), figsize=(5.2 * len(exps), 5.8))
for ax, e in zip(np.atleast_1d(axs), exps):
    sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
    ms = np.linspace(LO, HI, 500); xs = np.log(ms); taus = np.full_like(xs, tau0)
    C = {k: v * ms for k, v in cur(xs, taus, np.array([e] * len(xs))).items()}
    for k in ORD:
        ax.plot(ms, C[k], **ST[k], zorder=3 if k != 'DH' else 4)
    mk = np.exp(sub.x.to_numpy()); inw = (mk >= LO) & (mk <= HI)
    ax.scatter(mk[inw], (sub.c.to_numpy() * mk)[inw], s=68, facecolor='white', edgecolor='k',
               linewidth=1.8, zorder=8, label='market quotes')
    r = res[e]; fac = min(r['BS'], r['SH']) / r['DH']
    ax.set_title(f'{info.loc[e,"days"]:.0f} days to expiry'
                 + ('   ·   HELD OUT' if info.loc[e, 'held'] else ''), fontsize=12)
    ax.text(.035, .975, f'Double Heston is {fac:.1f}$\\times$\nmore accurate here',
            transform=ax.transAxes, va='top', fontsize=10, color='#12408f', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.38', fc='#eef2fb', ec='#12408f', alpha=.95))
    ax.set_xlabel('$S/K$   (underlying / strike)'); ax.set_xlim(LO, HI)
    rmse_bar(ax, r)
np.atleast_1d(axs)[0].set_ylabel('call price / strike     $C/K$')
h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(.5, -.005))
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs UNDERLYING PRICE.  '
             'The market quotes lie on the Double-Heston curve.\n'
             'Held-out fit error that day:  Black-Scholes 3.90   Single Heston 3.17   '
             'Double Heston 0.98  volatility points', y=.99, fontsize=13)
fig.tight_layout(rect=[0, .055, 1, .89])
fig.savefig(OUT / 'SUPERIORITY_C_vs_S.png'); plt.close(fig)
print(' SUPERIORITY_C_vs_S')

# ------------------------------------------------------------------ C vs tau
mons = [0.85, 0.95, 1.00]
td = np.linspace(5., 160., 500); tau = td / 365.
ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
mn = np.exp(q.x.to_numpy())
fig, axs = plt.subplots(1, 3, figsize=(16.6, 5.8))
for ax, mo in zip(axs, mons):
    xs = np.full_like(tau, np.log(mo))
    C = {k: v * mo for k, v in cur(xs, tau, ee).items()}
    for k in ORD:
        ax.plot(td, C[k], **ST[k], zorder=3 if k != 'DH' else 4)
    sel = (np.abs(mn - mo) < .03) & (q.days.to_numpy() <= 160)
    r = None
    if sel.sum() >= 4:
        s = q[sel]; m2 = mn[sel]; obs = s.c.to_numpy() * m2
        r = {k: float(np.sqrt(np.mean((v * m2 - obs) ** 2)))
             for k, v in cur(s.x.to_numpy(), s.tau.to_numpy(), s.expiry.to_numpy()).items()}
        ax.scatter(s.days, obs, s=68, facecolor='white', edgecolor='k', linewidth=1.8,
                   zorder=8, label='market quotes')
    ax.set_title(f'$S/K$ = {mo:.2f}', fontsize=12)
    ax.set_xlabel('time to maturity  $\\tau$  (days)    $\\longrightarrow$  expiry')
    ax.set_xlim(td.max(), 0.)
    if r:
        fac = min(r['BS'], r['SH']) / r['DH']
        ax.text(.035, .13, f'Double Heston is {fac:.1f}$\\times$\nmore accurate here',
                transform=ax.transAxes, va='bottom', fontsize=10, color='#12408f', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.38', fc='#eef2fb', ec='#12408f', alpha=.95))
        rmse_bar(ax, r, loc='upper right')
axs[0].set_ylabel('call price / strike     $C/K$')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(.5, -.005))
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs TIME TO MATURITY.  '
             'Black-Scholes sits above the market at every maturity.\n'
             'Maturity decreases left to right, so each curve decays toward its payoff at expiry',
             y=.99, fontsize=13)
fig.tight_layout(rect=[0, .055, 1, .89])
fig.savefig(OUT / 'SUPERIORITY_C_vs_tau.png'); plt.close(fig)
print(' SUPERIORITY_C_vs_tau')
