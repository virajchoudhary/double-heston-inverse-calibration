"""The two displays that DO separate the models against S and against tau.

A price curve cannot separate them: the payoff dominates, so the models agree to ~1%
of the plotted range. Both displays below are still functions of S and of tau, but with
the payoff divided out.

  1  IMPLIED VOLATILITY vs S/K      -- the smile. The market standard for model comparison.
  2  TIME VALUE (C - intrinsic) vs S -- still a price in currency units, payoff removed.
  3  ATM IMPLIED VOLATILITY vs tau  -- the term structure.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, iv as inv_iv
from amend01 import predict_bs_expiry

DATE = '2025-01-15'
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10.5, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
ST = {'BS': dict(color='#8a8a8a', ls='-', lw=2.0, label='Black-Scholes'),
      'SH': dict(color='#d9822b', ls='-', lw=2.0, label='Single Heston'),
      'DH': dict(color='#1f3f8a', ls='-', lw=2.8, label='Double Heston')}

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']


def c_of(model, x, tau, expiry=None):
    x = np.atleast_1d(np.asarray(x, float)); tau = np.atleast_1d(np.asarray(tau, float))
    if model == 'DH': return exact(dh, x, tau, False)
    if model == 'SH': return exact(sh, x, tau, False)
    return predict_bs_expiry(bs, np.asarray(expiry), x, tau)


top = q.groupby('expiry').agg(n=('c', 'size'), days=('days', 'median'))
exps = list(top[top.n >= 8].sort_values('days').index)[:4]

# ---------------------------------------------------------------- 1. IV smile vs S/K
fig, axs = plt.subplots(2, len(exps), figsize=(4.7 * len(exps), 8.4),
                        gridspec_kw={'height_ratios': [2.0, 1.1]})
for j, e in enumerate(exps):
    sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
    mk = np.exp(sub.x.to_numpy())
    a, b = axs[0, j], axs[1, j]
    a.scatter(mk, 100 * sub.market_iv, s=34, c='k', zorder=6, label='market', alpha=.85)
    xs = np.linspace(sub.x.min() - .01, sub.x.max() + .01, 500)
    taus = np.full_like(xs, tau0); ms = np.exp(xs)
    for k in ['BS', 'SH', 'DH']:
        c = c_of(k, xs, taus, np.array([e] * len(xs)))
        a.plot(ms, 100 * inv_iv(c, xs, taus), **ST[k])
        cm = c_of(k, sub.x.to_numpy(), sub.tau.to_numpy(), sub.expiry.to_numpy())
        d = 100 * (inv_iv(cm, sub.x.to_numpy(), sub.tau.to_numpy()) - sub.market_iv.to_numpy())
        ok = np.isfinite(d)
        b.scatter(mk[ok], d[ok], s=26, color=ST[k]['color'], alpha=.85)
    a.set_title(f'{e}   ·   {top.loc[e,"days"]:.0f} days', fontsize=10.5)
    b.axhline(0, color='k', lw=1.2); b.set_xlabel('$S/K$   (forward / strike)')
axs[0, 0].set_ylabel('implied volatility  (%)')
axs[1, 0].set_ylabel('model $-$ market  (vol points)')
axs[0, 0].legend(frameon=False, loc='upper right')
axs[0, -1].legend(frameon=True, framealpha=.9, loc='upper right', fontsize=8)
fig.suptitle(f'BITCOIN {DATE}  —  implied volatility vs underlying:  THE SAME OPTION PRICES, payoff divided out\n'
             'Black-Scholes is flat by construction and cannot bend. The market smiles. Double Heston follows it.',
             y=.99, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(OUT / 'BTC9_IV_smile_vs_S.png'); plt.close(fig)
print(' BTC9_IV_smile_vs_S')

# ---------------------------------------------------------------- 2. time value vs S
fig, axs = plt.subplots(1, len(exps), figsize=(4.7 * len(exps), 4.8))
for ax, e in zip(np.atleast_1d(axs), exps):
    sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
    mk = np.exp(sub.x.to_numpy())
    ax.scatter(mk, sub.c.to_numpy() * mk - np.maximum(mk - 1, 0), s=32, c='k', zorder=6,
               label='market', alpha=.85)
    xs = np.linspace(sub.x.min() - .01, sub.x.max() + .01, 500)
    taus = np.full_like(xs, tau0); ms = np.exp(xs)
    for k in ['BS', 'SH', 'DH']:
        c = c_of(k, xs, taus, np.array([e] * len(xs)))
        ax.plot(ms, c * ms - np.maximum(ms - 1, 0), **ST[k])
    ax.set_title(f'{e}   ·   {top.loc[e,"days"]:.0f} days', fontsize=10.5)
    ax.set_xlabel('$S/K$   (forward / strike)')
np.atleast_1d(axs)[0].set_ylabel('time value / strike   $(C-\\max(S-K,0))/K$')
np.atleast_1d(axs)[-1].legend(frameon=True, framealpha=.9, loc='upper right', fontsize=8)
fig.suptitle(f'BITCOIN {DATE}  —  TIME VALUE vs underlying: still a price in currency units, with the payoff subtracted\n'
             'Subtracting $\\max(S-K,0)$ removes the term that made the three curves coincide.',
             y=.99, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, .88])
fig.savefig(OUT / 'BTC10_time_value_vs_S.png'); plt.close(fig)
print(' BTC10_time_value_vs_S')

# ---------------------------------------------------------------- 3. ATM IV vs tau
td = np.linspace(4., 90., 400); tau = td / 365.
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
xs = np.zeros_like(tau)
for k in ['BS', 'SH', 'DH']:
    c = c_of(k, xs, tau, ee)
    ax.plot(td, 100 * inv_iv(c, xs, tau), **ST[k])
atm = q[np.abs(np.exp(q.x) - 1) < .03]
ax.scatter(atm.days, 100 * atm.market_iv, s=40, c='k', zorder=6, label='market, $|S/K-1|<0.03$')
ax.set_xlim(td.max(), 0.)
ax.set_xlabel('time to maturity  $\\tau$  (days)   $\\longrightarrow$  expiry')
ax.set_ylabel('at-the-money implied volatility  (%)')
ax.set_title(f'BITCOIN {DATE}  —  at-the-money implied volatility vs time to maturity\n'
             'Black-Scholes is a step function between expiry knots; only a two-factor model bends smoothly',
             fontsize=11)
ax.legend(frameon=False, loc='best')
fig.tight_layout(); fig.savefig(OUT / 'BTC11_ATM_IV_vs_tau.png'); plt.close(fig)
print(' BTC11_ATM_IV_vs_tau')
