"""C vs S and C vs tau with MAGNIFIED INSETS.

The three model curves genuinely coincide at full scale because the payoff dominates the
price. A magnified inset is the standard way to show a real difference that is smaller
than a line width: the outer axis keeps its true, undistorted scale, and the inset shows
the same curves over a narrow window. Nothing is offset or rescaled.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, iv as inv_iv
from amend01 import predict_bs_expiry

DATE = '2025-01-15'
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9.5, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
ST = {'BS': dict(color='#8a8a8a', ls='-', lw=1.9, label='Black-Scholes'),
      'SH': dict(color='#d9822b', ls='-', lw=1.9, label='Single Heston'),
      'DH': dict(color='#1f3f8a', ls='-', lw=2.6, label='Double Heston')}

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


# ------------------------------------------------------------------ C vs S with insets
def fig_s():
    top = q.groupby('expiry').agg(n=('c', 'size'), days=('days', 'median'))
    exps = list(top[top.n >= 8].sort_values('days').index)[:4]
    fig, axs = plt.subplots(1, len(exps), figsize=(5.0 * len(exps), 5.4))
    for ax, e in zip(np.atleast_1d(axs), exps):
        sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
        mk = np.exp(sub.x.to_numpy()); ck = sub.c.to_numpy() * mk
        xs = np.linspace(sub.x.min() - .01, sub.x.max() + .01, 700)
        taus = np.full_like(xs, tau0); ms = np.exp(xs)
        C = {k: c_of(k, xs, taus, np.array([e] * len(xs))) * ms for k in ['BS', 'SH', 'DH']}
        ax.scatter(mk, ck, s=26, c='k', zorder=6, label='market quotes', alpha=.8)
        for k in ['BS', 'SH', 'DH']:
            ax.plot(ms, C[k], **ST[k])
        ax.set_title(f'{e}   ·   {top.loc[e,"days"]:.0f} days', fontsize=10.5)
        ax.set_xlabel('$S/K$   (forward / strike)')

        # centre the inset on the largest model-vs-market gap
        gap = np.abs(c_of('BS', sub.x.to_numpy(), sub.tau.to_numpy(), sub.expiry.to_numpy()) * mk - ck)
        c0 = float(mk[int(np.argmax(gap))]); half = .085
        w = (ms > c0 - half) & (ms < c0 + half)
        ylo = min(C[k][w].min() for k in C); yhi = max(C[k][w].max() for k in C)
        pad = .18 * (yhi - ylo); ylo, yhi = ylo - pad, yhi + pad
        zoom = (ms.max() - ms.min()) / (2 * half)

        ia = inset_axes(ax, width='58%', height='42%', loc='upper left', borderpad=1.4)
        for k in ['BS', 'SH', 'DH']:
            ia.plot(ms, C[k], **{**ST[k], 'label': None})
        inw = (mk > c0 - half) & (mk < c0 + half)
        ia.scatter(mk[inw], ck[inw], s=30, c='k', zorder=6)
        ia.set_xlim(c0 - half, c0 + half); ia.set_ylim(ylo, yhi)
        ia.tick_params(labelsize=7); ia.grid(alpha=.3)
        ia.set_title(f'{zoom:.0f}$\\times$ zoom', fontsize=8.5, pad=2)
        mark_inset(ax, ia, loc1=3, loc2=4, fc='none', ec='#666', lw=1.0, ls='--')
    np.atleast_1d(axs)[0].set_ylabel('call price / strike   $C/K$')
    np.atleast_1d(axs)[-1].legend(frameon=True, framealpha=.92, loc='lower right', fontsize=8.5)
    fig.suptitle(f'BITCOIN {DATE}  —  call price vs underlying, with magnified insets\n'
                 'At full scale the three models coincide; the inset shows the same curves, same axes, '
                 'over a narrow window.', y=.99, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(OUT / 'BTC7_C_vs_S_zoom.png'); plt.close(fig)
    print(' BTC7_C_vs_S_zoom')


# ------------------------------------------------------------------ C vs tau with insets
def fig_tau():
    bands = [0.90, 1.00, 1.10]
    td = np.linspace(3., 90., 500); tau = td / 365.
    fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.4))
    mn = np.exp(q.x.to_numpy())
    for ax, mo in zip(axs, bands):
        xs = np.full_like(tau, np.log(mo))
        # BS needs an expiry label; use the nearest listed expiry per tau
        ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
        C = {k: c_of(k, xs, tau, ee) * mo for k in ['BS', 'SH', 'DH']}
        for k in ['BS', 'SH', 'DH']:
            ax.plot(td, C[k], **ST[k])
        sel = np.abs(mn - mo) < .035
        if sel.sum():
            ax.scatter(q.days[sel], q.c[sel] * mn[sel], s=30, c='k', zorder=6,
                       label='market quotes', alpha=.8)
        ax.set_xlim(td.max(), 0.)
        ax.set_title(f'$S/K$ = {mo:.2f}', fontsize=10.5)
        ax.set_xlabel('time to maturity  $\\tau$  (days)   $\\longrightarrow$  expiry')

        spread = np.max([C['BS'], C['SH'], C['DH']], 0) - np.min([C['BS'], C['SH'], C['DH']], 0)
        t0 = float(td[int(np.argmax(spread))]); half = 11.
        w = (td > t0 - half) & (td < t0 + half)
        ylo = min(C[k][w].min() for k in C); yhi = max(C[k][w].max() for k in C)
        pad = .22 * (yhi - ylo) + 1e-6; ylo, yhi = ylo - pad, yhi + pad
        zoom = (td.max() - td.min()) / (2 * half)
        ia = inset_axes(ax, width='56%', height='42%', loc='lower left', borderpad=1.5)
        for k in ['BS', 'SH', 'DH']:
            ia.plot(td, C[k], **{**ST[k], 'label': None})
        s2 = sel & (q.days.to_numpy() > t0 - half) & (q.days.to_numpy() < t0 + half)
        if s2.sum():
            ia.scatter(q.days[s2], q.c[s2] * mn[s2], s=30, c='k', zorder=6)
        ia.set_xlim(t0 + half, t0 - half); ia.set_ylim(ylo, yhi)
        ia.tick_params(labelsize=7); ia.grid(alpha=.3)
        ia.set_title(f'{zoom:.0f}$\\times$ zoom', fontsize=8.5, pad=2)
        mark_inset(ax, ia, loc1=2, loc2=1, fc='none', ec='#666', lw=1.0, ls='--')
    axs[0].set_ylabel('call price / strike   $C/K$')
    axs[-1].legend(frameon=True, framealpha=.92, loc='upper right', fontsize=8.5)
    fig.suptitle(f'BITCOIN {DATE}  —  call price vs time to maturity, with magnified insets\n'
                 'maturity decreases left to right; the inset magnifies the window of widest separation',
                 y=.99, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(OUT / 'BTC8_C_vs_tau_zoom.png'); plt.close(fig)
    print(' BTC8_C_vs_tau_zoom')


fig_s(); fig_tau()
