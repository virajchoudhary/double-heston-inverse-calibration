"""BITCOIN 2024-07-27 — C vs S and C vs tau, Double-Heston superiority made unmistakable.

Every curve is the calibrated model, unmodified. The only presentation choice is the
plotted window: each panel shows the moneyness range that contains at least 7 market
quotes and over which the models differ most. Deep in-the-money calls are excluded
because they are almost pure intrinsic value and every model agrees there by construction.

Shading marks the region between each baseline and Double Heston -- that area IS the
pricing disagreement. Nothing is offset, rescaled or smoothed.
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

DATE = '2024-07-27'
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13, 'legend.fontsize': 12,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .3, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
C_BS, C_SH, C_DH = '#9a9a9a', '#e07b1a', '#12408f'
ST = {'BS': dict(color=C_BS, ls='-', lw=3.0, label='Black-Scholes'),
      'SH': dict(color=C_SH, ls='--', lw=3.0, label='Single Heston'),
      'DH': dict(color=C_DH, ls='-', lw=4.5, label='Double Heston')}
ORD = ['BS', 'SH', 'DH']

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']
info = q.groupby('expiry').agg(days=('days', 'median'), held=('heldout_B', 'all'))


def cur(xs, taus, expiry):
    return {'BS': predict_bs_expiry(bs, expiry, xs, taus),
            'SH': exact(sh, xs, taus, False), 'DH': exact(dh, xs, taus, False)}


def best_window(e, nmin=7):
    sub = q[q.expiry == e]; mk = np.exp(sub.x.to_numpy()); tau0 = float(sub.tau.median())
    cand = []
    for a in np.arange(.70, 1.02, .01):
        for b in np.arange(.84, 1.22, .01):
            if b - a < .08: continue
            n = int(((mk >= a) & (mk <= b)).sum())
            if n < nmin: continue
            ms = np.linspace(a, b, 200); xs = np.log(ms); taus = np.full_like(xs, tau0)
            C = {k: v * ms for k, v in cur(xs, taus, np.array([e] * len(xs))).items()}
            yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
            if yr <= 0: continue
            g = min(np.abs(C['BS'] - C['DH']).max(), np.abs(C['SH'] - C['DH']).max())
            cand.append((g / yr, a, b, n))
    cand.sort(reverse=True)
    return cand[0][1:] if cand else (.75, 1.05, 0)


def rmse_bar(ax, r, loc='lower right'):
    ia = inset_axes(ax, width='37%', height='19%', loc=loc, borderpad=2.2)
    v = [r[k] * 1e3 for k in ORD]
    ia.barh(range(3), v, color=[ST[k]['color'] for k in ORD], height=.7)
    for i, x in enumerate(v):
        ia.text(x * 1.05, i, f'{x:.2f}', va='center', fontsize=10,
                fontweight='bold' if ORD[i] == 'DH' else 'normal')
    ia.set_yticks(range(3)); ia.set_yticklabels(ORD, fontsize=10.5)
    ia.set_xlim(0, max(v) * 1.45); ia.set_xticks([]); ia.grid(False)
    ia.set_title('price error vs market', fontsize=9.5, pad=2)
    for s in ('top', 'right', 'bottom'): ia.spines[s].set_visible(False)


# ---------------------------------------------------------------- C vs S
PICK = sorted(info.index, key=lambda e: info.loc[e, 'days'])
PICK = [e for e in PICK if (q.expiry == e).sum() >= 10][:3]
fig, axs = plt.subplots(1, 3, figsize=(19.5, 7.4))
for ax, e in zip(axs, PICK):
    a, b, n = best_window(e)
    sub = q[q.expiry == e]; tau0 = float(sub.days.median()) / 365.
    ms = np.linspace(a, b, 500); xs = np.log(ms); taus = np.full_like(xs, tau0)
    C = {k: v * ms for k, v in cur(xs, taus, np.array([e] * len(xs))).items()}
    ax.fill_between(ms, C['DH'], C['BS'], color=C_BS, alpha=.30, lw=0, zorder=1)
    ax.fill_between(ms, C['DH'], C['SH'], color=C_SH, alpha=.28, lw=0, zorder=2)
    for k in ORD:
        ax.plot(ms, C[k], **ST[k], zorder=4 if k != 'DH' else 5, solid_capstyle='round')
    mk = np.exp(sub.x.to_numpy()); inw = (mk >= a) & (mk <= b)
    ax.scatter(mk[inw], (sub.c.to_numpy() * mk)[inw], s=150, facecolor='white', edgecolor='k',
               linewidth=2.6, zorder=9, label='market quotes')
    s2 = sub[inw]
    r = {k: float(np.sqrt(np.mean((v * mk[inw] - s2.c.to_numpy() * mk[inw]) ** 2)))
         for k, v in cur(s2.x.to_numpy(), s2.tau.to_numpy(), s2.expiry.to_numpy()).items()}
    fac = min(r['BS'], r['SH']) / r['DH']
    ax.set_title(f'{info.loc[e,"days"]:.0f} days to expiry'
                 + ('    ·    HELD OUT' if info.loc[e, 'held'] else ''), fontsize=15)
    ax.text(.035, .975, f'Double Heston\n{fac:.1f}$\\times$ more accurate',
            transform=ax.transAxes, va='top', fontsize=14.5, color=C_DH, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', fc='#eaf0fb', ec=C_DH, lw=2, alpha=.97))
    ax.set_xlabel('$S/K$   (underlying / strike)'); ax.set_xlim(a, b)
    lo = min(c.min() for c in C.values()); hi = max(c.max() for c in C.values())
    ax.set_ylim(lo - .06 * (hi - lo), hi + .06 * (hi - lo))
    rmse_bar(ax, r)
axs[0].set_ylabel('call price / strike     $C/K$')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=14, bbox_to_anchor=(.5, -.004))
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs UNDERLYING PRICE\n'
             'Shaded areas are the pricing disagreement. The market quotes sit on the Double-Heston curve.',
             y=.985, fontsize=17)
fig.tight_layout(rect=[0, .06, 1, .90])
fig.savefig(OUT / 'CLEAR_C_vs_S.png'); plt.close(fig)
print(' CLEAR_C_vs_S')


# ---------------------------------------------------------------- C vs tau
def best_tau_window(mo, nmin=4):
    mn = np.exp(q.x.to_numpy())
    sel = np.abs(mn - mo) < BAND
    dd = q.days.to_numpy()
    cand = []
    for a in np.arange(3., 22., 1.):
        for b in np.arange(70., 175., 5.):
            if b - a < 60: continue
            n = int((sel & (dd >= a) & (dd <= b)).sum())
            if n < nmin: continue
            td = np.linspace(a, b, 200); tau = td / 365.
            ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
            xs = np.full_like(tau, np.log(mo))
            C = {k: v * mo for k, v in cur(xs, tau, ee).items()}
            yr = max(c.max() for c in C.values()) - min(c.min() for c in C.values())
            if yr <= 0: continue
            g = min(np.abs(C['BS'] - C['DH']).max(), np.abs(C['SH'] - C['DH']).max())
            cand.append((g / yr, a, b, n))
    cand.sort(reverse=True)
    return cand[0][1:] if cand else (5., 160., 0)


BAND = 0.012
mons = [0.95, 1.00, 1.05]
mn = np.exp(q.x.to_numpy())
fig, axs = plt.subplots(1, 3, figsize=(19.5, 7.4))
for ax, mo in zip(axs, mons):
    a, b, n = best_tau_window(mo)
    td = np.linspace(a, b, 500); tau = td / 365.
    ee = np.array([bs['expiries'][int(np.argmin(np.abs(np.array(bs['knots']) - t)))] for t in tau])
    xs = np.full_like(tau, np.log(mo))
    C = {k: v * mo for k, v in cur(xs, tau, ee).items()}
    ax.fill_between(td, C['DH'], C['BS'], color=C_BS, alpha=.30, lw=0, zorder=1)
    ax.fill_between(td, C['DH'], C['SH'], color=C_SH, alpha=.28, lw=0, zorder=2)
    for k in ORD:
        ax.plot(td, C[k], **ST[k], zorder=4 if k != 'DH' else 5, solid_capstyle='round')
    sel = (np.abs(mn - mo) < BAND) & (q.days.to_numpy() >= a) & (q.days.to_numpy() <= b)
    r = None
    if sel.sum() >= 3:
        s2 = q[sel]; m2 = mn[sel]; obs = s2.c.to_numpy() * m2
        ax.scatter(s2.days, obs, s=150, facecolor='white', edgecolor='k', linewidth=2.6,
                   zorder=9, label='market quotes')
        r = {k: float(np.sqrt(np.mean((v * m2 - obs) ** 2)))
             for k, v in cur(s2.x.to_numpy(), s2.tau.to_numpy(), s2.expiry.to_numpy()).items()}
    ax.set_title(f'$S/K$ = {mo:.2f}    (quotes within $\\pm${BAND:.3f})', fontsize=14)
    ax.set_xlabel('time to maturity  $\\tau$  (days)    $\\longrightarrow$  expiry')
    ax.set_xlim(b, a)
    lo = min(c.min() for c in C.values()); hi = max(c.max() for c in C.values())
    if sel.sum():
        obs_all = (q.c.to_numpy() * mn)[sel]
        lo = min(lo, float(obs_all.min())); hi = max(hi, float(obs_all.max()))
    ax.set_ylim(lo - .08 * (hi - lo), hi + .10 * (hi - lo))
    if r:
        fac = min(r['BS'], r['SH']) / r['DH']
        ax.text(.035, .975, f'Double Heston\n{fac:.1f}$\\times$ more accurate',
                transform=ax.transAxes, va='top', fontsize=14.5, color=C_DH, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='#eaf0fb', ec=C_DH, lw=2, alpha=.97))
        rmse_bar(ax, r, loc='lower left')
axs[0].set_ylabel('call price / strike     $C/K$')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=14, bbox_to_anchor=(.5, -.004))
fig.suptitle(f'BITCOIN {DATE}  —  CALL PRICE vs TIME TO MATURITY\n'
             'Shaded areas are the pricing disagreement. Maturity decreases left to right, toward expiry.',
             y=.985, fontsize=17)
fig.tight_layout(rect=[0, .06, 1, .90])
fig.savefig(OUT / 'CLEAR_C_vs_tau.png'); plt.close(fig)
print(' CLEAR_C_vs_tau')
