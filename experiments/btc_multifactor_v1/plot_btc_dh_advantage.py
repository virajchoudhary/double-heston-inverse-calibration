"""BTC: C vs S and C vs tau, three calibrated models against real market quotes.

Date chosen by a NEUTRAL rule stated before looking at any error: among shock-regime
design-B test dates, the one with the most held-out quotes (the richest surface).
2025-10-12, 156 quotes across 9 expiries.

Black-Scholes is the CORRECTED per-expiry fit from AMENDMENT_01, not the flawed v1
per-quote version. Using the flawed one would overstate Double Heston's advantage.
Single Heston and Double Heston use the selected feller_free variant.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine import exact, black, iv as inv_iv, bs_predict
from amend01 import predict_bs_expiry

DATE = sys.argv[1] if len(sys.argv) > 1 else '2025-10-12'
OUT = HERE / 'figures_dh_advantage'; OUT.mkdir(exist_ok=True)
TAG = sys.argv[2] if len(sys.argv) > 2 else 'BTC'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
SUB = sys.argv[3] if len(sys.argv) > 3 else 'maturity decreases left to right'
ST = {'BS': dict(color='#8a8a8a', ls='-', lw=1.8, label='Black-Scholes (one vol per expiry)'),
      'SH': dict(color='#d9822b', ls='-', lw=1.8, label='Single Heston (1 variance factor)'),
      'DH': dict(color='#1f3f8a', ls='-', lw=2.4, label='Double Heston (2 variance factors)')}

q = pd.read_csv(HERE / 'artifacts' / 'surfaces' / f'{DATE}.csv')
fb = HERE / 'artifacts' / 'fits' / 'B' / DATE
dh = json.load(open(fb / 'DH_feller_free.json'))['fit']['best']['params']
sh = json.load(open(fb / 'SH_feller_free.json'))['fit']['best']['params']
bs = json.load(open(HERE / 'artifacts' / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json'))['fit']


def price_c(model, x, tau, expiry=None):
    """Forward-normalised call c = C/F."""
    x = np.atleast_1d(np.asarray(x, float)); tau = np.atleast_1d(np.asarray(tau, float))
    if model == 'DH': return exact(dh, x, tau, False)
    if model == 'SH': return exact(sh, x, tau, False)
    if expiry is not None: return predict_bs_expiry(bs, np.asarray(expiry), x, tau)
    return bs_predict(bs, x, tau)


# ------------------------------------------------------------------ FIGURE 1: C vs S
def fig_c_vs_s():
    top = q.groupby('expiry').agg(n=('c', 'size'), days=('days', 'median')).sort_values('n', ascending=False)
    exps = list(top.head(4).sort_values('days').index)
    fig, axs = plt.subplots(2, 4, figsize=(17.4, 8.0), gridspec_kw={'height_ratios': [2.0, 1.15]})
    rows = []
    for j, e in enumerate(exps):
        m = q.expiry == e
        sub = q[m]; tau0 = float(sub.days.median()) / 365.
        F = float(sub.forward.median())
        a, b = axs[0, j], axs[1, j]
        # market quotes: y = C/K, x = F/K   (this is C vs S, both divided by the strike)
        a.scatter(np.exp(sub.x), sub.c * np.exp(sub.x), s=22, c='k', zorder=5,
                  label='market quotes', alpha=.75)
        xs = np.linspace(sub.x.min() - .02, sub.x.max() + .02, 400)
        taus = np.full_like(xs, tau0)
        for k in ['BS', 'SH', 'DH']:
            c = price_c(k, xs, taus, expiry=np.array([e] * len(xs)) if k == 'BS' else None)
            a.plot(np.exp(xs), c * np.exp(xs), **ST[k])
        a.set_title(f'{e}   ·   {top.loc[e,"days"]:.0f} days   ·   {int(top.loc[e,"n"])} quotes', fontsize=10)
        a.set_xlabel('$S/K$   (forward / strike)')
        # residual in implied-volatility points, the metric the study scored on
        for k in ['BS', 'SH', 'DH']:
            c = price_c(k, sub.x.to_numpy(), sub.tau.to_numpy(),
                        expiry=sub.expiry.to_numpy() if k == 'BS' else None)
            mi = inv_iv(c, sub.x.to_numpy(), sub.tau.to_numpy())
            d = 100 * (mi - sub.market_iv.to_numpy())
            ok = np.isfinite(d)
            b.scatter(np.exp(sub.x)[ok], d[ok], s=16, color=ST[k]['color'], alpha=.85)
            rows.append({'expiry': e, 'days': top.loc[e, 'days'], 'model': k, 'n': int(ok.sum()),
                         'IV_RMSE_volpts': float(np.sqrt(np.mean(d[ok] ** 2)))})
        b.axhline(0, color='k', lw=1.1)
        b.set_xlabel('$S/K$'); 
        r = {x['model']: x['IV_RMSE_volpts'] for x in rows if x['expiry'] == e}
        b.text(.02, .04, 'IV RMSE (vol pts)   BS %.2f   SH %.2f   DH %.2f' % (r['BS'], r['SH'], r['DH']),
               transform=b.transAxes, fontsize=8.5, va='bottom', fontweight='bold', color='#8a2020',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb', alpha=.92))
    axs[0, 0].set_ylabel('call price / strike   $C/K$')
    axs[1, 0].set_ylabel('model $-$ market  (vol points)')
    axs[0, 0].legend(frameon=False, loc='upper left', fontsize=8, handlelength=2.2)
    fig.suptitle(f'BITCOIN options, {DATE}  —  call price vs underlying, three calibrated models against real quotes\n'
                 + SUB, y=.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(OUT / f'{TAG}_C_vs_S.png'); plt.close(fig)
    print(f' {TAG}_C_vs_S')
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ FIGURE 2: C vs tau
def fig_c_vs_tau():
    bands = [(0.85, 'S/K = 0.85  (out of the money)'), (1.00, 'S/K = 1.00  (at the money)'),
             (1.15, 'S/K = 1.15  (in the money)')]
    td = np.linspace(2., 360., 400); tau = td / 365.
    fig, axs = plt.subplots(2, 3, figsize=(15.0, 7.4), gridspec_kw={'height_ratios': [2.0, 1.15]})
    mn = np.exp(q.x.to_numpy())
    for j, (mo, lab) in enumerate(bands):
        a, b = axs[0, j], axs[1, j]
        xs = np.full_like(tau, np.log(mo))
        for k in ['BS', 'SH', 'DH']:
            c = price_c(k, xs, tau)
            a.plot(td, c * mo, **ST[k])
        sel = np.abs(mn - mo) < 0.04
        if sel.sum():
            a.scatter(q.days[sel], q.c[sel] * mn[sel], s=26, c='k', zorder=5,
                      label=f'market quotes within $\\pm$0.04', alpha=.8)
        a.set_title(lab, fontsize=10)
        a.set_xlim(td.max(), 0.)
        for k in ['BS', 'SH', 'DH']:
            if sel.sum():
                sub = q[sel]
                c = price_c(k, sub.x.to_numpy(), sub.tau.to_numpy(),
                            expiry=sub.expiry.to_numpy() if k == 'BS' else None)
                mi = inv_iv(c, sub.x.to_numpy(), sub.tau.to_numpy())
                d = 100 * (mi - sub.market_iv.to_numpy()); ok = np.isfinite(d)
                b.scatter(sub.days.to_numpy()[ok], d[ok], s=22, color=ST[k]['color'], alpha=.85)
        b.axhline(0, color='k', lw=1.1); b.set_xlim(td.max(), 0.)
        b.set_xlabel('time to maturity  $\\tau$  (days)   $\\longrightarrow$  expiry')
    axs[0, 0].set_ylabel('call price / strike   $C/K$')
    axs[1, 0].set_ylabel('model $-$ market  (vol points)')
    axs[0, 0].legend(frameon=False, loc='upper right', fontsize=8, handlelength=2.2)
    fig.suptitle(f'BITCOIN options, {DATE}  —  call price vs time to maturity at fixed moneyness\n'
                 + SUB, y=.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(OUT / f'{TAG}_C_vs_tau.png'); plt.close(fig)
    print(f' {TAG}_C_vs_tau')


r = fig_c_vs_s(); fig_c_vs_tau()
r.to_csv(OUT / f'{TAG}_per_expiry_iv_rmse.csv', index=False)
print('\nIV RMSE by expiry (vol points), all quotes that day')
print(r.pivot_table(index=['days'], columns='model', values='IV_RMSE_volpts').round(3).to_string())
