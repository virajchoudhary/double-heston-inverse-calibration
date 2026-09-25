"""REAL MARKET HELD-OUT SURFACE figures, plus the two PINN validation figures.

Market surface: the one already frozen by dh_relevance_selection_v1 (IWM, selected ex ante).
Models: the calibrations already computed there - Black-Scholes one volatility per expiry, Single
Heston five parameters, Double Heston ten, all fitted on the calibration half. The DH-PINN appears at
its own capacity (published parameters plus one level scale) and its legend says so.

Validation: DH-PINN against exact Double Heston; BS-PINN against analytic Black-Scholes at the
BS-PINN's own rate, dividend and calibrated volatility. Never against each other's targets.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent; EXP = OUT.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / 'dh_pinn_v5')); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
import pricers as P
import common as CC
from engine import exact, iv as inv_iv, black
from amend01 import fit_bs_expiry, predict_bs_expiry
from track_b import prior_c

FIG, DATA = OUT / 'figures', OUT / 'data'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
SEL = EXP / 'dh_relevance_selection_v1'
R = json.loads((SEL / 'model_comparison.json').read_text())
SYM = R['symbol']; NAME = SYM.lstrip('_')
CFG = json.loads((EXP / 'btc_multifactor_v1' / 'config.json').read_text())
TAG = f'REAL MARKET HELD-OUT SURFACE  ·  {NAME}, selected ex ante by the frozen DH-relevance score'
LAB = {'BS': 'Black-Scholes, one vol per expiry', 'SH': 'Single Heston, 5 parameters',
       'DH': 'Double Heston, 10 parameters', 'PINN': 'DH-PINN (published params + level scale)'}
ROWS = []


def surface():
    q = pd.read_csv(SEL / 'surfaces' / f'{SYM}.csv')
    order = q.groupby('expiry').tau.median().sort_values().index.tolist()
    q['expiry_rank'] = q.expiry.map({e: i for i, e in enumerate(order)})
    q = q.sort_values(['expiry_rank', 'strike']).reset_index(drop=True)
    q['strike_rank'] = q.groupby('expiry_rank').cumcount()
    q['cal'] = (q.expiry_rank + q.strike_rank) % 2 == 0
    return q


def fitted(q):
    c = q[q.cal]
    bs = fit_bs_expiry(*(c[k].to_numpy() for k in ['expiry', 'x', 'tau', 'c', 'vega_w']), CFG['bounds'])
    par = {k: np.array(R['strongest_fair_heldout'][f'{k}_calibrated']['params']) for k in ['SH', 'DH']}
    scale = R['matched_capacity']['DH_PINN_NEW']['scale']
    return bs, par, scale


def model_c(m, ex, x, tau, bs, par, scale):
    """Forward-normalised call price c = C/F for each market model."""
    if m == 'BS': return predict_bs_expiry(bs, ex, x, tau)
    if m in ('SH', 'DH'): return exact(par[m], x, tau, feller=False)
    return prior_c('DH_PINN', x, tau, scale, P.CC_NETS)


def fig_market_time_value(q, bs, par, scale):
    ho = q[~q.cal]
    exp = (ho.groupby('expiry').days.median() - 90).abs().idxmin()
    g = ho[ho.expiry.eq(exp)].sort_values('strike')
    tau = float(g.tau.median()); F = float(g.forward.median()); K = float(g.strike.iloc[(g.strike - F).abs().argmin()])
    S_m = F * K / g.strike.to_numpy(); C_m = g.c.to_numpy() * S_m
    keep = np.abs(np.log(S_m / K)) <= .36; S_m, C_m = S_m[keep], C_m[keep]
    o = np.argsort(S_m); S_m, C_m = S_m[o], C_m[o]
    TV_m = C_m - np.maximum(S_m - K, 0)
    S = np.linspace(S_m.min(), S_m.max(), 400); x = np.log(S / K); t = np.full_like(S, tau)
    ex = np.full(len(S), exp, dtype=object)
    tv = {m: model_c(m, ex, x, t, bs, par, scale) * S - np.maximum(S - K, 0) for m in ['BS', 'SH', 'DH', 'PINN']}
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 7.8), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    a.plot(S_m, TV_m, 'o', ms=4.5, color='k', zorder=5, label='market, held-out quotes')
    for m in ['BS', 'SH', 'DH', 'PINN']: a.plot(S, tv[m], **{**P.STYLE[m], 'label': LAB[m]})
    a.set_ylabel(r'time value $C-\max(S-K,0)$'); a.legend(frameon=False, loc='upper right', handlelength=2.4)
    a.set_title(f'{NAME}: time value vs underlying price  ·  expiry {exp} ({tau*365:.0f} d)')
    for m in ['BS', 'SH', 'DH', 'PINN']:
        b.plot(S_m, np.interp(S_m, S, tv[m]) - TV_m, **{**P.STYLE[m], 'label': LAB[m].split(',')[0] + r' $-$ market'})
    b.axhline(0, color='k', lw=1.1)
    b.set_xlabel(f'underlying price $S$   ($K={K:g}$, quotes rescaled by homogeneity)')
    b.set_ylabel('time value $-$ market'); b.legend(frameon=False, loc='best', handlelength=2.4)
    fig.text(.5, .955, TAG, ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / 'market_time_value_vs_S.png'); plt.close(fig)
    pd.DataFrame({'S': S, 'tau_days': tau * 365, **{f'TV_{m}': tv[m] for m in tv}}).to_csv(DATA / 'market_time_value_vs_S.csv', index=False)
    return K


def fig_market_smile(q, bs, par, scale):
    ho = q[~q.cal]; days = ho.groupby('expiry').days.median()
    picks = [days.sub(d).abs().idxmin() for d in (30, 90, 365)]
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=False)
    rows = []
    for ax, exp in zip(axs, picks):
        g = ho[ho.expiry.eq(exp)].sort_values('x'); tau = float(g.tau.median())
        xs = np.linspace(g.x.min(), g.x.max(), 200); t = np.full_like(xs, tau); ex = np.full(len(xs), exp, dtype=object)
        ax.plot(g.x, 100 * g.market_iv, 'o', ms=4, color='k', zorder=5, label='market')
        for m in ['BS', 'SH', 'DH', 'PINN']:
            iv = 100 * inv_iv(model_c(m, ex, xs, t, bs, par, scale), xs, t)
            ax.plot(xs, iv, **{**P.STYLE[m], 'label': LAB[m]})
            e = 100 * inv_iv(model_c(m, g.expiry.to_numpy(), g.x.to_numpy(), g.tau.to_numpy(), bs, par, scale),
                             g.x.to_numpy(), g.tau.to_numpy()) - 100 * g.market_iv.to_numpy()
            rows.append({'figure': 'market_iv_smile', 'setting': 'market', 'quantity': 'implied vol (vol pts)',
                         'tau_days': tau * 365, 'model': m, 'RMSE_vs_market': float(np.sqrt(np.nanmean(e ** 2))),
                         'max_abs_vs_market': float(np.nanmax(np.abs(e))), 'P95_abs_vs_market': float(np.nanquantile(np.abs(e), .95))})
        ax.set_title(f'{exp}  ({tau*365:.0f} days)'); ax.set_xlabel(r'$x=\log(F/K)$')
    axs[0].set_ylabel('implied volatility (%)'); axs[0].legend(frameon=False, loc='best', handlelength=2.4)
    fig.suptitle(f'{NAME}: implied-volatility smile, held-out quotes', y=.98)
    fig.text(.5, .915, TAG, ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .90]); fig.savefig(FIG / 'market_iv_smile.png'); plt.close(fig)
    ROWS.extend(rows)


def fig_market_term_structure(q, bs, par, scale):
    ho = q[~q.cal]; rows = []
    for e, v in ho.groupby('expiry'):
        v = v.sort_values('x')
        if len(v) < 3 or not (v.x.min() < 0 < v.x.max()): continue
        rows.append({'expiry': e, 'days': float(v.days.median()), 'tau': float(v.tau.median()),
                     'market': float(np.interp(0., v.x.to_numpy(), v.market_iv.to_numpy()))})
    a = pd.DataFrame(rows).sort_values('days')
    for m in ['BS', 'SH', 'DH', 'PINN']:
        a[m] = [100 * inv_iv(model_c(m, np.array([r.expiry], dtype=object), np.array([0.]), np.array([r.tau]), bs, par, scale),
                             np.array([0.]), np.array([r.tau]))[0] for r in a.itertuples()]
    a['market'] *= 100
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.4, 7.8), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    ax.plot(a.days, a.market, 'o-', ms=4.5, color='k', lw=1.3, zorder=5, label='market ATM')
    for m in ['BS', 'SH', 'DH', 'PINN']: ax.plot(a.days, a[m], **{**P.STYLE[m], 'label': LAB[m], 'marker': 'o', 'ms': 3})
    ax.set_ylabel('at-the-money implied volatility (%)'); ax.legend(frameon=False, loc='best', handlelength=2.4)
    ax.set_title(f'{NAME}: at-the-money implied-volatility term structure, held-out quotes')
    for m in ['BS', 'SH', 'DH', 'PINN']:
        bx.plot(a.days, a[m] - a.market, **{**P.STYLE[m], 'label': LAB[m].split(',')[0] + r' $-$ market', 'marker': 'o', 'ms': 3})
        ROWS.append({'figure': 'market_atm_iv_term_structure', 'setting': 'market', 'quantity': 'ATM IV (vol pts)',
                     'tau_days': np.nan, 'model': m, 'RMSE_vs_market': float(np.sqrt(np.mean((a[m] - a.market) ** 2))),
                     'max_abs_vs_market': float(np.abs(a[m] - a.market).max()), 'P95_abs_vs_market': float(np.quantile(np.abs(a[m] - a.market), .95))})
    bx.axhline(0, color='k', lw=1.1); bx.set_xlabel(r'remaining maturity $\tau$ (days)')
    bx.set_ylabel('ATM IV $-$ market (vol pts)'); bx.legend(frameon=False, loc='best', handlelength=2.4)
    fig.text(.5, .955, TAG, ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / 'market_atm_iv_term_structure.png'); plt.close(fig)
    a.to_csv(DATA / 'market_atm_iv_term_structure.csv', index=False)


def fig_market_residual_panels(q, bs, par, scale):
    ho = q[~q.cal].copy()
    x, t, ex = ho.x.to_numpy(), ho.tau.to_numpy(), ho.expiry.to_numpy()
    err = {m: 100 * (inv_iv(model_c(m, ex, x, t, bs, par, scale), x, t) - ho.market_iv.to_numpy()) for m in ['BS', 'SH', 'DH', 'PINN']}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    xb = np.linspace(-.36, .36, 13); mid = (xb[:-1] + xb[1:]) / 2; idx = np.digitize(x, xb)
    tb = [7, 30, 90, 365, 9999]; tmid = ['<=30d', '30-90d', '90-365d', '>365d']
    for m in ['BS', 'SH', 'DH', 'PINN']:
        e = err[m]; ok = np.isfinite(e)
        a.plot(mid, [float(np.sqrt(np.mean(e[ok][idx[ok] == i] ** 2))) if (idx[ok] == i).any() else np.nan for i in range(1, len(xb))],
               **{**P.STYLE[m], 'label': LAB[m], 'marker': 'o', 'ms': 3.5})
        b.plot(range(4), [float(np.sqrt(np.mean(e[ok][(ho.days.to_numpy()[ok] >= tb[i]) & (ho.days.to_numpy()[ok] < tb[i + 1])] ** 2)))
                          for i in range(4)], **{**P.STYLE[m], 'label': LAB[m], 'marker': 'o', 'ms': 4})
        ROWS.append({'figure': 'market_residual_panels', 'setting': 'market', 'quantity': 'IV RMSE all held-out (vol pts)',
                     'tau_days': np.nan, 'model': m, 'RMSE_vs_market': float(np.sqrt(np.nanmean(e ** 2))),
                     'max_abs_vs_market': float(np.nanmax(np.abs(e))), 'P95_abs_vs_market': float(np.nanquantile(np.abs(e), .95))})
    a.set_xlabel(r'log-moneyness $x=\log(F/K)$'); a.set_ylabel('implied-vol RMSE (vol points)')
    a.set_title('Held-out error by moneyness'); a.legend(frameon=False, loc='upper center', handlelength=2.4)
    b.set_xticks(range(4), tmid); b.set_xlabel('maturity bucket'); b.set_ylabel('implied-vol RMSE (vol points)')
    b.set_title('Held-out error by maturity')
    fig.suptitle(f'{NAME}: held-out implied-volatility error', y=.98)
    fig.text(.5, .915, TAG, ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .90]); fig.savefig(FIG / 'market_residual_panels.png'); plt.close(fig)
    pd.DataFrame({'x': x, 'days': ho.days.to_numpy(), **{f'err_{m}': err[m] for m in err}}).to_csv(DATA / 'market_residuals.csv', index=False)


# ------------------------------------------------------------------ validation figures
def fig_dh_pinn_validation():
    S = np.linspace(.70 * P.K, 1.30 * P.K, 300); tds = [30., 90., 365., 730.]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 7.8), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    rows = {}
    for td, col in zip(tds, ['#9ecae1', '#6baed6', '#3182bd', '#08519c']):
        dh = P.price('DH', S, np.full_like(S, td / 365.)); pn = P.price('PINN', S, np.full_like(S, td / 365.))
        a.plot(S, dh, color=col, lw=2.0, label=fr'exact DH, $\tau$={td:g} d')
        a.plot(S, pn, color='#16a085', lw=1.2, ls='--', label='DH-PINN' if td == tds[0] else None)
        b.plot(S, pn - dh, color=col, lw=1.6, label=fr'$\tau$={td:g} d')
        rows[f'DH_{td:g}d'] = dh; rows[f'PINN_{td:g}d'] = pn
    a.set_ylabel('call price $C$'); a.legend(frameon=False, loc='upper left', ncol=2, handlelength=2.2)
    a.set_title('DH-PINN validated against its own target: exact Double Heston')
    b.axhline(0, color='k', lw=1.); b.set_xlabel('underlying price $S$   ($K=100$)')
    b.set_ylabel(r'$C_{PINN}-C_{DH}$'); b.legend(frameon=False, loc='best', ncol=2, handlelength=2.2)
    e = np.concatenate([rows[f'PINN_{td:g}d'] - rows[f'DH_{td:g}d'] for td in tds])
    fig.text(.5, .955, f'CONTROLLED TWO-TIMESCALE BENCHMARK  ·  max |error| {np.abs(e).max():.2e}, '
                       f'RMSE {np.sqrt(np.mean(e**2)):.2e}, P95 {np.quantile(np.abs(e), .95):.2e} in price units',
             ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / 'dh_pinn_vs_dh_validation.png'); plt.close(fig)
    pd.DataFrame({'S': S, **rows}).to_csv(DATA / 'dh_pinn_vs_dh_validation.csv', index=False)
    ROWS.append({'figure': 'dh_pinn_vs_dh_validation', 'setting': 'controlled', 'quantity': 'price vs exact DH',
                 'tau_days': np.nan, 'model': 'PINN', 'RMSE_vs_DH': float(np.sqrt(np.mean(e ** 2))),
                 'max_abs_vs_DH': float(np.abs(e).max()), 'P95_abs_vs_DH': float(np.quantile(np.abs(e), .95))})


def fig_bs_pinn_validation():
    meta = P.bs_pinn_meta()
    lo, hi = np.exp(meta['x_min']) * P.K, np.exp(meta['x_max']) * P.K
    S = np.linspace(lo, hi, 300); tds = [30., 90., 365., 730.]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 7.8), sharex=True, gridspec_kw={'height_ratios': [1.6, 1]})
    rows = {}; errs = []
    for td, col in zip(tds, ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']):
        an = P.bs_analytic(S, np.full_like(S, td / 365.)); pn = P.bs_pinn_price(S, np.full_like(S, td / 365.))
        a.plot(S, an, color=col, lw=2.0, label=fr'analytic BS, $\tau$={td:g} d')
        a.plot(S, pn, color='#000', lw=1.0, ls='--', label='BS-PINN' if td == tds[0] else None)
        b.plot(S, pn - an, color=col, lw=1.6, label=fr'$\tau$={td:g} d')
        rows[f'BS_{td:g}d'] = an; rows[f'BS_PINN_{td:g}d'] = pn; errs.append(pn - an)
    a.set_ylabel('call price $C$'); a.legend(frameon=False, loc='upper left', ncol=2, handlelength=2.2)
    a.set_title('BS-PINN validated against its own target: analytic Black-Scholes')
    b.axhline(0, color='k', lw=1.); b.set_xlabel('underlying price $S$   ($K=100$)')
    b.set_ylabel(r'$C_{BS\ PINN}-C_{BS}$'); b.legend(frameon=False, loc='best', ncol=2, handlelength=2.2)
    e = np.concatenate(errs)
    fig.text(.5, .955, f"BS-PINN own convention: $\\sigma$={meta['sigma']:.6f}, $r$={meta['rate']}, $q$={meta['dividend']}  ·  "
                       f"max |error| {np.abs(e).max():.2e}, RMSE {np.sqrt(np.mean(e**2)):.2e} in price units",
             ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / 'bs_pinn_vs_bs_validation.png'); plt.close(fig)
    pd.DataFrame({'S': S, **rows}).to_csv(DATA / 'bs_pinn_vs_bs_validation.csv', index=False)
    ROWS.append({'figure': 'bs_pinn_vs_bs_validation', 'setting': 'BS-PINN own convention', 'quantity': 'price vs analytic BS',
                 'tau_days': np.nan, 'model': 'BS_PINN', 'RMSE_vs_own_target': float(np.sqrt(np.mean(e ** 2))),
                 'max_abs_vs_own_target': float(np.abs(e).max()), 'P95_abs_vs_own_target': float(np.quantile(np.abs(e), .95))})


if __name__ == '__main__':
    P.CC_NETS = __import__('evaluate').load_stage('FINAL', 'final')
    q = surface(); bs, par, scale = fitted(q)
    fig_market_time_value(q, bs, par, scale); print('market time value done', flush=True)
    fig_market_smile(q, bs, par, scale); print('market smile done', flush=True)
    fig_market_term_structure(q, bs, par, scale); print('market term structure done', flush=True)
    fig_market_residual_panels(q, bs, par, scale); print('market residuals done', flush=True)
    fig_dh_pinn_validation(); fig_bs_pinn_validation(); print('validation done', flush=True)
    pd.DataFrame(ROWS).to_csv(DATA / 'market_and_validation_metrics.csv', index=False)
    print('\n', pd.DataFrame(ROWS)[['figure', 'model', 'RMSE_vs_market', 'RMSE_vs_DH', 'RMSE_vs_own_target']].to_string(index=False)
          if 'RMSE_vs_own_target' in pd.DataFrame(ROWS) else pd.DataFrame(ROWS).to_string(index=False))
