"""Headline figures on the ex-ante selected real surface.

Two capacity settings, never mixed in one panel:
  * strongest fair calibration (BS one vol per expiry, SH five parameters, DH ten), fitted on the
    checkerboard calibration half and drawn against the held-out half;
  * matched capacity (published parameters plus one level scale), the only setting the PINN enters.
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
sys.path.insert(0, str(EXP / 'dh_pinn_v5')); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
sys.path.insert(0, str(EXP / 'financial_curve_figures_v2' / 'scripts'))
from engine import exact, iv as inv_iv, fit_heston
from amend01 import fit_bs_expiry, predict_bs_expiry
from track_b import prior_c
from evaluate import load_stage
import common as CC

FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True); DATA = OUT / 'data'; DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
CFG = json.loads((EXP / 'btc_multifactor_v1' / 'config.json').read_text())
R = json.loads((OUT / 'model_comparison.json').read_text())
SYM = R['symbol']; NAME = SYM.lstrip('_')
STYLE = CC.STYLE
SEL_NOTE = (f'Real market surface {NAME}, selected ex ante by the frozen DH-relevance score '
            f'(rank 1 of 17 in-domain candidates)')


def split(q):
    q = q.copy()
    order = q.groupby('expiry').tau.median().sort_values().index.tolist()
    q['expiry_rank'] = q.expiry.map({e: i for i, e in enumerate(order)})
    q = q.sort_values(['expiry_rank', 'strike']).reset_index(drop=True)
    q['strike_rank'] = q.groupby('expiry_rank').cumcount()
    q['cal'] = (q.expiry_rank + q.strike_rank) % 2 == 0
    return q


def calibrated(q):
    """Refit on the calibration half exactly as compare_models.heldout() did."""
    c = q[q.cal]
    x, t, y, w, ex = (c[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w', 'expiry'])
    bs = fit_bs_expiry(ex, x, t, y, w, CFG['bounds'])
    p = {k: np.array(R['strongest_fair_heldout'][f'{k}_calibrated']['params']) for k in ['SH', 'DH']}
    return bs, p


def price_curve(kind, S, tau, K, bs=None, p=None, scale=None, net=None):
    x = np.log(S / K); t = np.full_like(S, tau)
    if kind == 'BS_cal': c = predict_bs_expiry(bs, np.full(len(S), '_', dtype=object), x, t)
    elif kind in ('SH_cal', 'DH_cal'): c = exact(p, x, t, feller=False)
    else: c = prior_c(kind, x, t, scale, net)
    return c * S


def fig_C_vs_S(q, bs, par):
    ho = q[~q.cal]
    e = ho.groupby('expiry').days.median()
    exp = (e - 90).abs().idxmin()
    g = ho[ho.expiry.eq(exp)].sort_values('strike')
    tau = float(g.tau.median()); F = float(g.forward.median()); K = float(g.strike.iloc[(g.strike - F).abs().argmin()])
    S_m = F * K / g.strike.to_numpy(); C_m = g.c.to_numpy() * S_m
    keep = np.abs(np.log(S_m / K)) <= .36; S_m, C_m = S_m[keep], C_m[keep]
    o = np.argsort(S_m); S_m, C_m = S_m[o], C_m[o]
    S = np.linspace(S_m.min(), S_m.max(), 400)
    cur = {'BS': price_curve('BS_cal', S, tau, K, bs=bs), 'SH': price_curve('SH_cal', S, tau, K, p=par['SH']),
           'DH': price_curve('DH_cal', S, tau, K, p=par['DH'])}
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    a.plot(S_m, C_m, 'o', ms=4.5, color='k', zorder=5, label=f'market, held-out quotes ({NAME})')
    for m in ['BS', 'SH', 'DH']:
        lab = {'BS': 'Black-Scholes, one vol per expiry', 'SH': 'Single Heston, 5 parameters', 'DH': 'Double Heston, 10 parameters'}[m]
        a.plot(S, cur[m], **{**STYLE[m], 'label': lab})
    a.plot(S, np.maximum(S - K, 0), ls=':', lw=1., color='k', label=r'payoff $\max(S-K,0)$')
    a.set_ylabel('call price $C$'); a.legend(frameon=False, loc='upper left', handlelength=2.4)
    a.set_title(f'{NAME} call price vs underlying price  ·  strongest fair calibration, held-out quotes')
    for m in ['BS', 'SH', 'DH']:
        b.plot(S_m, np.interp(S_m, S, cur[m]) - C_m, **{**STYLE[m], 'label': STYLE[m]['label'] + r' $-$ market'})
    b.axhline(0, color='k', lw=1.1)
    b.set_xlabel(f'underlying price $S$   (strike $K={K:g}$; quotes rescaled by homogeneity)')
    b.set_ylabel(r'$C_{model}-C_{market}$'); b.legend(frameon=False, loc='best', handlelength=2.4)
    fig.text(.5, .962, f'Real surface {NAME}, selected ex ante by the frozen DH-relevance score (rank 1 of 17)',
             ha='center', fontsize=8.5, color='#555')
    fig.text(.5, .942, f'expiry {exp} ({tau*365:.0f} days)  ·  held-out IV RMSE: BS 5.39, SH 1.89, DH 0.58 vol points',
             ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .932]); fig.savefig(FIG / f'{NAME}_C_vs_S_strongest_calibration.png'); plt.close(fig)
    pd.DataFrame({'S': S, 'tau_days': tau * 365, **{f'C_{m}': cur[m] for m in cur}}).to_csv(DATA / f'{NAME}_C_vs_S.csv', index=False)
    return K, tau


def fig_calendar(q, bs, par, K):
    T = 365.; t = np.linspace(0, T - 1., 400); S0 = 1.05 * K
    cur = {}
    for m, kind, kw in [('BS', 'BS_cal', {'bs': bs}), ('SH', 'SH_cal', {'p': par['SH']}), ('DH', 'DH_cal', {'p': par['DH']})]:
        cur[m] = np.array([price_curve(kind, np.array([S0]), (T - ti) / 365., K, **kw)[0] for ti in t])
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for m in ['BS', 'SH', 'DH']:
        lab = {'BS': 'Black-Scholes, one vol per expiry', 'SH': 'Single Heston, 5 parameters', 'DH': 'Double Heston, 10 parameters'}[m]
        ax.plot(t, cur[m], **{**STYLE[m], 'label': lab})
    ax.axhline(max(S0 - K, 0), color='k', ls=':', lw=1., label=r'payoff $\max(S-K,0)$')
    ax.set_xlabel('calendar time $t$ progressing toward expiry   (days, expiry at $t=T=365$)')
    ax.set_ylabel(f'call price $C$   ($S={S0:.0f}$, $K={K:g}$)')
    ax.set_title(f'{NAME}: European call value as expiry approaches')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, .88), handlelength=2.4)
    ax.text(.03, .30, r'$\tau = T-t$', transform=ax.transAxes, fontsize=10, color='#444')
    fig.text(.5, .93, f'Real surface {NAME}, selected ex ante by the frozen DH-relevance score  ·  '
                      'parameters calibrated to that surface', ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .925]); fig.savefig(FIG / f'{NAME}_C_vs_calendar_time.png'); plt.close(fig)
    pd.DataFrame({'t_days': t, 'tau_days': T - t, 'S': S0, **{f'C_{m}': cur[m] for m in cur}}).to_csv(DATA / f'{NAME}_C_vs_calendar_time.csv', index=False)


def fig_matched(q, K, tau):
    """Matched capacity: published parameters plus one level scale. The PINN enters here."""
    M = R['matched_capacity']; net = load_stage('FINAL', 'final')
    S = np.linspace(.75 * K, 1.25 * K, 400); x = np.log(S / K); t = np.full_like(S, tau)
    cur = {'BS': prior_c('BS', x, t, M['BS']['scale']) * S, 'SH': prior_c('SH', x, t, M['SH']['scale']) * S,
           'DH': prior_c('DH', x, t, M['DH']['scale']) * S,
           'PINN': prior_c('DH_PINN', x, t, M['DH_PINN_NEW']['scale'], net) * S}
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    for m in ['BS', 'SH', 'DH', 'PINN']: a.plot(S, cur[m], **STYLE[m])
    a.plot(S, np.maximum(S - K, 0), ls=':', lw=1., color='k', label=r'payoff $\max(S-K,0)$')
    a.set_ylabel('call price $C$'); a.legend(frameon=False, loc='upper left', handlelength=2.4)
    a.set_title(f'{NAME}: matched capacity  ·  published parameters, one level scale')
    for m in ['BS', 'SH', 'PINN']:
        b.plot(S, cur[m] - cur['DH'], **{**STYLE[m], 'label': STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=STYLE['DH']['color'], lw=1.4)
    b.set_xlabel(f'underlying price $S$   (strike $K={K:g}$)'); b.set_ylabel(r'$C_{model}-C_{DH}$')
    b.legend(frameon=False, loc='best', handlelength=2.4)
    fig.text(.5, .955, SEL_NOTE + fr'  ·  $\tau$ = {tau*365:.0f} days  ·  the PINN can only enter at this capacity',
             ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / f'{NAME}_C_vs_S_matched_capacity.png'); plt.close(fig)
    pd.DataFrame({'S': S, 'tau_days': tau * 365, **{f'C_{m}': cur[m] for m in cur}}).to_csv(DATA / f'{NAME}_matched_capacity.csv', index=False)


if __name__ == '__main__':
    q = split(pd.read_csv(OUT / 'surfaces' / f'{SYM}.csv'))
    bs, par = calibrated(q)
    K, tau = fig_C_vs_S(q, bs, par); print('C vs S done')
    fig_calendar(q, bs, par, K); print('calendar done')
    fig_matched(q, K, tau); print('matched done')
