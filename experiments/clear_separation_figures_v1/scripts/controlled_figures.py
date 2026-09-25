"""CONTROLLED TWO-TIMESCALE BENCHMARK figures: time value, IV smile, term structure, decay.

Reference for every residual panel is exact Double Heston. Nothing is shifted or rescaled.
The Black-Scholes PINN is absent here on purpose: it carries r = 0.03, q = 0.01 and its own
calibrated sigma, so placing it on this r = q = 0 benchmark would compare it to the wrong target.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(HERE))
import pricers as P
import common as CC

FIG, DATA = OUT / 'figures', OUT / 'data'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
TAG = 'CONTROLLED TWO-TIMESCALE BENCHMARK'
NOTE = (f'{TAG}  ·  fast-heavy state $v_f$={CC.SCENARIOS[CC.PRIMARY]["v_fast"]}, $v_s$={CC.SCENARIOS[CC.PRIMARY]["v_slow"]}'
        f'  ·  $\\kappa_f$={CC.KAPPA_FAST:.2f}, $\\kappa_s$={CC.KAPPA_SLOW:.2f} (ratio {CC.KAPPA_FAST/CC.KAPPA_SLOW:.1f}$\\times$)')
M = P.CONTROLLED_MODELS
CHECKS, ROWS = {}, []


def two_panel(figsize=(8.4, 7.8)):
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    return fig, a, b


def finish(fig, name, note=NOTE, rect=.945, y=.955):
    fig.text(.5, y, note, ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, rect]); fig.savefig(FIG / f'{name}.png'); plt.close(fig)


def save(df, name): df.to_csv(DATA / f'{name}.csv', index=False)


# ------------------------------------------------------------------ A. time value vs S
def time_value_panel(td, lo=.70, hi=1.30, name=None, zoom=False):
    S = np.linspace(lo * P.K, hi * P.K, 501); tau = td / 365.
    tv = {m: P.time_value(m, S, np.full_like(S, tau)) for m in M}
    for m in M: CHECKS[f'time_value tau={td:g}d:{m}'] = P.shape_checks(S, P.price(m, S, np.full_like(S, tau)))
    fig, a, b = two_panel()
    for m in M: a.plot(S, tv[m], **P.STYLE[m])
    a.set_ylabel('time value  $C(S,\\tau)-\\max(S-K,0)$')
    a.set_title(f'Time value vs underlying price   ·   $\\tau$ = {td:g} days' + ('   ·   at-the-money zoom' if zoom else ''))
    a.legend(frameon=False, loc='upper right' if not zoom else 'lower center', handlelength=2.4)
    for m in ['BS', 'SH', 'PINN']:
        b.plot(S, tv[m] - tv['DH'], **{**P.STYLE[m], 'label': P.STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=P.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel('underlying price $S$   ($K=100$)'); b.set_ylabel('time value $-$ exact DH')
    b.legend(frameon=False, loc='best', handlelength=2.4)
    nm = name or f'time_value_vs_S_{td:g}d'
    finish(fig, nm)
    d = pd.DataFrame({'S': S, 'tau_days': td, **{f'TV_{m}': tv[m] for m in M}})
    for m in ['BS', 'SH', 'PINN']: d[f'{m}_minus_DH'] = tv[m] - tv['DH']
    save(d, nm)
    for m in M:
        e = tv[m] - tv['DH']
        ROWS.append({'figure': nm, 'setting': 'controlled', 'quantity': 'time value', 'tau_days': td, 'model': m,
                     'RMSE_vs_DH': float(np.sqrt(np.mean(e ** 2))), 'max_abs_vs_DH': float(np.abs(e).max()),
                     'P95_abs_vs_DH': float(np.quantile(np.abs(e), .95))})
    return S, tv


def multipanel():
    tds = [30., 90., 180., 365.]
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 8.4), sharex=True)
    S = np.linspace(.70 * P.K, 1.30 * P.K, 501)
    for ax, td in zip(axs.ravel(), tds):
        for m in M: ax.plot(S, P.time_value(m, S, np.full_like(S, td / 365.)), **P.STYLE[m])
        ax.set_title(fr'$\tau$ = {td:g} days')
    for ax in axs[1]: ax.set_xlabel('underlying price $S$   ($K=100$)')
    for ax in axs[:, 0]: ax.set_ylabel('time value')
    axs[0, 0].legend(frameon=False, loc='upper right', handlelength=2.4)
    fig.suptitle('Time value vs underlying price', y=.985)
    finish(fig, 'controlled_time_value_vs_S_multipanel', rect=.925, y=.945)
    fig.clf()


# ------------------------------------------------------------------ B. implied-volatility smile
def smile_panel(td, lo=.70, hi=1.30, name=None, zoom=False):
    S = np.linspace(lo * P.K, hi * P.K, 401); tau = np.full_like(S, td / 365.); x = np.log(S / P.K)
    iv = {m: 100 * P.implied_vol(m, S, tau) for m in M}
    fig, a, b = two_panel()
    for m in M: a.plot(x, iv[m], **P.STYLE[m])
    a.set_ylabel('implied volatility (%)')
    a.set_title(f'Implied-volatility smile   ·   $\\tau$ = {td:g} days' + ('   ·   at-the-money zoom' if zoom else ''))
    a.legend(frameon=False, loc='best', handlelength=2.4)
    for m in ['BS', 'SH', 'PINN']:
        b.plot(x, iv[m] - iv['DH'], **{**P.STYLE[m], 'label': P.STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=P.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel(r'log-moneyness $x=\log(S/K)$'); b.set_ylabel('IV $-$ exact DH (vol pts)')
    b.legend(frameon=False, loc='best', handlelength=2.4)
    nm = name or f'iv_smile_{"short" if td <= 45 else "medium" if td <= 180 else "long"}'
    finish(fig, nm)
    d = pd.DataFrame({'S': S, 'x': x, 'tau_days': td, **{f'IV_{m}': iv[m] for m in M}})
    for m in ['BS', 'SH', 'PINN']: d[f'{m}_minus_DH'] = iv[m] - iv['DH']
    save(d, nm)
    for m in M:
        e = iv[m] - iv['DH']; ok = np.isfinite(e)
        ROWS.append({'figure': nm, 'setting': 'controlled', 'quantity': 'implied vol (vol pts)', 'tau_days': td, 'model': m,
                     'RMSE_vs_DH': float(np.sqrt(np.nanmean(e[ok] ** 2))) if ok.any() else np.nan,
                     'max_abs_vs_DH': float(np.abs(e[ok]).max()) if ok.any() else np.nan,
                     'P95_abs_vs_DH': float(np.quantile(np.abs(e[ok]), .95)) if ok.any() else np.nan,
                     'points_without_invertible_IV': int((~np.isfinite(iv[m])).sum())})


def smile_triple():
    tds = [30., 90., 365.]
    S = np.linspace(.70 * P.K, 1.30 * P.K, 401); x = np.log(S / P.K)
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.6), sharey=True)
    for ax, td in zip(axs, tds):
        for m in M: ax.plot(x, 100 * P.implied_vol(m, S, np.full_like(S, td / 365.)), **P.STYLE[m])
        ax.set_title(fr'$\tau$ = {td:g} days'); ax.set_xlabel(r'$x=\log(S/K)$')
    axs[0].set_ylabel('implied volatility (%)'); axs[0].legend(frameon=False, loc='best', handlelength=2.4)
    fig.suptitle('Implied-volatility smile: short, medium and long maturity', y=.98)
    finish(fig, 'controlled_iv_smile_short_medium_long', rect=.90, y=.915)


# ------------------------------------------------------------------ C. ATM IV term structure
def atm_term_structure(lo=7., hi=730., name='controlled_atm_iv_term_structure', zoom=False):
    td = np.linspace(lo, hi, 400)
    iv = {m: 100 * P.atm_iv(m, td / 365.) for m in M}
    fig, a, b = two_panel()
    for m in M: a.plot(td, iv[m], **P.STYLE[m])
    a.set_ylabel('at-the-money implied volatility (%)')
    a.set_title('At-the-money implied-volatility term structure' + ('   ·   short-maturity zoom' if zoom else ''))
    a.legend(frameon=False, loc='best', handlelength=2.4)
    for m in ['BS', 'SH', 'PINN']:
        b.plot(td, iv[m] - iv['DH'], **{**P.STYLE[m], 'label': P.STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=P.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel(r'remaining maturity $\tau$ (days)'); b.set_ylabel('IV $-$ exact DH (vol pts)')
    b.legend(frameon=False, loc='best', handlelength=2.4)
    finish(fig, name)
    d = pd.DataFrame({'tau_days': td, **{f'ATM_IV_{m}': iv[m] for m in M}})
    for m in ['BS', 'SH', 'PINN']: d[f'{m}_minus_DH'] = iv[m] - iv['DH']
    save(d, name)
    for m in M:
        e = iv[m] - iv['DH']
        ROWS.append({'figure': name, 'setting': 'controlled', 'quantity': 'ATM IV (vol pts)', 'tau_days': np.nan, 'model': m,
                     'RMSE_vs_DH': float(np.sqrt(np.mean(e ** 2))), 'max_abs_vs_DH': float(np.abs(e).max()),
                     'P95_abs_vs_DH': float(np.quantile(np.abs(e), .95))})


# ------------------------------------------------------------------ D. ATM time value vs maturity
def atm_time_value():
    td = np.linspace(7., 730., 400); S = np.full_like(td, P.K)
    tv = {m: P.time_value(m, S, td / 365.) for m in M}
    fig, a, b = two_panel()
    for m in M: a.plot(td, tv[m], **P.STYLE[m])
    a.set_ylabel('at-the-money time value   ($S=K=100$)')
    a.set_title('At-the-money time value vs remaining maturity')
    a.legend(frameon=False, loc='lower right', handlelength=2.4)
    for m in ['BS', 'SH', 'PINN']:
        b.plot(td, tv[m] - tv['DH'], **{**P.STYLE[m], 'label': P.STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=P.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel(r'remaining maturity $\tau$ (days)'); b.set_ylabel('time value $-$ exact DH')
    b.legend(frameon=False, loc='best', handlelength=2.4)
    finish(fig, 'controlled_atm_time_value_vs_tau')
    d = pd.DataFrame({'tau_days': td, **{f'TV_{m}': tv[m] for m in M}})
    for m in ['BS', 'SH', 'PINN']: d[f'{m}_minus_DH'] = tv[m] - tv['DH']
    save(d, 'controlled_atm_time_value_vs_tau')
    for m in M:
        e = tv[m] - tv['DH']
        ROWS.append({'figure': 'controlled_atm_time_value_vs_tau', 'setting': 'controlled', 'quantity': 'ATM time value',
                     'tau_days': np.nan, 'model': m, 'RMSE_vs_DH': float(np.sqrt(np.mean(e ** 2))),
                     'max_abs_vs_DH': float(np.abs(e).max()), 'P95_abs_vs_DH': float(np.quantile(np.abs(e), .95))})


# ------------------------------------------------------------------ E. calendar-time decay of time value
def decay():
    T = 730.; t = np.linspace(0., T - 7., 400)
    fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True)
    rows = {'t_days': t, 'tau_days': T - t}
    for ax, r in zip(axs, [0.9, 1.0, 1.1]):
        S0 = r * P.K
        for m in M:
            tv = P.time_value(m, np.full_like(t, S0), (T - t) / 365.)
            ax.plot(t, tv, **P.STYLE[m]); rows[f'TV_{m}_S_over_K_{r}'] = tv
        ax.set_title(f'$S/K$ = {r:g}'); ax.set_xlabel('calendar time $t$ toward expiry (days)')
        ax.axhline(0, color='k', lw=.8, ls=':')
    axs[0].set_ylabel('time value'); axs[0].legend(frameon=False, loc='lower left', handlelength=2.4)
    fig.suptitle(r'Time-value decay as calendar time advances   ($\tau=T-t$, expiry at $t=T=730$ d)', y=.98)
    finish(fig, 'controlled_time_value_decay_to_expiry', rect=.90, y=.915)
    save(pd.DataFrame(rows), 'controlled_time_value_decay_to_expiry')


# ------------------------------------------------------------------ fast vs slow, same total variance
def fast_vs_slow():
    td = np.linspace(7., 730., 400); S = np.full_like(td, P.K)
    cols = {'FAST_HEAVY': '#b2182b', 'SLOW_HEAVY': '#2166ac'}
    out = {'tau_days': td}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.0, 5.2))
    for sc, col in cols.items():
        v = CC.SCENARIOS[sc]; lab = 'fast-heavy' if sc == 'FAST_HEAVY' else 'slow-heavy'
        tv = P.time_value('DH', S, td / 365., sc); pn = P.time_value('PINN', S, td / 365., sc)
        iv = 100 * P.atm_iv('DH', td / 365., sc); ivp = 100 * P.atm_iv('PINN', td / 365., sc)
        a.plot(td, tv, color=col, lw=2.2, label=f'exact DH, {lab}  ($v_f$={v["v_fast"]}, $v_s$={v["v_slow"]})')
        a.plot(td, pn, color=col, lw=1.5, ls='--', label=f'DH-PINN, {lab}')
        b.plot(td, iv, color=col, lw=2.2, label=f'exact DH, {lab}')
        b.plot(td, ivp, color=col, lw=1.5, ls='--', label=f'DH-PINN, {lab}')
        out[f'TV_DH_{sc}'] = tv; out[f'TV_PINN_{sc}'] = pn; out[f'ATM_IV_DH_{sc}'] = iv
    a.set_xlabel(r'remaining maturity $\tau$ (days)'); a.set_ylabel('at-the-money time value')
    a.set_title('Same instantaneous variance, different allocation'); a.legend(frameon=False, loc='lower right')
    b.axhline(100 * np.sqrt(.04), color='k', lw=.8, ls=':')
    b.text(td[0] + 10, 100 * np.sqrt(.04), r'$\sqrt{v_f+v_s}=20\%$', fontsize=8.5, va='bottom', color='#444')
    b.set_xlabel(r'remaining maturity $\tau$ (days)'); b.set_ylabel('at-the-money implied volatility (%)')
    b.set_title('Implied-volatility term structure'); b.legend(frameon=False, loc='center right')
    fig.suptitle(r'Fast versus slow variance at identical total variance  ($v_f+v_s=0.04$, both 20% instantaneous)', y=.98)
    finish(fig, 'fast_vs_slow_same_total_variance', rect=.90, y=.915)
    save(pd.DataFrame(out), 'fast_vs_slow_same_total_variance')
    # dedicated ATM IV term-structure version
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for sc, col in cols.items():
        lab = 'fast-heavy' if sc == 'FAST_HEAVY' else 'slow-heavy'
        ax.plot(td, 100 * P.atm_iv('DH', td / 365., sc), color=col, lw=2.2, label=f'exact DH, {lab}')
        ax.plot(td, 100 * P.atm_iv('PINN', td / 365., sc), color=col, lw=1.5, ls='--', label=f'DH-PINN, {lab}')
    ax.axhline(100 * np.sqrt(.04), color='k', lw=.8, ls=':')
    ax.set_xlabel(r'remaining maturity $\tau$ (days)'); ax.set_ylabel('at-the-money implied volatility (%)')
    ax.set_title('Fast-heavy versus slow-heavy: ATM implied-volatility term structure')
    ax.legend(frameon=False, loc='center right', handlelength=2.4)
    finish(fig, 'fast_vs_slow_atm_iv_term_structure', rect=.93, y=.94)


if __name__ == '__main__':
    for td in [30., 90., 180., 365.]: time_value_panel(td)
    multipanel()
    time_value_panel(90., .85, 1.15, 'controlled_time_value_vs_S_atm_zoom', zoom=True)
    time_value_panel(90., .85, 1.15, 'time_value_vs_S_atm_zoom', zoom=True)
    time_value_panel(90., name='controlled_time_value_vs_S')
    for td, nm in [(30., 'iv_smile_short'), (90., 'iv_smile_medium'), (365., 'iv_smile_long')]: smile_panel(td, name=nm)
    smile_panel(90., name='controlled_iv_smile')
    smile_panel(90., .85, 1.15, 'iv_smile_atm_zoom', zoom=True)
    smile_triple()
    atm_term_structure(); atm_term_structure(name='atm_iv_term_structure')
    atm_term_structure(7., 120., 'atm_term_structure_zoom', zoom=True)
    atm_time_value(); decay(); fast_vs_slow()
    (DATA / 'controlled_shape_checks.json').write_text(json.dumps(CHECKS, indent=2, default=float) + '\n')
    pd.DataFrame(ROWS).to_csv(DATA / 'controlled_metrics.csv', index=False)
    bad = {k: v for k, v in CHECKS.items() if v['delta_violations'] or v['gamma_violations'] or v['below_intrinsic'] or v['above_spot'] or v['negative_time_value']}
    print('controlled figures done; shape violations:', bad or 'none')
