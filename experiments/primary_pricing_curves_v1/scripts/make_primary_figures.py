"""PRIMARY price-curve figures: C vs S and C vs remaining maturity tau.

Everything is computed from the pricing equations through the frozen pricers. No curve is
shifted, rescaled or reordered. The maturity ordering is CHECKED, never imposed.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(HERE))
import curvelib as L
import common as CC

FIG, DATA = OUT / 'figures', OUT / 'data'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

S  = L.S_grid()
TD = L.TAU_DAYS
FS = CC.SCENARIOS['FAST_HEAVY']
NOTE = (f'CONTROLLED TWO-TIMESCALE BENCHMARK  ·  $r=q=0$, $K=100$\n'
        f'fast-heavy state $v_f$={FS["v_fast"]}, $v_s$={FS["v_slow"]} (total 0.04 -> 20% instantaneous)'
        f'  ·  $\\kappa_f$={CC.KAPPA_FAST:.2f}, $\\kappa_s$={CC.KAPPA_SLOW:.2f}')
NOTE_BS = 'ANALYTIC BLACK-SCHOLES REFERENCE  ·  $r=q=0$, $K=100$, constant $\\sigma$ = 20%\ncomputed directly from the Black-Scholes formula, not inferred from implied-volatility data'
VAL, ORD, ROWS = [], [], []


def finish(fig, name, note=NOTE, rect=.935, y=.962):
    if note: fig.text(.5, y, note, ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=[0, 0, 1, rect]); fig.savefig(FIG / f'{name}.png'); plt.close(fig)
    print('  wrote', name)


def family(ax, model, scenario=CC.PRIMARY, show_payoff=True):
    """One C(S) curve per predeclared maturity, computed from the model, then validated."""
    cur = {}
    for c, td in zip(L.MAT_CMAP, TD):
        tau = np.full_like(S, td / 365.)
        C = L.price(model, S, tau, scenario)
        cur[td] = C
        ax.plot(S, C, color=c, lw=1.9, label=fr'$\tau$ = {td:g} d')
        VAL.append({'scenario': scenario, 'model': model, **L.validate(S, C, tau, f'{model} tau={td:g}d')})
    if show_payoff:
        ax.plot(S, L.payoff(S), **{k: v for k, v in L.STYLE['PAYOFF'].items()})
    ax.set_xlabel('underlying price  $S$        ($K=100$)'); ax.set_ylabel('call price  $C$')
    # maturity ordering: verified, not imposed
    for a, b in zip(TD[:-1], TD[1:]):
        d = cur[b] - cur[a]; ok = np.isfinite(d)
        ORD.append({'scenario': scenario, 'model': model, 'tau_short_d': a, 'tau_long_d': b,
                    'points': int(ok.sum()), 'min_gap': float(d[ok].min()), 'mean_gap': float(d[ok].mean()),
                    'monotone_increasing_in_tau': bool((d[ok] >= -1e-9).all())})
    return cur


# ---------------------------------------------------------------- FIGURE 1  analytic Black-Scholes
def fig01():
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    family(ax, 'BS20')
    ax.set_title('FIGURE 1   Black-Scholes call price vs underlying price\n'
                 'analytic formula, constant $\\sigma=20\\%$, $r=q=0$')
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.2)
    finish(fig, '01_BS_C_vs_S_multi_maturity', note=NOTE_BS)


# ---------------------------------------------------------------- FIGURE 2  BS-PINN vs analytic BS
def fig02():
    m = L.bs_pinn_meta()
    lo, hi = max(L.S_LO, np.exp(m['x_min'])) * L.K, min(L.S_HI, np.exp(m['x_max'])) * L.K
    Sb = np.linspace(lo, hi, 601)
    tds = [t for t in TD if t / 365. <= m['tau_max'] + 1e-9]
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 7.8), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    for c, td in zip(L.MAT_CMAP, tds):
        tau = np.full_like(Sb, td / 365.)
        ex, pn = L.bs_analytic_rq(Sb, tau), L.bs_pinn_price(Sb, tau)
        a.plot(Sb, ex, color=c, lw=1.9, label=fr'analytic BS, $\tau$={td:g} d')
        a.plot(Sb, pn, color=c, lw=1.6, ls='--', dashes=(4, 3), alpha=.95)
        b.plot(Sb, pn - ex, color=c, lw=1.5, label=fr'$\tau$={td:g} d')
        e = pn - ex
        ROWS.append({'figure': '02_BS_PINN', 'model': 'BS_PINN', 'tau_days': td,
                     'RMSE': float(np.sqrt(np.mean(e ** 2))), 'max_abs': float(np.abs(e).max())})
        VAL.append({'scenario': 'BS_PINN_own_convention', 'model': 'BS_PINN',
                    **L.validate(Sb, pn, tau, f'BS_PINN tau={td:g}d')})
    a.plot([], [], color='k', lw=1.6, ls='--', label='Black-Scholes PINN (dashed)')
    a.set_ylabel('call price  $C$')
    a.set_title('FIGURE 2   Black-Scholes PINN vs analytic Black-Scholes\n'
                'the PINN is shown on ITS OWN trained convention, not on the $r=q=0$ benchmark')
    a.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.4)
    b.axhline(0, color='k', lw=1.1)
    b.set_xlabel('underlying price  $S$        ($K=100$)'); b.set_ylabel('PINN $-$ analytic')
    b.legend(frameon=False, ncol=3, handlelength=2.2)
    finish(fig, '01b_BS_PINN_vs_BS_C_vs_S',
           note=(f'BLACK-SCHOLES PINN OWN CONVENTION  ·  $r$={m["rate"]}, $q$={m["dividend"]}, '
                 f'calibrated $\\sigma$={m["sigma"]:.6f}\ntrained domain $S/K\\in$'
                 f'[{np.exp(m["x_min"]):.3f}, {np.exp(m["x_max"]):.3f}], $\\tau\\leq${m["tau_max"]:g} y'))


# ---------------------------------------------------------------- FIGURES 3, 4
def fig03():
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    family(ax, 'SH')
    ax.set_title('FIGURE 3   Single Heston call price vs underlying price\n'
                 'exact pricer, frozen strongest Single-Heston fit, $r=q=0$')
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.2)
    finish(fig, '02_SH_C_vs_S_multi_maturity')


def fig04():
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    family(ax, 'DH')
    ax.set_title('FIGURE 4a   Double Heston call price vs underlying price\n'
                 'exact two-factor pricer, frozen published parameters, $r=q=0$')
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.2)
    finish(fig, '03_DH_C_vs_S_multi_maturity')


def fig05():
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    for c, td in zip(L.MAT_CMAP, TD):
        tau = np.full_like(S, td / 365.)
        dh, pn = L.price('DH', S, tau), L.price('PINN', S, tau)
        a.plot(S, dh, color=c, lw=2.0, label=fr'exact DH, $\tau$={td:g} d')
        a.plot(S, pn, color=c, lw=1.5, ls='--', dashes=(4, 3))
        e = pn - dh; ok = np.isfinite(e)
        b.plot(S, e, color=c, lw=1.4, label=fr'$\tau$={td:g} d')
        VAL.append({'scenario': CC.PRIMARY, 'model': 'PINN', **L.validate(S, pn, tau, f'PINN tau={td:g}d')})
        ROWS.append({'figure': '04_DH_PINN', 'model': 'PINN', 'tau_days': td,
                     'RMSE': float(np.sqrt(np.mean(e[ok] ** 2))), 'max_abs': float(np.abs(e[ok]).max())})
    a.plot(S, L.payoff(S), **L.STYLE['PAYOFF'])
    a.plot([], [], color='k', lw=1.5, ls='--', label='Double Heston PINN (dashed)')
    a.set_ylabel('call price  $C$')
    a.set_title('FIGURE 4b   Double Heston PINN vs exact Double Heston\n'
                'same $S$ grid, same maturities; PINN shown only inside its trained domain '
                '$|\\log(S/K)|\\leq0.36$')
    a.legend(frameon=False, ncol=2, loc='upper left', handlelength=2.4)
    b.axhline(0, color='k', lw=1.1)
    b.set_xlabel('underlying price  $S$        ($K=100$)'); b.set_ylabel('PINN $-$ exact DH')
    b.legend(frameon=False, ncol=3, handlelength=2.2)
    finish(fig, '04_DH_PINN_vs_DH_C_vs_S_multi_maturity')


# ---------------------------------------------------------------- FIGURE 5  model comparison
def fig06():
    tds = [30., 90., 365., 730.]
    fig, axs = plt.subplots(2, 4, figsize=(17.0, 8.2), sharex=True,
                            gridspec_kw={'height_ratios': [2.1, 1]})
    for j, td in enumerate(tds):
        tau = np.full_like(S, td / 365.)
        P = {m: L.price(m, S, tau) for m in L.MODELS}
        a, b = axs[0, j], axs[1, j]
        for m in L.MODELS:
            a.plot(S, P[m], **L.STYLE[m])
        a.plot(S, L.payoff(S), **L.STYLE['PAYOFF'])
        a.set_title(fr'$\tau$ = {td:g} days')
        for m in ['BS20', 'BS_TERM', 'SH', 'PINN']:
            e = P[m] - P['DH']; ok = np.isfinite(e)
            b.plot(S, e, **{**L.STYLE[m], 'label': L.STYLE[m]['label']})
            ROWS.append({'figure': '05_comparison', 'model': m, 'tau_days': td,
                         'RMSE': float(np.sqrt(np.mean(e[ok] ** 2))), 'max_abs': float(np.abs(e[ok]).max())})
            VAL.append({'scenario': CC.PRIMARY, 'model': m, **L.validate(S, P[m], tau, f'{m} cmp tau={td:g}d')})
        b.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
        b.set_xlabel('underlying price  $S$')
    axs[0, 0].set_ylabel('call price  $C$   (linear axis)')
    axs[1, 0].set_ylabel('model $-$ exact DH')
    axs[0, 0].legend(frameon=False, loc='upper left', handlelength=2.2, fontsize=8)
    axs[1, 0].legend(frameon=False, loc='best', handlelength=2.2, fontsize=7.5)
    fig.suptitle('FIGURE 5   Black-Scholes vs Single Heston vs exact Double Heston vs DH-PINN   ·   '
                 'call price vs underlying price, with residuals against exact Double Heston', y=.995, fontsize=12)
    finish(fig, '05_BS_SH_DH_C_vs_S_comparison', rect=.90, y=.925)


# ---------------------------------------------------------------- FIGURE 6  C vs tau, ATM
TAUD = np.linspace(7., 730., 601)


def fig07():
    tau = TAUD / 365.; Sa = np.full_like(tau, L.K)
    P = {m: L.price(m, Sa, tau) for m in L.MODELS}
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    for m in L.MODELS: a.plot(TAUD, P[m], **L.STYLE[m])
    a.set_ylabel('call price  $C$')
    a.set_title('FIGURE 6   Call price vs REMAINING time to maturity $\\tau$\n'
                'at the money, $S=K=100$   ·   this is price, not implied volatility')
    a.legend(frameon=False, loc='upper left', handlelength=2.4)
    for m in ['BS20', 'BS_TERM', 'SH', 'PINN']:
        e = P[m] - P['DH']; b.plot(TAUD, e, **L.STYLE[m])
        ok = np.isfinite(e)
        ROWS.append({'figure': '06_C_vs_tau_ATM', 'model': m, 'tau_days': np.nan,
                     'RMSE': float(np.sqrt(np.mean(e[ok] ** 2))), 'max_abs': float(np.abs(e[ok]).max())})
    b.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel('remaining time to maturity  $\\tau$  (days)   ->   more remaining optionality')
    b.set_ylabel('model $-$ exact DH')
    b.legend(frameon=False, ncol=2, handlelength=2.2)
    finish(fig, '06_C_vs_tau_ATM')
    pd.DataFrame({'tau_days': TAUD, **{f'C_{m}': P[m] for m in L.MODELS}}).to_csv(DATA / '06_C_vs_tau_ATM.csv', index=False)
    # monotonicity of C in tau
    for m in L.MODELS:
        d = np.diff(P[m]); ok = np.isfinite(d)
        ORD.append({'scenario': CC.PRIMARY, 'model': m, 'tau_short_d': 7., 'tau_long_d': 730.,
                    'points': int(ok.sum()), 'min_gap': float(d[ok].min()), 'mean_gap': float(d[ok].mean()),
                    'monotone_increasing_in_tau': bool((d[ok] >= -1e-9).all())})


def fig08():
    mons = [0.80, 0.90, 1.00, 1.10, 1.20]
    tau = TAUD / 365.
    fig, axs = plt.subplots(1, 5, figsize=(19.0, 4.4), sharex=True)
    for ax, mo in zip(axs, mons):
        Sm = np.full_like(tau, mo * L.K)
        for m in L.MODELS: ax.plot(TAUD, L.price(m, Sm, tau), **L.STYLE[m])
        ax.axhline(max(mo * L.K - L.K, 0.), color=L.STYLE['PAYOFF']['color'], ls='-.', lw=1.2,
                   label='payoff $\\max(S-K,0)$')
        ax.set_title(f'$S/K$ = {mo:.2f}'); ax.set_xlabel('remaining maturity $\\tau$ (days)')
    axs[0].set_ylabel('call price  $C$')
    axs[0].legend(frameon=False, loc='upper left', handlelength=2.2, fontsize=7.5)
    fig.suptitle('FIGURE 6b   Call price vs remaining time to maturity at fixed underlying price', y=.995)
    finish(fig, '07_C_vs_tau_multiple_moneyness', rect=.90, y=.925)


# ---------------------------------------------------------------- FIGURES 7, 8  fast vs slow
def fig09():
    fig, axs = plt.subplots(1, 3, figsize=(17.4, 5.4))
    for ax, sc, ttl in zip(axs[:2], ['FAST_HEAVY', 'SLOW_HEAVY'],
                           ['fast-heavy  $v_f$=0.035, $v_s$=0.005', 'slow-heavy  $v_f$=0.005, $v_s$=0.035']):
        family(ax, 'DH', scenario=sc)
        ax.set_title(f'exact Double Heston   ·   {ttl}')
    axs[0].legend(frameon=False, ncol=2, loc='upper left', handlelength=2.2)
    for c, td in zip(L.MAT_CMAP, TD):
        tau = np.full_like(S, td / 365.)
        d = L.price('DH', S, tau, 'FAST_HEAVY') - L.price('DH', S, tau, 'SLOW_HEAVY')
        axs[2].plot(S, d, color=c, lw=1.8, label=fr'$\tau$={td:g} d')
    axs[2].axhline(0, color='k', lw=1.1)
    axs[2].set_title('fast-heavy $-$ slow-heavy\nsame 20% instantaneous volatility, different price surface')
    axs[2].set_xlabel('underlying price  $S$'); axs[2].set_ylabel('price difference')
    axs[2].legend(frameon=False, ncol=2, handlelength=2.2)
    finish(fig, '08_fast_vs_slow_C_vs_S', rect=.90, y=.952,
           note=('CONTROLLED TWO-TIMESCALE BENCHMARK  ·  $r=q=0$, $K=100$  ·  BOTH frozen '
                 'FIXED_TOTAL_TWIST states, $v_f+v_s$=0.04 in each\n'
                 f'$\\kappa_f$={CC.KAPPA_FAST:.2f}, $\\kappa_s$={CC.KAPPA_SLOW:.2f} '
                 f'(ratio {CC.KAPPA_FAST/CC.KAPPA_SLOW:.1f}x)  ·  exact Double Heston'))


def fig10():
    tau = TAUD / 365.; Sa = np.full_like(tau, L.K)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    cols = {'FAST_HEAVY': '#c0392b', 'SLOW_HEAVY': '#1f3f8a'}
    out = {'tau_days': TAUD}
    for sc in ['FAST_HEAVY', 'SLOW_HEAVY']:
        dh, pn = L.price('DH', Sa, tau, sc), L.price('PINN', Sa, tau, sc)
        lab = 'fast-heavy' if sc == 'FAST_HEAVY' else 'slow-heavy'
        a.plot(TAUD, dh, color=cols[sc], lw=2.1, label=f'exact DH, {lab}')
        a.plot(TAUD, pn, color=cols[sc], lw=1.5, ls='--', dashes=(4, 3), label=f'DH-PINN, {lab}')
        b.plot(TAUD, pn - dh, color=cols[sc], lw=1.5, label=f'PINN $-$ exact, {lab}')
        out[f'C_DH_{sc}'] = dh; out[f'C_PINN_{sc}'] = pn
    a.set_ylabel('call price  $C$')
    a.set_title('FIGURE 8   Fast-heavy vs slow-heavy call price vs remaining maturity\n'
                'at the money, $S=K=100$   ·   actual option price, not implied volatility')
    a.legend(frameon=False, loc='upper left', handlelength=2.4)
    b.axhline(0, color='k', lw=1.1); b.legend(frameon=False, handlelength=2.2)
    b.set_xlabel('remaining time to maturity  $\\tau$  (days)'); b.set_ylabel('PINN $-$ exact DH')
    finish(fig, '09_fast_vs_slow_C_vs_tau',
           note=('CONTROLLED TWO-TIMESCALE BENCHMARK  ·  $r=q=0$, $K=100$  ·  BOTH frozen '
                 'FIXED_TOTAL_TWIST states, $v_f+v_s$=0.04 in each\n'
                 f'$\\kappa_f$={CC.KAPPA_FAST:.2f}, $\\kappa_s$={CC.KAPPA_SLOW:.2f}'))
    pd.DataFrame(out).to_csv(DATA / '09_fast_vs_slow_C_vs_tau.csv', index=False)


# ---------------------------------------------------------------- separate calendar-time decay
def fig11():
    T = 365.
    t = np.linspace(0., T - 7., 601)                      # calendar time elapsed
    tau = (T - t) / 365.                                  # remaining maturity
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    for mo, c in zip([0.90, 1.00, 1.10], ['#00a9d6', '#00d26a', '#b34bd0']):
        Sm = np.full_like(t, mo * L.K)
        ax.plot(t, L.price('DH', Sm, tau), color=c, lw=2.0, label=f'exact DH, $S/K$={mo:.2f}')
    ax.set_xlabel('calendar time $t$ elapsed since inception (days)   ->   toward expiry at $T$=365 d')
    ax.set_ylabel('call price  $C$')
    ax.set_title('SUPPORTING   Call price vs CALENDAR TIME progressing toward expiry\n'
                 'this is NOT $C$ vs $\\tau$: here $\\tau=T-t$ shrinks as $t$ grows')
    ax.legend(frameon=False, handlelength=2.4)
    finish(fig, '10_C_vs_calendar_time_decay')


# ---------------------------------------------------------------- numerical verification
def verify():
    rows = []
    # (a) analytic BS against independent quadrature of the risk-neutral expectation
    for td in TD:
        t = td / 365.
        for mo in [0.7, 0.85, 1.0, 1.15, 1.3]:
            S0 = mo * L.K
            f = lambda z: max(S0 * np.exp(-0.5 * L.SIGMA_CONST ** 2 * t + L.SIGMA_CONST * np.sqrt(t) * z) - L.K, 0.) \
                          * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
            num = quad(f, -12, 12, limit=500)[0]
            cf = float(L.bs_const(np.array([S0]), np.array([t]))[0])
            rows.append({'check': 'BS closed form vs quadrature', 'tau_days': td, 'S_over_K': mo,
                         'closed_form': cf, 'independent_numerical': num, 'abs_diff': abs(cf - num)})
    pd.DataFrame(rows).to_csv(DATA / 'bs_analytic_verification.csv', index=False)
    print('  BS closed-form vs quadrature: max abs diff =',
          max(r['abs_diff'] for r in rows))

    # (b) tau -> 0 convergence to the payoff (exact models only; the PINN is not trained below 7 d)
    z = []
    for td in [7., 3., 1., 0.25, 0.05, 0.01]:
        tau = np.full_like(S, td / 365.)
        for m in ['BS20', 'SH', 'DH']:
            try:
                e = L.price(m, S, tau) - L.payoff(S)
            except FloatingPointError:
                # The characteristic-function quadrature declares itself unreliable at this maturity.
                # Recorded, not silently skipped: it is a limit of the Fourier pricer, not of the model.
                z.append({'model': m, 'tau_days': td, 'status': 'quadrature declared unreliable',
                          'max_abs_minus_payoff': np.nan, 'mean_abs_minus_payoff': np.nan,
                          'min_time_value': np.nan})
                continue
            z.append({'model': m, 'tau_days': td, 'status': 'ok',
                      'max_abs_minus_payoff': float(np.abs(e).max()),
                      'mean_abs_minus_payoff': float(np.abs(e).mean()), 'min_time_value': float(e.min())})
    pd.DataFrame(z).to_csv(DATA / 'tau_to_zero_convergence.csv', index=False)
    for m in ['BS20', 'SH', 'DH']:
        ok = [r for r in z if r['model'] == m and r['status'] == 'ok']
        if ok:
            fin = min(ok, key=lambda r: r['tau_days'])
            print(f"  tau->0 {m}: smallest reliable tau = {fin['tau_days']:g} d, "
                  f"max |C - payoff| = {fin['max_abs_minus_payoff']:.3e}")
        bad = [r['tau_days'] for r in z if r['model'] == m and r['status'] != 'ok']
        if bad: print(f"    ({m} quadrature unreliable at tau = {bad} days -- pricer limit, recorded)")


def main():
    for f in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11):
        f()
    verify()
    v = pd.DataFrame(VAL); v.to_csv(DATA / 'financial_validation_C_vs_S.csv', index=False)
    o = pd.DataFrame(ORD); o.to_csv(DATA / 'maturity_ordering_check.csv', index=False)
    r = pd.DataFrame(ROWS); r.to_csv(DATA / 'residual_summary.csv', index=False)
    print('\nVIOLATION TOTALS over', len(v), 'model x maturity curves')
    print('  delta < 0 :', int(v['delta_violations'].sum()))
    print('  gamma < 0 :', int(v['gamma_violations'].sum()))
    print('  C < max(S-K,0) :', int(v['below_intrinsic'].sum()))
    print('  C > S :', int(v['above_spot'].sum()))
    print('  worst (most negative) time value :', float(v['min_time_value'].min()))
    print('  worst (most negative) gamma :', float(v['gamma_min'].min()))
    print('\nMATURITY ORDERING (verified, not imposed)')
    print(o.groupby('model')['monotone_increasing_in_tau'].agg(['sum', 'count']).to_string())
    bad = o[~o['monotone_increasing_in_tau']]
    if len(bad): print('  non-monotone cases:\n', bad.to_string(index=False))
    print('\nRESIDUALS vs exact DH (price units)')
    print(r.groupby(['figure', 'model'])[['RMSE', 'max_abs']].mean().to_string())


if __name__ == '__main__':
    main()
