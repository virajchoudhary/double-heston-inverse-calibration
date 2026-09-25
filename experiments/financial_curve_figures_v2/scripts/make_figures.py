"""All v2 presentation figures. Linear price axes, no offsets, no rescaling.

Every curve is a direct model output from common.price_S(). Residual panels show model minus
exact Double Heston; they never alter the price panels above them.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

OUT = Path(__file__).resolve().parents[1]
FIG, DATA = OUT / 'figures', OUT / 'data'
FIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
    'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'lines.solid_capstyle': 'round',
})
SUB = 'Controlled two-timescale stochastic-volatility benchmark'
ANNOT = (f'Fast factor  $\\kappa_f={C.KAPPA_FAST:.2f}$\nSlow factor  $\\kappa_s={C.KAPPA_SLOW:.2f}$\n'
         f'Ratio  ${C.KAPPA_FAST / C.KAPPA_SLOW:.1f}\\times$')
CHECKS = {}


def grid_S(n=401, lo=.70, hi=1.30):
    return np.linspace(lo * C.K, hi * C.K, n)


def curves_S(S, tau_days, scen=C.PRIMARY):
    tau = np.full_like(S, tau_days / 365.)
    return {m: C.price_S(m, scen, S, tau) for m in C.ORDER}


def record(tag, S, cur):
    for m, v in cur.items():
        CHECKS[f'{tag}:{m}'] = C.shape_checks(S, v)


def legend(ax, loc='upper left', payoff=True):
    ax.legend(frameon=False, loc=loc, handlelength=2.4)


# ------------------------------------------------------------------ Figure A: C vs S, panels by maturity
def fig_C_vs_S():
    S = grid_S(); rows = []
    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True)
    for ax, td in zip(axs.ravel(), C.PANEL_TAU_DAYS):
        cur = curves_S(S, td); record(f'C_vs_S tau={td:g}d', S, cur)
        for m in C.ORDER: ax.plot(S, cur[m], **C.STYLE[m])
        ax.plot(S, C.payoff(S), ls=':', lw=1.0, color='k', label=r'payoff $\max(S-K,0)$')
        ax.set_title(fr'$\tau$ = {td:g} days'); ax.set_xlim(S[0], S[-1])
        for m in C.ORDER: rows.append(pd.DataFrame({'tau_days': td, 'S': S, 'model': m, 'C': cur[m]}))
    for ax in axs[1]: ax.set_xlabel('underlying price $S$   (strike $K=100$)')
    for ax in axs[:, 0]: ax.set_ylabel('call price $C$')
    axs[1, 2].axis('off')
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[1, 2].legend(h, l, frameon=False, loc='center', handlelength=2.6)
    axs[1, 2].text(.5, .12, ANNOT, transform=axs[1, 2].transAxes, ha='center', va='center', fontsize=9, color='#444')
    fig.suptitle('European Call Price vs Underlying Price', y=.985)
    fig.text(.5, .945, SUB + f'   ·   fast-heavy state $v_f$={C.SCENARIOS[C.PRIMARY]["v_fast"]}, $v_s$={C.SCENARIOS[C.PRIMARY]["v_slow"]}, total 0.04',
             ha='center', fontsize=9.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .935]); fig.savefig(FIG / 'C_vs_S_controlled.png'); plt.close(fig)
    pd.concat(rows).pivot_table(index=['tau_days', 'S'], columns='model', values='C').reset_index().to_csv(DATA / 'C_vs_S_controlled.csv', index=False)


# ------------------------------------------------------------------ Figure A2 / A3: residual and ATM zoom
def fig_C_vs_S_residual(lo=.70, hi=1.30, name='C_vs_S_controlled_with_residual', zoom=False):
    S = grid_S(601, lo, hi); td = C.REPRESENTATIVE_TAU_DAYS
    cur = curves_S(S, td); record(f'{name}', S, cur)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    for m in C.ORDER: a.plot(S, cur[m], **C.STYLE[m])
    a.plot(S, C.payoff(S), ls=':', lw=1.0, color='k', label=r'payoff $\max(S-K,0)$')
    a.set_ylabel('call price $C$'); legend(a)
    a.set_title(f'European Call Price vs Underlying Price' + ('  ·  at-the-money zoom' if zoom else ''))
    a.text(.985, .06, ANNOT, transform=a.transAxes, ha='right', va='bottom', fontsize=8.5, color='#444')
    for m in ['BS', 'SH', 'PINN']:
        b.plot(S, cur[m] - cur['DH'], **{**C.STYLE[m], 'label': C.STYLE[m]['label'] + r' $-$ exact DH'})
    b.axhline(0, color=C.STYLE['DH']['color'], lw=1.4)
    b.set_xlabel('underlying price $S$   (strike $K=100$)'); b.set_ylabel(r'$C_{model}-C_{DH}$')
    b.legend(frameon=False, loc='upper right', handlelength=2.4); b.set_xlim(S[0], S[-1])
    fig.text(.5, .955, SUB + fr'   ·   $\tau$ = {td:g} days (predeclared)', ha='center', fontsize=9.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / f'{name}.png'); plt.close(fig)
    df = pd.DataFrame({'S': S, 'tau_days': td, **{f'C_{m}': cur[m] for m in C.ORDER}, 'payoff': C.payoff(S)})
    for m in ['BS', 'SH', 'PINN']: df[f'{m}_minus_DH'] = cur[m] - cur['DH']
    df.to_csv(DATA / f'{name}.csv', index=False)


# ------------------------------------------------------------------ Figure B: C vs remaining maturity
def fig_C_vs_tau(with_residual):
    td = np.linspace(C.PINN_TAU_MIN_DAYS, C.PINN_TAU_MAX_DAYS, 400)
    S = np.full_like(td, C.K)
    cur = {m: C.price_S(m, C.PRIMARY, S, td / 365.) for m in C.ORDER}
    name = 'C_vs_tau_controlled_with_residual' if with_residual else 'C_vs_tau_controlled'
    if with_residual:
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    else:
        fig, a = plt.subplots(figsize=(8.4, 5.6)); b = None
    for m in C.ORDER: a.plot(td, cur[m], **C.STYLE[m])
    a.set_ylabel('call price $C$   ($S=K=100$)'); legend(a, 'lower right')
    a.set_title('European Call Price vs Remaining Time to Maturity')
    a.text(.02, .96, ANNOT, transform=a.transAxes, ha='left', va='top', fontsize=8.5, color='#444')
    if b is None:
        a.set_xlabel(r'remaining time to maturity $\tau$   (days)')
    else:
        for m in ['BS', 'SH', 'PINN']:
            b.plot(td, cur[m] - cur['DH'], **{**C.STYLE[m], 'label': C.STYLE[m]['label'] + r' $-$ exact DH'})
        b.axhline(0, color=C.STYLE['DH']['color'], lw=1.4)
        b.set_xlabel(r'remaining time to maturity $\tau$   (days)'); b.set_ylabel(r'$C_{model}-C_{DH}$')
        b.legend(frameon=False, loc='upper left', handlelength=2.4)
    fig.text(.5, .955 if b is not None else .93, SUB, ha='center', fontsize=9.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945]); fig.savefig(FIG / f'{name}.png'); plt.close(fig)
    df = pd.DataFrame({'tau_days': td, 'S': C.K, **{f'C_{m}': cur[m] for m in C.ORDER}})
    for m in ['BS', 'SH', 'PINN']: df[f'{m}_minus_DH'] = cur[m] - cur['DH']
    df.to_csv(DATA / f'{name}.csv', index=False)
    d = np.diff(cur['DH'])
    CHECKS['C_vs_tau:DH_monotone_in_tau'] = {'violations': int((d < -1e-12).sum()), 'points': int(len(d))}


# ------------------------------------------------------------------ Figure C: calendar time toward expiry
def fig_calendar_time():
    T_days = 730.
    t = np.linspace(0., T_days - C.PINN_TAU_MIN_DAYS, 400)          # PINN valid while tau >= 7 days
    t_ext = np.linspace(0., T_days - 0.5, 500)                      # exact models can run closer to expiry
    S0 = 105.                                                       # slightly in the money so the limit is visible
    cur, cur_ext = {}, {}
    for m in C.ORDER:
        cur[m] = C.price_S(m, C.PRIMARY, np.full_like(t, S0), (T_days - t) / 365.)
        if m != 'PINN':
            cur_ext[m] = C.price_S(m, C.PRIMARY, np.full_like(t_ext, S0), (T_days - t_ext) / 365.)
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for m in C.ORDER:
        if m == 'PINN': ax.plot(t, cur[m], **C.STYLE[m])
        else: ax.plot(t_ext, cur_ext[m], **C.STYLE[m])
    ax.axhline(max(S0 - C.K, 0), color='k', ls=':', lw=1.0, label=r'payoff $\max(S-K,0)$')
    ax.axvline(T_days - C.PINN_TAU_MIN_DAYS, color='#16a085', lw=.8, ls=':')
    ax.text(T_days - C.PINN_TAU_MIN_DAYS - 12, ax.get_ylim()[1], 'PINN domain ends\n($\\tau=7$ d)  ', fontsize=8,
            color='#16a085', va='top', ha='right')
    ax.set_xlabel('calendar time $t$ progressing toward expiry   (days, expiry at $t=T=730$)')
    ax.set_ylabel(f'call price $C$   ($S={S0:g}$, $K=100$)')
    ax.set_title('European Call Value as Expiry Approaches')
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, .86), handlelength=2.4)
    ax.text(.03, .30, r'$\tau = T-t$', transform=ax.transAxes, fontsize=10, color='#444')
    fig.text(.5, .93, SUB, ha='center', fontsize=9.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .925]); fig.savefig(FIG / 'C_vs_calendar_time_to_expiry.png'); plt.close(fig)
    df = pd.DataFrame({'t_days': t_ext, 'tau_days': T_days - t_ext, 'S': S0,
                       **{f'C_{m}': cur_ext[m] for m in cur_ext}})
    df2 = pd.DataFrame({'t_days': t, 'tau_days': T_days - t, 'S': S0, 'C_PINN': cur['PINN']})
    df.merge(df2, on=['t_days', 'tau_days', 'S'], how='outer').sort_values('t_days').to_csv(DATA / 'C_vs_calendar_time_to_expiry.csv', index=False)
    CHECKS['calendar:terminal_gap_to_payoff'] = {m: float(abs(cur_ext[m][-1] - max(S0 - C.K, 0))) for m in cur_ext}


# ------------------------------------------------------------------ Figure: fast vs slow, same total variance
def fig_fast_vs_slow():
    td = np.linspace(C.PINN_TAU_MIN_DAYS, C.PINN_TAU_MAX_DAYS, 400); S = np.full_like(td, C.K)
    cols = {'FAST_HEAVY': '#b2182b', 'SLOW_HEAVY': '#2166ac'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    out = {'tau_days': td}
    for sc, col in cols.items():
        dh = C.price_S('DH', sc, S, td / 365.); pn = C.price_S('PINN', sc, S, td / 365.)
        lab = 'fast-heavy' if sc == 'FAST_HEAVY' else 'slow-heavy'
        v = C.SCENARIOS[sc]
        a.plot(td, dh, color=col, lw=2.2, label=f'exact DH, {lab}  ($v_f$={v["v_fast"]}, $v_s$={v["v_slow"]})')
        a.plot(td, pn, color=col, lw=1.6, ls='--', label=f'DH-PINN, {lab}')
        iv = np.sqrt(np.maximum(_implied_var(dh, td / 365.), 0)) * 100
        b.plot(td, iv, color=col, lw=2.2, label=f'exact DH, {lab}')
        out[f'C_DH_{sc}'] = dh; out[f'C_PINN_{sc}'] = pn; out[f'ATM_IV_pct_{sc}'] = iv
    a.set_xlabel(r'remaining time to maturity $\tau$   (days)'); a.set_ylabel('at-the-money call price $C$')
    a.set_title('Same instantaneous variance, different allocation'); a.legend(frameon=False, loc='lower right')
    b.set_xlabel(r'remaining time to maturity $\tau$   (days)'); b.set_ylabel('at-the-money implied volatility (%)')
    b.set_title('Implied-volatility term structure'); b.legend(frameon=False, loc='best')
    b.axhline(100 * np.sqrt(0.04), color='k', lw=.8, ls=':', label=None)
    b.text(td[0] + 8, 100 * np.sqrt(0.04), r'$\sqrt{v_f+v_s}=20\%$', fontsize=8.5, va='bottom', ha='left', color='#444')
    fig.suptitle('Fast versus slow variance at identical total variance  ($v_f+v_s=0.04$)', y=.98)
    fig.text(.5, .925, SUB + '   ·   ' + ANNOT.replace('\n', '   '), ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .915]); fig.savefig(FIG / 'fast_vs_slow_same_total_variance.png'); plt.close(fig)
    pd.DataFrame(out).to_csv(DATA / 'fast_vs_slow_same_total_variance.csv', index=False)


def _implied_var(C_atm, tau):
    """Total variance implied by an at-the-money call under Black-76 with F=K, inverted directly."""
    from scipy.stats import norm
    from scipy.optimize import brentq
    out = []
    for c, t in zip(np.atleast_1d(C_atm), np.atleast_1d(tau)):
        f = lambda s: C.K * (2 * norm.cdf(s * np.sqrt(t) / 2) - 1) - c
        try: out.append(brentq(f, 1e-6, 5.0) ** 2)
        except ValueError: out.append(np.nan)
    return np.array(out)


# ------------------------------------------------------------------ 3D surfaces
def surfaces():
    S = np.linspace(.70 * C.K, 1.30 * C.K, 70); td = np.geomspace(C.PINN_TAU_MIN_DAYS, C.PINN_TAU_MAX_DAYS, 70)
    SS, TT = np.meshgrid(S, td)
    Z = {m: C.price_S(m, C.PRIMARY, SS.ravel(), TT.ravel() / 365.).reshape(SS.shape) for m in C.ORDER}
    zlim = (0, max(z.max() for z in Z.values()) * 1.02)
    view = dict(elev=24, azim=-135)
    for m, fname in [('BS', 'BS_C_S_tau_3D'), ('SH', 'SH_C_S_tau_3D'), ('DH', 'DH_exact_C_S_tau_3D'), ('PINN', 'DH_PINN_C_S_tau_3D')]:
        fig = plt.figure(figsize=(7.4, 5.8)); ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(SS, TT, Z[m], cmap='viridis', linewidth=0, antialiased=True, vmin=zlim[0], vmax=zlim[1])
        ax.set_zlim(*zlim); ax.view_init(**view)
        ax.set_xlabel('$S$'); ax.set_ylabel(r'$\tau$ (days)'); ax.set_zlabel('$C$')
        ax.set_title(f'{C.STYLE[m]["label"]}   ·   $C(S,\\tau)$')
        fig.text(.5, .04, SUB, ha='center', fontsize=8.5, color='#555')
        fig.tight_layout(); fig.savefig(FIG / f'{fname}.png'); plt.close(fig)
    pairs = [('SH_minus_DH_surface', Z['SH'] - Z['DH'], r'$C_{SH}-C_{DH}$', 'Single Heston minus exact Double Heston', 'coolwarm'),
             ('PINN_minus_DH_error_surface', Z['PINN'] - Z['DH'], r'$C_{PINN}-C_{DH}$', 'DH-PINN minus exact Double Heston', 'coolwarm'),
             ('advantage_surface', np.abs(Z['SH'] - Z['DH']) - np.abs(Z['PINN'] - Z['DH']),
              r'$|C_{SH}-C_{DH}|-|C_{PINN}-C_{DH}|$', 'Where the PINN reproduces the two-factor target better than Single Heston', 'PiYG')]
    for fname, W, zlab, title, cmap in pairs:
        fig = plt.figure(figsize=(7.8, 5.8)); ax = fig.add_subplot(111, projection='3d')
        lim = float(np.abs(W).max())
        s3 = ax.plot_surface(SS, TT, W, cmap=cmap, linewidth=0, antialiased=True, vmin=-lim, vmax=lim)
        ax.view_init(**view); ax.set_xlabel('$S$'); ax.set_ylabel(r'$\tau$ (days)'); ax.set_zlabel(zlab)
        ax.set_title(title, fontsize=10.5); fig.colorbar(s3, ax=ax, shrink=.6, pad=.10)
        fig.text(.5, .04, SUB, ha='center', fontsize=8.5, color='#555')
        fig.tight_layout(); fig.savefig(FIG / f'{fname}.png'); plt.close(fig)
    df = pd.DataFrame({'S': SS.ravel(), 'tau_days': TT.ravel(), **{f'C_{m}': Z[m].ravel() for m in C.ORDER}})
    df['SH_minus_DH'] = df.C_SH - df.C_DH; df['BS_minus_DH'] = df.C_BS - df.C_DH; df['PINN_minus_DH'] = df.C_PINN - df.C_DH
    df['advantage'] = df.SH_minus_DH.abs() - df.PINN_minus_DH.abs()
    df.to_csv(DATA / 'surface_grid.csv', index=False)
    e = np.abs(Z['PINN'] - Z['DH']); j = np.unravel_index(np.argmax(e), e.shape)
    CHECKS['PINN_vs_DH_surface'] = {
        'max_abs': float(e.max()), 'rmse': float(np.sqrt(np.mean((Z['PINN'] - Z['DH']) ** 2))),
        'p95': float(np.quantile(e, .95)), 'at_S': float(SS[j]), 'at_tau_days': float(TT[j]),
        'max_abs_SH_minus_DH': float(np.abs(Z['SH'] - Z['DH']).max()),
        'max_abs_BS_minus_DH': float(np.abs(Z['BS'] - Z['DH']).max()),
        'advantage_positive_fraction': float((np.abs(Z['SH'] - Z['DH']) > e).mean()),
    }


if __name__ == '__main__':
    fig_C_vs_S(); print('A done', flush=True)
    fig_C_vs_S_residual(); print('A-residual done', flush=True)
    fig_C_vs_S_residual(.85, 1.15, 'C_vs_S_ATM_zoom', zoom=True); print('zoom done', flush=True)
    fig_C_vs_tau(False); fig_C_vs_tau(True); print('B done', flush=True)
    fig_calendar_time(); print('C done', flush=True)
    fig_fast_vs_slow(); print('fast/slow done', flush=True)
    surfaces(); print('surfaces done', flush=True)
    (OUT / 'data' / 'shape_checks.json').write_text(json.dumps(CHECKS, indent=2, default=float) + '\n')
    bad = {k: v for k, v in CHECKS.items() if isinstance(v, dict) and (v.get('delta_violations', 0) or v.get('gamma_violations', 0) or v.get('below_intrinsic', 0) or v.get('above_spot', 0))}
    print('\nshape-check failures:', json.dumps(bad, indent=1) if bad else 'none')
    print('PINN vs DH surface:', json.dumps(CHECKS['PINN_vs_DH_surface'], indent=1))
