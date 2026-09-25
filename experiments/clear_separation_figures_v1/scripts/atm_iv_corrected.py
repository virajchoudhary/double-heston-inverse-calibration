"""Corrected ATM implied-volatility term-structure figures.

Fixes a labelling defect: the grey curve previously labelled "Black-Scholes (exact)" is the frozen
BS_TERM baseline, one fitted volatility per maturity knot, which of course varies with maturity.
True constant-volatility Black-Scholes has a FLAT implied-volatility term structure, and is now
plotted and labelled separately.

Nothing is retrained, no model is redesigned, and no existing figure is overwritten: every output
here carries a new "_corrected" name.
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
from engine import black, iv as inv_iv

FIG, DATA = OUT / 'figures', OUT / 'data'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 8.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

# frozen constant-volatility baselines from nifty_multifactor_v4/artifacts/baselines/
SIG_FLAT = {s: float(CC.baseline(s)[3]['BS_FLAT']['volatility'][0]) for s in ['FAST_HEAVY', 'SLOW_HEAVY']}
SIG_INST = float(np.sqrt(CC.SCENARIOS[CC.PRIMARY]['v_fast'] + CC.SCENARIOS[CC.PRIMARY]['v_slow']))   # 0.20 exactly
STY = {
    'BS_FIX':  dict(color='#111111', ls='-',  lw=2.0, label=f'Black-Scholes, fixed $\\sigma$ = {100*SIG_FLAT[CC.PRIMARY]:.2f}%  (frozen BS_FLAT)'),
    'BS_INST': dict(color='#888888', ls=':',  lw=1.6, label=f'Black-Scholes, fixed $\\sigma$ = {100*SIG_INST:.0f}%  (instantaneous vol)'),
    'BS_TERM': dict(color='#7a7a7a', ls='-.', lw=1.9, label='Black-Scholes, maturity-dependent $\\sigma(\\tau)$  (one $\\sigma$ per expiry)'),
    'SH':      dict(color='#d9822b', ls='-',  lw=1.9, label='Single Heston'),
    'DH':      dict(color='#1f3f8a', ls='-',  lw=2.3, label='Double Heston (exact)'),
    'PINN':    dict(color='#16a085', ls='--', lw=2.1, label='Double Heston PINN'),
}
SUB = 'Controlled two-timescale benchmark'
NOTE = (f'$\\kappa_f$ = {CC.KAPPA_FAST:.2f},  $\\kappa_s$ = {CC.KAPPA_SLOW:.2f}   (ratio {CC.KAPPA_FAST/CC.KAPPA_SLOW:.1f}$\\times$)')


# ------------------------------------------------------------------ 3. mathematical validation
def validate_flat(td):
    """Generate ATM Black-Scholes prices at a fixed sigma, invert them, and check flatness."""
    rows, summary = [], {}
    for name, sig in [('frozen_BS_FLAT', SIG_FLAT[CC.PRIMARY]), ('sigma_0.20', SIG_INST)]:
        tau = td / 365.
        c = black(np.zeros_like(tau), tau, sig)                    # forward-normalised ATM call, r = q = 0, F = S = K
        rec = inv_iv(c, np.zeros_like(tau), tau)                   # invert the same prices
        dev = 100 * (rec - sig)                                    # deviation in volatility points
        summary[name] = {'sigma': sig, 'RMSE_volpts': float(np.sqrt(np.mean(dev ** 2))),
                         'MAE_volpts': float(np.mean(np.abs(dev))), 'max_abs_volpts': float(np.max(np.abs(dev))),
                         'points': int(len(tau))}
        rows.append(pd.DataFrame({'case': name, 'sigma': sig, 'tau_days': td, 'C_over_K': c * 100,
                                  'IV_recovered': rec, 'deviation_volpts': dev}))
    pd.concat(rows).to_csv(DATA / 'bs_fixed_sigma_validation.csv', index=False)
    (DATA / 'bs_fixed_sigma_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


# ------------------------------------------------------------------ 4 & 8. corrected term structure
def curves(td, scenario=CC.PRIMARY):
    tau = td / 365.
    out = {'BS_FIX': np.full_like(td, 100 * SIG_FLAT[scenario]),
           'BS_INST': np.full_like(td, 100 * SIG_INST),
           'BS_TERM': 100 * P.implied_vol('BS', np.full_like(td, P.K), tau, scenario),
           'SH': 100 * P.implied_vol('SH', np.full_like(td, P.K), tau, scenario),
           'DH': 100 * P.implied_vol('DH', np.full_like(td, P.K), tau, scenario),
           'PINN': 100 * P.implied_vol('PINN', np.full_like(td, P.K), tau, scenario)}
    return out


def fig_term_structure(with_residual):
    td = np.linspace(7., 730., 400); c = curves(td)
    order = ['BS_FIX', 'BS_INST', 'BS_TERM', 'SH', 'DH', 'PINN']
    if with_residual:
        fig, (a, b) = plt.subplots(2, 1, figsize=(8.6, 8.2), sharex=True, gridspec_kw={'height_ratios': [2.0, 1]})
    else:
        fig, a = plt.subplots(figsize=(8.6, 5.8)); b = None
    for m in order: a.plot(td, c[m], **STY[m])
    a.set_ylabel('ATM implied volatility (%)')
    a.set_title('ATM Implied Volatility vs Time to Maturity')
    a.legend(frameon=False, loc='lower right', handlelength=2.6)
    a.text(.015, .97, NOTE, transform=a.transAxes, fontsize=8.4, color='#444', va='top')
    if b is None:
        a.set_xlabel(r'remaining time to maturity $\tau$ (days)')
    else:
        RLAB = {'BS_FIX': 'BS fixed $\\sigma$=21.28%', 'BS_INST': 'BS fixed $\\sigma$=20%',
                'BS_TERM': 'BS $\\sigma(\\tau)$ per expiry', 'SH': 'Single Heston', 'PINN': 'DH-PINN'}
        for m in ['BS_FIX', 'BS_INST', 'BS_TERM', 'SH', 'PINN']:
            b.plot(td, c[m] - c['DH'], **{**STY[m], 'label': RLAB[m] + r' $-$ exact DH'})
        b.axhline(0, color=STY['DH']['color'], lw=1.5)
        b.set_xlabel(r'remaining time to maturity $\tau$ (days)')
        b.set_ylabel('IV $-$ exact DH (vol points)')
        b.legend(frameon=False, loc='lower right', ncol=2, handlelength=2.2, fontsize=8)
    fig.text(.5, .955 if b is not None else .93, SUB, ha='center', fontsize=9.3, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945 if b is not None else .92])
    name = 'atm_iv_term_structure_corrected' + ('_with_residual' if with_residual else '')
    fig.savefig(FIG / f'{name}.png'); fig.savefig(FIG / f'{name}.pdf'); plt.close(fig)
    if not with_residual:
        d = pd.DataFrame({'tau_days': td, **{f'ATM_IV_{m}': c[m] for m in order}})
        for m in ['BS_FIX', 'BS_INST', 'BS_TERM', 'SH', 'PINN']: d[f'{m}_minus_DH'] = c[m] - c['DH']
        d.to_csv(DATA / 'atm_iv_term_structure_corrected.csv', index=False)
    return c, td


# ------------------------------------------------------------------ 6 & 7. fast vs slow, corrected
def fig_fast_vs_slow():
    td = np.linspace(7., 730., 400); S = np.full_like(td, P.K)
    cols = {'FAST_HEAVY': '#b2182b', 'SLOW_HEAVY': '#2166ac'}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.2, 5.4))
    out = {'tau_days': td}
    for sc, col in cols.items():
        v = CC.SCENARIOS[sc]; lab = 'fast-heavy' if sc == 'FAST_HEAVY' else 'slow-heavy'
        ivd = 100 * P.atm_iv('DH', td / 365., sc); ivp = 100 * P.atm_iv('PINN', td / 365., sc)
        tv = P.time_value('DH', S, td / 365., sc)
        a.plot(td, ivd, color=col, lw=2.3, label=f'exact DH, {lab}  ($v_f$={v["v_fast"]}, $v_s$={v["v_slow"]})')
        a.plot(td, ivp, color=col, lw=1.4, ls='--', label=f'DH-PINN, {lab}')
        b.plot(td, tv, color=col, lw=2.3, label=f'exact DH, {lab}')
        b.plot(td, P.time_value('PINN', S, td / 365., sc), color=col, lw=1.4, ls='--', label=f'DH-PINN, {lab}')
        out[f'ATM_IV_DH_{sc}'] = ivd; out[f'ATM_IV_PINN_{sc}'] = ivp; out[f'TV_DH_{sc}'] = tv
    a.axhline(100 * SIG_INST, color='#111111', lw=1.8, ls='-',
              label=f'Black-Scholes, fixed $\\sigma$ = {100*SIG_INST:.0f}%  (flat by construction)')
    out['ATM_IV_BS_fixed'] = np.full_like(td, 100 * SIG_INST)
    a.set_xlabel(r'remaining time to maturity $\tau$ (days)'); a.set_ylabel('ATM implied volatility (%)')
    a.set_title('ATM implied-volatility term structure')
    a.legend(frameon=False, loc='center right', handlelength=2.4, fontsize=8.4)
    a.annotate('Both states start at 20% instantaneous volatility',
               xy=(20, 100 * SIG_INST), xytext=(210, 100 * SIG_INST + 1.15), fontsize=8.6, color='#333',
               arrowprops=dict(arrowstyle='->', lw=.8, color='#666'))
    a.text(.985, .035, 'Fast and slow variance factors mean-revert at different speeds',
           transform=a.transAxes, fontsize=8.6, color='#333', ha='right', va='bottom')
    a.set_ylim(19.4, 25.4)
    b.set_xlabel(r'remaining time to maturity $\tau$ (days)'); b.set_ylabel('at-the-money time value')
    b.set_title('At-the-money time value'); b.legend(frameon=False, loc='lower right', handlelength=2.4, fontsize=8.4)
    fig.suptitle('Same instantaneous variance ($v_f+v_s=0.04$), different allocation between the two factors', y=.98)
    fig.text(.5, .925, SUB + '   ·   ' + NOTE, ha='center', fontsize=9, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .915])
    fig.savefig(FIG / 'fast_vs_slow_same_variance_corrected.png')
    fig.savefig(FIG / 'fast_vs_slow_same_variance_corrected.pdf'); plt.close(fig)
    pd.DataFrame(out).to_csv(DATA / 'fast_vs_slow_same_variance_corrected.csv', index=False)


# ------------------------------------------------------------------ 13. final validation
def final_checks(c, td):
    r = {}
    S = np.linspace(.70 * P.K, 1.30 * P.K, 401); t90 = np.full_like(S, 90 / 365.)
    Cfix = black(np.log(S / P.K), t90, SIG_FLAT[CC.PRIMARY]) * S
    r['A_fixed_sigma_flat'] = {'max_spread_volpts': float(np.ptp(c['BS_FIX']))}
    r['A_shape_fixed_sigma_BS'] = P.shape_checks(S, Cfix)
    an = P.bs_analytic(np.linspace(60, 160, 200), np.full(200, .5))
    pn = P.bs_pinn_price(np.linspace(60, 160, 200), np.full(200, .5))
    r['B_bs_pinn_vs_analytic'] = {'RMSE': float(np.sqrt(np.mean((pn - an) ** 2))), 'max_abs': float(np.max(np.abs(pn - an)))}
    e = np.concatenate([P.price('PINN', S, np.full_like(S, d / 365.)) - P.price('DH', S, np.full_like(S, d / 365.))
                        for d in [30., 90., 365., 730.]])
    r['C_dh_pinn_vs_exact_dh'] = {'RMSE': float(np.sqrt(np.mean(e ** 2))), 'max_abs': float(np.max(np.abs(e)))}
    for m in ['SH', 'DH', 'PINN']:
        r[f'D_shape_{m}'] = P.shape_checks(S, P.price(m, S, t90))
    f = 100 * P.atm_iv('DH', td / 365., 'FAST_HEAVY'); s = 100 * P.atm_iv('DH', td / 365., 'SLOW_HEAVY')
    r['E_fast_slow_separation_volpts'] = {'at_30d': float(np.interp(30, td, s - f)), 'at_90d': float(np.interp(90, td, s - f)),
                                          'at_365d': float(np.interp(365, td, s - f)), 'min_gap': float(np.min(s - f))}
    (DATA / 'corrected_final_checks.json').write_text(json.dumps(r, indent=2, default=float) + '\n')
    return r


if __name__ == '__main__':
    td = np.linspace(7., 730., 400)
    v = validate_flat(td)
    print('--- fixed-sigma flatness validation (vol points)')
    for k, s in v.items():
        print(f"  {k:16} sigma {s['sigma']:.6f}  RMSE {s['RMSE_volpts']:.3e}  MAE {s['MAE_volpts']:.3e}  max {s['max_abs_volpts']:.3e}")
    assert max(s['max_abs_volpts'] for s in v.values()) < 1e-3, 'flatness validation FAILED - diagnose the inversion'
    c, td = fig_term_structure(False); fig_term_structure(True); fig_fast_vs_slow()
    r = final_checks(c, td)
    print('--- final checks')
    print('  A  fixed-sigma line spread: %.2e vol points' % r['A_fixed_sigma_flat']['max_spread_volpts'])
    print('  B  BS-PINN vs analytic BS:  RMSE %.4f  max %.4f' % (r['B_bs_pinn_vs_analytic']['RMSE'], r['B_bs_pinn_vs_analytic']['max_abs']))
    print('  C  DH-PINN vs exact DH:     RMSE %.2e  max %.2e' % (r['C_dh_pinn_vs_exact_dh']['RMSE'], r['C_dh_pinn_vs_exact_dh']['max_abs']))
    bad = {k: v2 for k, v2 in r.items() if k.startswith(('A_shape', 'D_shape')) and
           (v2['delta_violations'] or v2['gamma_violations'] or v2['below_intrinsic'] or v2['above_spot'])}
    print('  D  financial-shape violations:', bad or 'none')
    print('  E  slow-heavy minus fast-heavy ATM IV: 30d %.2f, 90d %.2f, 365d %.2f vol points'
          % (r['E_fast_slow_separation_volpts']['at_30d'], r['E_fast_slow_separation_volpts']['at_90d'],
             r['E_fast_slow_separation_volpts']['at_365d']))
