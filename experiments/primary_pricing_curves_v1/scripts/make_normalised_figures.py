"""NORMALISED companions to the primary price curves.

Why this is not cosmetic separation: every panel divides by a stated, financially meaningful
quantity applied IDENTICALLY to every model. No curve is shifted or offset vertically. The
un-normalised primaries in figures/01..10 remain the reference and are unchanged.

A normalisation was only kept if it was VERIFIED to separate the models. Two candidates were
tested and REJECTED, and the numbers are recorded in data/rejected_normalisations.csv:

  * C/S against standardised moneyness z = log(S/K)/(sigma sqrt(tau)).
    There is no single-variable scaling law: C/S depends on z AND on sigma sqrt(tau) separately,
    through K/S = exp(-z sigma sqrt(tau)). Measured maturity spread was 0.265 for Black-Scholes
    against 0.272 for Double Heston -- no contrast.
  * C/(S sigma sqrt(tau)) against z. Measured spread 13.7% for Black-Scholes against 11.1% for
    Single Heston, i.e. Black-Scholes spreads MORE. Rejected.

What is kept:
  N1  relative price difference (C_model - C_DH)/C_DH, in per cent.
  N2  vega-normalised price residual (C_model - C_DH)/vega_DH, in volatility points.
  N3  sqrt(tau)-normalised price vs maturity, AT THE MONEY ONLY -- verified to discriminate there
      (0.33% variation for constant-sigma Black-Scholes against 10.39% for Double Heston) and
      verified NOT to discriminate away from the money (~140% for every model), so it is not used there.
  N4  fast-heavy against slow-heavy, as a ratio and in volatility points.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(HERE))
import curvelib as L
import common as CC
from engine import iv as inv_iv

FIG, DATA = OUT / 'figures_normalised', OUT / 'data'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
S, TD, SIG = L.S_grid(), L.TAU_DAYS, L.SIGMA_CONST
NOTE = ('CONTROLLED TWO-TIMESCALE BENCHMARK  ·  $r=q=0$, $K=100$  ·  each panel divides by a stated '
        'quantity applied identically to every model\nno curve is shifted or offset  ·  '
        'un-normalised primaries in figures/01-10 are unchanged')
ROWS = []


def finish(fig, name, note=NOTE, rect=.87, y=.90):
    if note: fig.text(.5, y, note, ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=[0, 0, 1, rect]); fig.savefig(FIG / f'{name}.png'); plt.close(fig)
    print('  wrote', name)


def vega(S_, tau, sigma):
    v = sigma * np.sqrt(tau)
    d1 = (np.log(S_ / L.K) + .5 * v ** 2) / v
    return S_ * np.exp(-.5 * d1 ** 2) / np.sqrt(2 * np.pi) * np.sqrt(tau)


# ---------------------------------------------------------------- N1  relative price difference
def n1():
    tds = [30., 90., 365., 730.]
    FRAC = 0.01                       # keep C_DH >= 1% of the at-the-money DH price at that maturity
    fig, axs = plt.subplots(1, 4, figsize=(17.0, 4.6))
    for ax, td in zip(axs, tds):
        tau = np.full_like(S, td / 365.)
        dh = L.price('DH', S, tau)
        atm = float(L.price('DH', np.array([L.K]), np.array([td / 365.]))[0])
        keep = dh > FRAC * atm
        for m in ['BS20', 'BS_TERM', 'SH', 'PINN']:
            e = np.where(keep, 100. * (L.price(m, S, tau) - dh) / dh, np.nan)
            ax.plot(S, e, **L.STYLE[m]); ok = np.isfinite(e)
            ROWS.append({'figure': 'N1_relative', 'model': m, 'tau_days': td,
                         'RMSE_pct': float(np.sqrt(np.mean(e[ok] ** 2))), 'max_abs_pct': float(np.abs(e[ok]).max())})
        ax.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
        ax.set_title(fr'$\tau$ = {td:g} days'); ax.set_xlabel('underlying price  $S$')
        ax.set_ylabel('relative difference vs exact DH  (%)')
    axs[0].legend(frameon=False, handlelength=2.2, fontsize=8)
    fig.suptitle('N1   Price normalised by the exact Double-Heston price   ·   shown only where '
                 '$C_{DH}$ exceeds 1% of the at-the-money DH price, since the ratio diverges as '
                 '$C_{DH}\\to0$   ·   each panel has its own $y$ axis', y=.995)
    finish(fig, 'N1_relative_price_difference')


# ---------------------------------------------------------------- N2  vega-normalised residual
def n2():
    tds = [30., 90., 365., 730.]
    fig, axs = plt.subplots(1, 4, figsize=(17.0, 4.6))
    for ax, td in zip(axs, tds):
        tau = np.full_like(S, td / 365.)
        dh = L.price('DH', S, tau)
        vg = vega(S, tau, inv_iv(dh / S, np.log(S / L.K), tau))
        # vega -> 0 in the deep wings, where the ratio diverges and carries no information
        good = np.isfinite(vg) & (vg > 0.01 * np.nanmax(vg))
        for m in ['BS20', 'BS_TERM', 'SH', 'PINN']:
            e = np.where(good, 100. * (L.price(m, S, tau) - dh) / vg, np.nan)
            ax.plot(S, e, **L.STYLE[m]); ok = np.isfinite(e)
            ROWS.append({'figure': 'N2_vega', 'model': m, 'tau_days': td,
                         'RMSE_volpts': float(np.sqrt(np.mean(e[ok] ** 2))),
                         'max_abs_volpts': float(np.abs(e[ok]).max())})
        ax.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
        ax.set_title(fr'$\tau$ = {td:g} days'); ax.set_xlabel('underlying price  $S$')
        ax.set_ylabel('residual / DH vega   (vol points)')
    axs[0].legend(frameon=False, handlelength=2.2, fontsize=8)
    fig.suptitle('N2   Price residual normalised by Double-Heston vega   ·   a currency residual '
                 'expressed in volatility points   ·   restricted to vega $\\geq$ 1% of its peak, '
                 'and each panel has its own $y$ axis', y=.995)
    finish(fig, 'N2_vega_normalised_residual')


# ---------------------------------------------------------------- N3  sqrt(tau) normalisation, ATM
def n3():
    td = np.linspace(7., 730., 601); tau = td / 365.; Sa = np.full_like(tau, L.K)
    stats = {}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for m in L.MODELS:
        C = L.price(m, Sa, tau); y = C / np.sqrt(tau)
        stats[m] = 100. * (np.nanmax(y) - np.nanmin(y)) / np.nanmean(y)
        a.plot(td, y, **L.STYLE[m])
        b.plot(td, 100. * (C - L.price('DH', Sa, tau)) / L.price('DH', Sa, tau), **L.STYLE[m])
        ROWS.append({'figure': 'N3_sqrt_tau_ATM', 'model': m, 'tau_days': np.nan,
                     'variation_pct_of_C_over_sqrt_tau': float(stats[m])})
    a.set_ylabel(r'$C/\sqrt{\tau}$  (at the money)', fontsize=9)
    a.set_title('variation over 7-730 d:  '
                + ', '.join(f'{m} {stats[m]:.2f}%' for m in ['BS20', 'DH']), fontsize=8.5)
    b.set_ylabel('relative difference vs exact DH (%)', fontsize=9)
    b.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
    b.set_title('as a ratio to exact Double Heston', fontsize=8.5)
    for ax in (a, b):
        ax.set_xlabel(r'remaining maturity $\tau$ (days)', fontsize=9)
        ax.legend(frameon=False, handlelength=1.6, fontsize=6.5)
    fig.suptitle('N3   Call price divided by $\\sqrt{\\tau}$, at the money\n'
                 'removes the leading scaling, leaving variance term structure', fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .88]); fig.savefig(FIG / 'N3_sqrt_tau_normalised_ATM.png'); plt.close(fig)
    print('  wrote N3_sqrt_tau_normalised_ATM')


# ---------------------------------------------------------------- N4  fast vs slow, normalised
def n4():
    """Both normalisers are floored: the ratio diverges as the slow-heavy price goes to zero,
    and the vega ratio diverges in the deep wings. Floors are relative and stated on the figure."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for c, td in zip(L.MAT_CMAP, TD):
        tau = np.full_like(S, td / 365.)
        f = L.price('DH', S, tau, 'FAST_HEAVY'); s_ = L.price('DH', S, tau, 'SLOW_HEAVY')
        atm = float(L.price('DH', np.array([L.K]), np.array([td / 365.]), 'SLOW_HEAVY')[0])
        keep = s_ > 0.01 * atm                      # 1% of the at-the-money slow-heavy price
        r = np.where(keep, 100. * (f / s_ - 1.), np.nan)
        a.plot(S, r, color=c, lw=1.7, label=fr'$\tau$ = {td:g} d')
        vg = vega(S, tau, inv_iv(s_ / S, np.log(S / L.K), tau))
        good = np.isfinite(vg) & (vg > 0.01 * np.nanmax(vg))
        y = np.where(good, 100. * (f - s_) / vg, np.nan)
        b.plot(S, y, color=c, lw=1.7, label=fr'$\tau$ = {td:g} d')
        ok = np.isfinite(y)
        ROWS.append({'figure': 'N4_fast_vs_slow', 'model': 'DH', 'tau_days': td,
                     'max_abs_volpts': float(np.abs(y[ok]).max()),
                     'max_abs_pct': float(np.nanmax(np.abs(r)))})
    for ax, yl, ti in ((a, 'fast-heavy / slow-heavy $-$ 1  (%)', 'as a ratio'),
                       (b, 'price gap / vega  (vol points)', 'vega-normalised')):
        ax.axhline(0, color='k', lw=1.1); ax.set_xlabel('underlying price  $S$')
        ax.set_ylabel(yl, fontsize=9); ax.set_title(ti, fontsize=9.5)
        ax.legend(frameon=False, ncol=2, handlelength=1.8, fontsize=7)
    fig.suptitle('N4   Fast-heavy against slow-heavy, normalised   ·   identical 20% instantaneous '
                 'volatility\nratio floored at 1% of the ATM slow-heavy price; vega floored at 1% of peak',
                 fontsize=9, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .90]); fig.savefig(FIG / 'N4_fast_vs_slow_normalised.png'); plt.close(fig)
    print('  wrote N4_fast_vs_slow_normalised')


# ---------------------------------------------------------------- rejected normalisations, recorded
def rejected():
    lo = max(np.log(S.min() / L.K) / (SIG * np.sqrt(t / 365.)) for t in TD)
    hi = min(np.log(S.max() / L.K) / (SIG * np.sqrt(t / 365.)) for t in TD)
    z = np.linspace(lo, hi, 401); out = []
    for norm in ['C_over_S', 'C_over_S_sigma_sqrt_tau']:
        for m in ['BS20', 'SH', 'DH']:
            band = []
            for t in TD:
                tau = t / 365.; Sz = L.K * np.exp(z * SIG * np.sqrt(tau))
                C = L.price(m, Sz, np.full_like(Sz, tau))
                band.append(C / Sz if norm == 'C_over_S' else C / (Sz * SIG * np.sqrt(tau)))
            b = np.array(band); sp = b.max(0) - b.min(0)
            out.append({'normalisation': norm, 'model': m, 'max_maturity_spread': float(sp.max()),
                        'spread_pct_of_level': float(100 * sp.max() / np.abs(b).max()),
                        'verdict': 'rejected: does not separate models'})
    # off-the-money sqrt(tau) test
    tdv = np.linspace(7., 730., 601); tau = tdv / 365.
    for mo in [0.90, 1.10]:
        for m in ['BS20', 'SH', 'DH']:
            y = L.price(m, np.full_like(tau, mo * L.K), tau) / np.sqrt(tau)
            out.append({'normalisation': f'C_over_sqrt_tau at S/K={mo:.2f}', 'model': m,
                        'max_maturity_spread': float(np.nanmax(y) - np.nanmin(y)),
                        'spread_pct_of_level': float(100 * (np.nanmax(y) - np.nanmin(y)) / np.nanmean(y)),
                        'verdict': 'rejected away from the money: no contrast between models'})
    pd.DataFrame(out).to_csv(DATA / 'rejected_normalisations.csv', index=False)
    print('\nREJECTED normalisations recorded in data/rejected_normalisations.csv')
    print(pd.DataFrame(out)[['normalisation', 'model', 'spread_pct_of_level']].to_string(index=False))


if __name__ == '__main__':
    for f in (n1, n2, n3, n4): f()
    rejected()
    r = pd.DataFrame(ROWS); r.to_csv(DATA / 'normalised_summary.csv', index=False)
    print('\nN1 relative difference vs exact DH, max abs (%)')
    print(r[r.figure == 'N1_relative'].pivot_table(index='tau_days', columns='model', values='max_abs_pct').round(3).to_string())
    print('\nN2 vega-normalised residual, max abs (volatility points)')
    print(r[r.figure == 'N2_vega'].pivot_table(index='tau_days', columns='model', values='max_abs_volpts').round(4).to_string())
    print('\nN3 ATM variation of C/sqrt(tau) over 7-730 days (%)')
    print(r[r.figure == 'N3_sqrt_tau_ATM'][['model', 'variation_pct_of_C_over_sqrt_tau']].round(3).to_string(index=False))
