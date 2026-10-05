"""Three-model comparison: Black-Scholes, Single Heston, Double Heston.

No PINN curves. Each figure has a price panel (which must follow the Black-Scholes
solution shape) and a difference panel underneath that reveals where Double Heston
departs from the other two. Nothing is offset or rescaled.

Black-Scholes is shown twice so it is not strawmanned:
  solid  = constant sigma = 20%, the instantaneous volatility of the state
  dotted = the frozen per-maturity calibrated fit, i.e. the best Black-Scholes can do
Single Heston is the frozen strongest fit (global search + 12 restarts).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(HERE))
import curvelib as L
import common as CC

FIG = OUT / 'figures_three_model'; FIG.mkdir(exist_ok=True)
DATA = OUT / 'data'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

ST = {
    'BS20':    dict(color='#8a8a8a', ls='-',  lw=1.8, label='Black-Scholes  ($\\sigma$=20%, constant)'),
    'BS_TERM': dict(color='#4d4d4d', ls=':',  lw=1.8, label='Black-Scholes  (best per-maturity fit)'),
    'SH':      dict(color='#d9822b', ls='-',  lw=1.8, label='Single Heston  (one variance factor)'),
    'DH':      dict(color='#1f3f8a', ls='-',  lw=2.4, label='Double Heston  (two variance factors)'),
}
ORDER = ['BS20', 'BS_TERM', 'SH', 'DH']
S = L.S_grid()
NOTE = ('CONTROLLED TWO-TIMESCALE BENCHMARK   ·   $r=q=0$, $K=100$   ·   fast $\\kappa$=10.75, slow $\\kappa$=0.95 '
        '(ratio 11.3)   ·   all models on the same state, 20% instantaneous volatility')
ROWS = []


# ----------------------------------------------------------------- FIGURE 1: C vs S
def fig_c_vs_s():
    tds = [30., 90., 365., 730.]
    fig, axs = plt.subplots(2, 4, figsize=(17.4, 8.0), sharex=True,
                            gridspec_kw={'height_ratios': [2.0, 1.15]})
    for j, td in enumerate(tds):
        tau = np.full_like(S, td / 365.)
        P = {m: L.price(m, S, tau) for m in ORDER}
        a, b = axs[0, j], axs[1, j]
        for m in ORDER:
            a.plot(S, P[m], **ST[m])
        a.plot(S, L.payoff(S), color='#b03060', ls='-.', lw=1.3, label='payoff $\\max(S-K,0)$')
        a.set_title(fr'$\tau$ = {td:g} days', fontsize=11)
        worst = 0.
        for m in ['BS20', 'BS_TERM', 'SH']:
            d = P[m] - P['DH']
            b.plot(S, d, **{**ST[m], 'lw': 1.7})
            worst = max(worst, np.abs(d).max())
            ROWS.append({'figure': 'C_vs_S', 'tau_days': td, 'model': m,
                         'max_abs_gap': float(np.abs(d).max()),
                         'rmse_gap': float(np.sqrt(np.mean(d ** 2)))})
        b.axhline(0, color=ST['DH']['color'], lw=2.0)
        b.set_xlabel('underlying price  $S$')
        gap = np.abs(P['BS20'] - P['DH']).max()
        b.text(.03, .90, f'largest BS gap: {gap:.2f}', transform=b.transAxes, fontsize=9,
               color='#8a2020', fontweight='bold', va='top')
        b.set_ylim(-worst * 1.35, worst * 1.35)
    axs[0, 0].set_ylabel('call price  $C$')
    axs[1, 0].set_ylabel('model $-$ Double Heston')
    axs[0, 0].legend(frameon=False, loc='upper left', fontsize=8, handlelength=2.2)
    fig.suptitle('Call price vs underlying price: Black-Scholes, Single Heston, Double Heston\n'
                 'TOP: all three follow the Black-Scholes solution.   '
                 'BOTTOM: the gap to Double Heston, which widens with maturity.',
                 y=.995, fontsize=12)
    fig.text(.5, .915, NOTE, ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .895])
    fig.savefig(FIG / 'CMP1_C_vs_S_three_models.png'); plt.close(fig)
    print(' CMP1_C_vs_S_three_models')


# ----------------------------------------------------------------- FIGURE 2: C vs tau
def fig_c_vs_tau():
    td = np.linspace(7., 730., 601); tau = td / 365.
    mons = [0.90, 1.00, 1.10]
    fig, axs = plt.subplots(2, 3, figsize=(15.0, 7.4), sharex=True,
                            gridspec_kw={'height_ratios': [2.0, 1.15]})
    for j, mo in enumerate(mons):
        Sm = np.full_like(tau, mo * L.K)
        P = {m: L.price(m, Sm, tau) for m in ORDER}
        a, b = axs[0, j], axs[1, j]
        for m in ORDER:
            a.plot(td, P[m], **ST[m])
        a.axhline(max(mo * L.K - L.K, 0.), color='#b03060', ls='-.', lw=1.3,
                  label='payoff at expiry')
        a.set_title(fr'$S/K$ = {mo:.2f}', fontsize=11)
        worst = 0.
        for m in ['BS20', 'BS_TERM', 'SH']:
            d = P[m] - P['DH']
            b.plot(td, d, **{**ST[m], 'lw': 1.7})
            worst = max(worst, np.abs(d).max())
            ROWS.append({'figure': 'C_vs_tau', 'moneyness': mo, 'model': m,
                         'max_abs_gap': float(np.abs(d).max()),
                         'gap_at_730d': float(d[-1])})
        b.axhline(0, color=ST['DH']['color'], lw=2.0)
        b.set_xlabel('time to maturity  $\\tau$  (days)        $\\longrightarrow$  expiry')
        b.set_ylim(-worst * 1.35, worst * 1.35)
        g30 = abs(P['BS20'][np.argmin(np.abs(td - 30))] - P['DH'][np.argmin(np.abs(td - 30))])
        g730 = abs(P['BS20'][-1] - P['DH'][-1])
        b.text(.03, .90, f'BS gap   30 d: {g30:.3f}    730 d: {g730:.3f}', transform=b.transAxes,
               fontsize=8.5, color='#8a2020', fontweight='bold', va='top')
        for ax in (a, b):
            ax.set_xlim(td.max(), 0.)                      # maturity decreases toward expiry
    axs[0, 0].set_ylabel('call price  $C$')
    axs[1, 0].set_ylabel('model $-$ Double Heston')
    axs[0, 0].legend(frameon=False, loc='upper right', fontsize=8, handlelength=2.2)
    fig.suptitle('Call price vs time to maturity: Black-Scholes, Single Heston, Double Heston\n'
                 'maturity decreases left to right, so each curve decays to its payoff at expiry',
                 y=.995, fontsize=12)
    fig.text(.5, .912, NOTE, ha='center', fontsize=8, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .89])
    fig.savefig(FIG / 'CMP2_C_vs_tau_three_models.png'); plt.close(fig)
    print(' CMP2_C_vs_tau_three_models')


fig_c_vs_s(); fig_c_vs_tau()
d = pd.DataFrame(ROWS); d.to_csv(DATA / 'three_model_gaps.csv', index=False)
print('\nGAP TO DOUBLE HESTON, C vs S (price units, K=100)')
print(d[d.figure == 'C_vs_S'].pivot_table(index='tau_days', columns='model', values='max_abs_gap').round(4).to_string())
print('\nGAP TO DOUBLE HESTON, C vs tau (max over 7-730 d)')
print(d[d.figure == 'C_vs_tau'].pivot_table(index='moneyness', columns='model', values='max_abs_gap').round(4).to_string())
