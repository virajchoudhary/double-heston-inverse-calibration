"""C vs time-to-maturity drawn on a REVERSED tau axis, matching the standard
Black-Scholes textbook convention (tau decreasing left to right, expiry at the right edge).

The data are identical to the ascending-axis figures; only the axis direction changes.
C is still non-decreasing in tau -- reading right to left, the option gains value as more
time remains; reading left to right, it decays toward the payoff at expiry.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
sys.path.insert(0, str(HERE))
import curvelib as L
import common as CC

FIG = OUT / 'figures_decay_axis'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
TAUD = np.linspace(7., 730., 601); TAU = TAUD / 365.
NOTE = ('CONTROLLED TWO-TIMESCALE BENCHMARK  ·  $r=q=0$, $K=100$  ·  '
        '$\\tau$ DECREASES left to right: expiry at the right edge')


def atm():
    Sa = np.full_like(TAU, L.K)
    P = {m: L.price(m, Sa, TAU) for m in L.MODELS}
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.1, 6.4), sharex=True,
                               gridspec_kw={'height_ratios': [2.1, 1]})
    for m in L.MODELS:
        a.plot(TAUD, P[m], **L.STYLE[m])
    a.axhline(0., color=L.STYLE['PAYOFF']['color'], ls='-.', lw=1.2,
              label='payoff at expiry $\\max(S-K,0)=0$')
    a.set_ylabel('call price  $C$')
    a.set_title('Call price against time to maturity, at the money $S=K=100$\n'
                'time runs left to right toward expiry, so the curve decays to the payoff')
    a.legend(frameon=False, loc='upper right', handlelength=2.2, fontsize=7.5)
    for m in ['BS20', 'BS_TERM', 'SH', 'PINN']:
        b.plot(TAUD, P[m] - P['DH'], **L.STYLE[m])
    b.axhline(0, color=L.STYLE['DH']['color'], lw=1.4)
    b.set_ylabel('model $-$ exact DH')
    b.set_xlabel('time to maturity  $\\tau$  (days)        $\\longrightarrow$  expiry')
    b.legend(frameon=False, ncol=2, handlelength=2.0, fontsize=7)
    for ax in (a, b):
        ax.set_xlim(TAUD.max(), 0.)          # REVERSED axis, expiry at the right edge
    fig.text(.5, .965, NOTE, ha='center', fontsize=7.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .945])
    fig.savefig(FIG / 'A2r_C_vs_tau_ATM_decay_axis.png'); plt.close(fig)
    print(' A2r_C_vs_tau_ATM_decay_axis')


def moneyness():
    mons = [0.80, 0.90, 1.00, 1.10, 1.20]
    fig, axs = plt.subplots(1, 5, figsize=(17.0, 3.9), sharex=True)
    for ax, mo in zip(axs, mons):
        Sm = np.full_like(TAU, mo * L.K)
        for m in L.MODELS:
            ax.plot(TAUD, L.price(m, Sm, TAU), **L.STYLE[m])
        ax.axhline(max(mo * L.K - L.K, 0.), color=L.STYLE['PAYOFF']['color'], ls='-.', lw=1.2,
                   label='payoff $\\max(S-K,0)$')
        ax.set_title(f'$S/K$ = {mo:.2f}')
        ax.set_xlabel('$\\tau$ (days) $\\rightarrow$ expiry')
        ax.set_xlim(TAUD.max(), 0.)          # REVERSED axis
    axs[0].set_ylabel('call price  $C$')
    axs[0].legend(frameon=False, loc='upper right', handlelength=2.0, fontsize=6.8)
    fig.suptitle('Call price against time to maturity at five fixed underlying levels   ·   '
                 'time runs left to right toward expiry; each curve decays to its own payoff', y=.99)
    fig.text(.5, .925, NOTE, ha='center', fontsize=7.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .90])
    fig.savefig(FIG / 'A3r_C_vs_tau_moneyness_decay_axis.png'); plt.close(fig)
    print(' A3r_C_vs_tau_moneyness_decay_axis')


atm(); moneyness()
# sanity: the relation itself is unchanged
Sa = np.full_like(TAU, L.K); C = L.price('DH', Sa, TAU)
print(f'\n  C(tau=730d)={C[-1]:.4f}  C(tau=7d)={C[0]:.4f}  '
      f'non-decreasing in tau: {bool((np.diff(C) >= -1e-12).all())}')
print('  (axis reversed for display only; the data are the same)')
