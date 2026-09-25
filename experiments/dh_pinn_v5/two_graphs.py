"""The two requested graphs: C vs S and C vs time to maturity.

Market = today's SPX quotes. Models = fixed published parameters with the level scale fitted
on the full surface (as in the frozen benchmark): Black-Scholes, Single Heston, the previous
(locked v4) DH-PINN and the improved (v5) DH-PINN. Domain is the PINN's trained region,
|log(F/K)| <= 0.36 and 7 days <= tau <= 2 years.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; FIG = HERE / 'figures'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
from track_b import prior_c
from evaluate import load_stage, locked
from engine import black
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 140})

NETS = {'PINN_OLD': locked(), 'PINN_NEW': load_stage('FINAL', 'final')}
B = json.loads((HERE / 'artifacts' / 'spx_benchmark.json').read_text())['one_scale']
MODELS = [('BS', 'BS', B['BS']['scale'], '#9a9a9a', 'Black-Scholes'),
          ('SH', 'SH', B['SH']['scale'], '#d9822b', 'Single Heston'),
          ('PINN_OLD', 'DH_PINN', B['DH_PINN_LOCKED']['scale'], '#8e44ad', 'Double Heston PINN (previous)'),
          ('PINN_NEW', 'DH_PINN', B['DH_PINN_V5']['scale'], '#16a085', 'Double Heston PINN (improved)')]
q = pd.read_csv(EXP / 'hardcoded_v1' / 'artifacts' / 'spx_surface.csv')


def c_of(key, kind, s, x, tau):
    return prior_c(kind, np.asarray(x, float), np.asarray(tau, float), s, NETS.get(key))


# ---------------------------------------------------------------- graph 1: C vs S
g = q.loc[(q.days - 30).abs().sort_values().index]
exp = g.iloc[0].expiry; g = q[q.expiry.eq(exp)].sort_values('strike')
tau_a = float(g.tau.median()); F = float(g.forward.median()); K = float(g.strike.iloc[(g.strike - F).abs().argmin()])
S = K * np.exp(np.linspace(-.36, .36, 260)); x = np.log(S / K); t = np.full_like(S, tau_a)
fig, ax = plt.subplots(figsize=(8.6, 6))
s_eq = F * K / g.strike.to_numpy(); c_eq = g.c.to_numpy() * s_eq; o = np.argsort(s_eq)
keep = (np.abs(np.log(s_eq / K)) <= .36)[o]
ax.plot(s_eq[o][keep], c_eq[o][keep], 'o-', color='k', ms=4.5, lw=1.6, zorder=5, label='Market')
for key, kind, s, col, lab in MODELS:
    st = dict(lw=4.2, alpha=.55, ls='-') if key == 'PINN_OLD' else dict(lw=2.0, ls='--' if 'PINN' in key else '-')
    ax.plot(S, c_of(key, kind, s, x, t) * S, color=col, label=lab, **st)
ax.plot(S, np.maximum(S - K, 0), 'k:', lw=1.2, alpha=.7, label='payoff at expiry')
ax.set_yscale('log'); ax.set_ylim(1e-1, 4e3)
ax.set_xlabel('market price of the underlying, S'); ax.set_ylabel('call price C  (log scale)')
ax.set_title(f'C vs S   ·   SPX, strike {K:.0f}, {tau_a*365:.0f} days to expiry   (log price axis)', fontsize=12)
ax.legend(frameon=False, fontsize=10, loc='lower right'); fig.tight_layout()
fig.savefig(FIG / 'A_C_vs_S.png'); plt.close(fig)

# ---------------------------------------------------------------- graph 2: C vs time to maturity (at the money)
tau = np.geomspace(7 / 365, 2., 240); xz = np.zeros_like(tau); REF = 100.
rows = []
for e, v in q.groupby('expiry'):
    v = v.sort_values('x')
    if len(v) < 3 or not (v.x.min() < 0 < v.x.max()): continue
    iv0 = float(np.interp(0., v.x.to_numpy(), v.market_iv.to_numpy()))
    rows.append({'days': float(v.days.median()), 'tau': float(v.tau.median()), 'iv': iv0})
m = pd.DataFrame(rows).sort_values('days'); m = m[(m.days >= 7) & (m.days <= 730)]
fig, ax = plt.subplots(figsize=(8.6, 6))
ax.plot(m.days, black(np.zeros(len(m)), m.tau.to_numpy(), m.iv.to_numpy()) * REF, 'o-', color='k', ms=4.5, lw=1.6, zorder=5, label='Market')
for key, kind, s, col, lab in MODELS:
    st = dict(lw=4.2, alpha=.55, ls='-') if key == 'PINN_OLD' else dict(lw=2.0, ls='--' if 'PINN' in key else '-')
    ax.plot(tau * 365, c_of(key, kind, s, xz, tau) * REF, color=col, label=lab, **st)
ax.set_yscale('log'); ax.set_xlabel('time to maturity τ (days)'); ax.set_ylabel('at-the-money call price C  (S = K = 100, log scale)')
ax.set_title('C vs time to maturity   ·   SPX, at the money   (log price axis)', fontsize=12)
ax.legend(frameon=False, fontsize=10, loc='lower right'); fig.tight_layout()
fig.savefig(FIG / 'B_C_vs_tau.png'); plt.close(fig)

# ---------------------------------------------------------------- no-arbitrage validation of every plotted curve
print(f"{'model':32}{'C>=intrinsic':>13}{'C<=S':>7}{'delta in [0,1]':>17}{'convex':>9}{'dC/dtau>=0':>12}")
Sd = K * np.exp(np.linspace(-.36, .36, 1200)); xd = np.log(Sd / K); td = np.full_like(Sd, tau_a)
taud = np.geomspace(7 / 365, 2., 600); xzd = np.zeros_like(taud)
for key, kind, s, col, lab in MODELS:
    c = c_of(key, kind, s, xd, td) * Sd; d = np.gradient(c, Sd); g2 = np.gradient(d, Sd)
    ct = c_of(key, kind, s, xzd, taud) * REF
    print(f'{lab:32}{str(bool((c >= np.maximum(Sd - K, 0) - 1e-8).all())):>13}{str(bool((c <= Sd + 1e-8).all())):>7}'
          f'{f"[{d.min():.3f},{d.max():.3f}]":>17}{str(bool((g2 >= -1e-7).all())):>9}{str(bool((np.diff(ct) >= -1e-9).all())):>12}')
