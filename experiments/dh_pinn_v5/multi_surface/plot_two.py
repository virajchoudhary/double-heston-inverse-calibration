"""The two requested graphs for one surface: C vs S, and C vs time to maturity."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; V5 = HERE.parent
sys.path.insert(0, str(V5)); sys.path.insert(0, str(V5.parent / 'btc_multifactor_v1'))
from track_b import prior_c
from evaluate import load_stage, locked
from engine import black
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 140})
NETS = {'PINN_OLD': locked(), 'PINN_NEW': load_stage('FINAL', 'final')}
R = json.loads((HERE / 'results.json').read_text())['surfaces']
STYLE = [('BS', 'BS', None, '#9a9a9a', 'Black-Scholes'), ('SH', 'SH', None, '#d9822b', 'Single Heston'),
         ('PINN_OLD', 'DH_PINN', 'PINN_OLD', '#8e44ad', 'Double Heston PINN (previous)'),
         ('PINN_NEW', 'DH_PINN', 'PINN_NEW', '#16a085', 'Double Heston PINN (improved)')]


def draw(sym, tag):
    q = pd.read_csv(HERE / 'surfaces' / f'{sym}.csv'); M = R[sym]['models']
    lab_extra = f"SH {M['SH']['IV_RMSE']:.2f} vs DH {M['DH']['IV_RMSE']:.2f} vol pts"
    # graph 1
    exp = q.loc[(q.days - 30).abs().idxmin(), 'expiry']; g = q[q.expiry.eq(exp)].sort_values('strike')
    tau = float(g.tau.median()); F = float(g.forward.median()); K = float(g.strike.iloc[(g.strike - F).abs().argmin()])
    S = K * np.exp(np.linspace(-.36, .36, 260)); x = np.log(S / K); t = np.full_like(S, tau)
    fig, ax = plt.subplots(figsize=(8.6, 6))
    s_eq = F * K / g.strike.to_numpy(); c_eq = g.c.to_numpy() * s_eq; o = np.argsort(s_eq)
    ax.plot(s_eq[o], c_eq[o], 'o-', color='k', ms=4.5, lw=1.6, zorder=5, label='Market')
    for key, kind, net, col, lab in STYLE:
        st = dict(lw=4.2, alpha=.55, ls='-') if key == 'PINN_OLD' else dict(lw=2.0, ls='--' if 'PINN' in key else '-')
        ax.plot(S, prior_c(kind, x, t, M[key]['scale'], NETS.get(net)) * S, color=col, label=lab, **st)
    ax.plot(S, np.maximum(S - K, 0), 'k:', lw=1.2, alpha=.7, label='payoff at expiry')
    ax.set_yscale('log'); ax.set_xlabel('market price of the underlying, S'); ax.set_ylabel('call price C  (log scale)')
    ax.set_title(f'C vs S   ·   {sym.lstrip("_")}, strike {K:g}, {tau*365:.0f} days   [{tag}]\n{lab_extra}', fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc='lower right'); fig.tight_layout()
    fig.savefig(HERE / f'{sym.lstrip("_")}_1_C_vs_S.png'); plt.close(fig)
    # graph 2
    rows = []
    for e, v in q.groupby('expiry'):
        v = v.sort_values('x')
        if len(v) < 3 or not (v.x.min() < 0 < v.x.max()): continue
        rows.append({'days': float(v.days.median()), 'tau': float(v.tau.median()),
                     'iv': float(np.interp(0., v.x.to_numpy(), v.market_iv.to_numpy()))})
    m = pd.DataFrame(rows).sort_values('days'); REF = 100.
    tau2 = np.geomspace(7 / 365, min(2., m.tau.max()), 240); xz = np.zeros_like(tau2)
    fig, ax = plt.subplots(figsize=(8.6, 6))
    ax.plot(m.days, black(np.zeros(len(m)), m.tau.to_numpy(), m.iv.to_numpy()) * REF, 'o-', color='k', ms=4.5, lw=1.6, zorder=5, label='Market')
    for key, kind, net, col, lab in STYLE:
        st = dict(lw=4.2, alpha=.55, ls='-') if key == 'PINN_OLD' else dict(lw=2.0, ls='--' if 'PINN' in key else '-')
        ax.plot(tau2 * 365, prior_c(kind, xz, tau2, M[key]['scale'], NETS.get(net)) * REF, color=col, label=lab, **st)
    ax.set_xlabel('time to maturity τ (days)'); ax.set_ylabel('at-the-money call price C  (S = K = 100)')
    ax.set_title(f'C vs time to maturity   ·   {sym.lstrip("_")}, at the money   [{tag}]\n{lab_extra}', fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc='lower right'); fig.tight_layout()
    fig.savefig(HERE / f'{sym.lstrip("_")}_2_C_vs_tau.png'); plt.close(fig)
    print('drawn', sym)


if __name__ == '__main__':
    draw(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else '')
