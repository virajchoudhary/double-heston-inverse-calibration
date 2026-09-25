"""One graph: implied volatility vs moneyness for the same expiry as graph 1.
Same market data, same fitted level scales, same models - only the vertical axis changes."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; FIG = HERE / 'figures'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
from track_b import prior_c
from evaluate import load_stage, locked
from engine import iv as inv_iv
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 140})

NETS = {'PINN_OLD': locked(), 'PINN_NEW': load_stage('FINAL', 'final')}
B = json.loads((HERE / 'artifacts' / 'spx_benchmark.json').read_text())['one_scale']
MODELS = [('BS', 'BS', B['BS']['scale'], '#9a9a9a', 'Black-Scholes'),
          ('SH', 'SH', B['SH']['scale'], '#d9822b', 'Single Heston'),
          ('PINN_OLD', 'DH_PINN', B['DH_PINN_LOCKED']['scale'], '#8e44ad', 'Double Heston PINN (previous)'),
          ('PINN_NEW', 'DH_PINN', B['DH_PINN_V5']['scale'], '#16a085', 'Double Heston PINN (improved)')]

q = pd.read_csv(EXP / 'hardcoded_v1' / 'artifacts' / 'spx_surface.csv')
exp = q.loc[(q.days - 30).abs().idxmin(), 'expiry']; g = q[q.expiry.eq(exp)].sort_values('x')
tau = float(g.tau.median()); F = float(g.forward.median())
x = np.linspace(g.x.min(), g.x.max(), 300); t = np.full_like(x, tau)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.exp(-g.x.to_numpy()) * F, 100 * g.market_iv.to_numpy(), 'o', color='k', ms=5, zorder=5, label='Market')
for key, kind, s, col, lab in MODELS:
    c = prior_c(kind, x, t, s, NETS.get(key)); v = 100 * inv_iv(c, x, t)
    st = dict(lw=4.2, alpha=.55, ls='-') if key == 'PINN_OLD' else dict(lw=2.0, ls='--' if 'PINN' in key else '-')
    ax.plot(np.exp(-x) * F, v, color=col, label=lab, **st)
ax.axvline(F, color='k', lw=.6, alpha=.35); ax.text(F, ax.get_ylim()[1], ' at the money', fontsize=9, va='top', alpha=.6)
ax.set_xlabel('strike K'); ax.set_ylabel('implied volatility (%)')
ax.set_title(f'Implied volatility vs strike   ·   SPX {exp}, {tau*365:.0f} days   (same data as the C vs S graph)', fontsize=11.5)
ax.legend(frameon=False, fontsize=10, loc='upper right'); fig.tight_layout()
fig.savefig(FIG / 'C_iv_smile.png'); plt.close(fig)
print('written; expiry', exp, 'days', round(tau * 365))
