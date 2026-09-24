"""C vs market price and C vs time to maturity for FIXED (literature) parameters,
with the locked DH-PINN as a fourth curve.

Nothing here is calibrated: the Double Heston and Single Heston parameter sets are the
published ones used by nifty_multifactor_v4, Black-Scholes uses the flat volatility implied
by the published long-run variances, and the PINN is the locked C3 checkpoint (two-seed mean)
evaluated at the same published Double Heston parameters. Reads only.
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; ROOT = EXP.parent
import sys
sys.path.insert(0, str(EXP / 'btc_multifactor_v1')); sys.path.insert(0, str(ROOT))
from engine import exact, black
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN

V4 = EXP / 'nifty_multifactor_v4'; C = json.loads((V4 / 'config.json').read_text())
DH = np.array(C['published_double_slow_first'])          # DJIA literature, slow factor first
SH = np.array(C['published_single'])                     # DJIA literature single Heston
BS_VOL = float(np.sqrt(DH[1] + DH[6]))                   # flat vol = sqrt(theta_slow + theta_fast)
COL = {'BS': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8', 'PINN': '#8e44ad'}
LAB = {'BS': f'Black-Scholes, fixed vol {100*BS_VOL:.1f}%', 'SH': 'Single Heston (published)', 'DH': 'Double Heston (published, exact)', 'PINN': 'Double Heston PINN (locked C3)'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})

nets = []
for seed in C['pinn']['seeds']:
    n = TorchRegularVariancePINN(factors=2, width=256, depth=C['pinn']['depth'], tau_min=7 / 365, tau_max=2., x_half_width=.36)
    n.load_state_dict(torch.load(V4 / 'artifacts' / 'candidates' / 'C3' / f'pinn_s{seed}' / 'weights.pt')); n.eval(); n.requires_grad_(False); nets.append(n)
STRUCT = torch.tensor(np.stack([DH[0:4], DH[5:9]])[None], dtype=torch.float64)     # (1, 2, 4): kappa, theta, sigma, rho


def c_pinn(x, tau):
    """Forward-normalised call c = C/F from the two-seed mean network."""
    x, tau = np.broadcast_arrays(np.atleast_1d(x), np.atleast_1d(tau))
    co = torch.tensor(np.column_stack([x, np.full(len(x), DH[4]), np.full(len(x), DH[9]), tau]), dtype=torch.float64)
    st = STRUCT.expand(len(x), 2, 4)
    return np.mean([n.price(co, st).numpy() for n in nets], 0) * np.exp(-x)        # net.price is C/K


def c_model(m, x, tau):
    x, tau = np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(tau, float)))
    if m == 'BS': return black(x, tau, BS_VOL)
    if m == 'PINN': return c_pinn(x, tau)
    return exact(DH if m == 'DH' else SH, x, tau, feller=False)


K, TAU_A, XMAX = 100., 30 / 365, .36                      # PINN domain: |x| <= 0.36, tau in [7d, 2y]
fig, AX = plt.subplots(2, 2, figsize=(13.5, 9.2), gridspec_kw={'height_ratios': [1.35, 1]})

# (a) C vs the market price of the underlying
S = K * np.exp(np.linspace(-XMAX, XMAX, 200)); x = np.log(S / K); t = np.full_like(S, TAU_A)
ax = AX[0][0]
for m in COL: ax.plot(S, c_model(m, x, t) * S, color=COL[m], lw=2, ls='--' if m == 'PINN' else '-', label=LAB[m])
ax.plot(S, np.maximum(S - K, 0), 'k--', lw=1, alpha=.7, label='payoff at expiry  max(S − K, 0)')
ax.axvline(K, color='k', lw=.6, alpha=.35); ax.set_xlabel('market price of the underlying, S (strike K = 100)')
ax.set_ylabel('call price C'); ax.set_title(f'C vs market price  ·  fixed parameters, {TAU_A*365:.0f} days to expiry', fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc='upper left')

# (b) C vs time to maturity, at the money
tau = np.geomspace(7 / 365, 2., 200); xb = np.zeros_like(tau); ax = AX[0][1]
for m in COL: ax.plot(tau * 365, c_model(m, xb, tau) * K, color=COL[m], lw=2, ls='--' if m == 'PINN' else '-', label=LAB[m])
ax.set_xlabel('time to maturity τ (days)'); ax.set_ylabel('at-the-money call price C')
ax.set_title('C vs time to maturity  ·  at the money (K = S = 100)', fontsize=10.5); ax.legend(frameon=False, fontsize=8.5, loc='lower right')

# zoomed rows: difference from the exact Double Heston price
ax = AX[1][0]; ref = c_model('DH', x, t) * S
for m in ['BS', 'SH', 'PINN']: ax.plot(S, c_model(m, x, t) * S - ref, color=COL[m], lw=2, ls='--' if m == 'PINN' else '-', label=LAB[m])
ax.axhline(0, color=COL['DH'], lw=1.5); ax.set_xlabel('market price of the underlying, S'); ax.set_ylabel('price − exact Double Heston')
ax.set_title('Zoom: gap to the exact Double Heston price', fontsize=10.5); ax.legend(frameon=False, fontsize=8.5)
ax = AX[1][1]; ref = c_model('DH', xb, tau) * K
for m in ['BS', 'SH', 'PINN']: ax.plot(tau * 365, c_model(m, xb, tau) * K - ref, color=COL[m], lw=2, ls='--' if m == 'PINN' else '-', label=LAB[m])
ax.axhline(0, color=COL['DH'], lw=1.5); ax.set_xlabel('time to maturity τ (days)'); ax.set_ylabel('price − exact Double Heston')
ax.set_title('Zoom: gap to the exact Double Heston price', fontsize=10.5); ax.legend(frameon=False, fontsize=8.5)

fig.suptitle('Fixed published parameters, nothing calibrated: the PINN reproduces exact Double Heston; Single Heston and Black-Scholes do not', fontsize=11)
fig.tight_layout(); fig.savefig(HERE / 'hardcoded_price_curves.png'); plt.close(fig)
d = c_model('PINN', x, t) - c_model('DH', x, t)
print('PINN vs exact DH, forward-normalised: max abs', float(np.max(np.abs(d))), 'RMSE', float(np.sqrt(np.mean(d ** 2))))
