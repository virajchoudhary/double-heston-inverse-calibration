"""Call price C against market price S, and against time to maturity, for market + three models.

Parameters come from the frozen BTC fit of 2025-10-12 (design B). Market points are that
day's own Deribit quotes. Reads only; writes figures/7_price_curves.png.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import exact, bs_predict, iv, black

HERE = Path(__file__).resolve().parent; A = HERE / 'artifacts'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
COL = {'BS': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8'}
LAB = {'BS': 'Black-Scholes (per expiry)', 'SH': 'Single Heston', 'DH': 'Double Heston'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})

DATE = '2025-10-12'
P = {m: json.loads((A / 'fits' / 'B' / DATE / f'{m}_feller_free.json').read_text())['fit']['best']['params'] for m in ['SH', 'DH']}
BS = json.loads((A / 'amend01' / 'fits' / 'B' / DATE / 'BS_EXPIRY.json').read_text())['fit']
q = pd.read_csv(A / 'surfaces' / f'{DATE}.csv')


def c_model(model, x, tau):
    """Forward-normalised call price c = C/S."""
    x, tau = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(tau, float))
    x, tau = np.broadcast_arrays(x, tau)
    return bs_predict(BS, x, tau) if model == 'BS' else exact(P[model], x, tau, feller=False)


def atm_market(g):
    """Market ATM vol for one expiry: linear in vol across log-moneyness, evaluated at x = 0."""
    o = np.argsort(g.x.to_numpy())
    return float(np.interp(0., g.x.to_numpy()[o], g.market_iv.to_numpy()[o]))


fig, AX = plt.subplots(2, 2, figsize=(13.5, 9.2), gridspec_kw={'height_ratios': [1.35, 1]}); axs = AX[0]

# ---------------- (a) C vs market price S, fixed strike and maturity
EXP = '2025-10-31'                                        # a held-out expiry, 19 days
g = q[q.expiry.eq(EXP)]; tau_a = float(g.tau.median()); F = float(g.forward.median()); K = 112000.
S = np.linspace(.6 * K, 1.7 * K, 240); x = np.log(S / K); t = np.full_like(S, tau_a)
for m in COL: axs[0].plot(S / 1e3, c_model(m, x, t) * S, color=COL[m], lw=2, label=LAB[m])
axs[0].plot(S / 1e3, np.maximum(S - K, 0), 'k--', lw=1, label='payoff at expiry  max(S − K, 0)')
# market: the same expiry's quotes, rescaled to the fixed strike K (prices are homogeneous in S and K)
s_eq = F * K / g.strike.to_numpy(); c_eq = g.c.to_numpy() * s_eq
o = np.argsort(s_eq); axs[0].plot(s_eq[o] / 1e3, c_eq[o], 'o-', color='k', ms=4, lw=1.2, label=f'market ({EXP}, {len(g)} quotes)')
axs[0].axvline(F / 1e3, color='k', lw=.6, alpha=.35); axs[0].text(F / 1e3, axs[0].get_ylim()[1] * .02, ' today\'s price', fontsize=8, alpha=.6)
axs[0].set_xlabel('market price of BTC, S (thousand USD)'); axs[0].set_ylabel('call price C (USD)')
axs[0].set_title(f'C vs market price  ·  strike {K/1e3:.0f}k, {tau_a*365:.0f} days to expiry', fontsize=10.5)
axs[0].legend(frameon=False, fontsize=8.5, loc='upper left')

# ---------------- (b) C vs time to maturity, at the money
F0 = float(q[q.tau.eq(q.tau.min())].forward.median()); tau = np.geomspace(1 / 365, 1., 240); xb = np.zeros_like(tau)
for m in COL: axs[1].plot(tau * 365, c_model(m, xb, tau) * F0, color=COL[m], lw=2, label=LAB[m])
mk = q.groupby('expiry').apply(lambda g: pd.Series({'days': g.days.median(), 'iv': atm_market(g), 'F': g.forward.median()}), include_groups=False)
mk = mk[(mk.iv > 0) & (mk.days <= 365)]
axs[1].plot(mk.days, black(np.zeros(len(mk)), mk.days / 365, mk.iv) * F0, 'o-', color='k', ms=5, lw=1.2, label='market (ATM, each expiry)')
axs[1].set_xlabel('time to maturity τ (days)'); axs[1].set_ylabel('at-the-money call price C (USD)')
axs[1].set_title('C vs time to maturity  ·  at the money (K = S)', fontsize=10.5); axs[1].legend(frameon=False, fontsize=8.5, loc='lower right')


# ---------------- bottom row: the same curves, as the gap to the market (models are nearly identical above)
bx = AX[1][0]
mo = np.argsort(s_eq); s_srt, c_srt = s_eq[mo], c_eq[mo]
for m in COL: bx.plot(s_srt / 1e3, c_model(m, np.log(s_srt / K), np.full_like(s_srt, tau_a)) * s_srt - c_srt, color=COL[m], lw=2, label=LAB[m])
bx.axhline(0, color='k', lw=1); bx.set_xlabel('market price of BTC, S (thousand USD)'); bx.set_ylabel('model − market (USD)')
bx.set_title('Same curves, zoomed: error against the market quotes', fontsize=10.5); bx.legend(frameon=False, fontsize=8.5)
cx = AX[1][1]
mkt_c = black(np.zeros(len(mk)), mk.days.to_numpy() / 365, mk.iv.to_numpy()) * F0
for m in COL: cx.plot(mk.days, c_model(m, np.zeros(len(mk)), mk.days.to_numpy() / 365) * F0 - mkt_c, 'o-', color=COL[m], lw=2, ms=4, label=LAB[m])
cx.axhline(0, color='k', lw=1); cx.set_xlabel('time to maturity τ (days)'); cx.set_ylabel('model − market (USD)')
cx.set_title('Same curves, zoomed: error at each traded expiry', fontsize=10.5); cx.legend(frameon=False, fontsize=8.5)

fig.suptitle(f"Call price from each calibrated model vs the market, BTC {DATE}  (all parameters fitted to that day's options)", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / '7_price_curves.png'); plt.close(fig)
print('written; expiry', EXP, 'tau days', round(tau_a * 365, 1), 'F', round(F), 'market ATM points', len(mk))
