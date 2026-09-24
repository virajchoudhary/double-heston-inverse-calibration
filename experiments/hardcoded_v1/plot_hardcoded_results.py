"""Figure for the fixed-parameter comparison. Reads artifacts/results.json only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; R = json.loads((HERE / 'artifacts' / 'results.json').read_text())
COL = {'BS_FIXED': '#9a9a9a', 'SH_PUB': '#d9822b', 'DH_PUB': '#1f5fa8', 'DH_PINN': '#8e44ad'}
LAB = {'BS_FIXED': 'Black-Scholes, fixed vol', 'SH_PUB': 'Single Heston, published', 'DH_PUB': 'Double Heston, published', 'DH_PINN': 'Double Heston PINN'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
S = list(R['primary_nothing_fitted']); short = ['SPX today\n(calm, VIX 14.2)', 'BTC shock days\n(40 days)', 'ETH shock days\n(27 days)']

fig, axs = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1, 1.15]}); w = .2
for ax, key, title in [(axs[0], 'primary_nothing_fitted', 'Nothing fitted: published parameters as they stand'),
                       (axs[1], 'secondary_one_scale_fitted', 'One number fitted per surface (the level scale)')]:
    for i, m in enumerate(COL):
        v = [R[key][s][m]['IV_RMSE'] for s in S]
        b = ax.bar(np.arange(3) + (i - 1.5) * w, v, w, color=COL[m], label=LAB[m], edgecolor='white')
        ax.bar_label(b, fmt='%.1f', fontsize=7.5, padding=1)
    ax.set_xticks(range(3), short, fontsize=9); ax.set_ylabel('IV RMSE vs market (vol points)'); ax.set_ylim(0, 72)
    ax.set_title(title, fontsize=10)
axs[0].legend(frameon=False, fontsize=8.5, loc='upper left')
bk = R['buckets'][S[0]]; keys = list(bk)
for i, m in enumerate(COL):
    axs[2].bar(np.arange(len(keys)) + (i - 1.5) * w, [bk[k][m] for k in keys], w, color=COL[m], edgecolor='white')
axs[2].set_xticks(range(len(keys)), [f"{k}\n(n={bk[k]['quotes']})" for k in keys], fontsize=8.5)
axs[2].set_ylabel('IV RMSE vs market (vol points)'); axs[2].set_title('SPX by maturity, level scale fitted:\nthe second factor helps beyond 30 days', fontsize=10)
fig.suptitle('Published (hard-coded) parameters against real markets — the PINN reproduces Double Heston; the level, not the structure, dominates', fontsize=11)
fig.tight_layout(); fig.savefig(HERE / 'hardcoded_results.png'); plt.close(fig)
print('written')
