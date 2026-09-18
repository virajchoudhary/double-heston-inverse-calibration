"""Summary figure for sol (same layout as btc figure 6). Reads only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; A = HERE / 'artifacts'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
COL = {'BS_EXPIRY': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8'}; LAB = {'BS_EXPIRY': 'Black-Scholes (per expiry)', 'SH': 'Single Heston', 'DH': 'Double Heston'}; M = list(COL)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
s = pd.read_csv(A / 'test_scores.csv'); res = json.loads((A / 'results.json').read_text())
CELLS = [('shock', 'B', 'Shock days\nheld-out expiries (primary)'), ('shock', 'A', 'Shock days\nheld-out strikes'), ('calm', 'B', 'Calm days\nheld-out expiries'), ('calm', 'A', 'Calm days\nheld-out strikes')]


def piv(r, d):
    g = s[s.regime.eq(r) & s.design.eq(d)]
    return g.pivot_table(index=['date', 'episode'] if r == 'shock' else ['date'], columns='model', values='IV_RMSE_vol_points')[M].dropna().reset_index()


fig, ax = plt.subplots(figsize=(11, 6.3)); w = .27
for i, m in enumerate(M):
    b = ax.bar(np.arange(4) + (i - 1) * w, [piv(r, d)[m].median() for r, d, _ in CELLS], w, color=COL[m], label=LAB[m], edgecolor='white')
    ax.bar_label(b, fmt='%.2f', fontsize=9, padding=2, fontweight='bold' if m == 'DH' else 'normal')
for j, (r, d, _) in enumerate(CELLS):
    p = piv(r, d); c = next(x for x in res['cells'] if x['regime'] == r and x['design'] == d)
    verdict = lambda k: 'significant' if c[k]['DH_beats'] else 'not sig.'
    ax.text(j, -2.1, f"DH beats SH: {(p.SH > p.DH).sum()}/{len(p)} days\n({verdict('SH_minus_DH')})\nDH beats BS: {(p.BS_EXPIRY > p.DH).sum()}/{len(p)} days\n({verdict('BS_EXPIRY_minus_DH')})",
            ha='center', va='top', fontsize=8.3, linespacing=1.4, color=COL['DH'])
ax.set_xticks(range(4), [c[2] for c in CELLS]); ax.set_ylim(0, 11.5); ax.set_ylabel('median pricing error on unseen options\n(implied-vol points, lower is better)')
ax.set_title('Solana options (Deribit), 32 test days 2024–2026: Double Heston clearly beats Black-Scholes; vs Single Heston it is mixed', fontsize=10.5)
ax.legend(frameon=False, loc='upper right'); fig.subplots_adjust(bottom=.3, top=.9, left=.1, right=.97)
fig.savefig(FIG / 'sol_summary_comparison.png'); plt.close(fig)
