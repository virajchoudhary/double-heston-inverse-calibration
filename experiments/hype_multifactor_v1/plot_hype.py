"""Summary figure for HYPE (same style as BTC/XRP). Reads only."""
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
res = json.loads((A / 'results.json').read_text()); mark = json.loads((A / 'secondary' / 'secondary.json').read_text())['mark_target_cells']
CELLS = [(res, 'B', 'Held-out expiries\n(primary)'), (res, 'A', 'Held-out strikes'), (mark, 'B', 'Held-out expiries\nscored vs mark prices'), (mark, 'A', 'Held-out strikes\nscored vs mark prices')]
cell = lambda src, d: next(x for x in (src['cells'] if isinstance(src, dict) else src) if x['group'] == 'all' and x['design'] == d)
fig, ax = plt.subplots(figsize=(11, 6.3)); w = .27
for i, m in enumerate(M):
    b = ax.bar(np.arange(4) + (i - 1) * w, [cell(s, d)['median_IV_RMSE'][m] for s, d, _ in CELLS], w, color=COL[m], label=LAB[m], edgecolor='white')
    ax.bar_label(b, fmt='%.2f', fontsize=9, padding=2, fontweight='bold' if m == 'DH' else 'normal')
for j, (s, d, _) in enumerate(CELLS):
    c = cell(s, d); v = lambda k: 'significant' if c[k]['DH_beats'] else 'not sig.'
    ax.text(j, -1.0, f"DH beats SH: {c['SH_minus_DH']['DH_wins_dates']} days\n({v('SH_minus_DH')})\nDH beats BS: {c['BS_EXPIRY_minus_DH']['DH_wins_dates']} days\n({v('BS_EXPIRY_minus_DH')})",
            ha='center', va='top', fontsize=8.3, linespacing=1.4, color=COL['DH'])
ax.set_xticks(range(4), [c[2] for c in CELLS]); ax.set_ylim(0, 7); ax.set_ylabel('median pricing error on unseen options\n(implied-vol points, lower is better)')
ax.set_title('HYPE options (Deribit), 55 test days Jun–Sep 2026: Double Heston lowest everywhere; vs Single Heston significant in 2 of 4', fontsize=10.5)
ax.legend(frameon=False, loc='upper center', ncol=3); fig.subplots_adjust(bottom=.3, top=.9, left=.1, right=.97)
fig.savefig(FIG / 'hype_summary_comparison.png'); plt.close(fig)
