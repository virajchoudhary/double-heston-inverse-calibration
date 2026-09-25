"""Cross-asset summary of the frozen primary cells (held-out expiries). Reads only; writes cross_asset_figures/."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent; FIG = H / 'cross_asset_figures'; FIG.mkdir(exist_ok=True)
COL = {'BS_EXPIRY': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8'}; LAB = {'BS_EXPIRY': 'Black-Scholes (per expiry)', 'SH': 'Single Heston', 'DH': 'Double Heston'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
ASSETS = [('BTC', 'btc_multifactor_v1/artifacts/amend01/results.json', 'btc_multifactor_v1/artifacts/surfaces', lambda c: c['regime'] == 'shock' and c['design'] == 'B'),
          ('ETH', 'eth_multifactor_v1/artifacts/results.json', 'eth_multifactor_v1/artifacts/surfaces', lambda c: c['regime'] == 'shock' and c['design'] == 'B'),
          ('HYPE', 'hype_multifactor_v1/artifacts/results.json', 'hype_multifactor_v1/artifacts/surfaces', lambda c: c.get('group') == 'all' and c['design'] == 'B'),
          ('SOL', 'sol_multifactor_v1/artifacts/results.json', 'sol_multifactor_v1/artifacts/surfaces', lambda c: c['regime'] == 'shock' and c['design'] == 'B'),
          ('XRP', 'xrp_multifactor_v1/artifacts/results.json', 'xrp_multifactor_v1/artifacts/surfaces', lambda c: c['regime'] == 'shock' and c['design'] == 'B')]
rows = []
for name, res, surf, pick in ASSETS:
    c = next(x for x in json.loads((H / res).read_text())['cells'] if pick(x))
    exp = np.mean([pd.read_csv(f).expiry.nunique() for f in (H / surf).glob('*.csv')])
    bs = 'BS_EXPIRY_minus_DH'
    rows.append({'asset': name, 'expiries': exp, **{m: c['median_IV_RMSE'][m] for m in COL}, 'sh_gain': c['SH_minus_DH']['mean'], 'sh_ci': c['SH_minus_DH']['cluster_bootstrap95'],
                 'sh_sig': c['SH_minus_DH']['DH_beats'], 'bs_sig': c[bs]['DH_beats'], 'sh_wins': c['SH_minus_DH']['DH_wins_dates']})
d = pd.DataFrame(rows).sort_values('expiries', ascending=False).reset_index(drop=True); d.to_csv(FIG / 'cross_asset_primary.csv', index=False)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.35, 1]}); w = .27
for i, m in enumerate(COL):
    b = ax.bar(np.arange(len(d)) + (i - 1) * w, d[m], w, color=COL[m], label=LAB[m], edgecolor='white')
    ax.bar_label(b, fmt='%.2f', fontsize=8, padding=2, fontweight='bold' if m == 'DH' else 'normal')
ax.set_xticks(range(len(d)), [f"{r.asset}{' (all days)' if r.asset == 'HYPE' else ''}\n~{r.expiries:.1f} exp./day" for r in d.itertuples()])
ax.set_ylabel('median pricing error on unseen expiries\n(implied-vol points, lower is better)'); ax.set_ylim(0, 15)
ax.set_title('Primary test on each coin: held-out expiries', fontsize=10.5); ax.legend(frameon=False, loc='upper left', fontsize=9)
for j, r in d.iterrows():
    lo, hi = r.sh_ci; col = COL['DH'] if r.sh_sig else '#b0b0b0'
    bx.errorbar(r.sh_gain, j, xerr=[[r.sh_gain - lo], [hi - r.sh_gain]], fmt='o', color=col, capsize=4, ms=8)
    bx.text(hi + .05, j, f"{'significant' if r.sh_sig else 'not significant'} · DH better {r.sh_wins} days", va='center', fontsize=8.5, color=col)
bx.axvline(0, color='k', lw=.8); bx.set_yticks(range(len(d)), [f'{a} (~{e:.1f} exp.)' for a, e in zip(d.asset, d.expiries)]); bx.invert_yaxis()
bx.set_xlim(-.6, 3.6); bx.set_xlabel('Single Heston error − Double Heston error (vol points)\nmean with 95% cluster-bootstrap interval; right of 0 = DH better')
bx.set_title('How much Double Heston beats Single Heston', fontsize=10.5)
fig.suptitle('Double Heston vs Single Heston vs Black-Scholes across five crypto option markets: DH always beats BS; it beats SH significantly only where many liquid expiries exist', fontsize=11)
fig.tight_layout(); fig.savefig(FIG / 'cross_asset_summary.png'); plt.close(fig)
print(d.round(2).to_string())
