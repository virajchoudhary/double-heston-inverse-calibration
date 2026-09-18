"""Figures for REPORT.md from the frozen test artifacts (amended BS). Reads only; writes figures/."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import iv

HERE = Path(__file__).resolve().parent; A = HERE / 'artifacts'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
COL = {'BS_EXPIRY': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8'}
LAB = {'BS_EXPIRY': 'Black-Scholes (per expiry)', 'SH': 'Single Heston', 'DH': 'Double Heston'}
M = ['BS_EXPIRY', 'SH', 'DH']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})

s = pd.read_csv(A / 'amend01' / 'test_scores.csv'); s = s[s.model.isin(M)]
res = json.loads((A / 'amend01' / 'results.json').read_text())
CELLS = [('shock', 'B', 'Shock · held-out expiries\n(primary)'), ('shock', 'A', 'Shock · held-out strikes'), ('calm', 'B', 'Calm · held-out expiries'), ('calm', 'A', 'Calm · held-out strikes')]


def piv(regime, design, value='IV_RMSE_vol_points'):
    g = s[s.regime.eq(regime) & s.design.eq(design)]
    return g.pivot_table(index=['date', 'episode'] if regime == 'shock' else ['date'], columns='model', values=value)[M].dropna().reset_index()


# 1. Median held-out IV RMSE per cell
fig, ax = plt.subplots(figsize=(9, 4.2)); w = .26
for i, m in enumerate(M):
    vals = [piv(r, d)[m].median() for r, d, _ in CELLS]
    b = ax.bar(np.arange(4) + (i - 1) * w, vals, w, color=COL[m], label=LAB[m])
    ax.bar_label(b, fmt='%.2f', fontsize=8, padding=2)
ax.set_xticks(range(4), [c[2] for c in CELLS]); ax.set_ylabel('median held-out IV RMSE (vol points)')
ax.set_title('Pricing error on options not used in calibration: BTC test dates'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / '1_median_heldout_error.png'); plt.close(fig)

# 2. Paired per-date scatter: competitor error vs DH error (points above diagonal = DH better)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, comp in zip(axs, ['SH', 'BS_EXPIRY']):
    hi = 0
    for (r, d, _), mk in zip(CELLS, ['o', 's', '^', 'D']):
        p = piv(r, d); wins = (p[comp] > p.DH).sum()
        ax.scatter(p.DH, p[comp], s=22, marker=mk, alpha=.75, label=f'{r} {d}: DH better on {wins}/{len(p)}', color=['#1f5fa8', '#6aa0d8', '#2e8b57', '#8fcfa5'][CELLS.index((r, d, _))])
        hi = max(hi, p[[comp, 'DH']].to_numpy().max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], 'k--', lw=1); ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)
    ax.set_xlabel('Double Heston held-out IV RMSE'); ax.set_ylabel(f'{LAB[comp]} held-out IV RMSE')
    ax.set_title(f'Each dot is one test date · above the line = DH better than {"SH" if comp == "SH" else "BS"}'.replace(' · ', '\n'), fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(FIG / '2_paired_dates_scatter.png'); plt.close(fig)

# 3. Shock episodes: mean paired improvement with episode-cluster bootstrap CI (primary cell)
p = piv('shock', 'B'); fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, comp in zip(axs, ['SH', 'BS_EXPIRY']):
    p['d'] = p[comp] - p.DH; e = p.groupby('episode').agg(d=('d', 'mean'), date=('date', 'min')).sort_values('date')
    ax.bar(range(len(e)), e.d, color=np.where(e.d > 0, COL['DH'], '#c0392b'))
    ax.set_xticks(range(len(e)), [x[:7] for x in e.date], rotation=45, fontsize=8)
    c = next(x for x in res['cells'] if x['regime'] == 'shock' and x['design'] == 'B')[f'{comp}_minus_DH']
    ax.axhspan(*c['cluster_bootstrap95'], color=COL['DH'], alpha=.12, label=f"mean over episodes, bootstrap 95% [{c['cluster_bootstrap95'][0]:.2f}, {c['cluster_bootstrap95'][1]:.2f}]")
    ax.axhline(0, color='k', lw=.8); ax.set_ylabel(f'{"SH" if comp == "SH" else "BS"} error − DH error (vol points)')
    ax.set_title(f"Primary test: {c['unit_means_positive']} shock episodes favour DH · Wilcoxon p = {c['wilcoxon_one_sided_p']:.4f}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_xlabel('shock episode (onset month)')
fig.tight_layout(); fig.savefig(FIG / '3_primary_episode_gains.png'); plt.close(fig)

# 4. Maturity and moneyness buckets (pooled held-out IV RMSE, shock B)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, b in zip(axs, ['maturity', 'moneyness']):
    t = pd.read_csv(A / 'secondary' / f'bucket_{b}.csv'); t = t[t.regime.eq('shock') & t.design.eq('B')]
    x = np.arange(len(t))
    for i, m in enumerate(M): ax.bar(x + (i - 1) * .26, t[m], .26, color=COL[m], label=LAB[m])
    lab = t[b].str.replace(' (OTM call wing)', '\nOTM calls').str.replace(' (OTM put wing)', '\nOTM puts')
    ax.set_xticks(x, [f'{l}\n(n={n})' for l, n in zip(lab, t.quotes)], fontsize=8); ax.set_ylabel('pooled held-out IV RMSE (vol points)')
    ax.set_title(f'Shock, held-out expiries: by {b}' + (' (x = log F/K)' if b == 'moneyness' else ''), fontsize=10)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / '4_buckets.png'); plt.close(fig)

# 5. Representative smile: the primary-cell date whose SH−DH gap is the median (chosen by rule, not by look)
p = piv('shock', 'B'); p['d'] = p.SH - p.DH; date = p.sort_values('d').iloc[len(p) // 2].date
q = pd.read_csv(A / 'surfaces' / f'{date}.csv'); sel = json.loads((A / 'variant_selection.json').read_text())
pred = {'BS_EXPIRY': json.loads((A / 'amend01' / 'fits' / 'B' / date / 'BS_EXPIRY.json').read_text())['pred']}
for m in ['SH', 'DH']: pred[m] = json.loads((A / 'fits' / 'B' / date / f'{m}_{sel[m]}.json').read_text())['pred']
exps = sorted(q[q.heldout_B].expiry.unique()); fig, axs = plt.subplots(1, len(exps), figsize=(4 * len(exps), 3.8), squeeze=False)
for ax, e in zip(axs[0], exps):
    k = q.expiry.eq(e).to_numpy(); o = np.argsort(q.x.to_numpy()[k])
    ax.scatter(q.x[k].to_numpy()[o], 100 * q.market_iv[k].to_numpy()[o], color='k', s=16, zorder=3, label='market (held out)')
    for m in M: ax.plot(q.x[k].to_numpy()[o], 100 * iv(np.asarray(pred[m])[k], q.x[k].to_numpy(), q.tau[k].to_numpy())[o], color=COL[m], lw=2, label=LAB[m])
    ax.set_title(f'held-out expiry {e} ({q.days[k].iloc[0]:.0f}d)', fontsize=9.5); ax.set_xlabel('log-moneyness x = log(F/K)')
axs[0][0].set_ylabel('implied vol (%)'); axs[0][0].legend(frameon=False, fontsize=8)
fig.suptitle(f'{date} (shock; median DH-vs-SH gap among the 29 primary dates): smiles of expiries no model saw', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '5_example_heldout_smiles.png'); plt.close(fig)
print('figures written; example date', date)
