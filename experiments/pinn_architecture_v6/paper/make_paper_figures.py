"""Generate the three paper figures that did not exist yet, straight from the committed CSVs."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parent.parent / 'artifacts'
OUT = Path(__file__).resolve().parent / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
SHORT = {'BASELINE': 'baseline', 'A0_CONTINUE': 'A0', 'CONTROL_PLAIN': 'control',
         'ARCH_A_MODIFIED_MLP': 'ARCH-A', 'ARCH_B_MODIFIED_MLP_GRADBAL': 'ARCH-B',
         'ARCH_C_MODIFIED_MLP_GRADBAL_RAD': 'ARCH-C'}


def regions():
    d = pd.read_csv(A / 'final_fidelity' / 'regions.csv')
    lab = [b.replace('–', '-').replace('≤', '<=') for b in d['bucket']]
    v = d['improvement_percent'].values
    col = ['#b03060' if x < 0 else '#1f3f8a' for x in v]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    b = ax.bar(range(len(v)), v, color=col, width=.62)
    for i, (r, n) in enumerate(zip(v, d['quotes'])):
        ax.text(i, r + (.6 if r >= 0 else -1.4), f'{r:+.2f}%', ha='center', fontsize=8.5,
                fontweight='bold' if abs(r) > 15 else 'normal')
        ax.text(i, .4 if r < 0 else -1.2, f'n={n}', ha='center', fontsize=7, color='#666')
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(lab, fontsize=8.5)
    ax.set_ylabel('change in price RMSE (%)')
    ax.set_title('ARCH-A against the v5 baseline, final fidelity panel\npositive = improvement; the 7-30 day band regresses')
    ax.set_ylim(min(v) - 3.5, max(v) + 4)
    fig.tight_layout(); fig.savefig(OUT / 'fig_regions.png'); plt.close(fig); print(' fig_regions.png')


def seeds():
    d = pd.read_csv(A / 'seed_robustness.csv', header=[0, 1], index_col=0)
    d = d[d.index.notna()]
    pr = d['price_RMSE']
    arms = [a for a in ['BASELINE', 'A0_CONTINUE', 'CONTROL_PLAIN', 'ARCH_A_MODIFIED_MLP',
                        'ARCH_B_MODIFIED_MLP_GRADBAL', 'ARCH_C_MODIFIED_MLP_GRADBAL_RAD'] if a in pr.index]
    m = pr.loc[arms, 'mean'].astype(float).values * 1e5
    lo = pr.loc[arms, 'min'].astype(float).values * 1e5
    hi = pr.loc[arms, 'max'].astype(float).values * 1e5
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(arms))
    col = ['#16a085' if a == 'ARCH_A_MODIFIED_MLP' else '#7a7a7a' for a in arms]
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt='none', ecolor='#999', elinewidth=1.3, capsize=5)
    ax.scatter(x, m, c=col, s=64, zorder=3, label='seed mean')
    for i, a in enumerate(arms):
        ax.scatter([x[i], x[i]], [lo[i], hi[i]], c='#bbb', s=16, marker='_', zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[a] for a in arms], fontsize=8.5)
    ax.set_ylabel(r'price RMSE  ($\times 10^{-5}$)')
    ax.set_title('Seed robustness: mean over seeds 17 and 43, bars span the two seeds\n'
                 'two seeds only; the spread overlaps between the leading arms')
    fig.tight_layout(); fig.savefig(OUT / 'fig_seeds.png'); plt.close(fig); print(' fig_seeds.png')


def efficiency():
    d = pd.read_csv(A / 'efficiency.csv')
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for _, r in d.iterrows():
        a = r['arm']; c = '#16a085' if a.startswith('ARCH_A') else ('#1f3f8a' if a == 'BASELINE' else '#7a7a7a')
        ax.scatter(r['inference_1024_ms_per_seed'], r['price_RMSE'] * 1e5, s=90, color=c, zorder=3)
        ax.annotate(SHORT.get(a, a), (r['inference_1024_ms_per_seed'], r['price_RMSE'] * 1e5),
                    textcoords='offset points', xytext=(7, 5), fontsize=8.5)
    ax.set_xlabel('inference time (ms per 1024 quotes)')
    ax.set_ylabel(r'price RMSE  ($\times 10^{-5}$)')
    ax.set_title('Accuracy against inference cost\nlower and further left is better')
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / 'fig_efficiency.png'); plt.close(fig); print(' fig_efficiency.png')


if __name__ == '__main__':
    regions(); seeds(); efficiency()
