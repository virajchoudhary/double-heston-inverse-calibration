"""All v5 figures. Reads frozen artifacts only."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = Path(__file__).resolve().parent; A = HERE / 'artifacts'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'btc_multifactor_v1'))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
COL = {'BS': '#9a9a9a', 'SH': '#d9822b', 'DH': '#1f5fa8', 'DH_PINN_LOCKED': '#8e44ad', 'DH_PINN_V5': '#16a085', 'DH_PINN': '#8e44ad'}
LAB = {'BS': 'Black-Scholes', 'SH': 'Single Heston', 'DH': 'Double Heston (exact)', 'DH_PINN_LOCKED': 'DH-PINN (locked v4)', 'DH_PINN_V5': 'DH-PINN (improved v5)', 'DH_PINN': 'DH-PINN'}


def fig_ablation():
    r = json.loads((A / 'ablation' / 'fidelity.json').read_text())
    keys = [k for k in ['LOCKED_C3', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5'] if k in r]
    names = {'LOCKED_C3': 'locked v4\n(40k steps)', 'A0': 'A0 plain\n(12k)', 'A1': 'A1 +residual', 'A2': 'A2 +adaptive tanh',
             'A3': 'A3 +grad balance', 'A4': 'A4 +RAD', 'A5': 'A5 gated'}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key, lab in [(axs[0], 'price_RMSE', 'price RMSE vs exact DH'), (axs[1], 'IV_RMSE_volpts', 'IV RMSE vs exact DH (vol points)')]:
        v = [r[k]['development'][key] for k in keys]
        b = ax.bar(range(len(keys)), v, .6, color=['#555'] + ['#1f5fa8'] * (len(keys) - 1), edgecolor='white')
        ax.bar_label(b, fmt='%.3g' if key == 'price_RMSE' else '%.3f', fontsize=8.5, padding=2)
        ax.set_xticks(range(len(keys)), [names[k] for k in keys], fontsize=8.2); ax.set_ylabel(lab)
        ax.axhline(r['LOCKED_C3']['development'][key], color='#c0392b', lw=1, ls='--', label='locked baseline')
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Ablation on the development set (identical 12k-step budget for A0–A5)', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '1_ablation.png'); plt.close(fig)


def fig_old_vs_new():
    r = json.loads((A / 'final' / 'fidelity_final.json').read_text())
    keys = [k for k in ['LOCKED_C3', 'FINAL'] if k in r]
    if len(keys) < 2: return
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    bu = list(r[keys[0]]['final_fidelity']['by_maturity'])
    for i, (metric, lab) in enumerate([('price_RMSE', 'price RMSE'), ('IV_RMSE_volpts', 'IV RMSE (vol points)')]):
        for j, k in enumerate(keys):
            v = [r[k]['final_fidelity']['by_maturity'][b][metric] or np.nan for b in bu]
            axs[i].bar(np.arange(len(bu)) + (j - .5) * .35, v, .35, color=['#8e44ad', '#16a085'][j], label=['locked v4', 'improved v5'][j], edgecolor='white')
        axs[i].set_xticks(range(len(bu)), bu, fontsize=9); axs[i].set_ylabel(lab); axs[i].set_title(f'{lab} by maturity, untouched set', fontsize=10)
        axs[i].legend(frameon=False, fontsize=8.5)
    ov = [[r[k]['final_fidelity'][m] for k in keys] for m in ['price_RMSE', 'price_P95', 'price_max']]
    for j, k in enumerate(keys):
        axs[2].bar(np.arange(3) + (j - .5) * .35, [ov[i][j] for i in range(3)], .35, color=['#8e44ad', '#16a085'][j], label=['locked v4', 'improved v5'][j], edgecolor='white')
    axs[2].set_yscale('log'); axs[2].set_xticks(range(3), ['RMSE', 'P95', 'max'], fontsize=9); axs[2].set_ylabel('price error'); axs[2].set_title('Overall price error (log scale)', fontsize=10)
    axs[2].legend(frameon=False, fontsize=8.5)
    fig.suptitle('Old vs improved DH-PINN against exact Double Heston, untouched fidelity set', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '2_old_vs_new_pinn.png'); plt.close(fig)


def fig_market():
    r = json.loads((A / 'spx_benchmark.json').read_text())
    models = [k for k in ['BS', 'SH', 'DH', 'DH_PINN_LOCKED', 'DH_PINN_V5'] if k in r['one_scale']]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, tag in enumerate(['nothing_fitted', 'one_scale']):
        v = [r[tag][m]['IV_RMSE'] for m in models]
        b = axs[i].bar(range(len(models)), v, .6, color=[COL[m] for m in models], edgecolor='white')
        axs[i].bar_label(b, fmt='%.2f', fontsize=9, padding=2)
        axs[i].set_xticks(range(len(models)), [LAB[m].replace(' (', '\n(') for m in models], fontsize=8.5)
        axs[i].set_ylabel('SPX IV RMSE (vol points)'); axs[i].set_title(['Nothing fitted', 'One level scale fitted'][i], fontsize=10)
    fig.suptitle('Fixed-parameter SPX benchmark: the improved PINN tracks exact Double Heston, so the market error barely moves', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '3_market_benchmark.png'); plt.close(fig)
    bk = r['buckets_one_scale']; labs = list(bk[models[0]])
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); w = .8 / len(models)
    for i, m in enumerate(models):
        ax.bar(np.arange(len(labs)) + (i - (len(models) - 1) / 2) * w, [bk[m][l]['IV_RMSE'] for l in labs], w, color=COL[m], label=LAB[m], edgecolor='white')
    ax.set_xticks(range(len(labs)), [f"{l}\n(n={bk[models[0]][l]['n']})" for l in labs], fontsize=9)
    ax.set_ylabel('SPX IV RMSE (vol points)'); ax.set_title('SPX error by maturity, one level scale fitted', fontsize=10.5); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / '4_market_by_maturity.png'); plt.close(fig)


def fig_track_b():
    T = HERE / 'artifacts' / 'track_b'; r = json.loads((T / 'analysis.json').read_text()); res = json.loads((T / 'results.json').read_text())
    priors = [p for p in ['BS', 'SH', 'DH', 'DH_PINN'] if p in r['overall']]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    w = .38
    for i, p in enumerate(priors):
        axs[0].bar(i - w / 2, r['overall'][p]['prior_IV_RMSE'], w, color=COL[p], alpha=.45, edgecolor='white', label='prior only' if i == 0 else None)
        axs[0].bar(i + w / 2, r['overall'][p]['hybrid_IV_RMSE'], w, color=COL[p], edgecolor='white', label='+ identical residual head' if i == 0 else None)
        axs[0].text(i + w / 2, r['overall'][p]['hybrid_IV_RMSE'], f"{r['overall'][p]['hybrid_IV_RMSE']:.2f}", ha='center', va='bottom', fontsize=9)
        axs[0].text(i - w / 2, r['overall'][p]['prior_IV_RMSE'], f"{r['overall'][p]['prior_IV_RMSE']:.2f}", ha='center', va='bottom', fontsize=9)
    axs[0].set_xticks(range(len(priors)), [LAB[p] for p in priors], fontsize=8.5); axs[0].set_ylabel('held-out SPX IV RMSE (vol points)')
    axs[0].set_title('Equal-capacity residual head on each prior', fontsize=10.5); axs[0].legend(frameon=False, fontsize=8.5)
    for ax, key, title in [(axs[1], 'by_maturity', 'Hybrid error by maturity'), (axs[2], 'by_moneyness', 'Hybrid error by moneyness')]:
        labs = list(r[key][priors[0]]); ww = .8 / len(priors)
        for i, p in enumerate(priors):
            ax.bar(np.arange(len(labs)) + (i - (len(priors) - 1) / 2) * ww, [r[key][p][l]['hybrid'] for l in labs], ww, color=COL[p], label=LAB[p], edgecolor='white')
        ax.set_xticks(range(len(labs)), [l.replace(' ', '\n') for l in labs], fontsize=8); ax.set_ylabel('held-out IV RMSE (vol points)'); ax.set_title(title, fontsize=10.5)
    axs[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Track B: fixed-parameter prior + identical small residual head (held-out SPX quotes)', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '5_track_b.png'); plt.close(fig)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('which', nargs='*', default=['all']); a = ap.parse_args()
    todo = {'ablation': fig_ablation, 'oldnew': fig_old_vs_new, 'market': fig_market, 'trackb': fig_track_b}
    for k, f in todo.items():
        if 'all' in a.which or k in a.which:
            try: f(); print('ok', k)
            except FileNotFoundError as e: print('skip', k, e)
