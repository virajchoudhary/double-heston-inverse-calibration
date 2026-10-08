"""Publication figures for the v5 study, drawn from the verified report numbers."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'overleaf' / 'figures'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
TEAL, GREY, RED, BLUE = '#16a085', '#8a8a8a', '#b03060', '#1f3f8a'


def ablation():
    lab = ['locked\nv4', 'A0\nplain', 'A1\n+resid', 'A2\n+ad.tanh', 'A3\n+gradbal', 'A4\n+RAD', 'A5\ngated']
    pr = [1.081e-5, 6.144e-5, 1.985e-5, 1.572e-5, 2.549e-5, 2.475e-5, 1.471e-5]
    iv = [0.1125, 0.3513, 0.1904, 0.1575, 0.1157, 0.1197, 0.1051]
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.1))
    for ax, v, ttl, ylab, fmt in ((axs[0], pr, 'Price RMSE vs exact DH', r'price RMSE ($\times10^{-5}$)', '{:.2f}'),
                                  (axs[1], iv, 'IV RMSE vs exact DH', 'IV RMSE (vol points)', '{:.3f}')):
        vals = [x * 1e5 for x in v] if fmt == '{:.2f}' else v
        col = [GREY] + [BLUE] * 5 + [TEAL]
        ax.bar(range(7), vals, color=col, width=.66)
        ax.axhline(vals[0], color=RED, ls='--', lw=1.1, label='locked baseline')
        for i, x in enumerate(vals):
            ax.text(i, x + max(vals) * .02, fmt.format(x), ha='center', fontsize=6.8)
        ax.set_xticks(range(7)); ax.set_xticklabels(lab, fontsize=6.8)
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=9.5)
        ax.set_ylim(0, max(vals) * 1.16); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Ablation at an identical 12k-step budget (stages are cumulative)', fontsize=9.5, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(OUT / 'fig1_ablation.png'); plt.close(fig)
    print(' fig1_ablation')


def old_vs_new():
    names = ['Price\nRMSE', 'Price\nP95', 'Price\nmax', 'IV\nRMSE', '7-30d\nIV', 'short-ATM\nIV', 'PDE\nresidual']
    ratio = [1.61, 1.59, 2.56, 1.59, 1.56, 1.95, 1 / 1.8]
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    col = [TEAL] * 6 + [RED]
    ax.bar(range(7), ratio, color=col, width=.62)
    ax.axhline(1, color='k', lw=1.2)
    for i, r in enumerate(ratio):
        t = f'{r:.2f}x better' if r > 1 else f'{1/r:.1f}x worse'
        ax.text(i, r + .05, t, ha='center', fontsize=7)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=7.4)
    ax.set_ylabel('improvement factor vs locked baseline')
    ax.set_title('Selected architecture against the locked baseline,\nuntouched 16,384-point panel opened once', fontsize=9.5)
    ax.set_ylim(0, 2.95); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_old_vs_new.png'); plt.close(fig)
    print(' fig2_old_vs_new')


def market():
    lab = ['Black-\nScholes', 'Single\nHeston', 'Double\nHeston\n(exact)', 'Surrogate\nlocked', 'Surrogate\nselected']
    val = [7.271, 5.788, 5.818, 5.850, 5.847]
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    col = [GREY, GREY, BLUE, '#9aa7c7', TEAL]
    ax.bar(range(5), val, color=col, width=.62)
    for i, v in enumerate(val):
        ax.text(i, v + .1, f'{v:.3f}', ha='center', fontsize=7.6)
    ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=7.4)
    ax.set_ylabel('held-out IV RMSE (vol points)')
    ax.set_ylim(0, 8.4); ax.set_axisbelow(True)
    ax.annotate('', xy=(4, 5.847), xytext=(4, 5.818),
                arrowprops=dict(arrowstyle='<->', color=RED, lw=1.4))
    ax.text(4.35, 5.55, 'surrogate vs exact model:\n0.003 of 5.85', fontsize=7.2, color=RED, ha='left')
    ax.set_title('S&P 500, 10,706 quotes, published parameters, one level scale fitted\n'
                 'improving the surrogate 1.61x moves market error by 0.003', fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig4_market_benchmark.png'); plt.close(fig)
    print(' fig4_market_benchmark')


def track_b():
    lab = ['Black-Scholes', 'Single Heston', 'Double Heston\n(exact)', 'Double Heston\n(surrogate)']
    prior = [7.274, 5.773, 5.839, 5.850]
    hyb = [2.366, 1.917, 1.990, 1.970]
    x = np.arange(4); w = .36
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.bar(x - w / 2, prior, w, label='prior only', color=GREY)
    ax.bar(x + w / 2, hyb, w, label='+ identical 4,545-parameter head', color=TEAL)
    for i in range(4):
        ax.text(x[i] - w / 2, prior[i] + .12, f'{prior[i]:.2f}', ha='center', fontsize=7)
        ax.text(x[i] + w / 2, hyb[i] + .12, f'{hyb[i]:.2f}', ha='center', fontsize=7,
                fontweight='bold' if i == 1 else 'normal')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.6)
    ax.set_ylabel('held-out IV RMSE (vol points)'); ax.set_ylim(0, 8.6)
    ax.legend(frameon=False, loc='upper right'); ax.set_axisbelow(True)
    ax.set_title('Market-residual hybrid, 5,353 held-out quotes\n'
                 'the head does the work; the two-factor prior does not beat one-factor Heston', fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig5_track_b.png'); plt.close(fig)
    print(' fig5_track_b')


ablation(); old_vs_new(); market(); track_b()
