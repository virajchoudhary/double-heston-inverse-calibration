"""Rebuild the paper's Fig. 2 and Fig. 3 from the committed final-panel arrays."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parent.parent / 'artifacts' / 'final_fidelity'
OUT = Path(__file__).resolve().parent / 'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

d = np.load(A / 'data.npz')
x, tau = d['coords'][:, 0], d['coords'][:, 3]
teach = d['price']
pa = np.load(A / 'ARCH_A_MODIFIED_MLP_predictions.npy')
pb = np.load(A / 'BASELINE_predictions.npy')
ea, eb = np.abs(pa - teach), np.abs(pb - teach)
diff = ea - eb                                   # negative favours ARCH_A

TOL = 1e-15
win = (diff < -TOL).mean() * 100
tie = (np.abs(diff) <= TOL).mean() * 100
loss = (diff > TOL).mean() * 100
print(f'pointwise: improves {win:.2f}%  ties {tie:.2f}%  worsens {loss:.2f}%')
print(f'  paper states 49.88 / 3.32 / 46.80')
print(f'price RMSE baseline {np.sqrt((eb**2).mean()):.6e}  ARCH_A {np.sqrt((ea**2).mean()):.6e}')
print(f'  reduction {100*(1-np.sqrt((ea**2).mean())/np.sqrt((eb**2).mean())):.4f}%  (paper: 9.24%)')


def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9))
    rm = [1.057170e-5, 1.058803e-5, 9.610126e-6]
    lbl = ['v4 C3', 'v5 SHARED', 'v6 ARCH_A']
    col = ['#7a7a7a', '#7a7a7a', '#16a085']
    a.bar(range(3), [r * 1e5 for r in rm], color=col, width=.6)
    a.set_xticks(range(3)); a.set_xticklabels(lbl, fontsize=8)
    a.set_ylabel(r'Price RMSE ($\times 10^{-5}$)'); a.set_title('Price RMSE', fontsize=9.5)
    a.grid(axis='y', alpha=.25); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)

    names = ['Price RMSE', 'IV RMSE', 'PDE RMSE', 'PDE max', 'Parameters', 'Latency']
    ch = [-9.24, -1.57, 0.98, 7.87, 16.37, 35.11]
    cols = ['#16a085' if c < 0 else '#b03060' for c in ch]
    y = np.arange(len(ch))[::-1]
    b.barh(y, ch, color=cols, height=.62)
    b.set_yticks(y); b.set_yticklabels(names, fontsize=8)
    b.axvline(0, color='k', lw=1.1)
    b.set_xlabel('ARCH_A change vs v5 SHARED (%)', fontsize=8.5)
    b.set_title('Disclosed trade-offs', fontsize=9.5)
    b.grid(axis='x', alpha=.25); b.set_axisbelow(True)
    for s in ('top', 'right'): b.spines[s].set_visible(False)
    for yy, c in zip(y, ch):
        b.text(c + (1.2 if c >= 0 else -1.2), yy, f'{c:+.2f}', va='center',
               ha='left' if c >= 0 else 'right', fontsize=7.5)
    b.set_xlim(-21, 46)
    fig.tight_layout(); fig.savefig(OUT / 'fig2_common_panel.png'); plt.close(fig)
    print(' fig2_common_panel.png')


def fig3(nx=11, nt=11):
    xe = np.linspace(x.min(), x.max(), nx + 1)
    te = np.geomspace(tau.min(), tau.max(), nt + 1)
    M = np.full((nt, nx), np.nan)
    for i in range(nt):
        for j in range(nx):
            m = (tau >= te[i]) & (tau <= te[i + 1]) & (x >= xe[j]) & (x <= xe[j + 1])
            if m.sum(): M[i, j] = diff[m].mean()
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.pcolormesh(np.arange(nx + 1), np.arange(nt + 1), M * 1e6,
                       cmap='RdBu_r', vmin=-lim * 1e6, vmax=lim * 1e6)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('mean |err ARCH_A| - |err v5|   (x 1e-6)\nnegative favours ARCH_A',
                 fontsize=8)
    ax.set_xlabel('Log-forward moneyness bin'); ax.set_ylabel('Maturity bin (7-730 days, log-spaced)')
    ax.set_title(f'Paired absolute price-error change, v5 SHARED to ARCH_A\n'
                 f'improves {win:.2f}%, ties {tie:.2f}%, worsens {loss:.2f}% of rows', fontsize=9.5)
    fig.tight_layout(); fig.savefig(OUT / 'fig3_paired_error_map.png'); plt.close(fig)
    print(' fig3_paired_error_map.png')


fig2(); fig3()
