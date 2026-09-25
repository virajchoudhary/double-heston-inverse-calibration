"""Architecture diagrams: improved pure DH-PINN, and the hybrid with the market-residual head."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
INK, ACC, NET, OUTC, RES = '#2b2b2b', '#1f5fa8', '#8e44ad', '#2e8b57', '#c0392b'
plt.rcParams.update({'font.size': 9.5, 'figure.dpi': 140})


def canvas(w=12.6, h=7.4):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis('off'); return fig, ax


def box(ax, x, y, w, h, text, fc='white', ec=INK, fs=9.5, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight=weight, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, lw=1.2, color=c, shrinkA=2, shrinkB=2))


def improved(winner, params, flags):
    fig, ax = canvas()
    box(ax, 2, 55, 26, 13, 'INPUT  coordinates (4)\n' + r'$x=\log(F/K)$,  $v_{slow}$,  $v_{fast}$,  $\tau$', fc='#eef3fa', ec=ACC)
    box(ax, 2, 39, 26, 13, 'INPUT  parameters (8)\n' + r'$\kappa_i,\ \theta_i,\ \sigma_i,\ \rho_i$', fc='#eef3fa', ec=ACC)
    box(ax, 32, 42, 27, 26, 'FEATURE LAYER → 24 values\nunchanged from v4\n(4 general, 9 per factor, 2 cross)', fc='#f7f7f7')
    body = ('RESIDUAL BLOCKS  (new)\n\nprojection  Linear(24, 256)\n' + r'$h \leftarrow h + \alpha_\ell\,\tanh(a_\ell\,W_\ell h + b_\ell)$'
            + '\n× 4 blocks,  $\\alpha_\\ell$ learned (init 0.1)\n' + ('$a_\\ell$ learned per block (adaptive tanh)\n' if flags.get('adaptive') else '')
            + f'\n{params:,} parameters · float64') if flags.get('body') == 'residual' else (
        'GATED BLOCKS  (new)\n\n' + r'$U,V=\tanh(W_{1,2}f+b)$, $Z=\tanh(W_\ell h+b)$' + '\n' + r'$h \leftarrow (1-Z)\odot U + Z\odot V$'
        + f'\n\n{params:,} parameters · float64')
    box(ax, 63, 42, 35, 26, body, fc='#f4eef8', ec=NET)
    box(ax, 63, 27, 35, 11, 'BOUNDED CORRECTION\n' + r'$\mathrm{corr}=1.8\tanh(\mathrm{head})$', fc='#f4eef8', ec=NET)
    box(ax, 32, 27, 27, 11, 'ANALYTIC BASELINE\n' + r'$\bar v=\sum_i[\theta_i+(v_i-\theta_i)\frac{1-e^{-\kappa_i\tau}}{\kappa_i\tau}]$', fc='#eef3fa', ec=ACC)
    box(ax, 32, 13, 66, 10, 'OUTPUT   ' + r'$\sigma_{imp}=\sqrt{\bar v}\,e^{\mathrm{corr}}$' + '   →   ' + r'$C/K=\mathrm{Black}(x,\sigma_{imp}^2\tau)$', fc='#eaf5ee', ec=OUTC, weight='bold')
    extra = []
    if flags.get('gradbal'): extra.append('gradient-norm loss balancing every 100 steps')
    if flags.get('rad'): extra.append('residual-adaptive collocation resampling every 2,000 steps (same 18,000-point budget)')
    box(ax, 2, 2, 96, 8, 'TRAINING  same data and budget as v4' + (' · ' + ' · '.join(extra) if extra else ''), fc='#fbfbfb', fs=8.8)
    for a in [(28, 61.5, 32, 58), (28, 45.5, 32, 50), (59, 55, 63, 55), (80.5, 42, 80.5, 38), (45.5, 42, 45.5, 38), (80.5, 27, 80.5, 23.5), (45.5, 27, 45.5, 23.5)]:
        arrow(ax, *a)
    ax.text(50, 71.5, f'Improved pure DH-PINN ({winner}): same physics prior, better-conditioned body', ha='center', fontsize=12, weight='bold')
    fig.savefig(FIG / '6_improved_pinn_architecture.png', bbox_inches='tight'); plt.close(fig)


def hybrid(head_params):
    fig, ax = canvas(12.6, 6.4); ax.set_ylim(0, 62)
    box(ax, 2, 40, 30, 16, 'FIXED-PARAMETER PRIOR\npublished Double Heston\n(exact or improved PINN)\nplus one level scale fitted\non calibration quotes only', fc='#eef3fa', ec=ACC)
    box(ax, 36, 42, 24, 12, 'prior implied vol\n' + r'$IV_{DH}(x,\tau)$', fc='#f7f7f7')
    box(ax, 36, 22, 24, 14, 'RESIDUAL HEAD  (new, tiny)\n' + r'inputs $[x,\ \log\tau,\ IV_{DH},\ x/\sqrt{\tau}]$'
        + f'\nLinear(4,64)→tanh→Linear(64,64)\n→tanh→Linear(64,1)\n{head_params:,} parameters', fc='#fdeeec', ec=RES)
    box(ax, 64, 22, 34, 14, 'BOUNDED CORRECTION\n' + r'$\Delta IV=\delta_{max}\tanh(\cdot)$, $\delta_{max}$ from development data'
        + '\npenalised by ' + r'$\lambda\,\overline{\Delta IV^2}$' + '\nno held-out quote is ever an input', fc='#fdeeec', ec=RES)
    box(ax, 36, 5, 62, 11, 'HYBRID OUTPUT   ' + r'$IV=IV_{DH}+\Delta IV$' + '   →   ' + r'$C=\mathrm{Black}(x,IV^2\tau)$'
        + '\nlabelled DH_PINN_RESIDUAL — physics prior plus ML residual, NOT exact Double Heston', fc='#eaf5ee', ec=OUTC, weight='bold')
    for a in [(32, 48, 36, 48), (48, 42, 48, 36), (60, 29, 64, 29), (81, 22, 81, 16.5), (48, 22, 48, 16.5)]: arrow(ax, *a)
    ax.text(50, 59, 'Track B hybrid: fixed Double Heston physics + a small bounded market-residual head', ha='center', fontsize=12, weight='bold')
    ax.text(17, 33, 'the same head, optimiser,\nbudget and data are also\ngiven to the Black-Scholes\nand Single Heston priors', ha='center', fontsize=9, style='italic', color='#666')
    fig.savefig(FIG / '7_hybrid_architecture.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    sel = HERE / 'artifacts' / 'ablation' / 'selection.json'
    if sel.exists():
        s = json.loads(sel.read_text()); improved(s['winner'], s['params'], s['flags'])
    else:
        improved('A2 (placeholder)', 269834, {'body': 'residual', 'adaptive': True})
    r = json.loads((HERE / 'artifacts' / 'track_b' / 'results.json').read_text())
    hybrid(list(r['priors'].values())[0]['head_params'])
    print('diagrams written')
