"""Simplified side-by-side block architecture: locked v4 PINN and improved v5 PINN."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = Path(__file__).resolve().parent / 'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
INK, IN, NET, BASE, OUTC, NEW = '#2b2b2b', '#1f5fa8', '#8e44ad', '#1f5fa8', '#2e8b57', '#16a085'


def box(ax, x, y, w, h, text, fc='white', ec=INK, fs=8.6, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5,rounding_size=1.4', fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, weight=weight, linespacing=1.5)


def arrow(ax, x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=11, lw=1.1, color=c, shrinkA=2, shrinkB=2))


def panel(ax, improved):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ttl = ('IMPROVED PINN  (v5, gated)\n282,632 parameters' if improved else 'LOCKED PINN  (v4, plain)\n269,825 parameters')
    ax.text(50, 97, ttl, ha='center', va='top', fontsize=11, weight='bold', color=NEW if improved else INK)
    box(ax, 4, 78, 42, 9, 'INPUT coordinates (4)\n$x=\\log(F/K)$,  $v_s$,  $v_f$,  $\\tau$', fc='#eef3fa', ec=IN)
    box(ax, 54, 78, 42, 9, 'INPUT parameters (8)\n$\\kappa_i,\\theta_i,\\sigma_i,\\rho_i$   $i=s,f$', fc='#eef3fa', ec=IN)
    box(ax, 12, 65, 76, 8, 'FEATURE LAYER  →  24 values   (fixed formulas, identical in both)', fc='#f7f7f7')
    if improved:
        body = ('GATED BLOCKS   (new)\n'
                '$U=\\tanh(a\\,W_1f)$,   $V=\\tanh(a\\,W_2f)$\n'
                '$Z_\\ell=\\tanh(a_\\ell W_\\ell h)$\n'
                '$h \\leftarrow (1-Z_\\ell)\\odot U + Z_\\ell\\odot V$    ×5\n'
                '$a_\\ell$ learned  (adaptive tanh)')
        box(ax, 12, 45, 76, 16, body, fc='#e8f6f3', ec=NEW)
    else:
        box(ax, 12, 45, 76, 16, 'PLAIN STACK\nLinear(24,256) → tanh\n[ Linear(256,256) → tanh ] ×4\nno skip connections, fixed tanh',
            fc='#f4eef8', ec=NET)
    box(ax, 54, 32, 34, 9, 'head: Linear(256,1)\n$\\mathrm{corr}=1.8\\tanh(\\cdot)$', fc='#f4eef8', ec=NET)
    box(ax, 6, 32, 42, 9, 'ANALYTIC BASELINE\n$\\bar v=\\sum_i[\\theta_i+(v_i-\\theta_i)\\frac{1-e^{-\\kappa_i\\tau}}{\\kappa_i\\tau}]$',
        fc='#eef3fa', ec=BASE)
    box(ax, 12, 19, 76, 8, '$\\sigma_{imp}=\\sqrt{\\bar v}\\,e^{\\mathrm{corr}}$        $w=\\sigma_{imp}^2\\tau$', fc='#f7f7f7')
    box(ax, 12, 7, 76, 8, 'BLACK LAYER    $C/K=e^{x}\\Phi(d)-\\Phi(d-\\sqrt{w})$', fc='#eaf5ee', ec=OUTC, weight='bold')
    for a in [(25, 78, 30, 73), (75, 78, 70, 73), (50, 65, 50, 61), (35, 45, 27, 41), (68, 45, 70, 41),
              (27, 32, 40, 27), (70, 32, 60, 27), (50, 19, 50, 15)]:
        arrow(ax, *a)
    note = ('training: + gradient-norm loss balancing every 100 steps\n+ residual-adaptive collocation every 2,000 steps'
            if improved else 'training: fixed loss weights, fixed collocation set')
    ax.text(50, 2.5, note, ha='center', fontsize=7.6, color=NEW if improved else '#666', style='italic')


fig, axs = plt.subplots(1, 2, figsize=(13.6, 7.4))
panel(axs[0], False); panel(axs[1], True)
fig.suptitle('Double Heston PINN: prior-plus-correction architecture, before and after', y=.995, fontsize=12)
fig.text(.5, .015, 'Identical in both: 12 inputs, 24 engineered features, analytic variance baseline, bounded correction, Black pricing layer.'
                   '  Only the learned body and the training mechanics differ.', ha='center', fontsize=8.4, color='#555')
fig.tight_layout(rect=[0, .03, 1, .975])
fig.savefig(FIG / 'pinn_architecture_comparison.png'); plt.close(fig)
print('written')
