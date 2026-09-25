"""Architecture diagram of the locked DH-PINN (candidate C3). Drawn from the code, not from memory."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 9.5, 'figure.dpi': 140})
fig, ax = plt.subplots(figsize=(12.6, 7.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis('off')
INK, ACC, NET, OUTC = '#2b2b2b', '#1f5fa8', '#8e44ad', '#2e8b57'


def box(x, y, w, h, text, fc='white', ec=INK, fs=9.5, weight='normal', tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc or INK, weight=weight, linespacing=1.45)


def arrow(x1, y1, x2, y2, c=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, lw=1.2, color=c, shrinkA=2, shrinkB=2))


box(2, 56, 26, 13, 'INPUT  coordinates (4)\n' + r'$x=\log(F/K)$,  $v_{slow}$,  $v_{fast}$,  $\tau$', fc='#eef3fa', ec=ACC)
box(2, 40, 26, 13, 'INPUT  parameters (8)\n' + r'$\kappa_i,\ \theta_i,\ \sigma_i,\ \rho_i$  for  $i=$ slow, fast', fc='#eef3fa', ec=ACC)
box(33, 44, 31, 25, 'FEATURE LAYER  →  24 values\nfixed formulas, nothing learned\n\n'
                    '4 general: $x/0.36$,  $z=x/\\sqrt{\\tau\\bar v+x^2/64}$,\n$\\tanh(z/1.5)$,  scaled $\\log\\tau$\n\n'
                    '9 per factor (×2): $\\log v_i$, $\\log\\kappa_i$, $\\log\\theta_i$, $\\rho_i$,\n$\\tanh(\\kappa_i\\tau)$, $\\nu_i$, $\\rho_i\\nu_i$, $\\log(v_i/\\theta_i)$, Feller ratio\n\n'
                    '2 cross-factor: variance share,  skew × $\\tanh$', fc='#f7f7f7', fs=8.8)
box(68, 44, 30, 25, 'MLP  (the only learned part)\n\n'
                    'Linear 24→256  →  tanh\n[ Linear 256→256  →  tanh ] × 4\nLinear 256→1  (head, initialised to 0)\n\n'
                    '269,825 parameters · float64\nno residual connections', fc='#f4eef8', ec=NET)
box(68, 29, 30, 11, 'BOUNDED CORRECTION\n' + r'$\mathrm{corr}=1.8\cdot\tanh(\mathrm{head})$', fc='#f4eef8', ec=NET)
box(34, 29, 29, 11, 'BASELINE (analytic)\n' + r'$\bar v=\sum_i[\theta_i+(v_i-\theta_i)\frac{1-e^{-\kappa_i\tau}}{\kappa_i\tau}]$', fc='#eef3fa', ec=ACC)
box(34, 13, 64, 11, 'OUTPUT  implied volatility  ' + r'$\sigma_{imp}=\sqrt{\bar v}\;e^{\,\mathrm{corr}}$' +
    '      →      price  ' + r'$C/K=\mathrm{Black}(x,\ \sigma_{imp}^2\tau)$', fc='#eaf5ee', ec=OUTC, weight='bold')
box(2, 1, 96, 9, 'TRAINING   100,000 exact Fourier prices + 18,000 physics points   ·   loss = price²/(2e-5)² + IV²/(0.002)² + 0.1·PDE²/(0.01)² + 0.1·(negative convexity)²\n'
                 'Adam 40,000 steps, lr 1e-3 → 1e-5 cosine, batches 512 labels / 128 physics   ·   then L-BFGS 300 steps   ·   two seeds (17, 43), prices averaged', fc='#fbfbfb', fs=8.6)
for a in [(28, 62.5, 33, 60), (28, 46.5, 33, 52), (64, 56.5, 68, 56.5), (83, 44, 83, 40), (48.5, 44, 48.5, 40), (83, 29, 83, 24.5), (48.5, 29, 48.5, 24.5)]:
    arrow(*a)
ax.text(50, 71.5, 'Locked DH-PINN (candidate C3): a bounded correction to an analytic volatility baseline', ha='center', fontsize=12, weight='bold')
ax.text(50, 24.5, 'domain: |x| ≤ 0.36,  7 days ≤ τ ≤ 2 years,  published parameter family with one level scale s ∈ [0.7, 3.2]', ha='center', fontsize=8.5, style='italic', color='#666')
fig.savefig(HERE / 'pinn_architecture.png', bbox_inches='tight'); plt.close(fig); print('written')
