"""Price-curve, 3D-surface and moneyness figures for v5 (run after the final net exists)."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = Path(__file__).resolve().parent; A = HERE / 'artifacts'; FIG = HERE / 'figures'; FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'btc_multifactor_v1'))
from plots import COL, LAB
from track_b import prior_c
from evaluate import load_stage, locked
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 130})
K, TAU_A = 100., 30 / 365


def get_nets():
    n = {'DH_PINN_LOCKED': locked()}
    if (A / 'final' / 'FINAL').exists(): n['DH_PINN_V5'] = load_stage('FINAL', 'final')
    return n


def c_of(m, x, tau, nets):
    kind = 'DH_PINN' if m.startswith('DH_PINN') else m
    return prior_c(kind, np.asarray(x, float), np.asarray(tau, float), 1., nets.get(m))


def curves():
    nets = get_nets(); order = ['BS', 'SH', 'DH'] + [m for m in ['DH_PINN_LOCKED', 'DH_PINN_V5'] if m in nets]
    S = K * np.exp(np.linspace(-.36, .36, 220)); x = np.log(S / K); ta = np.full_like(S, TAU_A)
    tau = np.geomspace(7 / 365, 2., 220); xz = np.zeros_like(tau)
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9), gridspec_kw={'height_ratios': [1.3, 1]})
    for m in order:
        axs[0][0].plot(S, c_of(m, x, ta, nets) * S, color=COL[m], lw=2, ls='--' if 'PINN' in m else '-', label=LAB[m])
        axs[0][1].plot(tau * 365, c_of(m, xz, tau, nets) * K, color=COL[m], lw=2, ls='--' if 'PINN' in m else '-', label=LAB[m])
    axs[0][0].plot(S, np.maximum(S - K, 0), 'k--', lw=1, alpha=.7, label='payoff at expiry')
    axs[0][0].set_xlabel('underlying price S (strike K = 100)'); axs[0][0].set_ylabel('call price C')
    axs[0][0].set_title('C vs market price, 30 days to expiry', fontsize=10.5); axs[0][0].legend(frameon=False, fontsize=8.5)
    axs[0][1].set_xlabel('time to maturity (days)'); axs[0][1].set_ylabel('at-the-money call price C')
    axs[0][1].set_title('C vs time to maturity, at the money', fontsize=10.5); axs[0][1].legend(frameon=False, fontsize=8.5, loc='lower right')
    ref = c_of('DH', x, ta, nets) * S
    for m in order:
        if m == 'DH': continue
        axs[1][0].plot(S, c_of(m, x, ta, nets) * S - ref, color=COL[m], lw=2, ls='--' if 'PINN' in m else '-', label=LAB[m])
    axs[1][0].axhline(0, color=COL['DH'], lw=1.5); axs[1][0].set_yscale('symlog', linthresh=1e-5)
    axs[1][0].set_xlabel('underlying price S'); axs[1][0].set_ylabel('price − exact Double Heston (symlog)')
    axs[1][0].set_title('Gap to exact Double Heston', fontsize=10.5); axs[1][0].legend(frameon=False, fontsize=8)
    for m in [k for k in ['DH_PINN_LOCKED', 'DH_PINN_V5'] if k in nets]:
        axs[1][1].plot(tau * 365, np.abs(c_of(m, xz, tau, nets) - c_of('DH', xz, tau, nets)) * K, color=COL[m], lw=2, label=LAB[m])
    axs[1][1].set_yscale('log'); axs[1][1].set_xlabel('time to maturity (days)'); axs[1][1].set_ylabel('|PINN − exact DH| price')
    axs[1][1].set_title('Surrogate error by maturity, at the money', fontsize=10.5); axs[1][1].legend(frameon=False, fontsize=8.5)
    fig.suptitle('Published parameters: model price curves and surrogate error', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '8_price_curves.png'); plt.close(fig); print('ok curves')


def surface():
    nets = get_nets(); best = 'DH_PINN_V5' if 'DH_PINN_V5' in nets else 'DH_PINN_LOCKED'
    Sg = K * np.exp(np.linspace(-.36, .36, 60)); Tg = np.geomspace(7 / 365, 2., 60)
    XX, TT = np.meshgrid(np.log(Sg / K), Tg); fx, ft = XX.ravel(), TT.ravel()
    Z = (c_of(best, fx, ft, nets) * K * np.exp(fx)).reshape(XX.shape)
    E = ((c_of(best, fx, ft, nets) - c_of('DH', fx, ft, nets)) * K * np.exp(fx)).reshape(XX.shape)
    fig = plt.figure(figsize=(13, 5.2)); a1 = fig.add_subplot(121, projection='3d'); a2 = fig.add_subplot(122, projection='3d')
    a1.plot_surface(K * np.exp(XX), TT * 365, Z, cmap='viridis', linewidth=0)
    a1.set_xlabel('S'); a1.set_ylabel('days'); a1.set_zlabel('C'); a1.set_title(f'{LAB[best]} price surface', fontsize=10)
    s3 = a2.plot_surface(K * np.exp(XX), TT * 365, E, cmap='coolwarm', linewidth=0)
    a2.set_xlabel('S'); a2.set_ylabel('days'); a2.set_zlabel('error'); a2.set_title('minus exact Double Heston', fontsize=10)
    fig.colorbar(s3, ax=a2, shrink=.6); fig.suptitle('Improved DH-PINN surface and its deviation from exact Double Heston', fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / '9_surface_3d.png'); plt.close(fig); print('ok surface')


def moneyness():
    r = json.loads((A / 'spx_benchmark.json').read_text()); bk = r.get('moneyness_one_scale')
    if not bk: return print('skip moneyness')
    models = [m for m in ['BS', 'SH', 'DH', 'DH_PINN_LOCKED', 'DH_PINN_V5'] if m in bk]; labs = list(bk[models[0]])
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); w = .8 / len(models)
    for i, m in enumerate(models):
        ax.bar(np.arange(len(labs)) + (i - (len(models) - 1) / 2) * w, [bk[m][l]['IV_RMSE'] for l in labs], w, color=COL[m], label=LAB[m], edgecolor='white')
    ax.set_xticks(range(len(labs)), [f"{l}\n(n={bk[models[0]][l]['n']})" for l in labs], fontsize=8.5)
    ax.set_ylabel('SPX IV RMSE (vol points)'); ax.set_title('SPX error by moneyness, one level scale fitted', fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5); fig.tight_layout(); fig.savefig(FIG / '10_market_by_moneyness.png'); plt.close(fig); print('ok moneyness')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('which', nargs='*', default=['all']); a = ap.parse_args()
    for k, f in {'curves': curves, 'surface': surface, 'moneyness': moneyness}.items():
        if 'all' in a.which or k in a.which: f()
