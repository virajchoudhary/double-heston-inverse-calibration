"""TYPE 1 figures: real market quotes. No claim that Double Heston must win.

Surface choice is the one already fixed by experiments/dh_pinn_v5/multi_surface/RULE.md
(largest failure of a single-timescale term structure, chosen before any model error was seen).
Models carry the published parameters with one level scale fitted on that surface, exactly as in
the frozen benchmark; nothing is refitted here.
"""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; OUT = HERE.parent
EXP = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(EXP / 'dh_pinn_v5')); sys.path.insert(0, str(EXP / 'btc_multifactor_v1'))
import common as C
from track_b import prior_c
from evaluate import load_stage, locked
from engine import iv as inv_iv

FIG, DATA = OUT / 'figures', OUT / 'data'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'legend.fontsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .22,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
MS = EXP / 'dh_pinn_v5' / 'multi_surface'
R = json.loads((MS / 'results.json').read_text())
SYM = R['selected']
NETS = {'PINN': load_stage('FINAL', 'final')}
MAP = [('BS', 'BS', None), ('SH', 'SH', None), ('DH', 'DH', None), ('PINN', 'DH_PINN', 'PINN')]


def run(sym=SYM):
    q = pd.read_csv(MS / 'surfaces' / f'{sym}.csv'); M = R['surfaces'][sym]['models']
    name = sym.lstrip('_')
    exp = q.loc[(q.days - C.REPRESENTATIVE_TAU_DAYS).abs().idxmin(), 'expiry']
    g = q[q.expiry.eq(exp)].sort_values('strike').copy()
    tau = float(g.tau.median()); F = float(g.forward.median()); K = float(g.strike.iloc[(g.strike - F).abs().argmin()])
    # market points mapped to the common strike K using the degree-one homogeneity of C in (S, K)
    S_mkt = F * K / g.strike.to_numpy(); C_mkt = g.c.to_numpy() * S_mkt
    keep = np.abs(np.log(S_mkt / K)) <= C.PINN_X_HALF
    S_mkt, C_mkt = S_mkt[keep], C_mkt[keep]; o = np.argsort(S_mkt); S_mkt, C_mkt = S_mkt[o], C_mkt[o]
    S = np.linspace(S_mkt.min(), S_mkt.max(), 400); x = np.log(S / K); t = np.full_like(S, tau)
    cur = {m: prior_c(kind, x, t, M['DH_PINN' if m == 'PINN' else m]['scale'] if m != 'PINN' else M['PINN_NEW']['scale'],
                      NETS.get(net)) * S for m, kind, net in MAP}
    err = {m: np.interp(S_mkt, S, cur[m]) - C_mkt for m in cur}
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]})
    a.plot(S_mkt, C_mkt, 'o', ms=4, color='k', label=f'market quotes ({name})', zorder=5)
    for m in C.ORDER: a.plot(S, cur[m], **C.STYLE[m])
    a.plot(S, np.maximum(S - K, 0), ls=':', lw=1., color='k', label=r'payoff $\max(S-K,0)$')
    a.set_ylabel('call price $C$'); a.legend(frameon=False, loc='upper left', handlelength=2.4)
    a.set_title(f'Market figure: {name} call price vs underlying price')
    for m in C.ORDER: b.plot(S_mkt, err[m], **{**C.STYLE[m], 'label': C.STYLE[m]['label'] + r' $-$ market'})
    b.axhline(0, color='k', lw=1.1)
    b.set_xlabel(f'underlying price $S$   (strike $K={K:g}$, quotes rescaled by homogeneity)')
    b.set_ylabel(r'$C_{model}-C_{market}$'); b.legend(frameon=False, loc='best', handlelength=2.4)
    rms = {m: float(np.sqrt(np.mean(err[m] ** 2))) for m in err}
    fig.text(.5, .962, f'{name} expiry {exp} ({tau*365:.0f} d)  ·  frozen surface choice, published parameters, one level scale',
             ha='center', fontsize=8.5, color='#555')
    fig.text(.5, .942, 'price RMSE  ' + '   '.join(f'{m} {rms[m]:.2f}' for m in C.ORDER), ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .932]); fig.savefig(FIG / f'market_{name}_C_vs_S_with_error.png'); plt.close(fig)
    pd.DataFrame({'S': S, 'tau_days': tau * 365, **{f'C_{m}': cur[m] for m in cur}}).to_csv(DATA / f'market_{name}_C_vs_S.csv', index=False)

    # implied-volatility error against moneyness, whole surface
    fig, ax = plt.subplots(figsize=(8.4, 5.2)); rows = []
    xs = q.x.to_numpy(); ts = q.tau.to_numpy()
    for m, kind, net in MAP:
        sc = M['PINN_NEW']['scale'] if m == 'PINN' else M[m]['scale']
        mi = inv_iv(prior_c(kind, xs, ts, sc, NETS.get(net)), xs, ts)
        e = 100 * (mi - q.market_iv.to_numpy())
        ok = np.isfinite(e)
        bins = np.linspace(-.36, .36, 13); idx = np.digitize(xs[ok], bins)
        mid = [(bins[i - 1] + bins[i]) / 2 for i in range(1, len(bins))]
        val = [float(np.sqrt(np.mean(e[ok][idx == i] ** 2))) if (idx == i).any() else np.nan for i in range(1, len(bins))]
        ax.plot(mid, val, **{**C.STYLE[m], 'marker': 'o', 'ms': 3.5})
        rows.append(pd.DataFrame({'x_mid': mid, 'model': m, 'IV_RMSE_volpts': val}))
    ax.set_xlabel(r'log-moneyness $x=\log(F/K)$'); ax.set_ylabel('implied-vol RMSE (vol points)')
    ax.set_title(f'Market figure: {name} implied-volatility error by moneyness')
    ax.legend(frameon=False, loc='upper center', handlelength=2.4)
    fig.text(.5, .93, f'whole surface, {len(q)} quotes, {q.expiry.nunique()} expiries.  '
                      f'Overall: ' + '  '.join(f'{m} {M["PINN_NEW" if m == "PINN" else m]["IV_RMSE"]:.2f}' for m in C.ORDER),
             ha='center', fontsize=8.5, color='#555')
    fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(FIG / f'market_{name}_IV_error_by_moneyness.png'); plt.close(fig)
    pd.concat(rows).to_csv(DATA / f'market_{name}_IV_error_by_moneyness.csv', index=False)
    print(name, 'price RMSE', {k: round(v, 3) for k, v in rms.items()},
          '| surface IV RMSE', {m: round(M['PINN_NEW' if m == 'PINN' else m]['IV_RMSE'], 2) for m in C.ORDER})


if __name__ == '__main__':
    run()
