"""PINN development-split fidelity check. REPORTING/AUDIT ONLY.

Runs after training and BEFORE `run.py lock`. Uses only the predeclared development split
(teacher seed 93102, 4,096 points). It never generates or reads the held-out fidelity (93104)
or PDE-fidelity (93105) points that the frozen `evaluate()` uses, so that held-out test stays
untouched. Changes no model, checkpoint, threshold or selection.

Rule fixed before first run: evaluate the frozen fidelity gates on development points. If the
two-seed mean fails any gate, do not lock; document the failure and stop before any comparison.
A pass does not replace the frozen held-out fidelity test in evaluate().
"""
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from run import OUT, verify, read, save, sha, stamp, nets, neural, metrics


def bucket_masks(coords, p, P):
    x, vs, vf, tau = coords.T; days = 365 * tau; kf = np.exp(-x); s = p[:, 1] / P[1]
    m = {'all': np.ones(len(x), bool)}
    for a, b in [(7, 30), (30, 90), (90, 365), (365, 731)]: m[f'maturity_{a}-{b}d'] = (days >= a) & (days < b)
    for a, b in [(0, .9), (.9, .98), (.98, 1.02), (1.02, 1.1), (1.1, 9)]: m[f'K/F_{a}-{b}'] = (kf >= a) & (kf < b)
    m['ATM_|K/F-1|<=.02'] = abs(kf - 1) <= .02; m['wing_K/F<.9_or_>1.1'] = (kf < .9) | (kf > 1.1)
    m['fast_heavy_v_fast>v_slow'] = vf > vs; m['slow_heavy_v_slow>=v_fast'] = vs >= vf
    for a, b in [(.7, 1.5333), (1.5333, 2.3667), (2.3667, 3.21)]: m[f'scale_{a:.2f}-{b:.2f}'] = (s >= a) & (s < b)
    return m


def main():
    c = verify(); P = np.array(c['published_double_slow_first']); pc = c['pinn']; gates = pc['fidelity_gate']
    dest = OUT / 'pinn_development_check'; dest.mkdir(exist_ok=False)
    status = {}
    for s in pc['seeds']:
        done = read(OUT / f'pinn_s{s}/completed.json'); sd = torch.load(OUT / f'pinn_s{s}/weights.pt', weights_only=True)
        status[str(s)] = {'weights_sha256_matches_completion_record': done['weights_sha256'] == sha(OUT / f'pinn_s{s}/weights.pt'),
                          'all_weights_finite': bool(all(torch.isfinite(v).all() for v in sd.values())),
                          'training_seconds': done['seconds'], 'last_logged_adam_row': done['history'][-1]}
    models = nets(c)
    z = np.load(OUT / 'teacher/development.npz'); coords, p, y = z['coords'], z['params'], z['price']
    x, tau = coords[:, 0], coords[:, 3]
    pred, seeds = neural(models, p, x, tau)
    rows = []
    for label, mask in bucket_masks(coords, p, P).items():
        if not mask.any(): continue
        for name, q in [('two_seed_mean', pred)] + [(f'seed_{s}', v) for s, v in zip(pc['seeds'], seeds)]:
            m = metrics(y[mask], q[mask], x[mask], tau[mask]); big = mask & (y > 1e-3)
            m['relative_price_RMSE_where_price_gt_1e-3'] = float(np.sqrt(np.mean(((q[big] - y[big]) / y[big]) ** 2))) if big.any() else None
            m['teacher_price_RMS'] = float(np.sqrt(np.mean(y[mask] ** 2)))
            rows.append({'bucket': label, 'model': name, **m})
    table = pd.DataFrame(rows); table.to_csv(dest / 'development_fidelity_by_bucket.csv', index=False)
    mean = table[(table.bucket == 'all') & (table.model == 'two_seed_mean')].iloc[0]
    ivr = mean.IV_RMSE_volatility_points
    gate = {'forward_price_RMSE': [float(mean.price_RMSE), gates['forward_price_RMSE_max']],
            'forward_price_P95': [float(mean.price_P95), gates['forward_price_P95_max']],
            'forward_price_max': [float(mean.price_max), gates['forward_price_max_max']],
            'IV_RMSE_decimal': [float(ivr) / 100 if ivr is not None and np.isfinite(ivr) else float('inf'), gates['IV_RMSE_max']]}
    passed = {k: bool(v[0] <= v[1]) for k, v in gate.items()}
    result = {'utc': stamp(), 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'manifest.json'),
              'scope': 'development split only (teacher seed 93102); held-out fidelity (93104) and PDE-fidelity (93105) points untouched',
              'checkpoint_status': status, 'gate_value_vs_limit': gate, 'gate_passed': passed,
              'all_gates_passed_on_development': all(passed.values()),
              'decision': 'PROCEED_TO_LOCK' if all(passed.values()) else 'DO_NOT_LOCK_DOCUMENT_FAILURE'}
    save(dest / 'development_check.json', result)
    e = abs(pred - y); fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    ax[0].scatter(365 * tau, np.maximum(e, 1e-12), s=2, alpha=.4); ax[0].axhline(gates['forward_price_max_max'], c='r', ls='--', label='max gate')
    ax[0].axhline(gates['forward_price_P95_max'], c='orange', ls='--', label='P95 gate')
    ax[0].set(xscale='log', yscale='log', xlabel='maturity (days)', ylabel='|PINN - exact| (forward units)', title='Development split: error vs maturity'); ax[0].legend()
    ax[1].scatter(np.exp(-x), np.maximum(e, 1e-12), s=2, alpha=.4); ax[1].set(yscale='log', xlabel='K/F', ylabel='|PINN - exact|', title='Development split: error vs moneyness')
    fig.savefig(dest / 'development_errors.png', dpi=140); plt.close(fig)
    print(table[table.model.eq('two_seed_mean')][['bucket', 'price_RMSE', 'price_P95', 'price_max', 'IV_RMSE_volatility_points', 'quotes']].to_string(index=False))
    print(result)


if __name__ == '__main__':
    torch.set_num_threads(1); main()
