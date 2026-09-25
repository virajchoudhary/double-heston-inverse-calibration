"""Descriptive development-split report for one trained PINN candidate. REPORTING ONLY.

Written after C1's development metrics were seen and before any C2 development result.
It has NO role in selection: `run.py select-candidate` applies the frozen ladder rule on its own.
Uses only the development split (teacher seed 93102) and its coordinates for the PDE residual;
the held-out fidelity (93104) and PDE-fidelity (93105) sets are never generated or read.
"""
import argparse
import numpy as np
import pandas as pd
import torch
from run import OUT, require_exact, read, save, sha, stamp, candidate_nets, neural, metrics, tensor, structural
from src.mentor_dh_pinn.regular_pinn_torch import residual


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('candidate'); cand = ap.parse_args().candidate
    c = require_exact(); pc = c['pinn']; g = pc['fidelity_gate']; P = np.array(c['published_double_slow_first'])
    dest = OUT / 'candidates' / cand; out = dest / 'development_report.json'; assert not out.exists()
    models = candidate_nets(c, cand)
    z = np.load(OUT / 'teacher/development.npz'); coords, p, y = z['coords'], z['params'], z['price']; x, tau = coords[:, 0], coords[:, 3]
    pred, seeds = neural(models, p, x, tau); days = 365 * tau; kf = np.exp(-x)
    masks = {'all': np.ones(len(y), bool)}
    for a, b in [(7, 14), (14, 30), (30, 90), (90, 365), (365, 731)]: masks[f'maturity {a}-{b}d'] = (days >= a) & (days < b)
    for a, b in [(0, .9), (.9, .98), (.98, 1.02), (1.02, 1.1), (1.1, 9)]: masks[f'K/F {a}-{b}'] = (kf >= a) & (kf < b)
    masks['ATM |K/F-1|<=.02'] = abs(kf - 1) <= .02; masks['wing K/F<.9 or >1.1'] = (kf < .9) | (kf > 1.1)
    masks['fast-heavy v_fast>v_slow'] = p[:, 9] > p[:, 4]; masks['slow-heavy v_slow>=v_fast'] = p[:, 9] <= p[:, 4]
    rows = []
    for label, m in masks.items():
        for name, q in [('two_seed_mean', pred)] + [(f'seed_{s}', v) for s, v in zip(pc['seeds'], seeds)]:
            rows.append({'bucket': label, 'model': name, **metrics(y[m], q[m], x[m], tau[m])})
    table = pd.DataFrame(rows); table.to_csv(dest / 'development_report_buckets.csv', index=False)
    pde = {}
    for s, net in zip(pc['seeds'], models):
        eq = [residual(net, tensor(coords[j:j + 128]), structural(p[j:j + 128]))[0].detach().numpy() for j in range(0, len(coords), 128)]
        eq = np.concatenate(eq); pde[str(s)] = {'scaled_PDE_residual_RMSE': float(np.sqrt(np.mean(eq ** 2))), 'P95_abs': float(np.quantile(abs(eq), .95))}
    diff = seeds[0] - seeds[1]; e0, e1 = seeds[0] - y, seeds[1] - y
    mean = table[(table.bucket == 'all') & (table.model == 'two_seed_mean')].iloc[0]
    completion = {str(s): {k: read(OUT / 'candidates' / cand / f'pinn_s{s}/completed.json')[k] for k in ['seconds', 'weights_sha256', 'post_lbfgs_parts', 'post_lbfgs_loss_on_lbfgs_subset']} for s in pc['seeds']}
    report = {'utc': stamp(), 'candidate': cand, 'spec': pc['candidates'][cand], 'source_sha256': sha(__file__), 'protocol_sha256': sha(OUT / 'manifest.json'),
              'role': 'descriptive only; selection is made by run.py select-candidate', 'completion': completion,
              'all_finite_predictions': bool(np.isfinite(pred).all()),
              'gate_multiples_two_seed_mean': {'price_RMSE': float(mean.price_RMSE) / g['forward_price_RMSE_max'], 'price_P95': float(mean.price_P95) / g['forward_price_P95_max'],
                                               'price_max': float(mean.price_max) / g['forward_price_max_max'], 'IV_RMSE': float(mean.IV_RMSE_volatility_points) / 100 / g['IV_RMSE_max']},
              'seed_variation': {'RMS_difference_between_seed_prices': float(np.sqrt(np.mean(diff ** 2))), 'max_abs_difference': float(abs(diff).max()),
                                 'correlation_of_seed_errors': float(np.corrcoef(e0, e1)[0, 1])},
              'PDE_residual_on_development_coordinates_by_seed': pde}
    save(out, report)
    print(table[table.model.eq('two_seed_mean')][['bucket', 'price_RMSE', 'price_MAE', 'price_P95', 'price_max', 'IV_RMSE_volatility_points', 'quotes']].to_string(index=False))
    print({k: v for k, v in report.items() if k in ['gate_multiples_two_seed_mean', 'seed_variation', 'PDE_residual_on_development_coordinates_by_seed', 'all_finite_predictions']})
    print(table[table.bucket.eq('all')][['model', 'price_RMSE', 'price_P95', 'price_max', 'IV_RMSE_volatility_points']].to_string(index=False))


if __name__ == '__main__':
    torch.set_num_threads(2); main()
