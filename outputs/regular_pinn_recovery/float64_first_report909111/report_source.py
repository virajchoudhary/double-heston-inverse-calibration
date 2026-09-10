#!/usr/bin/env python3
"""Audit exposed-12 regular-PINN continuations and replay holdout isolation.

No training, reference pricing, parameter tuning or checkpoint selection occurs.
"""
import argparse
import csv
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.mentor_dh_pinn.assess_regular_pinn as assessor
import src.mentor_dh_pinn.regular_pinn_data as data
import src.mentor_dh_pinn.torch_pricer as pricer
from scripts.mentor_dh_pinn.finetune_regular_torch import tensor, validation_rmse


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def without_seconds(value):
    if isinstance(value, dict):
        return {k: without_seconds(v) for k, v in value.items() if k != 'seconds'}
    if isinstance(value, list):
        return [without_seconds(v) for v in value]
    return value


def collect(directory):
    manifest = read_json(directory/'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['evaluated_geometries'] == ['rich'] and manifest['evaluated_noise_levels'] == [0.]
    assert manifest['cases_per_model_family'] == 12
    assert len(manifest['checkpoints']) == 1
    info = manifest['checkpoints'][0]
    fits = read_json(directory/'fits_and_starts.json')
    params = read_csv(directory/'parameter_recovery.csv')
    metrics = [r for r in read_csv(directory/'metrics.csv') if r['model'] == info['label']]
    truths = read_json(directory/'synthetic_truth.json')
    observations = read_csv(directory/'observations.csv')
    assert len(metrics) == len(truths) == 12 and len(observations) == 12*126
    assert len({r['case'] for r in metrics}) == 12
    assert len({(r['case'], r['quote']) for r in observations}) == len(observations)
    assert len({(r['case'], r['factor'], r['parameter']) for r in params}) == len(params)
    checks, matrix, worst = {}, np.zeros((12, 10), dtype=bool), np.full(12, np.inf)
    order = ('kappa', 'theta', 'sigma', 'rho', 'v0')
    for i, truth in enumerate(truths):
        rows = [r for r in params if r['case'] == truth['case']]
        errors = []
        for r in rows:
            j = 5*(int(r['factor'])-1)+order.index(r['parameter'])
            actual, estimate = float(r['truth']), float(r['estimate'])
            assert actual == truth['physical'][j]
            gate = abs(estimate-actual)/(.05 if r['parameter'] == 'rho' else .05*abs(actual))
            passed = gate <= 1
            assert passed == (r['parameter_pass'] == 'True')
            matrix[i, j] = passed
            errors.append(gate)
        if len(rows) == 10:
            worst[i] = max(errors)
        metric = next(r for r in metrics if r['case'] == truth['case'])
        assert (metric['parameter_gate'] == 'True') == bool(matrix[i].all())
    for fit in fits:
        assert len(fit['starts']) == 5
        if fit['status'] != 'fitted':
            continue
        assert fit['calibration_iv_sse'] == min(s['sse'] for s in fit['starts'] if s.get('sse') is not None)
        np.testing.assert_array_equal(data.decode_unit(np.asarray(fit['unit']), 2), fit['physical'])
    checks['parameter_gates_counts_and_start_selection'] = True
    checks['checkpoint_hash'] = assessor.sha256(info['checkpoint']) == info['sha256']
    checks['config_hash'] = assessor.sha256(Path(info['checkpoint']).parent/'config.json') == info['config_sha256']
    assert all(checks.values()), checks
    return {'directory': str(directory), 'label': info['label'], 'manifest': manifest,
            'fits': fits, 'truths': truths, 'observations': observations, 'parameters': params,
            'checks': checks, 'matrix': matrix.tolist(), 'worst': worst.tolist(),
            'individual_passes': int(matrix.sum()), 'complete_passes': int(matrix.all(axis=1).sum()),
            'neural_price_passes': sum(r['neural_price_gate'] == 'True' for r in metrics),
            'exact_price_passes': sum(r['exact_price_gate'] == 'True' for r in metrics),
            'joint_passes': sum(r['joint_recovery_gate'] == 'True' for r in metrics),
            'invalid_neural_iv_quotes': sum(int(r.get('neural_invalid_iv_quotes') or 0) for r in metrics),
            'input_sha256': {n: assessor.sha256(directory/n) for n in (
                'manifest.json', 'metrics.csv', 'parameter_recovery.csv', 'fits_and_starts.json',
                'observations.csv', 'synthetic_truth.json', 'summary.json')}}


def audit_training_and_isolation(result, baseline):
    info = result['manifest']['checkpoints'][0]
    run = Path(info['checkpoint']).parent
    manifest = read_json(run/'manifest.json')
    config = read_json(run/'config.json')
    history = read_json(run/'history.json')
    selection = read_json(run/'selection.json')
    assert selection['metric'] == 'validation_iv_rmse'
    chosen = min(history, key=lambda r: r['validation_iv_rmse'])
    assert chosen['step'] == selection['step'] and chosen['validation_iv_rmse'] == selection['score']
    state = torch.load(run/'model.pt', weights_only=True)
    saved = torch.load(run/f'step_{selection["step"]:06d}.pt', weights_only=True)
    assert state.keys() == saved.keys()
    assert all(v.dtype == torch.float64 and torch.equal(v, saved[k]) for k, v in state.items())
    snapshots = read_json(run/'source_snapshot.json')
    assert all(hashlib.sha256(snapshots[n].encode()).hexdigest() == h for n, h in manifest['source_sha256'].items())
    dataset = Path(config['data'])
    assert all(assessor.sha256(dataset/n) == h for n, h in manifest['input_sha256'].items())
    parent = manifest['initial_checkpoint']
    assert assessor.sha256(parent['checkpoint']) == parent['sha256']
    network, _ = assessor.load_checkpoint(run)
    validation = dict(np.load(dataset/'validation.npz'))
    score = validation_rmse(network, tensor(validation['q']), tensor(np.sqrt(validation['w']/np.exp(validation['q'][:, 1]))))
    assert score == selection['score']
    assert result['truths'] == baseline['truths']
    assert result['observations'] == baseline['observations']
    for fit, original in zip(result['fits'], baseline['fits']):
        assert fit['case'] == original['case']
        assert [r['initial_unit'] for r in fit['starts']] == [r['initial_unit'] for r in original['starts']]
    units = np.array([r['unit'] for r in result['truths']])
    overlaps = {}
    for split in ('train', 'validation'):
        rows = np.load(dataset/f'{split}.npz')['q'][:, 2:]
        overlaps[split] = sum(bool(np.any(np.all(rows == u, axis=1))) for u in units)
    assert not any(overlaps.values())
    replays = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Reference pricing is forbidden in inverse-fit isolation replay')

    with ExitStack() as guards:
        for module, name in ((assessor, '_exact'), (assessor, 'exact_prices'),
                             (data, 'exact_prices'), (data, 'teacher_labels'),
                             (pricer, 'price_call'), (pricer, 'price_call_single')):
            guards.enter_context(patch.object(module, name, forbidden))
        for i, recorded in enumerate(result['fits']):
            quotes = [r for r in result['observations'] if r['case'] == recorded['case']]
            quotes.sort(key=lambda r: int(r['quote']))
            holdout = np.array([r['holdout'] == 'True' for r in quotes])
            assert holdout.sum() == 42
            x = np.log(1.0/np.array([float(r['strike']) for r in quotes]))
            tau = np.array([float(r['tau']) for r in quotes])
            iv = np.array([float(r['observed_iv']) for r in quotes])
            x[holdout], tau[holdout], iv[holdout] = np.nan, -1e99, np.inf
            replay = assessor.fit_network(network, x, tau, iv, fit_mask=~holdout,
                starts=5, max_nfev=400, seed=result['manifest']['seed']+100+i)
            expected = {k: recorded[k] for k in replay}
            assert without_seconds(replay) == without_seconds(expected), recorded['case']
            replays.append({'case': recorded['case'], 'all_fit_fields_identical_except_seconds': True})
    assert all(torch.equal(state[k], network.state_dict()[k]) for k in state)
    return {'selected_step': selection['step'], 'validation_iv_rmse': score,
            'initial_validation_iv_rmse': history[0]['validation_iv_rmse'],
            'checkpoint_tensors_exactly_match_selected_step': True,
            'training_source_snapshots_and_data_hashes_match': True,
            'identical_baseline_truths_quotes_and_starts': True,
            'exact_truth_overlap_counts': overlaps, 'holdout_corruption_replays': replays,
            'reference_pricer_runtime_guards_passed': True, 'frozen_weights_unchanged': True,
            'training_history': history}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment', type=Path, action='append', required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = collect(args.baseline)
    results = [baseline]
    for directory in args.assessment:
        result = collect(directory)
        result['training_audit'] = audit_training_and_isolation(result, baseline)
        results.append(result)
        print(json.dumps({'run': result['label'], 'individual': result['individual_passes'],
                          'complete': result['complete_passes'], 'isolation_cases': 12}), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    names = ['k1', 'th1', 'sig1', 'rho1', 'v01', 'k2', 'th2', 'sig2', 'rho2', 'v02']
    counts = np.array([np.asarray(r['matrix']).sum(axis=0) for r in results])
    axes[0].imshow(counts, vmin=0, vmax=12, cmap='Blues', aspect='auto')
    axes[0].set_yticks(range(len(results)), [r['label'] for r in results], fontsize=7)
    axes[0].set_xticks(range(10), names, rotation=45)
    for i in range(len(results)):
        for j in range(10):
            axes[0].text(j, i, f'{counts[i,j]}/12', ha='center', va='center', fontsize=7)
    axes[0].set_title('Individual parameter passes (not full-case success)')
    for r in results:
        axes[1].scatter(np.arange(12), r['worst'], label=r['label'], s=25, alpha=.75)
    axes[1].axhline(1, color='crimson', ls='--', label='Every parameter must be at/below 1')
    axes[1].set_yscale('log'); axes[1].set_xticks(range(12))
    axes[1].set_xlabel('Previously exposed case ID'); axes[1].set_ylabel('Worst parameter error / its tolerance')
    axes[1].legend(fontsize=6); axes[1].grid(alpha=.2)
    fig.savefig(args.out/'recovery_comparison.png', dpi=160); plt.close(fig)
    lines = ['# Regular-PINN float64 continuation — development evidence', '',
        'These are previously exposed synthetic Double Heston cases, not an unseen generalization test.', '',
        '| Checkpoint | Individual parameters | All ten | Neural price gate | Exact repricing | Joint |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r["label"]} | {r["individual_passes"]}/120 | {r["complete_passes"]}/12 | {r["neural_price_passes"]}/12 | {r["exact_price_passes"]}/12 | {r["joint_passes"]}/12 |')
    lines += ['', '![Recovery comparison](recovery_comparison.png)', '',
        'Interpretation: separate passing parameters do not add up to a successful case. A dot above 1 means at least one parameter failed its unchanged tolerance.', '',
        '## Training and integrity', '',
        'The architecture remains the regular five-hidden-layer, 160-unit tanh implied-variance price-PDE PINN. Training reuses 131,072 synthetic price/correction and parameter-derivative labels, 16,384 validation quotes and 18,000 collocation points. This is a conditional forward PINN followed by frozen-neural inverse calibration, not a direct parameter encoder. Synthetic sensitivity supervision is disclosed; prices and IV are not independent observations.', '',
        'Each recovery case uses 21 strike/forward ratios (0.8–1.2) at 30, 60, 90, 180, 365 and 730 days divided by 365. Only 84 quotes enter fitting; 42 strikes are withheld. This is synthetic maturity coverage, not a claim about available NSE expiries. Eight positive parameters require <=5% relative error; two correlations require <=0.05 absolute error. Canonical storage is slow factor first.', '',
        'Weights were selected solely by validation IV RMSE, including the initial checkpoint. Training-data hashes, source snapshots, selected tensors and the validation score were verified. Baseline truths, quotes and sixty blind starts are identical. Each actual fit was replayed after corrupting every held-out input while blocking six reference-pricer entry points: every fit field except elapsed time was identical. This is bounded leakage evidence, not a universal guarantee.', '']
    for r in results[1:]:
        a = r['training_audit']
        lines.append(f'- {r["label"]}: selected step {a["selected_step"]}; validation IV RMSE {a["initial_validation_iv_rmse"]:.10g} → {a["validation_iv_rmse"]:.10g}; invalid assessed neural IV quotes: {r["invalid_neural_iv_quotes"]}.')
    lines += ['', '## Limits and next work', '',
        'No complete-case recovery should be claimed unless the all-ten and joint columns actually pass. Lower validation price/IV error alone is insufficient. These are sequential warm starts, not independent initializations or a dtype-only causal ablation. Fresh sealed cases, noise tests and a matched Single-Heston comparison remain necessary. No NSE, PostgreSQL or Kaggle data were changed.', '',
        '[Full parameter table](PARAMETERS.md). Full fit/start records and input hashes are retained in audit.json.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    last = results[-1]
    table = ['# Latest checkpoint: all 120 parameter estimates', '',
             'Slow factor 1, fast factor 2. Previously exposed development cases.', '',
             '| Case | Factor | Parameter | Truth | Estimate | Pass |', '|---|---:|---|---:|---:|---|']
    for r in last['parameters']:
        table.append(f'| {r["case"]} | {r["factor"]} | {r["parameter"]} | {float(r["truth"]):.10g} | {float(r["estimate"]):.10g} | {r["parameter_pass"]} |')
    (args.out/'PARAMETERS.md').write_text('\n'.join(table)+'\n')
    (args.out/'audit.json').write_text(json.dumps({'results': results,
        'report_source_sha256': assessor.sha256(Path(__file__))}, indent=2, allow_nan=False))
    (args.out/'report_source.py').write_text(Path(__file__).read_text())


if __name__ == '__main__':
    main()
