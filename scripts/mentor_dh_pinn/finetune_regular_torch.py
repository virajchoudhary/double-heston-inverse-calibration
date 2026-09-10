#!/usr/bin/env python3
"""Float64 neural-weight refinement of the existing regular pricing-PDE PINN.

Only archived independent training labels and the PDE enter gradients. Validation
IV selects weights; recovery-case truths/quotes are not read by this script.
"""
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, sha256
from src.mentor_dh_pinn.regular_pinn_data import coordinates
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def anchor_loss(model, q, g, dg, scale, coordinate_map=coordinates):
    query = q.detach().requires_grad_(True)
    c, p = coordinate_map(query, model.factors, torch)
    predicted = model.correction(c, p)
    jac = torch.autograd.grad(predicted.sum(), query, create_graph=True)[0][:, 2:]
    return (((predicted - g) / .05)**2).mean() + .2 * (((jac - dg) / scale)**2).mean()


@torch.no_grad()
def relevance_weights(model, q, coordinate_map=coordinates):
    c, p = coordinate_map(q, model.factors, torch)
    w = model.iv(c, p)**2 * c[:, -1]
    d2 = c[:, 0] / torch.sqrt(w) - .5 * torch.sqrt(w)
    return torch.clamp_min(torch.exp(-d2*d2/2), .01)


def physics_loss(model, q, weights, mean_weight, coordinate_map=coordinates, pde_weight=.2):
    c, p = coordinate_map(q, model.factors, torch)
    r, d = residual(model, c, p)
    ar = r.abs()
    huber = torch.where(ar < .1, .5*r*r, .1*(ar-.05)) / .005
    shape = torch.clamp_min(-d['convexity'], 0)**2 + torch.clamp_min(-d['w']*d['l_tau'], 0)**2
    # A fixed full-pool normalizer gives unbiased uniform-minibatch estimates.
    return pde_weight * (weights*huber).mean()/mean_weight + .05 * shape.mean()


@torch.no_grad()
def validation_rmse(model, q, target, chunk=1024, coordinate_map=coordinates):
    errors = []
    for start in range(0, len(q), chunk):
        c, p = coordinate_map(q[start:start+chunk], model.factors, torch)
        errors.append(model.iv(c, p) - target[start:start+chunk])
    value = torch.cat(errors)
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite validation predictions; no quotes dropped")
    return float(torch.sqrt((value**2).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=909111)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--pde-batch', type=int, default=128)
    ap.add_argument('--collocation', type=int, default=18000)
    ap.add_argument('--save-every', type=int, default=250)
    ap.add_argument('--component-domain', action='store_true')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--pde-weight', type=float, default=.2)
    ap.add_argument('--add-residual-blocks', type=int, default=0,
                    help='Explicit separate deeper architecture; each block adds two tanh hidden layers')
    args = ap.parse_args()
    if min(args.steps, args.batch, args.pde_batch, args.collocation, args.save_every) < 1:
        ap.error('All budgets must be positive')
    if not math.isfinite(args.lr) or args.lr < 2e-7 or not math.isfinite(args.pde_weight) or args.pde_weight <= 0:
        ap.error('Finite learning rate >=2e-7 and strictly positive PDE weight required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    initial, info = load_checkpoint(args.checkpoint)
    model = (initial if isinstance(initial, TorchRegularVariancePINN)
             else TorchRegularVariancePINN.from_mlx(initial))
    if args.add_residual_blocks:
        model = model.deepened(args.add_residual_blocks)
    model.train().requires_grad_(True)
    from src.mentor_dh_pinn.component_pinn_data import component_coordinates, DOMAIN
    coordinate_map = component_coordinates if args.component_domain else coordinates
    if args.component_domain and model.factors != 1:
        raise ValueError('Component training requires a single-factor price PINN')
    names = ('train.npz', 'validation.npz', 'collocation.npz')
    data_hashes = {n: sha256(args.data/n) for n in names}
    train, validation, pool = [dict(np.load(args.data/n)) for n in names]
    data_manifest = json.loads((args.data/'manifest.json').read_text())
    if bool(data_manifest.get('component_domain', False)) != args.component_domain:
        raise ValueError('Training coordinate map must match archived data domain')
    label_counts = {}
    for name, dataset in [('train', train), ('validation', validation)]:
        mask = dataset['usable'].astype(bool)
        label_counts[name] = {'candidates': len(mask), 'usable': int(mask.sum()),
                              'excluded_reference_labels': int((~mask).sum())}
        if args.component_domain:
            dataset['archive_index'] = np.flatnonzero(mask)
            for key in list(dataset):
                if key != 'archive_index':
                    dataset[key] = dataset[key][mask]
    for dataset in (train, validation):
        if not dataset['usable'].all():
            raise ValueError('This declared clean DH trial requires every archived label usable')
        if not all(np.isfinite(dataset[k]).all() for k in ('q', 'g', 'w', 'dg_du')):
            raise ValueError('Nonfinite labels; refusing silent filtering')
        if (dataset['w'] <= 0).any():
            raise ValueError('Nonpositive total variance label')
        if len(np.unique(dataset['q'], axis=0)) != len(dataset['q']):
            raise ValueError('Duplicate sampled states')
    combined = np.concatenate([train['q'], validation['q']])
    if len(np.unique(combined, axis=0)) != len(combined):
        raise ValueError('Training/validation exact-state overlap')
    if not np.isfinite(pool['q']).all():
        raise ValueError('Nonfinite collocation coordinates')
    if args.batch > len(train['q']) or args.pde_batch > args.collocation:
        raise ValueError('Minibatch exceeds the declared data/pool size')
    indices = rng.choice(len(pool['q']), args.collocation, replace=False)
    q, g, dg = [tensor(train[k]) for k in ('q', 'g', 'dg_du')]
    pq = tensor(pool['q'][indices])
    scale = tensor(np.maximum(np.sqrt(np.mean(train['dg_du']**2, axis=0)), .02))
    pw = torch.cat([relevance_weights(model, pq[i:i+1024], coordinate_map) for i in range(0, len(pq), 1024)])
    mean_weight = pw.mean()
    vq = tensor(validation['q'])
    target = tensor(np.sqrt(validation['w']/np.exp(validation['q'][:, 1])))
    train_audit_count = min(16384, len(q))
    train_target = tensor(np.sqrt(train['w'][:train_audit_count]/np.exp(train['q'][:train_audit_count, 1])))
    architecture = {k: getattr(model, k) for k in (
        'factors', 'width', 'depth', 'tau_min', 'tau_max', 'x_half_width', 'correction_limit', 'residual_blocks')}
    cfg = {**architecture, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           'checkpoint_format': 'torch_state_dict_float64',
           'precision': 'float64 weights, values, gradients and AdamW moments; scalar step counter uses backend default',
           'architecture': ('deeper residual regular implied-variance pricing-PDE PINN' if model.residual_blocks
                            else 'unchanged regular implied-variance pricing-PDE PINN'),
           'total_hidden_layers': model.total_hidden_layers,
           'trainable_parameter_count': sum(p.numel() for p in model.parameters()),
           'training_iv_audit': f'first {train_audit_count} usable archived training rows; reporting only, not selection',
           'parameter_domain': DOMAIN if args.component_domain else 'original regular PINN domain',
           'optimizer': 'AdamW', 'lr': args.lr, 'min_lr': 2e-7, 'weight_decay': 1e-6,
           'sensitivity': .2, 'weight_pde': args.pde_weight, 'weight_shape': .05, 'anchor_scale': .05,
           'skip_recovery': True, 'selection': 'minimum archived validation IV RMSE; initial weights eligible',
           'resume': str(args.checkpoint), 'gradient_clipping': False}
    sources = [Path(__file__), ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
               ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py', ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
               ROOT/'src/mentor_dh_pinn/regular_pinn.py']
    if args.component_domain:
        sources.append(ROOT/'src/mentor_dh_pinn/component_pinn_data.py')
    manifest = {'initial_checkpoint': info, 'input_sha256': data_hashes,
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                'train_quotes': len(q), 'validation_quotes': len(vq), 'collocation_points': len(pq),
                'reference_label_counts': label_counts,
                'selection_scope': 'previously used development validation, not unseen generalization evidence',
                'training_information': 'archived correction and parameter-derivative labels plus price PDE; no recovery truth',
                'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                            'device': 'cpu', 'threads': 1}}
    write = lambda name, value: (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False))
    write('config.json', cfg)
    write('manifest.json', manifest)
    write('source_snapshot.json', {str(p.relative_to(ROOT)): p.read_text() for p in sources})
    np.savez_compressed(args.out/'fixed_training_state.npz', collocation_indices=indices,
                        sensitivity_scale=scale.numpy(), relevance_weights=pw.numpy(),
                        training_archive_indices=train.get('archive_index', np.arange(len(q))),
                        validation_archive_indices=validation.get('archive_index', np.arange(len(vq))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=2e-7)
    history, best, selected_step = [], math.inf, None
    visits = np.zeros(len(q), dtype=np.int64)
    started = time.perf_counter()

    def record(step, losses=None):
        nonlocal best, selected_step
        score = validation_rmse(model, vq, target, coordinate_map=coordinate_map)
        row = {'step': step, 'validation_iv_rmse': score, 'last_minibatch_losses': losses,
               'training_subset_iv_rmse': validation_rmse(model, q[:train_audit_count], train_target, coordinate_map=coordinate_map),
               'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-started}
        history.append(row)
        torch.save(model.state_dict(), args.out/f'step_{step:06d}.pt')
        if score < best:
            best, selected_step = score, step
            torch.save(model.state_dict(), args.out/'model.pt')
            write('selection.json', {'step': step, 'score': score, 'metric': 'validation_iv_rmse'})
        write('history.json', history)
        print(json.dumps(row), flush=True)

    try:
        record(0)
        for step in range(1, args.steps+1):
            ai = rng.choice(len(q), args.batch, replace=False)
            pi = rng.choice(len(pq), args.pde_batch, replace=False)
            visits[ai] += 1
            optimizer.zero_grad(set_to_none=True)
            anchor = anchor_loss(model, q[ai], g[ai], dg[ai], scale, coordinate_map)
            anchor.backward()
            physics = physics_loss(model, pq[pi], pw[pi], mean_weight, coordinate_map, args.pde_weight)
            physics.backward()
            if not torch.isfinite(anchor+physics) or any(
                    p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Nonfinite loss/gradient at step {step}')
            optimizer.step()
            scheduler.step()
            if step % args.save_every == 0 or step == args.steps:
                record(step, {'anchor_and_sobolev': float(anchor.detach()), 'physics_and_shape': float(physics.detach())})
        torch.save(model.state_dict(), args.out/'last.pt')
        torch.save(optimizer.state_dict(), args.out/'last_optimizer.pt')
        np.savez_compressed(args.out/'training_visits.npz', counts=visits)
        for n, digest in data_hashes.items():
            assert sha256(args.data/n) == digest, f'Changed training data: {n}'
        for name, digest in manifest['source_sha256'].items():
            assert sha256(ROOT/name) == digest, f'Changed source during training: {name}'
        assert sha256(info['checkpoint']) == info['sha256'], 'Changed initial checkpoint'
        assert sha256(Path(info['checkpoint']).parent/'config.json') == info['config_sha256']
        write('complete.json', {'steps': args.steps, 'selected_step': selected_step,
              'selected_validation_iv_rmse': best, 'checkpoint_sha256': sha256(args.out/'model.pt'),
              'distinct_training_quotes_visited': int((visits > 0).sum()),
              'seconds': time.perf_counter()-started, 'status': 'training complete; recovery not yet assessed'})
    except Exception as exc:
        write('FAILED.json', {'error': f'{type(exc).__name__}: {exc}', 'history_records': len(history)})
        raise


if __name__ == '__main__':
    main()
