"""Fidelity of any v5 checkpoint (or the locked v4 C3) against the exact Fourier teacher."""
import json, time
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; V4 = HERE.parent / 'nifty_multifactor_v4'; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from v5_models import V5Pinn, count
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN, residual
from train import STAGES, structural, tensor
sys.path.insert(0, str(V4)); from literature_exact import iv as exact_iv

DATA = HERE / 'artifacts' / 'data'
BUCKETS = [(7, 30, '7-30d'), (30, 90, '30-90d'), (90, 365, '90-365d'), (365, 731, '>365d')]
MB = [(-1, -.15, 'x<-0.15'), (-.15, -.05, '-0.15..-0.05'), (-.05, .05, 'ATM'), (.05, .15, '0.05..0.15'), (.15, 1, 'x>0.15')]


def locked():
    nets = []
    for s in [17, 43]:
        n = TorchRegularVariancePINN(factors=2, width=256, depth=5, tau_min=7 / 365, tau_max=2., x_half_width=.36)
        n.load_state_dict(torch.load(V4 / 'artifacts' / 'candidates' / 'C3' / f'pinn_s{s}' / 'weights.pt')); n.eval(); n.requires_grad_(False); nets.append(n)
    return nets


def load_stage(stage, folder='ablation'):
    d = HERE / 'artifacts' / folder / stage; nets = []
    for f in sorted(d.glob('weights_s*.pt')):
        meta = json.loads((d / f.name.replace('weights', 'meta').replace('.pt', '.json')).read_text())
        n = V5Pinn(body=meta['body'], adaptive=meta['adaptive'], width=256, depth=5)
        n.load_state_dict(torch.load(f)); n.eval(); n.requires_grad_(False); nets.append(n)
    return nets


def predict(nets, coords, params):
    co, st = tensor(coords), structural(params)
    with torch.no_grad():
        p = np.mean([n.price(co, st).numpy() for n in nets], 0) * np.exp(-coords[:, 0])   # forward-normalised c = C/F
    return p


def fidelity(nets, dataset):
    d = np.load(DATA / f'{dataset}.npz'); coords, params, price = d['coords'], d['params'], d['price']
    t0 = time.monotonic(); p = predict(nets, coords, params); latency = (time.monotonic() - t0) / len(price) * 1e6
    e = p - price; tau, x = coords[:, -1], coords[:, 0]
    mi = exact_iv(np.clip(p, 1e-12, None), x, tau); ok = d['iv_valid'] & np.isfinite(mi)
    iv_err = 100 * (mi[ok] - d['iv'][ok])
    out = {'points': len(price), 'price_RMSE': float(np.sqrt(np.mean(e ** 2))), 'price_MAE': float(np.mean(abs(e))),
           'price_P95': float(np.quantile(abs(e), .95)), 'price_max': float(np.max(abs(e))),
           'IV_RMSE_volpts': float(np.sqrt(np.mean(iv_err ** 2))), 'IV_P95_volpts': float(np.quantile(abs(iv_err), .95)),
           'IV_max_volpts': float(np.max(abs(iv_err))), 'IV_valid': int(ok.sum()), 'latency_us_per_quote': latency}
    days = tau * 365
    out['by_maturity'] = {lab: {'price_RMSE': float(np.sqrt(np.mean(e[m] ** 2))), 'n': int(m.sum()),
                                'IV_RMSE_volpts': float(np.sqrt(np.mean((100 * (mi[m & ok] - d['iv'][m & ok])) ** 2))) if (m & ok).any() else None}
                          for a, b, lab in BUCKETS for m in [(days >= a) & (days < b)]}
    out['by_moneyness'] = {lab: {'price_RMSE': float(np.sqrt(np.mean(e[m] ** 2))), 'n': int(m.sum())}
                           for a, b, lab in MB for m in [(x >= a) & (x < b)] if m.any()}
    return out


def pde(nets, dataset='pde_final', n=2048):
    d = np.load(DATA / f'{dataset}.npz'); co = tensor(d['coords'][:n]).requires_grad_(True); st = structural(d['params'][:n])
    vals = []
    for net in nets:
        for p in net.parameters(): p.requires_grad_(True)
        r, _ = residual(net, co.clone().requires_grad_(True), st); vals.append(float(r.detach().square().mean().sqrt()))
        for p in net.parameters(): p.requires_grad_(False)
    return {'scaled_PDE_residual_RMSE_per_seed': vals, 'mean': float(np.mean(vals))}


def report(which, folder='ablation', datasets=('development',)):
    nets = locked() if which == 'LOCKED_C3' else load_stage(which, folder)
    out = {'model': which, 'nets': len(nets), 'params': sum(p.numel() for p in nets[0].parameters())}
    d = HERE / 'artifacts' / folder / which
    if which != 'LOCKED_C3':
        metas = [json.loads(f.read_text()) for f in sorted(d.glob('meta_s*.json'))]
        out.update({'train_seconds': sum(m['train_seconds'] for m in metas), 'steps': metas[0]['steps'],
                    'alpha': metas[0]['alpha'], 'adaptive_a': metas[0]['adaptive_a'], 'stage_flags': {k: metas[0][k] for k in ['body', 'adaptive', 'gradbal', 'rad']}})
    for ds in datasets: out[ds] = fidelity(nets, ds)
    out['pde'] = pde(nets)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--folder', default='ablation')
    ap.add_argument('--datasets', nargs='+', default=['development']); ap.add_argument('--out', default=None)
    a = ap.parse_args(); res = {m: report(m, a.folder, a.datasets) for m in a.models}
    if a.out: Path(a.out).write_text(json.dumps(res, indent=2, default=float) + '\n')
    for m, r in res.items():
        for ds in a.datasets:
            print(f"{m:12} {ds:16} params {r['params']:,} price RMSE {r[ds]['price_RMSE']:.3e}  P95 {r[ds]['price_P95']:.3e}  max {r[ds]['price_max']:.3e}  IV RMSE {r[ds]['IV_RMSE_volpts']:.4f}  PDE {r['pde']['mean']:.4f}")
