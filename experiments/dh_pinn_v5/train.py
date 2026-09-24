"""Train one ablation stage. Identical budget and data for every stage (see config.json).

python train.py A1            # ablation budget, seed 17
python train.py FINAL --full  # winning architecture at the locked 40k-step recipe, both seeds
"""
import argparse, json, time
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; V4 = HERE.parent / 'nifty_multifactor_v4'; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from v5_models import V5Pinn, count
from src.mentor_dh_pinn.regular_pinn_torch import residual

C = json.loads((HERE / 'config.json').read_text()); C4 = json.loads((V4 / 'config.json').read_text())['pinn']
NORM = C4['loss_normalisers']
STAGES = {                    # body, adaptive tanh, gradient balancing, RAD sampling
    'A0': ('plain', False, False, False), 'A1': ('residual', False, False, False), 'A2': ('residual', True, False, False),
    'A3': ('residual', True, True, False), 'A4': ('residual', True, True, True), 'A5': ('gated', True, True, True)}


def tensor(a): return torch.tensor(np.asarray(a), dtype=torch.float64)
def structural(p): return tensor(np.stack([p[:, 0:4], p[:, 5:9]], axis=1))


def load():
    t = np.load(V4 / 'artifacts' / 'teacher' / 'train.npz'); col = np.load(V4 / 'artifacts' / 'teacher' / 'collocation.npz')
    pool = np.load(HERE / 'artifacts' / 'data' / 'rad_pool.npz')
    return t, col, pool


def grad_norm(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))


def train(stage, seed, steps, out):
    body, adaptive, gradbal, rad = STAGES[stage] if stage in STAGES else STAGES[json.loads((HERE / 'artifacts' / 'ablation' / 'selection.json').read_text())['winner']]
    torch.set_num_threads(2); torch.manual_seed(seed)
    net = V5Pinn(body=body, adaptive=adaptive, width=C['ablation']['width'], depth=C['ablation']['depth'])
    with torch.no_grad(): net.head.weight.zero_(); net.head.bias.zero_()
    t, col, pool = load()
    coords, st, price = tensor(t['coords']), structural(t['params']), tensor(t['price'])
    valid = torch.tensor(t['iv_valid']); vol = tensor(np.where(t['iv_valid'], t['iv'], 0.))
    cc, cs = tensor(col['coords']), structural(col['params'])
    pool_c, pool_s = tensor(pool['coords']), structural(pool['params'])
    params = [p for p in net.parameters() if p.requires_grad]
    w = {'iv': 1., 'pde': NORM['pde_weight'], 'conv': NORM['convexity_weight']}
    hist, gh = [], []

    def parts_of(di, ci):
        z, s = coords[di], st[di]
        pred = net.price(z, s) * torch.exp(-z[:, 0]); pv = net.iv(z, s); good = valid[di]
        eq, dg = residual(net, cc[ci], cs[ci])
        return {'price': (pred - price[di]).square().mean(), 'iv': (pv[good] - vol[di][good]).square().mean(),
                'pde': eq.square().mean(), 'conv': torch.relu(-dg['convexity']).square().mean()}

    def total(parts):
        return (parts['price'] / NORM['price'] ** 2 + w['iv'] * parts['iv'] / NORM['iv'] ** 2
                + w['pde'] * parts['pde'] / NORM['pde'] ** 2 + w['conv'] * parts['conv'])

    opt = torch.optim.Adam(params, lr=C4['adam_lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    t0 = time.monotonic(); start_step = 0; elapsed = 0.
    ck = out / f'ckpt_s{seed}.pt'
    if ck.exists():                                  # crash resilience only: identical schedule, seed and budget
        ck_state = torch.load(ck, weights_only=False)
        net.load_state_dict(ck_state['model']); opt.load_state_dict(ck_state['opt']); sched.load_state_dict(ck_state['sched'])
        w.update(ck_state['w']); cc, cs = ck_state['cc'], ck_state['cs']; hist, gh = ck_state['hist'], ck_state['gh']
        start_step = ck_state['step'] + 1; elapsed = ck_state['elapsed']
        print('RESUMED', stage, seed, 'at step', start_step, flush=True)
    for step in range(start_step, steps):
        di = torch.randint(len(coords), (C4['label_batch'],)); ci = torch.randint(len(cc), (C4['pde_batch'],))
        parts = parts_of(di, ci); loss = total(parts)
        if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
        if gradbal and step % 100 == 0:                       # Wang/Teng/Perdikaris learning-rate annealing
            gp = grad_norm(parts['price'] / NORM['price'] ** 2, params)
            for k, scale in [('iv', NORM['iv'] ** 2), ('pde', NORM['pde'] ** 2), ('conv', 1.)]:
                gi = grad_norm(parts[k] / scale, params)
                if gi > 1e-30: w[k] = .9 * w[k] + .1 * min(gp / gi, 1e3)
            gh.append({'step': step, 'grad_price': gp, 'weights': dict(w)})
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if rad and step and step % 2000 == 0:                 # residual-adaptive resampling, same budget
            with torch.no_grad(): pass
            idx = torch.randint(len(pool_c), (40000,))
            r = torch.cat([residual(net, pool_c[idx[j:j + 2500]].clone().requires_grad_(True), pool_s[idx[j:j + 2500]])[0].detach().abs()
                           for j in range(0, len(idx), 2500)])      # chunked: same candidates, bounded memory
            half = len(cc) // 2
            prob = (r / r.mean() + 1.).numpy(); prob = prob / prob.sum()
            g = np.random.default_rng(seed + step).gumbel(size=len(prob))          # Gumbel top-k = weighted draw without replacement
            pick = np.argsort(-(np.log(prob) + g))[:half]
            unif = torch.randint(len(pool_c), (len(cc) - half,))
            cc = torch.cat([pool_c[idx][pick], pool_c[unif]]); cs = torch.cat([pool_s[idx][pick], pool_s[unif]])
        if step % 2000 == 1999 or step == steps - 1:
            out.mkdir(parents=True, exist_ok=True)
            torch.save({'step': step, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'w': dict(w), 'cc': cc, 'cs': cs, 'hist': hist, 'gh': gh, 'elapsed': elapsed + time.monotonic() - t0}, out / f'ckpt_s{seed}.pt')
        if step % 500 == 0 or step == steps - 1:
            hist.append({'step': step + 1, 'loss': float(loss.detach()), 'parts': {k: float(v.detach()) for k, v in parts.items()},
                         'weights': dict(w), 'seconds': elapsed + time.monotonic() - t0})
            print(stage, seed, hist[-1]['step'], f"{hist[-1]['loss']:.3e}", flush=True)
    di = torch.arange(C4['lbfgs_labels']); ci = torch.arange(min(C4['lbfgs_pde'], len(cc)))
    lb = torch.optim.LBFGS(params, max_iter=C4['lbfgs_steps'], line_search_fn='strong_wolfe', history_size=30)

    def closure():
        lb.zero_grad(set_to_none=True); v = total(parts_of(di, ci)); v.backward(); return v
    lb.step(closure)
    out.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), out / f'weights_s{seed}.pt')
    meta = {'stage': stage, 'seed': seed, 'body': body, 'adaptive': adaptive, 'gradbal': gradbal, 'rad': rad,
            'steps': steps, 'params': count(net), 'train_seconds': elapsed + time.monotonic() - t0, 'history': hist,
            'grad_history': gh, 'alpha': net.body.alpha.tolist() if hasattr(net.body, 'alpha') else None,
            'adaptive_a': [float(m.a) for m in net.modules() if hasattr(m, 'a') and isinstance(getattr(m, 'a'), torch.nn.Parameter)]}
    (out / f'meta_s{seed}.json').write_text(json.dumps(meta, indent=2, default=float) + '\n')
    (out / f'ckpt_s{seed}.pt').unlink(missing_ok=True)
    print('DONE', stage, seed, f"{meta['train_seconds']:.0f}s", meta['params'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--full', action='store_true'); ap.add_argument('--seed', type=int, default=17)
    a = ap.parse_args()
    steps = 40000 if a.full else 12000
    train(a.stage, a.seed, steps, HERE / 'artifacts' / ('final' if a.full else 'ablation') / a.stage)
