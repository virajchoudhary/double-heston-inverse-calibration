"""Track B: fixed-parameter pricing prior + a tiny bounded market-residual head.

The SAME head, optimiser, data and budget are given to the Black-Scholes, Single Heston and
Double Heston priors, so the only difference is the prior. Structural parameters never change.
"""
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy import stats

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; ROOT = EXP.parents[0]
sys.path.insert(0, str(EXP / 'btc_multifactor_v1')); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from engine import exact, black, iv as invert_iv
from v5_models import V5Pinn
from train import structural, tensor

C = json.loads((HERE / 'config.json').read_text())
HC = EXP / 'hardcoded_v1'
PAR = json.loads((HC / 'config.json').read_text())['parameters']
DH = np.array(PAR['DH_PUB']); SH = np.array(PAR['SH_PUB']); BSVOL = float(np.sqrt(DH[4] + DH[9]))
OUT = HERE / 'artifacts' / 'track_b'; PRIORS = ['BS', 'SH', 'DH', 'DH_PINN']


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scaled(p, kind, s):
    p = p.copy()
    if kind == 'SH': p[[1, 4]] *= s; p[2] *= np.sqrt(s)
    else: p[[1, 6, 4, 9]] *= s; p[[2, 7]] *= np.sqrt(s)
    return p


def prior_c(kind, x, tau, s, net=None):
    """Forward-normalised prior call price."""
    if kind == 'BS': return black(x, tau, BSVOL * np.sqrt(s))
    if kind == 'SH': return exact(scaled(SH, 'SH', s), x, tau, feller=False)
    if kind == 'DH': return exact(scaled(DH, 'DH', s), x, tau, feller=False)
    p = scaled(DH, 'DH', s); co = tensor(np.column_stack([x, np.full(len(x), p[4]), np.full(len(x), p[9]), tau]))
    st = structural(p[None]).expand(len(x), 2, 4)
    with torch.no_grad(): return np.mean([n.price(co, st).numpy() for n in net], 0) * np.exp(-x)


def split():
    q = pd.read_csv(HC / 'artifacts' / 'spx_surface.csv')
    q['expiry_rank'] = q.groupby('expiry').ngroup()
    order = q.groupby('expiry').tau.median().sort_values().index.tolist()
    q['expiry_rank'] = q.expiry.map({e: i for i, e in enumerate(order)})
    q = q.sort_values(['expiry_rank', 'strike']).reset_index(drop=True)
    q['strike_rank'] = q.groupby('expiry_rank').cumcount()
    q['set'] = np.where((q.expiry_rank + q.strike_rank) % 2 == 0, 'calibration', 'heldout')
    q.loc[(q.set == 'calibration') & (q.strike_rank % 5 == 0), 'set'] = 'development'
    q.to_csv(OUT / 'spx_split.csv', index=False)
    audit = {'source_csv_sha256': sha(HC / 'artifacts' / 'spx_surface.csv'), 'rule': C['track_b']['split'],
             'counts': q.set.value_counts().to_dict(), 'expiries': int(q.expiry.nunique()), 'split_sha256': sha(OUT / 'spx_split.csv')}
    (OUT / 'split_audit.json').write_text(json.dumps(audit, indent=2) + '\n'); print(audit['counts']); return q


class Head(nn.Module):
    """4 -> 64 -> 64 -> 1, bounded output. Identical for every prior."""

    def __init__(self, dmax):
        super().__init__(); self.dmax = dmax
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)); self.double()
        with torch.no_grad(): self.net[-1].weight.zero_(); self.net[-1].bias.zero_()

    def forward(self, z): return self.dmax * torch.tanh(self.net(z)[:, 0])


def features(q, prior_iv, mu=None, sd=None):
    z = np.column_stack([q.x.to_numpy(), np.log(q.tau.to_numpy()), prior_iv, q.x.to_numpy() / np.sqrt(q.tau.to_numpy())])
    if mu is None: mu, sd = z.mean(0), z.std(0) + 1e-12
    return (z - mu) / sd, mu, sd


def fit_scale(kind, q, net=None):
    x, t, y, w = (q[k].to_numpy() for k in ['x', 'tau', 'c', 'vega_w'])
    r = minimize_scalar(lambda s: float(np.mean(((prior_c(kind, x, t, s, net) - y) / w) ** 2)), bounds=(0.7, 3.2), method='bounded', options={'xatol': 1e-6})
    return float(r.x)


def train_head(tr, dev, prior_iv_tr, prior_iv_dev, dmax, steps=4000, seed=7):
    keep_tr = np.isfinite(prior_iv_tr) & np.isfinite(tr.market_iv.to_numpy())
    keep_dev = np.isfinite(prior_iv_dev) & np.isfinite(dev.market_iv.to_numpy())
    tr, prior_iv_tr = tr[keep_tr], prior_iv_tr[keep_tr]; dev, prior_iv_dev = dev[keep_dev], prior_iv_dev[keep_dev]
    torch.manual_seed(seed); ztr, mu, sd = features(tr, prior_iv_tr); zdev, *_ = features(dev, prior_iv_dev, mu, sd)
    head = Head(dmax); opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    Ztr, Zdev = tensor(ztr), tensor(zdev)
    ytr, ydev = tensor(tr.market_iv.to_numpy()), tensor(dev.market_iv.to_numpy())
    ptr, pdev = tensor(prior_iv_tr), tensor(prior_iv_dev)
    best, best_state, hist = np.inf, {k: v.clone() for k, v in head.state_dict().items()}, []
    for step in range(steps):
        d = head(Ztr); loss = ((ptr + d - ytr) ** 2).mean() + .05 * (d ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad(): dv = float(((pdev + head(Zdev) - ydev) ** 2).mean().sqrt()) * 100
            hist.append({'step': step, 'train_loss': float(loss), 'dev_IV_RMSE': dv})
            if dv < best: best, best_state = dv, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, mu, sd, best, hist


def evaluate():
    OUT.mkdir(parents=True, exist_ok=True)
    q = split() if not (OUT / 'spx_split.csv').exists() else pd.read_csv(OUT / 'spx_split.csv')
    net = None
    fin = HERE / 'artifacts' / 'final'
    if (fin / 'FINAL').exists():
        from evaluate import load_stage; net = load_stage('FINAL', 'final')
    cal = q[q.set != 'heldout']; tr = q[q.set == 'calibration']; dev = q[q.set == 'development']; ho = q[q.set == 'heldout']
    res = {'counts': {k: int(v) for k, v in q.set.value_counts().items()}, 'priors': {}}
    for kind in PRIORS:
        if kind == 'DH_PINN' and net is None: continue
        s = fit_scale(kind, cal, net)                                   # calibration-only level fit
        pr = {name: invert_iv(prior_c(kind, g.x.to_numpy(), g.tau.to_numpy(), s, net), g.x.to_numpy(), g.tau.to_numpy())
              for name, g in [('train', tr), ('dev', dev), ('ho', ho)]}
        base = {name: float(np.sqrt(np.nanmean((100 * (pr[name] - g.market_iv.to_numpy())) ** 2)))
                for name, g in [('train', tr), ('dev', dev), ('ho', ho)]}
        picks = {}
        for dmax in [.02, .05, .10]:
            head, mu, sd, devrmse, hist = train_head(tr, dev, pr['train'], pr['dev'], dmax)
            picks[dmax] = {'dev_IV_RMSE': devrmse, 'head': head, 'mu': mu, 'sd': sd, 'hist': hist}
        dmax = min(picks, key=lambda d: picks[d]['dev_IV_RMSE']); P = picks[dmax]
        zho, *_ = features(ho, pr['ho'], P['mu'], P['sd'])
        with torch.no_grad(): d_ho = P['head'](tensor(zho)).numpy()
        corr = pr['ho'] + d_ho
        err_prior = 100 * (pr['ho'] - ho.market_iv.to_numpy()); err_corr = 100 * (corr - ho.market_iv.to_numpy())
        price_prior = black(ho.x.to_numpy(), ho.tau.to_numpy(), pr['ho']); price_corr = black(ho.x.to_numpy(), ho.tau.to_numpy(), np.clip(corr, 1e-4, None))
        res['priors'][kind] = {
            'scale': s, 'delta_max': dmax, 'dev_scan': {str(k): v['dev_IV_RMSE'] for k, v in picks.items()},
            'prior_only': base, 'heldout_prior_IV_RMSE': float(np.sqrt(np.nanmean(err_prior ** 2))),
            'heldout_residual_IV_RMSE': float(np.sqrt(np.nanmean(err_corr ** 2))),
            'heldout_prior_price_RMSE': float(np.sqrt(np.nanmean((price_prior - ho.c.to_numpy()) ** 2))),
            'heldout_residual_price_RMSE': float(np.sqrt(np.nanmean((price_corr - ho.c.to_numpy()) ** 2))),
            'heldout_IV_P95': float(np.nanquantile(abs(err_corr), .95)), 'mean_abs_correction_volpts': float(100 * np.mean(abs(d_ho))),
            'head_params': sum(p.numel() for p in P['head'].parameters()), 'history': P['hist']}
        np.savez_compressed(OUT / f'heldout_{kind}.npz', err_prior=err_prior, err_corr=err_corr, days=ho.days.to_numpy(),
                            x=ho.x.to_numpy(), expiry_rank=ho.expiry_rank.to_numpy(), corr_iv=corr, prior_iv=pr['ho'], delta=d_ho)
        torch.save(P['head'].state_dict(), OUT / f'head_{kind}.pt')
        print(kind, 'scale %.3f dmax %.2f | held-out prior %.3f -> residual %.3f' % (s, dmax, res['priors'][kind]['heldout_prior_IV_RMSE'], res['priors'][kind]['heldout_residual_IV_RMSE']), flush=True)
    (OUT / 'results.json').write_text(json.dumps(res, indent=2, default=float) + '\n')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['split', 'evaluate']); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (split() if a.command == 'split' else evaluate())
