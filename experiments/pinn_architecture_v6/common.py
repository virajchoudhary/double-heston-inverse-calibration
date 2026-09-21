"""Shared synthetic-only metrics and losses; original scientific code reused."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.nifty_multiscale_v5 import run as prior
from src.mentor_dh_pinn.regular_pinn_torch import residual

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts'
old=prior.old
T=old.tensor
save=old.save
read=old.read
sha=old.sha
SEEDS=[17,43]
LABELS=['teacher','iv','pde','convexity']
SCALES=np.array([1/(2e-5)**2,1/.002**2,.1/.01**2,.1])


def data(name):
    a=np.load(old.OUT/f'teacher/{name}.npz')
    return {k:a[k] for k in a.files}


def tensors(a):
    d={'z':T(a['coords']),'s':old.structural(a['params'])}
    if 'price' in a:
        d.update(y=T(a['price']),valid=torch.tensor(a['iv_valid']),iv=T(np.where(a['iv_valid'],a['iv'],0)))
    return d


def losses(net,d,c,di,ci):
    z,s=d['z'][di],d['s'][di]
    pred=net.price(z,s)*torch.exp(-z[:,0]);pv=net.iv(z,s);good=d['valid'][di]
    eq,dg=residual(net,c['z'][ci],c['s'][ci])
    ivloss=(pv[good]-d['iv'][di][good]).square().mean() if good.any() else pv.sum()*0
    return torch.stack([(pred-d['y'][di]).square().mean(),ivloss,eq.square().mean(),torch.relu(-dg['convexity']).square().mean()])


def gradients(parts,params):
    result={}
    for name,part,scale in zip(LABELS,parts,SCALES):
        gs=torch.autograd.grad(part*scale,params,retain_graph=True,allow_unused=True)
        v=torch.cat([g.flatten() for g in gs if g is not None])
        result[name]={'raw_loss':float(part.detach()),'scaled_loss':float(part.detach())*scale,
            'raw_grad_l2':float(v.norm())/scale,'scaled_grad_l2':float(v.norm()),
            'scaled_grad_max':float(v.abs().max()),'scaled_grad_mean':float(v.abs().mean())}
    return result


def physics(net,a):
    eq=[];conv=[];delta=[]
    for i in range(0,len(a['coords']),64):
        z=T(a['coords'][i:i+64]).requires_grad_(True);s=old.structural(a['params'][i:i+64])
        r,d=residual(net,z,s)
        price=net.price(z,s)
        first=torch.autograd.grad(price.sum(),z)[0][:,0]*torch.exp(-z[:,0])
        eq.extend(r.detach().numpy());conv.extend(d['convexity'].detach().numpy());delta.extend(first.detach().numpy())
    e=np.array(eq);ab=abs(e)
    return {'PDE_RMSE':float(np.sqrt(np.mean(e*e))),'PDE_mean_signed':float(e.mean()),
        'PDE_mean_absolute':float(ab.mean()),'PDE_median':float(np.median(ab)),
        'PDE_P95':float(np.quantile(ab,.95)),'PDE_max':float(ab.max()),
        'convexity_violation_percent':100*float((np.array(conv)<-1e-9).mean()),
        'monotonicity_violation_percent':100*float((np.array(delta)<-1e-9).mean())},e


def metrics(net,a):
    pred=old.neural([net],a['params'],a['coords'][:,0],a['coords'][:,-1])[0]
    return old.metrics(a['price'],pred,a['coords'][:,0],a['coords'][:,-1]),pred


def speed(net,a):
    z=T(a['coords'][:1024]);s=old.structural(a['params'][:1024]);times=[]
    with torch.no_grad():
        for _ in range(3):net.price(z,s)
        for _ in range(20):
            start=time.perf_counter();net.price(z,s);times.append(time.perf_counter()-start)
        single=[]
        for _ in range(20):
            start=time.perf_counter();net.price(z[:1],s[:1]);single.append(time.perf_counter()-start)
    return {'inference_1024_ms':1000*float(np.median(times)),'throughput_quotes_s':1024/float(np.median(times)),
            'inference_single_ms':1000*float(np.median(single))}
