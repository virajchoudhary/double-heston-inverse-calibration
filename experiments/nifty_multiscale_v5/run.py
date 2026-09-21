"""Reproducible additive extension; never modifies previous experiment artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.nifty_multifactor_v4 import run as old
from experiments.nifty_multifactor_v4.literature_exact import batch_teacher, iv
from src.mentor_dh_pinn.multiscale_pinn import TorchMultiscaleVariancePINN
from src.mentor_dh_pinn.regular_pinn_torch import residual

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'
SEEDS = [17, 43]
ARMS = ['SHARED', 'MULTISCALE']
T = old.tensor


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def freeze():
    OUT.mkdir(exist_ok=False)
    old.verify()
    paths = [HERE/'PROTOCOL.md', Path(__file__), ROOT/'src/mentor_dh_pinn/multiscale_pinn.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py', old.HERE/'literature_exact.py',
             old.HERE/'run.py',old.HERE/'config.json',
             ROOT/'outputs/nifty_fixed_crisis_20260917/clean_quotes.csv']
    paths += [old.OUT/f'pinn_s{s}/weights.pt' for s in SEEDS]
    paths += [old.OUT/f'teacher/{x}.npz' for x in ['train','development','collocation']]
    save(OUT/'manifest.json', {'utc':old.stamp(),'sources':{str(p.relative_to(ROOT)):old.sha(p) for p in paths},
         'parent_manifest_sha256':old.sha(old.OUT/'manifest.json'), 'seeds':SEEDS,
         'market_data_previously_exposed':True,'guaranteed_superiority':False})


def verify():
    manifest=old.read(OUT/'manifest.json')
    for p,digest in manifest['sources'].items():
        if old.sha(ROOT/p)!=digest:raise RuntimeError(f'Frozen dependency changed: {p}')
    return old.read(old.HERE/'config.json')


def model(seed,arm,trained=False):
    c=old.read(old.HERE/'config.json')
    base=old.model(c,256)
    base.load_state_dict(torch.load(old.OUT/f'pinn_s{seed}/weights.pt',weights_only=True))
    net=base if arm=='FROZEN' else TorchMultiscaleVariancePINN(base,gated=arm=='MULTISCALE')
    if trained and arm!='FROZEN':
        net.load_state_dict(torch.load(OUT/f'{arm}_s{seed}/weights.pt',weights_only=True))
    return net


def train(arm,seed):
    c=verify();torch.manual_seed(seed)
    dest=OUT/f'{arm}_s{seed}';dest.mkdir(exist_ok=False)
    net=model(seed,arm);a=np.load(old.OUT/'teacher/train.npz');col=np.load(old.OUT/'teacher/collocation.npz')
    z,s,y=T(a['coords']),old.structural(a['params']),T(a['price'])
    valid=torch.tensor(a['iv_valid']);vol=T(np.where(a['iv_valid'],a['iv'],0))
    cc,cs=T(col['coords']),old.structural(col['params'])
    n=c['pinn']['loss_normalisers']
    def loss(di,ci):
        pred=net.price(z[di],s[di])*torch.exp(-z[di,0]);pv=net.iv(z[di],s[di]);good=valid[di]
        eq,dg=residual(net,cc[ci],cs[ci])
        parts=torch.stack([(pred-y[di]).square().mean(),(pv[good]-vol[di][good]).square().mean(),
                           eq.square().mean(),torch.relu(-dg['convexity']).square().mean()])
        value=parts[0]/n['price']**2+parts[1]/n['iv']**2+n['pde_weight']*parts[2]/n['pde']**2+n['convexity_weight']*parts[3]
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
        return value,parts
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=2e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=800,eta_min=1e-5)
    started=time.monotonic();history=[]
    for step in range(800):
        di=torch.randint(len(z),(512,));ci=torch.randint(len(cc),(64,))
        optimizer.zero_grad(set_to_none=True);value,parts=loss(di,ci);value.backward();optimizer.step();schedule.step()
        if step%200==0 or step==799:
            row={'step':step+1,'loss':value.item(),'parts':parts.detach().tolist(),'seconds':time.monotonic()-started}
            history.append(row);save(dest/'progress.json',row);print(arm,seed,row,flush=True)
    optimizer=torch.optim.LBFGS(params,max_iter=60,line_search_fn='strong_wolfe',history_size=20)
    def closure():
        optimizer.zero_grad(set_to_none=True);value,_=loss(torch.arange(4096),torch.arange(256));value.backward();return value
    optimizer.step(closure)
    torch.save(net.state_dict(),dest/'weights.pt')
    save(dest/'completed.json',{'utc':old.stamp(),'seconds':time.monotonic()-started,'history':history,
         'weights_sha256':old.sha(dest/'weights.pt'),'trainable_parameters':sum(p.numel() for p in params),
         'market_weight_updates':False,'manifest_sha256':old.sha(OUT/'manifest.json')})


def prediction_metrics(models,a):
    mean,seeds=old.neural(models,a['params'],a['coords'][:,0],a['coords'][:,-1])
    rows=[]
    days=a['coords'][:,-1]*365
    buckets={'all':np.ones(len(days),bool),'7-30d':days<=30,'30-90d':(days>30)&(days<=90),
             '90-365d':(days>90)&(days<=365),'365-730d':days>365}
    for name,q in [('ensemble',mean)]+[(f'seed_{s}',q) for s,q in zip(SEEDS,seeds)]:
        for bucket,mask in buckets.items():
            if mask.any():rows.append({'model':name,'bucket':bucket,**old.metrics(a['price'][mask],q[mask],a['coords'][mask,0],a['coords'][mask,-1])})
    return rows,mean


def pde_metrics(models,a):
    values=[];negative=[]
    for net in models:
        eqs=[];conv=[]
        for start in range(0,len(a['coords']),64):
            eq,d=residual(net,T(a['coords'][start:start+64]),old.structural(a['params'][start:start+64]))
            eqs.extend(eq.detach().numpy());conv.extend(d['convexity'].detach().numpy())
        values.append(float(np.sqrt(np.mean(np.square(eqs)))))
        negative.append(int((np.array(conv)<-1e-9).sum()))
    return {'PDE_RMSE_per_seed':values,'PDE_RMSE_mean':float(np.mean(values)),
            'negative_convexity_per_seed':negative,'points_per_seed':len(a['coords'])}


def select():
    verify()
    if (OUT/'selection.json').exists():raise RuntimeError('Selection already frozen')
    a=np.load(old.OUT/'teacher/development.npz');col=np.load(old.OUT/'teacher/collocation.npz')
    col={k:col[k][:256] for k in col.files};rows=[];physics={};ensemble={}
    for arm in ['FROZEN']+ARMS:
        nets=[model(s,arm,True) for s in SEEDS]
        metrics,_=prediction_metrics(nets,a);rows.extend([{'arm':arm,**r} for r in metrics])
        ensemble[arm]=next(r for r in metrics if r['model']=='ensemble' and r['bucket']=='all')
        physics[arm]=pde_metrics(nets,col)
    eligible=[]
    for arm,r in ensemble.items():
        r['eligible']=(r['price_RMSE']<=2e-5 and r['price_P95']<=5e-5 and r['price_max']<=2e-4 and
                       r['IV_RMSE_volatility_points'] is not None and r['IV_RMSE_volatility_points']<=.2 and
                       physics[arm]['PDE_RMSE_mean']<=1.1*physics['FROZEN']['PDE_RMSE_mean'])
        if r['eligible']:eligible.append(arm)
    chosen=min(eligible,key=lambda arm:ensemble[arm]['price_RMSE']) if eligible else 'FROZEN'
    pd.DataFrame(rows).to_csv(OUT/'development_metrics.csv',index=False)
    save(OUT/'selection.json',{'utc':old.stamp(),'chosen':chosen,'development':ensemble,'physics':physics,
         'checkpoint_hashes':{str(p.relative_to(ROOT)):old.sha(p) for p in OUT.glob('*_s*/weights.pt')}})
    print('SELECTED',chosen,ensemble,flush=True)


def evaluate():
    c=verify();selection=old.read(OUT/'selection.json')
    for path,digest in selection['checkpoint_hashes'].items():
        if old.sha(ROOT/path)!=digest:raise RuntimeError('Checkpoint changed after selection')
    dest=OUT/'fresh_test';dest.mkdir(exist_ok=False)
    z,p=old.sample(c,4096,185104);y,audit=batch_teacher(p,z)
    a={'coords':z,'params':p,'price':y}
    zc,pc=old.sample(c,512,185105);col={'coords':zc,'params':pc}
    np.savez_compressed(dest/'data.npz',**a);save(dest/'teacher_audit.json',audit)
    rows=[];physics={};predictions={}
    for arm in ['FROZEN']+ARMS:
        nets=[model(s,arm,True) for s in SEEDS]
        metrics,pred=prediction_metrics(nets,a);rows.extend([{'arm':arm,**r} for r in metrics])
        physics[arm]=pde_metrics(nets,col);predictions[arm]=pred
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False);save(dest/'physics.json',physics)
    np.savez_compressed(dest/'predictions.npz',**predictions)
    save(dest/'completed.json',{'utc':old.stamp(),'selection_sha256':old.sha(OUT/'selection.json'),
          'no_reselection_allowed':True,'data_sha256':old.sha(dest/'data.npz')})
    print(pd.DataFrame(rows).query("model=='ensemble' and bucket=='all'").to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','train','select','evaluate'])
    ap.add_argument('--arm',choices=ARMS);ap.add_argument('--seed',type=int,choices=SEEDS)
    args=ap.parse_args();torch.set_num_threads(1)
    if args.command=='train':train(args.arm,args.seed)
    else:globals()[args.command]()
