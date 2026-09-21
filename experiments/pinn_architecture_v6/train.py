"""Configuration-driven ablation ladder. No market-data access."""
import argparse
import platform
import resource
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from .common import *
from src.mentor_dh_pinn.research_modified_pinn import ResearchCorrectionPINN

ARMS=['A0_CONTINUE','CONTROL_PLAIN','ARCH_A_MODIFIED_MLP',
      'ARCH_B_MODIFIED_MLP_GRADBAL','ARCH_C_MODIFIED_MLP_GRADBAL_RAD','ARCH_E_ADAPTIVE_ACTIVATION']


def initialize():
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen')
    prior.verify();old.verify()
    a=data('train');net=prior.model(17,'SHARED',True)
    with torch.no_grad():
        z=T(a['coords']);s=old.structural(a['params']);f=net.base.features(z,s)
        f=torch.cat([f,torch.exp(-s[:,:,0]*z[:,-1:])],-1)
        low,high=f.min(0).values,f.max(0).values
        scale=(high-low)/2;scale=torch.where(scale>1e-8,scale,torch.ones_like(scale))
    save(OUT/'conditioning.json',{'center':((high+low)/2).tolist(),'scale':scale.tolist(),'source':'training inputs only'})
    git_root=subprocess.run(['git','rev-parse','--show-toplevel'],capture_output=True,text=True).stdout.strip()
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() if git_root==str(ROOT) else None
    paths=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',ROOT/'CURRENT_PINN_BASELINE.md',
        ROOT/'src/mentor_dh_pinn/research_modified_pinn.py',ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',
        ROOT/'src/mentor_dh_pinn/multiscale_pinn.py',old.HERE/'run.py',old.HERE/'literature_exact.py',old.HERE/'config.json',OUT/'conditioning.json']
    paths+=list((HERE/'research').glob('*.py'))
    paths+=[old.OUT/f'teacher/{name}.npz' for name in ['train','development','collocation']]
    paths+=[prior.OUT/f'SHARED_s{s}/weights.pt' for s in SEEDS]
    save(OUT/'manifest.json',{'utc':old.stamp(),'git_commit':commit,'git_context':git_root,
         'note':'No project-local Git commit; source hashes are authoritative' if commit is None else '',
         'files':{str(p.relative_to(ROOT)):sha(p) for p in paths},'seeds':SEEDS,
         'adam_steps':2000,'lbfgs_iterations':80,'final_market_opened':False,
         'synthetic_fidelity_opened':False,'synthetic_fidelity_seed':206104,'synthetic_pde_seed':206105})


def verify():
    manifest=read(OUT/'manifest.json')
    for p,h in manifest['files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError(f'Frozen input changed: {p}')
    return manifest


def network(arm,seed,checkpoint=None):
    base=prior.model(seed,'SHARED',True)
    if arm in ['BASELINE','A0_CONTINUE']:net=base
    else:
        stats=read(OUT/'conditioning.json')
        net=ResearchCorrectionPINN(base,**stats_args(stats),modified=arm!='CONTROL_PLAIN',adaptive=arm=='ARCH_E_ADAPTIVE_ACTIVATION')
    if checkpoint:net.load_state_dict(torch.load(checkpoint,weights_only=True))
    return net


def stats_args(stats):return {'center':stats['center'],'scale':stats['scale']}


def strata(coords):
    x,vs,vf,t=coords.T;days=t*365
    masks=[days<=30,(days>30)&(days<=365),days>365,x<-.02,abs(x)<=.02,x>.02,
           vs<=np.sqrt(.00015*.18),vs>np.sqrt(.00015*.18),vf<=np.sqrt(.001*.22),vf>np.sqrt(.001*.22),vf>vs,vs>=vf]
    groups=[np.flatnonzero(mask) for mask in masks]
    if any(len(g)==0 for g in groups):raise RuntimeError('Collocation coverage collapsed')
    return groups


def sample_indices(rng,groups,n=64,total=18000):
    indices=[rng.choice(g) for g in groups]
    indices.extend(rng.integers(total,size=n-len(indices)))
    return torch.tensor(indices,dtype=torch.long)


def rad(net,c,seed,step,dest):
    started=time.perf_counter();cfg=read(old.HERE/'config.json')
    z,p=old.sample(cfg,8192,seed+206000+step);scores=[]
    for i in range(0,len(z),64):scores.extend(abs(residual(net,T(z[i:i+64]),old.structural(p[i:i+64]))[0].detach().numpy()))
    scores=np.array(scores)
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite RAD pool')
    weight=scores/scores.mean()+1 if scores.mean()>0 else np.ones_like(scores)
    probability=weight/weight.sum();rng=np.random.default_rng(seed+step)
    ids=rng.choice(len(z),size=4096,replace=False,p=probability)
    c['z'][-4096:]=T(z[ids]);c['s'][-4096:]=old.structural(p[ids])
    np.savez_compressed(dest/f'rad_{step}.npz',pool=z,probability=probability,selected=z[ids])
    groups=strata(c['z'].numpy())
    return groups,{'step':step,'seconds':time.perf_counter()-started,'unique_selected':len(np.unique(ids)),
                   'active_slots':len(c['z']),'probability_min':float(probability.min()),'probability_max':float(probability.max())}


def run(arm,seed):
    manifest=verify();torch.manual_seed(seed);rng=np.random.default_rng(seed)
    teacher_generator=torch.Generator().manual_seed(seed+206100)
    dest=OUT/arm/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
    save(dest/'config.json',{'arm':arm,'seed':seed,'manifest_sha256':sha(OUT/'manifest.json'),
                           'adam_steps':2000,'lbfgs_iterations':80,'git_commit':manifest['git_commit']})
    net=network(arm,seed);params=[p for p in net.parameters() if p.requires_grad]
    if sum(p.numel() for p in net.parameters())>1.25*275684:raise ValueError('Parameter budget exceeded')
    dev=data('development');d=tensors(data('train'));c=tensors(data('collocation'));groups=strata(c['z'].numpy())
    adaptive=arm in ['ARCH_B_MODIFIED_MLP_GRADBAL','ARCH_C_MODIFIED_MLP_GRADBAL_RAD']
    weights=np.ones(4);history=[];gradlog=[];radlog=[];started=time.perf_counter()
    optimizer=torch.optim.Adam(params,lr=2e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2000,eta_min=1e-5)
    try:
        for step in range(2000):
            if arm=='ARCH_C_MODIFIED_MLP_GRADBAL_RAD' and step in [500,1000,1500]:
                groups,row=rad(net,c,seed,step,dest);radlog.append(row)
            di=torch.randint(len(d['y']),(512,),generator=teacher_generator);ci=sample_indices(rng,groups)
            parts=losses(net,d,c,di,ci)
            if step%100==0:
                gs=gradients(parts,params)
                if adaptive:
                    norms=np.array([gs[name]['scaled_grad_l2'] for name in LABELS[:3]])
                    active=norms>1e-14
                    target=weights[:3].copy();target[active]=norms[active].mean()/norms[active]
                    weights[:3]=.9*weights[:3]+.1*target
                    if not np.isfinite(weights).all() or weights.min()<1e-6 or weights.max()>1e6:raise FloatingPointError('Unstable adaptive weight')
                for name,g,w in zip(LABELS,gs.values(),weights):gradlog.append({'step':step,'component':name,'weight':w,**g})
            value=(parts*T(SCALES*weights)).sum()
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();schedule.step()
            if step%100==0 or step==1999:
                row={'step':step+1,'total_loss':float(value.detach()),'seconds':time.perf_counter()-started,
                    **{name:float(v) for name,v in zip(LABELS,parts.detach())},
                    **{f'weight_{name}':float(w) for name,w in zip(LABELS,weights)},'terminal_loss':0.,'boundary_penalty_active':False}
                history.append(row);save(dest/'progress.json',row)
                if step%500==0 or step==1999:print(arm,seed,step+1,round(row['seconds'],1),row['total_loss'],flush=True)
        torch.save(net.state_dict(),dest/'before_lbfgs.pt')
        pre,_=metrics(net,dev);prephys,_=physics(net,dev)
        save(dest/'before_lbfgs_metrics.json',{**pre,**prephys})
        optimizer=torch.optim.LBFGS(params,max_iter=80,line_search_fn='strong_wolfe',history_size=20)
        lbfgslog=[]
        def closure():
            optimizer.zero_grad(set_to_none=True);parts=losses(net,d,c,torch.arange(4096),torch.arange(256))
            value=(parts*T(SCALES*weights)).sum()
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite LBFGS loss')
            value.backward();lbfgslog.append({'evaluation':len(lbfgslog)+1,'loss':float(value.detach()),
                                            **{name:float(v) for name,v in zip(LABELS,parts.detach())}})
            return value
        optimizer.step(closure)
        elapsed=time.perf_counter()-started
        torch.save(net.state_dict(),dest/'weights.pt')
        m,pred=metrics(net,dev);ph,e=physics(net,dev)
        save(dest/'metrics.json',{**m,**ph})
        np.savez_compressed(dest/'development_predictions.npz',price=pred,pde=e)
        pd.DataFrame(history).to_csv(dest/'training.csv',index=False)
        pd.DataFrame(gradlog).to_csv(dest/'gradients.csv',index=False)
        pd.DataFrame(lbfgslog).to_csv(dest/'lbfgs.csv',index=False);save(dest/'rad.json',radlog)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(dest/'completed.json',{'seconds':elapsed,'peak_process_RSS_MB':rss/(1024**2 if platform.system()=='Darwin' else 1024),
            'parameters':sum(p.numel() for p in net.parameters()),'trainable_parameters':sum(p.numel() for p in params),
            'weights_sha256':sha(dest/'weights.pt'),'final_weights':weights.tolist(),
            'activation_scales':net.core.scales.detach().tolist() if hasattr(net,'core') and net.core.scales is not None else None,
            'market_data_used':False,'precision':'float64','device':'CPU',**speed(net,dev)})
        print('COMPLETE',arm,seed,m,flush=True)
    except Exception as exc:
        save(dest/'failed.json',{'error':repr(exc),'seconds':time.perf_counter()-started})
        pd.DataFrame(history).to_csv(dest/'training.csv',index=False)
        pd.DataFrame(gradlog).to_csv(dest/'gradients.csv',index=False)
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['initialize','train']);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--seed',type=int,choices=SEEDS)
    args=ap.parse_args();torch.set_num_threads(1)
    if args.command=='initialize':initialize()
    elif args.arm is None or args.seed is None:ap.error('train needs arm and seed')
    else:run(args.arm,args.seed)
