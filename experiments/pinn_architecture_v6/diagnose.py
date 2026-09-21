"""Reproduce baseline and record a separately labelled training continuation."""
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def main():
    prior.verify();old.verify()
    dev=data('development');train=data('train');coll=data('collocation')
    d,c=tensors(train),tensors(coll);rows=[];summary=[];local=[]
    for seed in SEEDS:
        torch.manual_seed(seed);net=prior.model(seed,'SHARED',True)
        m,pred=metrics(net,dev);pde,e=physics(net,dev)
        summary.append({'seed':seed,**m,**pde,**speed(net,dev),'parameters':sum(p.numel() for p in net.parameters())})
        z=T(dev['coords']);s=old.structural(dev['params'])
        features=net.base.features(z,s).detach().numpy()
        pd.DataFrame({'minimum':features.min(0),'maximum':features.max(0),'std':features.std(0)}).to_csv(OUT/f'input_conditioning_s{seed}.csv',index_label='feature')
        for i in range(len(pred)):
            x,vs,vf,t=dev['coords'][i]
            local.append({'seed':seed,'x':x,'v_slow':vs,'v_fast':vf,'days':365*t,'price_error':pred[i]-dev['price'][i],
                          'pde_residual':e[i],'teacher_price':dev['price'][i]})
        params=[p for p in net.parameters() if p.requires_grad]
        opt=torch.optim.Adam(params,lr=2e-4)
        for step in range(201):
            di=torch.randint(len(train['price']),(512,));ci=torch.randint(len(coll['coords']),(64,))
            parts=losses(net,d,c,di,ci)
            if step%20==0:
                for name,g in gradients(parts,params).items():rows.append({'seed':seed,'step':step,'component':name,**g,
                     'scope':'new continuation diagnostic; historical gradients unavailable','terminal_grad_l2':0.,'boundary_penalty_active':False})
            if step<200:
                opt.zero_grad();(parts*T(SCALES)).sum().backward();opt.step()
        print('diagnosed',seed,m,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'baseline_reproduction.csv',index=False)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'gradient_balance_baseline.csv',index=False)
    pd.DataFrame(local).to_csv(OUT/'baseline_local_errors.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for (seed,name),g in frame.groupby(['seed','component']):
        axes[0].plot(g.step,np.maximum(g.scaled_grad_l2,1e-15),label=f'{name} s{seed}')
    axes[0].set(yscale='log',xlabel='Continuation step',ylabel='Scaled component gradient L2');axes[0].legend(fontsize=7,ncol=2)
    for seed,g in frame.groupby('seed'):
        pivot=g.pivot(index='step',columns='component',values='scaled_grad_l2')
        axes[1].plot(pivot.index,pivot.pde/pivot.teacher,label=f'PDE / teacher, s{seed}')
        axes[1].plot(pivot.index,pivot.iv/pivot.teacher,ls='--',label=f'IV / teacher, s{seed}')
    axes[1].set(yscale='log',xlabel='Continuation step',ylabel='Gradient ratio');axes[1].legend(fontsize=8)
    fig.suptitle('Baseline diagnostic continuation; historical training gradients were not logged')
    fig.savefig(OUT/'gradient_balance_baseline.png',dpi=150);plt.close(fig)
    z=np.array(local);loc=pd.DataFrame(local)
    groups=[]
    for name,mask in {'short':loc.days.le(30),'medium':loc.days.between(30,365,inclusive='right'),'long':loc.days.gt(365),
        'ATM':loc.x.abs().le(.02),'wings':loc.x.abs().gt(.15),'fast_heavy':loc.v_fast.gt(loc.v_slow),'slow_heavy':loc.v_slow.ge(loc.v_fast)}.items():
        g=loc[mask];groups.append({'bucket':name,'count':len(g),'price_RMSE':float(np.sqrt(np.mean(g.price_error**2))),
            'price_max':float(g.price_error.abs().max()),'PDE_RMSE':float(np.sqrt(np.mean(g.pde_residual**2)))})
    pd.DataFrame(groups).to_csv(OUT/'baseline_buckets.csv',index=False)


if __name__=='__main__':
    torch.set_num_threads(1);main()
