"""Retrospective daily anchor calibration; outer strikes never enter fitting."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc
import torch

from . import run
from experiments.nifty_multifactor_v4.literature_exact import (
    Grid, exact, fit_bs, bs_predict, fit_sh, iv,
)

OUT=run.OUT/'market'
SOURCE=run.ROOT/'outputs/nifty_fixed_crisis_20260917/clean_quotes.csv'


def panel():
    run.verify()
    a=pd.read_csv(SOURCE)
    if a.duplicated(['date','expiry','strike','option']).any():raise ValueError('Duplicate quote')
    return a[a.date_eligible].copy()


def split_day(day):
    anchor=day[day.split.eq('anchor')].sort_values(['expiry','strike'])
    test=day[day.split.eq('test')].sort_values(['expiry','strike'])
    keys=['expiry','strike']
    if len(anchor.merge(test,on=keys)):raise ValueError('Strike pair leaks across split')
    if len(anchor)<6 or len(test)<2:raise ValueError('Insufficient daily support')
    return anchor,test


def arrays(a):
    x,t=a.x.to_numpy(),a.tau.to_numpy()
    return x,t,a.market_call_normalized.to_numpy()*np.exp(-x)


def choose_bs(x,t,y):
    inner=np.arange(len(y))%5!=1
    candidates=[]
    for term in [False,True]:
        fitted=fit_bs(x[inner],t[inner],y[inner],term)
        score=float(np.mean((bs_predict(fitted,x[~inner],t[~inner])-y[~inner])**2))
        candidates.append({'term':term,'inner_MSE':score})
    chosen=min(candidates,key=lambda r:r['inner_MSE'])['term']
    return fit_bs(x,t,y,chosen),candidates


def baseline():
    a=panel();OUT.mkdir(exist_ok=True)
    if not (OUT/'manifest.json').exists():
        run.save(OUT/'manifest.json',{'utc':run.old.stamp(),'source_sha256':run.old.sha(SOURCE),
            'implementation_sha256':run.old.sha(Path(__file__)),'protocol_sha256':run.old.sha(run.HERE/'PROTOCOL.md')})
    verify()
    dest=OUT/'baselines';dest.mkdir(exist_ok=True)
    cfg=run.old.read(run.old.HERE/'config.json')['single_fit'].copy()
    cfg.update(multistarts=4,max_nfev=150)
    for (window,date),day in a.groupby(['window','date'],sort=True):
        path=dest/f'{date}.json'
        if path.exists():continue
        start=time.monotonic();anchor,test=split_day(day);x,t,y=arrays(anchor);xt,tt,yt=arrays(test)
        bs,inner=choose_bs(x,t,y);sh=fit_sh(x,t,y,cfg)
        audit=[];sp=exact(sh['best']['params'],xt,tt,audit)
        run.save(path,{'window':window,'date':date,'SH':sh,'BS':bs,'BS_inner':inner,
            'anchor_count':len(anchor),'test_count':len(test),'exact_fallbacks':audit,
            'max_SH_128_vs_checked_difference':float(np.max(abs(Grid(xt,tt,128)(sh['best']['params'])-sp))),
            'seconds':time.monotonic()-start})
        print(date,'baseline sec',round(time.monotonic()-start,2),'SH starts near best',sh['converged_near_best'],flush=True)


def verify():
    m=run.old.read(OUT/'manifest.json')
    for p,k in [(SOURCE,'source_sha256'),(Path(__file__),'implementation_sha256'),(run.HERE/'PROTOCOL.md','protocol_sha256')]:
        if run.old.sha(p)!=m[k]:raise RuntimeError(f'Market dependency changed: {p}')


def decode(z):
    scale,vs,vf=np.exp(z)
    p=np.array(run.old.read(run.old.HERE/'config.json')['published_double_slow_first'])
    p[[1,6]]*=scale;p[[2,7]]*=np.sqrt(scale);p[4]=vs;p[9]=vf
    return p


def predictor(nets,x,t):
    x,t=run.T(x),run.T(t)
    p=run.T(run.old.read(run.old.HERE/'config.json')['published_double_slow_first'])
    def predict(z):
        scale,vs,vf=torch.exp(z).unbind()
        st=torch.stack([torch.stack([p[0],p[1]*scale,p[2]*scale.sqrt(),p[3]]),
                        torch.stack([p[5],p[6]*scale,p[7]*scale.sqrt(),p[8]])])
        coords=torch.stack([x,torch.ones_like(x)*vs,torch.ones_like(x)*vf,t],dim=-1)
        return torch.stack([net.price(coords,st)*torch.exp(-x) for net in nets]).mean(0)
    return predict


def fit_pinn(nets,x,t,y,seed):
    pc=run.old.read(run.old.HERE/'config.json')['pinn']
    lo,hi=np.log(np.array([pc['scale_domain'],pc['slow_state_domain'],pc['fast_state_domain']])).T
    starts=lo+(hi-lo)*qmc.LatinHypercube(3,seed=seed).random(4)
    pred=predictor(nets,x,t)
    def fun(z):return (pred(run.T(z)).detach().numpy()-y)/.01
    def jac(z):return torch.func.jacfwd(pred)(run.T(z)).detach().numpy()/.01
    records=[]
    for j,z in enumerate(starts):
        result=least_squares(fun,z,jac=jac,bounds=(lo,hi),max_nfev=150,
                             ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
        records.append({'start':j,'success':bool(result.success),'status':int(result.status),
            'nfev':int(result.nfev),'objective':float(np.mean((result.fun*.01)**2)),
            'log_coordinates':result.x.tolist(),'params':decode(result.x).tolist(),
            'near_bound':bool(np.any(np.minimum((result.x-lo)/(hi-lo),(hi-result.x)/(hi-lo))<.005))})
    return {'best':min(records,key=lambda r:r['objective']),'starts':records,'global_optimum_proven':False}


def pinn():
    verify();a=panel();sel=run.old.read(run.OUT/'selection.json')
    for p,digest in sel['checkpoint_hashes'].items():
        if run.old.sha(run.ROOT/p)!=digest:raise RuntimeError('Selected checkpoints changed')
    nets=[run.model(s,sel['chosen'],True).eval().requires_grad_(False) for s in run.SEEDS]
    dest=OUT/'pinn';dest.mkdir(exist_ok=True)
    for index,((window,date),day) in enumerate(a.groupby(['window','date'],sort=True)):
        path=dest/f'{date}.json'
        if path.exists():continue
        start=time.monotonic();anchor,test=split_day(day);x,t,y=arrays(anchor)
        fit=fit_pinn(nets,x,t,y,185201+index)
        run.save(path,{'window':window,'date':date,'fit':fit,'seconds':time.monotonic()-start,
             'selection_sha256':run.old.sha(run.OUT/'selection.json')})
        print(date,'PINN sec',round(time.monotonic()-start,2),'boundary',fit['best']['near_bound'],flush=True)


def block_interval(values,seed=185119):
    values=np.asarray(values,float);n=len(values);rng=np.random.default_rng(seed)
    # Moving blocks without wrapping end-of-month to beginning-of-month.
    length=min(5,n);blocks=rng.integers(0,n-length+1,size=(5000,int(np.ceil(n/length))))
    indices=(blocks[...,None]+np.arange(length)).reshape(5000,-1)[:,:n]
    means=values[indices].mean(1)
    return {'mean':float(values.mean()),'CI95':np.quantile(means,[.025,.975]).tolist(),
            'CI_familywise_95_six_comparisons':np.quantile(means,[.05/12,1-.05/12]).tolist(),
            'positive_dates':int((values>0).sum()),'dates':n}


def report():
    verify();a=panel();selection=run.old.read(run.OUT/'selection.json')
    nets=[run.model(s,selection['chosen'],True).eval().requires_grad_(False) for s in run.SEEDS]
    predictions=[];diagnostics=[]
    for (window,date),day in a.groupby(['window','date'],sort=True):
        anchor,test=split_day(day);x,t,y=arrays(test)
        b=run.old.read(OUT/f'baselines/{date}.json');fit=run.old.read(OUT/f'pinn/{date}.json')['fit']
        q=test.copy();q['truth_forward']=y
        q['BS']=bs_predict(b['BS'],x,t);q['SH']=exact(b['SH']['best']['params'],x,t)
        q['DH_PINN']=predictor(nets,x,t)(run.T(fit['best']['log_coordinates'])).detach().numpy()
        q['DH_EXACT_AT_PINN_FIT']=exact(fit['best']['params'],x,t)
        for name in ['BS','SH','DH_PINN','DH_EXACT_AT_PINN_FIT']:
            q[name+'_error']=q[name]-y
            q[name+'_error_points']=q[name+'_error']*q.discount*q.forward
            q[name+'_iv']=iv(q[name].to_numpy(),x,t)
        predictions.append(q)
        diagnostics.append({'window':window,'date':date,'PINN_boundary':fit['best']['near_bound'],
            'PINN_success':fit['best']['success'],'SH_success':b['SH']['best']['success'],
            'SH_boundary':b['SH']['best']['near_bound'],'SH_converged_near_best':b['SH']['converged_near_best'],
            'neural_RMSE_forward':float(np.sqrt(np.mean((q.DH_PINN-q.DH_EXACT_AT_PINN_FIT)**2))),
            'SH_quadrature_max_gap':b['max_SH_128_vs_checked_difference']})
    q=pd.concat(predictions,ignore_index=True);q.to_csv(OUT/'predictions.csv',index=False)
    daily=[];summary=[];comparisons=[]
    for (window,date),g in q.groupby(['window','date']):
        daily.append({'window':window,'date':date,**{name:float(np.mean(g[name+'_error']**2))
            for name in ['BS','SH','DH_PINN','DH_EXACT_AT_PINN_FIT']}})
    daily=pd.DataFrame(daily);daily.to_csv(OUT/'daily_mse.csv',index=False)
    for window,g in q.groupby('window'):
        d=daily[daily.window.eq(window)]
        for bucket,mask in [('all',np.ones(len(g),bool)),('7-30d',g.tau.le(30/365)),
                            ('30-90d',g.tau.gt(30/365)&g.tau.le(90/365)),('90-100d',g.tau.gt(90/365))]:
            group=g[mask]
            if not len(group):continue
            joint=np.logical_and.reduce([np.isfinite(group[name+'_iv']) for name in ['BS','SH','DH_PINN','DH_EXACT_AT_PINN_FIT']])
            for name in ['BS','SH','DH_PINN','DH_EXACT_AT_PINN_FIT']:
                mse=group.groupby('date')[name+'_error'].apply(lambda x:np.mean(x*x))
                iv_error=group.loc[joint,name+'_iv']-group.loc[joint,'market_iv']
                summary.append({'window':window,'bucket':bucket,'model':name,'quotes':len(group),'dates':group.date.nunique(),
                    'equal_date_NRMSE':float(np.sqrt(mse.mean())),
                    'RMSE_index_points':float(np.sqrt(np.mean(group[name+'_error_points']**2))),
                    'IV_RMSE_vol_points':float(100*np.sqrt(np.mean(iv_error**2))) if joint.any() else None,
                    'joint_IV_quotes':int(joint.sum()),'invalid_IV_quotes':int((~np.isfinite(group[name+'_iv'])).sum())})
        for name in ['BS','SH']:
            comparisons.append({'window':window,'comparator':name,'difference':'comparator MSE minus PINN MSE',
                                **block_interval(d[name].to_numpy()-d.DH_PINN.to_numpy())})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    run.save(OUT/'comparisons.json',comparisons)
    run.save(OUT/'completed.json',{'utc':run.old.stamp(),'predictions_sha256':run.old.sha(OUT/'predictions.csv'),
             'rows':len(q),'dates':q.date.nunique(),'selected_arm':selection['chosen'],
             'retrospective_not_forecast':True,'all_ten_DH_parameters_calibrated':False})
    print(pd.DataFrame(summary).query("bucket=='all'").to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['baseline','pinn','report'])
    args=ap.parse_args();torch.set_num_threads(1);globals()[args.command]()
