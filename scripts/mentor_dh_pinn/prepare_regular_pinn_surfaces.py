#!/usr/bin/env python3
"""Synthetic training-only surfaces for a local, parameter-aware surrogate loss.

The fixed preconditioner maps IV errors into linearized parameter-tolerance
errors. It is a training target, never an input to the deployed inverse fit.
Truncated weak singular directions are recorded, not called identified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,expected_variance,teacher_labels
from scripts.mentor_dh_pinn.diagnose_regular_pinn_validation import exact_iv_jacobian


def preconditioner(jacobian,unit,factors,absolute_floor=1e-6):
    u=torch.tensor(unit,dtype=torch.float64,requires_grad=True)
    p=decode_unit(u,factors,torch).detach().numpy()
    dp=torch.autograd.functional.jacobian(lambda z:decode_unit(z,factors,torch),u).numpy()
    tolerance=.05*np.abs(p);tolerance[3::5]=.05
    transform=np.linalg.solve(dp,np.diag(tolerance))
    a=jacobian@transform
    left,s,right=np.linalg.svd(a,full_matrices=False)
    keep=s>max(absolute_floor,s[0]*1e-10)
    inverse=np.zeros_like(s);inverse[keep]=1/s[keep]
    return (right.T*inverse)@left.T,s,int(keep.sum()),a


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--factors',type=int,choices=(1,2),default=2)
    ap.add_argument('--surfaces',type=int,default=1024);ap.add_argument('--seed',type=int,default=907721)
    ap.add_argument('--singular-floor',type=float,default=1e-6)
    ap.add_argument('--fit-quotes-only',action='store_true',
                    help='Build the recovery loss from the same 84 quote positions used by calibration')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    units=qmc.LatinHypercube(5*args.factors,seed=args.seed).random(args.surfaces)
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6)
    tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    queries=np.stack([np.column_stack([x,np.log(tau),np.broadcast_to(u,(126,len(u)))]) for u in units])
    flat=queries.reshape(-1,queries.shape[-1]);labels=teacher_labels(flat,args.factors)
    iv=np.sqrt(labels['w']/np.exp(flat[:,1]));jac=[];jac_errors=[]
    for start in range(0,len(flat),512):
        q=torch.tensor(flat[start:start+512],dtype=torch.float64,requires_grad=True)
        back=expected_variance(q,args.factors,torch)
        dback=torch.autograd.grad(torch.log(back).sum(),q)[0].numpy()[:,2:]
        j=iv[start:start+512,None]*(labels['dg_du'][start:start+512]+.5*dback)
        _,j96=exact_iv_jacobian(flat[start:start+512],args.factors,96)
        jac.append(j);jac_errors.extend(np.max(np.abs(j-j96),axis=1))
    jac=np.concatenate(jac).reshape(args.surfaces,126,-1)
    jac_errors=np.asarray(jac_errors).reshape(args.surfaces,126)
    usable=labels['usable'].reshape(args.surfaces,126).all(axis=1)
    usable &= np.isfinite(jac).all(axis=(1,2)) & (jac_errors<=1e-6).all(axis=1)
    matrices=np.full((args.surfaces,5*args.factors,126),np.nan)
    singular=np.full((args.surfaces,5*args.factors),np.nan);ranks=np.zeros(args.surfaces,dtype=int)
    fit_mask=np.tile(np.arange(21)%3!=2,6) if args.fit_quotes_only else np.ones(126,dtype=bool)
    for i in np.flatnonzero(usable):
        b,singular[i],ranks[i],_=preconditioner(jac[i,fit_mask],units[i],args.factors,args.singular_floor)
        matrices[i]=0.
        matrices[i][:,fit_mask]=b
    data={'q':queries,'iv':iv.reshape(args.surfaces,126),'preconditioner':matrices,
          'singular_values':singular,'rank':ranks,'usable':usable,'unit':units,'fit_mask':fit_mask,
          'jacobian_quadrature_error':jac_errors,
          'quote_usable':labels['usable'].reshape(args.surfaces,126)}
    path=args.out/'surfaces.npz';np.savez_compressed(path,**data)
    report={'purpose':'Additional synthetic training only; no validation or final-test parameters are used',
            'seed':args.seed,'factors':args.factors,'surfaces':args.surfaces,'quotes_per_surface':126,
            'recovery_loss_quotes':int(fit_mask.sum()),
            'usable_surfaces':int(usable.sum()),'rejected_surfaces_retained':int((~usable).sum()),
            'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'singular_floor_absolute_iv':args.singular_floor,'singular_floor_relative':1e-10,
            'rank_counts_usable':{str(i):int(((ranks==i)&usable).sum()) for i in range(5*args.factors+1)},
            'metric':'B*(neural_IV-reference_IV); B is truncated pseudoinverse of exact IV Jacobian in physical recovery-tolerance coordinates',
            'limitations':'Local linear training surrogate, not actual recovered parameters. Weak directions below cutoff receive no parameter-aware loss; price/PDE losses remain. No guarantee of identifiability.',
            'bounds':'Full declared training unit cube, Latin hypercube; 21 fixed strikes .8..1.2; 30/60/90/180/365/730 days',
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',ROOT/'scripts/mentor_dh_pinn/diagnose_regular_pinn_validation.py',ROOT/'src/mentor_dh_pinn/torch_pricer.py']}}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
