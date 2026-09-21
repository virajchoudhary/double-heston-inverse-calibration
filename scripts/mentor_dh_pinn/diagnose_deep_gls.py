#!/usr/bin/env python3
"""Development-only calibration with training-estimated surrogate-error covariance.

This optional diagnostic uses no exact trial prices inside calibration. The
geometry is deliberately fixed; it does not claim arbitrary-quote support.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,_network_iv,_exact,sha256
from src.mentor_dh_pinn.regular_pinn_data import coordinates,decode_unit,invert_total_variance
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN


def fit_gls(network,x,tau,iv,mask,mean,covariance,shrinkage,seed=906777):
    # Mask before observing prices. Covariance/mean are frozen training statistics.
    xc,tc,target=x[mask],tau[mask],iv[mask]
    if not np.isfinite(target).all():raise ValueError('Invalid active observation')
    mean=mean[mask];covariance=covariance[np.ix_(mask,mask)]
    values,vectors=np.linalg.eigh(covariance)
    floor=max(values[-1]*shrinkage,1e-16)
    whitening=(vectors/np.sqrt(np.maximum(values,floor))).T
    cached=None;cached_u=None
    def evaluate(u):
        nonlocal cached,cached_u
        if cached_u is None or not np.array_equal(u,cached_u):
            q=torch.tensor(np.column_stack([xc,np.log(tc),np.broadcast_to(u,(len(xc),len(u)))]),dtype=torch.float64,requires_grad=True)
            c,p=coordinates(q,network.factors,torch);prediction=network.iv(c,p)
            jac=torch.autograd.grad(prediction.sum(),q)[0][:,2:].numpy()
            residual=prediction.detach().numpy()-mean-target
            cached_u=u.copy();cached=(whitening@residual,whitening@jac)
        return cached
    starts=.05+.9*qmc.LatinHypercube(5*network.factors,seed=seed).random(3)
    rows=[]
    for u in starts:
        result=least_squares(lambda z:evaluate(z)[0],u,jac=lambda z:evaluate(z)[1],
                             bounds=(1e-5,1-1e-5),x_scale='jac',max_nfev=400,
                             ftol=1e-12,xtol=1e-10,gtol=1e-10)
        r=evaluate(result.x)[0]
        rows.append({'unit':result.x.tolist(),'objective':float(r@r),'success':bool(result.success),'nfev':result.nfev})
    best=min(rows,key=lambda r:r['objective'])
    return {**best,'physical':decode_unit(np.array(best['unit']),network.factors).tolist(),'starts':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--surfaces',type=int,default=512)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    model,info=load_checkpoint(args.checkpoint);net=TorchRegularVariancePINN.from_mlx(model)
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6);tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    mask=np.tile(np.arange(21)%3!=2,6)
    units=.1+.8*qmc.LatinHypercube(10,seed=909451).random(args.surfaces)
    errors=[];start=time.perf_counter()
    for i,u in enumerate(units):
        price=_exact(x,tau,u,2);iv=np.sqrt(invert_total_variance(price,x)/tau)
        if not np.isfinite(iv).all():raise ValueError('Invalid training reference; preserve failed run')
        errors.append(_network_iv(net,x,tau,u)-iv)
    errors=np.array(errors);mean=errors.mean(0);cov=np.cov(errors,rowvar=False)
    np.savez_compressed(args.out/'training_error.npz',units=units,mean=mean,covariance=cov,errors=errors,x=x,tau=tau)
    manifest={'checkpoint':info,'scope':'development only','training_seed':909451,'training_surfaces':len(units),
              'recovery_seed':906311,'fit':'neural only with fixed training-error statistics','script_sha256':sha256(__file__)}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    records=[];summary=[]
    truths=np.random.default_rng(906311).uniform(.1,.9,(4,10))
    for shrinkage in [1e-2,1e-4,1e-6,1e-8]:
        all_errors=[];passes=0
        for case,u in enumerate(truths):
            truth=decode_unit(u,2);price=_exact(x,tau,u,2);iv=np.sqrt(invert_total_variance(price,x)/tau)
            fit=fit_gls(net,x,tau,iv,mask,mean,cov,shrinkage)
            scale=abs(truth).copy();scale[3::5]=.5
            error=(np.array(fit['physical'])-truth)/scale
            tolerance=.05*abs(truth);tolerance[3::5]=.05
            passed=bool((abs(np.array(fit['physical'])-truth)<=tolerance).all());passes+=passed
            records.append({'shrinkage':shrinkage,'case':case,'fit':fit,'parameter_pass':passed,'scaled_error':error.tolist()})
            all_errors.extend(error)
        row={'shrinkage':shrinkage,'scaled_parameter_rmse':float(np.sqrt(np.mean(np.square(all_errors)))),'parameter_passes':passes}
        summary.append(row);print(json.dumps(row),flush=True)
    (args.out/'fits.json').write_text(json.dumps(records,indent=2));(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('seconds',time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
