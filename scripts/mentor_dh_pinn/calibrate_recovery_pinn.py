#!/usr/bin/env python3
"""Calibrate one frozen PINN with the development-selected recovery profile.

Input JSON: x=log(F/K), tau in years, observed_iv in decimals, fit_mask.
The frozen covariance supports the documented 126-quote grid only. Withheld
IV entries may be null. No truth parameters or exact pricer enter calibration.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,fit_network,sha256
from src.mentor_dh_pinn.surrogate_error import SurrogateError
from src.mentor_dh_pinn.regular_pinn_data import black_call


def calibrate_quotes(payload, *, profile='clean', noise_fraction=0., starts=5, seed=71, max_nfev=400):
    if profile not in ('clean','noise'):raise ValueError('Profile must be clean or noise')
    if not np.isfinite(noise_fraction) or noise_fraction<0:raise ValueError('Invalid noise fraction')
    if (profile=='noise') != (noise_fraction>0):
        raise ValueError('Noise profile requires a positive supplied time-value noise fraction; clean requires zero')
    spec_path=ROOT/'outputs/deeper_pinn'/f'prior_{profile}_spec.json'
    spec=next(r for r in json.loads(spec_path.read_text()) if r['label']=='Selected regularized PINN')
    model,info=load_checkpoint(ROOT/spec['checkpoint'])
    path=ROOT/spec['statistics'];metadata=json.loads((path/'manifest.json').read_text())
    if metadata['checkpoint']['sha256']!=info['sha256']:raise ValueError('Checkpoint does not match error statistics')
    with np.load(path/'training_error.npz') as d:
        stats=SurrogateError(d['x'].copy(),d['tau'].copy(),d['mean'].copy(),d['covariance'].copy(),info['sha256'])
    x,tau,iv=[np.asarray(payload[k],dtype=float) for k in ['x','tau','observed_iv']]
    mask=np.asarray(payload.get('fit_mask',np.ones(len(x))),dtype=bool)
    if x.ndim!=1 or x.shape!=tau.shape or x.shape!=iv.shape or mask.shape!=x.shape:
        raise ValueError('Quote arrays and mask must have equal one-dimensional shape')
    std=None
    if noise_fraction:
        active_iv=iv[mask];active_tau=tau[mask];active_x=x[mask]
        if not np.isfinite(active_iv).all() or (active_iv<=0).any() or (active_tau<=0).any():
            raise ValueError('Invalid active quote')
        root=np.sqrt(active_tau)*active_iv;d2=active_x/root-.5*root
        vega=np.exp(-.5*d2*d2)/np.sqrt(2*np.pi)*np.sqrt(active_tau)
        observed=black_call(active_x,active_tau*active_iv**2)
        std=np.full(len(x),np.nan)
        std[mask]=noise_fraction*(observed-np.maximum(np.exp(active_x)-1,0))/np.maximum(vega,1e-12)
    fit=fit_network(model,x,tau,iv,fit_mask=mask,starts=starts,seed=seed,max_nfev=max_nfev,
                    surrogate_error=stats,covariance_floor=spec['floor'],observation_iv_std=std,
                    prior_strength=spec['prior_strength'])
    return {'profile':profile,'noise_fraction':noise_fraction,'checkpoint':info,
            'profile_sha256':sha256(spec_path),'statistics_sha256':sha256(path/'training_error.npz'),
            'parameter_order':'slow kappa,theta,sigma,rho,v0; fast kappa,theta,sigma,rho,v0',
            'interpretation':'Regularized point estimate. The fixed midpoint penalty introduces bias; a fitted status does not certify recovery of all ten parameters.',
            'fit':fit}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile',choices=['clean','noise'],default='clean')
    ap.add_argument('--noise-fraction',type=float,default=0.);ap.add_argument('--starts',type=int,default=5)
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--max-nfev',type=int,default=400)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Use an unused result path')
    import torch
    torch.set_num_threads(1)
    result=calibrate_quotes(json.loads(args.input.read_text()),profile=args.profile,noise_fraction=args.noise_fraction,
                            starts=args.starts,seed=args.seed,max_nfev=args.max_nfev)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'out':str(args.out),'status':result['fit']['status'],'physical':result['fit'].get('physical')}))


if __name__=='__main__':main()
