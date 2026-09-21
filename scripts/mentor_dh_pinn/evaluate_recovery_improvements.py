#!/usr/bin/env python3
"""Frozen synthetic comparison of raw neural fits and training-error GLS."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,fit_network,_network_iv,_exact,sha256
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,invert_total_variance,black_call
from src.mentor_dh_pinn.surrogate_error import SurrogateError


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--cases',type=int,default=24);ap.add_argument('--noise',type=float,default=0.)
    ap.add_argument('--boundary',action='store_true',help='Stress test two unit parameters at .02/.98 per surface')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    spec=json.loads(args.spec.read_text());models=[];inputs={str(args.spec):sha256(args.spec)}
    for item in spec:
        model,info=load_checkpoint(item['checkpoint']);net=TorchRegularVariancePINN.from_mlx(model)
        inputs[info['checkpoint']]=info['sha256'];stats=None
        if item.get('statistics'):
            path=Path(item['statistics']);data=np.load(path/'training_error.npz')
            manifest=json.loads((path/'manifest.json').read_text())
            if manifest['checkpoint']['sha256']!=info['sha256']:raise ValueError('Statistics/checkpoint mismatch')
            inputs[str(path/'training_error.npz')]=sha256(path/'training_error.npz')
            stats=SurrogateError(data['x'],data['tau'],data['mean'],data['covariance'],info['sha256'])
        models.append((item,net,info,stats))
    source_paths=[Path(__file__),ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
                  ROOT/'src/mentor_dh_pinn/surrogate_error.py',ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
                  ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',ROOT/'src/mentor_dh_pinn/deep_regular_pinn.py']
    snapshots={str(p):p.read_text() for p in source_paths}
    inputs.update({str(p):sha256(p) for p in source_paths})
    manifest={'status':'running','seed':args.seed,'cases_per_generator':args.cases,'noise':args.noise,'boundary':args.boundary,
              'selection':'Candidates and covariance floors fixed before generation of assessment truths',
              'spec':spec,'input_sha256':inputs,'starts':5,'max_nfev':400,
              'scope':'synthetic central domain; own-model parameters and same-data prices reported separately'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps(snapshots))
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6);tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    mask=np.tile(np.arange(21)%3!=2,6);rows=[];fits=[];truths=[];observations=[]
    for factors in [1,2]:
        units=np.random.default_rng(args.seed+10000*factors).uniform(.1,.9,(args.cases,5*factors))
        if args.boundary:
            for case in range(args.cases):
                units[case,case%(5*factors)]=.02
                units[case,(case+1)%(5*factors)]=.98
        # Explicitly verify no exact parameter overlap with training error surfaces.
        for item,_,_,stats in models:
            if stats is not None and factors==2:
                trained=np.load(Path(item['statistics'])/'training_error.npz')['units']
                assert not set(map(tuple,trained))&set(map(tuple,units))
        for case,u in enumerate(units):
            truth=decode_unit(u,factors);price=_exact(x,tau,u,factors)
            discrepancy=float(np.max(abs(price-_exact(x,tau,u,factors,96))))
            if discrepancy>1e-9:raise ValueError('Assessment reference fails quadrature check')
            noise_rng=np.random.default_rng(args.seed+100000*factors+case)
            observed=price+args.noise*(price-np.maximum(np.exp(x)-1,0))*noise_rng.normal(size=len(x))
            iv=np.sqrt(invert_total_variance(observed,x)/tau)
            if not np.isfinite(iv).all():raise ValueError('Invalid noisy surface; no clipping or resampling')
            truths.append({'factors':factors,'case':case,'unit':u.tolist(),'physical':truth.tolist(),'quadrature_error':discrepancy})
            observations.append({'factors':factors,'case':case,'iv':iv.tolist(),'price':observed.tolist()})
            for item,net,info,stats in models:
                # Single-family recovery plus all candidates on shared DH data.
                if factors==1 and net.factors!=1:continue
                observation_std=None
                if stats is not None and args.noise:
                    # Delta-method IV noise uses observed calibration data, never
                    # generating parameters or clean assessment prices.
                    root=np.sqrt(tau)*iv;d2=x/root-.5*root
                    vega=np.exp(-.5*d2*d2)/np.sqrt(2*np.pi)*np.sqrt(tau)
                    time_value=observed-np.maximum(np.exp(x)-1,0)
                    observation_std=args.noise*time_value/np.maximum(vega,1e-12)
                fit=fit_network(net,x,tau,iv,fit_mask=mask,starts=5,max_nfev=400,
                                seed=args.seed+case+90000,surrogate_error=stats,
                                covariance_floor=item.get('floor',1e-4),observation_iv_std=observation_std,
                                prior_strength=item.get('prior_strength',0.))
                if fit['status']!='fitted':raise RuntimeError('Failed fit retained; assessment not complete')
                est=np.array(fit['unit']);pred=_network_iv(net,x,tau,est)
                neural=black_call(x,tau*pred**2);exact=_exact(x,tau,est,net.factors)
                row={'factors':factors,'case':case,'model':item['label'],
                     'neural_price_rmse':float(np.sqrt(np.mean(((neural-price)*np.exp(-x))[~mask]**2))),
                     'exact_reprice_rmse':float(np.sqrt(np.mean(((exact-price)*np.exp(-x))[~mask]**2)))}
                if stats is not None:
                    corrected=black_call(x,tau*(pred-stats.mean)**2)
                    row['bias_corrected_price_rmse']=float(np.sqrt(np.mean(((corrected-price)*np.exp(-x))[~mask]**2)))
                if net.factors==factors:
                    delta=np.array(fit['physical'])-truth;scale=abs(truth);scale[3::5]=.5
                    tol=.05*abs(truth);tol[3::5]=.05
                    row.update(parameter_rmse=float(np.sqrt(np.mean((delta/scale)**2))),
                               parameter_pass=bool((abs(delta)<=tol).all()),
                               parameter_error=(delta/scale).tolist(),tolerance_error=(abs(delta)/tol).tolist())
                rows.append(row);fits.append({'factors':factors,'case':case,'model':item['label'],**fit})
            (args.out/'metrics.json').write_text(json.dumps(rows,indent=2))
            (args.out/'fits.json').write_text(json.dumps(fits,indent=2))
            print(json.dumps({'factors':factors,'case':case,'complete':True}),flush=True)
    summary=[]
    for factors in [1,2]:
        for item,net,_,_ in models:
            subset=[r for r in rows if r['factors']==factors and r['model']==item['label']]
            if not subset:continue
            result={'factors':factors,'model':item['label'],'cases':len(subset)}
            for key in ['neural_price_rmse','exact_reprice_rmse','parameter_rmse','bias_corrected_price_rmse']:
                if key in subset[0]:result[key]=float(np.sqrt(np.mean([r[key]**2 for r in subset])))
            if 'parameter_pass' in subset[0]:result['parameter_passes']=sum(r['parameter_pass'] for r in subset)
            summary.append(result)
    for p,digest in inputs.items():
        if sha256(p)!=digest:raise RuntimeError('Input changed during assessment')
    (args.out/'truths.json').write_text(json.dumps(truths,indent=2))
    (args.out/'observations.json').write_text(json.dumps({'x':x.tolist(),'tau':tau.tolist(),'fit_mask':mask.tolist(),'surfaces':observations}))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    manifest['status']='complete';(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
