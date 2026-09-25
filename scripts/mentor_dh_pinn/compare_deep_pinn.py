#!/usr/bin/env python3
"""Common-surface pricing and separate own-model recovery for frozen pure PINNs."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,fit_network,_network_iv,_exact,_write_csv,sha256
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,black_call,invert_total_variance


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--cases',type=int,default=12)
    ap.add_argument('--starts',type=int,default=5)
    ap.add_argument('--max-nfev',type=int,default=400)
    ap.add_argument('--noise',type=float,default=0.)
    ap.add_argument('--scope',choices=['development','assessment'],default='development')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if min(args.cases,args.starts,args.max_nfev)<1 or not np.isfinite(args.noise) or args.noise<0:
        raise ValueError('Positive budgets and finite nonnegative noise required')
    models=[load_checkpoint(p) for p in args.checkpoint]
    if len({info['label'] for _,info in models})!=len(models):raise ValueError('Unique run names required')
    sources=[Path(__file__),ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',ROOT/'src/mentor_dh_pinn/deep_regular_pinn.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',ROOT/'src/mentor_dh_pinn/torch_pricer.py']
    manifest={'status':'running','scope':args.scope,'seed':args.seed,'cases_per_generator':args.cases,
              'noise':args.noise,'starts':args.starts,'max_nfev':args.max_nfev,
              'checkpoints':[i for _,i in models],
              'calibration':'Frozen neural IV only; identical quote mask/start count/budget; no exact-pricer refinement',
              'comparison':'Same observed surfaces for pricing; generating parameter truth used only for own-model post-fit scoring',
              'parameter_gate':'Each positive parameter within 5%; each rho within .05',
              'geometry':'21 strikes .8..1.2; 30/60/90/180/365/730 days; every third strike held out',
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6);tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    holdout=np.tile(np.arange(21)%3==2,6)
    metrics=[];fits=[];parameters=[];observations=[];truths=[];start=time.perf_counter()
    for factors in [1,2]:
        rng=np.random.default_rng(args.seed+10000*factors)
        units=rng.uniform(.1,.9,(args.cases,5*factors))
        for case,unit in enumerate(units):
            tag={'generator_factors':factors,'case':case}
            truth=decode_unit(unit,factors)
            price=_exact(x,tau,unit,factors);price96=_exact(x,tau,unit,factors,96)
            quad=float(np.max(abs((price-price96)*np.exp(-x))))
            observed=price+args.noise*(price-np.maximum(np.expm1(x),0))*rng.normal(size=len(x))
            iv=np.sqrt(invert_total_variance(observed,x)/tau)
            truths.append({**tag,'unit':unit.tolist(),'physical':truth.tolist()})
            valid=np.isfinite(iv).all() and quad<=1e-8
            for j in range(len(x)):
                observations.append({**tag,'quote':j,'x':x[j],'tau':tau[j],'holdout':bool(holdout[j]),
                                     'true_price':price[j],'observed_price':observed[j],'observed_iv':iv[j]})
            for model,info in models:
                row={**tag,'model':info['label'],'model_factors':model.factors,'status':'invalid_observations',
                     'own_model':model.factors==factors,'parameter_gate':False if model.factors==factors else None}
                if valid:
                    fit=fit_network(model,x,tau,iv,fit_mask=~holdout,starts=args.starts,
                                    max_nfev=args.max_nfev,seed=args.seed+100+case)
                    fits.append({**tag,'model':info['label'],**fit});row['status']=fit['status']
                    if fit['status']=='fitted':
                        fitted=np.array(fit['unit']);pred=_network_iv(model,x,tau,fitted)
                        neural=black_call(x,pred**2*tau)
                        exact=_exact(x,tau,fitted,model.factors)
                        for prefix,estimate in [('neural',neural),('exact',exact)]:
                            error=(estimate-price)*np.exp(-x)
                            row[prefix+'_holdout_price_rmse']=float(np.sqrt(np.mean(error[holdout]**2)))
                            row[prefix+'_fit_price_rmse']=float(np.sqrt(np.mean(error[~holdout]**2)))
                        row.update(seconds=fit['seconds'],optimizer_success=fit['optimizer_success'])
                        if model.factors==factors:
                            estimate=np.asarray(fit['physical']);scale=abs(truth).copy();scale[3::5]=.5
                            tolerance=.05*abs(truth);tolerance[3::5]=.05
                            error=abs(estimate-truth)/tolerance
                            row.update(parameter_gate=bool((error<=1).all()),
                                       scaled_parameter_rmse=float(np.sqrt(np.mean(((estimate-truth)/scale)**2))),
                                       tolerance_scaled_rmse=float(np.sqrt(np.mean(error**2))),
                                       worst_tolerance_error=float(error.max()))
                            for j in range(len(truth)):
                                parameters.append({**tag,'model':info['label'],'parameter_index':j,'truth':truth[j],
                                                   'estimate':estimate[j],'tolerance_error':error[j]})
                metrics.append(row)
            _write_csv(args.out/'metrics.csv',metrics);_write_csv(args.out/'parameter_recovery.csv',parameters)
            _write_csv(args.out/'observations.csv',observations)
            (args.out/'fits_and_starts.json').write_text(json.dumps(fits,indent=2))
            (args.out/'truths.json').write_text(json.dumps(truths,indent=2))
            print(json.dumps({'completed':tag,'seconds':time.perf_counter()-start}),flush=True)
    summary=[]
    for factors in [1,2]:
        for _,info in models:
            rows=[r for r in metrics if r['generator_factors']==factors and r['model']==info['label']]
            complete=all(r['status']=='fitted'for r in rows)
            record={'generator_factors':factors,'model':info['label'],'cases':len(rows),
                    'failed_fits':sum(r['status']!='fitted'for r in rows),'own_model':rows[0]['own_model']}
            if complete:
                for field in ['neural_holdout_price_rmse','exact_holdout_price_rmse']:
                    record[field]=float(np.sqrt(np.mean([r[field]**2 for r in rows])))
                if record['own_model']:
                    record.update(parameter_passes=sum(r['parameter_gate']for r in rows),
                                  scaled_parameter_rmse=float(np.sqrt(np.mean([r['scaled_parameter_rmse']**2 for r in rows]))),
                                  tolerance_scaled_rmse=float(np.sqrt(np.mean([r['tolerance_scaled_rmse']**2 for r in rows]))))
            summary.append(record)
    for _,info in models:
        if sha256(info['checkpoint'])!=info['sha256']:raise RuntimeError('Checkpoint changed during frozen assessment')
    for p,d in manifest['source_sha256'].items():
        if sha256(ROOT/p)!=d:raise RuntimeError('Source changed during assessment')
    manifest.update(status='complete',seconds=time.perf_counter()-start)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
