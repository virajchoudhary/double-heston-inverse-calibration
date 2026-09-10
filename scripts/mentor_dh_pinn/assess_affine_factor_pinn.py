#!/usr/bin/env python3
"""Development recovery of the separately labelled factor-structured PINN."""
import argparse,json,sys,time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.affine_factor_calibration import fit_factor_pinn
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,invert_total_variance
from scripts.mentor_dh_pinn.assess_regular_pinn import _exact,sha256,_score_price_iv


def evaluate_case(job):
    i,truth,x,tau,holdout,starts,budget,start_seed,weights,cfg=job
    torch.set_num_threads(1)
    model=build_factor_pinn(cfg)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.requires_grad_(False)
    finite_list=lambda values:[float(v) if np.isfinite(v) else None for v in values]
    true_p=decode_unit(truth,2);reference=_exact(x,tau,truth,2);iv=np.sqrt(invert_total_variance(reference,x)/tau)
    quad=float(np.max(np.abs((_exact(x,tau,truth,2,96)-reference)*np.exp(-x))))
    if not np.isfinite(quad):quad=None
    row={'case':i,'true_unit':truth.tolist(),'true_physical':true_p.tolist(),
         'observed_iv':finite_list(iv),'reference_price':finite_list(reference),'true_quadrature_error_spot':quad,
         'all_parameter_pass':False,'joint_pass':False}
    if not np.isfinite(iv).all() or quad is None or quad>1e-8:
        row['status']='invalid_reference'
        return row
    fit=fit_factor_pinn(model,x,tau,iv,fit_mask=~holdout,starts=starts,seed=start_seed,max_nfev=budget)
    row.update(status=fit['status'],fit=fit)
    if fit['status']=='fitted':
        estimate=np.asarray(fit['physical']);tolerance=.05*true_p;tolerance[3::5]=.05
        error=np.abs(estimate-true_p)/tolerance
        row.update(parameter_gate_units=error.tolist(),all_parameter_pass=bool((error<=1).all()),
                   individual_parameter_passes=int((error<=1).sum()))
        with torch.no_grad():
            predicted=neural_call_prices(model,torch.tensor(estimate),torch.tensor(x),torch.tensor(tau)).numpy()
        exact=_exact(x,tau,np.asarray(fit['unit']),2)
        quad=float(np.max(np.abs((_exact(x,tau,np.asarray(fit['unit']),2,96)-exact)*np.exp(-x))))
        if not np.isfinite(quad):quad=None
        row['fitted_quadrature_error_spot']=quad
        for label,price in [('neural',predicted),('exact_reprice',exact)]:
            score,_=_score_price_iv(price,reference,iv,x,tau,holdout);row[label]=score
        row['joint_pass']=bool(row['all_parameter_pass'] and row['neural']['price_gate'] and row['exact_reprice']['price_gate']
            and row['neural']['invalid_iv_quotes']==0 and row['exact_reprice']['invalid_iv_quotes']==0
            and quad is not None and quad<=1e-8)
    return row


def evaluate_cases(jobs,workers):
    if workers==1:
        yield from map(evaluate_case,jobs)
    else:
        # Independent CPU cases; ordered map preserves the original case order.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            yield from pool.map(evaluate_case,jobs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--case-set',choices=('development4','exposed12'),default='development4')
    ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if args.workers<1:ap.error('--workers must be positive')
    run=args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
    weights=run/'model.pt' if args.checkpoint.is_dir() else args.checkpoint
    cfg=json.loads((run/'config.json').read_text())
    count,seed,starts,budget=(4,906311,3,200) if args.case_set=='development4' else (12,927931,5,400)
    units=np.random.default_rng(seed).uniform(.1,.9,(count,10))
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6);tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    holdout=np.tile(np.arange(21)%3==2,6)
    sources=[Path(__file__),*[ROOT/'src/mentor_dh_pinn'/name for name in ('affine_factor_pinn.py',
        'affine_factor_pricing.py','affine_factor_calibration.py','regular_pinn_data.py','torch_pricer.py','conjugate_factor_pinn.py')],
        ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py']
    if cfg.get('integrated',False):sources.append(ROOT/'src/mentor_dh_pinn/integrated_factor_pinn.py')
    if cfg.get('moment',False):sources.append(ROOT/'src/mentor_dh_pinn/moment_factor_pinn.py')
    if cfg.get('two_moment',False):sources.append(ROOT/'src/mentor_dh_pinn/two_moment_factor_pinn.py')
    manifest={'status':'running','variant':'factor-structured Riccati PINN, NOT regular pricing PINN',
        'evidence_level':'development; previously exposed cases, not unseen assessment',
        'case_set':args.case_set,'sampling_seed':seed,'cases':count,'starts':starts,'max_nfev_per_start':budget,
        'workers':args.workers,'cpu_threads_per_worker':1,
        'checkpoint':str(weights),'checkpoint_sha256':sha256(weights),'training_config':cfg,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'fit_information':'observed IV, log-moneyness and tau at calibration strikes only; blind starts',
        'target':'independent canonical Double Heston Fourier prices; clean six-expiry geometry',
        'geometry':{'x':x.tolist(),'tau':tau.tolist(),'holdout':holdout.tolist()},
        'gates':{'positive_relative':.05,'rho_absolute':.05,'heldout_price_rmse_spot':1e-5}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    rows=[];started=time.perf_counter()
    jobs=[(i,truth,x,tau,holdout,starts,budget,906777 if args.case_set=='development4' else 907931+100+i,
           weights,cfg) for i,truth in enumerate(units)]
    for row in evaluate_cases(jobs,args.workers):
        rows.append(row);(args.out/'cases.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
        print(json.dumps({'case':row['case'],'status':row['status'],'parameter_pass':row['all_parameter_pass'],
            'individual_passes':row.get('individual_parameter_passes',0),'seconds':time.perf_counter()-started}),flush=True)
    assert sha256(weights)==manifest['checkpoint_sha256'],'Checkpoint changed during evaluation'
    for p in sources:assert sha256(p)==manifest['source_sha256'][str(p.relative_to(ROOT))],f'Source changed: {p}'
    summary={'cases':count,'all_parameter_passes':sum(r['all_parameter_pass'] for r in rows),
        'individual_parameter_passes':sum(r.get('individual_parameter_passes',0) for r in rows),'individual_denominator':count*10,
        'joint_passes':sum(r['joint_pass'] for r in rows),'seconds':time.perf_counter()-started,
        'evidence_level':manifest['evidence_level']}
    manifest.update(status='complete',frozen_hashes_rechecked=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
