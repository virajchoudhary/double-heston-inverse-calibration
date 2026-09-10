"""Same-quote, exposed twelve-case pricing comparison; all fits use neural IV only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import json
import multiprocessing
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import scripts.mentor_dh_pinn.assess_regular_pinn as assessment
import src.mentor_dh_pinn.regular_pinn_data as data
import src.mentor_dh_pinn.torch_pricer as pricer
from src.mentor_dh_pinn.convolution_pinn import ConvolutionPINN


@contextmanager
def neural_only():
    def forbidden(*args,**kwargs):
        raise AssertionError('Exact-pricer evaluation forbidden inside inverse fitting')
    with ExitStack() as stack:
        for module,name in ((assessment,'_exact'),(assessment,'exact_prices'),
            (data,'exact_prices'),(data,'teacher_labels'),(pricer,'price_call'),(pricer,'price_call_single')):
            stack.enter_context(patch.object(module,name,forbidden))
        yield


def without_timing(value):
    if isinstance(value,dict):
        return {k:without_timing(v) for k,v in value.items() if k!='seconds'}
    if isinstance(value,list):return [without_timing(v) for v in value]
    return value


def geometry():
    strike=np.tile(np.linspace(.8,1.2,21),6)
    tau=np.repeat(np.array([30,60,90,180,365,730])/365.,21)
    return np.log(1/strike),tau,np.tile(np.arange(21)%3==2,6)


def case_worker(index,checkpoints,starts,max_nfev):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unit=np.random.default_rng(927931).uniform(.1,.9,(12,10))[index]
    truth=data.decode_unit(unit,2)
    x,tau,holdout=geometry()
    prices=assessment._exact(x,tau,unit,2)
    other=assessment._exact(x,tau,unit,2,96)
    iv=np.sqrt(data.invert_total_variance(prices,x)/tau)
    reference_error=float(np.max(np.abs((prices-other)*np.exp(-x))))
    result={'case':index,'truth_unit':unit.tolist(),'truth_parameters':truth.tolist(),
        'x':x.tolist(),'tau':tau.tolist(),'holdout':holdout.tolist(),
        'reference_prices':prices.tolist(),'reference_iv':iv.tolist(),
        'reference_quadrature_max_error_spot':reference_error,'models':[]}
    for checkpoint in checkpoints:
        model,info=assessment.load_checkpoint(checkpoint)
        row={'model':info['label'],'factors':model.factors,'checkpoint_sha256':info['sha256'],
             'complete_parameter_pass':False if model.factors==2 else None,
             'parameter_interpretation':'ten true DH parameters' if model.factors==2 else
                 'misspecified Single fit to DH data; no five-parameter truth exists'}
        if not np.isfinite(iv).all() or reference_error>1e-8:
            row['status']='invalid_reference';result['models'].append(row);continue
        with neural_only():
            fit=assessment.fit_network(model,x,tau,iv,fit_mask=~holdout,
                starts=starts,seed=908031+index,max_nfev=max_nfev)
            if index==0:
                corrupt=[a.copy() for a in (x,tau,iv)]
                for a in corrupt:a[holdout]=np.nan
                replay=assessment.fit_network(model,*corrupt,fit_mask=~holdout,
                    starts=starts,seed=908031+index,max_nfev=max_nfev)
                row['heldout_corruption_replay_identical']=without_timing(fit)==without_timing(replay)
                if not row['heldout_corruption_replay_identical']:
                    raise AssertionError('Held-out values changed calibration')
        row.update(status=fit['status'],fit=fit,exact_pricer_guard_active_for_all_starts=True)
        if fit['status']!='fitted':result['models'].append(row);continue
        fitted=np.array(fit['unit']);physical=np.array(fit['physical'])
        if model.factors==2:
            errors=np.abs(physical-truth)
            tolerance=.05*truth.copy();tolerance[[3,8]]=.05
            gates=errors<=tolerance
            row.update(parameter_passes=gates.tolist(),parameter_tolerance_units=(errors/tolerance).tolist(),
                individual_parameter_passes=int(gates.sum()),complete_parameter_pass=bool(gates.all()))
        # These exact calls happen only after fitting has finished.
        repriced=assessment._exact(x,tau,fitted,model.factors)
        repriced96=assessment._exact(x,tau,fitted,model.factors,96)
        row['exact_repricing']=assessment._score_price_iv(repriced,prices,iv,x,tau,holdout)[0]
        row['exact_repricing']['quadrature_max_error_spot']=float(np.max(np.abs((repriced-repriced96)*np.exp(-x))))
        row['exact_prices']=repriced.tolist()
        try:
            state,structural=data.coordinates(torch.tensor(assessment._query(x,tau,fitted),dtype=torch.float64),model.factors,torch)
            if isinstance(model,ConvolutionPINN):
                neural,diagnostics=model.price_and_diagnostics(state,structural)
                higher=ConvolutionPINN(model.component,128).price(state,structural)
                swapped=state[:,[0,2,1,3]]
                reverse=model.price(swapped,structural.flip(-2))
                row['composition_diagnostics']={
                    'max_density_mass_error':float((diagnostics['mass']-1).abs().max().detach()),
                    'max_martingale_moment_error':float((diagnostics['martingale_moment']-1).abs().max().detach()),
                    'minimum_density':float(diagnostics['minimum_density'].min().detach()),
                    'negative_density_quote_count':int((diagnostics['minimum_density']<0).sum()),
                    'quadrature_96_vs_128_max_error_spot':float(((neural-higher)*torch.exp(-state[:,0])).abs().max().detach()),
                    'factor_swap_max_error_spot':float(((neural-reverse)*torch.exp(-state[:,0])).abs().max().detach())}
                neural=neural.detach().numpy()
            else:
                neural_iv=assessment._network_iv(model,x,tau,fitted)
                neural=data.black_call(x,neural_iv**2*tau)
            row['neural_prices']=neural.tolist()
            row['neural']=assessment._score_price_iv(neural,prices,iv,x,tau,holdout)[0]
        except (FloatingPointError,ValueError,RuntimeError) as exc:
            row['neural_failure']=f'{type(exc).__name__}: {exc}'
        row['joint_pass']=bool(row['complete_parameter_pass'] and row.get('neural',{}).get('price_gate')
            and row['exact_repricing']['price_gate'] and row['exact_repricing']['quadrature_max_error_spot']<=1e-8
            and row.get('neural',{}).get('invalid_iv_quotes')==0)
        assert assessment.sha256(info['checkpoint'])==info['sha256']
        result['models'].append(row)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--cases',type=int,default=12)
    ap.add_argument('--starts',type=int,default=5)
    ap.add_argument('--max-nfev',type=int,default=400)
    args=ap.parse_args()
    if not 1<=args.cases<=12 or min(args.workers,args.starts,args.max_nfev)<1:ap.error('Invalid budget')
    args.out.mkdir(parents=True,exist_ok=False)
    infos=[assessment.load_checkpoint(p)[1] for p in args.checkpoint]
    sources=[Path(__file__),ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
        ROOT/'src/mentor_dh_pinn/convolution_pinn.py',ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
        ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',ROOT/'src/mentor_dh_pinn/torch_pricer.py']
    manifest={'status':'running','evidence':'exposed development cases, not unseen generalization',
        'truth_seed':927931,'case_count':args.cases,'checkpoints':infos,
        'starts':args.starts,'max_nfev':args.max_nfev,'workers':args.workers,
        'calibration_quotes_per_case':84,'heldout_quotes_per_case':42,
        'reference_family':'Double Heston only; favors correct model specification',
        'parameter_gates':'8 positive relative errors <=.05; 2 correlation absolute errors <=.05',
        'selection':'minimum calibration neural IV SSE only; identical quote split for all models',
        'source_sha256':{str(p.relative_to(ROOT)):assessment.sha256(p) for p in sources}}
    write=lambda name,obj:(args.out/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
    write('manifest.json',manifest)
    write('source_snapshot.json',{str(p.relative_to(ROOT)):p.read_text() for p in sources})
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(case_worker,i,args.checkpoint,args.starts,args.max_nfev):i for i in range(args.cases)}
        for future in as_completed(futures):
            result=future.result();write(f'case_{result["case"]:03d}.json',result)
            print(json.dumps({'case':result['case'],'models':[{k:r.get(k) for k in ('model','status','individual_parameter_passes','complete_parameter_pass')} for r in result['models']]}),flush=True)
    for path,digest in manifest['source_sha256'].items():assert assessment.sha256(ROOT/path)==digest
    for info in infos:
        assert assessment.sha256(info['checkpoint'])==info['sha256']
        assert assessment.sha256(Path(info['checkpoint']).parent/'config.json')==info['config_sha256']
    manifest.update(status='complete',source_and_checkpoint_hashes_rechecked=True)
    write('manifest.json',manifest)


if __name__=='__main__':main()
