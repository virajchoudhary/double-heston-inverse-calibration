#!/usr/bin/env python3
"""Post-fit development diagnostic only: surrogate bias versus local sensitivity.

Truths are used to inspect errors AFTER calibration, never to fit parameters or
select a network checkpoint. Linearized shifts are diagnostics, not estimates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.regular_pinn_data import decode_unit,invert_total_variance
from scripts.mentor_dh_pinn.assess_regular_pinn import _exact,sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    manifest=json.loads((args.assessment/'manifest.json').read_text())
    assert manifest['status']=='complete'
    cases=json.loads((args.assessment/'cases.json').read_text())
    weights=Path(manifest['checkpoint']);cfg=manifest['training_config']
    assert sha256(weights)==manifest['checkpoint_sha256']
    model=build_factor_pinn(cfg)
    model.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'));model.requires_grad_(False)
    geometry=manifest['geometry'];mask=~np.asarray(geometry['holdout'],bool)
    x=np.asarray(geometry['x'])[mask];tau=np.asarray(geometry['tau'])[mask]
    tx,tt=torch.tensor(x),torch.tensor(tau)
    price=lambda unit:neural_call_prices(model,decode_unit(unit,2,torch),tx,tt)
    results=[]
    for row in cases:
        unit=np.asarray(row['true_unit']);p=np.asarray(row['true_physical'])
        observed=np.asarray(row['observed_iv'])[mask]
        if not np.isfinite(observed).all():
            results.append({'case':row['case'],'status':'invalid_reference'});continue
        tu=torch.tensor(unit)
        predicted=price(tu).detach().numpy()
        neural_iv=np.sqrt(invert_total_variance(predicted,x)/tau)
        if not np.isfinite(neural_iv).all():
            results.append({'case':row['case'],'status':'invalid_neural_price_at_truth'});continue
        root=neural_iv*np.sqrt(tau);d2=x/root-root/2
        vega=np.exp(-d2*d2/2)/np.sqrt(2*np.pi)*np.sqrt(tau)
        neural_jac=torch.func.jacfwd(price)(tu).detach().numpy()/vega[:,None]
        physical_jac=torch.func.jacfwd(lambda u:decode_unit(u,2,torch))(tu).numpy()
        tolerance=.05*p;tolerance[3::5]=.05
        transform=np.linalg.solve(physical_jac,np.diag(tolerance))
        exact_jacs=[]
        for h in (1e-4,1e-5):
            jac=[]
            for j in range(10):
                bump=np.zeros(10);bump[j]=h
                ivs=[np.sqrt(invert_total_variance(_exact(x,tau,u,2),x)/tau) for u in (unit+bump,unit-bump)]
                jac.append((ivs[0]-ivs[1])/(2*h))
            exact_jacs.append(np.stack(jac,-1))
        exact_jac=exact_jacs[-1]
        bias=neural_iv-observed
        result={'case':row['case'],'status':'diagnosed','iv_bias_rmse_at_truth':float(np.sqrt(np.mean(bias*bias))),
                'neural_iv_sse_at_truth':float(bias@bias),
                'selected_fit_iv_sse':row.get('fit',{}).get('calibration_iv_sse'),
                'exact_jacobian_step_relative_difference':float(np.linalg.norm(exact_jacs[0]-exact_jac)/np.linalg.norm(exact_jac)),
                'neural_jacobian_relative_error':float(np.linalg.norm(neural_jac-exact_jac)/np.linalg.norm(exact_jac)),
                'actual_max_parameter_gate_units':max(row.get('parameter_gate_units',[float('inf')]))}
        if not np.isfinite(result['actual_max_parameter_gate_units']):result['actual_max_parameter_gate_units']=None
        for label,jac in [('exact',exact_jac),('neural',neural_jac)]:
            scaled=jac@transform
            shift,_,rank,singular=np.linalg.lstsq(scaled,-bias,rcond=None)
            result[label]={'singular_values_gate_scaled':singular.tolist(),'numerical_rank':int(rank),
                'condition_number':float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                'linear_bias_cancellation_shift_gate_units':shift.tolist(),
                'max_linear_shift_gate_units':float(np.max(np.abs(shift))),
                'remaining_linear_iv_rmse':float(np.sqrt(np.mean((bias+scaled@shift)**2)))}
        results.append(result);print(json.dumps(result),flush=True)
    assert sha256(weights)==manifest['checkpoint_sha256']
    output={'purpose':'Post-fit development sensitivity diagnostic; NOT inverse calibration, recovery result or identifiability proof',
        'assessment':str(args.assessment),'checkpoint_sha256':manifest['checkpoint_sha256'],
        'source_sha256':sha256(Path(__file__)),
        'interpretation':'A large linear shift means a small surrogate IV bias projects onto weak parameter directions. '
            'Local linearization can be inaccurate; these shifts must not be reported as fitted parameters.',
        'cases':results}
    (args.out/'diagnostic.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    (args.out/'source_snapshot.py').write_text(Path(__file__).read_text())


if __name__=='__main__':main()
