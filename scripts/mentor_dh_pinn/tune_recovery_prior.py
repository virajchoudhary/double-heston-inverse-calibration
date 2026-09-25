#!/usr/bin/env python3
"""Choose ridge strength on previously exposed DEVELOPMENT surfaces only."""
import json,sys,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,fit_network,sha256
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN
from src.mentor_dh_pinn.surrogate_error import SurrogateError


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    strengths=[0.,.1,1.,10.,100.];records=[];summary=[]
    manifest={'scope':'development; reuses previously exposed seed 910831',
              'cases':12,'strengths':strengths,'prior':'unit midpoint .5; no generating parameters in objective',
              'selection':'minimum mean squared scaled physical error, separately for clean and noisy',
              'script_sha256':sha256(__file__),'fit_sha256':sha256(ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py')}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for label,checkpoint,statistics in [('DH11','deep11_s17','deep11_gls_development'),('DH17','deep17_full_s29','deep17_gls_development')]:
        model,info=load_checkpoint(ROOT/'outputs/deeper_pinn'/checkpoint);net=TorchRegularVariancePINN.from_mlx(model)
        path=ROOT/'outputs/deeper_pinn'/statistics;d=np.load(path/'training_error.npz')
        assert json.loads((path/'manifest.json').read_text())['checkpoint']['sha256']==info['sha256']
        stats=SurrogateError(d['x'],d['tau'],d['mean'],d['covariance'],info['sha256'])
        for condition in ['clean','noise']:
            path=ROOT/'outputs/deeper_pinn'/f'recovery_{condition}_910831'
            observations=json.loads((path/'observations.json').read_text());truths=json.loads((path/'truths.json').read_text())
            x=np.array(observations['x']);tau=np.array(observations['tau']);mask=np.array(observations['fit_mask'])
            for strength in strengths:
                errors=[];passes=0
                for case in range(12):
                    obs=next(r for r in observations['surfaces'] if r['factors']==2 and r['case']==case)
                    truth=np.array(next(r for r in truths if r['factors']==2 and r['case']==case)['physical'])
                    iv=np.array(obs['iv']);std=None
                    if condition=='noise':
                        root=np.sqrt(tau)*iv;d2=x/root-.5*root
                        vega=np.exp(-.5*d2*d2)/np.sqrt(2*np.pi)*np.sqrt(tau)
                        std=.01*(np.array(obs['price'])-np.maximum(np.exp(x)-1,0))/np.maximum(vega,1e-12)
                    fit=fit_network(net,x,tau,iv,fit_mask=mask,starts=3,max_nfev=300,seed=911777+case,
                                    surrogate_error=stats,observation_iv_std=std,prior_strength=strength)
                    if fit['status']!='fitted':raise RuntimeError('Failed development fit')
                    delta=np.array(fit['physical'])-truth;scale=abs(truth);scale[3::5]=.5
                    tol=.05*abs(truth);tol[3::5]=.05;passed=bool((abs(delta)<=tol).all())
                    errors.extend(delta/scale);passes+=passed
                    records.append({'model':label,'condition':condition,'strength':strength,'case':case,
                                    'error':(delta/scale).tolist(),'pass':passed,'fit':fit})
                row={'model':label,'condition':condition,'strength':strength,
                     'parameter_rmse':float(np.sqrt(np.mean(np.square(errors)))),'passes':passes}
                summary.append(row);print(json.dumps(row),flush=True)
                (args.out/'fits.json').write_text(json.dumps(records));(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    selection={c:min([r for r in summary if r['condition']==c],key=lambda r:r['parameter_rmse']) for c in ['clean','noise']}
    (args.out/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
