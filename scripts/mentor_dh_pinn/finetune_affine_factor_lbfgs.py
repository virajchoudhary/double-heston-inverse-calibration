#!/usr/bin/env python3
"""Float64 deterministic factor-PINN weight refinement; no calibration cases."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.affine_factor_pinn import riccati_residual
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from scripts.mentor_dh_pinn.assess_regular_pinn import sha256


class FrozenFactorObjective:
    def __init__(self,model,training,physics,chunk=2048):
        self.model=model;self.q,self.y=training;self.physics=physics;self.chunk=chunk;self.calls=0
    def vector(self):return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().numpy().copy()
    def assign(self,vector):torch.nn.utils.vector_to_parameters(torch.tensor(vector,dtype=torch.float64),self.model.parameters())
    def __call__(self,vector):
        self.assign(vector);self.model.zero_grad(set_to_none=True);value=0.
        for start in range(0,len(self.q),self.chunk):
            sl=slice(start,start+self.chunk)
            loss=(self.model.raw_corrections(self.q[sl])-self.y[sl]).square().sum()/(4*len(self.q))
            value+=float(loss.detach());loss.backward()
        for start in range(0,len(self.physics),self.chunk):
            q=self.physics[start:start+self.chunk].detach().requires_grad_(True)
            loss=.2*riccati_residual(self.model.coefficients,q).square().sum()/(4*len(self.physics))
            value+=float(loss.detach());loss.backward()
        gradient=torch.cat([p.grad.reshape(-1) for p in self.model.parameters()]).detach().numpy().copy()
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite fixed factor objective')
        self.calls+=1
        return value,gradient


def gradient_check(objective,vector):
    _,g=objective(vector);direction=g/np.linalg.norm(g);expected=float(g@direction);checks=[]
    for h in (1e-4,1e-5,1e-6):
        fd=(objective(vector+h*direction)[0]-objective(vector-h*direction)[0])/(2*h)
        checks.append({'step':h,'analytic':expected,'finite_difference':fd,
                       'relative_error':abs(fd-expected)/max(abs(expected),1e-15)})
    objective.assign(vector)
    if min(c['relative_error'] for c in checks)>1e-4:raise RuntimeError('Factor objective gradient check failed')
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path);ap.add_argument('--iterations',type=int,default=400)
    ap.add_argument('--chunk',type=int,default=2048);ap.add_argument('--save-every',type=int,default=100)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=args.data or args.checkpoint
    parent=json.loads((args.checkpoint/'config.json').read_text())
    model=build_factor_pinn(parent)
    initial=args.checkpoint/'model.pt';model.load_state_dict(torch.load(initial,weights_only=True,map_location='cpu'))
    arrays={s:dict(np.load(data/f'{s}.npz')) for s in ('train','validation','collocation')}
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    objective=FrozenFactorObjective(model,(tensor(arrays['train']['q']),tensor(arrays['train']['targets'])),
        tensor(arrays['collocation']['q']),args.chunk)
    vq,vy=tensor(arrays['validation']['q']),tensor(arrays['validation']['targets'])
    cfg={'architecture':'factor-structured Riccati PINN; NOT regular price-PDE network','width':parent['width'],'depth':parent['depth'],
         'integrated':parent.get('integrated',False),'coefficient_target_type':parent.get('coefficient_target_type','log-real/amplitude-imaginary head corrections'),
         'conjugate':parent.get('conjugate',False),
         'moment':parent.get('moment',False),
         'two_moment':parent.get('two_moment',False),
         'optimizer':'float64 SciPy L-BFGS-B neural weights, fixed full objective','weight_decay':0.,'pde_weight':.2,
         'iterations':args.iterations,'chunk':args.chunk,'train':len(objective.q),'collocation':len(objective.physics),
         'selection':'independent coefficient validation RMSE; no recovery cases','resume':str(initial),'data':str(data)}
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    sources=[Path(__file__),ROOT/'src/mentor_dh_pinn/affine_factor_pinn.py',ROOT/'src/mentor_dh_pinn/conjugate_factor_pinn.py']
    if parent.get('integrated',False):sources.append(ROOT/'src/mentor_dh_pinn/integrated_factor_pinn.py')
    if parent.get('moment',False):sources.append(ROOT/'src/mentor_dh_pinn/moment_factor_pinn.py')
    if parent.get('two_moment',False):sources.append(ROOT/'src/mentor_dh_pinn/two_moment_factor_pinn.py')
    manifest={'initial_sha256':sha256(initial),'input_sha256':{s:sha256(data/f'{s}.npz') for s in arrays},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
              'no_resampling':True,'no_clipping':True,'no_case_truths':True}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    started=time.perf_counter();history=[];best=float('inf')
    def record(iteration,vector):
        nonlocal best
        value,_=objective(vector)
        with torch.no_grad():score=float((model.raw_corrections(vq)-vy).square().mean().sqrt())
        row={'iteration':iteration,'objective':value,'validation_coefficient_rmse':score,
             'objective_calls':objective.calls,'seconds':time.perf_counter()-started}
        history.append(row);(args.out/'history.json').write_text(json.dumps(history,indent=2))
        torch.save(model.state_dict(),args.out/f'iteration_{iteration:06d}.pt')
        if score<best:
            best=score;torch.save(model.state_dict(),args.out/'model.pt')
            (args.out/'selection.json').write_text(json.dumps({'iteration':iteration,'validation_coefficient_rmse':score},indent=2))
        print(json.dumps(row),flush=True)
    x=objective.vector();checks=gradient_check(objective,x)
    (args.out/'gradient_check.json').write_text(json.dumps(checks,indent=2))
    record(0,x);iteration=0
    def callback(vector):
        nonlocal iteration
        iteration+=1
        if iteration%args.save_every==0:record(iteration,vector)
    fit=minimize(objective,x,jac=True,method='L-BFGS-B',callback=callback,
        options={'maxiter':args.iterations,'maxcor':30,'maxls':30,'ftol':1e-14,'gtol':1e-10})
    if history[-1]['iteration']!=fit.nit:record(fit.nit,fit.x)
    objective.assign(fit.x);torch.save(model.state_dict(),args.out/'last.pt')
    (args.out/'complete.json').write_text(json.dumps({'iterations':int(fit.nit),'nfev':int(fit.nfev),
        'optimizer_success':bool(fit.success),'message':str(fit.message),'seconds':time.perf_counter()-started,
        'selected_coefficient_rmse':best,'parameter_recovery_evaluated':False},indent=2))


if __name__=='__main__':main()
