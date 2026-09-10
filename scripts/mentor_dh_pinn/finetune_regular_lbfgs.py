#!/usr/bin/env python3
"""Deterministic neural-weight L-BFGS; no assessment quotes enter training.

Samples, gradient-label normalization and PDE relevance weights are frozen for
the entire solve. The inverse calibration still evaluates only the neural model.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_unflatten
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.regular_pinn import residual
from src.mentor_dh_pinn.regular_pinn_data import coordinates
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,sha256
from scripts.mentor_dh_pinn.train_regular_pinn import make_validation_surfaces,recovery_validation,array


class FrozenObjective:
    def __init__(self,model,training,physics,scale,chunk=1024,pde_weight=.2,sensitivity=.2):
        self.model=model;self.chunk=chunk;self.factors=model.factors
        leaves=tree_flatten(model.trainable_parameters())
        self.layout=[(key,tuple(v.shape),v.size) for key,v in leaves]
        self.q,self.g,self.dg=(array(training[k]) for k in ('q','g','dg_du'))
        self.pq=array(physics);self.scale=array(scale)
        c,p=coordinates(self.pq,self.factors,mx)
        w=model.iv(c,p)**2*c[:,-1]
        d2=c[:,0]/mx.sqrt(w)-.5*mx.sqrt(w)
        # Freeze numerically, not just stop_gradient on a changing function.
        self.pw=array(np.asarray(mx.maximum(mx.exp(-d2*d2/2),.01)))
        self.mass=mx.sum(self.pw);mx.eval(self.q,self.g,self.dg,self.pq,self.pw,self.mass)
        n=len(self.q);npde=len(self.pq)
        def correction(q):
            c,p=coordinates(q,self.factors,mx);return model.correction(c,p)
        def anchor(q,g,dg):
            loss=mx.sum(((correction(q)-g)/.05)**2)/n
            if sensitivity:
                jac=mx.grad(lambda z:mx.sum(correction(z)))(q)[:,2:]
                loss=loss+sensitivity*mx.sum(((jac-dg)/self.scale)**2)/(n*5*self.factors)
            return loss
        def physics_loss(q,weight):
            c,p=coordinates(q,self.factors,mx);r,d=residual(model,c,p)
            ar=mx.abs(r);h=mx.where(ar<.1,.5*r*r,.1*(ar-.05))/.005
            shape=mx.maximum(-d['convexity'],0)**2+mx.maximum(-d['w']*d['l_tau'],0)**2
            return pde_weight*mx.sum(weight*h)/self.mass+.05*mx.sum(shape)/npde
        self.anchor_vg=mx.compile(nn.value_and_grad(model,anchor),inputs=model.state,outputs=model.state)
        self.physics_vg=mx.compile(nn.value_and_grad(model,physics_loss),inputs=model.state,outputs=model.state)
        self.calls=0;self.last=None

    def vector(self):return np.concatenate([np.asarray(v).reshape(-1) for _,v in tree_flatten(self.model.trainable_parameters())]).astype(float)

    def assign(self,vector):
        start=0;values=[]
        for key,shape,size in self.layout:
            values.append((key,array(vector[start:start+size]).reshape(shape)));start+=size
        if start!=len(vector):raise ValueError('Network vector size mismatch')
        self.model.update(tree_unflatten(values));mx.eval(self.model.parameters())

    def __call__(self,vector):
        self.assign(vector);value=mx.array(0.);gradient=mx.zeros((len(vector),))
        for start in range(0,len(self.q),self.chunk):
            sl=slice(start,start+self.chunk)
            val,gr=self.anchor_vg(self.q[sl],self.g[sl],self.dg[sl])
            value=value+val;gradient=gradient+mx.concatenate([v.reshape(-1) for _,v in tree_flatten(gr)])
            mx.eval(value,gradient)
        for start in range(0,len(self.pq),self.chunk):
            sl=slice(start,start+self.chunk);val,gr=self.physics_vg(self.pq[sl],self.pw[sl])
            value=value+val;gradient=gradient+mx.concatenate([v.reshape(-1) for _,v in tree_flatten(gr)])
            mx.eval(value,gradient)
        result=float(value),np.asarray(gradient,dtype=float)
        if not math.isfinite(result[0]) or not np.isfinite(result[1]).all():
            raise FloatingPointError('Nonfinite deterministic objective/gradient')
        self.calls+=1;self.last=result
        return result


def directional_check(objective,vector):
    value,gradient=objective(vector);direction=gradient/max(np.linalg.norm(gradient),1e-30)
    expected=float(gradient@direction);checks=[]
    for h in (1e-3,3e-4,1e-4):
        fd=(objective(vector+h*direction)[0]-objective(vector-h*direction)[0])/(2*h)
        checks.append({'step':h,'finite_difference':fd,'analytic':expected,
                       'relative_error':abs(fd-expected)/max(abs(expected),1e-12)})
    objective.assign(vector)
    if min(r['relative_error'] for r in checks)>.01:
        raise RuntimeError(f'Full-objective gradient check failed: {checks}')
    return checks


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--iterations',type=int,default=300)
    ap.add_argument('--anchors',type=int,default=16384);ap.add_argument('--collocation',type=int,default=18000)
    ap.add_argument('--chunk',type=int,default=1024);ap.add_argument('--seed',type=int,default=908021)
    ap.add_argument('--surfaces',type=Path,help='Independent grouped TRAINING surface references')
    ap.add_argument('--bias-gradient-share',type=float,default=.5,
                    help='Fixed initial quadratic-bias/base gradient ratio; zero is grouped-price control')
    ap.add_argument('--surface-chunk',type=int,default=64)
    ap.add_argument('--jacobian-gradient-share',type=float,
                    help='Use fixed derivative-consistency loss instead of quadratic bias; requires --surfaces')
    ap.add_argument('--save-every',type=int,default=100);ap.add_argument('--skip-recovery',action='store_true')
    args=ap.parse_args()
    if args.jacobian_gradient_share is not None:
        if not args.surfaces:ap.error('--jacobian-gradient-share requires --surfaces')
        args.bias_gradient_share=0.
    args.out.mkdir(parents=True,exist_ok=False)
    model,info=load_checkpoint(args.checkpoint)
    data=dict(np.load(args.data/'train.npz'));usable=np.flatnonzero(data['usable'])
    rng=np.random.default_rng(args.seed);indices=rng.choice(usable,args.anchors,replace=False)
    train={k:data[k][indices] for k in ('q','g','dg_du')}
    scale=np.maximum(np.sqrt(np.mean(data['dg_du'][usable]**2,axis=0)),.02)
    pool=np.load(args.data/'collocation.npz')['q'];pidx=rng.choice(len(pool),args.collocation,replace=False)
    if args.surfaces:
        if args.jacobian_gradient_share is not None:
            from scripts.mentor_dh_pinn.regular_lbfgs_jacobian_loss import FrozenJacobianObjective
            objective=FrozenJacobianObjective(model,train,pool[pidx],scale,args.chunk,
                surfaces=dict(np.load(args.surfaces)),jacobian_gradient_share=args.jacobian_gradient_share,
                surface_chunk=args.surface_chunk)
        else:
            from scripts.mentor_dh_pinn.regular_lbfgs_surface_loss import FrozenSurfaceObjective
            objective=FrozenSurfaceObjective(model,train,pool[pidx],scale,args.chunk,
                surfaces=dict(np.load(args.surfaces)),bias_gradient_share=args.bias_gradient_share,
                surface_chunk=args.surface_chunk)
    else:
        objective=FrozenObjective(model,train,pool[pidx],scale,args.chunk)
    # Describe this solve, rather than inheriting inapplicable Adam settings.
    # The complete parent checkpoint/config remains nested in the manifest.
    architecture={key:getattr(model,key) for key in ('factors','width','depth','tau_min','tau_max',
                                                   'x_half_width','correction_limit')}
    effective_config={'weight_decay':0.0,'resample_every':0,'gradient_clipping':False,
         'sensitivity':.2,'weight_pde':.2,'weight_shape':.05,'weight_anchor':1.0,'anchor_scale':.05,
         'iterations':args.iterations,'anchors':args.anchors,'collocation':args.collocation,'chunk':args.chunk,
         'optimizer':'L-BFGS-B on network weights, fixed objective; unbounded neural weights',
         'optimizer_options':{'maxiter':args.iterations,'maxls':40,'maxcor':20,'ftol':1e-12,'gtol':1e-8},
         'selection':('original four development recovery surfaces; never exposed-12 truth in optimizer'
                      if not args.skip_recovery else 'synthetic validation IV RMSE; recovery scoring skipped')}
    cfg={**architecture,**vars(args),**effective_config,'checkpoint':str(args.checkpoint),'data':str(args.data),'out':str(args.out),
         'optimizer':'L-BFGS-B on network weights, fixed objective; unbounded neural weights',
         'precision':'float32 MLX values/gradients, float64 SciPy optimizer vectors',
         'resume':str(args.checkpoint),'architecture':'regular implied-variance PDE PINN'}
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()}
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    (args.out/'effective_optimizer_settings.json').write_text(json.dumps({
         'effective_config':effective_config,
         'note':'Settings of this neural-weight L-BFGS solve. Parent Adam settings describe only the initial checkpoint; they are retained separately in manifest.initial_checkpoint.config.'},indent=2))
    sources=[Path(__file__),ROOT/'src/mentor_dh_pinn/regular_pinn.py',ROOT/'src/mentor_dh_pinn/regular_pinn_data.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
             ROOT/'scripts/mentor_dh_pinn/train_regular_pinn.py']
    if args.surfaces:sources.append(ROOT/'scripts/mentor_dh_pinn/regular_lbfgs_surface_loss.py')
    if args.jacobian_gradient_share is not None:
        sources.append(ROOT/'scripts/mentor_dh_pinn/regular_lbfgs_jacobian_loss.py')
    manifest={'initial_checkpoint':info,'seed':args.seed,'anchor_indices':indices.tolist(),'collocation_indices':pidx.tolist(),
              'input_sha256':{name:sha256(args.data/name) for name in ('train.npz','collocation.npz','validation.npz')},
              'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
              'frozen_weight_sha256':hashlib.sha256(np.asarray(objective.pw).tobytes()).hexdigest(),
              'gradient_scaling':scale.tolist(),'no_gradient_clipping':True,'no_objective_resampling':True,
              'exposed_cases_policy':'907931 cases are development from now on; not read by this training script'}
    if args.surfaces:
        manifest['surface_training']={**objective.surface_metadata,'path':str(args.surfaces),
                                     'input_sha256':sha256(args.surfaces)}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    vdata=dict(np.load(args.data/'validation.npz'));valid=vdata['usable'];vq=array(vdata['q'][valid])
    target=np.sqrt(vdata['w'][valid]/np.exp(vdata['q'][valid,1]))
    cases=[] if args.skip_recovery else make_validation_surfaces(model.factors)
    start=time.perf_counter();history=[];best=math.inf
    def record(iteration,vector):
        nonlocal best
        current_objective,_=objective(vector)
        c,p=coordinates(vq,model.factors,mx)
        iv=np.asarray(model.iv(c,p));rmse=float(np.sqrt(np.mean((iv-target)**2)))
        score,details=recovery_validation(model,model.factors,cases) if cases else (rmse,[])
        row={'iteration':iteration,'objective_calls':objective.calls,'objective':current_objective,
             'validation_iv_rmse':rmse,'validation_parameter_rmse':score if cases else None,'seconds':time.perf_counter()-start}
        history.append(row);(args.out/'history.json').write_text(json.dumps(history,indent=2))
        model.save_weights(str(args.out/f'iteration_{iteration:06d}.safetensors'))
        (args.out/f'recovery_validation_{iteration:06d}.json').write_text(json.dumps(details,indent=2))
        if score<best:
            best=score;model.save_weights(str(args.out/'model.safetensors'))
            (args.out/'selection.json').write_text(json.dumps({'iteration':iteration,'score':score},indent=2))
        print(json.dumps(row),flush=True)
    x0=objective.vector();checks=directional_check(objective,x0)
    (args.out/'directional_checks.json').write_text(json.dumps(checks,indent=2))
    record(0,x0);iteration=0
    def callback(vector):
        nonlocal iteration
        iteration+=1
        if iteration%args.save_every==0:record(iteration,vector)
    result=minimize(objective,x0,jac=True,method='L-BFGS-B',callback=callback,
                    options={'maxiter':args.iterations,'maxls':40,'maxcor':20,'ftol':1e-12,'gtol':1e-8})
    if history[-1]['iteration']!=result.nit:record(result.nit,result.x)
    objective.assign(result.x);model.save_weights(str(args.out/'last.safetensors'))
    (args.out/'complete.json').write_text(json.dumps({'iterations':int(result.nit),'function_evaluations':int(result.nfev),
            'optimizer_success':bool(result.success),'message':str(result.message),'final_objective':float(result.fun),
            'selected_score':best,'seconds':time.perf_counter()-start},indent=2))


if __name__=='__main__':main()
