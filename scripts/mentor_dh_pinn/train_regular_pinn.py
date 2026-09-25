#!/usr/bin/env python3
"""Train the same regular PDE PINN family for Single and Double Heston.

The sensitivity loss is optional Sobolev supervision from training references.
The exact engine is never used in the deployed neural calibration objective.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from functools import partial
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.regular_pinn import RegularVariancePINN,residual
from src.mentor_dh_pinn.regular_pinn_data import coordinates,draw_points,decode_unit,teacher_labels


def array(x):return mx.array(np.asarray(x,dtype=np.float32))


def recovery_validation(model,factors,points):
    from scripts.mentor_dh_pinn.assess_regular_pinn import fit_network
    from src.mentor_dh_pinn.regular_pinn_data import invert_total_variance
    errors=[];records=[]
    for i,(q,targets,true_p) in enumerate(points):
        t=np.exp(q[:,1]);iv=np.sqrt(targets["w"]/t)
        fit=np.tile(np.arange(21)%3!=2,6)
        result=fit_network(model,q[:,0],t,iv,fit_mask=fit,starts=3,seed=906777,max_nfev=200)
        if result["status"] != "fitted":
            errors.extend([math.inf]*(5*factors))
            records.append({"case":i,"true":true_p.tolist(),"status":result["status"],
                            "fit":result,"success":False})
            continue
        # API uses canonical physical output. Truth enters scoring only.
        estimated=np.asarray(result["physical"])
        scale=np.abs(true_p).copy();scale[3::5]=.5
        error=(estimated-true_p)/scale
        errors.extend(error.tolist());records.append({"case":i,"true":true_p.tolist(),
                 "estimated":estimated.tolist(),"scaled_error":error.tolist(),"success":result["optimizer_success"]})
    return float(np.sqrt(np.mean(np.asarray(errors)**2))),records


def make_validation_surfaces(factors):
    rng=np.random.default_rng(906311 if factors==2 else 906111)
    units=rng.uniform(.1,.9,(4,5*factors));out=[]
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6)
    tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    for unit in units:
        q=np.column_stack([x,np.log(tau),np.broadcast_to(unit,(len(x),len(unit)))])
        labels=teacher_labels(q,factors,gradients=False)
        if not labels["usable"].all():raise ValueError("Validation surface numerical labels invalid")
        out.append((q,labels,decode_unit(unit,factors)))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--factors",type=int,choices=(1,2),required=True)
    ap.add_argument("--steps",type=int,default=12000);ap.add_argument("--seed",type=int,default=17)
    ap.add_argument("--width",type=int,default=160);ap.add_argument("--depth",type=int,default=5)
    ap.add_argument('--residual-blocks',type=int,default=0)
    ap.add_argument('--residual-width',type=int,default=384)
    ap.add_argument("--batch",type=int,default=1024);ap.add_argument("--pde-batch",type=int,default=256)
    ap.add_argument("--sensitivity",type=float,default=0.);ap.add_argument("--weight-pde",type=float,default=.2)
    ap.add_argument('--geometry-sensitivity',type=float,default=0.)
    ap.add_argument("--weight-decay",type=float,default=1e-6);ap.add_argument("--lr",type=float,default=.001)
    ap.add_argument("--constant-lr",action="store_true",help="Small-LR continuation without another warmup/decay cycle")
    ap.add_argument("--surfaces",type=Path,help="Independent grouped synthetic TRAINING references")
    ap.add_argument("--surface-weight",type=float,default=0.,help="Local parameter-bias loss; zero is additional-surface price control")
    ap.add_argument("--surface-batch",type=int,default=8)
    ap.add_argument("--resample-every",type=int,default=1000);ap.add_argument("--save-every",type=int,default=2000)
    ap.add_argument("--resume",type=Path);ap.add_argument("--skip-recovery",action="store_true")
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    cfg={**vars(args),"data":str(args.data),"out":str(args.out),"resume":str(args.resume) if args.resume else None,
         "surfaces":str(args.surfaces) if args.surfaces else None,
         "factors":args.factors,"tau_max":2.,"architecture":"regular implied-variance PDE PINN",
         "selection":"clean synthetic validation parameter recovery RMSE, fixed 4 cases; not final assessment",
         "calibration":"optimize frozen neural IV predictions only; no exact pricing in fit"}
    (args.out/"config.json").write_text(json.dumps(cfg,indent=2))
    rng=np.random.default_rng(args.seed);mx.random.seed(args.seed)
    if args.residual_blocks:
        from src.mentor_dh_pinn.deep_regular_pinn import DeepRegularVariancePINN
        model=DeepRegularVariancePINN(args.factors,args.width,args.depth,
                    residual_blocks=args.residual_blocks,residual_width=args.residual_width)
        cfg['architecture']='one deep residual implied-variance PDE PINN'
        cfg['hidden_layer_count']=args.depth+2*args.residual_blocks
    else:
        model=RegularVariancePINN(args.factors,args.width,args.depth)
    if args.resume:
        if args.residual_blocks:
            initial=mx.load(str(args.resume));current=dict(tree_flatten(model.parameters()))
            missing=set(current)-set(initial);extra=set(initial)-set(current)
            if extra or any(not k.startswith('residual_blocks.') for k in missing):
                raise ValueError('Resume checkpoint must contain every backbone/head weight and no unexpected weights')
            if any(initial[k].shape!=current[k].shape for k in initial):
                raise ValueError('Resume layer shape mismatch')
        model.load_weights(str(args.resume),strict=not bool(args.residual_blocks))
    cfg['network_parameter_count']=sum(int(value.size) for _,value in tree_flatten(model.parameters()))
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    mx.eval(model.parameters())
    def read(name):
        d=dict(np.load(args.data/f"{name}.npz"));use=d["usable"]
        return {k:v[use] for k,v in d.items() if k!="usable"}
    train,val=read("train"),read("validation")
    tq,tg,tdg=(array(train[k]) for k in ("q","g","dg_du"))
    if args.geometry_sensitivity:
        if 'dg_dq' not in train:raise ValueError('Geometry supervision requires independently generated full derivative labels')
        tdg=array(train['dg_dq'])
    dq=array(np.load(args.data/"collocation.npz")["q"])
    sens_scale=array(np.maximum(np.sqrt(np.mean(train["dg_du"]**2,axis=0)),.02))
    if args.geometry_sensitivity:
        sens_scale=array(np.maximum(np.sqrt(np.mean(train['dg_dq']**2,axis=0)),.02))
    mx.eval(tq,tg,tdg,dq,sens_scale)
    surface_q=surface_iv=surface_b=mx.zeros((1,1));surface_count=1
    if args.surfaces:
        sd=dict(np.load(args.surfaces));use=sd['usable'];surface_count=int(use.sum())
        if not surface_count:raise ValueError('No usable grouped training surfaces')
        surface_q,surface_iv,surface_b=(array(sd[k][use]) for k in ('q','iv','preconditioner'))
        mx.eval(surface_q,surface_iv,surface_b)
    validation_surfaces=[] if args.skip_recovery else make_validation_surfaces(args.factors)
    warm=min(250,args.steps//10)
    schedule=optim.join_schedules([optim.linear_schedule(args.lr*.1,args.lr,warm),
                optim.cosine_decay(args.lr,max(args.steps-warm,1),args.lr*.02)],[warm])
    if args.constant_lr:schedule=args.lr
    opt=optim.AdamW(learning_rate=schedule,weight_decay=args.weight_decay)
    def correction(network,q):
        c,p=coordinates(q,args.factors,mx);return network.correction(c,p)
    def loss(aq,ag,adg,pq,sq,siv,sb):
        # nn.value_and_grad already injects the captured model's parameters;
        # passing that same model as an additional argument aliases its leaves
        # in MLX's compiled nested-AD graph.
        network=model
        predicted=correction(network,aq)
        anchor=mx.mean(((predicted-ag)/.05)**2)
        c,p=coordinates(pq,args.factors,mx);r,diag=residual(network,c,p)
        d2=c[:,0]/mx.sqrt(diag["w"])-.5*mx.sqrt(diag["w"])
        weight=mx.stop_gradient(mx.maximum(mx.exp(-d2*d2/2),.01))
        ar=mx.abs(r);huber=mx.where(ar<.1,.5*r*r,.1*(ar-.05))/.005
        physics=mx.sum(weight*huber)/mx.sum(weight)
        constraints=mx.mean(mx.maximum(-diag["convexity"],0)**2)
        constraints=constraints+mx.mean(mx.maximum(-diag["w"]*diag["l_tau"],0)**2)
        sensitivity=mx.array(0.)
        if args.sensitivity or args.geometry_sensitivity:
            # Pointwise rows: sum-gradient returns every row's parameter gradient.
            predicted_grad=mx.grad(lambda z:mx.sum(correction(network,z)))(aq)
            if args.geometry_sensitivity:
                scaled=((predicted_grad-adg)/sens_scale)**2
                sensitivity=mx.mean(scaled[:,2:])
                geometry=mx.mean(scaled[:,:2])
            else:
                sensitivity=mx.mean(((predicted_grad[:,2:]-adg)/sens_scale)**2)
                geometry=mx.array(0.)
        else:geometry=mx.array(0.)
        total=anchor+args.weight_pde*physics+args.sensitivity*sensitivity+.05*constraints+args.geometry_sensitivity*geometry
        group_price=bias=mx.array(0.)
        if args.surfaces:
            c,p=coordinates(sq.reshape(-1,sq.shape[-1]),args.factors,mx)
            error=network.iv(c,p).reshape(siv.shape)-siv
            group_price=mx.mean((error/.01)**2)
            local_shift=(sb@error[...,None])[...,0]
            # A robust local linear training metric, NOT recovered parameters.
            bias=mx.mean(mx.log1p(local_shift*local_shift))
            total=total+.1*group_price+args.surface_weight*bias
        return total,mx.stack([anchor,physics,sensitivity,constraints,group_price,bias,geometry])
    valuegrad=nn.value_and_grad(model,loss)
    state=[model.state,opt.state,mx.random.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(aq,ag,adg,pq,sq,siv,sb):
        (value,parts),grads=valuegrad(aq,ag,adg,pq,sq,siv,sb)
        grads,norm=optim.clip_grad_norm(grads,5.)
        opt.update(model,grads)
        return value,parts,norm
    start=time.perf_counter();history=[];best=math.inf;skips=0
    sources=[Path(__file__),ROOT/"src/mentor_dh_pinn/regular_pinn.py",ROOT/"src/mentor_dh_pinn/regular_pinn_data.py",
             ROOT/"src/mentor_dh_pinn/regular_pinn_torch.py",ROOT/"scripts/mentor_dh_pinn/assess_regular_pinn.py"]
    if args.residual_blocks:sources.append(ROOT/'src/mentor_dh_pinn/deep_regular_pinn.py')
    provenance={"config":cfg,"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "input_sha256":{str(args.data/p):hashlib.file_digest((args.data/p).open("rb"),"sha256").hexdigest() for p in ("train.npz","validation.npz","collocation.npz")},
                "gradient_label_scales":np.asarray(sens_scale).tolist(),"collocation_pool":18000}
    if args.resume:
        provenance["initial_checkpoint_sha256"]=hashlib.sha256(args.resume.read_bytes()).hexdigest()
        provenance["resume_policy"]="Continue network weights only; AdamW moments and learning-rate schedule restart. Not an independent initialization."
    if args.surfaces:
        provenance['input_sha256'][str(args.surfaces)]=hashlib.sha256(args.surfaces.read_bytes()).hexdigest()
        provenance['surface_training']={'usable':surface_count,'candidates':len(use),'group_price_weight':.1,
                'parameter_bias_weight':args.surface_weight,'loss':'mean(log1p((B*IV_error)^2)); local linear training proxy, not true recovery'}
    (args.out/"manifest.json").write_text(json.dumps(provenance,indent=2))
    (args.out/"source_snapshot.json").write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    for i in range(1,args.steps+1):
        a=array(rng.integers(len(train["q"]),size=args.batch)).astype(mx.int32)
        pi=array(rng.integers(len(dq),size=args.pde_batch)).astype(mx.int32)
        if args.surfaces:
            si=mx.array(rng.integers(surface_count,size=args.surface_batch),dtype=mx.int32)
            sq,siv,sb=surface_q[si],surface_iv[si],surface_b[si]
        else:sq,siv,sb=surface_q,surface_iv,surface_b
        value,parts,gn=step(tq[a],tg[a],tdg[a],dq[pi],sq,siv,sb);mx.eval(state,value,parts,gn)
        if not math.isfinite(float(value)) or not math.isfinite(float(gn)):
            # Do not keep training with corrupted optimizer state.
            (args.out/"FAILED.json").write_text(json.dumps({"step":i,"loss":str(value),"gradient_norm":str(gn)}))
            raise FloatingPointError(f"Nonfinite training state at step {i}")
        if args.resample_every and i%args.resample_every==0:
            dq=array(draw_points(18000,args.factors,args.seed+100000+i,collocation=True));mx.eval(dq)
        if i%200==0 or i==1 or i==args.steps:
            vq=array(val["q"]);c,p=coordinates(vq,args.factors,mx)
            iv=np.asarray(model.iv(c,p),dtype=float)
            target=np.sqrt(val["w"]/np.exp(val["q"][:,1]))
            iv_rmse=float(np.sqrt(np.mean((iv-target)**2)))
            rec={"step":i,"loss":float(value),"loss_parts":np.asarray(parts).tolist(),
                 "validation_iv_rmse":iv_rmse,"seconds":time.perf_counter()-start}
            if args.residual_blocks:
                rec['residual_output_weight_norm']=float(mx.sqrt(sum(mx.sum(b.down.weight**2) for b in model.residual_blocks)))
            if i%args.save_every==0 or i==args.steps:
                if validation_surfaces:
                    score,detail=recovery_validation(model,args.factors,validation_surfaces)
                    rec["validation_parameter_rmse"]=score
                    (args.out/f"recovery_validation_{i:06d}.json").write_text(json.dumps(detail,indent=2))
                else:score=iv_rmse
                model.save_weights(str(args.out/f"step_{i:06d}.safetensors"))
                if score<best:
                    best=score;model.save_weights(str(args.out/"model.safetensors"))
                    (args.out/"selection.json").write_text(json.dumps({"step":i,"score":score,
                            "metric":"validation_parameter_rmse" if validation_surfaces else "validation_iv_rmse"},indent=2))
            history.append(rec);(args.out/"history.json").write_text(json.dumps(history,indent=2))
            print(json.dumps(rec),flush=True)
    model.save_weights(str(args.out/"last.safetensors"))
    (args.out/"complete.json").write_text(json.dumps({"steps":args.steps,"seconds":time.perf_counter()-start,
                "selected_score":best,"finite_steps":args.steps,"optimizer":"AdamW"},indent=2))

if __name__=="__main__":main()
