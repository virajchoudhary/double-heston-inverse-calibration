#!/usr/bin/env python3
"""Train a separately labelled factor-structured Riccati PINN in float64.

Only independent factor-state references are read/generated. No unknown-case
quotes or parameter truths are available here. Selection uses coefficient
validation loss, not exposed-case recovery. This is not the regular price PINN.
"""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.affine_factor_pinn import AffineFactorPINN,riccati_residual
from src.mentor_dh_pinn.affine_factor_reference import draw_factor_points,correction_targets
from src.mentor_dh_pinn.integrated_factor_pinn import linear_coefficient_targets
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=908241);ap.add_argument('--train',type=int,default=65536)
    ap.add_argument('--validation',type=int,default=8192);ap.add_argument('--collocation',type=int,default=18000)
    ap.add_argument('--batch',type=int,default=512);ap.add_argument('--pde-batch',type=int,default=128)
    ap.add_argument('--width',type=int,default=64);ap.add_argument('--depth',type=int,default=4)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--save-every',type=int,default=200)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--integrated',action='store_true',help='Enforce D=A_tau by automatic differentiation')
    ap.add_argument('--conjugate',action='store_true',help='Also enforce shifted/unshifted coefficient conjugacy')
    ap.add_argument('--moment',action='store_true',help='Also enforce exact first cumulant through an O(u^2) real correction')
    ap.add_argument('--two-moment',action='store_true',help='Also enforce the second cumulant with an analytic imaginary slope')
    args=ap.parse_args()
    if args.conjugate and not args.integrated:ap.error('--conjugate requires --integrated')
    if args.moment and not args.conjugate:ap.error('--moment requires --conjugate and --integrated')
    if args.two_moment and not args.moment:ap.error('--two-moment requires --moment, --conjugate and --integrated')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    model=build_factor_pinn(vars(args))
    resume_policy='No warm start'
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if args.integrated and saved['head.weight'].shape[0]==4:
            original=AffineFactorPINN(args.width,args.depth).double();original.load_state_dict(saved)
            model.load_primitive_from(original);resume_policy='Copy shared hidden weights and A-output rows only; D becomes A_tau'
        else:model.load_state_dict(saved);resume_policy='Continue all matching neural weights'
        if args.conjugate:resume_policy+='; enforce shifted/unshifted conjugacy (architecture change)'
        if args.moment:resume_policy+='; enforce first cumulant (real correction O(u^2))'
        if args.two_moment:resume_policy+='; enforce second cumulant (analytic slope plus O(u^3) imaginary correction)'
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    cfg={**vars(args),'out':str(args.out),'resume':str(args.resume) if args.resume else None,
         'architecture':'factor-structured Riccati PINN; NOT regular price-PDE architecture',
         'precision':'float64 PyTorch CPU','optimizer':'AdamW','weight_decay':1e-6,'pde_weight':.2,
         'lr_schedule':'cosine from lr to .02*lr across declared steps',
         'selection':'independent factor-coefficient validation RMSE; not parameter recovery',
         'resume_policy':resume_policy,
         'coefficient_target_type':'linear amplitude-normalized D/A corrections' if args.integrated else 'log-real/amplitude-imaginary head corrections',
         'data_policy':'independent synthetic factor-state coefficients; no calibration-case inputs'}
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    datasets={}
    for split,n,offset in [('train',args.train,0),('validation',args.validation,1),('collocation',args.collocation,2)]:
        q=torch.tensor(draw_factor_points(n,args.seed+offset),dtype=torch.float64)
        with torch.no_grad():target=None if split=='collocation' else correction_targets(q)
        if args.integrated and target is not None:target=linear_coefficient_targets(q,target)
        datasets[split]=(q,target)
        arrays={'q':q.numpy()}
        if target is not None:arrays['targets']=target.numpy()
        np.savez_compressed(args.out/f'{split}.npz',**arrays)
    sources=[Path(__file__),ROOT/'src/mentor_dh_pinn/affine_factor_pinn.py',
             ROOT/'src/mentor_dh_pinn/affine_factor_reference.py',ROOT/'src/mentor_dh_pinn/conjugate_factor_pinn.py']
    if args.integrated:sources.append(ROOT/'src/mentor_dh_pinn/integrated_factor_pinn.py')
    if args.moment:sources.append(ROOT/'src/mentor_dh_pinn/moment_factor_pinn.py')
    if args.two_moment:sources.append(ROOT/'src/mentor_dh_pinn/two_moment_factor_pinn.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'config':cfg,'input_sha256':{s:sha(args.out/f'{s}.npz') for s in datasets},
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
              'initial_checkpoint_sha256':sha(args.resume) if args.resume else None,
              'sampling':'LHS, half log-frequency and half 128-node Laguerre frequencies; independent split seeds',
              'collocation_note':'18000 is a fixed factor-state/frequency pool, not regular four-dimensional price collocation',
              'no_case_truth_labels':True}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    tq,ty=datasets['train'];vq,vy=datasets['validation'];pq,_=datasets['collocation']
    best=float('inf');history=[];started=time.perf_counter()
    def record(step,loss=None):
        nonlocal best
        with torch.no_grad():error=model.raw_corrections(vq)-vy;score=float(error.square().mean().sqrt())
        row={'step':step,'validation_coefficient_rmse':score,'training_loss':loss,'seconds':time.perf_counter()-started}
        history.append(row);(args.out/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        torch.save(model.state_dict(),args.out/f'step_{step:06d}.pt')
        if score<best:
            best=score;torch.save(model.state_dict(),args.out/'model.pt')
            (args.out/'selection.json').write_text(json.dumps({'step':step,'validation_coefficient_rmse':score},indent=2))
        print(json.dumps(row),flush=True)
    record(0)
    for step in range(1,args.steps+1):
        optimizer.param_groups[0]['lr']=args.lr*(.02+.98*(1+math.cos(math.pi*(step-1)/max(args.steps-1,1)))/2)
        # One shared row selection keeps coordinates and reference values paired.
        index=rng.integers(len(tq),size=args.batch);aq,target=tq[index],ty[index]
        cq=pq[rng.integers(len(pq),size=args.pde_batch)].clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        anchor=(model.raw_corrections(aq)-target).square().mean()
        physics=riccati_residual(model.coefficients,cq).square().mean()
        loss=anchor+.2*physics
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite factor PINN loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite factor PINN weight gradients')
        optimizer.step()
        if step%args.save_every==0 or step==args.steps:record(step,float(loss.detach()))
    torch.save(model.state_dict(),args.out/'last.pt')
    (args.out/'complete.json').write_text(json.dumps({'steps':args.steps,'seconds':time.perf_counter()-started,
        'selected_coefficient_rmse':best,'parameter_recovery_evaluated':False},indent=2))


if __name__=='__main__':main()
