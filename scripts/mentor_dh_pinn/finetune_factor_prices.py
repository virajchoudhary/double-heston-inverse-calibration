#!/usr/bin/env python3
"""Price-aware neural-weight refinement using separate synthetic training surfaces.

The price loss is target-vega-weighted price MSE, a LOCAL approximation to IV
MSE, not exact IV MSE. No inverse calibration or recovery-case truths are used.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import sha256
from src.mentor_dh_pinn.affine_factor_pinn import riccati_residual
from src.mentor_dh_pinn.affine_factor_pricing import neural_call_prices
from src.mentor_dh_pinn.conjugate_factor_pinn import build_factor_pinn
from src.mentor_dh_pinn.regular_pinn_data import black_call,decode_unit


def load_surfaces(path,validation_count):
    with np.load(path,allow_pickle=False) as archive:
        q,iv,unit,usable=(archive[k] for k in ('q','iv','unit','usable'))
    if q.ndim!=3 or q.shape[2]!=12 or iv.shape!=q.shape[:2] or unit.shape!=(len(q),10):
        raise ValueError('Invalid Double Heston training surface shapes')
    if not 0<validation_count<len(q) or not usable.all():
        raise ValueError('Require a nonempty fixed split and all candidate references valid; no silent exclusions')
    if not np.isfinite(q).all() or not np.isfinite(iv).all() or not (iv>0).all():
        raise ValueError('Nonfinite/nonpositive surface inputs')
    if not ((unit>=0)&(unit<=1)).all():raise ValueError('Unit parameters outside training domain')
    np.testing.assert_array_equal(q[:,:,2:],np.broadcast_to(unit[:,None,:],q[:,:,2:].shape))
    np.testing.assert_array_equal(q[:,:,:2],np.broadcast_to(q[0,:,:2],q[:,:,:2].shape))
    if len({u.tobytes() for u in unit})!=len(unit):raise ValueError('Duplicate parameter surfaces across/within splits')
    x=q[0,:,0];tau=np.exp(q[0,:,1]);root=iv*np.sqrt(tau)
    target=black_call(x,root**2)
    vega=np.exp(-.5*(x/root-root/2)**2)/np.sqrt(2*np.pi)*np.sqrt(tau)
    if not np.isfinite(vega).all() or not (vega>0).all():raise ValueError('Invalid target vega; no clipping')
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    # Labels are Black reconstructions of the archived canonical synthetic IVs.
    return {'physical':tensor(decode_unit(unit,2)),'target':tensor(target),'vega':tensor(vega),
            'x':tensor(x),'tau':tensor(tau),'training_count':len(q)-validation_count,
            'validation_count':validation_count,'unit':unit}


def surface_loss(model,surfaces,indices):
    """Known training inputs, learned prices only; no exact-pricer call or inverse fit."""
    if not len(indices):raise ValueError('Empty surface minibatch')
    terms=[]
    for i in indices:
        price=neural_call_prices(model,surfaces['physical'][i],surfaces['x'],surfaces['tau'])
        terms.append(((price-surfaces['target'][i])/surfaces['vega'][i]).square().mean())
    return torch.stack(terms).mean()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--surfaces',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--save-every',type=int,default=500)
    ap.add_argument('--surface-batch',type=int,default=2)
    ap.add_argument('--validation-surfaces',type=int,default=128);ap.add_argument('--seed',type=int,default=909101)
    args=ap.parse_args()
    if args.steps<1 or args.save_every<1 or args.surface_batch<1:ap.error('Positive steps/save interval/surface batch required')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);parent=json.loads((args.checkpoint/'config.json').read_text())
    model=build_factor_pinn(parent);initial=args.checkpoint/'model.pt'
    model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
    surface_manifest=json.loads((args.surfaces.parent/'manifest.json').read_text())
    assert sha256(args.surfaces)==surface_manifest['data_sha256'], 'Surface corpus changed'
    surfaces=load_surfaces(args.surfaces,args.validation_surfaces)
    arrays={s:dict(np.load(args.data/f'{s}.npz',allow_pickle=False)) for s in ('train','validation','collocation')}
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    tq,ty=tensor(arrays['train']['q']),tensor(arrays['train']['targets'])
    vq,vy=tensor(arrays['validation']['q']),tensor(arrays['validation']['targets'])
    pq=tensor(arrays['collocation']['q'])
    cfg={k:parent[k] for k in ('width','depth','integrated','conjugate','moment','two_moment','coefficient_target_type')}
    cfg.update(architecture='factor-structured Riccati PINN; additional price-aware training, NOT regular price-PDE PINN',
        data=str(args.data),resume=str(initial),steps=args.steps,seed=args.seed,optimizer='AdamW float64 CPU',
        train=len(tq),validation=len(vq),collocation=len(pq),batch=512,pde_batch=128,surface_batch=args.surface_batch,
        lr=.00001,weight_decay=1e-6,pde_weight=.2,price_weight=10.,
        price_loss='mean(((learned_price-Black_reconstructed_reference_price)/target_vega)^2); local IV-error approximation',
        selection_metric='validation_vega_price_rmse',selection='fixed reserved surface price validation only; initial candidate included',
        price_surfaces={'path':str(args.surfaces),'sha256':sha256(args.surfaces),
            'train':surfaces['training_count'],'validation':surfaces['validation_count'],
            'split':'first train rows, final validation rows; no resampling or rejected rows',
            'limitation':'This corpus was used by earlier regular-PINN training variants; reserved rows are validation for this factor refinement, not globally unseen evidence.'})
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    sources=[Path(__file__),ROOT/'scripts/mentor_dh_pinn/assess_regular_pinn.py',
             *[ROOT/'src/mentor_dh_pinn'/n for n in ('affine_factor_pinn.py','affine_factor_pricing.py',
               'conjugate_factor_pinn.py','integrated_factor_pinn.py','moment_factor_pinn.py','two_moment_factor_pinn.py',
               'regular_pinn_data.py')]]
    manifest={'initial_checkpoint_sha256':sha256(initial),
        'input_sha256':{s:sha256(args.data/f'{s}.npz') for s in arrays},
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'surface_manifest':surface_manifest,'no_recovery_cases':True}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.json').write_text(json.dumps({str(p.relative_to(ROOT)):p.read_text() for p in sources},indent=2))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    started=time.perf_counter();history=[];best=float('inf');visits=np.zeros(surfaces['training_count'],dtype=int)
    def record(step):
        nonlocal best
        with torch.no_grad():
            coefficient=float((model.raw_corrections(vq)-vy).square().mean().sqrt())
            score=float(surface_loss(model,surfaces,range(surfaces['training_count'],len(surfaces['unit']))).sqrt())
        if not np.isfinite([coefficient,score]).all():raise FloatingPointError('Nonfinite validation')
        row={'step':step,'validation_coefficient_rmse':coefficient,'validation_vega_price_rmse':score,
             'seconds':time.perf_counter()-started}
        history.append(row);(args.out/'history.json').write_text(json.dumps(history,indent=2))
        torch.save(model.state_dict(),args.out/f'step_{step:06d}.pt')
        if score<best:
            best=score;torch.save(model.state_dict(),args.out/'model.pt')
            (args.out/'selection.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
    record(0)
    for step in range(1,args.steps+1):
        optimizer.param_groups[0]['lr']=cfg['lr']*(.02+.98*(1+math.cos(math.pi*(step-1)/max(args.steps-1,1)))/2)
        idx=rng.integers(len(tq),size=512);cq=pq[rng.integers(len(pq),size=128)].clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        loss=(model.raw_corrections(tq[idx])-ty[idx]).square().mean()+.2*riccati_residual(model.coefficients,cq).square().mean()
        surface_indices=rng.integers(surfaces['training_count'],size=args.surface_batch);np.add.at(visits,surface_indices,1)
        loss=loss+10*surface_loss(model,surfaces,surface_indices)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite training gradient')
        optimizer.step()
        if step%args.save_every==0 or step==args.steps:record(step)
    torch.save(model.state_dict(),args.out/'last.pt')
    for p in sources:assert sha256(p)==manifest['source_sha256'][str(p.relative_to(ROOT))]
    assert sha256(args.surfaces)==cfg['price_surfaces']['sha256']
    assert sha256(initial)==manifest['initial_checkpoint_sha256']
    for split in arrays:assert sha256(args.data/f'{split}.npz')==manifest['input_sha256'][split]
    (args.out/'complete.json').write_text(json.dumps({'steps':args.steps,'seconds':time.perf_counter()-started,
        'selected_vega_price_rmse':best,'parameter_recovery_evaluated':False,
        'training_surface_visit_counts':visits.tolist(),'unique_training_surfaces_visited':int((visits>0).sum()),
        'validation_surfaces_used_for_gradient_updates':0},indent=2))


if __name__=='__main__':main()
