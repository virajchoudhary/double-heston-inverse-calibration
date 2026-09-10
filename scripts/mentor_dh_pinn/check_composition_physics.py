"""Fresh-state raw Double Heston pricing-PDE audit of frozen neural compositions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,sha256
from src.mentor_dh_pinn.regular_pinn_data import draw_points,coordinates


@torch.enable_grad()
def raw_price_residual(model,state,structural):
    """Forward-price PDE at strictly positive maturities; structural inputs fixed."""
    c=state.detach().requires_grad_(True);p=structural.detach()
    price=model.price(c,p)
    derivative=lambda value:torch.autograd.grad(value.sum(),c,create_graph=True)[0]
    first=derivative(price);second_x=derivative(first[:,0])
    convexity=second_x[:,0]-first[:,0]
    residual=first[:,-1]-.5*c[:,1:-1].sum(-1)*convexity
    for i in range(2):
        k,theta,sigma,rho=p[:,i].unbind(-1)
        v=c[:,i+1];second_v=derivative(first[:,i+1])
        residual=residual-k*(theta-v)*first[:,i+1]-.5*sigma*sigma*v*second_v[:,i+1]-rho*sigma*v*second_x[:,i+1]
    return residual,{'price':price,'convexity':convexity,'calendar':first[:,-1]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--points',type=int,default=512)
    ap.add_argument('--seed',type=int,default=910101)
    ap.add_argument('--chunk',type=int,default=8)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    torch.set_num_threads(1)
    q=draw_points(args.points,2,args.seed,collocation=True)
    c,p=coordinates(torch.tensor(q,dtype=torch.float64),2,torch)
    result={'seed':args.seed,'points':len(q),'sampling':'independent diagnostic LHS; 1/4 wide log-moneyness',
        'selection_use':'none; frozen-checkpoint diagnostic, not training or selection',
        'pde':'C_tau - .5*(v1+v2)*(C_xx-C_x) - sum[k*(theta-v)*C_v + .5*sigma^2*v*C_vv + rho*sigma*v*C_xv]',
        'residual_units':'raw normalized-forward-call price per year; not normalized by small vega',
        'models':[],'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha256(__file__)}}
    arrays={'q':q}
    for path in args.checkpoint:
        model,info=load_checkpoint(path)
        if model.factors!=2:raise ValueError('This audit requires two factors')
        blocks=[]
        for start in range(0,len(c),args.chunk):
            residual,diagnostics=raw_price_residual(model,c[start:start+args.chunk],p[start:start+args.chunk])
            blocks.append({k:v.detach().numpy() for k,v in {'residual':residual,**diagnostics}.items()})
        values={key:np.concatenate([b[key] for b in blocks]) for key in blocks[0]}
        finite=all(np.isfinite(v).all() for v in values.values())
        rmse=lambda v:float(np.sqrt(np.mean(v*v))) if np.isfinite(v).all() else None
        r=values['residual'];wide=np.arange(len(r))%4==0
        row={'checkpoint':info,'all_finite':finite,'pde_rmse':rmse(r),'wide_pde_rmse':rmse(r[wide]),
             'core_pde_rmse':rmse(r[~wide]),'negative_convexity_count':int((values['convexity']< -1e-12).sum()),
             'negative_calendar_count':int((values['calendar']< -1e-12).sum()),
             'lower_price_bound_violations':int((values['price']<np.maximum(np.expm1(q[:,0]),0)-1e-12).sum()),
             'upper_price_bound_violations':int((values['price']>np.exp(q[:,0])+1e-12).sum())}
        result['models'].append(row)
        arrays.update({info['label']+'_'+k:v for k,v in values.items()})
        assert sha256(info['checkpoint'])==info['sha256']
        print(json.dumps({k:v for k,v in row.items() if k!='checkpoint'}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out.with_suffix('.npz'),**arrays)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
