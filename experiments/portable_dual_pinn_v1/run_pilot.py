"""Frozen-budget synthetic development comparison; never reads BTC market labels."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/btc_multifactor_v1'))
from engine import exact, iv
from src.mentor_dh_pinn.maturity_dual_pinn import PortableVariancePINN, MaturityDualPINN
from src.mentor_dh_pinn.regular_pinn_torch import residual

HERE = Path(__file__).resolve().parent
OUT = HERE/'pilot'
T = lambda a: torch.as_tensor(a,dtype=torch.float64)
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def structural(p): return T(np.stack([p[:,0:4],p[:,5:9]],axis=1))
def logmap(u,lo,hi): return lo*np.exp(u*np.log(hi/lo))


def parameters(n,seed):
    u=qmc.LatinHypercube(10,seed=seed).random(n)
    p=np.empty((n,10)); p[:,0]=logmap(u[:,0],.3,3)
    p[:,5]=p[:,0]+logmap(u[:,5],2,15)
    for offset in [0,5]:
        p[:,offset+1]=logmap(u[:,offset+1],.005,1.5)
        p[:,offset+2]=logmap(u[:,offset+2],.15,2)*np.sqrt(2*p[:,offset]*p[:,offset+1])
        p[:,offset+3]=-.9+1.8*u[:,offset+3]
        p[:,offset+4]=logmap(u[:,offset+4],.005,1.5)
    return p


def geometry(p,seed):
    u=qmc.LatinHypercube(2,seed=seed).random(len(p))
    return np.column_stack([-1+2*u[:,0],p[:,4],p[:,9],logmap(u[:,1],3/365,2)])


def dataset(n,seed):
    cases=parameters(n,seed); p=np.repeat(cases,96,axis=0)
    coords=geometry(p,seed+100); prices=[]; fallbacks=[]
    for i,case in enumerate(cases):
        z=coords[i*96:(i+1)*96]
        prices.extend(exact(case,z[:,0],z[:,-1],feller=False,audit=fallbacks))
    y=np.array(prices); v=iv(y,coords[:,0],coords[:,-1])
    lower=np.maximum(1-np.exp(-coords[:,0]),0)
    if not np.isfinite(y).all() or (y<lower-1e-9).any() or (y>1+1e-9).any():
        raise FloatingPointError('Teacher price validity failure')
    return dict(cases=cases,params=p,coords=coords,price=y,iv=v,
                case_id=np.repeat(np.arange(n),96)),len(fallbacks)


def tensors(a):
    valid=np.isfinite(a['iv']) & (a['iv']>0)
    return T(a['coords']),structural(a['params']),T(a['price']),T(np.where(valid,a['iv'],0)),torch.tensor(valid)


def evaluate(net,a):
    z,s,y,v,valid=tensors(a)
    with torch.no_grad():
        pred=net.price(z,s)*torch.exp(-z[:,0]); pv=net.iv(z,s)
    eq=[]
    for start in range(0,len(z),64):
        eq.append(residual(net,z[start:start+64],s[start:start+64])[0].detach())
    e=(pred-y).detach().numpy()
    return {'price_RMSE':float(np.sqrt(np.mean(e*e))),
            'price_MAE':float(np.mean(abs(e))),'price_max':float(abs(e).max()),
            'IV_RMSE_vol_points':float(100*torch.sqrt(((pv[valid]-v[valid])**2).mean())),
            'PDE_RMSE':float(torch.cat(eq).square().mean().sqrt()),
            'IV_valid':int(valid.sum()),'quotes':len(y)},pred.numpy()


def train(kind,seed,a,col):
    torch.manual_seed(seed)
    net=PortableVariancePINN() if kind=='single_branch' else MaturityDualPINN()
    with torch.no_grad():
        net.head.weight.zero_(); net.head.bias.zero_()
        if kind=='dual_branch': net.long_head.weight.zero_(); net.long_head.bias.zero_()
    z,s,y,v,valid=tensors(a); cc,cs=T(col['coords']),structural(col['params'])
    def loss(di,ci):
        pred=net.price(z[di],s[di])*torch.exp(-z[di,0]); pv=net.iv(z[di],s[di]); ok=valid[di]
        eq,dg=residual(net,cc[ci],cs[ci])
        il=(pv[ok]-v[di][ok]).square().mean() if ok.any() else pv.sum()*0
        parts=torch.stack([(pred-y[di]).square().mean(),il,eq.square().mean(),torch.relu(-dg['convexity']).square().mean()])
        value=parts[0]/.01**2+parts[1]/.05**2+.01*parts[2]+.01*parts[3]
        if not torch.isfinite(value): raise FloatingPointError('Nonfinite loss; no hidden restart')
        return value
    # Identical minibatch sequence per seed, independent of architecture init.
    rng=torch.Generator().manual_seed(seed+1000)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000,eta_min=.00001)
    start=time.monotonic(); history=[]
    for step in range(1000):
        di=torch.randint(len(z),(128,),generator=rng); ci=torch.randint(len(cc),(32,),generator=rng)
        opt.zero_grad(set_to_none=True); value=loss(di,ci); value.backward()
        if not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite gradient')
        opt.step(); sched.step()
        if step%200==0 or step==999:
            row={'model':kind,'seed':seed,'step':step+1,'loss':float(value.detach()),'seconds':time.monotonic()-start}
            history.append(row); print(row,flush=True)
    opt=torch.optim.LBFGS(net.parameters(),max_iter=30,history_size=20,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(set_to_none=True); value=loss(torch.arange(512),torch.arange(128));value.backward();return value
    opt.step(closure)
    torch.save(net.state_dict(),OUT/f'{kind}_{seed}.pt')
    save(f'{kind}_{seed}_history.json',history)
    return net,{'seconds':time.monotonic()-start,'parameters':sum(p.numel() for p in net.parameters())}


def main():
    OUT.mkdir(exist_ok=False)
    sources=[HERE/'PROTOCOL.md',Path(__file__),ROOT/'src/mentor_dh_pinn/maturity_dual_pinn.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',ROOT/'experiments/btc_multifactor_v1/engine.py',
             ROOT/'src/double_heston_reference.py',ROOT/'src/mentor_dh_pinn/regular_pinn_data.py']
    save('manifest.json',{'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
         'market_data_used':False,'role':'development pilot; no final test','torch':torch.__version__})
    torch.set_num_threads(1)
    a,fa=dataset(64,96001); b,fb=dataset(16,96002)
    assert not set(map(tuple,a['cases'])) & set(map(tuple,b['cases']))
    rng=np.random.default_rng(96003); p=a['cases'][rng.integers(64,size=18000)]
    col={'coords':geometry(p,96003),'params':p}
    for name,data in [('train',a),('development',b),('collocation',col)]: np.savez_compressed(OUT/f'{name}.npz',**data)
    save('data_audit.json',{'train_cases':64,'development_cases':16,'train_labels':len(a['price']),
         'development_labels':len(b['price']),'collocation':18000,'train_fallbacks':fa,'development_fallbacks':fb,
         'train_invalid_IV':int((~np.isfinite(a['iv'])|(a['iv']<=0)).sum()),
         'development_invalid_IV':int((~np.isfinite(b['iv'])|(b['iv']<=0)).sum()),
         'splits_disjoint_by_complete_parameter_case':True})
    rows=[]
    for seed in [17,43]:
        for kind in ['single_branch','dual_branch']:
            net,info=train(kind,seed,a,col); metrics,pred=evaluate(net,b)
            np.savez_compressed(OUT/f'{kind}_{seed}_development_predictions.npz',pred=pred,reference=b['price'],case_id=b['case_id'])
            row={'model':kind,'seed':seed,**info,**metrics};rows.append(row)
            save('progress_metrics.json',rows);print(row,flush=True)
    means={kind:{key:float(np.mean([r[key] for r in rows if r['model']==kind]))
                for key in ['price_RMSE','IV_RMSE_vol_points','PDE_RMSE']} for kind in ['single_branch','dual_branch']}
    a0,b0=means['single_branch'],means['dual_branch']
    promising=(b0['price_RMSE']<a0['price_RMSE'] and b0['IV_RMSE_vol_points']<a0['IV_RMSE_vol_points'] and b0['PDE_RMSE']<=1.1*a0['PDE_RMSE'])
    save('results.json',{'rows':rows,'mean_seed_metrics':means,'development_criterion_passed':promising,
         'market_improvement_proven':False,'ten_parameter_recovery_tested':False})
    print('COMPLETE',means,'promising:',promising,flush=True)


if __name__=='__main__': main()
