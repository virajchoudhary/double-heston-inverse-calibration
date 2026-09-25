"""Larger matched-budget development experiment; see EXTENDED_PROTOCOL.md."""
import json
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from run_pilot import (ROOT,HERE,sha,dataset,geometry,structural,tensors,evaluate,
                       PortableVariancePINN,MaturityDualPINN,residual,T)

OUT=HERE/'extended_stratified'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def stratified_label_indices(case_ids, per_case=2):
    """Cover every case rather than a prefix of case-grouped labels."""
    selected=[]
    for case in np.unique(case_ids):
        idx=np.flatnonzero(case_ids==case)
        if len(idx)<per_case:raise ValueError('Too few labels in a training case')
        selected.extend(idx[np.linspace(0,len(idx)-1,per_case,dtype=int)])
    return torch.tensor(selected,dtype=torch.long)


def fit(job):
    kind,seed=job;torch.set_num_threads(1);torch.manual_seed(seed)
    a=dict(np.load(OUT/'train.npz'));col=dict(np.load(OUT/'collocation.npz'))
    net=PortableVariancePINN() if kind=='single_branch' else MaturityDualPINN()
    with torch.no_grad():
        net.head.weight.zero_();net.head.bias.zero_()
        if kind=='dual_branch':net.long_head.weight.zero_();net.long_head.bias.zero_()
    z,s,y,v,valid=tensors(a);cc,cs=T(col['coords']),structural(col['params'])
    def loss(di,ci):
        pred=net.price(z[di],s[di])*torch.exp(-z[di,0]);pv=net.iv(z[di],s[di]);ok=valid[di]
        eq,dg=residual(net,cc[ci],cs[ci])
        il=(pv[ok]-v[di][ok]).square().mean() if ok.any() else pv.sum()*0
        value=(pred-y[di]).square().mean()/.01**2+il/.05**2+.01*eq.square().mean()+.01*torch.relu(-dg['convexity']).square().mean()
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite loss')
        return value
    rng=torch.Generator().manual_seed(seed+1000)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=.00001)
    start=time.monotonic();history=[]
    for step in range(4000):
        di=torch.randint(len(z),(128,),generator=rng);ci=torch.randint(len(cc),(32,),generator=rng)
        opt.zero_grad(set_to_none=True);value=loss(di,ci);value.backward()
        if not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite gradient')
        opt.step();scheduler.step()
        if step%500==0 or step==3999:
            row={'model':kind,'seed':seed,'step':step+1,'loss':float(value.detach()),'seconds':time.monotonic()-start}
            history.append(row);save(f'{kind}_{seed}_progress.json',history);print(row,flush=True)
    opt=torch.optim.LBFGS(net.parameters(),max_iter=100,history_size=20,line_search_fn='strong_wolfe')
    fine_labels=stratified_label_indices(a['case_id'])
    assert len(np.unique(a['case_id'][fine_labels.numpy()]))==len(a['cases'])
    def closure():
        opt.zero_grad(set_to_none=True);value=loss(fine_labels,torch.arange(128));value.backward();return value
    opt.step(closure)
    checkpoint=OUT/f'{kind}_{seed}.pt';torch.save(net.state_dict(),checkpoint)
    with torch.no_grad():
        pred=net.price(z,s)*torch.exp(-z[:,0]);vol=net.iv(z,s)
        train_rmse=float((pred-y).square().mean().sqrt())
        train_iv=float(100*(vol[valid]-v[valid]).square().mean().sqrt())
    b=dict(np.load(OUT/'development.npz'));metrics,pred=evaluate(net,b)
    dz,ds,*_=tensors(b);negative=0
    for i in range(0,len(dz),64):
        _,dg=residual(net,dz[i:i+64],ds[i:i+64]);negative+=int((dg['convexity'] < -1e-7).sum())
    np.savez_compressed(OUT/f'{kind}_{seed}_development_predictions.npz',pred=pred,reference=b['price'],case_id=b['case_id'])
    row={'model':kind,'seed':seed,'parameters':sum(p.numel() for p in net.parameters()),
         'seconds':time.monotonic()-start,'train_price_RMSE':train_rmse,'train_IV_RMSE_vol_points':train_iv,
         'development_negative_convexity_count':negative,'weights_sha256':sha(checkpoint),**metrics}
    save(f'{kind}_{seed}_metrics.json',row);print(row,flush=True);return row


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    sources=[HERE/'EXTENDED_PROTOCOL.md',HERE/'EXTENDED_AMENDMENT_01.md',HERE/'run_extended.py',HERE/'run_pilot.py',
             ROOT/'src/mentor_dh_pinn/maturity_dual_pinn.py',ROOT/'src/mentor_dh_pinn/regular_pinn_torch.py',
             ROOT/'experiments/btc_multifactor_v1/engine.py',ROOT/'src/double_heston_reference.py',
             ROOT/'src/mentor_dh_pinn/regular_pinn_data.py']
    save('manifest.json',{'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
         'market_labels_used':False,'role':'extended development, designed after pilot'})
    a,fa=dataset(256,97001);b,fb=dataset(64,97002)
    exposed=set(map(tuple,a['cases']))
    for name in ['train','development']:exposed|=set(map(tuple,np.load(HERE/'pilot'/f'{name}.npz')['cases']))
    assert not set(map(tuple,b['cases']))&exposed
    rng=np.random.default_rng(97003);p=a['cases'][rng.integers(256,size=18000)]
    col={'coords':geometry(p,97003),'params':p}
    for name,data in [('train',a),('development',b),('collocation',col)]:np.savez_compressed(OUT/f'{name}.npz',**data)
    save('data_audit.json',{'train_cases':256,'development_cases':64,'train_labels':len(a['price']),
         'development_labels':len(b['price']),'collocation':18000,'train_fallbacks':fa,'development_fallbacks':fb,
         'train_invalid_IV':int((~np.isfinite(a['iv'])|(a['iv']<=0)).sum()),
         'development_invalid_IV':int((~np.isfinite(b['iv'])|(b['iv']<=0)).sum()),
         'new_development_disjoint_from_all_training_and_old_development':True})
    jobs=[(kind,seed) for seed in [17,43] for kind in ['single_branch','dual_branch']]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(fit,jobs))
    means={kind:{k:float(np.mean([r[k] for r in rows if r['model']==kind]))
           for k in ['price_RMSE','IV_RMSE_vol_points','PDE_RMSE']} for kind in ['single_branch','dual_branch']}
    a0,b0=means['single_branch'],means['dual_branch']
    passed=b0['price_RMSE']<a0['price_RMSE'] and b0['IV_RMSE_vol_points']<a0['IV_RMSE_vol_points'] and b0['PDE_RMSE']<=1.1*a0['PDE_RMSE']
    save('results.json',{'rows':rows,'mean_seed_metrics':means,'development_criterion_passed':passed,
         'market_improvement_proven':False,'parameter_recovery_tested':False})
    print('COMPLETE',means,'development criterion:',passed,flush=True)


if __name__=='__main__':main()
