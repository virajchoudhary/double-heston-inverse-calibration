"""Read-only checkpoint diagnostics and figures. Does not select or retrain."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_pilot import ROOT, OUT, sha, tensors, PortableVariancePINN, MaturityDualPINN
from src.mentor_dh_pinn.regular_pinn_torch import residual


def main():
    dest=OUT.parent/'review';dest.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text())
    for f,h in manifest['source_sha256'].items(): assert sha(ROOT/f)==h,f
    r=json.loads((OUT/'results.json').read_text());torch.set_num_threads(1)
    train=dict(np.load(OUT/'train.npz'));dev=dict(np.load(OUT/'development.npz'))
    z,s,y,v,valid=tensors(train);dz,ds,*_=tensors(dev)
    diagnostics=[]
    for row in r['rows']:
        name,seed=row['model'],row['seed']
        net=PortableVariancePINN() if name=='single_branch' else MaturityDualPINN()
        path=OUT/f'{name}_{seed}.pt';net.load_state_dict(torch.load(path,weights_only=True));net.eval()
        with torch.no_grad():
            pred=net.price(z,s)*torch.exp(-z[:,0]);vol=net.iv(z,s)
            train_price=float((pred-y).square().mean().sqrt())
            train_iv=float(100*(vol[valid]-v[valid]).square().mean().sqrt())
        convex=[]
        for i in range(0,len(dz),64):
            _,d=residual(net,dz[i:i+64],ds[i:i+64]);convex.extend(d['convexity'].detach().numpy())
        convex=np.array(convex)
        diagnostics.append({**row,'train_price_RMSE':train_price,'train_IV_RMSE_vol_points':train_iv,
            'development_negative_convexity_count':int((convex < -1e-7).sum()),
            'development_convexity_min':float(convex.min()),'weights_sha256':sha(path)})
    paired=[]
    for seed in [17,43]:
        err={}
        for name in ['single_branch','dual_branch']:
            a=np.load(OUT/f'{name}_{seed}_development_predictions.npz')
            err[name]=[float(np.sqrt(np.mean((a['pred'][a['case_id']==c]-a['reference'][a['case_id']==c])**2))) for c in range(16)]
        paired.append({'seed':seed,'dual_wins_cases':int(np.sum(np.array(err['dual_branch'])<err['single_branch'])),
                       'case_price_RMSE':err})
    result={'checkpoint_diagnostics':diagnostics,'paired_cases':paired,
            'frozen_criterion_passed':r['development_criterion_passed'],
            'not_an_independent_market_test':True}
    (dest/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=['price_RMSE','IV_RMSE_vol_points','PDE_RMSE']
    titles=['Price RMSE (C / DF)','IV RMSE (volatility points)','Scaled PDE residual RMSE']
    fig,axs=plt.subplots(1,3,figsize=(12,4.4))
    for ax,key,title in zip(axs,keys,titles):
        for j,(name,color) in enumerate([('single_branch','#6c757d'),('dual_branch','#1f5fa8')]):
            vals=[row[key] for row in r['rows'] if row['model']==name]
            bar=ax.bar(j,np.mean(vals),width=.6,color=color)
            ax.scatter([j-.07,j+.07],vals,color='black',s=24,zorder=3)
            ax.bar_label(bar,fmt='%.4g',padding=3)
        ax.set_xticks([0,1],['Single branch','Dual branch']);ax.set_title(title,fontsize=11)
        ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('New synthetic development pilot — two seeds, 16 unseen parameter cases')
    fig.text(.5,.02,'Bars = mean across seeds; dots = individual seeds. Neither branch is Single Heston: both solve Double Heston.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.92]);fig.savefig(dest/'dual_pinn_pilot_errors.png',dpi=300);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
