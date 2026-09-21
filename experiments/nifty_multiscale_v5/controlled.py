"""Reused synthetic surfaces: descriptive maturity diagnostics, no reselection."""
import numpy as np
import pandas as pd
import torch

from . import run
from experiments.nifty_multifactor_v4.literature_exact import exact,bs_predict


def main():
    run.verify();selection=run.old.read(run.OUT/'selection.json')
    nets=[run.model(seed,selection['chosen'],True).eval() for seed in run.SEEDS]
    rows=[];source_hashes={}
    for case in run.old.read(run.old.OUT/'controlled_cases.json'):
        if case['kind']!='independent':continue
        path=run.old.OUT/'surfaces'/(case['id']+'.npz');bpath=run.old.OUT/'baselines'/(case['id']+'.json')
        source_hashes[str(path.relative_to(run.ROOT))]=run.old.sha(path)
        source_hashes[str(bpath.relative_to(run.ROOT))]=run.old.sha(bpath)
        a=np.load(path);b=run.old.read(bpath);hold=~a['calibration']
        x,t,y=(a[key][hold] for key in ['x','tau','price'])
        predictions={'DH_PINN':run.old.neural(nets,case['params'],x,t)[0],
                     'SH':exact(b['SH']['best']['params'],x,t),
                     'BS':bs_predict(b[b['BS_selected']],x,t)}
        days=365*t
        for bucket,mask in [('all',np.ones(len(y),bool)),('7-30d',days<=30),
                            ('30-90d',(days>30)&(days<=90)),('90-365d',(days>90)&(days<=365)),('365-730d',days>365)]:
            for name,pred in predictions.items():
                rows.append({'case':case['id'],'family':case['family'],'bucket':bucket,'model':name,
                    'quotes':int(mask.sum()),'price_RMSE':float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))})
    result=pd.DataFrame(rows);result.to_csv(run.OUT/'reused_controlled_maturity.csv',index=False)
    table=result.groupby(['family','bucket','model']).price_RMSE.mean().unstack()
    table.to_csv(run.OUT/'reused_controlled_summary.csv')
    run.save(run.OUT/'reused_controlled_manifest.json',{'utc':run.old.stamp(),'reused_not_fresh':True,
        'selection_sha256':run.old.sha(run.OUT/'selection.json'),'inputs':source_hashes,'source_sha256':run.old.sha(__file__)})
    print(table.loc[(slice(None),'all'),:].to_string(),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1);main()
