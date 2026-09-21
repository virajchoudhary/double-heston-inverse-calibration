"""Development-only selection, followed by one fresh synthetic fidelity opening."""
import argparse
import numpy as np
import pandas as pd
import torch
from .common import *
from .train import network,ARMS,verify
from experiments.nifty_multifactor_v4.literature_exact import batch_teacher,iv


def nets(arm):
    if arm!='BASELINE':
        for seed in SEEDS:
            folder=OUT/arm/f'seed_{seed}'
            if sha(folder/'weights.pt')!=read(folder/'completed.json')['weights_sha256']:
                raise RuntimeError(f'Checkpoint differs from completed training record: {folder}')
    return [network(arm,seed,None if arm=='BASELINE' else OUT/arm/f'seed_{seed}/weights.pt').eval() for seed in SEEDS]


def ensemble_metrics(models,a):
    pred=old.neural(models,a['params'],a['coords'][:,0],a['coords'][:,-1])[0]
    return old.metrics(a['price'],pred,a['coords'][:,0],a['coords'][:,-1]),pred


def passed(m):
    return (m['price_RMSE']<=2e-5 and m['price_P95']<=5e-5 and m['price_max']<=2e-4 and
            m['IV_RMSE_volatility_points'] is not None and m['IV_RMSE_volatility_points']<=.2)


def development():
    verify();a=data('development');col=data('collocation');col={k:v[:256] for k,v in col.items()}
    rows=[];seedrows=[];pre=[]
    for arm in ['BASELINE']+ARMS:
        if arm!='BASELINE' and not all((OUT/arm/f'seed_{s}/completed.json').exists() for s in SEEDS):continue
        models=nets(arm);m,pred=ensemble_metrics(models,a)
        phys=[];timing=[];counts=[];legacy=[];rss=[]
        for seed,net in zip(SEEDS,models):
            pde,_=physics(net,a);legacy.append(physics(net,col)[0]['PDE_RMSE']);phys.append(pde)
            mm,_=metrics(net,a);seedrows.append({'arm':arm,'seed':seed,**mm,**pde})
            if arm=='BASELINE':
                timing.append(read(prior.OUT/f'SHARED_s{seed}/completed.json')['seconds']);rss.append(float('nan'))
                counts.append(sum(p.numel() for p in net.parameters()))
            else:
                done=read(OUT/arm/f'seed_{seed}/completed.json');timing.append(done['seconds']);rss.append(done['peak_process_RSS_MB']);counts.append(done['parameters'])
                pre.append({'arm':arm,'seed':seed,**read(OUT/arm/f'seed_{seed}/before_lbfgs_metrics.json')})
        speedstats=[speed(n,a) for n in models]
        rows.append({'arm':arm,**m,'parameters_per_seed':counts[0],'ensemble_parameters':sum(counts),
            **{k:float(np.mean([p[k] for p in phys])) for k in phys[0]},'legacy_PDE_RMSE_mean':float(np.mean(legacy)),
            'training_seconds_mean':float(np.mean(timing)),'peak_RSS_MB':float(np.max(rss)),
            'inference_1024_ms_per_seed':float(np.mean([s['inference_1024_ms'] for s in speedstats])),
            'inference_single_ms_per_seed':float(np.mean([s['inference_single_ms'] for s in speedstats])),
            'throughput_per_seed':float(np.mean([s['throughput_quotes_s'] for s in speedstats])),
            'passes_four_gates':passed(m)})
    table=pd.DataFrame(rows);baseline=table[table.arm.eq('BASELINE')].iloc[0]
    table['passes_inherited_PDE_guard']=table.legacy_PDE_RMSE_mean<=1.1*baseline.legacy_PDE_RMSE_mean
    table['improves_baseline_price']=table.price_RMSE<baseline.price_RMSE
    table['eligible']=table.passes_four_gates & table.passes_inherited_PDE_guard & table.improves_baseline_price
    table.to_csv(OUT/'PINN_ARCHITECTURE_ABLATION.csv',index=False)
    pd.DataFrame(seedrows).to_csv(OUT/'per_seed_development.csv',index=False)
    pd.DataFrame(pre).to_csv(OUT/'before_lbfgs_comparison.csv',index=False)
    print(table[['arm','price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','eligible']].to_string(index=False),flush=True)
    return table


def select():
    verify()
    if (OUT/'selection.json').exists():raise RuntimeError('Already selected')
    table=development()
    required=ARMS[:5]
    for arm in required:
        if arm not in set(table.arm) and not any((OUT/arm).glob('seed_*/failed.json')):raise RuntimeError(f'Missing required arm {arm}')
    main=table[table.arm.isin(ARMS[2:5])]
    if not main.eligible.any() and 'ARCH_E_ADAPTIVE_ACTIVATION' not in set(table.arm) and not any((OUT/'ARCH_E_ADAPTIVE_ACTIVATION').glob('seed_*/failed.json')):
        raise RuntimeError('Predeclared conditional activation arm E is required')
    eligible=table[table.eligible].sort_values(['price_RMSE','IV_RMSE_volatility_points','parameters_per_seed','inference_1024_ms_per_seed'])
    chosen=None if len(eligible)==0 else eligible.iloc[0].arm
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.md'))+[ROOT/'src/mentor_dh_pinn/research_modified_pinn.py',OUT/'PINN_ARCHITECTURE_ABLATION.csv']
    files+=list(OUT.glob('*/seed_*/weights.pt'))
    save(OUT/'selection.json',{'utc':old.stamp(),'chosen':chosen,'reason':'all original gates plus inherited PDE guard and lower development price error required',
        'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'market_data_used':False,
        'final_synthetic_opened':False,'heldout_seed':206104,'heldout_PDE_seed':206105})
    print('SELECTED',chosen,flush=True)


def fidelity():
    verify();selected=read(OUT/'selection.json')
    for p,h in selected['files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError(f'Selection freeze mismatch: {p}')
    if selected['chosen'] is None:raise RuntimeError('NONE selected: final fidelity remains closed')
    dest=OUT/'final_fidelity';dest.mkdir(exist_ok=False)
    save(dest/'opened.json',{'utc':old.stamp(),'selection_sha256':sha(OUT/'selection.json')})
    cfg=read(old.HERE/'config.json');z,p=old.sample(cfg,4096,206104);y,audit=batch_teacher(p,z)
    vols=iv(y,z[:,0],z[:,-1]);a={'coords':z,'params':p,'price':y,'iv':vols,'iv_valid':np.isfinite(vols)}
    zc,pc=old.sample(cfg,512,206105);col={'coords':zc,'params':pc}
    np.savez_compressed(dest/'data.npz',**a);save(dest/'teacher_audit.json',audit)
    rows=[];sr=[]
    for arm in ['BASELINE',selected['chosen']]:
        models=nets(arm);m,pred=ensemble_metrics(models,a);np.save(dest/f'{arm}_predictions.npy',pred)
        ph=[]
        for seed,net in zip(SEEDS,models):
            pp,_=physics(net,col);ph.append(pp);mm,_=metrics(net,a);sr.append({'arm':arm,'seed':seed,**mm,**pp})
        rows.append({'arm':arm,**m,**{k:float(np.mean([p[k] for p in ph])) for k in ph[0]},'passes_four_gates':passed(m)})
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False);pd.DataFrame(sr).to_csv(dest/'per_seed.csv',index=False)
    save(dest/'completed.json',{'utc':old.stamp(),'data_sha256':sha(dest/'data.npz'),'selection_sha256':sha(OUT/'selection.json'),'no_reselection':True})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['development','select','fidelity']);args=ap.parse_args()
    torch.set_num_threads(1);globals()[args.command]()
