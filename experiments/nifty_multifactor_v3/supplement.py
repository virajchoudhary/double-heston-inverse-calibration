"""Reporting/audit only: no training, calibration, model selection or filtering changes.

Added before final evaluation to make the requested error decomposition explicit.
The original frozen runner/report remain untouched. This file has its own hash.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT, HERE, verify, read, save, sha, stamp, sample, nets, tensor, structural
from src.mentor_dh_pinn.regular_pinn_torch import residual
from literature_exact import iv
from report import markdown


def freeze():
    verify()
    assert not (OUT/'controlled_results.json').exists()
    path=OUT/'reporting_addendum_manifest.json'
    assert not path.exists()
    save(path,{'utc':stamp(),'source_sha256':sha(__file__),
        'scope':'Read-only auditing and requested error decomposition; no changed models, scoring rules or hypotheses',
        'protocol_sha256':sha(OUT/'manifest.json')})


def audit():
    c=verify();meta=read(OUT/'reporting_addendum_manifest.json')
    assert meta['source_sha256']==sha(__file__)
    assert meta['protocol_sha256']==sha(OUT/'manifest.json')
    lock=read(OUT/'evaluation_lock.json')
    assert lock['manifest_sha256']==sha(OUT/'manifest.json')
    assert lock['market_selection_sha256']==sha(OUT/'market_selection.json')
    for seed,digest in lock['models'].items():assert digest==sha(OUT/f'pinn_s{seed}/weights.pt')
    for name,digest in lock['baselines'].items():assert digest==sha(OUT/'baselines'/name)
    source_count=0
    for p in OUT.glob('market_*_download.json'):
        for row in read(p)['rows']:
            if row['status']=='downloaded':
                assert sha(OUT/'market_sources'/row['file'])==row['sha256'];source_count+=1
    a=np.load(OUT/'teacher/train.npz');b=np.load(OUT/'teacher/development.npz');q=np.load(OUT/'teacher/collocation.npz')
    def fingerprints(z):return set(map(tuple,np.column_stack([z['coords'],z['params']])))
    atrain,bdev,qcoll=fingerprints(a),fingerprints(b),fingerprints(q)
    assert len(atrain)==len(a['coords']) and len(bdev)==len(b['coords'])
    assert not atrain&bdev and not atrain&qcoll and not bdev&qcoll
    for z in [a,b,q]:
        assert np.isfinite(z['coords']).all() and np.isfinite(z['params']).all()
        p=z['params'];assert np.all(p[:,0]<p[:,5])
        assert np.all(2*p[:,0]*p[:,1]>p[:,2]**2) and np.all(2*p[:,5]*p[:,6]>p[:,7]**2)
        assert np.all(abs(p[:,[3,8]])<1)
    for z in [a,b]:assert np.isfinite(z['price']).all()
    checks=[]
    for case in read(OUT/'controlled_cases.json'):
        z=np.load(OUT/'surfaces'/(case['id']+'.npz'));mask=z['calibration']
        assert mask.any() and (~mask).any() and len(mask)==2013
        assert len(set(zip(z['x'],z['tau'])))==len(mask)
        assert np.isfinite(z['price']).all()
        checks.append({'case':case['id'],'calibration_quotes':int(mask.sum()),'test_quotes':int((~mask).sum())})
    market={}
    for label in ['validation','final']:
        path=OUT/f'market_{label}/clean_quotes.csv'
        if not path.exists():continue
        z=pd.read_csv(path);keys=['date','expiry','strike','option']
        assert not z.duplicated(keys).any()
        days=(pd.to_datetime(z.expiry)-pd.to_datetime(z.date)).dt.days
        assert np.allclose(z.tau,days/365,rtol=0,atol=1e-14)
        assert z.date.between(*c['market_'+label]).all()
        anchor=set(map(tuple,z[z.split.eq('anchor')][keys].to_numpy()))
        held=set(map(tuple,z[z.split.eq('test')][keys].to_numpy()))
        assert not anchor&held
        market[label]={'dates':sorted(z.date.unique().tolist()),'quotes':len(z),'anchor_quotes':len(anchor),'heldout_quotes':len(held)}
    if 'final' in market:
        assert max(market['validation']['dates'])<min(market['final']['dates'])
    result={'utc':stamp(),'passed':True,'protocol_and_all_locked_artifacts_unchanged':True,
        'official_source_files_verified':source_count,'training_rows':len(atrain),'development_rows':len(bdev),
        'collocation_rows':len(qcoll),'exact_train_development_collocation_overlap':0,
        'controlled_splits':checks,'market':market,
        'limitations':['Exact overlap checks do not prove statistical independence or absence of all leakage.',
            'No claim that these new market dates were unseen by external teammates.',
            'Market quote filters use observed price/IV quality criteria fixed before final evaluation; this defines a conditional eligible population.',
            'Daily closes are not synchronized executable bid/ask quotes.']}
    save(OUT/('integrity_final.json' if 'final' in market else 'integrity_pre_evaluation.json'),result)
    print('LOCK/DATA INTEGRITY VERIFIED',flush=True)
    return c


def report():
    c=audit();out=OUT/'supplement';out.mkdir(exist_ok=True)
    quotes=pd.read_csv(OUT/'controlled_predictions.csv');buckets=pd.read_csv(OUT/'bucket_metrics.csv')
    metadata=pd.DataFrame(read(OUT/'controlled_cases.json'))[['id','family','kind']].rename(columns={'id':'case'})
    buckets=buckets.merge(metadata,on='case',validate='many_to_one');ind=buckets[buckets.kind.eq('independent')]
    pivot=ind.pivot(index=['case','family','bucket'],columns='model',values='price_RMSE').reset_index()
    pivot['SH_minus_DH']=pivot.SH_BEST_FOUND-pivot.DH_PINN
    summary=pivot.groupby(['family','bucket'])[['DH_PINN','SH_BEST_FOUND','BS_SELECTED','SH_minus_DH']].mean().reset_index()
    summary.to_csv(out/'family_bucket_errors.csv',index=False)
    # Independent PDE coordinates are the original frozen test sample, not new selections.
    coords,p=sample(c,c['pinn']['pde_test_points'],c['teacher_seeds']['pde_fidelity'])
    eqframe=pd.DataFrame(coords,columns=['x','v_slow','v_fast','tau'])
    for seed,net in zip(c['pinn']['seeds'],nets(c)):
        vals=[]
        for j in range(0,len(coords),128):vals.append(residual(net,tensor(coords[j:j+128]),structural(p[j:j+128]))[0].detach().numpy())
        eqframe[f'scaled_PDE_residual_s{seed}']=np.concatenate(vals)
    eqframe.to_csv(out/'pde_fidelity_residuals.csv',index=False)
    shape=[]
    for name,a in quotes[quotes.kind.eq('representative')].groupby('case'):
        for tau,b in a.groupby('tau'):
            b=b.sort_values('strike_over_forward');vol=iv(b.DH_EXACT_TEACHER.to_numpy(),b.x.to_numpy(),b.tau.to_numpy())
            good=np.isfinite(vol);k=b.strike_over_forward.to_numpy()
            shape.append({'case':name,'days':tau*365,'ATM_IV':float(np.interp(1,k[good],vol[good])),
                'smirk_IV_slope_KF_09_to_11':float((np.interp(1.1,k[good],vol[good])-np.interp(.9,k[good],vol[good]))/.2)})
    shape=pd.DataFrame(shape);shape.to_csv(out/'representative_teacher_shapes.csv',index=False)
    maturity=summary[summary.bucket.isin(['short','medium','long','very_long'])]
    text=['# Additional integrity checks and error decomposition',
        'This is reporting-only code recorded before final evaluation. It does not change the frozen experiment, models, filters, parameter choices, thresholds or scores.',
        '## Controlled hypotheses: descriptive bucket evidence',
        'Mean errors across eight independently sampled surfaces per family. Positive SH_minus_DH favors the PINN. These bucket summaries are descriptive; the five predeclared family-level confidence intervals remain the inferential tests.',
        markdown(maturity),
        'The representative_teacher_shapes.csv file contains ATM IV levels and smirk slopes for all representative scenarios and maturities. It can establish different surface shapes, not prove unique recovery of ten structural parameters.',
        '## Integrity',
        'Verified frozen source/manifest/selection/checkpoint/baseline hashes, all downloaded official source hashes, no duplicate quote keys, ACT/365 using actual expiries, disjoint anchor/held-out contracts and chronological validation/final intervals. Exact training/development/collocation overlaps are zero. This is evidence of the checks performed, not a guarantee against every possible bias or unknown external exposure.']
    path=OUT/'market_final_predictions.csv'
    if path.exists():
        a=pd.read_csv(path);scale=a.discount.to_numpy()*a.forward.to_numpy();target=a.target_forward_call.to_numpy()
        names=['single_fixed_exact','double_fixed_exact','double_fixed_PINN','BS_fixed',
            'single_state_exact','double_state_exact','double_state_PINN','BS_state_flat','BS_state_term']
        for name in names:
            a[name+'_residual_forward']=a[name]-target
            a[name+'_residual_index_points']=(a[name]-target)*scale
            a[name+'_predicted_OTM_index_points']=a.price+(a[name]-target)*scale
        decomp=[]
        for mode in ['fixed','state']:
            exact=a[f'double_{mode}_exact'].to_numpy();pinn=a[f'double_{mode}_PINN'].to_numpy();ok=np.isfinite(pinn)
            model=(exact-target)*scale;neural=(pinn-exact)*scale;total=(pinn-target)*scale
            assert np.allclose(total[ok],(model+neural)[ok],rtol=1e-10,atol=1e-9)
            for label,values in [('model_parameter',model),('PINN_approximation',neural),('total',total)]:
                a[f'double_{mode}_{label}_error_index_points']=values
                decomp.append({'protocol':mode,'component':label,'common_quotes':int(ok.sum()),'omitted_PINN_domain':int((~ok).sum()),
                    'RMSE_index_points':float(np.sqrt(np.mean(values[ok]**2))),'MAE_index_points':float(np.mean(abs(values[ok])))})
        a.to_csv(out/'market_all_predictions_and_residuals.csv',index=False)
        d=pd.DataFrame(decomp);d.to_csv(out/'market_error_decomposition.csv',index=False)
        m=pd.read_csv(OUT/'market_summary.csv')
        fixed=m[m.model.isin(names[:4])];state=m[m.model.isin(names[4:])]
        fixed.to_csv(out/'market_fixed_summary.csv',index=False);state.to_csv(out/'market_state_summary.csv',index=False)
        reductions=[]
        for kind in ['single','double']:
            before=float(m.loc[m.model.eq(kind+'_fixed_exact'),'RMSE_index_points'].iloc[0]);after=float(m.loc[m.model.eq(kind+'_state_exact'),'RMSE_index_points'].iloc[0])
            reductions.append({'model':kind,'fixed_RMSE':before,'state_adaptive_RMSE':after,'relative_RMSE_reduction_percent':100*(1-after/before)})
        red=pd.DataFrame(reductions);red.to_csv(out/'state_adaptation_changes.csv',index=False)
        rows=[]
        for name in names:
            for date,g in a.groupby('date'):
                z=g[name+'_residual_index_points'].to_numpy();good=np.isfinite(z)
                rows.append({'date':date,'model':name,'quotes':len(g),'scored_quotes':int(good.sum()),'RMSE_index_points':float(np.sqrt(np.mean(z[good]**2))) if good.any() else np.nan})
        daily=pd.DataFrame(rows);daily.to_csv(out/'market_daily_errors.csv',index=False)
        # Same predeclared maturity/moneyness cuts, including empty-bucket disclosure.
        rows=[];kf=np.exp(-a.x.to_numpy());tau=a.tau.to_numpy()
        masks={'short':tau<=30/365,'medium':(tau>30/365)&(tau<=90/365),'long':(tau>90/365)&(tau<=1),'very_long':tau>1,
            'ATM':abs(kf-1)<=.02,'wing':(kf<.9)|(kf>1.1)}
        for lo,hi in zip([.7,.9,.98,1.02,1.1],[.9,.98,1.02,1.1,1.30000001]):masks[f'K/F_{lo:.2f}_{hi:.2f}']=(kf>=lo)&(kf<hi)
        for label,mask in masks.items():
            for name in names:
                z=a[name+'_residual_index_points'].to_numpy();good=mask&np.isfinite(z)
                rows.append({'model':name,'bucket':label,'eligible_quotes':int(mask.sum()),'scored_quotes':int(good.sum()),'RMSE_index_points':float(np.sqrt(np.mean(z[good]**2))) if good.any() else np.nan})
        pd.DataFrame(rows).to_csv(out/'market_bucket_errors.csv',index=False)
        cols=['model','dates','quotes','RMSE_index_points','MAE_index_points','equal_date_forward_RMSE','IV_RMSE_volatility_points','omitted_nonfinite']
        text+=['## Strict fixed-parameter market results',markdown(fixed[cols]),'## Separate state-adaptive diagnostic',markdown(state[cols]),
            'State-adaptive PINN rows may cover fewer quotes when fitted states are outside the trained domain. They must not be compared as if their sample were identical; exact-model fixed/state comparisons below use all eligible quotes.',
            '## Signed error decomposition',markdown(d),
            'Pointwise total error = approximation error + exact-model error. RMSE components are NOT additive because the errors can reinforce or cancel each other.',
            '## What changing the current variance state accomplished',markdown(red),
            'These are measured reductions from changing only initial variance states while retaining the selected structural coefficients and the same eligible test quotes. They are a within-protocol comparison, not a causal estimate that all remaining error comes from any one source. States are fitted on same-date anchor quotes; this is surface reconstruction, not future-date forecasting.',
            'No ten-parameter recovery is claimed. The continuous coefficients originated in the cited DJIA study; the chosen NIFTY scaling was selected on validation, not on final dates.']
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        labels=['Single Heston','Double Heston'];xx=np.arange(2)
        axes[0].bar(xx-.18,red.fixed_RMSE,.36,label='Fixed state');axes[0].bar(xx+.18,red.state_adaptive_RMSE,.36,label='Anchor-fitted state')
        axes[0].set(xticks=xx,xticklabels=labels,ylabel='Market price RMSE (index points)',title='Same held-out NIFTY quotes');axes[0].legend()
        dd=d[d.protocol.eq('fixed')];axes[1].bar(dd.component,dd.RMSE_index_points)
        axes[1].set(ylabel='RMSE (index points)',title='Fixed Double Heston error components (not additive)');axes[1].tick_params(axis='x',rotation=15)
        fig.savefig(out/'market_error_decomposition.png',dpi=150);plt.close(fig)
        text+=['![Market diagnostic](market_error_decomposition.png)',
            'Interpretation: the left panel isolates the improvement from updating initial states. The right panel distinguishes mathematical-model mismatch from neural approximation. A small total error can partly reflect cancellation; inspect both components.']
    text+=['## Limits of this experiment',
        'The synthetic truth is Double Heston by design. It is a controlled approximation test, not neutral evidence of market superiority. Single Heston searches impose strict Feller and finite parameter bounds; results are conditional on those bounds. Twelve convergent starts do not prove global optimality. Only eight independent surfaces per family enter confidence intervals. Two training seeds are not a broad seed sensitivity study. The new market interval is short and is not the previously examined COVID/war periods. The fixed validation candidates may all be misspecified for the final market. No spread-based claims are possible from closing bhavcopies.']
    (out/'REPORT_ADDENDUM.md').write_text('\n\n'.join(text)+'\n')
    save(out/'output_manifest.json',{'utc':stamp(),'source_sha256':sha(__file__),
        'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.glob('*')) if p.is_file() and p.name!='output_manifest.json'}})
    print('REPORT ADDENDUM',out/'REPORT_ADDENDUM.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','audit','report']);args=parser.parse_args()
    torch.set_num_threads(1)
    {'freeze':freeze,'audit':audit,'report':report}[args.command]()
