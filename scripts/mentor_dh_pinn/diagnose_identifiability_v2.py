#!/usr/bin/env python3
"""Symmetry audit, historical rescoring, and independent development information study.

No calibration or network training occurs here. Exact prices/Jacobians only
diagnose surfaces and evaluate fixed perturbations; no estimates are updated.
"""
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.identifiability import PARAMETERS,SWAP,canonicalize_parameters,recovery_metrics,parameter_error_summary
from src.mentor_dh_pinn.recovery_diagnostics import price_and_jacobian,physical_to_unit,information_diagnostics
from src.mentor_dh_pinn.regular_pinn_data import decode_unit


def read(p):return json.loads(p.read_text())
def write_csv(p,rows):
    if not rows:return
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def historical_records():
    records=[];sources=set();covered=set();unavailable=[]
    def add(source,group,case,p,t):
        if p is None or t is None:
            unavailable.append({'source':str(source),'case':str(case),'reason':'missing estimate or truth'});return
        if len(p)!=10 or len(t)!=10:return
        records.append({'source':str(source.relative_to(ROOT)),'group':group,'case':str(case),
                        'prediction':np.asarray(p),'truth':np.asarray(t)})
        sources.add(source)
    for name in ['truths.json','synthetic_truth.json']:
        for tp in (ROOT/'outputs').rglob(name):
            if 'identifiability_recovery_v2' in str(tp):continue
            fp=next((tp.parent/n for n in ['fits.json','fits_and_starts.json'] if (tp.parent/n).exists()),None)
            if fp is None:continue
            truths=read(tp)
            if not isinstance(truths,list):continue
            sources.add(tp);covered.add(tp.parent)
            truthmap={(str(t.get('generator_factors',t.get('factors'))),str(t['case'])):t for t in truths}
            for fit in read(fp):
                factors=str(fit.get('generator_factors',fit.get('factors')))
                if factors!='2' or len(fit.get('physical',[]))!=10:continue
                t=truthmap.get((factors,str(fit['case'])))
                group=str(fit.get('model','unknown'))+'|'+str(fit.get('geometry','rich'))+'|'+str(fit.get('noise','directory'))
                add(fp,group,fit['case'],fit.get('physical'),t.get('physical') if t else None)
    # Older experiments with only row-oriented physical parameter tables.
    for p in (ROOT/'outputs').rglob('parameter_recovery.csv'):
        if p.parent in covered or 'identifiability_recovery_v2' in str(p):continue
        grouped=defaultdict(list)
        with p.open() as f:
            for r in csv.DictReader(f):
                key=tuple(r.get(k,'') for k in ['case','model','arm','geometry','noise','generator_factors','factors'])
                grouped[key].append(r)
        for key,rows in grouped.items():
            if len(rows)!=10:continue
            def index(r):
                if 'parameter_index' in r:return int(r['parameter_index'])
                parameter=r['parameter'];base=parameter.split('_')[0]
                factor=int(r['factor'])-1 if 'factor' in r else int(parameter.endswith('_fast'))
                return 5*factor+['kappa','theta','sigma','rho','v0'].index(base)
            rows.sort(key=index)
            pred=[float(r.get('estimate',r.get('estimated'))) for r in rows]
            true=[float(r.get('truth',r.get('true'))) for r in rows]
            add(p,'|'.join(key[1:]),key[0],pred,true)
    # Every saved training recovery check contains its own explicit truth.
    for p in (ROOT/'outputs').rglob('recovery_validation_*.json'):
        if 'identifiability_recovery_v2' in str(p):continue
        for row in read(p):
            if len(row.get('true',[]))==10:add(p,p.parent.name,p.stem+'_'+str(row['case']),row.get('estimated'),row['true'])
    # GLS development diagnostics identify their truth seed in the manifest.
    for label in ['deep11_gls_development','deep17_gls_development']:
        folder=ROOT/'outputs/deeper_pinn'/label;p=folder/'fits.json'
        if not p.exists():continue
        m=read(folder/'manifest.json');sources.add(folder/'manifest.json')
        true=decode_unit(np.random.default_rng(m['recovery_seed']).uniform(.1,.9,(4,10)),2)
        for r in read(p):add(p,str(r['shrinkage']),r['case'],r['fit']['physical'],true[r['case']])
    p=ROOT/'outputs/deeper_pinn/prior_development_911/fits.json'
    if p.exists():
        tp=ROOT/'outputs/deeper_pinn/recovery_clean_910831/truths.json';sources.add(tp)
        truth={r['case']:r['physical'] for r in read(tp) if r['factors']==2}
        for r in read(p):add(p,f"{r['model']}|{r['condition']}|{r['strength']}",r['case'],r['fit']['physical'],truth[r['case']])
    return records,sources,unavailable


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cases',type=int,default=24);ap.add_argument('--seed',type=int,default=912201)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    x=np.tile(-np.log(np.linspace(.8,1.2,21)),6);tau=np.repeat(np.array([30,60,90,180,365,730])/365,21)
    mask=np.tile(np.arange(21)%3!=2,6)
    units=np.random.default_rng(args.seed).uniform(.1,.9,(args.cases,10))
    physical=decode_unit(units,2);prices=[];jacobians=[];conditioning=[];sensitivity=[];tradeoffs=[];spectra=[]
    max_swap=max_quad=max_jac_quad=0.
    for case,p in enumerate(physical):
        price,jac=price_and_jacobian(p,x,tau)
        swapped,sj=price_and_jacobian(p[SWAP],x,tau)
        p96,j96=price_and_jacobian(p,x,tau,96)
        max_swap=max(max_swap,float(np.max(abs(price-swapped))))
        max_quad=max(max_quad,float(np.max(abs(price-p96))))
        max_jac_quad=max(max_jac_quad,float(np.max(abs(jac-j96))))
        np.testing.assert_allclose(jac,sj[:,SWAP],atol=1e-11,rtol=1e-9)
        prices.append(price);jacobians.append(jac)
        time_value=price-np.maximum(1-np.exp(-x),0);std=.01*time_value
        if (std<=0).any():raise ValueError('Invalid reference time value; no quote clipping')
        for noise in [False,True]:
            info=information_diagnostics(jac[mask],p,standard_deviation=std[mask] if noise else None)
            s=info['singular_values'];conditioning.append({'case':case,'noise_model':'1% time value' if noise else 'unweighted',
                'condition_number':info['condition_number'],'smallest_singular_value':s[-1],
                'numerical_rank':info['numerical_rank'],'effective_rank':info['relative_effective_rank_1e_8'],
                'noise_resolved_tolerance_directions':info['noise_resolved_tolerance_directions'],
                **{f'eigenvalue_{i+1}':v for i,v in enumerate(info['fisher_eigenvalues'])}})
            if noise:spectra.append(s)
        full=information_diagnostics(jac,p,standard_deviation=std)
        for quote in range(len(x)):
            for j,name in enumerate(PARAMETERS):
                sensitivity.append({'case':case,'quote':quote,'strike_over_spot':float(np.exp(-x[quote])),
                    'maturity_days':float(tau[quote]*365),'calibration_quote':bool(mask[quote]),'parameter':name,
                    'price_derivative':jac[quote,j],'noise_scaled_tolerance_sensitivity':abs(full['scaled_jacobian'][quote,j]),
                    'conditional_information_contribution':full['conditional_quote_residuals'][quote,j]**2})
        # Fixed perturbations along the weakest direction, not optimized profiles.
        info=information_diagnostics(jac[mask],p,standard_deviation=std[mask]);tol=.05*abs(p);tol[3::5]=.05
        direction=info['right_vectors'][-1];direction/=max(abs(direction))
        for amplitude in [-8.,-4.,-2.,2.,4.,8.]:
            candidate=p+amplitude*tol*direction
            if (candidate[[0,1,2,4,5,6,7,9]]<=0).any():continue
            candidate_u=physical_to_unit(candidate)
            if not np.isfinite(candidate_u).all() or (candidate_u<0).any() or (candidate_u>1).any():continue
            alt,_=price_and_jacobian(candidate,x,tau)
            diff=alt-price;delta=diff[mask]/std[mask]
            tradeoffs.append({'case':case,'amplitude_tolerances':amplitude,'truth':p.tolist(),'alternative':candidate.tolist(),
                              'max_parameter_tolerance_error':float(np.max(abs((candidate-p)/tol))),
                              'price_rmse_spot':float(np.sqrt(np.mean(diff[mask]**2))),
                              'noise_standardized_rms':float(np.sqrt(np.mean(delta**2))),
                              'gaussian_known_variance_kl':float(.5*(delta@delta))})
    assert max_quad<1e-9
    np.savez_compressed(args.out/'development_jacobians.npz',units=units,physical=physical,x=x,tau=tau,
                        fit_mask=mask,prices=np.array(prices),jacobians=np.array(jacobians))
    write_csv(args.out/'conditioning.csv',conditioning);write_csv(args.out/'quote_sensitivity.csv',sensitivity)
    (args.out/'tradeoff_examples.json').write_text(json.dumps(tradeoffs,indent=2))
    records,sources,unavailable=historical_records();groups=defaultdict(list)
    for r in records:groups[(r['source'],r['group'])].append(r)
    audit_rows=[];parameter_rows=[];rescore=[]
    for (source,group),rs in groups.items():
        pred=np.array([r['prediction'] for r in rs]);true=np.array([r['truth'] for r in rs])
        direct=recovery_metrics(pred,true,permutation_invariant=False)
        pi=recovery_metrics(pred,true);canonical_pred=canonicalize_parameters(pred);canonical_true=canonicalize_parameters(true)
        canonical=recovery_metrics(canonical_pred,canonical_true,permutation_invariant=False)
        audit_rows.append({'source':source,'group':group,'cases':len(rs),
            'direct_all_ten':int(direct['all_ten'].sum()),'canonical_all_ten':int(canonical['all_ten'].sum()),
            'permutation_all_ten':int(pi['all_ten'].sum()),'permutation_rescued':int((pi['all_ten']&~direct['all_ten']).sum()),
            'lower_error_swapped_assignments':int(np.sum(pi['swapped'])),
            'direct_scaled_rmse':float(np.sqrt(np.mean(direct['historical_scaled_rmse']**2))),
            'permutation_scaled_rmse':float(np.sqrt(np.mean(pi['historical_scaled_rmse']**2)))})
        stats,corr,metrics=parameter_error_summary(canonical_pred,canonical_true,permutation_invariant=False)
        parameter_rows.extend({'source':source,'group':group,**r} for r in stats)
        for i,r in enumerate(rs):
            rescore.append({'source':source,'group':group,'case':r['case'],'direct_passes':int(direct['pass_count'][i]),
                'canonical_passes':int(canonical['pass_count'][i]),'permutation_passes':int(pi['pass_count'][i]),
                'permutation_swapped':bool(pi['swapped'][i]),'permutation_tolerance_rmse':float(pi['tolerance_rmse'][i])})
    write_csv(args.out/'historical_symmetry_summary.csv',audit_rows)
    write_csv(args.out/'historical_rescored_cases.csv',rescore)
    write_csv(args.out/'historical_parameter_errors.csv',parameter_rows)
    focus={}
    for condition,folder in [('Clean','prior_clean_911831'),('Noise','prior_noise_911831'),('Boundary','prior_boundary_clean_911931')]:
        rs=[r for r in records if folder+'/' in r['source'] and r['group'].startswith('Selected regularized PINN|')]
        pred=canonicalize_parameters(np.array([r['prediction'] for r in rs]));true=canonicalize_parameters(np.array([r['truth'] for r in rs]))
        stats,corr,metrics=parameter_error_summary(pred,true,permutation_invariant=False)
        focus[condition]={'statistics':stats,'correlation':corr.tolist(),'pass_counts':metrics['pass_count'].tolist()}
        write_csv(args.out/f'{condition.lower()}_parameter_errors.csv',stats)
        write_csv(args.out/f'{condition.lower()}_error_correlations.csv',
                  [{'parameter':name,**dict(zip(PARAMETERS,corr[j]))} for j,name in enumerate(PARAMETERS)])
    summary={'status':'diagnostic_complete; full recovery objective incomplete','development_seed':args.seed,
             'development_cases':args.cases,'exact_swap_max_price_difference':max_swap,
             'price_quadrature_max_difference':max_quad,'jacobian_quadrature_max_difference':max_jac_quad,
             'historical_estimates_rescored':len(records),'historical_groups':len(groups),
             'historical_permutation_rescued':sum(r['permutation_rescued'] for r in audit_rows),
             'unavailable_estimates':unavailable,'focus':focus,
             'interpretation':'Factor addition is commutative in this exact model. Existing unit map already enforces kappa_s < kappa_f. Full-rank ill-conditioning is not proof of structural nonidentifiability.',
             'history_scope':'All paired truth/fit JSON, parameter_recovery CSV and saved training-recovery JSON located under outputs; starts within a fit are not counted as separate calibrated results.',
             'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for c,color in [('Clean','#176b61'),('Noise','#ca8743'),('Boundary','#5e6da8')]:
        axes[0].plot(PARAMETERS,[r['rmse'] for r in focus[c]['statistics']],marker='o',label=c,color=color)
        axes[1].plot(PARAMETERS,[r['pass_percent'] for r in focus[c]['statistics']],marker='o',label=c,color=color)
    axes[0].set_ylabel('Relative-positive / absolute-rho RMSE');axes[0].set_yscale('log')
    axes[1].set_ylabel('Parameter tolerance pass (%)');axes[1].set_ylim(-2,102)
    for ax in axes:ax.tick_params(axis='x',rotation=60);ax.legend()
    fig.savefig(args.out/'parameter_diagnostics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');im=ax.imshow(focus['Noise']['correlation'],vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(10),PARAMETERS,rotation=60);ax.set_yticks(range(10),PARAMETERS)
    ax.set_title('Noisy estimation-error correlation (historical)');fig.colorbar(im,ax=ax)
    fig.savefig(args.out/'parameter_error_correlation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for s in spectra:ax.semilogy(range(1,11),s,color='#176b61',alpha=.25)
    ax.axhline(1,color='black',ls='--',label='One noise unit per tolerance displacement');ax.legend()
    ax.set_xlabel('Singular direction');ax.set_ylabel('Noise-scaled tolerance sensitivity')
    fig.savefig(args.out/'singular_value_spectrum.png',dpi=180);plt.close(fig)
    for field,title in [('maturity_days','maturity'),('strike_over_spot','strike')]:
        values=sorted({r[field] for r in sensitivity});table=[];heat=[]
        for parameter in PARAMETERS:
            row=[]
            for v in values:
                subset=[r for r in sensitivity if r['parameter']==parameter and r[field]==v]
                val=float(np.mean([r['conditional_information_contribution'] for r in subset]))
                table.append({'parameter':parameter,field:v,'mean_conditional_information':val});row.append(val)
            heat.append(np.array(row)/max(max(row),1e-300))
        write_csv(args.out/f'sensitivity_by_{title}.csv',table)
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained');im=ax.imshow(heat,aspect='auto',cmap='viridis')
        ax.set_yticks(range(10),PARAMETERS);ax.set_xticks(range(len(values)),[f'{v:.3g}' for v in values],rotation=60)
        ax.set_xlabel(title);ax.set_title('Conditional information, normalized within each parameter');fig.colorbar(im,ax=ax)
        fig.savefig(args.out/f'sensitivity_by_{title}.png',dpi=180);plt.close(fig)
    manifest={'purpose':'diagnosis only; no model selection from historical assessment examples',
              'status':'complete','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['focus','source_hashes']},indent=2))


if __name__=='__main__':main()
