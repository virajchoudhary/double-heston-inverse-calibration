#!/usr/bin/env python3
"""Audit and display completed factor-PINN development assessments, without fitting."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch


def audit_regular_comparison(folder,manifest,rows):
    """Same-case regression audit; reports roundoff instead of claiming bit identity."""
    truth=json.loads((folder/'synthetic_truth.json').read_text())
    fits=json.loads((folder/'fits_and_starts.json').read_text())
    fits=[r for r in fits if r['factors']==2 and r['geometry']=='rich' and r['noise']==0]
    assert len(truth)==len(fits)==len(rows)==12
    assert len({r['model'] for r in fits})==1
    with (folder/'observations.csv').open(newline='') as stream:quotes=list(csv.DictReader(stream))
    passed=0;complete=0;iv_difference=0.;price_difference=0.
    per_parameter=np.zeros(10,dtype=int);worst=[]
    geometry=manifest['geometry'];x=np.asarray(geometry['x'])
    for row in rows:
        tr=next(r for r in truth if int(r['case'].rsplit('_',1)[1])==row['case'])
        fit=next(r for r in fits if r['case']==tr['case'])
        assert fit['status']=='fitted' and row['status']=='fitted'
        np.testing.assert_array_equal(row['true_unit'],tr['unit'])
        np.testing.assert_array_equal(row['true_physical'],tr['physical'])
        assert len(fit['starts'])==len(row['fit']['starts'])==5
        for a,b in zip(fit['starts'],row['fit']['starts'],strict=True):
            np.testing.assert_array_equal(a['initial_unit'],b['initial_unit'])
        q=sorted((q for q in quotes if q['case']==tr['case']),key=lambda q:int(q['quote']))
        assert len(q)==126
        np.testing.assert_array_equal([int(q['quote']) for q in q],np.arange(126))
        assert np.isfinite([[float(q[key]) for key in ('observed_iv','observed_price_spot')] for q in q]).all()
        np.testing.assert_allclose([float(q['strike']) for q in q],np.exp(-x),rtol=0,atol=1e-15)
        np.testing.assert_array_equal([q['holdout']=='True' for q in q],geometry['holdout'])
        np.testing.assert_allclose([float(q['tau']) for q in q],geometry['tau'],rtol=0,atol=1e-15)
        iv_difference=max(iv_difference,float(np.max(np.abs(np.array([float(q['observed_iv']) for q in q])-row['observed_iv']))))
        price_difference=max(price_difference,float(np.max(np.abs(np.array([float(q['observed_price_spot']) for q in q])-
                              np.array(row['reference_price'])*np.exp(-x)))))
        p=np.asarray(tr['physical']);tol=.05*p;tol[3::5]=.05
        error=np.abs(np.asarray(fit['physical'])-p)/tol
        ok=error<=1;passed+=int(ok.sum());complete+=int(ok.all())
        per_parameter+=ok;worst.append(float(error.max()))
    return {'baseline':str(folder),'model':fits[0]['model'],'same_truth_vectors':True,'same_60_start_vectors':True,
            'input_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                            for name in ('synthetic_truth.json','fits_and_starts.json','observations.csv')},
            'same_holdout_masks':True,'max_observed_iv_difference':iv_difference,'max_spot_price_difference':price_difference,
            'per_parameter_pass_counts':per_parameter.tolist(),'per_case_max_gate_error':worst,
            'individual_parameter_passes':passed,'all_parameter_passes':complete}


def audit_training(manifest):
    folder=Path(manifest['checkpoint']).parent
    config=json.loads((folder/'config.json').read_text())
    assert config==manifest['training_config'], 'Training configuration changed after assessment'
    assert (folder/'complete.json').is_file(), 'Training did not finish'
    history=json.loads((folder/'history.json').read_text())
    assert np.isfinite([r['validation_coefficient_rmse'] for r in history]).all()
    selected=json.loads((folder/'selection.json').read_text())
    metric=config.get('selection_metric','validation_coefficient_rmse')
    assert metric in ('validation_coefficient_rmse','validation_vega_price_rmse')
    assert np.isfinite([r[metric] for r in history]).all()
    assert selected[metric]==min(r[metric] for r in history)
    key='step' if 'step' in selected else 'iteration'
    assert any(r[key]==selected[key] and r[metric]==selected[metric] for r in history)
    recorded=json.loads((folder/'manifest.json').read_text())
    snapshot=json.loads((folder/'source_snapshot.json').read_text())
    for name,digest in recorded['source_sha256'].items():
        assert hashlib.sha256(snapshot[name].encode()).hexdigest()==digest, f'Training source snapshot changed: {name}'
    chosen=folder/f'{key}_{selected[key]:06d}.pt'
    chosen_state=torch.load(chosen,map_location='cpu',weights_only=True)
    assessed_state=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=True)
    assert chosen_state.keys()==assessed_state.keys(), 'Selected checkpoint keys differ'
    assert all(torch.equal(chosen_state[k],assessed_state[k]) for k in chosen_state), 'Assessed weights differ from validation-selected weights'
    if config.get('resume'):
        digest=recorded.get('initial_checkpoint_sha256',recorded.get('initial_sha256'))
        assert digest and hashlib.sha256(Path(config['resume']).read_bytes()).hexdigest()==digest, 'Warm-start checkpoint changed'
    if 'price_surfaces' in config:
        price_config=config['price_surfaces'];path=Path(price_config['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==price_config['sha256']
        with np.load(path,allow_pickle=False) as archive:
            units=archive['unit'];iv=archive['iv'];q=archive['q']
            assert units.shape==(price_config['train']+price_config['validation'],10)
            assert iv.shape==q.shape[:2] and np.isfinite(iv).all() and (iv>0).all()
            assert archive['usable'].all() and np.isfinite(units).all() and np.isfinite(q).all()
            assert len({u.tobytes() for u in units})==len(units), 'Duplicate price-training/validation parameters'
            np.testing.assert_array_equal(q[:,:,2:],np.broadcast_to(units[:,None,:],q[:,:,2:].shape))
    data=Path(config.get('data',folder))
    keys={};counts={}
    for split in ('train','validation','collocation'):
        path=data/f'{split}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==recorded['input_sha256'][split]
        with np.load(path,allow_pickle=False) as arrays:
            q=arrays['q'];assert q.ndim==2 and q.shape[1]==6 and np.isfinite(q).all()
            assert (q[:,0]>0).all() and (q[:,1:3]>0).all()
            assert (np.abs(q[:,3])<1).all() and np.isin(q[:,5],[0,1]).all()
            if split!='collocation':
                assert arrays['targets'].shape==(len(q),4) and np.isfinite(arrays['targets']).all()
            keys[split]={row.tobytes() for row in q}
            assert len(keys[split])==len(q), f'Duplicate {split} states'
            counts[split]=len(q)
    assert not keys['train']&keys['validation']
    assert not keys['train']&keys['collocation']
    assert not keys['validation']&keys['collocation']
    return counts


def audit_assessment(folder):
    read=lambda name:json.loads((folder/name).read_text())
    manifest,rows,summary=read('manifest.json'),read('cases.json'),read('summary.json')
    assert manifest['status']=='complete' and manifest['frozen_hashes_rechecked']
    assert len(rows)==manifest['cases']==summary['cases']
    assert len({r['case'] for r in rows})==len(rows)
    assert manifest['gates']=={'positive_relative':.05,'rho_absolute':.05,'heldout_price_rmse_spot':1e-5}
    snapshot=read('source_snapshot.json')
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256(snapshot[name].encode()).hexdigest()==digest, name
    weights=Path(manifest['checkpoint'])
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    corpus=manifest['training_config'].get('price_surfaces')
    if corpus:
        path=Path(corpus['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==corpus['sha256']
        with np.load(path,allow_pickle=False) as archive:
            for row in rows:
                assert not (archive['unit']==np.asarray(row['true_unit'])).all(axis=1).any(), 'Recovery truth occurs in price corpus'
    expected_quotes=sum(not v for v in manifest['geometry']['holdout'])
    for row in rows:
        if row['status']!='fitted':
            assert not row['all_parameter_pass'] and not row['joint_pass']
            continue
        truth=np.asarray(row['true_physical']);estimate=np.asarray(row['fit']['physical'])
        tolerance=.05*truth;tolerance[3::5]=.05
        error=np.abs(estimate-truth)/tolerance
        np.testing.assert_allclose(error,row['parameter_gate_units'],rtol=1e-13,atol=1e-13)
        assert row['individual_parameter_passes']==int((error<=1).sum())
        assert row['all_parameter_pass']==bool((error<=1).all())
        assert row['fit']['fit_quotes']==expected_quotes
        starts=row['fit']['starts'];assert len(starts)==manifest['starts']
        selected=starts[row['fit']['selected_start']]
        assert selected['sse']==min(s['sse'] for s in starts if s['sse'] is not None)
        assert selected['physical']==row['fit']['physical']
        if row['joint_pass']:
            assert row['all_parameter_pass']
            for key in ('neural','exact_reprice'):
                assert row[key]['price_gate'] and row[key]['invalid_iv_quotes']==0
            assert row['fitted_quadrature_error_spot']<=1e-8
    assert summary['all_parameter_passes']==sum(r['all_parameter_pass'] for r in rows)
    assert summary['individual_parameter_passes']==sum(r.get('individual_parameter_passes',0) for r in rows)
    assert summary['individual_denominator']==10*len(rows)
    assert summary['joint_passes']==sum(r['joint_pass'] for r in rows)
    return manifest,rows,summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--regular-baseline',type=Path)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    runs=[audit_assessment(path) for path in args.assessment]
    first,first_rows,_=runs[0]
    regular=audit_regular_comparison(args.regular_baseline,first,first_rows) if args.regular_baseline else None
    for manifest,rows,_ in runs[1:]:
        assert manifest['geometry']==first['geometry'] and manifest['gates']==first['gates']
        assert manifest['sampling_seed']==first['sampling_seed']
        for a,b in zip(first_rows,rows,strict=True):
            for key in ('case','true_unit','true_physical','reference_price','observed_iv'):
                assert a[key]==b[key], f'Case/quote mismatch: {key}'
    labels=[p.name.removesuffix('_development4').removesuffix('_exposed12') for p in args.assessment]
    training_checks=[audit_training(m) for m,_,_ in runs]
    names=[f'{name}_{factor}' for factor in (1,2) for name in ('kappa','theta','sigma','rho','v0')]
    counts=np.array([[sum(r['status']=='fitted' and r['parameter_gate_units'][j]<=1 for r in rows)
                      for j in range(10)] for _,rows,_ in runs])
    plot_labels=labels
    if regular:
        counts=np.vstack([regular['per_parameter_pass_counts'],counts])
        plot_labels=['Regular PINN baseline',*labels]
    fig,ax=plt.subplots(figsize=(12,max(2.8,.65*len(plot_labels)+1.5)))
    im=ax.imshow(counts,vmin=0,vmax=len(first_rows),cmap='Blues',aspect='auto')
    ax.set_xticks(range(10),names);ax.set_yticks(range(len(plot_labels)),plot_labels)
    for i in range(len(plot_labels)):
        for j in range(10):ax.text(j,i,f'{counts[i,j]}/{len(first_rows)}',ha='center',va='center',
                                  color='white' if counts[i,j]>len(first_rows)/2 else 'black')
    ax.set_title('Exact-reference parameter passes — exposed development cases')
    fig.colorbar(im,ax=ax,label='Passing cases')
    fig.tight_layout();fig.savefig(args.out/'parameter_passes.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    if regular:ax.plot(range(len(first_rows)),regular['per_case_max_gate_error'],'s-',color='gray',label='Regular PINN baseline')
    for label,(_,rows,_) in zip(labels,runs):
        error=[max(r['parameter_gate_units']) if r['status']=='fitted' else np.nan for r in rows]
        ax.plot(range(len(rows)),error,'o-',label=label)
    ax.axhline(1,color='black',linestyle='--',label='All ten parameters must be at or below 1')
    ax.set_yscale('log');ax.set_xticks(range(len(first_rows)))
    ax.set(xlabel='Development case',ylabel='Largest parameter error / allowed tolerance',
           title='Worst parameter controls whether a complete case passes')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(args.out/'worst_parameter.png',dpi=160);plt.close(fig)
    lines=['# Factor-structured PINN: development evidence', '',
        'This is a separately approved Riccati/factor architecture, not the regular price-PDE PINN.',
        'All cases here have been exposed during development. These results do not establish generalization.', '',
        '| Training checkpoint | Individual parameters | All ten, by case | Joint price + parameter gates |',
        '|---|---:|---:|---:|']
    for label,(_,_,s) in zip(labels,runs):
        lines.append(f"| {label} | {s['individual_parameter_passes']}/{s['individual_denominator']} | "
                     f"{s['all_parameter_passes']}/{s['cases']} | {s['joint_passes']}/{s['cases']} |")
    lines+=['', '| Training checkpoint | Neural held-out price gate | Exact repricing gate | Selected optimizer reports convergence |',
            '|---|---:|---:|---:|']
    for label,(_,rows,s) in zip(labels,runs):
        neural=sum(r.get('neural',{}).get('price_gate',False) for r in rows)
        exact=sum(r.get('exact_reprice',{}).get('price_gate',False) for r in rows)
        converged=sum(r.get('fit',{}).get('optimizer_success',False) for r in rows)
        lines.append(f"| {label} | {neural}/{s['cases']} | {exact}/{s['cases']} | {converged}/{s['cases']} |")
    lines+=['', 'Price gates in this table concern price RMSE alone; joint success also rejects invalid IVs. '
            'Optimizer convergence means a numerical stopping criterion was met, not that parameters are correct '
            'or globally optimal. Budget-limited starts and invalid fits remain in the underlying records.']
    if regular:
        lines+=['',f"The preserved regular-PINN baseline passes {regular['individual_parameter_passes']}/120 individual "
            f"parameters and {regular['all_parameter_passes']}/12 complete cases. All twelve truths, sixty blind starting "
            'vectors and holdout masks match. Regenerated reference IV differs by at most '
            f"{regular['max_observed_iv_difference']:.3g}; spot-normalized price differs by at most "
            f"{regular['max_spot_price_difference']:.3g}. These are not bit-identical quote files. "
            'Architectures and training supervision differ; this is a development benchmark, not a matched generalization study.']
    lines+=['','![Parameter pass counts](parameter_passes.png)','',
        'Interpretation: darker cells mean more development cases passed that parameter. '
        'Separate parameter passes cannot be combined across cases to claim full recovery.', '',
        '![Worst parameter error](worst_parameter.png)','',
        'Interpretation: a point above the dashed line fails at least one parameter. '
        'Lines connect case identifiers for readability; they are not time-series forecasts. Missing fits are failures, not omitted successes.', '',
        '## Data and safeguards','',
        'Each case has 126 clean synthetic quotes: 21 strike/forward ratios from 0.8 to 1.2 '
        'at 30, 60, 90, 180, 365 and 730 days, divided by 365. '
        'Every third strike is withheld (42 quotes); 84 quotes enter calibration. '
        'These maturities are synthetic experimental coverage, not a claim about NSE contract availability.', '',
        'Eight positive parameters must each be within 5% relative error; both correlations within 0.05 absolute. '
        'Canonical storage is slow factor first, fast factor second. Joint success also requires '
        'held-out neural and independent exact-repriced price RMSE <=1e-5 of spot and valid quadrature.', '',
        'Calibration uses frozen learned coefficients, automatic parameter derivatives and blind multistart optimization. '
        'Exact references generate the observations and independently reprice the final estimate; '
        'they are not trial-price calls inside inverse fitting. Structural coefficient labels are used during '
        'synthetic neural training, which must be disclosed in comparisons with price-only training.', '',
        'This is a two-stage PINN-surrogate method: train a parameter-conditioned coefficient network, '
        'then freeze its weights and optimize the ten unknown parameter inputs through its learned prices. '
        'It is not a network that directly outputs ten parameters, and it is not simultaneous per-case '
        'optimization of both neural weights and model parameters.', '',
        'The report recomputed parameter gates, counts, minimum-SSE start selection and quote counts; '
        'verified unchanged truths/quotes across runs; checked checkpoint/data hashes and recorded source snapshots; '
        'checked that assessed weights exactly match the validation-selected saved checkpoint and verified warm-start hashes; '
        'and checked finite training labels, unique sampled states and no exact state overlap between splits. '
        'This is a bounded integrity audit, not a guarantee against every possible form of leakage or overfitting.', '',
        'Coefficient-validation RMSE is not parameter error. The independent-output and derivative-linked '
        'variants use different coefficient-loss normalizations, so those RMSE values are not directly comparable.', '',
        '## What still needs work','',
        'Any failed complete-case gate remains unresolved. A fresh sealed case set, sensitivity/stability testing '
        'and a disclosed matched Single-Heston comparison are still needed before claiming reliable superiority.', '']
    for label,(manifest,_,_) in zip(labels,runs):
        if 'price_surfaces' in manifest['training_config']:
            config=manifest['training_config'];corpus=config['price_surfaces']
            lines+=['',f"Additional supervision for {label}: {corpus['train']} synthetic training surfaces and "
                    f"{corpus['validation']} reserved validation surfaces, each with 126 quotes. "
                    'This refinement selects weights by reserved target-vega-weighted price RMSE, not coefficient RMSE. '
                    'The loss approximates IV error locally; it is not exact IV MSE. The corpus was previously '
                    'used by regular-PINN training variants, so its reserved subset is not globally unseen evidence.']
            diagnostic=Path(manifest['checkpoint']).parent/'validation_audit.json'
            if diagnostic.exists():
                check=json.loads(diagnostic.read_text())
                assert check['checkpoint_sha256']==manifest['checkpoint_sha256']
                assert check['surface_corpus_sha256']==corpus['sha256']
                source=diagnostic.parent/'validation_audit_source.py'
                assert hashlib.sha256(source.read_bytes()).hexdigest()==check['source_sha256']
                lines+=['',f"Actual-IV diagnostic for {label}: {check['invalid_iv_quotes']}/{check['quotes']} "
                        'reserved quotes failed inversion/round-trip checks. Whole-set actual IV RMSE: '
                        f"{check['actual_iv_rmse'] if check['actual_iv_rmse'] is not None else 'undefined (failed quotes not dropped)'}. "
                        'No checkpoint was reselected using this post-selection diagnostic.']
    lines+=['[Per-case parameter estimates for the final listed stage](PARAMETERS.md). '
            'Full-precision values and every start remain in the source assessment JSON.', '']
    parameter_lines=['# Parameter estimates: '+labels[-1], '',
        'Development estimates, not successfully recovered ground truth. Gate units <=1 pass. '
        'Factor 1 is slow; factor 2 is fast. Values below are rounded for display only.', '',
        '| Case | Parameter | Truth | Fitted | Error / tolerance | Pass |',
        '|---:|---|---:|---:|---:|---|']
    for row in runs[-1][1]:
        if row['status']!='fitted':
            parameter_lines.append(f"| {row['case']} | Invalid/failed fit | — | — | — | No |")
            continue
        for j,name in enumerate(names):
            error=row['parameter_gate_units'][j]
            parameter_lines.append(f"| {row['case']} | {name} | {row['true_physical'][j]:.8g} | "
                f"{row['fit']['physical'][j]:.8g} | {error:.6g} | {'Yes' if error<=1 else 'No'} |")
    (args.out/'PARAMETERS.md').write_text('\n'.join(parameter_lines)+'\n')
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    result={'status':'passed','assessments':[str(p) for p in args.assessment],
            'checks':'snapshot/checkpoint hashes, recomputed parameter gates, counts, quote identity, start selection',
            'training_split_counts':training_checks,
            'regular_baseline_comparison':regular,
            'report_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'report_runtime':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),
                              'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,
                              'matplotlib':matplotlib.__version__,
                              'note':'Captured at reporting time, not retrospectively asserted as every historical training environment.'},
            'summaries':[s for _,_,s in runs]}
    (args.out/'report_source_snapshot.py').write_text(Path(__file__).read_text())
    (args.out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
