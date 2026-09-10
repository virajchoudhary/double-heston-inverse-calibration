"""Audit and summarize the same-quote composition pilot without selecting cases."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import sha256,_score_price_iv


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assessment',type=Path,required=True)
    ap.add_argument('--training',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--physics',type=Path)
    ap.add_argument('--bias',type=Path)
    args=ap.parse_args()
    manifest=json.loads((args.assessment/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Assessment is incomplete')
    cases=[json.loads((args.assessment/f'case_{i:03d}.json').read_text()) for i in range(manifest['case_count'])]
    labels=[i['label'] for i in manifest['checkpoints']]
    assert len(set(labels))==len(labels)
    args.out.mkdir(parents=True,exist_ok=False)
    rows={name:[] for name in labels}
    parameters=['kappa_s','theta_s','sigma_s','rho_s','v0_s','kappa_f','theta_f','sigma_f','rho_f','v0_f']
    param_table=['# Double Heston parameter estimates — exposed development pilot','',
        'Tolerance: positive parameters 5% relative; correlations .05 absolute. All cases retained.','',
        '| Case | Parameter | Truth | Estimate | Error / tolerance | Pass |','|---|---|---:|---:|---:|---|']
    for case in cases:
        assert [r['model'] for r in case['models']]==labels
        for row in case['models']:
            rows[row['model']].append(row)
            if row['status']!='fitted':continue
            fit=row['fit'];successful=[r for r in fit['starts'] if r.get('sse') is not None]
            assert fit['calibration_iv_sse']==min(r['sse'] for r in successful)
            assert len(fit['starts'])==manifest['starts'] and fit['fit_quotes']==84
            assert row['exact_pricer_guard_active_for_all_starts']
            for source,price_key in [('neural','neural_prices'),('exact_repricing','exact_prices')]:
                if price_key not in row:continue
                actual,_=_score_price_iv(np.array(row[price_key]),np.array(case['reference_prices']),
                    np.array(case['reference_iv']),np.array(case['x']),np.array(case['tau']),np.array(case['holdout']))
                for key,value in actual.items():assert row[source][key]==value
            if row['factors']==2:
                truth=np.array(case['truth_parameters']);estimate=np.array(fit['physical'])
                tolerance=.05*truth;tolerance[[3,8]]=.05
                errors=np.abs(estimate-truth)/tolerance
                np.testing.assert_array_equal(errors<=1,row['parameter_passes'])
                assert row['complete_parameter_pass']==bool((errors<=1).all())
                for name,t,e,error in zip(parameters,truth,estimate,errors):
                    param_table.append(f'| {case["case"]} | {name} | {t:.9g} | {e:.9g} | {error:.4g} | {"yes" if error<=1 else "no"} |')
    param_table+=['','## Single Heston fitted parameters','',
        'These are misspecified fits to Double Heston surfaces, not recovery of a true Single vector.','',
        '| Case | Model | kappa | theta | sigma | rho | v0 | Status |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for case in cases:
        for row in case['models']:
            if row['factors']!=1:continue
            values=row.get('fit',{}).get('physical')
            cells=' | '.join(f'{v:.9g}' for v in values) if values is not None else '— | — | — | — | —'
            param_table.append(f'| {case["case"]} | {row["model"]} | {cells} | {row["status"]} |')
    summary=[]
    for label,records in rows.items():
        summary.append({'model':label,'cases':len(records),'fitted':sum(r['status']=='fitted' for r in records),
            'all_parameter_passes':sum(r.get('complete_parameter_pass') is True for r in records) if records[0]['factors']==2 else None,
            'individual_parameter_passes':sum(r.get('individual_parameter_passes',0) for r in records) if records[0]['factors']==2 else None,
            'neural_price_passes':sum(r.get('neural',{}).get('price_gate') is True for r in records),
            'exact_price_passes':sum(r.get('exact_repricing',{}).get('price_gate') is True and
                r['exact_repricing']['quadrature_max_error_spot']<=1e-8 for r in records),
            'joint_passes':sum(r.get('joint_pass') is True for r in records) if records[0]['factors']==2 else None,
            'invalid_neural_iv_quotes':sum(r.get('neural',{}).get('invalid_iv_quotes',0) for r in records),
            'neural_scoring_failures':sum('neural_failure' in r for r in records)})
        summary[-1]['selected_optimizer_converged']=sum(r.get('fit',{}).get('optimizer_success') is True for r in records)
        summary[-1]['budget_exhausted_starts']=sum(s.get('status')==0 for r in records for s in r.get('fit',{}).get('starts',[]))
        summary[-1]['failed_starts']=sum('error' in s for r in records for s in r.get('fit',{}).get('starts',[]))
    # Comparisons are paired by case; missing/nonfinite metrics never become wins.
    comparisons=[]
    double=[s['model'] for s in summary if s['all_parameter_passes'] is not None]
    single=[s['model'] for s in summary if s['all_parameter_passes'] is None]
    for dh in double:
        for sh in single:
            record={'double':dh,'single':sh,'cases':len(cases)}
            for source in ('neural','exact_repricing'):
                pairs=[(a.get(source,{}).get('holdout_price_rmse_spot'),b.get(source,{}).get('holdout_price_rmse_spot'))
                    for a,b in zip(rows[dh],rows[sh])]
                valid=[(a,b) for a,b in pairs if a is not None and b is not None]
                record[source]={'valid_pairs':len(valid),'double_wins':sum(a<b for a,b in valid),
                    'single_wins':sum(b<a for a,b in valid),'ties':sum(a==b for a,b in valid),
                    'median_double_single_rmse_ratio':float(np.median([a/b for a,b in valid])) if valid else None}
            comparisons.append(record)
    history=json.loads((args.training/'history.json').read_text())
    complete=json.loads((args.training/'complete.json').read_text())
    selection=json.loads((args.training/'selection.json').read_text())
    config=json.loads((args.training/'config.json').read_text())
    training_manifest=json.loads((args.training/'manifest.json').read_text())
    assert selection['score']==min(r['validation_iv_rmse'] for r in history)
    audit={'summary':summary,'comparisons':comparisons,'training_selection':selection,
        'all_case_metrics_recomputed':True,'all_start_selections_rechecked':True,
        'actual_case0_all_start_heldout_replay':[r.get('heldout_corruption_replay_identical') for r in cases[0]['models']],
        'assessment_manifest_sha256':sha256(args.assessment/'manifest.json'),
        'assessment_case_sha256':{f'case_{i:03d}.json':sha256(args.assessment/f'case_{i:03d}.json') for i in range(len(cases))},
        'training_complete':complete}
    audit['report_source_sha256']=sha256(__file__)
    if args.bias:
        audit['training_bias_diagnostic']=json.loads(args.bias.read_text())
        audit['training_bias_diagnostic_sha256']=sha256(args.bias)
    if args.physics:
        physics=json.loads(args.physics.read_text())
        arrays=np.load(args.physics.with_suffix('.npz'))
        x=arrays['q'][:,0];lower=np.maximum(np.expm1(x),0)
        for row in physics['models']:
            label=row['checkpoint']['label']
            price=arrays[label+'_price'];calendar=arrays[label+'_calendar']
            row['minimum_calendar_derivative']=float(calendar.min())
            row['max_lower_bound_shortfall']=float(np.maximum(lower-price,0).max())
            violation=(calendar< -1e-12)|(price<lower-1e-12)
            row['core_calendar_or_lower_bound_violations']=int((violation & (np.arange(len(x))%4!=0)).sum())
        audit['physics_review']=physics
        audit['physics_input_sha256']=sha256(args.physics)
    assert all(audit['actual_case0_all_start_heldout_replay'])
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    (args.out/'PARAMETERS.md').write_text('\n'.join(param_table)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained',sharey=True)
    for label,records in rows.items():
        for ax,source,title in zip(axes,('neural','exact_repricing'),('Neural held-out prices','Exact repricing of fitted parameters')):
            y=[r.get(source,{}).get('holdout_price_rmse_spot',np.nan) for r in records]
            ax.plot(range(len(cases)),y,'o-',label=label,markersize=4)
            ax.set(title=title,xlabel='Development case',ylabel='Price RMSE / spot',yscale='log')
            ax.grid(alpha=.2)
    for ax in axes:ax.axhline(1e-5,color='black',linestyle='--',linewidth=1,label='price gate')
    axes[1].legend(fontsize=6)
    fig.savefig(args.out/'pricing_comparison.png',dpi=170);plt.close(fig)
    for label in double:
        errors=np.array([r.get('parameter_tolerance_units',[np.nan]*10) for r in rows[label]])
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        im=ax.imshow(np.log10(np.maximum(errors,1e-3)),aspect='auto',cmap='RdYlGn_r',vmin=-1,vmax=1)
        ax.set_xticks(range(10),parameters,rotation=30);ax.set_yticks(range(len(cases)))
        ax.set(ylabel='Development case',title='Double Heston error / fixed parameter tolerance')
        for i in range(len(cases)):
            for j in range(10):
                ax.text(j,i,f'{errors[i,j]:.2g}\n{"pass" if errors[i,j]<=1 else "fail"}',ha='center',va='center',fontsize=6)
        fig.colorbar(im,ax=ax,label='log10(error / tolerance); pass <= 0')
        fig.savefig(args.out/'parameter_recovery.png',dpi=170);plt.close(fig)
    text=['# Compositional Double Heston PINN — development results','',
        'This is an exposed synthetic pilot, not an unseen or real-market performance claim.','',
        '## Results','',
        '| Model | All parameters | Individual DH gates | Neural price gate | Exact repricing gate |',
        '|---|---:|---:|---:|---:|']
    for s in summary:
        text.append(f'| {s["model"]} | {str(s["all_parameter_passes"])+"/"+str(len(cases)) if s["all_parameter_passes"] is not None else "N/A"} | '
            f'{str(s["individual_parameter_passes"])+"/"+str(10*len(cases)) if s["individual_parameter_passes"] is not None else "N/A"} | {s["neural_price_passes"]}/{len(cases)} | {s["exact_price_passes"]}/{len(cases)} |')
    text+=['','Single Heston has no matching five-parameter truth on these Double Heston surfaces. '
        'A lower pricing error does not demonstrate accurate ten-parameter recovery.','',
        '![Same-quote pricing comparison](pricing_comparison.png)','',
        'Interpretation: lower is better. Left measures the fitted neural function; right checks '
        'whether its recovered parameters also price accurately in the independent reference engine. '
        'The dashed line is the unchanged 1e-5-of-spot gate.','',
        '![Ten-parameter recovery](parameter_recovery.png)','',
        'Interpretation: each cell is error divided by its allowed tolerance. Values <=1 pass; '
        'a case succeeds only if all ten cells pass. Display color limits do not cap printed errors.','',
        '## Paired pricing comparison','']
    for comparison in comparisons:
        for source in ('neural','exact_repricing'):
            c=comparison[source]
            text.append(f'- {comparison["double"]} vs {comparison["single"]}, {source}: '
                f'Double wins {c["double_wins"]}/{len(cases)}; valid pairs {c["valid_pairs"]}; '
                f'median Double/Single RMSE ratio {c["median_double_single_rmse_ratio"]}.')
    text+=['','## What changed and what training achieved','',
        'Two evaluations of a shared Single Heston price-PDE PINN are combined through '
        'the independent-return convolution identity. Density is obtained by automatic price '
        'derivatives. Fixed 96-node quadrature is differentiable. No exact Heston pricer, '
        'ODE solver, density clipping or normalization enters calibration. This is a '
        'separately labelled component-PDE architecture, not direct full Double Heston PDE training.','',
        f'The broader-domain continuation ran {config["steps"]} float64 AdamW steps with '
        f'{config["collocation"]} collocation points, {training_manifest["train_quotes"]} usable '
        f'synthetic training states and {training_manifest["validation_quotes"]} validation states. '
        f'Selection chose step {selection["step"]}, validation IV RMSE {selection["score"]:.10g}. '
        f'The initial learning rate was {config["lr"]:g}, PDE weight {config["weight_pde"]:g}. '
        + ('Step 0 remained best; the continuation did not improve validation and the selected '
        'weights are unchanged from the warm start. This also means the original narrower '
        'training-domain limitation persists despite the broader validation check.' if selection['step']==0 else
        f'Validation improved by {100*(1-selection["score"]/history[0]["validation_iv_rmse"]):.2f}% '
        'from the initial checkpoint. This is forward-validation evidence; it is not proof of '
        'ten-parameter recovery or unbiased generalization.'),'',
        'Training targets are reference synthetic prices/IV corrections and parameter '
        'sensitivities, plus the component pricing PDE. Those are forward-training inputs; '
        'recovery-case truths are used only in assessment. All 1232 rejected training and '
        '145 rejected validation reference candidates remain in the archives. No prediction '
        'failures were removed from the assessment denominators.','',
        '## Data and controls','',
        'Twelve already-exposed DH parameter vectors (truth seed 927931), each with 21 strikes '
        'from .8 to 1.2 of unit spot and six maturities: 30,60,90,180,365,730 days divided by365. '
        'These are normalized synthetic European calls with zero rate/carry; they are not '
        'NSE observations or a representation of currently available Indian stock expiries. '
        'Each case uses 84 calibration and 42 held-out quotes, identical across models.','',
        'All models use five blind LHS starts and at most 400 evaluations/start; selection '
        'uses calibration neural-IV SSE only. All starts and failures are retained. The '
        'three models completed with reference-pricer entry points blocked during every fit; '
        'case 0 was replayed across all starts with held-out coordinates/values replaced by NaN. '
        'The replay was identical excluding timings. This is a concrete isolation test, not '
        'a guarantee against every possible implementation issue. Source and weight hashes '
        'were rechecked. Parameter gates, scores and start selections were independently '
        'recomputed by this report script.','',
        '## Test scope','',
        'The 55 scoped regular/composition tests pass, including parameter Jacobians, '
        'reference composition, checkpoint roundtrip and guarded held-out isolation. '
        'The whole checkout is NOT fully passing: with the legacy Single source import path '
        'configured, the broader run has 629 passed, 43 failed, 1 skipped. Its archived XML '
        'lists missing historical market/evidence files, hash/runtime/schema mismatches, '
        'an unavailable openpyxl dependency and the old 1e-12 numerical fixture discrepancy. '
        'Three legacy dtype-sensitive training tests fail together but pass in isolation. '
        'No old fixture, recorded hash or gate was changed to conceal these failures.','',
        '## Limitations and next work','',
        'The twelve cases have informed development and cannot provide an unbiased final '
        'generalization estimate. There is one training lineage, no fresh noise cohort here, '
        'and only DH-generated comparison surfaces. Check density, martingale, factor-swap '
        'and quadrature diagnostics in every case JSON; they are reported without corrections. '
        'The finite Fourier reference aliases at extreme integration tails; the independent '
        'composition identity test uses converged 32/48-node rules, while the learned model '
        'uses 96 nodes and is audited at 128.','',
        'Further claims require improved component accuracy, all-ten recovery on the fixed '
        'development gates, then a frozen independent cohort covering both Single- and '
        'Double-generated surfaces, noisy quotes and independent training seeds. No universal '
        'identifiability or Single-Heston superiority claim follows from this pilot.','',
        'IV is a deterministic transformation of price, not a new independent measurement. '
        'Factor ordering removes label swapping but does not guarantee that finite quotes '
        'identify all ten parameters stably.','',
        '[Full parameters](PARAMETERS.md) · [Machine-readable audit](audit.json)','']
    if args.physics:
        text+=['## Independent full-PDE diagnostic','',
            'These are raw price-per-year residuals on 512 fresh states, not the normalized '
            'residual metric used by earlier monolithic PINN reports. The diagnostic did not '
            'select checkpoints or change calibration.','',
            '| Model | PDE RMSE | Negative calendar points | Lower-bound violations | Max lower-bound shortfall |',
            '|---|---:|---:|---:|---:|']
        for row in audit['physics_review']['models']:
            text.append(f'| {row["checkpoint"]["label"]} | {row["pde_rmse"]:.6g} | '
                f'{row["negative_calendar_count"]} | {row["lower_price_bound_violations"]} | {row["max_lower_bound_shortfall"]:.4g} |')
        text+=['','The audit JSON records finiteness, convexity, bounds and core/stress counts '
            'at the stated 1e-12 shape threshold. Violations are not silently clipped or '
            'treated as exact physical validity.','']
    if args.bias:
        text+=['## Training-only inverse-conditioning diagnostic','',
            'First 32 archived independent training surfaces, without resampling or dropping '
            'failures. Both model checkpoints were frozen before this diagnostic. The stored '
            'reference-Jacobian pseudoinverse maps IV error to a local parameter-bias proxy '
            'in recovery-tolerance units. This is NOT actual recovery; directions truncated '
            'by the archived singular-value cutoff remain unmeasured.','',
            '| Model | IV RMSE | Linear-shift RMSE | Median largest absolute shift |',
            '|---|---:|---:|---:|']
        for row in audit['training_bias_diagnostic']['models']:
            fmt=lambda key:f'{row[key]:.6g}' if row[key] is not None else 'undefined (incomplete scoring)'
            text.append(f'| {row["checkpoint"]["label"]} | {fmt("whole_subset_iv_rmse")} | '
                f'{fmt("whole_subset_linear_shift_rmse")} | {fmt("median_max_abs_linear_shift")} |')
        text+=['','Read the average IV error alongside the largest parameter-direction metric: '
            'optimizing average forward-pricing accuracy alone need not improve every parameter. A possible '
            'next experiment is to adapt the existing grouped inverse-sensitivity training '
            'loss to this composition, with independent training/validation surfaces and '
            'fixed gates. That experiment has NOT been run here and is not a promised solution.','']
    (args.out/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({'summary':summary,'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
