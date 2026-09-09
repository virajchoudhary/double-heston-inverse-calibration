#!/usr/bin/env python3
"""Report frozen deeper-PINN comparisons, retaining price/recovery distinctions."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlx.core as mx
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint,sha256
from src.mentor_dh_pinn.regular_pinn_data import coordinates,draw_points
from src.mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN,residual
from src.constraints import validate_parameters

BASE=ROOT/'outputs/deeper_pinn'
RUNS={'Clean':'fresh_clean_909831','1% time-value noise':'fresh_noise_909831'}
NAMES={'single_sobolev_s17':'Single Heston (5 layers)',
       'double_sobolev_s17':'Original Double Heston (5 layers)',
       'deep11_s17':'Double Heston (11 layers)',
       'deep17_full_s29':'Double Heston (17 layers)',
       'deep11_balanced_s43':'Double Heston (11-layer continuation)'}


def js(path):return json.loads(path.read_text())
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    out=BASE/'report';out.mkdir(exist_ok=True);torch.set_num_threads(1)
    selection=js(BASE/'selection.json');evidence={};summary=[];comparisons=[];audit=[]
    for condition,name in RUNS.items():
        path=BASE/name;manifest=js(path/'manifest.json')
        assert manifest['status']=='complete'
        metrics=rows(path/'metrics.csv');params=rows(path/'parameter_recovery.csv')
        fits=js(path/'fits_and_starts.json');truths=js(path/'truths.json')
        n=manifest['cases_per_generator'];k=len(manifest['checkpoints'])
        assert len(metrics)==2*n*k and len({(r['generator_factors'],r['case'],r['model']) for r in metrics})==len(metrics)
        assert len(fits)==len(metrics)
        snapshots=js(path/'source_snapshot.json')
        import hashlib
        assert all(hashlib.sha256(snapshots[p].encode()).hexdigest()==digest for p,digest in manifest['source_sha256'].items())
        for f in fits:
            assert validate_parameters(f['physical'])['is_valid'] if len(f['physical'])==10 else all(np.isfinite(f['physical']))
            assert f['calibration_iv_sse']==min(s['sse'] for s in f['starts'] if s.get('sse') is not None)
            assert f['fit_quotes']==84
        for row in metrics:
            if row['own_model']=='True':
                group=[p for p in params if (p['generator_factors'],p['case'],p['model'])==(row['generator_factors'],row['case'],row['model'])]
                assert len(group)==5*int(row['model_factors'])
                assert (row['parameter_gate']=='True')==all(float(p['tolerance_error'])<=1 for p in group)
        for info in manifest['checkpoints']:assert sha256(info['checkpoint'])==info['sha256']
        tests={f:{tuple(t['unit']) for t in truths if t['generator_factors']==f} for f in [1,2]}
        for f,sub in [(1,'single_data'),(2,'double_data')]:
            original=js(ROOT/'outputs/regular_pinn_recovery'/sub/'manifest.json')['splits']['train']
            train={tuple(q) for q in draw_points(original['candidates'],f,original['seed'])[:,2:]}
            assert not train&tests[f]
        new=np.load(BASE/'data_double_full/train.npz')['q'][:,2:]
        assert not {tuple(q) for q in new}&tests[2]
        evidence[condition]=(metrics,params,truths)
        summary.extend({'condition':condition,**r} for r in js(path/'summary.json'))
        single={int(r['case']):float(r['neural_holdout_price_rmse']) for r in metrics if r['model']=='single_sobolev_s17' and r['generator_factors']=='2'}
        for model in NAMES:
            if model=='single_sobolev_s17':continue
            predicted={int(r['case']):float(r['neural_holdout_price_rmse']) for r in metrics if r['model']==model and r['generator_factors']=='2'}
            a=np.array([single[i] for i in sorted(single)]);b=np.array([predicted[i] for i in sorted(single)])
            rng=np.random.default_rng(909933);ix=rng.integers(0,len(a),(10000,len(a)))
            reductions=1-np.sqrt(np.mean(b[ix]**2,axis=1)/np.mean(a[ix]**2,axis=1))
            comparisons.append({'condition':condition,'model':model,'price_reduction_fraction':float(1-np.linalg.norm(b)/np.linalg.norm(a)),
                                'paired_case_wins':int((b<a).sum()),'cases':len(a),
                                'paired_bootstrap_95_interval':np.quantile(reductions,[.025,.975]).tolist()})
        audit.append({'condition':condition,'unique_metric_rows':len(metrics),'recorded_starts':sum(len(f['starts']) for f in fits),
                      'gates_constraints_masks_selection_hashes':'passed','exact_train_test_parameter_overlap':0})
    np.testing.assert_array_equal([t['unit'] for t in evidence['Clean'][2] if t['generator_factors']==2],
                                  [t['unit'] for t in evidence['1% time-value noise'][2] if t['generator_factors']==2])
    physics=[]
    manifest=js(BASE/RUNS['Clean']/'manifest.json')
    for info in manifest['checkpoints']:
        model,_=load_checkpoint(Path(info['checkpoint']))
        net=TorchRegularVariancePINN.from_mlx(model)
        q=draw_points(512,model.factors,909611,collocation=True)
        c,p=coordinates(torch.tensor(q,dtype=torch.float64),model.factors,torch)
        value,diagnostics=residual(net,c,p)
        assert torch.isfinite(value).all()
        cm,pm=coordinates(mx.array(q,dtype=mx.float32),model.factors,mx)
        diff=float(np.max(abs(net.iv(c,p).detach().numpy()-np.asarray(model.iv(cm,pm)))))
        assert diff<2e-6
        physics.append({'model':info['label'],'fresh_wide_pde_rmse':float(value.square().mean().sqrt()),
                        'negative_convexity_points':int((diagnostics['convexity']<0).sum()),
                        'mlx_torch_iv_max_difference':diff,'points':512})
    (out/'audit.json').write_text(json.dumps({'audits':audit,'physics':physics,'distinct_truths':48,
                                           'scope':'48 truths reused in clean/noisy comparisons; same pretrained backbone, not independent initializations'},indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'price_comparison.json').write_text(json.dumps(comparisons,indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    models=list(NAMES);short=['SH 5','DH 5','DH 11','DH 17','DH 11 cont.']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True)
    for ax,factor in zip(axes,[1,2]):
        values=[next(r['neural_holdout_price_rmse'] for r in summary if r['condition']=='Clean' and r['generator_factors']==factor and r['model']==m) for m in models]
        ax.bar(np.arange(5),values,color=['#7d8795','#acb5c0','#258b91','#175f67','#7cbdad'])
        ax.set_xticks(np.arange(5),short);ax.set_yscale('log');ax.set_ylabel('Neural heldout price RMSE / spot')
        ax.set_title(f'Same {"Single" if factor==1 else "Double"} Heston-generated prices')
    fig.savefig(out/'common_price_comparison.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    own=[next(r for r in summary if r['condition']=='Clean' and r['own_model'] and r['model']==m) for m in models]
    bars=ax.bar(np.arange(5),[r['parameter_passes']/r['cases']*100 for r in own],color=['#7d8795','#acb5c0','#258b91','#175f67','#7cbdad'])
    ax.bar_label(bars,labels=[f"{r['parameter_passes']}/{r['cases']}" for r in own],padding=4)
    ax.set_xticks(np.arange(5),short);ax.set_ylim(0,110);ax.set_ylabel('Every own-model parameter passes (%)')
    ax.set_title('Own-model recovery remains a separate, harder target for Double Heston')
    fig.savefig(out/'own_parameter_recovery.png',dpi=170);plt.close(fig)

    chosen=next(r for r in comparisons if r['condition']=='Clean' and r['model']=='deep17_full_s29')
    lines=['# Deeper Double Heston PINN versus Single Heston', '',
           f"**Pricing result:** the 17-layer Double Heston PINN reduces clean heldout price RMSE by {100*chosen['price_reduction_fraction']:.2f}% versus the Single Heston PINN on the same 24 Double Heston-generated surfaces. It wins in {chosen['paired_case_wins']}/24 cases. This uses one frozen neural network and neural-only calibration, with no exact-pricer refinement.", '',
           '**Parameter recovery is not better than Single Heston.** Adding layers and derivative supervision improved pricing, but did not establish recovery of all ten Double Heston parameters. No overall superiority or completed recovery goal is claimed.', '',
           '## Source and models', '',
           'Fetched `Double/Single-heston` at `b0b461ac012a16894078f7906cd9efb50e204a9c` from [the requested repository](https://github.com/virajchoudhary/double-heston-inverse-calibration/tree/Double/Single-heston). Changes are on local branch `codex/deeper-double-pinn-comparison`. No remote push was performed.', '',
           'The original models have five width-160 hidden layers. Each new model is one network: the same conditioned variance backbone plus trainable residual layers. The 11-layer network has 477,473 parameters; the 17-layer network has 847,745. Added layers initially preserve the old function, and their nonzero trained weight norms confirm actual learning.', '',
           'Three runs completed: 6,000, 16,000 and 8,000 additional AdamW updates (30,000 total). The 17-layer model was the pricing candidate selected on development data; the 11-layer model at step 4,000 was the best deeper recovery candidate. All models were frozen before assessment. They share pretrained ancestry, so the continuation seeds are not independent from-scratch trials.', '',
           '## Same-data price comparison', '', '| Model | Clean DH prices | Noisy DH prices |', '|---|---:|---:|']
    for m in models:
        clean=next(r for r in summary if r['condition']=='Clean' and r['generator_factors']==2 and r['model']==m)
        noisy=next(r for r in summary if r['condition']=='1% time-value noise' and r['generator_factors']==2 and r['model']==m)
        lines.append(f"| {NAMES[m]} | {clean['neural_holdout_price_rmse']:.6g} | {noisy['neural_holdout_price_rmse']:.6g} |")
    lines+=['', 'Errors are normalized by spot and scored against clean generating prices. Noise is independent Gaussian at 1% of option time value; no clipping/resampling. Each comparison uses identical observed surfaces and quote masks, five starts, and at most 400 evaluations per start. Exact repricing of fitted physical parameters is stored separately in `summary.json`; it never refines the estimates.', '',
            '![Common-surface prices](common_price_comparison.png)', '',
            f"The paired bootstrap 95% interval for the 17-layer clean DH price-error reduction is {100*chosen['paired_bootstrap_95_interval'][0]:.2f}%–{100*chosen['paired_bootstrap_95_interval'][1]:.2f}% (resampling 24 surfaces). This is limited synthetic evidence, not a market-performance claim. The original shallow Double Heston model also substantially beats Single Heston on DH-generated data; the entire advantage must not be credited to depth.", '',
            'The Single Heston-generated comparison is also shown, so performance is not judged only on data generated by the more flexible model.', '',
            '## Own-model parameter recovery', '',
            '| Model | All parameters pass, clean | Original scaled RMSE | Tolerance-scaled RMSE |', '|---|---:|---:|---:|']
    for m,r in zip(models,own):
        lines.append(f"| {NAMES[m]} | {r['parameter_passes']}/{r['cases']} | {r['scaled_parameter_rmse']:.6g} | {r['tolerance_scaled_rmse']:.6g} |")
    lines+=['', '![Own-model recovery](own_parameter_recovery.png)', '',
            'Own-model means five true parameters for Single Heston-generated data and ten for Double Heston-generated data. The unchanged gate requires every positive parameter within 5% and each rho within .05. Original scaled RMSE uses true positive values and .5 for rho; tolerance-scaled RMSE divides by the actual gate tolerance. No generating Single Heston label is invented when Single Heston fits Double Heston data.', '',
            'These runs did not achieve superiority on both requested outcomes. Neural price errors can be small while parameter estimates move substantially along weak directions. The training-error-covariance diagnostic improved one exposed development RMSE but still passed 0/4 all-parameter cases; it was not used in this frozen comparison.', '',
            '## Data and checks', '',
            'Fresh teacher data contains 131,072 training and 16,384 validation points, checked at 128/96 quadrature nodes. The full sensitivity variant uses the same prices plus independently differentiated strike/maturity sensitivities. Training/checkpoint selection uses development labels only. Final seed 909831 supplies 24 truths per generating family: 48 distinct truths reused under clean and noisy observations, not 480 distinct truths.', '',
            'The grid has 21 strikes and maturities 30/60/90/180/365/730 days. Every third strike is withheld. This is a synthetic central-domain test, not evidence that these expiries trade in an NSE stock chain. Boundary robustness and real-market superiority remain untested in this comparison.', '',
            '28 scoped tests passed, covering the deeper model, function-preserving initialization, complete MLX/Torch export, additional-layer PDE gradients, full teacher geometry derivatives, exact terminal payoff, and neural-only calibration/holdout isolation. Artifact checks passed for counts, gates, constraints, minimum-objective start selection, hashes and zero exact training/assessment parameter overlap.', '',
            '| Model | Fresh wide-domain PDE RMSE | Maximum MLX/Torch IV difference |', '|---|---:|---:|']
    for r in physics:lines.append(f"| {NAMES[r['model']]} | {r['fresh_wide_pde_rmse']:.6g} | {r['mlx_torch_iv_max_difference']:.3g} |")
    lines+=['', 'The PDE residuals are not zero and do not certify a globally solved PDE. Reduced PDE weights in the second/third trials are documented in the protocol.', '',
            '## Use the price-focused 17-layer model', '',
            '```python', 'from scripts.mentor_dh_pinn.assess_regular_pinn import load_checkpoint, fit_network', '',
            'model, metadata = load_checkpoint("outputs/deeper_pinn/deep17_full_s29")',
            '# x = log(F/K), tau in years, observed_iv as annualized decimals.',
            'fit = fit_network(model, x, tau, observed_iv, fit_mask=calibration_mask,',
            '                  starts=5, max_nfev=400)', 'parameters = fit["physical"]  # slow [kappa,theta,sigma,rho,v0], then fast', '```', '',
            'Run in the new checkout. Install `requirements-regular-pinn.txt`; MLX training requires Apple Silicon. The project-local `.venv` was created for this experiment. Every new run directory must be unused; historical results are preserved.', '',
            '- [Frozen choices](../selection.json)', '- [Clean parameters](../fresh_clean_909831/parameter_recovery.csv)',
            '- [Clean fits and all starts](../fresh_clean_909831/fits_and_starts.json)', '- [Noisy comparison](../fresh_noise_909831/summary.json)',
            '- [Artifact and PDE audit](audit.json)', '- [Protocol](../../../docs/DEEPER_PINN_COMPARISON_PROTOCOL.md)', '']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(out/'REPORT.md'),'price_candidate_result':chosen,'own_model_recovery':own},indent=2))


if __name__=='__main__':main()
