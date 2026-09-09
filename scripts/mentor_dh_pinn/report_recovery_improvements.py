#!/usr/bin/env python3
"""Report completed trials without conflating pricing and parameter recovery."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/deeper_pinn'


def read(p):return json.loads(p.read_text())


def main():
    out=BASE/'recovery_report';out.mkdir(exist_ok=True)
    runs={};audits=[]
    for label,folder in [('Clean','recovery_clean_910831'),('Noisy','recovery_noise_910831')]:
        path=BASE/folder;manifest=read(path/'manifest.json');assert manifest['status']=='complete'
        summary=read(path/'summary.json');metrics=read(path/'metrics.json');fits=read(path/'fits.json')
        assert len(metrics)==144 and len(fits)==144
        assert len({(r['factors'],r['case'],r['model']) for r in metrics})==144
        assert all(f['fit_quotes']==84 and len(f['starts'])==5 for f in fits)
        assert all(f['calibration_objective']==min(s['sse'] for s in f['starts'] if s['sse'] is not None) for f in fits)
        assert all(r['parameter_pass']==all(e<=1 for e in r['tolerance_error']) for r in metrics if 'parameter_pass' in r)
        snapshots=read(path/'source_snapshot.json')
        for p,content in snapshots.items():assert hashlib.sha256(content.encode()).hexdigest()==manifest['input_sha256'][p]
        for p,digest in manifest['input_sha256'].items():
            if p not in snapshots:assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
        assert max(t['quadrature_error'] for t in read(path/'truths.json'))<1e-9
        runs[label]=(summary,metrics)
        audits.append({'condition':label,'fits':len(fits),'starts':sum(len(f['starts']) for f in fits),'hashes_masks_gates_selection':'passed'})
    clean,metrics=runs['Clean'];dh=[r for r in clean if r['factors']==2]
    get=lambda name:next(r for r in dh if r['model']==name)
    original=get('Original Double Heston');deep=get('17-layer Double Heston');gls=get('17-layer Double Heston + GLS');single=get('Single Heston')
    improvement=1-gls['parameter_rmse']/original['parameter_rmse']
    price_improvement=1-deep['neural_price_rmse']/single['neural_price_rmse']
    bootstrap=[]
    for name in ['11-layer Double Heston + GLS','17-layer Double Heston + GLS']:
        a=np.array([r['parameter_rmse'] for r in metrics if r['factors']==2 and r['model']=='Original Double Heston'])
        b=np.array([r['parameter_rmse'] for r in metrics if r['factors']==2 and r['model']==name])
        ix=np.random.default_rng(910933).integers(0,len(a),(10000,len(a)))
        reduction=1-np.sqrt(np.mean(b[ix]**2,axis=1)/np.mean(a[ix]**2,axis=1))
        bootstrap.append({'model':name,'paired_recovery_wins':int((b<a).sum()),'cases':len(a),
                          'recovery_reduction_95_interval':np.quantile(reduction,[.025,.975]).tolist()})
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    names=['Original Double Heston','17-layer Double Heston','17-layer Double Heston + GLS']
    colors=['#99a5b1','#4291a8','#175c72'];short=['Original DH','17-layer DH','17-layer DH + GLS']
    vals=[get(n)['parameter_rmse'] for n in names]
    bars=axs[0].bar(short,vals,color=colors);axs[0].bar_label(bars,fmt='%.3f',padding=4)
    axs[0].set_ylim(0,max(vals)*1.25);axs[0].set_ylabel('Scaled physical-parameter RMSE')
    axs[0].set_title('Parameter error: 24 fresh clean DH surfaces')
    vals=[single['neural_price_rmse'],deep['neural_price_rmse'],gls['bias_corrected_price_rmse']]
    bars=axs[1].bar(['Single Heston','17-layer DH','DH + GLS\n(mean corrected)'],vals,color=colors)
    axs[1].set_yscale('log');axs[1].bar_label(bars,fmt='%.2g',padding=4)
    axs[1].set_ylim(min(vals)*.5,max(vals)*2);axs[1].set_ylabel('Heldout price RMSE / spot')
    axs[1].set_title('Pricing: identical Double Heston observations')
    fig.savefig(out/'results.png',dpi=180);plt.close(fig)
    lines=['# Implemented PINN improvements and measured results','',
           f'The 17-layer model with training-error GLS reduces clean Double Heston parameter RMSE by {100*improvement:.1f}% versus the original five-layer Double Heston PINN. The uncorrected 17-layer model reduces heldout price RMSE by {100*price_improvement:.2f}% versus Single Heston on the same Double Heston-generated observations. These are separate configurations and objectives.','',
           '**Full parameter recovery remains unsolved.** Single Heston recovers its own five parameters in 22/24 clean cases; the 17-layer Double Heston plus GLS recovers all ten in 2/24. The original Double Heston passes 1/24. The unchanged gate is 5% relative error for positive parameters and 0.05 absolute error for correlations.','',
           '![Measured results](results.png)','',
           '## What was implemented','',
           '- One 17-layer residual pricing PINN (847,745 parameters), with parameter and strike/maturity derivative supervision. No ensemble or exact-pricer refinement.','- Calibration can now use frozen training-error mean/covariance, eigenvalue regularization and observation uncertainty. The same whitening transforms residuals and their analytic Jacobian. Statistics are restricted to their exact quote geometry; assessment verifies the checkpoint hash.','- Recovery-sensitive training uses exact training Jacobians to penalize pricing errors that imply large local physical-parameter shifts. Its preconditioner now uses the same 84 quote positions as calibration. This is a local linear proxy, not an actual recovery loss.','- Two additional 4,000-update training trials used weights 0.01 and 0.001. Their fixed four-case development recovery scores did not beat the original 17-layer candidate, so they were not promoted. Both trials and their checkpoints are retained.','',
           '## Independent assessment','',
           'Seed 910831 supplies 24 new truths per generating model, reused for clean/noisy comparisons. Each surface has 126 quotes: 84 calibration and 42 heldout. Every fit receives five identical seeded starts and a 400-evaluation limit per start. Candidate networks, covariance statistics and floors were fixed before assessment. The 11-layer GLS candidate had the better four-case development recovery score; both predeclared GLS candidates are reported. No assessment-based retraining was performed.','',
           '| Condition / model | DH parameter RMSE | All-ten pass | Raw neural price RMSE | Exact post-fit reprice RMSE |','|---|---:|---:|---:|---:|']
    for condition,(summary,_) in runs.items():
        for r in summary:
            if r['factors']!=2:continue
            recovery=f"{r['parameter_rmse']:.5f}" if 'parameter_rmse' in r else 'N/A'
            passes=f"{r['parameter_passes']}/24" if 'parameter_passes' in r else 'N/A'
            lines.append(f"| {condition}: {r['model']} | {recovery} | {passes} | {r['neural_price_rmse']:.6g} | {r['exact_reprice_rmse']:.6g} |")
    lines+=['','Pricing errors are normalized by spot and scored against clean generating prices. Exact repricing happens only after fitting. GLS can trade raw neural price error for better physical parameters; its mean-corrected clean heldout price RMSE is '+f"{gls['bias_corrected_price_rmse']:.6g}."+' Single Heston on Double Heston data has no matching ten-parameter truth, so its parameter score is N/A.','',
            '**Noise limitation:** GLS did not improve noisy parameter recovery. The 17-layer GLS error rose to 0.787 versus 0.619 for the original Double Heston, and every Double Heston configuration passed 0/24 noisy cases. Do not treat the clean result as a robustness result.','',
            '**Sampling uncertainty:** the predeclared 11-layer GLS recovery candidate also reduced clean average parameter error by about 25%; its paired bootstrap 95% reduction interval is 5.5%–49.3%. The 17-layer GLS interval is −16.4%–52.7%, which includes no improvement. The 17-layer result alone does not establish a population-wide recovery advantage.','',
            'Noise is independent Gaussian at 1% of option time value. GLS adds a delta-method IV observation covariance estimated from observed quotes and the specified noise level. There is no clipping or resampling.','',
            '## Why recovery is still difficult','',
            'The tolerance-scaled teacher Jacobians of 1,024 training surfaces have median condition number about 5,238, rising above two million. Small pricing errors can therefore produce large parameter errors. More depth alone did not improve recovery on this fresh sample. Error-aware calibration helps average recovery but does not remove weak identification.','',
            'The covariance model uses 512 separate synthetic training surfaces. Their 128-versus-96-node price discrepancy is at most 2.36e-14. Fresh assessment prices also pass the 1e-9 quadrature check. This remains evidence for central-domain synthetic surfaces on a fixed grid, not market-data validation or a global identifiability guarantee.','',
            '30 scoped tests passed, including extra-layer gradients, MLX/Torch weight parity, teacher derivatives, holdout isolation, whitening algebra and the prohibition on exact pricing inside calibration. All 288 clean/noisy fits and 1,440 starts are retained. See `audit.json` for artifact checks and paired bootstrap intervals.','',
            '## Reproduce','',
            '```bash','.venv/bin/python scripts/mentor_dh_pinn/evaluate_recovery_improvements.py \\',
            '  --spec outputs/deeper_pinn/recovery_comparison_spec.json \\',
            '  --out outputs/deeper_pinn/NEW_UNUSED_DIRECTORY --seed 910831 --cases 24',
            '# Add --noise .01 for the noisy comparison.','```','',
            '- [Frozen candidate specification](../recovery_comparison_spec.json)',
            '- [Clean metrics and parameter errors](../recovery_clean_910831/metrics.json)',
            '- [Clean fits and all starts](../recovery_clean_910831/fits.json)',
            '- [Noisy results](../recovery_noise_910831/summary.json)',
            '- [Earlier depth/derivative comparison](../report/REPORT.md)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'audit.json').write_text(json.dumps({'artifact_checks':audits,'paired_bootstrap':bootstrap,
          'clean_recovery_reduction':improvement,'clean_price_reduction':price_improvement},indent=2))
    print(json.dumps({'report':str(out/'REPORT.md'),'recovery_reduction':improvement,'price_reduction':price_improvement,'bootstrap':bootstrap},indent=2))


if __name__=='__main__':main()
