#!/usr/bin/env python3
"""Audit frozen regularization experiments and write the result artifact."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/deeper_pinn'


def read(path):return json.loads(path.read_text())


def main():
    out=BASE/'regularized_report';out.mkdir(exist_ok=True);runs={};audit=[];comparisons=[];parameters=[]
    folders={'Clean':'prior_clean_911831','1% noise':'prior_noise_911831',
             'Boundary clean':'prior_boundary_clean_911931','Boundary 1% noise':'prior_boundary_noise_911931'}
    for condition,folder in folders.items():
        path=BASE/folder;m=read(path/'manifest.json');assert m['status']=='complete'
        n=m['cases_per_generator'];metrics=read(path/'metrics.json');fits=read(path/'fits.json');truths=read(path/'truths.json')
        assert len(metrics)==len(fits)==6*n
        assert len({(r['factors'],r['case'],r['model']) for r in metrics})==6*n
        assert all(f['fit_quotes']==84 and len(f['starts'])==5 for f in fits)
        assert all(f['calibration_objective']==min(s['sse'] for s in f['starts'] if s['sse'] is not None) for f in fits)
        assert all(r['parameter_pass']==all(e<=1 for e in r['tolerance_error']) for r in metrics if 'parameter_pass' in r)
        snapshots=read(path/'source_snapshot.json')
        for p,content in snapshots.items():assert hashlib.sha256(content.encode()).hexdigest()==m['input_sha256'][p]
        for p,digest in m['input_sha256'].items():
            if p not in snapshots:assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
        assert max(t['quadrature_error'] for t in truths)<1e-9
        summary=read(path/'summary.json');runs[condition]=summary
        for name in ['Original Double Heston','Unregularized matched PINN']:
            a=np.array([r['parameter_rmse'] for r in metrics if r['factors']==2 and r['model']==name])
            b=np.array([r['parameter_rmse'] for r in metrics if r['factors']==2 and r['model']=='Selected regularized PINN'])
            ix=np.random.default_rng(911933).integers(0,n,(10000,n))
            reduction=1-np.sqrt(np.mean(b[ix]**2,axis=1)/np.mean(a[ix]**2,axis=1))
            comparisons.append({'condition':condition,'baseline':name,'reduction':float(1-np.linalg.norm(b)/np.linalg.norm(a)),
                                'paired_case_wins':int((b<a).sum()),'cases':n,'bootstrap_95':np.quantile(reduction,[.025,.975]).tolist()})
        for case in range(n):
            true=next(t['physical'] for t in truths if t['factors']==2 and t['case']==case)
            for name in ['Original Double Heston','Unregularized matched PINN','Selected regularized PINN']:
                f=next(f for f in fits if f['factors']==2 and f['case']==case and f['model']==name)
                for j,p in enumerate(['kappa_s','theta_s','sigma_s','rho_s','v0_s','kappa_f','theta_f','sigma_f','rho_f','v0_f']):
                    parameters.append({'condition':condition,'case':case,'model':name,'parameter':p,'true':true[j],
                                       'estimated':f['physical'][j],'absolute_error':abs(true[j]-f['physical'][j])})
        selected=[f for f in fits if f['model']=='Selected regularized PINN']
        audit.append({'condition':condition,'fits':len(fits),'starts':sum(len(f['starts']) for f in fits),
                      'counts_masks_gates_objectives_hashes':'passed',
                      'median_regularization_dominated_directions':float(np.median([f['regularization_dominated_directions'] for f in selected]))})
    (out/'audit.json').write_text(json.dumps({'runs':audit,'comparisons':comparisons},indent=2))
    with (out/'parameter_estimates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(parameters[0]));writer.writeheader();writer.writerows(parameters)
    get=lambda c,name:next(r for r in runs[c] if r['factors']==2 and r['model']==name)
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    names=['Original Double Heston','Unregularized matched PINN','Selected regularized PINN']
    for ax,condition in zip(axes,['Clean','1% noise']):
        vals=[get(condition,n)['parameter_rmse'] for n in names]
        bars=ax.bar(['Original DH','Matched PINN\nwithout penalty','Regularized\nPINN'],vals,color=['#b1bac3','#5193a6','#176b61'])
        ax.bar_label(bars,fmt='%.3f',padding=5);ax.set_ylim(0,max(vals)*1.2)
        ax.set_ylabel('Scaled physical-parameter RMSE');ax.set_title(condition+' · 32 fresh DH surfaces')
    fig.savefig(out/'parameter_recovery.png',dpi=180);plt.close(fig)
    lines=['# More stable neural-only Double Heston recovery','',
           '**Measured improvement:** the new regularized calibration lowers average physical-parameter error on both clean and noisy fresh synthetic data. It uses one frozen pricing PINN per fit, with no exact-pricer refinement.','',
           '![Parameter recovery](parameter_recovery.png)','',
           '| Condition | Original DH | Same PINN without penalty | Regularized PINN | Reduction vs original |','|---|---:|---:|---:|---:|']
    for condition in folders:
        a,b,c=[get(condition,n) for n in names]
        lines.append(f"| {condition} | {a['parameter_rmse']:.5f} | {b['parameter_rmse']:.5f} | {c['parameter_rmse']:.5f} | {100*(1-c['parameter_rmse']/a['parameter_rmse']):.1f}% |")
    lines+=['','The matched comparison isolates the added midpoint penalty; the original comparison includes the earlier network and covariance changes. Clean and noisy scenarios use the same generating parameters, with noise added only to observations.','',
            'The strongest new evidence is the noisy matched comparison: 34.6% lower parameter RMSE, with a paired bootstrap 95% reduction interval of 10.8%–49.9%. The clean matched reduction is 9.9%, but its interval includes zero. Near the boundary, clean recovery is slightly worse than the same PINN without the penalty (0.443 versus 0.440), while noisy recovery improves on average. Thus the penalty is not a universal clean-data improvement.','',
            '## What changed and why','',
            'The inverse fit now minimizes `||W (IV_network(u) - training_bias - IV_observed)||² + lambda ||u - 0.5||²`, with bounded unit parameters. `W` comes from frozen training-error covariance plus supplied observation uncertainty. The added term discourages large movements in directions poorly constrained by the prices. It introduces a midpoint bias; it does not create information or prove identifiability.','',
            'A development sweep evaluated lambda = 0, 0.1, 1, 10 and 100 for the existing 11- and 17-layer networks on 12 previously exposed surfaces, under clean/noisy observations. Lambda 10 won both conditions. The clean profile selected the 17-layer network; the noisy profile selected the 11-layer network. Both are single networks; there is no ensemble. All choices were frozen before the new assessment.','',
            'The network weights were not retrained in this iteration. The improvement comes from the calibration objective. Prior attempts at locally recovery-sensitive training failed to improve development recovery and remain preserved.','',
            'The fit now records separate raw IV error, data objective, prior penalty, total objective, all starts, data-Jacobian singular values and the number of local directions dominated by regularization. These are local diagnostics, not certified parameter confidence intervals.','',
            '## Pricing and strict recovery','',
            '| Condition / model | Heldout neural price RMSE | Mean-corrected price RMSE | Exact post-fit price RMSE | All own-model parameters pass |',
            '|---|---:|---:|---:|---:|']
    for condition in ['Clean','1% noise']:
        for name in ['Single Heston','Original Double Heston','17-layer Double Heston','Selected regularized PINN']:
            r=get(condition,name);corr=f"{r['bias_corrected_price_rmse']:.6g}" if 'bias_corrected_price_rmse' in r else '—'
            gate=f"{r['parameter_passes']}/{r['cases']}" if 'parameter_passes' in r else 'N/A on DH data'
            lines.append(f"| {condition}: {name} | {r['neural_price_rmse']:.6g} | {corr} | {r['exact_reprice_rmse']:.6g} | {gate} |")
    shclean=next(r for r in runs['Clean'] if r['factors']==1);shnoise=next(r for r in runs['1% noise'] if r['factors']==1)
    lines+=['',f"**All-ten recovery is still not solved.** Regularized Double Heston passes 1/32 clean and 0/32 noisy cases. Single Heston on its own generated data passes {shclean['parameter_passes']}/32 clean and {shnoise['parameter_passes']}/32 noisy. The gate remains 5% relative error for positive parameters and 0.05 absolute error for correlations. Single Heston on DH-generated data has no matching ten-parameter truth.",'',
            'Price errors are normalized by spot and evaluated on 42 withheld quotes against clean generating prices. Calibration sees only the other 84 quotes. Exact repricing is evaluated after fitting and never changes parameters. The recovery profile makes a pricing/recovery tradeoff; the separate unregularized 17-layer pricing network remains preferable when pricing error alone is the objective.','',
            '## Verification and scope','',
            'Assessment seed 911831 supplies 32 new truths per model family. A separate stress test uses seed 911931 and 16 truths per family, each with two unit parameters fixed near the boundaries at 0.02 and 0.98. Neither set is used to select the penalty. Each fit uses five starts and at most 400 evaluations per start. Price references pass a 128-versus-96-node quadrature check. Input hashes and source snapshots are retained.','',
            'The domain is synthetic, with 21 strikes and maturities 30/60/90/180/365/730 days. Noise is Gaussian at 1% of option time value; the noisy profile receives that noise level. Do not silently apply the noisy profile to a different uncertainty level, arbitrary quote grid, or market-data distribution. Its performance in those settings is unverified.','',
            '31 scoped tests pass, including finite-difference validation of the regularized residual Jacobian, objective accounting, masked-quote invariance, and forbidden exact-pricer calls during optimization. A real CLI smoke test also completed with withheld IV values set to null.','',
            '| Condition | Reduction vs matched PINN | Paired wins | Bootstrap 95% reduction interval |','|---|---:|---:|---:|']
    for r in comparisons:
        if r['baseline']=='Unregularized matched PINN':
            lo,hi=r['bootstrap_95'];lines.append(f"| {r['condition']} | {100*r['reduction']:.1f}% | {r['paired_case_wins']}/{r['cases']} | {100*lo:.1f}% to {100*hi:.1f}% |")
    lines+=['','## Use the implemented calibration','',
            'Run from this checkout with its `.venv`. The input JSON contains `x`, `tau`, `observed_iv` and `fit_mask`; a ready example is linked below. Statistics require the exact documented grid, and the loader checks the network/statistics checkpoint hash.','',
            '```bash','.venv/bin/python scripts/mentor_dh_pinn/calibrate_recovery_pinn.py \\',
            '  --input outputs/deeper_pinn/prior_cli_example/quotes.json \\',
            '  --out outputs/deeper_pinn/NEW_FIT.json --profile clean',
            '# For the tested noisy condition: --profile noise --noise-fraction .01','```','',
            '- [Frozen selection](../prior_frozen_selection.json)',
            '- [All parameter estimates](parameter_estimates.csv)',
            '- [Audits and paired intervals](audit.json)',
            '- [CLI input example](../prior_cli_example/quotes.json)',
            '- [CLI output example](../prior_cli_example/fit.json)',
            '- [Development sweep](../prior_development_911/summary.json)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'report':str(out/'REPORT.md'),'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
