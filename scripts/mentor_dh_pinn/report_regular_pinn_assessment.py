#!/usr/bin/env python3
"""Audit and report the completed frozen assessment; never fits or trains."""
import argparse,csv,hashlib,json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.regular_pinn_data import decode_unit


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def yes(x):return x=='True'
def unique(data,keys):return len({tuple(r[k] for k in keys) for r in data})==len(data)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def byte_rows(a):
    a=np.ascontiguousarray(a);return a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).ravel()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assessment',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    source=args.assessment.resolve();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='complete'
    metrics=rows(source/'metrics.csv');params=rows(source/'parameter_recovery.csv');obs=rows(source/'observations.csv')
    summary=json.loads((source/'summary.json').read_text());fits=json.loads((source/'fits_and_starts.json').read_text())
    truths=json.loads((source/'synthetic_truth.json').read_text());n=manifest['cases_per_model_family']
    model_rows=[r for r in metrics if r['model']!='black_scholes_on_double_heston']
    checks={'metric_keys_unique':unique(metrics,['model','case','geometry','noise']),
            'parameter_keys_unique':unique(params,['model','case','geometry','noise','factor','parameter']),
            'observation_keys_unique':unique(obs,['case','geometry','noise','quote']),
            'expected_metric_rows':len(metrics)==n*20,'expected_parameter_rows':len(params)==n*120,
            'expected_observation_rows':len(obs)==n*756,'all_model_fits_retained':len(fits)==len(model_rows),
            'five_starts_per_fit':all(len(f['starts'])==5 for f in fits),
            'frozen_hash_recheck_recorded':manifest.get('frozen_checkpoint_and_source_hashes_rechecked') is True,
            'current_checkpoint_hashes_match':all(sha(Path(r['checkpoint']))==r['sha256'] for r in manifest['checkpoints']),
            'current_assessment_sources_match':all(sha(ROOT/p)==h for p,h in manifest['source_sha256'].items())}
    gates={}
    for r in params:
        actual,estimate=float(r['truth']),float(r['estimate'])
        passed=abs(estimate-actual)<=.05 if r['parameter']=='rho' else abs(estimate/actual-1)<=.05
        key=tuple(r[k] for k in ['model','case','geometry','noise'])
        gates.setdefault(key,[]).append(passed)
        assert passed==yes(r['parameter_pass'])
    checks['all_parameter_gates_recomputed']=all(yes(r['parameter_gate'])==all(gates[tuple(r[k] for k in ['model','case','geometry','noise'])]) for r in model_rows)
    valid_physical=True
    for fit in fits:
        p=np.array(fit['physical']).reshape(-1,5)
        valid_physical &= bool(np.isfinite(p).all() and (p[:,[0,1,2,4]]>0).all() and (np.abs(p[:,3])<1).all()
                               and (2*p[:,0]*p[:,1]>p[:,2]**2).all())
        if len(p)==2:valid_physical &= bool(p[0,0]<p[1,0])
        scores=[s['sse'] for s in fit['starts'] if s.get('sse') is not None]
        assert np.isclose(fit['calibration_iv_sse'],min(scores),rtol=1e-12,atol=1e-20)
    checks['fitted_physical_bounds_feller_and_order']=valid_physical
    checks['minimum_calibration_sse_start_selected']=True
    overlap={}
    base=source.parent
    for family,factors in [('single',1),('double',2)]:
        units=np.array([t['unit'] for t in truths if t['factors']==factors])
        for split in ['train','validation']:
            q=np.load(base/f'{family}_data'/f'{split}.npz')['q'][:,2:]
            overlap[f'{family}_{split}']=int(len(np.intersect1d(byte_rows(units),byte_rows(q))))
        if factors==2:
            q=np.load(base/'double_surface_train907721/surfaces.npz')['unit']
            overlap['double_extra_surface_train']=int(len(np.intersect1d(byte_rows(units),byte_rows(q))))
    checks['no_exact_assessment_parameter_overlap']=not any(overlap.values())
    assert all(checks.values()),checks
    audit={'checks':checks,'parameter_overlap_counts':overlap,'unique_truth_vectors':len(truths),
           'model_condition_fits':len(fits),'optimizer_starts':sum(len(f['starts']) for f in fits),
           'optimizer_converged_selected':sum(bool(f['optimizer_success']) for f in fits),
           'input_sha256':{p.name:sha(p) for p in source.iterdir() if p.is_file()},
           'report_script_sha256':sha(Path(__file__))}
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2))
    conditions=[('monthly',0.),('rich',0.),('monthly',.01),('rich',.01)]
    labels=['3 expiries\nclean','6 expiries\nclean','3 expiries\n1% noise','6 expiries\n1% noise']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True,layout='constrained')
    for ax,family in zip(axes,['single','double']):
        for offset,arm,color in [(-.18,'standard','#6b7280'),(.18,'sobolev','#007a87')]:
            values=[next(r['all_parameter_passes'] for r in summary if r['model']==f'{family}_{arm}_s17' and (r['geometry'],r['noise'])==c) for c in conditions]
            bars=ax.bar(np.arange(4)+offset,values,.34,label=arm,color=color)
            ax.bar_label(bars,labels=[f'{v}/{n}' for v in values],padding=3,fontsize=9)
        ax.set_xticks(np.arange(4),labels);ax.set_ylim(0,n+2);ax.set_title(f'{family.title()} Heston: own-model truths')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend()
    axes[0].set_ylabel('Cases passing every parameter tolerance')
    fig.suptitle('Frozen central-domain assessment — not a common-market model ranking')
    fig.savefig(args.out/'parameter_passes.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    models=['single_standard_s17','single_sobolev_s17','double_standard_s17','double_sobolev_s17']
    for i,name in enumerate(models):
        data=[float(r['max_positive_parameter_relative_error'])*100 for r in model_rows if r['model']==name and r['geometry']=='rich' and float(r['noise'])==0]
        ax.scatter(i+np.linspace(-.15,.15,len(data)),data,s=32,alpha=.8)
    ax.axhline(5,color='crimson',ls='--',lw=1,label='5% maximum positive-parameter error target')
    ax.set_xticks(range(4),['Single\nstandard','Single\nSobolev','Double\nstandard','Double\nSobolev'])
    ax.set_yscale('log');ax.set_ylabel('Worst positive-parameter relative error (%)')
    ax.set_title('Clean six-expiry cases: one dot per generating parameter vector')
    ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9)
    fig.savefig(args.out/'individual_recovery_errors.png',dpi=160);plt.close(fig)
    lines=['# Frozen regular-PINN assessment','',
           'Double Heston parameter recovery is **not achieved** in this experiment. Both frozen Double Heston variants failed the all-ten-parameter gate in every assessed condition. Neural price fitting alone must not be presented as parameter recovery.','',
           f'The assessment contains {len(truths)} distinct new synthetic parameter vectors (12 Single and 12 Double), reused across clean/noisy and monthly/rich conditions. There are {len(fits)} neural model-condition fits and {audit["optimizer_starts"]} recorded starting-point attempts, not that many independent truths. All four models use training seed 17.','',
           '## Results','',
           '| Model | Expiries | Noise | All parameters | Neural heldout prices | Exact repricing of fitted parameters | All gates |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['model'].startswith('black_'):continue
        lines.append(f'| {r["model"]} | {r["geometry"]} | {r["noise"]:.0%} | {r["all_parameter_passes"]}/{n} | {r["neural_price_passes"]}/{n} | {r["exact_price_passes"]}/{n} | {r["joint_recovery_passes"]}/{n} |')
    lines += ['', 'Parameter tolerances: each positive parameter within 5% of truth and each rho within 0.05. Price tolerance: heldout RMSE <=1e-5 of spot. Exact repricing is performed only after fitting the frozen neural network; it never refines the parameters. These are separate gates.','',
              '![All-parameter pass counts](parameter_passes.png)','',
              'Interpretation: adding training sensitivities improves Single Heston recovery, especially with six expiries. It does not establish ten-parameter recovery for Double Heston. Short maturities and noise are harder.','',
              '![Individual recovery errors](individual_recovery_errors.png)','',
              'Interpretation: each dot is an actual worst positive-parameter error, without aggregation into density bins. Points above 5% fail that part of the recovery target. Correlations have a separate absolute tolerance; all parameters are retained in the source CSV.','',
              '## Integrity and limits','',
              f'All {len(checks)} automated artifact checks passed, including unique row keys, expected counts, recomputed gates, valid fitted parameters, minimum calibration-SSE start selection, frozen hashes and zero exact parameter overlap with training/validation. Selected optimizers reported convergence in {audit["optimizer_converged_selected"]}/{len(fits)} fits; convergence is not recovery. See [audit.json](audit.json).', '',
              'Unit parameters are sampled in [0.1,0.9], only about 10.7% of the ten-dimensional training unit cube by volume. This is not full-boundary validation. One training initialization and 12 truths per family are insufficient for a broad reliability claim. No real NSE quotes were used or altered. The additional Black–Scholes rows are an analytic one-volatility pricing baseline, not a trained Black–Scholes PINN and not a DH parameter-recovery test. Single/Double own-model results do not establish which prices a common market dataset better.','',
              'Monthly means 30/60/90 days; rich adds 180/365/730 days. These are synthetic maturity scenarios, not a claim that all such NSE stock expiries exist. Tau is expiry-minus-trade days divided by 365. Noise is independent Gaussian with standard deviation 1% of option time value, without clipping/resampling. Every third strike is withheld.','',
              '## Evidence and continuation','',
              f'- [All fitted parameters and truths]({(source/"parameter_recovery.csv").as_uri()})',
              f'- [All prices/metrics]({(source/"metrics.csv").as_uri()})',
              f'- [All starts and convergence records]({(source/"fits_and_starts.json").as_uri()})',
              f'- [Frozen manifest]({(source/"manifest.json").as_uri()})','',
              'This assessment must not be reused as unseen evidence for later model changes chosen after inspecting it. Keep all negative trials. A later iteration needs fresh assessment data, repeated training initializations and boundary/noise tests. Perfect or universally unique recovery has not been demonstrated.']
    # Local filesystem links, not file:// URIs, work in the desktop preview.
    lines=[line.replace('file://','') for line in lines]
    (args.out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(audit['checks'],indent=2))


if __name__=='__main__':main()
