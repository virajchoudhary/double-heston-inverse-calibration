"""Figures and handoff from completed runs only; no fitting or data selection."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    result=json.loads((HERE/'extended_stratified/results.json').read_text())
    audit=json.loads((HERE/'btc_audit/audit.json').read_text())
    data=json.loads((HERE/'extended_stratified/data_audit.json').read_text())
    for folder in ['pilot','extended_stratified']:
        manifest=json.loads((HERE/folder/'manifest.json').read_text())
        for f,h in manifest['source_sha256'].items():
            if f=='src/mentor_dh_pinn/maturity_dual_pinn.py':
                assert hashlib.sha256((HERE/'model_training_snapshot.py').read_bytes()).hexdigest()==h,f
            else:
                assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    dest=HERE/'deliverables';dest.mkdir(exist_ok='--refresh-report' in sys.argv)
    rows=pd.DataFrame(result['rows']);rows.to_csv(dest/'extended_metrics.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(12,4.6))
    for ax,key,title in zip(axs,['price_RMSE','IV_RMSE_vol_points','PDE_RMSE'],
                           ['Price RMSE (C / DF)','IV RMSE (volatility points)','Scaled PDE residual RMSE']):
        for j,(kind,color) in enumerate([('single_branch','#6c757d'),('dual_branch','#1f5fa8')]):
            values=rows[rows.model.eq(kind)][key].to_numpy()
            bars=ax.bar(j,values.mean(),width=.6,color=color)
            ax.scatter([j-.07,j+.07],values,s=22,c='black',zorder=3)
            ax.text(j,max(values)*1.06,f'{values.mean():.4g}',ha='center',va='bottom')
        ax.set_xticks([0,1],['Single branch','Dual branch']);ax.set_title(title,fontsize=11)
        ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Extended synthetic development comparison — 64 new parameter cases, two seeds')
    fig.text(.5,.025,'Both architectures solve Double Heston. Bars: seed means; dots: seeds 17 and 43. Not a Bitcoin-market test.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(dest/'extended_pinn_errors.png',dpi=300);plt.close(fig)

    old=pd.read_csv(ROOT/'experiments/btc_multifactor_v1/artifacts/amend01/test_scores.csv')
    models=['BS_EXPIRY','SH','DH'];labels=['Black–Scholes per expiry','Single Heston','Double Heston']
    cells=[('shock','B'),('shock','A'),('calm','B'),('calm','A')]
    fig,ax=plt.subplots(figsize=(11,5.5));metric_rows=[]
    for i,(model,label,color) in enumerate(zip(models,labels,['#6c757d','#d9822b','#1f5fa8'])):
        values=[]
        for regime,design in cells:
            cell=old[old.model.eq(model)&old.regime.eq(regime)&old.design.eq(design)]
            value=float(cell.fwd_price_RMSE.median());values.append(value)
            metric_rows.append({'model':model,'regime':regime,'design':design,'median_normalised_price_RMSE':value,'dates':len(cell)})
        bars=ax.bar(np.arange(4)+(i-1)*.25,values,.25,label=label,color=color)
        ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=9)
    ax.set_xticks(range(4),['Shock\nheld-out expiries','Shock\nheld-out strikes','Calm\nheld-out expiries','Calm\nheld-out strikes'])
    ax.set_ylabel('Median per-date price RMSE (stored forward-normalised units)')
    ax.set_title('Existing Bitcoin price errors — corrected BS baseline, unchanged Heston fits')
    ax.legend(frameon=False);ax.set_ylim(bottom=0,top=ax.get_ylim()[1]*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.02,'Preprocessing caveat: forward inputs use held-out price/IV information. Not certified leakage-free; not PINN results.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(dest/'bitcoin_price_error.png',dpi=300);plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(dest/'bitcoin_price_metrics.csv',index=False)
    m=result['mean_seed_metrics'];b=m['single_branch'];d=m['dual_branch']
    improvements={k:100*(1-d[k]/b[k]) for k in b}
    table='\n'.join(f"| {r['model']} | {r['seed']} | {r['price_RMSE']:.6f} | {r['IV_RMSE_vol_points']:.3f} | {r['PDE_RMSE']:.3f} | {r['development_negative_convexity_count']} |" for r in result['rows'])
    text=f'''# Bitcoin comparison and portable dual-PINN research handoff

## Bottom line

The existing numerical BTC comparison was found and reproduced. It is NOT a
PINN comparison and is NOT certified leakage-free. A new dual-maturity PINN
was implemented and trained separately on synthetic data. Existing production
checkpoints, original BTC results, Kaggle and GitHub were not changed.

## Existing Bitcoin experiment

Sources: ../btc_multifactor_v1/REPORT.md, amended test_scores.csv, raw archives.
Verified {audit['raw_hashes_verified']} raw archive hashes and both protocol manifests;
reproduced all {audit['score_rows_reproduced']} stored metric rows. The displayed BS arm is the corrected
per-expiry version, not the flawed per-quote-volatility BS_TERM arm.
The market comparison has 29 shock dates and 21 calm dates, two holdout designs.
On the primary shock/held-out-expiry cell, median IV RMSE is BS 8.133,
SH 2.496, DH 1.857 volatility percentage points. These are cross-sectional
held-out option errors, NOT forecasts of future dates or parameter truth.

### Important data dependence

data.clean reconstructs each expiry's forward from trade price and exchange IV
before splitting. Thus held-out targets affect covariates, put-call conversion
and filters. An in-memory +5% perturbation of held-out prices at fixed IV changed
the inferred forwards by up to {audit['maximum_relative_forward_change']*100:.3f}% on
{audit['probe_date']}. This proves dependency, not the magnitude or direction of
its effect on the model ranking. No raw file was changed. The inspected archive
contains index_price but no independent underlying_price/expiry forward field.
Need independently timestamped forward data and verified settlement conversion
for a target-independent market rerun. The old p-values do not remove this issue.

Official convention references for a future adapter audit:
- https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options
- https://support.deribit.com/hc/en-us/articles/25944775983133-USD-Order-Options

![Existing market price error](bitcoin_price_error.png)
Interpretation: DH has the lowest stored median error, subject to the dependency
caveat above. This is not evidence that the NEW PINN beats these models.

## Architecture implemented

src/mentor_dh_pinn/maturity_dual_pinn.py:
- PortableVariancePINN: 3 x 96 tanh, 21,121 weights/biases, control architecture.
- MaturityDualPINN: two 3 x 64 tanh experts, 19,970 weights/biases total.
- 24 dimensionless features include log-forward moneyness, maturity,
  relative variance, mean-reversion horizons, skew and Feller-ratio quantities.
- Smooth short/long gate g = sigmoid(2 log(tau / (90/365))).
- Bounded log-IV correction delta = 1.8 tanh((1-g) h_short + g h_long).
- IV = sqrt(expected average total variance) exp(delta).
- w = tau IV²; Black price map returns C/(D K), then the adapter restores C.
- The same audited AD Double Heston PDE differentiates BOTH experts and gate.
- Loss: price MSE/.01² + IV MSE/.05² + .01 scaled PDE MSE
  + .01 negative-convexity squared penalty. Exact terminal payoff.
- No dropout, ReLU, fixed expiry grid, joint correlation-disk constraint or
  forced Feller constraint. No exact-pricer inference or ten-parameter output.

This adapts the maturity-specialist idea from the legacy dual inverse network,
not its fixed 27-short/18-long quote layout or structural restrictions.

## Data and fair comparison

Both controls are DOUBLE HESTON PINNs. “Single branch” does not mean Single Heston.
Extended training: {data['train_cases']} cases, {data['train_labels']} labels;
development: {data['development_cases']} disjoint new cases, {data['development_labels']} labels;
18,000 collocation points. New development cases are disjoint from both old
pilot splits and new training. All exact price targets are retained.
Undefined IVs are excluded from IV loss/metric only: train {data['train_invalid_IV']},
development {data['development_invalid_IV']}. Thus IV metrics do not cover every
price point. Independent adaptive quadrature fallbacks: training {data['train_fallbacks']},
development {data['development_fallbacks']}. Price bounds/nonfinites fail closed.

Domain: log(F/K) [-1,1], tau 3/365 to 2 years; theta,v0 .005–1.5;
slow kappa .3–3, fast = slow + gap 2–15; rho -.9 to .9;
sigma = eta sqrt(2 kappa theta), eta .15–2. LHS; positive quantities
sampled logarithmically except rho. All ten quantities vary across cases.

Both seeds 17/43: 4,000 Adam steps then max 100 L-BFGS iterations.
Identical loss, minibatch sequence, data and budgets; no development early stop.
L-BFGS uses two labels per training case, 512 labels spanning all 256 cases.

## Development results (NOT market results)

| Architecture | Seed | Price RMSE C/DF | IV RMSE vol points | Scaled PDE RMSE | Convexity violations |
|---|---:|---:|---:|---:|---:|
{table}

Mean-seed changes versus the matched single-branch control: price error
{improvements['price_RMSE']:.2f}% reduction, IV error
{improvements['IV_RMSE_vol_points']:.2f}% reduction, PDE error
{improvements['PDE_RMSE']:.2f}% reduction (negative means worse).
Predeclared development criterion passed: **{result['development_criterion_passed']}**.
This is not a statistical significance test; there are only two training seeds.
Convexity is a sampled diagnostic at tolerance 1e-7, not a global guarantee.

![Extended PINN errors](extended_pinn_errors.png)
Interpretation: compare approximation to the synthetic exact DH teacher, not
market pricing ability. Neither candidate recovers parameters in this test.

## Retained failures and limitations

The original pilot reduced mean price RMSE by 43.77% but slightly worsened IV
RMSE (8.844 to 8.930 points), so its promotion criterion failed. Its L-BFGS
prefix covered only six parameter cases, later identified as unintended.
The first extended attempt was stopped during data generation before training;
its manifest is preserved under extended/. EXTENDED_AMENDMENT_01.md corrected
the label subset before the corrected training began. No result was hidden.

A post-training unit check exposed a feature-broadcast bug when one unbatched
structural parameter set was shared across quotes. Adding an explicit zero
broadcast fixes that input path. Training used batched parameters, so no run
was affected. The exact original source is retained as model_training_snapshot.py
and matches both training manifests. Regression checks require bitwise identical
batched prices under the original and corrected implementations, and compare
the dual residual with an independently differentiated raw price PDE.

The new adapter accepts arbitrary quote counts, strikes, expiry dates converted
to year fractions, calls/puts, forwards and discount factors. It requires
independent forward inputs and prices in a consistent currency. It rejects
non-European exercise and out-of-domain maturity/moneyness. Supplied parameters
must also be inside the trained distribution: mathematical admissibility alone
does not certify coverage. BTC/ETH/indices are not interchangeable without
checking volatility coverage, carry and settlement conventions. No claim of
support for American, barrier, Asian or other exotic payoffs.

## Reproduction and continuation

Run from repository root using ../.venv/bin/python:

    -m pytest tests/test_maturity_dual_pinn.py experiments/portable_dual_pinn_v1/test_extended.py
    experiments/portable_dual_pinn_v1/run_pilot.py
    experiments/portable_dual_pinn_v1/audit_btc.py
    experiments/portable_dual_pinn_v1/run_extended.py
    experiments/portable_dual_pinn_v1/build_report.py

Scripts refuse to overwrite output directories. Reproduce in a separate checkout
or choose a NEW declared output directory; do not delete or overwrite evidence.
Manifests hash code and protocols before training. NPZs contain coords, params,
price, IV, case IDs; checkpoints, per-seed histories and metrics are retained.

Next justified step: independently sourced forward/settlement data, a fresh
frozen market protocol, parameter-domain/teacher-fidelity checks, and equal
calibration budgets for BS/SH/exact DH/PINN DH on only calibration quotes.
Do not tune on the old 50 Bitcoin test dates and call them untouched.
Do not interpret a better surrogate architecture as removing Heston model bias.
'''
    (dest/'REPORT.md').write_text(text)
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='DELIVERY_MANIFEST.json'}
    (dest/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'results':m,'reductions_percent':improvements,'report':str(dest/'REPORT.md')},indent=2))


if __name__=='__main__':main()
