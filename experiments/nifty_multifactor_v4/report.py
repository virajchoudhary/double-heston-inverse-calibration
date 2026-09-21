"""Read-only reporting of frozen experiment outputs; no fitting/selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT,HERE,read,verify,save,metrics
from literature_exact import iv


def markdown(frame):
    cols=list(frame.columns)
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']+
                     ['| '+' | '.join(fmt(v) for v in r)+' |' for r in frame.itertuples(index=False,name=None)])


def main():
    c=verify();r=read(OUT/'controlled_results.json');table=pd.read_csv(OUT/'controlled_metrics.csv');sel=read(OUT/'pinn_selection.json')
    quotes=pd.read_csv(OUT/'controlled_predictions.csv');cases=read(OUT/'controlled_cases.json')
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    for case in cases:
        a=quotes[quotes.case.eq(case['id'])].sort_values(['maturity_index','strike_index'])
        ni=int(a.maturity_index.max()+1);nj=int(a.strike_index.max()+1);k=a.strike_over_forward.unique();days=np.sort(a.tau.unique())*365
        errsh=abs(a.SH_BEST_FOUND-a.DH_EXACT_TEACHER).to_numpy().reshape(ni,nj)
        errdh=abs(a.DH_PINN-a.DH_EXACT_TEACHER).to_numpy().reshape(ni,nj)
        hold=(~a.calibration).to_numpy().reshape(ni,nj);adv=errsh-errdh
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for ax,z,title in zip(axes,[errsh,errdh,adv],['Best-found SH absolute error','DH-PINN absolute error','Advantage: SH error minus PINN error']):
            limit=max(float(abs(adv[hold]).max()),1e-10)
            opts={'cmap':'coolwarm','vmin':-limit,'vmax':limit} if title.startswith('Advantage') else {'cmap':'viridis','vmin':0,'vmax':max(float(errsh[hold].max()),float(errdh[hold].max()),1e-10)}
            im=ax.pcolormesh(k,days,np.where(hold,z,np.nan),shading='nearest',**opts)
            ax.set(xlabel='K/F',ylabel='Days to expiry (log scale)',title=title,yscale='log');fig.colorbar(im,ax=ax,label='Forward-normalized price error')
        fig.suptitle(case['id']+' — held-out cells only; blank cells were used for SH/BS calibration')
        fig.savefig(figures/(case['id']+'_errors.png'),dpi=130);plt.close(fig)
        if case['kind']!='representative':continue
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,d in zip(axes,[30,180,730]):
            idx=int(np.argmin(abs(days-d)));b=a[a.maturity_index.eq(idx)]
            for name in ['DH_EXACT_TEACHER','DH_PINN','SH_BEST_FOUND','BS_SELECTED']:
                ax.plot(b.strike_over_forward,100*iv(b[name].to_numpy(),b.x.to_numpy(),b.tau.to_numpy()),label=name,lw=1.3)
            ax.set(title=f'{days[idx]:.1f} days',xlabel='K/F',ylabel='Implied volatility (%)');ax.legend(fontsize=7)
        fig.suptitle(case['id']+' — smiles; calibration and held-out coordinates shown for shape only')
        fig.savefig(figures/(case['id']+'_smiles.png'),dpi=130);plt.close(fig)
        b=a[np.isclose(a.strike_over_forward,1)]
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for name in ['DH_EXACT_TEACHER','DH_PINN','SH_BEST_FOUND','BS_SELECTED']:
            ax.plot(b.tau*365,100*iv(b[name].to_numpy(),b.x.to_numpy(),b.tau.to_numpy()),label=name)
        ax.set(xlabel='Days to expiry',ylabel='ATM implied volatility (%)',title=case['id']);ax.legend(fontsize=8)
        fig.savefig(figures/(case['id']+'_term.png'),dpi=130);plt.close(fig)
        if case['id'].endswith('_0'):
            fig=plt.figure(figsize=(13,4),layout='constrained')
            X,Y=np.meshgrid(k,days)
            for i,name in enumerate(['DH_EXACT_TEACHER','DH_PINN','SH_BEST_FOUND']):
                ax=fig.add_subplot(1,3,i+1,projection='3d');ax.plot_surface(X,Y,a[name].to_numpy().reshape(ni,nj),cmap='viridis',linewidth=0)
                ax.set(xlabel='K/F',ylabel='Days',zlabel='C/(DF)',title=name)
            fig.suptitle(case['id']);fig.savefig(figures/(case['id']+'_surfaces.png'),dpi=130);plt.close(fig)
    independent=table[table.kind.eq('independent')]
    summary=independent.groupby(['family','model'],sort=False)[['price_RMSE','price_MAE','IV_RMSE_volatility_points']].mean().reset_index()
    summary.to_csv(OUT/'controlled_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    selected=summary[summary.model.isin(['BS_SELECTED','SH_BEST_FOUND','DH_PINN'])]
    pivot=selected.pivot(index='family',columns='model',values='price_RMSE').reindex(c['families'])
    pivot.plot.bar(ax=axes[0],rot=20);axes[0].set(ylabel='Mean held-out price RMSE / forward',title='All independently sampled surfaces')
    values=[]
    for fam in c['families']:
        a=independent[independent.family.eq(fam)].pivot(index='case',columns='model',values='price_RMSE')
        values.append((a.SH_BEST_FOUND-a.DH_PINN).to_numpy())
    axes[1].boxplot(values,tick_labels=c['families']);axes[1].axhline(0,c='black',ls='--');axes[1].tick_params(axis='x',rotation=20)
    axes[1].set(ylabel='SH RMSE minus DH-PINN RMSE',title='Paired surface differences: positive favors PINN')
    fig.savefig(figures/'controlled_summary.png',dpi=150);plt.close(fig)
    shdiag=[read(p)['SH'] for p in sorted((OUT/'baselines').glob('*.json'))]
    optimization={'surfaces':len(shdiag),'total_local_starts':sum(len(v['starts']) for v in shdiag),
        'successful_local_starts':sum(sum(x['success'] for x in v['starts']) for v in shdiag),
        'surfaces_two_successful_near_best':sum(v['converged_near_best']>=2 for v in shdiag),
        'best_near_bound':sum(v['best']['near_bound'] for v in shdiag),'global_optimum_proven':False}
    save(OUT/'optimization_summary.json',optimization)
    s=pd.DataFrame(r['statistics']);s.to_csv(OUT/'paired_statistics.csv',index=False)
    market_status='Real-market validation/final evaluation unavailable; no previously examined dates substituted.'
    if (OUT/'market_final_predictions.csv').exists():
        a=pd.read_csv(OUT/'market_final_predictions.csv');marketrows=[]
        names=['single_fixed_exact','double_fixed_exact','double_fixed_PINN','BS_fixed','single_state_exact','double_state_exact','double_state_PINN','BS_state_flat','BS_state_term']
        for name in names:
            ok=np.isfinite(a[name]);b=a[ok];y=b.target_forward_call.to_numpy();pred=b[name].to_numpy()
            m=metrics(y,pred,b.x.to_numpy(),b.tau.to_numpy());scale=b.discount*b.forward
            m['RMSE_index_points']=float(np.sqrt(np.mean(((pred-y)*scale)**2)))
            m['MAE_index_points']=float(np.mean(abs((pred-y)*scale)))
            m['equal_date_forward_RMSE']=float(np.sqrt(pd.Series((pred-y)**2).groupby(b.date.to_numpy()).mean().mean()))
            marketrows.append({'model':name,'dates':b.date.nunique(),'omitted_nonfinite':int((~ok).sum()),**m})
        pd.DataFrame(marketrows).to_csv(OUT/'market_summary.csv',index=False)
        market_status=markdown(pd.DataFrame(marketrows)[['model','dates','quotes','RMSE_index_points','equal_date_forward_RMSE','IV_RMSE_volatility_points','omitted_nonfinite']])
    passed=r['fidelity']['pass'];reliable=[v['family'] for v in r['statistics'] if v['reliable_structural_advantage']]
    bank=read(OUT/'scenario_bank.json');par=[]
    for kind,rows in bank['market'].items():
        for row in rows:par.append({'id':row['id'],'initial_vol':row['target_initial_vol'],'parameters_slow_first':row['params']})
    headline=('Neural fidelity gates passed. ' if passed else 'Neural fidelity gates FAILED; do not claim validated structural superiority. ')
    headline+='Families meeting the predeclared stringent advantage gate: '+(', '.join(reliable) if reliable else 'none')+'.'
    report=f'''# Predeclared NIFTY/multifactor experiment — v4

v1 stopped at the exact-pricer short-time test before any fitting. v2 passed but its PINN domain missed analytic scenario extremes; v3 widened only that domain. v3's PINN then failed the predeclared development-split fidelity gates because its loss normalisation made price accuracy invisible to the optimizer, so v3 was never locked or evaluated. v4 changes only the PINN objective normalisation, with a predeclared budget/width ladder, and inherits every other v3 artifact byte-identically. All failed records are preserved; see `../AMENDMENT.md`.

## Outcome

{headline}

This is not parameter recovery. Model coefficients are supplied to a teacher-assisted,
network-side PDE PINN; the network approximates prices. Single Heston's five
parameters are recalibrated separately to each controlled surface's calibration
portion. The exact DH teacher has zero target error by construction, not by a
fair contest against independently observed market truth.

## Correlation contract and literature

Read [CONTRACT_AUDIT.md](../CONTRACT_AUDIT.md). The existing CF and PDE are
the separable four-shock model. Individual correlation bounds suffice. The
canonical disk restriction is preserved untouched; published parameters are used
only through the isolated `literature_exact` adapter. Slow-first order swaps the
paper's factor labels but changes no numbers. The radius-.95 projected analogue
is a diagnostic, never described as published or selected for market fit.

Source: [Chang, Wang & Zhang (2021), Table 1](https://onlinelibrary.wiley.com/doi/10.1155/2021/6634779),
ordinary Heston comparators, not the fractional extension. Its reported MSEs are
DJIA ETF calibration results, not NIFTY results. We have not reproduced that
market dataset or imported those MSEs as our evidence.
[Christoffersen–Heston–Jacobs (2009), section 3](https://pure.au.dk/ws/files/17142435/rp09_34.pdf)
supplies the four-shock model and motivation for independent smirk level/slope movement.

## Scenarios and frozen protocol

Manifest SHA-256: `{(OUT/'manifest.sha256').read_text().strip()}`.
All sources are hashed and copied into `frozen_sources/`; all previous results
remain development evidence. The complete configuration, hypotheses H1–H5,
sampling seeds and limits are in `manifest.json` and `scenario_bank.json`.
The four market candidates per model target 14%, 18%, 22%, 28% initial volatility.
Theta and v0 scale by s; sigma by sqrt(s); kappa/rho stay fixed. Single uses its
own original total variance denominator .0244, Double .0255, for equal target
volatility levels. This preserves each CIR Feller ratio exactly.

{markdown(pd.DataFrame(par))}

The controlled set has 10 predeclared representative surfaces plus 40 independent
draws (8 per family), all retained. Each surface has 61 strikes K/F=.70–1.30 and
33 geometrically spaced maturities from 7 to 730 days: 2,013 quotes. Half-sized
alternating 3-strike x 2-maturity blocks form the calibration/held-out partition.
Continuous random states/scales come from the distributions in the manifest.
The representative surfaces are excluded from bootstrap estimates.

## Fair baselines and optimization

Single Heston: differential evolution (population 30, 25 generations), then 12
local starts per surface, maximum 400 evaluations each. All solutions, failures,
statuses and bound contacts are retained. Bounds: kappa .05–40, theta .0001–.5,
v0 .00001–.5, rho +/-.995, sigma=eta*sqrt(2*kappa*theta), eta .01–.999.
These impose strict Feller; conclusions are conditional on this constraint and
finite search. **Best found is not a mathematical proof of a global optimum.**

Optimization summary: `{json.dumps(optimization)}`.
Surfaces without repeated near-best convergence require extra caution, not removal.
Calibration minimizes forward-normalized price MSE over all calibration quotes.
BS_FLAT fits one volatility; BS_TERM fits maturity-specific volatilities with
linear total-variance interpolation and constant-volatility endpoint extrapolation.
Their family is selected on an inner calibration-only split, then refitted on
the complete calibration portion. No held-out option's IV prices itself.

## Actual PINN training and fidelity

Two fresh seeds (17,43), smooth tanh architecture, depth 5, width {sel["width"]} (candidate {sel["chosen"]}: {sel["reason"]}), float64,
100,000 synthetic labels, 18,000 PDE collocation points, {sel["adam_steps"]:,} Adam steps with
a predeclared cosine learning-rate schedule, then up to 300 L-BFGS iterations.
Final-checkpoint prices are averaged; no best seed or final-test-driven retraining.
Training spans both variance states and a continuous parameter-scale domain,
not the final controlled surfaces. The teacher sees no market labels.
The positive total-variance price ansatz enforces terminal payoff analytically.
Loss (v4): price MSE/(2e-5)² + valid IV MSE/(0.002)² + .1 scaled PDE MSE/.01²
+ .1 negative-convexity penalty. The existing PDE includes all x/variance mixed
and variance second derivatives. Initial states vary as inputs, not learned truth labels.
IVs that cannot be resolved numerically are excluded only from IV loss/metrics;
their prices remain in price loss and all headline price evaluations.

Held-out numerical fidelity (8,192 continuous points):

```
{json.dumps(r['fidelity'],indent=2)}
```

Absolute synthetic prices use F=1,D=1, so price errors equal forward-normalized
errors. They are not NIFTY index points. IV errors are in volatility percentage
points (100 x decimal-IV RMSE). Every per-quote prediction/residual and invalid-IV
count is retained. Phase A exact-pricer validation output is `exact_tests.txt`.
Cross-order Fourier checks and adaptive fallbacks are recorded; no teacher price
clipping. Tiny sub-tolerance negative/time-value roundoff may occur and is disclosed.

## Controlled held-out comparison

Means below are across the 8 independent surfaces per family, not pooled market data.

{markdown(summary)}

Paired statistics:

{markdown(s[['family','surfaces','mean_advantage','median_advantage','DH_win_fraction','hierarchy_DH_SH_BS_fraction','DH_over_SH_pooled_RMSE','reliable_structural_advantage']])}

Full 95% descriptive and 99% Bonferroni intervals are in `paired_statistics.csv`.
The sampling unit is the independent surface, not each strike as an independent
observation. Eight surfaces per family is a bounded initial study, not exhaustive
coverage. Strong claims require fidelity pass, DH error <=20% of SH error and
positive paired advantage lower bound. Negative/failed results are retained.

## Plots and interpretation

![Summary](figures/controlled_summary.png)

Interpretation: lower bars mean smaller held-out error. A positive paired
difference favors the PINN; negative differences favor recalibrated Single Heston.
Consult the fidelity gate before turning any ranking into a model claim.

Every one of the 50 surfaces has a three-panel error map in `figures/`:
SH absolute error, DH-PINN absolute error on the same color scale, and their
difference. Red/positive advantage means PINN is closer; blue/negative favors SH.
Blank cells are calibration cells, not removed difficult held-out quotes.
Representatives also have smiles, ATM term structures and 3D surfaces. The
representatives were fixed before fitting, not chosen for attractive outcomes.
All summary scores exclude calibration cells, even where the shape plots show both.

Hypotheses concern price-surface flexibility, not guaranteed ordering. Fast-state
shocks decay with exp(-10.7526*T); slow-state shocks with exp(-.9491*T).
This variance-state fact alone does not prove that the *pricing advantage* will
have the same maturity pattern. Bucket scores in `bucket_metrics.csv` test that.
Fixed-total twist cases have equal current variance but distinct future variance
curves; because SH is separately refitted, this is a stronger approximation test
than merely assigning SH the same current total variance. Factor weights/correlations
are never tuned after outcomes. These experiments do not establish universal superiority.

## Real-market fixed-parameter / state-adaptive legs

{market_status}

Validation reserved 2026-08-04–21; final reserved 2026-08-24–09-16. Only fresh
official NSE ZIPs after the old archive cutoff are eligible. Download status is
recorded even for errors/404s. Missing data are not replaced by synthetic quotes.
Final quotes can be opened only after validation choices, networks and baselines
are frozen. Daily closes are not synchronized bid/ask midpoints; bhavcopies do
not support spread-normalized accuracy claims. Source hashes and URLs, actual
expiry dates, anchor-only forward/discount fits, identical liquidity/IV filters,
rejected quotes and groups are retained by the existing cleaner.

The strict fixed-parameter market result must be reported separately from
`STATE_ADAPTIVE_DIAGNOSTIC_NOT_FIXED_PARAMETER_PRIMARY`. That optional leg
fits only date-specific initial variance(s) on anchor quotes with validation-selected
structural parameters fixed; outer strikes never estimate states. Any neural
state-domain violations are flagged, never extrapolated silently. A market model
error is exact-model minus market; neural approximation error is PINN minus the
same exact model. These are signed additive residuals, not additive RMSEs.

## Reproduce / continue

Run from this experiment directory with the existing workspace Python environment:

```
python run.py inherit    # copies and hash-verifies the v3 bank, market data and selection, surfaces, baselines and teacher
python run.py validate
python run.py train --seed 17 --candidate C1
python run.py train --seed 43 --candidate C1
python run.py train --seed 17 --candidate C2    # predeclared ladder; C3 only if C1 and C2 both fail
python run.py train --seed 43 --candidate C2
python run.py select-candidate
python run.py lock
python run.py evaluate
python run.py fetch-final
python run.py market-final
python run.py report
```

Existing completed output directories must not be overwritten. `freeze` fails
if artifacts already exist. Continue pending steps only; a changed mathematical
protocol requires a new experiment version, not overwriting this manifest.
No GitHub push, Kaggle upload, old checkpoint replacement or canonical contract
change was performed. This directory imports the hashed repository modules;
their frozen copies are included for provenance and reconstruction.
'''
    (OUT/'REPORT.md').write_text(report)
    print('REPORT',OUT/'REPORT.md',flush=True)


if __name__=='__main__':main()
