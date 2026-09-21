"""Assemble all measurements without changing selection or checkpoints."""
import numpy as np
import pandas as pd
from .common import *
from .evaluate import nets,ensemble_metrics
from .figures import SHORT


def md(frame):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return 'not recorded' if np.isnan(v) else f'{v:.6g}'
        return str(v)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(fmt(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def regional(table):
    a=data('development');z=a['coords'];days=z[:,-1]*365
    masks={'7–30d':days<=30,'30–90d':(days>30)&(days<=90),'90–365d':(days>90)&(days<=365),'365–730d':days>365,
        'ATM |x|≤.02':abs(z[:,0])<=.02,'wings |x|>.15':abs(z[:,0])>.15,
        'fast-heavy':z[:,2]>z[:,1],'slow-heavy':z[:,1]>=z[:,2],
        'low slow variance':z[:,1]<=np.sqrt(.00015*.18),'high slow variance':z[:,1]>np.sqrt(.00015*.18),
        'low fast variance':z[:,2]<=np.sqrt(.001*.22),'high fast variance':z[:,2]>np.sqrt(.001*.22)}
    rows=[]
    for arm in table.arm:
        if arm=='BASELINE':pred=ensemble_metrics(nets(arm),a)[1]
        else:pred=np.mean([np.load(OUT/arm/f'seed_{s}/development_predictions.npz')['price'] for s in SEEDS],axis=0)
        for name,mask in masks.items():
            m=old.metrics(a['price'][mask],pred[mask],z[mask,0],z[mask,-1]);rows.append({'arm':arm,'bucket':name,**m})
    result=pd.DataFrame(rows);result.to_csv(OUT/'regional_development.csv',index=False)
    return result


def main():
    table=pd.read_csv(OUT/'PINN_ARCHITECTURE_ABLATION.csv');selection=read(OUT/'selection.json');chosen=selection['chosen']
    base=table[table.arm.eq('BASELINE')].iloc[0];best=table[table.arm.eq(chosen)].iloc[0] if chosen else base
    reg=regional(table)
    r=reg[reg.arm.eq('BASELINE')].merge(reg[reg.arm.eq(chosen or 'BASELINE')],on='bucket',suffixes=('_base','_new'))
    region=r[['bucket','price_RMSE_base','price_RMSE_new','price_max_base','price_max_new']].copy()
    region['RMSE_improvement_percent']=100*(1-region.price_RMSE_new/region.price_RMSE_base)
    columns=['arm','parameters_per_seed','price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_P95','training_seconds_mean','inference_1024_ms_per_seed','eligible']
    display=table[columns].rename(columns={'parameters_per_seed':'Params/seed','IV_RMSE_volatility_points':'IV RMSE (vol pts)',
        'training_seconds_mean':'Train s/seed','inference_1024_ms_per_seed':'1024 prices ms/seed'})
    changes=[]
    for key in ['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE','PDE_P95','PDE_max']:
        changes.append({'Metric':key,'Baseline':base[key],'Selected':best[key],'Improvement %':100*(1-best[key]/base[key])})
    curve=pd.read_csv(OUT/'curves/curve_quality.csv')
    seeds=pd.read_csv(OUT/'per_seed_development.csv');stats=[]
    for arm,g in seeds.groupby('arm'):
        for key in ['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE']:
            stats.append({'arm':arm,'metric':key,'mean':g[key].mean(),'std':g[key].std(),'best':g[key].min(),'worst':g[key].max()})
    statframe=pd.DataFrame(stats);statframe.to_csv(OUT/'seed_robustness_long.csv',index=False)
    efficiency=table[['arm','parameters_per_seed','price_RMSE','training_seconds_mean','peak_RSS_MB','inference_1024_ms_per_seed','inference_single_ms_per_seed','throughput_per_seed']].copy()
    efficiency['inverse_RMSE_per_million_parameters']=1/(efficiency.price_RMSE*efficiency.parameters_per_seed/1e6)
    efficiency['inverse_RMSE_per_incremental_training_minute']=1/(efficiency.price_RMSE*efficiency.training_seconds_mean/60)
    efficiency.to_csv(OUT/'efficiency.csv',index=False)
    final_path=OUT/'final_fidelity/metrics.csv'
    if final_path.exists():
        final=pd.read_csv(final_path);fa=final[final.arm.eq('BASELINE')].iloc[0];fb=final[final.arm.eq(chosen)].iloc[0]
        finaltext=md(final[['arm','price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE','passes_four_gates']])
        finalgain=100*(1-fb.price_RMSE/fa.price_RMSE)
        finaltext+=f'\nFresh price RMSE improvement: **{finalgain:.3f}%**. Final measurements do not trigger reselection.\n'
        fchanges=pd.DataFrame([{'metric':k,'baseline':fa[k],'selected':fb[k],'improvement_percent':100*(1-fb[k]/fa[k])}
            for k in ['price_RMSE','IV_RMSE_volatility_points','price_P95','price_max','PDE_RMSE','PDE_P95','PDE_max']])
        fchanges.to_csv(OUT/'final_fidelity/improvements.csv',index=False)
        finaltext+='\n'+md(fchanges)
    else:finaltext='No new model selected: final synthetic fidelity remains closed.'
    ablation=[];index=table.set_index('arm')
    for a,b,label in [('BASELINE','A0_CONTINUE','Additional training and common stratified batching'),
          ('A0_CONTINUE','CONTROL_PLAIN','New conditioned plain correction (changes capacity and trainable representation)'),
          ('CONTROL_PLAIN','ARCH_A_MODIFIED_MLP','Modified recurrence vs comparable plain correction'),
          ('ARCH_A_MODIFIED_MLP','ARCH_B_MODIFIED_MLP_GRADBAL','Add gradient balancing'),
          ('ARCH_B_MODIFIED_MLP_GRADBAL','ARCH_C_MODIFIED_MLP_GRADBAL_RAD','Add RAD')]:
        if a in index.index and b in index.index:
            ablation.append({'Change':label,'Price RMSE improvement %':100*(1-index.loc[b,'price_RMSE']/index.loc[a,'price_RMSE']),
                'IV RMSE improvement %':100*(1-index.loc[b,'IV_RMSE_volatility_points']/index.loc[a,'IV_RMSE_volatility_points']),
                'PDE RMSE improvement %':100*(1-index.loc[b,'PDE_RMSE']/index.loc[a,'PDE_RMSE'])})
    diagnosis=pd.read_csv(OUT/'gradient_balance_baseline.csv');g=diagnosis.pivot(index=['seed','step'],columns='component',values='scaled_grad_l2')
    ratio=float((g.iv/g.teacher).median())
    rootrel='experiments/pinn_architecture_v6/artifacts'
    report=f'''# PINN architecture improvement report

Selected development candidate: **{chosen or 'NONE'}**. This study improves numerical
Double-Heston PINN fidelity only. The exact teacher, PDE, parameter/state domain,
controlled scenarios and four fidelity gates are unchanged. No market quotes were
parsed, calibrated or scored in this study. No final market test was opened.

## 1. Recovered original architecture

The read-only audit found v5 SHARED as the current best selected checkpoint,
inherited from v4 C3. C1 was not the baseline. All v3/v4/v5 source hashes matched.
See [CURRENT_PINN_BASELINE.md](CURRENT_PINN_BASELINE.md) for the full reproduced specification.
The core has five width-256 tanh hidden layers. Three width-32 two-layer correction
branches augment it. Each seed has 275,684 total parameters; predictions average
seeds 17 and 43. C3 used 40,000 Adam steps and 300 L-BFGS iterations; v5 added 800
Adam steps and 60 L-BFGS iterations, training only its 5,859 extension parameters.
The v6 study starts from these checkpoints and preserves them unchanged on disk.

The output is a positive learned implied total variance passed through an analytic
Black price transform. This is an ansatz, not Black–Scholes training labels. All
supervision uses the unchanged exact Double-Heston teacher. The normalized forward
coordinate is log(F/K), with separate spot/strike/carry removed by homogeneity.

![architectures]({rootrel}/figures/architecture_diagrams.png)

## 2. Diagnosed weaknesses

The baseline already passes all price/IV gates. Its failure is not inability to
cross the originally stated 2e-5 price gate. The observed opportunities were:

- IV gradients are small relative to price gradients: median IV/price L2 ratio
  {ratio:.4f} during a separately labelled 200-step continuation. Historical
  gradient trajectories were not logged and are not invented here.
- Short-maturity and wing PDE residuals are larger than medium-maturity residuals.
  The largest price errors occur at long maturities; short-maturity price error
  is not universally worst. Localization justifies testing RAD, not a claimed
  proof of spectral bias.
- Existing generic log-state transforms extend beyond [-1,1]. New branches use
  train-only affine conditioning of exactly the same 26 engineered features.
  Plain and modified branch controls share this conditioning. Its isolated causal
  effect was not measured, so we do not attribute a gain to conditioning alone.
- Terminal payoff is already exact. There is no learned terminal penalty to balance.
  Finite x-domain edges are not the asymptotic S=0 or S=infinity boundaries.

All gradient norms concern the trainable continuation/correction parameters, not
the frozen parent weights. Price, IV, PDE and convexity raw losses are logged
separately. No Greek loss or artificial boundary loss was introduced.

![baseline gradients]({rootrel}/gradient_balance_baseline.png)

## 3. Research-backed methods and faithful implementation

The recurrence in [Wang, Teng and Perdikaris (2021)](https://arxiv.org/abs/2001.04536)
was checked against the authors' [official implementation](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py).
Its two input encoders U,V are combined at every hidden layer as H←Z⊙U+(1−Z)⊙V,
where Z=tanh(W H+b). The gates are tanh, not sigmoid. We use that published core
inside a checkpoint-preserving correction branch; we did **not** replace or
retrain the entire inherited core. Zero-initialized output heads start at the
exact parent function. This residual use is our adaptation, disclosed separately
from the paper's architecture. Total model size stays below 1.25× baseline.

[Wang, Yu and Perdikaris (2022)](https://arxiv.org/abs/2007.14527) motivates checking
unequal component convergence; we did not compute an NTK eigendecomposition.
Candidate B uses the later official [JaxPI grad_norm implementation](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/jaxpi/models.py):
mean component gradient L2 norm divided by each norm, smoothed with EMA β=.9.
It balances normalized price/IV/PDE terms every 100 Adam steps. Convexity keeps
its original coefficient. Weights are fixed during L-BFGS. This is not described
as the original paper's max/mean statistic. Weight trajectories and failures are retained.

Candidate C uses [Wu et al. (2023)](https://arxiv.org/abs/2207.10289) RAD k=1,c=1:
probability proportional to |R|/mean|R|+1. At three fixed intervals, score 8,192
fresh candidate points and replace 4,096 of 18,000 active collocation locations,
without replacement. 13,904 original global points remain. Every minibatch in
every arm includes marginal coverage of maturity, moneyness, both variance states,
and fast-/slow-heavy states. This does not guarantee every joint stratum has a
point. All arms retain the same number of active collocation slots; RAD incurs
additional scoring cost, explicitly recorded.

Hard-terminal candidate D is not a distinct trained arm because the inherited
condition is already exact. The generic payoff+τN ansatz would leave a spatial
kink at positive τ. See [the mathematical assessment](experiments/pinn_architecture_v6/HARD_TERMINAL_ASSESSMENT.md),
motivated by [Sukumar and Srivastava](https://arxiv.org/abs/2104.08426).
Fourier features were not justified by a demonstrated spectral problem. gPINN's
higher derivatives were unnecessary for the initial ladder. Adaptive activation
E is conditional, as specified before training; untested methods are not invented
as result rows.

## 4. Ablation ladder and development results

All actual arms use 2,000 Adam steps, 512 labels/64 PDE points per step, then 80
strong-Wolfe L-BFGS iterations on the same 4,096-label/256-PDE polishing subset.
Learning rate decays from 2e-4 to 1e-5. Float64 CPU, no weight decay, no gradient
or output clipping. Teacher minibatches use an architecture-independent generator.
Both seeds are retained. A/B/C start independently from the same baseline, not
from each preceding trained arm. This keeps incremental methods attributable.

{md(display)}

Price errors are forward-normalized call units; IV errors are volatility points.
PDE summaries average per-seed residual statistics; the pricing row is an ensemble,
so it is not claimed that the averaged per-seed PDE statistic equals the ensemble
PDE statistic. Baseline training seconds above are the historical v5 increment,
not a fresh equal-compute run; A0_CONTINUE is the current equal-step control.
Inherited C3 training averaged about 2,697 seconds per seed and is shared by all
arms. Full process peak RSS includes libraries/data and is not isolated model memory.

Pairwise attribution (positive percentage means lower error):

{md(pd.DataFrame(ablation))}

A0 is continuation plus common stratified batching; CONTROL_PLAIN additionally
changes branch capacity and conditioning. Only A vs CONTROL_PLAIN isolates the
modified recurrence at approximately equal parameter/step budgets. B vs A and
C vs B isolate balancing and RAD respectively. Equal steps are not equal wall time.

![losses]({rootrel}/figures/training_losses.png)
![weights and gradients]({rootrel}/figures/adaptive_weights_and_gradients.png)
![development metrics]({rootrel}/figures/architecture_metrics.png)

## 5. Selected architecture and development change

Selection is based only on development price RMSE among candidates passing all
four original gates, the inherited PDE guard and baseline price improvement.
Tie-breakers are IV RMSE, parameter count, then latency. Original gates remain
RMSE≤2e-5, P95≤5e-5, max≤2e-4, IV RMSE≤.002 decimal. The inherited guard uses
the original first 256 collocation points and permits at most 10% PDE RMSE increase.
Other physics/curve/regional results remain visible, including regressions.

{md(pd.DataFrame(changes))}

## 6. Fresh synthetic fidelity, opened once after freezing

The old v4/v5 synthetic tests were already exposed and were not relabelled unseen.
After selection, the experiment hashes checkpoints, metrics and source, then
generates seed 206104 (4,096 prices) and 206105 (512 PDE points) within the same
unchanged domain. Only the baseline and selected model are evaluated. There is
no selection retry based on these results.

{finaltext}

## 7. Efficiency and seed robustness

{md(efficiency[['arm','parameters_per_seed','training_seconds_mean','peak_RSS_MB','inference_1024_ms_per_seed','inference_single_ms_per_seed']])}

The selected per-seed parameter change is {100*(best.parameters_per_seed/base.parameters_per_seed-1):.2f}%.
Latency is measured after warm-up, 20 repetitions, one CPU thread, with no gradients.
It is an environment-dependent measurement, not a universal deployment speed claim.
`efficiency.csv` defines two explicit descriptive scores: 1/(RMSE×million parameters)
and 1/(RMSE×incremental training minutes). They are not formal statistical accuracy
measures, and the baseline historical training-minute score is not directly comparable.
Lower RMSE alone does not establish that a larger architecture is more efficient.

There are **two** inherited independent training seeds, matching the existing
protocol. A new third seed would require a new parent-training arm and was not
fabricated. This limits robustness. Per-seed mean, sample standard deviation,
best and worst are provided in `seed_robustness_long.csv`; no statistical
architecture-superiority claim is based on two seeds alone.

{md(statframe[statframe.metric.eq('price_RMSE')])}

## 8. Optimizer contribution and difficult regions

`lbfgs_effect.csv` compares every seed immediately before and after L-BFGS.
Adaptive weights stay fixed throughout each polishing solve. Its benefit must
not be attributed solely to the architecture.

Selected versus baseline development regions:

{md(region)}

Negative improvements are regressions, not excluded quotes. The whole pricing
domain remains x∈[-.36,.36], τ∈[7,730] days with the inherited state/scale bounds.
No claim is made for τ below seven days except the exact expiry payoff itself.
Correlations and reversion speeds are fixed to the existing literature family;
there are no additional extreme-correlation states to sample outside that domain.

## 9. Curves, finite-domain edges and 3D surfaces

The common 81×41 fixed grid uses published baseline parameters, K=1 and r=q=0,
so S/K=exp(x). Exact prices are converted to C/K for curves; this differs from
the C/F units of the main fidelity table. Monotonicity uses C_S=exp(-x)c_x and
convexity uses C_SS=exp(-2x)(c_xx−c_x). Violations below −1e-8 are counted.
At τ=0 the actual payoff is measured, never smoothed for evaluation.

{md(curve)}

![baseline curves]({rootrel}/curves/BASELINE_curves.png)
![selected curves]({rootrel}/curves/{chosen or 'BASELINE'}_curves.png)
![baseline surface]({rootrel}/curves/BASELINE_surfaces.png)
![selected surface]({rootrel}/curves/{chosen or 'BASELINE'}_surfaces.png)
![before error]({rootrel}/curves/BASELINE_heatmap.png)
![after error]({rootrel}/curves/{chosen or 'BASELINE'}_heatmap.png)

Every tested arm has its own curves/surfaces on the same axes and error scale.
Relative error uses max(|exact price|,1e-8) as denominator; near-zero wing prices
can make relative errors large. Price, derivative, edge and terminal violations
are not removed from the plots. BS appears only as a qualitative shape reference.

## 10. Reproduction, preservation and interpretation

The code is under `experiments/pinn_architecture_v6/`; the published-core implementation
is `src/mentor_dh_pinn/research_modified_pinn.py`. The audit came first, then
diagnostics, a hashed predeclared protocol, ordered A/B/C training, selection,
and one-time fresh fidelity. Optional methods retain their declared trigger rules.
All previous experiments and unsuccessful candidates remain unchanged.

Run modules `diagnose`, `train initialize`, `train train --arm NAME --seed SEED`,
`evaluate development`, `curves`, `evaluate select`, `evaluate fidelity`,
`figures`, and `report` in a separate copy with a new output directory. Existing
training directories and final-test opening refuse overwrite. Source SHA-256
manifests are authoritative because this checkout has no project-local Git
commit; the enclosing home-directory repository is not claimed as provenance.

This study tests numerical improvements within a highly accurate existing
solver. A method that lowers price error while worsening PDE, IV, region-level
error or efficiency is a trade-off, not an unqualified improvement. The table
above identifies exactly which additions help and which do not. Market-model
mismatch and ten-parameter calibration remain separate problems.
'''
    (ROOT/'PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md').write_text(report)
    # Required top-level table and baseline diagnostic deliverables are copies;
    # original provenance-bearing artifacts remain in the experiment directory.
    for name in ['PINN_ARCHITECTURE_ABLATION.csv','gradient_balance_baseline.csv','gradient_balance_baseline.png']:
        (ROOT/name).write_bytes((OUT/name).read_bytes())
    print(ROOT/'PINN_ARCHITECTURE_IMPROVEMENT_REPORT.md')


if __name__=='__main__':
    import torch
    torch.set_num_threads(1);main()
