# Factor-structured PINN — separate research variant

The user explicitly approved testing this distinct architecture on 2026-09-08.
It must not be renamed as the previous regular pricing-PINN or included in a
matched-architecture comparison without disclosing the change. No success in
recovering the 12 requested exact-reference cases is established yet.

## Mathematical scope

The independent-factor exponential-affine characteristic function follows the
Double Heston formulation in [Christoffersen, Heston and Jacobs (2009), equations
7–11](https://pure.au.dk/ws/files/17142435/rp09_34.pdf). Each factor contributes
additively to the log characteristic function; the same one-factor coefficient
functions apply separately to both factors.

In this implementation's forward-normalized convention, for z=u-i*shift:

```
log(phi) = i*z*x + sum(kappa_i*theta_i*A_i + v0_i*D_i)
D_tau = 0.5*sigma^2*D^2 - (kappa-rho*sigma*i*z)*D - 0.5*(z^2+i*z)
A_tau = D
D(0) = A(0) = 0
```

Substitution into the characteristic-function pricing PDE gives these Riccati
equations. This variant constrains the transformed equations, not the original
regular network's four-dimensional price residual. Collocation counts across
these representations therefore do not represent identical workloads.

## What is learned versus known

Our new network learns the real/imaginary parts of two normalized coefficient
functions. Known affine dependence on theta/v0 and factor addition are enforced
algebraically; unknown numerical parameter values are not supplied at inverse
calibration. Five dimensionless features feed four hidden layers of 64 tanh
units and a four-real-output head. Analytic deterministic-variance backbones and
zero-time/frequency factors enforce the corresponding exact limits. No full
closed-form coefficient or numerical ODE solver appears in neural inference.

The exact coefficient code is a separate synthetic teacher. Its rationalized
formula was checked against the repository's canonical characteristic function
and its time derivative against the Riccati equations. Runtime guards and tests
ensure the neural core cannot call that reference code. Changing neural weights
changes characteristic predictions, so there is no hidden exact-price fallback.

This is explicitly a two-stage PINN-surrogate calibration method: train the
parameter-conditioned coefficient functions, freeze the neural weights, then
optimize ten constrained parameter inputs through the learned option prices.
It is not an encoder whose outputs are the ten parameters, or a simultaneous
per-case neural-weight/parameter inverse-PINN fit. All comparisons must disclose
this distinction and the additional coefficient supervision used in training.

## First training run, declared before execution

- Float64 PyTorch on CPU, seed 908241, 20,000 AdamW steps.
- 65,536 training factor states; 8,192 independent validation states (seed+1).
- 18,000 independent factor-state/frequency collocation points (seed+2), with
  128 selected per optimizer step; anchor minibatch 512.
- LHS continuous coordinates: tau 7/365 to 2 years, kappa .2 to 12, rho -.8
  to .5. Sigma comes from eta*sqrt(2*kappa*theta), eta .15–.9 and auxiliary
  theta .006–.24. That auxiliary theta sets a feasible sampling range only;
  it is not a network feature or an inverse-calibration answer.
- Half frequencies use 128-point Gauss-Laguerre nodes, half log sampling .01–500;
  shift is 0 or 1. These are synthetic transform states, not NSE quote rows.
- Four dimensionless coefficient-target MSE plus .2 times mean squared
  normalized Riccati residual. AdamW decay 1e-6; cosine LR .001 to .00002.
- Checkpoints every 2,000 steps, selected solely by independent coefficient
  validation RMSE. No exposed-case truth or quote enters this training script.

Coefficient accuracy alone is not recovery. After training, verify neural
Fourier-integral pricing and derivatives, then perform blind multistart
calibration with withheld strikes. Retain failed/invalid cases, keep eight
positive-parameter errors <=5% and both rho errors <=.05, and do not credit an
exact-pricer optimizer. The previously exposed 907931 cases remain development
evidence. Fresh cases and matched Single-Heston experiments would be required
before a fair generalization/model-comparison claim.

## First run result and fixed-objective continuation

The first 20,000-step run completed in 291.9 seconds. Selected coefficient
validation RMSE was 0.00375357. On the original four exact-reference development
cases, all four optimizations produced estimates, but only 6/40 individual
parameters passed and 0/4 complete cases passed. This is not recovery success.

Continue the selected weights with 400 float64 L-BFGS iterations. Every objective
call uses all 65,536 coefficient anchors and all 18,000 collocation points, with
unchanged data/PDE loss weights and no weight decay or gradient clipping. Samples
and loss weights remain fixed; an independent scalar/gradient finite-difference
check must pass before training. Select solely on the same independent 8,192
coefficient validation states, retaining the initial checkpoint. Record exact
reference recovery after selection, not as part of the training objective.

The 400-iteration continuation completed in 312.9 seconds, selecting coefficient
RMSE 0.00312974. The four development cases improved to 9/40 individual passes,
but still 0/4 complete cases and 0/4 joint gates. Both failed stages are retained.

## Derivative-linked primitive refinement

Next enforce A_tau=D exactly by learning only complex A and obtaining D through
automatic differentiation. This removes an independent pair of learned outputs;
it does not provide any unknown calibration parameter. The existing learned A
head supplies the warm start. Known zero-time, martingale and deterministic
limits remain; no numerical ODE or exact coefficient solver enters inference.
The PDE residual now contains the second time derivative of the primitive.

Use the same training/validation/collocation coordinates, 10,000 AdamW steps,
LR .0001 decaying to .000002, batch512 and PDE batch128. Keep the .2 physics
weight and 1e-6 decay. Re-express the same coefficient labels as linear
amplitude-normalized corrections, avoiding logarithms of potentially negative
learned derivative coefficients. This is a change of loss normalization, not
new information or changed truths. Its coefficient RMSE therefore has a
different scale and must not be directly ranked against the old log-target RMSE.
Selection remains independent coefficient validation; recovery is scored only
after selection. The initial A_tau=D identity, weight gradients, and nested
parameter Jacobian have passed independent tests before training.

The derivative-linked 10,000-step Adam run completed in 549.5 seconds, selecting
linear coefficient-validation RMSE 0.00315570 (initial 0.00432872). Its four-case
recovery assessment is recorded separately; coefficient improvement is not
credited as structural recovery. Before consuming its full recovery results,
continue with 400 deterministic float64 L-BFGS iterations using all 65,536 anchors
and all 18,000 collocation points. Keep loss weights, validation-only checkpoint
selection and the initial-checkpoint candidate unchanged. This is weight
refinement of the approved derivative-linked variant, not exact-pricer fitting.

The scoped regular/factor test suite now passes 51 tests (see
`outputs/affine_factor_pinn/testing_final908241/pytest.xml`). Independent cases
can run in CPU worker processes with unchanged seeds, budgets, geometry and
gates. The complete four-case derivative-linked Adam replay with three workers
matched every serial case/start/price/parameter field exactly after excluding
timing fields. This accelerates evaluation only; it does not change the model.

Future joint-pass scoring explicitly rejects noninvertible neural or exact IV
quotes in addition to the existing price/parameter/quadrature gates. No earlier
joint-pass count changes: every completed development stage already had zero.

The derivative-linked L-BFGS run completed its 400-iteration budget in 1181.3
seconds, selecting iteration 400 at linear coefficient-validation RMSE
0.00261242. It stopped at the iteration limit, not an optimizer-convergence
certificate. Four-case development and twelve-case exposed replay assessments
follow selection. All twelve unit/physical truth vectors were independently
checked to exactly match the preserved `exposed12_lbfgs908021` regular-PINN
baseline; the new architecture is not given easier replacement truths.

The selected refinement passed 6/40 individual gates on development4, down from
10/40 before refinement. The exposed12 replay passed 19/120 individual gates,
0/12 complete cases and 0/12 joint gates. The original regular-PINN baseline
is better on individual recovery (48/120), though it too passes 0/12 complete
cases. Lower coefficient validation error did not establish improved recovery.

## Shift-conjugacy refinement (declared before training)

The Riccati equations also imply, for real u,
`A(u-i;k,s,r)=conj(A(u;k-r*s,s,-r))`, and likewise for D. Enforce this algebraic
identity by mapping shifted inputs to the unshifted learned primitive and
conjugating its output. Keep the original kappa*theta prefactor in log CF.
This couples the two probability terms in the call-price formula without an
exact coefficient/ODE evaluation or supplying an unknown parameter answer.
It retains D=A_tau and removes a separately learned shifted branch.

The mapping was checked against the exact reference and Riccati residual on
1,024 independent states; neural conjugacy, weight gradients and a nested
parameter derivative also passed. Test 10,000 AdamW steps, LR .0001 to .000002,
starting from the selected derivative-linked L-BFGS weights. Use the SAME
65,536/8,192/18,000 coordinates and linear coefficient targets, batch512,
physics batch128, physics weight .2 and decay1e-6. Select only on coefficient
validation, including the initial candidate. This remains a separately labelled
factor variant; the exposed recovery cases remain development evidence.

The shift-conjugacy run completed 10,000 steps in 627.4 seconds, selecting
coefficient-validation RMSE 0.00262556. Recovery is assessed separately after
selection; that coefficient score is not a recovery result.

## First-cumulant refinement (declared before training)

The original real-output correction gate does not vanish at unshifted u=0.
Although A=D=0 there, their frequency derivatives can be biased. The known
first log-CF cumulant is minus one half of the expected integrated variance:
`tau * (theta + (v0-theta)*(1-exp(-kappa*tau))/(kappa*tau))` per factor.
Force the real normalized primitive correction to be O(u²), retaining its
O(u) imaginary correction. This makes the first cumulant exact for any neural
weights; it does not provide the unknown numerical parameter values. Higher
frequency coefficients still come from the neural network, not an exact solver.

A targeted test shows the old mapping can violate this moment and the new
mapping preserves it under arbitrary changed head weights; Riccati gradients
remain finite. Continue the selected conjugacy weights for 10,000 AdamW steps,
LR .0001 to .000002, with unchanged data, target normalization, minibatches,
physics weight and decay. Keep coefficient-validation-only selection including
iteration zero. All parameter/price gates and development labels are unchanged.

The conjugacy-only stage scored 21/120 individual passes, 0/12 complete and
0/12 joint passes. The first-cumulant stage finished 10,000 steps in 610.7
seconds at selected coefficient-validation RMSE 0.00273429, then scored 7/40
on development4 and 30/120 on exposed12, with zero complete/joint passes.
These sequential warm-started stages are not a controlled causal ablation or
independent restarts; do not attribute score changes solely to one constraint.

## Two-cumulant refinement (declared before training)

Match one additional power of u in the Riccati equation. For the unshifted
primitive, write `A/(c*tau²) = Q(a) + i*u*g1 + O(u²)`, a=kappa*tau. With
`R=(1-exp(-a))/a`, `Q=(1-R)/a`, `J=(R-exp(-a))/a`, the known slope is

```
g1 = (rho*sigma/kappa)*(Q-J)
   + (sigma²/(4*kappa²))*(-Q+2*J-R²/2).
```

Use stable small-a series for the cancelling combinations. Add this leading
imaginary term with a smooth high-frequency roll-off, and learn only an O(u³)
imaginary correction. Retain the O(u²) real correction, A_tau=D and shifted
conjugacy. This fixes the first two cumulants, not the full characteristic
function or option prices. Higher-order corrections still depend on neural
weights; no full exact coefficient formula or exact-pricer optimizer is used.

Tests compare the second derivative with the canonical reference under both
shifts after changing neural weights. First-moment, boundary, Riccati-gradient,
ten-parameter Jacobian and calibration-isolation tests remain required.
Continue the first-moment selected weights for 10,000 AdamW steps at .0001
decaying to .000002, using the same data, targets, batches, weights, decay and
validation-only selection. Recovery will be scored separately with unchanged
tolerances. This is another development-stage hypothesis, not proven recovery.

The two-cumulant Adam stage completed 10,000 steps in 893.4 seconds, selecting
coefficient-validation RMSE 0.00229791. Its twelve-case assessment is running;
that validation improvement is not a parameter-recovery result.

## Two-cumulant fixed-objective refinement (declared before execution)

On 2026-09-09, continue the selected two-cumulant weights for a bounded 200
float64 L-BFGS-B iterations, recording every 50 iterations. Reuse the existing
fixed-objective implementation, all 65,536 training anchors and 18,000 physics
points, unchanged linear coefficient targets and physics weight .2, with no
weight decay, clipping or resampling. Check the objective gradient numerically
before training. Select on the same independent 8,192 coefficient-validation
states, retaining iteration zero. Neither recovery quotes nor their true
parameters enter this optimization or checkpoint selection. Evaluate the
selected weights with all twelve original cases and unchanged starts/budgets/
gates afterwards, retaining failures. This is a sequential warm-started
development refinement, not a matched-budget ablation or unseen evaluation.

The two-cumulant Adam exposed12 assessment then completed: **27/120 individual
parameters, 0/12 complete cases and 0/12 joint passes**. This regressed from
the first-cumulant stage's 30/120 despite better coefficient-validation error;
both remain below the regular baseline's 48/120. The 200-iteration refinement
was declared and started before consuming this completed assessment. Preserve
the failed Adam result; do not use the exposed truths to choose its successor.

The 200-iteration two-cumulant L-BFGS continuation completed in 670.4 seconds,
selecting iteration 200 at coefficient-validation RMSE 0.00183506 (initial
0.00229791). It stopped at the iteration budget, not convergence. Full-objective
directional finite differences agreed with the analytical gradient to relative
1.23e-7 at step 1e-6. Recovery is evaluated after this selection, not inferred
from its approximately 20% coefficient-RMSE reduction.

Its exposed12 replay passed 30/120 individual gates, 0/12 complete cases and
0/12 joint gates. A float64 post-fit repricing audit exactly reproduced all
archived price/IV metrics. The first draft of that separate audit accidentally
constructed a parameter tensor as float32; its strict equality assertion failed.
The audit was fixed to use explicit float64, without changing model estimates,
assessment outputs, tolerances or the equality requirement.

## Price-aware refinement (declared before execution)

Coefficient MSE alone has not removed pricing bias. Test additional supervision
from the EXISTING independent synthetic `double_surface_train907721` corpus
(SHA256 affcea92146d791213194c6f0420c0367c84f86f5fb325def586318ce48251e8).
Reserve its final 128 parameter surfaces for this stage's validation and use
only the first 896 for training: 112,896 training and 16,128 validation quotes.
All 1,024 candidates passed their archived reference checks; retain this fact
and reject rather than silently exclude any newly invalid input. They were
previously used by some regular-PINN training variants, so the reserved subset
is NOT globally unseen evidence or a fair model-comparison test. None of the
exposed12 vectors or observed quotes is read by this training script.

Keep the selected two-cumulant architecture and initialize from its 200-step
L-BFGS checkpoint. The new loss is the original coefficient MSE + .2 Riccati
MSE + 10 times target-vega-weighted price MSE. Targets are Black reconstructions
of archived canonical synthetic IVs. This is a LOCAL approximation to IV MSE,
not exact IV MSE or independent additional IV information. No clipping of
vega or learned prices and no exact-pricer call enters the training loss.
Known synthetic parameter inputs are supplied during forward-function training;
unknown recovery parameters remain absent at inverse inference.

Use 4,000 float64 CPU AdamW steps, seed 909101, LR 1e-5 decaying to 2e-7,
weight decay 1e-6, coefficient batch512, physics batch128 from the same 18,000
pool, and two training surfaces per step. Evaluate/save every 500 steps and
retain the initial checkpoint. Select ONLY by reserved-surface weighted-price
RMSE; record coefficient validation separately. This changed selection rule is
declared before the run, not chosen by exposed12 parameter success. First run
a five-step operational pilot and a numerical gradient/validation-isolation
test, then start the full run anew from the L-BFGS checkpoint. Afterwards use
the unchanged twelve-case calibration protocol and gates, including failures.

### Minibatch-noise follow-up, declared before execution

A training-only diagnostic at the fixed L-BFGS warm start found the largest
surface contributed 28.78% of weighted-price loss and the largest ten of 896
contributed 83.90%. This suggests highly uneven stochastic updates; loss shares
alone do not prove gradient-noise causality. Do not discard, clip or downweight
these difficult examples. Once the two-surface trial completes, separately
test 1,000 AdamW steps with surface batch16, saving every250, starting again
from the same L-BFGS checkpoint. Keep LR1e-5 to2e-7, all loss weights, coefficient
and physics batches, data splits, seed909101, validation-only selection and
initial candidate unchanged. This provides 16,000 sampled surface visits
versus 8,000 in the first trial, and fewer coefficient/physics updates; it is
not an equal-budget causal ablation. Only this explicitly named batch-size and
step-budget change is being tested; the exposed recovery criterion is unchanged.

The first price-aware run completed 4,000 steps in 463.7 seconds and selected
step 3,500 at reserved weighted-price RMSE 0.00129313 (initial 0.00247705).
Coefficient validation at that checkpoint was worse, 0.00291677 versus the
initial 0.00183506, as openly permitted by the predeclared price-selection
criterion. No recovery result is inferred from either score. Its exposed12
assessment and the declared batch16 trial follow separately; the batch16 trial
restarts from the original L-BFGS weights, not from this price-aware run.

Post-selection actual Black inversion on the 16,128 reserved quotes found
12 noninvertible predictions; whole-set actual IV RMSE is undefined. The lower
weighted-price score is therefore NOT evidence of universally valid prices.
No values were clipped, quotes excluded or checkpoints retrospectively changed.
This limitation is retained in `price_aware_adam909101/VALIDATION_DIAGNOSTIC.md`.

The price-aware step-3500 exposed12 assessment completed: **32/120 individual
parameters, 0/12 complete cases and 0/12 joint passes**. This small development
gain from 30/120 does not establish reliable recovery and remains below the
regular-PINN baseline's 48/120. All twelve fits, every start and the separate
invalid-validation finding are retained. The batch16 trial remains separate.

The batch16 trial completed 1,000 steps in 685.5 seconds, selecting step1000
at reserved weighted-price RMSE 0.00108945 and coefficient RMSE 0.00208945.
All 896 training surfaces were visited, with 16,000 surface draws; the first
price-aware trial also visited all 896, with 8,000 draws. Its source/input,
selected-checkpoint and training-split audit passed before recovery assessment.
Actual-IV validity and the unchanged twelve-case recovery are evaluated after
selection, not used to retrospectively select another checkpoint.

The batch16 exposed12 run finished at **32/120 individual parameters, 0/12
complete and 0/12 joint passes**. Neural held-out price gates passed 3/12,
independent exact repricing 1/12. Actual-IV validation again found 12 invalid
quotes among 16,128; all twelve reference IVs were valid and the learned prices
were below intrinsic. No invalid predictions were dropped or clipped. Final
post-fit repricing reproduced all recorded scores. See the seven-variant report;
the regular baseline still has 48/120 individual and 0/12 complete passes.
