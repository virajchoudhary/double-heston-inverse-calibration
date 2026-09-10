# Conditional deeper regular-PINN trial

## Trigger and user authorization

The user authorized a much deeper PINN only if existing recovery remains poor.
That condition is met: the original five-layer baseline passes 48/120 individual
parameter gates but 0/12 complete cases; both float64 continuations pass 45/120
individual and 0/12 complete cases. No shallow model is overwritten or replaced.

The implemented 17-layer model has 416,321 neural weights/biases, versus 107,201
for the five-layer control. Those counts are network coefficients, not the ten
unknown Heston parameters calibrated after training. The default architecture
still has zero added residual blocks; deepening is an explicit experiment option.

## Architecture and paired experiment — declared before training

Keep the regular normalized price-PDE, features, implied-variance output map,
exact terminal condition, admissible parameter map and float64 precision.
Warm start `double_float64_extended909111/model.pt`. Add six residual blocks
between the existing five hidden layers and scalar correction head. Each block
has two width-160 linear/tanh layers and maps
`h -> h + tanh(W2*tanh(W1*h+b1)+b2)/sqrt(6)`.
W1 uses seeded PyTorch initialization; W2 and b2 start at zero. Thus the new
network has **17 hidden nonlinear layers along its longest path**, rather than
five, and exactly preserves the starting IV function and its PDE derivatives.
Skip connections are disclosed; this is not a plain 17-layer sequential MLP.
All weights are trained, not just the new blocks. No ReLU or dropout is used.

Run one 17-layer trial and one 5-layer control from the identical checkpoint.
Both: seed 909121, 4,000 AdamW updates, fresh optimizer, LR 1e-5 to 2e-7 cosine,
decay 1e-6, anchor/Sobolev/PDE/shape loss definitions unchanged. Use the same
131,072 synthetic training quotes, 16,384 validation quotes and fixed 18,000
collocation points, with batches 512 and 128. Identical NumPy seeds ensure
identical sampled training batches and collocation indices. The deep model's
additional initialization consumes only PyTorch RNG, not the sampling RNG.
This is a matched-update/data comparison, not equal wall-clock or compute cost.

Select weights every 500 updates solely by full validation IV RMSE, including
step zero. Record IV RMSE on the first 16,384 training rows for a development
train/validation comparison; never use that diagnostic to select checkpoints.
Run two-step operational pilots from the original warm start; pilot weights
are not used by either 4,000-step trial. Preserve unsuccessful trials.

## Recovery and integrity

After both selections are frozen, evaluate the same twelve previously exposed
clean six-expiry cases, five blind starts each, 400 evaluations per start.
Each case: 21 strike/forward ratios 0.8–1.2, tau=30/60/90/180/365/730 days
divided by 365, 84 calibration quotes and 42 held-out quotes. No generating
truth or held-out quote can enter calibration or checkpoint selection. The
eight positive parameters still require <=5% relative error; both correlations
require <=0.05 absolute error. Existing price and quadrature gates stay fixed.
Repeat checkpoint/source/selection verification and held-out corruption replays
with reference-pricer entry points blocked. Report all failed cases.

These synthetic cases and validation data have already been exposed during
development. The result cannot establish unseen generalization, zero overfitting,
market recovery, or a completed Single/Double/Black-Scholes research comparison.
Greater depth is a hypothesis, not a guarantee of better parameter recovery.

Pre-evaluation checks: both two-step pilots completed with finite gradients and
retained their initial checkpoints by validation score. Their initial IV scores,
losses and fixed physics weights matched exactly. New-block output matrices
changed during pilot training, so the added capacity is not frozen or bypassed.
The scoped suite passed **82 tests**, including full 17-layer raw-price versus
transformed-PDE checks after nonzero residual-weight perturbations, terminal and
zero-vol-of-vol limits, exact checkpoint loading, and objective-gradient checks.
The legacy diagnostics now preserve native Torch residual branches instead of
passing them through the MLX-only weight-copy path.

## One-use fresh comparison — declared while training, before recovery evaluation

In addition to the twelve exposed development cases, freeze a separate clean
twelve-case comparison using assessor seed **909131** (Double-Heston truth RNG
seed 929131), otherwise the identical geometry, starts, budgets and gates.
Do not generate or inspect these cases until both training selections finish.
Both selected models are evaluated regardless of their development scores.
This fresh sample does not select or fine-tune either checkpoint. If it is used
to inform later model changes, it becomes exposed and must not be reused as an
unseen test. Twelve central-domain draws still do not establish broad reliability.

## Completed outcome

Both selected update 4,000, strictly by minimum validation IV RMSE. Both visited
all 131,072 training rows with exactly 2,048,000 anchor draws. The recorded
visit-count arrays, fixed collocation indices, normalizers and initial relevance
weights are identical. No gates or training budgets were changed mid-run.

| Cohort | 5-layer individual gates | 17-layer individual gates | All ten, either model | Joint, either model |
|---|---:|---:|---:|---:|
| Previously exposed 12 cases, seed 907931 | 45/120 | 45/120 | 0/12 | 0/12 |
| One-use fresh 12 cases, seed 909131 | 55/120 | 55/120 | 0/12 | 0/12 |

Do not compare 55/120 against 45/120 as a training improvement: the cohorts have
different generating parameters. Within each cohort, every individual gate
classification is identical between models. Parameter estimates are not copied
or numerically identical: the largest shallow/deep difference is 0.0553 tolerance
units on the exposed cohort and 0.1011 on the fresh cohort. The selected deep
model's six added output matrices have nonzero norms (approximately .0097–.0131).

Neural held-out price gates pass 9/12 on the exposed cohort and 7/12 on the fresh
cohort for both models. Exact repricing of fitted parameters passes 0/12 in all
four model/cohort combinations. No assessed neural IV quote failed inversion.
This trial therefore **does not justify replacing the shallower model** and
does not achieve the all-ten-parameter goal. The default remains five layers.

Validation IV RMSE: shallow 0.0003331412723; deep 0.0003330637248. The difference
is only about 0.0233% relative and did not change parameter pass classifications.
Measured training times were 259.5 seconds and 557.4 seconds respectively on this
machine, with partly concurrent workloads; these are observed timings, not a
controlled standalone speed benchmark. More depth cost more compute here without
an observed recovery benefit. This is one warm-started architecture/budget trial,
not proof that every deeper PINN must fail.

### Integrity and physics diagnostics

All **82 scoped tests pass**. Training-source snapshots, data/weight hashes,
selected-step tensor equality and full validation scores were verified. Every
actual fit was replayed after corrupting all 42 held-out x/tau/IV inputs, with
six reference-pricer entry points blocked: all fields except timing matched
exactly, across both models and both cohorts (48 fit replays, 240 starts).
This is bounded leakage evidence, not a universal no-leakage guarantee.

Independent 4,096-point collocation diagnostics (seed 909141) are finite and
have no sampled negative convexity or calendar-total-variance derivatives for
either selected model. Terminal conditions pass. Nevertheless, all-point
dimensionless PDE RMSE is 0.07094 shallow and 0.06980 deep; wide-domain RMSE is
0.14127 and 0.13899. Finite arithmetic and sampled shape checks are not evidence
of an exactly solved PDE. These diagnostics did not alter training or selection.

### Artifacts and continuation

- [Exposed development comparison and all deep parameters](../outputs/regular_pinn_recovery/deep17_development_report909121/REPORT.md).
- [One-use fresh comparison and all deep parameters](../outputs/regular_pinn_recovery/deep17_fresh_report909131/REPORT.md).
- [Independent PDE diagnostics](../outputs/regular_pinn_recovery/deep17_pair_physics909141.json).
- [Executable scoped-test results](../outputs/regular_pinn_recovery/testing_deep17_final909121/pytest.xml).

The exposed truth seed is 927931; the new comparison truth seed is 929131.
Both cohorts are now visible. If these outcomes inform subsequent training,
neither cohort may be described as unseen for the next iteration. No further
training is running. No original dataset, Kaggle upload or GitHub branch was
changed by this experiment; local model/code/report files remain unpushed.

Selected weight hashes:

- `shallow5_control909121/model.pt`: `149a15b208b4fe3588ef44514210bfa192de756477c62bb5a1aa4d1a4afbd4c9`.
- `deep17_adam909121/model.pt`: `54d033e8b8b05ac9236fc7e19563f6a935bf387fa2315c08b759c727992101d3`.

The architecture option is `finetune_regular_torch.py --add-residual-blocks 6`;
omit it for the shallow control. This initializes the extra capacity with
`TorchRegularVariancePINN.deepened(6)` and does not retrain from scratch. Future
trials must use new output directories and explicitly document any new budget,
initialization, loss or sampling change. Do not present further tuning against
these same cases as a fresh confirmation test.
