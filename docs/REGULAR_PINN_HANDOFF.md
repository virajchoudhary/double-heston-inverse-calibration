# Regular Single/Double Heston PINN — continuation context

## Latest snapshot — September 10

**Current continuation:** the [compositional price-PDE PINN protocol](COMPOSITION_PINN_PROTOCOL.md)
defines a new separately labelled architecture, reusing a Single Heston pricing
PINN through the independent-factor convolution identity. The broader component
data has 129840 usable training states, 16239 validation states and 18000 PDE
collocation points. A 5000-step continuation finished but selected step 0:
validation did not improve, so the legacy Single and selected component weights
are identical. The first same-quote comparison has completed: **57/120 individual
parameter gates, 0/12 complete recoveries, 12/12 neural price gates and 2/12 exact
repricing gates**. Both pricing measures beat the Single control in all 12 cases,
but only on the exposed DH-generated surfaces. See
[the first composition report](../outputs/regular_pinn_recovery/composition_report910031/REPORT.md).
The next loss-balance trial, `component_balanced910071`, completed 10000 updates
and selected its final checkpoint at validation IV RMSE .0002100335908, about42%
below the initial error. Its separately frozen full comparison has completed:
**50/120 individual parameter gates, 0/12 complete recoveries, 12/12 neural price
gates and 0/12 exact-repricing gates**. Both pricing measures still beat both
Single controls in all twelve DH-generated cases, but parameter recovery got
worse than the first composition trial. The retrained variant is not promoted
as a recovery improvement. See
[the latest report, graphs and both models' parameter tables](../outputs/regular_pinn_recovery/composition_balanced_report910101/REPORT.md).
All jobs are complete. No original model or data was replaced, and no new work
was pushed. All 55 scoped tests pass. The broader checkout run has 629 passes, 43 failures
and one skip; see the protocol for missing historical evidence and other issues.

The full-PDE audit improves in average residual but retains small numerical
wide-moneyness shape/bound violations. A training-only inverse-conditioning
diagnostic shows that lower average IV error can coexist with a worse median
largest parameter-bias direction. Next proposed work is adapting the already
available grouped inverse-sensitivity loss to the composed network. That next
training experiment has NOT been implemented or run; it is not guaranteed to
solve recovery. Do not repeat depth/learning-rate trials without addressing this.

Earlier completed depth experiment (historical snapshot):

The user authorized a much deeper network only because prior recovery remained
poor. A **17-hidden-layer residual tanh PINN** and a matched **5-layer control**
have now completed 4,000 float64 AdamW updates each. Both score **45/120 individual
gates and 0/12 complete cases** on the exposed cohort, and **55/120 individual
gates and 0/12 complete cases** on a separately predeclared fresh cohort. Every
individual pass/fail classification matches between architectures. Depth did
not improve recovery in this trial; the original/default shallow architecture
is retained. See [the full deeper-model protocol and outcome](DEEP_REGULAR_PINN_PROTOCOL.md)
and [fresh comparison with graphs/parameters](../outputs/regular_pinn_recovery/deep17_fresh_report909131/REPORT.md).
All **82 scoped tests pass**. Both models and both cohorts passed actual-fit
held-out corruption replays with reference-pricer entry points blocked. All
current jobs have finished. The new seed-909131 comparison must not be called
unseen again if it informs subsequent development (truth RNG seed 929131).

Earlier completed stages follow for continuity:

Both completed price-aware factor trials pass **32/120 individual parameters,
0/12 complete cases, 0/12 joint passes**. Both have 12/16,128 invalid reserved
IV predictions; whole-set actual IV RMSE is undefined, not scored after dropping
failures. The unchanged regular baseline remains 48/120 and 0/12 complete cases.
[Seven-stage report and parameter tables](../outputs/affine_factor_pinn/seven_variant_exposed12_report909101/REPORT.md).
Two genuine float64 regular-PINN trials also completed: 1,000 updates followed
by a 5,000-update continuation. Both pass **45/120 individual parameters and
0/12 complete/joint cases**, not an improvement over the 48/120 baseline.
[Latest regular-PINN report, graph and parameter table](../outputs/regular_pinn_recovery/float64_combined_report909111/REPORT.md).
Validation IV RMSE decreased from 0.0003484951 to 0.0003354423, but this did not
translate into reliable recovery. Every actual fit in both trials passed a
held-out corruption replay with six exact-pricer entry points blocked.
At that earlier float64 stage, **75 scoped tests passed** (`testing_float64_final909111/pytest.xml`).
No training jobs remain running at this snapshot. All later work remains local
and unpushed; the earlier verified GitHub push is still commit `3927ad7`.
Later sections preserve the chronological development history, including failures.

## Outcome and scope

The regular Single Heston pricing-PINN workflow has been extended to Double
Heston and actually trained/tested. Accurate, reliable ten-parameter recovery
has **not** been achieved. Do not claim perfect calibration, universal
identification, no possible leakage, or a completed three-PINN comparison.
All new data in this iteration is explicitly synthetic; original NSE records,
PostgreSQL and Kaggle datasets were not changed.

Eight development runs completed: 112,000 optimizer updates in total, excluding
the 200-step runtime pilot. All start from training seed 17 or its checkpoints;
continuations are not independent restarts. Every unsuccessful trial remains.

See [all development results](../outputs/regular_pinn_recovery/all_development_report/REPORT.md)
and [locked assessment results](../outputs/regular_pinn_recovery/locked_assessment_report/REPORT.md).
Each report has plots with brief interpretation and machine-readable parameter
tables. [Protocol](REGULAR_PINN_RECOVERY_PROTOCOL.md) records sequential choices.

## What was implemented

- A shared 5-hidden-layer, 160-unit tanh conditional pricing PINN, for one or two
  variance factors. It predicts a correction to analytic expected integrated
  variance; Black's formula reconstructs option prices. It has no exact Fourier
  evaluation in its forward function or inverse fit.
- The correct one-/two-factor forward-measure pricing PDE, differentiated twice
  in state coordinates; exact terminal payoff, positive implied variance, shape
  penalties, strict factor ordering and Feller-feasible parameter transforms.
- A branchless `-expm1(-kappa*tau)/(kappa*tau)` fixed the MLX compiled
  higher-derivative failure. Raw-price AD and deterministic-variance tests check
  the transformed PDE independently.
- Standard synthetic price/IV-function training versus added Sobolev supervision
  of sensitivities to all 5/10 unit parameter coordinates. AdamW, weight decay
  1e-6, clipped gradient norm 5, fixed learning-rate schedules and saved histories.
- Identical trained weights copied into float64 PyTorch for inverse calibration,
  using bounded multistart SciPy least squares and analytic neural Jacobians.
  The old encoder plus exact-price polish remains a different, hybrid method.

The network is parameter-conditioned during forward training. At inverse use,
unknown parameters are optimized through that frozen network; generating truth
is not supplied to calibration. This is the existing regular Single Heston
workflow, not direct truth-supervised calibration at evaluation time.

## Data, coordinates and constraints

Core reference files are under `outputs/regular_pinn_recovery/{single,double}_data`:
131,072 candidate training quotes and 16,384 validation quotes per family.
Single training has 131,065 usable labels; all rejected candidates remain with
explicit masks. Double training and both validation sets are fully usable.
References use 128-node Fourier prices checked against 96 nodes; implicit Black
inversion gives IV and its parameter sensitivities in float64.

An 18,000-point PDE collocation pool is refreshed every 1,000 updates; 256 points
are used per optimizer step. The pool count is not the cumulative sampling count.
Ordinary anchor minibatches contain 1,024 quotes. Tau is calendar days / 365,
from 7/365 to 2 years. Half the sampled maturities are continuous, half use
30/60/90/180/365/730-day slices. These are synthetic scenarios, not exchange
expiry-day adjustments or a claim that all such stock maturities trade on NSE.

Coordinates are `x=log(F/K), v1[,v2], tau`, with known carry normalized into
the forward price. Canonical stored parameters are `[kappa,theta,sigma,rho,v0]`
per factor. **Factor 1 is slow, factor 2 fast** in this iteration; this is the
reverse naming order of the user's initial fast-first formulation, consistently
canonicalized throughout sampling, pricing, calibration and reporting.
Full numerical bounds are in `regular_pinn_data.DOMAIN` and the protocol.
These finite, Feller-feasible bounds are assumptions, not the entire Heston class.

An additional 1,024 independently sampled training surfaces (129,024 quotes,
seed 907721) supported the grouped-price control and parameter-aware loss trial.
All reference surfaces were numerically usable. A truncated Jacobian inverse
retained rank 10 in 961 surfaces and rank 9 in 63; omitted weak directions were
disclosed, not declared recovered. No validation/test parameters generated these
training surfaces. The parameter-aware loss did not improve actual recovery.

## Results — do not mix these evidence levels

**Development:** four separate fixed recovery surfaces selected checkpoints.
Double Sobolev scaled recovery RMSE improved from 0.163274 (standard) to 0.124257,
about 24%, but both passed 0/4 all-parameter cases. Extra low-LR training, lower
PDE weight and parameter-aware training did not beat that selected checkpoint.
The score is not an ordinary average percentage error: rho errors are scaled
by 0.5, other errors by their true positive values.

**Frozen assessment:** seed 907931, 12 new truths per family, monthly/rich
maturities, clean/1%-time-value-noise conditions, every third strike withheld.
Four frozen models, 192 model-condition fits and 960 starting-point attempts;
these reuse 24 distinct truths, not 192 independent parameter vectors.

| Selected Sobolev model | Clean six-expiry parameter target | All price + parameter gates |
|---|---:|---:|
| Single Heston, own-model truths | 10/12 | 1/12 |
| Double Heston, own-model truths | 0/12 | 0/12 |

Both Double Heston variants passed 0/12 all-parameter cases in each of all four
assessment conditions. In clean three-expiry DH cases, the Sobolev neural price
gate passed 12/12 but the parameter gate passed 0/12. This is precisely why a
close neural price fit cannot establish structural recovery. The strict price
gate is heldout RMSE <=1e-5 of spot; the parameter gate is <=5% for every positive
parameter and <=0.05 absolute for each rho. Keep exact repricing, neural pricing,
optimizer termination and parameter gates separate.

Assessment truths use central unit coordinates [0.1,0.9], approximately 10.7% of
the ten-dimensional unit cube by volume. Boundary robustness, repeated training
seeds, market calibration and a full common-data BS/SH/DH PINN comparison remain
unestablished. The BS rows are an analytic one-volatility price baseline only.

## Validity evidence and diagnosis

- 43 scoped regression tests passed (19 for the new regular pipeline); existing
  TorchScript deprecation warnings on Python 3.14 remain. This is not a claim
  that the complete historical repository suite was rerun.
- All 15 frozen-artifact audit checks passed, including row uniqueness, expected
  counts, recomputed gates, physical constraints, minimum calibration-SSE start
  selection, source/checkpoint hashes and zero exact train/validation/test
  parameter overlaps. 191/192 selected optimizers reported convergence.
- Calibrator tests corrupt heldout values without changing fitted results and
  forbid Fourier calls during fitting. Truth enters only generation and scoring.
- Train/validation IV errors are close: Double Sobolev 0.000336/0.000348. No
  strong memorization signal is observed on this distribution; this is not proof
  against all overfitting or distribution shift.
- Development exact Jacobian condition numbers in parameter-tolerance coordinates
  range about 1,469–15,806. Small surrogate errors project onto weak parameter
  directions and cause large inverse errors. Teacher sensitivity quadrature
  discrepancies were much smaller than the learned-function error.
- Fresh PDE diagnostics are finite, but the full wide-domain residual is not
  near zero. For selected Double Sobolev: near-smile RMSE 0.00660; wide-domain
  RMSE 0.17536. Terminal and identical-weight float32/float64 parity checks pass.
  These diagnostics do not certify a globally solved PDE.

Detailed files: `generalization_stage1.json`, `validation_diagnostics/`,
`physics_stage1_seed907611.json`, `physics_continuation_seed907611.json`,
`physics_grouped_seed907611.json`, and `locked_assessment_report/audit.json`.
The original stage-1 trainer source was recovered by exact SHA256 matching and
is embedded in `generalization_stage1.json`; newer runs have source snapshots.

## Code and safe next work

Core: `src/mentor_dh_pinn/regular_pinn.py`, `regular_pinn_data.py`,
`regular_pinn_torch.py`. Scripts are in `scripts/mentor_dh_pinn/`:
`prepare_regular_pinn.py`, `train_regular_pinn.py`, `assess_regular_pinn.py`,
`prepare_regular_pinn_surfaces.py`, the three diagnostics and the two report
generators. All commands expose `--help`; output directories must be new.

Training requires MLX on Apple Silicon; inverse inference uses identical-weight
PyTorch. Local Python: `/Users/dhruvaambhaikar/Documents/Options pricing/.venv/bin/python`.
Training and assessment entry points must run from the repository root.

Not yet attempted in this regular experiment: network-weight L-BFGS fine-tuning
or a float64 training phase. The inverse SciPy optimizer is not network-weight
L-BFGS. Those are sensible controlled next experiments before another major
architecture change. Extra dropout does not create missing price information
and introduces noise into PDE derivatives. If a factor-structured architecture
is explored, label its difference from this regular pricing-PINN honestly.

Do not tune to assessment seed 907931 and call it unseen again. Choose from the
declared development data, freeze new choices, use fresh assessment truths,
repeat training seeds and include boundary/noise cases. Retain failures and
all start logs. Do not substitute exact Fourier-polished parameters and call
them PINN recovery, nor weaken the already declared gates after seeing results.
No current recovery claim supports calling the project goal complete.

## Next iteration and repository publication

The user has now explicitly requested working toward all ten parameter targets
passing on the previously assessed 12 clean six-expiry cases. Those exposed cases
are development cases from now on; preserve the original assessment and require
fresh cases before another unseen-generalization claim. Do not feed their truth
parameters into initialization, training labels, or the calibration objective.

Float64 PyTorch PDE support and six independent tests have been added as
preparation for deterministic fine-tuning. The expanded scoped suite passes
49 tests. No L-BFGS network training result exists yet. Frozen assessment source
hashes refer to the source version at that assessment; newer PDE-support methods
are a later change, not a rewritten historical result.

The GitHub snapshot includes code, reports, regular PINN checkpoints, selected
hybrid-repair checkpoints, and small repair evaluation inputs. Regenerable
training arrays and local caches remain ignored. To regenerate the regular
reference files on a new checkout, use the existing preparation script with
`--factors 1 --seed 906110` for Single and `--factors 2 --seed 906210` for Double,
the default training/validation sizes, and the corresponding new output directory.
Generate extra grouped surfaces with seed 907721 only if reproducing those trials.

## Latest continuation: deterministic L-BFGS (2026-09-08)

The GitHub snapshot above is published at `3927ad7` on `Double/Single-heston`.
Subsequent L-BFGS experiments are local work, not part of that pushed commit.

The 1,000-iteration deterministic fine-tuning run completed in 562.7 seconds.
It used 16,384 fixed anchors and 18,000 collocation points on every objective
evaluation. Its full-validation IV RMSE improved by about 22%, but its
four-case parameter recovery worsened (0.124257 to 0.201380). The initial
checkpoint remains selected; no recovery improvement is credited. See the
[L-BFGS development report](../outputs/regular_pinn_recovery/lbfgs_development_report908021/REPORT.md).

`finetune_regular_lbfgs.py` now records effective optimizer settings explicitly.
The original pilot/full-run metadata corrections are separate artifacts; their
historical configs and executed source snapshots were not rewritten. Report
code understands iteration-zero selection and distinguishes L-BFGS iterations
from Adam steps. Condition filters in `assess_regular_pinn.py` preserve the
original random draws and label reused cases with `--development`.

Next work is the controlled quadratic training-surface loss trial recorded in
the protocol, not a change to the PINN family or an exact-pricer calibration.
The goal remains all 120 individual parameter gates across the 12 clean
six-expiry development cases, with unchanged thresholds and truth isolation.

The additional grouped-price, quadratic-bias and derivative-consistency L-BFGS
arms also completed (400 iterations each). None selected weights newer than the
starting Double Sobolev checkpoint. Their final four-case parameter scores were
0.141308, 0.220597 and 0.179836 respectively. See
[all L-BFGS results](../outputs/regular_pinn_recovery/lbfgs_all_trials_report908021/REPORT.md).
An exact replay reproduced all 120 exposed-case parameter rows unchanged:
48/120 individual passes, 0/12 complete cases.

The separate `neural_roundtrip908171` diagnostic passed all 12 newly sampled
cases when targets came from the SAME frozen network. It demonstrates numerical
self-consistency, not accuracy against genuine Double Heston reference prices,
and must not be counted toward the user's recovery target. All starts remain
archived, with reference-pricer calls forbidden during that diagnostic.

The user then explicitly approved testing a separately labelled factor-structured
PINN. Follow [its distinct protocol](AFFINE_FACTOR_PINN_PROTOCOL.md). It uses
Riccati coefficient equations rather than the old regular price residual and
requires its own transparent evaluation; do not pool the two architectures as
if they were a matched comparison.

### Factor variant continuation, 2026-09-08 (local, not pushed)

The independent-coefficient 20,000-step Adam model passed 6/40 individual
parameter gates on the original four development cases; 400 full-batch L-BFGS
iterations raised this to 9/40. The derivative-linked architecture learns A
and obtains D=A_tau by automatic differentiation. After 10,000 Adam steps it
passed 10/40 individual gates. **All three stages passed 0/4 complete cases.**
Neither synthetic network self-consistency nor coefficient validation is a
substitute for the required exact-reference recovery.

See the [audited interim report with plot interpretations](../outputs/affine_factor_pinn/development_audit908241/REPORT.md).
It independently checks recorded source snapshots, checkpoint/data hashes,
recovery counts, calibration-start selection and identical case observations
across stages. The 65,536/8,192/18,000 training/validation/collocation states are
finite, unique within splits, and have no exact cross-split state duplicates.
This does not rule out all statistical dependence, overfitting or leakage.

The derivative-linked full-batch L-BFGS continuation is stored under
`outputs/affine_factor_pinn/integrated_lbfgs908241`. Its declared budget is
400 iterations; inspect `complete.json` before calling it complete. Selection
uses coefficient validation only and includes iteration zero. Once complete,
run `assess_affine_factor_pinn.py` with the selected checkpoint and a new output
directory; first use `--case-set development4`, then the requested `exposed12`
development replay. Keep all outcomes, including invalid fits and failed starts.

`diagnose_factor_pinn_bias.py` is a **post-fit diagnostic**, not another
calibrator. Its `integrated_adam_bias908241/diagnostic.json` finds IV bias at
truth between 0.000240 and 0.002129, and exact gate-scaled local condition
numbers about 1,469–15,806. Exact finite-difference Jacobians at two step sizes
agree to relative 3.4e-9–3.1e-8. This supports sensitivity to surrogate error;
it does not prove global non-identifiability. Its linearized bias-cancellation
vectors are NOT fitted parameter estimates and must not enter inverse fitting.

The unchanged 12-case regular-PINN baseline remains 48/120 individual passes
and 0/12 complete cases. No factor-variant 12-case success is established by
the four-case results above. The ten-parameter goal is still unmet.

The derivative-linked full-batch L-BFGS run subsequently finished all 400
iterations in 1181.3 seconds (iteration limit, not optimizer convergence).
It selected coefficient-validation RMSE 0.00261242 but regressed to 6/40
individual passes on development4. Its exposed12 replay passed **19/120
individual parameters, 0/12 complete cases, 0/12 joint gates**. See the
[twelve-case comparison and parameter table](../outputs/affine_factor_pinn/exposed12_comparison_report908241/REPORT.md)
and [all four completed development stages](../outputs/affine_factor_pinn/four_stage_report908241/REPORT.md).
The regular-PINN baseline remains better at 48/120 individual passes.

The comparison audit confirms identical truth vectors, all 60 starting vectors
and holdout masks. Regenerated IVs differ by at most 2.9144e-14 and normalized
spot prices by 3.8858e-16; do not describe the quote files as bit-identical.
Both architectures still fail all complete-case recovery gates.

Next, `ConjugateFactorPINN` additionally ties shifted/unshifted coefficient
branches through the exact input/conjugation identity in the factor protocol.
It still calls only a learned primitive during inference, never the exact
teacher. `conjugate_adam908241` is a 10,000-step continuation with the same
data/targets; inspect its `complete.json` before reporting completion. This
new constraint has tests but is not yet evidence of parameter recovery.
The expanded scoped suite passed 57 tests; JUnit evidence is under
`outputs/affine_factor_pinn/testing_conjugate908241/pytest.xml`.

### September 9 continuation

The completed shift-conjugacy model passed 21/120 individual parameters on
exposed12, 0/12 complete cases and 0/12 joint gates. `MomentFactorPINN` then
enforced the exact first log-CF cumulant (expected integrated variance) by an
O(u²) real primitive correction. Its selected 10,000-step model passed 30/120
individual parameters, still 0/12 complete and joint gates; development4 was
7/40 with no complete cases. See the [three-stage twelve-case report and
per-case parameters](../outputs/affine_factor_pinn/three_variant_exposed12_report909031/REPORT.md).
None of these factor variants has beaten the regular baseline's 48/120.

`TwoMomentFactorPINN` also fixes the second low-frequency cumulant with a finite
analytical coefficient backbone and an O(u³) learned imaginary correction.
The derivation and declared 10,000-step continuation are in the factor protocol.
It is NOT a full characteristic-function formula, and exact reference/pricing
code is still excluded from inverse fitting. Its first/second cumulants remain
correct after changing neural weights, while higher-frequency predictions still
change. The training run is `two_moment_adam908241`; check `complete.json`
before calling it complete, then run the unchanged exposed12 assessment.

The refinements are sequential warm starts, not independent restarts or a
controlled causal ablation. A score change cannot be attributed solely to a
constraint without a corresponding equal-budget control. These remain
development-stage results, not a fair matched Single/Double/Black-Scholes
generalization comparison. No market data or Kaggle dataset was changed in
this continuation. Post-3927ad7 changes remain local and unpushed.

### September 9: two-cumulant assessment and stricter runtime audit

`two_moment_adam908241` completed 10,000 steps in 893.4 seconds and selected
coefficient RMSE 0.00229791. Its unchanged exposed12 assessment scored **27/120
individual parameters, 0/12 complete cases, 0/12 joint gates**. Neural held-out
price gates passed 4/12 and independent exact repricing passed 2/12; all twelve
selected optimizers reported convergence. Numerical convergence and small
pricing errors therefore must not be called successful structural recovery.

The post-fit diagnostic `two_moment_adam_bias909091/diagnostic.json` found the
wrong fitted parameters had lower neural IV loss than truth in all twelve
cases. Neural IV bias at truth ranged from 3.4713e-5 to 0.00138709; exact local
gate-scaled Jacobian condition numbers ranged from 1,361 to 28,852. These are
post-fit diagnostics, not calibration inputs or a proof of global ambiguity.

The actual trained Adam checkpoint also passed
`two_moment_isolation909091/audit.json`: case 0, all five original starts and
budgets, after corrupting all 42 held-out x/tau/IV rows and blocking six known
exact-reference entry points. Every fit/start field matched the archived run
exactly except seconds, and neural weights were unchanged. This is one bounded
runtime isolation test, not universal leakage or unseen-generalization proof.

The 71-test scoped suite passed again; see
`outputs/affine_factor_pinn/testing_full909091/pytest.xml`. The reporting audit
now additionally verifies saved training-source hashes, warm-start hashes and
tensor equality between assessed and validation-selected saved checkpoints.
It rejects deliberate weight replacement, source tampering and split overlap.

`two_moment_lbfgs909091` is the declared 200-iteration fixed-objective
continuation, begun before consuming the completed Adam recovery assessment.
It uses the same coefficient data and physics pool, no recovery truth, and
validation-only selection including iteration zero. Check its `complete.json`
before calling training complete, then assess the selected checkpoint with the
unchanged exposed12 protocol and a new output directory. All failures must
remain in the denominator. The target of 12/12 complete recoveries is unmet.

The two-cumulant L-BFGS continuation subsequently completed all 200 iterations
in 670.4 seconds, selecting coefficient RMSE 0.00183506 at iteration 200. Its
exposed12 assessment passed **30/120 individual parameters, 0/12 complete and
joint gates**, with neural price gates 4/12 and exact repricing 1/12. All selected
calibration optimizers reported convergence; this is still failed recovery.
The [five-stage comparison](../outputs/affine_factor_pinn/five_variant_exposed12_report909101/REPORT.md)
contains all parameters and concise plot interpretations. The
[expiry-level audit](../outputs/affine_factor_pinn/two_moment_lbfgs_repricing909091/REPORT.md)
recomputed every archived price/IV metric exactly in float64, without inverse
optimization. A failed first draft used a float32 replay tensor; that failed
audit attempt is retained, and no underlying assessment values were changed.

### Price-aware forward-function refinement

`finetune_factor_prices.py` adds target-vega-weighted price MSE to the existing
coefficient and Riccati losses. It is a local IV-error approximation, not exact
IV MSE. The same two-cumulant architecture is retained, with the 200-iteration
L-BFGS checkpoint as warm start. Read the predeclared price-aware section of
the factor protocol for exact weights, learning rate, batches and budgets.

The existing `double_surface_train907721/surfaces.npz` supplies 896 training
surfaces (112,896 quotes) and 128 reserved validation surfaces (16,128 quotes).
No candidates were silently removed. These rows were used by earlier regular-
PINN training variants, so the reserved subset is stage-specific validation,
not a globally unseen test. Price-aware selection deliberately uses reserved
weighted-price RMSE, not coefficient RMSE; the initial weights remain eligible.
The reporting audit checks the declared selection metric and rejects exact
recovery-vector overlap with this corpus. The known parameter inputs are
training inputs for the forward function, never revealed recovery answers.

The five-step operational pilot `price_aware_pilot909101` passed its artifact
audit. The full predeclared run is `price_aware_adam909101` (4,000 AdamW steps,
seed 909101), initialized afresh from L-BFGS, not from pilot weights. Inspect
`complete.json` and `selection.json` before claiming a result. Its early saved
validation scores were worse than initialization. If iteration zero wins,
report that no new weights were accepted, not a successful refinement.

The new price-loss gradient/validation-isolation check passed, and all 72
then-scoped tests passed (`testing_price_aware909101/pytest.xml`). A further
selection-metric regression case was added afterwards; rerun the current suite
for the updated count. Corrupting reserved training-validation labels does not
change a training minibatch's value or gradient. All scripts and post-3927ad7
results remain local and unpushed. No NSE, PostgreSQL or Kaggle data was changed.

### Genuine float64 regular-PINN continuation: completed and audited

`finetune_regular_torch.py` reuses `regular_pinn_torch.residual` and the existing
conditional price-PDE architecture. `.pt` checkpoints are loaded without a
float32 MLX round trip. A regression test stores a weight difference of 2^-40
from one and verifies exact preservation; float32 checkpoints falsely labelled
float64 are rejected. Full objective directional derivatives and chunk-weight
normalization pass in float64. Reference-pricer calls are blocked in the
training-loss regression check.

Runs (seed 909111, AdamW, validation-only selection):

| Run | Updates | Selected IV RMSE | Individual checks | Complete cases |
|---|---:|---:|---:|---:|
| Original unchanged baseline | historical | 0.0003484951214 | 48/120 | 0/12 |
| `double_float64_adam909111` | 1,000 | 0.0003410810733 | 45/120 | 0/12 |
| `double_float64_extended909111` | 5,000 additional | 0.0003354423479 | 45/120 | 0/12 |

Both selected the final saved step by minimum full validation IV RMSE, with
step zero eligible. Both neural held-out price gates passed 8/12; exact
repricing passed 0/12. There were no invalid assessed neural IV quotes. The
extended run visited every one of the 131,072 training quotes; the first visited
128,446. Same corpus, not new independent data. Fixed 18,000 collocation points;
512 anchor and 128 physics points per update. The two-step operational pilot
is separate and never used as the main-run warm start.

Selected checkpoint hashes:

- Initial float64 run: `35875776bd0ea23d74a04b4ad6df94ca00fc2ea295b3273df8498d866a1f7cb0`.
- Extended run: `fcb5f3ceaf5b23464e951b9c73c7169c07c4e1afcb33d95b217ae805a3430985`.

`audit_regular_continuation.py` recomputes gates and selection, verifies training
source snapshots, data/checkpoint hashes, exact selected tensor identity and
full validation IV RMSE. Baseline truths, quote rows and all sixty blind starts
are exactly identical. All twelve actual fits in each run were replayed after
setting held-out x to NaN, tau to -1e99 and IV to infinity. Six known exact-pricer
entry points were patched to throw; every fit field except timing remained
exactly identical. This is bounded evidence, not universal proof of no leakage.

Post-fit-only diagnostic `diagnose_factor_pinn_bias.py --regular` was added by
reusing the existing factor diagnostic, not altering inverse fitting. In both
regular trials, **all twelve selected wrong parameter vectors have lower neural
calibration SSE than the generating truth**. For the extended checkpoint, IV
bias at truth ranges 7.53e-5–3.15e-4. Tolerance-scaled reference-Jacobian condition
numbers range 1,361–28,852; all local numerical ranks are ten. Two finite-
difference step sizes agree to at most 1.15e-8 relative Frobenius difference.
This supports surrogate-bias amplification in weak parameter directions; it
does not prove global non-identifiability or make local linear shifts into
calibrated estimates. Diagnostic truth/Jacobians were never fed back to fitting.

Artifacts: `float64_adam_bias909111/diagnostic.json`,
`float64_extended_bias909111/diagnostic.json`, and
`float64_combined_report909111/{REPORT.md,PARAMETERS.md,audit.json}` under
`outputs/regular_pinn_recovery`. The report includes all 120 latest estimates
and concise plot interpretation. The original broad optimizer-state precision
description is clarified in the protocol: parameter moments are float64, while
AdamW's scalar step counter is float32. Historical configs were not overwritten.

Continuation guidance: do not repeat these two Adam budgets and claim a new
experiment, promote these exposed cases to unseen evidence, or select weights
by their recovered truth scores. A useful untested same-architecture hypothesis
is a deterministic float64 neural-weight refinement with fixed training-only
normalization; predeclare its budget and selection before running. The older
L-BFGS experiments used float32 MLX values/gradients, and remain negative
results. No such float64 deterministic continuation has been started here.
