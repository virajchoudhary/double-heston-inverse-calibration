You are working on my existing Double Heston inverse-calibration repository.

Current project status:

- A deeper 17-layer Double Heston PINN achieved excellent pricing accuracy.
- Pricing error is already much lower than Single Heston.
- Current regularized neural-only calibration results on fresh Double Heston surfaces are approximately:
  - Clean parameter RMSE: 0.217 → 0.153
  - 1% noisy parameter RMSE: 0.608 → 0.388
  - Clean all-10-parameter recovery: only 1/32
  - Noisy all-10-parameter recovery: 0/32
- Parameter tolerances:
  - Positive parameters: within 5% relative error
  - Correlation parameters: within 0.05 absolute error
- Exact-pricer refinement during calibration is NOT allowed.
- Calibration must remain neural-only.
- Exact pricing may only be used independently for generating truth/evaluation and must never modify calibrated parameters.
- Do not optimize only for pricing RMSE. The primary objective is accurate recovery of the true Double Heston parameters.

# MAIN OBJECTIVE

Improve Double Heston parameter recovery substantially, especially the all-10-parameter recovery rate.

The core issue appears to be that many different parameter combinations generate almost identical option surfaces. Therefore, further increasing network depth alone is unlikely to solve the problem.

Treat this primarily as an:

1. identifiability problem,
2. inverse-problem conditioning problem,
3. parameter-symmetry problem,
4. calibration initialization problem,
5. loss-design problem,
6. noisy-observation robustness problem.

Implement and experimentally test the following improvements.

---

# PHASE 1 — DIAGNOSE WHY EACH PARAMETER IS FAILING

Before changing the model again, perform a detailed parameter-recovery diagnostic.

For every Double Heston parameter, calculate:

- mean relative/absolute error,
- median error,
- RMSE,
- 90th percentile error,
- pass percentage,
- bias,
- error standard deviation,
- correlation between errors of different parameters.

Generate a table similar to:

Parameter | Mean error | Median | RMSE | Pass % | Bias | Error SD

Determine exactly which parameters are responsible for most all-ten failures.

Also calculate the correlation matrix of parameter-estimation errors.

Identify parameter pairs where one parameter increases while another decreases with almost no change in option prices.

This is extremely important.

Do NOT continue treating all ten parameters as equally identifiable if the data shows otherwise.

---

# PHASE 2 — CHECK DOUBLE HESTON FACTOR-SWAPPING SYMMETRY

This must be investigated before concluding that parameter recovery is poor.

Double Heston contains two stochastic-volatility factors.

Check whether exchanging factor 1 and factor 2 produces the same or nearly identical option surface.

For example, determine whether:

(kappa1, theta1, sigma1, rho1, v01)
↔
(kappa2, theta2, sigma2, rho2, v02)

represents an economically equivalent solution.

If factor swapping is mathematically equivalent or nearly equivalent, our current parameter-recovery metric may incorrectly classify a correct calibration as wrong.

Implement BOTH:

1. permutation-invariant evaluation:
   compare prediction to the original truth and factor-swapped truth and use the lower error,

AND

2. canonical factor ordering.

For example, enforce a deterministic ordering such as:

theta1 <= theta2

or

kappa1 >= kappa2

or whichever ordering is most mathematically appropriate and stable.

Apply the SAME canonicalization:

- when generating training data,
- to training labels,
- to calibration outputs,
- to ground-truth parameters,
- during evaluation.

Report how much of the current recovery problem was caused purely by label symmetry.

This experiment has very high priority.

---

# PHASE 3 — PARAMETER SENSITIVITY / IDENTIFIABILITY ANALYSIS

For each synthetic surface calculate the Jacobian:

J = d(option surface) / d(parameters)

using all calibration quotes.

Use the Jacobian to calculate:

Jᵀ W J

and analyze:

- singular values,
- condition number,
- eigenvalues,
- approximate Fisher information,
- sensitivity of prices to each parameter.

Determine which parameters have almost no independent information in the current strike/maturity grid.

Create sensitivity plots/tables by:

- strike,
- maturity,
- parameter.

Identify which quotes are most useful for recovering each individual parameter.

For example, determine whether:

- short maturities help v0,
- long maturities help theta,
- wings help sigma/rho,
- ATM quotes help other parameters.

Do not assume this. Measure it.

---

# PHASE 4 — OPTIMIZE THE QUOTE GRID FOR PARAMETER RECOVERY

The current grid uses many option quotes, but more quotes do not necessarily mean more independent information.

Use the sensitivity/Jacobian analysis to select or weight quotes that maximize parameter information.

Experiment with:

- D-optimal design,
- A-optimal design,
- determinant of Fisher information,
- smallest singular value maximization,
- condition-number minimization,
- greedy sensitivity-based quote selection.

Instead of simply treating all 84 calibration quotes equally, create parameter-information-aware weights.

Quotes that contribute unique parameter information should receive greater importance.

Evaluate:

1. original uniform weighting,
2. sensitivity weighting,
3. Fisher-information weighting,
4. optimized quote subset,
5. optimized quote weights.

Do not select weights using the final assessment data.

---

# PHASE 5 — REPARAMETERIZE THE DOUBLE HESTON PARAMETERS

The optimizer should not work directly in poorly scaled physical parameter space.

Create unconstrained latent variables z and transform them to physical parameters.

Use appropriate transforms such as:

positive parameters:
    softplus(z) + epsilon

bounded parameters:
    lower + (upper-lower) * sigmoid(z)

correlations:
    rho = tanh(z)

If training has known parameter bounds, use those exact bounds.

Normalize every parameter so that optimization occurs on comparable scales.

Also investigate physically meaningful transformed variables that reduce coupling, such as:

- long-run variance,
- mean-reversion timescale = 1/kappa,
- vol-of-vol ratios,
- Feller-related quantities,
- total initial variance,
- relative contribution of factor 1 vs factor 2.

Try parameterizations that make the two volatility factors easier to distinguish.

---

# PHASE 6 — STAGED / HIERARCHICAL CALIBRATION

Do not immediately optimize all ten parameters simultaneously.

The current all-at-once optimization may create a badly conditioned inverse problem.

Use the sensitivity analysis to divide parameters into strongly and weakly identifiable groups.

Experiment with staged optimization such as:

Stage 1:
recover parameters controlling broad volatility level.

Stage 2:
recover mean-reversion / long-run parameters.

Stage 3:
recover vol-of-vol and correlations.

Stage 4:
jointly optimize all ten parameters starting from the staged solution.

Do not hard-code the groups without evidence.

Use sensitivity and parameter-error correlation results to choose them.

Compare:

- joint-only optimization,
- staged → joint optimization,
- block-coordinate optimization.

---

# PHASE 7 — IMPROVE INITIALIZATION

Five generic random starts are probably insufficient for a ten-dimensional multimodal inverse problem.

Create a learned parameter initializer.

Train a separate inverse network:

option surface → approximate Double Heston parameters

The inverse network does NOT replace calibration.

Its purpose is only to produce high-quality starting points for the existing neural pricing-surrogate optimizer.

Generate multiple starts around the inverse-network prediction.

For example:

Start 1:
inverse-network prediction

Starts 2–5:
small perturbations around prediction

Additional starts:
diverse candidates selected using parameter-space distance.

Compare against the current random/multistart strategy.

Measure:

- initial parameter error,
- final parameter error,
- convergence rate,
- percentage reaching good basins.

---

# PHASE 8 — TRAIN DIRECTLY FOR RECOVERY SENSITIVITY

Pure price MSE does not sufficiently penalize pricing errors in directions that cause large parameter errors.

Implement a recovery-aware surrogate loss.

The loss should include ordinary pricing error PLUS a sensitivity-weighted component.

Conceptually:

L =
L_price
+ lambda1 * L_derivative
+ lambda2 * L_parameter_sensitive
+ lambda3 * L_physics

Use the local inverse sensitivity:

(Jᵀ W J + λI)^(-1) Jᵀ W

to estimate how pricing errors translate into parameter errors.

Penalize surrogate pricing errors more strongly when they correspond to large implied parameter shifts.

Improve the existing local recovery-sensitive loss rather than simply increasing its weight.

Tune lambda values ONLY on a development dataset.

---

# PHASE 9 — PARAMETER-SPECIFIC LOSS NORMALIZATION

Do not use one raw parameter RMSE across parameters with different physical scales.

For positive parameters use normalized error such as:

(pred - true) / characteristic_scale

or log-space error:

log(pred / true)

For correlations use ordinary absolute error.

Construct a recovery objective that approximates the actual pass criterion:

positive parameters:
5% relative tolerance

correlations:
0.05 absolute tolerance

Investigate smooth approximations to the pass criterion so training emphasizes parameters close to failing the tolerance gate.

For example, use a soft threshold loss that strongly penalizes errors approaching/exceeding the tolerance.

---

# PHASE 10 — NOISE-AWARE CALIBRATION

Current noisy recovery improved in RMSE but still gives 0/32 full passes.

Implement proper heteroscedastic noise handling.

Noise is 1% of option time value.

Therefore do not assume identical uncertainty for every quote.

Use:

residual_i / sigma_i

where sigma_i reflects the known quote uncertainty.

Combine this with the surrogate covariance model already implemented.

Test:

- ordinary least squares,
- GLS,
- heteroscedastic weighted least squares,
- GLS + heteroscedastic observation noise,
- robust loss such as Huber where justified.

Do not tune using final assessment data.

---

# PHASE 11 — ADAPTIVE REGULARIZATION

The current prior helps average RMSE but may bias difficult/boundary parameters.

Replace one fixed regularization strength with adaptive regularization.

Use parameter identifiability to determine prior strength.

Strongly identified parameter:
weak regularization.

Weakly identified parameter:
stronger regularization.

Possible form:

lambda_j ∝ 1 / information_j

where information_j comes from the Fisher/Jacobian analysis.

Test whether this improves noisy recovery without damaging clean/boundary recovery.

---

# PHASE 12 — BOUNDARY PERFORMANCE

Current boundary performance is poor.

Investigate why.

Increase training density near parameter boundaries without contaminating the final assessment.

Use a mixture distribution such as:

- majority interior samples,
- dedicated boundary samples,
- near-Feller-condition cases where relevant.

Do not only generate uniformly random parameters.

Include hard examples intentionally.

Measure boundary recovery separately.

---

# PHASE 13 — CURRICULUM TRAINING

Try curriculum training.

Begin with easier parameter regions where the inverse mapping is well-conditioned.

Gradually introduce:

1. interior clean cases,
2. difficult interior cases,
3. boundary cases,
4. noisy cases.

Do not immediately train equally on all difficulty levels if this destabilizes learning.

---

# PHASE 14 — HARD-EXAMPLE MINING

After each development evaluation, identify surfaces with the worst parameter recovery.

Determine why they fail:

- symmetry,
- low Jacobian rank,
- boundary parameter,
- correlated parameters,
- bad initialization,
- surrogate error.

Generate additional TRAINING cases similar to these failure regions.

Never add final-assessment examples to training.

---

# PHASE 15 — MULTI-OBJECTIVE CHECKPOINT SELECTION

Do NOT select the network based only on pricing validation loss.

Create a development score based on parameter recovery.

For example include:

- parameter RMSE,
- median normalized parameter error,
- all-ten pass rate,
- worst-parameter error,
- clean/noisy performance.

Pricing accuracy should remain a constraint, but parameter recovery should determine checkpoint selection.

---

# STRICT DATA SPLITTING

Use completely separate datasets:

TRAINING
for network optimization.

DEVELOPMENT
for:
- hyperparameter tuning,
- prior selection,
- quote weighting,
- checkpoint selection,
- architecture decisions.

FINAL ASSESSMENT
fresh random seed never used for model decisions.

STRESS ASSESSMENT
fresh boundary and noisy cases.

Never retrain based on final-assessment outcomes.

---

# REQUIRED BASELINES

Compare every final method against:

1. Single Heston PINN
2. Original Double Heston PINN
3. current 17-layer Double Heston PINN
4. current GLS Double Heston
5. current regularized Double Heston
6. new improved method

Use exactly the same fresh Double Heston truths where comparisons require paired testing.

---

# REQUIRED METRICS

For clean, 1% noise, and boundary cases report:

### Parameter metrics

- overall parameter RMSE
- median normalized error
- each parameter's RMSE
- each parameter's median error
- each parameter's pass rate
- all-10 pass rate
- 9/10 pass rate
- 8/10 pass rate

The 9/10 and 8/10 results are important because all-ten is an extremely strict multiplicative gate.

### Pricing metrics

- calibration quote RMSE
- heldout quote RMSE
- exact post-fit repricing RMSE for evaluation only

### Statistical metrics

- paired wins
- bootstrap 95% confidence intervals
- mean improvement
- median improvement

### Conditioning metrics

- Jacobian condition number
- smallest singular value
- effective rank
- Fisher-information eigenvalues.

---

# ADD AN ERROR-DISTRIBUTION TABLE

For every final method produce:

Number of parameters passing tolerance:

10/10
9/10
8/10
7/10
<=6/10

This will tell us whether calibrations are completely wrong or usually missing only one/two difficult parameters.

---

# VERY IMPORTANT EXPERIMENT

After implementing factor-symmetry handling, recalculate ALL previous results.

It is possible that:

0/32 or 1/32

is artificially low because the optimizer correctly recovered the two volatility factors but in reversed order.

Do this BEFORE spending large compute on additional network training.

---

# ITERATION STRATEGY

Do not implement everything blindly and then report one result.

Follow this order:

1. factor-symmetry test
2. parameter-wise error diagnostic
3. Jacobian/Fisher identifiability analysis
4. reparameterization
5. information-aware quote weighting
6. staged calibration
7. learned initialization
8. adaptive regularization
9. recovery-sensitive training
10. noise robustness
11. boundary-focused training

After each major change, run a development experiment.

Keep a table:

Method | Clean RMSE | Clean 10/10 | Noise RMSE | Noise 10/10 | Boundary RMSE

Discard changes that consistently worsen parameter recovery.

Combine only complementary improvements.

---

# SUCCESS CRITERIA

The primary objective is NOT another 1–5% pricing improvement.

Pricing is already sufficiently accurate.

The desired outcome is a major improvement in physical parameter recovery.

Aim for:

Clean:
- substantial reduction from current RMSE ~0.153
- dramatically higher than 1/32 all-ten recovery

1% noise:
- substantial reduction from current RMSE ~0.388
- achieve non-zero all-ten recovery if the problem is identifiable

Boundary:
- measurable improvement over the current model.

A particularly strong result would be:

Clean all-ten recovery > 50%

and

Noisy all-ten recovery > 20%

BUT DO NOT fabricate, overfit, leak assessment information, or manipulate the test distribution to achieve these targets.

If these targets are mathematically unrealistic because Double Heston parameters are structurally non-identifiable from this quote grid, demonstrate that scientifically using:

- Jacobian rank,
- Fisher information,
- condition numbers,
- profile likelihood,
- parameter trade-off examples,
- near-identical surfaces from substantially different parameter vectors.

If structural non-identifiability is demonstrated, then redefine and report the scientifically valid target, such as recovery of identifiable parameter combinations rather than pretending all ten parameters are recoverable.

---

# TESTING REQUIREMENTS

Add automated tests for every new component:

- canonical parameter ordering,
- factor permutation matching,
- transformed parameter bounds,
- Jacobian correctness via finite differences,
- quote weighting,
- staged calibration,
- adaptive regularization,
- inverse initializer,
- no exact-pricer calls during calibration,
- deterministic seeded evaluation,
- no train/development/assessment overlap.

All existing tests must continue to pass.

---

# OUTPUT FILES

Create a new results directory such as:

outputs/deeper_pinn/identifiability_recovery_v2/

Save:

REPORT.md
SUMMARY.json
parameter_errors.csv
parameter_pass_rates.csv
conditioning.csv
paired_comparison.csv
bootstrap_intervals.csv
clean_results.json
noise_results.json
boundary_results.json

and charts for:

- parameter error by parameter,
- predicted vs true parameter,
- sensitivity by maturity,
- sensitivity by strike,
- singular-value spectrum,
- parameter-error correlation,
- clean/noisy comparison,
- all-parameter recovery distribution.

---

# FINAL REPORT

At the end tell me clearly:

1. What was actually causing poor parameter recovery?
2. Was factor swapping affecting the all-ten metric?
3. Which parameters are hardest to recover?
4. Why are they difficult?
5. Which implementation change helped the most?
6. Best clean parameter RMSE.
7. Best noisy parameter RMSE.
8. Clean 10/10 pass rate.
9. Noisy 10/10 pass rate.
10. Boundary pass rate.
11. Per-parameter pass rates.
12. Price RMSE.
13. Whether improvement is statistically significant.
14. Whether all ten Double Heston parameters are identifiable from our current option grid.
15. Exact files changed.
16. Exact commands needed to reproduce the result.

Most importantly:

DO NOT stop merely because pricing improves.

Continue experimenting on the development set until the major parameter-recovery approaches above have been tested.

Only call a method "better" when it improves parameter recovery on completely fresh assessment cases.

Do not hide negative results.

Preserve every important experiment so the final research report can explain both what worked and what failed.