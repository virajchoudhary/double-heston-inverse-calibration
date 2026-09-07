# Double Heston calibration repair and continuation context

Starting code: `b24c642`, branch `Double/Single-heston`. Repairs live on
`codex/dh-calibration-repair-20260905`. This is a persistent checkout; the former
temporary research checkout no longer exists. Previous result files are preserved.

## What the current model actually does

An unordered set of European option quotes enters a Set-Transformer. It predicts
ten latent coordinates and a Cholesky factor for an approximate Gaussian. The
coordinates decode to the ordered Double Heston parameters:

`kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow,
kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast`.

Ordering, positive parameters, both Feller conditions and the repository's joint
correlation-disk convention remain enforced. The network is physics-informed by
the differentiable characteristic-function pricing engine. This iteration does
not train the old four-coordinate forward PDE PINN or create a new collocation
pool. Synthetic surfaces/quotes must not be described as PDE collocation points.

The `refine` method fits the same supplied quotes with damped Gauss–Newton steps.
It now uses explicit price-unit `quote_sigma` where available, tests the complete
per-surface objective, backtracks, and accepts only reductions. An invalid active
quote freezes that surface; it is not removed from the objective. The output `L`
describes uncertainty around the encoder's `mu_z`, not around refined parameters.
`local_identifiability` returns the scaled weighted-Jacobian SVD and weak bases.
These are local practical sensitivity diagnostics, not proofs of global uniqueness.

`precise_calibration.exact_polish` optionally continues optimisation with analytic
derivatives and three initialisations. This is an optimiser-assisted hybrid, not
network-only parameter recovery. It receives calibration prices and an initial
latent estimate; it has no argument for true parameters or holdout prices.

## Loss repair

The old variance of log-evidence ratios did not provide the claimed barrier
against covariance collapse. It has been replaced by

`E_q[0.5 * sum((model_price - quote_price)^2 / quote_sigma^2)] + KL(q || prior)`.

The Gaussian KL is analytic and contains the entropy term `-log det L`. On an
uninformative likelihood, its optimum covariance is the prior covariance: narrow
estimates widen, overly broad estimates shrink. Requiring a negative covariance
gradient in every case would itself be wrong. The actual synthetic prior is a
mixture; the Gaussian prior used here approximates its latent training moments.
The combined anchor/projection/ELBO loss with moving scale weights is not exact
Bayesian posterior inference. A Gaussian does not describe every nonlinear ridge.

Rejected nonfinite losses no longer poison the running loss scales. Disabled
objectives are not executed. Invalid predictions cannot earn a zero residual.

## Data and selection

Fresh synthetic generation produced 19,904 usable training surfaces with 595,822
quotes and 1,996 validation surfaces with 60,579 quotes. There were 96/4 rejected
candidate surfaces; they are recorded rather than relabelled as valid. The saved
geometry and prices use float64. Train and validation parameter sets have no
exact overlap. These are explicitly synthetic prices, not downloaded NSE records.

The rebuilt real corpus contains 1,219 training surfaces/25,219 quotes and 203
validation surfaces/4,429 quotes, with 1,607 quote holdouts. Sources are the shipped
clean stock panel and 70 original local NSE bhavcopies. NIFTY uses settlement
prices, individual-quote open interest >=10,000, actual expiry dates, and carry
estimated from calibration strikes before quote holds are scored. Validation
noise and weighting curves use calibration IV values only. Stock parity anchor
keys for holdouts were verified empty. Source and output SHA-256 hashes, removed
dates and quote caps are in `outputs/calibration_repair/real_corpus/manifest.json`.

All new training/validation/reserved-test calendar intersections are empty, across
symbols. NIFTY train dates span 2024-07-08–2026-01-27; validation spans
2026-02-03–2026-03-24. Strict liquidity leaves 3–8 training expiries and 3–7 validation
expiries. Ten stocks remain; ADANIPOWER has no training surfaces after the embargo.
The 100-quote stratified cap retains expiry/fold coverage but removes 12,815/2,408
input quotes in train/validation; those counts are not missing-data fabrication.

The historical market evaluations were repeatedly inspected and the old fine-tune
had already seen some newly assigned NIFTY validation dates. New masks cannot erase
that history. All market results in this repair are development evidence. There is
no new forward test, untouched-market generalisation claim, or Phase 3B experiment.

Three fine-tunes start from the same base `unified.pt`, with seeds 17/29/43 and
600-step budgets. Checkpoint selection uses pooled **heldout vega-scaled price**
RMSE after three refinement steps, plus synthetic recovery/coverage gates. That
quantity approximates IV error; the separate final evaluator inverts actual IV.
Every scheduled checkpoint and every seed is saved. No seed is chosen using test
results. The assessed primary is selected solely by development validation.

## Reproduce

Run from the repository root with Python containing the dependencies in
`requirements.txt`. The original machine used Python 3.14.5, PyTorch 2.13.0,
NumPy 2.4.6, SciPy 1.17.1 and pandas 3.0.3; every run saves runtime and hashes.

```sh
python scripts/mentor_dh_pinn/prepare_repair_synthetic.py
python scripts/mentor_dh_pinn/build_real_corpus.py --bhav-dir /path/to/nse_fo_bhavcopies
python scripts/mentor_dh_pinn/run_finetune.py --ckpt outputs/unified_v6/unified.pt --syn outputs/calibration_repair/synthetic --real outputs/calibration_repair/real_corpus --out outputs/calibration_repair/new_seed_17 --tag repaired --seed 17 --steps 600 --batch-syn 48 --batch-real 16 --val-every 100 --threads 1
python -m pytest tests/test_projection_finetune.py tests/test_refinement_objective.py tests/test_unified_calibrator.py tests/test_real_corpus_integrity.py tests/test_precise_calibration.py tests/test_repair_selection.py -q
```

Use new output directories for new experiments; training and assessment refuse
to overwrite an existing run manifest. The full original bhavcopy tree is external
data, explicitly supplied on the command line. Inference from shipped checkpoints
does not need it. Historical evaluator paths are not the recommended repair workflow.

## Reading results honestly

Completed results are in [RESULTS.md](../outputs/calibration_repair/RESULTS.md), with
four interpreted figures and the executable
[results notebook](../outputs/calibration_repair/Double_Heston_Repair_Results.ipynb).
Use actual artefacts, not preliminary chat measurements. The recovery
assessment shows clean/noisy rich and single-expiry geometry; true parameters are
used only by generation and scoring. Price and parameter accuracy have separate
fixed gates, and quadrature, no-arbitrage and failed-estimate checks accompany them.

No real NSE quote supplies a true ten-parameter label. Even synthetic prices can
have nearly equivalent parameter solutions on a restricted geometry. Small price
errors, a passing PDE/pricer test, or a local optimiser success flag cannot alone
establish perfect parameter recovery. Changes to future experiment choices after
examining assessment outcomes require a fresh assessment rather than relabelling
the same test. The unresolved POWERGRID mentor decision is outside this work.

## Measured outcome and remaining limits

All three 600-step runs completed. Development selection chose seed 43 at step 500;
the other seeds and every scheduled checkpoint are retained. Selected primary:
`outputs/calibration_repair/seed_43/repaired.pt`, SHA-256
`f6e502c0ca0f581cabe3bce6d076562a959eeafa9a56d5356a0d3bcd3347da8d`.
The approximate price/vega error improves about 22%, but actual all-quote IV RMSE
improves only 1.20% (0.0356952 to 0.0352670). NIFTY and JSWENERGY regress.
No seed was reselected using this actual-IV or final synthetic assessment.

The final synthetic assessment has 24 truths, 96 geometry/noise conditions and
864 model-arm records. Primary hybrid clean-rich recovery passes both gates in
21/24 cases; price alone passes 23/24. Clean single-expiry prices pass 24/24 while
all-parameter recovery passes 0/24. Noisy cases fail the strict parameter target;
unregularised polishing can overfit. Network-only recovery passes 0/24 in every
condition. Two noisy-rich hybrid outputs fail 96/128-node convergence; they remain
in the data and fail their price gate. Perfect calibration has not been achieved.

Three original tiny negative validation prices were traced to float64 cancellation.
The original evaluation is preserved in `market_validation`. An identical numerical
guard across all four models reevaluated seven near-zero prices using independent
Carr–Madan inversion at 40/60 decimal precision, with tail checks. All 6,428 quote
rows remain; no market price or model parameter changed. The checked evaluation
is in `market_precision_checked`; a failed early postprocessing attempt is retained
in `market_precision_attempt_1` (pandas dtype error, not a different model run).
The numerical guard was introduced after observing the failures and is not claimed
as preregistered. Final scoped tests: 46 passed, 18 deprecation warnings.

To reproduce the frozen assessment, use `assess_calibration_repair.py --help` with
all four checkpoints, a new output directory, seeds 905101/905102/905103, eight
cases per seed, and `--polish-model seed_43/repaired`. This reruns existing assessment
evidence, not a new untouched test. The recorded manifests specify all settings.
`evaluate_repair_validation.py` reproduces the development quote evaluation;
`diagnose_tail_precision.py` performs its separate numerical check; then
`report_calibration_repair.py` rebuilds the figures and notebook. The precision
script refuses to overwrite its existing output directory; preserve previous
results before an intentional rerun. The report notebook only displays results.
