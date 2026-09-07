# Double Heston calibration repair — 5 September 2026

Starting revision: b24c642a35bb9978221fb8b790196085ac8383fe.
This protocol is written before measuring the repaired models. The historical
test results have already influenced development and are not a fresh confirmatory
test. No market forecasting or Phase 3B experiment is authorised by this protocol.

## Questions and scope

1. Does corrected refinement decrease its actual per-surface calibration objective?
2. Does the repaired uncertainty loss resist artificial covariance collapse in
   genuinely uninformative directions, while permitting contraction when informed?
3. Can all ten ordered parameters be recovered on new, noiseless synthetic
   multi-expiry surfaces? Separately, how accurately are prices reconstructed?
4. What changes under noise and short/single-expiry geometry?
5. Do three independently seeded fine-tunes improve development validation without
   losing synthetic recovery performance? Report every seed, including regressions.

No true Double Heston parameter labels exist for NSE quotes. Model calibration is
a projection onto a restricted model class. Numerical sensitivity thresholds are
local practical diagnostics, not proofs that parameters are structurally absent.

## Fixed choices

- Reuse the committed unified.pt encoder, exact Fourier engine and strict ordering,
  positivity, Feller and repository correlation-disk constraints.
- Repair the uncertainty loss with expected negative log likelihood plus analytic
  Gaussian KL to the fixed training-prior approximation. The combined supervised,
  projection and ELBO loss is a training objective, not an exact Bayesian posterior.
- Inference uses explicit quote uncertainty when supplied, with objective-accepted
  steps. Existing synthetic relative-noise weighting is the fallback.
- Fresh synthetic training: 20,000 candidate surfaces, generator seed 905001;
  development validation: 2,000, seed 905002. Float64 saved geometry and prices,
  64-node quadrature. Keep the base training-only latent standardisation unchanged.
- Fine-tune seeds 17, 29, 43; same starting checkpoint, data, architecture and budget.
  Initial budget: 600 AdamW steps, learning rate 5e-5, 150-step warmup, cosine decay,
  batches 48 synthetic and 16 real, four Monte Carlo draws. Save all validation
  checkpoints and an initial checkpoint, along with configuration and source hashes.
- Select each seed using pooled quote squared error in vega units on development
  validation; report both pre-refinement and deployed refined performance. Selection
  must explicitly use the deployed rule where validation quote folds are available.
  Synthetic z-MAE may worsen by at most 5% and marginal 90% coverage must remain
  between 0.85 and 0.95 for a candidate to replace the initial checkpoint.
- No test-based hyperparameter or seed selection. Previously inspected test files
  are excluded from training and selection, and are labelled historical development
  evidence if subsequently displayed. Market calendar overlap is audited across
  symbols; base checkpoint history remains a limitation even after a new embargo.
- Numerical failures remain in denominators and are recorded. Any failed prediction
  makes the all-quote RMSE unavailable/infinite; successful-subset errors are labelled.

## Final recovery assessment

Use independent synthetic seeds 905101, 905102 and 905103, generated only after code
and training choices are frozen. Each supplies eight independently drawn parameter
sets, examined under a rich maturity ladder and a single expiry, clean and noisy.
Report all ten parameters, standardised latent errors, canonical physical errors,
price error, exact-IV error where inversion is valid, and local weighted-Jacobian
singular values. Keep input calibration quotes separate from quote holdouts.
Truth labels are available only to the scorer, never the calibrator or selector.

For noise-free multi-expiry recovery, the numerical target is maximum scaled latent
error <= 0.01 and price RMSE <= 1e-6 of spot. These are declared targets, not promised
outcomes. Compare encoder, normal refinement and an explicitly labelled exact
optimiser polish with a fixed budget; optimisation time is included. A polished
hybrid is not described as network-only recovery. No accuracy gate is loosened
after seeing results.

For real data, report pooled quote RMSE, surface median and per-symbol breakdown,
and identify the data source, settlement convention, date ranges, expiry counts,
quote exclusion counts and unresolved preprocessing assumptions. No repeated-test
p-values, universal superiority claims or '100% perfect' parameter claims.
