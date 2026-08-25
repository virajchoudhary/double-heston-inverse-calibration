# G8 Final Untouched Real-Market Protocol

Status: **FROZEN_PENDING_UNTOUCHED_DATA_ACQUISITION** — 25 August 2026.
Machine-readable twin: `configs/g8_final_real_market.yaml`.
Pre-execution audit: `docs/G8_PREEXECUTION_AUDIT.md`.

This document is committed before any G8 market observation is acquired or inspected. It separates two comparisons:

1. **Pricing-model family:** Black-Scholes vs standard Heston vs Double Heston.
2. **Inverse-calibration method:** traditional optimization vs ordinary ANN (Model 1) vs constraint/repricing-informed network (Model 2) vs optional future PDE-informed Model 3.

A finding about one comparison never substitutes for the other. Neither Double Heston nor Model 3 is given a preferred prior.

## 1. Non-negotiable boundaries

```text
DATA_ACQUISITION_NOW = FALSE
G8_MODEL_OUTPUTS_EXIST = FALSE
NEURAL_WEIGHT_UPDATES_ON_G8 = FALSE
ARCHITECTURE_OR_HYPERPARAMETER_TUNING_ON_G8 = FALSE
NTPC_2026-07-15 = DEVELOPMENT_PILOT
FIVE_JULY_NTPC_DATES = DEVELOPMENT
R2_REPRESENTATION = UNCHANGED
PRODUCTION_DH_PRICER = UNCHANGED
CANONICAL_PARAMETER_ORDER = UNCHANGED
PRACTICAL_NON_IDENTIFIABILITY = RETAINED
```

No G8 archive may be downloaded until all participating inverse methods are frozen. If Model 3 is not frozen by then, G8 reports `MODEL3_NOT_FROZEN_NOT_EVALUATED`; it does not wait indefinitely and does not add Model 3 post hoc under a new design. Adding it later requires a new protocol revision and a different untouched panel.

## 2. Eligible universe and deterministic dates

Primary symbols remain the reviewed sector primaries: **NTPC, CIPLA, INFY, and HDFCBANK**. Sector backups in fixed order are POWERGRID, SUNPHARMA, TCS, and ICICIBANK. A primary is permanently unavailable only after the complete 2026-09-30 through 2026-12-31 structural scan finds zero eligible surfaces for it. That declaration, the fixed backup choice, and a fresh deterministic scan must occur before any G8 model output. A support failure on one date creates no substitution.

Starting **2026-09-30**, scan official NSE trading dates in ascending order through 2026-12-31. Select the first **two** dates on which **all four** current primaries pass structural eligibility. This yields up to eight symbol-date surfaces. Record every scanned failure and reason. Fewer than two common dates resolves `G8_BLOCKED_INSUFFICIENT_ELIGIBLE_DATES`; no outcome-based date change is permitted.

All five NTPC development dates, all Stage-A touched symbol/date observations, NTPC realized-close inputs through 28 July 2026, and NIFTY are outside G8 as specified in the config. The 2026-09-30 floor prevents reuse of the July pilot's completed expiries.

## 3. Source and acquisition gate

Use only official NSE UDiFF daily bhavcopy ZIP files:

```text
BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
```

For every archive, retain immutable bytes plus official URL, UTC retrieval timestamp, filename, size, ZIP SHA-256/integrity, member name, extracted CSV SHA-256, encoding, delimiter, and trading date. Third-party mirrors and Bloomberg are prohibited. Bid/ask fields absent from UDiFF are never inferred. Raw bytes stay in an ignored/local immutable store; only hashes and validated manifests enter Git.

The latest official RBI 91-day T-bill auction result on or before each valuation date supplies `y`. Its HTML/PDF and normalized extract must be acquired before evaluation, then hash-sealed and field-validated. Future rates are forbidden; an older July artifact may not be silently reused when a later eligible observation exists.

## 4. Surface eligibility and cleaning

Construct the unchanged R2 representation: first two listed stock-option expiries having one active matched future, central log-moneyness targets `[-0.10,-0.05,0,+0.05,+0.10]`, calls and puts, 20 nominal slots, and actual `T=DTE/365`.

Rules:

- Exactly one positive-official CM equity close per symbol/date; it must equal the corresponding F&O `UndrlygPric` exactly.
- A matching stock future must exist for each selected expiry with positive close, traded quantity, open interest, and executed-trade count.
- Option candidates use `ClsPric`, the same four positive activity fields, finite Black IV on the matched-futures forward, `abs(log(K/S)) <= 0.10+1e-12`, and actual listed strikes only.
- For each expiry/type, Hungarian assignment minimizes absolute distance to the target set with a strict `0.05` gate. Ties break by strike then `FinInstrmId`. Failed matches become explicit masks with reasons.
- A usable surface needs at least 12 valid slots overall, including at least 6 pricing-family calibration slots and 3 holdout slots, at least one valid slot in each expiry rank, and at least one call and put.
- Never interpolate, extrapolate, clip, arbitrage-adjust, fill, or substitute a quote. Impossible prices fail IV eligibility; every nonselected raw row receives a rejection reason.

Carry is `q=r-log(F/S)/T` from the matched active future and is labelled futures-implied carry, not observed dividend yield. Discounting is `D=1/(1+yT)` and `r=-log(D)/T`. These conventions intentionally preserve the reviewed pilot/G2 contract and disclose the short-end proxy limitation rather than invent a curve.

## 5. Role separation

For the **pricing-family comparison**, inner nominal slots (`[-0.05,0,+0.05]`, both ranks/types) fit BS/Heston/DH; outer wings (`[-0.10,+0.10]`) are holdout. Masked slots are removed from both fit and score.

For the **inverse-method comparison**, every method sees the identical full valid R2 mask, including wings, because that is the trained canonical representation. There is no real-parameter truth, so no calibration/holdout split can measure parameter recovery.

## 6. Pricing-model family protocol

Black-Scholes has one volatility parameter with bounds `[0.01,2.0]` and bounded scalar optimization. Standard Heston uses the reviewed pilot path: `src.double_heston.heston_log_characteristic_exponent` plus the pilot put-call-parity wrapper (`scripts.run_ntpc_single_stock_pilot.price_heston_option`) at 64 Gauss-Laguerre nodes. Its external order is `[kappa,theta,sigma,rho,v0]`. Optimize five unconstrained coordinates clipped to `[-35,35]` through logistic sigmoid: kappa `0.05+u0*(12-0.05)`, theta `0.002+u1*(0.30-0.002)`, v0 `0.002+u2*(0.35-0.002)`, sigma `0.005+u3*(min(1.5,sqrt(2*kappa*theta)*(1-1e-7))-0.005)`, and rho `0.95*tanh(raw x4)`. The deterministic NumPy PCG64 generator has seed `20260912`: the first 5-vector is zero and the next seven are sequential `Normal(0,1.25^2)` draws. Double Heston maps unconstrained coordinates through `src.calibrate_double_heston.unconstrained_to_parameters` with the provisional canonical hard bounds; NumPy PCG64 seed `20260922` creates a zero 10-vector followed by eleven sequential `Normal(0,1.25^2)` 10-vectors. Fit stochastic families on inner slots with SciPy `least_squares(method="trf")`, tolerances `1e-10`, `diff_step=2e-5`, and `max_nfev=300`; residual is `(model-observed)/max(observed,1)`. Fixed start counts are Heston 8, DH 12. Representative is lowest final objective, ties by lowest index. All starts/failures are retained.

Report normalized and dollar price RMSE/MAE/max error, mean relative price error, median/p95, separately for calibration, holdout, and full valid slots. Ranking aggregates are unweighted at surface level: one per-surface holdout RMSE value per model, then the median across eligible surfaces. Slot-level values are reported but do not replace this rule. A method-surface failure means the representative fit or a required valid price/IV result is absent or nonfinite; reaching the evaluation cap is diagnostic, not automatically failure. Use the same bracketed forward-Black routine to compute market and model IV; report annualized RMSE/MAE/bias/max, success rate, and reasons. Never clip or replace an IV.

Default classification is `NO_CLEAR_PRICING_FAMILY_WINNER`. A family may be called preferred under this protocol only if it lowers median holdout normalized-price RMSE **and** median holdout annualized IV RMSE by at least 5% versus every comparator, with failure rate no more than 10 percentage points worse than the best comparator. Price fit alone never wins.

## 7. Inverse-method protocol

Traditional calibration follows the frozen R2-primary optimizer/bounds except that its synthetic three-start schedule is not executable on real data: the third start is truth-informed and forbidden. Before acquisition, add and review an adapter that returns only neutral-midpoint and broad deterministic starts from seed 42; run exactly those two starts at `max_nfev=300`, report both, and choose lowest objective. Model 1 and Model 2 load their already-frozen seed 11/22/33 best-validation checkpoints. Model 3 participates only if frozen and committed before acquisition; otherwise report it unavailable.

Before acquisition, run `python scripts/validate_g8_protocol.py check-checkpoints`. This lightweight gate verifies the frozen synthetic dataset hash, then all six exact neural checkpoints named in the machine contract: file SHA-256, successful load after hash approval, embedded standardizer state, research run kind, model/seed/path agreement, recorded git SHA, architecture spec presence, canonical parameter order, validation-only selection. Missing/mismatched checkpoints block acquisition. It performs no pricing, calibration, or evaluation.

Neural paths run eval mode with gradients disabled. No optimizer updates weights, selects a checkpoint, changes a loss/architecture/hyperparameter, or chooses a seed on G8. Traditional settings also cannot change after the first result. Failed methods/surfaces are reported, never replaced.

Parameter recovery is explicitly `NOT_APPLICABLE_NO_REAL_TRUTH`. Report predicted vectors, structural validity and individual violation rates, repricing and IV errors over the identical full valid mask, cross-seed prediction/headline dispersion for neural methods, multi-start dispersion/clusters/boundary hits/cap rates for tradition, and runtime. Complete-linkage cutoff remains 0.10 range-scaled units; disagreement means any pair exceeds 0.50 range-scaled units; material displacement threshold remains 0.05.

For every real surface, report usable-slot count overall and by expiry rank/option type. Models 1/2 were trained on complete synthetic masks while G8 uses observed partial masks; disclose this distribution shift and never describe G8 as ordinary in-distribution generalization.

No inverse method can win a real parameter-truth claim. Descriptive rankings of price error, IV error, validity, stability, and runtime are allowed and must remain separate. Low repricing error never certifies correct Double Heston parameters.

## 8. Runtime and evidence

Record wall seconds per model and surface (all traditional starts included), batched CPU inference milliseconds in eval/no-grad mode, totals, hardware, Python/library versions, git commit, config hash, dataset-manifest hash, and checkpoint hashes. Produce atomic JSON/CSV evidence and SHA-256 for every input/output/report. Archive/schema/hash mismatch aborts before execution; failures are retained with reasons and no silent reruns.

## 9. Interpretation discipline

The default inverse conclusion is `NO_REAL_PARAMETER_TRUTH_WINNER`, while practical non-identifiability remains retained. A preference label may describe measured real-surface performance only under the predeclared family rule or an identically defined joint price+IV/failure/stability rule stated before execution. It may not claim unique recovery or true parameters. Any threshold, seed, mask, budget, architecture, or selection change after a G8 result invalidates the protocol.

## 10. Stop condition

This milestone freezes the protocol only. Suitable untouched market data remains unproven and must not be acquired in this task. Execution requires the frozen data manifest and a separate explicit authorization.

```text
G8_PROTOCOL_FROZEN
```
