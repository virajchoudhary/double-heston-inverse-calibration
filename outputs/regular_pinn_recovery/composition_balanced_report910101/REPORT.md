# Compositional Double Heston PINN — development results

This is an exposed synthetic pilot, not an unseen or real-market performance claim.

## Results

| Model | All parameters | Individual DH gates | Neural price gate | Exact repricing gate |
|---|---:|---:|---:|---:|
| single_sobolev_s17 | N/A | N/A | 0/12 | 0/12 |
| component_balanced910071 | N/A | N/A | 0/12 | 0/12 |
| composition_balanced910071 | 0/12 | 50/120 | 12/12 | 0/12 |

Single Heston has no matching five-parameter truth on these Double Heston surfaces. A lower pricing error does not demonstrate accurate ten-parameter recovery.

![Same-quote pricing comparison](pricing_comparison.png)

Interpretation: lower is better. Left measures the fitted neural function; right checks whether its recovered parameters also price accurately in the independent reference engine. The dashed line is the unchanged 1e-5-of-spot gate.

![Ten-parameter recovery](parameter_recovery.png)

Interpretation: each cell is error divided by its allowed tolerance. Values <=1 pass; a case succeeds only if all ten cells pass. Display color limits do not cap printed errors.

## Paired pricing comparison

- composition_balanced910071 vs single_sobolev_s17, neural: Double wins 12/12; valid pairs 12; median Double/Single RMSE ratio 0.01427754642790166.
- composition_balanced910071 vs single_sobolev_s17, exact_repricing: Double wins 12/12; valid pairs 12; median Double/Single RMSE ratio 0.06978693985026319.
- composition_balanced910071 vs component_balanced910071, neural: Double wins 12/12; valid pairs 12; median Double/Single RMSE ratio 0.014104946189384852.
- composition_balanced910071 vs component_balanced910071, exact_repricing: Double wins 12/12; valid pairs 12; median Double/Single RMSE ratio 0.07000542381490935.

## What changed and what training achieved

Two evaluations of a shared Single Heston price-PDE PINN are combined through the independent-return convolution identity. Density is obtained by automatic price derivatives. Fixed 96-node quadrature is differentiable. No exact Heston pricer, ODE solver, density clipping or normalization enters calibration. This is a separately labelled component-PDE architecture, not direct full Double Heston PDE training.

The broader-domain continuation ran 10000 float64 AdamW steps with 18000 collocation points, 129840 usable synthetic training states and 16239 validation states. Selection chose step 10000, validation IV RMSE 0.0002100335908. The initial learning rate was 5e-05, PDE weight 0.02. Validation improved by 42.32% from the initial checkpoint. This is forward-validation evidence; it is not proof of ten-parameter recovery or unbiased generalization.

Training targets are reference synthetic prices/IV corrections and parameter sensitivities, plus the component pricing PDE. Those are forward-training inputs; recovery-case truths are used only in assessment. All 1232 rejected training and 145 rejected validation reference candidates remain in the archives. No prediction failures were removed from the assessment denominators.

## Data and controls

Twelve already-exposed DH parameter vectors (truth seed 927931), each with 21 strikes from .8 to 1.2 of unit spot and six maturities: 30,60,90,180,365,730 days divided by365. These are normalized synthetic European calls with zero rate/carry; they are not NSE observations or a representation of currently available Indian stock expiries. Each case uses 84 calibration and 42 held-out quotes, identical across models.

All models use five blind LHS starts and at most 400 evaluations/start; selection uses calibration neural-IV SSE only. All starts and failures are retained. The three models completed with reference-pricer entry points blocked during every fit; case 0 was replayed across all starts with held-out coordinates/values replaced by NaN. The replay was identical excluding timings. This is a concrete isolation test, not a guarantee against every possible implementation issue. Source and weight hashes were rechecked. Parameter gates, scores and start selections were independently recomputed by this report script.

## Test scope

The 55 scoped regular/composition tests pass, including parameter Jacobians, reference composition, checkpoint roundtrip and guarded held-out isolation. The whole checkout is NOT fully passing: with the legacy Single source import path configured, the broader run has 629 passed, 43 failed, 1 skipped. Its archived XML lists missing historical market/evidence files, hash/runtime/schema mismatches, an unavailable openpyxl dependency and the old 1e-12 numerical fixture discrepancy. Three legacy dtype-sensitive training tests fail together but pass in isolation. No old fixture, recorded hash or gate was changed to conceal these failures.

## Limitations and next work

The twelve cases have informed development and cannot provide an unbiased final generalization estimate. There is one training lineage, no fresh noise cohort here, and only DH-generated comparison surfaces. Check density, martingale, factor-swap and quadrature diagnostics in every case JSON; they are reported without corrections. The finite Fourier reference aliases at extreme integration tails; the independent composition identity test uses converged 32/48-node rules, while the learned model uses 96 nodes and is audited at 128.

Further claims require improved component accuracy, all-ten recovery on the fixed development gates, then a frozen independent cohort covering both Single- and Double-generated surfaces, noisy quotes and independent training seeds. No universal identifiability or Single-Heston superiority claim follows from this pilot.

IV is a deterministic transformation of price, not a new independent measurement. Factor ordering removes label swapping but does not guarantee that finite quotes identify all ten parameters stably.

[Full parameters](PARAMETERS.md) · [Machine-readable audit](audit.json)

## Independent full-PDE diagnostic

These are raw price-per-year residuals on 512 fresh states, not the normalized residual metric used by earlier monolithic PINN reports. The diagnostic did not select checkpoints or change calibration.

| Model | PDE RMSE | Negative calendar points | Lower-bound violations | Max lower-bound shortfall |
|---|---:|---:|---:|---:|
| composition910031 | 0.000143622 | 14 | 13 | 5.84e-09 |
| composition_balanced910071 | 9.61513e-05 | 13 | 13 | 4.051e-09 |

The audit JSON records finiteness, convexity, bounds and core/stress counts at the stated 1e-12 shape threshold. Violations are not silently clipped or treated as exact physical validity.

## Training-only inverse-conditioning diagnostic

First 32 archived independent training surfaces, without resampling or dropping failures. Both model checkpoints were frozen before this diagnostic. The stored reference-Jacobian pseudoinverse maps IV error to a local parameter-bias proxy in recovery-tolerance units. This is NOT actual recovery; directions truncated by the archived singular-value cutoff remain unmeasured.

| Model | IV RMSE | Linear-shift RMSE | Median largest absolute shift |
|---|---:|---:|---:|
| composition910031 | 0.000138309 | 9.87865 | 5.39664 |
| composition_balanced910071 | 9.48106e-05 | 7.85887 | 5.95951 |

Read the average IV error alongside the largest parameter-direction metric: optimizing average forward-pricing accuracy alone need not improve every parameter. A possible next experiment is to adapt the existing grouped inverse-sensitivity training loss to this composition, with independent training/validation surfaces and fixed gates. That experiment has NOT been run here and is not a promised solution.
