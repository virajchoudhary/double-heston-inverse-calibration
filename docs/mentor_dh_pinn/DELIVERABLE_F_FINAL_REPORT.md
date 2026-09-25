# Deliverable F — final research report

Unified physics-informed set calibrator for Double Heston. Every number below is read from a stored artifact in `outputs/unified_v6/`; every price is produced by the exact Fourier engine, never by a surrogate.

## F.1 Verdict against the six stated objectives

| objective | outcome |
|---|---|
| 1. match or beat exact repricing | **met on synthetic** (6.3102e-04 vs 8.7570e-04 cold-start Double Heston, 8.6147e-04 single Heston). **Not met on real markets** (NIFTY 0.02902 vs 0.02523; ADANIPOWER 0.01754 vs 0.00398) |
| 2. materially improve network-only repricing | **partly**. Pre-refinement 1.2576e-03; the gain comes from the physics layer, not the network |
| 3. process all single-expiry ADANIPOWER dates | **met**. All 11 dates ingested and calibrated; the fixed-grid architecture could not form an input for any of them |
| 4. eliminate manual market-specific lambda tuning | **met**. There is no lambda. The scalar spring is replaced by the predicted covariance, and nothing was retuned per market |
| 5. produce calibrated uncertainty | **met for coverage, not for discrimination**. 50/90/95% intervals cover 0.536/0.910/0.953, but Spearman(predicted sd, realised error) = 0.045 |
| 6. remain competitive in end-to-end latency | **met**. 50.5 ms per surface against 2888 ms for cold-start Double Heston (57x) |

## F.2 Synthetic benchmark

4,000 held-out surfaces; classical arms on a 100-surface subsample. Parameter error is range-scaled by the **training-prior 1st-99th percentile** per parameter (there is no PARAM_BOX any more); the scale vector is stored in the evaluation JSON so it can be audited.

| method | reprice vs clean | parameter RMSE | s/surface |
|---|---:|---:|---:|
| **unified, 3 physics steps** | **6.3102e-04** | **0.1593** | 0.0505 |
| single Heston (5 params) | 8.6147e-04 | -- | 0.5592 |
| Double Heston, cold 5-start | 8.7570e-04 | 0.4367 | 2.8880 |
| Black-Scholes, 1 sigma | 1.7313e-03 | -- | 0.0100 |

The unified calibrator is **1.39x** better than cold-start Double Heston on repricing, **2.74x** better on parameter recovery, and **57x** faster.

Refinement moves repricing from 1.2576e-03 to 6.4379e-04 (1.95x) and parameter error from 0.15265 to 0.14626. 65.7% of surfaces land within 0.10 range-scaled, 49.7% within 0.05.

Per parameter, the classic identifiability ordering is visible and is *not* hidden:

| parameter | median range-scaled error |
|---|---:|
| theta_fast | 0.00973 |
| v0_slow | 0.01443 |
| theta_slow | 0.01519 |
| v0_fast | 0.01650 |
| sigma_fast | 0.04522 |
| sigma_slow | 0.06473 |
| kappa_fast | 0.09859 |
| rho_fast | 0.11223 |
| rho_slow | 0.12267 |
| kappa_slow | 0.17621 |

`v0` and `theta` are recovered an order of magnitude better than `kappa` and `rho`, which is what the sensitivity structure predicts.

## F.3 Where it works and where it does not

| bucket | n | parameter | reprice | within 0.10 |
|---|---:|---:|---:|---:|
| D single expiry | 1014 | 0.1612 | 4.692e-04 | 0.621 |
| E two expiries | 602 | 0.1568 | 5.440e-04 | 0.639 |
| dense expiries | 1593 | 0.1351 | 7.813e-04 | 0.684 |
| historical 5x9 | 407 | 0.1412 | 5.933e-04 | 0.652 |
| B low-vol index regime | 857 | 0.1254 | 3.180e-04 | 0.753 |
| ordinary equity | 1768 | 0.1290 | 5.069e-04 | 0.724 |
| C high-vol stock regime | 1375 | 0.1920 | 1.439e-03 | 0.511 |
| low noise <=0.5% | 1279 | 0.1326 | 3.138e-04 | 0.686 |
| moderate noise | 1533 | 0.1453 | 6.548e-04 | 0.657 |
| J >2% noise | 1188 | 0.1603 | 1.656e-03 | 0.627 |
| H sparse (<=10 quotes) | 775 | 0.1603 | 4.583e-04 | 0.625 |
| dense (>=50 quotes) | 636 | 0.1343 | 7.840e-04 | 0.686 |

* **The prior redesign worked.** The low-volatility index regime -- the one NIFTY sits in, and the one the old prior effectively excluded at the 0.23rd percentile -- is now the **best** bucket (0.1254, 75.3% within 0.10).
* **Single-expiry surfaces are processed** (1014 of them, 4.692e-04 repricing). The fixed-grid architecture could not form an input at all.
* **High volatility is the weak regime** (0.1920, 51.1% within 0.10), roughly half as accurate as the index regime.
* Noise robustness is graceful: repricing degrades from 3.138e-04 below 0.5% noise to 1.656e-03 above 2%.

## F.4 Real markets

### NIFTY -- 10 dates, 1,750 held-out real quotes

| method | median holdout IV RMSE | best on |
|---|---:|---:|
| Black-Scholes | 0.02982 | 2/10 |
| single Heston | 0.02625 | 2/10 |
| Double Heston cold | 0.02523 | 2/10 |
| unified, network only | 0.03179 | 0/10 |
| unified + 3 physics steps | 0.02902 | 4/10 |

OOD status: {'severe_extrapolation': 10}. Latency 0.854 s against 17.874 s for cold-start Double Heston (21x).

### ADANIPOWER -- 11 dates, 51 held-out real quotes

| method | median holdout IV RMSE | best on |
|---|---:|---:|
| Black-Scholes | 0.02579 | 0/11 |
| single Heston | 0.00503 | 3/11 |
| Double Heston cold | 0.00398 | 6/11 |
| unified, network only | 0.02918 | 1/11 |
| unified + 3 physics steps | 0.01754 | 1/11 |

OOD status: {'in_distribution': 9, 'mild_extrapolation': 1, 'severe_extrapolation': 1}. Latency 0.509 s against 1.609 s for cold-start Double Heston (3x).

**This is the honest headline: the unified model does not beat classical calibration on real market data.**

On NIFTY it is competitive -- best on 4/10 dates, more than any single classical arm -- but its median (0.02902) is behind cold-start Double Heston (0.02523). On ADANIPOWER it is decisively behind: 0.01754 against 0.00398, a factor of 4.4.

**The OOD output explains the NIFTY gap and should be believed.** All 10 NIFTY dates are flagged `severe_extrapolation`: NIFTY carries 10-11 expiries and over a thousand quotes reaching 800+ days, while training geometry topped out at 8 expiries and 100 quotes. The model reports that it is extrapolating, and it is indeed weakest exactly there. ADANIPOWER, by contrast, is `in_distribution` on 9 of 11 dates -- single-expiry surfaces were 24% of training.

## F.5 Ablations

Only the ablations that are genuine inference-time switches were run. The rest each require training a separate model and are listed as not run.

| ablation | parameter | reprice | ms/surface |
|---|---:|---:|---:|
| I_refine_0 | 0.15519 | 1.2191e-03 | 3.0 |
| I_refine_1 | 0.14967 | 8.9874e-04 | 18.6 |
| I_refine_3 | 0.14768 | 6.3683e-04 | 50.0 |
| I_refine_5 | 0.14859 | 5.9506e-04 | 82.6 |
| E_spring_full | 0.14768 | 6.3683e-04 | 51.6 |
| E_spring_diagonal | 0.15294 | 6.4160e-04 | 51.8 |
| E_spring_scalar | 0.16765 | 6.7131e-04 | 52.9 |
| C_rounds_1 | 0.19223 | 2.0601e-03 | 53.9 |
| C_rounds_2 | 0.16114 | 7.5903e-04 | 53.6 |
| C_rounds_3 | 0.14768 | 6.3683e-04 | 53.8 |

* **The exact-physics layer is the single largest contributor.** Repricing improves 2.05x from 0 to 5 steps, most of it by step 3.
* **The covariance spring beats a scalar one**, which is the brief's central claim about replacing lambda: parameter error 0.14768 (full) against 0.15294 (diagonal) and 0.16765 (global scalar) -- 13.5% better than the scalar.
* **Iterated parameter-token communication matters**: 0.19223 at one round against 0.14768 at three, and repricing 3.2x better.

Not run, each requiring a separately trained model: A_fixed_vector_encoder, B_pooled_head_vs_parameter_tokens, F_iid_noise_vs_correlated, G_old_prior_vs_regime_balanced, H_surrogate_vs_exact_physics, J_sensitivity_informed_routing.

## F.6 What broke on the way, and what it cost

Three defects were found by testing rather than assumption. All three were mine.

1. **The first training run died silently at epoch 8.** One surface produced 186% total volatility on an 8.4-day option, where the characteristic function genuinely overflows; that single surface made the batch loss non-finite, and from epoch 9 every batch was skipped -- six epochs of nothing, with validation frozen. Fixed by masking non-finite quotes out of the loss, penalising unpriceable predictions explicitly, checking gradient finiteness after `backward()`, and saving the best-by-validation checkpoint rather than overwriting blindly.
2. **The latent coordinates were badly conditioned.** `softplus^-1(kappa_fast - kappa_slow)` is nearly linear in kappa and had a spread of 13.0 against about 1.2 for every other coordinate; it dominated the parameter loss and the model barely beat a constant predictor (1.681 against 1.726). Fixed by log-scaling both speeds with a multiplicative gap, which cut the spread ratio from 33.6 to 4.75 and the baseline z-MAE from 1.82 to 0.86. The brief warned about exactly this; I had applied log scaling to the prior but not to the coordinate.
3. **The unrolled refinement was inert.** Residuals were weighted by `1/spot` instead of by the reciprocal quote noise, which left the prior precision about 3,000x stiffer than the data term; every Gauss-Newton step moved the latent by 1e-4 against a trust region of 1.5. Pre- and post-refinement numbers were identical to four decimals, which is the symptom I should have caught sooner -- the phase D training log shows `ref` exactly equalling `par` every epoch. Fixed; repricing then improved 1.95x.

### The retrain, and a prediction that was wrong

Phase D of the reported model backpropagated through the *inert* refinement, so it was never trained to exploit a working physics layer. I predicted that re-running the curriculum with the corrected weighting would improve every number. **It did not.** A second full run (`unified_v2`, 3,211 s, identical settings, differing only in initialisation and in having a working refinement during phase D) came out worse on the held-out test set:

| model | refine steps | parameter | reprice |
|---|---:|---:|---:|
| v1, reported here | 0 | 0.15330 | 1.2352e-03 |
| **v1, reported here** | **3** | **0.14699** | **6.4113e-04** |
| v2, retrained | 0 | 0.15914 | 1.7675e-03 |
| v2, retrained | 3 | 0.16370 | 7.7621e-04 |

The mechanism behaved as predicted -- v2's refinement is **more** effective, improving repricing 2.28x against v1's 1.93x, and its phase-D log shows `ref` genuinely differing from `par` where v1's showed them identical. But v2's pre-refinement output is worse by more than that gain recovers. Its parameter error even *degrades* under refinement (0.15914 to 0.16370), which suggests it learned to place `mu_z` as a good launch point for the price fit at the cost of parameter accuracy -- the post-refinement objective rewards exactly that trade.

**Caveat on this comparison:** two runs, differing in random initialisation as well as in the refinement. I cannot separate 'training through refinement hurts' from initialisation variance without several more runs, and I have not done them. What I can say is that the predicted improvement did not appear, and the reported model is the better of the two.

## F.7 Remaining scientific limitations

1. **Uncertainty is calibrated but not discriminating.** Coverage is close to nominal (0.536/0.910/0.953 against 0.50/0.90/0.95), but Spearman(predicted sd, realised error) is only 0.045, and the theta standard deviation for single-expiry surfaces is only about 21% wider than for ten-expiry ones. The brief's requirement is that the head 'empirically know when the inverse problem is weak'. By that standard it does not yet. Average width is right; per-surface resolution is not.
2. **Real-market repricing is behind classical calibration**, decisively so on single-expiry ADANIPOWER. Ingestion is not identification, and this is the number that shows it.
3. **Six of ten ablations were not run**, so the attribution of gains is partial. The encoder, the parameter-token head, the noise model and the prior are all unvalidated against their alternatives.
4. **The earlier repricing benchmarks remain archived, not reproduced.** The two baseline scripts lost with the session scratchpad have been rewritten, but the historical 2.109e-4 figure was measured on a different test distribution (fixed 5x9 geometry, noise capped at 1%) and is not comparable to anything here.
5. **One volatility episode.** All real-market evidence comes from April 2026 NIFTY and a July-August 2026 ADANIPOWER window; NSE publishes no bid-ask, so every quote is an exchange settlement price.

## F.8 What the redesign did fix

| old weakness | status |
|---|---|
| fixed 5x9 grid | removed: variable-length set encoder, permutation invariance 6.1e-16 |
| interpolation required for real surfaces | removed: actual quote locations are fed directly |
| single expiry impossible | removed: all ADANIPOWER dates processed |
| low-vol NIFTY outside the prior | removed: the index regime is now the best bucket |
| global lambda retuned per market | removed: replaced by the predicted covariance |
| no uncertainty | added: full covariance, near-nominal coverage |
| hidden PARAM_BOX clipping | removed: decode is a bijection onto the model class; encode raises |
| surrogate error floor | removed: the exact engine is differentiable and is the teacher |
| price RMSE unrelated to recovery | partly: parameter and physics objectives are both trained |
| iid <=1% noise | removed: correlated, heteroskedastic, heavy-tailed, to 6% |
| neural latency quoted as calibration latency | removed: 50.5 ms end-to-end is the number reported |

## F.9 Reproduce

```bash
python scripts/mentor_dh_pinn/build_v6_data.py
python scripts/mentor_dh_pinn/train_unified.py --train-cap 80000 --batch 96 --lr 1.2e-3
python scripts/mentor_dh_pinn/evaluate_unified.py --n-full 4000 --n-baseline 100
python scripts/mentor_dh_pinn/evaluate_real_markets.py
python scripts/mentor_dh_pinn/run_ablations.py --n 1200
python -m unittest discover -s tests -p 'test_unified*.py'
```

Training: 14 epochs, 72 minutes, 0 skipped batches. Final gradient balance theta/v0 = 1.308, which was the acceptance test for the loss-domination failure this redesign set out to remove.