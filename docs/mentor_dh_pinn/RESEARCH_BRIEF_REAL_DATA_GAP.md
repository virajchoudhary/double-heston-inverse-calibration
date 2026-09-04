# Research brief — closing the real-data gap for the unified Double Heston calibrator

Written so you can gather literature efficiently. Section 4 is the part that matters: the
failure is diagnosed, not guessed, and the diagnosis points at a specific and well-studied
problem. Sections 6 and 7 say what is already ruled out and what must not be broken, so no
effort is spent on dead paths.

## 1. The one question

> A neural calibrator trained on surfaces generated **by** Double Heston beats classical
> calibration on synthetic data and loses to it on real quotes, **even where its own
> out-of-distribution detector says the input is in-distribution.** How do we fix that?

## 2. What the model is

One model, one call: an arbitrary set of option quotes -> a permutation-invariant set
encoder -> ten parameter-query tokens, each cross-attending to the quotes -> a location
`mu_z` and a full 10x10 covariance `Sigma_z` in unconstrained latent coordinates -> an
unrolled damped Gauss-Newton refinement against the **exact** Fourier pricer, inside
`forward()`.

Details that matter for reading the literature:

* Strike and maturity are token *features*, so N varies freely (3 to 100+ quotes, 1 to 8
  expiries). Permutation invariance measured at 6.1e-16.
* The pricer is a PyTorch port of the production engine, not a surrogate: agreement
  3.1e-15 batched, autograd verified against 4th-order finite differences at 1.6e-8.
* The parameter transform is a bijection onto the engine's valid set (Feller, kappa
  ordering, correlation disk are all engine-enforced). No box clipping.
* Training: 150,000 synthetic surfaces, regime-balanced prior over four volatility regimes,
  correlated/heteroskedastic/heavy-tailed noise to 6%. Loss is
  `L_parameter + L_uncertainty(Gaussian NLL) + L_clean_physics + L_refined`.
* **The supervised target is the generating parameter vector `p*`.**

## 3. Measured performance

Synthetic, 4,000 held-out surfaces (all prices from the exact engine):

| method | reprice | parameter RMSE | s/surface |
|---|---:|---:|---:|
| **unified, 3 physics steps** | **6.310e-04** | **0.1593** | **0.0505** |
| single Heston | 8.615e-04 | -- | 0.559 |
| Double Heston, cold 5-start | 8.757e-04 | 0.4367 | 2.888 |
| Black-Scholes | 1.731e-03 | -- | 0.010 |

Real markets, median holdout implied-volatility RMSE:

| market | unified | classical Double Heston | ratio |
|---|---:|---:|---:|
| NIFTY, 10 dates, 1,750 quotes | 0.02902 | 0.02523 | 1.15x worse |
| ADANIPOWER, 11 dates, 51 quotes | 0.01754 | 0.00398 | **4.4x worse** |

Uncertainty on synthetic is well calibrated: 50/90/95% intervals cover 0.536/0.910/0.953.

## 4. The diagnosis — two different failures

### 4.0 The accuracy gap can be bought with compute -- at a price that removes the point

Before the per-market diagnosis, one result frames everything. The same trained network,
given a large enough inference-time budget (12 starts drawn from its own `Sigma`, 30
refinement steps each), reaches:

| market | network 12x30 | classical cold 5-start | accuracy | wall clock |
|---|---:|---:|---|---|
| NIFTY | **0.02478** in 107.6 s | 0.02523 in 21.8 s | 1.8% **better** | **4.9x slower** |
| ADANIPOWER | 0.01477 in 69.7 s | 0.00398 in 1.5 s | 3.7x worse | **48x slower** |

So the NIFTY deficit *is* reachable -- but only by spending about 3,960 exact pricer
evaluations, which costs five times what classical calibration costs. **The amortised model's
entire advantage was speed, and any configuration that spends enough compute to fix the
accuracy gives that advantage away.** In its fast configuration (3 steps, single start:
0.854 s NIFTY, 0.509 s ADANIPOWER) it is 21x and 3.2x faster than classical but less accurate.

The research question is therefore sharper than "make it more accurate":

> **Make the amortised model land near the projection answer in a small, fixed number of
> exact-physics steps -- so accuracy is bought by a better starting point, not by search.**

That is a statement about the quality of the network's posterior, not about the optimiser,
and 4.2 shows exactly how far off that posterior currently is.

### 4.1 NIFTY is a step-budget problem, and is nearly solved

| configuration | median IV RMSE |
|---|---:|
| 3 refinement steps (as reported) | 0.02902 |
| 10 steps | 0.02664 |
| 30 steps | 0.02588 |
| 30 steps, prior weakened 100x | 0.02549 |
| classical cold 5-start | 0.02523 |

Raising the step budget and loosening the prior brings it to within **1%** of classical.
NIFTY is also flagged `severe_extrapolation` on 10/10 dates -- correctly, since it carries
10-11 expiries and 1,000+ quotes to 800 days while training topped out at 8 expiries and
100 quotes. Widening the training geometry plus more steps should close this. **This is
engineering, not research.**

### 4.2 ADANIPOWER is the research problem: the network is confidently wrong

ADANIPOWER is flagged `in_distribution` on 9 of 11 dates, so OOD does not explain it.
Each hypothesis was tested and eliminated:

| hypothesis | test | result |
|---|---|---|
| too few refinement steps | 3 -> 10 -> 30 steps | plateaus at 0.0167. **Not it.** |
| prior too stiff | weaken `Sigma` 100x, 10,000x | gets *worse* (0.0178, 0.0242). **Not it.** |
| bad basin, fixable by search | 4 and 12 starts drawn from the model's own `Sigma` | 0.01610, 0.01477. Helps 12%, still 3.7x off, and 48x slower than classical. **Not sufficient.** |

The last row is the clue. If the good region were inside the predicted posterior, a dozen
draws would find it. Measuring directly where the classical optimum sits **in the network's
own predicted units**, on real ADANIPOWER surfaces:

| latent coordinate | median distance from `mu_z` |
|---|---:|
| `v0_total` | **10.10 sd** (max 20.84) |
| `rho_a` | 6.00 sd |
| `rho_b` | 5.30 sd |
| `eta_fast` | 4.30 sd |
| `theta_total` | 3.81 sd |

**43% of coordinates lie beyond 3 sd; the median across all ten is 2.60 sd.** The posterior
is not merely uninformative -- it is centred in the wrong place with intervals far too narrow
to reach the answer. On synthetic data the same head is calibrated to within a percentage
point of nominal. **The calibration collapses under real-data misspecification.**

### 4.3 The mechanism, stated plainly

The network is trained to invert a data-generating process: given a surface *produced by*
Double Heston, recover the parameters that produced it. Real surfaces are not produced by
Double Heston. On real data there is no `p*` to recover -- only a **projection**, the
parameter vector minimising fit error under a misspecified model. Those two problems have
different answers, and the network was only ever trained on the first.

Classical calibration solves the second problem directly, which is why it wins on real
quotes and loses on synthetic ones (where the network's extra information -- knowing the
prior -- is genuinely worth 2.74x on parameter recovery).

This is sharpest on **single-expiry** surfaces because theta and kappa are unidentified
there: the fit-error surface has long flat ridges, so the recovery answer and the projection
answer can be arbitrarily far apart while fitting almost equally well.

## 5. Research directions, in priority order

### D1. Misspecification-aware amortised inference  (highest value)

*Question:* how do you train an amortised inverse model so its posterior stays honest when
the observation is not from the simulator?

This is a known and active failure mode in simulation-based inference, and what was measured
above -- near-nominal coverage in-distribution, gross overconfidence out of it -- is the
documented signature.

Search terms: `neural posterior estimation misspecification`, `robust simulation-based
inference`, `SBI model criticism`, `overconfident posterior SBI`, `misspecified simulator
amortized inference`, `generalised Bayes / power posterior misspecification`.

Starting points to verify: Hermans et al., *A Trust Crisis in Simulation-Based Inference?
Your Posterior Approximations Can Be Unfaithful* (TMLR 2022) -- directly on overconfidence;
Ward et al., *Robust Neural Posterior Estimation and Statistical Model Criticism*
(NeurIPS 2022); Cannon, Ward & Schmon, *Investigating the Impact of Model Misspecification
in Neural Simulation-Based Inference* (2022); Frazier, Robert & Rousseau on misspecified ABC.

### D2. Train on the projection, not the recovery  (most directly actionable)

*Question:* can the network be trained, or fine-tuned, on the objective actually used at
evaluation -- fit error against observed quotes -- rather than on parameter recovery?

**This needs no labels.** The exact pricer is differentiable, so the fit residual is
computable on real surfaces. A self-supervised fine-tune on real NSE quotes is possible
today with the code as it stands. That is the single cheapest experiment available.

Search terms: `amortized optimization`, `learning to optimize`, `self-supervised
calibration`, `deep equilibrium models inverse problems`, `unrolled optimization learned
initialization`, `test-time adaptation`.

Starting points: Amos, *Tutorial on Amortized Optimization* (2023); Chen et al., *Learning
to Optimize: A Primer and A Benchmark* (JMLR 2022).

### D3. Posteriors that actually cover

*Question:* replace the single Gaussian with a density that can be multimodal and wide
enough to contain the projection answer.

The full-covariance Gaussian is unimodal by construction. On a single-expiry surface the
fit-error landscape has ridges and multiple near-equivalent optima; one Gaussian cannot
represent that, which is consistent with 12 covariance-sampled starts failing.

Search terms: `conditional normalizing flow posterior`, `mixture density network
calibration`, `simulation-based calibration SBC`, `conformal prediction inverse problems`,
`coverage-calibrated neural posterior`, `WALDO confidence regions`.

### D4. Hybrid: network proposes, classical disposes

*Question:* what is the best division of labour between the amortised model and a real
optimiser, **under a fixed compute budget**?

Tested and quantified: 12 covariance-sampled starts improve ADANIPOWER 12% and NIFTY 4%,
but at 48x and 4.9x the classical wall clock. Naive search is not the answer. The useful
version of this question is how to get a *good* start cheaply -- one or two well-chosen
initialisations rather than twelve sampled ones. Any proposal must be benchmarked on the
accuracy-per-second frontier, not on accuracy alone.

Search terms: `multi-start global optimization calibration`, `learned initialization
nonlinear least squares`, `basin hopping`, `neural network warm start optimization`.

### D5. Identifiability on weakly informative surfaces

*Question:* on one expiry, what is actually identified, and should the model refuse to
report the rest?

This project already measured the sensitivity structure: `|d log C / d log p|` spans
30-300x, and on a 7-day quote `dC/dkappa_slow` is 5.4e5 times smaller than `dC/dv0_slow`.
A defensible answer for a single-expiry surface may be a *set*, not a point.

Search terms: `practical identifiability stochastic volatility`, `Fisher information
Heston calibration`, `ill-posed inverse problem regularization option pricing`,
`profile likelihood identifiability`.

## 5.1 D2 TESTED: projection fine-tuning works on multi-expiry, not on single-expiry

Implemented and run. `src/mentor_dh_pinn/finetune_projection.py`,
`scripts/mentor_dh_pinn/{build_real_corpus,run_finetune,evaluate_finetune,evaluate_panel_test}.py`.

**Setup.** 1,401 real surfaces / 27,949 quotes for self-supervised fine-tuning, assembled
from the stock panel's `split == "train"` plus 60 NIFTY dates, with all 20 NIFTY test and
validation dates and the whole panel test split withheld. Objective triad: supervised
recovery anchor on synthetic, vega-weighted projection loss on unlabelled real quotes, and
a variance-based Bayesian self-consistency term. All three standardised by a running EMA of
their own scale -- raw magnitudes were 10, 8e-4 and 12.5, so a fixed weighted sum would have
handed the entire gradient to self-consistency.

**Result on the real validation corpus:** IV RMSE 0.03081 -> 0.02516, **18.3% better**, with
synthetic 90% coverage holding at 0.906 and latent z-MAE drifting only 0.8%. Minimal
forgetting, 20 minutes of fine-tuning.

**Result on the frozen test sets:**

| evaluation | geometry | n | base | fine-tuned | classical DH | verdict |
|---|---|---:|---:|---:|---:|---|
| NIFTY | 10-11 expiries | 10 dates, 1,750 quotes | 0.02902 | **0.02478** | 0.02523 | **beats classical** |
| stock panel test | 1-2 expiries | 140 surfaces, 820 quotes | 0.01920 | 0.01760 | **0.00690** | no effect |

* **NIFTY: the fix works.** Better than base on **9 of 10 dates (Wilcoxon p = 0.006)**, and it
  now beats classical Double Heston on 7 of 10 dates (p = 0.064) while running in 1.16 s
  against 25.26 s -- **21.7x faster and more accurate**. This is the first configuration in
  the project that wins on real data without giving away the speed advantage.
* **Single expiry: no effect.** 8.3% median improvement, but better on only 74 of 140
  surfaces, **Wilcoxon p = 0.55**. Indistinguishable from noise. Both base and fine-tuned
  remain about 2.6x behind classical (p = 2.5e-23).

**The important negative.** The fine-tuning corpus was already **96% single- and two-expiry**
(1,341 of 1,401 surfaces came from the stock panel). Single-expiry failure therefore is *not*
a data-volume problem, and more single-expiry fine-tuning data will not fix it. Projection
training repairs the misspecification gap **only where the surface identifies the parameters
well enough for a projection to be well defined.** On one expiry, theta and kappa are not
identified, the fit-error landscape is a ridge rather than a basin, and moving the training
objective does not create information that the quotes do not contain.

This narrows the remaining problem sharply: **D3 (posteriors that can represent a ridge) and
D5 (report a set, not a point, when the surface cannot identify one)** are now the live
directions for single-expiry surfaces. D1 remains relevant to both regimes.

Note on measurement: the ADANIPOWER test set has a median of **four** holdout quotes per
date, and shows no significant effect either way (6/11, p = 1.000). It is too small to guide
anything. The 140-surface panel test split is the evaluation to use for single-expiry work.

## 6. Already ruled out -- do not spend effort here

* **More refinement steps.** ADANIPOWER plateaus by 10-30 steps.
* **Brute-force multi-start from the predicted covariance.** Measured on both markets: it
  helps 12% on ADANIPOWER and 4% on NIFTY, and costs 48x and 4.9x the classical wall clock
  respectively. It buys accuracy with compute the amortised design exists to avoid.
* **Weakening the prior.** Makes ADANIPOWER worse, monotonically.
* **Retraining with a working physics layer.** Tried (`unified_v2`); it came out worse
  (0.16370 vs 0.14699 parameter, 7.7621e-04 vs 6.4113e-04 reprice). Caveat: n=1 vs n=1,
  confounded with initialisation.
* **A better surrogate pricer.** There is no surrogate any more -- the exact engine is in
  the graph. Separately, an earlier experiment showed a 35% better surrogate producing
  *worse* parameter recovery.
* **Wider parameter boxes.** The transform is already a bijection onto the full valid set;
  nothing is clipped.

## 7. Constraints that must not be broken

1. **Feller, `kappa_slow < kappa_fast`, and the correlation disk are enforced by the pricing
   engine itself.** A vector violating them cannot be priced. Any new parameterisation must
   stay inside the model class.
2. The two variance factors are **exactly permutation-symmetric** (measured 1.07e-14), so an
   ordering convention is mandatory or the target is ill-defined.
3. Final accuracy must be evaluated with the **exact** engine, never a surrogate.
4. Reported latency must include the physics steps, not just the encoder.
5. Real-market splits are chronological and must stay that way; quotes from one surface must
   never be split across train and test.

## 8. What success looks like

Beating classical Double Heston on ADANIPOWER holdout IV RMSE -- below **0.00398** -- while
keeping the properties the redesign bought: arbitrary quote geometry, no per-market tuning,
end-to-end latency far below the classical 1.6-18 s, and an uncertainty output whose
coverage survives contact with real data.

A weaker but still valuable result: matching classical accuracy at 20x the speed, with
honest intervals.

**Any claimed improvement must report wall clock.** The accuracy gap is already closable on
NIFTY by brute force at 4.9x the classical cost; that is not progress, and a result that
does not state its compute budget cannot be distinguished from it.

## 9. Where everything lives

Branch `Double/Single-heston` of `virajchoudhary/double-heston-inverse-calibration`.
Code `src/mentor_dh_pinn/`, experiments `scripts/mentor_dh_pinn/`, trained networks
`outputs/unified_v6/unified.pt`, full results `docs/mentor_dh_pinn/DELIVERABLE_F_FINAL_REPORT.md`,
loading instructions `HESTON_PINN_MODELS.md`.
