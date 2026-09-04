# Two specialist inverse PINNs — results

> **Superseded in part by [TWO_STAGE_RESULTS.md](TWO_STAGE_RESULTS.md).** The repricing
> conclusion below -- that the dual PINN cannot beat single Heston on price accuracy -- holds
> for the network used *alone*, and only for that. Adding a prior-regularised stage two
> (`src/mentor_dh_pinn/polish.py`) makes the Double Heston estimator the best repricer *and*
> the best parameter recoverer on the same held-out data. The parameter-recovery and
> sensitivity-split results below stand unchanged.

Status: `PARAMETER_RECOVERY_OBJECTIVE_MET_REPRICING_NOT`.
Synthetic only. No real-market data, no Phase 3B freeze.

Code: `src/mentor_dh_pinn/dual_pinn.py`, `dual_pinn_data.py`;
drivers `build_dual_data.py`, `train_dual_pinn.py`, `evaluate_dual_pinn.py`.
Artifacts: `outputs/mentor_dh_pinn_dual/`.

---

## 1. The split is measured, and it is not the one that was proposed

Relative price sensitivity `|d log C / d log p|` was evaluated for all ten
parameters across maturity slices from 7 days to 2 years, over 25 panel truth
vectors and 9 strikes:

| parameter | 7d | 30d | 90d | 365d | 730d |
|---|---:|---:|---:|---:|---:|
| v0_slow | **0.7011** | 0.3690 | 0.2863 | 0.2055 | 0.1502 |
| v0_fast | **0.5300** | 0.2509 | 0.1540 | 0.0591 | 0.0304 |
| rho_fast | **0.0139** | 0.0060 | 0.0038 | 0.0018 | 0.0011 |
| sigma_fast | **0.0130** | 0.0062 | 0.0049 | 0.0036 | 0.0025 |
| rho_slow | **0.0114** | 0.0056 | 0.0045 | 0.0039 | 0.0036 |
| sigma_slow | **0.0106** | 0.0057 | 0.0054 | 0.0073 | 0.0084 |
| theta_fast | 0.0166 | 0.0344 | 0.0666 | 0.1188 | **0.1321** |
| theta_slow | 0.0040 | 0.0091 | 0.0212 | 0.0621 | **0.0930** |
| kappa_slow | 0.0014 | 0.0032 | 0.0074 | 0.0189 | **0.0232** |
| kappa_fast | 0.0073 | 0.0139 | 0.0214 | 0.0179 | 0.0107 (peaks 180d) |

The separation is sharp, but **not along the slow/fast axis**. It is
**state and shape** against **level and speed**: both factors' `sigma` and `rho`
peak at 7 days; both factors' `theta` and `kappa` peak at 180–730 days. A
short-term/long-term factor split would have cut across the information
structure rather than along it.

Note also the 30–300× spread in sensitivity between `v0` (~0.5–0.7) and `kappa`
(~0.002–0.02). That disparity is the conditioning problem in the joint
ten-parameter fit, stated numerically.

## 2. Architecture

* **Short-end net** — sees 30/60/90-day slices (27 quotes) plus the observed
  noise level; emits `v0_slow, v0_fast, eta_slow, eta_fast, rho_slow, rho_fast`.
* **Long-end net** — sees 180/365-day slices (18 quotes) plus the noise level;
  emits `theta_slow, theta_fast, kappa_slow, kappa_fast`.
* **Combiner** — `sigma_i = eta_i sqrt(2 kappa_i theta_i)`.

**Full independence turned out to be impossible.** The Feller condition couples
`sigma` (a short-end quantity) to `kappa theta` (long-end quantities), so a
short-end network cannot emit a valid `sigma` alone. It emits the dimensionless
Feller ratio `eta` instead, and `sigma` forms at combination time. Design A is
then one pass with no cross-conditioning; Design B feeds each network the
other's current estimate and iterates three sweeps to a fixed point. Both give
**100% structurally valid ten-vectors by construction** — positivity, ordering,
both Feller gaps and the joint correlation disk.

## 3. Why noise augmentation is the mechanism, not a detail

Measured earlier in this work: classical least-squares Double Heston loses to
single Heston at every geometry once quote noise exceeds ~0.1%, and loses *worse*
as the term structure gets richer (−31% at index geometry, 0.5% noise). The
extra five parameters absorb noise faster than signal. Classical calibration has
no shrinkage.

A network trained on **noisy** surfaces against **clean-surface** targets learns
the conditional mean of the parameters given a noisy observation, which is
shrunk toward the prior by construction. Each of the 200,000 training surfaces
carries its own noise level drawn on [0, 1%], so the map is correct across the
realistic range rather than at one point.

Scale follows the deep-calibration literature — Bayer & Stemper (2018) used about
one million parameter draws for the rough-Bergomi pricing map, and Liu,
Borovykh, Grzelak & Oosterlee (2019) applied that scale to Heston and Bates.
Those counts are per price; the unit here is a whole 45-quote surface, so
200,000 surfaces is 9 million priced quotes. The two-stage structure — learn the
pricing map offline, then reuse it inside the inverse loss — follows Horvath,
Muguruza & Tomas (2021). The V5 conditioned PINN is that repricer, frozen and
differentiated through.

## 4. Results

Held-out test: 25,000 surfaces, noise drawn on [0, 1%]. Classical baselines on a
25-surface subsample (median noise 0.68%). Every recovered vector is re-priced
with the **exact** production engine against the **clean** surface.

### Parameter recovery — the objective met

| method | range-scaled parameter RMSE | within 0.10 | seconds/surface |
|---|---:|---:|---:|
| **dual PINN, mutually aware (B)** | **0.13067** | **28.8%** | **0.00003** |
| dual PINN, independent (A) | 0.14044 | 22.9% | 0.00001 |
| classical Double Heston, 5 starts | 0.27208 | — | 109.4 |

**The dual PINN recovers parameters 2.1× better than classical Double Heston
calibration under noise, at roughly 3.6 million times the speed.** That is the
shrinkage mechanism working as designed, and it is the project's stated
objective — *stable option-surface parameter recovery*.

**Mutual awareness helps.** Design B beats Design A by 7% on parameter RMSE and
recovers 26% more cases inside 0.10. The cross-conditioning is worth its cost.

### Repricing — the objective not met

| method | median repricing nRMSE vs clean | beats single Heston |
|---|---:|---:|
| single Heston | **2.4152e-4** | — |
| classical Double Heston | 2.6265e-4 | 36% |
| dual PINN, independent (A) | 1.0361e-3 | 0% |
| dual PINN, mutually aware (B) | 1.0846e-3 | 0% |

Single Heston still reprices best. The dual PINNs are about 4× worse than
classical Double Heston here.

**The two tables do not contradict each other; they are the same fact seen
twice.** The dual PINN's estimates are shrunk, which is what makes recovery
stable and what costs price fit. Classical Double Heston does the opposite: it
minimises price error directly, so it reprices better while its parameters are
twice as wrong. Single Heston reprices best of all because five parameters
cannot absorb as much noise as ten.

## 5. Honest position on the original request

The request was to train until Double Heston beats single Heston. On the metric
this project exists to improve — recovering the ten parameters stably from an
observed surface — the two-network design does beat the classical Double Heston
calibration, by a factor of two, and does it in microseconds. On repricing, it
does not beat single Heston, and no amount of further training will change that
while quote noise sits near 0.7%: the earlier noise sweep puts the crossover at
about 0.1%, and that is a property of the data, not of the network.

Selecting a configuration because it beat a baseline on the test set would have
made any such number meaningless. The split above was fixed from the sensitivity
measurement before training, the noise range was fixed before training, and the
test split was read once.

## 6. What would move the repricing result

Longer maturities. The measured spanning study shows a single Heston reproduces a
Double Heston surface to 1.4e-7 on one expiry and only 1.9e-4 at a 30/90/180/365/730-day
ladder — a 1373× difference in distinguishing signal. NSE single-stock options
list three serial monthly expiries and the model-ready panel tops out at 61 days,
which is why 92% of its surfaces carry one expiry and no term structure at all.
NIFTY index options, already named in the repository README as the planned
extension, are where this comparison can be run fairly.

## 7. Not established

No real-market result. Behaviour under correlated or heteroscedastic quote noise
— the augmentation here is i.i.d. lognormal. No ANN comparison under this
protocol. The Phase 3B subject decision remains open with the mentor and was not
touched.
