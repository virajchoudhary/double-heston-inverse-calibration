# Two-stage calibration: the dual PINN as a prior, not as an answer

Generated from `outputs/mentor_dh_pinn_dual/`. Held-out test split, 60 surfaces repriced with the exact production engine; median quote noise 0.60%. Parameter recovery for the networks alone is over the full 25,000 test surfaces.

## The problem the ridge solves

The dual networks are trained on noisy surfaces against clean targets, so what they learn is the *conditional mean* of the parameters given a noisy surface. That estimator is shrunk toward the prior. Shrinkage is why it beats classical least squares on parameter recovery under noise, and it is equally why it reprices badly: a shrunk vector does not sit at the minimum of the pricing residual for the one surface in front of it.

Classical calibration is the opposite estimator -- unshrunk, and it absorbs the noise. Neither endpoint is right. Stage two interpolates between them, warm-started at the network's own answer, in the constraint-free coordinates of `calibrate.decode`:

```
argmin_z  || (C(decode(z)) - C_obs) / S ||^2  +  lam * eps^2 * || z - z_net ||^2
```

`eps` is the typical per-quote noise magnitude for *this* surface, which the network is already told, so `lam` is dimensionless and comparable across surfaces. `lam = 0` is the two-stage design of Horvath, Muguruza & Tomas (2021) -- network as initialiser. `lam -> inf` is the network answer, unmoved. Every reachable point is structurally valid, because the optimiser never leaves `decode`'s image.

## Choosing the ridge, on validation only

Rule, declared before the test split was read: minimise validation parameter RMSE subject to validation repricing within 5% of the best achievable; chosen before the test split was read.

| start | ridge | validation param RMSE | validation reprice |
|---|---:|---:|---:|
| aware | 0 | 0.21902 | 1.9776e-04 |
| aware | 0.3 | 0.12828 | 1.6709e-04 |
| aware | 1 | 0.11140 | 1.5578e-04 |
| aware | 3 | 0.11375 | 1.4813e-04 |
| aware | 10 | 0.12053 | 1.5034e-04 |
| aware | 30 | 0.12279 | 1.6352e-04 |
| aware | 100 | 0.12759 | 2.0171e-04 |
| aware | 300 | 0.12707 | 2.8302e-04 |
| aware | 1000 | 0.12747 | 3.3691e-04 |
| aware | inf (net alone) | 0.12701 | 1.4630e-03 |
| ensemble | 0 | 0.21750 | 1.9770e-04 |
| ensemble | 0.3 | 0.14317 | 1.6616e-04 |
| ensemble | 1 | 0.11284 | 1.5603e-04 |
| ensemble | 3 | 0.10509 | 1.5034e-04  **<- chosen** |
| ensemble | 10 | 0.11766 | 1.5483e-04 |
| ensemble | 30 | 0.12059 | 1.7367e-04 |
| ensemble | 100 | 0.12683 | 2.0819e-04 |
| ensemble | 300 | 0.12543 | 2.8081e-04 |
| ensemble | 1000 | 0.12506 | 3.2345e-04 |
| ensemble | inf (net alone) | 0.12991 | 1.1904e-03 |

Both metrics have an interior optimum, and it is the same region. Pure warm-starting (`lam = 0`) reprices *worse* than `lam = 3` despite fitting the noisy surface harder -- it is fitting the noise. That is the whole case for the ridge: warm-starting alone is not enough.

## Held-out results

| method | median reprice vs clean | beats single Heston | param RMSE | s/surface |
|---|---:|---:|---:|---:|
| two-stage, ridge 3 (chosen on validation) | 2.1089e-04 | 78% | 0.12934 | 4.65 |
| classical single Heston | 2.5102e-04 | 0% | -- | 16.29 |
| two-stage, ridge 0 (network as initialiser) | 2.6350e-04 | 35% | 0.25263 | 24.74 |
| classical Double Heston, cold 5-start | 2.7332e-04 | 37% | 0.26257 | 249.28 |
| dual PINN A+B ensemble | 1.0170e-03 | 3% | 0.14121 | 0.091 ms |
| dual PINN B, mutually aware | 1.0424e-03 | 3% | 0.13943 | 0.068 ms |
| dual PINN A, independent | 1.0509e-03 | 3% | 0.14862 | 0.023 ms |

**The two-stage estimator wins on both axes at once.** Repricing 2.1089e-04 against single Heston's 2.5102e-04 (1.19x) and cold-start Double Heston's 2.7332e-04 (1.30x); parameter RMSE 0.12934 against cold-start Double Heston's 0.26257 (2.03x); and it does it in 4.6 s against 249 s (54x faster).

## Where the gain actually comes from -- not the warm start

The obvious hypothesis is that the network rescues a cold optimiser stuck in a bad basin. **That hypothesis is false, and the fit residual against the noisy surface each method was handed says so directly.**

| method | median fit RMSE vs the noisy surface it was given | median reprice vs clean |
|---|---:|---:|
| classical Double Heston, cold 5-start | 6.8337e-04 | 2.7332e-04 |
| two-stage, ridge 0 (network as initialiser) | 6.8337e-04 | 2.6350e-04 |
| two-stage, ridge 3 (chosen on validation) | 7.0271e-04 | 2.1089e-04 |
| classical single Heston | 7.2682e-04 | 2.5102e-04 |

Cold 5-start and warm-started stage two reach the **same** optimum -- 6.8337e-04 against 6.8337e-04, agreeing to within 1% on 90% of surfaces -- and the median objective spread across the five cold starts is 1.00x, i.e. the starts all agree with each other too. Cold-start Double Heston was never start-limited. Warm-starting buys speed, and only speed.

What buys accuracy is the ridge, and it does so by fitting the observed surface **worse**. At `lam = 3` the fit residual rises to 7.0271e-04 while repricing against the clean surface falls to 2.1089e-04. That is the signature of correct regularisation: the extra residual is noise the estimator declined to absorb. Ten free parameters against 45 noisy quotes will absorb noise unless something stops them, and the network -- which learned the shrunk conditional mean over 200,000 noisy surfaces -- is what supplies the direction to shrink toward.

So the division of labour is: the network provides the prior, the ridge provides the shrinkage, and the exact engine provides the fit. Neither piece works alone. The network alone reprices 1.0170e-03; the fit alone (two-stage, ridge 0 (network as initialiser)) reprices 2.6350e-04; together they reprice 2.1089e-04.

## Significance

The two-stage estimator reprices better than single Heston on **47 of 60** held-out surfaces (exact binomial p = 6.1e-06; paired Wilcoxon p = 5.3e-07). Cold-start Double Heston manages 22 of 60, which is not distinguishable from a coin. Against cold-start Double Heston the two-stage estimator wins with paired Wilcoxon p = 6e-08.

## What this changes

The earlier reading -- that Double Heston loses to single Heston under realistic quote noise, and that no amount of further training fixes it -- was right about the *network alone* and right that training is not the lever, but wrong as a statement about the model. The binding constraint at ~0.6% noise was **estimation variance**, exactly as the noise-crossover study said, and the fix is a better estimator rather than a better optimiser or a longer run. Continuing to train the networks remains pointless: epochs 15 to 21 moved validation parameter RMSE only 0.15928 to 0.15783.

Unchanged, and still the real limit: 91.7% of the NSE panel has a single expiry and a 61-day maximum tenor, and single Heston reproduces a Double Heston surface to 1.4e-7 on one expiry against 1.9e-4 on a 730-day ladder. Everything above is measured on a five-expiry ladder out to one year. It says the estimator is no longer the bottleneck. It does not say the current NSE panel can tell the two models apart -- NIFTY index options are still where that question gets settled.

## Reproduce

```bash
python scripts/mentor_dh_pinn/sweep_polish_lambda.py          # validation; writes polish_choice.json
python scripts/mentor_dh_pinn/evaluate_dual_pinn_v2.py --cases 60 --workers 8
python scripts/mentor_dh_pinn/write_two_stage_report.py
```

Literature: Christoffersen, Heston & Jacobs (2009); Bayer & Stemper (2018); Liu, Borovykh, Grzelak & Oosterlee (2019); Horvath, Muguruza & Tomas (2021).
