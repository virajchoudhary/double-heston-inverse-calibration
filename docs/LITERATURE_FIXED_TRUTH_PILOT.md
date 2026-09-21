# Literature fixed-truth Double Heston PINN pilot

This is a **separate development experiment**, not a change to frozen V2 B0/B1/B2.
The goal is to recover a known fixed ten-parameter vector from its synthetic
option surface. Ground truth is used in generation and final scoring only.
Hard-coding the answer into the inverse optimizer would not demonstrate recovery.

## Provenance and cases

Source: Kyriakou, Brignone and Fusai (2024), *Unified moment-based modelling of
integrated stochastic processes*, Operations Research 72(4), 1630-1653,
[DOI 10.1287/opre.2022.2422](https://doi.org/10.1287/opre.2022.2422).
[Accepted manuscript](https://openaccess.city.ac.uk/id/eprint/29416/1/manuscript_cro.pdf),
Table 2, printed p.22. Here lowercase paper `v` means sigma, while `V(0)` is
initial variance. Values are NOT squared again. The paper's DH1 call price
at S=K=100, r=3%, T=1 is 26.9504 (Table 3, p.27); our teacher reproduces it
as 26.95035571889789. Table layout was visually checked before transcription.

Canonical ordering in this repository is **slow first**, fast second. A complete
factor swap is only a relabelling; it must include all five values of each factor.

| Case | Slow: kappa, theta, sigma, rho, v0 | Fast: kappa, theta, sigma, rho, v0 | Status |
|---|---|---|---|
| published_DH1 | .9, .1, .36, -.5, .36 | 1.2, .15, .2, -.5, .2 | Exact published set |
| derived_faster_factor | .9, .1, .36, -.5, .36 | 3.6, .15, .2, -.5, .2 | Study-defined: fast kappa x3 |
| derived_lower_volvol | .9, .1, .18, -.5, .36 | 1.2, .15, .1, -.5, .2 | Study-defined: both sigmas x0.5 |

The two derived cases are **not additional published calibrations**. They test
factor time-scale separation and weak volatility-of-variance around a published
anchor. The multipliers are experiment design choices, fixed before recovery
results, not numbers attributed to the authors. These sets are not calibrated
Nifty 50 or ADANIPOWER market parameters and contain no NSE observations.

The paper's DH2 is retained unchanged in the config, but excluded from this
strict-Feller experiment because both factors violate 2*kappa*theta > sigma^2.
This is an incompatibility with our experiment, not proof that DH2 is an invalid
stochastic model. No published values were clipped to make them fit the old domain.

## Data and isolation

- Each truth is fixed, not drawn at random. Only initialization and LHS coordinates
  use pseudo-random seeds.
- 246 fitting quotes per case: 41 S/K ratios uniformly spaced over [0.5,1.5],
  crossed with six synthetic expiries [30,60,90,180,365,730] days.
- 240 held-out quotes per case: interleaved midpoint ratios at the same six
  maturities. No coordinate overlap. This measures strike interpolation, not
  unseen-expiry, market or date generalization.
- tau = days/365; 30 days is about .0822 years, not 1 year. Long expiries here
  are synthetic research cases, not claims about NSE single-stock listing rules.
- Constant K=100, r=.03, q=0. Store forward x=log(S/K)+(r-q)*tau and normalized
  call c=C/(K*exp(-r*tau)). Actual currency price is c*K*exp(-r*tau).
- Quote NPZ columns: `xt[:,0]` forward log-moneyness, `xt[:,1]` tau years,
  `price` normalized call and `iv` annualized Black implied volatility.
- 18,000 four-dimensional LHS collocation points over forward x [-1,1], each
  state variance [.0001,1], tau [7/365,2], uniform in these coordinates. They
  carry no price, IV or true-parameter labels. Adam uses 128 per step; L-BFGS
  uses a fixed 512-point subset. The pool is not 18,000 observations per step.
- Teacher: existing float64 Double Heston Fourier inversion, 128 nodes, checked
  against 96 nodes. Reject the experiment if any quote fails numerical or IV
  validity gates; no silent dropping, filling, clipping or invented market data.
- `prepare` writes truth separately. The `train` function reads only the public
  protocol, fitting quotes, and collocation array. The parameter vector is NOT
  a supervision label. Holdout data and true values are read in `score` only.

## PINN and loss

Reuse the existing regular PyTorch **full Double Heston price-PDE** residual and
smooth 5x128 tanh network, trained jointly with ten latent variables. This is a
new joint inverse run, **not** reuse of a pretrained B0 checkpoint, not the older
factor-convolution variant, and not an exact-pricer calibration optimizer.

The network represents positive implied variance as a multiplicative correction
to expected average model variance; its price uses the Black formula and has an
exact terminal payoff. This inherited output representation is not the original
raw-price Softplus/Fourier-feature architecture.

In forward-normalized coordinates the full pricing residual is

    R = c_tau - .5*(v_s+v_f)*(c_xx-c_x)
        - sum_i [kappa_i*(theta_i-v_i)*c_vi
                 + .5*sigma_i^2*v_i*c_vivi + rho_i*sigma_i*v_i*c_xvi]

The existing implementation divides R by Black_c_w times
`sum(v_i)+expected_average_variance` for numerical scaling; derivatives retain
network and latent-parameter gradients.

    loss = mean(((IV_PINN-IV_target)/.01)^2)
         + mean(((c_PINN-c_target)/.01)^2)
         + mean((scaled_PDE_residual/.01)^2)
         + .1*mean(relu(-normalized_convexity)^2)

Prices and IV are two weightings of the same observations, **not independent
information**. They do not mathematically resolve non-identifiability.
Terminal payoff and strict Feller/order constraints are enforced by construction,
so there is no separate parameter-label MSE, Feller penalty, or terminal penalty.

Pilot-only smooth bounds: slow kappa [.02,5], fast kappa > slow+.01 and <20;
each theta [.001,1], each v0 [.0001,1]; |rho|<.999;
sigma/sqrt(2*kappa*theta) in (.01,.999). The old V2 decoder is untouched.
All ten quantities can move; sigma's admissible range depends on kappa and theta.

## Fixed protocol and interpretation

Two generic initialization seeds (17,43), identical seed-specific guesses across
all cases. 1,000 Adam steps then at most 100 L-BFGS iterations per fit. Report the
fixed final checkpoint for **all six fits**, not the seed closest to truth.
This bounded pilot does not claim optimization to convergence or robust recovery.

Scoring reports neural and independent Fourier-repriced price/IV errors, alongside
all ten parameter errors. The Fourier pricer is used after training, with no
gradient refinement. A good neural surface can coexist with wrong parameters.
Success requires all eight positive parameters within 5% relative error and
both correlations within .05 absolute error; no gates are relaxed after results.
Shortfall is retained as a failure, not relabelled as recovery.

To reproduce, use the project Python environment from the repository root:

```sh
python scripts/mentor_dh_pinn/literature_fixed_truth_pilot.py prepare --data outputs/NEW_UNIQUE_DIRECTORY
python scripts/mentor_dh_pinn/literature_fixed_truth_pilot.py train --data outputs/NEW_UNIQUE_DIRECTORY --case published_DH1 --seed 17
# Repeat train for all three case IDs and both frozen seeds.
python scripts/mentor_dh_pinn/literature_fixed_truth_pilot.py score --data outputs/NEW_UNIQUE_DIRECTORY
python -m pytest -q tests/test_literature_fixed_truth_pilot.py tests/test_regular_pinn_torch_physics.py tests/test_torch_pricer.py
```

Existing output directories are never overwritten. Source/data hashes, truth,
initial values, all training logs, checkpoints and per-parameter results are
retained under the experiment directory. Published-case numerical accuracy and
code checks do not establish statistical generalization or parameter identifiability.
