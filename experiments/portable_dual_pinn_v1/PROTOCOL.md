# Portable maturity-dual PINN: development pilot

This is a NEW research variant, not an amendment of a frozen BTC or NIFTY test.
The old market results have already been seen. They cannot be an untouched
test of this new architecture. No existing checkpoint, fit, split or result is
overwritten. No claim of improved market accuracy or parameter identification
will follow from this pilot.

## Architecture

Two smooth maturity specialists (3 x 64 tanh each) share the same 24
dimensionless input features and average-variance/Black price transformation.
Their scalar log-IV corrections blend using g = sigmoid(2 log(tau/(90/365))).
Both branches and the gate are differentiated in the Double Heston PDE.
Compare with one 3 x 96 tanh branch using the SAME features and loss. Record
parameter counts, timing and all seeds. The comparison is not against the old
NIFTY 5 x 256 checkpoint trained on a different, much narrower domain.

This is a forward pricing surrogate, unlike the legacy fixed-grid inverse
DualPINN. It does not directly recover ten parameters. Asset independence comes
from log-forward moneyness and currency normalization, not universal learned
coverage. Calls and puts at arbitrary strikes/expiries are supported for EUROPEAN
exercise only, within the trained domain. New payoff physics requires a new PDE.
Ten parameters remain supplied inputs. No exact pricer is used at inference.

## Fixed pilot settings (before the first training run)

- Synthetic training: 64 independent parameter cases x 96 LHS coordinates.
- Development: 16 entirely different parameter cases x 96 LHS coordinates.
- Coordinates: x in [-1,1], tau log-uniform 3/365 to 2 years.
- Parameter sampling: independent LHS in 10 dimensions; slow kappa [.3,3],
  fast kappa = slow + log-uniform gap [2,15]; theta and v0 [.005,1.5]
  log-uniform, rho [-.9,.9], sigma = eta sqrt(2 kappa theta) with
  log-uniform eta [.15,2]. Feller violation is allowed and disclosed.
- Distinct seeds: train 96001, development 96002, collocation 96003.
- 18,000 LHS collocation points; variance states and structure sampled from
  TRAINING parameter cases only. No development labels in the training loop.
- Optimisation: seeds 17 and 43, Adam 1,000 steps, cosine LR .001 to .00001,
  batches 128 teacher / 32 PDE, then L-BFGS max_iter 30 using the first
  512 training labels and 128 collocation points. No early stopping or retries.
- Loss: price MSE / .01² + IV MSE / .05² + .01 PDE MSE
  + .01 mean squared negative-convexity penalty. Terminal payoff is exact.
- Every price is retained. Undefined/numerically saturated teacher IVs are
  excluded ONLY from IV loss and counted. 96/128-node disagreement >1e-7
  triggers independent adaptive reference pricing; failures stop, not resample.
- Report all-seed mean price RMSE and IV RMSE and individual seed results.
  Promote as promising only if both mean errors improve AND development PDE
  RMSE is no more than 10% worse than the single-branch control. This is a
  development criterion, NOT a statistical test or deployment certification.

## Market integrity caveat discovered during inspection

The existing BTC pipeline calls solve_k(price, trade_iv, tau, cp), aggregates
implied forwards over all trades by expiry, and only THEN creates the quote
holdout flags. Thus held-out target information participates in covariates,
put-call conversion and filtering. The stored ranking can be reproduced, but
must not be presented as a demonstrated target-independent/leakage-free test.
The magnitude of this issue has not been measured. Currency conversion also
needs independent validation against actual exchange forward/index fields.

Before a new market test, obtain independently timestamped forwards, discounts
and settlement conversions; freeze a new chronological date set; split before
fitting data-dependent transformations; calibrate only calibration quotes.
Keep this pilot and all old market dates classified as development exposure.

## Run

From repository root using the project Python environment:

    python experiments/portable_dual_pinn_v1/run_pilot.py
    python experiments/portable_dual_pinn_v1/audit_btc.py

Both refuse to overwrite their output directory. Generated artifacts include
the protocol/code hashes, source data, both models/seeds, history and metrics.
