# Extended development run — separate from pilot

Written after reviewing the pilot; therefore development, not confirmatory.
The pilot reduced price/PDE errors but worsened mean IV RMSE, failing its gate.
Twelve low-vega development targets were repriced using adaptive quadrature at
1e-12 tolerance: price differences were below 1e-13 and the largest observed
IV shift was about .000187 in volatility units. This small convenience sample
does not prove all teacher IVs reliable. It does not explain the broad errors.

Change ONLY synthetic sample size and optimisation budget: 256 training
parameter cases x 96 states, 64 new development parameter cases x 96 states,
4,000 Adam steps followed by 100 L-BFGS iterations. Same domain, architecture,
feature map, loss weights, batches, 18,000 collocation points and seeds 17/43.
Data seeds train 97001 / development 97002 / collocation 97003. Collocation
parameters are drawn only from new training cases. Development parameter vectors
must be disjoint from both old splits and the new training cases.

Both candidates restart from scratch. The single-branch model remains the
fair control; both are Double Heston forward PINNs, NOT SH versus DH models.
No market observations or old development labels train either candidate.
All four fits are reported, with price, IV and PDE errors. Same promotion rule:
mean seed price and IV RMSE both lower, mean PDE RMSE at most 10% worse.
Also report each seed separately, train errors and convexity violations.
No extension or retry conditional on a disappointing result. Frozen artifacts
are preserved under extended/; no pre-existing results are overwritten.
Even a passed gate is only promising development evidence, not a market win,
proof of recovering ten parameters, or generalisation to all European markets.
