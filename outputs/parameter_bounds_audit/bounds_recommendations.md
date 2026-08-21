# Parameter bounds audit recommendations

- REVIEW: provisional uniform sampling has acceptance rate 0.5552; it is not appropriate without further evidence.
- REQUIRE_FINANCIAL_REVIEW: accepted-valid near-any-boundary rate is 0.3267 (907/2776); no real-market calibration validates these ranges.
- SPLIT_SAMPLING_RANGE: accepted-valid near-Feller rate is 0.0706 (196/2776); retain boundary-near cases as challenge sampling rather than silently mixing them.
- REVIEW: 17 priced parameter-pair comparisons met surface RMSE <= 1e-3 and normalized parameter distance >= 0.10; this is a sampling-design exposure, not a statistical-identifiability claim.
- KEEP: surface validity had 0 finite-price failures, 0 no-arbitrage failures, 0 call-monotonicity failures, 0 put-monotonicity failures, and 0 convexity failures across 21000 prices.
- KEEP: strict positivity, slow/fast ordering, strict Feller, individual correlation, joint-disk, and hard numerical-safety constraints remain enforced.
- REVIEW: implied-volatility diagnostics are unavailable because this repository has no validated IV inversion.
