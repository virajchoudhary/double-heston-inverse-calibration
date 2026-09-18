# ETH multifactor replication: protocol

Written and frozen before any ETH option surface was fetched or any model fitted.

This is identical to `../btc_multifactor_v1`, with these differences:

- **Options:** Deribit ETH options, quoted in ETH, which is the same convention as BTC and uses the same data code.
- **Shock dates:** taken from ETH DVOL, using the BTC rule verbatim with a new seed for drawing calm dates.
- **No validation stage:** every date is a test date. Both Heston models use `feller_free`, inherited from the BTC validation.
- **Black-Scholes:** the corrected per-expiry BS_EXPIRY.

## Primary endpoint

Shock dates, design B, with the episode as the unit. The test is the one-sided Wilcoxon over episodes with p < 0.05, **and** a positive lower bound from the episode-cluster bootstrap.

## Integrity

The results are reported whichever way they come out. Nothing is changed after fits are seen. Unusable dates are disclosed and never replaced.
