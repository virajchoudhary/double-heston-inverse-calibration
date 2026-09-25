# SOL multifactor replication: protocol

Written and frozen before any SOL option surface was fetched or any model fitted.

This is identical to `../xrp_multifactor_v1`, including its pre-fetch calm-rule amendment, with three differences:

- **Instruments:** SOL_USDC options.
- **Spot series:** the SOL_USDC perpetual.
- **Price floor:** 0.2 USDC, which is 2 ticks of the current 0.1 USDC tick.

Deribit SOL_USDC option history starts around March 2024, so dates begin in 2024-04.

## Inherited from BTC and XRP

- engine, bounds, optimiser and seed
- BS_EXPIRY as the Black-Scholes model
- `feller_free` for both Heston models
- metrics, endpoint and statistics

Every selected date is a test date. The primary endpoint is shock dates under design B, with the episode as the unit.

The results are reported whichever way they come out. Nothing is changed after fits are seen. Unusable dates are disclosed and never replaced.
