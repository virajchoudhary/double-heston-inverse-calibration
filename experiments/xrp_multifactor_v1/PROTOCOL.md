# XRP multifactor replication: protocol

Written and frozen before any XRP option surface was fetched or any model fitted. `config.json` is the complete specification.

## Question

Does the BTC result carry over to a different, thinner crypto options market? The BTC result was that per-date calibrated Double Heston prices held-out options better than Single Heston and per-expiry Black-Scholes. ENJ was requested first, but it has no listed options on Deribit, Binance, OKX or Bybit. XRP was chosen instead.

## What is inherited unchanged from BTC

- the pricing engine and the calibration objective
- the optimiser, including its seed
- the parameter bounds (σ ≤ 10 included, so this stays a clean replication)
- the Black-Scholes model (the corrected per-expiry BS_EXPIRY)
- the metrics, the primary endpoint and its statistics
- the Feller choice: both Heston models use the free variant, selected on BTC validation dates

No XRP date is used for any choice, so every selected date is a test date.

## What differs, and why

These changes follow from the market structure, not from any result.

- **Price convention.** XRP options are linear and quoted in USDC as the undiscounted Black-76 price. This was checked on one trade.
- **Trading window.** A full UTC day instead of 2 hours, because trading is thin.
- **Minimum expiries.** At least 4 per date instead of 5, because fewer expiries are listed (6 at present).
- **Price floor.** 0.001 USDC, which is 2 ticks.
- **Shock dates.** Deribit publishes no XRP volatility index, so shocks are defined from spot realised volatility: 10-day realised vol at least 1.5× its 60-day median. The threshold is higher than BTC's 1.2 because realised vol is noisier than implied vol.
- **History.** The data run from 2024-04 onward, because Deribit XRP options history starts around March 2024.

## Integrity

The results are reported whichever way they come out. Nothing is changed after fits are seen. Unusable dates are disclosed and never replaced.

## Amendment 01 (before any option data was fetched)

The first date selection (`artifacts/dates_v0_empty_calm.json`, manifest `manifest_v0.json`) found 11 shock episodes but **0 calm dates**. The fixed range cap of 1.3 was copied from BTC's DVOL rule, and only 1 of 899 days met it, because 10-day realised volatility is much noisier than an implied-volatility index (the median 21-day max/min is 2.38).

The cap is now the 25th percentile of that range over the eligible days, which keeps the calmest quarter. Nothing else changed: the shock dates are identical. No option data had been fetched.
