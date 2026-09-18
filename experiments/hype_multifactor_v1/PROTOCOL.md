# HYPE multifactor replication: protocol

Written and frozen before any HYPE option surface was fetched or any model fitted. `config.json` is the complete specification.

## Question

Does the BTC result carry over to Hyperliquid (HYPE) options? The BTC result was that per-date calibrated Double Heston beats Single Heston and per-expiry Black-Scholes on held-out options.

## Inherited unchanged from BTC and XRP

- the engine and the objective
- the optimiser and its seed
- the parameter bounds
- BS_EXPIRY as the Black-Scholes model
- the metrics
- `feller_free` for both Heston models (chosen on BTC validation)
- the data pipeline and filters from XRP, except the price floor

## What differs, and why

These changes follow from the market structure and were decided before any option data was fetched.

- **Short history.** Deribit HYPE options were listed on 2026-06-22. That leaves under three months, too short for a sample of shock episodes. **Every calendar day from 2026-06-23 to 2026-09-16 is therefore a test date**, and none is chosen.
- **Primary unit: the ISO week.** Days in the same week share a market state and are not independent. The one-sided Wilcoxon test and the cluster bootstrap both run over weeks.
- **Shock label (secondary only).** A day counts as a shock day if it falls 1–8 days after a realised-volatility onset. The onset uses the same RV rule as XRP, computed on Hyperliquid's own HYPE perpetual daily closes (history from December 2024).
- **Price floor.** 0.004 USDC, which is 2 ticks.

## Integrity

- The results are reported whichever way they come out.
- Nothing is changed after fits are seen.
- Unusable dates are disclosed and never replaced.
