# Physics-Informed Inverse Calibration of the Canonical Double Heston Model for Stable Option-Surface Parameter Recovery

This folder contains the complete starter project for the **ordinary ANN inverse-calibration baseline**. It is the non-physics neural comparator for the approved capstone. It does not contain a Double Heston PDE residual, a PINN loss, or claimed research results.

> **The current smoke test is infrastructure validation only and is not a research result.**

| Component | Status |
|---|---|
| ANN infrastructure | Complete |
| Automated tests | Passing |
| Smoke test | Passing |
| Double Heston pricer | Awaiting teammate source |
| Research synthetic data | Blocked |
| ANN research training | Not started |
| NIFTY validation | Not started |
| PINN comparison | Not started |

## Project documentation

- [Current status](docs/CURRENT_STATUS.md)
- [Results to date](docs/RESULTS_TO_DATE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Next steps](docs/NEXT_STEPS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Team handoff requirements](docs/TEAM_HANDOFF_REQUIREMENTS.md)
- [Repository audit](docs/REPOSITORY_AUDIT.md)

## Purpose and scope

The ANN maps one fixed-size normalized option surface to ten canonical Double Heston parameters. A future frozen pricing engine will then reconstruct the option surface from the ANN-predicted parameters. The ordinary ANN uses parameter-supervised mean-squared error. A future PINN is a separate model and must not be conflated with this baseline.

The teammate handoff validates a locked prototype built on 11 power-sector stocks. The final approved project must use NIFTY end-of-day European option surfaces, synthetic parameter-recovery validation, chronological real-market validation, and comparisons against traditional Double Heston calibration, this ordinary ANN, a PINN, and Standard Heston.

## Ten outputs

The exact order is:

```text
kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow,
kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast
```

This is eight structural parameters plus two surface-specific initial variance states. All kappa, theta, sigma, and v0 values must be positive. The slow factor must satisfy `kappa_slow < kappa_fast`. Both factors require positive Feller gaps, and correlations must satisfy individual bounds and `rho_slow^2 + rho_fast^2 < 1`.

## Synthetic-first design

Historical calibrated parameters are not treated as labels. Research labels must be known synthetic parameter vectors sampled from teammate-confirmed bounds and priced with the frozen canonical Double Heston engine. Complete surfaces, not quote rows, are assigned to train/validation/test splits.

There are two strictly separated generation paths:

- Research mode requires the real pricing engine and confirmed bounds. It fails clearly if either is missing.
- Smoke-test mode uses `dummy_surface_generator_for_smoke_test`, writes only under `smoke_test` paths, and records `NOT_RESEARCH_DATA`. Genuine loaders reject it unless an explicit override is provided.

## Surface representation

The deterministic grid contains nine log-moneyness values `[-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]`, six maturities `[7, 14, 30, 60, 90, 180]` days, and separate call and put blocks. The flattened normalized-price input therefore has `9 * 6 * 2 = 108` values. Strikes are `spot * exp(log-moneyness)`, maturities are `days / 365`, calls precede puts, and optional masks preserve fixed shapes.

## Dataset fields

The row-level synthetic schema stores surface ID, market inputs, seed, noise level, surface split, log-moneyness, strike, maturity in days and years, option type, generated and spot-normalized prices, mask, data status, and all ten targets. The PyTorch dataset groups these rows into one fixed-length feature vector and one ten-parameter target per surface.

## PDF handoff conversion

The only teammate file available was copied to `handoff/` and extracted with a local PDF parser. `handoff/HESTON_DOUBLE_HESTON_TEAM_CONTEXT.md` is a readable technical companion containing the audit verdict, equations, locked data lineage, historical results, synthetic-recovery findings, identifiability warnings, continuation protocol, and missing files. The original PDF remains the authoritative validated handoff.

## Installation and validation

From the directory containing `ann_inverse_calibration`:

```powershell
python -m pip install -r ann_inverse_calibration/requirements.txt
python -m compileall ann_inverse_calibration
python -m pytest ann_inverse_calibration/tests -q
python -m ann_inverse_calibration.src.run_smoke_test
```

The smoke test creates a small development-only dataset, trains the ordinary ANN for three CPU epochs, computes a loss, saves the best validation checkpoint, verifies `(test_surfaces, 10)` predictions, and writes under `outputs/metrics/smoke_test/`.

## Genuine synthetic generation after source arrival

1. Verify the frozen teammate source and checksum without altering its mathematics.
2. Implement only the thin adapter in `src/pricing_interface.py` and retain shape, finite-value, and option-bound checks.
3. Record teammate-confirmed bounds and provenance in a non-template configuration marked `TEAMMATE_CONFIRMED`.
4. Rerun the teammate pricing and controlled recovery tests.
5. Call research generation; it will sample known valid vectors, price each complete grid, add only declared noise, and keep whole surfaces in one split.
6. Train using train-only target normalization and validation-only checkpoint selection.
7. Report parameter recovery and reconstructed-price errors, retaining all failures.

## Current limitations and missing dependencies

- The frozen validated `double_heston.py` was not available, so real pricing and repricing are blocked.
- Genuine parameter bounds were not available; the template contains nulls and requires teammate confirmation.
- The final NIFTY data contract and chronological dates were not available.
- The smoke-test dummy mapping is not Double Heston pricing and cannot support financial or research conclusions.
- No ANN or PINN research result is claimed.
- Existing historical Heston files and results were not retuned or modified.
