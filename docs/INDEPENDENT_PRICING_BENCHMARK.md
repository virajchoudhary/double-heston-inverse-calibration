# Independent Double Heston Pricing Benchmark

Status date: 06 August 2026

## Method

`src/double_heston_reference.py` is a deliberately slow validation reference. It independently implements the two-factor affine characteristic function and evaluates the two Fourier probability integrals with `scipy.integrate.quad` using `epsabs=1e-10`, `epsrel=1e-10`, and `limit=500`. Calls are evaluated from the integrals; puts use put-call parity.

The reference module may reuse the shared parameter order and constraint validator, but it does not import or call `src/double_heston.py` or `src/pricing_interface.py`. The production module does not import the reference. The benchmark runner is the only module that imports both.

Integration warnings, exceptions, error estimates, evaluation counts, subdivisions, and reliability status are retained. Prices are not clipped, and failed integrations are not replaced.

## Frozen cases and acceptance rules

The fixture contains 36 hand-authored controlled cases: 18 calls and 18 paired puts. It covers short, medium, and long maturities; ITM, ATM, and OTM options; low and high total variance; weak and strong negative skew; moderate Feller-boundary proximity; safely interior parameters; five spot levels; six rate values; and five dividend-yield values.

The immutable case-level tolerance for both production node counts is:

```text
abs(production - reference) <= 2e-5 + 2e-6 * abs(reference)
```

No-arbitrage tolerance is `1e-8`; paired put-call parity tolerance is `1e-10`. A benchmark pass requires every reference integration to be reliable and every 64-node and 96-node case, no-arbitrage check, and parity check to pass. Aggregates cannot hide an individual failure.

## Results

| Result | 64 nodes | 96 nodes |
|---|---:|---:|
| Cases passing | 36 / 36 | 36 / 36 |
| RMSE vs reference | `5.458369984817452e-13` | `4.2228670813888515e-12` |
| MAE vs reference | `5.18369298103178e-13` | `4.0641980521745795e-12` |
| Maximum absolute difference | `8.100187187665142e-13` | `5.6985527407960035e-12` |

The maximum relative difference across non-negligible reference prices was `1.064640411670892e-9`. Reference integration failures, unreliable results, and warnings were all zero. Reference, 64-node, and 96-node no-arbitrage failure counts were all zero. Maximum absolute parity error was `7.105427357601002e-15` for each method.

On the final primary verification run, total reference runtime was about `3.232` seconds, versus `0.0430` seconds for 64 nodes and `0.0449` seconds for 96 nodes. The adaptive reference was therefore about 75 times slower than the 64-node production path and 72 times slower than the 96-node path on this small benchmark.

## Interpretation and limitations

The production and independent numerical methods agree on the frozen controlled cases, and no unexplained pricing discrepancy remains in this benchmark. Agreement is necessary evidence, not proof that either implementation is universally correct. The benchmark does not compare with the unavailable teammate code and does not validate calibration or performance on real NIFTY data.

Detailed case rows, grouped errors, failures, quadrature comparisons, and runtime tables are stored under `outputs/double_heston_benchmark/`.
