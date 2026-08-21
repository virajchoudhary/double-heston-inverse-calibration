# Architecture

The ordinary ANN baseline is a parameter-supervised, non-physics neural network. It intentionally contains no PDE residual.

```mermaid
flowchart LR
    A["Option surface"] --> B["Deterministic call and put grid"]
    B --> C["108 spot-normalized price inputs"]
    C --> D["Ordinary PyTorch MLP"]
    D --> E["Ten Double Heston outputs"]
    E --> F["Independent canonical Double Heston repricer"]
```

## Fixed surface contract

The grid combines nine log-moneyness coordinates, six maturity coordinates, and separate call and put blocks: `9 x 6 x 2 = 108` normalized inputs. Complete surfaces remain together in exactly one train, validation, or test split.

## Exact parameter order

1. `kappa_slow`
2. `theta_slow`
3. `sigma_slow`
4. `rho_slow`
5. `v0_slow`
6. `kappa_fast`
7. `theta_fast`
8. `sigma_fast`
9. `rho_fast`
10. `v0_fast`

These outputs represent eight structural parameters and two surface-specific initial variance states.

## Declared constraints

- Positive `kappa`, `theta`, `sigma`, and `v0` for both factors
- `kappa_slow < kappa_fast`
- `2 * kappa_slow * theta_slow - sigma_slow^2 > 0`
- `2 * kappa_fast * theta_fast - sigma_fast^2 > 0`
- Each correlation strictly inside `(-1, 1)`
- `rho_slow^2 + rho_fast^2 < 1`

## Research boundary

`src/pricing_interface.py` routes research generation and repricing to `src/double_heston.py`. The development dummy remains available only through the explicitly labelled smoke-test path and is never an implicit fallback. Full ANN research training has not started.

The repository also contains a PINN infrastructure path for inverse calibration. Unlike the ANN baseline, it uses a differentiable torch repricer that mirrors the canonical Double Heston surface contract and combines parameter supervision with full-surface repricing consistency. This is infrastructure, not a dated research-training result, and it does not change the historical G2 status recorded in the dated status documents.
