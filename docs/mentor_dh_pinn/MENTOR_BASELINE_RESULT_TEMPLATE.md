# Mentor Double Heston PINN Baseline V1 Result Template

## Run identity

- Git SHA:
- Dataset SHA256:
- Parameter surface ID:
- Parameter vector hash:
- Seed: 3407
- Test evaluated once: YES/NO

## Fixed parameter vector

Canonical order:
`[kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow, kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast]`

Values:

## Baseline contract

- Inputs: `S, v_slow, v_fast, tau, K, r, q`
- Output: raw CALL price
- Architecture: 5 x 128 tanh, 67,201 parameters
- Lambdas: `lambda_PDE=lambda_B=lambda_T=lambda_data=1`
- Optimizer / learning rate / weight decay: AdamW / 1e-3 / 1e-6
- Train / validation / test counts:
- Epochs completed / best epoch:

## Numerical health

- Finite gradients:
- NaN/Inf observed:
- PDE autograd status:

## One-shot synthetic test metrics

- Price RMSE:
- Price MAE:
- Normalized RMSE:
- Relative price error mean:
- Relative price error p95:
- PDE residual RMS:
- PDE residual max:
- Terminal RMSE:
- Terminal max error:
- Boundary-low RMSE:
- Boundary-high RMSE:
- Inference runtime total:
- Inference runtime per contract:

## Figures

1. Option Price vs Strike:
2. Option Price vs Maturity:
3. Absolute Pricing Error vs Strike:
4. Training Losses vs Epoch:
5. Validation Price Error vs Epoch:
6. PDE Residual Diagnostics:

## What this proves

Record only conclusions directly supported by the synthetic fixed-parameter
forward-pricing evidence.

## What this does not prove

This does not establish inverse calibration, parameter identifiability,
global recovery, real-market validity, superiority, or representation
freezing.

## Next scientific step

Validation-only lambda-weight study after mentor review; keep test sealed.
