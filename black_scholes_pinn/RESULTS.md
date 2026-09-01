# Trained calibration result

The supplied checkpoint was trained from an initial volatility of `0.35` against
180 synthetic market call quotes generated with reference volatility `0.20`.
The reference volatility was used only to generate quotes and evaluate the final
answer; it was not included in the loss or supplied to the optimizer.

| Quantity | Result |
|---|---:|
| Joint optimizer volatility | 0.1993663 |
| Final robust PDE-calibrated volatility | 0.1999991 |
| Reference volatility (evaluation only) | 0.2000000 |
| Final volatility relative error | 0.000468% |
| Market quote price MAE | 0.00607 |
| Out-of-sample dense price MAE | 0.00555 |
| Out-of-sample dense price RMSE | 0.00794 |
| Test PDE residual RMSE | 0.001820 |

The authoritative machine-readable values are in
`outputs/high_accuracy_run/metrics.json`. The checkpoint is
`outputs/high_accuracy_run/black_scholes_pinn.pt`.
