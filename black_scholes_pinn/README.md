# True inverse PINN for Black–Scholes

This directory is deliberately standalone. It does not import from or modify the
Double Heston implementation in the parent project.

The model learns the normalized call-price field `c(x,tau) = C/K`, where
`x = log(S/K)`. Its loss combines:

1. the Black–Scholes PDE residual, evaluated using automatic differentiation;
2. observed market call prices;
3. the expiry payoff condition; and
4. the low- and high-asset boundary conditions.

The expiry and boundary conditions are explicit PINN losses. The unknown constant
volatility is a bounded, trainable parameter. Training first learns a smooth price
field from quotes and conditions, initializes volatility using only the learned
field's PDE derivatives, then jointly refines the field and volatility with Adam
and full-batch L-BFGS. Finally, it robustly inverts the PDE at pre-specified stable
points (near the money, away from expiry, with sufficient gamma). This staged
design prevents premature inverse-parameter collapse without supervising volatility.

The analytical formula is isolated in `market.py`. It is used only to create a
reproducible synthetic market and to measure out-of-sample accuracy. It is never
called by the neural network or the loss, and the true volatility is never supplied
to the optimizer.

## Run the supplied calibration

From the repository root:

```bash
python3 -m black_scholes_pinn.train
```

Results are written only below `black_scholes_pinn/outputs/high_accuracy_run/`, including
the trained checkpoint, full metrics, market fit, dense predictions, convergence
history, and plots.

## Calibrate a market CSV

```bash
python3 -m black_scholes_pinn.train --market-csv /absolute/path/to/calls.csv \
  --output black_scholes_pinn/outputs/my_market_run
```

Required columns are `spot,strike,tau,rate,dividend,call_price`. This version assumes
one constant rate, dividend yield, and volatility across the supplied surface.

## Tests

```bash
python3 -m unittest discover -s black_scholes_pinn/tests -v
```

The tests verify the analytical reference, boundary/terminal loss construction,
volatility bounds, and that the PDE loss differentiates through both second-order
price derivatives and the inverse volatility parameter.
