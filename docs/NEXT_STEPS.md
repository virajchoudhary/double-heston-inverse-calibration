# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

1. Review the feature branch, engine equations, correlation-convention note, tests, and saved validation evidence.
2. Independently benchmark representative prices against a second implementation or adaptive quadrature, without using the new regression fixture as its own oracle.
3. Review the provisional hard and empirical ranges with domain supervision; record new provenance before any change.
4. Expand controlled convergence and stress tests near short maturities and valid constraint boundaries.
5. Run predeclared multi-seed recovery experiments at 0%, 0.5%, 1%, and 2% noise.
6. Freeze the reviewed generation configuration and generate immutable train, validation, and test surfaces.
7. Train the ordinary ANN across multiple seeds; retain all parameter and repricing failures.
8. Evaluate all ten parameters and reconstructed prices without treating low price error as unique recovery.
9. Define the NIFTY EOD data contract, quote filters, masks, rates/dividends, and chronological split dates.
10. Perform chronological NIFTY validation.
11. Build and evaluate the separate PINN only after the ordinary ANN baseline is complete.
12. Compare ANN, PINN, numerical Double Heston calibration, and Standard Heston under one fixed protocol.

The exact next action is **independent review and external numerical benchmarking of `feat/double-heston-engine` before expanding synthetic generation or starting ANN research training**.
