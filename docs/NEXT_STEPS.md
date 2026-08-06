# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

1. Review `NEEDS_BOUNDS_REVIEW` with quantitative-finance/domain supervision.
2. Decide interior constraint margins and the proportion assigned to explicitly labelled boundary-near challenge sampling.
3. Review economically plausible training, noise-test, and out-of-distribution ranges; record provenance for every approved change.
4. Re-run the deterministic bounds audit and require a new freeze decision before generating a research dataset.
5. Expand controlled convergence and stress tests near short maturities and valid constraint boundaries if the revised sampling design introduces new regions.
6. Run predeclared multi-seed recovery experiments at 0%, 0.5%, 1%, and 2% noise.
7. Only after a `READY_FOR_SYNTHETIC_GENERATION` gate, generate immutable train, validation, and test surfaces.
8. Train the ordinary ANN across multiple seeds; retain all parameter and repricing failures.
9. Evaluate all ten parameters and reconstructed prices without treating low price error as unique recovery.
10. Define the NIFTY EOD data contract, quote filters, masks, rates/dividends, and chronological split dates.
11. Perform chronological NIFTY validation.
12. Build and evaluate the separate PINN only after the ordinary ANN baseline is complete.
13. Compare ANN, PINN, numerical Double Heston calibration, and Standard Heston under one fixed protocol.

The exact next action is **financial/domain review and revision of `configs/parameter_sampling_REVIEWED.yaml`, followed by a repeat audit and freeze decision**. Do not begin large synthetic generation or ANN training while the gate is `NEEDS_BOUNDS_REVIEW`.
