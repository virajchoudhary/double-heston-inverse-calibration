# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

1. Resolve the four retained challenge pricing failures under `NEEDS_SAMPLER_CORRECTION` without dropping rows or tuning ranges post hoc.
2. Review the interior, wide-valid, challenge, noise-test, and OOD ranges with quantitative-finance/domain supervision.
3. Review economically plausible training and out-of-distribution ranges; record provenance for every approved change.
4. Re-run the deterministic reviewed-sampling audit and require a new freeze decision before generating any combined or challenge/stress-inclusive research dataset.
5. Expand controlled convergence and stress tests near short maturities and valid constraint boundaries if the revised sampling design introduces new regions.
6. Run predeclared multi-seed recovery experiments at 0%, 0.5%, 1%, and 2% noise.
7. Only after a global `READY_FOR_SYNTHETIC_GENERATION` gate, generate combined or challenge/stress-inclusive train, validation, and test surfaces; the separately scoped clean core follows its own readiness contract.
8. Train the ordinary ANN across multiple seeds; retain all parameter and repricing failures.
9. Evaluate all ten parameters and reconstructed prices without treating low price error as unique recovery.
10. Define the NIFTY EOD data contract, quote filters, masks, rates/dividends, and chronological split dates.
11. Perform chronological NIFTY validation.
12. Build and evaluate the separate PINN only after the ordinary ANN baseline is complete.
13. Compare ANN, PINN, numerical Double Heston calibration, and Standard Heston under one fixed protocol.

The global next action remains **sampler correction for the retained challenge
pricing failures, followed by financial/domain review and a repeat audit**. The
separate normal-core milestone is now scientifically ready: implement and test
the reviewed-core generator as the next milestone. The generator module does
not yet exist, so no generation command is currently available or runnable.
No large synthetic generation or ANN training may be claimed. Challenge/OOD/noise remain
separate populations and the global gate remains `NEEDS_SAMPLER_CORRECTION`.
