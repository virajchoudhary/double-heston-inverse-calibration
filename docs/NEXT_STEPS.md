# Next Steps

1. Receive the frozen validated Double Heston implementation.
2. Verify checksums and imported helper modules.
3. Confirm the exact pricing interface.
4. Confirm parameter bounds and their provenance.
5. Integrate the pricing adapter.
6. Reproduce controlled Double Heston recovery tests.
7. Generate genuine labelled synthetic surfaces.
8. Create immutable train, validation, and test splits.
9. Train the ordinary ANN.
10. Evaluate all ten recovered parameters.
11. Run 0%, 0.5%, 1%, and 2% noise experiments.
12. Reprice surfaces using ANN-predicted parameters.
13. Repeat training across five seeds.
14. Create the NIFTY EOD surface dataset contract.
15. Perform chronological NIFTY validation.
16. Compare ANN, PINN, numerical Double Heston, and Standard Heston.

Each research stage must retain parameter vectors, surface IDs, seeds, noise levels, masks, split identities, failures, and relevant source/model hashes.
