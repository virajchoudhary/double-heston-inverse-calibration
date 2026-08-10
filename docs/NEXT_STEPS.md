# Next Steps

The unavailable teammate source is no longer an expected dependency. The project proceeds from the independent canonical reimplementation without claiming source equivalence.

## Active mentor-updated v2 sequence

1. Analyze common observed and active moneyness/maturity support across NTPC, CIPLA, INFY, and HDFCBANK.
2. Decide and freeze the replacement representation only through G2.
3. Update and revalidate the synthetic surface contract for the G2-approved representation.
4. Generate the final synthetic research dataset.
5. Train and evaluate the ordinary ANN baseline across predeclared seeds.
6. Build and evaluate the separate PDE/PINN approach after the ANN baseline.
7. Run the frozen chronological real-market evaluation.
8. Compare ANN, PINN, numerical Double Heston calibration, and Standard Heston under one fixed protocol.

Stage A candidate selection is complete: NTPC, CIPLA, INFY, and HDFCBANK are the primaries, with NTPC selected at moderate confidence. The original three-date Power comparison was unresolved; the predeclared five-Wednesday July extension resolved the choice without Bloomberg. The immediate global next milestone is **common-support analysis across the four primaries**. No maturity grid, moneyness grid, feature count, or replacement representation is selected yet.

## Secondary sampler, challenge, and OOD follow-up

The still-valid sampling work remains required but is no longer described as the immediate global pipeline step:

- resolve or formally disposition the four retained challenge pricing-tolerance failures under `NEEDS_SAMPLER_CORRECTION` without dropping rows or tuning ranges post hoc;
- review interior, wide-valid, challenge, noise-test, and OOD ranges with quantitative-finance/domain supervision;
- re-run the deterministic reviewed-sampling audit before including challenge/stress populations in research data;
- expand controlled short-maturity and boundary convergence tests if an approved sampling revision introduces new regions; and
- run the predeclared multi-seed 0%, 0.5%, 1%, and 2% noise experiments.

The normal clean core remains ready only under its current scoped contract. No final 10,000-surface dataset, ANN research training, or PINN result exists. The current 108-grid is rejected as the final unchanged representation, the replacement is open, and `G2 = NOT_PASSED`.
