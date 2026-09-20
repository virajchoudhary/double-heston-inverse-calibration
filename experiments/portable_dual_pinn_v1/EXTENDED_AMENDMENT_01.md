# Pre-training correction: stratify the L-BFGS labels

The initial extended run was terminated during reference-data generation,
before any training process was created. Its original manifest remains in
extended/ (the only artifact in that directory). No extended evaluation
results existed when this correction was made.

The synthetic builder stores 96 consecutive labels per parameter case. A
512-label prefix, used in the completed pilot, spans only six of 64 cases.
This is an unintended mismatch between the fine-tuning subset and the full
domain, not evidence of a flawed PDE. Its effect on the pilot is not quantified.
The completed pilot remains unmodified and is not certified as definitive.

For the new larger run, choose two evenly spaced labels per training case,
giving 512 labels across ALL 256 cases, with an executable coverage assertion.
Both architectures get the identical subset. The 128 collocation rows already
have independently randomised parameter-case assignments, not grouped order;
those remain unchanged. No labels from development are used.

All other EXTENDED_PROTOCOL.md settings remain unchanged. Corrected artifacts
go to extended_stratified/. This is still development work, not an untouched
market test. Both original and corrected manifests are retained.
