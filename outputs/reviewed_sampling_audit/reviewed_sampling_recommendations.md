# Reviewed sampling audit

- Deterministic latent-coordinate LHS is not physical-space LHS after conditional transforms.
- Fixed candidate populations retain every rejection; no accepted-row refill occurs.
- Challenge is excluded from ordinary training unless explicitly opted in.
- OOD is evaluation-only and uses disjoint high-tail kappa_fast support.
- Every declared parameter envelope is checked against every generated candidate.
- Raw noise diagnostics are retained in priced_surface_metrics.csv with evidence_kind=raw_noise_diagnostic; they use no clipping, projection, or dropped rows, do not transfer clean-price validity, and do not participate in the READY gate.
