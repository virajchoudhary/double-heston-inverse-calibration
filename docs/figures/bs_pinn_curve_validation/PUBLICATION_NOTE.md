# Publication checksum audit — 20 September 2026

When packaging these existing artifacts for GitHub, every `files` entry in
OUTPUT_MANIFEST.json was compared with the exact staged bytes. All entries
matched except PLOT_REPORT.md. This mismatch already existed in the local files
before publication; it was not introduced by Git line-ending conversion.

- Original recorded report SHA-256:
  `fd2fb1ff4824ae6901fa7566710209c39eb7d276135c09fceab7a86c921ca73f`
- Published/current report SHA-256:
  `02b4788d86e316505ac5df1b446e2fc3fcab204f65df45303116085f89f6d7c2`

The original manifest is retained unchanged, not retroactively rewritten.
The reason for the older Markdown checksum discrepancy is not established by
this check. The remaining listed artifacts match their recorded hashes; no
prices, plots, metrics, model parameters or checkpoints were altered to resolve
this packaging issue. Consult numerical_checks.json and curve_shape_checks.csv
for machine-readable results, and use the published report checksum above when
verifying this handoff. This note is not a fresh scientific evaluation.
