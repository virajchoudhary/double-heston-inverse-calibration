# NTPC canonical-160 reproduction correction

## Correction status

Root cause: **ARTIFACT_PROVENANCE_MISMATCH**.

The reviewed canonical-coordinate `max_nfev=160` solution artifact is internally consistent. Its 12 vectors independently produce 66 unique unordered distances, median `0.35733879424203197`, maximum `0.5641491074467359`, maximum distance from best `0.4910854918863381`, 11 materially displaced solutions, and 7 complete-linkage clusters.

The later optimizer-cap replay used a different numerical input representation. It produced median `0.3169421650862818`, maximum `0.5641666171316797`, maximum distance from best `0.4914112891055993`, 11 materially displaced solutions, and 7 clusters. These values describe that replay artifact; they are not corrections to the reviewed baseline.

## Exact defect

PR #13 constructed `T` in memory as binary floating-point `DTE / 365` and calibrated before writing `selected_options.csv`. PR #14 and the later replay loaded the decimal CSV representation without reconstructing derived pricing inputs. Every calibration row's loaded `T` differs from the in-memory value by at most `8.326672684688674e-17`.

That difference is economically negligible but not trajectory-neutral for a capped, ill-conditioned nonlinear least-squares problem. Under a tight range-scaled RMS replay tolerance of `1e-4`, 5 starts are tolerance-only and 7 have different optimizer endpoints; the first is start 3. Start 6 is the dominant divergence and proves causality:

- reconstructed in-memory `T`: reviewed start-6 vector reproduced with maximum parameter difference `2.22e-16`;
- CSV-loaded `T`: later replay start-6 vector reproduced with maximum parameter difference `9.02e-17`;
- replacing only CSV `T` with `DTE/365` restored the reviewed basin;
- replacing any other differing serialized field alone did not.

Three repeated current-runtime start-6 fits on the CSV input were byte-identical. The defect is therefore not stochastic optimizer behavior under identical inputs. It is an input-artifact provenance mismatch.

These counterfactuals are reproducible through `scripts/run_ntpc_canonical160_causal_probe.py`. The bounded runner verifies the endpoint artifact hashes, rebuilds the selected rows from their hash-recorded raw NSE source, executes only canonical start 6 at the frozen `max_nfev=160` contract, records every single-field probe and three CSV repeats, and writes the ignored evidence artifact `canonical160_causal_probe.json`. The tracked forensic manifest seals that artifact's hash. No 320-budget fit is part of this correction or probe.

## Provenance chain

### PR #13 pilot

- Commit: `dd539150898bf5ca4d168c5dba3f3a33c69628e2`.
- Script SHA-256: `1E641C02A8493177F64B6FBFC7FABAF28AB3725E61A1AFE81017D95C6301F908`.
- The tracked pilot manifest SHA-256 is `A19800FBEDBF00F8226A3F413A17D861420E82064D94384C9BA521FD2F1B9ADC`.
- It binds the ignored `double_heston_multistart.csv` artifact to SHA-256 `4E092F2BEC5F53033E61EFB1D2B2D761C9D3AB8F72F17F33D6E989946FC1EB70`.
- The pilot report used its stored best/stability artifacts; it did not publish the later complete-linkage median.

### PR #14 reparameterization comparison

- Commit: `f9e1155ffae64492ace2899efb5221d1df1e2bf3`.
- It hash-verified and loaded the PR #13 12-start artifact, then computed `baseline_pairwise_distances.csv` using full configured ranges, Euclidean distance divided by `sqrt(10)`, unique `i < j` pairs, and the full near-equivalent population.
- The tracked pairwise CSV SHA-256 is `C5568814B0496E8FA4A8F8485D8A345193FA37F8F21AC08CAE6F5136545F2B29`.
- The tracked `stability_comparison.json` SHA-256 is `9EB702A24B978959BECC159B4C2CFCABBE81B8A0CCECC887A460AE57E15192F3` and stores `0.35733879424203197`.
- The generated report table renders that stored/recomputed field as `0.357338794`.

### Invalid optimizer-cap replay

- Branch: `feat/ntpc-dh-optimizer-cap-sensitivity`; no commit or push.
- Canonical-160 replay artifact SHA-256: `CF148D54639EA194E620BABB5E6CF741A91AB77C269A2C1E3BA3CFA25B33926E`.
- It loaded `selected_options.csv` directly and computed `0.3169421650862818` from its materially different 12-vector population.
- The invalid replay remains preserved and must not be committed or used as a corrected baseline.

## Rejected alternative explanations

- **Metric reduction:** rejected. Independent calculation reproduces each disputed median from its own vectors. Diagonal inclusion, full symmetric flattening, duplicate pairs, success-only filtering, distance-from-best, row order, and cluster filtering reproduce neither disputed value.
- **Start population:** rejected. All four intended cells use start IDs 0–11 and the canonical start-population digest `3AC1C30FF1B5416987D2103EA70B9262BBB8B4991F18F7A06C98E3A41C86ABA1`.
- **Code or bounds drift:** rejected. The pilot script hash matches its manifest; relevant source and bounds did not change through PR #14/main.
- **Runtime drift:** rejected as primary cause. The current runtime matches the recorded Python/NumPy/SciPy/pandas/matplotlib contract, and controlled repeats are deterministic for fixed CSV input.

## Correction

Future numerical replay from CSV must reconstruct derived pricing inputs from stable primitives before optimization:

- `T = DTE / 365`;
- `discount_factor = 1 / (1 + y*T)`;
- `continuous_rate = -log(discount_factor) / T`;
- `futures_implied_carry = continuous_rate - log(F/S) / T`.

`src/ntpc_pricing_input_contract.py` implements this contract. The PR #14 runner now applies it immediately after loading `selected_options.csv`. Historical reviewed evidence is not modified.

## Scientific impact

- NTPC pilot conclusion changed: **NO**.
- PR #14 `INSUFFICIENT` classification changed: **NO**.
- Materially displaced count changed: **NO** (`11` under both populations).
- Cluster count changed: **NO** (`7` under both populations).
- Slow/fast allocation ambiguity changed: **NO**.
- Heston versus Double Heston comparison changed: **NO**.
- G2/global-ambiguity conclusions changed: **NO**; those use separate experiments and evidence.

This is a numerical replay/provenance correction, not a scientific-conclusion change. The already-predeclared 160-versus-320 sensitivity experiment must be rerun separately from the corrected frozen baseline only after this correction is reviewed.
