# G8 Readiness Review Package

## Identities

- Frozen protocol: `7eecc7188c54f9d4505d32ccf5c51069a4c3a97c`
- Readiness base reviewed: `746060bc32a79d4d1a15ba2de1c27e7334fc8103`
- Current review branch: `research/g8-acquisition-readiness`
- Config SHA-256: `d6107bf7c1b5404e59130d99b5e0f12aef4352c1452b83235187caa7628d4f37`

## Source Map

| Concern | Entry points |
|---|---|
| Frozen boundaries | `configs/g8_final_real_market.yaml`, `docs/G8_FINAL_REAL_MARKET_PROTOCOL.md`, `scripts/validate_g8_protocol.py` |
| Date/source contracts | `src/g8_readiness/contracts.py`, `src/g8_readiness/acquisition.py` |
| Checkpoints | `src/g8_readiness/checkpoints.py`, `outputs/g8_checkpoint_forensics/manifest.json` |
| Optional Model3 | `src/g8_readiness/model3.py` |
| Selection/backups | `src/g8_readiness/scanner.py`, `src/g8_readiness/surfaces.py` |
| Manifest/state gates | `src/g8_readiness/manifests.py`, `src/g8_readiness/state_machine.py` |
| Harnesses/adapters | `src/g8_readiness/harness.py` |
| Fixture orchestration | `src/g8_readiness/pipeline.py`, `scripts/run_g8_readiness_pipeline.py` |

## Invariants To Assess

1. Protocol fidelity and unchanged frozen settings.
2. Hard date floor `2026-09-30` through `2026-12-31`.
3. Development-data exclusion, especially July/NTPC observations.
4. Official NSE UDiFF-only and RBI-domain source restriction.
5. Immutable acquisition provenance and failed-artifact retention.
6. Latest official RBI observation on or before valuation date.
7. Deterministic two-date structural selection.
8. Full-window zero-support backup trigger only.
9. Canonical R2 semantics: 20 slots, masks, activity, spot equality, Black IV, Hungarian assignment, 0.10 moneyness screen versus 0.05 assignment ceiling.
10. Pricing-family calibration/holdout isolation.
11. Traditional two-start no-truth rule.
12. Neural eval/no-grad inference with no checkpoint selection or weight updates.
13. Hash-first checkpoint provenance gate.
14. Model3 inclusion only from valid pre-acquisition freeze evidence.
15. Separate acquisition and final-evaluation authorizations.
16. Synthetic fixture labels versus real-result classification.
17. No future-market or future-rate leakage.

## Evidence At Packet Creation

- Real verifier command: `python scripts/validate_g8_protocol.py check-checkpoints`
- Verifier result: `CHECKPOINT_GATE_PASS`; six of six checkpoints loaded after hash approval.
- Focused G8/R2 baseline before state-gate addition: `64 passed`; 13 legacy R2 integration tests deselected because ignored July development archives are absent in this clean checkout.
- Focused current readiness/protocol suite after state-gate addition: `37 passed`.
- Fixture replay: two synthetic dates, eight synthetic surfaces, `NOT_REAL_MARKET_DATA=true`, `NOT_A_RESEARCH_RESULT=true`.

## Post-Review Remediation Included In This Packet

- Pricing rows now emit only valid mask slots.
- Selected-data freeze requires two common dates, four surfaces per date, and successful scan completion.
- Backup replacement now requires explicit complete official-calendar coverage.
- Surface construction carries explicit development-contract overlap checking; real construction additionally requires a nonempty registry and separate authorization.
- Required holdout IV failures now count as method-surface failures in winner-rule accounting.
- Nonselected option rows receive explicit rejection reasons; generic expiry reasons cannot overwrite more specific outcomes.
- Synthetic role mappings use the disjoint canonical calibration/holdout masks.
- `MODEL3_INCLUDED` is accepted only with complete bound evidence checks.
- Pre-acquisition freeze verifies protocol/config/tool identities and current-date floor before it can become ready.
- RBI completeness now requires an official release calendar matching supplied rate records.
- Rejection ledgers are nonselected-only and cover invalid strikes, outside-moneyness rows, invalid types, IV failures, distance failures, and Hungarian non-assignments.
- The assignment ceiling is the exact frozen `0.05 + 1e-12` boundary with no additional tolerance.
- Selected-data seals prove primary/backup symbol composition, CM/FO archive coverage and identities, latest RBI chronology, two dates, four symbols per date, and eight surfaces.
- RBI intake detects conflicting reused HTML or normalized artifacts and rejects unsafe release identifiers.
- Real surface construction uses a distinct real source label; synthetic remains explicitly synthetic.
- Nonfinite required calibration or holdout model prices mark the method-surface as failed.
- `complete_window_scanned` reflects explicit calendar coverage, and preflight diagnostics report protocol/config/tool identity blockers.
- Third-review P1/P2/P3 fixes add seal/source consistency, current-vs-valuation date checks, Model3 committed-artifact hash proof, state-machine identity gates, duplicate archive/rate conflict rejection, complete-window backup enforcement, retained-failure conflict detection, explicit pricing-result provenance, and canonical finite development-contract strikes.
- Fourth review's two findings are fixed: every incomplete-window seal now proves target success with no backup substitutions, and pricing-run provenance is derived from and reconciled against the originating surface.
- Fifth substantive review findings are fixed: pricing rows and quotes are bound to the source surface, RBI record/calendar dates are bound to lookup keys, and the scanner accepts a deterministic four-symbol primary-or-fixed-backup rescan path.

## Known Limitations

- Independent review is pending; timeout is not approval.
- Current date remains before the frozen floor.
- Model3 final pre-acquisition eligibility remains a separate decision.
- A sealed one-command real acquisition workflow intentionally remains unavailable until review, checkpoints, calendar, and explicit human authorization align.
- No real NSE/RBI acquisition or availability scan was performed.

## Prohibited Reviewer Actions

No reviewer should acquire data, inspect future availability, run pricing/calibration/evaluation, update weights, modify files, commit, push, or alter frozen science. Review is read-only code/provenance inspection.
