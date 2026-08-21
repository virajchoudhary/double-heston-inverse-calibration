# Architecture Risk Register — Node A (with Node C inputs), Overnight 2026-08-22

Consolidated from Node A findings F1–F20 and Node C F1–F10 (integrated). Severity =
scientific/credibility impact if unmanaged; Likelihood = chance of occurrence in the
planned research path. Owner = who should act (HUMAN decision required, or engineering
once approved).

| # | Risk | Evidence | Severity | Likelihood | Mitigation | Owner |
|---|---|---|---|---|---|---|
| R1 | Positional cross-stack parameter passing silently swaps slow/fast factors and reorders within factors (no error raised) | F2#1; adapter verified F5/F12; Node C F5 | CRITICAL | Medium | Approve adapter as the ONLY sanctioned interop (Decision #5); add named-field assertions at any future seam | HUMAN |
| R2 | Archive-2 `real_finetune`/`--continuous` updates neural weights on real market data (control violation; already exercised at smoke scale) | F3/F10/F14; Node C F7 | CRITICAL | High while code is reachable from a named mode | Quarantine/remove from canonical entry points (Decision #1); if ablation ever authorized, separate labeled experiment | HUMAN |
| R3 | Archive-2 PDE loss imported as "physics" — as shipped it is a silently-wrong PDE (variance terms zeroed); fixed, it is non-discriminating | F19 (A-013/A-014); Node C F2/F3 REPRODUCED | HIGH | Medium (tempting code to reuse) | DEPRECATE; Model 3 must be fresh construction with Node C-verified wiring (Decision #4) | HUMAN + eng |
| R4 | Constraint-map OOD reach: predictions can sit far outside reviewed sampling ranges (~0% box inclusion from untrained raws; 8–500x violations) | F11 (A-007) | HIGH | Medium | OOD-reporting layer at evaluation (never silent clipping); Decision #3 on box policy | HUMAN |
| R5 | Recovery metrics on boundary-challenge populations carry a map-imposed floor (unrepresentable 0.995-margin shell: 0.44%/0.33% pilot occupancy, over-represented near boundaries) | F16 (A-011) | MEDIUM | Medium | Disclose shell membership of challenge targets; consider margin-width research decision | eng after approval |
| R6 | Repo-level `configs/default_experiment.json` IS the archive-2 config (byte-identical; loaded by `dheston/config.py` as its default) — naming hazard invites accidental archive-semantics runs | F14; Node C F9 | MEDIUM | Medium | Rename/relocate archive configs during quarantine PR (Decision #1 scope) | eng after approval |
| R7 | Production pricer emits small NEGATIVE deep-OTM prices at ultra-short maturities (tau ≲ 5e-3); benign within research grid (worst −6.3e-12) | Node C F10; spot-verified by Node A (A-015) | LOW | Low | Document validated domain (tau ≥ 7 days grid safe); avoid ultra-short expiries in any future grid | eng |
| R8 | Cross-stack metric comparisons conflate raw-scale and range-scaled RMSE | F18 | MEDIUM | Medium | State normalization convention in every metric table; canonical eval stays range-scaled | eng |
| R9 | G2 grid change lands as a rushed rewrite (coupling is data-layer-only today: derived input_size) | F9/F13 | MEDIUM | Low-Medium | Keep model-layer data-derived input_size; representation-interface design before 10k generation | eng after mentor gate |
| R10 | Stale PINN status tokens in two status docs mislead readers about infrastructure vs milestone state | F8 | LOW-MEDIUM | High (until fixed) | Two-axis vocabulary docs PR (Decision #2) | HUMAN |
| R11 | Validation objective mismatch if a PDE term is ever trained but not validated (archive-2 pattern) | Node C F6 | MEDIUM | Low (canonical clean today) | Validation/objective parity rule in the fairness contract (F7) | eng |
| R12 | docs/ARCHITECTURE.md does not cover the archive-2 stack; future readers miss the quarantine rules | F17 | LOW | Medium | Extend ARCHITECTURE.md with the seam/quarantine section after review | eng after approval |

Top actions for the morning team map to R1–R4 (Human Decisions #1–#4) and R6/R10 (quick
hygiene PRs once approved).
