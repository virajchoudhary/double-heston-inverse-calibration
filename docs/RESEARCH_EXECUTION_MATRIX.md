# Research Execution Matrix

Generated deterministically from the registry and dependency graph.

- Snapshot: `2026-08-25T17:35:00+00:00`
- Audit base: `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- CAN RUN NOW applies to this coordination run; a fresh human authorization is separately required where stated.

## canonical_double_heston_engine

- CURRENT STATUS: SEALED_RESULT / COMPLETE / SEALED_AND_FROZEN
- TEST EXPOSURE: NOT_APPLICABLE_CONTROLLED_BENCHMARK (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Sealed; read-only maintenance only.
- DEPENDENCIES: none
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `src/run_independent_pricing_benchmark.py`
- COMPUTE CLASS: LIGHT_LOCAL_CPU
- RECOMMENDED LOCATION: local Windows Python 3.13
- EXPECTED SCALE: train=NOT_APPLICABLE; validation=NOT_APPLICABLE; test=36 controlled benchmark cases; compute=LIGHT_LOCAL_CPU
- REQUIRED INPUT IDENTITIES: config=`configs/parameter_bounds_PROVISIONAL.yaml` (3E30429A3B69A98C6DDAFDD3276E4E0A3EDEBD113CDACFFFEE9D7A2C032C274C); dataset=`tests/fixtures/double_heston_benchmark_cases.json` (5EE9F07D4250F9661164E7B370F5572E043670BC3E20F0A6FC065F0F75C8688B)
- STOP GATE: Any engine semantic change requires a new predeclared protocol and independent review.
- FORBIDDEN ACTIONS:
  - Change the frozen pricer without a new reviewed protocol.
  - Use repricing quality as unique-parameter-recovery evidence.

## final_r2_10k_dataset

- CURRENT STATUS: SEALED_RESULT / COMPLETE / FINAL_DATASET_FROZEN_BYTE_IDENTICAL_FULL_REPLAY
- TEST EXPOSURE: OPENED_ONLY_AFTER_PRIMARY_MODEL_FREEZE_PER_PROTOCOL (`test_opened=true`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Immutable input; regeneration would be a scientific mutation.
- DEPENDENCIES: frozen_r2_representation
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `b13caf0747d339c7a6bdf7a0400465bcfaad814f`
- RUNNER: `src/r2_final_generation.py`
- COMPUTE CLASS: LIGHT_TO_MODERATE_LOCAL_CPU_GENERATION
- RECOMMENDED LOCATION: local Windows Python 3.13.4
- EXPECTED SCALE: train=7,500 surfaces (6,250 interior + 1,250 wide-valid); validation=1,250 surfaces (1,042 interior + 208 wide-valid); test=1,250 untouched surfaces (1,042 interior + 208 wide-valid); compute=LIGHT_TO_MODERATE_LOCAL_CPU_GENERATION
- REQUIRED INPUT IDENTITIES: config=`configs/r2_synthetic_generation_FINAL.yaml` (D8E705CD424C56189B01C14723DF0E24F750543D904D5A4593A171483AEF0393); dataset=`data/final_r2_clean_10000/surfaces.jsonl` (148B579A4F6CE572E34796E872479C4C016C89BBCD20438C2BB62D6B6960F1F6)
- STOP GATE: Any identity mismatch stops all downstream consumption.
- FORBIDDEN ACTIONS:
  - Regenerate, edit, refill, re-split, or reinterpret the dataset.
  - Start G8 using these synthetic rows.

## frozen_r2_representation

- CURRENT STATUS: SEALED_RESULT / COMPLETE / FROZEN_INTERFACE
- TEST EXPOSURE: NOT_APPLICABLE_INTERFACE_FREEZE (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Frozen interface; consumers must adapt to it, not change it.
- DEPENDENCIES: r2_r3_representation_selection
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `src/r2_representation/contract.py`
- COMPUTE CLASS: LIGHT_LOCAL_CPU
- RECOMMENDED LOCATION: local contract validation
- EXPECTED SCALE: train=240-surface deterministic development pilot; validation=Included in the 240/40 pilot boundary as documented; test=NOT_APPLICABLE_INTERFACE_FREEZE; compute=LIGHT_LOCAL_CPU
- REQUIRED INPUT IDENTITIES: config=`src/r2_representation/contract.py` (D7E85317CB8D8AB2D05AAF21026D32BB620EFAF9F3176465371D2EAC02EA80FD); dataset=`evidence/final_r2_synthetic_pilot_20260822/surfaces.jsonl` (275C97A80F63C011959E309FE7F70C85AA16175423602D2D2AE96058F17BA942)
- STOP GATE: Interface change requires a new reviewed representation protocol.
- FORBIDDEN ACTIONS:
  - Introduce legacy-108 or rejected-R3 vectors through the canonical boundary.
  - Impute masked real quotes.

## g8_final_real_market_evaluation

- CURRENT STATUS: BLOCKED / NOT_STARTED / NO_PROTOCOL_NO_DATES_NO_RESULT
- TEST EXPOSURE: UNTOUCHED_BY_CONSTRUCTION (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Blocked: no protocol, runner, dates, or approved evaluation design exists.
- DEPENDENCIES: traditional_ann_model2_primary_comparison
- WORKTREE / BRANCH: `research/g8-final-eval-protocol` at `MISSING_REF`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `MISSING_RUNNER`
- COMPUTE CLASS: TO_BE_DETERMINED_BY_FROZEN_PROTOCOL
- RECOMMENDED LOCATION: TO_BE_DETERMINED_BY_FROZEN_PROTOCOL
- EXPECTED SCALE: train=NO_REAL_MARKET_WEIGHT_UPDATE; validation=NOT_DEFINED_PENDING_PROTOCOL; test=NOT_SELECTED_PENDING_PROTOCOL; compute=TO_BE_DETERMINED_BY_FROZEN_PROTOCOL
- REQUIRED INPUT IDENTITIES: config=`MISSING_CONFIG` (NOT_APPLICABLE); dataset=`future untouched official-NSE observations` (NOT_YET_SELECTED)
- STOP GATE: Reviewed frozen protocol, reserved untouched cohort, implementation readiness, and explicit human authorization.
- FORBIDDEN ACTIONS:
  - Select G8 dates using model performance.
  - Reuse 2026-07-01, 07-08, 07-15, 07-22, or 07-29.
  - Update primary neural weights with real-market data.
  - Run final evaluation without a pushed frozen protocol.

## identifiability_research

- CURRENT STATUS: SUPERSEDED_HISTORICAL / COMPLETE / HISTORICAL_PRACTICAL_NONIDENTIFIABILITY_RETAINED
- TEST EXPOSURE: NOT_APPLICABLE_SYNTHETIC_DIAGNOSTIC (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Superseded historical diagnostics; retained finding only.
- DEPENDENCIES: ntpc_bs_heston_dh_pilot
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `scripts/run_g2_global_ambiguity_analysis.py`
- COMPUTE CLASS: MODERATE_LOCAL_CPU_DIAGNOSTIC
- RECOMMENDED LOCATION: local Windows Python 3.13
- EXPECTED SCALE: train=Synthetic truths and four primary-stock market support; not research training; validation=NOT_APPLICABLE_DIAGNOSTIC; test=NOT_APPLICABLE_DIAGNOSTIC; compute=MODERATE_LOCAL_CPU_DIAGNOSTIC
- REQUIRED INPUT IDENTITIES: config=`configs/parameter_sampling_REVIEWED.yaml` (2634446A3E11DB8A620D24A4974CA9FC190D29271BEF2A8965C10C01453FD29A); dataset=`synthetic truth panels and official-NSE Stage A support` (UNKNOWN_MULTI_ARTIFACT_PANELS)
- STOP GATE: Superseded by R2/R3 selection; no new runs without a new protocol.
- FORBIDDEN ACTIONS:
  - Reopen the superseded 108-grid representation as current.
  - Treat price fit as parameter recovery.

## model3_genuine_pde

- CURRENT STATUS: IMPLEMENTATION_READY / PENDING / PILOT_READY_AFTER_AUDIT_FIXES_NO_RESULT_EXISTS
- TEST EXPOSURE: TEST_CLOSED (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Implementation-ready but launch requires fresh human authorization and the GPU identity gate.
- DEPENDENCIES: final_r2_10k_dataset, traditional_ann_model2_primary_comparison
- WORKTREE / BRANCH: `research/model3-pde-protocol` at `a01ddc1db854f823eb02b91193eecb4dc6698974`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `scripts/run_model3_pde_pilot.py`
- COMPUTE CLASS: SINGLE_GPU_FLOAT64
- RECOMMENDED LOCATION: Kaggle T4/P100 pilot, then isolated single-GPU sessions for Stage B
- EXPECTED SCALE: train=Stage A first 240 train surfaces; Stage B all 7,500 train surfaces; validation=Stage A first 40 validation surfaces; Stage B all 1,250 validation surfaces; test=1,250 untouched test surfaces, closed until all final checkpoints are verified; compute=SINGLE_GPU_FLOAT64
- REQUIRED INPUT IDENTITIES: config=`configs/model3_pde_protocol.yaml` (D38482381BD3021BAFF80333B40A0770941A79D80FD5E0DA3B4BC314A4F10361); dataset=`data/final_r2_clean_10000/surfaces.jsonl` (148B579A4F6CE572E34796E872479C4C016C89BBCD20438C2BB62D6B6960F1F6)
- STOP GATE: Fresh human authorization; exact HEAD/config/data checks; clean tracked tree; focused foundation tests; smoke diagnostics; then isolated GPU pilot.
- FORBIDDEN ACTIONS:
  - Run Stage A or Stage B during this orchestration sprint.
  - Open the test split before all three final checkpoints are frozen.
  - Change architecture, losses, seeds, epochs, collocation policy, or optimizer settings after results.

## nse_stage_a_development_selection

- CURRENT STATUS: DEVELOPMENT_RESULT / COMPLETE / COMPLETE_DEVELOPMENT_PROVENANCE
- TEST EXPOSURE: NOT_APPLICABLE_DEVELOPMENT_SCREEN (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Completed development screen; historical/provenance use only.
- DEPENDENCIES: canonical_double_heston_engine
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `scripts/run_nse_stage_a_screen.py`
- COMPUTE CLASS: LIGHT_LOCAL_CPU
- RECOMMENDED LOCATION: local Windows Python 3.13
- EXPECTED SCALE: train=NTPC, CIPLA, INFY, HDFCBANK; 2026-07-01, 2026-07-08, 2026-07-15, 2026-07-22, 2026-07-29; validation=NOT_APPLICABLE_DEVELOPMENT_SCREEN; test=NOT_APPLICABLE_DEVELOPMENT_SCREEN; compute=LIGHT_LOCAL_CPU
- REQUIRED INPUT IDENTITIES: config=`configs/market_data_audit_stage_a.yaml` (D91D1C0164E6CF70BF6D17850DBA97470F9471D1F4DD20D63F9CD317368C3523); dataset=`market_data_audit/stage_a/raw/nse` (UNKNOWN_LOCAL_RAW_DATA)
- STOP GATE: Historical lane is closed except for read-only maintenance.
- FORBIDDEN ACTIONS:
  - Use any of the five dates in final G8 evaluation.
  - Claim real-market model performance from this screen.

## ntpc_bs_heston_dh_pilot

- CURRENT STATUS: DEVELOPMENT_RESULT / COMPLETE / COMPLETE_NO_CLEAR_WINNER
- TEST EXPOSURE: NOT_APPLICABLE_DEVELOPMENT_PILOT (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Completed development result; NO_CLEAR_WINNER is final for this lane.
- DEPENDENCIES: nse_stage_a_development_selection
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `scripts/run_ntpc_single_stock_pilot.py`
- COMPUTE CLASS: LIGHT_LOCAL_CPU
- RECOMMENDED LOCATION: local Windows Python 3.13.4
- EXPECTED SCALE: train=12 calibration targets on 2026-07-15; validation=7 holdout targets on 2026-07-15; test=NOT_APPLICABLE_DEVELOPMENT_PILOT; compute=LIGHT_LOCAL_CPU
- REQUIRED INPUT IDENTITIES: config=`configs/parameter_bounds_PROVISIONAL.yaml` (3E30429A3B69A98C6DDAFDD3276E4E0A3EDEBD113CDACFFFEE9D7A2C032C274C); dataset=`official NSE CM/FO records for NTPC on 2026-07-15` (UNKNOWN_LOCAL_RAW_DATA)
- STOP GATE: Closed pending a separately reviewed multi-date or real-market protocol.
- FORBIDDEN ACTIONS:
  - Interpret fitted parameters as ground truth.
  - Promote this pilot to a final real-market performance claim.

## ood_boundary_robustness

- CURRENT STATUS: PARTIAL_CHECKPOINT / PARTIAL / COHORT_GENERATION_AND_REPLAY_DONE_METHOD_EVALUATION_PENDING
- TEST EXPOSURE: PRIMARY_BASELINE_HASHES_RECORDED_VALUES_NOT_INSPECTED_FOR_TUNING (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Cohort generation/replay is complete; expensive method evaluations need fresh authorization and a reviewed aligned harness.
- DEPENDENCIES: traditional_ann_model2_primary_comparison
- WORKTREE / BRANCH: `research/ood-boundary-protocol` at `b6c5e5d0c60d5a99d767ebb3db5175859f310293`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `python -m src.ood_boundary_protocol`
- COMPUTE CLASS: METHOD_DEPENDENT_EVALUATION_AFTER_FROZEN_HARNESS
- RECOMMENDED LOCATION: local light generation/replay; later bounded method evaluations separately authorized
- EXPECTED SCALE: train=NOT_USED_EVALUATION_ONLY; validation=NOT_USED_EVALUATION_ONLY; test=420 evaluation-only research records; 360 clean pricing calls; no model predictions yet; compute=METHOD_DEPENDENT_EVALUATION_AFTER_FROZEN_HARNESS
- REQUIRED INPUT IDENTITIES: config=`configs/ood_boundary_protocol.yaml` (948A23E7D30F762D9D6D85BFF79C5C83C51624E943B4F1F5DD94E7038A348E7C); dataset=`evidence/ood_boundary_protocol_v1/all_research_surfaces.jsonl` (E8B117AC93F6319E634FA28D6DD5ED884E86E130CF420E65EB8EF8DA0276B7E4)
- STOP GATE: Fresh human authorization; reviewed aligned adapter/harness; unchanged cohort/config/threshold identities; no threshold tuning.
- FORBIDDEN ACTIONS:
  - Run expensive method evaluations or reinterpret cohort generation as a model-performance result.
  - Use primary metric values to tune cohorts or thresholds.

## paper_synthesis

- CURRENT STATUS: PARTIAL_CHECKPOINT / PARTIAL / DOCUMENTATION_SYNTHESIS_PARTIAL_STALE_BRANCH_POINTERS
- TEST EXPOSURE: READ_ONLY_EVIDENCE_USE (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Partial synthesis can continue only as scoped documentation after live lanes settle; it may never create science.
- DEPENDENCIES: traditional_ann_model2_primary_comparison, r2_observation_noise_robustness, model3_genuine_pde, ood_boundary_robustness
- WORKTREE / BRANCH: `research/paper-synthesis` at `1fe18e6650bee29f0c4cd731b45ebd198699dde0`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `paper/scripts/generate_results_assets.py`
- COMPUTE CLASS: LIGHT_LOCAL_CPU
- RECOMMENDED LOCATION: local documentation host
- EXPECTED SCALE: train=NOT_APPLICABLE_SYNTHESIS; validation=NOT_APPLICABLE_SYNTHESIS; test=NOT_APPLICABLE_SYNTHESIS; compute=LIGHT_LOCAL_CPU
- REQUIRED INPUT IDENTITIES: config=`paper/scripts/generate_results_assets.py` (C6D78DE0AA85C61E2DE3DF760FEAC788C6C5DAC9A7D969EB40CFB74ED2D817B0); dataset=`committed evidence referenced by paper/RESULTS_INVENTORY.md` (UNKNOWN_MULTI_SOURCE_INVENTORY)
- STOP GATE: Every manuscript claim must map to an immutable artifact hash/ref and pass structural validation.
- FORBIDDEN ACTIONS:
  - Claim completed positive-noise three-way robustness, OOD, G8, or Model 3 results.
  - Edit source scientific evidence to fit prose.
  - Compile claims from stale refs without disclosure.

## r2_observation_noise_robustness

- CURRENT STATUS: PARTIAL_CHECKPOINT / PARTIAL / NEURAL_AND_ZERO_PERCENT_DONE_POSITIVE_TRADITIONAL_PARTIAL
- TEST EXPOSURE: STORED_TEST_SPLIT_OPENED_BY_FROZEN_PROTOCOL (`test_opened=true`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Prepared multicore CPU continuation is outside this coordination run and needs fresh authorization.
- DEPENDENCIES: traditional_ann_model2_primary_comparison
- WORKTREE / BRANCH: `codex/r2-noise-recovery` at `e58b36d1ff6cbf39115d46698ad0fb45b1ac1b8b`; base `91b66b63af7fbda6ad425fe7beeddf045e6b99c0`
- RUNNER: `scripts/run_r2_noise_robustness.py`
- COMPUTE CLASS: MULTICORE_CLOUD_CPU_CALIBRATION
- RECOMMENDED LOCATION: persistent Linux or Windows x86-64 cloud VM
- EXPECTED SCALE: train=NOT_USED_TEST_SPLIT_ONLY_N1250_PER_LEVEL; validation=NOT_USED; test=1,250 stored test surfaces per level; traditional subset exactly 250; compute=MULTICORE_CLOUD_CPU_CALIBRATION
- REQUIRED INPUT IDENTITIES: config=`configs/r2_noise_robustness_FINAL.yaml` (2FA49B3EB885D3427C01AB0CFE447FC6DDD7F19957DB73C4B4ED782476C57C5A); dataset=`data/r2_noise_robustness/levels/*/noisy_surfaces.jsonl plus frozen clean dataset` (MULTIPLE_LEVEL_IDENTITIES_IN_REQUIRED_DATA_MANIFEST)
- STOP GATE: Exact commit b94447d71d8418a218e0e5f2a91ba807f2dcb687, clean tree, all hashes, checkpoint archive identity, and fresh authorization.
- FORBIDDEN ACTIONS:
  - Resume heavy Issue #34 calibration during this sprint.
  - Edit journals, thresholds, subsets, cohorts, checkpoints, or optimizer settings.
  - Interpret a partial journal as a complete level.

## r2_r3_representation_selection

- CURRENT STATUS: SEALED_RESULT / COMPLETE / R2_SELECTED_AND_FROZEN_WITH_NONIDENTIFIABILITY_RETAINED
- TEST EXPOSURE: NOT_APPLICABLE_REPRESENTATION_SELECTION (`test_opened=false`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Sealed once; R2 selected and frozen.
- DEPENDENCIES: identifiability_research
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`
- RUNNER: `scripts/run_g2_r2r3_matrix.py`
- COMPUTE CLASS: MODERATE_LOCAL_CPU_DIAGNOSTIC
- RECOMMENDED LOCATION: local Windows Python 3.13
- EXPECTED SCALE: train=Market-support diagnostics; 20 truths x 2 representations x 4 noise levels x 12 starts; validation=NOT_APPLICABLE_REPRESENTATION_SELECTION; test=NOT_APPLICABLE_REPRESENTATION_SELECTION; compute=MODERATE_LOCAL_CPU_DIAGNOSTIC
- REQUIRED INPUT IDENTITIES: config=`configs/parameter_sampling_REVIEWED.yaml` (2634446A3E11DB8A620D24A4974CA9FC190D29271BEF2A8965C10C01453FD29A); dataset=`five designated NTPC development dates plus sealed synthetic truth panel` (UNKNOWN_SEE_REQUIRED_ARTIFACT_HASHES)
- STOP GATE: Representation changes require a new reviewed protocol.
- FORBIDDEN ACTIONS:
  - Reopen R2 versus R3.
  - Claim unique ten-parameter recovery.

## traditional_ann_model2_primary_comparison

- CURRENT STATUS: SEALED_RESULT / COMPLETE / CANONICAL_COMPLETED_COMPARISON
- TEST EXPOSURE: TEST_OPENED_ONCE_AFTER_FREEZE (`test_opened=true`)
- CAN RUN NOW? NO
- WHY / WHY NOT? Sealed baseline; rerunning cannot alter the accepted result.
- DEPENDENCIES: final_r2_10k_dataset
- WORKTREE / BRANCH: `main` at `72ad8e1aa845ec4c6f0fc61fc526df75438639bb`; base `285a7247eed5096756caec0d561f2f69be140d52`
- RUNNER: `src/r2_primary/final_evaluation.py`
- COMPUTE CLASS: MIXED_LOCAL_CPU_AND_GPU_TRAINING
- RECOMMENDED LOCATION: completed local and P100 cloud sessions
- EXPECTED SCALE: train=7,500 stored train surfaces; validation=1,250 stored validation surfaces; test=1,250 stored untouched test surfaces; compute=MIXED_LOCAL_CPU_AND_GPU_TRAINING
- REQUIRED INPUT IDENTITIES: config=`configs/r2_primary_comparison_FINAL.yaml` (33CA0F763EC10BB2424EEFB02448C9C8E50021854B96A948E420F44BDBA70781); dataset=`data/final_r2_clean_10000/surfaces.jsonl` (148B579A4F6CE572E34796E872479C4C016C89BBCD20438C2BB62D6B6960F1F6)
- STOP GATE: Changes require a new protocol and cannot rewrite this sealed comparison.
- FORBIDDEN ACTIONS:
  - Alter completed evidence or thresholds.
  - Claim unique parameter recovery from repricing quality.
