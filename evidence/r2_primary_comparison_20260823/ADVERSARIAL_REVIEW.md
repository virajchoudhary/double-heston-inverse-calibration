# R2 Primary Comparison — Adversarial Review

Executed before opening the PR, attacking the 15 mandated failure modes
(milestone runbook section 20). Each attack is answered with repository
evidence, not intent. Result sections marked TBD until evidence exists.

## 1. Test leakage?

TBD — attack: does any training/tuning path read test rows?
Evidence: training entrypoints index `indices_for_split("train"/"validation")`
only; standardizer fit on train rows; early stopping on validation only;
checkpoints record `test_set_used_for_selection: false`; test predictions
computed once in `final_evaluation.py` after models frozen. Verify by test +
grep + checkpoint inspection.

## 2. Validation leakage?

TBD — attack: validation reused for anything beyond early stopping/checkpoint?

## 3. Real-market leakage?

TBD — attack: any real observation reachable? Quarantine active; dataset
synthetic-only; loader rejects `real_market_inputs_used != False`.

## 4. Hidden 108 assumptions?

TBD — attack: any legacy-grid code on the R2 path? Feature size 100; contract
guards; import-surface test.

## 5. Parameter-order mismatch?

TBD — attack: any consumer using a different order? Canonical order pinned by
tests in constants, dataset records (name-keyed), checkpoints, metrics.

## 6. Unfair information advantage for Method 2?

TBD — attack: repricing term sees more than Model 1? Same 100-dim features;
repricing targets are the same 20 observed normalized prices; geometry is
representation-deterministic.

## 7. Loss weights changed after seeing results?

TBD — weights frozen in protocol config before training (commit 5ea9fd0);
training specs mirrored and asserted by test; checkpoints record weights.

## 8. Seed cherry-picking?

TBD — seeds 11/22/33 predeclared; every run retained; per-seed results
reported; aggregation includes all seeds.

## 9. Failed calibration starts dropped?

TBD — every start recorded in traditional_calibration_starts.csv (including
success=False and exceptions); representative rule never deletes rows.

## 10. Runtime comparison unfair?

TBD — runtimes measured and reported per method with hardware documented;
neural inference is amortized full-split batch; traditional includes all
three starts (disclosed); training runtime reported separately. No
per-surface runtime claim is made across heterogeneous workloads without
qualification.

## 11. Repricing mistaken for parameter recovery?

TBD — identifiability-aware metrics cross-tab both; comparison table carries
the warning; claim discipline forbids equivalence.

## 12. Practical non-identifiability weakened?

TBD — retained finding; near-equivalence reported (repricing success vs
parameter error); factor-swap confusion measured.

## 13. Output checkpoint accidentally trained on test?

TBD — checkpoints predate evaluation; provenance in each checkpoint; final
evaluation loads best-validation checkpoints without further training.

## 14. Real fine-tune bypass?

TBD — quarantine fail-closed test green; no `--continuous`/real-finetune path
executed anywhere in this milestone.

## 15. Claims stronger than evidence?

TBD — results doc language checked against allowed/forbidden claim lists;
every number traced to committed evidence.
