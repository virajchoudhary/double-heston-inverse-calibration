# Result Intake Contract

This contract is for returned computational research results only. It does not
authorize execution and never converts a partial run into a completed result.
Existing sealed evidence remains immutable; use this schema when a separately
authorized lane returns new artifacts for acceptance review.

## Required base fields

Every computational intake record must contain:

- `experiment_id`
- `git_sha` (40-character commit identity)
- `branch`
- exact `command`
- `environment` (operating system, accelerator, key runtime facts)
- `package_versions` (at least NumPy, pandas, SciPy, PyTorch when imported)
- `hardware` (CPU/GPU model, memory, and dedicated-core count where relevant)
- `seed`
- `classification`: exactly `COMPLETE`, `PARTIAL`, or `FAILED`
- `stdout_provenance`
- `stderr_provenance`

## Completed-result additions

A record classified `COMPLETE` must additionally contain:

- `protocol_config_sha256`
- `dataset_sha256`
- `started_at_utc`
- `ended_at_utc`
- `exit_status`
- `checkpoint_identity`
- nonempty `output_files`, each with a path and SHA-256
- `metric_manifest`

Completion also requires independent recomputation of every listed file hash,
confirmation of Git/config/data identities, and reload or replay checks required
by the lane protocol. A green process exit alone is not completion evidence.

## Partial-result additions

A record classified `PARTIAL` must additionally contain:

- `protocol_config_sha256`
- `dataset_sha256`
- `started_at_utc`
- non-`SUCCESS` `exit_status`
- `checkpoint_identity`
- `partial_outputs`
- `failure_reason`

The validator rejects a partial record whose `result_status` or
`scientific_claim_allowed` contains `COMPLETE`. Partial journals and checkpoints
are preserved as evidence; only the lane's unmodified runner may complete them.

## Failed-result rule

A `FAILED` classification requires a concrete `failure_reason`. Preserve failed
starts, interrupted journals, and rejected candidates. Do not delete or rewrite
them to manufacture a clean run.

## Validation

Save one record as JSON (or YAML) and run:

```powershell
C:\Python313\python.exe scripts\verify_research_state.py --validate-result-intake <record.json>
```

The command prints `RESULT_INTAKE_PASS` or `RESULT_INTAKE_FAIL` and exits zero
only on a schema-valid record. Schema validity is necessary, not sufficient;
the receiving reviewer must still verify hashes, ancestry, population boundaries,
stop gates, and scientific interpretation under the experiment protocol.
