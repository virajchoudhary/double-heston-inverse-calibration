# Repository Audit

Audit date: 06 August 2026

## Scope

- Working-tree files considered: 86, excluding `.git/` and ignored local backups
- Generated text/source/configuration files scanned: 58
- Python and pytest cache files identified for exclusion: 30
- Reproducible smoke artifacts identified for exclusion: 5
- Files classified as ignored: 38
- Local timestamped backups: ignored and excluded from the count
- Validated handoff PDF: retained unchanged as an explicit binary exception

The scan covered generated Markdown, Python, YAML, JSON, CSV, configuration, and repository-support files. Binary cache/checkpoint files were classified by path and extension. The validated PDF was not modified.

## Security and privacy results

| Check | Result |
|---|---|
| API-key and access-token signatures | No matches |
| Assignment-like passwords or secrets | No matches |
| Private-key headers | No matches |
| Absolute user-profile paths in generated text | No matches |
| Interpreter-installation paths in generated text | No matches |
| Local download or temporary-machine paths in generated text | No matches |
| Raw NSE dataset files | 0 |
| Repository visibility requirement | Must remain private |

The scan intentionally reports match counts and paths rather than printing possible secret values. No raw NSE option dataset is being prepared for Git. The tracked handoff contains project context and a validated PDF, not the underlying raw NSE rows.

## Largest files considered

| Relative path | Size (bytes) | Intended Git treatment |
|---|---:|---|
| `outputs/metrics/smoke_test/data/surfaces.csv` | 2,061,513 | Excluded; reproducible dummy surfaces |
| `handoff/Heston_Double_Heston_Validated_Teammate_Handoff_FINAL.pdf` | 1,443,242 | Included; validated teammate handoff |
| `outputs/metrics/smoke_test/best_validation_checkpoint.pt` | 921,301 | Excluded; reproducible smoke checkpoint |
| `handoff/HESTON_DOUBLE_HESTON_TEAM_CONTEXT.md` | 11,259 | Included |
| `src/synthetic_dataset.py` | 10,844 | Included |
| `outputs/metrics/smoke_test/parameter_evaluation/parameter_predictions_and_errors.csv` | 7,650 | Excluded; dummy row-level output |

## Selected smoke evidence retained

Only lightweight milestone evidence is intended for Git:

- `outputs/metrics/smoke_test/NOT_RESEARCH_DATA`
- `outputs/metrics/smoke_test/smoke_test_summary.json`
- `outputs/metrics/smoke_test/training_summary.json`
- `outputs/metrics/smoke_test/training_history.csv`
- `outputs/metrics/smoke_test/parameter_evaluation/parameter_metrics.csv`
- `outputs/metrics/smoke_test/parameter_evaluation/parameter_metrics.json`

These files document infrastructure execution only. Their dummy metrics are not financial-model results.

## Publication boundary

This is a private academic project. No open-source licence is included, no permission for outside reuse is implied, and any GitHub repository created for it must remain private. Secrets, raw private datasets, generated bulk datasets, machine-specific metadata, cache files, and reproducible checkpoints must remain excluded.
