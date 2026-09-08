# Single Heston — classical calibration research

The companion to `single_heston_pinn/{src,scripts}`, which hold the PINN. This directory holds
the classical single-Heston research the PINN was measured against: the overfitting audit, the
next-day forecast study, the generalisation work, and the single-vs-double comparison.

## Layout

```
research/
  audit_single_heston_overfitting.py    three-fold cross-fit; metric inflation vs honest baselines
  forecast_single_heston_next_day.py    next-session forecast; candidate selection, variance propagation
  improve_heston_generalization.py      post-fit calibrator over the forecast models
  compare_single_double_heston.py       single vs double on an identical test split
  audit_heston_handoff.py               machine audit of the teammate handoff pack
  build_heston_handoff_report.py        builds the handoff .docx / .pdf
  build_heston_results_notebook.py      builds Heston_Double_Heston_Results.ipynb
  Heston_Double_Heston_Results.ipynb    the rendered results notebook
  checks/                               deterministic numerical checks (see below)
  outputs/                              the results these scripts produced
```

Each script roots itself at `Path(__file__).resolve().parent`, so `outputs/` resolves correctly
as long as the script is run **from this directory**:

```bash
cd single_heston_pinn/research
PYTHONPATH=../src:. python audit_single_heston_overfitting.py
```

## checks/ — not pytest

`checks/*.py` are named `test_*.py` for historical reasons but are **script-style checks with a
`main()`, not pytest modules** — pytest collects zero tests from them. Run them directly:

```bash
cd single_heston_pinn/research
for f in checks/test_*.py; do PYTHONPATH=../src:. python "$f"; done
```

All four pass in this layout. The PINN's own pytest suite is separate and lives at
`tests/test_pinn_single_heston.py` (12 tests); `tests/conftest.py` puts `../src` and this
directory on `sys.path` so that suite collects.

## What is shipped, and what is not

`outputs/` carries every report, metric, summary table and figure — about 22 MB.

**Deliberately excluded: nine raw per-quote prediction dumps, ~81 MB.** They are regenerable by
re-running the script that wrote them:

| excluded file | size |
|---|---:|
| `single_heston_next_day_forecast/all_candidate_predictions.csv` | 27.4 MB |
| `single_double_heston_comparison/double_heston_validation_predictions.csv` | 25.8 MB |
| `single_heston/single_heston_predictions.csv` | 5.5 MB |
| `single_double_heston_comparison/single_double_identical_test_predictions.csv` | 4.8 MB |
| `single_heston_generalized_forecast/generalized_heston_test_predictions.csv` | 4.8 MB |
| `single_heston_next_day_forecast/selected_heston_test_predictions.csv` | 4.4 MB |
| `single_heston_generalized_forecast/generalized_next_session_full_surface.csv` | 3.1 MB |
| `single_heston_next_day_forecast/next_session_normalized_full_surface.csv` | 2.7 MB |
| `single_heston_overfit_audit/three_fold_crossfit_predictions.csv` | 2.5 MB |

**Two scripts therefore cannot re-run end to end here.**
`build_heston_handoff_report.py` and `build_heston_results_notebook.py` read
`outputs/019fc8a0/`, `outputs/double_heston_stability_optimized/`,
`outputs/double_heston_state_scale_audit/` and `outputs/double_heston_train_cv_optimized/` —
Double Heston and raw-input directories outside this branch's single-Heston scope. Their
**products are shipped** (`outputs/heston_handoff/`, the notebook, the .docx and .pdf), so the
results are readable; only regeneration needs those inputs.
