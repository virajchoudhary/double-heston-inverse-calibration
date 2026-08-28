# Mentor Double Heston Forward PINN Baseline

This isolated experiment asks one narrow question: for one known valid Double
Heston parameter vector, can a simple forward PINN learn synthetic European
CALL prices while explicitly minimizing PDE, stock-boundary, terminal-payoff,
and supervised-data losses?

It is not an inverse calibration experiment. It uses no real market data, does
not tune lambda weights, and does not modify or import the Model 3 inverse
encoder.

## Reproducible workflow

```powershell
C:\Python313\python.exe scripts\mentor_dh_pinn\generate_dataset.py
C:\Python313\python.exe scripts\mentor_dh_pinn\run_baseline.py --skip-data-generation --device cpu
C:\Python313\python.exe scripts\mentor_dh_pinn\evaluate_baseline.py
C:\Python313\python.exe scripts\mentor_dh_pinn\make_figures.py
```

For a local wiring check:

```powershell
C:\Python313\python.exe scripts\mentor_dh_pinn\run_baseline.py --smoke
C:\Python313\python.exe scripts\mentor_dh_pinn\evaluate_baseline.py
C:\Python313\python.exe scripts\mentor_dh_pinn\make_figures.py
```

The smoke split is a reduced development dataset and is not a scientific
result. Full intended counts are 4096 train, 1024 validation, and 1024 test.

Runtime artifacts are written under
`outputs/mentor_dh_pinn_baseline_v1/` and ignored by Git. Training never
requests test rows. Before touching test rows, the explicit evaluator creates
`test_evaluation_claim.json` atomically. An existing claim or metrics artifact
always refuses a repeated evaluation; there is no override flag.

See [MENTOR_BASELINE_PROTOCOL.md](MENTOR_BASELINE_PROTOCOL.md) for the
mathematical contract and
[MENTOR_BASELINE_RESULT_TEMPLATE.md](MENTOR_BASELINE_RESULT_TEMPLATE.md) for
the result-report structure.
