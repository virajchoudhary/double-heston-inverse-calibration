# Team Handoff Requirements

Genuine research remains blocked until the teammate provides:

- Frozen `double_heston.py`
- Frozen `test_double_heston.py`
- Every imported helper module
- Controlled synthetic-recovery fixtures
- Exact lower and upper bounds for all ten parameters
- Parameter-bound provenance and version
- Pricing invocation example
- Batch and shape contract for calls and puts
- Quadrature settings
- Numerical tolerances
- Explicit failure behavior
- Requirements or environment file
- Checksum manifest for every frozen source file
- One sample genuine generated surface with its known parameter vector
- Final NIFTY EOD surface contract

The NIFTY contract must define timestamps, spot, risk-free rates, dividends, quote filters, exact expiries, grid mapping, missing-value masks, interpolation policy, chronological split dates, and retained failure records.

Historical calibrated Double Heston parameters are reproducibility artifacts, not supervised ANN truth labels. Temporary smoke ranges must never be promoted into research bounds.
