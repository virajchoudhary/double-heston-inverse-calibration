# Handoff Status and Remaining Requirements

The previously expected teammate source will not be provided. The project no longer waits for it.

## Reproduced from the validated handoff contract

- exact ten-parameter order;
- positive `kappa`, `theta`, `sigma`, and `v0` values;
- slow/fast `kappa` ordering;
- strict factorwise Feller conditions;
- individual correlation bounds and the joint correlation-disk rule;
- total instantaneous variance as the sum of two factors;
- Little-Heston-Trap-style characteristic pricing;
- 64-node Gauss-Laguerre default;
- put-call parity;
- documented variance-state propagation equation;
- warnings about non-identifiability and the limited status of historical recovery numbers.

## Newly implemented in this repository

- `src/double_heston.py` and its callable API;
- the ANN pricing adapter integration;
- all engine and integration tests;
- the `CANONICAL_REIMPLEMENTATION_FIXTURE` and its expected prices;
- constrained SciPy calibration and repeated-start logging;
- clean and 1% noise validation scripts and outputs;
- pilot genuine-engine surface generation;
- provisional numerical-safety and sampling ranges;
- implementation and validation documentation.

None of these newly implemented artifacts is claimed to be copied from or equivalent to unavailable teammate code.

## Permanently unavailable original artifacts

- teammate `double_heston.py` and imported helpers;
- teammate test implementation and original controlled fixture;
- exact original lower and upper parameter bounds and their source provenance;
- checksum manifest and original invocation details.

The historical handoff results remain context only and are not forced targets for the new engine.

## Remaining external research requirements

- independent numerical benchmarking of the canonical implementation;
- domain review of the provisional bounds;
- a final NIFTY EOD surface contract covering timestamps, spot, rates, dividends, quote filters, expiries, grid mapping, missing-value masks, interpolation policy, chronological splits, and retained failures;
- predeclared multi-seed/noise protocols before ANN research training.

Historical calibrated Double Heston parameters remain reproducibility artifacts, not supervised ANN truth labels.
