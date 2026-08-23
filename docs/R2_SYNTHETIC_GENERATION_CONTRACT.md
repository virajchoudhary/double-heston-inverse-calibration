# R2 Final Synthetic Generation Contract

Status: `FROZEN_BEFORE_PILOT` (22 August 2026)
Config: [../configs/r2_synthetic_generation_FINAL.yaml](../configs/r2_synthetic_generation_FINAL.yaml)
Tracking issue: #27

This contract freezes the final clean synthetic-generation policy on the
already-sealed R2 representation. It does not reopen R2 versus R3, alter the
production pricer, change parameter semantics, or generate the final 10,000
surfaces.

```text
R2_INTERFACE = FROZEN_R2_RANKED_TWO_EXPIRY_CENTRAL_FIVE
FINAL_10K_NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
MODEL2_RESEARCH_TRAINING = NOT_STARTED
```

## 1. Representation and parameters

Primary data must be constructed only through `src.r2_representation` as an
`R2Surface`: exactly 20 nominal slots in canonical option-major, expiry-rank,
moneyness order; central log-moneyness `[-0.10,-0.05,0,+0.05,+0.10]`; calls
and puts; spot-normalized prices at synthetic spot 100; actual two-rank
maturity conditioning; finite rate/carry conditioning; explicit mask; and all
synthetic masks true. The legacy fixed-calendar 108 grid is compatibility and
historical evidence only; it cannot enter this primary path.

The canonical ten-parameter order is:

1. `kappa_slow`
2. `theta_slow`
3. `sigma_slow`
4. `rho_slow`
5. `v0_slow`
6. `kappa_fast`
7. `theta_fast`
8. `sigma_fast`
9. `rho_fast`
10. `v0_fast`

Candidate generation reuses the reviewed latent LHS transforms, ranges, hard
constraints, distribution-specific acceptance-margin gates, and canonical
validation from `configs/parameter_sampling_REVIEWED.yaml`. No range or
threshold is tuned by this milestone.

## 2. Candidate pools and selection

The entire fixed candidate pool is generated before selection. Every accepted
and rejected candidate is retained with all rejection reasons. Accepted rows
are sorted by zero-based integer `candidate_id`, then the first rows through
the frozen quota are selected. If the fixed pool has too few accepted rows,
generation fails closed. There is no refill loop, second seed, range change,
threshold change, or replacement of difficult selected rows.

| cohort | distribution | pool | required | parameter seed |
|---|---|---:|---:|---:|
| pilot | interior_train | 400 | 200 | 20260822 |
| pilot | wide_valid_train | 200 | 40 | 20260823 |
| final | interior_train | 15,000 | 8,334 | 20260807 |
| final | wide_valid_train | 5,000 | 1,666 | 20260808 |

The final wide seed follows the existing reviewed-audit convention of
advancing one seed between separately sampled distributions. It remains fixed
before any final-pool outcome is observed.

## 3. Exact quotas and split assignment

The future clean core is exactly 10,000 surfaces: 8,334 interior and 1,666
wide. Splits are train `6,250 I + 1,250 W = 7,500`; validation `1,042 I +
208 W = 1,250`; test `1,042 I + 208 W = 1,250`.

The development pilot is exactly 240 surfaces: 200 interior and 40 wide.
Splits are train `150 I + 30 W = 180`; validation `25 I + 5 W = 30`; test
`25 I + 5 W = 30`. Pilot records are permanently labelled
`DEVELOPMENT_PILOT_NOT_FINAL_RESEARCH_DATASET`.

Within each distribution, selected candidates are ordered ascending by
candidate ID and sliced into exact train/validation/test quotas in that order.
No stochastic permutation is used. A whole parameter vector plus its complete
surface is the atomic unit. Surface IDs, vectors, and candidate IDs may not
cross splits.

## 4. Synthetic-only conditioning

Conditioning is a narrow engineering/research design, not a claim about the
real-market distribution. No observed NSE option price, stock spot, futures
price/forward/carry, development-date rate, or date-specific market value is
used to construct a primary synthetic surface.

The deterministic lattice uses rank-1 DTE days `[7,14,21,30,45,60,75,90]`,
rank-2 gaps `[7,14,21,30,45,60,90]`, continuous rates
`[0.01,0.02,0.03,0.04,0.05,0.06]`, and carry offsets
`[-0.02,-0.01,0.00,0.01,0.02,0.03]`. Carry equals rate plus offset. Thus DTE2
is always strictly greater than DTE1 and lies in `[14,180]`.

A generation index maps to `(index * stride) mod 2016`; mixed-radix digits
select the four lattice dimensions in the order documented in config. Pilot
seed 20260822 uses stride 997; final seed 20260807 uses predeclared coprime
stride 1103. This varies conditioning without random replay drift. Spot is
100 for normalization and pricing because normalized target-moneyness prices
are invariant to that engineering normalization under Black-Scholes
homogeneity.

## 5. Pricing, noise, exclusions, and failure retention

Prices come only from unchanged production source `src/double_heston.py`,
through `price_double_heston_surface` with 64 nodes. Each expiry-rank piece is
priced at constant rate/carry and target strikes `100 * exp(k)`. Noise is zero
for this clean core. Future 0.5%, 1%, and 2% populations must be separate
controlled derivatives. Boundary challenge, OOD, noisy copies, real-market
data, G8 selection, and neural training are excluded here. The historical
sampler status remains `NEEDS_SAMPLER_CORRECTION`, and
`CHALLENGE_STRESS_READY` remains false.

A pricing or numerical failure preserves the candidate identity, vector,
conditioning, surface ID, and error, then fails closed. No clipping,
imputation, repricing substitution, hidden retry, or replacement is allowed.
All candidate rejection reasons remain in provenance.

## 6. Provenance and execution gates

Each surface payload retains dataset status, distribution, split, candidate
ID, canonical parameters, seeds, conditioning values/index, R2 name/version,
canonical slot keys/order, mask, generator/config identities, and production
pricer hash. Dataset manifests retain counts, exact quotas, acceptance and
rejection summaries, all relevant hashes, environment metadata, command, and
a timestamp used only as provenance, never RNG input.

The contract checkpoint must be committed before any pilot output exists.
Pilot execution requires that checkpoint. Final readiness additionally requires
a complete authoritative pilot whose replay report is `VERIFIED_IDENTICAL`,
then evaluates the predeclared 15,000/5,000 pools without pricing surfaces.
A future final-10k run requires a separate explicitly authorized command/gate
and is not present in this milestone. Running the pilot can never create final
research data.

```text
FINAL_CANDIDATE_POOL_READINESS = TO_BE_VERIFIED_WITHOUT_PRICING
R2_PILOT = NOT_EXECUTED_AT_CONTRACT_FREEZE
FINAL_10K = NOT_GENERATED
G8 = NOT_STARTED
```
