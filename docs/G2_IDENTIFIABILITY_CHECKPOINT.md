# G2 Identifiability Checkpoint

Status date: 11 August 2026

Canonical base: `5e5a829828c3009ea42475c2e2ff8e9681995a5a`

This checkpoint preserves the completed G2 research history on
`feat/g2-common-support`. It does not reopen the experiment, select a new
representation, generate the final 10,000-surface dataset, or report ANN/PINN
research results.

## Canonical state

```text
STAGE_A_CANDIDATE_SELECTION = COMPLETE
SELECTED_PRIMARY_SET = NTPC | CIPLA | INFY | HDFCBANK
G2_MARKET_SUPPORTED_GEOMETRY = ESTABLISHED
G2_FINAL_REPRESENTATION = NOT_FROZEN
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
MULTI_DATE_DIAGNOSTIC = INSUFFICIENT
FINAL_10K = NOT_GENERATED
ANN_RESEARCH_TRAINING = NOT_STARTED
PINN = NOT_DERIVED_OR_TRAINED
```

The market-supported geometry is known, but stable recovery of the canonical
ten Double Heston parameters has not been demonstrated. Market support and
inverse identifiability remain separate gates.

## Chronological research record

### 1. Initial provisional representation

The initial candidate ANN surface used:

- nine log-moneyness nodes;
- six maturities;
- calls and puts; and
- `9 × 6 × 2 = 108` normalized option prices.

Stage A rejected this 108-price grid as the final unchanged representation.
The 180-day node was outside the observed expiry support, and the extreme
moneyness wings were too sparse. This rejection did not itself define a
replacement.

### 2. Market-supported geometry

The balanced official-NSE panel was restricted to NTPC, CIPLA, INFY, and
HDFCBANK on `2026-07-01`, `2026-07-15`, and `2026-07-22`. The support analysis
established:

- the near and middle revised/actual listed expiries;
- five central log-moneyness nodes
  `[-0.10, -0.05, 0.00, +0.05, +0.10]`;
- calls and puts; and
- actual maturity coordinates, using `T = DTE / 365`.

This geometry contains `2 × 5 × 2 = 20` normalized option-price observations.
Maturity and carry conditioning are separate coordinates, not additional
option-price observations. The observed near/middle maturity profiles were
`(27, 55)`, `(13, 41)`, and `(6, 34)` calendar days.

The geometry is market-supported, but the final inverse representation was not
frozen. At this point the carry contract was unresolved and practical
identifiability of the canonical ten targets had not been verified.

### 3. Reduced-grid identifiability

The reduced-grid diagnostic fixed the canonical ten-parameter order, used the
full hard-bound widths for parameter scaling, evaluated eight deterministic
maximin parameter vectors across the three observed maturity profiles, and
used 64-node Gauss-Laguerre pricing.

For the market-supported central-five calls-and-puts representation:

- algebraic rank was 10 for every evaluated case;
- practical full-rank frequency was `0/24` across the eight parameter vectors
  and three maturity profiles;
- median condition number was approximately `5.107e7`;
- clean recovery was `0/12`;
- 0.5% noise recovery was `0/12`; and
- 1.0% noise recovery was `0/12`.

The clean median normalized-price RMSE was approximately `2.515e-8`, while the
median range-scaled parameter RMSE was approximately `7.444e-2`. Excellent
repricing therefore did not establish parameter recovery. Under 0.5% and 1.0%
price noise, parameter errors and boundary pressure became severe.

The tested carry-conditioned contract had 20 prices plus maturity-aligned
discount/forward and maturity coordinates, but the historical discount source
remained unresolved. Independently of that provenance blocker, the numerical
identifiability gate failed.

### 4. Third-expiry information remediation

The next bounded experiment compared the same central-five geometry using two
and three listed expiries. It did not add strike wings, change the ten targets,
change the thresholds, or redefine the Stage A activity rule.

Adding the far listed expiry produced:

- approximately `136.93×` improvement in the median smallest singular value;
- approximately `65.47×` reduction in the median condition number;
- practical full rank in `12/24` parameter-profile cases (`50%`);
- clean recovery `2/12`;
- 0.5% noise recovery `0/12`; and
- 1.0% noise recovery `0/12`.

The far expiry was structurally observed and price-usable, but it failed the
existing 75% Stage A activity rule. It was not admitted merely because it
improved local conditioning. The third expiry also remained below the
predeclared practical-rank and recovery gates.

### 5. Multi-date A/B/C/D diagnostic

The multi-date experiment fixed the dates, 14-day and 7-day gaps, the three
near/middle maturity profiles, central-five calls and puts, controlled carry,
the ten targets, parameter scaling, practical-rank threshold, recovery gates,
optimizer configuration, sample counts, starts, and noise levels.

The designs were:

- **A — anchor-date baseline:** the `2026-07-01` option surface only.
- **B — multi-date oracle states:** later slow/fast variance states were
  supplied as oracle conditioning.
- **C — multi-date latent states:** later slow/fast variance states were fitted
  as four nuisance variables and projected from target information.
- **D — exact CIR physics:** Design C plus the exact stochastic CIR transition
  densities, with nuisance states profiled through the Schur complement.

Later variance states were never assumed equal to anchor
`v0_slow`/`v0_fast`. They were generated by exact noncentral-chi-square CIR
transitions and were date-specific.

| Design | Practical full rank | Median smallest singular value | Median condition number | Clean recovery |
|---|---:|---:|---:|---:|
| A | 0/8 | `5.994e-09` | `4.504e7` | 0/6 |
| B | 8/8 | `2.999e-05` | `7.261e3` | 1/6 |
| C | 6/8 | `5.334e-06` | `5.120e4` | 0/6 |
| D | 8/8 | `3.320e-05` | `6.279e3` | 0/6 |

Every design recovered `0/6` at 0.5% noise and `0/6` at 1.0% noise. The
multi-date and exact-CIR additions strongly improved local conditioning, but no
design met the stable-recovery gate.

### 6. Independent exact-CIR path replication

The independent replication changed only the stochastic CIR path seed:

- original CIR path seed: `20260811`;
- replication CIR path seed: `27182818`; and
- recovery/noise/start seed retained: `20260811`.

All other scientific and numerical choices were frozen. The replication found:

| Hypothesis | Status |
|---|---|
| H1: multi-date oracle information greatly improves local conditioning over A | `REPLICATED` |
| H2: latent states weaken target identifiability relative to oracle B | `PARTIALLY_REPLICATED` |
| H3: exact CIR physics materially improves local conditioning relative to C | `PARTIALLY_REPLICATED` |
| H4: stable ten-parameter recovery remains poor | `REPLICATED` |

`REPLICATION = MIXED` refers mainly to the practical-rank sensitivity of
latent-state Design C: C changed from `6/8` to `8/8` full rank. The conditioning
comparisons persisted—C remained materially worse conditioned than oracle B,
and exact-CIR D remained materially better conditioned than C.

The core global-recovery conclusion replicated. Maximum clean recovery
remained `1/6`, and every design again recovered `0/6` at both 0.5% and 1.0%
noise.

## Scientific interpretation

Multi-date observations and exact CIR dynamics materially improve **local**
conditioning. That improvement has not translated into stable **global**
recovery of all ten canonical parameters. Optimizer convergence and low price
RMSE are not identification evidence.

Accordingly:

```text
G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
MULTI_DATE_DIAGNOSTIC = INSUFFICIENT
```

The canonical research objective and ten-parameter target are unchanged. The
final representation remains unfrozen. There is no basis here to claim PINN
success, ANN success, or superiority of any inverse method.

## Reproducibility checkpoint

The tracked machine-readable manifest is
[`docs/evidence/G2_CHECKPOINT_MANIFEST.json`](evidence/G2_CHECKPOINT_MANIFEST.json).
It records the ignored artifact roots, deterministic aggregate hashes, CSV row
counts, seeds, node counts, parameter samples, optimizers, starts, iteration
caps, noise levels, analysis commands, and replay status.

The manifest references 67 ignored G2 files in five evidence sets. It does not
embed or stage the ignored CSV/PNG evidence. Before this checkpoint was written,
all 98 files under `market_data_audit/stage_a` had aggregate SHA-256
`FC80D6A8F7B36C2FF0A4D6DE4ED1AB2EF241F7FA3ABAA38249E0B1E22D3BFAF0`
under the manifest's documented path-and-file-hash algorithm.

Canonical analysis commands:

```powershell
python -B scripts/run_g2_common_support_analysis.py
python -B scripts/run_g2_identifiability_analysis.py
python -B scripts/run_g2_information_remediation.py
python -B scripts/run_g2_multi_date_identifiability.py --node-count 64 --maxiter 80 --start-count 3 --per-distribution 1
python -B scripts/run_g2_multi_date_replication.py --node-count 64 --maxiter 80 --start-count 3 --per-distribution 1
```

The checkpoint task does not rerun these canonical scientific experiments. It
rehashes the existing verified artifacts, runs the focused G2 tests, runs the
full suite once, and confirms that prior Stage A and G2 evidence are unchanged.

## Tracked G2 implementation history

The checkpoint includes five reports, five deterministic analysis scripts, and
five focused test modules covering:

1. common-support geometry;
2. reduced-grid identifiability;
3. third-expiry/information remediation;
4. multi-date A/B/C/D identifiability; and
5. independent CIR-path replication.

The checkpoint and evidence manifest are additional tracked documentation. No
raw NSE files, ignored derived CSV/PNG evidence, final synthetic dataset, ANN
training output, or PINN work is included.

## Next research decision

The next research question is to characterize the remaining global parameter
ambiguity and, preferably with mentor review, decide whether the next inverse
formulation should use:

- defensible priors or regularization;
- reparameterization that preserves the canonical scientific interpretation;
- joint historical inference;
- complementary observables; or
- another justified inverse formulation.

The replicated negative recovery result is sufficient to stop blindly adding
ordinary option-price features. A large CIR-seed panel is not automatically the
next implementation.
