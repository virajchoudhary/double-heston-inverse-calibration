# G2 Multi-Date Independent Replication

## Frozen replication contract

- Original exact-CIR path seed: `20260811`.
- Independent replication path seed, predeclared before computation: `27182818`.
- Recovery/noise/start seed remains `20260811`.
- The only changed scientific field is `cir_path_seed`. Dates, gaps, maturity profiles, central-five calls/puts, controlled carry, targets, A-D definitions, scaling, projection/Schur method, gates, optimizer, target samples, starts, noise levels, and observables are unchanged.

All original Stage A/G2 reports and evidence were hashed before and after replication and remained byte-identical. The replication writes only to its separate report and derived output root.

## Original versus replication

| Design | Practical target rank | Median smallest singular value | Median condition number | Weak-dir cosine |
|---|---:|---:|---:|---:|
| A | 0/8 → 0/8 | 5.994e-09 → 5.994e-09 | 4.504e+07 → 4.504e+07 | 1.000 |
| B | 8/8 → 8/8 | 2.999e-05 → 3.504e-05 | 7.261e+03 → 7.602e+03 | 0.772 |
| C | 6/8 → 8/8 | 5.334e-06 → 4.646e-06 | 5.120e+04 → 5.025e+04 | 0.908 |
| D | 8/8 → 8/8 | 3.320e-05 → 3.311e-05 | 6.279e+03 → 6.581e+03 | 0.731 |

## Recovery comparison

| Design | Noise | Recovery pass | Median target RMSE | Median nuisance RMSE | Optimizer success |
|---|---:|---:|---:|---:|---:|
| A | 0.0% | 0/6 → 0/6 | 1.389e-01 → 1.389e-01 | N/A | 4/6 → 4/6 |
| A | 0.5% | 0/6 → 0/6 | 2.564e-01 → 2.564e-01 | N/A | 1/6 → 1/6 |
| A | 1.0% | 0/6 → 0/6 | 3.578e-01 → 3.578e-01 | N/A | 1/6 → 1/6 |
| B | 0.0% | 1/6 → 1/6 | 5.609e-02 → 4.994e-02 | N/A | 2/6 → 2/6 |
| B | 0.5% | 0/6 → 0/6 | 2.981e-01 → 2.947e-01 | N/A | 0/6 → 0/6 |
| B | 1.0% | 0/6 → 0/6 | 3.579e-01 → 2.566e-01 | N/A | 1/6 → 0/6 |
| C | 0.0% | 0/6 → 0/6 | 7.473e-02 → 1.394e-01 | 1.233e-02 → 1.091e-01 | 0/6 → 0/6 |
| C | 0.5% | 0/6 → 0/6 | 1.469e-01 → 2.179e-01 | 9.787e-02 → 1.922e-01 | 0/6 → 0/6 |
| C | 1.0% | 0/6 → 0/6 | 2.159e-01 → 1.845e-01 | 8.926e-02 → 1.601e-01 | 0/6 → 0/6 |
| D | 0.0% | 0/6 → 0/6 | 1.944e-01 → 1.967e-01 | 1.233e-01 → 2.176e-01 | 0/6 → 0/6 |
| D | 0.5% | 0/6 → 0/6 | 2.210e-01 → 2.207e-01 | 6.899e-02 → 7.391e-02 | 0/6 → 0/6 |
| D | 1.0% | 0/6 → 0/6 | 2.320e-01 → 2.865e-01 | 1.017e-01 → 1.321e-01 | 1/6 → 0/6 |

## Weakest-direction stability

- Design A: median absolute cosine `1.000`, minimum `1.000`, top-three overlap `3/3`; original `kappa_slow|theta_slow|theta_fast`, replication `kappa_slow|theta_slow|theta_fast`.
- Design B: median absolute cosine `0.772`, minimum `0.302`, top-three overlap `2/3`; original `theta_slow|kappa_slow|sigma_slow`, replication `kappa_slow|theta_slow|sigma_fast`.
- Design C: median absolute cosine `0.908`, minimum `0.645`, top-three overlap `3/3`; original `kappa_slow|theta_fast|theta_slow`, replication `theta_slow|kappa_slow|theta_fast`.
- Design D: median absolute cosine `0.731`, minimum `0.125`, top-three overlap `3/3`; original `theta_slow|kappa_slow|theta_fast`, replication `theta_slow|kappa_slow|theta_fast`.

## Hypothesis assessment

| Hypothesis | Status | Decisive replication evidence |
|---|---|---|
| H1: Oracle multi-date information greatly improves local conditioning over A | **REPLICATED** | `condition_reduction=5925.15;rank_improved=True` |
| H2: Latent states weaken target identifiability relative to B | **PARTIALLY_REPLICATED** | `rank_weaker=False;condition_worse=True` |
| H3: Exact CIR physics materially improves local conditioning relative to C | **PARTIALLY_REPLICATED** | `condition_reduction=7.63612;rank_improved=False` |
| H4: Stable ten-parameter recovery remains poor | **REPLICATED** | `maximum_recovery_frequency=0.166667;complete_design=False` |

## Decision

**REPLICATION = MIXED**

Replication diagnostic: `MULTI_DATE_DIAGNOSTIC = INSUFFICIENT`. Seed-sensitive conclusions: `H2|H3`.

The seed sensitivity is confined to the practical-rank component: C changed from `6/8` to `8/8`, so C no longer ranked below B and D no longer ranked above C. The conditioning components persisted: C remained worse conditioned than B and exact-CIR D remained materially better conditioned than C.

**G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS**

One replication does not reopen or pass G2. The scientific question is whether the four qualitative conclusions reproduce, not whether individual floating-point metrics are identical.

## Recommended next research decision

Ask the mentor whether to authorize a small, pre-registered CIR-seed panel of the unchanged experiment solely to estimate the stability of C's practical-rank frequency before considering any redesign.

## Reproducibility and artifacts

| Artifact | SHA-256 |
|---|---|
| `decision.json` | `B1DD862ABE5AC81DE820252A366F5081A44E47E0E2F8332F733BA69C52D054EE` |
| `figures/conditioning_replication.png` | `56744B8DFFA2DEF4E5D7DC94CB325AF19BEEB38C31867160BAEA0F4F36F4082A` |
| `figures/mentor_replication_summary.png` | `7C25CDD21C537E33C4A663DB886B9F7B6157A0681230FDD395FF04D8FB224CF3` |
| `figures/recovery_replication.png` | `E3B5A865FE24467F10141C6B1B051C921B97909D593C7A60B874B2FD986C5FA8` |
| `figures/weakest_direction_stability.png` | `976D7B1B0A61E7E276515E99C8EA8857E7D216F34610DC75B7FFF64AB7A93142` |
| `hypothesis_status.csv` | `F8F79790D2BA873494E7DE4B0BB8BC2C5285788E9EACABCCBB172120C22FEA85` |
| `original_vs_replication.csv` | `2FF364BCD8303B2076CE71A7FC46A54A0211E761ECD02D5BD2D62E4BFD894FC0` |
| `replication_contract.csv` | `041BD0769E3C20CC5D7F1ABC51A6261CA6DD28DFB66C123B8DBE140BEFE24584` |
| `replication_identifiability.csv` | `F87D8AAAF9FF42FFA8FC22862B6100B077F4912CFE651F7F20E10A9FC466BCD3` |
| `replication_nuisance_state_recovery.csv` | `36680654E677E51847AC8C3E1C4737E1DC15BB98B154075D0E145DF09507036B` |
| `replication_parameter_errors.csv` | `45688F73140713AC4710E9059671C95F2D1C02AE1965448A61744CC1B744532B` |
| `replication_parameter_sensitivity.csv` | `A51766BBD214905676170C2E0A1EF5916760B0A082D6E3BA05DC99FC940CA4F2` |
| `replication_recovery_starts.csv` | `F99C4EE7543B459B773C88961F00E367FAC7A9D8D4F980C54A9CBF22E257B004` |
| `replication_recovery_summary.csv` | `22C241FEFFC62F4C7206164B0EA58EE4516BFFADB75E72A14F8126DFE4EA25D4` |
| `replication_state_paths.csv` | `79A4C25BF6D8D62D4C7A8413E55E42FA4DB430668467F887267527E5913DB2A0` |
| `replication_weakest_directions.csv` | `5AA019CC2FED05E741F6C6E09BB14DB8143011724CCC9BAB49524A70D2F4EEE3` |
| `weakest_direction_stability.csv` | `737D87D0CE530567BB3AF830A787734315549BF0B57FA201C07668FCBD1AB90F` |

REPLICATION = MIXED

G2 = NOT_PASSED — STRUCTURAL IDENTIFIABILITY PROBLEM REMAINS
