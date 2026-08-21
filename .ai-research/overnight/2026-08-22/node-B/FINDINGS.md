# Node B Findings

## Phase D — Jacobian conditioning across geometries (representative 4 cases + 16 interior samples)

Conventions identical to committed G2 diagnostics (spot-normalized prices, range-scaled parameters,
central differences, practical-rank tolerance 1e-6 relative). Median over the 4 representative cases:

| Geometry | Quotes | Condition number | sigma_min | Practical rank |
| --- | ---: | ---: | ---: | ---: |
| full108 (provisional grid) | 108 | 2.55e4 | 2.0e-5 | 10/10 |
| full108 calls only (= puts only) | 54 | 2.55e4 | 1.4e-5 | 10/10 |
| wings4x6 (±0.20,±0.30; 6 maturities) | 48 | 2.74e4 | 8.7e-6 | 10/10 |
| central5x6 (5 central nodes; 6 maturities) | 60 | 6.42e4 | 6.0e-6 | 10/10 |
| long3 (60/90/180d) | 54 | 1.27e5 | 1.2e-5 | 10/10 |
| short3 (7/14/30d) | 54 | 1.70e6 | 3.2e-7 | 9.5 |
| central5 market 27/55 (G2 anchor) | 20 | 6.54e8 | 1.0e-8 | 7.5 |
| single maturity (7/30/90/180d) | 18 | 1.4e11–5.0e11 | ~5e-13 | ~5 |

Key conclusions:
1. The G2 committed ill-conditioning (central-5, two maturities) REPLICATES (6.5e8 vs committed 5.1e7 median on their reduced-grid panel; both catastrophically noise-dominated, rank 7.5).
2. The FULL provisional 108 grid is locally full practical rank with condition ~2.6e4 — the provisional grid is NOT locally rank-deficient. Maturity span drives conditioning (2 -> 6 maturities improves ~4.5 orders); moneyness width adds little beyond ~4 nodes.
3. Calls and puts are informationally redundant under the carry contract: calls-only, puts-only, and calls+puts share the same condition number to 5 digits (puts are exact parity transforms of calls).
4. Noise-to-sensitivity: sigma_min ~ 1.4e-5 in normalized units sits BELOW realistic noise (0.5% of a 0.05-normalized ATM price = 2.5e-4): naive linear error propagation displaces the estimate by O(10) full-range widths along the weakest direction even on the full grid. Local "full practical rank" therefore coexists with noise-driven parameter collapse.
5. Exact factor-swap symmetry verified: swapping the slow and fast parameter blocks reproduces the 108-quote surface bitwise (max diff 0.0). This exact permutation degeneracy is excluded from the declared space only by kappa_slow < kappa_fast.

Implication: on the provisional 108 grid the failure is NOT classical local rank deficiency; it is (a) noise scale versus weakest sensitivities and (b) global structure (multi-modality / near-equivalent distant solutions), tested next in Phases B/C/F.
