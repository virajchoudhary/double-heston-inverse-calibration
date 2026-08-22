"""Frozen constants for the predeclared G2 R2-vs-R3 representation-selection study.

Everything in this module is part of the predeclared contract from
``docs/G2_SELF_GOVERNED_REPRESENTATION_PROTOCOL.md`` (merged before any R2/R3
outcome was computed) and MUST NOT be edited after results are seen.  The
decision evaluator (``decision.py``) imports its thresholds only from here and
exposes no runtime override.

Provenance of reused conventions:
- optimizer settings, latent transform, boundary diagnostics:
  ``scripts/run_g2_global_ambiguity_analysis.py`` and
  ``src/calibrate_double_heston.py`` (committed G2 diagnostics);
- fast vectorized pricer arithmetic: Node B overnight toolkit
  (archive/overnight-20260822-node-b, commits 77b8f2e/61905d0), validated
  against the frozen production pricer before use;
- practical-rank tolerance (1e-6 relative): ``run_g2_identifiability_analysis``;
- clustering cutoff / material displacement / near-equivalence RMSE:
  ``run_g2_global_ambiguity_analysis`` and the Node B toolkit;
- interpretation bands (strong/partial improvement): existing project bands
  recorded in the protocol document itself.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# --- frozen randomization (protocol section 3 / task contract) ---------------
TRUTH_SELECTION_SEED: Final[int] = 20260822
MULTISTART_SEED: Final[int] = 20260823
NOISE_BASE_SEED: Final[int] = 20260824

NOISE_LEVELS: Final[tuple[float, ...]] = (0.0, 0.005, 0.01, 0.02)
START_COUNT: Final[int] = 12
BROAD_START_SCALE: Final[float] = 1.25  # N(0, 1.25^2) latent draws, existing convention

# --- candidate representations -----------------------------------------------
CENTRAL_FIVE: Final[tuple[float, ...]] = (-0.10, -0.05, 0.0, 0.05, 0.10)
R2_EXPIRY_RANKS: Final[int] = 2
R3_EXPIRY_RANKS: Final[int] = 3
R2_NOMINAL_SLOTS: Final[int] = R2_EXPIRY_RANKS * len(CENTRAL_FIVE) * 2  # 20
R3_NOMINAL_SLOTS: Final[int] = R3_EXPIRY_RANKS * len(CENTRAL_FIVE) * 2  # 30
REPRESENTATIONS: Final[tuple[str, ...]] = ("R2", "R3")

# --- synthetic panel construction ---------------------------------------------
TRUTH_PANEL_SIZE: Final[int] = 20
STANDING_TRUTH_COUNT: Final[int] = 4
ADDITIONAL_TRUTH_COUNT: Final[int] = 16
# Frozen draw size for the seeded reviewed-interior selection: one LHS draw of
# 64 rows from the existing interior_train population, existing margin gate,
# first 16 accepted rows in row order.
ADDITIONAL_TRUTH_DRAW_COUNT: Final[int] = 64

# Synthetic spot normalization.  Black-Scholes homogeneity makes spot-normalized
# prices on strikes S*exp(k) independent of S, so 100 is a pure normalization.
SYNTHETIC_SPOT: Final[float] = 100.0

# --- calibration (existing G2-ambiguity / Node B diagnostic convention) --------
OPTIMIZER_METHOD: Final[str] = "trf"
OPTIMIZER_MAX_NFEV: Final[int] = 120
OPTIMIZER_FTOL: Final[float] = 1e-10
OPTIMIZER_XTOL: Final[float] = 1e-10
OPTIMIZER_GTOL: Final[float] = 1e-10
OPTIMIZER_DIFF_STEP: Final[float] = 2e-5
NODE_COUNT: Final[int] = 64

# --- local-information diagnostics --------------------------------------------
JACOBIAN_RELATIVE_STEP: Final[float] = 1.0e-4
PRACTICAL_RANK_RELATIVE_TOLERANCE: Final[float] = 1.0e-6

# --- global / cluster diagnostics ----------------------------------------------
CLUSTER_DISTANCE_CUTOFF: Final[float] = 0.10
MATERIAL_DISPLACEMENT_RMSE: Final[float] = 0.05
NEAR_PRICE_EQUIVALENCE_RMSE: Final[float] = 2.5e-7
NEAR_EQUIVALENCE_RELATIVE_MARGIN: Final[float] = 1.05

# --- predeclared interpretation bands (existing project bands) -----------------
STRONG_IMPROVEMENT_MEDIAN: Final[float] = 0.25
STRONG_IMPROVEMENT_MAXIMUM: Final[float] = 0.25
PARTIAL_IMPROVEMENT_MEDIAN: Final[float] = 0.10
PARTIAL_IMPROVEMENT_MAXIMUM: Final[float] = 0.10
HOLDOUT_GUARDRAIL_CEILING: Final[float] = 0.05

# --- holdout guardrail applicability -------------------------------------------
# The existing 5% holdout-deterioration ceiling was defined for a design that
# calibrated on the inner three moneyness targets and held out the +-0.10 wings.
# R2 and R3 both include +-0.10 as calibration slots, so that exact holdout
# metric does not exist for this comparison.  Per the protocol it is recorded
# as NOT APPLICABLE rather than replaced with a manufactured substitute.
HOLDOUT_GUARDRAIL_STATUS: Final[str] = "NOT_APPLICABLE_NO_DIRECTLY_COMPARABLE_HOLDOUT_METRIC"

# --- predeclared practical-non-identifiability operationalization ---------------
# Applied to the representation selected by freeze rules 1-2, at the smallest
# realistic noise level (0.5%): the median across truths of the best-start
# range-scaled parameter RMSE exceeds MATERIAL_DISPLACEMENT_RMSE (the existing
# material-displacement convention) while the median best-start relative
# repricing RMSE stays at or below 2x the noise level (the surface is fitted at
# noise scale yet parameters move materially).  This reuses only existing
# thresholds; it introduces no new tunable number.
NON_IDENTIFIABILITY_NOISE_LEVEL: Final[float] = 0.005
NON_IDENTIFIABILITY_REPRICING_FACTOR: Final[float] = 2.0

# --- G2 completion labels -------------------------------------------------------
G2_LABEL_IDENTIFIABILITY_ACCEPTABLE: Final[str] = (
    "G2 = PASSED_REPRESENTATION_FROZEN_IDENTIFIABILITY_ACCEPTABLE"
)
G2_LABEL_PRACTICAL_NON_IDENTIFIABILITY: Final[str] = (
    "G2 = PASSED_REPRESENTATION_FROZEN_WITH_PRACTICAL_NON_IDENTIFIABILITY"
)
G2_LABEL_FAILED_MARKET_CONSTRUCTION: Final[str] = (
    "G2 = FAILED_MARKET_CONSTRUCTION_REQUIREMENTS"
)

# --- the four standing representative G2 truth cases ---------------------------
# Exact deterministic output of the committed maximin representative selection
# (``run_g2_identifiability_analysis.select_representative_parameters`` with
# per_distribution=4, first two rows per distribution), matching
# ``market_data_audit/stage_a/derived/g2_global_ambiguity/cases.csv`` and
# docs/G2_GLOBAL_AMBIGUITY_ANALYSIS.md at full double precision.
STANDING_TRUTH_VECTORS: Final[dict[str, np.ndarray]] = {
    "case_1": np.asarray(
        [
            0.7201001270341181, 0.0627343918595344, 0.2119791852951291,
            -0.1802978834235966, 0.0730662426958352, 3.703463446201858,
            0.0433194740041468, 0.2627966047238745, -0.1036039245690306,
            0.0678851614425259,
        ]
    ),
    "case_2": np.asarray(
        [
            1.3814304349374713, 0.1191704658387904, 0.2028616298687553,
            -0.6619571308153547, 0.1075045663356258, 5.940280012505328,
            0.0738837487753257, 0.7311424537922181, -0.0610983933146679,
            0.077538064427117,
        ]
    ),
    "case_3": np.asarray(
        [
            1.345471123521795, 0.10402368329319, 0.1927738068021157,
            -0.1243602019590688, 0.1830631299606303, 6.931444137752705,
            0.0773359823305846, 0.5905528253398824, -0.2870652967954047,
            0.1220541374473158,
        ]
    ),
    "case_4": np.asarray(
        [
            2.297187948528226, 0.1999311918616672, 0.7611044302387986,
            0.1827516134971883, 0.0523569672586306, 7.478763310652926,
            0.0644440362118678, 0.9115907741472162, 0.6143052142846812,
            0.1958509664540857,
        ]
    ),
}
STANDING_TRUTH_SAMPLE_IDS: Final[dict[str, str]] = {
    "case_1": "interior_train_4151",
    "case_2": "interior_train_1450",
    "case_3": "wide_valid_train_744",
    "case_4": "wide_valid_train_4264",
}

# --- development market panel ---------------------------------------------------
MARKET_DATES: Final[tuple[str, ...]] = (
    "2026-07-01", "2026-07-08", "2026-07-15", "2026-07-22", "2026-07-29",
)
# Raw-evidence roots (existing official-NSE UDiFF archives on disk; the 07-08
# and 07-29 archives live in the power-tiebreak acquisition tree).
MARKET_RAW_ROOTS: Final[dict[str, str]] = {
    "2026-07-01": "market_data_audit/stage_a/raw/nse",
    "2026-07-15": "market_data_audit/stage_a/raw/nse",
    "2026-07-22": "market_data_audit/stage_a/raw/nse",
    "2026-07-08": "market_data_audit/stage_a/power_tiebreak/raw/nse",
    "2026-07-29": "market_data_audit/stage_a/power_tiebreak/raw/nse",
}
# RBI 91-day T-bill simple yields already validated and hash-sealed by the
# committed multi-date contract, extended ONLY by that contract's own
# carry-forward convention (latest preserved observation on or before the
# valuation date) to the two dates whose auction HTML was never preserved.
# No new acquisition, no fabrication; the extension is documented per protocol.
RATE_OBSERVATIONS: Final[dict[str, dict[str, object]]] = {
    "2026-07-01": {
        "yield": 0.052521,
        "observed": "2026-07-01",
        "source_identifier": "RBI Press Release 2026-2027/584 (validated, hash-sealed in NTPC_DH_MULTI_DATE_CALIBRATION_MANIFEST.json)",
    },
    "2026-07-15": {
        "yield": 0.053324,
        "observed": "2026-07-15",
        "source_identifier": "RBI Press Release 2026-2027/672 (validated, hash-sealed in NTPC_DH_MULTI_DATE_CALIBRATION_MANIFEST.json)",
    },
}
RATE_SOURCE_BY_VALUATION: Final[dict[str, str]] = {
    "2026-07-01": "2026-07-01",
    "2026-07-08": "2026-07-01",  # carry-forward: no preserved auction artifact for this date
    "2026-07-15": "2026-07-15",
    "2026-07-22": "2026-07-15",  # committed carry-forward convention
    "2026-07-29": "2026-07-15",  # carry-forward: no preserved auction artifact for this date
}
