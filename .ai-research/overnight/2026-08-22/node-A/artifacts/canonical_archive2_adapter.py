"""Node A artifact: the ONLY sanctioned canonical <-> archive-2 parameter permutation.

Evidence artifact for future review — deliberately NOT installed into src/. If the team
approves cross-stack interop, this module (or its reviewed equivalent) should be promoted
into canonical src/ with focused tests.

Mapping (verified numerically via exact gradient permutation, Node A experiment A-004):

  canonical order (src/constants.py):
      [kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow,
       kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast]

  archive-2 order (src/dheston/calibration/transforms.py):
      [v01, kappa1, theta1, sigma1, rho1, v02, kappa2, theta2, sigma2, rho2]
      with factor1 = FAST (kappa2 <= kappa1 enforced) and factor2 = SLOW.

  So: canonical slow factor  <-> archive factor 2
      canonical fast factor  <-> archive factor 1
      within-factor: canonical [k, theta, sigma, rho, v0] <-> archive [v0, k, theta, sigma, rho]

Positional tensor passing between the stacks is FORBIDDEN; always route through this
permutation (or its reviewed promotion).

Self-test: round-trip identity + spot check against named fields. Run directly.
"""

from __future__ import annotations

import numpy as np

# canonical index of (kappa, theta, sigma, rho, v0) for slow = [0,1,2,3,4], fast = [5,6,7,8,9]
# archive index of (v0, kappa, theta, sigma, rho) for fast = [0,1,2,3,4], slow = [5,6,7,8,9]
_CANONICAL_TO_ARCHIVE = [
    # kappa_slow -> kappa2 (archive 6)
    6,
    # theta_slow -> theta2 (archive 7)
    7,
    # sigma_slow -> sigma2 (archive 8)
    8,
    # rho_slow -> rho2 (archive 9)
    9,
    # v0_slow -> v02 (archive 5)
    5,
    # kappa_fast -> kappa1 (archive 1)
    1,
    # theta_fast -> theta1 (archive 2)
    2,
    # sigma_fast -> sigma1 (archive 3)
    3,
    # rho_fast -> rho1 (archive 4)
    4,
    # v0_fast -> v01 (archive 0)
    0,
]
_ARCHIVE_TO_CANONICAL = np.argsort(_CANONICAL_TO_ARCHIVE).tolist()

assert sorted(_CANONICAL_TO_ARCHIVE) == list(range(10)), "not a permutation"


def canonical_to_archive2(canonical: np.ndarray) -> np.ndarray:
    """Map canonical-order parameters to archive-2 order (pure permutation)."""
    values = np.asarray(canonical, dtype=np.float64)
    return values[..., _CANONICAL_TO_ARCHIVE]


def archive2_to_canonical(archive: np.ndarray) -> np.ndarray:
    """Map archive-2-order parameters to canonical order (pure permutation)."""
    values = np.asarray(archive, dtype=np.float64)
    return values[..., _ARCHIVE_TO_CANONICAL]


if __name__ == "__main__":
    named = {
        "kappa_slow": 0.8, "theta_slow": 0.04, "sigma_slow": 0.2, "rho_slow": -0.4, "v0_slow": 0.03,
        "kappa_fast": 3.0, "theta_fast": 0.05, "sigma_fast": 0.3, "rho_fast": -0.6, "v0_fast": 0.04,
    }
    canonical = np.array(list(named.values()))
    archive = canonical_to_archive2(canonical)
    # spot check named fields in archive order [v01,k1,th1,s1,r1, v02,k2,th2,s2,r2]
    expected_archive = np.array([
        named["v0_fast"], named["kappa_fast"], named["theta_fast"], named["sigma_fast"], named["rho_fast"],
        named["v0_slow"], named["kappa_slow"], named["theta_slow"], named["sigma_slow"], named["rho_slow"],
    ])
    assert np.allclose(archive, expected_archive), "named-field spot check failed"
    assert np.allclose(archive2_to_canonical(archive), canonical), "round trip failed"
    print("adapter self-test passed: named fields verified, round trip exact")
