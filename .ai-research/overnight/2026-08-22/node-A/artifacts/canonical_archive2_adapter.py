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

# archive[i] = canonical[_ARCHIVE_FROM_CANONICAL[i]]:
#   archive = [v01,kappa1,theta1,sigma1,rho1, v02,kappa2,theta2,sigma2,rho2]
#   = [v0_fast, kappa_fast, theta_fast, sigma_fast, rho_fast,
#      v0_slow, kappa_slow, theta_slow, sigma_slow, rho_slow]
_ARCHIVE_FROM_CANONICAL = [9, 5, 6, 7, 8, 4, 0, 1, 2, 3]
_CANONICAL_FROM_ARCHIVE = [6, 7, 8, 9, 5, 1, 2, 3, 4, 0]

assert sorted(_ARCHIVE_FROM_CANONICAL) == list(range(10)), "not a permutation"
assert sorted(_CANONICAL_FROM_ARCHIVE) == list(range(10)), "not a permutation"


def canonical_to_archive2(canonical: np.ndarray) -> np.ndarray:
    """Map canonical-order parameters to archive-2 order (pure permutation)."""
    values = np.asarray(canonical, dtype=np.float64)
    return values[..., _ARCHIVE_FROM_CANONICAL]


def archive2_to_canonical(archive: np.ndarray) -> np.ndarray:
    """Map archive-2-order parameters to canonical order (pure permutation)."""
    values = np.asarray(archive, dtype=np.float64)
    return values[..., _CANONICAL_FROM_ARCHIVE]


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
