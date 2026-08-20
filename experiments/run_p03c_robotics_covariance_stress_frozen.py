"""Final P0.3c robotics covariance stress with a frozen harmful source regime.

This wrapper intentionally leaves the RACE/Safe-CV estimator unchanged.  The
only change from ``run_p03c_robotics_covariance_stress.py`` is the synthetic
negative-transfer benchmark: the development-only regime calibration selected
``permuted_gain_4x`` as the least-severe predeclared source construction that
was genuinely harmful against the strong target-only baseline.

Calibration/evaluation separation
----------------------------------
The calibration script used target seeds in the 95,000,000 namespace.  The
base P0.3c runner uses the disjoint 9,000,000 namespace, so this final stress
run does not reuse calibration target samples.

The selected harmful source construction is frozen exactly as calibrated:
  1. deterministically permute complete six-statistic signal blocks;
  2. apply alternating per-signal gains 0.25 and 4.0;
  3. sign-flip every third signal subgroup as in the predeclared candidate.

No anomaly labels or synthetic truth enter RACE fitting or Safe-CV selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# When this file is executed directly as
#   python experiments/run_p03c_robotics_covariance_stress_frozen.py
# Python places ``experiments/`` rather than the repository root on sys.path.
# Add the project root explicitly before importing sibling experiment modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_p03c_robotics_covariance_stress as base
from experiments.run_p03c_harmful_regime_calibration import (
    _congruence_with_signal_gains,
    _signal_permutation,
)

FROZEN_HARMFUL_CANDIDATE = "permuted_gain_4x"
FROZEN_CALIBRATION_SEED_BASE = 20260820
FROZEN_PERMUTATION_SEED_OFFSET = 17


def _frozen_source_covariances(target_cov: np.ndarray, seed: int = 20260819) -> dict[str, np.ndarray]:
    """Return base source regimes with only adversarial replaced by frozen stress.

    ``seed`` is accepted for signature compatibility with the base runner but
    does not alter the frozen harmful regime.  The permutation is exactly the
    one used during development calibration for a given dimensionality.
    """
    out = base._source_covariances(target_cov, seed=seed)
    p = int(target_cov.shape[0])

    calibration_seed = FROZEN_CALIBRATION_SEED_BASE + p
    idx = _signal_permutation(
        p,
        calibration_seed + FROZEN_PERMUTATION_SEED_OFFSET,
    )
    permuted = target_cov[np.ix_(idx, idx)]
    harmful = _congruence_with_signal_gains(
        permuted,
        0.25,
        4.00,
        sign_flip=True,
    )

    eig = np.linalg.eigvalsh(harmful)
    if float(eig.min()) <= 1e-10:
        raise RuntimeError(
            f"Frozen {FROZEN_HARMFUL_CANDIDATE} covariance is not SPD: "
            f"min eig={float(eig.min())}"
        )

    out["adversarial"] = harmful
    return out


def main() -> None:
    # Monkey-patch only the benchmark source generator used by base.run.
    # Estimator, target-only baseline, Safe-CV, metrics, thresholds, seeds, and
    # all other protocol logic remain exactly the frozen P0.3c implementation.
    base._source_covariances = _frozen_source_covariances
    base.PROTOCOL_VERSION = "p03c-robotics-covariance-stress-v2-frozen-harmful"
    args = base.parse_args()
    base.run(args)


if __name__ == "__main__":
    main()
