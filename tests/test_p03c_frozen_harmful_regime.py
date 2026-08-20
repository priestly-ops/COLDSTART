from __future__ import annotations

import numpy as np

from experiments.run_p03c_harmful_regime_calibration import _candidate_regimes
from experiments.run_p03c_robotics_covariance_stress import _make_robotics_covariance
from experiments.run_p03c_robotics_covariance_stress_frozen import (
    FROZEN_HARMFUL_CANDIDATE,
    _frozen_source_covariances,
)


def test_frozen_adversarial_matches_calibrated_candidate() -> None:
    p = 128
    target_cov = _make_robotics_covariance(p)
    frozen = _frozen_source_covariances(target_cov)["adversarial"]

    candidates = dict(
        (name, cov)
        for name, cov, _meta in _candidate_regimes(
            target_cov,
            seed=20260820 + p,
        )
    )
    expected = candidates[FROZEN_HARMFUL_CANDIDATE]

    assert np.allclose(frozen, expected)


def test_frozen_adversarial_is_spd_and_nontrivial() -> None:
    p = 128
    target_cov = _make_robotics_covariance(p)
    frozen = _frozen_source_covariances(target_cov)["adversarial"]

    assert float(np.min(np.linalg.eigvalsh(frozen))) > 0.0
    assert not np.allclose(frozen, target_cov)


def test_other_source_regimes_remain_available() -> None:
    target_cov = _make_robotics_covariance(128)
    regimes = _frozen_source_covariances(target_cov)
    assert set(regimes) == {
        "identical",
        "mild",
        "moderate",
        "block_mismatch",
        "adversarial",
    }
