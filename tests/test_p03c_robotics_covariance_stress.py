from __future__ import annotations

import numpy as np

from experiments.run_p03c_robotics_covariance_stress import (
    MIN_TRANSFER_N,
    STATISTICS_PER_SIGNAL,
    _make_robotics_covariance,
    _sample_multivariate_t,
    _source_covariances,
)


def test_robotics_covariance_is_spd_and_heteroscedastic() -> None:
    cov = _make_robotics_covariance(128)
    eig = np.linalg.eigvalsh(cov)
    assert cov.shape == (128, 128)
    assert eig.min() > 1e-8
    variances = np.diag(cov)
    assert float(np.max(variances) / np.min(variances)) > 1.2


def test_signal_major_statistic_groups_are_correlated() -> None:
    cov = _make_robotics_covariance(128)
    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    within = []
    for start in range(0, 120, STATISTICS_PER_SIGNAL):
        block = corr[start:start + STATISTICS_PER_SIGNAL, start:start + STATISTICS_PER_SIGNAL]
        within.extend(np.abs(block[np.triu_indices_from(block, k=1)]).tolist())
    assert np.median(within) > 0.25


def test_source_regimes_are_spd_and_distinct() -> None:
    target = _make_robotics_covariance(128)
    sources = _source_covariances(target)
    assert set(sources) == {"identical", "mild", "moderate", "block_mismatch", "adversarial"}
    assert np.allclose(sources["identical"], target)
    for name, cov in sources.items():
        assert cov.shape == target.shape
        assert np.linalg.eigvalsh(cov).min() > 1e-8, name
    assert not np.allclose(sources["mild"], target)
    assert not np.allclose(sources["moderate"], target)
    assert not np.allclose(sources["adversarial"], target)


def test_multivariate_t_sampling_is_finite_and_correct_shape() -> None:
    cov = _make_robotics_covariance(24)
    x = _sample_multivariate_t(np.random.default_rng(1), cov, 50, 8.0)
    assert x.shape == (50, 24)
    assert np.isfinite(x).all()


def test_min_transfer_policy_is_frozen_at_25() -> None:
    assert MIN_TRANSFER_N == 25
