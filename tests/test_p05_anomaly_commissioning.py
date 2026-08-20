from __future__ import annotations

import numpy as np

from experiments.run_p05_anomaly_commissioning import (
    MIN_TRANSFER_N,
    _fit_estimate,
    _scores,
)
from src.base_detector import BaseDetector


class Args:
    ridge_gammas = [0.05, 0.10, 0.20, 0.40, 0.70, 1.0]
    race_lambdas = [0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
    cv_folds = 5
    se_multiplier = 1.0


def test_safe_policy_fallback_is_exact_target_only_below_25() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 12))
    target = rng.normal(size=(10, 12))
    target_est, target_mu, _ = _fit_estimate(
        "BestTargetOnlySafeCV", source, target, n=10, cv_seed=123, args=Args()
    )
    safe_est, safe_mu, diag = _fit_estimate(
        "RACECovSafeCV", source, target, n=10, cv_seed=123, args=Args()
    )
    assert MIN_TRANSFER_N == 25
    assert diag["policy_fallback"] is True
    assert diag["accepted_transfer"] is False
    assert diag["source_weight"] == 0.0
    assert diag["selected_lambda"] == 0.0
    np.testing.assert_allclose(safe_est.covariance, target_est.covariance)
    np.testing.assert_allclose(safe_est.precision, target_est.precision)
    np.testing.assert_allclose(safe_mu, target_mu)


def test_alpha_001_with_100_calibration_scores_uses_maximum() -> None:
    scores = np.arange(100, dtype=float)
    threshold = BaseDetector.conformal_quantile(scores, alpha=0.01)
    assert threshold == float(np.max(scores))


def test_mahalanobis_scores_are_finite_nonnegative() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(20, 5))
    mu = np.mean(x, axis=0)
    precision = np.eye(5)
    scores = _scores(x, mu, precision)
    assert scores.shape == (20,)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)
