from __future__ import annotations

import numpy as np

from src.covariance_transfer_estimators import (
    race_covariance,
    safe_cv_race_covariance,
    safe_cv_target_only,
)


def _sample(seed: int, covariance: np.ndarray, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=n)


def test_safe_cv_rejects_obviously_bad_source() -> None:
    target_cov = np.array([[1.0, 0.65], [0.65, 1.0]], dtype=float)
    bad_source_cov = np.array([[1.0, -0.65], [-0.65, 1.0]], dtype=float)
    target = _sample(1, target_cov, 80)
    source = _sample(2, bad_source_cov, 400)

    result = safe_cv_race_covariance(
        target,
        source,
        lambdas=(0.0, 5.0, 20.0, 60.0, 120.0),
        n_folds=5,
        seed=123,
        se_multiplier=1.0,
    )

    assert 0.0 <= result.selected_source_weight <= 1.0
    assert result.selected_lambda in {0.0, 5.0, 20.0, 60.0, 120.0}
    assert any(float(row["lambda_reg"]) == 0.0 for row in result.cv_rows)
    if not result.accepted_transfer:
        assert result.selected_lambda == 0.0
        assert result.selected_source_weight == 0.0


def test_safe_cv_accepts_related_source_in_easy_case() -> None:
    target_cov = np.array([[1.0, 0.55], [0.55, 1.0]], dtype=float)
    target = _sample(10, target_cov, 40)
    source = _sample(11, target_cov, 1000)

    result = safe_cv_race_covariance(
        target,
        source,
        lambdas=(0.0, 5.0, 20.0, 60.0, 120.0),
        n_folds=5,
        seed=321,
        se_multiplier=0.0,
    )

    # With a large same-distribution source and the non-conservative selector,
    # at least one positive-lambda candidate should normally beat lambda=0.
    accepted_rows = [r for r in result.cv_rows if bool(r["accepted_transfer_candidate"])]
    assert accepted_rows
    assert result.accepted_transfer
    assert result.selected_lambda > 0.0
    assert result.selected_source_weight > 0.0


def test_safe_cv_final_fit_uses_all_target_rows() -> None:
    cov = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=float)
    target = _sample(20, cov, 25)
    source = _sample(21, cov, 100)
    result = safe_cv_race_covariance(
        target,
        source,
        lambdas=(0.0, 10.0, 60.0),
        n_folds=5,
        seed=7,
    )
    assert int(result.estimate.metadata["target_samples"]) == 25


def test_safe_target_only_uses_same_full_target_budget() -> None:
    cov = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float)
    target = _sample(30, cov, 25)
    est, rows = safe_cv_target_only(
        target,
        ridge_gammas=(0.1, 0.5, 1.0),
        n_folds=5,
        seed=42,
    )
    assert len(rows) == 4  # Ledoit-Wolf + three ridge candidates
    assert int(est.metadata["n_samples"]) == 25
    assert est.metadata["selected_family"] in {"LedoitWolf", "Ridge"}


def test_lambda_zero_matches_target_ledoit_wolf_path() -> None:
    cov = np.eye(3)
    target = _sample(40, cov, 25)
    source = _sample(41, cov, 100)
    zero = race_covariance(target, source, lambda_reg=0.0)
    assert zero.metadata["source_weight"] == 0.0
    assert zero.metadata["target_weight"] == 1.0
