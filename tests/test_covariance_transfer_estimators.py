from __future__ import annotations

import numpy as np

from src.covariance_transfer_estimators import (
    choose_by_healthy_risk,
    ledoit_wolf_covariance,
    pooled_ledoit_wolf,
    race_covariance,
    ridge_covariance,
)


def _data(seed: int, n: int = 40, p: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega = np.eye(p)
    for i in range(p - 1):
        omega[i, i + 1] = omega[i + 1, i] = -0.2
    cov = np.linalg.inv(omega)
    return rng.multivariate_normal(np.zeros(p), cov, size=n)


def test_ledoit_and_ridge_are_spd() -> None:
    x = _data(1)
    for est in (
        ledoit_wolf_covariance(x),
        ridge_covariance(x, gamma=0.2),
    ):
        assert np.min(np.linalg.eigvalsh(est.covariance)) > 0.0
        assert np.isfinite(est.precision).all()


def test_race_lambda_zero_equals_target_ledoit_wolf() -> None:
    target = _data(2, n=30)
    source = _data(3, n=80)
    target_lw = ledoit_wolf_covariance(target)
    race = race_covariance(target, source, lambda_reg=0.0)
    assert np.allclose(race.covariance, target_lw.covariance)
    assert float(race.metadata["source_weight"]) == 0.0
    assert float(race.metadata["target_weight"]) == 1.0


def test_race_weight_matches_original_formula() -> None:
    target = _data(4, n=25)
    source = _data(5, n=100)
    race = race_covariance(target, source, lambda_reg=60.0)
    expected_target = 25.0 / 85.0
    assert np.isclose(float(race.metadata["target_weight"]), expected_target)
    assert np.isclose(float(race.metadata["source_weight"]), 1.0 - expected_target)


def test_pooled_ledoit_uses_both_domains() -> None:
    target = _data(6, n=20)
    source = _data(7, n=50)
    est = pooled_ledoit_wolf(target, source)
    assert est.metadata["target_samples"] == 20
    assert est.metadata["source_samples"] == 50
    assert est.covariance.shape == (6, 6)


def test_healthy_risk_selection_is_deterministic() -> None:
    target = _data(8, n=30)
    tune = _data(9, n=15)
    candidates = [
        ridge_covariance(target, gamma=0.1, method="ridge_01"),
        ridge_covariance(target, gamma=0.5, method="ridge_05"),
    ]
    first, rows1 = choose_by_healthy_risk(candidates, tune)
    second, rows2 = choose_by_healthy_risk(candidates, tune)
    assert first.method == second.method
    assert rows1 == rows2
