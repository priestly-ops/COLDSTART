from __future__ import annotations

import numpy as np

from src.precision_transfer_estimators import (
    clime,
    crossfit_trans_clime,
    fit_robust_scaler,
    gaussian_precision_risk,
    reference_trans_clime,
    relative_frobenius_error,
    spd_project,
    support_metrics,
)


def _make_data(seed: int = 0, n: int = 80, p: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    precision = np.eye(p)
    for i in range(p - 1):
        precision[i, i + 1] = precision[i + 1, i] = -0.18
    covariance = np.linalg.inv(precision)
    return rng.multivariate_normal(np.zeros(p), covariance, size=n)


def test_spd_projection_is_positive_definite_and_reports_change() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    result = spd_project(matrix, eigen_floor=1e-5)
    assert np.min(np.linalg.eigvalsh(result.matrix)) >= 0.999e-5
    assert result.relative_frobenius_change > 0.0


def test_clime_is_feasible_on_small_well_conditioned_problem() -> None:
    x = _make_data()
    estimate = clime(x, lam=0.35)
    assert estimate.raw.shape == (6, 6)
    assert estimate.lp_diagnostics.success_fraction == 1.0
    assert estimate.lp_diagnostics.max_constraint_violation < 1e-7
    assert np.min(np.linalg.eigvalsh(estimate.spd)) > 0.0


def test_reference_trans_clime_runs_without_anomaly_information() -> None:
    source = _make_data(seed=1, n=120)
    target = _make_data(seed=2, n=40)
    estimate = reference_trans_clime(
        target[:30],
        target[30:],
        source,
        target_lambda=0.4,
        transfer_lambda_const=1.0,
    )
    assert estimate.raw.shape == (6, 6)
    assert estimate.lp_diagnostics.success_fraction > 0.8
    assert 0.0 <= float(estimate.metadata["transfer_selected_fraction"]) <= 1.0
    assert np.min(np.linalg.eigvalsh(estimate.spd)) > 0.0


def test_crossfit_trans_clime_is_deterministic() -> None:
    source = _make_data(seed=3, n=100)
    target = _make_data(seed=4, n=30)
    first = crossfit_trans_clime(target, source, target_lambda=0.4, n_folds=5, seed=42)
    second = crossfit_trans_clime(target, source, target_lambda=0.4, n_folds=5, seed=42)
    assert np.allclose(first.raw, second.raw)
    assert first.metadata["estimator"] == "Trans-CLIME-crossfit-COLDSTART-extension"


def test_scaler_fit_uses_only_passed_training_rows() -> None:
    source = _make_data(seed=5, n=80)
    target = _make_data(seed=6, n=20)
    target_with_changed_unseen_rows = target.copy()
    # Fit API receives only the first 10 rows in both cases, so changing rows
    # outside that fit set cannot affect the fitted scaler.
    target_with_changed_unseen_rows[10:] += 1000.0
    a = fit_robust_scaler(target[:10], source_train=source, mode="shrink")
    b = fit_robust_scaler(target_with_changed_unseen_rows[:10], source_train=source, mode="shrink")
    assert np.allclose(a.center, b.center)
    assert np.allclose(a.scale, b.scale)


def test_error_and_risk_metrics_are_well_behaved() -> None:
    truth = np.eye(3)
    assert relative_frobenius_error(truth, truth) == 0.0
    metrics = support_metrics(truth, truth)
    assert metrics["support_f1"] == 1.0
    x = np.random.default_rng(0).normal(size=(30, 3))
    assert np.isfinite(gaussian_precision_risk(x, truth))
