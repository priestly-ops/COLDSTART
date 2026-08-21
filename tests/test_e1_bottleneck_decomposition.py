from __future__ import annotations

import numpy as np
import pytest

from src.bottleneck_decomposition import (
    CALIBRATION_LIMITED,
    CERTIFICATION_LIMITED,
    CERTIFIED,
    REPRESENTATION_LIMITED,
    classify_bottleneck,
)
from src.oracle_feasibility import empirical_oracle_feasibility


RECALL_TARGET = 0.90
FPR_BUDGET = 0.01


def _oracle(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
):
    return empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=FPR_BUDGET,
        recall_target=RECALL_TARGET,
    )


def test_representation_limited_when_oracle_cannot_reach_target() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(0.2, 0.8, 100)
    oracle = _oracle(healthy, anomaly)

    assert not oracle.empirically_feasible

    result = classify_bottleneck(
        oracle=oracle,
        deployed_recall=0.40,
        deployed_fpr=0.00,
        recall_lower=0.30,
        fpr_upper=0.04,
        recall_target=RECALL_TARGET,
        fpr_budget=FPR_BUDGET,
    )

    assert result.bottleneck_label == REPRESENTATION_LIMITED


def test_calibration_limited_when_oracle_feasible_but_deployment_fails() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(2.0, 3.0, 100)
    oracle = _oracle(healthy, anomaly)

    assert oracle.empirically_feasible

    result = classify_bottleneck(
        oracle=oracle,
        deployed_recall=0.70,
        deployed_fpr=0.00,
        recall_lower=0.60,
        fpr_upper=0.04,
        recall_target=RECALL_TARGET,
        fpr_budget=FPR_BUDGET,
    )

    assert result.bottleneck_label == CALIBRATION_LIMITED


def test_calibration_limited_when_deployed_fpr_exceeds_budget() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(2.0, 3.0, 100)
    oracle = _oracle(healthy, anomaly)

    result = classify_bottleneck(
        oracle=oracle,
        deployed_recall=1.00,
        deployed_fpr=0.04,
        recall_lower=0.95,
        fpr_upper=0.10,
        recall_target=RECALL_TARGET,
        fpr_budget=FPR_BUDGET,
    )

    assert result.bottleneck_label == CALIBRATION_LIMITED
    assert np.isclose(result.deployed_fpr_excess, 0.03)


def test_certification_limited_when_empirical_point_passes_but_bounds_fail() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(2.0, 3.0, 100)
    oracle = _oracle(healthy, anomaly)

    result = classify_bottleneck(
        oracle=oracle,
        deployed_recall=0.95,
        deployed_fpr=0.00,
        recall_lower=0.91,
        fpr_upper=0.036,
        recall_target=RECALL_TARGET,
        fpr_budget=FPR_BUDGET,
    )

    assert result.bottleneck_label == CERTIFICATION_LIMITED
    assert result.deployed_empirical_success
    assert not result.deployed_certified_success


def test_certified_when_both_exact_bounds_pass() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(2.0, 3.0, 100)
    oracle = _oracle(healthy, anomaly)

    result = classify_bottleneck(
        oracle=oracle,
        deployed_recall=0.98,
        deployed_fpr=0.00,
        recall_lower=0.92,
        fpr_upper=0.009,
        recall_target=RECALL_TARGET,
        fpr_budget=FPR_BUDGET,
    )

    assert result.bottleneck_label == CERTIFIED
    assert result.deployed_certified_success


def test_deployed_success_cannot_contradict_oracle_infeasibility() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(0.2, 0.8, 100)
    oracle = _oracle(healthy, anomaly)
    assert not oracle.empirically_feasible

    with pytest.raises(RuntimeError, match="oracle infeasibility"):
        classify_bottleneck(
            oracle=oracle,
            deployed_recall=0.95,
            deployed_fpr=0.00,
            recall_lower=0.91,
            fpr_upper=0.009,
            recall_target=RECALL_TARGET,
            fpr_budget=FPR_BUDGET,
        )


def test_oracle_uses_strict_greater_than_threshold_tie_convention() -> None:
    # At threshold=1.0, healthy scores tied at 1.0 are not false positives and
    # anomaly scores tied at 1.0 are not detections because the deployed rule
    # is score > threshold, not >= threshold.
    healthy = np.array([0.0] * 99 + [1.0])
    anomaly = np.array([1.0] * 10 + [2.0] * 90)
    oracle = _oracle(healthy, anomaly)

    assert oracle.empirically_feasible
    assert np.isclose(oracle.max_recall_at_fpr_budget, 0.90)
    assert np.isclose(oracle.fpr_at_max_recall, 0.00)
    assert np.isclose(oracle.threshold_at_fpr_budget, 1.0)
