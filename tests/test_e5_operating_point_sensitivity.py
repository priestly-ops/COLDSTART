from __future__ import annotations

import numpy as np

from src.e5_operating_point import (
    OperatingPoint,
    evaluate_operating_point_from_scores,
    minimum_best_case_anomaly_evidence,
    minimum_best_case_healthy_evidence,
)


def _scores():
    calibration = np.arange(100, dtype=np.float64)
    healthy = np.linspace(0.0, 120.0, 100, dtype=np.float64)
    anomaly = np.linspace(80.0, 180.0, 200, dtype=np.float64)
    return calibration, healthy, anomaly


def test_mcal_100_regimes_change_with_fpr_budget() -> None:
    calibration, healthy, anomaly = _scores()

    strict = evaluate_operating_point_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        operating_point=OperatingPoint("strict", 0.90, 0.005),
    )
    primary = evaluate_operating_point_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        operating_point=OperatingPoint("primary", 0.90, 0.01),
    )
    loose = evaluate_operating_point_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        operating_point=OperatingPoint("loose", 0.90, 0.02),
    )

    assert strict.conformal_rank == 101
    assert strict.conformal_regime == "infinite"
    assert np.isinf(strict.threshold)
    assert strict.recall == 0.0
    assert strict.fpr == 0.0

    assert primary.conformal_rank == 100
    assert primary.conformal_regime == "maximum"
    assert primary.threshold == 99.0

    assert loose.conformal_rank == 99
    assert loose.conformal_regime == "submaximum"
    assert loose.threshold == 98.0


def test_recall_target_does_not_change_conformal_threshold_when_budget_fixed() -> None:
    calibration, healthy, anomaly = _scores()
    r80 = evaluate_operating_point_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        operating_point=OperatingPoint("r80", 0.80, 0.01),
    )
    r95 = evaluate_operating_point_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        operating_point=OperatingPoint("r95", 0.95, 0.01),
    )

    assert r80.threshold == r95.threshold
    assert r80.conformal_rank == r95.conformal_rank == 100
    assert r80.tp == r95.tp
    assert r80.fp == r95.fp


def test_best_case_healthy_certification_boundaries() -> None:
    assert minimum_best_case_healthy_evidence(false_alert_budget=0.005) == 736
    assert minimum_best_case_healthy_evidence(false_alert_budget=0.01) == 368
    assert minimum_best_case_healthy_evidence(false_alert_budget=0.02) == 183


def test_best_case_anomaly_certification_boundaries() -> None:
    assert minimum_best_case_anomaly_evidence(recall_target=0.80) == 17
    assert minimum_best_case_anomaly_evidence(recall_target=0.90) == 36
    assert minimum_best_case_anomaly_evidence(recall_target=0.95) == 72
