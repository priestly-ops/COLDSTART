from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bottleneck_decomposition import classify_bottleneck
from src.calibration_tail import conformal_threshold_info
from src.certification import certify_operating_point
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority


@dataclass(frozen=True)
class OperatingPoint:
    """Predeclared deployment specification for E5 sensitivity analysis."""

    name: str
    recall_target: float
    false_alert_budget: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Operating-point name cannot be empty.")
        if not 0.0 <= self.recall_target <= 1.0:
            raise ValueError("recall_target must lie in [0, 1].")
        if not 0.0 < self.false_alert_budget < 1.0:
            raise ValueError("false_alert_budget must lie strictly in (0, 1).")


@dataclass(frozen=True)
class OperatingPointResult:
    threshold: float
    conformal_rank: int
    conformal_regime: str
    minimum_attainable_alpha: float
    tp: int
    fn: int
    fp: int
    tn: int
    recall: float
    fpr: float
    recall_lower: float
    fpr_upper: float
    empirical_success: bool
    certified_success: bool
    auroc: float
    oracle_recall_at_fpr_budget: float
    oracle_fpr_at_max_recall: float
    oracle_empirically_feasible: bool
    bottleneck_label: str


def evaluate_operating_point_from_scores(
    *,
    calibration_scores: np.ndarray,
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    operating_point: OperatingPoint,
    joint_confidence: float = 0.95,
) -> OperatingPointResult:
    """Evaluate one endpoint using a fixed fitted score representation.

    E5 deliberately changes only the deployment specification. The model,
    preprocessing, commissioning set, calibration-score vector, and evaluation
    score vectors are held fixed. The deterministic conformal threshold is
    recomputed with alpha equal to the endpoint's false-alert budget.
    """
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    healthy_scores = np.asarray(healthy_scores, dtype=np.float64).reshape(-1)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)

    for name, values in {
        "calibration_scores": calibration_scores,
        "healthy_scores": healthy_scores,
        "anomaly_scores": anomaly_scores,
    }.items():
        if values.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or Inf.")

    info = conformal_threshold_info(
        calibration_scores,
        alpha=operating_point.false_alert_budget,
    )
    if not info.finite_sample_feasible:
        regime = "infinite"
    elif info.threshold_is_maximum:
        regime = "maximum"
    else:
        regime = "submaximum"

    threshold = float(info.strict_threshold)
    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold

    fp = int(np.sum(healthy_pred))
    tn = int(healthy_scores.size - fp)
    tp = int(np.sum(anomaly_pred))
    fn = int(anomaly_scores.size - tp)
    recall = float(tp / (tp + fn))
    fpr = float(fp / (fp + tn))

    bounds = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=operating_point.recall_target,
        fpr_budget=operating_point.false_alert_budget,
        joint_confidence=joint_confidence,
    )
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=operating_point.false_alert_budget,
        recall_target=operating_point.recall_target,
    )
    bottleneck = classify_bottleneck(
        oracle=oracle,
        deployed_recall=recall,
        deployed_fpr=fpr,
        recall_lower=bounds.recall_lower,
        fpr_upper=bounds.fpr_upper,
        recall_target=operating_point.recall_target,
        fpr_budget=operating_point.false_alert_budget,
    )

    return OperatingPointResult(
        threshold=threshold,
        conformal_rank=int(info.raw_rank),
        conformal_regime=regime,
        minimum_attainable_alpha=float(info.minimum_attainable_alpha),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=recall,
        fpr=fpr,
        recall_lower=float(bounds.recall_lower),
        fpr_upper=float(bounds.fpr_upper),
        empirical_success=bool(bottleneck.deployed_empirical_success),
        certified_success=bool(bottleneck.deployed_certified_success),
        auroc=float(probability_of_superiority(healthy_scores, anomaly_scores)),
        oracle_recall_at_fpr_budget=float(oracle.max_recall_at_fpr_budget),
        oracle_fpr_at_max_recall=float(oracle.fpr_at_max_recall),
        oracle_empirically_feasible=bool(oracle.empirically_feasible),
        bottleneck_label=str(bottleneck.bottleneck_label),
    )


def minimum_best_case_healthy_evidence(
    *,
    false_alert_budget: float,
    joint_confidence: float = 0.95,
) -> int:
    """Minimum healthy n that can certify the FPR target when FP=0."""
    if not 0.0 < false_alert_budget < 1.0:
        raise ValueError("false_alert_budget must lie strictly in (0, 1).")
    delta_fpr = (1.0 - joint_confidence) / 2.0
    n = 1
    while 1.0 - delta_fpr ** (1.0 / n) > false_alert_budget:
        n += 1
    return n


def minimum_best_case_anomaly_evidence(
    *,
    recall_target: float,
    joint_confidence: float = 0.95,
) -> int:
    """Minimum anomaly n that can certify recall when FN=0."""
    if not 0.0 < recall_target < 1.0:
        raise ValueError("recall_target must lie strictly in (0, 1).")
    delta_recall = (1.0 - joint_confidence) / 2.0
    n = 1
    while delta_recall ** (1.0 / n) < recall_target:
        n += 1
    return n
