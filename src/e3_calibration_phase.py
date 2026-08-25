from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.calibration_tail import conformal_threshold_info
from src.certification import certify_operating_point
from src.bottleneck_decomposition import classify_bottleneck
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.voraus_loader import RobotCycle


@dataclass(frozen=True)
class E3ThresholdResult:
    calibration_size: int
    conformal_rank: int
    conformal_regime: str
    threshold: float
    minimum_attainable_alpha: float
    tp: int
    fn: int
    fp: int
    tn: int
    recall: float
    false_positive_rate: float
    recall_lower: float
    fpr_upper: float
    empirical_success: bool
    certified_success: bool
    auroc: float
    oracle_recall_at_fpr_budget: float
    oracle_fpr_at_max_recall: float
    oracle_empirically_feasible: bool
    bottleneck_label: str


def nested_calibration_order(
    base_calibration: Sequence[RobotCycle],
    extra_calibration: Sequence[RobotCycle],
    *,
    ordering_seed: int,
) -> tuple[RobotCycle, ...]:
    """Create a nested calibration order while preserving the E0 M=100 set.

    The first len(base_calibration) entries are a deterministic permutation of
    the exact E0 frozen calibration set. Any additional independent healthy
    cycles are appended in a separately permuted block. Consequently M=100
    reproduces the same calibration *set* as E0, while smaller M values are
    nested random subsets and larger M values add only previously unused cycles.
    """
    base = list(base_calibration)
    extra = list(extra_calibration)

    base_ids = {cycle.episode_id for cycle in base}
    extra_ids = {cycle.episode_id for cycle in extra}
    if base_ids & extra_ids:
        raise RuntimeError("Base and extra calibration pools overlap.")

    rng = np.random.default_rng(ordering_seed)
    base_perm = rng.permutation(len(base))
    extra_perm = rng.permutation(len(extra)) if extra else np.empty(0, dtype=int)

    ordered = [base[int(i)] for i in base_perm]
    ordered.extend(extra[int(i)] for i in extra_perm)
    return tuple(ordered)


def evaluate_threshold_from_scores(
    *,
    calibration_scores: np.ndarray,
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    false_alert_budget: float,
    recall_target: float,
    joint_confidence: float,
) -> E3ThresholdResult:
    """Evaluate one calibration-prefix threshold with frozen evaluation scores."""
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64).reshape(-1)
    healthy_scores = np.asarray(healthy_scores, dtype=np.float64).reshape(-1)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)

    info = conformal_threshold_info(calibration_scores, alpha=false_alert_budget)
    threshold = float(info.strict_threshold)
    if not info.finite_sample_feasible:
        regime = "infinite"
    elif info.threshold_is_maximum:
        regime = "maximum"
    else:
        regime = "submaximum"

    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold

    fp = int(healthy_pred.sum())
    tn = int((~healthy_pred).sum())
    tp = int(anomaly_pred.sum())
    fn = int((~anomaly_pred).sum())

    fpr = float(fp / (fp + tn))
    recall = float(tp / (tp + fn))

    certification = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=recall_target,
        fpr_budget=false_alert_budget,
        joint_confidence=joint_confidence,
    )

    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=false_alert_budget,
        recall_target=recall_target,
    )

    decomposition = classify_bottleneck(
        oracle=oracle,
        deployed_recall=recall,
        deployed_fpr=fpr,
        recall_lower=float(certification.recall_lower),
        fpr_upper=float(certification.fpr_upper),
        recall_target=recall_target,
        fpr_budget=false_alert_budget,
    )

    return E3ThresholdResult(
        calibration_size=int(info.calibration_size),
        conformal_rank=int(info.raw_rank),
        conformal_regime=regime,
        threshold=threshold,
        minimum_attainable_alpha=float(info.minimum_attainable_alpha),
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=recall,
        false_positive_rate=fpr,
        recall_lower=float(certification.recall_lower),
        fpr_upper=float(certification.fpr_upper),
        empirical_success=bool(decomposition.deployed_empirical_success),
        certified_success=bool(decomposition.deployed_certified_success),
        auroc=float(probability_of_superiority(healthy_scores, anomaly_scores)),
        oracle_recall_at_fpr_budget=float(oracle.max_recall_at_fpr_budget),
        oracle_fpr_at_max_recall=float(oracle.fpr_at_max_recall),
        oracle_empirically_feasible=bool(oracle.empirically_feasible),
        bottleneck_label=str(decomposition.bottleneck_label),
    )
