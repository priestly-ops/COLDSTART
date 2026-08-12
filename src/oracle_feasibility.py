from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OracleFeasibilityResult:
    """Retrospective empirical operating-point geometry.

    This object is deliberately diagnostic rather than deployable. Thresholds
    are selected using held-out evaluation scores themselves. Therefore:

    * failure is strong descriptive evidence that the requested operating point
      is absent from the observed score geometry;
    * success only shows that an empirical operating point exists in this
      evaluation sample. It does not provide a calibration or future-sample
      guarantee.
    """

    healthy_count: int
    anomaly_count: int
    false_alert_budget: float
    recall_target: float
    allowed_false_positives: int
    fpr_resolution: float
    recall_resolution: float

    max_recall_at_fpr_budget: float
    fpr_at_max_recall: float
    threshold_at_fpr_budget: float

    min_fpr_at_recall_target: float
    recall_at_min_fpr: float
    threshold_at_recall_target: float

    empirically_feasible: bool
    recall_slack: float
    fpr_slack: float


def _validate_scores(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} scores cannot be empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} scores contain NaN or Inf.")
    return values


def _metrics_at_threshold(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    # Keep the detector convention used throughout COLDSTART: score > threshold
    # is anomalous. This matters for ties at empirical order statistics.
    fpr = float(np.mean(healthy_scores > threshold))
    recall = float(np.mean(anomaly_scores > threshold))
    return fpr, recall


def candidate_thresholds(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> np.ndarray:
    """Return exact empirical decision thresholds, including both extremes.

    For predictions defined by score > threshold, decisions only change when
    the threshold crosses a unique observed score. Evaluating every unique
    score plus -inf/+inf therefore exhausts all empirical classification
    patterns without interpolation assumptions.
    """
    healthy_scores = _validate_scores(healthy_scores, "healthy")
    anomaly_scores = _validate_scores(anomaly_scores, "anomaly")
    unique = np.unique(np.concatenate([healthy_scores, anomaly_scores]))
    return np.concatenate(([-np.inf], unique, [np.inf])).astype(np.float64)


def empirical_oracle_feasibility(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    false_alert_budget: float = 0.01,
    recall_target: float = 0.90,
) -> OracleFeasibilityResult:
    """Compute exact retrospective empirical feasibility of a joint target.

    Two reciprocal diagnostics are calculated over all empirical thresholds:

    1. Maximum recall achievable while empirical FPR <= budget.
    2. Minimum empirical FPR achievable while recall >= target.

    They should agree on feasibility, but both are retained because they make
    tie behavior and distance from the operating boundary transparent.
    """
    healthy_scores = _validate_scores(healthy_scores, "healthy")
    anomaly_scores = _validate_scores(anomaly_scores, "anomaly")

    if not (0.0 <= false_alert_budget <= 1.0):
        raise ValueError("false_alert_budget must lie in [0, 1].")
    if not (0.0 <= recall_target <= 1.0):
        raise ValueError("recall_target must lie in [0, 1].")

    thresholds = candidate_thresholds(healthy_scores, anomaly_scores)
    fprs = np.empty(thresholds.size, dtype=np.float64)
    recalls = np.empty(thresholds.size, dtype=np.float64)
    for i, threshold in enumerate(thresholds):
        fprs[i], recalls[i] = _metrics_at_threshold(
            healthy_scores, anomaly_scores, float(threshold)
        )

    # Best recall under the FPR budget. If several thresholds tie in recall,
    # prefer lower FPR, then the largest threshold for deterministic behavior.
    valid_budget = np.flatnonzero(fprs <= false_alert_budget + 1e-15)
    if valid_budget.size == 0:  # +inf always has FPR=0, defensive only.
        raise RuntimeError("No threshold satisfies the empirical FPR budget.")
    best_recall = float(np.max(recalls[valid_budget]))
    tied = valid_budget[np.isclose(recalls[valid_budget], best_recall, atol=1e-15)]
    min_fpr_among_tied = float(np.min(fprs[tied]))
    tied2 = tied[np.isclose(fprs[tied], min_fpr_among_tied, atol=1e-15)]
    budget_index = int(tied2[np.argmax(thresholds[tied2])])

    # Best (minimum) FPR while retaining the target recall. If target recall is
    # impossible (only relevant for malformed target >1, already rejected),
    # keep explicit NaNs rather than silently substituting another quantity.
    valid_recall = np.flatnonzero(recalls >= recall_target - 1e-15)
    if valid_recall.size == 0:
        min_fpr = math.nan
        recall_at_min_fpr = math.nan
        threshold_at_recall = math.nan
    else:
        min_fpr = float(np.min(fprs[valid_recall]))
        tied = valid_recall[np.isclose(fprs[valid_recall], min_fpr, atol=1e-15)]
        # Among equal-FPR thresholds, choose the largest threshold that still
        # satisfies recall target; this is the most conservative representative.
        recall_index = int(tied[np.argmax(thresholds[tied])])
        recall_at_min_fpr = float(recalls[recall_index])
        threshold_at_recall = float(thresholds[recall_index])

    max_recall = float(recalls[budget_index])
    fpr_at_max = float(fprs[budget_index])
    feasible = bool(max_recall >= recall_target - 1e-15)

    # Reciprocal diagnostics must agree whenever min_fpr is finite.
    if np.isfinite(min_fpr):
        reciprocal = bool(min_fpr <= false_alert_budget + 1e-15)
        if reciprocal != feasible:
            raise RuntimeError(
                "Reciprocal oracle feasibility diagnostics disagree; "
                "check threshold/tie handling."
            )

    return OracleFeasibilityResult(
        healthy_count=int(healthy_scores.size),
        anomaly_count=int(anomaly_scores.size),
        false_alert_budget=float(false_alert_budget),
        recall_target=float(recall_target),
        allowed_false_positives=int(math.floor(false_alert_budget * healthy_scores.size + 1e-12)),
        fpr_resolution=float(1.0 / healthy_scores.size),
        recall_resolution=float(1.0 / anomaly_scores.size),
        max_recall_at_fpr_budget=max_recall,
        fpr_at_max_recall=fpr_at_max,
        threshold_at_fpr_budget=float(thresholds[budget_index]),
        min_fpr_at_recall_target=min_fpr,
        recall_at_min_fpr=recall_at_min_fpr,
        threshold_at_recall_target=threshold_at_recall,
        empirically_feasible=feasible,
        recall_slack=float(max_recall - recall_target),
        fpr_slack=float(false_alert_budget - min_fpr) if np.isfinite(min_fpr) else math.nan,
    )


def probability_of_superiority(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> float:
    """P(A > H) + 0.5 P(A == H), equivalent to empirical ROC AUC."""
    healthy_scores = _validate_scores(healthy_scores, "healthy")
    anomaly_scores = _validate_scores(anomaly_scores, "anomaly")

    # Sorting avoids an O(n_healthy*n_anomaly) pairwise matrix.
    healthy_sorted = np.sort(healthy_scores)
    left = np.searchsorted(healthy_sorted, anomaly_scores, side="left")
    right = np.searchsorted(healthy_sorted, anomaly_scores, side="right")
    wins = left.astype(np.float64)
    ties = (right - left).astype(np.float64)
    return float(np.mean((wins + 0.5 * ties) / healthy_scores.size))
