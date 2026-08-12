
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IndependentOracleResult:
    max_recall_at_budget: float
    fpr_at_max_recall: float
    threshold_at_budget: float
    min_fpr_at_target: float
    recall_at_min_fpr: float
    threshold_at_target: float
    feasible: bool
    max_recall_count_at_budget: int
    required_recall_count: int
    min_fp_count_at_target: int
    allowed_fp_count: int


def sha256_array(values: np.ndarray) -> str:
    arr = np.asarray(values)
    payload = (
        str(arr.dtype).encode("utf-8")
        + str(arr.shape).encode("utf-8")
        + np.ascontiguousarray(arr).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def sha256_strings(values: Iterable[object]) -> str:
    text = "\n".join(str(v) for v in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return x


def metrics_at_threshold(
    healthy: np.ndarray,
    anomaly: np.ndarray,
    threshold: float,
) -> tuple[int, int, float, float]:
    healthy = _validate(healthy, "healthy")
    anomaly = _validate(anomaly, "anomaly")
    fp = int(np.sum(healthy > threshold))
    tp = int(np.sum(anomaly > threshold))
    return fp, tp, fp / len(healthy), tp / len(anomaly)


def independent_bruteforce_oracle(
    healthy: np.ndarray,
    anomaly: np.ndarray,
    *,
    false_alert_budget: float,
    recall_target: float,
) -> IndependentOracleResult:
    """Independent implementation used to audit the primary M2 oracle.

    This deliberately does NOT import src.oracle_feasibility.

    It enumerates intervals between sorted unique scores using midpoints,
    plus +/- infinity. That is algorithmically different from evaluating
    the observed scores themselves and catches score/tie boundary mistakes.
    """
    h = _validate(healthy, "healthy")
    a = _validate(anomaly, "anomaly")
    unique = np.unique(np.concatenate([h, a]))
    thresholds = [-np.inf]
    if unique.size:
        thresholds.append(float(unique[0]))
        if unique.size > 1:
            mids = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
            thresholds.extend(float(x) for x in mids)
        thresholds.append(float(unique[-1]))
    thresholds.append(np.inf)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    allowed_fp = int(math.floor(false_alert_budget * len(h) + 1e-12))
    required_tp = int(math.ceil(recall_target * len(a) - 1e-12))

    records = []
    for t in thresholds:
        fp, tp, fpr, recall = metrics_at_threshold(h, a, float(t))
        records.append((float(t), fp, tp, fpr, recall))

    budget_records = [r for r in records if r[1] <= allowed_fp]
    if not budget_records:
        raise RuntimeError("No threshold meets allowed FP count.")

    best_tp = max(r[2] for r in budget_records)
    best = [r for r in budget_records if r[2] == best_tp]
    min_fp = min(r[1] for r in best)
    best = [r for r in best if r[1] == min_fp]
    budget = max(best, key=lambda r: r[0])

    target_records = [r for r in records if r[2] >= required_tp]
    if target_records:
        min_fp_target = min(r[1] for r in target_records)
        target_best = [r for r in target_records if r[1] == min_fp_target]
        target = max(target_best, key=lambda r: r[0])
        min_fpr = target[3]
        recall_at_min = target[4]
        threshold_target = target[0]
    else:
        min_fp_target = len(h) + 1
        min_fpr = math.nan
        recall_at_min = math.nan
        threshold_target = math.nan

    feasible = best_tp >= required_tp
    if target_records:
        reciprocal = min_fp_target <= allowed_fp
        if reciprocal != feasible:
            raise RuntimeError("Independent reciprocal oracle checks disagree.")

    return IndependentOracleResult(
        max_recall_at_budget=float(budget[4]),
        fpr_at_max_recall=float(budget[3]),
        threshold_at_budget=float(budget[0]),
        min_fpr_at_target=float(min_fpr),
        recall_at_min_fpr=float(recall_at_min),
        threshold_at_target=float(threshold_target),
        feasible=bool(feasible),
        max_recall_count_at_budget=int(best_tp),
        required_recall_count=int(required_tp),
        min_fp_count_at_target=int(min_fp_target),
        allowed_fp_count=int(allowed_fp),
    )


def sklearn_auc_check(healthy: np.ndarray, anomaly: np.ndarray) -> float:
    """Independent AUROC check if sklearn is installed."""
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return math.nan
    h = _validate(healthy, "healthy")
    a = _validate(anomaly, "anomaly")
    y = np.concatenate([np.zeros(len(h), dtype=int), np.ones(len(a), dtype=int)])
    scores = np.concatenate([h, a])
    return float(roc_auc_score(y, scores))


def count_based_sensitivity(
    healthy: np.ndarray,
    anomaly: np.ndarray,
    *,
    recall_targets=(0.80, 0.90, 0.95),
    allowed_fp_counts=(0, 1, 2),
) -> list[dict]:
    """Small criterion-sensitivity grid expressed in counts.

    Using FP counts avoids pretending that n_H=100 supports finer FPR precision
    than it actually does.
    """
    h = _validate(healthy, "healthy")
    a = _validate(anomaly, "anomaly")
    rows = []
    for fp_allowed in allowed_fp_counts:
        budget = fp_allowed / len(h)
        for target in recall_targets:
            r = independent_bruteforce_oracle(
                h, a,
                false_alert_budget=budget,
                recall_target=target,
            )
            d = asdict(r)
            d.update({
                "allowed_fp_count_requested": int(fp_allowed),
                "empirical_fpr_budget": float(budget),
                "recall_target_requested": float(target),
                "healthy_count": int(len(h)),
                "anomaly_count": int(len(a)),
            })
            rows.append(d)
    return rows


def score_ordering_audit(
    healthy: np.ndarray,
    anomaly: np.ndarray,
) -> dict:
    h = np.sort(_validate(healthy, "healthy"))
    a = np.sort(_validate(anomaly, "anomaly"))
    return {
        "healthy_min": float(h[0]),
        "healthy_median": float(np.median(h)),
        "healthy_second_max": float(h[-2]) if len(h) >= 2 else math.nan,
        "healthy_max": float(h[-1]),
        "anomaly_min": float(a[0]),
        "anomaly_q10": float(np.quantile(a, 0.10)),
        "anomaly_median": float(np.median(a)),
        "anomaly_max": float(a[-1]),
        "healthy_above_anomaly_min": int(np.sum(h > a[0])),
        "healthy_above_anomaly_q10": int(np.sum(h > np.quantile(a, 0.10))),
        "anomaly_below_healthy_max": int(np.sum(a <= h[-1])),
        "anomaly_below_healthy_second_max": (
            int(np.sum(a <= h[-2])) if len(h) >= 2 else -1
        ),
        "strict_complete_separation": bool(a[0] > h[-1]),
    }
