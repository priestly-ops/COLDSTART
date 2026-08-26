from __future__ import annotations

"""Bootstrap sensitivity utilities for empirical low-FPR oracle recall.

These routines quantify how the retrospective empirical oracle changes when the
*fixed evaluation executions* are resampled with replacement. They are a
sample-stability diagnostic only: the resulting percentile intervals are not
post-freeze certification bounds and must not be interpreted as population
confidence intervals under arbitrary sampling/dependence assumptions.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OracleBootstrapResult:
    healthy_count: int
    anomaly_count: int
    false_alert_budget: float
    recall_target: float
    allowed_false_positives: int
    observed_oracle_recall: float
    bootstrap_repetitions: int
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_median: float
    bootstrap_interval_lower: float
    bootstrap_interval_upper: float
    bootstrap_feasibility_rate: float
    bootstrap_below_target_rate: float
    q01: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    q99: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_scores(values: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} scores cannot be empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} scores contain NaN or Inf")
    return x


def oracle_recall_at_fpr_budget_fast(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    false_alert_budget: float = 0.01,
) -> tuple[float, float, int]:
    """Exact empirical max recall subject to empirical FPR <= budget.

    For the repository-wide decision rule ``score > threshold`` and n healthy
    observations, at most ``floor(B*n)`` healthy scores may exceed threshold.
    The smallest feasible threshold is therefore the (k+1)-th largest healthy
    score, where k is that allowed false-positive count. This is the exact
    threshold that maximizes empirical anomaly recall while respecting the FPR
    budget, including ties because equality is classified healthy.

    Returns ``(recall, threshold, allowed_false_positives)``.
    """
    h = _validate_scores(healthy_scores, "healthy")
    a = _validate_scores(anomaly_scores, "anomaly")
    if not 0.0 <= float(false_alert_budget) <= 1.0:
        raise ValueError("false_alert_budget must lie in [0,1]")

    n_h = int(h.size)
    allowed = int(math.floor(float(false_alert_budget) * n_h + 1e-12))
    if allowed >= n_h:
        threshold = -math.inf
        recall = float(np.mean(a > threshold))
        return recall, threshold, allowed

    # Ascending order-statistic index for the (allowed+1)-th largest value.
    kth = n_h - allowed - 1
    threshold = float(np.partition(h, kth)[kth])
    recall = float(np.mean(a > threshold))
    return recall, threshold, allowed


def bootstrap_oracle_recall(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    false_alert_budget: float = 0.01,
    recall_target: float = 0.90,
    repetitions: int = 5_000,
    confidence: float = 0.95,
    seed: int = 42,
    chunk_size: int = 250,
) -> OracleBootstrapResult:
    """Percentile bootstrap sensitivity for empirical oracle recall.

    Healthy and anomalous evaluation executions are resampled independently,
    with replacement, preserving their original sample sizes. For each
    bootstrap replicate the empirical low-FPR oracle is recomputed exactly.
    Chunked vectorization keeps memory bounded for the 100-healthy / ~755-fault
    voraus-AD evaluation split.
    """
    h = _validate_scores(healthy_scores, "healthy")
    a = _validate_scores(anomaly_scores, "anomaly")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if not 0.0 <= recall_target <= 1.0:
        raise ValueError("recall_target must lie in [0,1]")

    observed, _, allowed = oracle_recall_at_fpr_budget_fast(
        h,
        a,
        false_alert_budget=false_alert_budget,
    )

    n_h = int(h.size)
    n_a = int(a.size)
    kth = n_h - allowed - 1
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(repetitions), dtype=np.float64)

    start = 0
    while start < int(repetitions):
        stop = min(start + int(chunk_size), int(repetitions))
        m = stop - start
        h_idx = rng.integers(0, n_h, size=(m, n_h))
        h_sample = h[h_idx]
        if allowed >= n_h:
            thresholds = np.full(m, -np.inf, dtype=np.float64)
        else:
            thresholds = np.partition(h_sample, kth, axis=1)[:, kth]

        a_idx = rng.integers(0, n_a, size=(m, n_a))
        a_sample = a[a_idx]
        boot[start:stop] = np.mean(a_sample > thresholds[:, None], axis=1)
        start = stop

    alpha = 1.0 - float(confidence)
    lower, upper = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    q = np.quantile(boot, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    return OracleBootstrapResult(
        healthy_count=n_h,
        anomaly_count=n_a,
        false_alert_budget=float(false_alert_budget),
        recall_target=float(recall_target),
        allowed_false_positives=int(allowed),
        observed_oracle_recall=float(observed),
        bootstrap_repetitions=int(repetitions),
        bootstrap_mean=float(np.mean(boot)),
        bootstrap_std=float(np.std(boot, ddof=1)) if repetitions > 1 else 0.0,
        bootstrap_median=float(np.median(boot)),
        bootstrap_interval_lower=float(lower),
        bootstrap_interval_upper=float(upper),
        bootstrap_feasibility_rate=float(np.mean(boot >= recall_target - 1e-15)),
        bootstrap_below_target_rate=float(np.mean(boot < recall_target - 1e-15)),
        q01=float(q[0]),
        q05=float(q[1]),
        q25=float(q[2]),
        q50=float(q[3]),
        q75=float(q[4]),
        q95=float(q[5]),
        q99=float(q[6]),
    )
