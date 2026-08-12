from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalThresholdInfo:
    """Finite-sample split-conformal threshold diagnostics.

    ``raw_rank`` is the one-indexed order statistic requested by

        ceil((m + 1) * (1 - alpha)).

    When ``raw_rank > m``, no finite calibration score can provide the
    requested deterministic split-conformal guarantee. In strict mode the
    appropriate threshold is therefore +inf (predict no anomalies), rather
    than silently clipping the rank to m and claiming the requested alpha.
    ``legacy_clipped_threshold`` is also returned solely to audit the behavior
    of the historical implementation.
    """

    calibration_size: int
    alpha: float
    raw_rank: int
    used_rank: int | None
    finite_sample_feasible: bool
    strict_threshold: float
    legacy_clipped_threshold: float
    minimum_attainable_alpha: float
    threshold_is_maximum: bool


def conformal_threshold_info(
    scores: np.ndarray,
    alpha: float,
) -> ConformalThresholdInfo:
    """Compute strict finite-sample split-conformal threshold diagnostics."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)

    if values.size == 0:
        raise ValueError("scores cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("scores contain NaN or Inf")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    sorted_scores = np.sort(values)
    m = int(sorted_scores.size)
    raw_rank = int(np.ceil((m + 1) * (1.0 - float(alpha))))
    feasible = raw_rank <= m

    # With a threshold equal to the largest finite calibration score, the
    # smallest standard split-conformal miscoverage granularity is 1/(m+1).
    minimum_attainable_alpha = 1.0 / float(m + 1)

    legacy_rank = min(raw_rank, m)
    legacy_threshold = float(sorted_scores[legacy_rank - 1])

    if feasible:
        strict_threshold = float(sorted_scores[raw_rank - 1])
        used_rank: int | None = raw_rank
        threshold_is_maximum = raw_rank == m
    else:
        strict_threshold = float("inf")
        used_rank = None
        threshold_is_maximum = False

    return ConformalThresholdInfo(
        calibration_size=m,
        alpha=float(alpha),
        raw_rank=raw_rank,
        used_rank=used_rank,
        finite_sample_feasible=bool(feasible),
        strict_threshold=strict_threshold,
        legacy_clipped_threshold=legacy_threshold,
        minimum_attainable_alpha=minimum_attainable_alpha,
        threshold_is_maximum=bool(threshold_is_maximum),
    )
