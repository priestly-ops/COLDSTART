from __future__ import annotations

"""Calibration-conditional conformal p-value utilities for E19.

The implementation follows the conformal-anomaly-detection construction used by
Kundacina et al. (IEEE Access, 2025) and the calibration-conditional adjustment
framework of Bates et al. (Annals of Statistics, 2023).

COLDSTART anomaly scores increase with abnormality, whereas the paper writes its
formula in terms of conformity scores that decrease with abnormality. We therefore
compute the equivalent upper-tail conformal p-value

    p = (1 + #{calibration_score >= test_score}) / (n_cal + 1).

The DKWM calibration-conditional adjustment uses

    b_i = min(i / n_cal + sqrt(log(2/delta)/(2 n_cal)), 1),

with b_0=0 and b_{n_cal+1}=1, and maps marginal p-values through
h(p)=b_{ceil((n_cal+1)p)}.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CCVFeasibility:
    calibration_size: int
    alpha: float
    delta: float
    minimum_marginal_pvalue: float
    minimum_dkwm_adjusted_pvalue: float
    marginal_alpha_feasible: bool
    dkwm_alpha_feasible: bool


def _validate_scores(values: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} scores cannot be empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} scores contain NaN or Inf")
    return x


def marginal_conformal_pvalues(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    """Upper-tail conformal p-values for anomaly scores (larger = stranger)."""
    calibration = np.sort(_validate_scores(calibration_scores, "calibration"))
    test = _validate_scores(test_scores, "test")
    n = calibration.size
    # Number of calibration scores >= each test score. side='left' ensures ties
    # are counted conservatively in the calibration tail.
    lower_index = np.searchsorted(calibration, test, side="left")
    tail_count = n - lower_index
    return (1.0 + tail_count.astype(np.float64)) / float(n + 1)


def dkwm_thresholds(calibration_size: int, delta: float) -> np.ndarray:
    """Return b_0,...,b_(n+1) for the DKWM CCV adjustment."""
    n = int(calibration_size)
    if n < 1:
        raise ValueError("calibration_size must be >= 1")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must lie in (0,1)")
    correction = math.sqrt(math.log(2.0 / float(delta)) / (2.0 * n))
    b = np.empty(n + 2, dtype=np.float64)
    b[0] = 0.0
    i = np.arange(1, n + 1, dtype=np.float64)
    b[1 : n + 1] = np.minimum(i / float(n) + correction, 1.0)
    b[n + 1] = 1.0
    return np.maximum.accumulate(b)


def dkwm_adjust_pvalues(
    marginal_pvalues: np.ndarray,
    *,
    calibration_size: int,
    delta: float = 0.05,
) -> np.ndarray:
    p = np.asarray(marginal_pvalues, dtype=np.float64).reshape(-1)
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("marginal p-values must be finite and lie in [0,1]")
    n = int(calibration_size)
    b = dkwm_thresholds(n, delta)
    indices = np.ceil((n + 1) * p - 1e-15).astype(int)
    indices = np.clip(indices, 0, n + 1)
    return b[indices]


def ccv_feasibility(
    calibration_size: int,
    *,
    alpha: float,
    delta: float = 0.05,
) -> CCVFeasibility:
    n = int(calibration_size)
    if n < 1:
        raise ValueError("calibration_size must be >= 1")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    marginal_min = 1.0 / float(n + 1)
    dkwm_min = float(dkwm_adjust_pvalues(
        np.asarray([marginal_min], dtype=np.float64),
        calibration_size=n,
        delta=delta,
    )[0])
    return CCVFeasibility(
        calibration_size=n,
        alpha=float(alpha),
        delta=float(delta),
        minimum_marginal_pvalue=float(marginal_min),
        minimum_dkwm_adjusted_pvalue=dkwm_min,
        marginal_alpha_feasible=bool(marginal_min <= alpha + 1e-15),
        dkwm_alpha_feasible=bool(dkwm_min <= alpha + 1e-15),
    )


def classify_from_pvalues(pvalues: np.ndarray, *, alpha: float) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if not np.isfinite(p).all():
        raise ValueError("p-values contain NaN or Inf")
    return p <= float(alpha) + 1e-15
