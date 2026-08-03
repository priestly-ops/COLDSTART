from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """Common interface for all cycle-level anomaly detectors.

    A detector produces a continuous anomaly score where larger values
    indicate a more anomalous cycle. The calibration step converts scores
    into binary predictions by fitting a threshold using healthy calibration
    cycles only.
    """

    def __init__(self, false_alert_budget: float = 0.01) -> None:
        if not 0.0 < false_alert_budget < 1.0:
            raise ValueError(
                "false_alert_budget must be between 0 and 1."
            )

        self.false_alert_budget = float(false_alert_budget)
        self.threshold_: float | None = None
        self.is_fitted_: bool = False
        self.is_calibrated_: bool = False

    @abstractmethod
    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "BaseDetector":
        """Fit the detector using permitted source and target data."""

    @abstractmethod
    def score_samples(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Return one anomaly score per input cycle."""

    def calibrate(
        self,
        calibration_features: np.ndarray,
    ) -> "BaseDetector":
        """Fit a split-conformal threshold on healthy calibration data.

        Uses the finite-sample conformal order statistic:

            ceil((n + 1) * (1 - alpha))

        where alpha is the allowed false-alert rate.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Detector must be fitted before calibration."
            )

        scores = self.score_samples(calibration_features)
        scores = self._validate_scores(scores)

        self.threshold_ = self.conformal_quantile(
            scores=scores,
            alpha=self.false_alert_budget,
        )
        self.is_calibrated_ = True
        return self
    def calibrate_from_scores(
        self,
        calibration_scores: np.ndarray,
    ) -> "BaseDetector":
        """Calibrate from precomputed healthy anomaly scores."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Detector must be fitted before calibration."
            )

        scores = self._validate_scores(
            calibration_scores
        )

        self.threshold_ = self.conformal_quantile(
            scores=scores,
            alpha=self.false_alert_budget,
        )

        self.is_calibrated_ = True
        return self

    def predict(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Return binary anomaly predictions."""
        if not self.is_calibrated_ or self.threshold_ is None:
            raise RuntimeError(
                "Detector must be calibrated before prediction."
            )

        scores = self.score_samples(features)
        return (scores > self.threshold_).astype(np.int64)

    @staticmethod
    def conformal_quantile(
        scores: np.ndarray,
        alpha: float,
    ) -> float:
        """Return the finite-sample split-conformal threshold.

    The threshold is the k-th ordered healthy calibration score, where

        k = ceil((n + 1) * (1 - alpha))

    If k > n, deterministic split conformal cannot provide the requested
    coverage with the available calibration size. In that case the largest
    calibration score is returned and the result is conservative.
    """
        scores = BaseDetector._validate_scores(scores)

        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        sorted_scores = np.sort(scores)
        sample_count = len(sorted_scores)

        rank = int(
            np.ceil(
                (sample_count + 1)
                * (1.0 - alpha)
            )
        )

        if rank > sample_count:
            rank = sample_count

        return float(sorted_scores[rank - 1])

    @staticmethod
    def _validate_features(
        features: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2D matrix, received {matrix.shape}."
            )

        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Feature matrix cannot be empty.")

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Feature matrix contains NaN or Inf."
            )

        return matrix

    @staticmethod
    def _validate_scores(
        scores: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)

        if len(values) == 0:
            raise ValueError("Score vector cannot be empty.")

        if not np.isfinite(values).all():
            raise ValueError(
                "Score vector contains NaN or Inf."
            )

        return values

    def get_params(self) -> dict[str, Any]:
        return {
            "false_alert_budget": self.false_alert_budget,
            "threshold": self.threshold_,
            "is_fitted": self.is_fitted_,
            "is_calibrated": self.is_calibrated_,
        }