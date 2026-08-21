from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.calibration_tail import conformal_threshold_info


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
        self.calibration_rank_: int | None = None
        self.calibration_regime_: str | None = None
        self.calibration_size_: int | None = None
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

    def _calibrate_scores(
        self,
        scores: np.ndarray,
    ) -> "BaseDetector":
        """Apply the frozen deterministic nonrandomized conformal rule."""
        values = self._validate_scores(scores)
        info = conformal_threshold_info(
            values,
            alpha=self.false_alert_budget,
        )

        self.threshold_ = float(info.strict_threshold)
        self.calibration_rank_ = int(info.raw_rank)
        self.calibration_size_ = int(info.calibration_size)

        if not info.finite_sample_feasible:
            self.calibration_regime_ = "infinite"
        elif info.threshold_is_maximum:
            self.calibration_regime_ = "maximum"
        else:
            self.calibration_regime_ = "submaximum"

        self.is_calibrated_ = True
        return self

    def calibrate(
        self,
        calibration_features: np.ndarray,
    ) -> "BaseDetector":
        """Fit the frozen split-conformal threshold on healthy data.

        The calibration target is alpha == false_alert_budget. The requested
        one-indexed order-statistic rank is

            ceil((m + 1) * (1 - alpha)).

        If that rank exceeds m, the strict deterministic threshold is +inf,
        which produces no alarms for finite anomaly scores.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "Detector must be fitted before calibration."
            )

        scores = self.score_samples(calibration_features)
        return self._calibrate_scores(scores)

    def calibrate_from_scores(
        self,
        calibration_scores: np.ndarray,
    ) -> "BaseDetector":
        """Calibrate from precomputed healthy anomaly scores."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Detector must be fitted before calibration."
            )

        return self._calibrate_scores(calibration_scores)

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
        """Return the frozen strict finite-sample conformal threshold."""
        info = conformal_threshold_info(
            BaseDetector._validate_scores(scores),
            alpha=alpha,
        )
        return float(info.strict_threshold)

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
            "calibration_rank": self.calibration_rank_,
            "calibration_regime": self.calibration_regime_,
            "calibration_size": self.calibration_size_,
            "is_fitted": self.is_fitted_,
            "is_calibrated": self.is_calibrated_,
        }
