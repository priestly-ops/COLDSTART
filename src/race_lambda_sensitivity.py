from __future__ import annotations

"""RACE shrinkage-family sensitivity helpers.

This module exists so the frozen :class:`src.detectors.RACEDetector` implementation
can remain untouched while reviewer-facing lambda sensitivity is evaluated over
its full algebraic family, including the limiting endpoints lambda=0 and
lambda=+infinity.

For finite positive lambda, this implementation is intentionally identical to
RACEDetector when fed the same preprocessed source/target matrices.
"""

import math

import numpy as np
from sklearn.covariance import LedoitWolf

from src.detectors import GaussianMahalanobisDetector, RACEDetector


RACE_LAMBDA_GRID: tuple[float, ...] = (
    0.0,
    10.0,
    30.0,
    60.0,
    100.0,
    300.0,
    math.inf,
)
FROZEN_RACE_LAMBDA = 60.0


def lambda_label(value: float) -> str:
    value = float(value)
    if math.isnan(value) or value < 0.0:
        raise ValueError("lambda must be non-negative or +infinity")
    if math.isinf(value):
        if value < 0.0:
            raise ValueError("lambda must be non-negative or +infinity")
        return "inf"
    if value.is_integer():
        return str(int(value))
    return format(value, ".12g")


def target_weight_for_lambda(target_count: int, lambda_reg: float) -> float:
    if int(target_count) < 1:
        raise ValueError("target_count must be >= 1")
    value = float(lambda_reg)
    if math.isnan(value) or value < 0.0:
        raise ValueError("lambda_reg must be non-negative or +infinity")
    if math.isinf(value):
        if value < 0.0:
            raise ValueError("lambda_reg must be non-negative or +infinity")
        return 0.0
    if value == 0.0:
        return 1.0
    return float(target_count / (target_count + value))


class RACELambdaSensitivityDetector(GaussianMahalanobisDetector):
    """RACE Gaussian shrinkage family with explicit endpoint support.

    lambda=0 gives the target Gaussian endpoint and lambda=+infinity gives the
    source Gaussian endpoint. Both endpoints retain the RACE experiment's pooled
    preprocessing when used through ``fit_detector(detector_name='RACE', ...)``.
    They should therefore be described as *RACE-family endpoints*, not silently
    substituted for the standalone TargetOnly/SourceOnly experiment pipelines.
    """

    def __init__(
        self,
        lambda_reg: float,
        false_alert_budget: float = 0.01,
    ) -> None:
        super().__init__(false_alert_budget=false_alert_budget)
        value = float(lambda_reg)
        if math.isnan(value) or value < 0.0:
            raise ValueError("lambda_reg must be non-negative or +infinity")
        if math.isinf(value) and value < 0.0:
            raise ValueError("lambda_reg must be non-negative or +infinity")
        self.lambda_reg = value
        self.target_weight_: float | None = None
        self.source_location_: np.ndarray | None = None
        self.target_location_: np.ndarray | None = None
        self.source_covariance_: np.ndarray | None = None
        self.target_covariance_: np.ndarray | None = None

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "RACELambdaSensitivityDetector":
        source_features = self._validate_features(source_features)
        target_features = self._validate_features(target_features)

        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError("Source and target feature dimensions differ.")
        if source_features.shape[0] < 2:
            raise ValueError("At least two source cycles are required.")
        if target_features.shape[0] < 2:
            raise ValueError("At least two target cycles are required.")

        source_estimator = LedoitWolf(
            assume_centered=False,
            store_precision=False,
        ).fit(source_features)
        target_estimator = LedoitWolf(
            assume_centered=False,
            store_precision=False,
        ).fit(target_features)

        source_mean = np.asarray(source_estimator.location_, dtype=np.float64)
        target_mean = np.asarray(target_estimator.location_, dtype=np.float64)
        source_covariance = np.asarray(source_estimator.covariance_, dtype=np.float64)
        target_covariance = np.asarray(target_estimator.covariance_, dtype=np.float64)

        weight = target_weight_for_lambda(target_features.shape[0], self.lambda_reg)
        adapted_mean = (1.0 - weight) * source_mean + weight * target_mean
        adapted_covariance = (
            (1.0 - weight) * source_covariance
            + weight * target_covariance
        )
        adapted_covariance = RACEDetector._stabilize_covariance(adapted_covariance)
        precision = np.linalg.pinv(adapted_covariance, hermitian=True)
        if not np.isfinite(precision).all():
            raise RuntimeError("RACE sensitivity precision contains NaN or Inf.")

        self.source_location_ = source_mean
        self.target_location_ = target_mean
        self.source_covariance_ = source_covariance
        self.target_covariance_ = target_covariance
        self.location_ = adapted_mean
        self.precision_ = precision
        self.feature_count_ = source_features.shape[1]
        self.target_weight_ = float(weight)
        self.is_fitted_ = True
        return self

    def get_params(self) -> dict[str, object]:
        params = super().get_params()
        params.update(
            {
                "lambda_reg": self.lambda_reg,
                "lambda_label": lambda_label(self.lambda_reg),
                "target_weight": self.target_weight_,
            }
        )
        return params
