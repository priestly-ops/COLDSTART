from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from src.base_detector import BaseDetector


class GaussianMahalanobisDetector(BaseDetector):
    """Gaussian anomaly detector using Ledoit-Wolf covariance."""

    def __init__(
        self,
        false_alert_budget: float = 0.01,
    ) -> None:
        super().__init__(
            false_alert_budget=false_alert_budget
        )

        self.location_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.feature_count_: int | None = None

    def _fit_gaussian(
        self,
        training_features: np.ndarray,
    ) -> None:
        training_features = self._validate_features(
            training_features
        )

        if training_features.shape[0] < 2:
            raise ValueError(
                "At least two training cycles are required."
            )

        estimator = LedoitWolf(
            assume_centered=False,
            store_precision=True,
        )
        estimator.fit(training_features)

        self.location_ = np.asarray(
            estimator.location_,
            dtype=np.float64,
        )
        self.precision_ = np.asarray(
            estimator.precision_,
            dtype=np.float64,
        )
        self.feature_count_ = training_features.shape[1]

        if not np.isfinite(self.location_).all():
            raise RuntimeError(
                "Gaussian location contains NaN or Inf."
            )

        if not np.isfinite(self.precision_).all():
            raise RuntimeError(
                "Gaussian precision contains NaN or Inf."
            )

        self.is_fitted_ = True

    def score_samples(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if (
            not self.is_fitted_
            or self.location_ is None
            or self.precision_ is None
            or self.feature_count_ is None
        ):
            raise RuntimeError(
                "Detector must be fitted before scoring."
            )

        features = self._validate_features(features)

        if features.shape[1] != self.feature_count_:
            raise ValueError(
                f"Expected {self.feature_count_} features, "
                f"received {features.shape[1]}."
            )

        centered = features - self.location_

        squared_distances = np.einsum(
            "ij,jk,ik->i",
            centered,
            self.precision_,
            centered,
            optimize=True,
        )

        squared_distances = np.maximum(
            squared_distances,
            0.0,
        )

        return np.sqrt(squared_distances)


class TargetOnlyDetector(GaussianMahalanobisDetector):
    """Fit exclusively on target commissioning cycles."""

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "TargetOnlyDetector":
        del source_features

        target_features = self._validate_features(
            target_features
        )
        self._fit_gaussian(target_features)

        return self


class SourceOnlyDetector(GaussianMahalanobisDetector):
    """Fit exclusively on source healthy cycles."""

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "SourceOnlyDetector":
        del target_features

        source_features = self._validate_features(
            source_features
        )
        self._fit_gaussian(source_features)

        return self


class PooledDetector(GaussianMahalanobisDetector):
    """Fit on concatenated source and target healthy cycles."""

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "PooledDetector":
        source_features = self._validate_features(
            source_features
        )
        target_features = self._validate_features(
            target_features
        )

        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError(
                "Source and target feature dimensions differ."
            )

        pooled = np.vstack(
            (source_features, target_features)
        )

        self._fit_gaussian(pooled)
        return self


class RACEDetector(GaussianMahalanobisDetector):
    """Robust Adaptation under Covariate Shift.

    Source and target Gaussians are estimated independently using
    Ledoit-Wolf covariance. Their means and covariances are combined using:

        weight = N / (N + lambda_reg)

    The target estimate therefore receives increasing influence as more
    commissioning cycles become available.
    """

    def __init__(
        self,
        lambda_reg: float = 60.0,
        false_alert_budget: float = 0.01,
    ) -> None:
        super().__init__(
            false_alert_budget=false_alert_budget
        )

        if lambda_reg <= 0:
            raise ValueError(
                "lambda_reg must be positive."
            )

        self.lambda_reg = float(lambda_reg)
        self.target_weight_: float | None = None
        self.source_location_: np.ndarray | None = None
        self.target_location_: np.ndarray | None = None
        self.source_covariance_: np.ndarray | None = None
        self.target_covariance_: np.ndarray | None = None

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "RACEDetector":
        source_features = self._validate_features(
            source_features
        )
        target_features = self._validate_features(
            target_features
        )

        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError(
                "Source and target feature dimensions differ."
            )

        if source_features.shape[0] < 2:
            raise ValueError(
                "At least two source cycles are required."
            )

        if target_features.shape[0] < 2:
            raise ValueError(
                "At least two target cycles are required."
            )

        source_estimator = LedoitWolf(
            assume_centered=False,
            store_precision=False,
        )
        target_estimator = LedoitWolf(
            assume_centered=False,
            store_precision=False,
        )

        source_estimator.fit(source_features)
        target_estimator.fit(target_features)

        source_mean = np.asarray(
            source_estimator.location_,
            dtype=np.float64,
        )
        target_mean = np.asarray(
            target_estimator.location_,
            dtype=np.float64,
        )

        source_covariance = np.asarray(
            source_estimator.covariance_,
            dtype=np.float64,
        )
        target_covariance = np.asarray(
            target_estimator.covariance_,
            dtype=np.float64,
        )

        target_count = target_features.shape[0]
        weight = target_count / (
            target_count + self.lambda_reg
        )

        adapted_mean = (
            (1.0 - weight) * source_mean
            + weight * target_mean
        )

        adapted_covariance = (
            (1.0 - weight) * source_covariance
            + weight * target_covariance
        )

        adapted_covariance = self._stabilize_covariance(
            adapted_covariance
        )

        precision = np.linalg.pinv(
            adapted_covariance,
            hermitian=True,
        )

        if not np.isfinite(precision).all():
            raise RuntimeError(
                "RACE precision matrix contains NaN or Inf."
            )

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
                "target_weight": self.target_weight_,
            }
        )
        return params

    @staticmethod
    def _stabilize_covariance(
        covariance: np.ndarray,
        minimum_eigenvalue: float = 1e-8,
    ) -> np.ndarray:
        covariance = np.asarray(
            covariance,
            dtype=np.float64,
        )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        eigenvalues = np.maximum(
            eigenvalues,
            minimum_eigenvalue,
        )

        stabilized = (
            eigenvectors
            @ np.diag(eigenvalues)
            @ eigenvectors.T
        )

        return 0.5 * (
            stabilized + stabilized.T
        )