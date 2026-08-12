from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf

from src.base_detector import BaseDetector


OriginalRaceVariant = Literal[
    "TargetOnly",
    "OriginalRACE",
    "MeanTransferOnly",
    "CovarianceTransferOnly",
    "EigenvectorsOnly",
    "EigenvaluesOnly",
    "TargetMeanSourceCovariance",
    "SourceMeanTargetCovariance",
]


FROZEN_ORIGINAL_RACE_ABLATIONS: tuple[OriginalRaceVariant, ...] = (
    "TargetOnly",
    "OriginalRACE",
    "MeanTransferOnly",
    "CovarianceTransferOnly",
    "EigenvectorsOnly",
    "EigenvaluesOnly",
    "TargetMeanSourceCovariance",
    "SourceMeanTargetCovariance",
)


@dataclass(frozen=True)
class GaussianFit:
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_covariance: np.ndarray
    target_covariance: np.ndarray
    race_mean: np.ndarray
    race_covariance: np.ndarray
    target_weight: float
    source_shrinkage: float
    target_shrinkage: float


def fit_source_target_gaussians(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    lambda_reg: float = 60.0,
) -> GaussianFit:
    source = _validate_matrix(source_features)
    target = _validate_matrix(target_features)
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature dimensions differ.")
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("At least two source and target cycles are required.")
    if lambda_reg <= 0.0:
        raise ValueError("lambda_reg must be positive.")

    source_estimator = LedoitWolf(assume_centered=False, store_precision=False)
    target_estimator = LedoitWolf(assume_centered=False, store_precision=False)
    source_estimator.fit(source)
    target_estimator.fit(target)

    source_mean = np.asarray(source_estimator.location_, dtype=np.float64)
    target_mean = np.asarray(target_estimator.location_, dtype=np.float64)
    source_covariance = stabilize_covariance(source_estimator.covariance_)
    target_covariance = stabilize_covariance(target_estimator.covariance_)
    target_weight = float(target.shape[0] / (target.shape[0] + lambda_reg))
    race_mean = (1.0 - target_weight) * source_mean + target_weight * target_mean
    race_covariance = stabilize_covariance(
        (1.0 - target_weight) * source_covariance
        + target_weight * target_covariance
    )

    return GaussianFit(
        source_mean=source_mean,
        target_mean=target_mean,
        source_covariance=source_covariance,
        target_covariance=target_covariance,
        race_mean=race_mean,
        race_covariance=race_covariance,
        target_weight=target_weight,
        source_shrinkage=float(getattr(source_estimator, "shrinkage_", np.nan)),
        target_shrinkage=float(getattr(target_estimator, "shrinkage_", np.nan)),
    )


def build_original_race_component(
    fit: GaussianFit,
    variant: OriginalRaceVariant,
) -> tuple[np.ndarray, np.ndarray]:
    target_values, target_vectors = sorted_eigh(fit.target_covariance)
    source_values, source_vectors = sorted_eigh(fit.source_covariance)

    if variant == "TargetOnly":
        mean = fit.target_mean
        covariance = fit.target_covariance
    elif variant == "OriginalRACE":
        mean = fit.race_mean
        covariance = fit.race_covariance
    elif variant == "MeanTransferOnly":
        mean = fit.race_mean
        covariance = fit.target_covariance
    elif variant == "CovarianceTransferOnly":
        mean = fit.target_mean
        covariance = fit.race_covariance
    elif variant == "EigenvectorsOnly":
        mean = fit.target_mean
        covariance = source_vectors @ np.diag(target_values) @ source_vectors.T
    elif variant == "EigenvaluesOnly":
        mean = fit.target_mean
        covariance = target_vectors @ np.diag(source_values) @ target_vectors.T
    elif variant == "TargetMeanSourceCovariance":
        mean = fit.target_mean
        covariance = fit.source_covariance
    elif variant == "SourceMeanTargetCovariance":
        mean = fit.source_mean
        covariance = fit.target_covariance
    else:
        raise ValueError(f"Unknown Original RACE ablation variant: {variant}")

    return np.asarray(mean, dtype=np.float64), stabilize_covariance(covariance)


class OriginalRaceComponentDetector(BaseDetector):
    def __init__(
        self,
        *,
        variant: OriginalRaceVariant,
        lambda_reg: float = 60.0,
        false_alert_budget: float = 0.01,
    ) -> None:
        super().__init__(false_alert_budget=false_alert_budget)
        self.variant = variant
        self.lambda_reg = float(lambda_reg)
        self.fit_: GaussianFit | None = None
        self.location_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.feature_count_: int | None = None

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "OriginalRaceComponentDetector":
        fit = fit_source_target_gaussians(
            source_features,
            target_features,
            lambda_reg=self.lambda_reg,
        )
        mean, covariance = build_original_race_component(fit, self.variant)
        precision = np.linalg.pinv(covariance, hermitian=True)
        if not np.isfinite(precision).all():
            raise RuntimeError("Component precision contains NaN or Inf.")
        self.fit_ = fit
        self.location_ = mean
        self.covariance_ = covariance
        self.precision_ = precision
        self.feature_count_ = mean.shape[0]
        self.is_fitted_ = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if (
            not self.is_fitted_
            or self.location_ is None
            or self.precision_ is None
            or self.feature_count_ is None
        ):
            raise RuntimeError("Detector must be fitted before scoring.")
        x = _validate_matrix(features)
        if x.shape[1] != self.feature_count_:
            raise ValueError(
                f"Expected {self.feature_count_} features, received {x.shape[1]}."
            )
        centered = x - self.location_
        squared = np.einsum("ij,jk,ik->i", centered, self.precision_, centered, optimize=True)
        return np.sqrt(np.maximum(squared, 0.0))


def directional_original_race_audit(
    fit: GaussianFit,
    *,
    calibration: np.ndarray,
    healthy_eval: np.ndarray,
    anomaly_eval: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    calibration = _validate_matrix(calibration)
    healthy_eval = _validate_matrix(healthy_eval)
    anomaly = None if anomaly_eval is None else _validate_matrix(anomaly_eval)

    target_values, target_vectors = sorted_eigh(fit.target_covariance)
    source_values, source_vectors = sorted_eigh(fit.source_covariance)
    delta_mean = fit.race_mean - fit.target_mean
    eps = 1e-12
    rows: list[dict[str, float | int]] = []

    for j in range(target_vectors.shape[1]):
        direction = target_vectors[:, j]
        target_var = float(max(target_values[j], eps))
        source_projected_var = float(direction @ fit.source_covariance @ direction)
        race_projected_var = float(direction @ fit.race_covariance @ direction)
        alignment = float(np.max((source_vectors.T @ direction) ** 2))
        var_ratio = float((source_projected_var + eps) / (target_var + eps))
        variance_compatibility = float(np.exp(-abs(np.log(var_ratio))))
        healthy_compatibility = float(alignment * variance_compatibility)

        cal_target = _directional_terms(calibration, fit.target_mean, direction, target_var)
        cal_race = _directional_terms(calibration, fit.race_mean, direction, race_projected_var)
        healthy_target = _directional_terms(healthy_eval, fit.target_mean, direction, target_var)
        healthy_race = _directional_terms(healthy_eval, fit.race_mean, direction, race_projected_var)

        row: dict[str, float | int] = {
            "direction": int(j),
            "target_eigenvalue": target_var,
            "source_eigenvalue_same_rank": float(source_values[j]),
            "source_projected_variance": source_projected_var,
            "race_projected_variance": race_projected_var,
            "covariance_delta_projected": float(race_projected_var - target_var),
            "source_target_eigenvector_alignment_max_cos2": alignment,
            "variance_compatibility": variance_compatibility,
            "healthy_compatibility": healthy_compatibility,
            "mean_transfer_projection": float(delta_mean @ direction),
            "calibration_score_contribution_change_mean": float(np.mean(cal_race - cal_target)),
            "healthy_score_contribution_change_mean": float(np.mean(healthy_race - healthy_target)),
            "healthy_score_contribution_change_abs_mean": float(np.mean(np.abs(healthy_race - healthy_target))),
        }
        if anomaly is not None:
            anomaly_target = _directional_terms(anomaly, fit.target_mean, direction, target_var)
            anomaly_race = _directional_terms(anomaly, fit.race_mean, direction, race_projected_var)
            row.update(
                {
                    "posthoc_target_direction_separation": float(
                        np.mean(anomaly_target) - np.mean(healthy_target)
                    ),
                    "posthoc_race_direction_separation": float(
                        np.mean(anomaly_race) - np.mean(healthy_race)
                    ),
                    "posthoc_direction_separation_change": float(
                        (np.mean(anomaly_race) - np.mean(healthy_race))
                        - (np.mean(anomaly_target) - np.mean(healthy_target))
                    ),
                }
            )
        rows.append(row)
    return rows


def sorted_eigh(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(stabilize_covariance(covariance))
    order = np.argsort(values)[::-1]
    return np.asarray(values[order], dtype=np.float64), np.asarray(vectors[:, order], dtype=np.float64)


def stabilize_covariance(
    covariance: np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-8,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, minimum_eigenvalue)
    stabilized = vectors @ np.diag(values) @ vectors.T
    return 0.5 * (stabilized + stabilized.T)


def covariance_condition(covariance: np.ndarray) -> tuple[float, float, float]:
    values = np.linalg.eigvalsh(stabilize_covariance(covariance))
    return float(values.min()), float(values.max()), float(values.max() / max(values.min(), 1e-12))


def _directional_terms(
    features: np.ndarray,
    mean: np.ndarray,
    direction: np.ndarray,
    variance: float,
) -> np.ndarray:
    projected = (features - mean) @ direction
    return (projected * projected) / max(float(variance), 1e-12)


def _validate_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Expected a non-empty 2D feature matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or Inf.")
    return matrix
