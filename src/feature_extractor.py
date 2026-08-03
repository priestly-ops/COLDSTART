from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from src.voraus_loader import RobotCycle


STATISTIC_NAMES = (
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "total_variation",
)


def extract_cycle_features(
    cycle: RobotCycle,
) -> np.ndarray:
    """Extract fixed-length statistical features from one cycle.

    Features are calculated from the original unpadded sequence.
    The output order is signal-major:

        signal_1_mean
        signal_1_std
        ...
        signal_2_mean
        ...

    Args:
        cycle: One robot execution.

    Returns:
        A one-dimensional finite NumPy feature vector.
    """
    values = np.asarray(cycle.values, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(
            f"Episode {cycle.episode_id} must be 2D, "
            f"received shape {values.shape}."
        )

    if values.shape[0] < 2:
        raise ValueError(
            f"Episode {cycle.episode_id} contains fewer than "
            "two time steps."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Episode {cycle.episode_id} contains NaN or Inf."
        )

    means = np.mean(values, axis=0)
    standard_deviations = np.std(
        values,
        axis=0,
        ddof=1,
    )
    medians = np.median(values, axis=0)
    q25 = np.quantile(values, 0.25, axis=0)
    q75 = np.quantile(values, 0.75, axis=0)

    total_variation = np.sum(
        np.abs(np.diff(values, axis=0)),
        axis=0,
    )

    feature_matrix = np.column_stack(
        (
            means,
            standard_deviations,
            medians,
            q25,
            q75,
            total_variation,
        )
    )

    features = feature_matrix.reshape(-1)

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite features generated for episode "
            f"{cycle.episode_id}."
        )

    return features


def extract_feature_matrix(
    cycles: Sequence[RobotCycle],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and episode IDs from a cycle collection."""
    if not cycles:
        raise ValueError("At least one cycle is required.")

    expected_columns = cycles[0].columns

    for cycle in cycles:
        if cycle.columns != expected_columns:
            raise ValueError(
                f"Episode {cycle.episode_id} has a different "
                "signal schema."
            )

    matrix = np.vstack(
        [
            extract_cycle_features(cycle)
            for cycle in cycles
        ]
    )

    episode_ids = np.asarray(
        [cycle.episode_id for cycle in cycles],
        dtype=np.int64,
    )

    if matrix.shape[0] != len(cycles):
        raise RuntimeError(
            "Feature row count does not match cycle count."
        )

    return matrix, episode_ids


def make_feature_names(
    signal_columns: Sequence[str],
) -> tuple[str, ...]:
    """Create feature names matching extract_cycle_features."""
    names: list[str] = []

    for signal in signal_columns:
        for statistic in STATISTIC_NAMES:
            names.append(f"{signal}__{statistic}")

    return tuple(names)


@dataclass
class FeaturePreprocessor:
    """Training-only variance filtering and standardization."""

    variance_threshold: float = 1e-12

    def __post_init__(self) -> None:
        if self.variance_threshold < 0:
            raise ValueError(
                "variance_threshold cannot be negative."
            )

        self.variance_filter = VarianceThreshold(
            threshold=self.variance_threshold
        )
        self.scaler = StandardScaler()

        self.is_fitted = False
        self.input_feature_count_: int | None = None
        self.output_feature_count_: int | None = None

    def fit(
        self,
        training_features: np.ndarray,
    ) -> "FeaturePreprocessor":
        training_features = self._validate_matrix(
            training_features
        )

        self.input_feature_count_ = training_features.shape[1]

        filtered = self.variance_filter.fit_transform(
            training_features
        )

        if filtered.shape[1] == 0:
            raise ValueError(
                "All features were removed by variance filtering."
            )

        self.scaler.fit(filtered)

        self.output_feature_count_ = filtered.shape[1]
        self.is_fitted = True

        return self

    def transform(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor must be fitted before transform()."
            )

        features = self._validate_matrix(features)

        if features.shape[1] != self.input_feature_count_:
            raise ValueError(
                f"Expected {self.input_feature_count_} features, "
                f"received {features.shape[1]}."
            )

        filtered = self.variance_filter.transform(features)
        transformed = self.scaler.transform(filtered)

        if not np.isfinite(transformed).all():
            raise RuntimeError(
                "Preprocessing produced NaN or Inf."
            )

        return transformed

    def fit_transform(
        self,
        training_features: np.ndarray,
    ) -> np.ndarray:
        self.fit(training_features)
        return self.transform(training_features)

    def selected_feature_mask(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor has not been fitted."
            )

        return self.variance_filter.get_support()

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, received "
                f"shape {matrix.shape}."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Feature matrix cannot contain zero rows."
            )

        if matrix.shape[1] == 0:
            raise ValueError(
                "Feature matrix cannot contain zero columns."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Feature matrix contains NaN or Inf."
            )

        return matrix