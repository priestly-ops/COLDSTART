from __future__ import annotations

"""
Dataset-agnostic statistical feature extraction for robot executions.

This module works with the common cycle interface returned by both:

- ``src.voraus_loader``
- ``src.aursad_loader``

It intentionally does not import either loader. This avoids pulling
dataset-specific dependencies into feature extraction and lets both datasets
reuse the same downstream evaluation code.

Feature order
-------------
For backward compatibility with the existing voraus-AD experiments, features
are emitted signal-major in this frozen order:

    mean, std, median, q25, q75, total_variation

For 48 AURSAD channels, this produces 48 * 6 = 288 features per execution.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


FEATURE_CACHE_VERSION = "robot-cycle-statistical-features-v1"

STATISTIC_NAMES = (
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "total_variation",
)


@runtime_checkable
class RobotCycleLike(Protocol):
    """Structural interface required by this feature extractor."""

    episode_id: int
    values: np.ndarray
    columns: tuple[str, ...]
    anomaly: bool
    category: int
    setting: int


def _validate_cycle(
    cycle: RobotCycleLike,
) -> np.ndarray:
    """Validate one cycle and return its values as float64."""
    values = np.asarray(
        cycle.values,
        dtype=np.float64,
    )

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

    if values.shape[1] == 0:
        raise ValueError(
            f"Episode {cycle.episode_id} contains zero signals."
        )

    if len(cycle.columns) != values.shape[1]:
        raise ValueError(
            f"Episode {cycle.episode_id} has {values.shape[1]} "
            f"value columns but {len(cycle.columns)} column names."
        )

    if len(set(cycle.columns)) != len(cycle.columns):
        raise ValueError(
            f"Episode {cycle.episode_id} contains duplicate "
            "signal-column names."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Episode {cycle.episode_id} contains NaN or Inf."
        )

    return values


def extract_cycle_features(
    cycle: RobotCycleLike,
) -> np.ndarray:
    """Extract fixed-length statistical features from one execution.

    Features are calculated from the original, unpadded sequence. The output
    is signal-major:

        signal_1_mean
        signal_1_std
        signal_1_median
        signal_1_q25
        signal_1_q75
        signal_1_total_variation
        signal_2_mean
        ...

    Args:
        cycle: One robot execution exposing the RobotCycleLike interface.

    Returns:
        A finite one-dimensional NumPy feature vector.
    """
    values = _validate_cycle(
        cycle
    )

    means = np.mean(
        values,
        axis=0,
    )

    standard_deviations = np.std(
        values,
        axis=0,
        ddof=1,
    )

    medians = np.median(
        values,
        axis=0,
    )

    q25 = np.quantile(
        values,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        values,
        0.75,
        axis=0,
    )

    total_variation = np.sum(
        np.abs(
            np.diff(
                values,
                axis=0,
            )
        ),
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

    features = feature_matrix.reshape(
        -1
    )

    expected_size = (
        values.shape[1]
        * len(STATISTIC_NAMES)
    )

    if features.shape != (
        expected_size,
    ):
        raise RuntimeError(
            f"Episode {cycle.episode_id} produced feature shape "
            f"{features.shape}; expected ({expected_size},)."
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite features generated for episode "
            f"{cycle.episode_id}."
        )

    return features


def extract_feature_matrix(
    cycles: Sequence[RobotCycleLike],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and execution IDs from a cycle collection.

    This function preserves the original public return type used by the
    existing voraus-AD pipeline.

    Args:
        cycles: Non-empty sequence of complete robot executions.

    Returns:
        ``(feature_matrix, episode_ids)``.
    """
    batch = extract_feature_batch(
        cycles
    )

    return (
        batch.features,
        batch.episode_ids,
    )


def make_feature_names(
    signal_columns: Sequence[str],
) -> tuple[str, ...]:
    """Create names matching the exact order of extract_cycle_features."""
    columns = tuple(
        str(column)
        for column in signal_columns
    )

    if not columns:
        raise ValueError(
            "At least one signal column is required."
        )

    if len(set(columns)) != len(columns):
        raise ValueError(
            "Signal column names must be unique."
        )

    names: list[str] = []

    for signal in columns:
        for statistic in STATISTIC_NAMES:
            names.append(
                f"{signal}__{statistic}"
            )

    return tuple(names)


@dataclass(frozen=True)
class FeatureBatch:
    """Feature matrix plus aligned episode metadata."""

    features: np.ndarray
    episode_ids: np.ndarray
    anomaly_labels: np.ndarray
    categories: np.ndarray
    settings: np.ndarray
    feature_names: tuple[str, ...]
    signal_columns: tuple[str, ...]
    statistic_names: tuple[str, ...] = STATISTIC_NAMES

    def __post_init__(
        self,
    ) -> None:
        features = np.asarray(
            self.features,
            dtype=np.float64,
        )

        episode_ids = np.asarray(
            self.episode_ids,
            dtype=np.int64,
        )

        anomaly_labels = np.asarray(
            self.anomaly_labels,
            dtype=np.bool_,
        )

        categories = np.asarray(
            self.categories,
            dtype=np.int64,
        )

        settings = np.asarray(
            self.settings,
            dtype=np.int64,
        )

        if features.ndim != 2:
            raise ValueError(
                f"features must be 2D; received shape "
                f"{features.shape}."
            )

        row_count = features.shape[0]

        for name, values in (
            (
                "episode_ids",
                episode_ids,
            ),
            (
                "anomaly_labels",
                anomaly_labels,
            ),
            (
                "categories",
                categories,
            ),
            (
                "settings",
                settings,
            ),
        ):
            if values.ndim != 1:
                raise ValueError(
                    f"{name} must be 1D; received shape "
                    f"{values.shape}."
                )

            if len(values) != row_count:
                raise ValueError(
                    f"{name} contains {len(values)} rows but "
                    f"features contains {row_count}."
                )

        if row_count == 0:
            raise ValueError(
                "FeatureBatch cannot contain zero executions."
            )

        if features.shape[1] == 0:
            raise ValueError(
                "FeatureBatch cannot contain zero features."
            )

        if len(set(
            episode_ids.tolist()
        )) != row_count:
            raise ValueError(
                "FeatureBatch episode_ids must be unique."
            )

        if features.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                f"Feature matrix contains {features.shape[1]} "
                f"columns but {len(self.feature_names)} feature "
                "names were supplied."
            )

        expected_feature_count = (
            len(self.signal_columns)
            * len(self.statistic_names)
        )

        if features.shape[1] != expected_feature_count:
            raise ValueError(
                "Feature width does not match the signal/statistic "
                f"schema: {features.shape[1]} != "
                f"{len(self.signal_columns)} * "
                f"{len(self.statistic_names)}."
            )

        if len(set(
            self.feature_names
        )) != len(self.feature_names):
            raise ValueError(
                "feature_names must be unique."
            )

        if len(set(
            self.signal_columns
        )) != len(self.signal_columns):
            raise ValueError(
                "signal_columns must be unique."
            )

        if not np.isfinite(
            features
        ).all():
            raise ValueError(
                "FeatureBatch contains NaN or Inf."
            )

        object.__setattr__(
            self,
            "features",
            features,
        )

        object.__setattr__(
            self,
            "episode_ids",
            episode_ids,
        )

        object.__setattr__(
            self,
            "anomaly_labels",
            anomaly_labels,
        )

        object.__setattr__(
            self,
            "categories",
            categories,
        )

        object.__setattr__(
            self,
            "settings",
            settings,
        )

    def select_episode_ids(
        self,
        episode_ids: Sequence[int],
        *,
        preserve_requested_order: bool = True,
        require_all: bool = True,
    ) -> "FeatureBatch":
        """Return a subset without recomputing time-series features."""
        requested = [
            int(value)
            for value in episode_ids
        ]

        if not requested:
            raise ValueError(
                "At least one episode ID must be requested."
            )

        if len(set(
            requested
        )) != len(requested):
            raise ValueError(
                "Requested episode IDs contain duplicates."
            )

        row_by_id = {
            int(episode_id): index
            for index, episode_id
            in enumerate(
                self.episode_ids
            )
        }

        missing = [
            episode_id
            for episode_id in requested
            if episode_id not in row_by_id
        ]

        if missing and require_all:
            raise KeyError(
                "Feature cache is missing requested episode IDs: "
                f"{missing[:20]}"
            )

        requested_set = set(
            requested
        )

        if preserve_requested_order:
            selected_ids = [
                episode_id
                for episode_id in requested
                if episode_id in row_by_id
            ]

            row_indices = np.asarray(
                [
                    row_by_id[episode_id]
                    for episode_id
                    in selected_ids
                ],
                dtype=np.int64,
            )
        else:
            mask = np.asarray(
                [
                    int(episode_id)
                    in requested_set
                    for episode_id
                    in self.episode_ids
                ],
                dtype=bool,
            )

            row_indices = np.flatnonzero(
                mask
            )

        if row_indices.size == 0:
            raise ValueError(
                "No requested episode IDs were found."
            )

        return FeatureBatch(
            features=self.features[
                row_indices
            ],
            episode_ids=self.episode_ids[
                row_indices
            ],
            anomaly_labels=self.anomaly_labels[
                row_indices
            ],
            categories=self.categories[
                row_indices
            ],
            settings=self.settings[
                row_indices
            ],
            feature_names=self.feature_names,
            signal_columns=self.signal_columns,
            statistic_names=self.statistic_names,
        )


def extract_feature_batch(
    cycles: Sequence[RobotCycleLike],
) -> FeatureBatch:
    """Extract features and aligned metadata from complete executions."""
    if not cycles:
        raise ValueError(
            "At least one cycle is required."
        )

    expected_columns = tuple(
        cycles[0].columns
    )

    if not expected_columns:
        raise ValueError(
            "Cycles must contain at least one signal column."
        )

    seen_episode_ids: set[int] = set()

    for cycle in cycles:
        episode_id = int(
            cycle.episode_id
        )

        if episode_id in seen_episode_ids:
            raise ValueError(
                f"Duplicate episode ID found: {episode_id}"
            )

        seen_episode_ids.add(
            episode_id
        )

        if tuple(
            cycle.columns
        ) != expected_columns:
            raise ValueError(
                f"Episode {episode_id} has a different "
                "signal schema or column order."
            )

    matrix = np.vstack(
        [
            extract_cycle_features(
                cycle
            )
            for cycle in cycles
        ]
    )

    episode_ids = np.asarray(
        [
            int(
                cycle.episode_id
            )
            for cycle in cycles
        ],
        dtype=np.int64,
    )

    anomaly_labels = np.asarray(
        [
            bool(
                cycle.anomaly
            )
            for cycle in cycles
        ],
        dtype=np.bool_,
    )

    categories = np.asarray(
        [
            int(
                cycle.category
            )
            for cycle in cycles
        ],
        dtype=np.int64,
    )

    settings = np.asarray(
        [
            int(
                cycle.setting
            )
            for cycle in cycles
        ],
        dtype=np.int64,
    )

    return FeatureBatch(
        features=matrix,
        episode_ids=episode_ids,
        anomaly_labels=anomaly_labels,
        categories=categories,
        settings=settings,
        feature_names=make_feature_names(
            expected_columns
        ),
        signal_columns=expected_columns,
    )


def save_feature_batch(
    batch: FeatureBatch,
    path: Path | str,
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Save a deterministic compressed feature cache."""
    output_path = Path(
        path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload_metadata = {
        "cache_version": (
            FEATURE_CACHE_VERSION
        ),
        "execution_count": int(
            batch.features.shape[0]
        ),
        "feature_count": int(
            batch.features.shape[1]
        ),
        "signal_count": int(
            len(
                batch.signal_columns
            )
        ),
        "statistic_names": list(
            batch.statistic_names
        ),
    }

    if metadata:
        payload_metadata[
            "user_metadata"
        ] = metadata

    np.savez_compressed(
        output_path,
        features=batch.features,
        episode_ids=batch.episode_ids,
        anomaly_labels=(
            batch.anomaly_labels
        ),
        categories=batch.categories,
        settings=batch.settings,
        feature_names=np.asarray(
            batch.feature_names,
            dtype=np.str_,
        ),
        signal_columns=np.asarray(
            batch.signal_columns,
            dtype=np.str_,
        ),
        statistic_names=np.asarray(
            batch.statistic_names,
            dtype=np.str_,
        ),
        metadata_json=np.asarray(
            json.dumps(
                payload_metadata,
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    )

    return output_path


def load_feature_batch(
    path: Path | str,
) -> FeatureBatch:
    """Load and validate a compressed feature cache."""
    input_path = Path(
        path
    ).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature cache not found: {input_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as archive:
        required = {
            "features",
            "episode_ids",
            "anomaly_labels",
            "categories",
            "settings",
            "feature_names",
            "signal_columns",
            "statistic_names",
            "metadata_json",
        }

        missing = sorted(
            required
            - set(
                archive.files
            )
        )

        if missing:
            raise ValueError(
                "Feature cache is missing arrays: "
                f"{missing}"
            )

        metadata = json.loads(
            str(
                archive[
                    "metadata_json"
                ].item()
            )
        )

        if metadata.get(
            "cache_version"
        ) != FEATURE_CACHE_VERSION:
            raise ValueError(
                "Unsupported feature-cache version: "
                f"{metadata.get('cache_version')!r}"
            )

        batch = FeatureBatch(
            features=np.asarray(
                archive["features"],
                dtype=np.float64,
            ),
            episode_ids=np.asarray(
                archive["episode_ids"],
                dtype=np.int64,
            ),
            anomaly_labels=np.asarray(
                archive[
                    "anomaly_labels"
                ],
                dtype=np.bool_,
            ),
            categories=np.asarray(
                archive["categories"],
                dtype=np.int64,
            ),
            settings=np.asarray(
                archive["settings"],
                dtype=np.int64,
            ),
            feature_names=tuple(
                str(value)
                for value
                in archive[
                    "feature_names"
                ].tolist()
            ),
            signal_columns=tuple(
                str(value)
                for value
                in archive[
                    "signal_columns"
                ].tolist()
            ),
            statistic_names=tuple(
                str(value)
                for value
                in archive[
                    "statistic_names"
                ].tolist()
            ),
        )

    expected_shape = (
        int(
            metadata[
                "execution_count"
            ]
        ),
        int(
            metadata[
                "feature_count"
            ]
        ),
    )

    if batch.features.shape != expected_shape:
        raise ValueError(
            "Feature-cache shape does not match metadata: "
            f"{batch.features.shape} != {expected_shape}."
        )

    return batch


@dataclass
class FeaturePreprocessor:
    """Training-only variance filtering and standardization."""

    variance_threshold: float = 1e-12

    def __post_init__(
        self,
    ) -> None:
        if self.variance_threshold < 0:
            raise ValueError(
                "variance_threshold cannot be negative."
            )

        self.variance_filter = VarianceThreshold(
            threshold=self.variance_threshold
        )

        self.scaler = StandardScaler()

        self.is_fitted = False
        self.input_feature_count_: (
            int | None
        ) = None
        self.output_feature_count_: (
            int | None
        ) = None

    def fit(
        self,
        training_features: np.ndarray,
    ) -> "FeaturePreprocessor":
        """Fit preprocessing using training data only."""
        training_features = (
            self._validate_matrix(
                training_features
            )
        )

        self.input_feature_count_ = (
            training_features.shape[1]
        )

        try:
            filtered = (
                self.variance_filter
                .fit_transform(
                    training_features
                )
            )
        except ValueError as error:
            raise ValueError(
                "Variance filtering failed. This commonly occurs "
                "when all training features are constant."
            ) from error

        if filtered.shape[1] == 0:
            raise ValueError(
                "All features were removed by variance filtering."
            )

        self.scaler.fit(
            filtered
        )

        self.output_feature_count_ = (
            filtered.shape[1]
        )

        self.is_fitted = True

        return self

    def transform(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Apply fitted filtering and scaling."""
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor must be fitted before "
                "transform()."
            )

        features = self._validate_matrix(
            features
        )

        if (
            features.shape[1]
            != self.input_feature_count_
        ):
            raise ValueError(
                f"Expected {self.input_feature_count_} features, "
                f"received {features.shape[1]}."
            )

        filtered = (
            self.variance_filter
            .transform(
                features
            )
        )

        transformed = self.scaler.transform(
            filtered
        )

        if not np.isfinite(
            transformed
        ).all():
            raise RuntimeError(
                "Preprocessing produced NaN or Inf."
            )

        return transformed

    def fit_transform(
        self,
        training_features: np.ndarray,
    ) -> np.ndarray:
        """Fit on and transform the training matrix."""
        self.fit(
            training_features
        )

        return self.transform(
            training_features
        )

    def selected_feature_mask(
        self,
    ) -> np.ndarray:
        """Return the fitted variance-selection mask."""
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor has not been fitted."
            )

        return self.variance_filter.get_support()

    def selected_feature_names(
        self,
        feature_names: Sequence[str],
    ) -> tuple[str, ...]:
        """Return names retained by fitted variance filtering."""
        if not self.is_fitted:
            raise RuntimeError(
                "FeaturePreprocessor has not been fitted."
            )

        names = tuple(
            str(name)
            for name in feature_names
        )

        if len(
            names
        ) != self.input_feature_count_:
            raise ValueError(
                f"Expected {self.input_feature_count_} feature "
                f"names, received {len(names)}."
            )

        mask = self.selected_feature_mask()

        return tuple(
            name
            for name, keep
            in zip(
                names,
                mask,
            )
            if keep
        )

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )

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

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "Feature matrix contains NaN or Inf."
            )

        return matrix