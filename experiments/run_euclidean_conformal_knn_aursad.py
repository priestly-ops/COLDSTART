#!/usr/bin/env python3
"""
experiments/run_euclidean_conformal_knn_aursad.py

Run a target-only Euclidean conformal k-NN commissioning baseline on AURSAD.

Representation
--------------
This baseline operates on the frozen 288-dimensional execution-level
statistical feature vectors:

    48 measured channels * 6 statistics

For every seed and commissioning size:

1. Select healthy target commissioning executions.
2. Fit variance filtering and standardization on commissioning data only.
3. Use Euclidean distance in the transformed feature space.
4. Score each sample by its distance to the k-th nearest healthy
   commissioning neighbor.
5. Select a finite-sample split-conformal threshold from the fixed healthy
   calibration set.
6. Evaluate on the fixed healthy and anomaly evaluation sets.

The nominal k defaults to 10. For commissioning N < k, the effective k is
clipped to N so the detector remains defined:

    effective_k = min(k, N)

This choice is recorded for every run.

Example
-------
Smoke test:

    .\\.venv\\Scripts\\python.exe ^
        experiments\\run_euclidean_conformal_knn_aursad.py ^
        --grid 10,25 ^
        --seeds 0 ^
        --bootstrap-samples 1000 ^
        --overwrite

Full experiment:

    .\\.venv\\Scripts\\python.exe ^
        experiments\\run_euclidean_conformal_knn_aursad.py ^
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import (
    FeatureBatch,
    FeaturePreprocessor,
    load_feature_batch,
)
from src.reproducibility import reproducibility_metadata

DEFAULT_CACHE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "aursad"
    / "feature_cache"
    / "aursad_features.npz"
)

DEFAULT_PROTOCOL_DIR = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
    / "protocol"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "aursad"
    / "euclidean_conformal_knn"
)

DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))

DEFAULT_K = 10
DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_GLOBAL_SEED = 42

PROTOCOL_VERSION = "aursad-euclidean-conformal-knn-v1"
DETECTOR_NAME = "EuclideanConformalKNN"

SEED_RESULT_COLUMNS = (
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "commissioning_count",
    "calibration_count",
    "healthy_eval_count",
    "anomaly_eval_count",
    "retained_features",
    "requested_k",
    "effective_k",
    "conformal_rank",
    "threshold",
    "false_positive_rate",
    "recall",
    "auroc",
    "success",
)

PER_CLASS_COLUMNS = (
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "label",
    "label_name",
    "execution_count",
    "recall",
)


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a regular file: {path}"
        )


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "Duplicate integer values are not allowed."
        )

    return values


def validate_grid(
    values: Iterable[int],
) -> tuple[int, ...]:
    grid = tuple(int(value) for value in values)

    if any(value <= 0 for value in grid):
        raise ValueError(
            "Commissioning sizes must be positive."
        )

    if tuple(sorted(grid)) != grid:
        raise ValueError(
            "Commissioning sizes must be strictly increasing."
        )

    if len(set(grid)) != len(grid):
        raise ValueError(
            "Commissioning sizes must be unique."
        )

    return grid


def validate_seeds(
    values: Iterable[int],
) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)

    if any(seed < 0 for seed in seeds):
        raise ValueError(
            "Seed IDs must be non-negative."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Seed IDs must be unique."
        )

    return seeds


def load_protocol_csv(
    path: Path,
    *,
    expected_partition: str,
) -> pd.DataFrame:
    require_file(
        path,
        f"{expected_partition} protocol CSV",
    )

    frame = pd.read_csv(path)

    required = {
        "sample_nr",
        "partition",
        "label",
        "label_name",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    frame = frame.copy()

    frame["sample_nr"] = pd.to_numeric(
        frame["sample_nr"],
        errors="raise",
    ).astype(np.int64)

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(np.int64)

    frame["partition"] = (
        frame["partition"]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(
        set(frame["partition"].unique())
        - {expected_partition}
    )

    if unexpected:
        raise ValueError(
            f"{path.name} contains unexpected partitions: "
            f"{unexpected}"
        )

    return frame


def load_protocol_tables(
    protocol_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables = {
        "commissioning": load_protocol_csv(
            protocol_dir / "commissioning_ids.csv",
            expected_partition="commissioning",
        ),
        "calibration": load_protocol_csv(
            protocol_dir / "calibration_ids.csv",
            expected_partition="calibration",
        ),
        "healthy_eval": load_protocol_csv(
            protocol_dir / "healthy_eval_ids.csv",
            expected_partition="healthy_eval",
        ),
        "anomaly_eval": load_protocol_csv(
            protocol_dir / "anomaly_eval_ids.csv",
            expected_partition="anomaly_eval",
        ),
    }

    commissioning = tables["commissioning"]

    required = {
        "seed",
        "commissioning_n",
        "selection_rank",
    }

    missing = sorted(
        required - set(commissioning.columns)
    )

    if missing:
        raise ValueError(
            "commissioning_ids.csv is missing columns: "
            f"{missing}"
        )

    for column in (
        "seed",
        "commissioning_n",
        "selection_rank",
    ):
        commissioning[column] = pd.to_numeric(
            commissioning[column],
            errors="raise",
        ).astype(np.int64)

    return tables


def unique_ids(frame: pd.DataFrame) -> np.ndarray:
    ids = frame["sample_nr"].to_numpy(
        dtype=np.int64
    )

    if len(ids) != len(set(ids.tolist())):
        duplicated = (
            frame.loc[
                frame["sample_nr"].duplicated(
                    keep=False
                ),
                "sample_nr",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Duplicate sample_nr values found: {duplicated[:20]}"
        )

    return ids


def validate_protocol(
    tables: dict[str, pd.DataFrame],
    grid: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    commissioning = tables["commissioning"]
    calibration = tables["calibration"]
    healthy_eval = tables["healthy_eval"]
    anomaly_eval = tables["anomaly_eval"]

    if not commissioning["label"].eq(0).all():
        raise ValueError(
            "Commissioning contains non-normal executions."
        )

    if not calibration["label"].eq(0).all():
        raise ValueError(
            "Calibration contains non-normal executions."
        )

    if not healthy_eval["label"].eq(0).all():
        raise ValueError(
            "Healthy evaluation contains non-normal executions."
        )

    if not anomaly_eval["label"].isin(
        [1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Anomaly evaluation contains unsupported labels."
        )

    fixed_sets = {
        "calibration": set(
            unique_ids(calibration).tolist()
        ),
        "healthy_eval": set(
            unique_ids(healthy_eval).tolist()
        ),
        "anomaly_eval": set(
            unique_ids(anomaly_eval).tolist()
        ),
    }

    names = list(fixed_sets)

    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = (
                fixed_sets[left_name]
                & fixed_sets[right_name]
            )

            if overlap:
                raise RuntimeError(
                    f"Protocol leakage between {left_name} and "
                    f"{right_name}: {sorted(overlap)[:20]}"
                )

    for seed in seeds:
        seed_rows = commissioning[
            commissioning["seed"].eq(seed)
        ]

        if seed_rows.empty:
            raise ValueError(
                f"Commissioning protocol is missing seed {seed}."
            )

        previous_ids: set[int] = set()

        for n_value in grid:
            subset = seed_rows[
                seed_rows["commissioning_n"].eq(
                    n_value
                )
            ].sort_values(
                "selection_rank"
            )

            ids = subset["sample_nr"].astype(int).tolist()

            if len(ids) != n_value:
                raise ValueError(
                    f"Seed {seed}, N={n_value}: expected "
                    f"{n_value} rows, found {len(ids)}."
                )

            if len(set(ids)) != n_value:
                raise ValueError(
                    f"Seed {seed}, N={n_value}: duplicate IDs."
                )

            current_ids = set(ids)

            if previous_ids and not previous_ids.issubset(
                current_ids
            ):
                raise ValueError(
                    f"Seed {seed}, N={n_value}: commissioning "
                    "sets are not nested."
                )

            for fixed_name, fixed_ids in fixed_sets.items():
                overlap = current_ids & fixed_ids

                if overlap:
                    raise RuntimeError(
                        f"Seed {seed}, N={n_value}: leakage with "
                        f"{fixed_name}: {sorted(overlap)[:20]}"
                    )

            previous_ids = current_ids


def get_commissioning_ids(
    commissioning: pd.DataFrame,
    *,
    seed: int,
    n_value: int,
) -> np.ndarray:
    subset = commissioning[
        commissioning["seed"].eq(seed)
        & commissioning["commissioning_n"].eq(
            n_value
        )
    ].sort_values(
        "selection_rank"
    )

    ids = subset["sample_nr"].to_numpy(
        dtype=np.int64
    )

    if len(ids) != n_value:
        raise ValueError(
            f"Seed {seed}, N={n_value}: expected {n_value} "
            f"commissioning IDs, found {len(ids)}."
        )

    return ids


def subset_batch(
    cache: FeatureBatch,
    ids: Iterable[int],
) -> FeatureBatch:
    return cache.select_episode_ids(
        [int(value) for value in ids],
        preserve_requested_order=True,
        require_all=True,
    )


def split_conformal_threshold(
    calibration_scores: np.ndarray,
    alpha: float,
) -> tuple[float, int]:
    scores = np.asarray(
        calibration_scores,
        dtype=np.float64,
    )

    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError(
            "Calibration scores must be a non-empty 1D array."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Calibration scores contain NaN or Inf."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    rank = int(
        np.ceil(
            (len(scores) + 1)
            * (1.0 - alpha)
        )
    )

    rank = min(
        rank,
        len(scores),
    )

    ordered = np.sort(scores)

    return (
        float(
            ordered[rank - 1]
        ),
        rank,
    )


def kth_neighbor_scores(
    model: NearestNeighbors,
    features: np.ndarray,
    effective_k: int,
) -> np.ndarray:
    """
    Return distance to the effective_k-th nearest training neighbor.

    For external calibration/evaluation samples, there is no self-neighbor,
    so requesting exactly effective_k neighbors is correct.
    """
    distances, _ = model.kneighbors(
        features,
        n_neighbors=effective_k,
        return_distance=True,
    )

    scores = distances[
        :,
        effective_k - 1,
    ]

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise RuntimeError(
            f"k-NN produced score shape {scores.shape}."
        )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "k-NN produced NaN or Inf scores."
        )

    return scores


def run_one(
    cache: FeatureBatch,
    commissioning_ids: np.ndarray,
    calibration_ids: np.ndarray,
    healthy_eval_ids: np.ndarray,
    anomaly_eval_ids: np.ndarray,
    anomaly_eval_table: pd.DataFrame,
    *,
    commissioning_size: int,
    seed: int,
    requested_k: int,
    false_alert_budget: float,
    recall_target: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    commissioning = subset_batch(
        cache,
        commissioning_ids,
    )

    calibration = subset_batch(
        cache,
        calibration_ids,
    )

    healthy_eval = subset_batch(
        cache,
        healthy_eval_ids,
    )

    anomaly_eval = subset_batch(
        cache,
        anomaly_eval_ids,
    )

    if commissioning.anomaly_labels.any():
        raise RuntimeError(
            "Commissioning subset contains anomalies."
        )

    if calibration.anomaly_labels.any():
        raise RuntimeError(
            "Calibration subset contains anomalies."
        )

    if healthy_eval.anomaly_labels.any():
        raise RuntimeError(
            "Healthy evaluation subset contains anomalies."
        )

    if not anomaly_eval.anomaly_labels.all():
        raise RuntimeError(
            "Anomaly evaluation subset contains healthy rows."
        )

    preprocessor = FeaturePreprocessor(
        variance_threshold=1e-12
    )

    train_features = preprocessor.fit_transform(
        commissioning.features
    )

    calibration_features = preprocessor.transform(
        calibration.features
    )

    healthy_features = preprocessor.transform(
        healthy_eval.features
    )

    anomaly_features = preprocessor.transform(
        anomaly_eval.features
    )

    effective_k = min(
        int(requested_k),
        len(train_features),
    )

    if effective_k <= 0:
        raise RuntimeError(
            "effective_k must be positive."
        )

    model = NearestNeighbors(
        n_neighbors=effective_k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    model.fit(
        train_features
    )

    calibration_scores = kth_neighbor_scores(
        model,
        calibration_features,
        effective_k,
    )

    threshold, conformal_rank = (
        split_conformal_threshold(
            calibration_scores,
            false_alert_budget,
        )
    )

    healthy_scores = kth_neighbor_scores(
        model,
        healthy_features,
        effective_k,
    )

    anomaly_scores = kth_neighbor_scores(
        model,
        anomaly_features,
        effective_k,
    )

    healthy_predictions = (
        healthy_scores > threshold
    )

    anomaly_predictions = (
        anomaly_scores > threshold
    )

    false_positive_rate = float(
        healthy_predictions.mean()
    )

    recall = float(
        anomaly_predictions.mean()
    )

    labels = np.concatenate(
        (
            np.zeros(
                len(healthy_scores),
                dtype=np.int64,
            ),
            np.ones(
                len(anomaly_scores),
                dtype=np.int64,
            ),
        )
    )

    scores = np.concatenate(
        (
            healthy_scores,
            anomaly_scores,
        )
    )

    auroc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": DETECTOR_NAME,
        "commissioning_size": int(
            commissioning_size
        ),
        "seed": int(seed),
        "commissioning_count": int(
            len(commissioning_ids)
        ),
        "calibration_count": int(
            len(calibration_ids)
        ),
        "healthy_eval_count": int(
            len(healthy_eval_ids)
        ),
        "anomaly_eval_count": int(
            len(anomaly_eval_ids)
        ),
        "retained_features": int(
            preprocessor.output_feature_count_
        ),
        "requested_k": int(
            requested_k
        ),
        "effective_k": int(
            effective_k
        ),
        "conformal_rank": int(
            conformal_rank
        ),
        "threshold": float(
            threshold
        ),
        "false_positive_rate": (
            false_positive_rate
        ),
        "recall": recall,
        "auroc": auroc,
        "success": bool(
            recall >= recall_target
            and false_positive_rate
            <= false_alert_budget
        ),
    }

    prediction_by_id = {
        int(episode_id): bool(prediction)
        for episode_id, prediction
        in zip(
            anomaly_eval.episode_ids,
            anomaly_predictions,
        )
    }

    class_rows: list[dict[str, Any]] = []

    for (
        label,
        label_name,
    ), group in anomaly_eval_table.groupby(
        [
            "label",
            "label_name",
        ],
        sort=True,
    ):
        class_ids = (
            group["sample_nr"]
            .astype(int)
            .tolist()
        )

        predictions = np.asarray(
            [
                prediction_by_id[
                    episode_id
                ]
                for episode_id in class_ids
            ],
            dtype=bool,
        )

        class_rows.append(
            {
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "detector": DETECTOR_NAME,
                "commissioning_size": int(
                    commissioning_size
                ),
                "seed": int(seed),
                "label": int(label),
                "label_name": str(
                    label_name
                ),
                "execution_count": int(
                    len(class_ids)
                ),
                "recall": float(
                    predictions.mean()
                ),
            }
        )

    return result, class_rows


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    confidence: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "Bootstrap values must be a non-empty 1D array."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap values contain NaN or Inf."
        )

    rng = np.random.default_rng(seed)

    sampled = rng.choice(
        values,
        size=(
            bootstrap_samples,
            len(values),
        ),
        replace=True,
    )

    means = sampled.mean(axis=1)
    alpha = 1.0 - confidence

    return (
        float(
            np.quantile(
                means,
                alpha / 2.0,
            )
        ),
        float(
            np.quantile(
                means,
                1.0 - alpha / 2.0,
            )
        ),
    )


def build_summary(
    results: pd.DataFrame,
    *,
    false_alert_budget: float,
    recall_target: float,
    confidence: float,
    bootstrap_samples: int,
    global_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for n_value, group in results.groupby(
        "commissioning_size",
        sort=True,
    ):
        recall_values = group["recall"].to_numpy(
            dtype=np.float64
        )

        fpr_values = group[
            "false_positive_rate"
        ].to_numpy(
            dtype=np.float64
        )

        auroc_values = group["auroc"].to_numpy(
            dtype=np.float64
        )

        recall_lower, recall_upper = (
            bootstrap_mean_interval(
                recall_values,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=(
                    global_seed
                    + int(n_value) * 100
                ),
            )
        )

        fpr_lower, fpr_upper = (
            bootstrap_mean_interval(
                fpr_values,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=(
                    global_seed
                    + int(n_value) * 100
                    + 1
                ),
            )
        )

        auroc_lower, auroc_upper = (
            bootstrap_mean_interval(
                auroc_values,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=(
                    global_seed
                    + int(n_value) * 100
                    + 2
                ),
            )
        )

        rows.append(
            {
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "detector": DETECTOR_NAME,
                "commissioning_size": int(
                    n_value
                ),
                "seed_count": int(
                    len(group)
                ),
                "mean_recall": float(
                    recall_values.mean()
                ),
                "recall_ci_lower": (
                    recall_lower
                ),
                "recall_ci_upper": (
                    recall_upper
                ),
                "mean_false_positive_rate": float(
                    fpr_values.mean()
                ),
                "fpr_ci_lower": fpr_lower,
                "fpr_ci_upper": fpr_upper,
                "mean_auroc": float(
                    auroc_values.mean()
                ),
                "auroc_ci_lower": (
                    auroc_lower
                ),
                "auroc_ci_upper": (
                    auroc_upper
                ),
                "success_rate": float(
                    group["success"]
                    .astype(bool)
                    .mean()
                ),
                "mean_retained_features": float(
                    group[
                        "retained_features"
                    ].mean()
                ),
                "mean_effective_k": float(
                    group[
                        "effective_k"
                    ].mean()
                ),
                "meets_joint_ci_criterion": bool(
                    recall_lower
                    >= recall_target
                    and fpr_upper
                    <= false_alert_budget
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "commissioning_size"
    ).reset_index(drop=True)


def build_per_class_summary(
    per_class: pd.DataFrame,
    *,
    confidence: float,
    bootstrap_samples: int,
    global_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (
        n_value,
        label,
        label_name,
    ), group in per_class.groupby(
        [
            "commissioning_size",
            "label",
            "label_name",
        ],
        sort=True,
    ):
        values = group["recall"].to_numpy(
            dtype=np.float64
        )

        lower, upper = bootstrap_mean_interval(
            values,
            confidence=confidence,
            bootstrap_samples=bootstrap_samples,
            seed=(
                global_seed
                + int(n_value) * 100
                + int(label) * 10
                + 5
            ),
        )

        rows.append(
            {
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "detector": DETECTOR_NAME,
                "commissioning_size": int(
                    n_value
                ),
                "label": int(label),
                "label_name": str(
                    label_name
                ),
                "execution_count_per_seed": int(
                    group[
                        "execution_count"
                    ].iloc[0]
                ),
                "seed_count": int(
                    len(group)
                ),
                "mean_recall": float(
                    values.mean()
                ),
                "recall_ci_lower": lower,
                "recall_ci_upper": upper,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "commissioning_size",
            "label",
        ]
    ).reset_index(drop=True)


def estimate_n_star(
    summary: pd.DataFrame,
) -> dict[str, Any]:
    eligible = summary[
        summary[
            "meets_joint_ci_criterion"
        ].astype(bool)
    ].sort_values(
        "commissioning_size"
    )

    maximum_n = int(
        summary[
            "commissioning_size"
        ].max()
    )

    if eligible.empty:
        return {
            "status": "censored",
            "n_star": None,
            "display": f"Censored (>{maximum_n})",
            "maximum_tested_n": maximum_n,
        }

    n_star = int(
        eligible.iloc[0][
            "commissioning_size"
        ]
    )

    return {
        "status": "observed",
        "n_star": n_star,
        "display": str(n_star),
        "maximum_tested_n": maximum_n,
    }


def load_checkpoint_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if overwrite or not path.exists():
        return []

    frame = pd.read_csv(path)

    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Existing checkpoint {path} is incompatible. "
            f"Missing columns: {missing}"
        )

    versions = set(
        frame[
            "protocol_version"
        ].dropna().astype(str).unique()
    )

    if versions and versions != {
        PROTOCOL_VERSION
    }:
        raise ValueError(
            f"Existing checkpoint uses protocol versions "
            f"{sorted(versions)}, expected {PROTOCOL_VERSION!r}."
        )

    return frame[
        list(columns)
    ].to_dict(
        orient="records"
    )


def completed_keys(
    rows: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    return {
        (
            int(row["commissioning_size"]),
            int(row["seed"]),
        )
        for row in rows
    }


def atomic_write_csv(
    rows: list[dict[str, Any]],
    path: Path,
    columns: tuple[str, ...],
    sort_columns: list[str],
) -> None:
    frame = pd.DataFrame(rows)

    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame[
        list(columns)
    ].sort_values(
        sort_columns
    ).reset_index(drop=True)

    temporary = path.with_suffix(
        ".tmp.csv"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Euclidean conformal k-NN experiments on AURSAD."
        )
    )

    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
    )

    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--grid",
        type=parse_int_csv,
        default=DEFAULT_GRID,
    )

    parser.add_argument(
        "--seeds",
        type=parse_int_csv,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
    )

    parser.add_argument(
        "--false-alert-budget",
        type=float,
        default=DEFAULT_FALSE_ALERT_BUDGET,
    )

    parser.add_argument(
        "--recall-target",
        type=float,
        default=DEFAULT_RECALL_TARGET,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--global-seed",
        type=int,
        default=DEFAULT_GLOBAL_SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_path = (
        args.cache_path
        .expanduser()
        .resolve()
    )

    protocol_dir = (
        args.protocol_dir
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    grid = validate_grid(
        args.grid
    )

    seeds = validate_seeds(
        args.seeds
    )

    if args.k <= 0:
        raise ValueError(
            "--k must be positive."
        )

    if not 0.0 < args.false_alert_budget < 1.0:
        raise ValueError(
            "--false-alert-budget must be between 0 and 1."
        )

    if not 0.0 < args.recall_target <= 1.0:
        raise ValueError(
            "--recall-target must be in (0, 1]."
        )

    if not 0.0 < args.confidence < 1.0:
        raise ValueError(
            "--confidence must be between 0 and 1."
        )

    if args.bootstrap_samples <= 0:
        raise ValueError(
            "--bootstrap-samples must be positive."
        )

    require_file(
        cache_path,
        "AURSAD feature cache",
    )

    if not protocol_dir.exists():
        raise FileNotFoundError(
            f"Protocol directory does not exist: {protocol_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_results_path = (
        output_dir
        / "euclidean_knn_seed_results.csv"
    )

    per_class_seed_path = (
        output_dir
        / "euclidean_knn_per_class_seed_results.csv"
    )

    summary_path = (
        output_dir
        / "euclidean_knn_summary.csv"
    )

    per_class_summary_path = (
        output_dir
        / "euclidean_knn_per_class_recall.csv"
    )

    n_star_path = (
        output_dir
        / "euclidean_knn_n_star.json"
    )

    manifest_path = (
        output_dir
        / "euclidean_knn_run_manifest.json"
    )

    print("=" * 76)
    print("AURSAD EUCLIDEAN CONFORMAL k-NN")
    print("=" * 76)
    print(f"Cache:       {cache_path}")
    print(f"Protocol:    {protocol_dir}")
    print(f"Output:      {output_dir}")
    print(f"Grid:        {list(grid)}")
    print(f"Seeds:       {list(seeds)}")
    print(f"Requested k: {args.k}")
    print(
        f"Criteria:    recall >= {args.recall_target:.3f}, "
        f"FPR <= {args.false_alert_budget:.3f}"
    )

    started = time.perf_counter()

    cache = load_feature_batch(
        cache_path
    )

    if cache.features.shape[1] != 288:
        raise ValueError(
            f"Expected 288 cached features, found "
            f"{cache.features.shape[1]}."
        )

    protocol = load_protocol_tables(
        protocol_dir
    )

    validate_protocol(
        protocol,
        grid=grid,
        seeds=seeds,
    )

    calibration_ids = unique_ids(
        protocol["calibration"]
    )

    healthy_eval_ids = unique_ids(
        protocol["healthy_eval"]
    )

    anomaly_eval_ids = unique_ids(
        protocol["anomaly_eval"]
    )

    required_cache_ids = (
        set(calibration_ids.tolist())
        | set(healthy_eval_ids.tolist())
        | set(anomaly_eval_ids.tolist())
        | set(
            protocol["commissioning"][
                "sample_nr"
            ].astype(int).tolist()
        )
    )

    available_cache_ids = set(
        cache.episode_ids.astype(int).tolist()
    )

    missing_ids = sorted(
        required_cache_ids
        - available_cache_ids
    )

    if missing_ids:
        raise ValueError(
            "Feature cache is missing protocol IDs: "
            f"{missing_ids[:20]}"
        )

    nominal_rank = int(
        min(
            np.ceil(
                (len(calibration_ids) + 1)
                * (
                    1.0
                    - args.false_alert_budget
                )
            ),
            len(calibration_ids),
        )
    )

    print("\nFixed partitions:")
    print(
        f"  Calibration:      {len(calibration_ids):,}"
    )
    print(
        f"  Healthy eval:     {len(healthy_eval_ids):,}"
    )
    print(
        f"  Anomaly eval:     {len(anomaly_eval_ids):,}"
    )
    print(
        f"  Conformal rank:   {nominal_rank}/"
        f"{len(calibration_ids)}"
    )

    result_rows = load_checkpoint_rows(
        seed_results_path,
        SEED_RESULT_COLUMNS,
        overwrite=bool(
            args.overwrite
        ),
    )

    class_rows = load_checkpoint_rows(
        per_class_seed_path,
        PER_CLASS_COLUMNS,
        overwrite=bool(
            args.overwrite
        ),
    )

    done = completed_keys(
        result_rows
    )

    total_runs = len(grid) * len(seeds)
    run_number = len(done)

    print(
        f"\nCompleted checkpoint runs: "
        f"{len(done)}/{total_runs}"
    )

    for n_value in grid:
        for seed in seeds:
            key = (
                int(n_value),
                int(seed),
            )

            if key in done:
                continue

            run_number += 1

            print(
                f"Processing N={n_value} seed={seed} "
                f"({run_number}/{total_runs})..."
            )

            commissioning_ids = (
                get_commissioning_ids(
                    protocol[
                        "commissioning"
                    ],
                    seed=seed,
                    n_value=n_value,
                )
            )

            result, run_class_rows = run_one(
                cache=cache,
                commissioning_ids=(
                    commissioning_ids
                ),
                calibration_ids=(
                    calibration_ids
                ),
                healthy_eval_ids=(
                    healthy_eval_ids
                ),
                anomaly_eval_ids=(
                    anomaly_eval_ids
                ),
                anomaly_eval_table=(
                    protocol[
                        "anomaly_eval"
                    ]
                ),
                commissioning_size=n_value,
                seed=seed,
                requested_k=int(
                    args.k
                ),
                false_alert_budget=(
                    args.false_alert_budget
                ),
                recall_target=(
                    args.recall_target
                ),
            )

            result_rows.append(
                result
            )

            class_rows.extend(
                run_class_rows
            )

            atomic_write_csv(
                result_rows,
                seed_results_path,
                SEED_RESULT_COLUMNS,
                [
                    "commissioning_size",
                    "seed",
                ],
            )

            atomic_write_csv(
                class_rows,
                per_class_seed_path,
                PER_CLASS_COLUMNS,
                [
                    "commissioning_size",
                    "seed",
                    "label",
                ],
            )

            print(
                "  "
                f"k={result['effective_k']} "
                f"recall={result['recall']:.4f} "
                f"FPR={result['false_positive_rate']:.4f} "
                f"AUROC={result['auroc']:.4f} "
                f"success={result['success']} "
                f"features={result['retained_features']}"
            )

    results = pd.read_csv(
        seed_results_path
    )

    expected_keys = {
        (
            int(n_value),
            int(seed),
        )
        for n_value in grid
        for seed in seeds
    }

    observed_keys = {
        (
            int(row.commissioning_size),
            int(row.seed),
        )
        for row in results.itertuples(
            index=False
        )
    }

    missing_runs = sorted(
        expected_keys - observed_keys
    )

    if missing_runs:
        raise RuntimeError(
            f"Experiment finished with missing runs: "
            f"{missing_runs[:20]}"
        )

    results = results[
        results["commissioning_size"].isin(
            grid
        )
        & results["seed"].isin(
            seeds
        )
    ].copy()

    per_class_seed = pd.read_csv(
        per_class_seed_path
    )

    per_class_seed = per_class_seed[
        per_class_seed[
            "commissioning_size"
        ].isin(
            grid
        )
        & per_class_seed["seed"].isin(
            seeds
        )
    ].copy()

    summary = build_summary(
        results,
        false_alert_budget=(
            args.false_alert_budget
        ),
        recall_target=(
            args.recall_target
        ),
        confidence=float(
            args.confidence
        ),
        bootstrap_samples=int(
            args.bootstrap_samples
        ),
        global_seed=int(
            args.global_seed
        ),
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    per_class_summary = (
        build_per_class_summary(
            per_class_seed,
            confidence=float(
                args.confidence
            ),
            bootstrap_samples=int(
                args.bootstrap_samples
            ),
            global_seed=int(
                args.global_seed
            ),
        )
    )

    per_class_summary.to_csv(
        per_class_summary_path,
        index=False,
    )

    n_star = estimate_n_star(
        summary
    )

    n_star_payload = {
        "protocol_version": (
            PROTOCOL_VERSION
        ),
        "detector": DETECTOR_NAME,
        "criterion": {
            "recall_target": float(
                args.recall_target
            ),
            "false_alert_budget": float(
                args.false_alert_budget
            ),
            "confidence": float(
                args.confidence
            ),
            "rule": (
                "smallest N with recall CI lower bound >= target "
                "and FPR CI upper bound <= budget"
            ),
        },
        "estimate": n_star,
    }

    n_star_path.write_text(
        json.dumps(
            json_safe(
                n_star_payload
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    protocol_input_paths = {
        "commissioning": protocol_dir / "commissioning_ids.csv",
        "calibration": protocol_dir / "calibration_ids.csv",
        "healthy_eval": protocol_dir / "healthy_eval_ids.csv",
        "anomaly_eval": protocol_dir / "anomaly_eval_ids.csv",
    }

    manifest = {
        "run_version": PROTOCOL_VERSION,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "command": " ".join(
            sys.argv
        ),
        "dataset": "AURSAD",
        "detector": {
            "name": DETECTOR_NAME,
            "representation": (
                "288-dimensional statistical execution features"
            ),
            "metric": "euclidean",
            "requested_k": int(
                args.k
            ),
            "small_n_rule": (
                "effective_k=min(requested_k, commissioning_size)"
            ),
            "training_data": (
                "target commissioning normal executions only"
            ),
            "calibration": (
                "fixed healthy split-conformal calibration set"
            ),
            "score": (
                "distance to effective_k-th nearest commissioning "
                "neighbor"
            ),
            "prediction_rule": (
                "anomaly when score > conformal threshold"
            ),
        },
        "protocol": {
            "commissioning_grid": list(
                grid
            ),
            "seeds": list(
                seeds
            ),
            "false_alert_budget": float(
                args.false_alert_budget
            ),
            "recall_target": float(
                args.recall_target
            ),
            "confidence": float(
                args.confidence
            ),
            "bootstrap_samples": int(
                args.bootstrap_samples
            ),
            "global_seed": int(
                args.global_seed
            ),
            "calibration_count": int(
                len(calibration_ids)
            ),
            "healthy_eval_count": int(
                len(healthy_eval_ids)
            ),
            "anomaly_eval_count": int(
                len(anomaly_eval_ids)
            ),
            "conformal_rank": int(
                nominal_rank
            ),
        },
        "input": {
            "cache_path": str(
                cache_path
            ),
            "cache_sha256": sha256_file(
                cache_path
            ),
            "cache_shape": [
                int(
                    cache.features.shape[0]
                ),
                int(
                    cache.features.shape[1]
                ),
            ],
        },
        "result": {
            "run_count": int(
                len(results)
            ),
            "n_star": n_star,
        },
        "outputs": {
            "seed_results": str(
                seed_results_path
            ),
            "summary": str(
                summary_path
            ),
            "per_class_seed_results": str(
                per_class_seed_path
            ),
            "per_class_summary": str(
                per_class_summary_path
            ),
            "n_star": str(
                n_star_path
            ),
        },
        "timing_seconds": {
            "total": float(
                elapsed
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": (
                platform.platform()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "validation": {
            "protocol_disjointness_checked": True,
            "nested_commissioning_checked": True,
            "cache_coverage_checked": True,
            "normal_only_training_checked": True,
            "normal_only_calibration_checked": True,
            "normal_only_healthy_eval_checked": True,
            "anomaly_only_fault_eval_checked": True,
            "training_only_preprocessing": True,
            "finite_sample_conformal_calibration": True,
            "all_requested_runs_completed": True,
        },
        "limitations": [
            (
                "This baseline uses Euclidean distance on fixed-length "
                "statistical features, not raw time-series trajectories."
            ),
            (
                "The effective k is clipped when commissioning N is "
                "smaller than the requested k."
            ),
            (
                "Damaged-thread label 4 has only three evaluation "
                "executions."
            ),
        ],
    }
    manifest.update(
        reproducibility_metadata(
            repo_root=PROJECT_ROOT,
            input_paths={
                "feature_cache": cache_path,
                **{
                    f"protocol_{name}": path
                    for name, path in protocol_input_paths.items()
                },
            },
            artifact_paths={
                "seed_results": seed_results_path,
                "summary": summary_path,
                "per_class_seed_results": per_class_seed_path,
                "per_class_summary": per_class_summary_path,
                "n_star": n_star_path,
            },
        )
    )

    manifest_path.write_text(
        json.dumps(
            json_safe(
                manifest
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("EUCLIDEAN CONFORMAL k-NN COMPLETE")
    print("=" * 76)

    print(
        summary[
            [
                "commissioning_size",
                "mean_recall",
                "recall_ci_lower",
                "recall_ci_upper",
                "mean_false_positive_rate",
                "fpr_ci_lower",
                "fpr_ci_upper",
                "mean_auroc",
                "success_rate",
                "mean_effective_k",
                "meets_joint_ci_criterion",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nEstimated N*: {n_star['display']}"
    )

    print("\nArtifacts:")
    print(f"  {seed_results_path}")
    print(f"  {summary_path}")
    print(f"  {per_class_seed_path}")
    print(f"  {per_class_summary_path}")
    print(f"  {n_star_path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
