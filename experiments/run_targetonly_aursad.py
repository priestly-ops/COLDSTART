#!/usr/bin/env python3
"""
experiments/run_targetonly_aursad.py

Run the TargetOnly commissioning experiment on the frozen AURSAD protocol.

TargetOnly fits a Ledoit-Wolf Gaussian exclusively on healthy target
commissioning features. The detector is calibrated using the fixed healthy
calibration partition and evaluated on the fixed healthy and anomaly
evaluation partitions.

Protocol
--------
Commissioning sizes:
    10, 25, 50, 100, 250, 500

Seeds:
    0 through 19

False-alert budget:
    0.01

Recall target:
    0.90

A run is successful when:
    recall >= 0.90 and FPR <= 0.01

The reported commissioning estimator N* is the smallest N where:
    lower 95% bootstrap CI for mean recall >= 0.90
    upper 95% bootstrap CI for mean FPR <= 0.01

Inputs
------
outputs/aursad/feature_cache/aursad_features.npz
reports/aursad/protocol/commissioning_ids.csv
reports/aursad/protocol/calibration_ids.csv
reports/aursad/protocol/healthy_eval_ids.csv
reports/aursad/protocol/anomaly_eval_ids.csv

Outputs
-------
outputs/aursad/targetonly/
├── targetonly_seed_results.csv
├── targetonly_summary.csv
├── targetonly_per_class_recall.csv
├── targetonly_n_star.json
└── targetonly_run_manifest.json

Example
-------
From the repository root:

    .\\.venv\\Scripts\\python.exe experiments\\run_targetonly_aursad.py

To restart from scratch:

    .\\.venv\\Scripts\\python.exe experiments\\run_targetonly_aursad.py ^
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import TargetOnlyDetector
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
    / "targetonly"
)

DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))

DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_GLOBAL_SEED = 42

PROTOCOL_VERSION = "aursad-targetonly-split-conformal-v1"
DETECTOR_NAME = "TargetOnly"

CHECKPOINT_COLUMNS = (
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "commissioning_count",
    "calibration_count",
    "healthy_eval_count",
    "anomaly_eval_count",
    "retained_features",
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
    """Compute SHA-256 without loading a whole file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy and Pandas values into JSON-safe Python objects."""
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


def require_file(path: Path, description: str) -> None:
    """Raise a clear error when an input artifact is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a regular file: {path}"
        )


def parse_int_csv(value: str) -> tuple[int, ...]:
    """Parse comma-separated unique integer values."""
    parsed: list[int] = []

    for raw in value.split(","):
        item = raw.strip()

        if item:
            parsed.append(int(item))

    if not parsed:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "Duplicate integer values are not allowed."
        )

    return tuple(parsed)


def validate_grid(values: Iterable[int]) -> tuple[int, ...]:
    """Validate an increasing commissioning grid."""
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


def validate_seeds(values: Iterable[int]) -> tuple[int, ...]:
    """Validate unique non-negative seed IDs."""
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
    """Load one frozen protocol CSV with strict validation."""
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
    """Load all protocol membership tables used in evaluation."""
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

    required_commissioning = {
        "seed",
        "commissioning_n",
        "selection_rank",
    }

    missing = sorted(
        required_commissioning
        - set(commissioning.columns)
    )

    if missing:
        raise ValueError(
            "commissioning_ids.csv is missing columns: "
            f"{missing}"
        )

    commissioning["seed"] = pd.to_numeric(
        commissioning["seed"],
        errors="raise",
    ).astype(np.int64)

    commissioning["commissioning_n"] = pd.to_numeric(
        commissioning["commissioning_n"],
        errors="raise",
    ).astype(np.int64)

    commissioning["selection_rank"] = pd.to_numeric(
        commissioning["selection_rank"],
        errors="raise",
    ).astype(np.int64)

    return tables


def unique_ids(frame: pd.DataFrame) -> np.ndarray:
    """Return sample IDs while rejecting duplicate fixed memberships."""
    values = frame["sample_nr"].to_numpy(
        dtype=np.int64
    )

    if len(set(values.tolist())) != len(values):
        duplicates = (
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
            f"Duplicate sample_nr values found: {duplicates[:20]}"
        )

    return values


def validate_protocol(
    tables: dict[str, pd.DataFrame],
    grid: tuple[int, ...],
    seeds: tuple[int, ...],
) -> None:
    """Verify labels, fixed partitions, nestedness, and zero leakage."""
    calibration = tables["calibration"]
    healthy_eval = tables["healthy_eval"]
    anomaly_eval = tables["anomaly_eval"]
    commissioning = tables["commissioning"]

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

    if not commissioning["label"].eq(0).all():
        raise ValueError(
            "Commissioning contains non-normal executions."
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

    fixed_names = list(fixed_sets)

    for index, left_name in enumerate(fixed_names):
        for right_name in fixed_names[index + 1:]:
            overlap = (
                fixed_sets[left_name]
                & fixed_sets[right_name]
            )

            if overlap:
                raise RuntimeError(
                    f"Protocol leakage between {left_name} and "
                    f"{right_name}: {sorted(overlap)[:20]}"
                )

    observed_seeds = set(
        commissioning["seed"].astype(int).unique()
    )

    missing_seeds = sorted(
        set(seeds) - observed_seeds
    )

    if missing_seeds:
        raise ValueError(
            f"Commissioning CSV is missing seeds: {missing_seeds}"
        )

    for seed in seeds:
        seed_rows = commissioning[
            commissioning["seed"].eq(seed)
        ]

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
    seed: int,
    n_value: int,
) -> np.ndarray:
    """Return one seed/N membership in frozen selection order."""
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
    episode_ids: Iterable[int],
) -> FeatureBatch:
    """Select cache rows in requested protocol order."""
    return cache.select_episode_ids(
        list(
            int(value)
            for value in episode_ids
        ),
        preserve_requested_order=True,
        require_all=True,
    )


def conformal_rank(
    sample_count: int,
    alpha: float,
) -> int:
    """Return the one-indexed finite-sample conformal rank."""
    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    rank = int(
        np.ceil(
            (sample_count + 1)
            * (1.0 - alpha)
        )
    )

    return min(
        rank,
        sample_count,
    )


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
    false_alert_budget: float,
    recall_target: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Fit, calibrate, and evaluate one TargetOnly seed/N run."""
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
            "Commissioning cache subset contains anomalies."
        )

    if calibration.anomaly_labels.any():
        raise RuntimeError(
            "Calibration cache subset contains anomalies."
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

    target_train = preprocessor.fit_transform(
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

    detector = TargetOnlyDetector(
        false_alert_budget=false_alert_budget
    )

    # TargetOnly ignores source_features. Passing the target matrix preserves
    # the common BaseDetector fit signature without introducing source data.
    detector.fit(
        source_features=target_train,
        target_features=target_train,
    )

    detector.calibrate(
        calibration_features
    )

    if detector.threshold_ is None:
        raise RuntimeError(
            "TargetOnly threshold was not produced."
        )

    healthy_scores = detector.score_samples(
        healthy_features
    )

    anomaly_scores = detector.score_samples(
        anomaly_features
    )

    healthy_predictions = (
        healthy_scores > detector.threshold_
    )

    anomaly_predictions = (
        anomaly_scores > detector.threshold_
    )

    false_positive_rate = float(
        np.mean(healthy_predictions)
    )

    recall = float(
        np.mean(anomaly_predictions)
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

    all_scores = np.concatenate(
        (
            healthy_scores,
            anomaly_scores,
        )
    )

    auroc = float(
        roc_auc_score(
            labels,
            all_scores,
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
        "threshold": float(
            detector.threshold_
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

        class_predictions = np.asarray(
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
                    np.mean(
                        class_predictions
                    )
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
    """Percentile-bootstrap confidence interval for the seed mean."""
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "Bootstrap input must be a non-empty 1D array."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap input contains NaN or Inf."
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    if bootstrap_samples <= 0:
        raise ValueError(
            "bootstrap_samples must be positive."
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

    means = sampled.mean(
        axis=1
    )

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


def group_bootstrap_seed(
    commissioning_size: int,
    metric_offset: int,
    global_seed: int,
) -> int:
    """Create a deterministic bootstrap seed for each N/metric."""
    return int(
        global_seed
        + commissioning_size * 100
        + metric_offset
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
    """Aggregate seed-level metrics and confidence intervals."""
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
                seed=group_bootstrap_seed(
                    int(n_value),
                    metric_offset=0,
                    global_seed=global_seed,
                ),
            )
        )

        fpr_lower, fpr_upper = (
            bootstrap_mean_interval(
                fpr_values,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=group_bootstrap_seed(
                    int(n_value),
                    metric_offset=1,
                    global_seed=global_seed,
                ),
            )
        )

        auroc_lower, auroc_upper = (
            bootstrap_mean_interval(
                auroc_values,
                confidence=confidence,
                bootstrap_samples=bootstrap_samples,
                seed=group_bootstrap_seed(
                    int(n_value),
                    metric_offset=2,
                    global_seed=global_seed,
                ),
            )
        )

        success_rate = float(
            group["success"]
            .astype(bool)
            .mean()
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
                "fpr_ci_lower": (
                    fpr_lower
                ),
                "fpr_ci_upper": (
                    fpr_upper
                ),
                "mean_auroc": float(
                    auroc_values.mean()
                ),
                "auroc_ci_lower": (
                    auroc_lower
                ),
                "auroc_ci_upper": (
                    auroc_upper
                ),
                "success_rate": (
                    success_rate
                ),
                "mean_retained_features": float(
                    group[
                        "retained_features"
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
    """Aggregate fault-class recall across seeds."""
    rows: list[dict[str, Any]] = []

    grouped = per_class.groupby(
        [
            "commissioning_size",
            "label",
            "label_name",
        ],
        sort=True,
    )

    for (
        n_value,
        label,
        label_name,
    ), group in grouped:
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
    """Return the smallest commissioning N meeting the joint CI rule."""
    eligible = summary[
        summary[
            "meets_joint_ci_criterion"
        ].astype(bool)
    ].sort_values(
        "commissioning_size"
    )

    if eligible.empty:
        maximum_n = int(
            summary[
                "commissioning_size"
            ].max()
        )

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
        "maximum_tested_n": int(
            summary[
                "commissioning_size"
            ].max()
        ),
    }


def load_existing_rows(
    path: Path,
    columns: tuple[str, ...],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Load compatible checkpoint rows unless starting over."""
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

    protocol_values = set(
        frame["protocol_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    if protocol_values and protocol_values != {
        PROTOCOL_VERSION
    }:
        raise ValueError(
            f"Existing checkpoint {path} uses protocol versions "
            f"{sorted(protocol_values)}, expected "
            f"{PROTOCOL_VERSION!r}."
        )

    return frame[
        list(columns)
    ].to_dict(
        orient="records"
    )


def completed_keys(
    rows: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    """Return completed (N, seed) run keys."""
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
    """Write checkpoint CSV atomically."""
    frame = pd.DataFrame(
        rows
    )

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
            "Run TargetOnly commissioning experiments on AURSAD."
        )
    )

    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to aursad_features.npz.",
    )

    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR,
        help="Directory containing frozen protocol CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for TargetOnly results.",
    )

    parser.add_argument(
        "--grid",
        type=parse_int_csv,
        default=DEFAULT_GRID,
        help=(
            "Comma-separated commissioning sizes. "
            "Default: 10,25,50,100,250,500"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=parse_int_csv,
        default=DEFAULT_SEEDS,
        help=(
            "Comma-separated seed IDs. Default: 0 through 19"
        ),
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
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--global-seed",
        type=int,
        default=DEFAULT_GLOBAL_SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard compatible existing checkpoints.",
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
        / "targetonly_seed_results.csv"
    )

    per_class_seed_path = (
        output_dir
        / "targetonly_per_class_seed_results.csv"
    )

    summary_path = (
        output_dir
        / "targetonly_summary.csv"
    )

    per_class_summary_path = (
        output_dir
        / "targetonly_per_class_recall.csv"
    )

    n_star_path = (
        output_dir
        / "targetonly_n_star.json"
    )

    manifest_path = (
        output_dir
        / "targetonly_run_manifest.json"
    )

    print("=" * 76)
    print("AURSAD TARGETONLY COMMISSIONING EXPERIMENT")
    print("=" * 76)
    print(f"Cache:     {cache_path}")
    print(f"Protocol:  {protocol_dir}")
    print(f"Output:    {output_dir}")
    print(f"Grid:      {list(grid)}")
    print(f"Seeds:     {list(seeds)}")
    print(
        f"Criteria:  recall >= {args.recall_target:.3f}, "
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

    expected_cache_ids = (
        set(calibration_ids.tolist())
        | set(healthy_eval_ids.tolist())
        | set(anomaly_eval_ids.tolist())
        | set(
            protocol["commissioning"][
                "sample_nr"
            ].astype(int).tolist()
        )
    )

    actual_cache_ids = set(
        cache.episode_ids.astype(int).tolist()
    )

    missing_cache_ids = sorted(
        expected_cache_ids - actual_cache_ids
    )

    if missing_cache_ids:
        raise ValueError(
            "Feature cache is missing protocol IDs: "
            f"{missing_cache_ids[:20]}"
        )

    calibration_rank = conformal_rank(
        len(calibration_ids),
        args.false_alert_budget,
    )

    print(
        "\nFixed partitions:"
    )
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
        f"  Conformal rank:   {calibration_rank}/"
        f"{len(calibration_ids)}"
    )

    result_rows = load_existing_rows(
        seed_results_path,
        CHECKPOINT_COLUMNS,
        overwrite=bool(
            args.overwrite
        ),
    )

    class_rows = load_existing_rows(
        per_class_seed_path,
        PER_CLASS_COLUMNS,
        overwrite=bool(
            args.overwrite
        ),
    )

    done = completed_keys(
        result_rows
    )

    total_runs = (
        len(grid)
        * len(seeds)
    )

    completed_before = len(done)

    print(
        f"\nCompleted checkpoint runs: "
        f"{completed_before}/{total_runs}"
    )

    run_number = completed_before

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
                CHECKPOINT_COLUMNS,
                sort_columns=[
                    "commissioning_size",
                    "seed",
                ],
            )

            atomic_write_csv(
                class_rows,
                per_class_seed_path,
                PER_CLASS_COLUMNS,
                sort_columns=[
                    "commissioning_size",
                    "seed",
                    "label",
                ],
            )

            print(
                "  "
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
        name: protocol_dir
        / (
            "commissioning_ids.csv"
            if name == "commissioning"
            else f"{name}_ids.csv"
        )
        for name in (
            "commissioning",
            "calibration",
            "healthy_eval",
            "anomaly_eval",
        )
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
            "model": (
                "LedoitWolf Gaussian Mahalanobis"
            ),
            "training_data": (
                "target commissioning normal executions only"
            ),
            "calibration": (
                "fixed healthy split-conformal calibration set"
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
                calibration_rank
            ),
        },
        "inputs": {
            "feature_cache": {
                "path": str(
                    cache_path
                ),
                "sha256": sha256_file(
                    cache_path
                ),
                "shape": [
                    int(
                        cache.features.shape[0]
                    ),
                    int(
                        cache.features.shape[1]
                    ),
                ],
            },
            "protocol_files": {
                name: {
                    "path": str(
                        path
                    ),
                    "sha256": sha256_file(
                        path
                    ),
                }
                for name, path in protocol_input_paths.items()
            },
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
        "result": {
            "run_count": int(
                len(results)
            ),
            "n_star": n_star,
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
            "all_requested_runs_completed": True,
        },
        "limitations": [
            (
                "Damaged-thread label 4 has only three executions, "
                "so its per-class recall estimate is underpowered."
            ),
            (
                "TargetOnly uses no source-domain data and is the "
                "commissioning-from-scratch baseline."
            ),
            (
                "The global feature cache is unscaled; variance "
                "filtering and standardization are fitted independently "
                "within each seed/N training set."
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
    print("TARGETONLY COMPLETE")
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
