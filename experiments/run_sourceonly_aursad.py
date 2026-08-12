#!/usr/bin/env python3
"""
experiments/run_sourceonly_aursad.py

Leakage-safe SourceOnly evaluation on the frozen AURSAD target protocol.

IMPORTANT
---------
The frozen AURSAD protocol does not contain an independent source-training
partition. Its 1,420 healthy tightening executions are already assigned to:

- commissioning reservoir: 520
- calibration: 600
- healthy evaluation: 300

Therefore this script intentionally requires a separate source feature cache.
It will not reuse AURSAD commissioning, calibration, or evaluation data as
"source" data.

A valid source cache must:

1. contain healthy source-domain executions only, or be accompanied by
   --source-ids-path selecting healthy source executions;
2. use exactly the same feature names and ordering as the AURSAD target cache;
3. have episode IDs disjoint from target protocol IDs when both caches use the
   same episode-ID namespace;
4. be created without target calibration/evaluation leakage.

Typical invocation
------------------
python experiments/run_sourceonly_aursad.py ^
  --source-cache-path outputs/source/feature_cache/source_features.npz ^
  --source-ids-path reports/source/source_train_ids.csv

If every execution in the source cache is healthy source training data, omit
--source-ids-path.

Outputs
-------
outputs/aursad/sourceonly/
├── sourceonly_seed_results.csv
├── sourceonly_summary.csv
├── sourceonly_per_class_seed_results.csv
├── sourceonly_per_class_recall.csv
├── sourceonly_n_star.json
└── sourceonly_run_manifest.json

Although SourceOnly does not adapt with N, results are repeated over the frozen
AURSAD commissioning grid and seeds so its horizontal baseline can be compared
directly with TargetOnly, Pooled, and RACE tables.
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

from src.detectors import SourceOnlyDetector
from src.feature_extractor import (
    FeatureBatch,
    FeaturePreprocessor,
    load_feature_batch,
)

DEFAULT_TARGET_CACHE_PATH = (
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
    / "sourceonly"
)

DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))

DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_GLOBAL_SEED = 42

PROTOCOL_VERSION = "aursad-sourceonly-split-conformal-v1"
DETECTOR_NAME = "SourceOnly"

SEED_RESULT_COLUMNS = (
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "source_count",
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


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a regular file: {path}"
        )


def parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )

    if not parsed:
        raise argparse.ArgumentTypeError(
            "At least one integer is required."
        )

    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "Duplicate values are not allowed."
        )

    return parsed


def validate_grid(values: Iterable[int]) -> tuple[int, ...]:
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


def load_target_protocol(
    protocol_dir: Path,
) -> dict[str, pd.DataFrame]:
    return {
        "commissioning": load_protocol_csv(
            protocol_dir / "commissioning_ids.csv",
            "commissioning",
        ),
        "calibration": load_protocol_csv(
            protocol_dir / "calibration_ids.csv",
            "calibration",
        ),
        "healthy_eval": load_protocol_csv(
            protocol_dir / "healthy_eval_ids.csv",
            "healthy_eval",
        ),
        "anomaly_eval": load_protocol_csv(
            protocol_dir / "anomaly_eval_ids.csv",
            "anomaly_eval",
        ),
    }


def unique_ids(frame: pd.DataFrame) -> np.ndarray:
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


def validate_target_protocol(
    protocol: dict[str, pd.DataFrame],
) -> None:
    if not protocol["calibration"]["label"].eq(0).all():
        raise ValueError(
            "Calibration contains non-normal executions."
        )

    if not protocol["healthy_eval"]["label"].eq(0).all():
        raise ValueError(
            "Healthy evaluation contains non-normal executions."
        )

    if not protocol["anomaly_eval"]["label"].isin(
        [1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Anomaly evaluation contains unsupported labels."
        )

    if not protocol["commissioning"]["label"].eq(0).all():
        raise ValueError(
            "Commissioning contains non-normal executions."
        )

    fixed_sets = {
        "calibration": set(
            unique_ids(
                protocol["calibration"]
            ).tolist()
        ),
        "healthy_eval": set(
            unique_ids(
                protocol["healthy_eval"]
            ).tolist()
        ),
        "anomaly_eval": set(
            unique_ids(
                protocol["anomaly_eval"]
            ).tolist()
        ),
    }

    names = list(fixed_sets)

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = (
                fixed_sets[left]
                & fixed_sets[right]
            )

            if overlap:
                raise RuntimeError(
                    f"Target protocol leakage between {left} and "
                    f"{right}: {sorted(overlap)[:20]}"
                )


def load_source_ids(
    source_cache: FeatureBatch,
    source_ids_path: Path | None,
) -> np.ndarray:
    """
    Select source training IDs.

    If no CSV is provided, every cache execution must be healthy.
    """
    if source_ids_path is None:
        if source_cache.anomaly_labels.any():
            raise ValueError(
                "The source cache contains anomalous executions. "
                "Provide --source-ids-path selecting healthy source "
                "training executions only."
            )

        return source_cache.episode_ids.copy()

    require_file(
        source_ids_path,
        "Source ID CSV",
    )

    frame = pd.read_csv(source_ids_path)

    candidate_columns = (
        "episode_id",
        "sample_nr",
        "source_id",
    )

    id_column = next(
        (
            column
            for column in candidate_columns
            if column in frame.columns
        ),
        None,
    )

    if id_column is None:
        raise ValueError(
            "Source ID CSV must contain one of: "
            f"{candidate_columns}"
        )

    ids = pd.to_numeric(
        frame[id_column],
        errors="raise",
    ).astype(np.int64).to_numpy()

    if len(ids) == 0:
        raise ValueError(
            "Source ID CSV contains no executions."
        )

    if len(set(ids.tolist())) != len(ids):
        raise ValueError(
            "Source ID CSV contains duplicate IDs."
        )

    selected = source_cache.select_episode_ids(
        ids.tolist(),
        preserve_requested_order=True,
        require_all=True,
    )

    if selected.anomaly_labels.any():
        bad_ids = selected.episode_ids[
            selected.anomaly_labels
        ].astype(int).tolist()

        raise ValueError(
            "Selected source training data contains anomalies: "
            f"{bad_ids[:20]}"
        )

    return ids


def validate_feature_schema(
    source_cache: FeatureBatch,
    target_cache: FeatureBatch,
) -> None:
    """Require exact feature-space compatibility."""
    if source_cache.feature_names != target_cache.feature_names:
        source_names = set(
            source_cache.feature_names
        )

        target_names = set(
            target_cache.feature_names
        )

        missing_from_source = [
            name
            for name in target_cache.feature_names
            if name not in source_names
        ]

        extra_in_source = [
            name
            for name in source_cache.feature_names
            if name not in target_names
        ]

        raise ValueError(
            "Source and AURSAD target caches do not use the same "
            "feature schema and order. "
            f"Missing from source: {missing_from_source[:10]}; "
            f"extra in source: {extra_in_source[:10]}. "
            "SourceOnly is not meaningful until both domains share "
            "an identical feature representation."
        )

    if source_cache.signal_columns != target_cache.signal_columns:
        raise ValueError(
            "Source and target caches have different signal columns "
            "or signal ordering."
        )

    if source_cache.statistic_names != target_cache.statistic_names:
        raise ValueError(
            "Source and target caches use different statistic order."
        )


def subset_batch(
    cache: FeatureBatch,
    episode_ids: Iterable[int],
) -> FeatureBatch:
    return cache.select_episode_ids(
        [int(value) for value in episode_ids],
        preserve_requested_order=True,
        require_all=True,
    )


def conformal_rank(
    sample_count: int,
    alpha: float,
) -> int:
    rank = int(
        np.ceil(
            (sample_count + 1)
            * (1.0 - alpha)
        )
    )

    return min(rank, sample_count)


def run_one(
    source_cache: FeatureBatch,
    target_cache: FeatureBatch,
    source_ids: np.ndarray,
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
    """Fit SourceOnly once conceptually and report one grid/seed row."""
    source = subset_batch(
        source_cache,
        source_ids,
    )

    calibration = subset_batch(
        target_cache,
        calibration_ids,
    )

    healthy_eval = subset_batch(
        target_cache,
        healthy_eval_ids,
    )

    anomaly_eval = subset_batch(
        target_cache,
        anomaly_eval_ids,
    )

    if source.anomaly_labels.any():
        raise RuntimeError(
            "Source training subset contains anomalies."
        )

    if calibration.anomaly_labels.any():
        raise RuntimeError(
            "Target calibration contains anomalies."
        )

    if healthy_eval.anomaly_labels.any():
        raise RuntimeError(
            "Healthy target evaluation contains anomalies."
        )

    if not anomaly_eval.anomaly_labels.all():
        raise RuntimeError(
            "Target anomaly evaluation contains healthy rows."
        )

    preprocessor = FeaturePreprocessor(
        variance_threshold=1e-12
    )

    source_features = preprocessor.fit_transform(
        source.features
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

    detector = SourceOnlyDetector(
        false_alert_budget=false_alert_budget
    )

    # SourceOnly must ignore target_features. Passing an empty matrix would
    # violate some shared detector validators, so source_features is supplied
    # in both positions while the detector implementation uses the source.
    detector.fit(
        source_features=source_features,
        target_features=source_features,
    )

    detector.calibrate(
        calibration_features
    )

    if detector.threshold_ is None:
        raise RuntimeError(
            "SourceOnly threshold was not produced."
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
        "source_count": int(
            len(source_ids)
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
        ["label", "label_name"],
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

    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe SourceOnly evaluation on AURSAD."
        )
    )

    parser.add_argument(
        "--source-cache-path",
        type=Path,
        required=True,
        help=(
            "Independent source-domain FeatureBatch NPZ with the "
            "same 288-feature schema as AURSAD."
        ),
    )

    parser.add_argument(
        "--source-ids-path",
        type=Path,
        default=None,
        help=(
            "Optional CSV selecting healthy source training IDs. "
            "If omitted, the entire source cache must be healthy."
        ),
    )

    parser.add_argument(
        "--target-cache-path",
        type=Path,
        default=DEFAULT_TARGET_CACHE_PATH,
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
        "--allow-overlapping-id-namespace",
        action="store_true",
        help=(
            "Allow source and target caches to reuse numeric episode "
            "IDs. Use only when domains have independent namespaces."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_cache_path = (
        args.source_cache_path
        .expanduser()
        .resolve()
    )

    source_ids_path = (
        args.source_ids_path
        .expanduser()
        .resolve()
        if args.source_ids_path
        is not None
        else None
    )

    target_cache_path = (
        args.target_cache_path
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
        source_cache_path,
        "Source feature cache",
    )

    require_file(
        target_cache_path,
        "AURSAD target feature cache",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_results_path = (
        output_dir
        / "sourceonly_seed_results.csv"
    )

    per_class_seed_path = (
        output_dir
        / "sourceonly_per_class_seed_results.csv"
    )

    summary_path = (
        output_dir
        / "sourceonly_summary.csv"
    )

    per_class_summary_path = (
        output_dir
        / "sourceonly_per_class_recall.csv"
    )

    n_star_path = (
        output_dir
        / "sourceonly_n_star.json"
    )

    manifest_path = (
        output_dir
        / "sourceonly_run_manifest.json"
    )

    if args.overwrite:
        for path in (
            seed_results_path,
            per_class_seed_path,
            summary_path,
            per_class_summary_path,
            n_star_path,
            manifest_path,
        ):
            if path.exists():
                path.unlink()

    print("=" * 76)
    print("AURSAD SOURCEONLY COMMISSIONING EXPERIMENT")
    print("=" * 76)
    print(f"Source cache: {source_cache_path}")
    print(f"Target cache: {target_cache_path}")
    print(f"Protocol:     {protocol_dir}")
    print(f"Output:       {output_dir}")

    started = time.perf_counter()

    source_cache = load_feature_batch(
        source_cache_path
    )

    target_cache = load_feature_batch(
        target_cache_path
    )

    validate_feature_schema(
        source_cache,
        target_cache,
    )

    source_ids = load_source_ids(
        source_cache,
        source_ids_path,
    )

    protocol = load_target_protocol(
        protocol_dir
    )

    validate_target_protocol(
        protocol
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

    target_protocol_ids = (
        set(calibration_ids.tolist())
        | set(healthy_eval_ids.tolist())
        | set(anomaly_eval_ids.tolist())
        | set(
            protocol["commissioning"][
                "sample_nr"
            ].astype(int).tolist()
        )
    )

    source_id_set = set(
        source_ids.astype(int).tolist()
    )

    overlap = (
        source_id_set
        & target_protocol_ids
    )

    if (
        overlap
        and not args.allow_overlapping_id_namespace
    ):
        raise RuntimeError(
            "Source IDs overlap AURSAD target protocol IDs: "
            f"{sorted(overlap)[:20]}. "
            "If these are independent dataset namespaces that merely "
            "reuse integers, rerun with "
            "--allow-overlapping-id-namespace."
        )

    calibration_rank = conformal_rank(
        len(calibration_ids),
        args.false_alert_budget,
    )

    print(
        f"\nSource training executions: {len(source_ids):,}"
    )
    print(
        f"Calibration:               {len(calibration_ids):,}"
    )
    print(
        f"Healthy evaluation:        {len(healthy_eval_ids):,}"
    )
    print(
        f"Anomaly evaluation:        {len(anomaly_eval_ids):,}"
    )
    print(
        f"Conformal rank:            {calibration_rank}/"
        f"{len(calibration_ids)}"
    )

    result_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    total_runs = len(grid) * len(seeds)
    run_number = 0

    for n_value in grid:
        for seed in seeds:
            run_number += 1

            print(
                f"Processing N={n_value} seed={seed} "
                f"({run_number}/{total_runs})..."
            )

            result, run_class_rows = run_one(
                source_cache=source_cache,
                target_cache=target_cache,
                source_ids=source_ids,
                calibration_ids=calibration_ids,
                healthy_eval_ids=healthy_eval_ids,
                anomaly_eval_ids=anomaly_eval_ids,
                anomaly_eval_table=(
                    protocol["anomaly_eval"]
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

            result_rows.append(result)
            class_rows.extend(run_class_rows)

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
                f"recall={result['recall']:.4f} "
                f"FPR={result['false_positive_rate']:.4f} "
                f"AUROC={result['auroc']:.4f} "
                f"success={result['success']}"
            )

    results = pd.read_csv(
        seed_results_path
    )

    per_class_seed = pd.read_csv(
        per_class_seed_path
    )

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

    per_class_summary = build_per_class_summary(
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
        "note": (
            "SourceOnly does not depend on target commissioning N; "
            "rows across N and seeds should be identical."
        ),
    }

    n_star_path.write_text(
        json.dumps(
            json_safe(n_star_payload),
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started

    manifest = {
        "run_version": PROTOCOL_VERSION,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "command": " ".join(sys.argv),
        "dataset": "AURSAD",
        "detector": {
            "name": DETECTOR_NAME,
            "model": (
                "LedoitWolf Gaussian Mahalanobis"
            ),
            "training_data": (
                "independent healthy source-domain executions only"
            ),
            "target_commissioning_used": False,
            "calibration": (
                "fixed AURSAD healthy split-conformal calibration set"
            ),
        },
        "source": {
            "cache_path": str(
                source_cache_path
            ),
            "cache_sha256": sha256_file(
                source_cache_path
            ),
            "source_ids_path": (
                str(source_ids_path)
                if source_ids_path
                is not None
                else None
            ),
            "source_count": int(
                len(source_ids)
            ),
            "all_selected_source_rows_healthy": True,
            "schema_matches_target_exactly": True,
            "numeric_id_overlap_with_target": int(
                len(overlap)
            ),
            "overlapping_id_namespace_allowed": bool(
                args.allow_overlapping_id_namespace
            ),
        },
        "target": {
            "cache_path": str(
                target_cache_path
            ),
            "cache_sha256": sha256_file(
                target_cache_path
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
        },
        "result": {
            "run_count": int(
                len(results)
            ),
            "n_star": n_star,
            "sourceonly_is_horizontal_baseline": True,
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
            "total": float(elapsed),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "validation": {
            "independent_source_required": True,
            "healthy_source_only_checked": True,
            "exact_feature_schema_checked": True,
            "target_protocol_disjointness_checked": True,
            "target_calibration_is_healthy": True,
            "target_healthy_eval_is_healthy": True,
            "target_anomaly_eval_is_anomalous": True,
            "preprocessing_fitted_on_source_only": True,
            "target_commissioning_not_used": True,
        },
        "limitations": [
            (
                "SourceOnly is meaningful only when the supplied "
                "source domain represents the intended transfer setting."
            ),
            (
                "Source and target must share exactly the same feature "
                "representation; the current 48-channel AURSAD schema "
                "does not automatically match voraus-AD."
            ),
            (
                "Because SourceOnly ignores commissioning data, all "
                "reported N/seed rows are expected to be identical."
            ),
            (
                "Damaged-thread label 4 has only three AURSAD "
                "evaluation executions."
            ),
        ],
    }

    manifest_path.write_text(
        json.dumps(
            json_safe(manifest),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("SOURCEONLY COMPLETE")
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