#!/usr/bin/env python3
"""Frozen diagnostic-only context-stratified analysis for voraus-AD.

This script reconstructs the exact detector fits, preprocessing, split-conformal
thresholds, and episode-level scores used by ``experiments/run_commissioning.py``.
Before writing diagnostic results, every detector/commissioning-size/seed run is
checked against ``outputs/commissioning_seed_results.csv``. A mismatch aborts the
analysis, preventing accidental analysis of a changed protocol.

Primary context analysis uses only scientifically admissible metadata fields that
are available without anomaly labels and are constant within each episode. The
candidate fields are ``action`` and ``active``. Their admissibility is audited at
runtime; unsupported fields are not silently converted into semantic contexts.

An optional exploratory healthy-only KMeans sensitivity analysis is available via
``--exploratory-clusters``. It is deliberately labeled exploratory, uses a small
predeclared routing representation separate from the full anomaly-score feature
vector, is fitted once on the union of target commissioning pools only, and never
uses calibration, healthy-evaluation, anomaly-evaluation, or anomaly labels.

No detector, split, calibration rule, threshold rule, or hyperparameter is tuned.
All outputs are diagnostic and must not be used for method selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector
from src.detectors import RACEDetector
from src.evaluation import detector_factories, fit_detector
from src.feature_extractor import (
    STATISTIC_NAMES,
    extract_feature_matrix,
    make_feature_names,
)
from src.split_generator import create_experiment_split
from src.voraus_loader import RobotCycle, load_cycles, select_signal_columns


# ---------------------------------------------------------------------------
# Frozen protocol
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "nested-split-conformal-v2"
ANALYSIS_VERSION = "context-diagnostic-v1"
GLOBAL_SEED = 42
SEEDS = tuple(range(20))
COMMISSIONING_GRID = (10, 25, 50, 100)
MAXIMUM_COMMISSIONING_SIZE = 100
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
SIGNAL_SET = "measured"
VARIANCE_THRESHOLD = 1e-12
DETECTOR_ORDER = ("SourceOnly", "TargetOnly", "Pooled", "RACE")

# Numerical agreement tolerances for reconstruction versus frozen CSV.
ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-9

# Candidate semantic context fields. ``anomaly`` and ``category`` are excluded
# because they are outcome/label related; ``setting`` is constant in target data.
SEMANTIC_CONTEXT_CANDIDATES = ("action", "active")

# Separate, predeclared routing representation for the optional exploratory
# sensitivity analysis. These summaries do not use the full anomaly-score vector.
ROUTING_STATISTICS = ("mean", "std", "total_variation")
EXPLORATORY_CLUSTER_COUNT = 3

np.random.seed(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunReproduction:
    detector: str
    commissioning_size: int
    seed: int
    reconstructed_threshold: float
    frozen_threshold: float
    reconstructed_fpr: float
    frozen_fpr: float
    reconstructed_recall: float
    frozen_recall: float
    reconstructed_retained_features: int
    frozen_retained_features: int
    reconstructed_target_weight: float | None
    frozen_target_weight: float | None
    passed: bool


@dataclass(frozen=True)
class ContextAudit:
    candidate: str
    present: bool
    constant_within_episode: bool
    missing_episode_count: int
    unique_healthy_target_values: tuple[str, ...]
    unique_anomaly_values: tuple[str, ...]
    admitted: bool
    reason: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen diagnostic context analysis for voraus-AD."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet",
        help="Path to the official 100 Hz voraus-AD Parquet file.",
    )
    parser.add_argument(
        "--frozen-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "commissioning_seed_results.csv",
        help="Authoritative frozen seed-level result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "context_analysis",
        help="Directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--exploratory-clusters",
        action="store_true",
        help=(
            "Enable explicitly exploratory healthy-only KMeans contexts. "
            "Semantic metadata contexts remain the primary analysis."
        ),
    )
    parser.add_argument(
        "--skip-dataset-hash",
        action="store_true",
        help="Skip SHA-256 hashing of the large raw Parquet file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing context-analysis directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        if math.isfinite(scalar):
            return scalar
        return str(scalar)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def close_enough(actual: float, expected: float) -> bool:
    return bool(
        np.isclose(
            actual,
            expected,
            atol=ABS_TOLERANCE,
            rtol=REL_TOLERANCE,
            equal_nan=True,
        )
    )


def normalize_optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def scalar_to_context(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:.12g}"
    return str(value)


def episode_ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def matrix_for(
    cycles: Sequence[RobotCycle],
    feature_by_episode: dict[int, np.ndarray],
) -> np.ndarray:
    if not cycles:
        raise ValueError("Cannot build a feature matrix for an empty cycle group.")
    return np.vstack([feature_by_episode[int(cycle.episode_id)] for cycle in cycles])


def all_pairwise_overlap_counts(groups: dict[str, Sequence[int]]) -> dict[str, int]:
    names = list(groups)
    result: dict[str, int] = {}
    for index, first in enumerate(names):
        first_ids = set(map(int, groups[first]))
        for second in names[index + 1 :]:
            second_ids = set(map(int, groups[second]))
            result[f"{first}__{second}"] = len(first_ids & second_ids)
    return result


def finite_ratio(maximum: float, second_maximum: float) -> float:
    if second_maximum > 0.0:
        return float(maximum / second_maximum)
    if maximum == 0.0:
        return 1.0
    return float("inf")


def empirical_rate(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=np.float64) > threshold))


# ---------------------------------------------------------------------------
# Metadata context audit
# ---------------------------------------------------------------------------

def read_episode_metadata(
    data_path: Path,
    candidates: Sequence[str],
) -> tuple[pd.DataFrame, list[ContextAudit], list[str]]:
    schema_columns = list(pq.ParquetFile(data_path).schema.names)
    present_candidates = [column for column in candidates if column in schema_columns]
    columns = ["sample", "anomaly", "setting"] + present_candidates

    frame = pd.read_parquet(data_path, columns=columns)
    if frame.empty:
        raise ValueError("Raw metadata read returned no rows.")

    frame["sample"] = frame["sample"].astype(np.int64)
    episode_base = (
        frame.groupby("sample", sort=True)
        .agg(anomaly=("anomaly", "first"), setting=("setting", "first"))
        .reset_index()
    )

    audits: list[ContextAudit] = []
    admitted: list[str] = []
    episode_metadata = episode_base.copy()

    for candidate in candidates:
        if candidate not in present_candidates:
            audits.append(
                ContextAudit(
                    candidate=candidate,
                    present=False,
                    constant_within_episode=False,
                    missing_episode_count=len(episode_base),
                    unique_healthy_target_values=tuple(),
                    unique_anomaly_values=tuple(),
                    admitted=False,
                    reason="Column is absent from the Parquet schema.",
                )
            )
            continue

        grouped = frame.groupby("sample", sort=True)[candidate]
        nunique = grouped.nunique(dropna=False)
        constant = bool((nunique <= 1).all())
        first_values = grouped.first()
        missing_count = int(first_values.isna().sum())
        episode_metadata = episode_metadata.merge(
            first_values.rename(candidate),
            left_on="sample",
            right_index=True,
            how="left",
            validate="one_to_one",
        )

        healthy_target_mask = (
            ~episode_metadata["anomaly"].astype(bool)
            & (episode_metadata["setting"].astype(int) == 73)
        )
        anomaly_mask = episode_metadata["anomaly"].astype(bool)
        healthy_values = tuple(
            sorted(
                scalar_to_context(value)
                for value in episode_metadata.loc[healthy_target_mask, candidate].drop_duplicates()
            )
        )
        anomaly_values = tuple(
            sorted(
                scalar_to_context(value)
                for value in episode_metadata.loc[anomaly_mask, candidate].drop_duplicates()
            )
        )

        admitted_flag = constant and missing_count == 0 and len(healthy_values) >= 2
        if not constant:
            reason = (
                "Rejected: values change within at least one episode, so a single "
                "pre-decision episode context is not established."
            )
        elif missing_count > 0:
            reason = "Rejected: one or more episodes have missing context values."
        elif len(healthy_values) < 2:
            reason = (
                "Rejected: fewer than two values occur in target healthy episodes, "
                "so the field cannot stratify the target healthy regime."
            )
        else:
            reason = (
                "Admitted as a diagnostic metadata context: present without using "
                "anomaly/category labels, constant within episode, complete, and "
                "non-constant across target healthy episodes. Semantics still require "
                "dataset-documentation interpretation before causal naming."
            )
            admitted.append(candidate)

        audits.append(
            ContextAudit(
                candidate=candidate,
                present=True,
                constant_within_episode=constant,
                missing_episode_count=missing_count,
                unique_healthy_target_values=healthy_values,
                unique_anomaly_values=anomaly_values,
                admitted=admitted_flag,
                reason=reason,
            )
        )

    if admitted:
        episode_metadata["semantic_context"] = episode_metadata.apply(
            lambda row: "|".join(
                f"{column}={scalar_to_context(row[column])}" for column in admitted
            ),
            axis=1,
        )
    else:
        episode_metadata["semantic_context"] = "UNAVAILABLE"

    return episode_metadata, audits, admitted


# ---------------------------------------------------------------------------
# Optional exploratory context model
# ---------------------------------------------------------------------------

def routing_feature_indices(signal_columns: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    all_names = make_feature_names(signal_columns)
    selected_indices: list[int] = []
    selected_names: list[str] = []
    for index, name in enumerate(all_names):
        statistic = name.rsplit("__", maxsplit=1)[-1]
        if statistic in ROUTING_STATISTICS:
            selected_indices.append(index)
            selected_names.append(name)
    if not selected_indices:
        raise RuntimeError("No routing features matched the predeclared statistics.")
    return np.asarray(selected_indices, dtype=np.int64), tuple(selected_names)


def fit_exploratory_context_model(
    cycles: Sequence[RobotCycle],
    feature_by_episode: dict[int, np.ndarray],
    signal_columns: Sequence[str],
) -> tuple[StandardScaler, KMeans, np.ndarray, tuple[str, ...], list[int]]:
    """Fit once on the union of target commissioning pools over frozen seeds.

    This fitting population contains healthy target episodes only and excludes all
    calibration/evaluation/anomaly episodes from the corresponding seed-local role.
    It is a sensitivity analysis, not a semantic operating-context model.
    """
    reference_ids: set[int] = set()
    for seed in SEEDS:
        split = create_experiment_split(
            cycles=cycles,
            commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            seed=seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=NORMAL_EVALUATION_SIZE,
            maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
        )
        reference_ids.update(episode_ids(split.target_commissioning))

    routing_indices, routing_names = routing_feature_indices(signal_columns)
    reference_matrix = np.vstack(
        [feature_by_episode[episode_id][routing_indices] for episode_id in sorted(reference_ids)]
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(reference_matrix)
    model = KMeans(
        n_clusters=EXPLORATORY_CLUSTER_COUNT,
        random_state=GLOBAL_SEED,
        n_init=20,
        algorithm="lloyd",
    )
    model.fit(scaled)
    return scaler, model, routing_indices, routing_names, sorted(reference_ids)


def assign_exploratory_clusters(
    episode_id_values: Sequence[int],
    feature_by_episode: dict[int, np.ndarray],
    scaler: StandardScaler,
    model: KMeans,
    routing_indices: np.ndarray,
) -> np.ndarray:
    matrix = np.vstack(
        [feature_by_episode[int(episode_id)][routing_indices] for episode_id in episode_id_values]
    )
    return model.predict(scaler.transform(matrix)).astype(np.int64)


# ---------------------------------------------------------------------------
# Frozen result loading and validation
# ---------------------------------------------------------------------------

def load_frozen_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Frozen result CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "protocol_version",
        "detector",
        "commissioning_size",
        "seed",
        "false_positive_rate",
        "recall",
        "success",
        "threshold",
        "retained_features",
        "target_weight",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Frozen result CSV is missing columns: {sorted(missing)}")
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions != {PROTOCOL_VERSION}:
        raise ValueError(
            f"Expected protocol {PROTOCOL_VERSION!r}; found {sorted(versions)}."
        )
    key_counts = frame.groupby(["detector", "commissioning_size", "seed"]).size()
    duplicates = key_counts[key_counts != 1]
    if not duplicates.empty:
        raise ValueError(f"Frozen result keys are not unique:\n{duplicates.head()}")
    expected_count = len(DETECTOR_ORDER) * len(COMMISSIONING_GRID) * len(SEEDS)
    if len(frame) != expected_count:
        raise ValueError(f"Expected {expected_count} frozen rows, found {len(frame)}.")
    return frame


def frozen_row(
    frozen: pd.DataFrame,
    detector: str,
    commissioning_size: int,
    seed: int,
) -> pd.Series:
    selection = frozen[
        (frozen["detector"] == detector)
        & (frozen["commissioning_size"] == commissioning_size)
        & (frozen["seed"] == seed)
    ]
    if len(selection) != 1:
        raise RuntimeError(
            f"Expected one frozen row for {(detector, commissioning_size, seed)}, "
            f"found {len(selection)}."
        )
    return selection.iloc[0]


def compare_reconstruction(
    *,
    detector: str,
    commissioning_size: int,
    seed: int,
    threshold: float,
    fpr: float,
    recall: float,
    retained_features: int,
    target_weight: float | None,
    frozen: pd.DataFrame,
) -> RunReproduction:
    row = frozen_row(frozen, detector, commissioning_size, seed)
    expected_weight = normalize_optional_float(row["target_weight"])
    weight_ok = (
        target_weight is None and expected_weight is None
    ) or (
        target_weight is not None
        and expected_weight is not None
        and close_enough(target_weight, expected_weight)
    )
    passed = all(
        [
            close_enough(threshold, float(row["threshold"])),
            close_enough(fpr, float(row["false_positive_rate"])),
            close_enough(recall, float(row["recall"])),
            retained_features == int(row["retained_features"]),
            weight_ok,
        ]
    )
    return RunReproduction(
        detector=detector,
        commissioning_size=commissioning_size,
        seed=seed,
        reconstructed_threshold=threshold,
        frozen_threshold=float(row["threshold"]),
        reconstructed_fpr=fpr,
        frozen_fpr=float(row["false_positive_rate"]),
        reconstructed_recall=recall,
        frozen_recall=float(row["recall"]),
        reconstructed_retained_features=retained_features,
        frozen_retained_features=int(row["retained_features"]),
        reconstructed_target_weight=target_weight,
        frozen_target_weight=expected_weight,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Context and score summaries
# ---------------------------------------------------------------------------

def context_columns(assignments: pd.DataFrame, use_clusters: bool) -> list[str]:
    columns: list[str] = []
    if "semantic_context" in assignments and not (
        assignments["semantic_context"] == "UNAVAILABLE"
    ).all():
        columns.append("semantic_context")
    if use_clusters:
        columns.append("exploratory_cluster")
    return columns


def summarize_context_scores(per_episode_scores: pd.DataFrame) -> pd.DataFrame:
    context_types: list[tuple[str, str]] = []
    if "semantic_context" in per_episode_scores.columns and not (
        per_episode_scores["semantic_context"] == "UNAVAILABLE"
    ).all():
        context_types.append(("semantic", "semantic_context"))
    if "exploratory_cluster" in per_episode_scores.columns:
        context_types.append(("exploratory_cluster", "exploratory_cluster"))

    rows: list[dict[str, Any]] = []
    for context_type, context_column in context_types:
        grouped = per_episode_scores.groupby(
            ["seed", "commissioning_size", "detector", "split", context_column],
            dropna=False,
            sort=True,
        )
        for keys, group in grouped:
            seed, commissioning_size, detector, split, context_value = keys
            scores = np.sort(group["score"].to_numpy(dtype=np.float64))
            maximum = float(scores[-1])
            second = float(scores[-2]) if len(scores) >= 2 else float("nan")
            rows.append(
                {
                    "seed": int(seed),
                    "commissioning_size": int(commissioning_size),
                    "detector": str(detector),
                    "split": str(split),
                    "context_type": context_type,
                    "context": scalar_to_context(context_value),
                    "episode_count": int(len(scores)),
                    "score_mean": float(np.mean(scores)),
                    "score_std": float(np.std(scores, ddof=1)) if len(scores) >= 2 else 0.0,
                    "score_median": float(np.median(scores)),
                    "score_q90": float(np.quantile(scores, 0.90)),
                    "score_q95": float(np.quantile(scores, 0.95)),
                    "score_q99": float(np.quantile(scores, 0.99)),
                    "score_max": maximum,
                    "score_second_max": second,
                    "max_second_ratio": finite_ratio(maximum, second)
                    if len(scores) >= 2
                    else float("nan"),
                    "max_second_gap": maximum - second if len(scores) >= 2 else float("nan"),
                    "predicted_anomaly_count": int(group["prediction"].sum()),
                    "predicted_anomaly_rate": float(group["prediction"].mean()),
                    "threshold_episode_count": int(group["is_threshold_episode"].sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_assignment_shift(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    context_specs: list[tuple[str, str]] = []
    if not (assignments["semantic_context"] == "UNAVAILABLE").all():
        context_specs.append(("semantic", "semantic_context"))
    if "exploratory_cluster" in assignments.columns:
        context_specs.append(("exploratory_cluster", "exploratory_cluster"))

    for seed, seed_frame in assignments.groupby("seed", sort=True):
        for context_type, context_column in context_specs:
            calibration = seed_frame[seed_frame["split"] == "calibration"]
            healthy_eval = seed_frame[seed_frame["split"] == "healthy_evaluation"]
            anomaly_eval = seed_frame[seed_frame["split"] == "anomaly_evaluation"]
            contexts = sorted(
                set(calibration[context_column].astype(str))
                | set(healthy_eval[context_column].astype(str))
                | set(anomaly_eval[context_column].astype(str))
            )
            for context in contexts:
                p_cal = float(np.mean(calibration[context_column].astype(str) == context))
                p_healthy = float(np.mean(healthy_eval[context_column].astype(str) == context))
                p_anomaly = float(np.mean(anomaly_eval[context_column].astype(str) == context))
                rows.append(
                    {
                        "seed": int(seed),
                        "context_type": context_type,
                        "context": context,
                        "p_calibration_healthy": p_cal,
                        "p_evaluation_healthy": p_healthy,
                        "p_evaluation_anomaly": p_anomaly,
                        "anomaly_minus_calibration": p_anomaly - p_cal,
                        "anomaly_minus_healthy_evaluation": p_anomaly - p_healthy,
                        "healthy_evaluation_minus_calibration": p_healthy - p_cal,
                    }
                )
    return pd.DataFrame(rows)


def threshold_context_summary(threshold_rows: pd.DataFrame) -> pd.DataFrame:
    context_specs: list[tuple[str, str]] = []
    if not (threshold_rows["semantic_context"] == "UNAVAILABLE").all():
        context_specs.append(("semantic", "semantic_context"))
    if "exploratory_cluster" in threshold_rows.columns:
        context_specs.append(("exploratory_cluster", "exploratory_cluster"))

    rows: list[dict[str, Any]] = []
    for context_type, context_column in context_specs:
        grouped = threshold_rows.groupby(
            ["detector", "commissioning_size", context_column],
            dropna=False,
            sort=True,
        )
        for keys, group in grouped:
            detector, commissioning_size, context = keys
            rows.append(
                {
                    "detector": str(detector),
                    "commissioning_size": int(commissioning_size),
                    "context_type": context_type,
                    "context": scalar_to_context(context),
                    "threshold_run_count": int(len(group)),
                    "threshold_run_fraction": float(len(group) / len(SEEDS)),
                    "unique_threshold_episode_count": int(group["representative_threshold_episode_id"].nunique()),
                    "mean_threshold": float(group["threshold"].mean()),
                    "median_threshold": float(group["threshold"].median()),
                    "mean_max_second_ratio": float(group["max_second_ratio"].mean()),
                    "median_max_second_ratio": float(group["max_second_ratio"].median()),
                    "mean_max_second_gap": float(group["max_second_gap"].mean()),
                    "tied_threshold_run_count": int((group["threshold_tie_count"] > 1).sum()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figures(
    threshold_runs: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    assignment_shift: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    recurring = (
        threshold_runs.groupby(["detector", "representative_threshold_episode_id"])
        .size()
        .rename("count")
        .reset_index()
    )
    for detector in DETECTOR_ORDER:
        subset = recurring[recurring["detector"] == detector].nlargest(10, "count")
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.bar(subset["representative_threshold_episode_id"].astype(str), subset["count"])
        ax.set_title(f"Recurring threshold-setting episodes: {detector}")
        ax.set_xlabel("Healthy calibration episode ID")
        ax.set_ylabel("Number of detector–N–seed runs")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = figure_dir / f"threshold_episode_frequency_{detector.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path.relative_to(output_dir)))

    if not threshold_summary.empty:
        for context_type in threshold_summary["context_type"].unique():
            subset = threshold_summary[threshold_summary["context_type"] == context_type].copy()
            if subset.empty:
                continue
            pivot = subset.pivot_table(
                index="context",
                columns="detector",
                values="threshold_run_fraction",
                aggfunc="mean",
                fill_value=0.0,
            )
            fig, ax = plt.subplots(figsize=(9.0, max(4.5, 0.45 * len(pivot))))
            image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
            ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
            ax.set_xlabel("Detector")
            ax.set_ylabel("Context")
            ax.set_title(f"Threshold-setting frequency by {context_type} context")
            fig.colorbar(image, ax=ax, label="Mean fraction across N")
            fig.tight_layout()
            path = figure_dir / f"threshold_context_frequency_{context_type}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            created.append(str(path.relative_to(output_dir)))

    if not assignment_shift.empty:
        for context_type in assignment_shift["context_type"].unique():
            subset = assignment_shift[assignment_shift["context_type"] == context_type]
            summary = (
                subset.groupby("context")["anomaly_minus_calibration"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("mean")
            )
            fig, ax = plt.subplots(figsize=(8.0, max(4.5, 0.45 * len(summary))))
            ax.barh(summary["context"].astype(str), summary["mean"], xerr=summary["std"].fillna(0.0))
            ax.axvline(0.0, linewidth=1.0)
            ax.set_xlabel("Anomaly assignment share − healthy calibration share")
            ax.set_ylabel("Context")
            ax.set_title(f"Context-assignment shift ({context_type})")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            path = figure_dir / f"context_assignment_shift_{context_type}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            created.append(str(path.relative_to(output_dir)))

    return created


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    data_path = args.data_path.resolve()
    frozen_path = args.frozen_results.resolve()
    output_dir = args.output_dir.resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Pass the external file explicitly, for example:\n"
            "python experiments/run_context_stratified_analysis.py "
            "--data-path D:/datasets/voraus-ad-dataset-100hz.parquet"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace diagnostic artifacts."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Frozen context-stratified diagnostic analysis ===")
    print(f"Dataset: {data_path}")
    print(f"Frozen results: {frozen_path}")
    print(f"Output directory: {output_dir}")

    frozen = load_frozen_results(frozen_path)
    print("Loading measured-signal cycles...")
    cycles = load_cycles(path=data_path, signal_set=SIGNAL_SET)
    raw_features, feature_episode_ids = extract_feature_matrix(cycles)
    feature_by_episode = {
        int(episode_id): raw_features[index]
        for index, episode_id in enumerate(feature_episode_ids)
    }
    cycle_by_episode = {int(cycle.episode_id): cycle for cycle in cycles}
    if len(cycle_by_episode) != len(cycles):
        raise RuntimeError("Episode IDs are not unique.")

    signal_columns = select_signal_columns(data_path, signal_set=SIGNAL_SET)
    print(
        f"Loaded {len(cycles)} episodes, {len(signal_columns)} measured signals, "
        f"and {raw_features.shape[1]} statistical features."
    )

    print("Auditing candidate pre-fault metadata contexts...")
    metadata, metadata_audits, admitted_context_fields = read_episode_metadata(
        data_path, SEMANTIC_CONTEXT_CANDIDATES
    )
    metadata_by_episode = metadata.set_index("sample", verify_integrity=True)
    for audit in metadata_audits:
        print(f"  {audit.candidate}: {'ADMITTED' if audit.admitted else 'REJECTED'} — {audit.reason}")

    cluster_scaler: StandardScaler | None = None
    cluster_model: KMeans | None = None
    routing_indices: np.ndarray | None = None
    routing_names: tuple[str, ...] = tuple()
    cluster_reference_ids: list[int] = []
    if args.exploratory_clusters:
        print("Fitting one exploratory healthy-only context model...")
        (
            cluster_scaler,
            cluster_model,
            routing_indices,
            routing_names,
            cluster_reference_ids,
        ) = fit_exploratory_context_model(
            cycles=cycles,
            feature_by_episode=feature_by_episode,
            signal_columns=signal_columns,
        )
        print(
            f"Exploratory model fitted on {len(cluster_reference_ids)} unique target "
            f"commissioning-pool episodes using {len(routing_names)} routing features."
        )

    factories = detector_factories(FALSE_ALERT_BUDGET)
    assignments_records: list[dict[str, Any]] = []
    membership_records: list[dict[str, Any]] = []
    threshold_run_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    reproduction_records: list[RunReproduction] = []
    leakage_audit: dict[str, Any] = {}
    membership_hashes: dict[str, str] = {}

    # Build seed-level assignments once; they are invariant across N.
    split_cache: dict[tuple[int, int], Any] = {}
    for seed in SEEDS:
        max_split = create_experiment_split(
            cycles=cycles,
            commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            seed=seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=NORMAL_EVALUATION_SIZE,
            maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
        )
        split_cache[(seed, MAXIMUM_COMMISSIONING_SIZE)] = max_split
        max_split.verify_no_overlap()

        seed_groups = {
            "source_train": episode_ids(max_split.source_train),
            "commissioning_pool": episode_ids(max_split.target_commissioning),
            "calibration": episode_ids(max_split.target_calibration),
            "healthy_evaluation": episode_ids(max_split.target_normal_evaluation),
            "anomaly_evaluation": episode_ids(max_split.target_anomaly_evaluation),
        }
        overlaps = all_pairwise_overlap_counts(seed_groups)
        if any(overlaps.values()):
            raise RuntimeError(f"Leakage detected for seed {seed}: {overlaps}")
        leakage_audit[str(seed)] = overlaps
        membership_hashes[str(seed)] = sha256_json(seed_groups)

        for split_name, ids in seed_groups.items():
            for position, episode_id in enumerate(ids):
                cycle = cycle_by_episode[episode_id]
                membership_records.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "position": position,
                        "episode_id": episode_id,
                        "is_anomaly": int(cycle.anomaly),
                        "category": int(cycle.category),
                        "setting": int(cycle.setting),
                    }
                )

        assignment_groups = {
            "calibration": seed_groups["calibration"],
            "healthy_evaluation": seed_groups["healthy_evaluation"],
            "anomaly_evaluation": seed_groups["anomaly_evaluation"],
        }
        for split_name, ids in assignment_groups.items():
            clusters: np.ndarray | None = None
            if args.exploratory_clusters:
                assert cluster_scaler is not None
                assert cluster_model is not None
                assert routing_indices is not None
                clusters = assign_exploratory_clusters(
                    ids,
                    feature_by_episode,
                    cluster_scaler,
                    cluster_model,
                    routing_indices,
                )
            for position, episode_id in enumerate(ids):
                cycle = cycle_by_episode[episode_id]
                meta_row = metadata_by_episode.loc[episode_id]
                record = {
                    "seed": seed,
                    "split": split_name,
                    "position": position,
                    "episode_id": episode_id,
                    "is_anomaly": int(cycle.anomaly),
                    "category": int(cycle.category),
                    "setting": int(cycle.setting),
                    "semantic_context": str(meta_row["semantic_context"]),
                }
                for column in admitted_context_fields:
                    record[column] = scalar_to_context(meta_row[column])
                if clusters is not None:
                    record["exploratory_cluster"] = int(clusters[position])
                assignments_records.append(record)

    assignments = pd.DataFrame(assignments_records)
    assignment_lookup = assignments.set_index(["seed", "split", "episode_id"])

    total_runs = len(SEEDS) * len(COMMISSIONING_GRID) * len(DETECTOR_ORDER)
    run_index = 0
    for seed in SEEDS:
        for commissioning_size in COMMISSIONING_GRID:
            if commissioning_size == MAXIMUM_COMMISSIONING_SIZE:
                split = split_cache[(seed, MAXIMUM_COMMISSIONING_SIZE)]
            else:
                split = create_experiment_split(
                    cycles=cycles,
                    commissioning_size=commissioning_size,
                    seed=seed,
                    calibration_size=CALIBRATION_SIZE,
                    normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                    maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
                )
                split.verify_no_overlap()

            # Verify nested prefix membership against the saved N=100 pool.
            expected_prefix = episode_ids(
                split_cache[(seed, MAXIMUM_COMMISSIONING_SIZE)].target_commissioning
            )[:commissioning_size]
            observed_prefix = episode_ids(split.target_commissioning)
            if observed_prefix != expected_prefix:
                raise RuntimeError(
                    f"Nested commissioning prefix failed for seed={seed}, N={commissioning_size}."
                )

            source_raw = matrix_for(split.source_train, feature_by_episode)
            target_raw = matrix_for(split.target_commissioning, feature_by_episode)
            calibration_raw = matrix_for(split.target_calibration, feature_by_episode)
            healthy_eval_raw = matrix_for(split.target_normal_evaluation, feature_by_episode)
            anomaly_eval_raw = matrix_for(split.target_anomaly_evaluation, feature_by_episode)

            split_map = {
                "calibration": split.target_calibration,
                "healthy_evaluation": split.target_normal_evaluation,
                "anomaly_evaluation": split.target_anomaly_evaluation,
            }

            for detector_name in DETECTOR_ORDER:
                run_index += 1
                print(
                    f"Processing run {run_index}/{total_runs}: seed={seed}, "
                    f"N={commissioning_size}, detector={detector_name}"
                )
                detector, preprocessor, _, _ = fit_detector(
                    detector_name=detector_name,
                    detector_factory=factories[detector_name],
                    source_raw=source_raw,
                    target_raw=target_raw,
                )
                transformed = {
                    "calibration": preprocessor.transform(calibration_raw),
                    "healthy_evaluation": preprocessor.transform(healthy_eval_raw),
                    "anomaly_evaluation": preprocessor.transform(anomaly_eval_raw),
                }
                scores = {
                    split_name: detector.score_samples(features)
                    for split_name, features in transformed.items()
                }
                threshold = BaseDetector.conformal_quantile(
                    scores["calibration"], FALSE_ALERT_BUDGET
                )
                detector.calibrate_from_scores(scores["calibration"])
                if detector.threshold_ is None or not close_enough(detector.threshold_, threshold):
                    raise RuntimeError("Detector calibration disagrees with conformal_quantile().")

                fpr = empirical_rate(scores["healthy_evaluation"], threshold)
                recall = empirical_rate(scores["anomaly_evaluation"], threshold)
                retained_features = int(preprocessor.output_feature_count_ or 0)
                target_weight = (
                    float(detector.target_weight_)
                    if isinstance(detector, RACEDetector)
                    and detector.target_weight_ is not None
                    else None
                )
                reproduction = compare_reconstruction(
                    detector=detector_name,
                    commissioning_size=commissioning_size,
                    seed=seed,
                    threshold=threshold,
                    fpr=fpr,
                    recall=recall,
                    retained_features=retained_features,
                    target_weight=target_weight,
                    frozen=frozen,
                )
                reproduction_records.append(reproduction)
                if not reproduction.passed:
                    reproduction_path = output_dir / "reproduction_check_failed.csv"
                    pd.DataFrame([asdict(item) for item in reproduction_records]).to_csv(
                        reproduction_path, index=False
                    )
                    raise RuntimeError(
                        "Frozen-result reconstruction failed. Diagnostic analysis aborted. "
                        f"See {reproduction_path}. Last mismatch: {reproduction}"
                    )

                calibration_scores = scores["calibration"]
                calibration_ids = episode_ids(split.target_calibration)
                sorted_scores = np.sort(calibration_scores)
                maximum = float(sorted_scores[-1])
                second_maximum = float(sorted_scores[-2])
                tied_mask = np.isclose(
                    calibration_scores,
                    threshold,
                    atol=ABS_TOLERANCE,
                    rtol=REL_TOLERANCE,
                )
                tied_ids = sorted(
                    int(calibration_ids[index])
                    for index in np.flatnonzero(tied_mask)
                )
                representative_id = tied_ids[0]
                threshold_assignment = assignment_lookup.loc[
                    (seed, "calibration", representative_id)
                ]
                threshold_record = {
                    "seed": seed,
                    "commissioning_size": commissioning_size,
                    "detector": detector_name,
                    "threshold": threshold,
                    "representative_threshold_episode_id": representative_id,
                    "all_threshold_episode_ids": ";".join(map(str, tied_ids)),
                    "threshold_tie_count": len(tied_ids),
                    "max_score": maximum,
                    "second_max_score": second_maximum,
                    "max_second_ratio": finite_ratio(maximum, second_maximum),
                    "max_second_gap": maximum - second_maximum,
                    "semantic_context": str(threshold_assignment["semantic_context"]),
                    "false_positive_rate": fpr,
                    "recall": recall,
                    "retained_features": retained_features,
                    "target_weight": target_weight,
                }
                if args.exploratory_clusters:
                    threshold_record["exploratory_cluster"] = int(
                        threshold_assignment["exploratory_cluster"]
                    )
                threshold_run_records.append(threshold_record)

                for split_name, split_cycles in split_map.items():
                    split_scores = scores[split_name]
                    for index, cycle in enumerate(split_cycles):
                        episode_id = int(cycle.episode_id)
                        assignment = assignment_lookup.loc[(seed, split_name, episode_id)]
                        record = {
                            "seed": seed,
                            "commissioning_size": commissioning_size,
                            "detector": detector_name,
                            "split": split_name,
                            "episode_id": episode_id,
                            "is_anomaly": int(cycle.anomaly),
                            "category": int(cycle.category),
                            "setting": int(cycle.setting),
                            "score": float(split_scores[index]),
                            "threshold": threshold,
                            "prediction": int(split_scores[index] > threshold),
                            "is_threshold_episode": int(
                                split_name == "calibration" and episode_id in tied_ids
                            ),
                            "semantic_context": str(assignment["semantic_context"]),
                        }
                        if args.exploratory_clusters:
                            record["exploratory_cluster"] = int(
                                assignment["exploratory_cluster"]
                            )
                        score_records.append(record)

    reproduction_df = pd.DataFrame([asdict(item) for item in reproduction_records])
    if not reproduction_df["passed"].all():
        raise RuntimeError("At least one frozen reconstruction failed.")

    memberships = pd.DataFrame(membership_records)
    threshold_runs = pd.DataFrame(threshold_run_records)
    per_episode_scores = pd.DataFrame(score_records)
    threshold_summary = threshold_context_summary(threshold_runs)
    context_scores = summarize_context_scores(per_episode_scores)
    assignment_shift = summarize_assignment_shift(assignments)

    # Required artifacts.
    required_paths = {
        "context_episode_assignments.csv": output_dir / "context_episode_assignments.csv",
        "threshold_context_summary.csv": output_dir / "threshold_context_summary.csv",
        "context_score_summary.csv": output_dir / "context_score_summary.csv",
        "context_assignment_shift.csv": output_dir / "context_assignment_shift.csv",
    }
    assignments.to_csv(required_paths["context_episode_assignments.csv"], index=False)
    threshold_summary.to_csv(required_paths["threshold_context_summary.csv"], index=False)
    context_scores.to_csv(required_paths["context_score_summary.csv"], index=False)
    assignment_shift.to_csv(required_paths["context_assignment_shift.csv"], index=False)

    # Additional audit-friendly artifacts.
    memberships_path = output_dir / "frozen_memberships.csv"
    threshold_runs_path = output_dir / "threshold_run_details.csv"
    per_episode_scores_path = output_dir / "per_episode_scores.csv"
    reproduction_path = output_dir / "frozen_result_reproduction.csv"
    metadata_audit_path = output_dir / "context_metadata_audit.csv"
    memberships.to_csv(memberships_path, index=False)
    threshold_runs.to_csv(threshold_runs_path, index=False)
    per_episode_scores.to_csv(per_episode_scores_path, index=False)
    reproduction_df.to_csv(reproduction_path, index=False)
    pd.DataFrame([asdict(item) for item in metadata_audits]).to_csv(
        metadata_audit_path, index=False
    )

    figure_paths = save_figures(
        threshold_runs=threshold_runs,
        threshold_summary=threshold_summary,
        assignment_shift=assignment_shift,
        output_dir=output_dir,
    )

    artifact_paths = {
        **required_paths,
        "frozen_memberships.csv": memberships_path,
        "threshold_run_details.csv": threshold_runs_path,
        "per_episode_scores.csv": per_episode_scores_path,
        "frozen_result_reproduction.csv": reproduction_path,
        "context_metadata_audit.csv": metadata_audit_path,
    }
    artifact_hashes = {
        name: sha256_file(path) for name, path in artifact_paths.items()
    }

    recurring_episode_counts = {
        str(key): int(value)
        for key, value in Counter(
            threshold_runs["representative_threshold_episode_id"].astype(int)
        ).most_common()
    }
    audit_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_status": "diagnostic_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "project_root": str(PROJECT_ROOT),
        "git_commit": git_commit(PROJECT_ROOT),
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "global_seed": GLOBAL_SEED,
            "seeds": list(SEEDS),
            "commissioning_grid": list(COMMISSIONING_GRID),
            "maximum_commissioning_size": MAXIMUM_COMMISSIONING_SIZE,
            "calibration_size": CALIBRATION_SIZE,
            "normal_evaluation_size": NORMAL_EVALUATION_SIZE,
            "false_alert_budget": FALSE_ALERT_BUDGET,
            "recall_target": RECALL_TARGET,
            "signal_set": SIGNAL_SET,
            "variance_threshold": VARIANCE_THRESHOLD,
            "detectors": list(DETECTOR_ORDER),
            "calibration_rule": "finite-sample split conformal order statistic",
            "prediction_rule": "score > threshold",
            "absolute_tolerance": ABS_TOLERANCE,
            "relative_tolerance": REL_TOLERANCE,
        },
        "dataset": {
            "path": str(data_path),
            "sha256": None if args.skip_dataset_hash else sha256_file(data_path),
            "hash_skipped": bool(args.skip_dataset_hash),
            "episode_count": len(cycles),
            "measured_signal_count": len(signal_columns),
            "measured_signals": list(signal_columns),
            "raw_statistical_feature_count": int(raw_features.shape[1]),
            "statistics": list(STATISTIC_NAMES),
        },
        "frozen_results": {
            "path": str(frozen_path),
            "sha256": sha256_file(frozen_path),
            "row_count": len(frozen),
            "all_reconstructed_runs_passed": bool(reproduction_df["passed"].all()),
        },
        "memberships": {
            "membership_hash_by_seed": membership_hashes,
            "pairwise_overlap_counts_by_seed": leakage_audit,
            "all_overlap_counts_zero": all(
                value == 0
                for seed_audit in leakage_audit.values()
                for value in seed_audit.values()
            ),
            "nested_commissioning_prefixes_verified": True,
        },
        "semantic_context": {
            "candidate_fields": list(SEMANTIC_CONTEXT_CANDIDATES),
            "admitted_fields": admitted_context_fields,
            "status": "available" if admitted_context_fields else "not_established",
            "audits": [asdict(item) for item in metadata_audits],
            "prohibited_fields": ["anomaly", "category"],
            "interpretation_warning": (
                "Admitted metadata values are diagnostic group labels only. Dataset "
                "documentation must support any semantic names or causal interpretation."
            ),
        },
        "exploratory_context": {
            "enabled": bool(args.exploratory_clusters),
            "status": "exploratory_sensitivity_only" if args.exploratory_clusters else "disabled",
            "algorithm": "KMeans" if args.exploratory_clusters else None,
            "cluster_count": EXPLORATORY_CLUSTER_COUNT if args.exploratory_clusters else None,
            "random_state": GLOBAL_SEED if args.exploratory_clusters else None,
            "routing_statistics": list(ROUTING_STATISTICS) if args.exploratory_clusters else [],
            "routing_feature_count": len(routing_names),
            "routing_feature_names": list(routing_names),
            "fit_population": (
                "union of target commissioning pools across frozen seeds; healthy-only"
                if args.exploratory_clusters
                else None
            ),
            "fit_episode_ids": cluster_reference_ids,
            "fit_episode_ids_sha256": sha256_json(cluster_reference_ids)
            if args.exploratory_clusters
            else None,
            "not_for_method_selection": True,
        },
        "threshold_diagnostics": {
            "run_count": len(threshold_runs),
            "recurring_representative_threshold_episode_counts": recurring_episode_counts,
            "ties_reported_explicitly": True,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": artifact_hashes[name]}
            for name, path in artifact_paths.items()
        },
        "figures": figure_paths,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "limitations": [
            "This analysis is diagnostic and does not establish a successful method.",
            "No detector or context hyperparameter may be selected from these outputs.",
            "Semantic context admission checks structural validity, not causal meaning.",
            "Exploratory clusters, when enabled, are not semantic operating regimes.",
            "Anomaly context shifts are routing diagnostics, not evidence of fault cause.",
        ],
    }
    audit_path = output_dir / "audit_manifest.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_manifest, handle, indent=2, sort_keys=True, default=json_default)

    print("\n=== Analysis complete ===")
    print(f"All {len(reproduction_df)} reconstructed frozen runs matched.")
    print(f"Semantic context status: {audit_manifest['semantic_context']['status']}")
    print(f"Artifacts written to: {output_dir}")
    print(f"Audit manifest: {audit_path}")


if __name__ == "__main__":
    run(parse_args())