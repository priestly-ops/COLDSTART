#!/usr/bin/env python3
"""
experiments/run_pakct_aursad.py

Run PAKCT: Phase-Aligned k-NN Conformal Test on AURSAD.

PAKCT operates on raw multivariate execution trajectories rather than the
cached execution-level statistical features.

For each commissioning size and seed:

1. Load complete healthy commissioning executions.
2. Fit per-channel standardization using commissioning samples only.
3. Use the first commissioning execution as the phase reference.
4. Align every execution to the reference with multivariate FastDTW.
5. Aggregate the DTW path onto the reference phase.
6. Resample the aligned trajectory to a fixed number of phase bins.
7. Flatten the phase-aligned trajectory.
8. Score samples using Euclidean distance to the k-th nearest aligned
   commissioning execution.
9. Select a finite-sample split-conformal threshold on the fixed healthy
   calibration set.
10. Evaluate on the fixed healthy and anomaly evaluation sets.

Leakage safety
--------------
- Channel scaling is fitted on commissioning data only.
- The phase reference is selected from commissioning data only.
- Calibration is used only for threshold selection.
- Evaluation data never affects preprocessing, alignment reference,
  k-NN fitting, or threshold construction.
- Protocol memberships remain at complete sample_nr execution level.

Computational note
------------------
FastDTW alignment is substantially more expensive than the feature-space
baselines. This runner caches aligned vectors separately for every seed/N run
and saves CSV checkpoints after each completed run.

Default parameters
------------------
k = 10
FastDTW radius = 2
phase bins = 128
distance channels = all frozen 48 measured channels

Smoke test
----------
    .\\.venv\\Scripts\\python.exe experiments\\run_pakct_aursad.py ^
        --grid 10 ^
        --seeds 0 ^
        --calibration-limit 50 ^
        --healthy-eval-limit 50 ^
        --anomaly-eval-limit 100 ^
        --phase-bins 64 ^
        --bootstrap-samples 1000 ^
        --overwrite

Full protocol
-------------
    .\\.venv\\Scripts\\python.exe experiments\\run_pakct_aursad.py ^
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aursad_loader import (
    RobotCycle,
    load_executions,
)
from src.reproducibility import reproducibility_metadata

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aursad"
    / "AURSAD.h5"
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
    / "pakct"
)

DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))

DEFAULT_K = 10
DEFAULT_DTW_RADIUS = 2
DEFAULT_PHASE_BINS = 128
DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_GLOBAL_SEED = 42
DEFAULT_EPSILON = 1e-12

PROTOCOL_VERSION = "aursad-pakct-fastdtw-v1"
DETECTOR_NAME = "PAKCT"

SEED_RESULT_COLUMNS = (
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "commissioning_count",
    "calibration_count",
    "healthy_eval_count",
    "anomaly_eval_count",
    "signal_count",
    "phase_bins",
    "aligned_feature_count",
    "requested_k",
    "effective_k",
    "dtw_radius",
    "conformal_rank",
    "threshold",
    "false_positive_rate",
    "recall",
    "auroc",
    "success",
    "alignment_seconds",
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


@dataclass(frozen=True)
class ChannelStandardizer:
    """Training-only per-channel standardization for raw trajectories."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        cycles: Sequence[RobotCycle],
        epsilon: float = DEFAULT_EPSILON,
    ) -> "ChannelStandardizer":
        if not cycles:
            raise ValueError(
                "At least one commissioning cycle is required."
            )

        expected_columns = tuple(
            cycles[0].columns
        )

        sums = np.zeros(
            len(expected_columns),
            dtype=np.float64,
        )

        sums_squared = np.zeros_like(
            sums
        )

        sample_count = 0

        for cycle in cycles:
            if tuple(cycle.columns) != expected_columns:
                raise ValueError(
                    f"Execution {cycle.episode_id} has a different "
                    "signal schema."
                )

            values = np.asarray(
                cycle.values,
                dtype=np.float64,
            )

            if values.ndim != 2:
                raise ValueError(
                    f"Execution {cycle.episode_id} must be 2D."
                )

            if len(values) < 2:
                raise ValueError(
                    f"Execution {cycle.episode_id} has fewer than "
                    "two time steps."
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Execution {cycle.episode_id} contains NaN or Inf."
                )

            sums += values.sum(
                axis=0
            )

            sums_squared += np.square(
                values
            ).sum(
                axis=0
            )

            sample_count += len(values)

        mean = sums / sample_count

        variance = np.maximum(
            sums_squared / sample_count
            - np.square(mean),
            0.0,
        )

        scale = np.sqrt(
            variance
        )

        scale = np.where(
            scale > epsilon,
            scale,
            1.0,
        )

        return cls(
            mean=mean,
            scale=scale,
        )

    def transform(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        transformed = (
            values - self.mean
        ) / self.scale

        if not np.isfinite(
            transformed
        ).all():
            raise RuntimeError(
                "Channel standardization produced NaN or Inf."
            )

        return transformed


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


def unique_ids(
    frame: pd.DataFrame,
) -> np.ndarray:
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


def limit_ids(
    ids: np.ndarray,
    limit: int | None,
) -> np.ndarray:
    if limit is None:
        return ids

    if limit <= 0:
        raise ValueError(
            "Protocol limits must be positive."
        )

    return ids[
        : min(
            limit,
            len(ids),
        )
    ]


def align_to_reference(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """
    Align one standardized trajectory to the standardized reference phase.

    The returned array has exactly the reference length. Multiple source
    samples mapped to one reference index are averaged. Empty reference bins
    are filled by linear interpolation along phase.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    _, path = fastdtw(
        reference,
        values,
        radius=radius,
        dist=euclidean,
    )

    reference_length = len(reference)
    signal_count = reference.shape[1]

    sums = np.zeros(
        (
            reference_length,
            signal_count,
        ),
        dtype=np.float64,
    )

    counts = np.zeros(
        reference_length,
        dtype=np.int64,
    )

    for reference_index, value_index in path:
        sums[
            reference_index
        ] += values[
            value_index
        ]

        counts[
            reference_index
        ] += 1

    aligned = np.empty_like(
        sums
    )

    populated = counts > 0

    aligned[
        populated
    ] = (
        sums[
            populated
        ]
        / counts[
            populated,
            None,
        ]
    )

    missing_indices = np.flatnonzero(
        ~populated
    )

    if len(missing_indices):
        known_indices = np.flatnonzero(
            populated
        )

        if len(known_indices) == 0:
            raise RuntimeError(
                "DTW path did not populate any reference phase."
            )

        for channel_index in range(
            signal_count
        ):
            aligned[
                missing_indices,
                channel_index,
            ] = np.interp(
                missing_indices,
                known_indices,
                aligned[
                    known_indices,
                    channel_index,
                ],
            )

    if not np.isfinite(
        aligned
    ).all():
        raise RuntimeError(
            "DTW alignment produced NaN or Inf."
        )

    return aligned


def resample_phase(
    aligned: np.ndarray,
    phase_bins: int,
) -> np.ndarray:
    """Linearly resample an aligned trajectory to fixed phase bins."""
    if phase_bins < 2:
        raise ValueError(
            "phase_bins must be at least 2."
        )

    source_phase = np.linspace(
        0.0,
        1.0,
        num=len(aligned),
        dtype=np.float64,
    )

    target_phase = np.linspace(
        0.0,
        1.0,
        num=phase_bins,
        dtype=np.float64,
    )

    output = np.empty(
        (
            phase_bins,
            aligned.shape[1],
        ),
        dtype=np.float64,
    )

    for channel_index in range(
        aligned.shape[1]
    ):
        output[
            :,
            channel_index,
        ] = np.interp(
            target_phase,
            source_phase,
            aligned[
                :,
                channel_index,
            ],
        )

    if not np.isfinite(
        output
    ).all():
        raise RuntimeError(
            "Phase resampling produced NaN or Inf."
        )

    return output


def aligned_vector(
    cycle: RobotCycle,
    *,
    standardizer: ChannelStandardizer,
    reference_standardized: np.ndarray,
    radius: int,
    phase_bins: int,
) -> np.ndarray:
    standardized = standardizer.transform(
        cycle.values
    )

    aligned = align_to_reference(
        standardized,
        reference_standardized,
        radius=radius,
    )

    phase_resampled = resample_phase(
        aligned,
        phase_bins,
    )

    vector = phase_resampled.reshape(
        -1
    )

    if not np.isfinite(
        vector
    ).all():
        raise RuntimeError(
            f"Execution {cycle.episode_id} produced non-finite "
            "aligned features."
        )

    return vector


def load_or_build_aligned_matrix(
    cycles: Sequence[RobotCycle],
    *,
    cache_path: Path,
    standardizer: ChannelStandardizer,
    reference_standardized: np.ndarray,
    reference_id: int,
    radius: int,
    phase_bins: int,
    overwrite: bool,
) -> np.ndarray:
    """Load an aligned matrix cache or construct it deterministically."""
    expected_ids = np.asarray(
        [
            int(cycle.episode_id)
            for cycle in cycles
        ],
        dtype=np.int64,
    )

    if cache_path.exists() and not overwrite:
        with np.load(
            cache_path,
            allow_pickle=False,
        ) as archive:
            cached_ids = np.asarray(
                archive["episode_ids"],
                dtype=np.int64,
            )

            matrix = np.asarray(
                archive["features"],
                dtype=np.float64,
            )

            metadata = json.loads(
                str(
                    archive[
                        "metadata_json"
                    ].item()
                )
            )

        expected_metadata = {
            "reference_id": int(
                reference_id
            ),
            "radius": int(
                radius
            ),
            "phase_bins": int(
                phase_bins
            ),
            "signal_count": int(
                reference_standardized.shape[1]
            ),
        }

        if metadata != expected_metadata:
            raise ValueError(
                f"Aligned cache metadata mismatch at {cache_path}. "
                f"Found {metadata}, expected {expected_metadata}."
            )

        if not np.array_equal(
            cached_ids,
            expected_ids,
        ):
            raise ValueError(
                f"Aligned cache episode IDs do not match at "
                f"{cache_path}."
            )

        expected_shape = (
            len(cycles),
            phase_bins
            * reference_standardized.shape[1],
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Aligned cache shape {matrix.shape} does not match "
                f"{expected_shape}."
            )

        return matrix

    rows = []

    for index, cycle in enumerate(
        cycles,
        start=1,
    ):
        print(
            f"    aligning {index}/{len(cycles)} "
            f"sample_nr={cycle.episode_id}"
        )

        rows.append(
            aligned_vector(
                cycle,
                standardizer=standardizer,
                reference_standardized=(
                    reference_standardized
                ),
                radius=radius,
                phase_bins=phase_bins,
            )
        )

    matrix = np.vstack(
        rows
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "reference_id": int(
            reference_id
        ),
        "radius": int(
            radius
        ),
        "phase_bins": int(
            phase_bins
        ),
        "signal_count": int(
            reference_standardized.shape[1]
        ),
    }

    np.savez_compressed(
        cache_path,
        episode_ids=expected_ids,
        features=matrix,
        metadata_json=np.asarray(
            json.dumps(
                metadata,
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    )

    return matrix


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

    ordered = np.sort(
        scores
    )

    return (
        float(
            ordered[
                rank - 1
            ]
        ),
        rank,
    )


def kth_neighbor_scores(
    model: NearestNeighbors,
    features: np.ndarray,
    effective_k: int,
) -> np.ndarray:
    distances, _ = model.kneighbors(
        features,
        n_neighbors=effective_k,
        return_distance=True,
    )

    scores = distances[
        :,
        effective_k - 1,
    ]

    if not np.isfinite(
        scores
    ).all():
        raise RuntimeError(
            "k-NN produced NaN or Inf scores."
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def run_one(
    *,
    data_path: Path,
    aligned_cache_dir: Path,
    commissioning_ids: np.ndarray,
    calibration_ids: np.ndarray,
    healthy_eval_ids: np.ndarray,
    anomaly_eval_ids: np.ndarray,
    anomaly_eval_table: pd.DataFrame,
    commissioning_size: int,
    seed: int,
    requested_k: int,
    radius: int,
    phase_bins: int,
    false_alert_budget: float,
    recall_target: float,
    overwrite_aligned_cache: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    all_ids = np.concatenate(
        (
            commissioning_ids,
            calibration_ids,
            healthy_eval_ids,
            anomaly_eval_ids,
        )
    )

    if len(set(
        all_ids.tolist()
    )) != len(all_ids):
        raise RuntimeError(
            "Run partitions overlap at execution level."
        )

    cycles = load_executions(
        episode_ids=all_ids.tolist(),
        data_path=data_path,
    )

    by_id = {
        int(cycle.episode_id): cycle
        for cycle in cycles
    }

    commissioning_cycles = [
        by_id[
            int(episode_id)
        ]
        for episode_id in commissioning_ids
    ]

    calibration_cycles = [
        by_id[
            int(episode_id)
        ]
        for episode_id in calibration_ids
    ]

    healthy_eval_cycles = [
        by_id[
            int(episode_id)
        ]
        for episode_id in healthy_eval_ids
    ]

    anomaly_eval_cycles = [
        by_id[
            int(episode_id)
        ]
        for episode_id in anomaly_eval_ids
    ]

    for cycle in commissioning_cycles:
        if cycle.anomaly:
            raise RuntimeError(
                "Commissioning contains an anomalous execution."
            )

    for cycle in calibration_cycles:
        if cycle.anomaly:
            raise RuntimeError(
                "Calibration contains an anomalous execution."
            )

    for cycle in healthy_eval_cycles:
        if cycle.anomaly:
            raise RuntimeError(
                "Healthy evaluation contains an anomaly."
            )

    for cycle in anomaly_eval_cycles:
        if not cycle.anomaly:
            raise RuntimeError(
                "Anomaly evaluation contains a healthy execution."
            )

    standardizer = ChannelStandardizer.fit(
        commissioning_cycles
    )

    reference_cycle = commissioning_cycles[
        0
    ]

    reference_standardized = (
        standardizer.transform(
            reference_cycle.values
        )
    )

    run_cache_dir = (
        aligned_cache_dir
        / f"N_{commissioning_size}"
        / f"seed_{seed:02d}"
    )

    alignment_started = time.perf_counter()

    commissioning_matrix = (
        load_or_build_aligned_matrix(
            commissioning_cycles,
            cache_path=(
                run_cache_dir
                / "commissioning.npz"
            ),
            standardizer=standardizer,
            reference_standardized=(
                reference_standardized
            ),
            reference_id=int(
                reference_cycle.episode_id
            ),
            radius=radius,
            phase_bins=phase_bins,
            overwrite=overwrite_aligned_cache,
        )
    )

    calibration_matrix = (
        load_or_build_aligned_matrix(
            calibration_cycles,
            cache_path=(
                run_cache_dir
                / "calibration.npz"
            ),
            standardizer=standardizer,
            reference_standardized=(
                reference_standardized
            ),
            reference_id=int(
                reference_cycle.episode_id
            ),
            radius=radius,
            phase_bins=phase_bins,
            overwrite=overwrite_aligned_cache,
        )
    )

    healthy_eval_matrix = (
        load_or_build_aligned_matrix(
            healthy_eval_cycles,
            cache_path=(
                run_cache_dir
                / "healthy_eval.npz"
            ),
            standardizer=standardizer,
            reference_standardized=(
                reference_standardized
            ),
            reference_id=int(
                reference_cycle.episode_id
            ),
            radius=radius,
            phase_bins=phase_bins,
            overwrite=overwrite_aligned_cache,
        )
    )

    anomaly_eval_matrix = (
        load_or_build_aligned_matrix(
            anomaly_eval_cycles,
            cache_path=(
                run_cache_dir
                / "anomaly_eval.npz"
            ),
            standardizer=standardizer,
            reference_standardized=(
                reference_standardized
            ),
            reference_id=int(
                reference_cycle.episode_id
            ),
            radius=radius,
            phase_bins=phase_bins,
            overwrite=overwrite_aligned_cache,
        )
    )

    alignment_seconds = (
        time.perf_counter()
        - alignment_started
    )

    effective_k = min(
        requested_k,
        len(commissioning_matrix),
    )

    model = NearestNeighbors(
        n_neighbors=effective_k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    model.fit(
        commissioning_matrix
    )

    calibration_scores = kth_neighbor_scores(
        model,
        calibration_matrix,
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
        healthy_eval_matrix,
        effective_k,
    )

    anomaly_scores = kth_neighbor_scores(
        model,
        anomaly_eval_matrix,
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
        "signal_count": int(
            reference_standardized.shape[1]
        ),
        "phase_bins": int(
            phase_bins
        ),
        "aligned_feature_count": int(
            commissioning_matrix.shape[1]
        ),
        "requested_k": int(
            requested_k
        ),
        "effective_k": int(
            effective_k
        ),
        "dtw_radius": int(
            radius
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
        "alignment_seconds": float(
            alignment_seconds
        ),
    }

    prediction_by_id = {
        int(episode_id): bool(prediction)
        for episode_id, prediction
        in zip(
            anomaly_eval_ids,
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

        class_ids = [
            episode_id
            for episode_id in class_ids
            if episode_id
            in prediction_by_id
        ]

        if not class_ids:
            continue

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

    rng = np.random.default_rng(
        seed
    )

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
        recall_values = group[
            "recall"
        ].to_numpy(
            dtype=np.float64
        )

        fpr_values = group[
            "false_positive_rate"
        ].to_numpy(
            dtype=np.float64
        )

        auroc_values = group[
            "auroc"
        ].to_numpy(
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
                "success_rate": float(
                    group[
                        "success"
                    ].astype(bool).mean()
                ),
                "mean_alignment_seconds": float(
                    group[
                        "alignment_seconds"
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
        values = group[
            "recall"
        ].to_numpy(
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
                "recall_ci_lower": (
                    lower
                ),
                "recall_ci_upper": (
                    upper
                ),
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

    frame = pd.read_csv(
        path
    )

    missing = sorted(
        set(columns)
        - set(frame.columns)
    )

    if missing:
        raise ValueError(
            f"Existing checkpoint {path} is incompatible. "
            f"Missing columns: {missing}"
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
            int(row[
                "commissioning_size"
            ]),
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
    ).reset_index(
        drop=True
    )

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
            "Run PAKCT on the frozen AURSAD protocol."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
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
        "--dtw-radius",
        type=int,
        default=DEFAULT_DTW_RADIUS,
    )

    parser.add_argument(
        "--phase-bins",
        type=int,
        default=DEFAULT_PHASE_BINS,
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
        "--calibration-limit",
        type=int,
        default=None,
        help=(
            "Smoke-test only: use the first K calibration IDs."
        ),
    )

    parser.add_argument(
        "--healthy-eval-limit",
        type=int,
        default=None,
        help=(
            "Smoke-test only: use the first K healthy evaluation IDs."
        ),
    )

    parser.add_argument(
        "--anomaly-eval-limit",
        type=int,
        default=None,
        help=(
            "Smoke-test only: use the first K anomaly evaluation IDs."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Restart result CSVs and rebuild aligned caches."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = (
        args.data_path
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

    if args.dtw_radius < 0:
        raise ValueError(
            "--dtw-radius cannot be negative."
        )

    if args.phase_bins < 2:
        raise ValueError(
            "--phase-bins must be at least 2."
        )

    if not 0.0 < args.false_alert_budget < 1.0:
        raise ValueError(
            "--false-alert-budget must be between 0 and 1."
        )

    require_file(
        data_path,
        "AURSAD raw HDF5 file",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned_cache_dir = (
        output_dir
        / "aligned_cache"
    )

    seed_results_path = (
        output_dir
        / "pakct_seed_results.csv"
    )

    per_class_seed_path = (
        output_dir
        / "pakct_per_class_seed_results.csv"
    )

    summary_path = (
        output_dir
        / "pakct_summary.csv"
    )

    per_class_summary_path = (
        output_dir
        / "pakct_per_class_recall.csv"
    )

    n_star_path = (
        output_dir
        / "pakct_n_star.json"
    )

    manifest_path = (
        output_dir
        / "pakct_run_manifest.json"
    )

    protocol = load_protocol_tables(
        protocol_dir
    )

    validate_protocol(
        protocol,
        grid=grid,
        seeds=seeds,
    )

    calibration_ids = limit_ids(
        unique_ids(
            protocol["calibration"]
        ),
        args.calibration_limit,
    )

    healthy_eval_ids = limit_ids(
        unique_ids(
            protocol["healthy_eval"]
        ),
        args.healthy_eval_limit,
    )

    anomaly_eval_ids = limit_ids(
        unique_ids(
            protocol["anomaly_eval"]
        ),
        args.anomaly_eval_limit,
    )

    smoke_limited = any(
        value is not None
        for value in (
            args.calibration_limit,
            args.healthy_eval_limit,
            args.anomaly_eval_limit,
        )
    )

    print("=" * 76)
    print("AURSAD PAKCT")
    print("=" * 76)
    print(f"Raw data:     {data_path}")
    print(f"Protocol:     {protocol_dir}")
    print(f"Output:       {output_dir}")
    print(f"Grid:         {list(grid)}")
    print(f"Seeds:        {list(seeds)}")
    print(f"k:            {args.k}")
    print(f"DTW radius:   {args.dtw_radius}")
    print(f"Phase bins:   {args.phase_bins}")
    print(
        f"Calibration:  {len(calibration_ids)}"
    )
    print(
        f"Healthy eval: {len(healthy_eval_ids)}"
    )
    print(
        f"Anomaly eval: {len(anomaly_eval_ids)}"
    )

    if smoke_limited:
        print(
            "WARNING: protocol limits are active; results are "
            "smoke-test results only."
        )

    started = time.perf_counter()

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

    total_runs = (
        len(grid)
        * len(seeds)
    )

    run_number = len(done)

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
                f"\nProcessing N={n_value} seed={seed} "
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
                data_path=data_path,
                aligned_cache_dir=(
                    aligned_cache_dir
                ),
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
                radius=int(
                    args.dtw_radius
                ),
                phase_bins=int(
                    args.phase_bins
                ),
                false_alert_budget=float(
                    args.false_alert_budget
                ),
                recall_target=float(
                    args.recall_target
                ),
                overwrite_aligned_cache=bool(
                    args.overwrite
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
                f"recall={result['recall']:.4f} "
                f"FPR={result['false_positive_rate']:.4f} "
                f"AUROC={result['auroc']:.4f} "
                f"success={result['success']} "
                f"alignment={result['alignment_seconds']:.1f}s"
            )

    results = pd.read_csv(
        seed_results_path
    )

    per_class_seed = pd.read_csv(
        per_class_seed_path
    )

    summary = build_summary(
        results,
        false_alert_budget=float(
            args.false_alert_budget
        ),
        recall_target=float(
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

    n_star_path.write_text(
        json.dumps(
            {
                "protocol_version": (
                    PROTOCOL_VERSION
                ),
                "detector": DETECTOR_NAME,
                "smoke_limited": (
                    smoke_limited
                ),
                "estimate": n_star,
            },
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
            "raw_signal_count": 48,
            "phase_reference": (
                "first commissioning execution"
            ),
            "channel_scaling": (
                "commissioning-only per-channel standardization"
            ),
            "alignment": "multivariate FastDTW",
            "dtw_radius": int(
                args.dtw_radius
            ),
            "phase_bins": int(
                args.phase_bins
            ),
            "requested_k": int(
                args.k
            ),
            "metric": (
                "Euclidean distance on flattened phase-aligned "
                "trajectories"
            ),
            "calibration": (
                "finite-sample split-conformal on fixed healthy "
                "calibration executions"
            ),
        },
        "protocol": {
            "grid": list(
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
            "calibration_count": int(
                len(calibration_ids)
            ),
            "healthy_eval_count": int(
                len(healthy_eval_ids)
            ),
            "anomaly_eval_count": int(
                len(anomaly_eval_ids)
            ),
            "smoke_limited": bool(
                smoke_limited
            ),
        },
        "input": {
            "raw_data_path": str(
                data_path
            ),
            "raw_data_sha256": sha256_file(
                data_path
            ),
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
            "aligned_cache_dir": str(
                aligned_cache_dir
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
            "complete_execution_membership": True,
            "zero_partition_overlap": True,
            "normal_only_commissioning": True,
            "commissioning_only_scaling": True,
            "commissioning_only_reference": True,
            "fixed_split_conformal_calibration": True,
            "evaluation_not_used_for_fitting": True,
        },
        "limitations": [
            (
                "FastDTW is an approximate DTW algorithm."
            ),
            (
                "All 48 channels contribute equally after "
                "commissioning-only standardization."
            ),
            (
                "Aligned trajectories are resampled to a fixed phase "
                "grid before Euclidean k-NN scoring."
            ),
            (
                "Damaged-thread label 4 contains only three "
                "evaluation executions."
            ),
        ],
    }
    manifest.update(
        reproducibility_metadata(
            repo_root=PROJECT_ROOT,
            input_paths={
                "raw_data": data_path,
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
    print("PAKCT COMPLETE")
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
                "mean_alignment_seconds",
                "meets_joint_ci_criterion",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nEstimated N*: {n_star['display']}"
    )

    if smoke_limited:
        print(
            "This was a limited smoke test. Do not use its N* "
            "as a paper result."
        )

    print("\nArtifacts:")
    print(f"  {seed_results_path}")
    print(f"  {summary_path}")
    print(f"  {per_class_seed_path}")
    print(f"  {per_class_summary_path}")
    print(f"  {n_star_path}")
    print(f"  {manifest_path}")
    print(f"  {aligned_cache_dir}")


if __name__ == "__main__":
    main()
