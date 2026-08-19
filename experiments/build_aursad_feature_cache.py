#!/usr/bin/env python3
"""
experiments/build_aursad_feature_cache.py

Build a single deterministic statistical-feature cache for every AURSAD
execution used by the frozen commissioning protocol.

The script:

1. Reads the frozen protocol membership CSV.
2. Collects every unique active sample_nr exactly once.
3. Excludes protocol rows marked as "excluded".
4. Loads complete executions from AURSAD.h5 with the measured-signal policy.
5. Extracts the shared six-statistic feature representation.
6. Saves a compressed NPZ FeatureBatch.
7. Saves a human-readable metadata JSON and episode inventory CSV.
8. Reopens the cache and verifies exact round-trip consistency.

The resulting cache can be subset by episode ID for all commissioning seeds,
calibration, healthy evaluation, anomaly evaluation, and detector runs without
reopening the roughly 6 GB raw HDF5 file.

Expected feature width
----------------------
48 measured signals * 6 statistics = 288 features.

Example
-------
From the repository root:

    .\\.venv\\Scripts\\python.exe experiments\\build_aursad_feature_cache.py

To replace an existing cache:

    .\\.venv\\Scripts\\python.exe experiments\\build_aursad_feature_cache.py ^
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aursad_loader import (
    load_executions,
)
from src.feature_extractor import (
    FEATURE_CACHE_VERSION,
    STATISTIC_NAMES,
    extract_feature_batch,
    load_feature_batch,
    save_feature_batch,
)
from src.reproducibility import reproducibility_metadata


DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aursad"
    / "AURSAD.h5"
)

DEFAULT_INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
    / "aursad_episode_inventory.csv"
)

DEFAULT_MEMBERSHIP_PATH = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
    / "protocol"
    / "aursad_protocol_membership.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "aursad"
    / "feature_cache"
)

DEFAULT_CACHE_NAME = "aursad_features.npz"
DEFAULT_METADATA_NAME = "feature_metadata.json"
DEFAULT_EPISODE_INDEX_NAME = "feature_episode_index.csv"

ACTIVE_PARTITIONS = frozenset(
    {
        "commissioning",
        "calibration",
        "healthy_eval",
        "anomaly_eval",
    }
)

EXPECTED_SIGNAL_COUNT = 48
EXPECTED_FEATURE_COUNT = (
    EXPECTED_SIGNAL_COUNT
    * len(STATISTIC_NAMES)
)

REQUIRED_MEMBERSHIP_COLUMNS = (
    "sample_nr",
    "partition",
    "label",
    "label_name",
)

REQUIRED_INVENTORY_COLUMNS = (
    "sample_nr",
    "label",
    "label_name",
    "row_count",
)


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Compute a streaming SHA-256 digest."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(
    value: Any,
) -> Any:
    """Convert common NumPy and Pandas values into JSON-safe objects."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def require_file(
    path: Path,
    description: str,
) -> None:
    """Raise a clear error when a required input is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{description} is not a regular file: {path}"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    description: str,
) -> None:
    """Validate required DataFrame columns."""
    required = tuple(columns)

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{description} is missing required columns: {missing}"
        )


def load_protocol_membership(
    path: Path,
) -> pd.DataFrame:
    """Load and validate the frozen protocol membership table."""
    membership = pd.read_csv(path)

    require_columns(
        membership,
        REQUIRED_MEMBERSHIP_COLUMNS,
        "Protocol membership CSV",
    )

    membership = membership.copy()

    membership["sample_nr"] = pd.to_numeric(
        membership["sample_nr"],
        errors="raise",
    ).astype(np.int64)

    membership["label"] = pd.to_numeric(
        membership["label"],
        errors="raise",
    ).astype(np.int64)

    membership["partition"] = (
        membership["partition"]
        .astype(str)
        .str.strip()
    )

    unknown_partitions = sorted(
        set(
            membership["partition"].unique()
        )
        - ACTIVE_PARTITIONS
        - {"excluded"}
    )

    if unknown_partitions:
        raise ValueError(
            "Protocol membership contains unknown partitions: "
            f"{unknown_partitions}"
        )

    active = membership[
        membership["partition"].isin(
            ACTIVE_PARTITIONS
        )
    ].copy()

    if active.empty:
        raise ValueError(
            "The protocol contains no active executions."
        )

    label_counts_per_id = (
        active.groupby(
            "sample_nr",
            sort=False,
        )["label"]
        .nunique()
    )

    conflicting_ids = (
        label_counts_per_id[
            label_counts_per_id > 1
        ]
        .index
        .astype(int)
        .tolist()
    )

    if conflicting_ids:
        raise ValueError(
            "Some active sample_nr values have conflicting labels: "
            f"{conflicting_ids[:20]}"
        )

    return membership


def active_episode_table(
    membership: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one deterministic record per unique active execution.

    Commissioning IDs appear repeatedly across N and seeds in the membership
    table. They are deduplicated here so each raw execution is loaded and
    featurized exactly once.
    """
    active = membership[
        membership["partition"].isin(
            ACTIVE_PARTITIONS
        )
    ].copy()

    partition_lists = (
        active.groupby(
            "sample_nr",
            sort=True,
        )["partition"]
        .agg(
            lambda values: ",".join(
                sorted(
                    set(
                        str(value)
                        for value in values
                    )
                )
            )
        )
        .rename(
            "active_partitions"
        )
        .reset_index()
    )

    protocol_metadata = (
        active.sort_values(
            [
                "sample_nr",
                "partition",
            ]
        )
        .drop_duplicates(
            subset=["sample_nr"],
            keep="first",
        )[
            [
                "sample_nr",
                "label",
                "label_name",
            ]
        ]
    )

    unique_active = protocol_metadata.merge(
        partition_lists,
        on="sample_nr",
        how="left",
        validate="one_to_one",
    )

    require_columns(
        inventory,
        REQUIRED_INVENTORY_COLUMNS,
        "Episode inventory CSV",
    )

    inventory = inventory.copy()

    inventory["sample_nr"] = pd.to_numeric(
        inventory["sample_nr"],
        errors="raise",
    ).astype(np.int64)

    inventory["label"] = pd.to_numeric(
        inventory["label"],
        errors="raise",
    ).astype(np.int64)

    inventory["row_count"] = pd.to_numeric(
        inventory["row_count"],
        errors="raise",
    ).astype(np.int64)

    if inventory["sample_nr"].duplicated().any():
        duplicated = (
            inventory.loc[
                inventory["sample_nr"].duplicated(
                    keep=False
                ),
                "sample_nr",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "Episode inventory contains duplicate sample_nr values: "
            f"{duplicated[:20]}"
        )

    inventory_columns = [
        column
        for column in (
            "sample_nr",
            "label",
            "label_name",
            "row_count",
            "first_global_row",
            "last_global_row",
            "duration_seconds",
            "estimated_sampling_rate_hz",
            "timestamps_monotonic_nondecreasing",
        )
        if column in inventory.columns
    ]

    merged = unique_active.merge(
        inventory[
            inventory_columns
        ],
        on="sample_nr",
        how="left",
        suffixes=(
            "_protocol",
            "_inventory",
        ),
        validate="one_to_one",
        indicator=True,
    )

    missing_inventory = (
        merged.loc[
            merged["_merge"].ne("both"),
            "sample_nr",
        ]
        .astype(int)
        .tolist()
    )

    if missing_inventory:
        raise ValueError(
            "Active protocol executions are missing from the episode "
            f"inventory: {missing_inventory[:20]}"
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    if (
        "label_protocol" in merged.columns
        and "label_inventory" in merged.columns
    ):
        disagreement = merged[
            merged["label_protocol"].ne(
                merged["label_inventory"]
            )
        ]

        if not disagreement.empty:
            raise ValueError(
                "Protocol and inventory labels disagree for sample_nr "
                f"values: "
                f"{disagreement['sample_nr'].astype(int).tolist()[:20]}"
            )

        merged["label"] = (
            merged["label_protocol"]
            .astype(np.int64)
        )

        merged = merged.drop(
            columns=[
                "label_protocol",
                "label_inventory",
            ]
        )

    if (
        "label_name_protocol" in merged.columns
        and "label_name_inventory" in merged.columns
    ):
        merged["label_name"] = (
            merged[
                "label_name_protocol"
            ]
            .astype(str)
        )

        merged = merged.drop(
            columns=[
                "label_name_protocol",
                "label_name_inventory",
            ]
        )

    merged = merged.sort_values(
        "sample_nr"
    ).reset_index(drop=True)

    if merged["sample_nr"].duplicated().any():
        raise RuntimeError(
            "Internal error: active episode table is not unique by "
            "sample_nr."
        )

    return merged


def ensure_output_paths(
    output_dir: Path,
    cache_name: str,
    metadata_name: str,
    episode_index_name: str,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    """Construct output paths and protect existing artifacts."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / cache_name
    )

    metadata_path = (
        output_dir
        / metadata_name
    )

    episode_index_path = (
        output_dir
        / episode_index_name
    )

    existing = [
        path
        for path in (
            cache_path,
            metadata_path,
            episode_index_path,
        )
        if path.exists()
    ]

    if existing and not overwrite:
        rendered = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "Feature-cache artifacts already exist. Use --overwrite "
            f"to replace them:\n{rendered}"
        )

    return (
        cache_path,
        metadata_path,
        episode_index_path,
    )


def validate_loaded_cycles(
    cycles: list[Any],
    expected: pd.DataFrame,
) -> None:
    """Validate loader output before feature extraction."""
    expected_ids = (
        expected["sample_nr"]
        .astype(np.int64)
        .to_numpy()
    )

    loaded_ids = np.asarray(
        [
            int(cycle.episode_id)
            for cycle in cycles
        ],
        dtype=np.int64,
    )

    if len(cycles) != len(expected_ids):
        raise RuntimeError(
            f"Loader returned {len(cycles)} executions, expected "
            f"{len(expected_ids)}."
        )

    if not np.array_equal(
        loaded_ids,
        expected_ids,
    ):
        mismatch_positions = np.flatnonzero(
            loaded_ids != expected_ids
        )

        preview = [
            {
                "position": int(index),
                "expected": int(
                    expected_ids[index]
                ),
                "loaded": int(
                    loaded_ids[index]
                ),
            }
            for index in mismatch_positions[:10]
        ]

        raise RuntimeError(
            "AURSAD loader did not preserve requested execution order. "
            f"Mismatches: {preview}"
        )

    if len(set(
        loaded_ids.tolist()
    )) != len(loaded_ids):
        raise RuntimeError(
            "AURSAD loader returned duplicate episode IDs."
        )

    schemas = {
        tuple(cycle.columns)
        for cycle in cycles
    }

    if len(schemas) != 1:
        raise RuntimeError(
            "Loaded AURSAD executions do not share one signal schema."
        )

    signal_columns = next(
        iter(schemas)
    )

    if len(signal_columns) != EXPECTED_SIGNAL_COUNT:
        raise RuntimeError(
            "Unexpected measured-signal count. "
            f"Expected {EXPECTED_SIGNAL_COUNT}, found "
            f"{len(signal_columns)}. "
            "Confirm that src/aursad_loader.py is using the frozen "
            "48-channel measured policy."
        )

    expected_labels = (
        expected.set_index(
            "sample_nr"
        )["label"]
        .astype(int)
        .to_dict()
    )

    for cycle in cycles:
        sample_nr = int(
            cycle.episode_id
        )

        expected_label = int(
            expected_labels[
                sample_nr
            ]
        )

        if int(
            cycle.category
        ) != expected_label:
            raise RuntimeError(
                f"Execution {sample_nr}: loader category "
                f"{cycle.category} does not match protocol label "
                f"{expected_label}."
            )

        expected_anomaly = (
            expected_label
            in {1, 2, 3, 4}
        )

        if bool(
            cycle.anomaly
        ) != expected_anomaly:
            raise RuntimeError(
                f"Execution {sample_nr}: loader anomaly flag "
                f"{cycle.anomaly} does not match label "
                f"{expected_label}."
            )


def validate_feature_batch(
    batch: Any,
    expected: pd.DataFrame,
) -> None:
    """Validate extracted feature dimensions and metadata."""
    expected_ids = (
        expected["sample_nr"]
        .astype(np.int64)
        .to_numpy()
    )

    if batch.features.shape != (
        len(expected_ids),
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            "Unexpected AURSAD feature-matrix shape: "
            f"{batch.features.shape}; expected "
            f"({len(expected_ids)}, {EXPECTED_FEATURE_COUNT})."
        )

    if not np.array_equal(
        batch.episode_ids,
        expected_ids,
    ):
        raise RuntimeError(
            "FeatureBatch episode ordering differs from the active "
            "episode index."
        )

    if len(
        batch.signal_columns
    ) != EXPECTED_SIGNAL_COUNT:
        raise RuntimeError(
            f"FeatureBatch has {len(batch.signal_columns)} signals; "
            f"expected {EXPECTED_SIGNAL_COUNT}."
        )

    if tuple(
        batch.statistic_names
    ) != tuple(
        STATISTIC_NAMES
    ):
        raise RuntimeError(
            "FeatureBatch statistic order differs from the frozen "
            f"order: {STATISTIC_NAMES}."
        )

    if len(
        batch.feature_names
    ) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"FeatureBatch has {len(batch.feature_names)} feature "
            f"names; expected {EXPECTED_FEATURE_COUNT}."
        )

    if not np.isfinite(
        batch.features
    ).all():
        raise RuntimeError(
            "Extracted features contain NaN or Inf."
        )

    expected_anomaly = (
        expected["label"]
        .isin(
            [1, 2, 3, 4]
        )
        .to_numpy(
            dtype=np.bool_
        )
    )

    if not np.array_equal(
        batch.anomaly_labels,
        expected_anomaly,
    ):
        raise RuntimeError(
            "FeatureBatch anomaly labels differ from protocol labels."
        )

    expected_categories = (
        expected["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        batch.categories,
        expected_categories,
    ):
        raise RuntimeError(
            "FeatureBatch categories differ from protocol labels."
        )


def build_episode_index(
    active: pd.DataFrame,
    batch: Any,
) -> pd.DataFrame:
    """Create a human-readable row index aligned with the NPZ cache."""
    row_by_id = {
        int(episode_id): int(row_index)
        for row_index, episode_id
        in enumerate(
            batch.episode_ids
        )
    }

    index = active.copy()

    index.insert(
        0,
        "feature_row",
        index["sample_nr"]
        .map(row_by_id)
        .astype(np.int64),
    )

    index["anomaly"] = (
        index["label"]
        .isin(
            [1, 2, 3, 4]
        )
    )

    index = index.sort_values(
        "feature_row"
    ).reset_index(drop=True)

    if not np.array_equal(
        index["feature_row"].to_numpy(),
        np.arange(
            len(index),
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "Feature episode index is not aligned with cache rows."
        )

    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic AURSAD statistical-feature cache."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to raw AURSAD.h5.",
    )

    parser.add_argument(
        "--inventory-path",
        type=Path,
        default=DEFAULT_INVENTORY_PATH,
        help=(
            "Episode inventory produced by "
            "experiments/audit_aursad_episodes.py."
        ),
    )

    parser.add_argument(
        "--membership-path",
        type=Path,
        default=DEFAULT_MEMBERSHIP_PATH,
        help=(
            "Frozen protocol membership CSV produced by "
            "experiments/build_aursad_protocol.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for feature-cache artifacts.",
    )

    parser.add_argument(
        "--cache-name",
        default=DEFAULT_CACHE_NAME,
        help=(
            f"Compressed cache filename. Default: {DEFAULT_CACHE_NAME}"
        ),
    )

    parser.add_argument(
        "--metadata-name",
        default=DEFAULT_METADATA_NAME,
        help=(
            "Human-readable metadata filename. "
            f"Default: {DEFAULT_METADATA_NAME}"
        ),
    )

    parser.add_argument(
        "--episode-index-name",
        default=DEFAULT_EPISODE_INDEX_NAME,
        help=(
            "Cache row-index CSV filename. "
            f"Default: {DEFAULT_EPISODE_INDEX_NAME}"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing feature-cache artifacts.",
    )

    parser.add_argument(
        "--skip-raw-hash",
        action="store_true",
        help=(
            "Skip hashing the roughly 6 GB raw HDF5 file. The protocol "
            "and inventory files are always hashed."
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

    inventory_path = (
        args.inventory_path
        .expanduser()
        .resolve()
    )

    membership_path = (
        args.membership_path
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    require_file(
        data_path,
        "Raw AURSAD HDF5 file",
    )

    require_file(
        inventory_path,
        "AURSAD episode inventory",
    )

    require_file(
        membership_path,
        "AURSAD protocol membership",
    )

    (
        cache_path,
        metadata_path,
        episode_index_path,
    ) = ensure_output_paths(
        output_dir=output_dir,
        cache_name=str(
            args.cache_name
        ),
        metadata_name=str(
            args.metadata_name
        ),
        episode_index_name=str(
            args.episode_index_name
        ),
        overwrite=bool(
            args.overwrite
        ),
    )

    started = time.perf_counter()

    print("=" * 76)
    print("AURSAD FEATURE CACHE BUILDER")
    print("=" * 76)
    print(f"Raw data:    {data_path}")
    print(f"Inventory:   {inventory_path}")
    print(f"Membership:  {membership_path}")
    print(f"Output:      {output_dir}")
    print(
        f"Expected:    {EXPECTED_SIGNAL_COUNT} signals, "
        f"{EXPECTED_FEATURE_COUNT} features/execution"
    )

    membership = load_protocol_membership(
        membership_path
    )

    inventory = pd.read_csv(
        inventory_path
    )

    active = active_episode_table(
        membership=membership,
        inventory=inventory,
    )

    episode_ids = (
        active["sample_nr"]
        .astype(np.int64)
        .tolist()
    )

    partition_counts = (
        active["active_partitions"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"\nUnique active executions: {len(episode_ids):,}"
    )

    print(
        "Labels:"
    )

    print(
        active.groupby(
            [
                "label",
                "label_name",
            ]
        )["sample_nr"]
        .nunique()
        .reset_index(
            name="execution_count"
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nLoading complete executions from HDF5..."
    )

    load_started = time.perf_counter()

    # The AURSAD loader is expected to preserve the requested ID order and use
    # its default frozen measured-signal policy.
    cycles = load_executions(
        episode_ids=episode_ids,
        data_path=data_path,
    )

    load_seconds = (
        time.perf_counter()
        - load_started
    )

    validate_loaded_cycles(
        cycles=cycles,
        expected=active,
    )

    print(
        f"Loaded {len(cycles):,} executions in "
        f"{load_seconds:.2f} seconds."
    )

    print(
        "Extracting statistical features..."
    )

    feature_started = time.perf_counter()

    batch = extract_feature_batch(
        cycles
    )

    feature_seconds = (
        time.perf_counter()
        - feature_started
    )

    validate_feature_batch(
        batch=batch,
        expected=active,
    )

    print(
        f"Feature shape: {batch.features.shape}"
    )

    print(
        f"Feature extraction completed in "
        f"{feature_seconds:.2f} seconds."
    )

    # Release raw time-series memory before cache serialization.
    del cycles

    print(
        "Saving compressed feature cache..."
    )

    save_feature_batch(
        batch,
        cache_path,
        metadata={
            "dataset": "AURSAD",
            "episode_unit": "sample_nr",
            "raw_data_path": str(
                data_path
            ),
            "inventory_path": str(
                inventory_path
            ),
            "membership_path": str(
                membership_path
            ),
            "active_partitions": sorted(
                ACTIVE_PARTITIONS
            ),
            "measured_signal_policy": (
                "frozen_48_channel_policy"
            ),
        },
    )

    episode_index = build_episode_index(
        active=active,
        batch=batch,
    )

    episode_index.to_csv(
        episode_index_path,
        index=False,
    )

    print(
        "Reopening cache for round-trip verification..."
    )

    restored = load_feature_batch(
        cache_path
    )

    if not np.array_equal(
        restored.episode_ids,
        batch.episode_ids,
    ):
        raise RuntimeError(
            "Round-trip verification failed for episode IDs."
        )

    if not np.array_equal(
        restored.anomaly_labels,
        batch.anomaly_labels,
    ):
        raise RuntimeError(
            "Round-trip verification failed for anomaly labels."
        )

    if not np.array_equal(
        restored.categories,
        batch.categories,
    ):
        raise RuntimeError(
            "Round-trip verification failed for categories."
        )

    if not np.array_equal(
        restored.settings,
        batch.settings,
    ):
        raise RuntimeError(
            "Round-trip verification failed for settings."
        )

    if restored.feature_names != batch.feature_names:
        raise RuntimeError(
            "Round-trip verification failed for feature names."
        )

    if restored.signal_columns != batch.signal_columns:
        raise RuntimeError(
            "Round-trip verification failed for signal columns."
        )

    if not np.array_equal(
        restored.features,
        batch.features,
    ):
        maximum_error = float(
            np.max(
                np.abs(
                    restored.features
                    - batch.features
                )
            )
        )

        raise RuntimeError(
            "Round-trip verification failed for feature values. "
            f"Maximum absolute difference: {maximum_error}"
        )

    raw_hash = None

    if args.skip_raw_hash:
        print(
            "Raw HDF5 SHA-256: skipped"
        )
    else:
        print(
            "Computing raw HDF5 SHA-256..."
        )

        raw_hash = sha256_file(
            data_path
        )

        print(
            f"Raw HDF5 SHA-256: {raw_hash}"
        )

    total_seconds = (
        time.perf_counter()
        - started
    )

    label_counts = (
        episode_index.groupby(
            [
                "label",
                "label_name",
            ]
        )["sample_nr"]
        .nunique()
        .reset_index(
            name="execution_count"
        )
        .to_dict(
            orient="records"
        )
    )

    metadata = {
        "cache_version": (
            FEATURE_CACHE_VERSION
        ),
        "dataset": "AURSAD",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "command": " ".join(
            sys.argv
        ),
        "episode_unit": "sample_nr",
        "inputs": {
            "raw_data": {
                "path": str(
                    data_path
                ),
                "size_bytes": int(
                    data_path.stat().st_size
                ),
                "sha256": raw_hash,
            },
            "episode_inventory": {
                "path": str(
                    inventory_path
                ),
                "sha256": sha256_file(
                    inventory_path
                ),
            },
            "protocol_membership": {
                "path": str(
                    membership_path
                ),
                "sha256": sha256_file(
                    membership_path
                ),
            },
        },
        "selection": {
            "active_partitions": sorted(
                ACTIVE_PARTITIONS
            ),
            "excluded_partition_cached": False,
            "unique_execution_count": int(
                len(
                    episode_index
                )
            ),
            "partition_combination_counts": (
                partition_counts
            ),
            "label_counts": (
                label_counts
            ),
        },
        "feature_schema": {
            "signal_count": int(
                len(
                    batch.signal_columns
                )
            ),
            "signal_columns": list(
                batch.signal_columns
            ),
            "statistic_count": int(
                len(
                    batch.statistic_names
                )
            ),
            "statistic_names": list(
                batch.statistic_names
            ),
            "feature_count": int(
                batch.features.shape[1]
            ),
            "feature_names": list(
                batch.feature_names
            ),
            "feature_order": (
                "signal-major"
            ),
            "standard_deviation_ddof": 1,
            "total_variation_definition": (
                "sum(abs(diff(values, axis=0)), axis=0)"
            ),
            "measured_signal_policy": (
                "frozen_48_channel_policy"
            ),
        },
        "cache": {
            "path": str(
                cache_path
            ),
            "size_bytes": int(
                cache_path.stat().st_size
            ),
            "sha256": sha256_file(
                cache_path
            ),
            "shape": [
                int(
                    batch.features.shape[0]
                ),
                int(
                    batch.features.shape[1]
                ),
            ],
            "dtype": str(
                batch.features.dtype
            ),
            "round_trip_verified": True,
        },
        "episode_index": {
            "path": str(
                episode_index_path
            ),
            "sha256": sha256_file(
                episode_index_path
            ),
        },
        "timing_seconds": {
            "load_executions": float(
                load_seconds
            ),
            "extract_features": float(
                feature_seconds
            ),
            "total": float(
                total_seconds
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
            "all_active_protocol_ids_present": True,
            "one_cache_row_per_execution": True,
            "episode_order_verified": True,
            "label_alignment_verified": True,
            "anomaly_alignment_verified": True,
            "shared_signal_schema_verified": True,
            "expected_signal_count_verified": True,
            "expected_feature_count_verified": True,
            "finite_features_verified": True,
            "cache_round_trip_verified": True,
        },
        "notes": [
            (
                "The same raw execution is cached only once even when "
                "it appears in multiple commissioning seeds and N values."
            ),
            (
                "Protocol membership is applied later by selecting cache "
                "rows via sample_nr."
            ),
            (
                "No variance filtering or standardization is applied "
                "while building the global cache. Those transformations "
                "must be fitted using detector training data only."
            ),
            (
                "Label 5 supplementary operations and other excluded "
                "protocol rows are not cached."
            ),
        ],
    }
    metadata.update(
        reproducibility_metadata(
            repo_root=PROJECT_ROOT,
            input_paths={
                "raw_data": data_path,
                "episode_inventory": inventory_path,
                "protocol_membership": membership_path,
            },
            artifact_paths={
                "feature_cache": cache_path,
                "episode_index": episode_index_path,
            },
        )
    )

    metadata_path.write_text(
        json.dumps(
            json_safe(
                metadata
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("FEATURE CACHE COMPLETE")
    print("=" * 76)
    print(
        f"Executions:       {batch.features.shape[0]:,}"
    )
    print(
        f"Signals:          {len(batch.signal_columns):,}"
    )
    print(
        f"Features:         {batch.features.shape[1]:,}"
    )
    print(
        f"Normal:           "
        f"{int((~batch.anomaly_labels).sum()):,}"
    )
    print(
        f"Anomalous:        "
        f"{int(batch.anomaly_labels.sum()):,}"
    )
    print(
        f"Cache size:       "
        f"{cache_path.stat().st_size / (1024 ** 2):.2f} MiB"
    )
    print(
        f"Total time:       {total_seconds:.2f} seconds"
    )
    print(
        "Round-trip check: PASSED"
    )

    print("\nArtifacts:")
    print(f"  {cache_path}")
    print(f"  {metadata_path}")
    print(f"  {episode_index_path}")


if __name__ == "__main__":
    main()
