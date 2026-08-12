#!/usr/bin/env python3
"""
experiments/audit_aursad_episodes.py

Episode-level audit for AURSAD.

The raw AURSAD HDF5 file is stored as a Pandas/PyTables DataFrame under
/complete_data using dtype-specific blocks. This script reconstructs only the
columns needed for episode auditing and streams them in chunks.

The leakage-safe unit is expected to be one complete `sample_nr` execution.

This script verifies:

- whether every sample_nr has exactly one label;
- execution row counts and durations;
- timestamp monotonicity within each execution;
- estimated sampling interval and sampling frequency;
- whether execution rows are contiguous in the HDF5 table;
- runtime_state values observed per execution;
- class counts at execution level;
- whether the observed label counts match published expectations.

It does not train detectors or construct commissioning/calibration/evaluation
splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aursad"
    / "AURSAD.h5"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
)

DEFAULT_CHUNK_ROWS = 100_000

REQUIRED_COLUMNS = (
    "sample_nr",
    "label",
    "timestamp",
    "runtime_state",
)

# Published expectations recorded for audit comparison only.
EXPECTED_TOTAL_EXECUTIONS = 4094
EXPECTED_TIGHTENING_EXECUTIONS = 2045
EXPECTED_SUPPLEMENTARY_EXECUTIONS = 2049

EXPECTED_EXECUTION_COUNTS_BY_LABEL = {
    0: 1420,
    1: 221,
    2: 183,
    3: 218,
    4: 3,
    5: 2049,
}

DEFAULT_LABEL_NAMES = {
    0: "normal_tightening",
    1: "damaged_screw",
    2: "extra_component",
    3: "missing_screw",
    4: "damaged_thread",
    5: "supplementary_operation",
}


def decode_text(value: Any) -> Any:
    """Decode byte-like values safely."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.bytes_):
        return bytes(value).decode(
            "utf-8",
            errors="replace",
        )

    return value


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats into JSON-safe objects."""
    if isinstance(value, (bytes, np.bytes_)):
        return decode_text(value)

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

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Compute a streaming SHA-256 hash."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True)
class ColumnLocation:
    """Location of one logical DataFrame column in the PyTables blocks."""

    column_name: str
    block_dataset_name: str
    block_column_index: int
    dtype: str


@dataclass
class EpisodeAccumulator:
    """Streaming statistics for one AURSAD execution."""

    sample_nr: int

    row_count: int = 0

    labels: set[int] = field(
        default_factory=set
    )

    runtime_states: set[int] = field(
        default_factory=set
    )

    first_global_row: int | None = None
    last_global_row: int | None = None

    contiguous_segment_count: int = 0
    previous_global_row: int | None = None

    first_timestamp: float | None = None
    last_timestamp: float | None = None
    minimum_timestamp: float | None = None
    maximum_timestamp: float | None = None

    previous_timestamp: float | None = None

    timestamp_decrease_count: int = 0
    timestamp_equal_count: int = 0
    positive_delta_count: int = 0

    delta_sum: float = 0.0
    delta_sum_squares: float = 0.0
    minimum_positive_delta: float | None = None
    maximum_positive_delta: float | None = None

    def update(
        self,
        global_row_index: int,
        label: int,
        timestamp: float,
        runtime_state: int,
    ) -> None:
        """Update this episode with one row."""
        self.row_count += 1
        self.labels.add(int(label))
        self.runtime_states.add(int(runtime_state))

        if self.first_global_row is None:
            self.first_global_row = int(
                global_row_index
            )
            self.contiguous_segment_count = 1
        else:
            assert self.previous_global_row is not None

            if (
                global_row_index
                != self.previous_global_row + 1
            ):
                self.contiguous_segment_count += 1

        self.last_global_row = int(
            global_row_index
        )
        self.previous_global_row = int(
            global_row_index
        )

        if np.isfinite(timestamp):
            timestamp = float(timestamp)

            if self.first_timestamp is None:
                self.first_timestamp = timestamp

            self.last_timestamp = timestamp

            self.minimum_timestamp = (
                timestamp
                if self.minimum_timestamp is None
                else min(
                    self.minimum_timestamp,
                    timestamp,
                )
            )

            self.maximum_timestamp = (
                timestamp
                if self.maximum_timestamp is None
                else max(
                    self.maximum_timestamp,
                    timestamp,
                )
            )

            if self.previous_timestamp is not None:
                delta = (
                    timestamp
                    - self.previous_timestamp
                )

                if delta < 0:
                    self.timestamp_decrease_count += 1
                elif delta == 0:
                    self.timestamp_equal_count += 1
                else:
                    self.positive_delta_count += 1
                    self.delta_sum += delta
                    self.delta_sum_squares += (
                        delta * delta
                    )

                    self.minimum_positive_delta = (
                        delta
                        if self.minimum_positive_delta
                        is None
                        else min(
                            self.minimum_positive_delta,
                            delta,
                        )
                    )

                    self.maximum_positive_delta = (
                        delta
                        if self.maximum_positive_delta
                        is None
                        else max(
                            self.maximum_positive_delta,
                            delta,
                        )
                    )

            self.previous_timestamp = timestamp

    def to_record(
        self,
        label_names: dict[int, str],
    ) -> dict[str, Any]:
        """Convert the accumulator into one episode inventory record."""
        unique_labels = sorted(self.labels)
        unique_runtime_states = sorted(
            self.runtime_states
        )

        single_label = len(unique_labels) == 1

        label_value = (
            unique_labels[0]
            if single_label
            else None
        )

        label_name = (
            label_names.get(
                int(label_value),
                f"label_{label_value}",
            )
            if label_value is not None
            else "multiple_labels"
        )

        timestamp_duration = None

        if (
            self.minimum_timestamp is not None
            and self.maximum_timestamp is not None
        ):
            timestamp_duration = (
                self.maximum_timestamp
                - self.minimum_timestamp
            )

        mean_positive_delta = None
        std_positive_delta = None
        estimated_sampling_rate_hz = None

        if self.positive_delta_count > 0:
            mean_positive_delta = (
                self.delta_sum
                / self.positive_delta_count
            )

            variance = max(
                0.0,
                (
                    self.delta_sum_squares
                    / self.positive_delta_count
                )
                - mean_positive_delta**2,
            )

            std_positive_delta = math.sqrt(
                variance
            )

            if mean_positive_delta > 0:
                estimated_sampling_rate_hz = (
                    1.0
                    / mean_positive_delta
                )

        return {
            "sample_nr": int(self.sample_nr),
            "label": label_value,
            "label_name": label_name,
            "unique_label_count": len(
                unique_labels
            ),
            "labels_json": json.dumps(
                unique_labels
            ),
            "has_single_label": single_label,
            "row_count": int(
                self.row_count
            ),
            "first_global_row": (
                int(self.first_global_row)
                if self.first_global_row
                is not None
                else None
            ),
            "last_global_row": (
                int(self.last_global_row)
                if self.last_global_row
                is not None
                else None
            ),
            "contiguous_segment_count": int(
                self.contiguous_segment_count
            ),
            "rows_are_contiguous": (
                self.contiguous_segment_count == 1
            ),
            "first_timestamp": (
                float(self.first_timestamp)
                if self.first_timestamp
                is not None
                else None
            ),
            "last_timestamp": (
                float(self.last_timestamp)
                if self.last_timestamp
                is not None
                else None
            ),
            "minimum_timestamp": (
                float(self.minimum_timestamp)
                if self.minimum_timestamp
                is not None
                else None
            ),
            "maximum_timestamp": (
                float(self.maximum_timestamp)
                if self.maximum_timestamp
                is not None
                else None
            ),
            "duration_seconds": (
                float(timestamp_duration)
                if timestamp_duration
                is not None
                else None
            ),
            "timestamp_decrease_count": int(
                self.timestamp_decrease_count
            ),
            "timestamp_equal_count": int(
                self.timestamp_equal_count
            ),
            "positive_delta_count": int(
                self.positive_delta_count
            ),
            "timestamps_monotonic_nondecreasing": (
                self.timestamp_decrease_count == 0
            ),
            "mean_positive_delta_seconds": (
                float(mean_positive_delta)
                if mean_positive_delta
                is not None
                else None
            ),
            "std_positive_delta_seconds": (
                float(std_positive_delta)
                if std_positive_delta
                is not None
                else None
            ),
            "minimum_positive_delta_seconds": (
                float(
                    self.minimum_positive_delta
                )
                if self.minimum_positive_delta
                is not None
                else None
            ),
            "maximum_positive_delta_seconds": (
                float(
                    self.maximum_positive_delta
                )
                if self.maximum_positive_delta
                is not None
                else None
            ),
            "estimated_sampling_rate_hz": (
                float(
                    estimated_sampling_rate_hz
                )
                if estimated_sampling_rate_hz
                is not None
                else None
            ),
            "runtime_state_count": len(
                unique_runtime_states
            ),
            "runtime_states_json": json.dumps(
                unique_runtime_states
            ),
        }


def read_column_locations(
    handle: h5py.File,
) -> dict[str, ColumnLocation]:
    """
    Reconstruct logical DataFrame column locations from PyTables blocks.
    """
    group = handle["/complete_data"]

    locations: dict[
        str,
        ColumnLocation,
    ] = {}

    for dataset_name in sorted(
        group.keys()
    ):
        if not dataset_name.startswith(
            "block"
        ):
            continue

        if not dataset_name.endswith(
            "_items"
        ):
            continue

        block_prefix = dataset_name[
            : -len("_items")
        ]

        values_name = (
            f"{block_prefix}_values"
        )

        if values_name not in group:
            raise ValueError(
                f"Missing dataset: "
                f"/complete_data/{values_name}"
            )

        items_dataset = group[
            dataset_name
        ]
        values_dataset = group[
            values_name
        ]

        names = [
            str(decode_text(value))
            for value in items_dataset[:]
        ]

        if values_dataset.ndim != 2:
            raise ValueError(
                f"{values_name} must be 2-D; "
                f"found shape "
                f"{values_dataset.shape}"
            )

        if len(names) != values_dataset.shape[1]:
            raise ValueError(
                f"{dataset_name} contains "
                f"{len(names)} names but "
                f"{values_name} has "
                f"{values_dataset.shape[1]} "
                f"columns."
            )

        for column_index, name in enumerate(
            names
        ):
            if name in locations:
                raise ValueError(
                    f"Duplicate logical column "
                    f"name detected: {name}"
                )

            locations[name] = (
                ColumnLocation(
                    column_name=name,
                    block_dataset_name=(
                        values_name
                    ),
                    block_column_index=(
                        column_index
                    ),
                    dtype=str(
                        values_dataset.dtype
                    ),
                )
            )

    return locations


def read_required_chunk(
    group: h5py.Group,
    locations: dict[str, ColumnLocation],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    """
    Read only the required columns for one row chunk.

    Columns are grouped by physical block so each HDF5 block slice is read
    once per chunk.
    """
    by_block: dict[
        str,
        list[ColumnLocation],
    ] = defaultdict(list)

    for column_name in REQUIRED_COLUMNS:
        by_block[
            locations[
                column_name
            ].block_dataset_name
        ].append(
            locations[column_name]
        )

    result: dict[
        str,
        np.ndarray,
    ] = {}

    for block_name, block_locations in (
        by_block.items()
    ):
        dataset = group[
            block_name
        ]

        ordered_locations = sorted(
            block_locations,
            key=lambda location: location.block_column_index,
        )

        physical_indices = [
            location.block_column_index
            for location in ordered_locations
        ]

        block_chunk = dataset[
            start:stop,
            physical_indices,
        ]

        if block_chunk.ndim == 1:
            block_chunk = (
                block_chunk[:, None]
            )

        for local_index, location in enumerate(
            ordered_locations
        ):
            result[
                location.column_name
            ] = np.asarray(
                block_chunk[
                    :,
                    local_index,
                ]
            )

    return result


def audit_episodes(
    data_path: Path,
    chunk_rows: int,
    label_names: dict[int, str],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """Stream the HDF5 rows and construct one record per sample_nr."""
    accumulators: dict[
        int,
        EpisodeAccumulator,
    ] = {}

    with h5py.File(
        data_path,
        "r",
    ) as handle:
        if "/complete_data" not in handle:
            raise ValueError(
                "Missing /complete_data group."
            )

        group = handle[
            "/complete_data"
        ]

        locations = read_column_locations(
            handle
        )

        missing_columns = [
            name
            for name in REQUIRED_COLUMNS
            if name not in locations
        ]

        if missing_columns:
            raise ValueError(
                "Missing required AURSAD "
                f"columns: {missing_columns}"
            )

        block_row_counts = {
            int(
                group[
                    location.block_dataset_name
                ].shape[0]
            )
            for location
            in locations.values()
        }

        if len(block_row_counts) != 1:
            raise ValueError(
                "Inconsistent row counts "
                f"across blocks: "
                f"{sorted(block_row_counts)}"
            )

        total_rows = block_row_counts.pop()

        axis1_rows = int(
            group["axis1"].shape[0]
        )

        if total_rows != axis1_rows:
            raise ValueError(
                f"axis1 has {axis1_rows} rows "
                f"but value blocks have "
                f"{total_rows}."
            )

        print(
            f"Total rows: {total_rows:,}"
        )
        print(
            f"Chunk size: {chunk_rows:,}"
        )

        for start in range(
            0,
            total_rows,
            chunk_rows,
        ):
            stop = min(
                start + chunk_rows,
                total_rows,
            )

            chunk = read_required_chunk(
                group=group,
                locations=locations,
                start=start,
                stop=stop,
            )

            sample_values = (
                chunk["sample_nr"]
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            label_values = (
                chunk["label"]
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            timestamp_values = (
                chunk["timestamp"]
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            runtime_values = (
                chunk["runtime_state"]
                .astype(
                    np.int64,
                    copy=False,
                )
            )

            for local_index in range(
                stop - start
            ):
                sample_nr = int(
                    sample_values[
                        local_index
                    ]
                )

                accumulator = (
                    accumulators.get(
                        sample_nr
                    )
                )

                if accumulator is None:
                    accumulator = (
                        EpisodeAccumulator(
                            sample_nr=(
                                sample_nr
                            )
                        )
                    )

                    accumulators[
                        sample_nr
                    ] = accumulator

                accumulator.update(
                    global_row_index=(
                        start
                        + local_index
                    ),
                    label=int(
                        label_values[
                            local_index
                        ]
                    ),
                    timestamp=float(
                        timestamp_values[
                            local_index
                        ]
                    ),
                    runtime_state=int(
                        runtime_values[
                            local_index
                        ]
                    ),
                )

            print(
                f"Processed rows "
                f"{start:,}:{stop:,} "
                f"("
                f"{100.0 * stop / total_rows:.1f}%"
                f") — "
                f"{len(accumulators):,} "
                f"executions discovered"
            )

    records = [
        accumulator.to_record(
            label_names=label_names
        )
        for sample_nr, accumulator
        in sorted(
            accumulators.items()
        )
    ]

    inventory = pd.DataFrame(
        records
    )

    summary = {
        "row_count": int(
            inventory["row_count"].sum()
        ),
        "execution_count": int(
            len(inventory)
        ),
        "minimum_sample_nr": int(
            inventory["sample_nr"].min()
        ),
        "maximum_sample_nr": int(
            inventory["sample_nr"].max()
        ),
        "multi_label_execution_count": int(
            (
                inventory[
                    "unique_label_count"
                ]
                > 1
            ).sum()
        ),
        "noncontiguous_execution_count": int(
            (
                ~inventory[
                    "rows_are_contiguous"
                ]
            ).sum()
        ),
        "nonmonotonic_timestamp_execution_count": int(
            (
                ~inventory[
                    "timestamps_monotonic_nondecreasing"
                ]
            ).sum()
        ),
    }

    return inventory, summary


def build_label_counts(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Create execution-level label counts."""
    valid = inventory[
        inventory["has_single_label"]
    ].copy()

    counts = (
        valid.groupby(
            [
                "label",
                "label_name",
            ],
            dropna=False,
        )
        .agg(
            execution_count=(
                "sample_nr",
                "nunique",
            ),
            total_rows=(
                "row_count",
                "sum",
            ),
            median_rows=(
                "row_count",
                "median",
            ),
            minimum_rows=(
                "row_count",
                "min",
            ),
            maximum_rows=(
                "row_count",
                "max",
            ),
            median_duration_seconds=(
                "duration_seconds",
                "median",
            ),
            median_sampling_rate_hz=(
                "estimated_sampling_rate_hz",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "label"
        )
    )

    counts["label"] = (
        counts["label"]
        .astype(int)
    )

    counts[
        "expected_execution_count"
    ] = counts["label"].map(
        EXPECTED_EXECUTION_COUNTS_BY_LABEL
    )

    counts[
        "matches_expected_execution_count"
    ] = (
        counts["execution_count"]
        == counts[
            "expected_execution_count"
        ]
    )

    return counts


def build_length_summary(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize episode sizes and durations by label."""
    valid = inventory[
        inventory["has_single_label"]
    ].copy()

    rows: list[
        dict[str, Any]
    ] = []

    for (
        label,
        label_name,
    ), group in valid.groupby(
        [
            "label",
            "label_name",
        ],
        sort=True,
    ):
        row_counts = group[
            "row_count"
        ].to_numpy(
            dtype=np.float64
        )

        durations = group[
            "duration_seconds"
        ].dropna().to_numpy(
            dtype=np.float64
        )

        sampling_rates = group[
            "estimated_sampling_rate_hz"
        ].replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna().to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "label": int(label),
                "label_name": label_name,
                "execution_count": int(
                    len(group)
                ),
                "row_count_min": float(
                    np.min(row_counts)
                ),
                "row_count_q25": float(
                    np.quantile(
                        row_counts,
                        0.25,
                    )
                ),
                "row_count_median": float(
                    np.median(
                        row_counts
                    )
                ),
                "row_count_q75": float(
                    np.quantile(
                        row_counts,
                        0.75,
                    )
                ),
                "row_count_max": float(
                    np.max(row_counts)
                ),
                "duration_min_seconds": (
                    float(
                        np.min(
                            durations
                        )
                    )
                    if durations.size
                    else None
                ),
                "duration_median_seconds": (
                    float(
                        np.median(
                            durations
                        )
                    )
                    if durations.size
                    else None
                ),
                "duration_max_seconds": (
                    float(
                        np.max(
                            durations
                        )
                    )
                    if durations.size
                    else None
                ),
                "sampling_rate_median_hz": (
                    float(
                        np.median(
                            sampling_rates
                        )
                    )
                    if sampling_rates.size
                    else None
                ),
                "sampling_rate_q25_hz": (
                    float(
                        np.quantile(
                            sampling_rates,
                            0.25,
                        )
                    )
                    if sampling_rates.size
                    else None
                ),
                "sampling_rate_q75_hz": (
                    float(
                        np.quantile(
                            sampling_rates,
                            0.75,
                        )
                    )
                    if sampling_rates.size
                    else None
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "label"
    )


def build_runtime_state_counts(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize observed runtime-state combinations."""
    return (
        inventory.groupby(
            [
                "label",
                "label_name",
                "runtime_states_json",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "execution_count"
        )
        .reset_index()
        .sort_values(
            [
                "label",
                "execution_count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


def parse_label_names(
    mapping_path: Path | None,
) -> dict[int, str]:
    """
    Load an optional JSON label-name mapping.

    Expected format:
        {
          "0": "normal_tightening",
          "1": "damaged_screw"
        }
    """
    if mapping_path is None:
        return dict(
            DEFAULT_LABEL_NAMES
        )

    with mapping_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    return {
        int(key): str(value)
        for key, value in raw.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AURSAD at execution level "
            "using sample_nr."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=(
            "Path to AURSAD.h5."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for audit outputs."
        ),
    )

    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help=(
            "Number of HDF5 rows "
            "processed per chunk."
        ),
    )

    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help=(
            "Skip dataset SHA-256 "
            "calculation."
        ),
    )

    parser.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from "
            "integer labels to names."
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

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"AURSAD file does not "
            f"exist: {data_path}"
        )

    if not data_path.is_file():
        raise ValueError(
            f"AURSAD path is not a "
            f"file: {data_path}"
        )

    if args.chunk_rows <= 0:
        raise ValueError(
            "--chunk-rows must be "
            "positive."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_names = parse_label_names(
        args.label_map
    )

    print("=" * 72)
    print("AURSAD EPISODE-LEVEL AUDIT")
    print("=" * 72)
    print(
        f"Input: {data_path}"
    )
    print(
        "Size: "
        f"{data_path.stat().st_size / (1024 ** 3):.3f} GiB"
    )

    if args.skip_hash:
        dataset_hash = None
        print(
            "SHA-256: skipped"
        )
    else:
        print(
            "Computing SHA-256..."
        )
        dataset_hash = sha256_file(
            data_path
        )
        print(
            f"SHA-256: "
            f"{dataset_hash}"
        )

    inventory, audit_summary = (
        audit_episodes(
            data_path=data_path,
            chunk_rows=args.chunk_rows,
            label_names=label_names,
        )
    )

    label_counts = build_label_counts(
        inventory
    )

    length_summary = build_length_summary(
        inventory
    )

    runtime_state_counts = (
        build_runtime_state_counts(
            inventory
        )
    )

    inventory_path = (
        output_dir
        / "aursad_episode_inventory.csv"
    )

    label_counts_path = (
        output_dir
        / "aursad_label_execution_counts.csv"
    )

    length_summary_path = (
        output_dir
        / "aursad_episode_length_summary.csv"
    )

    runtime_states_path = (
        output_dir
        / "aursad_runtime_state_summary.csv"
    )

    manifest_path = (
        output_dir
        / "aursad_episode_audit.json"
    )

    inventory.to_csv(
        inventory_path,
        index=False,
        float_format="%.12g",
    )

    label_counts.to_csv(
        label_counts_path,
        index=False,
        float_format="%.12g",
    )

    length_summary.to_csv(
        length_summary_path,
        index=False,
        float_format="%.12g",
    )

    runtime_state_counts.to_csv(
        runtime_states_path,
        index=False,
        float_format="%.12g",
    )

    tightening_count = int(
        inventory.loc[
            inventory[
                "label"
            ].isin(
                [0, 1, 2, 3, 4]
            ),
            "sample_nr",
        ].nunique()
    )

    supplementary_count = int(
        inventory.loc[
            inventory[
                "label"
            ].eq(5),
            "sample_nr",
        ].nunique()
    )

    observed_counts = {
        int(row.label): int(
            row.execution_count
        )
        for row in label_counts.itertuples(
            index=False
        )
    }

    expected_counts_match = {
        str(label): (
            observed_counts.get(label)
            == expected_count
        )
        for label, expected_count
        in EXPECTED_EXECUTION_COUNTS_BY_LABEL.items()
    }

    all_expected_counts_match = all(
        expected_counts_match.values()
    )

    manifest = {
        "audit_version": (
            "aursad-episode-audit-v1"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "command": " ".join(
            sys.argv
        ),
        "dataset": {
            "path": str(
                data_path
            ),
            "size_bytes": int(
                data_path.stat().st_size
            ),
            "sha256": dataset_hash,
        },
        "episode_unit": (
            "sample_nr"
        ),
        "label_mapping": {
            str(key): value
            for key, value
            in sorted(
                label_names.items()
            )
        },
        "audit": {
            **audit_summary,
            "tightening_execution_count": (
                tightening_count
            ),
            "supplementary_execution_count": (
                supplementary_count
            ),
            "observed_execution_counts_by_label": {
                str(key): value
                for key, value
                in sorted(
                    observed_counts.items()
                )
            },
            "expected_execution_counts_by_label": {
                str(key): value
                for key, value
                in sorted(
                    EXPECTED_EXECUTION_COUNTS_BY_LABEL.items()
                )
            },
            "expected_count_match_by_label": (
                expected_counts_match
            ),
            "all_expected_counts_match": (
                all_expected_counts_match
            ),
            "total_execution_count_matches_expected": (
                len(inventory)
                == EXPECTED_TOTAL_EXECUTIONS
            ),
            "tightening_count_matches_expected": (
                tightening_count
                == EXPECTED_TIGHTENING_EXECUTIONS
            ),
            "supplementary_count_matches_expected": (
                supplementary_count
                == EXPECTED_SUPPLEMENTARY_EXECUTIONS
            ),
        },
        "artifacts": {
            "episode_inventory": str(
                inventory_path
            ),
            "label_execution_counts": str(
                label_counts_path
            ),
            "episode_length_summary": str(
                length_summary_path
            ),
            "runtime_state_summary": str(
                runtime_states_path
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": (
                platform.platform()
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "protocol_readiness": {
            "single_label_per_execution": (
                audit_summary[
                    "multi_label_execution_count"
                ]
                == 0
            ),
            "contiguous_rows_per_execution": (
                audit_summary[
                    "noncontiguous_execution_count"
                ]
                == 0
            ),
            "monotonic_timestamps_per_execution": (
                audit_summary[
                    "nonmonotonic_timestamp_execution_count"
                ]
                == 0
            ),
            "published_counts_reproduced": (
                all_expected_counts_match
            ),
            "recommended_leakage_safe_unit": (
                "one complete sample_nr execution"
            ),
        },
        "limitations": [
            (
                "Label names are based on the "
                "predeclared mapping and must be "
                "checked against official AURSAD "
                "documentation."
            ),
            (
                "Label 5 is treated as a "
                "supplementary operation class and "
                "must not automatically be used as "
                "healthy source data without a "
                "separate protocol decision."
            ),
            (
                "This audit does not create source, "
                "commissioning, calibration, or "
                "evaluation memberships."
            ),
            (
                "No detector has been trained."
            ),
        ],
    }

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(manifest),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print(
        f"Executions: "
        f"{len(inventory):,}"
    )

    print(
        f"Tightening executions: "
        f"{tightening_count:,}"
    )

    print(
        f"Supplementary executions: "
        f"{supplementary_count:,}"
    )

    print(
        "Multi-label executions: "
        f"{audit_summary['multi_label_execution_count']}"
    )

    print(
        "Noncontiguous executions: "
        f"{audit_summary['noncontiguous_execution_count']}"
    )

    print(
        "Nonmonotonic timestamp executions: "
        f"{audit_summary['nonmonotonic_timestamp_execution_count']}"
    )

    print(
        "Published label counts reproduced: "
        f"{all_expected_counts_match}"
    )

    print("\nExecution counts by label:")
    print(
        label_counts.to_string(
            index=False
        )
    )

    print("\nArtifacts:")
    print(
        f"  {inventory_path}"
    )
    print(
        f"  {label_counts_path}"
    )
    print(
        f"  {length_summary_path}"
    )
    print(
        f"  {runtime_states_path}"
    )
    print(
        f"  {manifest_path}"
    )

    if (
        audit_summary[
            "multi_label_execution_count"
        ]
        > 0
    ):
        print(
            "\n[WARNING] Some sample_nr "
            "executions contain multiple labels."
        )

    if (
        audit_summary[
            "noncontiguous_execution_count"
        ]
        > 0
    ):
        print(
            "\n[WARNING] Some executions "
            "appear in multiple disjoint row segments."
        )

    if (
        audit_summary[
            "nonmonotonic_timestamp_execution_count"
        ]
        > 0
    ):
        print(
            "\n[WARNING] Some executions "
            "have decreasing timestamps."
        )

    if not all_expected_counts_match:
        print(
            "\n[WARNING] Observed execution-level "
            "label counts do not match the "
            "predeclared published expectations."
        )


if __name__ == "__main__":
    main()