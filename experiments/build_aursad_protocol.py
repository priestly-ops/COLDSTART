#!/usr/bin/env python3
"""
experiments/build_aursad_protocol.py

Construct a deterministic, leakage-safe, episode-level AURSAD commissioning
protocol from the audited episode inventory.

The protocol is intentionally built at the `sample_nr` execution level.
Rows from one execution are never split across commissioning, calibration,
healthy evaluation, or anomaly evaluation partitions.

Expected input
--------------
reports/aursad/aursad_episode_inventory.csv

This file is produced by:
    experiments/audit_aursad_episodes.py

Default protocol
----------------
Healthy tightening executions (label 0) are divided into:

1. fixed calibration pool
2. fixed healthy evaluation set
3. per-seed commissioning reservoir

Anomalous tightening executions (labels 1, 2, 3, 4) are assigned to a fixed
anomaly evaluation set.

Supplementary-operation executions (label 5) are excluded by default.

Commissioning sets are nested within each seed:
    N=10 ⊂ N=25 ⊂ N=50 ⊂ N=100 ⊂ N=250 ⊂ N=500

The calibration and evaluation memberships are fixed across seeds so that
variation across seeds is attributable to commissioning-sample selection
rather than changing test sets.

Outputs
-------
reports/aursad/protocol/
├── aursad_protocol_membership.csv
├── commissioning_ids.csv
├── calibration_ids.csv
├── healthy_eval_ids.csv
├── anomaly_eval_ids.csv
├── excluded_ids.csv
├── protocol_summary.csv
├── protocol_manifest.json
└── seeds/
    ├── seed_00_commissioning_ids.csv
    ├── ...
    └── seed_19_commissioning_ids.csv

Example
-------
python experiments/build_aursad_protocol.py ^
  --inventory-path reports/aursad/aursad_episode_inventory.csv ^
  --output-dir reports/aursad/protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
    / "aursad_episode_inventory.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "aursad"
    / "protocol"
)

DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_MASTER_SEED = 42

NORMAL_LABEL = 0
ANOMALY_LABELS = (1, 2, 3, 4)
SUPPLEMENTARY_LABEL = 5

REQUIRED_COLUMNS = (
    "sample_nr",
    "label",
    "label_name",
    "has_single_label",
    "rows_are_contiguous",
    "timestamps_monotonic_nondecreasing",
    "row_count",
)

PARTITION_COMMISSIONING = "commissioning"
PARTITION_CALIBRATION = "calibration"
PARTITION_HEALTHY_EVAL = "healthy_eval"
PARTITION_ANOMALY_EVAL = "anomaly_eval"
PARTITION_EXCLUDED = "excluded"


@dataclass(frozen=True)
class ProtocolConfig:
    """Frozen configuration for one protocol build."""

    grid: tuple[int, ...]
    seeds: tuple[int, ...]
    master_seed: int
    calibration_count: int
    healthy_eval_count: int
    timestamp_policy: str
    include_supplementary: bool

    @property
    def maximum_commissioning_n(self) -> int:
        return max(self.grid)


def sha256_file(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Compute SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert common NumPy/Pandas values to JSON-safe Python objects."""
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

    if pd.isna(value):
        return None

    return value


def parse_int_csv(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers while preserving input order."""
    parsed: list[int] = []

    for raw_item in value.split(","):
        item = raw_item.strip()

        if not item:
            continue

        parsed.append(int(item))

    if not parsed:
        raise argparse.ArgumentTypeError(
            "Expected at least one integer."
        )

    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(
            "Duplicate integer values are not allowed."
        )

    return tuple(parsed)


def validate_grid(grid: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize the commissioning grid."""
    normalized = tuple(int(value) for value in grid)

    if any(value <= 0 for value in normalized):
        raise ValueError(
            "Every commissioning-grid value must be positive."
        )

    if tuple(sorted(normalized)) != normalized:
        raise ValueError(
            "The commissioning grid must be strictly increasing."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "The commissioning grid contains duplicate values."
        )

    return normalized


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate seed identifiers."""
    normalized = tuple(int(value) for value in seeds)

    if any(value < 0 for value in normalized):
        raise ValueError(
            "Seed identifiers must be non-negative integers."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Seed identifiers must be unique."
        )

    return normalized


def coerce_bool_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert common CSV boolean representations to bool.

    Pandas may read True/False columns as booleans or strings depending on
    file history and missing values.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknown = sorted(
        set(normalized.dropna().unique())
        - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"Column {column_name!r} contains unsupported boolean "
            f"values: {unknown[:10]}"
        )

    return normalized.map(mapping).astype(bool)


def load_inventory(path: Path) -> pd.DataFrame:
    """Load and validate the episode-level audit inventory."""
    if not path.exists():
        raise FileNotFoundError(
            f"AURSAD episode inventory not found: {path}"
        )

    inventory = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in inventory.columns
    ]

    if missing:
        raise ValueError(
            "Episode inventory is missing required columns: "
            f"{missing}"
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

    for column in (
        "has_single_label",
        "rows_are_contiguous",
        "timestamps_monotonic_nondecreasing",
    ):
        inventory[column] = coerce_bool_series(
            inventory[column],
            column,
        )

    if inventory["sample_nr"].duplicated().any():
        duplicates = (
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
            "Each sample_nr must appear exactly once in the episode "
            f"inventory. Duplicates include: {duplicates[:20]}"
        )

    if (inventory["row_count"] <= 0).any():
        bad_ids = (
            inventory.loc[
                inventory["row_count"] <= 0,
                "sample_nr",
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "Every execution must have at least one row. Invalid "
            f"sample_nr values include: {bad_ids[:20]}"
        )

    inventory = inventory.sort_values(
        "sample_nr"
    ).reset_index(drop=True)

    return inventory


def classify_eligibility(
    inventory: pd.DataFrame,
    timestamp_policy: str,
    include_supplementary: bool,
) -> pd.DataFrame:
    """
    Add protocol eligibility and exclusion-reason columns.

    Timestamp policy:
      - exclude: remove nonmonotonic executions from all partitions
      - keep: retain them and record that the protocol accepted them
      - error: stop if any nonmonotonic execution exists
    """
    if timestamp_policy not in {
        "exclude",
        "keep",
        "error",
    }:
        raise ValueError(
            "timestamp_policy must be one of: exclude, keep, error"
        )

    result = inventory.copy()

    nonmonotonic = (
        ~result[
            "timestamps_monotonic_nondecreasing"
        ]
    )

    if timestamp_policy == "error" and nonmonotonic.any():
        affected = (
            result.loc[
                nonmonotonic,
                [
                    "sample_nr",
                    "label",
                    "label_name",
                ],
            ]
            .to_dict(orient="records")
        )

        raise ValueError(
            "Nonmonotonic timestamp executions were found and "
            f"--timestamp-policy=error was selected: {affected}"
        )

    reasons: list[str] = []

    for row in result.itertuples(index=False):
        row_reasons: list[str] = []

        if not bool(row.has_single_label):
            row_reasons.append("multiple_labels")

        if not bool(row.rows_are_contiguous):
            row_reasons.append("noncontiguous_rows")

        if (
            timestamp_policy == "exclude"
            and not bool(
                row.timestamps_monotonic_nondecreasing
            )
        ):
            row_reasons.append("nonmonotonic_timestamps")

        if (
            int(row.label) == SUPPLEMENTARY_LABEL
            and not include_supplementary
        ):
            row_reasons.append("supplementary_operation")

        if int(row.label) not in {
            NORMAL_LABEL,
            *ANOMALY_LABELS,
            SUPPLEMENTARY_LABEL,
        }:
            row_reasons.append("unknown_label")

        reasons.append(
            ";".join(row_reasons)
        )

    result["exclusion_reason"] = reasons
    result["protocol_eligible"] = (
        result["exclusion_reason"].eq("")
    )

    return result


def split_fixed_partitions(
    inventory: pd.DataFrame,
    config: ProtocolConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create fixed calibration, healthy-evaluation, anomaly-evaluation, and
    commissioning-reservoir memberships.
    """
    eligible = inventory[
        inventory["protocol_eligible"]
    ].copy()

    healthy = eligible[
        eligible["label"].eq(NORMAL_LABEL)
    ].copy()

    anomalies = eligible[
        eligible["label"].isin(ANOMALY_LABELS)
    ].copy()

    if anomalies.empty:
        raise ValueError(
            "No eligible anomaly executions were found."
        )

    required_healthy = (
        config.calibration_count
        + config.healthy_eval_count
        + config.maximum_commissioning_n
    )

    if len(healthy) < required_healthy:
        raise ValueError(
            "Insufficient eligible healthy executions for the requested "
            "protocol. "
            f"Available={len(healthy)}, "
            f"calibration={config.calibration_count}, "
            f"healthy_eval={config.healthy_eval_count}, "
            f"max_commissioning={config.maximum_commissioning_n}, "
            f"required_total={required_healthy}."
        )

    rng = np.random.default_rng(
        config.master_seed
    )

    healthy_ids = healthy[
        "sample_nr"
    ].to_numpy(
        dtype=np.int64
    )

    shuffled_ids = rng.permutation(
        healthy_ids
    )

    calibration_stop = (
        config.calibration_count
    )

    healthy_eval_stop = (
        calibration_stop
        + config.healthy_eval_count
    )

    calibration_ids = shuffled_ids[
        :calibration_stop
    ]

    healthy_eval_ids = shuffled_ids[
        calibration_stop:healthy_eval_stop
    ]

    commissioning_reservoir_ids = shuffled_ids[
        healthy_eval_stop:
    ]

    by_id = inventory.set_index(
        "sample_nr",
        drop=False,
    )

    calibration = (
        by_id.loc[calibration_ids]
        .reset_index(drop=True)
        .copy()
    )

    healthy_eval = (
        by_id.loc[healthy_eval_ids]
        .reset_index(drop=True)
        .copy()
    )

    commissioning_reservoir = (
        by_id.loc[commissioning_reservoir_ids]
        .reset_index(drop=True)
        .copy()
    )

    anomaly_eval = anomalies.sort_values(
        [
            "label",
            "sample_nr",
        ]
    ).reset_index(drop=True)

    return (
        calibration,
        healthy_eval,
        anomaly_eval,
        commissioning_reservoir,
    )


def build_nested_commissioning_sets(
    reservoir: pd.DataFrame,
    config: ProtocolConfig,
) -> pd.DataFrame:
    """
    Generate nested commissioning memberships independently for every seed.
    """
    reservoir_ids = reservoir[
        "sample_nr"
    ].to_numpy(
        dtype=np.int64
    )

    if len(reservoir_ids) < config.maximum_commissioning_n:
        raise ValueError(
            "Commissioning reservoir is smaller than the largest "
            f"requested N. reservoir={len(reservoir_ids)}, "
            f"max_N={config.maximum_commissioning_n}"
        )

    metadata = reservoir.set_index(
        "sample_nr",
        drop=False,
    )

    records: list[pd.DataFrame] = []

    for seed in config.seeds:
        rng = np.random.default_rng(
            seed
        )

        ordered_ids = rng.permutation(
            reservoir_ids
        )

        for n_value in config.grid:
            selected_ids = ordered_ids[
                :n_value
            ]

            selected = (
                metadata.loc[selected_ids]
                .reset_index(drop=True)
                .copy()
            )

            selected.insert(
                0,
                "seed",
                int(seed),
            )

            selected.insert(
                1,
                "commissioning_n",
                int(n_value),
            )

            selected.insert(
                2,
                "selection_rank",
                np.arange(
                    1,
                    len(selected) + 1,
                    dtype=np.int64,
                ),
            )

            selected["partition"] = (
                PARTITION_COMMISSIONING
            )

            records.append(selected)

    if not records:
        raise ValueError(
            "No commissioning memberships were created."
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def make_partition_table(
    frame: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    """Attach standard fixed-partition metadata."""
    result = frame.copy()

    result.insert(
        0,
        "seed",
        -1,
    )

    result.insert(
        1,
        "commissioning_n",
        -1,
    )

    result.insert(
        2,
        "selection_rank",
        np.arange(
            1,
            len(result) + 1,
            dtype=np.int64,
        ),
    )

    result["partition"] = partition

    return result


def assert_pairwise_disjoint(
    named_id_sets: dict[str, Iterable[int]],
) -> None:
    """Assert pairwise disjointness among named partitions."""
    normalized = {
        name: set(int(value) for value in values)
        for name, values in named_id_sets.items()
    }

    names = list(normalized)

    for left_index, left_name in enumerate(
        names
    ):
        for right_name in names[
            left_index + 1:
        ]:
            overlap = (
                normalized[left_name]
                & normalized[right_name]
            )

            if overlap:
                preview = sorted(overlap)[:20]

                raise AssertionError(
                    "Protocol leakage detected between "
                    f"{left_name!r} and {right_name!r}. "
                    f"Overlapping sample_nr values: {preview}"
                )


def validate_nestedness(
    commissioning: pd.DataFrame,
    config: ProtocolConfig,
) -> None:
    """Verify strict nestedness within each seed."""
    expected_sizes = set(
        config.grid
    )

    for seed, seed_group in commissioning.groupby(
        "seed",
        sort=True,
    ):
        observed_sizes = set(
            seed_group[
                "commissioning_n"
            ].astype(int).unique()
        )

        if observed_sizes != expected_sizes:
            raise AssertionError(
                f"Seed {seed} has commissioning sizes "
                f"{sorted(observed_sizes)}, expected "
                f"{sorted(expected_sizes)}."
            )

        previous_ids: set[int] = set()

        for n_value in config.grid:
            current_group = seed_group[
                seed_group[
                    "commissioning_n"
                ].eq(n_value)
            ]

            current_ids = set(
                current_group[
                    "sample_nr"
                ].astype(int)
            )

            if len(current_ids) != n_value:
                raise AssertionError(
                    f"Seed {seed}, N={n_value}: expected "
                    f"{n_value} unique IDs, found "
                    f"{len(current_ids)}."
                )

            if previous_ids and not previous_ids.issubset(
                current_ids
            ):
                missing = sorted(
                    previous_ids - current_ids
                )[:20]

                raise AssertionError(
                    f"Nestedness violation for seed {seed}, "
                    f"N={n_value}. Missing prior IDs: {missing}"
                )

            previous_ids = current_ids


def validate_protocol(
    calibration: pd.DataFrame,
    healthy_eval: pd.DataFrame,
    anomaly_eval: pd.DataFrame,
    commissioning: pd.DataFrame,
    reservoir: pd.DataFrame,
    config: ProtocolConfig,
) -> None:
    """Run all leakage and composition assertions."""
    calibration_ids = set(
        calibration[
            "sample_nr"
        ].astype(int)
    )

    healthy_eval_ids = set(
        healthy_eval[
            "sample_nr"
        ].astype(int)
    )

    anomaly_eval_ids = set(
        anomaly_eval[
            "sample_nr"
        ].astype(int)
    )

    reservoir_ids = set(
        reservoir[
            "sample_nr"
        ].astype(int)
    )

    assert_pairwise_disjoint(
        {
            "calibration": calibration_ids,
            "healthy_eval": healthy_eval_ids,
            "anomaly_eval": anomaly_eval_ids,
            "commissioning_reservoir": reservoir_ids,
        }
    )

    if not calibration["label"].eq(
        NORMAL_LABEL
    ).all():
        raise AssertionError(
            "Calibration contains non-normal executions."
        )

    if not healthy_eval["label"].eq(
        NORMAL_LABEL
    ).all():
        raise AssertionError(
            "Healthy evaluation contains non-normal executions."
        )

    if not reservoir["label"].eq(
        NORMAL_LABEL
    ).all():
        raise AssertionError(
            "Commissioning reservoir contains non-normal executions."
        )

    if not anomaly_eval["label"].isin(
        ANOMALY_LABELS
    ).all():
        raise AssertionError(
            "Anomaly evaluation contains unsupported labels."
        )

    if len(calibration_ids) != config.calibration_count:
        raise AssertionError(
            "Calibration partition size mismatch."
        )

    if len(healthy_eval_ids) != config.healthy_eval_count:
        raise AssertionError(
            "Healthy-evaluation partition size mismatch."
        )

    validate_nestedness(
        commissioning,
        config,
    )

    for seed in config.seeds:
        seed_rows = commissioning[
            commissioning["seed"].eq(seed)
        ]

        seed_max_ids = set(
            seed_rows.loc[
                seed_rows[
                    "commissioning_n"
                ].eq(
                    config.maximum_commissioning_n
                ),
                "sample_nr",
            ].astype(int)
        )

        if not seed_max_ids.issubset(
            reservoir_ids
        ):
            raise AssertionError(
                f"Seed {seed} commissioning IDs are not a "
                "subset of the commissioning reservoir."
            )

        assert_pairwise_disjoint(
            {
                f"seed_{seed}_commissioning": seed_max_ids,
                "calibration": calibration_ids,
                "healthy_eval": healthy_eval_ids,
                "anomaly_eval": anomaly_eval_ids,
            }
        )


def build_summary(
    calibration: pd.DataFrame,
    healthy_eval: pd.DataFrame,
    anomaly_eval: pd.DataFrame,
    commissioning: pd.DataFrame,
    excluded: pd.DataFrame,
    config: ProtocolConfig,
) -> pd.DataFrame:
    """Create a compact protocol summary table."""
    rows: list[dict[str, Any]] = []

    fixed_partitions = {
        PARTITION_CALIBRATION: calibration,
        PARTITION_HEALTHY_EVAL: healthy_eval,
        PARTITION_ANOMALY_EVAL: anomaly_eval,
        PARTITION_EXCLUDED: excluded,
    }

    for partition, frame in fixed_partitions.items():
        if frame.empty:
            rows.append(
                {
                    "partition": partition,
                    "seed": -1,
                    "commissioning_n": -1,
                    "label": None,
                    "label_name": None,
                    "execution_count": 0,
                    "total_rows": 0,
                }
            )
            continue

        grouped = (
            frame.groupby(
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
            )
            .reset_index()
        )

        grouped.insert(
            0,
            "partition",
            partition,
        )

        grouped.insert(
            1,
            "seed",
            -1,
        )

        grouped.insert(
            2,
            "commissioning_n",
            -1,
        )

        rows.extend(
            grouped.to_dict(
                orient="records"
            )
        )

    commissioning_summary = (
        commissioning.groupby(
            [
                "seed",
                "commissioning_n",
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
        )
        .reset_index()
    )

    commissioning_summary.insert(
        0,
        "partition",
        PARTITION_COMMISSIONING,
    )

    rows.extend(
        commissioning_summary.to_dict(
            orient="records"
        )
    )

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        [
            "partition",
            "seed",
            "commissioning_n",
            "label",
        ],
        na_position="last",
    ).reset_index(drop=True)


def write_seed_files(
    commissioning: pd.DataFrame,
    seeds_dir: Path,
    config: ProtocolConfig,
) -> dict[str, str]:
    """Write one long-form commissioning-membership file per seed."""
    seeds_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs: dict[str, str] = {}

    width = max(
        2,
        len(str(max(config.seeds))),
    )

    for seed in config.seeds:
        seed_frame = (
            commissioning[
                commissioning["seed"].eq(seed)
            ]
            .sort_values(
                [
                    "commissioning_n",
                    "selection_rank",
                ]
            )
            .reset_index(drop=True)
        )

        path = (
            seeds_dir
            / (
                f"seed_{seed:0{width}d}"
                "_commissioning_ids.csv"
            )
        )

        seed_frame.to_csv(
            path,
            index=False,
        )

        outputs[str(seed)] = str(path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, leakage-safe AURSAD "
            "commissioning protocol."
        )
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for protocol artifacts.",
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
            "Comma-separated seed identifiers. "
            "Default: 0 through 19"
        ),
    )

    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help=(
            "Seed used only for fixed calibration/evaluation "
            "partition construction."
        ),
    )

    parser.add_argument(
        "--calibration-count",
        type=int,
        default=600,
        help=(
            "Number of fixed healthy calibration executions. "
            "Default: 600"
        ),
    )

    parser.add_argument(
        "--healthy-eval-count",
        type=int,
        default=300,
        help=(
            "Number of fixed healthy evaluation executions. "
            "Default: 300"
        ),
    )

    parser.add_argument(
        "--timestamp-policy",
        choices=(
            "exclude",
            "keep",
            "error",
        ),
        default="exclude",
        help=(
            "How to handle executions with decreasing timestamps. "
            "Default: exclude"
        ),
    )

    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        help=(
            "Mark label-5 supplementary operations as eligible. "
            "They are still not assigned to the normal or anomaly "
            "tightening protocol by this script."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow writing into a non-empty output directory."
        ),
    )

    return parser.parse_args()


def ensure_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> None:
    """
    Protect existing protocol artifacts unless --overwrite is selected.
    """
    if output_dir.exists():
        existing = [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
        ]

        if existing and not overwrite:
            preview = "\n".join(
                f"  - {path}"
                for path in existing[:10]
            )

            raise FileExistsError(
                "Output directory already contains files. "
                "Use --overwrite to replace protocol artifacts.\n"
                f"{preview}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()

    inventory_path = (
        args.inventory_path
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

    if args.master_seed < 0:
        raise ValueError(
            "--master-seed must be non-negative."
        )

    if args.calibration_count <= 0:
        raise ValueError(
            "--calibration-count must be positive."
        )

    if args.healthy_eval_count <= 0:
        raise ValueError(
            "--healthy-eval-count must be positive."
        )

    config = ProtocolConfig(
        grid=grid,
        seeds=seeds,
        master_seed=int(
            args.master_seed
        ),
        calibration_count=int(
            args.calibration_count
        ),
        healthy_eval_count=int(
            args.healthy_eval_count
        ),
        timestamp_policy=str(
            args.timestamp_policy
        ),
        include_supplementary=bool(
            args.include_supplementary
        ),
    )

    ensure_output_directory(
        output_dir,
        overwrite=bool(
            args.overwrite
        ),
    )

    print("=" * 76)
    print("AURSAD LEAKAGE-SAFE PROTOCOL BUILDER")
    print("=" * 76)
    print(f"Inventory: {inventory_path}")
    print(f"Output:    {output_dir}")
    print(f"Grid:      {list(config.grid)}")
    print(f"Seeds:     {list(config.seeds)}")
    print(
        "Fixed healthy partitions: "
        f"calibration={config.calibration_count}, "
        f"healthy_eval={config.healthy_eval_count}"
    )
    print(
        "Timestamp policy: "
        f"{config.timestamp_policy}"
    )

    inventory_hash = sha256_file(
        inventory_path
    )

    inventory = load_inventory(
        inventory_path
    )

    classified = classify_eligibility(
        inventory=inventory,
        timestamp_policy=(
            config.timestamp_policy
        ),
        include_supplementary=(
            config.include_supplementary
        ),
    )

    excluded = classified[
        ~classified["protocol_eligible"]
    ].copy()

    (
        calibration,
        healthy_eval,
        anomaly_eval,
        commissioning_reservoir,
    ) = split_fixed_partitions(
        classified,
        config,
    )

    commissioning = (
        build_nested_commissioning_sets(
            reservoir=commissioning_reservoir,
            config=config,
        )
    )

    validate_protocol(
        calibration=calibration,
        healthy_eval=healthy_eval,
        anomaly_eval=anomaly_eval,
        commissioning=commissioning,
        reservoir=commissioning_reservoir,
        config=config,
    )

    calibration_table = (
        make_partition_table(
            calibration,
            PARTITION_CALIBRATION,
        )
    )

    healthy_eval_table = (
        make_partition_table(
            healthy_eval,
            PARTITION_HEALTHY_EVAL,
        )
    )

    anomaly_eval_table = (
        make_partition_table(
            anomaly_eval,
            PARTITION_ANOMALY_EVAL,
        )
    )

    excluded_table = (
        make_partition_table(
            excluded,
            PARTITION_EXCLUDED,
        )
    )

    membership = pd.concat(
        [
            commissioning,
            calibration_table,
            healthy_eval_table,
            anomaly_eval_table,
            excluded_table,
        ],
        ignore_index=True,
        sort=False,
    )

    membership = membership.sort_values(
        [
            "partition",
            "seed",
            "commissioning_n",
            "selection_rank",
            "sample_nr",
        ]
    ).reset_index(drop=True)

    summary = build_summary(
        calibration=calibration,
        healthy_eval=healthy_eval,
        anomaly_eval=anomaly_eval,
        commissioning=commissioning,
        excluded=excluded,
        config=config,
    )

    membership_path = (
        output_dir
        / "aursad_protocol_membership.csv"
    )

    commissioning_path = (
        output_dir
        / "commissioning_ids.csv"
    )

    calibration_path = (
        output_dir
        / "calibration_ids.csv"
    )

    healthy_eval_path = (
        output_dir
        / "healthy_eval_ids.csv"
    )

    anomaly_eval_path = (
        output_dir
        / "anomaly_eval_ids.csv"
    )

    excluded_path = (
        output_dir
        / "excluded_ids.csv"
    )

    summary_path = (
        output_dir
        / "protocol_summary.csv"
    )

    manifest_path = (
        output_dir
        / "protocol_manifest.json"
    )

    seeds_dir = (
        output_dir
        / "seeds"
    )

    membership.to_csv(
        membership_path,
        index=False,
    )

    commissioning.to_csv(
        commissioning_path,
        index=False,
    )

    calibration_table.to_csv(
        calibration_path,
        index=False,
    )

    healthy_eval_table.to_csv(
        healthy_eval_path,
        index=False,
    )

    anomaly_eval_table.to_csv(
        anomaly_eval_path,
        index=False,
    )

    excluded_table.to_csv(
        excluded_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    seed_outputs = write_seed_files(
        commissioning=commissioning,
        seeds_dir=seeds_dir,
        config=config,
    )

    anomaly_counts = (
        anomaly_eval.groupby(
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

    nonmonotonic = classified[
        ~classified[
            "timestamps_monotonic_nondecreasing"
        ]
    ][
        [
            "sample_nr",
            "label",
            "label_name",
            "protocol_eligible",
            "exclusion_reason",
        ]
    ].to_dict(
        orient="records"
    )

    manifest = {
        "protocol_version": (
            "aursad-commissioning-protocol-v1"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "command": " ".join(
            sys.argv
        ),
        "input": {
            "episode_inventory_path": str(
                inventory_path
            ),
            "episode_inventory_sha256": (
                inventory_hash
            ),
            "episode_count": int(
                len(inventory)
            ),
        },
        "episode_unit": "sample_nr",
        "configuration": {
            "commissioning_grid": list(
                config.grid
            ),
            "seeds": list(
                config.seeds
            ),
            "master_seed": int(
                config.master_seed
            ),
            "calibration_count": int(
                config.calibration_count
            ),
            "healthy_eval_count": int(
                config.healthy_eval_count
            ),
            "maximum_commissioning_n": int(
                config.maximum_commissioning_n
            ),
            "timestamp_policy": (
                config.timestamp_policy
            ),
            "include_supplementary": (
                config.include_supplementary
            ),
            "normal_label": NORMAL_LABEL,
            "anomaly_labels": list(
                ANOMALY_LABELS
            ),
            "supplementary_label": (
                SUPPLEMENTARY_LABEL
            ),
        },
        "partition_counts": {
            "calibration": int(
                calibration[
                    "sample_nr"
                ].nunique()
            ),
            "healthy_eval": int(
                healthy_eval[
                    "sample_nr"
                ].nunique()
            ),
            "anomaly_eval": int(
                anomaly_eval[
                    "sample_nr"
                ].nunique()
            ),
            "commissioning_reservoir": int(
                commissioning_reservoir[
                    "sample_nr"
                ].nunique()
            ),
            "excluded": int(
                excluded[
                    "sample_nr"
                ].nunique()
            ),
        },
        "anomaly_evaluation_counts": (
            anomaly_counts
        ),
        "nonmonotonic_executions": (
            nonmonotonic
        ),
        "validation": {
            "episode_level_membership": True,
            "fixed_calibration_across_seeds": True,
            "fixed_evaluation_across_seeds": True,
            "nested_commissioning_within_seed": True,
            "pairwise_partition_overlap_count": 0,
            "commissioning_contains_only_label_0": True,
            "calibration_contains_only_label_0": True,
            "healthy_eval_contains_only_label_0": True,
            "anomaly_eval_contains_only_labels_1_to_4": True,
        },
        "outputs": {
            "membership": str(
                membership_path
            ),
            "commissioning": str(
                commissioning_path
            ),
            "calibration": str(
                calibration_path
            ),
            "healthy_eval": str(
                healthy_eval_path
            ),
            "anomaly_eval": str(
                anomaly_eval_path
            ),
            "excluded": str(
                excluded_path
            ),
            "summary": str(
                summary_path
            ),
            "seed_commissioning_files": (
                seed_outputs
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
        "protocol_notes": [
            (
                "All memberships are defined at complete sample_nr "
                "execution level."
            ),
            (
                "Calibration and evaluation sets are fixed across "
                "commissioning seeds."
            ),
            (
                "Commissioning sets are nested within each seed."
            ),
            (
                "All eligible anomaly tightening executions are used "
                "for anomaly evaluation."
            ),
            (
                "Supplementary-operation label 5 is not used as "
                "healthy tightening data."
            ),
            (
                "Damaged-thread label 4 has only three executions; "
                "per-class estimates for this label are underpowered."
            ),
            (
                "The protocol manifest records the timestamp policy "
                "and every affected execution."
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
    print("PROTOCOL SUMMARY")
    print("=" * 76)
    print(
        f"Calibration executions: "
        f"{len(calibration):,}"
    )
    print(
        f"Healthy evaluation executions: "
        f"{len(healthy_eval):,}"
    )
    print(
        f"Anomaly evaluation executions: "
        f"{len(anomaly_eval):,}"
    )
    print(
        "Commissioning reservoir executions: "
        f"{len(commissioning_reservoir):,}"
    )
    print(
        f"Excluded executions: "
        f"{len(excluded):,}"
    )
    print(
        "Per-seed maximum commissioning N: "
        f"{config.maximum_commissioning_n}"
    )
    print(
        "Seeds generated: "
        f"{len(config.seeds)}"
    )
    print(
        "Leakage checks: PASSED"
    )
    print(
        "Nestedness checks: PASSED"
    )

    print("\nAnomaly evaluation counts:")
    print(
        anomaly_eval.groupby(
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

    print("\nArtifacts:")
    for path in (
        membership_path,
        commissioning_path,
        calibration_path,
        healthy_eval_path,
        anomaly_eval_path,
        excluded_path,
        summary_path,
        manifest_path,
    ):
        print(f"  {path}")

    print(
        f"  {seeds_dir}"
    )


if __name__ == "__main__":
    main()