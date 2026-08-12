from __future__ import annotations

"""Efficient episode-level loader for the raw AURSAD HDF5 dataset.

The raw dataset is a Pandas/PyTables table stored under ``/complete_data``.
Logical columns are distributed across dtype-specific ``block*_values``
datasets, with names stored in matching ``block*_items`` datasets.

This module never loads the complete 6 GB table. It uses the episode audit
inventory to locate each complete ``sample_nr`` execution and reads only the
requested rows and signal columns.

The public ``load_cycles`` function returns the same ``RobotCycle`` dataclass
used by ``src.voraus_loader`` so existing feature extraction and evaluation
code can be reused without conversion.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd



@dataclass(frozen=True)
class RobotCycle:
    """Dataset-neutral cycle container matching src.voraus_loader.RobotCycle."""

    episode_id: int
    values: np.ndarray
    columns: tuple[str, ...]
    anomaly: bool
    category: int
    setting: int


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "aursad" / "AURSAD.h5"
)
DEFAULT_INVENTORY_PATH = (
    PROJECT_ROOT / "reports" / "aursad" / "aursad_episode_inventory.csv"
)
DEFAULT_PROTOCOL_DIR = (
    PROJECT_ROOT / "reports" / "aursad" / "protocol"
)

HDF5_GROUP = "/complete_data"

# These columns describe an execution rather than sensor measurements.
NON_SIGNAL_COLUMNS = frozenset(
    {
        "sample_nr",
        "label",
        "timestamp",
        "runtime_state",
        "index",
        "axis1",
    }
)

AURSAD_MEASURED_SIGNAL_COLUMNS = (
    "actual_q_0",
    "actual_q_1",
    "actual_q_2",
    "actual_q_3",
    "actual_q_4",
    "actual_q_5",
    "actual_qd_0",
    "actual_qd_1",
    "actual_qd_2",
    "actual_qd_3",
    "actual_qd_4",
    "actual_qd_5",
    "actual_current_0",
    "actual_current_1",
    "actual_current_2",
    "actual_current_3",
    "actual_current_4",
    "actual_current_5",
    "actual_TCP_pose_0",
    "actual_TCP_pose_1",
    "actual_TCP_pose_2",
    "actual_TCP_pose_3",
    "actual_TCP_pose_4",
    "actual_TCP_pose_5",
    "actual_TCP_speed_0",
    "actual_TCP_speed_1",
    "actual_TCP_speed_2",
    "actual_TCP_speed_3",
    "actual_TCP_speed_4",
    "actual_TCP_speed_5",
    "actual_TCP_force_0",
    "actual_TCP_force_1",
    "actual_TCP_force_2",
    "actual_TCP_force_3",
    "actual_TCP_force_4",
    "actual_TCP_force_5",
    "actual_tool_accelerometer_0",
    "actual_tool_accelerometer_1",
    "actual_tool_accelerometer_2",
    "actual_robot_current",
    "actual_main_voltage",
    "actual_robot_voltage",
    "actual_joint_voltage_0",
    "actual_joint_voltage_1",
    "actual_joint_voltage_2",
    "actual_joint_voltage_3",
    "actual_joint_voltage_4",
    "actual_joint_voltage_5",
)

# Name fragments used for optional signal subsets. The default ``measured``
# policy now uses the explicit AURSAD signal list above.
SCREWDRIVER_TERMS = (
    "screwdriver",
    "current",
    "voltage",
)
ROBOT_TERMS = (
    "joint",
    "q_",
    "qd_",
    "actual",
    "torque",
    "force",
    "speed",
    "velocity",
    "position",
    "tcp",
)


@dataclass(frozen=True)
class ColumnLocation:
    """Physical location of one logical column in the PyTables layout."""

    name: str
    block_name: str
    block_column_index: int
    dtype: np.dtype


@dataclass(frozen=True)
class EpisodeSpan:
    """Inclusive global-row span and audit metadata for one execution."""

    sample_nr: int
    first_row: int
    last_row: int
    row_count: int
    label: int
    label_name: str
    timestamps_monotonic: bool

    @property
    def stop_row(self) -> int:
        """Exclusive stop row suitable for NumPy/HDF5 slicing."""
        return self.last_row + 1


@dataclass(frozen=True)
class ProtocolSelection:
    """Resolved protocol membership used to load a set of executions."""

    path: Path
    episode_ids: tuple[int, ...]
    partition: str | None
    seed: int | None
    commissioning_n: int | None


def _decode(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _coerce_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized.unique()) - set(mapping))
    if unknown:
        raise ValueError(
            f"Column {name!r} contains unsupported boolean values: "
            f"{unknown[:10]}"
        )
    return normalized.map(mapping).astype(bool)


def read_column_locations(handle: h5py.File) -> dict[str, ColumnLocation]:
    """Reconstruct logical columns from the Pandas/PyTables HDF5 blocks."""
    if HDF5_GROUP not in handle:
        raise ValueError(f"Missing required HDF5 group {HDF5_GROUP!r}.")

    group = handle[HDF5_GROUP]
    locations: dict[str, ColumnLocation] = {}

    item_names = sorted(
        name
        for name in group.keys()
        if re.fullmatch(r"block\d+_items", name)
    )
    if not item_names:
        raise ValueError(
            f"No block*_items datasets were found under {HDF5_GROUP}."
        )

    for items_name in item_names:
        prefix = items_name[: -len("_items")]
        values_name = f"{prefix}_values"
        if values_name not in group:
            raise ValueError(
                f"Missing values dataset {HDF5_GROUP}/{values_name}."
            )

        values = group[values_name]
        names = [_decode(value) for value in group[items_name][:]]

        if values.ndim != 2:
            raise ValueError(
                f"{values_name} must be two-dimensional; got {values.shape}."
            )
        if len(names) != values.shape[1]:
            raise ValueError(
                f"{items_name} has {len(names)} names but {values_name} "
                f"has {values.shape[1]} columns."
            )

        for index, name in enumerate(names):
            if name in locations:
                raise ValueError(f"Duplicate logical column name: {name!r}.")
            locations[name] = ColumnLocation(
                name=name,
                block_name=values_name,
                block_column_index=index,
                dtype=np.dtype(values.dtype),
            )

    return locations


def get_dataset_columns(path: Path | str) -> list[str]:
    """Return logical AURSAD columns without reading table rows."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AURSAD dataset not found: {path}")

    with h5py.File(path, "r") as handle:
        return list(read_column_locations(handle))


def _is_numeric_location(location: ColumnLocation) -> bool:
    return np.issubdtype(location.dtype, np.number)


def select_signal_columns(
    path: Path | str,
    signal_set: str = "measured",
    signal_columns: Sequence[str] | None = None,
) -> list[str]:
    """Select sensor columns while excluding execution metadata.

    Parameters
    ----------
    path:
        Raw ``AURSAD.h5`` path.
    signal_set:
        ``"measured"`` or ``"all"`` selects all numeric non-metadata
        columns. ``"screwdriver"`` selects names associated with screwdriver
        current/voltage channels. ``"robot"`` selects robot-state names.
    signal_columns:
        Optional explicit logical column list. When provided, it overrides
        ``signal_set`` and is validated against the HDF5 schema.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AURSAD dataset not found: {path}")

    with h5py.File(path, "r") as handle:
        locations = read_column_locations(handle)

    if signal_columns is not None:
        requested = [str(name) for name in signal_columns]
        if not requested:
            raise ValueError("signal_columns cannot be empty.")
        if len(set(requested)) != len(requested):
            raise ValueError("signal_columns contains duplicate names.")

        missing = [name for name in requested if name not in locations]
        if missing:
            raise ValueError(f"Unknown AURSAD signal columns: {missing}")

        nonnumeric = [
            name for name in requested if not _is_numeric_location(locations[name])
        ]
        if nonnumeric:
            raise ValueError(f"Requested signal columns are nonnumeric: {nonnumeric}")

        metadata = [name for name in requested if _normalize(name) in NON_SIGNAL_COLUMNS]
        if metadata:
            raise ValueError(
                f"Execution metadata cannot be used as signals: {metadata}"
            )
        return requested

    candidates = [
        name
        for name, location in locations.items()
        if _is_numeric_location(location)
        and _normalize(name) not in NON_SIGNAL_COLUMNS
    ]

    policy = signal_set.strip().lower()
    if policy in {"measured", "all", "all_numeric"}:
        if policy == "measured":
            selected = list(AURSAD_MEASURED_SIGNAL_COLUMNS)
            available_columns = set(locations)
            missing = [
                name
                for name in AURSAD_MEASURED_SIGNAL_COLUMNS
                if name not in available_columns
            ]
            if missing:
                raise ValueError(
                    f"Missing required AURSAD measured signals: {missing}"
                )
        else:
            selected = candidates
    elif policy == "screwdriver":
        selected = [
            name
            for name in candidates
            if any(term in _normalize(name) for term in SCREWDRIVER_TERMS)
        ]
    elif policy == "robot":
        selected = [
            name
            for name in candidates
            if any(term in _normalize(name) for term in ROBOT_TERMS)
        ]
    else:
        raise ValueError(
            f"Unknown signal_set={signal_set!r}. Expected one of "
            "'measured', 'all', 'screwdriver', or 'robot'."
        )

    if not selected:
        raise ValueError(
            f"Signal policy {signal_set!r} selected no columns. "
            "Use get_dataset_columns() and pass signal_columns explicitly."
        )

    return selected


def load_episode_inventory(
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
) -> pd.DataFrame:
    """Load and validate the execution inventory produced by the audit."""
    path = Path(inventory_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "AURSAD episode inventory not found. Run "
            "experiments/audit_aursad_episodes.py first: "
            f"{path}"
        )

    inventory = pd.read_csv(path)
    required = {
        "sample_nr",
        "label",
        "label_name",
        "row_count",
        "first_global_row",
        "last_global_row",
        "has_single_label",
        "rows_are_contiguous",
        "timestamps_monotonic_nondecreasing",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Episode inventory is missing columns: {missing}")

    inventory = inventory.copy()
    for name in (
        "sample_nr",
        "label",
        "row_count",
        "first_global_row",
        "last_global_row",
    ):
        inventory[name] = pd.to_numeric(inventory[name], errors="raise").astype(np.int64)

    for name in (
        "has_single_label",
        "rows_are_contiguous",
        "timestamps_monotonic_nondecreasing",
    ):
        inventory[name] = _coerce_bool(inventory[name], name)

    if inventory["sample_nr"].duplicated().any():
        duplicates = inventory.loc[
            inventory["sample_nr"].duplicated(keep=False), "sample_nr"
        ].tolist()
        raise ValueError(f"Duplicate sample_nr values in inventory: {duplicates[:20]}")

    expected_count = inventory["last_global_row"] - inventory["first_global_row"] + 1
    inconsistent = inventory["row_count"].ne(expected_count)
    if inconsistent.any():
        bad = inventory.loc[
            inconsistent,
            ["sample_nr", "row_count", "first_global_row", "last_global_row"],
        ].head(20)
        raise ValueError(
            "Inventory row spans disagree with row_count:\n"
            + bad.to_string(index=False)
        )

    return inventory.sort_values("sample_nr").reset_index(drop=True)


def build_episode_spans(
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    *,
    require_single_label: bool = True,
    require_contiguous: bool = True,
    require_monotonic_timestamps: bool = False,
) -> dict[int, EpisodeSpan]:
    """Build a sample_nr-to-row-span lookup from the audit inventory."""
    inventory = load_episode_inventory(inventory_path)

    eligible = pd.Series(True, index=inventory.index)
    if require_single_label:
        eligible &= inventory["has_single_label"]
    if require_contiguous:
        eligible &= inventory["rows_are_contiguous"]
    if require_monotonic_timestamps:
        eligible &= inventory["timestamps_monotonic_nondecreasing"]

    spans: dict[int, EpisodeSpan] = {}
    for row in inventory.loc[eligible].itertuples(index=False):
        sample_nr = int(row.sample_nr)
        spans[sample_nr] = EpisodeSpan(
            sample_nr=sample_nr,
            first_row=int(row.first_global_row),
            last_row=int(row.last_global_row),
            row_count=int(row.row_count),
            label=int(row.label),
            label_name=str(row.label_name),
            timestamps_monotonic=bool(row.timestamps_monotonic_nondecreasing),
        )

    if not spans:
        raise ValueError("No eligible AURSAD executions remain after validation.")
    return spans


def _validate_requested_ids(
    episode_ids: Sequence[int] | np.ndarray,
    spans: Mapping[int, EpisodeSpan],
) -> tuple[int, ...]:
    ids = tuple(int(value) for value in episode_ids)
    if not ids:
        raise ValueError("episode_ids cannot be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("episode_ids contains duplicates.")

    missing = sorted(set(ids) - set(spans))
    if missing:
        raise ValueError(
            "Requested sample_nr values are absent or ineligible according "
            f"to the inventory: {missing[:20]}"
        )
    return ids


def _read_episode_values(
    group: h5py.Group,
    locations: Mapping[str, ColumnLocation],
    signal_columns: Sequence[str],
    span: EpisodeSpan,
) -> np.ndarray:
    """Read one execution, grouping requested columns by physical block."""
    by_block: dict[str, list[ColumnLocation]] = {}
    for name in signal_columns:
        location = locations[name]
        by_block.setdefault(location.block_name, []).append(location)

    output = np.empty(
        (span.row_count, len(signal_columns)),
        dtype=np.float64,
    )
    output_positions = {name: index for index, name in enumerate(signal_columns)}

    for block_name, block_locations in by_block.items():
        # h5py fancy column indexing requires increasing physical indices.
        ordered = sorted(block_locations, key=lambda item: item.block_column_index)
        physical_indices = [item.block_column_index for item in ordered]
        block_chunk = group[block_name][
            span.first_row : span.stop_row,
            physical_indices,
        ]
        if block_chunk.ndim == 1:
            block_chunk = block_chunk[:, None]

        for local_index, location in enumerate(ordered):
            output[:, output_positions[location.name]] = np.asarray(
                block_chunk[:, local_index], dtype=np.float64
            )

    return output


def load_cycles(
    path: Path | str = DEFAULT_DATA_PATH,
    signal_set: str = "measured",
    episode_ids: Sequence[int] | None = None,
    *,
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    signal_columns: Sequence[str] | None = None,
    require_monotonic_timestamps: bool = False,
    preserve_requested_order: bool = True,
    protocol_paths: Sequence[Path | str] | None = None,
) -> list[RobotCycle]:
    """Load complete AURSAD executions as voraus-compatible RobotCycle objects.

    ``category`` is the original AURSAD integer label, ``anomaly`` is
    ``label != 0``, and ``setting`` is set to 0 because AURSAD does not expose
    the voraus experimental-setting field.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"AURSAD dataset not found: {path}")

    spans = build_episode_spans(
        inventory_path,
        require_single_label=True,
        require_contiguous=True,
        require_monotonic_timestamps=require_monotonic_timestamps,
    )

    if protocol_paths:
        protocol_ids: set[int] = set()
        for raw_path in protocol_paths:
            protocol_ids.update(read_protocol_ids(raw_path).episode_ids)
        if episode_ids is None:
            requested_ids = tuple(sorted(protocol_ids & set(spans)))
        else:
            requested_ids = tuple(
                int(value)
                for value in episode_ids
                if int(value) in protocol_ids and int(value) in spans
            )
            if not preserve_requested_order:
                requested_ids = tuple(sorted(requested_ids))
    elif episode_ids is None:
        requested_ids = tuple(sorted(spans))
    else:
        requested_ids = _validate_requested_ids(episode_ids, spans)
        if not preserve_requested_order:
            requested_ids = tuple(sorted(requested_ids))

    selected_columns = select_signal_columns(
        path,
        signal_set=signal_set,
        signal_columns=signal_columns,
    )

    cycles: list[RobotCycle] = []
    with h5py.File(path, "r") as handle:
        group = handle[HDF5_GROUP]
        locations = read_column_locations(handle)

        missing = [name for name in selected_columns if name not in locations]
        if missing:
            raise RuntimeError(f"Selected signal columns disappeared: {missing}")

        for sample_nr in requested_ids:
            span = spans[sample_nr]
            values = _read_episode_values(
                group,
                locations,
                selected_columns,
                span,
            )

            if values.shape != (span.row_count, len(selected_columns)):
                raise RuntimeError(
                    f"Execution {sample_nr} loaded shape {values.shape}; "
                    f"expected {(span.row_count, len(selected_columns))}."
                )
            if values.shape[0] < 2:
                raise ValueError(
                    f"Execution {sample_nr} has fewer than two rows."
                )
            if not np.isfinite(values).all():
                bad_count = int((~np.isfinite(values)).sum())
                raise ValueError(
                    f"Execution {sample_nr} contains {bad_count} NaN/Inf "
                    "signal values."
                )

            cycles.append(
                RobotCycle(
                    episode_id=sample_nr,
                    values=values,
                    columns=tuple(selected_columns),
                    anomaly=span.label != 0,
                    category=span.label,
                    setting=0,
                )
            )

    if not cycles:
        raise ValueError("No AURSAD cycles were loaded.")
    return cycles


def load_execution(
    sample_nr: int,
    path: Path | str = DEFAULT_DATA_PATH,
    signal_set: str = "measured",
    *,
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    signal_columns: Sequence[str] | None = None,
    require_monotonic_timestamps: bool = False,
) -> RobotCycle:
    """Convenience wrapper that loads exactly one complete execution."""
    return load_cycles(
        path=path,
        signal_set=signal_set,
        episode_ids=[int(sample_nr)],
        inventory_path=inventory_path,
        signal_columns=signal_columns,
        require_monotonic_timestamps=require_monotonic_timestamps,
    )[0]


def load_executions(
    episode_ids: Sequence[int],
    data_path: Path | str = DEFAULT_DATA_PATH,
    signal_set: str = "measured",
    *,
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    signal_columns: Sequence[str] | None = None,
    require_monotonic_timestamps: bool = False,
) -> list[RobotCycle]:
    """Compatibility wrapper used by the feature-cache experiment."""
    return load_cycles(
        path=data_path,
        signal_set=signal_set,
        episode_ids=episode_ids,
        inventory_path=inventory_path,
        signal_columns=signal_columns,
        require_monotonic_timestamps=require_monotonic_timestamps,
    )


def read_protocol_ids(
    protocol_csv: Path | str,
    *,
    seed: int | None = None,
    commissioning_n: int | None = None,
    partition: str | None = None,
) -> ProtocolSelection:
    """Resolve unique sample_nr values from a protocol membership CSV.

    For ``commissioning_ids.csv`` or the combined membership file, provide
    both ``seed`` and ``commissioning_n``. Fixed-partition files need no
    filters. Duplicate rows after filtering are rejected rather than silently
    collapsed, except where the combined membership file legitimately contains
    different commissioning N values and the caller selected one N.
    """
    path = Path(protocol_csv).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Protocol CSV not found: {path}")

    frame = pd.read_csv(path)
    if "sample_nr" not in frame.columns:
        raise ValueError(f"Protocol CSV lacks sample_nr column: {path}")

    filtered = frame.copy()

    if partition is not None:
        if "partition" not in filtered.columns:
            raise ValueError(
                "partition filter was requested but the CSV has no "
                "partition column."
            )
        filtered = filtered[filtered["partition"].eq(partition)]

    if seed is not None:
        if "seed" not in filtered.columns:
            raise ValueError("seed filter requested but CSV has no seed column.")
        filtered = filtered[
            pd.to_numeric(filtered["seed"], errors="raise").eq(int(seed))
        ]

    if commissioning_n is not None:
        if "commissioning_n" not in filtered.columns:
            raise ValueError(
                "commissioning_n filter requested but CSV has no "
                "commissioning_n column."
            )
        filtered = filtered[
            pd.to_numeric(
                filtered["commissioning_n"], errors="raise"
            ).eq(int(commissioning_n))
        ]

    if filtered.empty:
        raise ValueError(
            f"No protocol rows matched path={path}, partition={partition!r}, "
            f"seed={seed}, commissioning_n={commissioning_n}."
        )

    ids = pd.to_numeric(filtered["sample_nr"], errors="raise").astype(np.int64)
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].astype(int).tolist()
        raise ValueError(
            "Filtered protocol membership contains duplicate sample_nr "
            f"values: {duplicates[:20]}"
        )

    return ProtocolSelection(
        path=path,
        episode_ids=tuple(ids.astype(int).tolist()),
        partition=partition,
        seed=None if seed is None else int(seed),
        commissioning_n=(
            None if commissioning_n is None else int(commissioning_n)
        ),
    )


def load_protocol_cycles(
    protocol_csv: Path | str,
    *,
    data_path: Path | str = DEFAULT_DATA_PATH,
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    seed: int | None = None,
    commissioning_n: int | None = None,
    partition: str | None = None,
    signal_set: str = "measured",
    signal_columns: Sequence[str] | None = None,
    require_monotonic_timestamps: bool = False,
) -> list[RobotCycle]:
    """Load only executions selected by a protocol CSV membership."""
    selection = read_protocol_ids(
        protocol_csv,
        seed=seed,
        commissioning_n=commissioning_n,
        partition=partition,
    )
    return load_cycles(
        path=data_path,
        signal_set=signal_set,
        episode_ids=selection.episode_ids,
        inventory_path=inventory_path,
        signal_columns=signal_columns,
        require_monotonic_timestamps=require_monotonic_timestamps,
        preserve_requested_order=True,
    )


def load_named_protocol_split(
    split_name: str,
    *,
    protocol_dir: Path | str = DEFAULT_PROTOCOL_DIR,
    data_path: Path | str = DEFAULT_DATA_PATH,
    inventory_path: Path | str = DEFAULT_INVENTORY_PATH,
    seed: int | None = None,
    commissioning_n: int | None = None,
    signal_set: str = "measured",
    signal_columns: Sequence[str] | None = None,
) -> list[RobotCycle]:
    """Load a standard protocol split by a concise stable name.

    Valid names are ``commissioning``, ``calibration``, ``healthy_eval``, and
    ``anomaly_eval``.
    """
    filenames = {
        "commissioning": "commissioning_ids.csv",
        "calibration": "calibration_ids.csv",
        "healthy_eval": "healthy_eval_ids.csv",
        "anomaly_eval": "anomaly_eval_ids.csv",
    }
    if split_name not in filenames:
        raise ValueError(
            f"Unknown split_name={split_name!r}. Expected one of "
            f"{sorted(filenames)}."
        )

    if split_name == "commissioning":
        if seed is None or commissioning_n is None:
            raise ValueError(
                "commissioning split requires both seed and commissioning_n."
            )
    elif seed is not None or commissioning_n is not None:
        raise ValueError(
            f"Fixed split {split_name!r} does not accept seed or "
            "commissioning_n."
        )

    protocol_path = Path(protocol_dir).expanduser().resolve() / filenames[split_name]
    return load_protocol_cycles(
        protocol_path,
        data_path=data_path,
        inventory_path=inventory_path,
        seed=seed,
        commissioning_n=commissioning_n,
        signal_set=signal_set,
        signal_columns=signal_columns,
    )


def assert_disjoint_cycles(*cycle_groups: Iterable[RobotCycle]) -> None:
    """Assert that multiple loaded split collections share no episode IDs."""
    seen: set[int] = set()
    for group_index, cycles in enumerate(cycle_groups):
        current = {int(cycle.episode_id) for cycle in cycles}
        overlap = seen & current
        if overlap:
            raise AssertionError(
                f"Cycle group {group_index} overlaps an earlier group: "
                f"{sorted(overlap)[:20]}"
            )
        seen.update(current)