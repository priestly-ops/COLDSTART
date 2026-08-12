from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.signal_policy import (
    select_machine_signals,
    select_measured_signals,
)


@dataclass(frozen=True)
class RobotCycle:
    episode_id: int
    values: np.ndarray
    columns: tuple[str, ...]
    anomaly: bool
    category: int
    setting: int


def get_dataset_columns(path: Path) -> list[str]:
    parquet = pq.ParquetFile(path)
    return list(parquet.schema.names)


def select_signal_columns(
    path: Path,
    signal_set: str = "measured",
) -> list[str]:
    columns = get_dataset_columns(path)

    if signal_set == "measured":
        return select_measured_signals(columns)

    if signal_set == "machine":
        return select_machine_signals(columns)

    raise ValueError(
        f"Unknown signal_set={signal_set!r}. "
        "Expected 'measured' or 'machine'."
    )


def load_cycles(
    path: Path,
    signal_set: str = "measured",
    episode_ids: Sequence[int] | None = None,
) -> list[RobotCycle]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    signal_columns = select_signal_columns(
        path=path,
        signal_set=signal_set,
    )

    metadata_columns = [
        "sample",
        "anomaly",
        "category",
        "setting",
    ]

    selected_columns = metadata_columns + signal_columns

    if episode_ids is not None:
        episode_id_set = {
            int(value)
            for value in episode_ids
        }
        filters = [
            (
                "sample",
                "in",
                sorted(episode_id_set),
            )
        ]
    else:
        episode_id_set = None
        filters = None

    if episode_id_set is not None:
        frames: list[pd.DataFrame] = []
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=selected_columns,
            batch_size=250_000,
        ):
            chunk = batch.to_pandas()
            chunk = chunk[
                chunk["sample"].isin(episode_id_set)
            ]
            if not chunk.empty:
                frames.append(chunk)
        if frames:
            dataframe = pd.concat(
                frames,
                ignore_index=True,
            )
        else:
            dataframe = pd.DataFrame(
                columns=selected_columns,
            )
    else:
        dataframe = pd.read_parquet(
            path,
            columns=selected_columns,
            filters=filters,
        )

    if episode_id_set is not None:
        dataframe = dataframe[
            dataframe["sample"].isin(episode_id_set)
        ]

    cycles: list[RobotCycle] = []

    for episode_id, cycle_df in dataframe.groupby(
        "sample",
        sort=True,
    ):
        for column in [
            "anomaly",
            "category",
            "setting",
        ]:
            unique_count = cycle_df[column].nunique(
                dropna=False
            )

            if unique_count != 1:
                raise ValueError(
                    f"Episode {episode_id} has "
                    f"{unique_count} values for {column}."
                )

        values = cycle_df[
            signal_columns
        ].to_numpy(dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                f"Episode {episode_id} has invalid shape "
                f"{values.shape}."
            )

        if len(values) == 0:
            raise ValueError(
                f"Episode {episode_id} is empty."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Episode {episode_id} contains NaN or Inf."
            )

        cycles.append(
            RobotCycle(
                episode_id=int(episode_id),
                values=values,
                columns=tuple(signal_columns),
                anomaly=bool(
                    cycle_df["anomaly"].iloc[0]
                ),
                category=int(
                    cycle_df["category"].iloc[0]
                ),
                setting=int(
                    cycle_df["setting"].iloc[0]
                ),
            )
        )

    if not cycles:
        raise ValueError("No cycles were loaded.")

    return cycles


def load_cycle_metadata(path: Path) -> list[RobotCycle]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataframe = pd.read_parquet(
        path,
        columns=[
            "sample",
            "anomaly",
            "category",
            "setting",
        ],
    )

    cycles: list[RobotCycle] = []
    for episode_id, cycle_df in dataframe.groupby(
        "sample",
        sort=True,
    ):
        for column in [
            "anomaly",
            "category",
            "setting",
        ]:
            unique_count = cycle_df[column].nunique(
                dropna=False
            )
            if unique_count != 1:
                raise ValueError(
                    f"Episode {episode_id} has "
                    f"{unique_count} values for {column}."
                )

        cycles.append(
            RobotCycle(
                episode_id=int(episode_id),
                values=np.empty((0, 0), dtype=np.float64),
                columns=(),
                anomaly=bool(
                    cycle_df["anomaly"].iloc[0]
                ),
                category=int(
                    cycle_df["category"].iloc[0]
                ),
                setting=int(
                    cycle_df["setting"].iloc[0]
                ),
            )
        )

    if not cycles:
        raise ValueError("No cycle metadata were loaded.")

    return cycles
