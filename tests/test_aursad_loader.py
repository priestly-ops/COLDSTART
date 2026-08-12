from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.aursad_loader import (
    AURSAD_MEASURED_SIGNAL_COLUMNS,
    load_cycles,
    load_executions,
)


def _write_fixture_h5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group("complete_data")

        signal_names = list(AURSAD_MEASURED_SIGNAL_COLUMNS)
        rows = []
        for sample_nr, label, timestamp, runtime_state in [(100, 0, 1.0, 0), (100, 0, 1.1, 0), (200, 1, 2.0, 1), (200, 1, 2.1, 1)]:
            row = [sample_nr, label, timestamp, runtime_state]
            row.extend(float(value + 1) for value in range(len(signal_names)))
            rows.append(row)

        group.create_dataset(
            "block0_values",
            data=np.array(rows, dtype=np.float64),
        )
        group.create_dataset(
            "block0_items",
            data=np.array(
                [
                    b"sample_nr",
                    b"label",
                    b"timestamp",
                    b"runtime_state",
                    *[name.encode("utf-8") for name in signal_names],
                ],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )
        group.create_dataset("axis1", data=np.array([0, 1], dtype=np.int64))


def _write_inventory(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "sample_nr": 100,
                "label": 0,
                "label_name": "normal",
                "row_count": 2,
                "first_global_row": 0,
                "last_global_row": 1,
                "has_single_label": True,
                "rows_are_contiguous": True,
                "timestamps_monotonic_nondecreasing": True,
            }
        ]
    ).to_csv(path, index=False)


def test_load_cycles_reads_protocol_selected_executions(tmp_path: Path):
    data_path = tmp_path / "sample_aursad.h5"
    protocol_path = tmp_path / "protocol.csv"
    inventory_path = tmp_path / "inventory.csv"
    _write_fixture_h5(data_path)
    _write_inventory(inventory_path)

    pd.DataFrame({"sample_nr": [100]}).to_csv(protocol_path, index=False)

    cycles = load_cycles(
        path=data_path,
        signal_set="measured",
        protocol_paths=[protocol_path],
        inventory_path=inventory_path,
    )

    assert len(cycles) == 1
    assert cycles[0].episode_id == 100
    assert cycles[0].values.shape == (2, 48)
    assert cycles[0].columns[0] == "actual_q_0"
    assert cycles[0].anomaly is False
    assert cycles[0].category == 0


def test_load_executions_alias_loads_requested_ids(tmp_path: Path):
    data_path = tmp_path / "sample_aursad.h5"
    inventory_path = tmp_path / "inventory.csv"
    _write_fixture_h5(data_path)
    _write_inventory(inventory_path)

    cycles = load_executions(episode_ids=[100], data_path=data_path, inventory_path=inventory_path)

    assert len(cycles) == 1
    assert cycles[0].episode_id == 100
