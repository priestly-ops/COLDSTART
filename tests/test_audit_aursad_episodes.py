import h5py
import numpy as np

from experiments.audit_aursad_episodes import ColumnLocation, read_required_chunk


def test_read_required_chunk_handles_unsorted_column_indices(tmp_path):
    data_path = tmp_path / "sample.h5"

    with h5py.File(data_path, "w") as handle:
        group = handle.create_group("complete_data")
        group.create_dataset(
            "block0_values",
            data=np.array(
                [[10, 20, 30, 40], [50, 60, 70, 80]],
                dtype=np.int64,
            ),
        )
        group.create_dataset(
            "block0_items",
            data=np.array(
                [b"sample_nr", b"label", b"timestamp", b"runtime_state"],
                dtype=h5py.string_dtype("utf-8"),
            ),
        )

    with h5py.File(data_path, "r") as handle:
        group = handle["/complete_data"]
        locations = {
            "sample_nr": ColumnLocation("sample_nr", "block0_values", 1, "int64"),
            "label": ColumnLocation("label", "block0_values", 0, "int64"),
            "timestamp": ColumnLocation("timestamp", "block0_values", 3, "int64"),
            "runtime_state": ColumnLocation("runtime_state", "block0_values", 2, "int64"),
        }

        result = read_required_chunk(group, locations, 0, 2)

    np.testing.assert_array_equal(result["label"], np.array([10, 50], dtype=np.int64))
    np.testing.assert_array_equal(result["sample_nr"], np.array([20, 60], dtype=np.int64))
    np.testing.assert_array_equal(result["timestamp"], np.array([40, 80], dtype=np.int64))
    np.testing.assert_array_equal(result["runtime_state"], np.array([30, 70], dtype=np.int64))
