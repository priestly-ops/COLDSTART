from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "voraus_signal_audit.json"
)

META_COLUMNS = {
    "time",
    "sample",
    "anomaly",
    "category",
    "setting",
    "action",
    "active",
}


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet = pq.ParquetFile(DATASET_PATH)

    signal_columns = [
        column
        for column in parquet.schema.names
        if column not in META_COLUMNS
    ]

    null_counts = {
        column: 0
        for column in signal_columns
    }

    non_finite_counts = {
        column: 0
        for column in signal_columns
    }

    minimums = {
        column: np.inf
        for column in signal_columns
    }

    maximums = {
        column: -np.inf
        for column in signal_columns
    }

    total_rows = 0

    for row_group_index in range(
        parquet.num_row_groups
    ):
        print(
            f"Auditing row group "
            f"{row_group_index + 1}/"
            f"{parquet.num_row_groups}"
        )

        table = parquet.read_row_group(
            row_group_index,
            columns=signal_columns,
        )

        dataframe = table.to_pandas()
        total_rows += len(dataframe)

        for column in signal_columns:
            series = dataframe[column]

            null_counts[column] += int(
                series.isna().sum()
            )

            numeric = series.to_numpy(
                dtype=np.float64,
                copy=False,
            )

            finite_mask = np.isfinite(numeric)

            non_finite_counts[column] += int(
                (~finite_mask).sum()
            )

            if finite_mask.any():
                finite_values = numeric[finite_mask]

                minimums[column] = min(
                    minimums[column],
                    float(finite_values.min()),
                )

                maximums[column] = max(
                    maximums[column],
                    float(finite_values.max()),
                )

    constant_columns = [
        column
        for column in signal_columns
        if np.isfinite(minimums[column])
        and minimums[column] == maximums[column]
    ]

    columns_with_nulls = {
        column: count
        for column, count in null_counts.items()
        if count > 0
    }

    columns_with_non_finite = {
        column: count
        for column, count
        in non_finite_counts.items()
        if count > 0
    }

    report = {
        "dataset_path": str(DATASET_PATH),
        "total_rows": total_rows,
        "signal_column_count": len(signal_columns),
        "columns_with_nulls": columns_with_nulls,
        "columns_with_non_finite_values": (
            columns_with_non_finite
        ),
        "constant_columns": constant_columns,
        "column_ranges": {
            column: {
                "minimum": (
                    None
                    if not np.isfinite(minimums[column])
                    else minimums[column]
                ),
                "maximum": (
                    None
                    if not np.isfinite(maximums[column])
                    else maximums[column]
                ),
            }
            for column in signal_columns
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("VORAUS-AD SIGNAL AUDIT")
    print("=" * 70)
    print(f"Rows audited: {total_rows:,}")
    print(
        f"Signal columns: {len(signal_columns)}"
    )
    print(
        "Columns with nulls:",
        len(columns_with_nulls),
    )
    print(
        "Columns with non-finite values:",
        len(columns_with_non_finite),
    )
    print(
        "Constant columns:",
        len(constant_columns),
    )
    print(f"Report written to: {REPORT_PATH}")

    if columns_with_non_finite:
        raise RuntimeError(
            "Non-finite sensor values were found. "
            "Inspect the generated report."
        )


if __name__ == "__main__":
    main()