from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "voraus_data_audit.json"

META_COLUMNS = [
    "time",
    "sample",
    "anomaly",
    "category",
    "setting",
    "action",
    "active",
]


def python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(DATASET_PATH)
    schema_names = parquet.schema.names

    available_meta = [
        column for column in META_COLUMNS
        if column in schema_names
    ]

    dataframe = pd.read_parquet(
        DATASET_PATH,
        columns=available_meta,
    )

    required = {
        "sample",
        "anomaly",
        "category",
        "setting",
    }

    missing_required = sorted(
        required.difference(dataframe.columns)
    )

    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}"
        )

    grouped = dataframe.groupby("sample", sort=True)

    inconsistent_cycles: dict[str, list[int]] = {}

    for column in ["anomaly", "category", "setting"]:
        nunique = grouped[column].nunique(dropna=False)
        inconsistent = nunique[nunique != 1].index.tolist()
        inconsistent_cycles[column] = [
            int(value) for value in inconsistent
        ]

    cycle_lengths = grouped.size()

    duplicate_rows = int(dataframe.duplicated().sum())

    report = {
        "dataset_path": str(DATASET_PATH),
        "row_count": int(len(dataframe)),
        "column_count": int(len(schema_names)),
        "columns": schema_names,
        "unique_cycles": int(dataframe["sample"].nunique()),
        "missing_values": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "duplicate_metadata_rows": duplicate_rows,
        "cycle_length": {
            "minimum": int(cycle_lengths.min()),
            "maximum": int(cycle_lengths.max()),
            "mean": float(cycle_lengths.mean()),
            "median": float(cycle_lengths.median()),
            "p01": float(cycle_lengths.quantile(0.01)),
            "p99": float(cycle_lengths.quantile(0.99)),
        },
        "inconsistent_cycles": inconsistent_cycles,
        "normal_cycle_count": int(
            grouped["anomaly"].first().eq(False).sum()
        ),
        "anomalous_cycle_count": int(
            grouped["anomaly"].first().eq(True).sum()
        ),
        "category_cycle_counts": {
            str(key): int(value)
            for key, value in grouped["category"]
            .first()
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "setting_cycle_counts": {
            str(key): int(value)
            for key, value in grouped["setting"]
            .first()
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
    }

    clean_report = {
        key: python_value(value)
        for key, value in report.items()
    }

    REPORT_PATH.write_text(
        json.dumps(clean_report, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("VORAUS-AD DATA AUDIT")
    print("=" * 70)
    print(f"Rows:              {report['row_count']:,}")
    print(f"Columns:           {report['column_count']}")
    print(f"Cycles:            {report['unique_cycles']}")
    print(f"Normal cycles:     {report['normal_cycle_count']}")
    print(f"Anomalous cycles:  {report['anomalous_cycle_count']}")
    print(
        "Cycle lengths:     "
        f"{report['cycle_length']['minimum']} to "
        f"{report['cycle_length']['maximum']}"
    )
    print(
        "Metadata conflicts:",
        {
            key: len(value)
            for key, value in inconsistent_cycles.items()
        },
    )
    print(f"Report written to: {REPORT_PATH}")

    has_conflicts = any(
        inconsistent_cycles[column]
        for column in inconsistent_cycles
    )

    if has_conflicts:
        raise RuntimeError(
            "Some cycles contain inconsistent metadata. "
            "Inspect the audit report before continuing."
        )


if __name__ == "__main__":
    main()