from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    parquet_file = pq.ParquetFile(DATASET_PATH)

    print("Dataset:", DATASET_PATH)
    print("Row groups:", parquet_file.num_row_groups)
    print("Schema:")
    print(parquet_file.schema)

    metadata_columns = [
        "sample",
        "anomaly",
        "category",
        "setting",
    ]

    metadata = pd.read_parquet(
        DATASET_PATH,
        columns=metadata_columns,
    )

    print("\nShape of metadata table:", metadata.shape)
    print("Unique cycles:", metadata["sample"].nunique())
    print("\nAnomaly counts by cycle:")
    print(
        metadata.groupby("sample")["anomaly"]
        .first()
        .value_counts(dropna=False)
    )

    print("\nCategory counts by cycle:")
    print(
        metadata.groupby("sample")["category"]
        .first()
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nSetting counts by cycle:")
    print(
        metadata.groupby("sample")["setting"]
        .first()
        .value_counts(dropna=False)
        .sort_index()
    )


if __name__ == "__main__":
    main()