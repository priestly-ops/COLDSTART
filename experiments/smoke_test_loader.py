from pathlib import Path

from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)


def main() -> None:
    cycles = load_cycles(
        path=DATASET_PATH,
        signal_set="measured",
    )

    normal_count = sum(not cycle.anomaly for cycle in cycles)
    anomalous_count = sum(cycle.anomaly for cycle in cycles)

    print(f"Loaded cycles: {len(cycles)}")
    print(f"Normal cycles: {normal_count}")
    print(f"Anomalous cycles: {anomalous_count}")
    print(f"Measured channels: {len(cycles[0].columns)}")
    print(f"First cycle shape: {cycles[0].values.shape}")
    print(f"First episode ID: {cycles[0].episode_id}")


if __name__ == "__main__":
    main()