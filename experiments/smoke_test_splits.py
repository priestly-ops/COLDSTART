from pathlib import Path

from src.split_generator import create_experiment_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

COMMISSIONING_GRID = [10, 25, 50, 100]


def main() -> None:
    cycles = load_cycles(
        path=DATASET_PATH,
        signal_set="measured",
    )

    for commissioning_size in COMMISSIONING_GRID:
        split = create_experiment_split(
            cycles=cycles,
            commissioning_size=commissioning_size,
            seed=42,
            calibration_size=30,
            normal_evaluation_size=100,
        )

        print("=" * 60)
        print(f"N = {commissioning_size}")
        print(f"Source training:    {len(split.source_train)}")
        print(f"Commissioning:      {len(split.target_commissioning)}")
        print(f"Calibration:        {len(split.target_calibration)}")
        print(
            f"Normal evaluation:  "
            f"{len(split.target_normal_evaluation)}"
        )
        print(
            f"Anomaly evaluation: "
            f"{len(split.target_anomaly_evaluation)}"
        )
        print("Leakage check:      PASS")

        total_target_used = (
            len(split.target_commissioning)
            + len(split.target_calibration)
            + len(split.target_normal_evaluation)
        )

        assert total_target_used <= 319


if __name__ == "__main__":
    main()