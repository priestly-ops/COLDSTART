from pathlib import Path

import numpy as np

from src.feature_extractor import (
    FeaturePreprocessor,
    extract_feature_matrix,
    make_feature_names,
)
from src.split_generator import create_experiment_split
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

    split = create_experiment_split(
        cycles=cycles,
        commissioning_size=100,
        seed=42,
        calibration_size=30,
        normal_evaluation_size=100,
    )

    source_features, source_ids = extract_feature_matrix(
        split.source_train
    )

    target_features, target_ids = extract_feature_matrix(
        split.target_commissioning
    )

    calibration_features, calibration_ids = (
        extract_feature_matrix(
            split.target_calibration
        )
    )

    normal_eval_features, normal_eval_ids = (
        extract_feature_matrix(
            split.target_normal_evaluation
        )
    )

    anomaly_features, anomaly_ids = extract_feature_matrix(
        split.target_anomaly_evaluation
    )

    feature_names = make_feature_names(
        split.source_train[0].columns
    )

    assert source_features.shape[1] == len(feature_names)
    assert source_features.shape[1] == 94 * 6
    assert len(source_ids) == 948
    assert len(target_ids) == 100
    assert len(calibration_ids) == 30
    assert len(normal_eval_ids) == 100
    assert len(anomaly_ids) == 755

    # Demonstration using the pooled allowed training information.
    # Each detector will later fit its own appropriate preprocessor.
    allowed_training_features = np.vstack(
        (
            source_features,
            target_features,
        )
    )

    preprocessor = FeaturePreprocessor(
        variance_threshold=1e-12
    )

    transformed_training = (
        preprocessor.fit_transform(
            allowed_training_features
        )
    )

    transformed_calibration = (
        preprocessor.transform(
            calibration_features
        )
    )

    transformed_normal_eval = (
        preprocessor.transform(
            normal_eval_features
        )
    )

    transformed_anomalies = (
        preprocessor.transform(
            anomaly_features
        )
    )

    print("=" * 68)
    print("FEATURE EXTRACTION SMOKE TEST")
    print("=" * 68)
    print(
        f"Raw features per cycle:       "
        f"{source_features.shape[1]}"
    )
    print(
        f"Retained training features:   "
        f"{preprocessor.output_feature_count_}"
    )
    print(
        f"Removed low-variance features:"
        f" {source_features.shape[1] - preprocessor.output_feature_count_}"
    )
    print(
        f"Training matrix:              "
        f"{transformed_training.shape}"
    )
    print(
        f"Calibration matrix:           "
        f"{transformed_calibration.shape}"
    )
    print(
        f"Normal evaluation matrix:     "
        f"{transformed_normal_eval.shape}"
    )
    print(
        f"Anomaly evaluation matrix:    "
        f"{transformed_anomalies.shape}"
    )
    print(
        "All transformed values finite:",
        bool(
            np.isfinite(
                transformed_anomalies
            ).all()
        ),
    )


if __name__ == "__main__":
    main()