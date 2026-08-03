from pathlib import Path

import numpy as np

from src.detectors import (
    PooledDetector,
    RACEDetector,
    SourceOnlyDetector,
    TargetOnlyDetector,
)
from src.feature_extractor import (
    FeaturePreprocessor,
    extract_feature_matrix,
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


def calculate_metrics(
    normal_predictions: np.ndarray,
    anomaly_predictions: np.ndarray,
) -> tuple[float, float]:
    false_positive_rate = float(
        np.mean(normal_predictions == 1)
    )
    recall = float(
        np.mean(anomaly_predictions == 1)
    )

    return false_positive_rate, recall


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

    source_raw, _ = extract_feature_matrix(
        split.source_train
    )
    target_raw, _ = extract_feature_matrix(
        split.target_commissioning
    )
    calibration_raw, _ = extract_feature_matrix(
        split.target_calibration
    )
    normal_raw, _ = extract_feature_matrix(
        split.target_normal_evaluation
    )
    anomaly_raw, _ = extract_feature_matrix(
        split.target_anomaly_evaluation
    )

    detectors = {
        "TargetOnly": TargetOnlyDetector(),
        "SourceOnly": SourceOnlyDetector(),
        "Pooled": PooledDetector(),
        "RACE": RACEDetector(lambda_reg=60.0),
    }

    print("=" * 92)
    print(
        f"{'Detector':<14} | "
        f"{'Dim':>5} | "
        f"{'Threshold':>12} | "
        f"{'FPR':>7} | "
        f"{'Recall':>7} | "
        f"{'Weight':>7}"
    )
    print("=" * 92)

    for name, detector in detectors.items():
        if name == "TargetOnly":
            preprocessor_training = target_raw
        elif name == "SourceOnly":
            preprocessor_training = source_raw
        else:
            preprocessor_training = np.vstack(
                (source_raw, target_raw)
            )

        preprocessor = FeaturePreprocessor(
            variance_threshold=1e-12
        )

        preprocessor.fit(preprocessor_training)

        source_features = preprocessor.transform(
            source_raw
        )
        target_features = preprocessor.transform(
            target_raw
        )
        calibration_features = preprocessor.transform(
            calibration_raw
        )
        normal_features = preprocessor.transform(
            normal_raw
        )
        anomaly_features = preprocessor.transform(
            anomaly_raw
        )

        detector.fit(
            source_features=source_features,
            target_features=target_features,
        )

        detector.calibrate(calibration_features)

        normal_predictions = detector.predict(
            normal_features
        )
        anomaly_predictions = detector.predict(
            anomaly_features
        )

        fpr, recall = calculate_metrics(
            normal_predictions,
            anomaly_predictions,
        )

        race_weight = (
            detector.target_weight_
            if isinstance(detector, RACEDetector)
            else np.nan
        )

        print(
            f"{name:<14} | "
            f"{preprocessor.output_feature_count_:>5} | "
            f"{detector.threshold_:>12.4f} | "
            f"{fpr:>7.3f} | "
            f"{recall:>7.3f} | "
            f"{race_weight:>7.3f}"
        )

        assert detector.threshold_ is not None
        assert normal_predictions.shape == (100,)
        assert anomaly_predictions.shape == (755,)
        assert np.isfinite(
            detector.score_samples(anomaly_features)
        ).all()

    print("=" * 92)
    
    print("Detector smoke test: PASS")


if __name__ == "__main__":
    main()