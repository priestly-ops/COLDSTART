from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from src.base_detector import BaseDetector
from src.detectors import (
    PooledDetector,
    RACEDetector,
    SourceOnlyDetector,
    TargetOnlyDetector,
)
from src.feature_extractor import FeaturePreprocessor
from src.voraus_loader import RobotCycle


DetectorFactory = Callable[[], BaseDetector]


@dataclass(frozen=True)
class EvaluationResult:
    detector: str
    commissioning_size: int
    seed: int
    false_positive_rate: float
    recall: float
    success: bool
    threshold: float
    retained_features: int
    target_weight: float | None


def _check_disjoint_ids(
    groups: dict[str, Sequence[RobotCycle]],
) -> None:
    id_sets = {
        name: {cycle.episode_id for cycle in cycles}
        for name, cycles in groups.items()
    }

    names = list(id_sets)

    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = id_sets[first] & id_sets[second]

            if overlap:
                raise RuntimeError(
                    f"Episode overlap between {first} and "
                    f"{second}: {sorted(overlap)[:10]}"
                )


def _make_preprocessor_training(
    detector_name: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    if detector_name == "TargetOnly":
        return target_features

    if detector_name == "SourceOnly":
        return source_features

    if detector_name in {"Pooled", "RACE"}:
        return np.vstack(
            (source_features, target_features)
        )

    raise ValueError(
        f"Unsupported detector: {detector_name}"
    )


def fit_detector(
    detector_name: str,
    detector_factory: DetectorFactory,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
) -> tuple[
    BaseDetector,
    FeaturePreprocessor,
    np.ndarray,
    np.ndarray,
]:
    preprocessor_training = _make_preprocessor_training(
        detector_name=detector_name,
        source_features=source_raw,
        target_features=target_raw,
    )

    preprocessor = FeaturePreprocessor(
        variance_threshold=1e-12
    )
    preprocessor.fit(preprocessor_training)

    source_features = preprocessor.transform(source_raw)
    target_features = preprocessor.transform(target_raw)

    detector = detector_factory()
    detector.fit(
        source_features=source_features,
        target_features=target_features,
    )

    return (
        detector,
        preprocessor,
        source_features,
        target_features,
    )


def leave_one_out_scores(
    detector_name: str,
    detector_factory: DetectorFactory,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
) -> np.ndarray:
    """Generate target-normal leave-one-out conformity scores.

    Each target cycle is excluded from both preprocessing and detector
    fitting before its score is calculated.
    """
    target_raw = np.asarray(
        target_raw,
        dtype=np.float64,
    )

    if target_raw.ndim != 2:
        raise ValueError(
            "target_raw must be a 2D feature matrix."
        )

    if target_raw.shape[0] < 3:
        raise ValueError(
            "At least three target cycles are required "
            "for leave-one-out calibration."
        )

    scores = np.empty(
        target_raw.shape[0],
        dtype=np.float64,
    )

    for held_out_index in range(target_raw.shape[0]):
        keep_mask = np.ones(
            target_raw.shape[0],
            dtype=bool,
        )
        keep_mask[held_out_index] = False

        target_training = target_raw[keep_mask]
        held_out = target_raw[
            held_out_index : held_out_index + 1
        ]

        detector, preprocessor, _, _ = fit_detector(
            detector_name=detector_name,
            detector_factory=detector_factory,
            source_raw=source_raw,
            target_raw=target_training,
        )

        held_out_features = preprocessor.transform(
            held_out
        )

        scores[held_out_index] = (
            detector.score_samples(
                held_out_features
            )[0]
        )

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Leave-one-out scoring generated NaN or Inf."
        )

    return scores


def evaluate_detector(
    detector_name: str,
    detector_factory: DetectorFactory,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    calibration_raw: np.ndarray,
    normal_evaluation_raw: np.ndarray,
    anomaly_evaluation_raw: np.ndarray,
    commissioning_size: int,
    seed: int,
    false_alert_budget: float = 0.01,
    recall_target: float = 0.90,
) -> EvaluationResult:
    """Fit, calibrate, and evaluate one detector."""

    detector, preprocessor, _, _ = fit_detector(
        detector_name=detector_name,
        detector_factory=detector_factory,
        source_raw=source_raw,
        target_raw=target_raw,
    )

    calibration_features = preprocessor.transform(
        calibration_raw
    )

    normal_features = preprocessor.transform(
        normal_evaluation_raw
    )

    anomaly_features = preprocessor.transform(
        anomaly_evaluation_raw
    )

    detector.calibrate(
        calibration_features
    )

    normal_predictions = detector.predict(
        normal_features
    )

    anomaly_predictions = detector.predict(
        anomaly_features
    )

    false_positive_rate = float(
        np.mean(normal_predictions == 1)
    )

    recall = float(
        np.mean(anomaly_predictions == 1)
    )

    if detector.threshold_ is None:
        raise RuntimeError(
            "Detector threshold is unavailable."
        )

    target_weight: float | None = None

    if isinstance(detector, RACEDetector):
        target_weight = detector.target_weight_

    return EvaluationResult(
        detector=detector_name,
        commissioning_size=commissioning_size,
        seed=seed,
        false_positive_rate=false_positive_rate,
        recall=recall,
        success=(
            recall >= recall_target
            and false_positive_rate
            <= false_alert_budget
        ),
        threshold=float(detector.threshold_),
        retained_features=int(
            preprocessor.output_feature_count_
        ),
        target_weight=target_weight,
    )

def detector_factories(
    false_alert_budget: float,
) -> dict[str, DetectorFactory]:
    return {
        "TargetOnly": lambda: TargetOnlyDetector(
            false_alert_budget=false_alert_budget
        ),
        "SourceOnly": lambda: SourceOnlyDetector(
            false_alert_budget=false_alert_budget
        ),
        "Pooled": lambda: PooledDetector(
            false_alert_budget=false_alert_budget
        ),
        "RACE": lambda: RACEDetector(
            lambda_reg=60.0,
            false_alert_budget=false_alert_budget,
        ),
    }