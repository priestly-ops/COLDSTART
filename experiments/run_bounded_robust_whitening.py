#!/usr/bin/env python3
"""Frozen-split bounded robust-whitening experiment.

This is the predeclared score-stability intervention following the negative
calibration-tail study.  It never uses anomaly labels for fitting, feature
selection, scaling, covariance estimation, clipping, or calibration.

Primary comparison (c=3):
  * TargetOnly / CURRENT_MAHALANOBIS
  * TargetOnly / ROBUST_DIAGONAL
  * TargetOnly / BOUNDED_ROBUST_WHITENING
  * RACE       / CURRENT_MAHALANOBIS
  * RACE       / BOUNDED_ROBUST_WHITENING

Sensitivity-only bounded scores use c in {2.5, 4.0}.  The commissioning,
calibration, and evaluation memberships exactly match the main experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import RACEDetector, TargetOnlyDetector  # noqa: E402
from src.evaluation import fit_detector  # noqa: E402
from src.feature_extractor import extract_feature_matrix  # noqa: E402
from src.split_generator import create_experiment_split  # noqa: E402
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("bounded_robust_whitening")
GLOBAL_SEED = 42
SEEDS = tuple(range(20))
GRID = (10, 25, 50, 100)
MAX_N = 100
CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
ALPHA = 0.01
RECALL_TARGET = 0.90
PRIMARY_CLIP = 3.0
SENSITIVITY_CLIPS = (2.5, 4.0)
MAD_NORMALIZER = 1.4826
MIN_SCALE = 1e-12
MIN_EIGENVALUE = 1e-8
LAMBDA_REG = 60.0
BOOTSTRAP_SAMPLES = 10_000
PROTOCOL_VERSION = "bounded-robust-whitening-v1.0.0"


def cycle_ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def digest(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def raw_features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, ids = extract_feature_matrix(cycles)
    if ids.tolist() != cycle_ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or Inf.")
    return matrix


def assert_disjoint(**groups: Sequence[RobotCycle]) -> None:
    memberships = {name: set(cycle_ids(value)) for name, value in groups.items()}
    names = list(memberships)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = memberships[left] & memberships[right]
            if overlap:
                raise AssertionError(
                    f"Episode leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )


@dataclass
class ScoreOutput:
    scores: np.ndarray
    clipped_fraction: np.ndarray


class RobustTransform:
    """Commissioning-only median/MAD transform and stable feature filter."""

    def fit(self, target_commissioning: np.ndarray) -> "RobustTransform":
        values = np.asarray(target_commissioning, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("RobustTransform requires at least two 2D samples.")
        self.median_ = np.median(values, axis=0)
        mad = np.median(np.abs(values - self.median_), axis=0)
        scale = MAD_NORMALIZER * mad
        # MAD can be zero for quantized but nonconstant features. Such features
        # are excluded rather than assigned an arbitrarily tiny denominator.
        self.keep_ = np.isfinite(scale) & (scale > MIN_SCALE)
        if not np.any(self.keep_):
            raise RuntimeError("No features have a positive commissioning MAD.")
        self.scale_ = scale[self.keep_]
        self.center_ = self.median_[self.keep_]
        self.output_feature_count_ = int(np.sum(self.keep_))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "keep_"):
            raise RuntimeError("RobustTransform must be fitted first.")
        transformed = (
            np.asarray(values, dtype=np.float64)[:, self.keep_] - self.center_
        ) / self.scale_
        if not np.isfinite(transformed).all():
            raise RuntimeError("Robust transformation produced NaN or Inf.")
        return transformed


class BoundedWhiteningScore:
    """Ledoit-Wolf whitening with bounded per-component score influence."""

    def __init__(self, detector: str, clip: float) -> None:
        if detector not in {"TargetOnly", "RACE"}:
            raise ValueError(f"Unsupported detector: {detector}")
        if clip <= 0:
            raise ValueError("clip must be positive.")
        self.detector = detector
        self.clip = float(clip)

    @staticmethod
    def _estimate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fitted = LedoitWolf(store_precision=False).fit(values)
        return (
            np.asarray(fitted.location_, dtype=np.float64),
            np.asarray(fitted.covariance_, dtype=np.float64),
        )

    def fit(self, source: np.ndarray, target: np.ndarray) -> "BoundedWhiteningScore":
        target_mean, target_cov = self._estimate(target)
        if self.detector == "TargetOnly":
            location, covariance = target_mean, target_cov
            self.target_weight_ = 1.0
        else:
            source_mean, source_cov = self._estimate(source)
            weight = len(target) / (len(target) + LAMBDA_REG)
            location = (1.0 - weight) * source_mean + weight * target_mean
            covariance = (1.0 - weight) * source_cov + weight * target_cov
            self.target_weight_ = float(weight)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, MIN_EIGENVALUE)
        # Row-vector convention: (x - mu) @ whitening_.
        self.whitening_ = eigenvectors / np.sqrt(eigenvalues)[None, :]
        self.location_ = location
        self.feature_count_ = target.shape[1]
        return self

    def score(self, values: np.ndarray) -> ScoreOutput:
        whitened = (np.asarray(values, dtype=np.float64) - self.location_) @ self.whitening_
        clipped = np.clip(whitened, -self.clip, self.clip)
        scores = np.sqrt(np.sum(clipped * clipped, axis=1))
        fraction = np.mean(np.abs(whitened) > self.clip, axis=1)
        return ScoreOutput(scores=scores, clipped_fraction=fraction)


class RobustDiagonalScore:
    """Bounded Euclidean score in commissioning median/MAD coordinates."""

    def __init__(self, clip: float) -> None:
        self.clip = float(clip)

    def score(self, values: np.ndarray) -> ScoreOutput:
        clipped = np.clip(values, -self.clip, self.clip)
        return ScoreOutput(
            scores=np.sqrt(np.sum(clipped * clipped, axis=1)),
            clipped_fraction=np.mean(np.abs(values) > self.clip, axis=1),
        )


def conformal_threshold(scores: np.ndarray) -> tuple[float, int, bool]:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    raw_rank = int(np.ceil((len(values) + 1) * (1.0 - ALPHA)))
    rank = min(raw_rank, len(values))
    return float(values[rank - 1]), rank, raw_rank <= len(values)


def score_ratio(scores: np.ndarray) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    if len(ordered) < 2:
        return np.nan
    denominator = max(float(ordered[-2]), np.finfo(np.float64).tiny)
    return float(ordered[-1] / denominator)


def current_scores(
    detector_name: str,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    healthy: np.ndarray,
    anomaly: np.ndarray,
) -> tuple[ScoreOutput, ScoreOutput, ScoreOutput, int, float | None]:
    factory = TargetOnlyDetector if detector_name == "TargetOnly" else RACEDetector
    detector, preprocessor, _, _ = fit_detector(
        detector_name, factory, source, target
    )
    outputs = []
    for matrix in (calibration, healthy, anomaly):
        scores = detector.score_samples(preprocessor.transform(matrix))
        outputs.append(ScoreOutput(scores, np.zeros(len(scores), dtype=np.float64)))
    weight = detector.target_weight_ if isinstance(detector, RACEDetector) else None
    return (*outputs, int(preprocessor.output_feature_count_), weight)


def robust_scores(
    detector_name: str,
    variant: str,
    clip: float,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    healthy: np.ndarray,
    anomaly: np.ndarray,
) -> tuple[ScoreOutput, ScoreOutput, ScoreOutput, int, float | None]:
    transform = RobustTransform().fit(target)
    source_z, target_z = transform.transform(source), transform.transform(target)
    matrices = [transform.transform(x) for x in (calibration, healthy, anomaly)]
    if variant == "ROBUST_DIAGONAL":
        if detector_name != "TargetOnly":
            raise ValueError("ROBUST_DIAGONAL is predeclared for TargetOnly only.")
        scorer = RobustDiagonalScore(clip)
        weight: float | None = None
    else:
        scorer = BoundedWhiteningScore(detector_name, clip).fit(source_z, target_z)
        weight = scorer.target_weight_
    outputs = tuple(scorer.score(matrix) for matrix in matrices)
    return (*outputs, transform.output_feature_count_, weight)


def variants() -> tuple[tuple[str, str, float | None, str], ...]:
    rows: list[tuple[str, str, float | None, str]] = [
        ("TargetOnly", "CURRENT_MAHALANOBIS", None, "reference"),
        ("RACE", "CURRENT_MAHALANOBIS", None, "reference"),
    ]
    for detector, variant in (
        ("TargetOnly", "ROBUST_DIAGONAL"),
        ("TargetOnly", "BOUNDED_ROBUST_WHITENING"),
        ("RACE", "BOUNDED_ROBUST_WHITENING"),
    ):
        rows.append((detector, variant, PRIMARY_CLIP, "primary"))
        rows.extend((detector, variant, clip, "sensitivity") for clip in SENSITIVITY_CLIPS)
    return tuple(rows)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_experiment(cycles: Sequence[RobotCycle]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    result_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    memberships: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "alpha": ALPHA,
        "primary_clip": PRIMARY_CLIP,
        "sensitivity_clips": list(SENSITIVITY_CLIPS),
        "seeds": {},
    }
    for seed in SEEDS:
        split = create_experiment_split(
            cycles,
            commissioning_size=MAX_N,
            seed=seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
            maximum_commissioning_size=MAX_N,
        )
        assert_disjoint(
            source=split.source_train,
            commissioning=split.target_commissioning,
            calibration=split.target_calibration,
            healthy_evaluation=split.target_normal_evaluation,
            anomaly_evaluation=split.target_anomaly_evaluation,
        )
        memberships["seeds"][str(seed)] = {
            "commissioning_pool_ids": cycle_ids(split.target_commissioning),
            "calibration_ids": cycle_ids(split.target_calibration),
            "healthy_evaluation_ids": cycle_ids(split.target_normal_evaluation),
            "anomaly_evaluation_digest": digest(cycle_ids(split.target_anomaly_evaluation)),
        }
        source = raw_features(split.source_train)
        calibration = raw_features(split.target_calibration)
        healthy = raw_features(split.target_normal_evaluation)
        anomaly = raw_features(split.target_anomaly_evaluation)
        labels = np.concatenate((np.zeros(len(healthy)), np.ones(len(anomaly))))

        for n in GRID:
            LOGGER.info("Processing seed=%d N=%d", seed, n)
            target = raw_features(split.target_commissioning[:n])
            for detector_name, variant, clip, role in variants():
                if variant == "CURRENT_MAHALANOBIS":
                    scored = current_scores(
                        detector_name, source, target, calibration, healthy, anomaly
                    )
                    clip_label = "none"
                else:
                    assert clip is not None
                    scored = robust_scores(
                        detector_name, variant, clip, source, target,
                        calibration, healthy, anomaly,
                    )
                    clip_label = f"{clip:g}"
                calibration_out, healthy_out, anomaly_out, dimension, weight = scored
                threshold, rank, supported = conformal_threshold(calibration_out.scores)
                healthy_predictions = healthy_out.scores > threshold
                anomaly_predictions = anomaly_out.scores > threshold
                fpr = float(np.mean(healthy_predictions))
                recall = float(np.mean(anomaly_predictions))
                combined_scores = np.concatenate((healthy_out.scores, anomaly_out.scores))
                threshold_setters = np.flatnonzero(
                    np.isclose(calibration_out.scores, threshold, rtol=1e-12, atol=1e-12)
                )
                setter_ids = [cycle_ids(split.target_calibration)[i] for i in threshold_setters]
                base = {
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                    "N": n,
                    "detector": detector_name,
                    "score_variant": variant,
                    "clip": clip_label,
                    "analysis_role": role,
                }
                result_rows.append({
                    **base,
                    "threshold": threshold,
                    "threshold_rank": rank,
                    "finite_sample_alpha_supported": supported,
                    "interior_order_statistic": bool(rank < len(calibration_out.scores)),
                    "recall": recall,
                    "false_positive_rate": fpr,
                    "success": bool(recall >= RECALL_TARGET and fpr <= ALPHA),
                    "auroc": float(roc_auc_score(labels, combined_scores)),
                    "auprc": float(average_precision_score(labels, combined_scores)),
                    "max_to_second_calibration_ratio": score_ratio(calibration_out.scores),
                    "threshold_setter_ids": ";".join(map(str, setter_ids)),
                    "threshold_setter_count": len(setter_ids),
                    "effective_feature_dimension": dimension,
                    "mean_calibration_clipped_fraction": float(np.mean(calibration_out.clipped_fraction)),
                    "mean_healthy_clipped_fraction": float(np.mean(healthy_out.clipped_fraction)),
                    "mean_anomaly_clipped_fraction": float(np.mean(anomaly_out.clipped_fraction)),
                    "target_weight": weight,
                })
                for split_name, ids, output in (
                    ("calibration", cycle_ids(split.target_calibration), calibration_out),
                    ("healthy_evaluation", cycle_ids(split.target_normal_evaluation), healthy_out),
                    ("anomaly_evaluation", cycle_ids(split.target_anomaly_evaluation), anomaly_out),
                ):
                    score_rows.extend({
                        **base,
                        "split": split_name,
                        "episode_id": episode_id,
                        "score": float(score),
                        "clipped_fraction": float(fraction),
                        "threshold": threshold,
                    } for episode_id, score, fraction in zip(
                        ids, output.scores, output.clipped_fraction, strict=True
                    ))

    results = pd.DataFrame(result_rows)
    summary_rows: list[dict[str, object]] = []
    group_cols = ["detector", "score_variant", "clip", "analysis_role", "N"]
    for keys, group in results.groupby(group_cols, sort=True, dropna=False):
        n = int(keys[-1])
        recall_low, recall_high = bootstrap_interval(group["recall"].to_numpy(), GLOBAL_SEED + n)
        fpr_low, fpr_high = bootstrap_interval(group["false_positive_rate"].to_numpy(), GLOBAL_SEED + 1000 + n)
        summary_rows.append({
            **dict(zip(group_cols, keys, strict=True)),
            "seed_count": len(group),
            "mean_recall": float(group["recall"].mean()),
            "recall_ci_lower": recall_low,
            "recall_ci_upper": recall_high,
            "mean_false_positive_rate": float(group["false_positive_rate"].mean()),
            "fpr_ci_lower": fpr_low,
            "fpr_ci_upper": fpr_high,
            "success_rate": float(group["success"].mean()),
            "mean_auroc": float(group["auroc"].mean()),
            "mean_auprc": float(group["auprc"].mean()),
            "mean_max_to_second_ratio": float(group["max_to_second_calibration_ratio"].mean()),
            "mean_effective_feature_dimension": float(group["effective_feature_dimension"].mean()),
            "mean_calibration_clipped_fraction": float(group["mean_calibration_clipped_fraction"].mean()),
            "joint_ci_success": bool(recall_low >= RECALL_TARGET and fpr_high <= ALPHA),
        })
    return results, pd.DataFrame(summary_rows), pd.DataFrame(score_rows), memberships


def write_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    memberships: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "bounded_whitening_results.csv", index=False, float_format="%.12g")
    summary.to_csv(output_dir / "bounded_whitening_summary.csv", index=False, float_format="%.12g")
    scores.to_csv(output_dir / "bounded_whitening_score_distributions.csv", index=False, float_format="%.12g")
    (output_dir / "bounded_whitening_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )


def verify_determinism(cycles: Sequence[RobotCycle]) -> None:
    first = run_experiment(cycles)
    second = run_experiment(cycles)
    for index, columns in enumerate((
        ["seed", "N", "detector", "score_variant", "clip"],
        ["detector", "score_variant", "clip", "N"],
        ["seed", "N", "detector", "score_variant", "clip", "split", "episode_id"],
    )):
        left = first[index].sort_values(columns, kind="stable").to_csv(index=False)
        right = second[index].sort_values(columns, kind="stable").to_csv(index=False)
        if left != right:
            raise AssertionError(f"Determinism failed for table {index}.")
    if json.dumps(first[3], sort_keys=True) != json.dumps(second[3], sort_keys=True):
        raise AssertionError("Determinism failed for episode memberships.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--verify-determinism", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    np.random.seed(GLOBAL_SEED)
    LOGGER.info("Loading %s", args.data_path)
    cycles = load_cycles(args.data_path)
    if args.verify_determinism:
        verify_determinism(cycles)
    outputs = run_experiment(cycles)
    write_outputs(*outputs, args.output_dir)
    LOGGER.info("Saved bounded-whitening outputs to %s", args.output_dir)
    print(outputs[1].to_string(index=False))


if __name__ == "__main__":
    main()