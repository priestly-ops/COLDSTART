#!/usr/bin/env python3
"""Leakage-safe TargetOnly covariance/effective-dimension ablation.

Copy this file to ``experiments/run_targetonly_regularization_ablation.py``.
It reuses the exact voraus-AD split and feature extraction APIs used by the
commissioning experiment. Every transform is fitted on commissioning cycles
only; calibration and evaluation memberships remain untouched.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "experiments" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector  # noqa: E402
from src.feature_extractor import extract_feature_matrix  # noqa: E402
from src.split_generator import create_experiment_split  # noqa: E402
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("targetonly_regularization")
GLOBAL_SEED = 42
GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
MAX_N = 100
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
ALPHA = 0.01
RECALL_TARGET = 0.90
PROTOCOL_VERSION = "targetonly-regularization-v1.0.0"


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def raw_features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, matrix_ids = extract_feature_matrix(cycles)
    if matrix_ids.tolist() != ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    return np.asarray(matrix, dtype=np.float64)


def assert_disjoint(**groups: Sequence[RobotCycle]) -> None:
    sets = {name: set(ids(value)) for name, value in groups.items()}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"Leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )


@dataclass
class TrainingTransform:
    """Variance filtering, optional training-only feature cap, scaling and PCA."""

    pca_spec: float | int | None = None
    feature_cap: int | None = None

    def fit(self, X: np.ndarray) -> "TrainingTransform":
        self.variance_filter_ = VarianceThreshold(1e-12).fit(X)
        filtered = self.variance_filter_.transform(X)

        self.selected_: np.ndarray | None = None
        if self.feature_cap is not None and filtered.shape[1] > self.feature_cap:
            # Selection uses commissioning variance only. Stable sorting makes
            # ties deterministic. This happens before scaling.
            variance = np.var(filtered, axis=0, ddof=1)
            ranked = np.argsort(-variance, kind="stable")
            self.selected_ = np.sort(ranked[: self.feature_cap])
            filtered = filtered[:, self.selected_]

        self.scaler_ = StandardScaler().fit(filtered)
        scaled = self.scaler_.transform(filtered)

        self.pca_: PCA | None = None
        if self.pca_spec is not None:
            max_components = min(scaled.shape)
            if isinstance(self.pca_spec, int):
                components = min(self.pca_spec, max_components)
            else:
                components = self.pca_spec
            self.pca_ = PCA(n_components=components, svd_solver="full")
            self.pca_.fit(scaled)

        transformed = self.transform(X)
        self.output_feature_count_ = int(transformed.shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filtered = self.variance_filter_.transform(X)
        if self.selected_ is not None:
            filtered = filtered[:, self.selected_]
        result = self.scaler_.transform(filtered)
        if self.pca_ is not None:
            result = self.pca_.transform(result)
        if not np.isfinite(result).all():
            raise RuntimeError("Transformation produced NaN or Inf.")
        return np.asarray(result, dtype=np.float64)


class DiagonalMahalanobis(BaseDetector):
    def fit(self, source_features: np.ndarray, target_features: np.ndarray):
        del source_features
        X = self._validate_features(target_features)
        self.location_ = np.mean(X, axis=0)
        variance = np.var(X, axis=0, ddof=1)
        # Scale-aware floor; standardized inputs normally make this 1e-8.
        floor = max(1e-8, 1e-8 * float(np.median(variance[variance > 0])))
        self.variance_ = np.maximum(variance, floor)
        self.is_fitted_ = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        X = self._validate_features(features)
        return np.sqrt(np.sum((X - self.location_) ** 2 / self.variance_, axis=1))


class LedoitWolfMahalanobis(BaseDetector):
    def fit(self, source_features: np.ndarray, target_features: np.ndarray):
        del source_features
        X = self._validate_features(target_features)
        estimator = LedoitWolf(store_precision=True).fit(X)
        self.location_ = np.asarray(estimator.location_, dtype=np.float64)
        self.precision_ = np.asarray(estimator.precision_, dtype=np.float64)
        self.shrinkage_ = float(estimator.shrinkage_)
        self.covariance_ = np.asarray(estimator.covariance_, dtype=np.float64)
        self.is_fitted_ = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        X = self._validate_features(features)
        centered = X - self.location_
        squared = np.einsum(
            "ij,jk,ik->i", centered, self.precision_, centered, optimize=True
        )
        return np.sqrt(np.maximum(squared, 0.0))


@dataclass(frozen=True)
class Variant:
    name: str
    detector_factory: Callable[[], BaseDetector]
    pca_spec: float | int | None = None
    feature_cap: int | None = None


VARIANTS = (
    Variant("LW_CURRENT", lambda: LedoitWolfMahalanobis(ALPHA)),
    Variant("DIAGONAL", lambda: DiagonalMahalanobis(ALPHA)),
    Variant("PCA_95_LW", lambda: LedoitWolfMahalanobis(ALPHA), pca_spec=0.95),
    Variant("PCA_99_LW", lambda: LedoitWolfMahalanobis(ALPHA), pca_spec=0.99),
    Variant("PCA_10_LW", lambda: LedoitWolfMahalanobis(ALPHA), pca_spec=10),
    Variant("PCA_25_LW", lambda: LedoitWolfMahalanobis(ALPHA), pca_spec=25),
    Variant("PCA_50_LW", lambda: LedoitWolfMahalanobis(ALPHA), pca_spec=50),
    Variant("TOPVAR_50_LW", lambda: LedoitWolfMahalanobis(ALPHA), feature_cap=50),
    Variant("TOPVAR_100_LW", lambda: LedoitWolfMahalanobis(ALPHA), feature_cap=100),
    Variant("TOPVAR_250_LW", lambda: LedoitWolfMahalanobis(ALPHA), feature_cap=250),
)


def quantiles(prefix: str, scores: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_q50": float(np.quantile(scores, 0.50)),
        f"{prefix}_q90": float(np.quantile(scores, 0.90)),
        f"{prefix}_q95": float(np.quantile(scores, 0.95)),
        f"{prefix}_q99": float(np.quantile(scores, 0.99)),
        f"{prefix}_max": float(np.max(scores)),
    }


def evaluate_variant(
    variant: Variant,
    source_raw: np.ndarray,
    commissioning_raw: np.ndarray,
    calibration_raw: np.ndarray,
    healthy_raw: np.ndarray,
    anomaly_raw: np.ndarray,
) -> dict[str, object]:
    del source_raw  # TargetOnly must never use source values.
    transform = TrainingTransform(
        pca_spec=variant.pca_spec, feature_cap=variant.feature_cap
    ).fit(commissioning_raw)
    commissioning = transform.transform(commissioning_raw)
    calibration = transform.transform(calibration_raw)
    healthy = transform.transform(healthy_raw)
    anomaly = transform.transform(anomaly_raw)

    detector = variant.detector_factory()
    detector.fit(np.empty((1, commissioning.shape[1])), commissioning)
    calibration_scores = detector.score_samples(calibration)
    detector.calibrate_from_scores(calibration_scores)
    healthy_scores = detector.score_samples(healthy)
    anomaly_scores = detector.score_samples(anomaly)
    if detector.threshold_ is None:
        raise RuntimeError("Calibration did not expose a threshold.")

    fpr = float(np.mean(healthy_scores > detector.threshold_))
    recall = float(np.mean(anomaly_scores > detector.threshold_))
    covariance_condition = np.nan
    shrinkage = np.nan
    if isinstance(detector, LedoitWolfMahalanobis):
        covariance_condition = float(np.linalg.cond(detector.covariance_))
        shrinkage = detector.shrinkage_

    return {
        "retained_features": transform.output_feature_count_,
        "threshold": float(detector.threshold_),
        "FPR": fpr,
        "recall": recall,
        "success": bool(fpr <= ALPHA and recall >= RECALL_TARGET),
        "covariance_condition": covariance_condition,
        "lw_shrinkage": shrinkage,
        **quantiles("calibration", calibration_scores),
        **quantiles("healthy", healthy_scores),
        **quantiles("anomaly", anomaly_scores),
    }


def run(cycles: Sequence[RobotCycle]) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    memberships: dict[str, object] = {"protocol_version": PROTOCOL_VERSION, "seeds": {}}
    for seed in SEEDS:
        split_at_max = create_experiment_split(
            cycles, MAX_N, seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=NORMAL_EVALUATION_SIZE,
            maximum_commissioning_size=MAX_N,
        )
        assert_disjoint(
            source=split_at_max.source_train,
            commissioning=split_at_max.target_commissioning,
            calibration=split_at_max.target_calibration,
            healthy_evaluation=split_at_max.target_normal_evaluation,
            anomaly_evaluation=split_at_max.target_anomaly_evaluation,
        )
        memberships["seeds"][str(seed)] = {
            "commissioning_pool": ids(split_at_max.target_commissioning),
            "calibration": ids(split_at_max.target_calibration),
            "healthy_evaluation": ids(split_at_max.target_normal_evaluation),
            "anomaly_evaluation": ids(split_at_max.target_anomaly_evaluation),
        }
        source_raw = raw_features(split_at_max.source_train)
        calibration_raw = raw_features(split_at_max.target_calibration)
        healthy_raw = raw_features(split_at_max.target_normal_evaluation)
        anomaly_raw = raw_features(split_at_max.target_anomaly_evaluation)

        for N in GRID:
            commissioning_cycles = split_at_max.target_commissioning[:N]
            commissioning_raw = raw_features(commissioning_cycles)
            for variant in VARIANTS:
                LOGGER.info("Processing N=%d seed=%d/19 variant=%s", N, seed, variant.name)
                metrics = evaluate_variant(
                    variant, source_raw, commissioning_raw,
                    calibration_raw, healthy_raw, anomaly_raw,
                )
                rows.append({
                    "seed": seed, "N": N, "detector": "TargetOnly",
                    "variant": variant.name, **metrics,
                    "protocol_version": PROTOCOL_VERSION,
                })
    return pd.DataFrame(rows), memberships


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, N), group in results.groupby(["variant", "N"], sort=False):
        rng = np.random.default_rng(np.random.SeedSequence([GLOBAL_SEED, int(N), len(rows)]))
        recall_low, recall_high = bootstrap_ci(group.recall.to_numpy(), rng)
        fpr_low, fpr_high = bootstrap_ci(group.FPR.to_numpy(), rng)
        rows.append({
            "variant": variant,
            "N": int(N),
            "mean_recall": float(group.recall.mean()),
            "recall_ci_low": recall_low,
            "recall_ci_high": recall_high,
            "mean_FPR": float(group.FPR.mean()),
            "FPR_ci_low": fpr_low,
            "FPR_ci_high": fpr_high,
            "success_rate": float(group.success.mean()),
            "median_threshold": float(group.threshold.median()),
            "median_retained_features": float(group.retained_features.median()),
            "median_covariance_condition": float(group.covariance_condition.median()),
        })
    return pd.DataFrame(rows).sort_values(["N", "mean_recall"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--signal-set", choices=("measured", "machine"), default="measured")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed_everything()
    cycles = load_cycles(args.data_path, signal_set=args.signal_set)
    results, memberships = run(cycles)
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "targetonly_regularization_results.csv", index=False)
    summary.to_csv(args.output_dir / "targetonly_regularization_summary.csv", index=False)
    (args.output_dir / "targetonly_regularization_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nTargetOnly regularization ablation complete:\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()