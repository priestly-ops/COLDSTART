#!/usr/bin/env python3
"""Post-hoc feature-stability diagnostic for TargetOnly seed failures.

This script intentionally uses anomaly-evaluation labels only for diagnosis.
All preprocessing, Top-250 selection, Gaussian fitting, and calibration use
healthy commissioning/calibration cycles exactly as in the main protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector  # noqa: E402
from src.feature_extractor import (  # noqa: E402
    extract_feature_matrix,
    make_feature_names,
)
from src.split_generator import create_experiment_split  # noqa: E402
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("feature_stability")
GLOBAL_SEED = 42
SEEDS = (4, 9, 19, 0, 3, 13)
FAILED_SEEDS = frozenset((4, 9, 19))
GRID = (10, 25, 50, 100)
MAX_N = 100
TOP_K = 250
CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
ALPHA = 0.01
PROTOCOL_VERSION = "feature-stability-v1.0.0"


def cycle_ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, ids = extract_feature_matrix(cycles)
    if ids.tolist() != cycle_ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    return np.asarray(matrix, dtype=np.float64)


def assert_disjoint(**groups: Sequence[RobotCycle]) -> None:
    sets = {name: set(cycle_ids(value)) for name, value in groups.items()}
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"Leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )


def safe_effect_size(healthy: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """Signed standardized mean difference using pooled evaluation SD."""
    nh, na = healthy.shape[0], anomaly.shape[0]
    vh = np.var(healthy, axis=0, ddof=1)
    va = np.var(anomaly, axis=0, ddof=1)
    pooled = np.sqrt(((nh - 1) * vh + (na - 1) * va) / (nh + na - 2))
    difference = np.mean(anomaly, axis=0) - np.mean(healthy, axis=0)
    result = np.zeros_like(difference)
    valid = pooled > 1e-12
    result[valid] = difference[valid] / pooled[valid]
    result[~valid & (np.abs(difference) > 1e-12)] = np.sign(
        difference[~valid & (np.abs(difference) > 1e-12)]
    ) * np.inf
    return result


def univariate_aurocs(healthy: np.ndarray, anomaly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.concatenate((np.zeros(len(healthy)), np.ones(len(anomaly))))
    values = np.vstack((healthy, anomaly))
    raw = np.full(values.shape[1], 0.5, dtype=np.float64)
    for feature_index in range(values.shape[1]):
        column = values[:, feature_index]
        if np.ptp(column) > 0:
            raw[feature_index] = roc_auc_score(labels, column)
    oriented = np.maximum(raw, 1.0 - raw)
    return raw, oriented


class DiagnosticTransform:
    """Commissioning-only variance filter, optional Top-K, and scaler."""

    def __init__(self, top_k: int | None) -> None:
        self.top_k = top_k

    def fit(self, commissioning: np.ndarray) -> "DiagnosticTransform":
        self.filter_ = VarianceThreshold(1e-12).fit(commissioning)
        self.nonconstant_indices_ = np.flatnonzero(self.filter_.get_support())
        filtered = self.filter_.transform(commissioning)
        filtered_variance = np.var(filtered, axis=0, ddof=1)
        ranked_local = np.argsort(-filtered_variance, kind="stable")
        keep_count = len(ranked_local) if self.top_k is None else min(self.top_k, len(ranked_local))
        self.selected_local_ = np.sort(ranked_local[:keep_count])
        self.selected_raw_indices_ = self.nonconstant_indices_[self.selected_local_]
        selected = filtered[:, self.selected_local_]
        self.scaler_ = StandardScaler().fit(selected)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        filtered = self.filter_.transform(matrix)[:, self.selected_local_]
        transformed = self.scaler_.transform(filtered)
        if not np.isfinite(transformed).all():
            raise RuntimeError("Transformation produced NaN or Inf.")
        return transformed


def fit_model(transform: DiagnosticTransform, commissioning_raw: np.ndarray) -> LedoitWolf:
    commissioning = transform.transform(commissioning_raw)
    return LedoitWolf(store_precision=True).fit(commissioning)


def scores_and_contributions(
    model: LedoitWolf,
    transformed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered = transformed - model.location_
    # Per-feature signed terms sum exactly to squared Mahalanobis distance.
    contributions = centered * (centered @ model.precision_)
    squared = np.maximum(np.sum(contributions, axis=1), 0.0)
    return np.sqrt(squared), contributions


def quantile_rows(seed: int, n: int, variant: str, split_name: str,
                  episode_ids: Sequence[int], scores: np.ndarray,
                  threshold: float) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "seed_group": "failed" if seed in FAILED_SEEDS else "control",
            "N": n,
            "variant": variant,
            "split": split_name,
            "episode_id": int(episode_id),
            "score": float(score),
            "threshold": threshold,
            "above_threshold": bool(score > threshold),
        }
        for episode_id, score in zip(episode_ids, scores, strict=True)
    ]


def run(cycles: Sequence[RobotCycle]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    feature_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    selections: dict[tuple[int, int], set[int]] = {}
    variance_ranks: dict[tuple[int, int], np.ndarray] = {}
    memberships: dict[str, object] = {"protocol_version": PROTOCOL_VERSION, "seeds": {}}

    for seed in SEEDS:
        split = create_experiment_split(
            cycles, MAX_N, seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
            maximum_commissioning_size=MAX_N,
        )
        assert_disjoint(
            source=split.source_train,
            commissioning=split.target_commissioning,
            calibration=split.target_calibration,
            healthy=split.target_normal_evaluation,
            anomaly=split.target_anomaly_evaluation,
        )
        feature_names = make_feature_names(split.target_commissioning[0].columns)
        calibration_raw = features(split.target_calibration)
        healthy_raw = features(split.target_normal_evaluation)
        anomaly_raw = features(split.target_anomaly_evaluation)
        effect = safe_effect_size(healthy_raw, anomaly_raw)
        auc_raw, auc_oriented = univariate_aurocs(healthy_raw, anomaly_raw)
        memberships["seeds"][str(seed)] = {
            "commissioning_pool_ids": cycle_ids(split.target_commissioning),
            "calibration_ids": cycle_ids(split.target_calibration),
            "healthy_evaluation_ids": cycle_ids(split.target_normal_evaluation),
            "anomaly_evaluation_ids": cycle_ids(split.target_anomaly_evaluation),
        }

        for n in GRID:
            LOGGER.info("Processing seed=%d N=%d", seed, n)
            commissioning_cycles = split.target_commissioning[:n]
            commissioning_raw = features(commissioning_cycles)
            variance = np.var(commissioning_raw, axis=0, ddof=1)
            ranks = rankdata(-variance, method="min").astype(int)
            variance_ranks[(seed, n)] = ranks

            transforms = {
                "LW_CURRENT": DiagnosticTransform(None).fit(commissioning_raw),
                "TOPVAR_250_LW": DiagnosticTransform(TOP_K).fit(commissioning_raw),
            }
            selections[(seed, n)] = set(transforms["TOPVAR_250_LW"].selected_raw_indices_.tolist())
            contribution_stats: dict[str, dict[str, np.ndarray]] = {}

            for variant, transform in transforms.items():
                model = fit_model(transform, commissioning_raw)
                selected = transform.selected_raw_indices_
                calibration = transform.transform(calibration_raw)
                healthy = transform.transform(healthy_raw)
                anomaly = transform.transform(anomaly_raw)
                calibration_scores, calibration_contrib = scores_and_contributions(model, calibration)
                healthy_scores, healthy_contrib = scores_and_contributions(model, healthy)
                anomaly_scores, anomaly_contrib = scores_and_contributions(model, anomaly)
                threshold = BaseDetector.conformal_quantile(calibration_scores, ALPHA)
                score_rows.extend(quantile_rows(seed, n, variant, "calibration", cycle_ids(split.target_calibration), calibration_scores, threshold))
                score_rows.extend(quantile_rows(seed, n, variant, "healthy_evaluation", cycle_ids(split.target_normal_evaluation), healthy_scores, threshold))
                score_rows.extend(quantile_rows(seed, n, variant, "anomaly_evaluation", cycle_ids(split.target_anomaly_evaluation), anomaly_scores, threshold))
                contribution_stats[variant] = {
                    "selected": selected,
                    "calibration": np.mean(calibration_contrib, axis=0),
                    "healthy": np.mean(healthy_contrib, axis=0),
                    "anomaly": np.mean(anomaly_contrib, axis=0),
                    "anomaly_abs": np.mean(np.abs(anomaly_contrib), axis=0),
                }

            selected_top = selections[(seed, n)]
            maps: dict[str, dict[int, int]] = {}
            for variant, stats in contribution_stats.items():
                maps[variant] = {int(raw): local for local, raw in enumerate(stats["selected"])}

            for index, name in enumerate(feature_names):
                row: dict[str, object] = {
                    "seed": seed,
                    "seed_group": "failed" if seed in FAILED_SEEDS else "control",
                    "N": n,
                    "feature_index": index,
                    "feature_name": name,
                    "commissioning_variance": float(variance[index]),
                    "commissioning_variance_rank": int(ranks[index]),
                    "standardized_effect_size": float(effect[index]),
                    "absolute_effect_size": float(abs(effect[index])),
                    "univariate_auroc_raw": float(auc_raw[index]),
                    "univariate_auroc_oriented": float(auc_oriented[index]),
                    "retained_by_top250": index in selected_top,
                    "excluded_by_top250": index not in selected_top,
                }
                for variant, prefix in (("LW_CURRENT", "lw"), ("TOPVAR_250_LW", "top250")):
                    local = maps[variant].get(index)
                    stats = contribution_stats[variant]
                    row[f"{prefix}_retained"] = local is not None
                    for split_name in ("calibration", "healthy", "anomaly", "anomaly_abs"):
                        row[f"{prefix}_{split_name}_mean_mahalanobis_contribution"] = (
                            np.nan if local is None else float(stats[split_name][local])
                        )
                feature_rows.append(row)

    overlap_rows: list[dict[str, object]] = []
    keys = sorted(selections)
    for left_position, left in enumerate(keys):
        for right in keys[left_position + 1 :]:
            left_set, right_set = selections[left], selections[right]
            union = left_set | right_set
            rho = spearmanr(variance_ranks[left], variance_ranks[right]).statistic
            overlap_rows.append({
                "left_seed": left[0], "left_N": left[1],
                "right_seed": right[0], "right_N": right[1],
                "same_seed": left[0] == right[0], "same_N": left[1] == right[1],
                "top250_intersection": len(left_set & right_set),
                "top250_union": len(union),
                "top250_jaccard": float(len(left_set & right_set) / len(union)) if union else 1.0,
                "top250_overlap_coefficient": float(len(left_set & right_set) / min(len(left_set), len(right_set))),
                "variance_rank_spearman": float(rho),
            })

    return pd.DataFrame(feature_rows), pd.DataFrame(overlap_rows), pd.DataFrame(score_rows), memberships


def write_outputs(feature_frame: pd.DataFrame, overlap_frame: pd.DataFrame,
                  score_frame: pd.DataFrame, memberships: dict[str, object],
                  output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output_dir / "feature_stability_features.csv", index=False, float_format="%.12g")
    overlap_frame.to_csv(output_dir / "feature_stability_rank_overlap.csv", index=False, float_format="%.12g")
    score_frame.to_csv(output_dir / "feature_stability_score_distributions.csv", index=False, float_format="%.12g")
    (output_dir / "feature_stability_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    np.random.seed(GLOBAL_SEED)
    LOGGER.info("Loading %s", args.data_path)
    cycles = load_cycles(args.data_path)
    feature_frame, overlap_frame, score_frame, memberships = run(cycles)
    write_outputs(feature_frame, overlap_frame, score_frame, memberships, args.output_dir)
    LOGGER.info("Saved diagnostic outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()