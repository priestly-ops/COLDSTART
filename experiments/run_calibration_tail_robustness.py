#!/usr/bin/env python3
"""Predeclared calibration-tail robustness experiment.

Across 20 seeds, N in {10, 25, 50, 100}, TargetOnly and RACE, and
calibration sizes {100, all available}, compare:

* MAX_SCORE: the current conservative maximum calibration score;
* FINITE_SAMPLE: the standard finite-sample split-conformal order statistic;
* CONTEXT_CONDITIONAL: a finite-sample threshold from the 100 calibration
  cycles with most similar phase-resampled joint trajectories;
* TRIMMED_SENSITIVITY: remove the largest 1% of calibration scores before
  applying the finite-sample quantile (sensitivity analysis only).

The original 100-cycle commissioning pool, 100-cycle calibration set, and
100-cycle healthy evaluation set are reconstructed exactly. Larger calibration
sets add only target-healthy cycles that were unused by the original split;
commissioning and evaluation memberships never move.

NOTE: the predeclared 200-cycle calibration condition was dropped. It would
require 400 target-healthy cycles (100 commissioning + 100 base calibration
+ 100 healthy evaluation + 100 unused, to reach a pool of >=200), but the
target domain has only 319 target-healthy cycles available in total. The
"all available" condition (~119 calibration cycles) is the largest feasible
size under this fixed protocol structure and stands in for the 200 tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import RACEDetector, TargetOnlyDetector  # noqa: E402
from src.evaluation import fit_detector  # noqa: E402
from src.feature_extractor import extract_feature_matrix  # noqa: E402
from src.split_generator import TARGET_SETTING, create_experiment_split  # noqa: E402
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("calibration_tail_robustness")

GLOBAL_SEED = 42
SEEDS = tuple(range(20))
COMMISSIONING_GRID = (10, 25, 50, 100)
MAXIMUM_COMMISSIONING_SIZE = 100
BASE_CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
CALIBRATION_REQUESTS: tuple[int | str, ...] = (100, "all")  # 200 dropped: infeasible given the 319-cycle target-healthy pool (see note below)
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
SUSPECT_EPISODE_ID = 1710
CONTEXT_NEIGHBORS = 100
PHASE_BINS = 20
TRIM_FRACTION = 0.01
BOOTSTRAP_SAMPLES = 10_000
PROTOCOL_VERSION = "calibration-tail-robustness-v1.0.0"

METHODS = (
    "MAX_SCORE",
    "FINITE_SAMPLE",
    "CONTEXT_CONDITIONAL",
    "TRIMMED_SENSITIVITY",
)


def cycle_ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def digest(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, ids = extract_feature_matrix(cycles)
    if ids.tolist() != cycle_ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    return np.asarray(matrix, dtype=np.float64)


def target_healthy_shuffle(
    cycles: Sequence[RobotCycle], seed: int
) -> tuple[RobotCycle, ...]:
    healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly and cycle.setting == TARGET_SETTING
    ]
    order = np.random.default_rng(seed).permutation(len(healthy))
    return tuple(healthy[int(index)] for index in order)


def calibration_pool_preserving_original_split(
    cycles: Sequence[RobotCycle], seed: int
) -> tuple[tuple[RobotCycle, ...], tuple[RobotCycle, ...]]:
    """Return nested calibration pool and fixed original evaluation set."""
    shuffled = target_healthy_shuffle(cycles, seed)
    commission_end = MAXIMUM_COMMISSIONING_SIZE
    base_calibration_end = commission_end + BASE_CALIBRATION_SIZE
    evaluation_end = base_calibration_end + HEALTHY_EVALUATION_SIZE
    if len(shuffled) < evaluation_end:
        raise ValueError("Not enough target healthy cycles for fixed protocol.")

    base = shuffled[commission_end:base_calibration_end]
    evaluation = shuffled[base_calibration_end:evaluation_end]
    unused = shuffled[evaluation_end:]
    pool = tuple(base) + tuple(unused)
    if len(pool) < 1:
        raise ValueError(
            f"Seed {seed} has only {len(pool)} eligible calibration cycles; "
            "no calibration condition can run."
        )
    return pool, tuple(evaluation)


def select_calibration(
    pool: Sequence[RobotCycle], request: int | str
) -> tuple[RobotCycle, ...]:
    if request == "all":
        return tuple(pool)
    count = int(request)
    if count > len(pool):
        raise ValueError(f"Requested {count} calibration cycles; have {len(pool)}.")
    return tuple(pool[:count])


def assert_disjoint(**groups: Sequence[RobotCycle]) -> None:
    sets = {name: set(cycle_ids(value)) for name, value in groups.items()}
    names = list(sets)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"Episode leakage between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )


def finite_sample_rank(sample_count: int, alpha: float) -> tuple[int, bool]:
    raw_rank = int(np.ceil((sample_count + 1) * (1.0 - alpha)))
    return min(raw_rank, sample_count), raw_rank <= sample_count


def finite_threshold(scores: np.ndarray) -> tuple[float, int, bool]:
    values = np.asarray(scores, dtype=np.float64)
    rank, supported = finite_sample_rank(len(values), FALSE_ALERT_BUDGET)
    return float(np.sort(values)[rank - 1]), rank, supported


def trimmed_threshold(scores: np.ndarray) -> tuple[float, int, int, bool]:
    values = np.sort(np.asarray(scores, dtype=np.float64))
    trim_count = max(1, int(np.floor(len(values) * TRIM_FRACTION)))
    if trim_count >= len(values):
        raise ValueError("Trimming removed the entire calibration set.")
    retained = values[:-trim_count]
    threshold, rank, supported = finite_threshold(retained)
    return threshold, rank, trim_count, supported


def joint_phase_indices(columns: Sequence[str]) -> list[int]:
    lowered = [name.lower() for name in columns]
    preferred = [
        index
        for index, name in enumerate(lowered)
        if "joint_position" in name or "motor_position" in name
    ]
    if preferred:
        return preferred
    fallback = [index for index, name in enumerate(lowered) if "position" in name]
    if fallback:
        return fallback
    raise ValueError("No joint/motor position signals found for phase context.")


def phase_descriptor(cycle: RobotCycle, indices: Sequence[int]) -> np.ndarray:
    values = np.asarray(cycle.values[:, indices], dtype=np.float64)
    source_phase = np.linspace(0.0, 1.0, values.shape[0])
    target_phase = np.linspace(0.0, 1.0, PHASE_BINS)
    resampled = np.column_stack(
        [np.interp(target_phase, source_phase, values[:, column])
         for column in range(values.shape[1])]
    )
    return resampled.reshape(-1)


def context_matrices(
    calibration: Sequence[RobotCycle],
    queries: Sequence[RobotCycle],
) -> tuple[np.ndarray, np.ndarray]:
    if calibration[0].columns != queries[0].columns:
        raise ValueError("Calibration and query signal schemas differ.")
    indices = joint_phase_indices(calibration[0].columns)
    calibration_raw = np.vstack([phase_descriptor(cycle, indices) for cycle in calibration])
    query_raw = np.vstack([phase_descriptor(cycle, indices) for cycle in queries])
    location = np.mean(calibration_raw, axis=0)
    scale = np.std(calibration_raw, axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (calibration_raw - location) / scale, (query_raw - location) / scale


def context_thresholds(
    calibration_scores: np.ndarray,
    calibration_cycles: Sequence[RobotCycle],
    query_cycles: Sequence[RobotCycle],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    calibration_context, query_context = context_matrices(
        calibration_cycles, query_cycles
    )
    neighbor_count = min(CONTEXT_NEIGHBORS, len(calibration_cycles))
    rank, supported = finite_sample_rank(neighbor_count, FALSE_ALERT_BUDGET)
    thresholds = np.empty(len(query_cycles), dtype=np.float64)
    includes_1710 = np.zeros(len(query_cycles), dtype=bool)
    determined_by_1710 = np.zeros(len(query_cycles), dtype=bool)
    calibration_ids = np.asarray(cycle_ids(calibration_cycles), dtype=np.int64)

    for query_index, query in enumerate(query_context):
        squared = np.sum((calibration_context - query) ** 2, axis=1)
        # Stable ordering makes ties deterministic.
        neighbors = np.argsort(squared, kind="stable")[:neighbor_count]
        neighbor_scores = calibration_scores[neighbors]
        threshold = float(np.sort(neighbor_scores)[rank - 1])
        thresholds[query_index] = threshold
        suspect_positions = neighbors[calibration_ids[neighbors] == SUSPECT_EPISODE_ID]
        includes_1710[query_index] = len(suspect_positions) > 0
        if len(suspect_positions):
            determined_by_1710[query_index] = bool(
                np.isclose(
                    calibration_scores[int(suspect_positions[0])],
                    threshold,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
    return thresholds, includes_1710, determined_by_1710, rank, supported


def detector_specs():
    return (
        ("TargetOnly", TargetOnlyDetector),
        ("RACE", RACEDetector),
    )


def global_method_threshold(
    method: str, calibration_scores: np.ndarray
) -> tuple[float, int, int, bool]:
    if method == "MAX_SCORE":
        return float(np.max(calibration_scores)), len(calibration_scores), 0, False
    if method == "FINITE_SAMPLE":
        threshold, rank, supported = finite_threshold(calibration_scores)
        return threshold, rank, 0, supported
    if method == "TRIMMED_SENSITIVITY":
        threshold, rank, trimmed, supported = trimmed_threshold(calibration_scores)
        return threshold, rank, trimmed, supported
    raise ValueError(f"Not a global threshold method: {method}")


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_experiment(
    cycles: Sequence[RobotCycle],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    result_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    memberships: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "seeds": {},
    }

    anomaly_cycles = tuple(cycle for cycle in cycles if cycle.anomaly)
    anomaly_raw = features(anomaly_cycles)

    for seed in SEEDS:
        pool, fixed_evaluation = calibration_pool_preserving_original_split(cycles, seed)
        reference = create_experiment_split(
            cycles,
            commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            seed=seed,
            calibration_size=BASE_CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
            maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
        )
        if cycle_ids(pool[:BASE_CALIBRATION_SIZE]) != cycle_ids(reference.target_calibration):
            raise AssertionError("Original 100-cycle calibration membership changed.")
        if cycle_ids(fixed_evaluation) != cycle_ids(reference.target_normal_evaluation):
            raise AssertionError("Original healthy evaluation membership changed.")

        memberships["seeds"][str(seed)] = {
            "commissioning_pool_ids": cycle_ids(reference.target_commissioning),
            "healthy_evaluation_ids": cycle_ids(fixed_evaluation),
            "anomaly_evaluation_digest": digest(cycle_ids(anomaly_cycles)),
            "calibration_pool_ids": cycle_ids(pool),
            "calibration_sizes": {
                str(request): cycle_ids(select_calibration(pool, request))
                for request in CALIBRATION_REQUESTS
            },
        }
        source_raw = features(reference.source_train)
        evaluation_raw = features(fixed_evaluation)

        for n in COMMISSIONING_GRID:
            commissioning = reference.target_commissioning[:n]
            commissioning_raw = features(commissioning)
            assert_disjoint(
                source=reference.source_train,
                commissioning=commissioning,
                calibration=pool,
                healthy_evaluation=fixed_evaluation,
                anomaly_evaluation=anomaly_cycles,
            )
            for detector_name, detector_factory in detector_specs():
                LOGGER.info("Fitting seed=%d N=%d detector=%s", seed, n, detector_name)
                detector, preprocessor, _, _ = fit_detector(
                    detector_name=detector_name,
                    detector_factory=detector_factory,
                    source_raw=source_raw,
                    target_raw=commissioning_raw,
                )
                healthy_scores = detector.score_samples(preprocessor.transform(evaluation_raw))
                anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

                for request in CALIBRATION_REQUESTS:
                    calibration = select_calibration(pool, request)
                    calibration_scores = detector.score_samples(
                        preprocessor.transform(features(calibration))
                    )
                    calibration_ids = np.asarray(cycle_ids(calibration), dtype=np.int64)
                    suspect_mask = calibration_ids == SUSPECT_EPISODE_ID
                    suspect_present = bool(np.any(suspect_mask))
                    suspect_score = (
                        float(calibration_scores[suspect_mask][0]) if suspect_present else np.nan
                    )
                    label = str(request)

                    for split_name, ids, values in (
                        ("calibration", calibration_ids, calibration_scores),
                        ("healthy_evaluation", np.asarray(cycle_ids(fixed_evaluation)), healthy_scores),
                        ("anomaly_evaluation", np.asarray(cycle_ids(anomaly_cycles)), anomaly_scores),
                    ):
                        score_rows.extend(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "seed": seed,
                                "N": n,
                                "detector": detector_name,
                                "calibration_request": label,
                                "calibration_size": len(calibration),
                                "split": split_name,
                                "episode_id": int(episode_id),
                                "score": float(score),
                                "is_episode_1710": int(episode_id) == SUSPECT_EPISODE_ID,
                            }
                            for episode_id, score in zip(ids, values, strict=True)
                        )

                    all_queries = tuple(fixed_evaluation) + anomaly_cycles
                    context_values = context_thresholds(
                        calibration_scores, calibration, all_queries
                    )
                    context_threshold_values, context_includes, context_determines, context_rank, context_supported = context_values
                    healthy_count = len(fixed_evaluation)

                    for method in METHODS:
                        if method == "CONTEXT_CONDITIONAL":
                            healthy_thresholds = context_threshold_values[:healthy_count]
                            anomaly_thresholds = context_threshold_values[healthy_count:]
                            predictions_healthy = healthy_scores > healthy_thresholds
                            predictions_anomaly = anomaly_scores > anomaly_thresholds
                            thresholds = context_threshold_values
                            determines = context_determines
                            includes = context_includes
                            rank = context_rank
                            trim_count = 0
                            supported = context_supported
                        else:
                            threshold, rank, trim_count, supported = global_method_threshold(
                                method, calibration_scores
                            )
                            predictions_healthy = healthy_scores > threshold
                            predictions_anomaly = anomaly_scores > threshold
                            thresholds = np.full(len(all_queries), threshold)
                            includes = np.full(len(all_queries), suspect_present)
                            determines_flag = bool(
                                suspect_present
                                and np.isclose(suspect_score, threshold, rtol=1e-12, atol=1e-12)
                            )
                            determines = np.full(len(all_queries), determines_flag)

                        fpr = float(np.mean(predictions_healthy))
                        recall = float(np.mean(predictions_anomaly))
                        coverage = 1.0 - fpr
                        base = {
                            "protocol_version": PROTOCOL_VERSION,
                            "seed": seed,
                            "N": n,
                            "detector": detector_name,
                            "calibration_method": method,
                            "calibration_request": label,
                            "calibration_size": len(calibration),
                        }
                        result_rows.append(
                            {
                                **base,
                                "threshold": float(np.median(thresholds)),
                                "threshold_min": float(np.min(thresholds)),
                                "threshold_max": float(np.max(thresholds)),
                                "threshold_rank": rank,
                                "finite_sample_alpha_supported": supported,
                                "trimmed_count": trim_count,
                                "recall": recall,
                                "false_positive_rate": fpr,
                                "empirical_coverage": coverage,
                                "success": bool(recall >= RECALL_TARGET and fpr <= FALSE_ALERT_BUDGET),
                                "episode_1710_present": suspect_present,
                                "episode_1710_score": suspect_score,
                                "episode_1710_determines_any_threshold": bool(np.any(determines)),
                                "episode_1710_threshold_fraction": float(np.mean(determines)),
                                "episode_1710_neighbor_fraction": float(np.mean(includes)),
                                "retained_features": int(preprocessor.output_feature_count_),
                            }
                        )
                        for query_index, query_cycle in enumerate(all_queries):
                            threshold_rows.append(
                                {
                                    **base,
                                    "query_split": (
                                        "healthy_evaluation" if query_index < healthy_count
                                        else "anomaly_evaluation"
                                    ),
                                    "query_episode_id": int(query_cycle.episode_id),
                                    "threshold": float(thresholds[query_index]),
                                    "episode_1710_in_context": bool(includes[query_index]),
                                    "episode_1710_determines_threshold": bool(determines[query_index]),
                                }
                            )

    results = pd.DataFrame(result_rows)
    summary_rows: list[dict[str, object]] = []
    group_columns = [
        "detector", "N", "calibration_method", "calibration_request", "calibration_size"
    ]
    for keys, group in results.groupby(group_columns, sort=True, dropna=False):
        recall_low, recall_high = bootstrap_interval(
            group["recall"].to_numpy(), GLOBAL_SEED + int(keys[1])
        )
        fpr_low, fpr_high = bootstrap_interval(
            group["false_positive_rate"].to_numpy(), GLOBAL_SEED + 1000 + int(keys[1])
        )
        summary_rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "seed_count": len(group),
                "mean_recall": float(group["recall"].mean()),
                "recall_ci_lower": recall_low,
                "recall_ci_upper": recall_high,
                "mean_false_positive_rate": float(group["false_positive_rate"].mean()),
                "fpr_ci_lower": fpr_low,
                "fpr_ci_upper": fpr_high,
                "mean_empirical_coverage": float(group["empirical_coverage"].mean()),
                "success_rate": float(group["success"].mean()),
                "episode_1710_threshold_seed_rate": float(
                    group["episode_1710_determines_any_threshold"].mean()
                ),
            }
        )
    return (
        results,
        pd.DataFrame(summary_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(threshold_rows),
        memberships,
    )


def canonical_csv(frame: pd.DataFrame, columns: list[str]) -> bytes:
    return frame.sort_values(columns, kind="stable").to_csv(
        index=False, float_format="%.12g", lineterminator="\n"
    ).encode("utf-8")


def verify_determinism(cycles: Sequence[RobotCycle]) -> None:
    LOGGER.info("Running complete deterministic replay")
    first = run_experiment(cycles)
    second = run_experiment(cycles)
    sort_sets = (
        ["seed", "N", "detector", "calibration_request", "calibration_method"],
        ["detector", "N", "calibration_request", "calibration_method"],
        ["seed", "N", "detector", "calibration_request", "split", "episode_id"],
        ["seed", "N", "detector", "calibration_request", "calibration_method", "query_split", "query_episode_id"],
    )
    for index, sort_columns in enumerate(sort_sets):
        if canonical_csv(first[index], sort_columns) != canonical_csv(second[index], sort_columns):
            raise AssertionError(f"Determinism failed for output table {index}.")
    if json.dumps(first[4], sort_keys=True) != json.dumps(second[4], sort_keys=True):
        raise AssertionError("Determinism failed for memberships.")
    LOGGER.info("Determinism check passed")


def write_outputs(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    thresholds: pd.DataFrame,
    memberships: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "calibration_tail_results.csv", index=False, float_format="%.12g")
    summary.to_csv(output_dir / "calibration_tail_summary.csv", index=False, float_format="%.12g")
    scores.to_csv(output_dir / "calibration_tail_score_distributions.csv", index=False, float_format="%.12g")
    thresholds.to_csv(output_dir / "calibration_tail_query_thresholds.csv", index=False, float_format="%.12g")
    (output_dir / "calibration_tail_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
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
    LOGGER.info("Saved calibration-tail outputs to %s", args.output_dir)
    print(outputs[1].to_string(index=False))


if __name__ == "__main__":
    main()