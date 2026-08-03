#!/usr/bin/env python3
"""Same-size calibration intervention and raw-cycle audit for episode 1710.

For seeds 4, 9, and 19 at N=100, compare:
  * ORIGINAL: unchanged 100-cycle calibration set;
  * REMOVE_1710: replace calibration episode 1710 with the first unused
    target-healthy episode in the seed's deterministic shuffled order;
  * RANDOM_CONTROL: replace one deterministically selected non-1710
    calibration episode with that same unused episode.

Commissioning, source, healthy evaluation, anomaly evaluation, and calibration
size are invariant across conditions. The raw-cycle audit is descriptive and
does not alter any experiment membership.
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
from src.split_generator import (  # noqa: E402
    TARGET_SETTING,
    create_experiment_split,
)
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("calibration_1710_intervention")

GLOBAL_SEED = 42
SEEDS = (4, 9, 19)
N_COMMISSIONING = 100
CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
SUSPECT_EPISODE_ID = 1710
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
PROTOCOL_VERSION = "calibration-1710-intervention-v1.0.0"

ORIGINAL = "ORIGINAL"
REMOVE_SUSPECT = "REMOVE_1710"
RANDOM_CONTROL = "RANDOM_CONTROL"
CONDITION_ORDER = (ORIGINAL, REMOVE_SUSPECT, RANDOM_CONTROL)


def cycle_ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, ids = extract_feature_matrix(cycles)
    if ids.tolist() != cycle_ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    return np.asarray(matrix, dtype=np.float64)


def digest(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_pairwise_disjoint(**groups: Sequence[RobotCycle]) -> None:
    id_sets = {name: set(cycle_ids(value)) for name, value in groups.items()}
    names = list(id_sets)
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            overlap = id_sets[left] & id_sets[right]
            if overlap:
                raise AssertionError(
                    f"Episode leakage between {left} and {right}: "
                    f"{sorted(overlap)[:10]}"
                )


def reconstruct_unused_target(
    cycles: Sequence[RobotCycle], seed: int
) -> tuple[RobotCycle, ...]:
    """Recover target healthy cycles after commission/calibration/evaluation."""
    target_healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly and cycle.setting == TARGET_SETTING
    ]
    permutation = np.random.default_rng(seed).permutation(len(target_healthy))
    shuffled = [target_healthy[int(index)] for index in permutation]
    used_count = (
        N_COMMISSIONING + CALIBRATION_SIZE + HEALTHY_EVALUATION_SIZE
    )
    unused = tuple(shuffled[used_count:])
    if not unused:
        raise RuntimeError(f"Seed {seed} has no unused target healthy episode.")
    return unused


def replace_cycle(
    original: Sequence[RobotCycle], removed_id: int, replacement: RobotCycle
) -> tuple[RobotCycle, ...]:
    ids = cycle_ids(original)
    if ids.count(removed_id) != 1:
        raise AssertionError(
            f"Expected episode {removed_id} exactly once; observed "
            f"{ids.count(removed_id)} times."
        )
    if replacement.episode_id in ids:
        raise AssertionError("Replacement is already in the calibration set.")
    result = tuple(
        replacement if cycle.episode_id == removed_id else cycle
        for cycle in original
    )
    if len(result) != len(original):
        raise AssertionError("Calibration size changed during replacement.")
    return result


def choose_control_id(calibration: Sequence[RobotCycle], seed: int) -> int:
    candidates = sorted(
        cycle.episode_id
        for cycle in calibration
        if cycle.episode_id != SUSPECT_EPISODE_ID
    )
    if not candidates:
        raise RuntimeError("No eligible random-control calibration episode.")
    rng = np.random.default_rng(GLOBAL_SEED + 10_000 + seed)
    return int(candidates[int(rng.integers(0, len(candidates)))])


def detector_specs():
    return (
        ("TargetOnly", TargetOnlyDetector),
        ("RACE", RACEDetector),
    )


def evaluate_condition(
    detector_name: str,
    detector_factory,
    source_raw: np.ndarray,
    commissioning_raw: np.ndarray,
    calibration_raw: np.ndarray,
    healthy_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    episode_ids_by_split: dict[str, list[int]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    detector, preprocessor, _, _ = fit_detector(
        detector_name=detector_name,
        detector_factory=detector_factory,
        source_raw=source_raw,
        target_raw=commissioning_raw,
    )
    transformed = {
        "calibration": preprocessor.transform(calibration_raw),
        "healthy_evaluation": preprocessor.transform(healthy_raw),
        "anomaly_evaluation": preprocessor.transform(anomaly_raw),
    }
    scores = {
        name: detector.score_samples(matrix)
        for name, matrix in transformed.items()
    }
    detector.calibrate_from_scores(scores["calibration"])
    if detector.threshold_ is None:
        raise RuntimeError("Calibration did not produce a threshold.")
    threshold = float(detector.threshold_)
    fpr = float(np.mean(scores["healthy_evaluation"] > threshold))
    recall = float(np.mean(scores["anomaly_evaluation"] > threshold))
    metrics = {
        "threshold": threshold,
        "recall": recall,
        "false_positive_rate": fpr,
        "success": bool(fpr <= FALSE_ALERT_BUDGET and recall >= RECALL_TARGET),
        "retained_features": int(preprocessor.output_feature_count_),
    }
    score_rows: list[dict[str, object]] = []
    for split_name, split_scores in scores.items():
        for episode_id, score in zip(
            episode_ids_by_split[split_name], split_scores, strict=True
        ):
            score_rows.append(
                {
                    "split": split_name,
                    "episode_id": int(episode_id),
                    "score": float(score),
                    "threshold": threshold,
                    "above_threshold": bool(score > threshold),
                }
            )
    return metrics, score_rows


def run_intervention(
    cycles: Sequence[RobotCycle],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    result_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    memberships: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "suspect_episode_id": SUSPECT_EPISODE_ID,
        "seeds": {},
    }

    for seed in SEEDS:
        LOGGER.info("Preparing seed=%d N=%d", seed, N_COMMISSIONING)
        split = create_experiment_split(
            cycles,
            commissioning_size=N_COMMISSIONING,
            seed=seed,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
            maximum_commissioning_size=N_COMMISSIONING,
        )
        split.verify_no_overlap()
        original = split.target_calibration
        if SUSPECT_EPISODE_ID not in cycle_ids(original):
            raise AssertionError(
                f"Episode {SUSPECT_EPISODE_ID} is absent from seed {seed} "
                "calibration; protocol reconstruction does not match."
            )
        unused = reconstruct_unused_target(cycles, seed)
        replacement = unused[0]
        control_id = choose_control_id(original, seed)
        conditions = {
            ORIGINAL: tuple(original),
            REMOVE_SUSPECT: replace_cycle(
                original, SUSPECT_EPISODE_ID, replacement
            ),
            RANDOM_CONTROL: replace_cycle(original, control_id, replacement),
        }
        for name, calibration in conditions.items():
            assert_pairwise_disjoint(
                source=split.source_train,
                commissioning=split.target_commissioning,
                calibration=calibration,
                healthy=split.target_normal_evaluation,
                anomaly=split.target_anomaly_evaluation,
            )
            if len(calibration) != CALIBRATION_SIZE:
                raise AssertionError("Calibration size is not invariant.")

        memberships["seeds"][str(seed)] = {
            "source_digest": digest(cycle_ids(split.source_train)),
            "commissioning_ids": cycle_ids(split.target_commissioning),
            "healthy_evaluation_ids": cycle_ids(
                split.target_normal_evaluation
            ),
            "anomaly_evaluation_ids": cycle_ids(
                split.target_anomaly_evaluation
            ),
            "replacement_episode_id": int(replacement.episode_id),
            "random_control_removed_episode_id": control_id,
            "conditions": {
                name: cycle_ids(calibration)
                for name, calibration in conditions.items()
            },
        }

        source_raw = features(split.source_train)
        commissioning_raw = features(split.target_commissioning)
        healthy_raw = features(split.target_normal_evaluation)
        anomaly_raw = features(split.target_anomaly_evaluation)

        for detector_name, detector_factory in detector_specs():
            for condition_name in CONDITION_ORDER:
                LOGGER.info(
                    "Running seed=%d detector=%s condition=%s",
                    seed,
                    detector_name,
                    condition_name,
                )
                calibration = conditions[condition_name]
                calibration_raw = features(calibration)
                episode_ids_by_split = {
                    "calibration": cycle_ids(calibration),
                    "healthy_evaluation": cycle_ids(
                        split.target_normal_evaluation
                    ),
                    "anomaly_evaluation": cycle_ids(
                        split.target_anomaly_evaluation
                    ),
                }
                metrics, condition_scores = evaluate_condition(
                    detector_name,
                    detector_factory,
                    source_raw,
                    commissioning_raw,
                    calibration_raw,
                    healthy_raw,
                    anomaly_raw,
                    episode_ids_by_split,
                )
                removed_id = {
                    ORIGINAL: None,
                    REMOVE_SUSPECT: SUSPECT_EPISODE_ID,
                    RANDOM_CONTROL: control_id,
                }[condition_name]
                base = {
                    "protocol_version": PROTOCOL_VERSION,
                    "seed": seed,
                    "N": N_COMMISSIONING,
                    "detector": detector_name,
                    "condition": condition_name,
                    "removed_episode_id": removed_id,
                    "replacement_episode_id": (
                        None if condition_name == ORIGINAL
                        else int(replacement.episode_id)
                    ),
                }
                result_rows.append({**base, **metrics})
                score_rows.extend({**base, **row} for row in condition_scores)

    results = pd.DataFrame(result_rows)
    originals = results.loc[
        results["condition"] == ORIGINAL,
        ["seed", "detector", "threshold", "recall", "false_positive_rate"],
    ].rename(
        columns={
            "threshold": "original_threshold",
            "recall": "original_recall",
            "false_positive_rate": "original_false_positive_rate",
        }
    )
    results = results.merge(
        originals, on=["seed", "detector"], how="left", validate="many_to_one"
    )
    results["threshold_delta"] = (
        results["threshold"] - results["original_threshold"]
    )
    results["recall_delta"] = results["recall"] - results["original_recall"]
    results["false_positive_rate_delta"] = (
        results["false_positive_rate"]
        - results["original_false_positive_rate"]
    )
    return results, pd.DataFrame(score_rows), memberships


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return 0.6744897501960817 * (values - median) / mad


def audit_episode_1710(
    cycles: Sequence[RobotCycle],
) -> tuple[pd.DataFrame, dict[str, object]]:
    target = [cycle for cycle in cycles if cycle.episode_id == SUSPECT_EPISODE_ID]
    if len(target) != 1:
        raise AssertionError(
            f"Expected one episode {SUSPECT_EPISODE_ID}; found {len(target)}."
        )
    suspect = target[0]
    peers = [
        cycle
        for cycle in cycles
        if not cycle.anomaly and cycle.setting == TARGET_SETTING
        and cycle.columns == suspect.columns
    ]
    if suspect.anomaly or suspect.setting != TARGET_SETTING:
        raise AssertionError("Episode 1710 is not target-setting healthy data.")
    suspect_peer_index = next(
        index
        for index, cycle in enumerate(peers)
        if cycle.episode_id == SUSPECT_EPISODE_ID
    )

    peer_lengths = np.asarray([len(cycle.values) for cycle in peers], dtype=float)
    signal_rows: list[dict[str, object]] = []
    suspect_values = np.asarray(suspect.values, dtype=np.float64)
    for index, name in enumerate(suspect.columns):
        column = suspect_values[:, index]
        peer_range = np.asarray(
            [np.ptp(cycle.values[:, index]) for cycle in peers], dtype=float
        )
        peer_tv = np.asarray(
            [np.sum(np.abs(np.diff(cycle.values[:, index]))) for cycle in peers],
            dtype=float,
        )
        max_jump = float(np.max(np.abs(np.diff(column))))
        peer_jump = np.asarray(
            [np.max(np.abs(np.diff(cycle.values[:, index]))) for cycle in peers],
            dtype=float,
        )
        signal_rows.append(
            {
                "signal_index": index,
                "signal_name": name,
                "sample_count": len(column),
                "nan_count": int(np.isnan(column).sum()),
                "inf_count": int(np.isinf(column).sum()),
                "unique_count": int(np.unique(column).size),
                "constant": bool(np.ptp(column) <= 1e-12),
                "minimum": float(np.min(column)),
                "maximum": float(np.max(column)),
                "mean": float(np.mean(column)),
                "standard_deviation": float(np.std(column, ddof=1)),
                "range": float(np.ptp(column)),
                "range_robust_z_vs_target_healthy": float(robust_z(peer_range)[suspect_peer_index]),
                "total_variation": float(np.sum(np.abs(np.diff(column)))),
                "total_variation_robust_z_vs_target_healthy": float(robust_z(peer_tv)[suspect_peer_index]),
                "maximum_absolute_jump": max_jump,
                "maximum_jump_robust_z_vs_target_healthy": float(robust_z(peer_jump)[suspect_peer_index]),
                "start_end_difference": float(column[-1] - column[0]),
            }
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "episode_id": SUSPECT_EPISODE_ID,
        "anomaly_label": bool(suspect.anomaly),
        "category": int(suspect.category),
        "setting": int(suspect.setting),
        "sample_count": int(len(suspect.values)),
        "signal_count": int(suspect.values.shape[1]),
        "finite": bool(np.isfinite(suspect.values).all()),
        "sample_count_robust_z_vs_target_healthy": float(
            robust_z(peer_lengths)[suspect_peer_index]
        ),
        "target_healthy_peer_count": len(peers),
        "constant_signal_count": int(
            np.sum(np.ptp(suspect.values, axis=0) <= 1e-12)
        ),
    }
    return pd.DataFrame(signal_rows), summary


def canonical_csv(frame: pd.DataFrame, sort_columns: list[str]) -> bytes:
    return frame.sort_values(sort_columns, kind="stable").to_csv(
        index=False, float_format="%.12g", lineterminator="\n"
    ).encode("utf-8")


def write_outputs(
    results: pd.DataFrame,
    scores: pd.DataFrame,
    memberships: dict[str, object],
    audit: pd.DataFrame,
    audit_summary: dict[str, object],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_dir / "calibration_1710_results.csv",
        index=False,
        float_format="%.12g",
    )
    scores.to_csv(
        output_dir / "calibration_1710_score_distributions.csv",
        index=False,
        float_format="%.12g",
    )
    audit.to_csv(
        output_dir / "calibration_1710_raw_cycle_audit.csv",
        index=False,
        float_format="%.12g",
    )
    (output_dir / "calibration_1710_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "calibration_1710_raw_cycle_summary.json").write_text(
        json.dumps(audit_summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def verify_determinism(cycles: Sequence[RobotCycle]) -> None:
    LOGGER.info("Running deterministic replay")
    first_results, first_scores, first_memberships = run_intervention(cycles)
    second_results, second_scores, second_memberships = run_intervention(cycles)
    if canonical_csv(first_results, ["seed", "detector", "condition"]) != canonical_csv(
        second_results, ["seed", "detector", "condition"]
    ):
        raise AssertionError("Result metrics changed on deterministic replay.")
    if canonical_csv(
        first_scores,
        ["seed", "detector", "condition", "split", "episode_id"],
    ) != canonical_csv(
        second_scores,
        ["seed", "detector", "condition", "split", "episode_id"],
    ):
        raise AssertionError("Score distributions changed on deterministic replay.")
    if json.dumps(first_memberships, sort_keys=True) != json.dumps(
        second_memberships, sort_keys=True
    ):
        raise AssertionError("Memberships changed on deterministic replay.")
    LOGGER.info("Determinism check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs"
    )
    parser.add_argument("--verify-determinism", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    np.random.seed(GLOBAL_SEED)
    LOGGER.info("Loading %s", args.data_path)
    cycles = load_cycles(args.data_path)
    if args.verify_determinism:
        verify_determinism(cycles)
    results, scores, memberships = run_intervention(cycles)
    audit, audit_summary = audit_episode_1710(cycles)
    write_outputs(
        results, scores, memberships, audit, audit_summary, args.output_dir
    )
    LOGGER.info("Saved calibration-intervention outputs to %s", args.output_dir)
    print(
        results[
            [
                "seed",
                "detector",
                "condition",
                "removed_episode_id",
                "replacement_episode_id",
                "threshold",
                "recall",
                "false_positive_rate",
                "success",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()