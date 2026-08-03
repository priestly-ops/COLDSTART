#!/usr/bin/env python3
"""Same-N remove-and-replace ablation for the existing coldstart APIs.

Runs ORIGINAL, ABLATE_SUSPECT, and RANDOM_CONTROL for TargetOnly and RACE
at N=100 for seeds 4 and 19. All non-commissioning memberships are held
fixed, and every intervention replaces exactly one commissioning cycle with
one previously unused healthy target cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "experiments" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import RACEDetector, TargetOnlyDetector  # noqa: E402
from src.evaluation import evaluate_detector  # noqa: E402
from src.feature_extractor import extract_feature_matrix  # noqa: E402
from src.split_generator import (  # noqa: E402
    TARGET_SETTING,
    ExperimentSplit,
    create_experiment_split,
)
from src.voraus_loader import RobotCycle, load_cycles  # noqa: E402

LOGGER = logging.getLogger("outlier_ablation")

GLOBAL_SEED = 42
N = 100
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
PROTOCOL_VERSION = "outlier-ablation-v1.1.0-coldstart-api"

ORIGINAL = "ORIGINAL"
ABLATE_SUSPECT = "ABLATE_SUSPECT"
RANDOM_CONTROL = "RANDOM_CONTROL"
EXPERIMENTS = {4: 1840, 19: 1962}


@dataclass(frozen=True)
class Membership:
    seed: int
    condition: str
    commissioning_ids: list[int]
    calibration_ids: list[int]
    normal_evaluation_ids: list[int]
    anomaly_evaluation_ids: list[int]
    unused_target_ids: list[int]
    removed_episode_id: int | None
    replacement_episode_id: int | None


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ids(cycles: Sequence[RobotCycle]) -> list[int]:
    return [int(cycle.episode_id) for cycle in cycles]


def digest(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_disjoint(**groups: Sequence[int]) -> None:
    for name, values in groups.items():
        if len(values) != len(set(values)):
            raise AssertionError(f"Duplicate episode ID in {name}.")
    names = list(groups)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = set(groups[left_name]) & set(groups[right_name])
            if overlap:
                raise AssertionError(
                    f"Leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:10]}"
                )


def target_order_and_unused(
    cycles: Sequence[RobotCycle], seed: int, split: ExperimentSplit
) -> tuple[list[RobotCycle], list[RobotCycle]]:
    """Reconstruct the target shuffle and verify it against the official split."""
    target_healthy = [
        cycle for cycle in cycles
        if not cycle.anomaly and cycle.setting == TARGET_SETTING
    ]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(target_healthy))
    ordered = [target_healthy[int(index)] for index in permutation]

    pool_end = N
    calibration_end = pool_end + CALIBRATION_SIZE
    evaluation_end = calibration_end + NORMAL_EVALUATION_SIZE

    checks = {
        "commissioning": (ordered[:pool_end], split.target_commissioning),
        "calibration": (
            ordered[pool_end:calibration_end], split.target_calibration
        ),
        "normal_evaluation": (
            ordered[calibration_end:evaluation_end],
            split.target_normal_evaluation,
        ),
    }
    for name, (reconstructed, official) in checks.items():
        if ids(reconstructed) != ids(official):
            raise AssertionError(
                f"Could not reproduce {name} for seed {seed}; do not run "
                "the ablation with a different split protocol."
            )
    return ordered, ordered[evaluation_end:]


def choose_control(commissioning_ids: Sequence[int], suspect: int, seed: int) -> int:
    candidates = sorted(set(commissioning_ids) - {suspect})
    rng = np.random.default_rng(
        np.random.SeedSequence([GLOBAL_SEED, seed, 910_247])
    )
    return int(rng.choice(candidates))


def replace_one(
    original: Sequence[RobotCycle], removed_id: int, replacement: RobotCycle
) -> list[RobotCycle]:
    positions = [
        index for index, cycle in enumerate(original)
        if cycle.episode_id == removed_id
    ]
    if len(positions) != 1:
        raise AssertionError(
            f"Episode {removed_id} occurs {len(positions)} times; expected once."
        )
    if replacement.episode_id in set(ids(original)):
        raise AssertionError("Replacement is already in commissioning set.")
    result = list(original)
    result[positions[0]] = replacement
    if len(result) != N:
        raise AssertionError("Intervention changed commissioning size.")
    return result


def build_conditions(
    cycles: Sequence[RobotCycle], seed: int, suspect: int
) -> tuple[ExperimentSplit, dict[str, list[RobotCycle]], dict[str, Membership]]:
    split = create_experiment_split(
        cycles=cycles,
        commissioning_size=N,
        seed=seed,
        calibration_size=CALIBRATION_SIZE,
        normal_evaluation_size=NORMAL_EVALUATION_SIZE,
        maximum_commissioning_size=N,
    )
    split.verify_no_overlap()
    _, unused = target_order_and_unused(cycles, seed, split)
    if len(unused) < 2:
        raise ValueError("At least two unused healthy target cycles are required.")

    original = list(split.target_commissioning)
    original_ids = ids(original)
    if suspect not in original_ids:
        raise AssertionError(
            f"Suspect {suspect} is absent from seed {seed}, N={N}."
        )

    control = choose_control(original_ids, suspect, seed)
    suspect_replacement = unused[0]
    control_replacement = unused[1]
    condition_cycles = {
        ORIGINAL: original,
        ABLATE_SUSPECT: replace_one(original, suspect, suspect_replacement),
        RANDOM_CONTROL: replace_one(original, control, control_replacement),
    }

    common = {
        "seed": seed,
        "calibration_ids": ids(split.target_calibration),
        "normal_evaluation_ids": ids(split.target_normal_evaluation),
        "anomaly_evaluation_ids": ids(split.target_anomaly_evaluation),
        "unused_target_ids": ids(unused),
    }
    memberships = {
        ORIGINAL: Membership(
            condition=ORIGINAL,
            commissioning_ids=original_ids,
            removed_episode_id=None,
            replacement_episode_id=None,
            **common,
        ),
        ABLATE_SUSPECT: Membership(
            condition=ABLATE_SUSPECT,
            commissioning_ids=ids(condition_cycles[ABLATE_SUSPECT]),
            removed_episode_id=suspect,
            replacement_episode_id=suspect_replacement.episode_id,
            **common,
        ),
        RANDOM_CONTROL: Membership(
            condition=RANDOM_CONTROL,
            commissioning_ids=ids(condition_cycles[RANDOM_CONTROL]),
            removed_episode_id=control,
            replacement_episode_id=control_replacement.episode_id,
            **common,
        ),
    }
    validate_memberships(memberships)
    return split, condition_cycles, memberships


def validate_memberships(memberships: dict[str, Membership]) -> None:
    base = memberships[ORIGINAL]
    invariant_fields = (
        "calibration_ids",
        "normal_evaluation_ids",
        "anomaly_evaluation_ids",
    )
    for condition, membership in memberships.items():
        if len(membership.commissioning_ids) != N:
            raise AssertionError(f"{condition} did not preserve N={N}.")
        for field in invariant_fields:
            if getattr(membership, field) != getattr(base, field):
                raise AssertionError(
                    f"{field} changed under {condition}; "
                    f"base={digest(getattr(base, field))}, "
                    f"condition={digest(getattr(membership, field))}."
                )
        assert_disjoint(
            commissioning=membership.commissioning_ids,
            calibration=membership.calibration_ids,
            normal_evaluation=membership.normal_evaluation_ids,
            anomaly_evaluation=membership.anomaly_evaluation_ids,
        )
        if condition != ORIGINAL:
            removed = set(base.commissioning_ids) - set(membership.commissioning_ids)
            added = set(membership.commissioning_ids) - set(base.commissioning_ids)
            if removed != {membership.removed_episode_id}:
                raise AssertionError(f"{condition} did not remove exactly one expected ID.")
            if added != {membership.replacement_episode_id}:
                raise AssertionError(f"{condition} did not add exactly one expected ID.")


def raw_features(cycles: Sequence[RobotCycle]) -> np.ndarray:
    matrix, extracted_ids = extract_feature_matrix(cycles)
    if extracted_ids.tolist() != ids(cycles):
        raise AssertionError("Feature extraction changed episode ordering.")
    return matrix


def evaluate_condition(
    detector_name: str,
    split: ExperimentSplit,
    commissioning: Sequence[RobotCycle],
    seed: int,
):
    factory = (
        (lambda: TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET))
        if detector_name == "TargetOnly"
        else (lambda: RACEDetector(
            lambda_reg=60.0, false_alert_budget=FALSE_ALERT_BUDGET
        ))
    )
    return evaluate_detector(
        detector_name=detector_name,
        detector_factory=factory,
        source_raw=raw_features(split.source_train),
        target_raw=raw_features(commissioning),
        calibration_raw=raw_features(split.target_calibration),
        normal_evaluation_raw=raw_features(split.target_normal_evaluation),
        anomaly_evaluation_raw=raw_features(split.target_anomaly_evaluation),
        commissioning_size=N,
        seed=seed,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )


def run_once(cycles: Sequence[RobotCycle]):
    rows: list[dict[str, object]] = []
    episode_output = {
        "protocol_version": PROTOCOL_VERSION,
        "N": N,
        "global_seed": GLOBAL_SEED,
        "seeds": {},
    }
    for seed, suspect in EXPERIMENTS.items():
        split, condition_cycles, memberships = build_conditions(cycles, seed, suspect)
        episode_output["seeds"][str(seed)] = {
            name: asdict(value) for name, value in memberships.items()
        }
        for detector_name in ("TargetOnly", "RACE"):
            for condition in (ORIGINAL, ABLATE_SUSPECT, RANDOM_CONTROL):
                LOGGER.info(
                    "seed=%d detector=%s condition=%s", seed, detector_name, condition
                )
                result = evaluate_condition(
                    detector_name, split, condition_cycles[condition], seed
                )
                membership = memberships[condition]
                rows.append({
                    "seed": seed,
                    "N": N,
                    "detector": detector_name,
                    "condition": condition,
                    "removed_episode_id": membership.removed_episode_id,
                    "replacement_episode_id": membership.replacement_episode_id,
                    "threshold": result.threshold,
                    "recall": result.recall,
                    "FPR": result.false_positive_rate,
                    "success": result.success,
                    "retained_features": result.retained_features,
                    "target_weight": result.target_weight,
                })
    frame = pd.DataFrame(rows)
    originals = frame[frame.condition == ORIGINAL][
        ["seed", "detector", "recall", "FPR"]
    ].rename(columns={"recall": "original_recall", "FPR": "original_FPR"})
    frame = frame.merge(originals, on=["seed", "detector"], validate="many_to_one")
    frame["recall_delta"] = frame.recall - frame.original_recall
    frame["FPR_delta"] = frame.FPR - frame.original_FPR
    frame["protocol_version"] = PROTOCOL_VERSION
    frame = frame.sort_values(["seed", "detector", "condition"], kind="stable")
    return frame, episode_output


def summary_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Same-N Outlier Remove-and-Replace Ablation",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`. Each intervention preserves N={N} and "
        "holds calibration and evaluation memberships fixed.",
        "",
        "| Seed | Detector | Condition | Removed | Replacement | Recall | Δ Recall | FPR | Δ FPR | Success |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in frame.itertuples(index=False):
        removed = "—" if pd.isna(row.removed_episode_id) else str(int(row.removed_episode_id))
        replacement = "—" if pd.isna(row.replacement_episode_id) else str(int(row.replacement_episode_id))
        lines.append(
            f"| {row.seed} | {row.detector} | {row.condition} | {removed} | "
            f"{replacement} | {row.recall:.4f} | {row.recall_delta:+.4f} | "
            f"{row.FPR:.4f} | {row.FPR_delta:+.4f} | {'Yes' if row.success else 'No'} |"
        )
    lines.extend(["", "## Matched-control interpretation", ""])
    indexed = frame.set_index(["seed", "detector", "condition"])
    for seed in EXPERIMENTS:
        for detector in ("TargetOnly", "RACE"):
            suspect_delta = float(indexed.loc[(seed, detector, ABLATE_SUSPECT), "recall_delta"])
            control_delta = float(indexed.loc[(seed, detector, RANDOM_CONTROL), "recall_delta"])
            difference = suspect_delta - control_delta
            if difference > 1e-12:
                verdict = "suspect removal improved recall more than random control"
            elif difference < -1e-12:
                verdict = "random control improved recall more than suspect removal"
            else:
                verdict = "both removals produced the same recall change"
            lines.append(
                f"- Seed {seed}, {detector}: {verdict} "
                f"(difference {difference:+.4f})."
            )
    lines.extend([
        "",
        "These are controlled seed-level mechanism tests, not proof of a general causal effect.",
    ])
    return "\n".join(lines) + "\n"


def canonical(frame: pd.DataFrame, memberships: dict[str, object]) -> tuple[bytes, str]:
    csv_bytes = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    json_text = json.dumps(memberships, sort_keys=True, separators=(",", ":"))
    return csv_bytes, json_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="Path to the voraus parquet file.")
    parser.add_argument("--signal-set", choices=("measured", "machine"), default="measured")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--verify-determinism", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    seed_everything()
    cycles = load_cycles(args.data_path, signal_set=args.signal_set)
    frame, memberships = run_once(cycles)
    if args.verify_determinism:
        seed_everything()
        second_frame, second_memberships = run_once(cycles)
        if canonical(frame, memberships) != canonical(second_frame, second_memberships):
            raise AssertionError("Determinism verification failed.")
        LOGGER.info("Determinism verification passed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "outlier_ablation_results.csv", index=False, float_format="%.12g")
    (args.output_dir / "outlier_ablation_episode_ids.json").write_text(
        json.dumps(memberships, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "outlier_ablation_summary.md").write_text(
        summary_markdown(frame), encoding="utf-8"
    )
    print(frame[[
        "seed", "detector", "condition", "removed_episode_id",
        "replacement_episode_id", "recall", "FPR", "recall_delta",
        "FPR_delta", "success"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()