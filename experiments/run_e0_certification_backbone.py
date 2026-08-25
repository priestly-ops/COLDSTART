from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detectors import RACEDetector, TargetOnlyDetector
from src.evaluation import evaluate_detector
from src.feature_extractor import extract_feature_matrix
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e0_certification"
SEED_RESULTS_PATH = OUTPUT_DIR / "e0_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e0_summary.csv"
N_STAR_PATH = OUTPUT_DIR / "e0_n_star.json"
MANIFEST_PATH = OUTPUT_DIR / "e0_manifest.json"

COMMISSIONING_GRID = [10, 25, 50, 100]
SEEDS = list(range(20))
EVALUATION_SEED = 42
GLOBAL_SEED = 42

FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
COMMISSIONING_SUCCESS_THRESHOLD = 0.80

CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100

PROTOCOL_VERSION = "coldstart-e0-certification-v1"
DETECTOR_NAMES = ("TargetOnly", "RACE")

np.random.seed(GLOBAL_SEED)


CHECKPOINT_COLUMNS = [
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "false_positive_rate",
    "recall",
    "success",
    "empirical_success",
    "certified_success",
    "tp",
    "fn",
    "fp",
    "tn",
    "recall_lower",
    "fpr_upper",
    "joint_confidence",
    "delta_recall",
    "delta_fpr",
    "calibration_alpha",
    "conformal_rank",
    "conformal_regime",
    "calibration_size",
    "threshold",
    "retained_features",
    "target_weight",
]


def detector_factory(detector_name: str):
    if detector_name == "TargetOnly":
        return lambda: TargetOnlyDetector(
            false_alert_budget=FALSE_ALERT_BUDGET
        )
    if detector_name == "RACE":
        return lambda: RACEDetector(
            lambda_reg=60.0,
            false_alert_budget=FALSE_ALERT_BUDGET,
        )
    raise ValueError(f"Unsupported E0 detector: {detector_name}")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)

    frame = pd.read_csv(SEED_RESULTS_PATH)
    required = {
        "protocol_version",
        "detector",
        "commissioning_size",
        "seed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Existing E0 checkpoint is incompatible; missing columns: "
            f"{sorted(missing)}"
        )

    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(
            "Existing E0 checkpoint uses a different protocol version: "
            f"{sorted(versions)}"
        )

    return frame


def completed_keys(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    if frame.empty:
        return set()
    return {
        (str(row.detector), int(row.commissioning_size), int(row.seed))
        for row in frame.itertuples(index=False)
    }


def save_rows(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[CHECKPOINT_COLUMNS].sort_values(
        ["commissioning_size", "seed", "detector"]
    )
    _atomic_write_csv(frame, SEED_RESULTS_PATH)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, n_value), group in results.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        certified_rate = float(group["certified_success"].astype(bool).mean())
        empirical_rate = float(group["empirical_success"].astype(bool).mean())
        rows.append(
            {
                "detector": str(detector),
                "commissioning_size": int(n_value),
                "number_of_seeds": int(len(group)),
                "mean_recall": float(group["recall"].mean()),
                "mean_fpr": float(group["false_positive_rate"].mean()),
                "mean_recall_lower": float(group["recall_lower"].mean()),
                "mean_fpr_upper": float(group["fpr_upper"].mean()),
                "empirical_success_rate": empirical_rate,
                "certified_success_rate": certified_rate,
                "certified_seed_count": int(
                    group["certified_success"].astype(bool).sum()
                ),
                "infinite_calibration_rate": float(
                    (group["conformal_regime"] == "infinite").mean()
                ),
                "maximum_calibration_rate": float(
                    (group["conformal_regime"] == "maximum").mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["detector", "commissioning_size"]
    ).reset_index(drop=True)


def estimate_n_star(summary: pd.DataFrame) -> dict[str, str | int]:
    estimates: dict[str, str | int] = {}
    for detector in DETECTOR_NAMES:
        subset = summary[summary["detector"] == detector].sort_values(
            "commissioning_size"
        )
        qualifying = subset[
            subset["certified_success_rate"]
            >= COMMISSIONING_SUCCESS_THRESHOLD
        ]
        if qualifying.empty:
            estimates[detector] = (
                f"Censored (>{max(COMMISSIONING_GRID)})"
            )
        else:
            estimates[detector] = int(
                qualifying.iloc[0]["commissioning_size"]
            )
    return estimates


def minimum_zero_failure_healthy_eval_size() -> int:
    delta_fpr = (1.0 - JOINT_CONFIDENCE) / 2.0
    return int(
        math.ceil(
            math.log(delta_fpr)
            / math.log(1.0 - FALSE_ALERT_BUDGET)
        )
    )


def minimum_perfect_recall_anomaly_eval_size() -> int:
    delta_recall = (1.0 - JOINT_CONFIDENCE) / 2.0
    return int(
        math.ceil(
            math.log(delta_recall)
            / math.log(RECALL_TARGET)
        )
    )


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cycles = load_cycles(
        path=DATASET_PATH,
        signal_set="measured",
    )

    checkpoint = load_checkpoint()
    rows = checkpoint.to_dict(orient="records") if not checkpoint.empty else []
    finished = completed_keys(checkpoint)

    expected = {
        (detector, n_value, seed)
        for detector in DETECTOR_NAMES
        for n_value in COMMISSIONING_GRID
        for seed in SEEDS
    }

    print("=" * 78)
    print("COLDSTART E0: EXACT CERTIFICATION BACKBONE")
    print("=" * 78)
    print(f"Protocol:                  {PROTOCOL_VERSION}")
    print(f"Detectors:                 {list(DETECTOR_NAMES)}")
    print(f"Commissioning grid:        {COMMISSIONING_GRID}")
    print(f"Commissioning seeds:       {SEEDS}")
    print(f"Evaluation seed:           {EVALUATION_SEED}")
    print(f"Recall target:             {RECALL_TARGET}")
    print(f"FPR budget / alpha:        {FALSE_ALERT_BUDGET}")
    print(f"Joint confidence:          {JOINT_CONFIDENCE}")
    print(f"Commission success q0:     {COMMISSIONING_SUCCESS_THRESHOLD}")
    print(f"Calibration size:          {CALIBRATION_SIZE}")
    print(f"Healthy evaluation size:   {NORMAL_EVALUATION_SIZE}")
    print(
        "Best-case healthy n needed for exact joint certification: "
        f"{minimum_zero_failure_healthy_eval_size()}"
    )
    print(
        "Best-case anomaly n needed for exact joint certification: "
        f"{minimum_perfect_recall_anomaly_eval_size()}"
    )
    print(f"Completed rows:            {len(finished & expected)}/{len(expected)}")
    print("=" * 78)

    fixed_calibration_ids: tuple[int, ...] | None = None
    fixed_healthy_eval_ids: tuple[int, ...] | None = None
    fixed_anomaly_eval_ids: tuple[int, ...] | None = None
    observed_anomaly_eval_size: int | None = None

    completed_counter = len(finished & expected)

    for seed in SEEDS:
        for n_value in COMMISSIONING_GRID:
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n_value,
                commissioning_seed=seed,
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )

            calibration_ids = tuple(
                cycle.episode_id for cycle in split.target_calibration
            )
            healthy_eval_ids = tuple(
                cycle.episode_id for cycle in split.target_normal_evaluation
            )
            anomaly_eval_ids = tuple(
                cycle.episode_id for cycle in split.target_anomaly_evaluation
            )

            if fixed_calibration_ids is None:
                fixed_calibration_ids = calibration_ids
                fixed_healthy_eval_ids = healthy_eval_ids
                fixed_anomaly_eval_ids = anomaly_eval_ids
                observed_anomaly_eval_size = len(anomaly_eval_ids)
            else:
                if calibration_ids != fixed_calibration_ids:
                    raise RuntimeError("Calibration IDs changed across E0 seeds/N.")
                if healthy_eval_ids != fixed_healthy_eval_ids:
                    raise RuntimeError("Healthy evaluation IDs changed across E0 seeds/N.")
                if anomaly_eval_ids != fixed_anomaly_eval_ids:
                    raise RuntimeError("Anomaly evaluation IDs changed across E0 seeds/N.")

            source_raw, _ = extract_feature_matrix(split.source_train)
            target_raw, _ = extract_feature_matrix(split.target_commissioning)
            calibration_raw, _ = extract_feature_matrix(split.target_calibration)
            healthy_eval_raw, _ = extract_feature_matrix(
                split.target_normal_evaluation
            )
            anomaly_eval_raw, _ = extract_feature_matrix(
                split.target_anomaly_evaluation
            )

            for detector in DETECTOR_NAMES:
                key = (detector, n_value, seed)
                if key in finished:
                    continue

                completed_counter += 1
                print(
                    f"E0 {completed_counter}/{len(expected)} | "
                    f"detector={detector} N={n_value} seed={seed}"
                )

                result = evaluate_detector(
                    detector_name=detector,
                    detector_factory=detector_factory(detector),
                    source_raw=source_raw,
                    target_raw=target_raw,
                    calibration_raw=calibration_raw,
                    normal_evaluation_raw=healthy_eval_raw,
                    anomaly_evaluation_raw=anomaly_eval_raw,
                    commissioning_size=n_value,
                    seed=seed,
                    false_alert_budget=FALSE_ALERT_BUDGET,
                    recall_target=RECALL_TARGET,
                    joint_confidence=JOINT_CONFIDENCE,
                )

                row = asdict(result)
                row["protocol_version"] = PROTOCOL_VERSION
                rows.append(row)
                finished.add(key)
                save_rows(rows)

    results = pd.DataFrame(rows)
    summary = build_summary(results)
    _atomic_write_csv(summary, SUMMARY_PATH)

    n_star = estimate_n_star(summary)
    N_STAR_PATH.write_text(
        json.dumps(n_star, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if observed_anomaly_eval_size is None:
        raise RuntimeError("E0 did not observe an anomaly evaluation set.")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "commissioning_grid": COMMISSIONING_GRID,
        "commissioning_seeds": SEEDS,
        "detectors": list(DETECTOR_NAMES),
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "delta_recall": (1.0 - JOINT_CONFIDENCE) / 2.0,
        "delta_fpr": (1.0 - JOINT_CONFIDENCE) / 2.0,
        "commissioning_success_threshold": COMMISSIONING_SUCCESS_THRESHOLD,
        "calibration_alpha_mapping": "alpha_equals_B",
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": NORMAL_EVALUATION_SIZE,
        "anomaly_evaluation_size": observed_anomaly_eval_size,
        "best_case_minimum_healthy_eval_size": (
            minimum_zero_failure_healthy_eval_size()
        ),
        "best_case_minimum_anomaly_eval_size": (
            minimum_perfect_recall_anomaly_eval_size()
        ),
        "healthy_certification_possible_even_with_zero_fp": bool(
            NORMAL_EVALUATION_SIZE
            >= minimum_zero_failure_healthy_eval_size()
        ),
        "frozen_calibration_ids": list(fixed_calibration_ids or ()),
        "frozen_healthy_eval_ids": list(fixed_healthy_eval_ids or ()),
        "frozen_anomaly_eval_ids": list(fixed_anomaly_eval_ids or ()),
        "n_star": n_star,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nE0 complete.")
    print(summary.to_string(index=False))
    print(f"\nN*: {n_star}")
    if not manifest["healthy_certification_possible_even_with_zero_fp"]:
        print(
            "IMPORTANT: the frozen healthy evaluation set is too small to "
            "certify FPR <= 0.01 at simultaneous 95% confidence even if "
            "zero false positives are observed. This is a protocol result, "
            "not a detector failure."
        )


if __name__ == "__main__":
    main()
