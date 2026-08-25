from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.e4_certification_sample_size import (
    healthy_bound_curve,
    minimum_anomaly_sample_size,
    minimum_healthy_sample_size,
    recall_bound_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e4_certification_sample_size"
HEALTHY_CURVE_PATH = OUTPUT_DIR / "e4_healthy_bound_curve.csv"
RECALL_CURVE_PATH = OUTPUT_DIR / "e4_recall_bound_curve.csv"
MINIMUMS_PATH = OUTPUT_DIR / "e4_minimum_evidence.csv"
FEASIBILITY_PATH = OUTPUT_DIR / "e4_dataset_feasibility.csv"
MANIFEST_PATH = OUTPUT_DIR / "e4_manifest.json"

PROTOCOL_VERSION = "coldstart-e4-certification-sample-size-v1"
JOINT_CONFIDENCE = 0.95
FAMILYWISE_DELTA = 1.0 - JOINT_CONFIDENCE
DELTA_RECALL = FAMILYWISE_DELTA / 2.0
DELTA_FPR = FAMILYWISE_DELTA / 2.0
FPR_BUDGET = 0.01
RECALL_TARGET = 0.90

# Frozen voraus-AD facts inherited from E3/E0 protocol.
TARGET_HEALTHY_TOTAL = 319
TARGET_ANOMALY_TOTAL = 755
PRIMARY_COMMISSIONING_N = 100
PRIMARY_CALIBRATION_M = 119
FROZEN_HEALTHY_EVAL_N = 100

HEALTHY_GRID = sorted(
    set(
        [25, 50, 75, 99, 100, 119, 150, 200, 250, 300, 350, 360, 365, 366, 367,
         368, 369, 375, 400, 500, 555, 600, 720, 750, 1000]
    )
)
ANOMALY_GRID = sorted(
    set([10, 20, 30, 34, 35, 36, 37, 40, 50, 54, 70, 85, 100, 200, 500, 755])
)
FP_SCENARIOS = [0, 1, 2, 3]
FN_SCENARIOS = [0, 1, 2, 3]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    healthy_rows = []
    minimum_rows = []
    for fp in FP_SCENARIOS:
        healthy_rows.extend(
            healthy_bound_curve(
                false_positives=fp,
                sample_sizes=HEALTHY_GRID,
                delta_fpr=DELTA_FPR,
            )
        )
        result = minimum_healthy_sample_size(
            false_positives=fp,
            fpr_budget=FPR_BUDGET,
            delta_fpr=DELTA_FPR,
        )
        minimum_rows.append(
            {
                "side": "healthy_fpr",
                "error_count": fp,
                "minimum_sample_size": result.minimum_sample_size,
                "bound_at_minimum": result.bound_at_minimum,
                "target": result.target,
                "delta": result.delta,
            }
        )

    recall_rows = []
    for fn in FN_SCENARIOS:
        recall_rows.extend(
            recall_bound_curve(
                false_negatives=fn,
                sample_sizes=ANOMALY_GRID,
                delta_recall=DELTA_RECALL,
            )
        )
        result = minimum_anomaly_sample_size(
            false_negatives=fn,
            recall_target=RECALL_TARGET,
            delta_recall=DELTA_RECALL,
        )
        minimum_rows.append(
            {
                "side": "anomaly_recall",
                "error_count": fn,
                "minimum_sample_size": result.minimum_sample_size,
                "bound_at_minimum": result.bound_at_minimum,
                "target": result.target,
                "delta": result.delta,
            }
        )

    pd.DataFrame(healthy_rows).to_csv(HEALTHY_CURVE_PATH, index=False)
    pd.DataFrame(recall_rows).to_csv(RECALL_CURVE_PATH, index=False)
    pd.DataFrame(minimum_rows).to_csv(MINIMUMS_PATH, index=False)

    remaining_after_primary = (
        TARGET_HEALTHY_TOTAL - PRIMARY_COMMISSIONING_N - PRIMARY_CALIBRATION_M
    )
    feasibility = pd.DataFrame(
        [
            {
                "scenario": "entire_target_healthy_dataset",
                "available_independent_healthy_episodes": TARGET_HEALTHY_TOTAL,
                "required_zero_fp_episodes": 368,
                "can_certify_zero_fp_best_case": TARGET_HEALTHY_TOTAL >= 368,
                "note": "Even assigning every target-healthy episode to certification would not reach the exact zero-FP requirement.",
            },
            {
                "scenario": "E3_primary_N100_M119",
                "available_independent_healthy_episodes": remaining_after_primary,
                "required_zero_fp_episodes": 368,
                "can_certify_zero_fp_best_case": remaining_after_primary >= 368,
                "note": "N=100 commissioning and Mcal=119 leave only 100 healthy episodes for independent certification.",
            },
            {
                "scenario": "frozen_E0_E1_E3_eval",
                "available_independent_healthy_episodes": FROZEN_HEALTHY_EVAL_N,
                "required_zero_fp_episodes": 368,
                "can_certify_zero_fp_best_case": FROZEN_HEALTHY_EVAL_N >= 368,
                "note": "This is the reviewer-facing frozen evaluation budget used in E0-E3.",
            },
        ]
    )
    feasibility.to_csv(FEASIBILITY_PATH, index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_type": "exact_finite_sample_evidence_budget",
        "detector_training": "none; detector and threshold are conceptually frozen before certification",
        "joint_confidence": JOINT_CONFIDENCE,
        "bonferroni_delta_recall": DELTA_RECALL,
        "bonferroni_delta_fpr": DELTA_FPR,
        "false_alert_budget": FPR_BUDGET,
        "recall_target": RECALL_TARGET,
        "healthy_fp_scenarios": FP_SCENARIOS,
        "anomaly_fn_scenarios": FN_SCENARIOS,
        "healthy_grid": HEALTHY_GRID,
        "anomaly_grid": ANOMALY_GRID,
        "voraus_target_healthy_total": TARGET_HEALTHY_TOTAL,
        "voraus_anomaly_total": TARGET_ANOMALY_TOTAL,
        "primary_E3_commissioning_size": PRIMARY_COMMISSIONING_N,
        "primary_E3_calibration_size": PRIMARY_CALIBRATION_M,
        "frozen_healthy_evaluation_size": FROZEN_HEALTHY_EVAL_N,
        "important_interpretation": (
            "E4 estimates independent evaluation evidence requirements after a detector and threshold are frozen. "
            "It is not a model-training or calibration sample-size curve, and bootstrap/resampling duplicates are not counted as independent certification episodes."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("COLDSTART E4: EXACT CERTIFICATION SAMPLE-SIZE EXPERIMENT")
    print("=" * 78)
    print(pd.DataFrame(minimum_rows).to_string(index=False))
    print("\nDataset feasibility:")
    print(feasibility.to_string(index=False))
    print("\nE4 complete.")


if __name__ == "__main__":
    main()
