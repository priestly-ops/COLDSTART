from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detectors import TargetOnlyDetector
from src.e3_calibration_phase import evaluate_threshold_from_scores, nested_calibration_order
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility
from src.split_generator import TARGET_SETTING, create_frozen_evaluation_split
from src.voraus_loader import RobotCycle, load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e3_calibration_phase"
SEED_RESULTS_PATH = OUTPUT_DIR / "e3_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e3_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e3_manifest.json"

PROTOCOL_VERSION = "coldstart-e3-calibration-phase-v1"
GLOBAL_SEED = 42
EVALUATION_SEED = 42
CALIBRATION_ORDER_SEED = 314159
SEEDS = list(range(20))
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
HEALTHY_EVALUATION_SIZE = 100
E0_CALIBRATION_SIZE = 100
E0_MAX_COMMISSIONING_SIZE = 100

# Primary arm: the E1-proven calibration-limited TargetOnly setting.
PRIMARY_N = 100
PRIMARY_CALIBRATION_GRID = [25, 50, 75, 99, 100, 119]

# Structural arm: N=10 is the only frozen-evaluation setting that leaves enough
# independent target-healthy cycles to cross the theoretical M=198 -> 199
# maximum-to-submaximum transition while retaining 100 healthy eval episodes.
STRUCTURAL_N = 10
STRUCTURAL_CALIBRATION_GRID = [150, 198, 199, 209]

np.random.seed(GLOBAL_SEED)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _target_healthy(cycles: list[RobotCycle]) -> list[RobotCycle]:
    return sorted(
        [c for c in cycles if not c.anomaly and c.setting == TARGET_SETTING],
        key=lambda c: c.episode_id,
    )


def _extra_pool(
    all_target_healthy: list[RobotCycle],
    commissioning: tuple[RobotCycle, ...],
    base_calibration: tuple[RobotCycle, ...],
    healthy_eval: tuple[RobotCycle, ...],
) -> tuple[RobotCycle, ...]:
    used_ids = {
        c.episode_id
        for c in commissioning + base_calibration + healthy_eval
    }
    return tuple(c for c in all_target_healthy if c.episode_id not in used_ids)


def _detector_factory():
    return TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET)


def _load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SEED_RESULTS_PATH)
    if "protocol_version" not in frame.columns:
        raise ValueError("Existing E3 checkpoint lacks protocol_version.")
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(f"Incompatible E3 checkpoint versions: {sorted(versions)}")
    return frame


def _completed_keys(frame: pd.DataFrame) -> set[tuple[str, int, int, int]]:
    if frame.empty:
        return set()
    return {
        (
            str(row.arm),
            int(row.commissioning_size),
            int(row.calibration_size),
            int(row.seed),
        )
        for row in frame.itertuples(index=False)
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, n_value, m_value), group in results.groupby(
        ["arm", "commissioning_size", "calibration_size"], sort=True
    ):
        rows.append(
            {
                "arm": str(arm),
                "commissioning_size": int(n_value),
                "calibration_size": int(m_value),
                "number_of_seeds": int(len(group)),
                "conformal_rank": int(group["conformal_rank"].iloc[0]),
                "conformal_regime": str(group["conformal_regime"].iloc[0]),
                "minimum_attainable_alpha": float(
                    group["minimum_attainable_alpha"].iloc[0]
                ),
                "mean_recall": float(group["recall"].mean()),
                "mean_fpr": float(group["false_positive_rate"].mean()),
                "mean_auroc": float(group["auroc"].mean()),
                "mean_oracle_recall_at_fpr_budget": float(
                    group["oracle_recall_at_fpr_budget"].mean()
                ),
                "oracle_feasibility_rate": float(
                    group["oracle_empirically_feasible"].astype(bool).mean()
                ),
                "empirical_success_rate": float(
                    group["empirical_success"].astype(bool).mean()
                ),
                "certified_success_rate": float(
                    group["certified_success"].astype(bool).mean()
                ),
                "representation_limited_rate": float(
                    (group["bottleneck_label"] == "representation_limited").mean()
                ),
                "calibration_limited_rate": float(
                    (group["bottleneck_label"] == "calibration_limited").mean()
                ),
                "certification_limited_rate": float(
                    (group["bottleneck_label"] == "certification_limited").mean()
                ),
                "certified_rate": float(
                    (group["bottleneck_label"] == "certified").mean()
                ),
                "mean_threshold": float(group["threshold"].replace(np.inf, np.nan).mean())
                if np.isfinite(group["threshold"]).any()
                else np.inf,
                "infinite_threshold_rate": float(np.isinf(group["threshold"]).mean()),
                "mean_seedwise_recall_lower": float(group["recall_lower"].mean()),
                "mean_seedwise_fpr_upper": float(group["fpr_upper"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["arm", "commissioning_size", "calibration_size"]
    ).reset_index(drop=True)


def _run_arm(
    *,
    cycles: list[RobotCycle],
    target_healthy: list[RobotCycle],
    arm: str,
    commissioning_size: int,
    calibration_grid: list[int],
    rows: list[dict[str, Any]],
    completed: set[tuple[str, int, int, int]],
) -> None:
    for seed in SEEDS:
        base_split = create_frozen_evaluation_split(
            cycles=cycles,
            commissioning_size=commissioning_size,
            commissioning_seed=seed,
            evaluation_seed=EVALUATION_SEED,
            calibration_size=E0_CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
            maximum_commissioning_size=E0_MAX_COMMISSIONING_SIZE,
        )

        extra = _extra_pool(
            target_healthy,
            base_split.target_commissioning,
            base_split.target_calibration,
            base_split.target_normal_evaluation,
        )
        calibration_master = nested_calibration_order(
            base_split.target_calibration,
            extra,
            ordering_seed=CALIBRATION_ORDER_SEED + commissioning_size,
        )

        max_requested = max(calibration_grid)
        if len(calibration_master) < max_requested:
            raise RuntimeError(
                f"Arm {arm} seed {seed} has only {len(calibration_master)} "
                f"independent calibration candidates; requested {max_requested}."
            )

        source_raw, _ = extract_feature_matrix(base_split.source_train)
        target_raw, _ = extract_feature_matrix(base_split.target_commissioning)
        healthy_raw, _ = extract_feature_matrix(base_split.target_normal_evaluation)
        anomaly_raw, _ = extract_feature_matrix(base_split.target_anomaly_evaluation)
        calibration_raw, _ = extract_feature_matrix(calibration_master)

        detector, preprocessor, _, _ = fit_detector(
            detector_name="TargetOnly",
            detector_factory=_detector_factory,
            source_raw=source_raw,
            target_raw=target_raw,
        )

        calibration_features = preprocessor.transform(calibration_raw)
        healthy_features = preprocessor.transform(healthy_raw)
        anomaly_features = preprocessor.transform(anomaly_raw)

        calibration_scores = detector.score_samples(calibration_features)
        healthy_scores = detector.score_samples(healthy_features)
        anomaly_scores = detector.score_samples(anomaly_features)

        oracle = empirical_oracle_feasibility(
            healthy_scores=healthy_scores,
            anomaly_scores=anomaly_scores,
            false_alert_budget=FALSE_ALERT_BUDGET,
            recall_target=RECALL_TARGET,
        )

        for m_value in calibration_grid:
            key = (arm, commissioning_size, m_value, seed)
            if key in completed:
                continue

            result = evaluate_threshold_from_scores(
                calibration_scores=calibration_scores[:m_value],
                healthy_scores=healthy_scores,
                anomaly_scores=anomaly_scores,
                false_alert_budget=FALSE_ALERT_BUDGET,
                recall_target=RECALL_TARGET,
                joint_confidence=JOINT_CONFIDENCE,
            )

            # Oracle depends only on the fitted score and frozen evaluation set,
            # not on calibration size. This guard catches accidental model drift.
            if not np.isclose(
                result.oracle_recall_at_fpr_budget,
                oracle.max_recall_at_fpr_budget,
                atol=1e-12,
            ):
                raise RuntimeError("Oracle recall changed across calibration prefixes.")

            row = {
                "protocol_version": PROTOCOL_VERSION,
                "arm": arm,
                "detector": "TargetOnly-Stat",
                "commissioning_size": commissioning_size,
                "calibration_size": m_value,
                "seed": seed,
                "calibration_candidate_pool_size": len(calibration_master),
                "conformal_rank": result.conformal_rank,
                "conformal_regime": result.conformal_regime,
                "minimum_attainable_alpha": result.minimum_attainable_alpha,
                "threshold": result.threshold,
                "tp": result.tp,
                "fn": result.fn,
                "fp": result.fp,
                "tn": result.tn,
                "recall": result.recall,
                "false_positive_rate": result.false_positive_rate,
                "recall_lower": result.recall_lower,
                "fpr_upper": result.fpr_upper,
                "empirical_success": result.empirical_success,
                "certified_success": result.certified_success,
                "auroc": result.auroc,
                "oracle_recall_at_fpr_budget": result.oracle_recall_at_fpr_budget,
                "oracle_fpr_at_max_recall": result.oracle_fpr_at_max_recall,
                "oracle_empirically_feasible": result.oracle_empirically_feasible,
                "bottleneck_label": result.bottleneck_label,
                "commissioning_ids": "|".join(
                    str(c.episode_id) for c in base_split.target_commissioning
                ),
                "calibration_ids": "|".join(
                    str(c.episode_id) for c in calibration_master[:m_value]
                ),
                "healthy_eval_ids": "|".join(
                    str(c.episode_id) for c in base_split.target_normal_evaluation
                ),
            }
            rows.append(row)
            completed.add(key)
            _atomic_write_csv(
                pd.DataFrame(rows).sort_values(
                    ["arm", "commissioning_size", "calibration_size", "seed"]
                ),
                SEED_RESULTS_PATH,
            )
            print(
                f"E3 arm={arm} N={commissioning_size} Mcal={m_value} "
                f"seed={seed} regime={result.conformal_regime} "
                f"recall={result.recall:.3f} fpr={result.false_positive_rate:.3f}"
            )


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    target_healthy = _target_healthy(cycles)

    checkpoint = _load_checkpoint()
    rows = checkpoint.to_dict(orient="records") if not checkpoint.empty else []
    completed = _completed_keys(checkpoint)

    print("=" * 78)
    print("COLDSTART E3: CALIBRATION-SIZE PHASE TRANSITION")
    print("=" * 78)
    print(f"Target healthy cycles: {len(target_healthy)}")
    print(f"Primary arm: N={PRIMARY_N}, Mcal={PRIMARY_CALIBRATION_GRID}")
    print(
        f"Structural arm: N={STRUCTURAL_N}, Mcal={STRUCTURAL_CALIBRATION_GRID}"
    )
    print(f"Healthy evaluation size: {HEALTHY_EVALUATION_SIZE}")
    print(f"FPR budget / alpha: {FALSE_ALERT_BUDGET}")
    print("The M=199 structural arm is diagnostic; N=10 is representation-limited.")
    print("=" * 78)

    _run_arm(
        cycles=cycles,
        target_healthy=target_healthy,
        arm="primary_calibration_limited",
        commissioning_size=PRIMARY_N,
        calibration_grid=PRIMARY_CALIBRATION_GRID,
        rows=rows,
        completed=completed,
    )
    _run_arm(
        cycles=cycles,
        target_healthy=target_healthy,
        arm="structural_rank_transition",
        commissioning_size=STRUCTURAL_N,
        calibration_grid=STRUCTURAL_CALIBRATION_GRID,
        rows=rows,
        completed=completed,
    )

    results = pd.DataFrame(rows)
    summary = _build_summary(results)
    _atomic_write_csv(summary, SUMMARY_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": "TargetOnly-Stat",
        "global_seed": GLOBAL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "calibration_order_seed": CALIBRATION_ORDER_SEED,
        "commissioning_seeds": SEEDS,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "healthy_evaluation_size": HEALTHY_EVALUATION_SIZE,
        "target_healthy_cycle_count": len(target_healthy),
        "primary_arm": {
            "purpose": "empirically test calibration size where E1 found TargetOnly calibration-limited",
            "commissioning_size": PRIMARY_N,
            "calibration_grid": PRIMARY_CALIBRATION_GRID,
            "maximum_independent_calibration_size": 119,
        },
        "structural_arm": {
            "purpose": "observe exact deterministic conformal maximum-to-submaximum rank transition",
            "commissioning_size": STRUCTURAL_N,
            "calibration_grid": STRUCTURAL_CALIBRATION_GRID,
            "maximum_independent_calibration_size": 209,
            "performance_interpretation_warning": (
                "N=10 is representation-limited in E1; use this arm to validate "
                "order-statistic structure, not to claim calibration solves performance."
            ),
        },
        "conformal_boundaries_at_alpha_0_01": {
            "first_finite_threshold_m": 99,
            "maximum_threshold_examples": [99, 100, 119, 198],
            "first_submaximum_threshold_m": 199,
        },
        "data_limit_note": (
            "With 319 target-healthy cycles, N=100 commissioning and 100 healthy "
            "evaluation cycles leave at most 119 independent calibration cycles. "
            "M=199 cannot be tested at N=100 without leakage."
        ),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nE3 complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
