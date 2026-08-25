from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bottleneck_decomposition import (
    CALIBRATION_LIMITED,
    CERTIFICATION_LIMITED,
    CERTIFIED,
    REPRESENTATION_LIMITED,
    classify_bottleneck,
)
from src.detectors import RACEDetector, TargetOnlyDetector
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import (
    empirical_oracle_feasibility,
    probability_of_superiority,
)
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
)
E0_RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "e0_certification" / "e0_seed_results.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e1_oracle_decomposition"
SEED_RESULTS_PATH = OUTPUT_DIR / "e1_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e1_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e1_manifest.json"

COMMISSIONING_GRID = [10, 25, 50, 100]
SEEDS = list(range(20))
EVALUATION_SEED = 42
GLOBAL_SEED = 42

FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100

DETECTOR_NAMES = ("TargetOnly", "RACE")
PROTOCOL_VERSION = "coldstart-e1-oracle-decomposition-v1"
E0_PROTOCOL_VERSION = "coldstart-e0-certification-v1"

np.random.seed(GLOBAL_SEED)


CHECKPOINT_COLUMNS = [
    "protocol_version",
    "e0_protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "healthy_eval_count",
    "anomaly_eval_count",
    "allowed_false_positives",
    "fpr_resolution",
    "recall_resolution",
    "deployed_threshold",
    "deployed_recall",
    "deployed_fpr",
    "deployed_tp",
    "deployed_fn",
    "deployed_fp",
    "deployed_tn",
    "recall_lower",
    "fpr_upper",
    "deployed_empirical_success",
    "deployed_certified_success",
    "oracle_empirically_feasible",
    "oracle_recall_at_fpr_budget",
    "oracle_fpr_at_max_recall",
    "oracle_threshold_at_fpr_budget",
    "oracle_min_fpr_at_recall_target",
    "oracle_recall_at_min_fpr",
    "oracle_threshold_at_recall_target",
    "oracle_recall_slack",
    "oracle_fpr_slack",
    "oracle_minus_deployed_recall",
    "deployed_recall_deficit",
    "deployed_fpr_excess",
    "bottleneck_label",
    "empirical_auroc",
    "calibration_alpha",
    "conformal_rank",
    "conformal_regime",
    "calibration_size",
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
    raise ValueError(f"Unsupported E1 detector: {detector_name}")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_e0_results() -> pd.DataFrame:
    if not E0_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "E0 seed results are required before E1. Expected: "
            f"{E0_RESULTS_PATH}"
        )

    frame = pd.read_csv(E0_RESULTS_PATH)
    required = {
        "protocol_version",
        "detector",
        "commissioning_size",
        "seed",
        "recall",
        "false_positive_rate",
        "tp",
        "fn",
        "fp",
        "tn",
        "recall_lower",
        "fpr_upper",
        "empirical_success",
        "certified_success",
        "threshold",
        "calibration_alpha",
        "conformal_rank",
        "conformal_regime",
        "calibration_size",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "E0 output is incompatible with E1; missing columns: "
            f"{sorted(missing)}"
        )

    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions != {E0_PROTOCOL_VERSION}:
        raise ValueError(
            "E1 requires the frozen E0 protocol. Found protocol versions: "
            f"{sorted(versions)}"
        )

    expected_rows = len(DETECTOR_NAMES) * len(COMMISSIONING_GRID) * len(SEEDS)
    subset = frame[
        frame["detector"].isin(DETECTOR_NAMES)
        & frame["commissioning_size"].isin(COMMISSIONING_GRID)
        & frame["seed"].isin(SEEDS)
    ].copy()
    if len(subset) != expected_rows:
        raise ValueError(
            "E0 does not contain the complete E1 matrix: "
            f"expected {expected_rows} rows, found {len(subset)}."
        )

    duplicated = subset.duplicated(
        ["detector", "commissioning_size", "seed"], keep=False
    )
    if duplicated.any():
        raise ValueError("E0 contains duplicate detector/N/seed rows.")

    return subset


def load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    frame = pd.read_csv(SEED_RESULTS_PATH)
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(
            "Existing E1 checkpoint uses a different protocol: "
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


def _e0_row_for(
    e0: pd.DataFrame,
    detector: str,
    n_value: int,
    seed: int,
) -> pd.Series:
    rows = e0[
        (e0["detector"] == detector)
        & (e0["commissioning_size"] == n_value)
        & (e0["seed"] == seed)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one E0 row for {detector}, N={n_value}, seed={seed}; "
            f"found {len(rows)}."
        )
    return rows.iloc[0]


def _assert_reconstruction_matches_e0(
    *,
    e0_row: pd.Series,
    threshold: float,
    recall: float,
    fpr: float,
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    conformal_rank: int,
    conformal_regime: str,
    calibration_size: int,
    tolerance: float = 1e-10,
) -> None:
    checks = {
        "threshold": (float(threshold), float(e0_row["threshold"])),
        "recall": (float(recall), float(e0_row["recall"])),
        "fpr": (float(fpr), float(e0_row["false_positive_rate"])),
    }
    for name, (reconstructed, frozen) in checks.items():
        if not np.isclose(reconstructed, frozen, atol=tolerance, rtol=0.0):
            raise RuntimeError(
                f"E1 reconstruction mismatch for {name}: "
                f"reconstructed={reconstructed}, E0={frozen}."
            )

    integer_checks = {
        "tp": (tp, int(e0_row["tp"])),
        "fn": (fn, int(e0_row["fn"])),
        "fp": (fp, int(e0_row["fp"])),
        "tn": (tn, int(e0_row["tn"])),
        "conformal_rank": (conformal_rank, int(e0_row["conformal_rank"])),
        "calibration_size": (calibration_size, int(e0_row["calibration_size"])),
    }
    for name, (reconstructed, frozen) in integer_checks.items():
        if int(reconstructed) != int(frozen):
            raise RuntimeError(
                f"E1 reconstruction mismatch for {name}: "
                f"reconstructed={reconstructed}, E0={frozen}."
            )

    if str(conformal_regime) != str(e0_row["conformal_regime"]):
        raise RuntimeError(
            "E1 reconstruction mismatch for conformal regime: "
            f"{conformal_regime!r} vs {e0_row['conformal_regime']!r}."
        )


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [
        REPRESENTATION_LIMITED,
        CALIBRATION_LIMITED,
        CERTIFICATION_LIMITED,
        CERTIFIED,
    ]

    for (detector, n_value), group in results.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        counts = group["bottleneck_label"].value_counts()
        row: dict[str, Any] = {
            "detector": str(detector),
            "commissioning_size": int(n_value),
            "number_of_seeds": int(len(group)),
            "mean_empirical_auroc": float(group["empirical_auroc"].mean()),
            "mean_deployed_recall": float(group["deployed_recall"].mean()),
            "mean_deployed_fpr": float(group["deployed_fpr"].mean()),
            "mean_oracle_recall_at_fpr_budget": float(
                group["oracle_recall_at_fpr_budget"].mean()
            ),
            "oracle_feasibility_rate": float(
                group["oracle_empirically_feasible"].astype(bool).mean()
            ),
            "mean_oracle_minus_deployed_recall": float(
                group["oracle_minus_deployed_recall"].mean()
            ),
            "empirical_success_rate": float(
                group["deployed_empirical_success"].astype(bool).mean()
            ),
            "certified_success_rate": float(
                group["deployed_certified_success"].astype(bool).mean()
            ),
        }
        for label in labels:
            row[f"{label}_count"] = int(counts.get(label, 0))
            row[f"{label}_rate"] = float(counts.get(label, 0) / len(group))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["detector", "commissioning_size"]
    ).reset_index(drop=True)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    e0 = load_e0_results()
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")

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
    print("COLDSTART E1: ORACLE-VS-DEPLOYED BOTTLENECK DECOMPOSITION")
    print("=" * 78)
    print(f"Protocol:             {PROTOCOL_VERSION}")
    print(f"E0 protocol:          {E0_PROTOCOL_VERSION}")
    print(f"Detectors:            {list(DETECTOR_NAMES)}")
    print(f"Commissioning grid:   {COMMISSIONING_GRID}")
    print(f"Seeds:                {SEEDS}")
    print(f"Evaluation seed:      {EVALUATION_SEED}")
    print(f"Recall target:        {RECALL_TARGET}")
    print(f"FPR budget:           {FALSE_ALERT_BUDGET}")
    print(
        "Oracle rule: same scalar score and same `score > threshold` family; "
        "evaluation labels used diagnostically only."
    )
    print(f"Completed rows:       {len(finished & expected)}/{len(expected)}")
    print("=" * 78)

    fixed_calibration_ids: tuple[int, ...] | None = None
    fixed_healthy_eval_ids: tuple[int, ...] | None = None
    fixed_anomaly_eval_ids: tuple[int, ...] | None = None
    counter = len(finished & expected)

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

            calibration_ids = tuple(c.episode_id for c in split.target_calibration)
            healthy_ids = tuple(
                c.episode_id for c in split.target_normal_evaluation
            )
            anomaly_ids = tuple(c.episode_id for c in split.target_anomaly_evaluation)
            if fixed_calibration_ids is None:
                fixed_calibration_ids = calibration_ids
                fixed_healthy_eval_ids = healthy_ids
                fixed_anomaly_eval_ids = anomaly_ids
            else:
                if calibration_ids != fixed_calibration_ids:
                    raise RuntimeError("Calibration IDs changed across E1 runs.")
                if healthy_ids != fixed_healthy_eval_ids:
                    raise RuntimeError("Healthy evaluation IDs changed across E1 runs.")
                if anomaly_ids != fixed_anomaly_eval_ids:
                    raise RuntimeError("Anomaly evaluation IDs changed across E1 runs.")

            source_raw, _ = extract_feature_matrix(split.source_train)
            target_raw, _ = extract_feature_matrix(split.target_commissioning)
            calibration_raw, _ = extract_feature_matrix(split.target_calibration)
            healthy_raw, _ = extract_feature_matrix(split.target_normal_evaluation)
            anomaly_raw, _ = extract_feature_matrix(split.target_anomaly_evaluation)

            for detector_name in DETECTOR_NAMES:
                key = (detector_name, n_value, seed)
                if key in finished:
                    continue

                counter += 1
                print(
                    f"E1 {counter}/{len(expected)} | detector={detector_name} "
                    f"N={n_value} seed={seed}"
                )

                e0_row = _e0_row_for(e0, detector_name, n_value, seed)
                detector, preprocessor, _, _ = fit_detector(
                    detector_name=detector_name,
                    detector_factory=detector_factory(detector_name),
                    source_raw=source_raw,
                    target_raw=target_raw,
                )

                calibration_features = preprocessor.transform(calibration_raw)
                healthy_features = preprocessor.transform(healthy_raw)
                anomaly_features = preprocessor.transform(anomaly_raw)
                detector.calibrate(calibration_features)

                healthy_scores = detector.score_samples(healthy_features)
                anomaly_scores = detector.score_samples(anomaly_features)
                threshold = float(detector.threshold_)

                healthy_predictions = (healthy_scores > threshold).astype(np.int64)
                anomaly_predictions = (anomaly_scores > threshold).astype(np.int64)
                fp = int(healthy_predictions.sum())
                tn = int(len(healthy_predictions) - fp)
                tp = int(anomaly_predictions.sum())
                fn = int(len(anomaly_predictions) - tp)
                fpr = float(fp / (fp + tn))
                recall = float(tp / (tp + fn))

                if detector.calibration_rank_ is None:
                    raise RuntimeError("Missing conformal rank after calibration.")
                if detector.calibration_regime_ is None:
                    raise RuntimeError("Missing conformal regime after calibration.")
                if detector.calibration_size_ is None:
                    raise RuntimeError("Missing calibration size after calibration.")

                _assert_reconstruction_matches_e0(
                    e0_row=e0_row,
                    threshold=threshold,
                    recall=recall,
                    fpr=fpr,
                    tp=tp,
                    fn=fn,
                    fp=fp,
                    tn=tn,
                    conformal_rank=int(detector.calibration_rank_),
                    conformal_regime=str(detector.calibration_regime_),
                    calibration_size=int(detector.calibration_size_),
                )

                oracle = empirical_oracle_feasibility(
                    healthy_scores=healthy_scores,
                    anomaly_scores=anomaly_scores,
                    false_alert_budget=FALSE_ALERT_BUDGET,
                    recall_target=RECALL_TARGET,
                )
                decomposition = classify_bottleneck(
                    oracle=oracle,
                    deployed_recall=recall,
                    deployed_fpr=fpr,
                    recall_lower=float(e0_row["recall_lower"]),
                    fpr_upper=float(e0_row["fpr_upper"]),
                    recall_target=RECALL_TARGET,
                    fpr_budget=FALSE_ALERT_BUDGET,
                )

                target_weight: float | None = None
                if isinstance(detector, RACEDetector):
                    target_weight = detector.target_weight_

                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "e0_protocol_version": E0_PROTOCOL_VERSION,
                    "detector": detector_name,
                    "commissioning_size": n_value,
                    "seed": seed,
                    "healthy_eval_count": oracle.healthy_count,
                    "anomaly_eval_count": oracle.anomaly_count,
                    "allowed_false_positives": oracle.allowed_false_positives,
                    "fpr_resolution": oracle.fpr_resolution,
                    "recall_resolution": oracle.recall_resolution,
                    "deployed_threshold": threshold,
                    "deployed_recall": recall,
                    "deployed_fpr": fpr,
                    "deployed_tp": tp,
                    "deployed_fn": fn,
                    "deployed_fp": fp,
                    "deployed_tn": tn,
                    "recall_lower": float(e0_row["recall_lower"]),
                    "fpr_upper": float(e0_row["fpr_upper"]),
                    "deployed_empirical_success": decomposition.deployed_empirical_success,
                    "deployed_certified_success": decomposition.deployed_certified_success,
                    "oracle_empirically_feasible": oracle.empirically_feasible,
                    "oracle_recall_at_fpr_budget": oracle.max_recall_at_fpr_budget,
                    "oracle_fpr_at_max_recall": oracle.fpr_at_max_recall,
                    "oracle_threshold_at_fpr_budget": oracle.threshold_at_fpr_budget,
                    "oracle_min_fpr_at_recall_target": oracle.min_fpr_at_recall_target,
                    "oracle_recall_at_min_fpr": oracle.recall_at_min_fpr,
                    "oracle_threshold_at_recall_target": oracle.threshold_at_recall_target,
                    "oracle_recall_slack": oracle.recall_slack,
                    "oracle_fpr_slack": oracle.fpr_slack,
                    "oracle_minus_deployed_recall": decomposition.oracle_minus_deployed_recall,
                    "deployed_recall_deficit": decomposition.deployed_recall_deficit,
                    "deployed_fpr_excess": decomposition.deployed_fpr_excess,
                    "bottleneck_label": decomposition.bottleneck_label,
                    "empirical_auroc": probability_of_superiority(
                        healthy_scores, anomaly_scores
                    ),
                    "calibration_alpha": float(e0_row["calibration_alpha"]),
                    "conformal_rank": int(detector.calibration_rank_),
                    "conformal_regime": str(detector.calibration_regime_),
                    "calibration_size": int(detector.calibration_size_),
                    "target_weight": target_weight,
                }
                rows.append(row)
                finished.add(key)
                save_rows(rows)

    results = pd.DataFrame(rows)
    summary = build_summary(results)
    _atomic_write_csv(summary, SUMMARY_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "e0_protocol_version": E0_PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "commissioning_grid": COMMISSIONING_GRID,
        "commissioning_seeds": SEEDS,
        "detectors": list(DETECTOR_NAMES),
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": NORMAL_EVALUATION_SIZE,
        "oracle_is_diagnostic_only": True,
        "oracle_uses_evaluation_labels": True,
        "oracle_decision_family": "same_scalar_score_strict_greater_than_threshold",
        "classification_precedence": [
            REPRESENTATION_LIMITED,
            CALIBRATION_LIMITED,
            CERTIFICATION_LIMITED,
            CERTIFIED,
        ],
        "classification_rules": {
            REPRESENTATION_LIMITED: "oracle empirical recall at FPR budget < recall target",
            CALIBRATION_LIMITED: "oracle feasible but deployed empirical operating point fails",
            CERTIFICATION_LIMITED: "deployed empirical operating point passes but exact joint certification fails",
            CERTIFIED: "exact joint recall and FPR certification passes",
        },
        "frozen_calibration_ids": list(fixed_calibration_ids or ()),
        "frozen_healthy_eval_ids": list(fixed_healthy_eval_ids or ()),
        "frozen_anomaly_eval_ids": list(fixed_anomaly_eval_ids or ()),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nE1 complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
