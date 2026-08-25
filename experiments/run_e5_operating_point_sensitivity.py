from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detectors import RACEDetector, TargetOnlyDetector
from src.e5_operating_point import (
    OperatingPoint,
    evaluate_operating_point_from_scores,
    minimum_best_case_anomaly_evidence,
    minimum_best_case_healthy_evidence,
)
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
E0_RESULTS_PATH = PROJECT_ROOT / "outputs" / "e0_certification" / "e0_seed_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e5_operating_point_sensitivity"
SEED_RESULTS_PATH = OUTPUT_DIR / "e5_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e5_summary.csv"
NSTAR_PATH = OUTPUT_DIR / "e5_nstar_sensitivity.csv"
MANIFEST_PATH = OUTPUT_DIR / "e5_manifest.json"

PROTOCOL_VERSION = "coldstart-e5-operating-point-sensitivity-v1"
E0_PROTOCOL_VERSION = "coldstart-e0-certification-v1"
GLOBAL_SEED = 42
EVALUATION_SEED = 42
SEEDS = list(range(20))
COMMISSIONING_GRID = [10, 25, 50, 100]
CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
JOINT_CONFIDENCE = 0.95
Q0 = 0.80

# Predeclared sensitivity grid. The first entry is the frozen primary endpoint.
OPERATING_POINTS = (
    OperatingPoint("primary_R90_B01", recall_target=0.90, false_alert_budget=0.01),
    OperatingPoint("lower_recall_R80_B01", recall_target=0.80, false_alert_budget=0.01),
    OperatingPoint("strict_fpr_R90_B005", recall_target=0.90, false_alert_budget=0.005),
    OperatingPoint("loose_fpr_R90_B02", recall_target=0.90, false_alert_budget=0.02),
    OperatingPoint("strict_recall_R95_B01", recall_target=0.95, false_alert_budget=0.01),
)
DETECTORS = ("TargetOnly", "RACE")
PRIMARY_ENDPOINT = "primary_R90_B01"

np.random.seed(GLOBAL_SEED)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _detector_factory(detector_name: str):
    # false_alert_budget affects calibration only; E5 calibrates from the fixed
    # score vectors independently for every operating point.
    if detector_name == "TargetOnly":
        return lambda: TargetOnlyDetector(false_alert_budget=0.01)
    if detector_name == "RACE":
        return lambda: RACEDetector(lambda_reg=60.0, false_alert_budget=0.01)
    raise ValueError(f"Unsupported detector: {detector_name}")


def _load_e0() -> pd.DataFrame:
    if not E0_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "E5 requires frozen E0 outputs to verify the primary endpoint: "
            f"{E0_RESULTS_PATH}"
        )
    frame = pd.read_csv(E0_RESULTS_PATH)
    required = {
        "protocol_version", "detector", "commissioning_size", "seed",
        "threshold", "recall", "false_positive_rate", "tp", "fn", "fp", "tn",
        "conformal_rank", "conformal_regime", "calibration_size",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"E0 output missing required columns: {sorted(missing)}")
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions != {E0_PROTOCOL_VERSION}:
        raise ValueError(f"E5 requires frozen E0 protocol; found {sorted(versions)}")
    return frame


def _e0_row(e0: pd.DataFrame, detector: str, n_value: int, seed: int) -> pd.Series:
    subset = e0[
        (e0["detector"] == detector)
        & (e0["commissioning_size"] == n_value)
        & (e0["seed"] == seed)
    ]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected exactly one E0 row for {detector}, N={n_value}, seed={seed}; "
            f"found {len(subset)}."
        )
    return subset.iloc[0]


def _assert_primary_matches_e0(result, row: pd.Series) -> None:
    numeric = {
        "threshold": (result.threshold, float(row["threshold"])),
        "recall": (result.recall, float(row["recall"])),
        "fpr": (result.fpr, float(row["false_positive_rate"])),
    }
    for name, (actual, expected) in numeric.items():
        if np.isinf(actual) or np.isinf(expected):
            if not (np.isinf(actual) and np.isinf(expected) and np.sign(actual) == np.sign(expected)):
                raise RuntimeError(f"E5 primary {name} does not reproduce E0.")
        elif not np.isclose(actual, expected, atol=1e-10, rtol=0.0):
            raise RuntimeError(
                f"E5 primary {name} does not reproduce E0: {actual} vs {expected}."
            )

    integer_checks = {
        "tp": (result.tp, int(row["tp"])),
        "fn": (result.fn, int(row["fn"])),
        "fp": (result.fp, int(row["fp"])),
        "tn": (result.tn, int(row["tn"])),
        "conformal_rank": (result.conformal_rank, int(row["conformal_rank"])),
        "calibration_size": (CALIBRATION_SIZE, int(row["calibration_size"])),
    }
    for name, (actual, expected) in integer_checks.items():
        if actual != expected:
            raise RuntimeError(
                f"E5 primary {name} does not reproduce E0: {actual} vs {expected}."
            )
    if result.conformal_regime != str(row["conformal_regime"]):
        raise RuntimeError(
            "E5 primary conformal regime does not reproduce E0: "
            f"{result.conformal_regime} vs {row['conformal_regime']}."
        )


def _load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SEED_RESULTS_PATH)
    if "protocol_version" not in frame.columns:
        raise ValueError("Existing E5 checkpoint lacks protocol_version.")
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(f"Incompatible E5 checkpoint versions: {sorted(versions)}")
    return frame


def _completed_keys(frame: pd.DataFrame) -> set[tuple[str, int, int, str]]:
    if frame.empty:
        return set()
    return {
        (str(r.detector), int(r.commissioning_size), int(r.seed), str(r.endpoint))
        for r in frame.itertuples(index=False)
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = results.groupby(
        ["detector", "endpoint", "recall_target", "false_alert_budget", "commissioning_size"],
        sort=True,
    )
    for (detector, endpoint, r0, budget, n_value), group in grouped:
        rows.append(
            {
                "detector": detector,
                "endpoint": endpoint,
                "recall_target": float(r0),
                "false_alert_budget": float(budget),
                "commissioning_size": int(n_value),
                "number_of_seeds": int(len(group)),
                "conformal_rank": int(group["conformal_rank"].iloc[0]),
                "conformal_regime": str(group["conformal_regime"].iloc[0]),
                "mean_recall": float(group["recall"].mean()),
                "mean_fpr": float(group["fpr"].mean()),
                "mean_auroc": float(group["auroc"].mean()),
                "mean_oracle_recall_at_fpr_budget": float(
                    group["oracle_recall_at_fpr_budget"].mean()
                ),
                "oracle_feasibility_rate": float(
                    group["oracle_empirically_feasible"].astype(bool).mean()
                ),
                "empirical_success_rate": float(group["empirical_success"].astype(bool).mean()),
                "certified_success_rate": float(group["certified_success"].astype(bool).mean()),
                "representation_limited_rate": float(
                    (group["bottleneck_label"] == "representation_limited").mean()
                ),
                "calibration_limited_rate": float(
                    (group["bottleneck_label"] == "calibration_limited").mean()
                ),
                "certification_limited_rate": float(
                    (group["bottleneck_label"] == "certification_limited").mean()
                ),
                "certified_rate": float((group["bottleneck_label"] == "certified").mean()),
                "mean_seedwise_recall_lower": float(group["recall_lower"].mean()),
                "mean_seedwise_fpr_upper": float(group["fpr_upper"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["endpoint", "detector", "commissioning_size"]
    ).reset_index(drop=True)


def _first_n_at_rate(group: pd.DataFrame, column: str, target: float) -> str:
    ordered = group.sort_values("commissioning_size")
    passing = ordered[ordered[column] >= target - 1e-12]
    if passing.empty:
        return f"Censored (>{max(COMMISSIONING_GRID)})"
    return str(int(passing.iloc[0]["commissioning_size"]))


def _build_nstar(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, endpoint, r0, budget), group in summary.groupby(
        ["detector", "endpoint", "recall_target", "false_alert_budget"], sort=True
    ):
        rows.append(
            {
                "detector": detector,
                "endpoint": endpoint,
                "recall_target": float(r0),
                "false_alert_budget": float(budget),
                "q0": Q0,
                "empirical_Nstar_q80": _first_n_at_rate(group, "empirical_success_rate", Q0),
                "certified_Nstar_q80": _first_n_at_rate(group, "certified_success_rate", Q0),
                "oracle_feasible_Nstar_q80": _first_n_at_rate(group, "oracle_feasibility_rate", Q0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    e0 = _load_e0()
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    checkpoint = _load_checkpoint()
    rows = checkpoint.to_dict(orient="records") if not checkpoint.empty else []
    completed = _completed_keys(checkpoint)

    total_fits = len(DETECTORS) * len(COMMISSIONING_GRID) * len(SEEDS)
    fit_index = 0
    print("=" * 82)
    print("COLDSTART E5: OPERATING-POINT SENSITIVITY")
    print("=" * 82)
    print(f"Endpoints: {[p.name for p in OPERATING_POINTS]}")
    print(f"Detector fits: {total_fits}; each fit is reused across all endpoints.")
    print("Primary endpoint must reproduce E0 exactly before sensitivity rows are trusted.")
    print("=" * 82)

    for n_value in COMMISSIONING_GRID:
        for seed in SEEDS:
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n_value,
                commissioning_seed=seed,
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )
            source_raw, _ = extract_feature_matrix(split.source_train)
            target_raw, _ = extract_feature_matrix(split.target_commissioning)
            calibration_raw, _ = extract_feature_matrix(split.target_calibration)
            healthy_raw, _ = extract_feature_matrix(split.target_normal_evaluation)
            anomaly_raw, _ = extract_feature_matrix(split.target_anomaly_evaluation)

            for detector_name in DETECTORS:
                fit_index += 1
                detector, preprocessor, _, _ = fit_detector(
                    detector_name=detector_name,
                    detector_factory=_detector_factory(detector_name),
                    source_raw=source_raw,
                    target_raw=target_raw,
                )
                calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
                healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
                anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

                for point in OPERATING_POINTS:
                    key = (detector_name, n_value, seed, point.name)
                    if key in completed:
                        continue

                    result = evaluate_operating_point_from_scores(
                        calibration_scores=calibration_scores,
                        healthy_scores=healthy_scores,
                        anomaly_scores=anomaly_scores,
                        operating_point=point,
                        joint_confidence=JOINT_CONFIDENCE,
                    )

                    if point.name == PRIMARY_ENDPOINT:
                        _assert_primary_matches_e0(
                            result,
                            _e0_row(e0, detector_name, n_value, seed),
                        )

                    rows.append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "detector": detector_name,
                            "commissioning_size": n_value,
                            "seed": seed,
                            "endpoint": point.name,
                            "recall_target": point.recall_target,
                            "false_alert_budget": point.false_alert_budget,
                            "calibration_alpha": point.false_alert_budget,
                            "calibration_size": CALIBRATION_SIZE,
                            "healthy_eval_count": len(healthy_scores),
                            "anomaly_eval_count": len(anomaly_scores),
                            "conformal_rank": result.conformal_rank,
                            "conformal_regime": result.conformal_regime,
                            "minimum_attainable_alpha": result.minimum_attainable_alpha,
                            "threshold": result.threshold,
                            "tp": result.tp,
                            "fn": result.fn,
                            "fp": result.fp,
                            "tn": result.tn,
                            "recall": result.recall,
                            "fpr": result.fpr,
                            "recall_lower": result.recall_lower,
                            "fpr_upper": result.fpr_upper,
                            "empirical_success": result.empirical_success,
                            "certified_success": result.certified_success,
                            "auroc": result.auroc,
                            "oracle_recall_at_fpr_budget": result.oracle_recall_at_fpr_budget,
                            "oracle_fpr_at_max_recall": result.oracle_fpr_at_max_recall,
                            "oracle_empirically_feasible": result.oracle_empirically_feasible,
                            "bottleneck_label": result.bottleneck_label,
                        }
                    )
                    completed.add(key)
                    _atomic_write_csv(
                        pd.DataFrame(rows).sort_values(
                            ["endpoint", "commissioning_size", "seed", "detector"]
                        ),
                        SEED_RESULTS_PATH,
                    )

                print(
                    f"E5 fit {fit_index}/{total_fits}: {detector_name} "
                    f"N={n_value} seed={seed}; all endpoints complete/checkpointed."
                )

    results = pd.DataFrame(rows)
    summary = _build_summary(results)
    nstar = _build_nstar(summary)
    _atomic_write_csv(summary, SUMMARY_PATH)
    _atomic_write_csv(nstar, NSTAR_PATH)

    endpoint_manifest = []
    for point in OPERATING_POINTS:
        endpoint_manifest.append(
            {
                "name": point.name,
                "recall_target": point.recall_target,
                "false_alert_budget": point.false_alert_budget,
                "best_case_min_healthy_certification_n_fp0": minimum_best_case_healthy_evidence(
                    false_alert_budget=point.false_alert_budget,
                    joint_confidence=JOINT_CONFIDENCE,
                ),
                "best_case_min_anomaly_certification_n_fn0": minimum_best_case_anomaly_evidence(
                    recall_target=point.recall_target,
                    joint_confidence=JOINT_CONFIDENCE,
                ),
            }
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "test whether COLDSTART bottleneck conclusions are robust to predeclared recall/FPR endpoints",
        "global_seed": GLOBAL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "commissioning_grid": COMMISSIONING_GRID,
        "commissioning_seeds": SEEDS,
        "detectors": list(DETECTORS),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": HEALTHY_EVALUATION_SIZE,
        "joint_confidence": JOINT_CONFIDENCE,
        "q0": Q0,
        "alpha_mapping": "deterministic split-conformal alpha equals endpoint false-alert budget",
        "score_reuse": "fit once per detector/N/seed; reuse identical calibration/evaluation scores across all endpoints",
        "primary_endpoint_reproduction_guard": "primary_R90_B01 must exactly reproduce frozen E0 threshold/counts/metrics/regime",
        "operating_points": endpoint_manifest,
        "fixed_Mcal_100_tail_regimes": {
            "B_0.005": "infinite (raw rank 101)",
            "B_0.01": "maximum (raw rank 100)",
            "B_0.02": "submaximum (raw rank 99)",
        },
        "certification_interpretation": (
            "With n_H=100, even zero false positives cannot certify any tested FPR budget; "
            "best-case healthy requirements are 736 for B=.005, 368 for B=.01, and 183 for B=.02."
        ),
        "oracle_is_diagnostic_only": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\nE5 complete.")
    print(summary.to_string(index=False))
    print("\nN* sensitivity:")
    print(nstar.to_string(index=False))


if __name__ == "__main__":
    main()
