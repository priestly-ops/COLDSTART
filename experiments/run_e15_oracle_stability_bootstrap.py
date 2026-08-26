from __future__ import annotations

"""E15: evaluation-sample stability of the empirical low-FPR oracle.

This experiment does not retrain or tune detectors using anomaly labels. It fits
frozen TargetOnly and RACE(lambda=60) models under the established voraus-AD
commissioning protocol, obtains the fixed healthy/anomaly evaluation scores,
and then bootstraps those *evaluation executions* to quantify sensitivity of the
retrospective empirical oracle recall at FPR <= 1%.

The bootstrap intervals are diagnostic sample-stability intervals only. They are
not post-freeze certification bounds and are not used to redefine N*, tune RACE,
or make population-coverage claims.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.detectors import RACEDetector, TargetOnlyDetector
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility
from src.oracle_stability import bootstrap_oracle_recall
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e15_oracle_stability"
SEED_RESULTS_PATH = OUTPUT_DIR / "e15_seed_oracle_stability.csv"
SUMMARY_PATH = OUTPUT_DIR / "e15_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e15_manifest.json"

PROTOCOL_VERSION = "coldstart-e15-oracle-stability-v1"
N_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
DETECTORS = ("TargetOnly", "RACE")
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
B = 0.01
R0 = 0.90
RACE_LAMBDA = 60.0
GLOBAL_SEED = 42
DEFAULT_BOOTSTRAP_REPETITIONS = 5_000

np.random.seed(GLOBAL_SEED)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _parse_detectors(text: str) -> tuple[str, ...]:
    values = tuple(v.strip() for v in text.split(",") if v.strip())
    unknown = sorted(set(values) - set(DETECTORS))
    if unknown:
        raise ValueError(f"unsupported detector(s): {unknown}; expected subset of {DETECTORS}")
    return values


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _rows_for_ids(matrix: np.ndarray, row_by_id: dict[int, int], cycles) -> np.ndarray:
    ids = [int(c.episode_id) for c in cycles]
    return matrix[np.asarray([row_by_id[i] for i in ids], dtype=np.int64)]


def _factory(detector_name: str):
    if detector_name == "TargetOnly":
        return lambda: TargetOnlyDetector(false_alert_budget=B)
    if detector_name == "RACE":
        return lambda: RACEDetector(lambda_reg=RACE_LAMBDA, false_alert_budget=B)
    raise ValueError(detector_name)


def _bootstrap_seed(detector_name: str, n_value: int, seed: int) -> int:
    detector_offset = 0 if detector_name == "TargetOnly" else 1_000_000
    return int(GLOBAL_SEED + detector_offset + 10_000 * int(n_value) + int(seed))


def _evaluate_cell(
    *,
    detector_name: str,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    healthy_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    n_value: int,
    seed: int,
    bootstrap_repetitions: int,
) -> dict[str, Any]:
    detector, preprocessor, _, _ = fit_detector(
        detector_name=detector_name,
        detector_factory=_factory(detector_name),
        source_raw=source_raw,
        target_raw=target_raw,
    )
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

    exact = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
    )
    stability = bootstrap_oracle_recall(
        healthy_scores,
        anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
        repetitions=int(bootstrap_repetitions),
        confidence=0.95,
        seed=_bootstrap_seed(detector_name, n_value, seed),
    )

    if not np.isclose(
        stability.observed_oracle_recall,
        exact.max_recall_at_fpr_budget,
        rtol=0.0,
        atol=1e-15,
    ):
        raise RuntimeError(
            f"fast oracle disagrees with frozen oracle for {detector_name} "
            f"N={n_value} seed={seed}: fast={stability.observed_oracle_recall} "
            f"exact={exact.max_recall_at_fpr_budget}"
        )

    lower = float(stability.bootstrap_interval_lower)
    upper = float(stability.bootstrap_interval_upper)
    relation = (
        "entirely_below_target"
        if upper < R0 - 1e-15
        else "entirely_above_target"
        if lower >= R0 - 1e-15
        else "straddles_target"
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "detector": detector_name,
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "bootstrap_seed": _bootstrap_seed(detector_name, n_value, seed),
        **stability.to_dict(),
        "observed_oracle_empirically_feasible": bool(exact.empirically_feasible),
        "bootstrap_interval_relation_to_target": relation,
        "bootstrap_interval_width": float(upper - lower),
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (detector, n_value), g in df.groupby(["detector", "commissioning_size"], sort=True):
        rows.append({
            "detector": str(detector),
            "commissioning_size": int(n_value),
            "n_commissioning_draws": int(len(g)),
            "mean_observed_oracle_recall": float(g.observed_oracle_recall.mean()),
            "median_observed_oracle_recall": float(g.observed_oracle_recall.median()),
            "min_observed_oracle_recall": float(g.observed_oracle_recall.min()),
            "max_observed_oracle_recall": float(g.observed_oracle_recall.max()),
            "observed_oracle_feasibility_rate": float(g.observed_oracle_empirically_feasible.astype(bool).mean()),
            "mean_bootstrap_mean_oracle_recall": float(g.bootstrap_mean.mean()),
            "mean_bootstrap_std": float(g.bootstrap_std.mean()),
            "mean_bootstrap_interval_width": float(g.bootstrap_interval_width.mean()),
            "median_bootstrap_interval_width": float(g.bootstrap_interval_width.median()),
            "mean_bootstrap_feasibility_rate": float(g.bootstrap_feasibility_rate.mean()),
            "fraction_intervals_entirely_below_target": float((g.bootstrap_interval_relation_to_target == "entirely_below_target").mean()),
            "fraction_intervals_straddling_target": float((g.bootstrap_interval_relation_to_target == "straddles_target").mean()),
            "fraction_intervals_entirely_above_target": float((g.bootstrap_interval_relation_to_target == "entirely_above_target").mean()),
        })
    return pd.DataFrame(rows).sort_values(["detector", "commissioning_size"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E15 oracle stability bootstrap")
    parser.add_argument("--n-values", default=",".join(map(str, N_GRID)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--detectors", default=",".join(DETECTORS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    n_values = _parse_ints(args.n_values)
    seeds = _parse_ints(args.seeds)
    detectors = _parse_detectors(args.detectors)
    if not set(n_values).issubset(N_GRID):
        raise ValueError(f"n-values must be a subset of {N_GRID}")
    if not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be a subset of 0..19")
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap-repetitions must be >= 1")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    all_raw, all_ids = extract_feature_matrix(cycles)
    row_by_id = {int(i): j for j, i in enumerate(all_ids)}

    existing = pd.DataFrame()
    if SEED_RESULTS_PATH.exists() and not args.no_resume:
        existing = pd.read_csv(SEED_RESULTS_PATH)
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (str(r["detector"]), int(r["commissioning_size"]), int(r["seed"]), int(r["bootstrap_repetitions"]))
        for r in rows
    }

    total = len(detectors) * len(n_values) * len(seeds)
    done = 0
    for n_value in n_values:
        for seed in seeds:
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n_value,
                commissioning_seed=seed,
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=HEALTHY_EVAL_SIZE,
                maximum_commissioning_size=MAX_COMMISSIONING_SIZE,
            )
            source_raw = _rows_for_ids(all_raw, row_by_id, split.source_train)
            target_raw = _rows_for_ids(all_raw, row_by_id, split.target_commissioning)
            healthy_raw = _rows_for_ids(all_raw, row_by_id, split.target_normal_evaluation)
            anomaly_raw = _rows_for_ids(all_raw, row_by_id, split.target_anomaly_evaluation)

            for detector_name in detectors:
                done += 1
                key = (detector_name, int(n_value), int(seed), int(args.bootstrap_repetitions))
                if key in completed:
                    print(f"E15 {done}/{total}: skip completed {detector_name} N={n_value} seed={seed}")
                    continue

                row = _evaluate_cell(
                    detector_name=detector_name,
                    source_raw=source_raw,
                    target_raw=target_raw,
                    healthy_raw=healthy_raw,
                    anomaly_raw=anomaly_raw,
                    n_value=n_value,
                    seed=seed,
                    bootstrap_repetitions=int(args.bootstrap_repetitions),
                )
                rows.append(row)
                completed.add(key)
                _atomic_csv(pd.DataFrame(rows), SEED_RESULTS_PATH)
                print(
                    f"E15 {done}/{total}: {detector_name} N={n_value} seed={seed} "
                    f"oracle={row['observed_oracle_recall']:.3f} "
                    f"boot95=[{row['bootstrap_interval_lower']:.3f},"
                    f"{row['bootstrap_interval_upper']:.3f}] "
                    f"feasRate={row['bootstrap_feasibility_rate']:.3f}"
                )

    results = pd.DataFrame(rows)
    selected = results[
        results.detector.isin(detectors)
        & results.commissioning_size.isin(n_values)
        & results.seed.isin(seeds)
        & (results.bootstrap_repetitions == int(args.bootstrap_repetitions))
    ].copy()
    _atomic_csv(_summary(selected), SUMMARY_PATH)

    full_design = (
        set(detectors) == set(DETECTORS)
        and set(n_values) == set(N_GRID)
        and set(seeds) == set(SEEDS)
        and int(args.bootstrap_repetitions) == DEFAULT_BOOTSTRAP_REPETITIONS
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "evaluation-sample stability audit of the retrospective empirical oracle",
        "dataset": "voraus-AD",
        "source_setting": 72,
        "target_setting": 73,
        "detectors": list(detectors),
        "race_lambda": RACE_LAMBDA,
        "commissioning_grid": list(n_values),
        "commissioning_seeds": list(seeds),
        "evaluation_seed": EVALUATION_SEED,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "false_alert_budget": B,
        "recall_target": R0,
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_resampling_unit": "held-out robot execution/cycle",
        "bootstrap_scheme": "healthy and anomalous evaluation executions resampled independently with replacement at their original sample sizes",
        "interval": "95% percentile bootstrap sensitivity interval",
        "oracle_definition": "maximum empirical anomaly recall over thresholds with empirical healthy FPR <= 0.01",
        "fast_oracle_guard": "every observed fast oracle value is checked against src.oracle_feasibility.empirical_oracle_feasibility before bootstrap reporting",
        "interpretation_guard": "Bootstrap intervals quantify sensitivity to the finite fixed evaluation sample only. They are not population confidence intervals, do not replace exact post-freeze binomial certification, and are not used for tuning or N* redefinition.",
        "full_frozen_design_complete": bool(full_design),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(_summary(selected).to_string(index=False))
    print(f"Wrote E15 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
