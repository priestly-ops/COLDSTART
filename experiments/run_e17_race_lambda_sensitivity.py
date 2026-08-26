from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.bottleneck_decomposition import classify_bottleneck
from src.calibration_tail import conformal_threshold_info
from src.certification import certify_operating_point
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.paired_inference import summarize_paired_deltas
from src.race_lambda_sensitivity import (
    FROZEN_RACE_LAMBDA,
    RACE_LAMBDA_GRID,
    RACELambdaSensitivityDetector,
    lambda_label,
)
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e17_race_lambda_sensitivity"
SEED_RESULTS_PATH = OUTPUT_DIR / "e17_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e17_summary.csv"
PAIRED_PATH = OUTPUT_DIR / "e17_paired_inference_vs_lambda60.csv"
MANIFEST_PATH = OUTPUT_DIR / "e17_manifest.json"

PROTOCOL_VERSION = "coldstart-e17-race-lambda-sensitivity-v1"
N_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
B = 0.01
R0 = 0.90
CONFIDENCE = 0.95
GLOBAL_SEED = 42

np.random.seed(GLOBAL_SEED)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _parse_lambdas(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip().lower()
        if not token:
            continue
        value = math.inf if token in {"inf", "+inf", "infinity"} else float(token)
        if math.isnan(value) or value < 0.0:
            raise ValueError("lambda values must be non-negative or inf")
        values.append(value)
    return tuple(values)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _rows_for_ids(matrix: np.ndarray, row_by_id: dict[int, int], cycles) -> np.ndarray:
    ids = [int(c.episode_id) for c in cycles]
    return matrix[np.asarray([row_by_id[i] for i in ids], dtype=np.int64)]


def _evaluate_lambda(
    *,
    lambda_reg: float,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    calibration_raw: np.ndarray,
    healthy_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    n_value: int,
    seed: int,
) -> dict[str, Any]:
    detector, preprocessor, _, _ = fit_detector(
        detector_name="RACE",
        detector_factory=lambda: RACELambdaSensitivityDetector(
            lambda_reg=lambda_reg,
            false_alert_budget=B,
        ),
        source_raw=source_raw,
        target_raw=target_raw,
    )
    calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

    threshold_info = conformal_threshold_info(calibration_scores, alpha=B)
    threshold = float(threshold_info.strict_threshold)
    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold
    fp = int(healthy_pred.sum())
    tn = int(len(healthy_pred) - fp)
    tp = int(anomaly_pred.sum())
    fn = int(len(anomaly_pred) - tp)
    recall = float(tp / (tp + fn))
    fpr = float(fp / (fp + tn))

    bounds = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=R0,
        fpr_budget=B,
        joint_confidence=CONFIDENCE,
    )
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
    )
    bottleneck = classify_bottleneck(
        oracle=oracle,
        deployed_recall=recall,
        deployed_fpr=fpr,
        recall_lower=bounds.recall_lower,
        fpr_upper=bounds.fpr_upper,
        recall_target=R0,
        fpr_budget=B,
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "lambda_reg": "inf" if math.isinf(lambda_reg) else float(lambda_reg),
        "lambda_label": lambda_label(lambda_reg),
        "target_weight": float(detector.target_weight_),
        "calibration_size": int(len(calibration_scores)),
        "healthy_eval_count": int(len(healthy_scores)),
        "anomaly_eval_count": int(len(anomaly_scores)),
        "threshold": threshold,
        "conformal_rank": int(threshold_info.raw_rank),
        "conformal_regime": (
            "infinite" if not threshold_info.finite_sample_feasible
            else "maximum" if threshold_info.threshold_is_maximum
            else "submaximum"
        ),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "false_positive_rate": fpr,
        "recall_lower": float(bounds.recall_lower),
        "fpr_upper": float(bounds.fpr_upper),
        "empirical_success": bool(recall >= R0 and fpr <= B),
        "certified_success": bool(bounds.certified),
        "certification_valid": True,
        "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": str(bottleneck.bottleneck_label),
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (n_value, label), g in df.groupby(["commissioning_size", "lambda_label"], sort=False):
        rows.append({
            "commissioning_size": int(n_value),
            "lambda_label": str(label),
            "n_runs": int(len(g)),
            "mean_target_weight": float(g.target_weight.mean()),
            "mean_recall": float(g.recall.mean()),
            "mean_fpr": float(g.false_positive_rate.mean()),
            "mean_auroc": float(g.auroc.mean()),
            "mean_oracle_recall_at_fpr_budget": float(g.oracle_recall_at_fpr_budget.mean()),
            "oracle_feasibility_rate": float(g.oracle_empirically_feasible.astype(bool).mean()),
            "empirical_success_rate": float(g.empirical_success.astype(bool).mean()),
            "certified_success_rate": float(g.certified_success.astype(bool).mean()),
            "representation_limited_rate": float((g.bottleneck_label == "representation_limited").mean()),
            "calibration_limited_rate": float((g.bottleneck_label == "calibration_limited").mean()),
            "certification_limited_rate": float((g.bottleneck_label == "certification_limited").mean()),
        })
    out = pd.DataFrame(rows)
    order = {lambda_label(v): i for i, v in enumerate(RACE_LAMBDA_GRID)}
    out["_lambda_order"] = out.lambda_label.map(order)
    return out.sort_values(["commissioning_size", "_lambda_order"]).drop(columns="_lambda_order").reset_index(drop=True)


def _paired_inference(df: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "oracle_recall_at_fpr_budget",
        "auroc",
        "recall",
        "false_positive_rate",
    )
    rows: list[dict[str, Any]] = []
    reference_label = lambda_label(FROZEN_RACE_LAMBDA)
    for n_value, n_group in df.groupby("commissioning_size", sort=True):
        reference = n_group[n_group.lambda_label == reference_label].set_index("seed")
        if len(reference) != len(SEEDS):
            raise RuntimeError(f"N={n_value}: incomplete lambda=60 reference")
        for label, group in n_group.groupby("lambda_label", sort=False):
            current = group.set_index("seed")
            common = sorted(set(reference.index) & set(current.index))
            if len(common) != len(SEEDS):
                raise RuntimeError(f"N={n_value} lambda={label}: incomplete paired seeds")
            for metric in metrics:
                delta = current.loc[common, metric].to_numpy(dtype=float) - reference.loc[common, metric].to_numpy(dtype=float)
                result = summarize_paired_deltas(
                    delta,
                    confidence=0.95,
                    bootstrap_repetitions=10_000,
                    signflip_repetitions=100_000,
                    seed=GLOBAL_SEED + int(n_value) + sum(ord(c) for c in str(label)) + len(metric),
                )
                rows.append({
                    "commissioning_size": int(n_value),
                    "lambda_label": str(label),
                    "reference_lambda_label": reference_label,
                    "metric": metric,
                    "delta_definition": f"lambda={label} minus lambda={reference_label}",
                    **result.to_dict(),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="E17 RACE lambda sensitivity under the frozen voraus-AD protocol")
    parser.add_argument("--n-values", default=",".join(map(str, N_GRID)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--lambdas", default="0,10,30,60,100,300,inf")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    n_values = _parse_ints(args.n_values)
    seeds = _parse_ints(args.seeds)
    lambdas = _parse_lambdas(args.lambdas)
    if not set(n_values).issubset(N_GRID):
        raise ValueError(f"n-values must be a subset of {N_GRID}")
    if not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be a subset of 0..19")
    if not any((not math.isinf(v)) and v == FROZEN_RACE_LAMBDA for v in lambdas):
        raise ValueError("lambda=60 must be included because it is the frozen RACE reference")
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
        (int(r["commissioning_size"]), int(r["seed"]), str(r["lambda_label"]))
        for r in rows
    }

    total = len(n_values) * len(seeds) * len(lambdas)
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
            calibration_raw = _rows_for_ids(all_raw, row_by_id, split.target_calibration)
            healthy_raw = _rows_for_ids(all_raw, row_by_id, split.target_normal_evaluation)
            anomaly_raw = _rows_for_ids(all_raw, row_by_id, split.target_anomaly_evaluation)

            for lambda_reg in lambdas:
                done += 1
                label = lambda_label(lambda_reg)
                key = (int(n_value), int(seed), label)
                if key in completed:
                    print(f"E17 {done}/{total}: skip completed N={n_value} seed={seed} lambda={label}")
                    continue
                row = _evaluate_lambda(
                    lambda_reg=lambda_reg,
                    source_raw=source_raw,
                    target_raw=target_raw,
                    calibration_raw=calibration_raw,
                    healthy_raw=healthy_raw,
                    anomaly_raw=anomaly_raw,
                    n_value=n_value,
                    seed=seed,
                )
                rows.append(row)
                completed.add(key)
                _atomic_csv(pd.DataFrame(rows), SEED_RESULTS_PATH)
                print(
                    f"E17 {done}/{total}: N={n_value} seed={seed} lambda={label} "
                    f"wT={row['target_weight']:.3f} recall={row['recall']:.3f} "
                    f"FPR={row['false_positive_rate']:.3f} oracle={row['oracle_recall_at_fpr_budget']:.3f}"
                )

    results = pd.DataFrame(rows)
    selected = results[
        results.commissioning_size.isin(n_values)
        & results.seed.isin(seeds)
        & results.lambda_label.isin([lambda_label(v) for v in lambdas])
    ].copy()
    _atomic_csv(_summary(selected), SUMMARY_PATH)

    full_design = (
        set(n_values) == set(N_GRID)
        and set(seeds) == set(SEEDS)
        and set(lambda_label(v) for v in lambdas) == set(lambda_label(v) for v in RACE_LAMBDA_GRID)
    )
    if full_design:
        _atomic_csv(_paired_inference(selected), PAIRED_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "robustness audit of the frozen RACE shrinkage strength; not hyperparameter tuning",
        "dataset": "voraus-AD",
        "source_setting": 72,
        "target_setting": 73,
        "commissioning_grid": list(n_values),
        "commissioning_seeds": list(seeds),
        "lambda_grid": [lambda_label(v) for v in lambdas],
        "frozen_race_lambda": lambda_label(FROZEN_RACE_LAMBDA),
        "endpoint_semantics": {
            "0": "RACE-family target-Gaussian endpoint under pooled RACE preprocessing",
            "inf": "RACE-family source-Gaussian endpoint under pooled RACE preprocessing",
        },
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "false_alert_budget": B,
        "recall_target": R0,
        "joint_confidence": CONFIDENCE,
        "evaluation_seed": EVALUATION_SEED,
        "bootstrap_repetitions": 10_000,
        "signflip_repetitions": 100_000,
        "interpretation_guard": "lambda=60 remains the original frozen method. No lambda selected from this sweep may be used to redefine RACE or replace the primary result.",
        "full_frozen_design_complete": bool(full_design),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(_summary(selected).to_string(index=False))
    print(f"Wrote E17 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
