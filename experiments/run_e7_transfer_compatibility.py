from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.calibration_tail import conformal_threshold_info
from src.detectors import RACEDetector, TargetOnlyDetector
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.split_generator import create_frozen_evaluation_split
from src.transfer_compatibility import compute_compatibility_metrics
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e7_transfer_compatibility"
PAIR_RESULTS_PATH = OUTPUT_DIR / "e7_paired_results.csv"
ASSOCIATION_PATH = OUTPUT_DIR / "e7_associations.csv"
SUMMARY_PATH = OUTPUT_DIR / "e7_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e7_manifest.json"

PROTOCOL_VERSION = "coldstart-e7-transfer-compatibility-v1"
N_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
B = 0.01
R0 = 0.90
GLOBAL_SEED = 42

COMPATIBILITY_COLUMNS = (
    "mean_shift_rms",
    "covariance_correlation_frobenius",
    "rbf_mmd2",
    "domain_classifier_auc",
)
OUTCOME_COLUMNS = (
    "delta_oracle_recall_at_fpr_budget",
    "delta_auroc",
    "delta_deployed_recall",
    "delta_deployed_fpr",
)

np.random.seed(GLOBAL_SEED)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _rows_for_ids(matrix: np.ndarray, row_by_id: dict[int, int], cycles) -> np.ndarray:
    ids = [int(c.episode_id) for c in cycles]
    return matrix[np.asarray([row_by_id[i] for i in ids], dtype=np.int64)]


def _detector_metrics(
    *,
    detector_name: str,
    detector_factory,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    calibration_raw: np.ndarray,
    healthy_raw: np.ndarray,
    anomaly_raw: np.ndarray,
) -> dict[str, float]:
    detector, preprocessor, _, _ = fit_detector(
        detector_name=detector_name,
        detector_factory=detector_factory,
        source_raw=source_raw,
        target_raw=target_raw,
    )
    calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))
    threshold_info = conformal_threshold_info(calibration_scores, alpha=B)
    threshold = float(threshold_info.strict_threshold)
    recall = float(np.mean(anomaly_scores > threshold))
    fpr = float(np.mean(healthy_scores > threshold))
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
    )
    return {
        "recall": recall,
        "fpr": fpr,
        "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
        "oracle_recall": float(oracle.max_recall_at_fpr_budget),
        "oracle_feasible": float(oracle.empirically_feasible),
        "threshold": threshold,
    }


def _within_n_rank_correlation(df: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    xr = np.empty(len(df), dtype=np.float64)
    yr = np.empty(len(df), dtype=np.float64)
    for _, idx in df.groupby("commissioning_size").groups.items():
        loc = np.asarray(list(idx), dtype=int)
        xr[loc] = rankdata(df.loc[loc, x].to_numpy(dtype=float), method="average")
        yr[loc] = rankdata(df.loc[loc, y].to_numpy(dtype=float), method="average")
        xr[loc] -= xr[loc].mean()
        yr[loc] -= yr[loc].mean()
    if np.std(xr) <= 0 or np.std(yr) <= 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(xr, yr)[0, 1])
    # Permutation p-value preserving N groups.
    rng = np.random.default_rng(GLOBAL_SEED)
    more_extreme = 0
    repetitions = 5000
    for _ in range(repetitions):
        yp = yr.copy()
        for _, idx in df.groupby("commissioning_size").groups.items():
            loc = np.asarray(list(idx), dtype=int)
            yp[loc] = rng.permutation(yp[loc])
        rp = float(np.corrcoef(xr, yp)[0, 1])
        if abs(rp) >= abs(r) - 1e-15:
            more_extreme += 1
    p = float((more_extreme + 1) / (repetitions + 1))
    return r, p


def _association_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in COMPATIBILITY_COLUMNS:
        for outcome in OUTCOME_COLUMNS:
            for n_value, g in df.groupby("commissioning_size", sort=True):
                result = spearmanr(g[metric], g[outcome])
                rows.append({
                    "scope": f"N={int(n_value)}",
                    "commissioning_size": int(n_value),
                    "compatibility_metric": metric,
                    "outcome": outcome,
                    "spearman_rho": float(result.statistic),
                    "pvalue": float(result.pvalue),
                    "n": int(len(g)),
                })
            rho, p = _within_n_rank_correlation(df, metric, outcome)
            rows.append({
                "scope": "within_N_pooled",
                "commissioning_size": np.nan,
                "compatibility_metric": metric,
                "outcome": outcome,
                "spearman_rho": rho,
                "pvalue": p,
                "n": int(len(df)),
            })
    return pd.DataFrame(rows)


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_value, g in df.groupby("commissioning_size", sort=True):
        delta = g["delta_oracle_recall_at_fpr_budget"].to_numpy(dtype=float)
        rows.append({
            "commissioning_size": int(n_value),
            "n_runs": int(len(g)),
            "mean_targetonly_oracle_recall": float(g.targetonly_oracle_recall_at_fpr_budget.mean()),
            "mean_race_oracle_recall": float(g.race_oracle_recall_at_fpr_budget.mean()),
            "mean_delta_oracle_recall": float(delta.mean()),
            "median_delta_oracle_recall": float(np.median(delta)),
            "positive_transfer_rate": float(np.mean(delta > 1e-12)),
            "negative_transfer_rate": float(np.mean(delta < -1e-12)),
            "neutral_transfer_rate": float(np.mean(np.abs(delta) <= 1e-12)),
            "mean_delta_auroc": float(g.delta_auroc.mean()),
            "mean_delta_deployed_recall": float(g.delta_deployed_recall.mean()),
            "mean_delta_deployed_fpr": float(g.delta_deployed_fpr.mean()),
            "mean_mean_shift_rms": float(g.mean_shift_rms.mean()),
            "mean_covariance_discrepancy": float(g.covariance_correlation_frobenius.mean()),
            "mean_rbf_mmd2": float(g.rbf_mmd2.mean()),
            "mean_domain_classifier_auc": float(g.domain_classifier_auc.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default=",".join(map(str, N_GRID)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    n_values = _parse_ints(args.n_values)
    seeds = _parse_ints(args.seeds)
    if not set(n_values).issubset(N_GRID):
        raise ValueError(f"n-values must be a subset of {N_GRID}")
    if not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be a subset of 0..19")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    all_raw, all_ids = extract_feature_matrix(cycles)
    row_by_id = {int(i): j for j, i in enumerate(all_ids)}

    rows: list[dict[str, Any]] = []
    total = len(n_values) * len(seeds)
    done = 0
    for n_value in n_values:
        for seed in seeds:
            done += 1
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

            compatibility = compute_compatibility_metrics(
                source_raw,
                target_raw,
                seed=GLOBAL_SEED + 1000 * n_value + seed,
            )
            targetonly = _detector_metrics(
                detector_name="TargetOnly",
                detector_factory=lambda: TargetOnlyDetector(false_alert_budget=B),
                source_raw=source_raw,
                target_raw=target_raw,
                calibration_raw=calibration_raw,
                healthy_raw=healthy_raw,
                anomaly_raw=anomaly_raw,
            )
            race = _detector_metrics(
                detector_name="RACE",
                detector_factory=lambda: RACEDetector(false_alert_budget=B),
                source_raw=source_raw,
                target_raw=target_raw,
                calibration_raw=calibration_raw,
                healthy_raw=healthy_raw,
                anomaly_raw=anomaly_raw,
            )
            row = {
                "protocol_version": PROTOCOL_VERSION,
                "commissioning_size": int(n_value),
                "seed": int(seed),
                **compatibility.__dict__,
                "targetonly_recall": targetonly["recall"],
                "targetonly_fpr": targetonly["fpr"],
                "targetonly_auroc": targetonly["auroc"],
                "targetonly_oracle_recall_at_fpr_budget": targetonly["oracle_recall"],
                "targetonly_oracle_feasible": bool(targetonly["oracle_feasible"]),
                "race_recall": race["recall"],
                "race_fpr": race["fpr"],
                "race_auroc": race["auroc"],
                "race_oracle_recall_at_fpr_budget": race["oracle_recall"],
                "race_oracle_feasible": bool(race["oracle_feasible"]),
                "delta_oracle_recall_at_fpr_budget": race["oracle_recall"] - targetonly["oracle_recall"],
                "delta_auroc": race["auroc"] - targetonly["auroc"],
                "delta_deployed_recall": race["recall"] - targetonly["recall"],
                "delta_deployed_fpr": race["fpr"] - targetonly["fpr"],
            }
            rows.append(row)
            _atomic_csv(pd.DataFrame(rows), PAIR_RESULTS_PATH)
            print(
                f"E7 {done}/{total}: N={n_value} seed={seed} "
                f"domainAUC={compatibility.domain_classifier_auc:.3f} "
                f"dOracle={row['delta_oracle_recall_at_fpr_budget']:+.3f} "
                f"dAUROC={row['delta_auroc']:+.3f}"
            )

    df = pd.DataFrame(rows).reset_index(drop=True)
    associations = _association_table(df)
    summary = _summary_table(df)
    _atomic_csv(associations, ASSOCIATION_PATH)
    _atomic_csv(summary, SUMMARY_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "healthy-only source-target compatibility and RACE negative-transfer audit",
        "dataset": "voraus-AD",
        "source_setting": 72,
        "target_setting": 73,
        "commissioning_grid": list(n_values),
        "seeds": list(seeds),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "false_alert_budget": B,
        "recall_target": R0,
        "primary_transfer_outcome": "RACE minus TargetOnly oracle recall at empirical FPR <= 0.01",
        "negative_transfer_definition": "delta_oracle_recall_at_fpr_budget < 0",
        "compatibility_metrics": {
            "mean_shift_rms": "RMS target-minus-source mean shift after source standardization",
            "covariance_correlation_frobenius": "normalized Frobenius difference between Ledoit-Wolf correlation matrices",
            "rbf_mmd2": "balanced RBF MMD^2 with median-distance bandwidth",
            "domain_classifier_auc": "balanced cross-validated source-vs-target logistic AUC, folded so >=0.5",
        },
        "association_policy": "Spearman within each N plus pooled within-N rank correlation with 5000 group-preserving permutations",
        "leakage_policy": "compatibility metrics use source healthy and target commissioning healthy data only; anomaly labels enter only retrospective transfer outcomes",
        "interpretation": "larger compatibility metrics mean greater source-target discrepancy; a negative association with RACE benefit is consistent with compatibility-limited transfer",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote E7 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
