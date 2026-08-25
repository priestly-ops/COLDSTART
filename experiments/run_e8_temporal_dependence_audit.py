from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.calibration_tail import conformal_threshold_info
from src.certification import exact_one_sided_fpr_upper
from src.detectors import TargetOnlyDetector
from src.e8_dependence import (
    autocorrelation,
    block_bootstrap_interval,
    permutation_pvalue_lag1,
    summarize_dependence,
)
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.split_generator import TARGET_SETTING, create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e8_temporal_dependence"
SEED_RESULTS_PATH = OUTPUT_DIR / "e8_seed_results.csv"
BLOCK_RESULTS_PATH = OUTPUT_DIR / "e8_block_bootstrap.csv"
FEATURE_RESULTS_PATH = OUTPUT_DIR / "e8_feature_dependence.csv"
SUMMARY_PATH = OUTPUT_DIR / "e8_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e8_manifest.json"

PROTOCOL_VERSION = "coldstart-e8-temporal-dependence-v1"
GLOBAL_SEED = 42
EVALUATION_SEED = 42
SEEDS = tuple(range(20))
COMMISSIONING_SIZE = 100
CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
MAX_LAG = 20
PERMUTATIONS = 5000
BOOTSTRAPS = 5000
BLOCK_SIZES = (1, 2, 5, 10, 20)
JOINT_CONFIDENCE = 0.95

np.random.seed(GLOBAL_SEED)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _lag1_per_feature(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    values: list[float] = []
    for j in range(x.shape[1]):
        column = x[:, j]
        if np.std(column) <= 0.0:
            continue
        values.append(float(autocorrelation(column, max_lag=1)[0]))
    if not values:
        raise RuntimeError("No non-constant features were available for E8.")
    return np.asarray(values, dtype=np.float64)


def _ordered_scores(ids: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(ids, kind="stable")
    return ids[order], scores[order]


def _feature_level_audit(cycles) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_healthy = sorted(
        [c for c in cycles if (not c.anomaly and c.setting == TARGET_SETTING)],
        key=lambda c: c.episode_id,
    )
    raw, ids = extract_feature_matrix(target_healthy)
    means = np.mean(raw, axis=0)
    stds = np.std(raw, axis=0, ddof=1)
    keep = stds > 1e-12
    z = (raw[:, keep] - means[keep]) / stds[keep]

    lag1 = _lag1_per_feature(z)
    pca1 = PCA(n_components=1, svd_solver="full").fit_transform(z).reshape(-1)
    pca_summary = summarize_dependence(pca1, max_lag=MAX_LAG)
    pca_perm = permutation_pvalue_lag1(
        pca1,
        n_permutations=PERMUTATIONS,
        seed=GLOBAL_SEED,
    )

    gaps = np.diff(np.asarray(ids, dtype=np.int64))
    gap_info = {
        "n_target_healthy": int(len(ids)),
        "episode_id_min": int(np.min(ids)),
        "episode_id_max": int(np.max(ids)),
        "median_episode_id_gap": float(np.median(gaps)) if gaps.size else None,
        "max_episode_id_gap": int(np.max(gaps)) if gaps.size else None,
        "fraction_unit_episode_id_gaps": float(np.mean(gaps == 1)) if gaps.size else None,
        "pca1_lag1_autocorrelation": pca_summary.lag1_autocorrelation,
        "pca1_ljung_box_pvalue": pca_summary.ljung_box_pvalue,
        "pca1_effective_sample_size": pca_summary.effective_sample_size,
        "pca1_effective_sample_fraction": pca_summary.effective_sample_size / pca_summary.n,
        "pca1_lag1_permutation_pvalue": pca_perm,
        "median_abs_feature_lag1": float(np.median(np.abs(lag1))),
        "p90_abs_feature_lag1": float(np.quantile(np.abs(lag1), 0.90)),
        "max_abs_feature_lag1": float(np.max(np.abs(lag1))),
        "number_nonconstant_features": int(lag1.size),
    }

    rows = pd.DataFrame(
        {
            "protocol_version": PROTOCOL_VERSION,
            "feature_index_nonconstant": np.arange(lag1.size, dtype=int),
            "lag1_autocorrelation": lag1,
            "abs_lag1_autocorrelation": np.abs(lag1),
        }
    )
    return rows, gap_info


def _seed_audit(cycles, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = create_frozen_evaluation_split(
        cycles=cycles,
        commissioning_size=COMMISSIONING_SIZE,
        commissioning_seed=seed,
        evaluation_seed=EVALUATION_SEED,
        calibration_size=CALIBRATION_SIZE,
        normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
        maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
    )

    source_raw, _ = extract_feature_matrix(split.source_train)
    target_raw, _ = extract_feature_matrix(split.target_commissioning)
    calibration_raw, calibration_ids = extract_feature_matrix(split.target_calibration)
    healthy_raw, healthy_ids = extract_feature_matrix(split.target_normal_evaluation)

    detector, preprocessor, _, _ = fit_detector(
        detector_name="TargetOnly",
        detector_factory=lambda: TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET),
        source_raw=source_raw,
        target_raw=target_raw,
    )
    calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))

    cal_ids_ordered, cal_scores_ordered = _ordered_scores(calibration_ids, calibration_scores)
    eval_ids_ordered, eval_scores_ordered = _ordered_scores(healthy_ids, healthy_scores)
    fixed_ids = np.concatenate([cal_ids_ordered, eval_ids_ordered])
    fixed_scores = np.concatenate([cal_scores_ordered, eval_scores_ordered])
    fixed_order = np.argsort(fixed_ids, kind="stable")
    fixed_ids = fixed_ids[fixed_order]
    fixed_scores = fixed_scores[fixed_order]

    cal_dep = summarize_dependence(cal_scores_ordered, max_lag=MAX_LAG)
    eval_dep = summarize_dependence(eval_scores_ordered, max_lag=MAX_LAG)
    fixed_dep = summarize_dependence(fixed_scores, max_lag=MAX_LAG)
    fixed_perm = permutation_pvalue_lag1(
        fixed_scores,
        n_permutations=PERMUTATIONS,
        seed=GLOBAL_SEED + seed,
    )

    info = conformal_threshold_info(calibration_scores, alpha=FALSE_ALERT_BUDGET)
    threshold = float(info.strict_threshold)
    alarms = (eval_scores_ordered > threshold).astype(np.float64)
    fp = int(np.sum(alarms))
    tn = int(len(alarms) - fp)
    iid_fpr_upper = exact_one_sided_fpr_upper(
        fp=fp,
        tn=tn,
        delta=(1.0 - JOINT_CONFIDENCE) / 2.0,
    )

    if 0 < fp < len(alarms):
        alarm_dep = summarize_dependence(alarms, max_lag=min(MAX_LAG, len(alarms) - 2))
        alarm_lag1 = alarm_dep.lag1_autocorrelation
        alarm_ess = alarm_dep.effective_sample_size
        alarm_ljung_p = alarm_dep.ljung_box_pvalue
    else:
        # A constant 0/1 sequence has zero variance, so autocorrelation and
        # ESS are not statistically identifiable. Keep these explicitly NaN.
        alarm_lag1 = float("nan")
        alarm_ess = float("nan")
        alarm_ljung_p = float("nan")

    row = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": "TargetOnly",
        "commissioning_size": COMMISSIONING_SIZE,
        "seed": int(seed),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_count": HEALTHY_EVALUATION_SIZE,
        "threshold": threshold,
        "fp": fp,
        "tn": tn,
        "empirical_fpr": float(fp / len(alarms)),
        "iid_exact_fpr_upper": float(iid_fpr_upper),
        "calibration_score_lag1": cal_dep.lag1_autocorrelation,
        "calibration_score_ljung_p": cal_dep.ljung_box_pvalue,
        "calibration_score_ess": cal_dep.effective_sample_size,
        "evaluation_score_lag1": eval_dep.lag1_autocorrelation,
        "evaluation_score_ljung_p": eval_dep.ljung_box_pvalue,
        "evaluation_score_ess": eval_dep.effective_sample_size,
        "fixed_healthy_score_lag1": fixed_dep.lag1_autocorrelation,
        "fixed_healthy_score_max_abs_acf": fixed_dep.max_abs_autocorrelation,
        "fixed_healthy_score_ljung_p": fixed_dep.ljung_box_pvalue,
        "fixed_healthy_score_ess": fixed_dep.effective_sample_size,
        "fixed_healthy_score_ess_fraction": fixed_dep.effective_sample_size / fixed_dep.n,
        "fixed_healthy_lag1_permutation_p": fixed_perm,
        "alarm_lag1": alarm_lag1,
        "alarm_ljung_p": alarm_ljung_p,
        "alarm_ess": alarm_ess,
    }

    block_rows: list[dict[str, Any]] = []
    for block_size in BLOCK_SIZES:
        low, high = block_bootstrap_interval(
            alarms,
            block_size=block_size,
            confidence=JOINT_CONFIDENCE,
            n_bootstrap=BOOTSTRAPS,
            seed=GLOBAL_SEED + 1000 * seed + block_size,
        )
        block_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "seed": int(seed),
                "block_size": int(block_size),
                "empirical_fpr": float(np.mean(alarms)),
                "bootstrap_fpr_lower": low,
                "bootstrap_fpr_upper": high,
                "iid_exact_fpr_upper": float(iid_fpr_upper),
            }
        )
    return row, block_rows


def _build_summary(seed_df: pd.DataFrame, block_df: pd.DataFrame, feature_info: dict[str, Any]) -> pd.DataFrame:
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "n_seeds": int(len(seed_df)),
        "mean_empirical_fpr": float(seed_df.empirical_fpr.mean()),
        "median_fixed_score_lag1": float(seed_df.fixed_healthy_score_lag1.median()),
        "mean_abs_fixed_score_lag1": float(seed_df.fixed_healthy_score_lag1.abs().mean()),
        "fraction_fixed_score_ljung_p_lt_005": float((seed_df.fixed_healthy_score_ljung_p < 0.05).mean()),
        "fraction_fixed_score_perm_p_lt_005": float((seed_df.fixed_healthy_lag1_permutation_p < 0.05).mean()),
        "median_fixed_score_ess_fraction": float(seed_df.fixed_healthy_score_ess_fraction.median()),
        "mean_iid_exact_fpr_upper": float(seed_df.iid_exact_fpr_upper.mean()),
    }
    for block_size in BLOCK_SIZES:
        g = block_df[block_df.block_size == block_size]
        row[f"mean_block{block_size}_fpr_upper"] = float(g.bootstrap_fpr_upper.mean())
        row[f"max_block{block_size}_fpr_upper"] = float(g.bootstrap_fpr_upper.max())
    row.update(feature_info)
    return pd.DataFrame([row])


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    feature_df, feature_info = _feature_level_audit(cycles)
    _atomic_csv(feature_df, FEATURE_RESULTS_PATH)

    seed_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS, start=1):
        row, blocks = _seed_audit(cycles, seed)
        seed_rows.append(row)
        block_rows.extend(blocks)
        _atomic_csv(pd.DataFrame(seed_rows), SEED_RESULTS_PATH)
        _atomic_csv(pd.DataFrame(block_rows), BLOCK_RESULTS_PATH)
        print(
            f"E8 seed {index}/{len(SEEDS)}: seed={seed} "
            f"lag1={row['fixed_healthy_score_lag1']:.3f} "
            f"ESS={row['fixed_healthy_score_ess']:.1f}/200 "
            f"FP={row['fp']}"
        )

    seed_df = pd.DataFrame(seed_rows)
    block_df = pd.DataFrame(block_rows)
    summary = _build_summary(seed_df, block_df, feature_info)
    _atomic_csv(summary, SUMMARY_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "audit sample-order dependence underlying exchangeability/independence assumptions",
        "dataset": "voraus-AD target setting PRE_B / setting 73",
        "ordering": "ascending episode/sample ID used as acquisition-order proxy; not asserted to be a wall-clock timestamp",
        "target_healthy_count_for_feature_audit": feature_info["n_target_healthy"],
        "detector": "TargetOnly",
        "commissioning_size": COMMISSIONING_SIZE,
        "commissioning_seeds": list(SEEDS),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": HEALTHY_EVALUATION_SIZE,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "max_lag": MAX_LAG,
        "lag1_permutation_replicates": PERMUTATIONS,
        "moving_block_bootstrap_replicates": BOOTSTRAPS,
        "block_sizes": list(BLOCK_SIZES),
        "notes": [
            "Feature-level dependence is diagnostic only and uses all healthy target executions.",
            "Score-level audit uses the frozen E0 calibration/evaluation populations and excludes commissioning episodes.",
            "The moving-block bootstrap is a sensitivity analysis, not a replacement finite-sample certification guarantee.",
            "Standard conformal/binomial guarantees are not claimed under arbitrary temporal dependence.",
            "Alarm-sequence autocorrelation/ESS is reported as NaN when the binary sequence is constant.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote E8 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
