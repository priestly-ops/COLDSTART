from __future__ import annotations

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

from src.bottleneck_decomposition import classify_bottleneck
from src.ccv_conformal import (
    ccv_feasibility,
    classify_from_pvalues,
    dkwm_adjust_pvalues,
    marginal_conformal_pvalues,
)
from src.certification import certify_operating_point
from src.detectors import TargetOnlyDetector
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e19_modern_conformal_ccv"
SEED_RESULTS_PATH = OUTPUT_DIR / "e19_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e19_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e19_manifest.json"

PROTOCOL_VERSION = "coldstart-e19-modern-conformal-ccv-v1"
N_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
B = 0.01
R0 = 0.90
CONFIDENCE = 0.95
CCV_DELTA = 0.05
GLOBAL_SEED = 42

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


def _metric_row(
    *,
    variant: str,
    n_value: int,
    seed: int,
    base_auroc: float,
    oracle,
    healthy_pred: np.ndarray,
    anomaly_pred: np.ndarray,
    min_pvalue: float,
    alpha_feasible: bool,
) -> dict[str, Any]:
    fp = int(np.sum(healthy_pred))
    tn = int(len(healthy_pred) - fp)
    tp = int(np.sum(anomaly_pred))
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
        "variant": variant,
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_count": int(len(healthy_pred)),
        "anomaly_eval_count": int(len(anomaly_pred)),
        "alpha": B,
        "ccv_delta": CCV_DELTA if variant == "CAD-CCV-DKWM" else np.nan,
        "minimum_attainable_pvalue": float(min_pvalue),
        "alpha_feasible_from_calibration_granularity": bool(alpha_feasible),
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
        "auroc": float(base_auroc),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": str(bottleneck.bottleneck_label),
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, n_value), g in df.groupby(["variant", "commissioning_size"], sort=True):
        rows.append({
            "variant": str(variant),
            "commissioning_size": int(n_value),
            "n_runs": int(len(g)),
            "mean_recall": float(g.recall.mean()),
            "mean_fpr": float(g.false_positive_rate.mean()),
            "mean_auroc": float(g.auroc.mean()),
            "mean_oracle_recall_at_fpr_budget": float(g.oracle_recall_at_fpr_budget.mean()),
            "oracle_feasibility_rate": float(g.oracle_empirically_feasible.astype(bool).mean()),
            "empirical_success_rate": float(g.empirical_success.astype(bool).mean()),
            "certified_success_rate": float(g.certified_success.astype(bool).mean()),
            "alpha_feasibility_rate": float(g.alpha_feasible_from_calibration_granularity.astype(bool).mean()),
            "mean_minimum_attainable_pvalue": float(g.minimum_attainable_pvalue.mean()),
            "representation_limited_rate": float((g.bottleneck_label == "representation_limited").mean()),
            "calibration_limited_rate": float((g.bottleneck_label == "calibration_limited").mean()),
            "certification_limited_rate": float((g.bottleneck_label == "certification_limited").mean()),
        })
    return pd.DataFrame(rows).sort_values(["variant", "commissioning_size"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E19 modern CAD calibration-conditional conformal comparator")
    parser.add_argument("--n-values", default=",".join(map(str, N_GRID)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--no-resume", action="store_true")
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

    existing = pd.DataFrame()
    if SEED_RESULTS_PATH.exists() and not args.no_resume:
        existing = pd.read_csv(SEED_RESULTS_PATH)
    rows = existing.to_dict("records") if not existing.empty else []
    completed = {
        (str(r["variant"]), int(r["commissioning_size"]), int(r["seed"]))
        for r in rows
    }

    variants = ("CAD-Marginal", "CAD-CCV-DKWM")
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

            if all((v, int(n_value), int(seed)) in completed for v in variants):
                print(f"E19 {done}/{total}: skip completed N={n_value} seed={seed}")
                continue

            detector, preprocessor, _, _ = fit_detector(
                detector_name="TargetOnly",
                detector_factory=lambda: TargetOnlyDetector(false_alert_budget=B),
                source_raw=source_raw,
                target_raw=target_raw,
            )
            cal_scores = detector.score_samples(preprocessor.transform(calibration_raw))
            healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
            anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))
            base_auroc = probability_of_superiority(healthy_scores, anomaly_scores)
            oracle = empirical_oracle_feasibility(
                healthy_scores=healthy_scores,
                anomaly_scores=anomaly_scores,
                false_alert_budget=B,
                recall_target=R0,
            )

            healthy_pm = marginal_conformal_pvalues(cal_scores, healthy_scores)
            anomaly_pm = marginal_conformal_pvalues(cal_scores, anomaly_scores)
            feasibility = ccv_feasibility(CALIBRATION_SIZE, alpha=B, delta=CCV_DELTA)

            if ("CAD-Marginal", int(n_value), int(seed)) not in completed:
                rows.append(_metric_row(
                    variant="CAD-Marginal",
                    n_value=n_value,
                    seed=seed,
                    base_auroc=base_auroc,
                    oracle=oracle,
                    healthy_pred=classify_from_pvalues(healthy_pm, alpha=B),
                    anomaly_pred=classify_from_pvalues(anomaly_pm, alpha=B),
                    min_pvalue=feasibility.minimum_marginal_pvalue,
                    alpha_feasible=feasibility.marginal_alpha_feasible,
                ))
                completed.add(("CAD-Marginal", int(n_value), int(seed)))

            if ("CAD-CCV-DKWM", int(n_value), int(seed)) not in completed:
                healthy_pccv = dkwm_adjust_pvalues(healthy_pm, calibration_size=CALIBRATION_SIZE, delta=CCV_DELTA)
                anomaly_pccv = dkwm_adjust_pvalues(anomaly_pm, calibration_size=CALIBRATION_SIZE, delta=CCV_DELTA)
                rows.append(_metric_row(
                    variant="CAD-CCV-DKWM",
                    n_value=n_value,
                    seed=seed,
                    base_auroc=base_auroc,
                    oracle=oracle,
                    healthy_pred=classify_from_pvalues(healthy_pccv, alpha=B),
                    anomaly_pred=classify_from_pvalues(anomaly_pccv, alpha=B),
                    min_pvalue=feasibility.minimum_dkwm_adjusted_pvalue,
                    alpha_feasible=feasibility.dkwm_alpha_feasible,
                ))
                completed.add(("CAD-CCV-DKWM", int(n_value), int(seed)))

            _atomic_csv(pd.DataFrame(rows), SEED_RESULTS_PATH)
            latest = pd.DataFrame(rows)
            g = latest[(latest.commissioning_size == n_value) & (latest.seed == seed)]
            print(
                f"E19 {done}/{total}: N={n_value} seed={seed} "
                + " | ".join(
                    f"{r.variant}: recall={r.recall:.3f} FPR={r.false_positive_rate:.3f} minp={r.minimum_attainable_pvalue:.4f}"
                    for r in g.itertuples(index=False)
                )
            )

    results = pd.DataFrame(rows)
    selected = results[
        results.commissioning_size.isin(n_values)
        & results.seed.isin(seeds)
        & results.variant.isin(variants)
    ].copy()
    _atomic_csv(_summary(selected), SUMMARY_PATH)

    feasibility = ccv_feasibility(CALIBRATION_SIZE, alpha=B, delta=CCV_DELTA)
    full_design = set(n_values) == set(N_GRID) and set(seeds) == set(SEEDS)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "modern industrial conformal anomaly-detection comparator using calibration-conditional p-value adjustment",
        "literature_mapping": {
            "application": "Kundacina et al., Conformal Anomaly Detection for Predictive Maintenance in Thermal Power Plants, IEEE Access, 2025, DOI 10.1109/ACCESS.2025.3546451",
            "statistical_basis": "Bates et al., Testing for outliers with conformal p-values, Annals of Statistics, 2023",
            "implemented_adjustment": "DKWM calibration-conditional adjustment",
        },
        "dataset": "voraus-AD",
        "source_setting": 72,
        "target_setting": 73,
        "base_anomaly_score": "frozen TargetOnly Ledoit-Wolf Mahalanobis score; same score used for both conformal variants",
        "commissioning_grid": list(n_values),
        "commissioning_seeds": list(seeds),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "alpha": B,
        "recall_target": R0,
        "joint_confidence": CONFIDENCE,
        "ccv_delta": CCV_DELTA,
        "variants": {
            "CAD-Marginal": "standard marginal conformal p-values",
            "CAD-CCV-DKWM": "DKWM calibration-conditional adjustment applied to the same marginal conformal p-values",
        },
        "calibration_granularity": {
            "minimum_marginal_pvalue": feasibility.minimum_marginal_pvalue,
            "minimum_dkwm_adjusted_pvalue": feasibility.minimum_dkwm_adjusted_pvalue,
            "marginal_alpha_feasible": feasibility.marginal_alpha_feasible,
            "dkwm_alpha_feasible": feasibility.dkwm_alpha_feasible,
        },
        "certification_policy": "detector and p-value mapping are frozen before independent evaluation; exact post-freeze binomial certification remains valid as a separate deployment claim",
        "interpretation_guard": "E19 isolates calibration-method effects by holding the TargetOnly anomaly score fixed. Failure of CAD-CCV at alpha=0.01 with only 100 calibration cycles must not be described as failure of the underlying score model.",
        "full_frozen_design_complete": bool(full_design),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(_summary(selected).to_string(index=False))
    print(f"Wrote E19 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
