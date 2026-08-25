from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bottleneck_decomposition import classify_bottleneck
from src.calibration_tail import conformal_threshold_info
from src.certification import certify_operating_point
from src.detectors import TargetOnlyDetector
from src.feature_extractor import FeatureBatch, FeaturePreprocessor, load_feature_batch
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "outputs" / "aursad" / "feature_cache" / "aursad_features.npz"
PROTOCOL_DIR = PROJECT_ROOT / "reports" / "aursad" / "coldstart_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "e9_coldstart"
SEED_RESULTS_PATH = OUTPUT_DIR / "e9_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e9_summary.csv"
PER_CLASS_PATH = OUTPUT_DIR / "e9_per_class_recall.csv"
NSTAR_PATH = OUTPUT_DIR / "e9_nstar.csv"
MANIFEST_PATH = OUTPUT_DIR / "e9_manifest.json"

PROTOCOL_VERSION = "coldstart-e9-aursad-v1"
GRID = [10, 25, 50, 100, 250, 500]
SEEDS = list(range(20))
CALIBRATION_GRID = [50, 99, 100, 198, 199, 250, 500]
PRIMARY_MCAL = 199
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
Q0 = 0.80
EXPECTED_CALIBRATION_POOL = 500
EXPECTED_HEALTHY_EVAL = 368
EXPECTED_ANOMALY_EVAL = 625
GLOBAL_SEED = 42

np.random.seed(GLOBAL_SEED)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _load_csv(name: str, partition: str) -> pd.DataFrame:
    path = PROTOCOL_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing E9 protocol file: {path}. Build reports/aursad/coldstart_v2 first."
        )
    frame = pd.read_csv(path)
    required = {"sample_nr", "partition", "label", "label_name", "selection_rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_nr"] = pd.to_numeric(frame["sample_nr"], errors="raise").astype(np.int64)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    frame["selection_rank"] = pd.to_numeric(frame["selection_rank"], errors="raise").astype(np.int64)
    if set(frame["partition"].astype(str)) != {partition}:
        raise ValueError(f"{name} does not contain only partition={partition!r}")
    return frame


def _load_protocol() -> dict[str, pd.DataFrame]:
    tables = {
        "commissioning": _load_csv("commissioning_ids.csv", "commissioning"),
        "calibration": _load_csv("calibration_ids.csv", "calibration"),
        "healthy_eval": _load_csv("healthy_eval_ids.csv", "healthy_eval"),
        "anomaly_eval": _load_csv("anomaly_eval_ids.csv", "anomaly_eval"),
    }
    commissioning = tables["commissioning"]
    for col in ("seed", "commissioning_n"):
        if col not in commissioning.columns:
            raise ValueError(f"commissioning_ids.csv missing {col}")
        commissioning[col] = pd.to_numeric(commissioning[col], errors="raise").astype(np.int64)

    calibration = tables["calibration"].sort_values("selection_rank").reset_index(drop=True)
    healthy_eval = tables["healthy_eval"].sort_values("selection_rank").reset_index(drop=True)
    anomaly_eval = tables["anomaly_eval"].sort_values("sample_nr").reset_index(drop=True)
    tables["calibration"] = calibration
    tables["healthy_eval"] = healthy_eval
    tables["anomaly_eval"] = anomaly_eval

    if len(calibration) != EXPECTED_CALIBRATION_POOL:
        raise ValueError(f"Expected {EXPECTED_CALIBRATION_POOL} calibration executions, found {len(calibration)}")
    if len(healthy_eval) != EXPECTED_HEALTHY_EVAL:
        raise ValueError(f"Expected {EXPECTED_HEALTHY_EVAL} healthy eval executions, found {len(healthy_eval)}")
    if len(anomaly_eval) != EXPECTED_ANOMALY_EVAL:
        raise ValueError(f"Expected {EXPECTED_ANOMALY_EVAL} anomaly executions, found {len(anomaly_eval)}")
    if not calibration["label"].eq(0).all() or not healthy_eval["label"].eq(0).all():
        raise ValueError("Healthy protocol partitions contain non-normal labels")
    if not anomaly_eval["label"].isin([1, 2, 3, 4]).all():
        raise ValueError("Anomaly evaluation contains unsupported labels")

    fixed = {
        "calibration": set(calibration["sample_nr"].astype(int)),
        "healthy_eval": set(healthy_eval["sample_nr"].astype(int)),
        "anomaly_eval": set(anomaly_eval["sample_nr"].astype(int)),
    }
    names = list(fixed)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = fixed[left] & fixed[right]
            if overlap:
                raise RuntimeError(f"Leakage between {left} and {right}: {sorted(overlap)[:10]}")

    for seed in SEEDS:
        previous: set[int] = set()
        for n_value in GRID:
            rows = commissioning[(commissioning["seed"] == seed) & (commissioning["commissioning_n"] == n_value)]
            ids = set(rows["sample_nr"].astype(int))
            if len(ids) != n_value:
                raise ValueError(f"seed={seed}, N={n_value}: expected {n_value} unique commissioning IDs, found {len(ids)}")
            if previous and not previous.issubset(ids):
                raise RuntimeError(f"Commissioning sets are not nested for seed={seed}, N={n_value}")
            if any(ids & fixed_ids for fixed_ids in fixed.values()):
                raise RuntimeError(f"Commissioning leakage for seed={seed}, N={n_value}")
            previous = ids
    return tables


def _subset(cache: FeatureBatch, ids: list[int]) -> FeatureBatch:
    return cache.select_episode_ids(ids, preserve_requested_order=True, require_all=True)


def _commissioning_ids(table: pd.DataFrame, seed: int, n_value: int) -> list[int]:
    rows = table[(table["seed"] == seed) & (table["commissioning_n"] == n_value)].sort_values("selection_rank")
    return rows["sample_nr"].astype(int).tolist()


def _fit_and_score(
    cache: FeatureBatch,
    commissioning_ids: list[int],
    calibration_ids: list[int],
    healthy_ids: list[int],
    anomaly_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    commissioning = _subset(cache, commissioning_ids)
    calibration = _subset(cache, calibration_ids)
    healthy = _subset(cache, healthy_ids)
    anomaly = _subset(cache, anomaly_ids)

    pre = FeaturePreprocessor(variance_threshold=1e-12)
    target_train = pre.fit_transform(commissioning.features)
    calibration_features = pre.transform(calibration.features)
    healthy_features = pre.transform(healthy.features)
    anomaly_features = pre.transform(anomaly.features)

    detector = TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET)
    detector.fit(source_features=target_train, target_features=target_train)

    return (
        detector.score_samples(calibration_features),
        detector.score_samples(healthy_features),
        detector.score_samples(anomaly_features),
        int(pre.output_feature_count_),
    )


def _evaluate_prefix(
    calibration_scores: np.ndarray,
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    mcal: int,
) -> dict[str, Any]:
    info = conformal_threshold_info(calibration_scores[:mcal], alpha=FALSE_ALERT_BUDGET)
    threshold = float(info.strict_threshold)
    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold
    fp = int(healthy_pred.sum())
    tn = int((~healthy_pred).sum())
    tp = int(anomaly_pred.sum())
    fn = int((~anomaly_pred).sum())

    cert = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=RECALL_TARGET,
        fpr_budget=FALSE_ALERT_BUDGET,
        joint_confidence=JOINT_CONFIDENCE,
    )
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )
    bottleneck = classify_bottleneck(
        oracle=oracle,
        deployed_recall=cert.recall,
        deployed_fpr=cert.fpr,
        recall_lower=cert.recall_lower,
        fpr_upper=cert.fpr_upper,
        recall_target=RECALL_TARGET,
        fpr_budget=FALSE_ALERT_BUDGET,
    )
    if not info.finite_sample_feasible:
        regime = "infinite"
    elif info.threshold_is_maximum:
        regime = "maximum"
    else:
        regime = "submaximum"

    return {
        "threshold": threshold,
        "conformal_rank": int(info.raw_rank),
        "conformal_regime": regime,
        "minimum_attainable_alpha": float(info.minimum_attainable_alpha),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": float(cert.recall),
        "false_positive_rate": float(cert.fpr),
        "recall_lower": float(cert.recall_lower),
        "fpr_upper": float(cert.fpr_upper),
        "empirical_success": bool(cert.recall >= RECALL_TARGET and cert.fpr <= FALSE_ALERT_BUDGET),
        "certified_success": bool(cert.certified),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": str(bottleneck.bottleneck_label),
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (n_value, mcal), group in results.groupby(["commissioning_size", "calibration_size"], sort=True):
        rows.append({
            "commissioning_size": int(n_value),
            "calibration_size": int(mcal),
            "n_seeds": int(len(group)),
            "conformal_regime": str(group["conformal_regime"].iloc[0]),
            "mean_recall": float(group["recall"].mean()),
            "mean_fpr": float(group["false_positive_rate"].mean()),
            "mean_auroc": float(group["auroc"].mean()),
            "mean_oracle_recall_at_fpr_budget": float(group["oracle_recall_at_fpr_budget"].mean()),
            "oracle_feasibility_rate": float(group["oracle_empirically_feasible"].astype(bool).mean()),
            "empirical_success_rate": float(group["empirical_success"].astype(bool).mean()),
            "certified_success_rate": float(group["certified_success"].astype(bool).mean()),
            "representation_limited_rate": float((group["bottleneck_label"] == "representation_limited").mean()),
            "calibration_limited_rate": float((group["bottleneck_label"] == "calibration_limited").mean()),
            "certification_limited_rate": float((group["bottleneck_label"] == "certification_limited").mean()),
            "certified_rate": float((group["bottleneck_label"] == "certified").mean()),
            "mean_seedwise_recall_lower": float(group["recall_lower"].mean()),
            "mean_seedwise_fpr_upper": float(group["fpr_upper"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["commissioning_size", "calibration_size"]).reset_index(drop=True)


def _build_nstar(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[summary["calibration_size"] == PRIMARY_MCAL].copy()
    rows = []
    for metric in ["oracle_feasibility_rate", "empirical_success_rate", "certified_success_rate"]:
        passing = primary[primary[metric] >= Q0].sort_values("commissioning_size")
        rows.append({
            "calibration_size": PRIMARY_MCAL,
            "criterion": metric,
            "q0": Q0,
            "n_star": int(passing.iloc[0]["commissioning_size"]) if not passing.empty else np.nan,
            "censored": bool(passing.empty),
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"AURSAD feature cache missing: {CACHE_PATH}")
    tables = _load_protocol()
    cache = load_feature_batch(CACHE_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_ids = tables["calibration"]["sample_nr"].astype(int).tolist()
    healthy_ids = tables["healthy_eval"]["sample_nr"].astype(int).tolist()
    anomaly_ids = tables["anomaly_eval"]["sample_nr"].astype(int).tolist()
    anomaly_table = tables["anomaly_eval"]

    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for n_value in GRID:
            print(f"E9 TargetOnly-Stat N={n_value} seed={seed}/19")
            commission_ids = _commissioning_ids(tables["commissioning"], seed, n_value)
            calibration_scores, healthy_scores, anomaly_scores, retained = _fit_and_score(
                cache, commission_ids, calibration_ids, healthy_ids, anomaly_ids
            )
            auroc = probability_of_superiority(healthy_scores, anomaly_scores)

            for mcal in CALIBRATION_GRID:
                result = _evaluate_prefix(calibration_scores, healthy_scores, anomaly_scores, mcal)
                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "detector": "TargetOnly-Stat",
                    "commissioning_size": n_value,
                    "calibration_size": mcal,
                    "healthy_eval_count": len(healthy_ids),
                    "anomaly_eval_count": len(anomaly_ids),
                    "seed": seed,
                    "retained_features": retained,
                    "auroc": auroc,
                    **result,
                }
                rows.append(row)

                threshold = result["threshold"]
                prediction_by_id = {
                    int(ep): bool(score > threshold)
                    for ep, score in zip(anomaly_ids, anomaly_scores)
                }
                for (label, label_name), group in anomaly_table.groupby(["label", "label_name"], sort=True):
                    ids = group["sample_nr"].astype(int).tolist()
                    preds = np.asarray([prediction_by_id[i] for i in ids], dtype=bool)
                    class_rows.append({
                        "protocol_version": PROTOCOL_VERSION,
                        "commissioning_size": n_value,
                        "calibration_size": mcal,
                        "seed": seed,
                        "label": int(label),
                        "label_name": str(label_name),
                        "execution_count": int(len(ids)),
                        "recall": float(preds.mean()),
                        "descriptive_only": bool(int(label) == 4),
                    })

            _atomic_write_csv(pd.DataFrame(rows), SEED_RESULTS_PATH)
            _atomic_write_csv(pd.DataFrame(class_rows), PER_CLASS_PATH)

    results = pd.DataFrame(rows)
    summary = _build_summary(results)
    nstar = _build_nstar(summary)
    _atomic_write_csv(summary, SUMMARY_PATH)
    _atomic_write_csv(nstar, NSTAR_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": "TargetOnly-Stat",
        "commissioning_grid": GRID,
        "commissioning_seeds": SEEDS,
        "calibration_grid": CALIBRATION_GRID,
        "primary_calibration_size": PRIMARY_MCAL,
        "healthy_evaluation_size": EXPECTED_HEALTHY_EVAL,
        "anomaly_evaluation_size": EXPECTED_ANOMALY_EVAL,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "q0": Q0,
        "episode_unit": "sample_nr execution",
        "alarm_rule": "score > threshold",
        "calibration_rule": "strict deterministic split conformal",
        "oracle_is_diagnostic_only": True,
        "damaged_thread_label_4_descriptive_only": True,
        "purpose": "external AURSAD validation of representation/calibration/certification bottlenecks",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\nE9 complete")
    print(summary.to_string(index=False))
    print("\nN* sensitivity at Mcal=199")
    print(nstar.to_string(index=False))


if __name__ == "__main__":
    main()
