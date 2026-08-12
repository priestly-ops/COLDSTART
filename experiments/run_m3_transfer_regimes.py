"""Run M3 transferability-regime experiment on voraus-AD.

The primary detector comparison is frozen to the existing Aligned RACE-A0
implementation and its target-only, source-permutation, and weight-permutation
controls. Source regimes are constructed from healthy-only geometry before any
anomaly scoring is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligned_race_a0 import AlignedRACEA0Detector
from src.feature_extractor import extract_feature_batch
from src.m3_transfer_regimes import (
    TRANSFERABILITY_COLUMNS,
    add_transferability_regimes,
    assert_no_episode_leakage,
    construct_source_regimes,
    correlation_table,
    detector_summary,
    episode_ids,
    paired_deltas,
    paired_summary,
    scientific_decision,
    transfer_weight_diagnostics,
)
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
FROZEN_EVALUATION_SEED = 42

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PROTOCOL_VERSION = "m3-transfer-regimes-race-a0-v1"
OUTPUT_SCHEMA_VERSION = "m3-score-and-prediction-equivalence-audit-v3"
COMMISSIONING_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
SOURCE_SUBSET_SIZE = 100

K_MAX = 16
BETA = 0.50
LAMBDA_WEIGHT = 0.25
DIRECTION_MIN_COS2 = 0.20
GLOBAL_ALIGNMENT_MIN = 0.20

DETECTOR_MODES = {
    "TargetOnly": "target_only",
    "RACE": "aligned",
    "SourcePermutation": "feature_permuted",
    "WeightPermutation": "weight_permuted",
}


def _requested_run_config(
    *,
    data_path: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    source_subset_size: int,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset_path": str(data_path),
        "dataset_hash_sha256": _dataset_hash(data_path),
        "global_seed": GLOBAL_SEED,
        "frozen_evaluation_seed": FROZEN_EVALUATION_SEED,
        "split_generator": "create_frozen_evaluation_split",
        "commissioning_grid": list(n_values),
        "seeds": list(seeds),
        "calibration_size": CALIBRATION_SIZE,
        "normal_evaluation_size": NORMAL_EVALUATION_SIZE,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "detectors": DETECTOR_MODES,
        "race_detector_version": "AlignedRACEA0Detector",
        "race_protocol_version": "race-a0-v2-principal-vector-soft-weights",
        "race_hyperparameters": {
            "k_max": K_MAX,
            "beta": BETA,
            "lambda_weight": LAMBDA_WEIGHT,
            "direction_min_cos2": DIRECTION_MIN_COS2,
            "global_alignment_min": GLOBAL_ALIGNMENT_MIN,
        },
        "source_regime_construction": {
            "method": "healthy-only robust distance from source episodes to target commissioning center",
            "regimes": ["near", "moderate", "high"],
            "source_subset_size": source_subset_size,
            "no_anomaly_labels_used": True,
        },
        "transferability_metrics": list(TRANSFERABILITY_COLUMNS),
    }


def _assert_output_dir_compatible(output_dir: Path, requested: dict[str, object]) -> None:
    manifest_path = output_dir / "m3_manifest.json"
    existing_outputs = list(output_dir.glob("m3_*"))
    if not existing_outputs:
        return
    if not manifest_path.exists():
        raise RuntimeError(
            f"Output directory already contains M3 files but no manifest: {output_dir}. "
            "Use a fresh --output-dir or remove the stale files explicitly."
        )
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = [
        key
        for key, value in requested.items()
        if existing.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Output directory contains an incompatible M3 run: {output_dir}. "
            f"Mismatched manifest fields: {', '.join(mismatches)}. "
            "Use a fresh --output-dir to avoid mixing runs."
        )


def _dataset_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _feature_lookup(cycles) -> dict[int, np.ndarray]:
    batch = extract_feature_batch(cycles)
    return {int(eid): batch.features[i] for i, eid in enumerate(batch.episode_ids)}


def _matrix(cycles, features_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([features_by_episode[int(cycle.episode_id)] for cycle in cycles])


def _matrix_from_ids(ids, features_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([features_by_episode[int(eid)] for eid in ids])


def _build_detector(mode: str, seed: int) -> AlignedRACEA0Detector:
    return AlignedRACEA0Detector(
        k_max=K_MAX,
        beta=BETA,
        lambda_weight=LAMBDA_WEIGHT,
        direction_min_cos2=DIRECTION_MIN_COS2,
        global_alignment_min=GLOBAL_ALIGNMENT_MIN,
        mode=mode,  # type: ignore[arg-type]
        false_alert_budget=FALSE_ALERT_BUDGET,
        random_state=10_000 + seed,
    )


def _ranking_metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float]:
    labels = np.r_[
        np.zeros(len(normal_scores), dtype=np.int64),
        np.ones(len(anomaly_scores), dtype=np.int64),
    ]
    scores = np.r_[normal_scores, anomaly_scores]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _score_finiteness(*arrays: np.ndarray) -> bool:
    return all(bool(np.isfinite(values).all()) for values in arrays)


def _score_summary(prefix: str, scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        f"{prefix}_score_min": float(np.min(values)),
        f"{prefix}_score_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_score_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_score_max": float(np.max(values)),
        f"{prefix}_score_mean": float(np.mean(values)),
        f"{prefix}_score_std": float(np.std(values, ddof=1 if len(values) > 1 else 0)),
    }


def _rank_hash(scores: np.ndarray) -> str:
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="mergesort").astype(np.int64)
    return hashlib.sha256(order.tobytes()).hexdigest()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _kendall_tau_and_order_changes(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    n = len(x)
    if n < 2:
        return float("nan"), 0
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        product = dx * dy
        concordant += int(np.sum(product > 0.0))
        discordant += int(np.sum(product < 0.0))
    denom = n * (n - 1) / 2
    return float((concordant - discordant) / denom), discordant


def _score_equivalence_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Score vectors must have matching shapes.")
    finite_ratio = np.divide(y, x, out=np.full_like(y, np.nan), where=np.abs(x) > 1e-12)
    finite_ratio = finite_ratio[np.isfinite(finite_ratio)]
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    scalar = float(np.dot(x, y) / max(float(np.dot(x, x)), 1e-12))
    scalar_residual = y - scalar * x
    rank_x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    rank_y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    kendall, order_changes = _kendall_tau_and_order_changes(x, y)
    return {
        "pearson": _safe_corr(x, y),
        "spearman": _safe_corr(rank_x, rank_y),
        "kendall_tau": kendall,
        "pairwise_order_changes": order_changes,
        "ratio_median": float(np.median(finite_ratio)) if finite_ratio.size else float("nan"),
        "ratio_mean": float(np.mean(finite_ratio)) if finite_ratio.size else float("nan"),
        "ratio_std": float(np.std(finite_ratio, ddof=1)) if finite_ratio.size > 1 else 0.0,
        "best_scalar": scalar,
        "scalar_max_abs_residual": float(np.max(np.abs(scalar_residual))),
        "scalar_median_abs_residual": float(np.median(np.abs(scalar_residual))),
        "affine_slope": float(slope),
        "affine_intercept": float(intercept),
        "affine_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "affine_max_abs_residual": float(np.max(np.abs(residual))),
        "affine_median_abs_residual": float(np.median(np.abs(residual))),
    }


def _score_equivalence_rows(
    *,
    source_pair_id: str,
    source_group: str,
    target_group: str,
    commissioning_size: int,
    seed: int,
    detector_payloads: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("TargetOnly", "RACE"),
        ("TargetOnly", "SourcePermutation"),
        ("TargetOnly", "WeightPermutation"),
        ("RACE", "SourcePermutation"),
        ("RACE", "WeightPermutation"),
    ]
    for reference_name, candidate_name in comparisons:
        if reference_name not in detector_payloads or candidate_name not in detector_payloads:
            continue
        reference = detector_payloads[reference_name]
        candidate = detector_payloads[candidate_name]
        for split_name in ("calibration", "healthy_eval", "anomaly_eval", "eval"):
            row: dict[str, object] = {
                "source_pair_id": source_pair_id,
                "source_group": source_group,
                "target_group": target_group,
                "commissioning_size": commissioning_size,
                "seed": seed,
                "reference_detector": reference_name,
                "candidate_detector": candidate_name,
                "score_split": split_name,
                "n_scores": int(len(reference[split_name])),  # type: ignore[arg-type]
                "reference_threshold": float(reference["threshold"]),
                "candidate_threshold": float(candidate["threshold"]),
                "threshold_ratio": float(candidate["threshold"]) / float(reference["threshold"]),
            }
            row.update(
                _score_equivalence_stats(
                    np.asarray(reference[split_name], dtype=np.float64),
                    np.asarray(candidate[split_name], dtype=np.float64),
                )
            )
            rows.append(row)
    return rows


def _prediction_equivalence_rows(
    *,
    source_pair_id: str,
    source_group: str,
    target_group: str,
    commissioning_size: int,
    seed: int,
    healthy_eval_ids: tuple[int, ...],
    anomaly_eval_ids: tuple[int, ...],
    detector_payloads: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("TargetOnly", "RACE"),
        ("TargetOnly", "SourcePermutation"),
        ("TargetOnly", "WeightPermutation"),
        ("RACE", "SourcePermutation"),
        ("RACE", "WeightPermutation"),
    ]
    eval_ids = np.asarray(healthy_eval_ids + anomaly_eval_ids, dtype=np.int64)
    labels = np.r_[
        np.zeros(len(healthy_eval_ids), dtype=np.int64),
        np.ones(len(anomaly_eval_ids), dtype=np.int64),
    ]
    for reference_name, candidate_name in comparisons:
        if reference_name not in detector_payloads or candidate_name not in detector_payloads:
            continue
        reference = detector_payloads[reference_name]
        candidate = detector_payloads[candidate_name]
        reference_pred = np.asarray(reference["eval"], dtype=np.float64) > float(reference["threshold"])
        candidate_pred = np.asarray(candidate["eval"], dtype=np.float64) > float(candidate["threshold"])
        changed = reference_pred != candidate_pred
        changed_healthy = changed & (labels == 0)
        changed_anomaly = changed & (labels == 1)
        reference_fp = eval_ids[(labels == 0) & reference_pred]
        candidate_fp = eval_ids[(labels == 0) & candidate_pred]
        reference_fn = eval_ids[(labels == 1) & ~reference_pred]
        candidate_fn = eval_ids[(labels == 1) & ~candidate_pred]
        rows.append(
            {
                "source_pair_id": source_pair_id,
                "source_group": source_group,
                "target_group": target_group,
                "commissioning_size": commissioning_size,
                "seed": seed,
                "reference_detector": reference_name,
                "candidate_detector": candidate_name,
                "n_eval": int(len(eval_ids)),
                "n_changed_predictions": int(np.sum(changed)),
                "n_changed_healthy_predictions": int(np.sum(changed_healthy)),
                "n_changed_anomaly_predictions": int(np.sum(changed_anomaly)),
                "reference_false_positives": int(len(reference_fp)),
                "candidate_false_positives": int(len(candidate_fp)),
                "reference_false_negatives": int(len(reference_fn)),
                "candidate_false_negatives": int(len(candidate_fn)),
                "changed_episode_ids": ";".join(map(str, eval_ids[changed])),
                "changed_healthy_episode_ids": ";".join(map(str, eval_ids[changed_healthy])),
                "changed_anomaly_episode_ids": ";".join(map(str, eval_ids[changed_anomaly])),
                "reference_false_positive_episode_ids": ";".join(map(str, reference_fp)),
                "candidate_false_positive_episode_ids": ";".join(map(str, candidate_fp)),
                "reference_false_negative_episode_ids": ";".join(map(str, reference_fn)),
                "candidate_false_negative_episode_ids": ";".join(map(str, candidate_fn)),
                "decision_equivalent": bool(not np.any(changed)),
            }
        )
    return rows


def evaluate_detector(
    *,
    detector_name: str,
    mode: str,
    source_features: np.ndarray,
    target_features: np.ndarray,
    calibration_features: np.ndarray,
    healthy_eval_features: np.ndarray,
    anomaly_eval_features: np.ndarray,
    commissioning_size: int,
    seed: int,
    source_pair_id: str,
    source_group: str,
    target_group: str,
    transfer_metrics: dict[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    model = _build_detector(mode, seed)
    model.fit(source_features, target_features)
    calibration_scores = model.score_samples(calibration_features)
    model.calibrate_from_scores(calibration_scores)
    if model.threshold_ is None or not np.isfinite(model.threshold_):
        raise RuntimeError(f"{detector_name} produced a non-finite threshold.")
    healthy_scores = model.score_samples(healthy_eval_features)
    anomaly_scores = model.score_samples(anomaly_eval_features)
    if not _score_finiteness(calibration_scores, healthy_scores, anomaly_scores):
        raise RuntimeError(f"{detector_name} produced non-finite scores.")

    recall = float(np.mean(anomaly_scores > model.threshold_))
    fpr = float(np.mean(healthy_scores > model.threshold_))
    ranking = _ranking_metrics(healthy_scores, anomaly_scores)
    diag = model.diagnostics_
    if diag is None:
        raise RuntimeError(f"{detector_name} diagnostics are missing.")

    row: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "source_pair_id": source_pair_id,
        "source_group": source_group,
        "target_group": target_group,
        "commissioning_size": commissioning_size,
        "seed": seed,
        "detector": detector_name,
        "recall": recall,
        "fpr": fpr,
        "auroc": ranking["auroc"],
        "auprc": ranking["auprc"],
        "success": float(recall >= RECALL_TARGET and fpr <= FALSE_ALERT_BUDGET),
        "threshold": float(model.threshold_),
        "calibration_size": int(len(calibration_scores)),
        "calibration_exceedance_count": int(np.sum(calibration_scores > model.threshold_)),
        "healthy_eval_alert_count": int(np.sum(healthy_scores > model.threshold_)),
        "anomaly_eval_alert_count": int(np.sum(anomaly_scores > model.threshold_)),
        "commissioning_actual_size": int(len(target_features)),
        "n_source": int(len(source_features)),
        "n_features": int(source_features.shape[1]),
        "k_effective": int(diag.k_effective),
        "transfer_weight": float(np.mean(diag.effective_weights)) if diag.effective_weights else np.nan,
        "transfer_weight_mass": float(np.sum(diag.effective_weights)),
        "alignment_mean_cos2": float(diag.alignment_mean_cos2),
        "n_shared_directions": int(diag.n_shared_directions),
        "fallback": bool(diag.fallback),
        "fallback_reason": diag.fallback_reason,
        "score_finite": True,
        "threshold_finite": True,
        "calibration_rank_hash": _rank_hash(calibration_scores),
        "healthy_eval_rank_hash": _rank_hash(healthy_scores),
        "anomaly_eval_rank_hash": _rank_hash(anomaly_scores),
        "eval_rank_hash": _rank_hash(np.r_[healthy_scores, anomaly_scores]),
        "covariance_conditioning_note": "robust scale floors and variance floors active in AlignedRACEA0Detector",
    }
    row.update(_score_summary("calibration", calibration_scores))
    row.update(_score_summary("healthy_eval", healthy_scores))
    row.update(_score_summary("anomaly_eval", anomaly_scores))
    row.update({key: float(transfer_metrics[key]) for key in TRANSFERABILITY_COLUMNS})
    payload: dict[str, object] = {
        "threshold": float(model.threshold_),
        "calibration": calibration_scores,
        "healthy_eval": healthy_scores,
        "anomaly_eval": anomaly_scores,
        "eval": np.r_[healthy_scores, anomaly_scores],
    }
    return row, payload


def _partition_row(
    *,
    source_pair_id: str,
    source_group: str,
    target_group: str,
    commissioning_size: int,
    seed: int,
    source_ids: tuple[int, ...],
    commissioning_ids: tuple[int, ...],
    calibration_ids: tuple[int, ...],
    healthy_eval_ids: tuple[int, ...],
    anomaly_eval_ids: tuple[int, ...],
) -> dict[str, object]:
    groups = {
        "source": source_ids,
        "target_commissioning": commissioning_ids,
        "target_calibration": calibration_ids,
        "healthy_evaluation": healthy_eval_ids,
        "anomaly_evaluation": anomaly_eval_ids,
    }
    assert_no_episode_leakage(groups)
    return {
        "source_pair_id": source_pair_id,
        "source_group": source_group,
        "target_group": target_group,
        "commissioning_size": commissioning_size,
        "seed": seed,
        "source_count": len(source_ids),
        "commissioning_count": len(commissioning_ids),
        "calibration_count": len(calibration_ids),
        "healthy_eval_count": len(healthy_eval_ids),
        "anomaly_eval_count": len(anomaly_eval_ids),
        "source_episode_ids": ";".join(map(str, source_ids)),
        "commissioning_episode_ids": ";".join(map(str, commissioning_ids)),
        "calibration_episode_ids": ";".join(map(str, calibration_ids)),
        "healthy_eval_episode_ids": ";".join(map(str, healthy_eval_ids)),
        "anomaly_eval_episode_ids": ";".join(map(str, anomaly_eval_ids)),
        "no_overlap": True,
    }


def _write_figures(results: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(name: str) -> None:
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            plt.savefig(path, bbox_inches="tight", dpi=180)
            written.append(str(path))
        plt.close()

    plt.figure(figsize=(6.5, 4.3))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.scatter(deltas["standardized_mean_shift"], deltas["delta_recall"], alpha=0.65, s=22)
    plt.xlabel("Healthy standardized mean shift")
    plt.ylabel("Recall(RACE) - Recall(TargetOnly)")
    save("m3_transferability_vs_delta_recall")

    plt.figure(figsize=(6.5, 4.3))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.scatter(deltas["standardized_mean_shift"], deltas["delta_success"], alpha=0.65, s=22)
    plt.xlabel("Healthy standardized mean shift")
    plt.ylabel("Success(RACE) - Success(TargetOnly)")
    save("m3_transferability_vs_delta_success")

    plt.figure(figsize=(6.5, 4.3))
    race = results[results["detector"] == "RACE"]
    plt.scatter(race["standardized_mean_shift"], race["transfer_weight"], alpha=0.65, s=22)
    plt.xlabel("Healthy standardized mean shift")
    plt.ylabel("Mean source transfer weight")
    save("m3_source_weight_vs_transferability")

    plt.figure(figsize=(6.5, 4.3))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.scatter(deltas["transfer_weight"], deltas["delta_recall"], alpha=0.65, s=22)
    plt.xlabel("Mean source transfer weight")
    plt.ylabel("Recall(RACE) - Recall(TargetOnly)")
    save("m3_source_weight_vs_delta_recall")

    curve = (
        results.groupby(["detector", "transferability_regime", "commissioning_size"], as_index=False)
        .agg(recall=("recall", "mean"))
    )
    plt.figure(figsize=(7.2, 4.8))
    for (detector, regime), group in curve.groupby(["detector", "transferability_regime"]):
        if detector not in {"TargetOnly", "RACE"}:
            continue
        group = group.sort_values("commissioning_size")
        plt.plot(
            group["commissioning_size"],
            group["recall"],
            marker="o",
            label=f"{detector} {regime}",
        )
    plt.xlabel("Commissioning healthy cycles")
    plt.ylabel("Mean recall")
    plt.legend(fontsize=8, ncol=2)
    save("m3_commissioning_curves_by_regime")

    plt.figure(figsize=(6.5, 4.3))
    data = [g["delta_recall"].to_numpy() for _, g in deltas.groupby("commissioning_size")]
    labels = [str(k) for k in sorted(deltas["commissioning_size"].unique())]
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("Commissioning healthy cycles")
    plt.ylabel("Recall(RACE) - Recall(TargetOnly)")
    save("m3_paired_recall_differences_by_N")

    plt.figure(figsize=(6.5, 4.3))
    data = [g["delta_fpr"].to_numpy() for _, g in deltas.groupby("commissioning_size")]
    labels = [str(k) for k in sorted(deltas["commissioning_size"].unique())]
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xlabel("Commissioning healthy cycles")
    plt.ylabel("FPR(RACE) - FPR(TargetOnly)")
    save("m3_fpr_differences_by_N")

    return written


def _robustness(results: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"check": "all_runs", "n": len(deltas), "mean_delta_recall": float(deltas["delta_recall"].mean())})
    metric = "standardized_mean_shift"
    lo, hi = deltas[metric].quantile([0.05, 0.95])
    trimmed = deltas[(deltas[metric] >= lo) & (deltas[metric] <= hi)]
    rows.append({"check": "exclude_extreme_5pct_pairs", "n": len(trimmed), "mean_delta_recall": float(trimmed["delta_recall"].mean())})
    pair_level = deltas.groupby(["source_pair_id", "commissioning_size"], as_index=False).mean(numeric_only=True)
    rows.append({"check": "pair_level_aggregation", "n": len(pair_level), "mean_delta_recall": float(pair_level["delta_recall"].mean())})
    seed_means = []
    for seed in sorted(deltas["seed"].unique()):
        leave_one = deltas[deltas["seed"] != seed]
        seed_means.append(float(leave_one["delta_recall"].mean()))
    rows.append({"check": "leave_one_seed_min", "n": len(seed_means), "mean_delta_recall": float(np.min(seed_means))})
    rows.append({"check": "leave_one_seed_max", "n": len(seed_means), "mean_delta_recall": float(np.max(seed_means))})
    rows.append({"check": "score_finiteness", "n": len(results), "mean_delta_recall": float(results["score_finite"].mean())})
    rows.append({"check": "threshold_finiteness", "n": len(results), "mean_delta_recall": float(results["threshold_finite"].mean())})
    return pd.DataFrame(rows)


def run_m3(
    *,
    data_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
    n_values: tuple[int, ...] = COMMISSIONING_GRID,
    seeds: tuple[int, ...] = SEEDS,
    source_subset_size: int = SOURCE_SUBSET_SIZE,
) -> dict[str, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = _requested_run_config(
        data_path=data_path,
        n_values=n_values,
        seeds=seeds,
        source_subset_size=source_subset_size,
    )
    _assert_output_dir_compatible(output_dir, run_config)

    print("Loading voraus-AD cycles...")
    cycles = load_cycles(path=data_path, signal_set="measured")
    print(f"Loaded cycles: {len(cycles)}")
    print("Extracting episode features once...")
    features_by_episode = _feature_lookup(cycles)

    rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    score_equivalence_rows: list[dict[str, object]] = []
    prediction_equivalence_rows: list[dict[str, object]] = []
    total = len(n_values) * len(seeds) * 3 * len(DETECTOR_MODES)
    counter = 0
    for n in n_values:
        for seed in seeds:
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n,
                commissioning_seed=seed,
                evaluation_seed=FROZEN_EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )
            split.verify_no_overlap()
            source_ids_all = episode_ids(split.source_train)
            commissioning_ids = episode_ids(split.target_commissioning)
            calibration_ids = episode_ids(split.target_calibration)
            healthy_eval_ids = episode_ids(split.target_normal_evaluation)
            anomaly_eval_ids = episode_ids(split.target_anomaly_evaluation)
            source_all = _matrix(split.source_train, features_by_episode)
            target = _matrix(split.target_commissioning, features_by_episode)
            calibration = _matrix(split.target_calibration, features_by_episode)
            healthy_eval = _matrix(split.target_normal_evaluation, features_by_episode)
            anomaly_eval = _matrix(split.target_anomaly_evaluation, features_by_episode)
            regimes = construct_source_regimes(
                source_episode_ids=source_ids_all,
                source_features=source_all,
                target_episode_ids=commissioning_ids,
                target_features=target,
                commissioning_size=n,
                seed=seed,
                subset_size=source_subset_size,
            )
            for regime in regimes:
                source = _matrix_from_ids(regime.source_episode_ids, features_by_episode)
                partition_rows.append(
                    _partition_row(
                        source_pair_id=regime.source_pair_id,
                        source_group=regime.source_group,
                        target_group=regime.target_group,
                        commissioning_size=n,
                        seed=seed,
                        source_ids=regime.source_episode_ids,
                        commissioning_ids=commissioning_ids,
                        calibration_ids=calibration_ids,
                        healthy_eval_ids=healthy_eval_ids,
                        anomaly_eval_ids=anomaly_eval_ids,
                    )
                )
                detector_payloads: dict[str, dict[str, object]] = {}
                for detector_name, mode in DETECTOR_MODES.items():
                    counter += 1
                    print(
                        f"Processing pair {counter}/{total} | source_pair={regime.source_pair_id} "
                        f"| N={n} | seed={seed}/19 | detector={detector_name}"
                    )
                    row, payload = evaluate_detector(
                        detector_name=detector_name,
                        mode=mode,
                        source_features=source,
                        target_features=target,
                        calibration_features=calibration,
                        healthy_eval_features=healthy_eval,
                        anomaly_eval_features=anomaly_eval,
                        commissioning_size=n,
                        seed=seed,
                        source_pair_id=regime.source_pair_id,
                        source_group=regime.source_group,
                        target_group=regime.target_group,
                        transfer_metrics=regime.metrics,
                    )
                    rows.append(row)
                    detector_payloads[detector_name] = payload
                score_equivalence_rows.extend(
                    _score_equivalence_rows(
                        source_pair_id=regime.source_pair_id,
                        source_group=regime.source_group,
                        target_group=regime.target_group,
                        commissioning_size=n,
                        seed=seed,
                        detector_payloads=detector_payloads,
                    )
                )
                prediction_equivalence_rows.extend(
                    _prediction_equivalence_rows(
                        source_pair_id=regime.source_pair_id,
                        source_group=regime.source_group,
                        target_group=regime.target_group,
                        commissioning_size=n,
                        seed=seed,
                        healthy_eval_ids=healthy_eval_ids,
                        anomaly_eval_ids=anomaly_eval_ids,
                        detector_payloads=detector_payloads,
                    )
                )

    results = add_transferability_regimes(pd.DataFrame(rows))
    deltas = add_transferability_regimes(paired_deltas(results))
    summary = detector_summary(results)
    paired = paired_summary(deltas)
    transfer = deltas.copy()
    correlations = correlation_table(deltas)
    weights = transfer_weight_diagnostics(results, deltas)
    robustness = _robustness(results, deltas)
    decision = scientific_decision(deltas)
    figure_paths = _write_figures(results, deltas, output_dir / "figures")

    paths = {
        "seed_results": output_dir / "m3_seed_results.csv",
        "summary": output_dir / "m3_summary.csv",
        "paired_summary": output_dir / "m3_paired_summary.csv",
        "paired_deltas": output_dir / "m3_paired_deltas.csv",
        "transferability_vs_benefit": output_dir / "m3_transferability_vs_benefit.csv",
        "transferability_correlations": output_dir / "m3_transferability_correlations.csv",
        "transfer_weight_diagnostics": output_dir / "m3_transfer_weight_diagnostics.csv",
        "score_equivalence_audit": output_dir / "m3_score_equivalence_audit.csv",
        "prediction_equivalence_audit": output_dir / "m3_prediction_equivalence_audit.csv",
        "partition_audit": output_dir / "m3_frozen_partition_audit.csv",
        "robustness": output_dir / "m3_robustness_checks.csv",
        "decision": output_dir / "m3_decision.json",
        "manifest": output_dir / "m3_manifest.json",
    }
    results.to_csv(paths["seed_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired_summary"], index=False)
    deltas.to_csv(paths["paired_deltas"], index=False)
    transfer.to_csv(paths["transferability_vs_benefit"], index=False)
    correlations.to_csv(paths["transferability_correlations"], index=False)
    weights.to_csv(paths["transfer_weight_diagnostics"], index=False)
    pd.DataFrame(score_equivalence_rows).to_csv(paths["score_equivalence_audit"], index=False)
    pd.DataFrame(prediction_equivalence_rows).to_csv(paths["prediction_equivalence_audit"], index=False)
    pd.DataFrame(partition_rows).to_csv(paths["partition_audit"], index=False)
    robustness.to_csv(paths["robustness"], index=False)
    paths["decision"].write_text(json.dumps(decision, indent=2), encoding="utf-8")

    manifest = {
        **run_config,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "full_frozen_protocol": tuple(n_values) == COMMISSIONING_GRID and tuple(seeds) == SEEDS,
        "outputs": {key: str(value) for key, value in paths.items()},
        "figures": figure_paths,
        "decision_file": str(paths["decision"]),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = _final_report(deltas, correlations, results, decision)
    print(report)
    return paths


def _final_report(
    deltas: pd.DataFrame,
    correlations: pd.DataFrame,
    results: pd.DataFrame,
    decision: dict[str, object],
) -> str:
    corr = correlations[
        (correlations["level"] == "pair_N")
        & (correlations["transferability_metric"] == "standardized_mean_shift")
        & (correlations["benefit_metric"] == "delta_recall")
    ]
    spearman = float(corr["spearman"].iloc[0]) if len(corr) else float("nan")
    by_regime = deltas.groupby("transferability_regime").agg(
        delta_recall=("delta_recall", "mean"),
        delta_fpr=("delta_fpr", "mean"),
    )
    perm = results[results["detector"].isin(["SourcePermutation", "WeightPermutation"])].groupby("detector")[
        "recall"
    ].mean()
    lines = [
        "",
        "M3 FINAL RESULT",
        "------------------------------",
        f"Real source-target pairs: {deltas['source_pair_id'].nunique()}",
        f"Seeds: {results['seed'].nunique()}",
        "N: " + ",".join(str(v) for v in sorted(results["commissioning_size"].unique())),
        "",
        f"Spearman(transferability, delta_recall): {spearman:.4f}",
    ]
    for regime, row in by_regime.iterrows():
        lines.append(
            f"Mean delta recall {regime}: {row['delta_recall']:.4f} | "
            f"mean delta FPR: {row['delta_fpr']:.4f}"
        )
    lines.extend(
        [
            "",
            "Source permutation control mean recall: "
            + (f"{perm.get('SourcePermutation', np.nan):.4f}"),
            "Weight permutation control mean recall: "
            + (f"{perm.get('WeightPermutation', np.nan):.4f}"),
            "",
            "Scientific decision:",
            str(decision["decision"]),
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n", type=int, nargs="+", default=list(COMMISSIONING_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--source-subset-size", type=int, default=SOURCE_SUBSET_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_m3(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_values=tuple(args.n),
        seeds=tuple(args.seeds),
        source_subset_size=args.source_subset_size,
    )


if __name__ == "__main__":
    main()
