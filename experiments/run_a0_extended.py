"""Controlled five-seed RACE-A0 diagnostic pass with principled variants."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.a0_transferability import audit_pair
from src.aligned_race_a0 import AlignedRACEA0Detector
from src.aligned_race_a0_variants import (
    FeaturePermutedSourceRACEA0Detector,
    StabilityAwareRACEA0Detector,
    VarianceAwareRACEA0Detector,
)
from src.feature_extractor import extract_feature_batch
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


DEFAULT_VORAUS = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
N_VALUES = (10, 25)
SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.01
K_MAX = 16
BETA = 0.5
BOOTSTRAP_RESAMPLES = 50


def _episode_feature_map(cycles) -> dict[int, np.ndarray]:
    batch = extract_feature_batch(cycles)
    return {int(episode_id): batch.features[i] for i, episode_id in enumerate(batch.episode_ids)}


def _matrix_for(cycles, features_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([features_by_episode[int(cycle.episode_id)] for cycle in cycles])


def _oracle_threshold_at_fpr(normal_scores: np.ndarray, alpha: float) -> float:
    rank = int(np.ceil((1.0 - alpha) * len(normal_scores))) - 1
    rank = int(np.clip(rank, 0, len(normal_scores) - 1))
    return float(np.sort(normal_scores)[rank])


def _metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray, calibration_scores: np.ndarray, threshold: float):
    labels = np.r_[np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]
    scores = np.r_[normal_scores, anomaly_scores]
    oracle_threshold = _oracle_threshold_at_fpr(normal_scores, ALPHA)
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "pauroc_0_01_standardized": float(roc_auc_score(labels, scores, max_fpr=ALPHA)),
        "oracle_recall_1pct": float(np.mean(anomaly_scores > oracle_threshold)),
        "oracle_fpr_1pct": float(np.mean(normal_scores > oracle_threshold)),
        "conformal_recall": float(np.mean(anomaly_scores > threshold)),
        "conformal_fpr": float(np.mean(normal_scores > threshold)),
        "threshold": float(threshold),
        "threshold_rank": int(np.sum(calibration_scores <= threshold)),
    }


def _detectors(seed: int) -> dict[str, AlignedRACEA0Detector]:
    common = {"k_max": K_MAX, "beta": BETA, "random_state": seed}
    return {
        "A0TargetOnly": AlignedRACEA0Detector(mode="target_only", **common),
        "TargetPCA": AlignedRACEA0Detector(mode="target_pca", **common),
        "RACE-A0": AlignedRACEA0Detector(mode="aligned", **common),
        "WeightPermutedRACE": AlignedRACEA0Detector(mode="weight_permuted", **common),
        "FeaturePermutedSource": FeaturePermutedSourceRACEA0Detector(mode="aligned", **common),
        "A0-VarianceAware": VarianceAwareRACEA0Detector(mode="aligned", **common),
        "A0-Stability": StabilityAwareRACEA0Detector(
            mode="aligned",
            bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            **common,
        ),
    }


def _evaluate_detector(
    name: str,
    detector: AlignedRACEA0Detector,
    *,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    normal: np.ndarray,
    anomaly: np.ndarray,
    dataset: str,
    n_target: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    detector.fit(source, target)
    calibration_scores = detector.score_samples(calibration)
    detector.calibrate_from_scores(calibration_scores)
    normal_scores = detector.score_samples(normal)
    anomaly_scores = detector.score_samples(anomaly)
    metric_values = _metrics(normal_scores, anomaly_scores, calibration_scores, detector.threshold_)
    normal_components = detector.score_components(normal)
    anomaly_components = detector.score_components(anomaly)
    row = {
        "dataset": dataset,
        "domain": "voraus_protocol",
        "detector": name,
        "n_target": n_target,
        "seed": seed,
        **metric_values,
        "shared_score_mean": float(
            np.mean(np.r_[normal_components["shared_score"], anomaly_components["shared_score"]])
        ),
        "shared_score_std": float(
            np.std(np.r_[normal_components["shared_score"], anomaly_components["shared_score"]])
        ),
    }
    diagnostics = getattr(detector, "compatibility_diagnostics_", None)
    if diagnostics is not None:
        row.update({f"compat_{key}": value for key, value in asdict(diagnostics).items()})
    if hasattr(detector, "variance_tau_"):
        row["variance_tau"] = float(detector.variance_tau_)

    angle_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    singular_values = getattr(detector, "singular_values_", np.asarray([], dtype=float))
    raw_weights = getattr(detector, "raw_cos2_weights_", np.asarray([], dtype=float))
    effective_weights = getattr(detector, "effective_weights_", np.asarray([], dtype=float))
    for j, singular in enumerate(singular_values):
        cos2 = float(np.clip(singular, 0.0, 1.0) ** 2)
        angle_rows.append(
            {
                "dataset": dataset,
                "domain": "voraus_protocol",
                "detector": name,
                "n_target": n_target,
                "seed": seed,
                "mode_index": j,
                "singular_value": float(singular),
                "cos2": cos2,
                "principal_angle_degrees": float(
                    np.degrees(np.arccos(np.sqrt(np.clip(cos2, 0.0, 1.0))))
                ),
            }
        )
        weight_rows.append(
            {
                "dataset": dataset,
                "domain": "voraus_protocol",
                "detector": name,
                "n_target": n_target,
                "seed": seed,
                "mode_index": j,
                "raw_cos2_weight": float(raw_weights[j]) if j < len(raw_weights) else np.nan,
                "effective_weight": float(effective_weights[j]) if j < len(effective_weights) else np.nan,
            }
        )
    return row, angle_rows, weight_rows


def run_extended(*, voraus_path: Path = DEFAULT_VORAUS, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    cycles = load_cycles(voraus_path)
    features_by_episode = _episode_feature_map(cycles)
    result_rows: list[dict[str, object]] = []
    angle_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []

    total = len(N_VALUES) * len(SEEDS) * len(_detectors(0))
    counter = 0
    for n_target in N_VALUES:
        for seed in SEEDS:
            split = create_frozen_evaluation_split(cycles, n_target, seed)
            source = _matrix_for(split.source_train, features_by_episode)
            target = _matrix_for(split.target_commissioning, features_by_episode)
            calibration = _matrix_for(split.target_calibration, features_by_episode)
            normal = _matrix_for(split.target_normal_evaluation, features_by_episode)
            anomaly = _matrix_for(split.target_anomaly_evaluation, features_by_episode)
            audit, _ = audit_pair(
                source,
                target,
                dataset="voraus-ad",
                source_domain="source_train_protocol",
                target_domain=f"target_commissioning_N{n_target}_seed{seed}",
                n_target=n_target,
                seed=seed,
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
            )
            for name, detector in _detectors(seed).items():
                counter += 1
                print(f"[{counter:03d}/{total:03d}] N={n_target} seed={seed} detector={name}")
                row, angles, weights = _evaluate_detector(
                    name,
                    detector,
                    source=source,
                    target=target,
                    calibration=calibration,
                    normal=normal,
                    anomaly=anomaly,
                    dataset="voraus-ad",
                    n_target=n_target,
                    seed=seed,
                )
                result_rows.append(row)
                angle_rows.extend(angles)
                weight_rows.extend(weights)
                transfer_rows.append(
                    {
                        "dataset": "voraus-ad",
                        "domain": "voraus_protocol",
                        "n_target": n_target,
                        "seed": seed,
                        "detector": name,
                        "alignment_mean_cos2": audit.alignment_mean_cos2,
                        "target_subspace_stability": audit.bootstrap_alignment_mean,
                        "variance_agreement_mean": audit.variance_agreement_mean,
                        "AUROC": row["AUROC"],
                        "pauroc_0_01_standardized": row["pauroc_0_01_standardized"],
                        "oracle_recall_1pct": row["oracle_recall_1pct"],
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby(["dataset", "domain", "n_target", "detector"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUPRC_mean=("AUPRC", "mean"),
            pauroc_1pct_mean=("pauroc_0_01_standardized", "mean"),
            oracle_recall_1pct_mean=("oracle_recall_1pct", "mean"),
            conformal_recall_mean=("conformal_recall", "mean"),
            conformal_fpr_mean=("conformal_fpr", "mean"),
        )
    )
    baseline_metrics = ["AUROC", "pauroc_0_01_standardized", "oracle_recall_1pct"]
    paired_rows: list[dict[str, object]] = []
    baselines = ("TargetPCA", "RACE-A0", "WeightPermutedRACE", "FeaturePermutedSource")
    for (_, _, n_target, seed), group in results.groupby(["dataset", "domain", "n_target", "seed"]):
        by_detector = group.set_index("detector")
        for detector in group["detector"]:
            for baseline in baselines:
                if detector == baseline or baseline not in by_detector.index:
                    continue
                row = {
                    "dataset": "voraus-ad",
                    "domain": "voraus_protocol",
                    "n_target": n_target,
                    "seed": seed,
                    "detector": detector,
                    "baseline": baseline,
                }
                for metric in baseline_metrics:
                    row[f"delta_{metric}"] = float(
                        by_detector.loc[detector, metric] - by_detector.loc[baseline, metric]
                    )
                paired_rows.append(row)
    transfer = pd.DataFrame(transfer_rows)
    target_pca = transfer[transfer["detector"] == "TargetPCA"].set_index(["n_target", "seed"])
    for metric in baseline_metrics:
        transfer[f"delta_{metric}_vs_TargetPCA"] = [
            float(value - target_pca.loc[(n, seed), metric])
            for value, n, seed in zip(transfer[metric], transfer["n_target"], transfer["seed"])
        ]

    paths = {
        "results": output_dir / "a0_extended_seed_results.csv",
        "summary": output_dir / "a0_extended_summary.csv",
        "principal_angles": output_dir / "a0_extended_principal_angles.csv",
        "compatibility_weights": output_dir / "a0_extended_compatibility_weights.csv",
        "paired_deltas": output_dir / "a0_extended_paired_deltas.csv",
        "transferability_vs_benefit": output_dir / "a0_transferability_vs_benefit.csv",
        "manifest": output_dir / "a0_extended_manifest.json",
    }
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(angle_rows).to_csv(paths["principal_angles"], index=False)
    pd.DataFrame(weight_rows).to_csv(paths["compatibility_weights"], index=False)
    pd.DataFrame(paired_rows).to_csv(paths["paired_deltas"], index=False)
    transfer.to_csv(paths["transferability_vs_benefit"], index=False)
    manifest = {
        "configuration": {
            "datasets": ["voraus-ad"],
            "n_values": list(N_VALUES),
            "seeds": list(SEEDS),
            "detectors": list(_detectors(0).keys()),
            "alpha": ALPHA,
            "k_max": K_MAX,
            "beta": BETA,
            "variance_tau_grid": [0.5, 1.0, 2.0],
            "stability_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "selection_policy": "all variant hyperparameters selected with healthy-only source/target data",
            "a0_conditional": "skipped: current cycle-level feature matrix has no validated phase/context variable",
        },
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voraus-path", type=Path, default=DEFAULT_VORAUS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = run_extended(voraus_path=args.voraus_path, output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
