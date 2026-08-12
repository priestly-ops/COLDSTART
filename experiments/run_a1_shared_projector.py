"""Run frozen A1 shared-projector diagnostics on the voraus protocol."""

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
from src.feature_extractor import extract_feature_batch
from src.shared_projector_a1 import SharedProjectorA1Detector
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


DEFAULT_VORAUS = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
N_VALUES = (10, 25)
SEEDS = (0, 1, 2, 3, 4)
GAMMA_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)
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


def _detectors(seed: int) -> dict[str, object]:
    common = {"k_max": K_MAX, "beta": BETA, "random_state": seed}
    detectors: dict[str, object] = {
        "TargetPCA": SharedProjectorA1Detector(gamma=0.0, **common),
        "RACE-A0": AlignedRACEA0Detector(mode="aligned", **common),
        "A1-SelectedGamma": SharedProjectorA1Detector(gamma=None, gamma_grid=GAMMA_GRID, **common),
    }
    for gamma in GAMMA_GRID:
        detectors[f"A1-gamma={gamma:.2f}"] = SharedProjectorA1Detector(gamma=gamma, **common)
    return detectors


def _evaluate(
    name: str,
    detector,
    *,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    normal: np.ndarray,
    anomaly: np.ndarray,
    n_target: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    detector.fit(source, target)
    calibration_scores = detector.score_samples(calibration)
    detector.calibrate_from_scores(calibration_scores)
    normal_scores = detector.score_samples(normal)
    anomaly_scores = detector.score_samples(anomaly)
    row = {
        "dataset": "voraus-ad",
        "domain": "voraus_protocol",
        "detector": name,
        "n_target": n_target,
        "seed": seed,
        **_metrics(normal_scores, anomaly_scores, calibration_scores, detector.threshold_),
    }
    diagnostics = getattr(detector, "diagnostics_", None)
    gamma_rows: list[dict[str, object]] = []
    projector_rows: list[dict[str, object]] = []
    if isinstance(detector, SharedProjectorA1Detector) and diagnostics is not None:
        diag = asdict(diagnostics)
        row.update(
            {
                "selected_gamma": diag["selected_gamma"],
                "k_effective": diag["k_effective"],
                "projector_alignment": diag["projector_alignment"],
                "target_projector_mass": diag["target_projector_mass"],
                "source_projector_mass": diag["source_projector_mass"],
            }
        )
        for gamma, risk in diagnostics.gamma_risks.items():
            gamma_rows.append(
                {
                    "dataset": "voraus-ad",
                    "domain": "voraus_protocol",
                    "detector": name,
                    "n_target": n_target,
                    "seed": seed,
                    "gamma": gamma,
                    "risk": risk,
                    "risk_delta_vs_0": diagnostics.gamma_risk_deltas_vs_0.get(gamma, np.nan),
                    "selected": gamma == diagnostics.selected_gamma,
                }
            )
    eigenvalues = getattr(detector, "projector_eigenvalues_", None)
    if eigenvalues is not None:
        for index, value in enumerate(eigenvalues):
            projector_rows.append(
                {
                    "dataset": "voraus-ad",
                    "domain": "voraus_protocol",
                    "detector": name,
                    "n_target": n_target,
                    "seed": seed,
                    "mode_index": index,
                    "projector_eigenvalue": float(value),
                }
            )
    return row, gamma_rows, projector_rows


def run_a1(*, voraus_path: Path = DEFAULT_VORAUS, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    cycles = load_cycles(voraus_path)
    features_by_episode = _episode_feature_map(cycles)
    rows: list[dict[str, object]] = []
    gamma_rows: list[dict[str, object]] = []
    projector_rows: list[dict[str, object]] = []
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
                row, gamma_diag, projector_diag = _evaluate(
                    name,
                    detector,
                    source=source,
                    target=target,
                    calibration=calibration,
                    normal=normal,
                    anomaly=anomaly,
                    n_target=n_target,
                    seed=seed,
                )
                rows.append(row)
                gamma_rows.extend(gamma_diag)
                projector_rows.extend(projector_diag)
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
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["dataset", "domain", "n_target", "detector"], as_index=False)
        .agg(
            AUROC_mean=("AUROC", "mean"),
            AUPRC_mean=("AUPRC", "mean"),
            pauroc_1pct_mean=("pauroc_0_01_standardized", "mean"),
            oracle_recall_1pct_mean=("oracle_recall_1pct", "mean"),
            conformal_recall_mean=("conformal_recall", "mean"),
            conformal_fpr_mean=("conformal_fpr", "mean"),
            selected_gamma_mean=("selected_gamma", "mean"),
        )
    )
    baselines = ("TargetPCA", "RACE-A0", "A1-gamma=0.00")
    paired_rows: list[dict[str, object]] = []
    metrics = ("AUROC", "pauroc_0_01_standardized", "oracle_recall_1pct")
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
                for metric in metrics:
                    row[f"delta_{metric}"] = float(
                        by_detector.loc[detector, metric] - by_detector.loc[baseline, metric]
                    )
                paired_rows.append(row)
    transfer = pd.DataFrame(transfer_rows)
    target_pca = transfer[transfer["detector"] == "TargetPCA"].set_index(["n_target", "seed"])
    for metric in metrics:
        transfer[f"delta_{metric}_vs_TargetPCA"] = [
            float(value - target_pca.loc[(n, seed), metric])
            for value, n, seed in zip(transfer[metric], transfer["n_target"], transfer["seed"])
        ]

    paths = {
        "results": output_dir / "a1_seed_results.csv",
        "summary": output_dir / "a1_summary.csv",
        "gamma_diagnostics": output_dir / "a1_gamma_diagnostics.csv",
        "projector_spectrum": output_dir / "a1_projector_spectrum.csv",
        "paired_deltas": output_dir / "a1_paired_deltas.csv",
        "transferability_vs_benefit": output_dir / "a1_transferability_vs_benefit.csv",
        "manifest": output_dir / "a1_manifest.json",
    }
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(gamma_rows).to_csv(paths["gamma_diagnostics"], index=False)
    pd.DataFrame(projector_rows).to_csv(paths["projector_spectrum"], index=False)
    pd.DataFrame(paired_rows).to_csv(paths["paired_deltas"], index=False)
    transfer.to_csv(paths["transferability_vs_benefit"], index=False)
    manifest = {
        "configuration": {
            "datasets": ["voraus-ad"],
            "n_values": list(N_VALUES),
            "seeds": list(SEEDS),
            "detectors": list(_detectors(0).keys()),
            "gamma_grid": list(GAMMA_GRID),
            "gamma_selection": "healthy-only deterministic leave-one-out target reconstruction risk",
            "alpha": ALPHA,
            "k_max": K_MAX,
            "beta": BETA,
            "stability_bootstrap_resamples_for_transferability": BOOTSTRAP_RESAMPLES,
            "no_outcome_tuning": True,
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
    paths = run_a1(voraus_path=args.voraus_path, output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
