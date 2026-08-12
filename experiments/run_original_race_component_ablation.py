"""Run a frozen Original RACE component ablation.

This diagnostic isolates why historical/global RACE can outperform TargetOnly
in a source regime without using anomaly outcomes to tune any compatibility,
gate, threshold, or ablation definition.
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

from src.feature_extractor import extract_feature_batch
from src.m3_transfer_regimes import (
    TRANSFERABILITY_COLUMNS,
    assert_no_episode_leakage,
    construct_source_regimes,
    episode_ids,
)
from src.original_race_ablation import (
    FROZEN_ORIGINAL_RACE_ABLATIONS,
    OriginalRaceComponentDetector,
    covariance_condition,
    directional_original_race_audit,
    fit_source_target_gaussians,
)
from src.split_generator import create_frozen_evaluation_split
from src.srace import score_equivalence_stats
from src.voraus_loader import load_cycle_metadata, load_cycles


PROTOCOL_VERSION = "original-race-component-ablation-v1"
OUTPUT_SCHEMA_VERSION = "original-race-component-diagnostic-v1"
GLOBAL_SEED = 42
FROZEN_EVALUATION_SEED = 42
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "original_race_component_ablation_seed0"
DEFAULT_N = (10,)
DEFAULT_SEEDS = (0,)
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
SOURCE_SUBSET_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
LAMBDA_REG = 60.0


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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True).strip())
    except Exception:
        return True


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ["numpy", "pandas", "scikit-learn", "pyarrow"]:
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


def _split_episode_ids(split) -> tuple[int, ...]:
    ids = set()
    for cycles in [
        split.source_train,
        split.target_commissioning,
        split.target_calibration,
        split.target_normal_evaluation,
        split.target_anomaly_evaluation,
    ]:
        ids.update(episode_ids(cycles))
    return tuple(sorted(ids))


def _ranking_metrics(healthy_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float]:
    labels = np.r_[np.zeros(len(healthy_scores), dtype=np.int64), np.ones(len(anomaly_scores), dtype=np.int64)]
    scores = np.r_[healthy_scores, anomaly_scores]
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
    }


def _evaluate_variant(
    *,
    variant: str,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    healthy_eval: np.ndarray,
    anomaly_eval: np.ndarray,
    keys: dict[str, object],
    transfer_metrics: dict[str, float],
) -> tuple[dict[str, object], dict[str, object]]:
    model = OriginalRaceComponentDetector(
        variant=variant,
        lambda_reg=LAMBDA_REG,
        false_alert_budget=FALSE_ALERT_BUDGET,
    ).fit(source, target)
    calibration_scores = model.score_samples(calibration)
    model.calibrate_from_scores(calibration_scores)
    healthy_scores = model.score_samples(healthy_eval)
    anomaly_scores = model.score_samples(anomaly_eval)
    threshold = float(model.threshold_)
    recall = float(np.mean(anomaly_scores > threshold))
    fpr = float(np.mean(healthy_scores > threshold))
    ranking = _ranking_metrics(healthy_scores, anomaly_scores)
    min_eig, max_eig, condition = covariance_condition(model.covariance_)
    fit = model.fit_
    row = {
        **keys,
        "protocol_version": PROTOCOL_VERSION,
        "detector": variant,
        "source_size": int(len(source)),
        "commissioning_size": int(len(target)),
        "calibration_size": int(len(calibration)),
        "healthy_eval_size": int(len(healthy_eval)),
        "anomaly_eval_size": int(len(anomaly_eval)),
        "n_features": int(source.shape[1]),
        "lambda_reg": LAMBDA_REG,
        "target_weight": float(fit.target_weight),
        "source_weight": float(1.0 - fit.target_weight),
        "source_shrinkage": float(fit.source_shrinkage),
        "target_shrinkage": float(fit.target_shrinkage),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": condition,
        "threshold": threshold,
        "recall": recall,
        "FPR": fpr,
        "fpr": fpr,
        "AUROC": ranking["AUROC"],
        "AUPRC": ranking["AUPRC"],
        "auroc": ranking["AUROC"],
        "auprc": ranking["AUPRC"],
        "success": float(recall >= RECALL_TARGET and fpr <= FALSE_ALERT_BUDGET),
        "calibration_exceedance_count": int(np.sum(calibration_scores > threshold)),
        "healthy_eval_alert_count": int(np.sum(healthy_scores > threshold)),
        "anomaly_eval_alert_count": int(np.sum(anomaly_scores > threshold)),
    }
    for column in TRANSFERABILITY_COLUMNS:
        row[column] = float(transfer_metrics[column])
    payload = {
        "threshold": threshold,
        "calibration": calibration_scores,
        "healthy_eval": healthy_scores,
        "anomaly_eval": anomaly_scores,
        "eval": np.r_[healthy_scores, anomaly_scores],
        "pred_eval": np.r_[healthy_scores, anomaly_scores] > threshold,
    }
    return row, payload


def _score_equivalence_rows(keys: dict[str, object], payloads: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reference in ["TargetOnly", "OriginalRACE"]:
        for candidate in FROZEN_ORIGINAL_RACE_ABLATIONS:
            if reference == candidate:
                continue
            for split in ["calibration", "healthy_eval", "anomaly_eval", "eval"]:
                stats = score_equivalence_stats(payloads[reference][split], payloads[candidate][split])
                pred_ref = np.asarray(payloads[reference]["pred_eval"], dtype=bool)
                pred_cand = np.asarray(payloads[candidate]["pred_eval"], dtype=bool)
                row = {
                    **keys,
                    "reference_detector": reference,
                    "candidate_detector": candidate,
                    "score_split": split,
                    "n_scores": int(len(payloads[reference][split])),
                    "reference_threshold": float(payloads[reference]["threshold"]),
                    "candidate_threshold": float(payloads[candidate]["threshold"]),
                    "threshold_ratio": float(payloads[candidate]["threshold"])
                    / max(float(payloads[reference]["threshold"]), 1e-12),
                    "number_changed_predictions": int(np.sum(pred_ref != pred_cand)),
                }
                row.update(stats)
                rows.append(row)
    return rows


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_pair_id", "source_group", "target_group", "N", "seed"]
    rows: list[dict[str, object]] = []
    for reference in ["TargetOnly", "OriginalRACE"]:
        baseline = results[results["detector"] == reference].set_index(keys)
        for detector in FROZEN_ORIGINAL_RACE_ABLATIONS:
            if detector == reference:
                continue
            current = results[results["detector"] == detector].set_index(keys)
            for key in current.index.intersection(baseline.index):
                row = dict(zip(keys, key))
                row["reference_detector"] = reference
                row["candidate_detector"] = detector
                for metric in ["recall", "FPR", "AUROC", "AUPRC", "success", "threshold"]:
                    row[f"delta_{metric}"] = float(current.loc[key, metric] - baseline.loc[key, metric])
                rows.append(row)
    return pd.DataFrame(rows)


def _partition_row(keys: dict[str, object], groups: dict[str, tuple[int, ...]]) -> dict[str, object]:
    assert_no_episode_leakage(groups)
    row = {**keys, "no_overlap": True}
    for name, values in groups.items():
        row[f"{name}_count"] = len(values)
        row[f"{name}_episode_ids"] = ";".join(map(str, values))
    return row


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "original_race_manifest.json",
        "component_ablation": output_dir / "original_race_component_ablation.csv",
        "summary": output_dir / "original_race_summary.csv",
        "paired_deltas": output_dir / "original_race_paired_deltas.csv",
        "direction_audit": output_dir / "original_race_direction_audit.csv",
        "score_equivalence": output_dir / "original_race_score_equivalence.csv",
        "source_compatibility": output_dir / "original_race_source_compatibility.csv",
        "partition_audit": output_dir / "original_race_partition_audit.csv",
    }


def _write_current_outputs(
    *,
    paths: dict[str, Path],
    rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    partition_rows: list[dict[str, object]],
    direction_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
) -> None:
    results = pd.DataFrame(rows)
    if results.empty:
        return
    summary = results.groupby(["detector", "N"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        FPR_mean=("FPR", "mean"),
        AUROC_mean=("AUROC", "mean"),
        AUPRC_mean=("AUPRC", "mean"),
        success_rate=("success", "mean"),
        threshold_mean=("threshold", "mean"),
        condition_number_mean=("condition_number", "mean"),
    )
    results.to_csv(paths["component_ablation"], index=False)
    summary.to_csv(paths["summary"], index=False)
    _paired_deltas(results).to_csv(paths["paired_deltas"], index=False)
    pd.DataFrame(direction_rows).to_csv(paths["direction_audit"], index=False)
    pd.DataFrame(score_rows).to_csv(paths["score_equivalence"], index=False)
    pd.DataFrame(source_rows).to_csv(paths["source_compatibility"], index=False)
    pd.DataFrame(partition_rows).to_csv(paths["partition_audit"], index=False)


def run_original_race_component_ablation(
    data_path: Path,
    output_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)
    print("Loading VORAUS metadata for Original RACE component ablation...", flush=True)
    cycles = load_cycle_metadata(path=data_path)

    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

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
            selected_ids = _split_episode_ids(split)
            print(f"Loading signals/features for N={n}, seed={seed}: {len(selected_ids)} episodes...", flush=True)
            selected_cycles = load_cycles(path=data_path, signal_set="measured", episode_ids=selected_ids)
            features_by_episode = _feature_lookup(selected_cycles)
            source_all = _matrix(split.source_train, features_by_episode)
            target = _matrix(split.target_commissioning, features_by_episode)
            calibration = _matrix(split.target_calibration, features_by_episode)
            healthy_eval = _matrix(split.target_normal_evaluation, features_by_episode)
            anomaly_eval = _matrix(split.target_anomaly_evaluation, features_by_episode)
            regimes = construct_source_regimes(
                source_episode_ids=episode_ids(split.source_train),
                source_features=source_all,
                target_episode_ids=episode_ids(split.target_commissioning),
                target_features=target,
                commissioning_size=n,
                seed=seed,
                subset_size=SOURCE_SUBSET_SIZE,
            )
            for regime in regimes:
                source = _matrix_from_ids(regime.source_episode_ids, features_by_episode)
                keys = {
                    "source_pair_id": regime.source_pair_id,
                    "source_group": regime.source_group,
                    "target_group": regime.target_group,
                    "N": n,
                    "seed": seed,
                }
                source_rows.append({**keys, **{k: float(v) for k, v in regime.metrics.items()}, "source_size": len(regime.source_episode_ids)})
                partition_rows.append(
                    _partition_row(
                        keys,
                        {
                            "source": regime.source_episode_ids,
                            "commissioning": episode_ids(split.target_commissioning),
                            "calibration": episode_ids(split.target_calibration),
                            "healthy_eval": episode_ids(split.target_normal_evaluation),
                            "anomaly_eval": episode_ids(split.target_anomaly_evaluation),
                        },
                    )
                )
                fit = fit_source_target_gaussians(source, target, lambda_reg=LAMBDA_REG)
                for direction_row in directional_original_race_audit(
                    fit,
                    calibration=calibration,
                    healthy_eval=healthy_eval,
                    anomaly_eval=anomaly_eval,
                ):
                    direction_rows.append({**keys, **direction_row})

                payloads: dict[str, dict[str, object]] = {}
                for variant in FROZEN_ORIGINAL_RACE_ABLATIONS:
                    row, payload = _evaluate_variant(
                        variant=variant,
                        source=source,
                        target=target,
                        calibration=calibration,
                        healthy_eval=healthy_eval,
                        anomaly_eval=anomaly_eval,
                        keys=keys,
                        transfer_metrics=regime.metrics,
                    )
                    rows.append(row)
                    payloads[variant] = payload
                score_rows.extend(_score_equivalence_rows(keys, payloads))
                _write_current_outputs(
                    paths=paths,
                    rows=rows,
                    source_rows=source_rows,
                    partition_rows=partition_rows,
                    direction_rows=direction_rows,
                    score_rows=score_rows,
                )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "dataset_path": str(data_path),
        "dataset_hash_sha256": _dataset_hash(data_path),
        "global_seed": GLOBAL_SEED,
        "frozen_evaluation_seed": FROZEN_EVALUATION_SEED,
        "commissioning_grid": list(n_values),
        "seeds": list(seeds),
        "calibration_size": CALIBRATION_SIZE,
        "normal_evaluation_size": NORMAL_EVALUATION_SIZE,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "source_subset_size": SOURCE_SUBSET_SIZE,
        "lambda_reg": LAMBDA_REG,
        "frozen_ablation_definitions": list(FROZEN_ORIGINAL_RACE_ABLATIONS),
        "no_additional_anomaly_based_tuning": True,
        "anomaly_separation_policy": "Anomaly labels are used only after fit/calibration to report recall, AUROC/AUPRC, and posthoc_direction_separation columns.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote Original RACE component ablation outputs to {output_dir}", flush=True)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_original_race_component_ablation(args.data_path, args.output_dir, tuple(args.n), tuple(args.seeds))


if __name__ == "__main__":
    main()
