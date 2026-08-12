"""Run the literature-grounded S-RACE diagnostic experiment.

This is a small, predeclared validation run. It deliberately does not replace
or overwrite the frozen RACE/M3 outputs.
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

from src.detectors import RACEDetector, TargetOnlyDetector
from src.feature_extractor import extract_feature_batch
from src.m3_transfer_regimes import (
    TRANSFERABILITY_COLUMNS,
    assert_no_episode_leakage,
    construct_source_regimes,
    episode_ids,
)
from src.split_generator import create_frozen_evaluation_split
from src.srace import SelectiveRACEDetector, score_equivalence_stats
from src.voraus_loader import load_cycle_metadata, load_cycles


PROTOCOL_VERSION = "srace-selective-covariance-v1"
OUTPUT_SCHEMA_VERSION = "srace-diagnostic-v1"
GLOBAL_SEED = 42
FROZEN_EVALUATION_SEED = 42
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "srace_small_validation"
DEFAULT_N = (10, 25, 50)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
SOURCE_SUBSET_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90

SRACE_PARAMS = {
    "source_prior_strength": 20.0,
    "compatibility_log_tau": 1.0,
    "median_shift_tau": 2.0,
    "compatibility_floor": 0.05,
    "safe_gate_tolerance": 0.01,
    "min_eigenvalue": 1e-8,
}

LITERATURE_TO_DESIGN = [
    {
        "paper": "Kim et al., When Model Meets New Normals, AAAI 2024",
        "relevant_idea": "normal behavior shifts at test time; adapt to new normals",
        "why_it_matters": "robot commissioning target normals may be shifted from source normals",
        "adopt": "target healthy commissioning data anchors mean, covariance basis, calibration, and safe gate",
        "do_not_adopt": "online self-supervised representation updates; no target anomaly labels or streaming updates",
    },
    {
        "paper": "Sun and Saenko, Deep CORAL, ECCV 2016",
        "relevant_idea": "unsupervised domain adaptation can align second-order statistics",
        "why_it_matters": "RACE transfer is a covariance-structure question, not a global score-scale question",
        "adopt": "second-order compatibility via projected variance agreement",
        "do_not_adopt": "global covariance alignment transform that could overwrite target-specific structure",
    },
    {
        "paper": "Ledoit and Wolf, Journal of Multivariate Analysis 2004",
        "relevant_idea": "well-conditioned covariance shrinkage for high-dimensional small-sample regimes",
        "why_it_matters": "commissioning N is small relative to feature dimension",
        "adopt": "Ledoit-Wolf source and target covariance estimates plus eigenvalue floors",
        "do_not_adopt": "raw sample covariance inversion",
    },
    {
        "paper": "Context-aware Domain Adaptation for Time Series Anomaly Detection, SDM 2023",
        "relevant_idea": "misaligned context can cause negative transfer",
        "why_it_matters": "source directions should transfer only when healthy target behavior is compatible",
        "adopt": "healthy-only compatibility and source-regime diagnostics",
        "do_not_adopt": "source-label-driven DQN context sampler",
    },
    {
        "paper": "DACAD, Domain Adaptation Contrastive Learning for MTS Anomaly Detection, TKDE 2025",
        "relevant_idea": "align normal behavior while avoiding anomaly-class mismatch",
        "why_it_matters": "target anomalies are unavailable and must stay evaluation-only",
        "adopt": "normal-only source/target compatibility and anomaly-free adaptation",
        "do_not_adopt": "contrastive representation training or synthetic anomaly tuning",
    },
    {
        "paper": "COFT-AD, few-shot anomaly detection, CoRR 2024",
        "relevant_idea": "pretrained/source knowledge can help few-shot normal-only adaptation under covariate shift",
        "why_it_matters": "COLDSTART has many source normals and few target normals",
        "adopt": "few-shot healthy-only adaptation framing",
        "do_not_adopt": "deep fine-tuning with generated negatives",
    },
    {
        "paper": "Selective/attention transfer and negative-transfer work",
        "relevant_idea": "not all source features are transferable",
        "why_it_matters": "M3 controls showed source transfer can be operationally irrelevant",
        "adopt": "per-direction compatibility weights and compatibility permutation control",
        "do_not_adopt": "unconstrained attention learned from evaluation outcomes",
    },
    {
        "paper": "Split conformal anomaly detection literature",
        "relevant_idea": "finite-sample FPR control uses a calibration score quantile and is invariant to monotone score transforms",
        "why_it_matters": "M1/M3 showed low-alpha tail bottlenecks and score-rescaling cancellation",
        "adopt": "explicit score-equivalence diagnostics and final healthy calibration only",
        "do_not_adopt": "post-hoc threshold tuning or anomaly-aware calibration",
    },
]


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
        out = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def _requested_run_config(
    data_path: Path,
    output_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
) -> dict[str, object]:
    del output_dir
    return {
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
        "srace_hyperparameters": SRACE_PARAMS,
    }


def _assert_output_dir_compatible(output_dir: Path, requested: dict[str, object]) -> None:
    manifest_path = output_dir / "srace_manifest.json"
    existing = list(output_dir.glob("srace_*"))
    if not existing:
        return
    if not manifest_path.exists():
        raise RuntimeError(
            f"Output directory already contains S-RACE files but no manifest: {output_dir}. "
            "Use a fresh --output-dir to avoid mixing incompatible runs."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = [
        key
        for key, value in requested.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Output directory contains an incompatible S-RACE run: {output_dir}. "
            f"Mismatched manifest fields: {', '.join(mismatches)}. "
            "Use a fresh --output-dir."
        )


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
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _build_detector(name: str, seed: int):
    if name == "TargetOnly":
        return TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET)
    if name == "OriginalRACE":
        return RACEDetector(false_alert_budget=FALSE_ALERT_BUDGET)
    mode = {
        "S-RACE": "srace",
        "S-RACE WrongSource": "srace",
        "S-RACE SourcePermutation": "source_permutation",
        "S-RACE CompatibilityPermutation": "compatibility_permutation",
    }[name]
    return SelectiveRACEDetector(**SRACE_PARAMS, mode=mode, random_state=10_000 + seed, false_alert_budget=FALSE_ALERT_BUDGET)


def _evaluate_detector(
    *,
    name: str,
    source: np.ndarray,
    target: np.ndarray,
    calibration: np.ndarray,
    healthy_eval: np.ndarray,
    anomaly_eval: np.ndarray,
    n: int,
    seed: int,
    source_pair_id: str,
    source_group: str,
    target_group: str,
    transfer_metrics: dict[str, float],
    fit_source_pair_id: str | None = None,
    fit_source_group: str | None = None,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    model = _build_detector(name, seed)
    model.fit(source, target)
    calibration_scores = model.score_samples(calibration)
    model.calibrate_from_scores(calibration_scores)
    healthy_scores = model.score_samples(healthy_eval)
    anomaly_scores = model.score_samples(anomaly_eval)
    threshold = float(model.threshold_)
    recall = float(np.mean(anomaly_scores > threshold))
    fpr = float(np.mean(healthy_scores > threshold))
    ranking = _ranking_metrics(healthy_scores, anomaly_scores)
    diag = getattr(model, "diagnostics_", None)
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": name,
        "N": n,
        "seed": seed,
        "source_pair_id": source_pair_id,
        "source_group": source_group,
        "fit_source_pair_id": fit_source_pair_id or source_pair_id,
        "fit_source_group": fit_source_group or source_group,
        "target_group": target_group,
        "source_size": int(len(source)),
        "commissioning_size": int(len(target)),
        "calibration_size": int(len(calibration)),
        "healthy_eval_size": int(len(healthy_eval)),
        "anomaly_eval_size": int(len(anomaly_eval)),
        "n_features": int(source.shape[1]),
        "recall": recall,
        "FPR": fpr,
        "fpr": fpr,
        "AUROC": ranking["auroc"],
        "AUPRC": ranking["auprc"],
        "auroc": ranking["auroc"],
        "auprc": ranking["auprc"],
        "success": float(recall >= RECALL_TARGET and fpr <= FALSE_ALERT_BUDGET),
        "threshold": threshold,
        "calibration_exceedance_count": int(np.sum(calibration_scores > threshold)),
        "healthy_eval_alert_count": int(np.sum(healthy_scores > threshold)),
        "anomaly_eval_alert_count": int(np.sum(anomaly_scores > threshold)),
    }
    for column in TRANSFERABILITY_COLUMNS:
        row[column] = float(transfer_metrics[column])
    if diag is not None:
        row.update({key: value for key, value in diag.__dict__.items() if not isinstance(value, tuple)})
    else:
        row.update(
            {
                "transferred_dimensions": 0,
                "weight_mean": 0.0,
                "weight_median": 0.0,
                "weight_max": 0.0,
                "compatibility_mean": np.nan,
                "compatibility_median": np.nan,
                "compatibility_max": np.nan,
                "structural_compatibility_mean": np.nan,
                "structural_compatibility_median": np.nan,
                "structural_compatibility_max": np.nan,
                "active_structural_compatibility_mean": np.nan,
                "active_structural_compatibility_median": np.nan,
                "active_structural_compatibility_max": np.nan,
                "principal_cos2_mean": np.nan,
                "principal_cos2_median": np.nan,
                "principal_cos2_min": np.nan,
                "principal_cos2_max": np.nan,
                "pre_gate_weight_mean": np.nan,
                "pre_gate_weight_median": np.nan,
                "pre_gate_weight_max": np.nan,
                "pre_gate_compatibility_mean": np.nan,
                "pre_gate_compatibility_median": np.nan,
                "pre_gate_compatibility_max": np.nan,
                "variance_compatibility_mean": np.nan,
                "variance_compatibility_median": np.nan,
                "variance_compatibility_max": np.nan,
                "location_compatibility_mean": np.nan,
                "location_compatibility_median": np.nan,
                "location_compatibility_max": np.nan,
                "target_uncertainty": np.nan,
                "shared_rank": np.nan,
                "private_dimensions": np.nan,
                "condition_number": np.nan,
                "effective_rank": np.nan,
                "min_eigenvalue": np.nan,
                "max_eigenvalue": np.nan,
                "source_shrinkage": np.nan,
                "target_shrinkage": np.nan,
                "safe_gate_open": np.nan,
                "safe_gate_margin": np.nan,
                "fallback": np.nan,
                "fallback_reason": "",
            }
        )
    payload = {
        "threshold": threshold,
        "calibration": calibration_scores,
        "healthy_eval": healthy_scores,
        "anomaly_eval": anomaly_scores,
        "eval": np.r_[healthy_scores, anomaly_scores],
        "pred_eval": np.r_[healthy_scores, anomaly_scores] > threshold,
    }
    weights = pd.DataFrame()
    if isinstance(model, SelectiveRACEDetector) and model.transfer_weights_ is not None:
        weights = pd.DataFrame(
            {
                "source_pair_id": source_pair_id,
                "source_group": source_group,
                "fit_source_pair_id": fit_source_pair_id or source_pair_id,
                "fit_source_group": fit_source_group or source_group,
                "target_group": target_group,
                "N": n,
                "seed": seed,
                "detector": name,
                "direction": np.arange(len(model.transfer_weights_)),
                "transfer_weight": model.transfer_weights_,
                "pre_gate_transfer_weight": model.pre_gate_transfer_weights_,
                "structural_compatibility": model.structural_compatibility_,
                "variance_compatibility": model.variance_compatibility_,
                "location_compatibility": model.location_compatibility_,
                "pre_gate_compatibility": model.pre_gate_compatibility_,
                "compatibility": model.compatibility_,
                "target_variance": model.target_variance_,
                "source_projected_variance": model.source_projected_variance_,
                "adapted_variance": model.adapted_variance_,
            }
        )
    return row, payload, weights


def _score_equivalence_rows(keys: dict[str, object], payloads: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("TargetOnly", "OriginalRACE"),
        ("TargetOnly", "S-RACE"),
        ("TargetOnly", "S-RACE WrongSource"),
        ("TargetOnly", "S-RACE SourcePermutation"),
        ("TargetOnly", "S-RACE CompatibilityPermutation"),
        ("S-RACE", "S-RACE WrongSource"),
        ("S-RACE", "S-RACE SourcePermutation"),
        ("S-RACE", "S-RACE CompatibilityPermutation"),
    ]
    for reference, candidate in comparisons:
        if reference not in payloads or candidate not in payloads:
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
                "threshold_ratio": float(payloads[candidate]["threshold"]) / max(float(payloads[reference]["threshold"]), 1e-12),
                "number_changed_predictions": int(np.sum(pred_ref != pred_cand)),
            }
            row.update(stats)
            rows.append(row)
    return rows


def _partition_row(keys: dict[str, object], groups: dict[str, tuple[int, ...]]) -> dict[str, object]:
    assert_no_episode_leakage(groups)
    row = {**keys, "no_overlap": True}
    for name, values in groups.items():
        row[f"{name}_count"] = len(values)
        row[f"{name}_episode_ids"] = ";".join(map(str, values))
    return row


def _per_class_rows(keys: dict[str, object], anomaly_cycles, payloads: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    categories = np.asarray([int(c.category) for c in anomaly_cycles], dtype=np.int64)
    rows: list[dict[str, object]] = []
    for detector, payload in payloads.items():
        anomaly_scores = np.asarray(payload["anomaly_eval"], dtype=np.float64)
        threshold = float(payload["threshold"])
        for category in sorted(np.unique(categories)):
            mask = categories == category
            rows.append(
                {
                    **keys,
                    "detector": detector,
                    "category": int(category),
                    "anomaly_count": int(np.sum(mask)),
                    "recall": float(np.mean(anomaly_scores[mask] > threshold)),
                }
            )
    return rows


def _score_scatter_rows(
    keys: dict[str, object],
    eval_ids: tuple[int, ...],
    labels: np.ndarray,
    payloads: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("TargetOnly", "S-RACE"),
        ("S-RACE", "S-RACE WrongSource"),
        ("S-RACE", "S-RACE SourcePermutation"),
    ]
    for reference, candidate in comparisons:
        if reference not in payloads or candidate not in payloads:
            continue
        reference_scores = np.asarray(payloads[reference]["eval"], dtype=np.float64)
        candidate_scores = np.asarray(payloads[candidate]["eval"], dtype=np.float64)
        for episode_id, label, reference_score, candidate_score in zip(
            eval_ids,
            labels,
            reference_scores,
            candidate_scores,
        ):
            rows.append(
                {
                    **keys,
                    "reference_detector": reference,
                    "candidate_detector": candidate,
                    "episode_id": int(episode_id),
                    "label": int(label),
                    "reference_score": float(reference_score),
                    "candidate_score": float(candidate_score),
                    "reference_threshold": float(payloads[reference]["threshold"]),
                    "candidate_threshold": float(payloads[candidate]["threshold"]),
                }
            )
    return rows


def _paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_pair_id", "source_group", "target_group", "N", "seed"]
    target = results[results["detector"] == "TargetOnly"].set_index(keys)
    rows: list[dict[str, object]] = []
    for detector in [
        "OriginalRACE",
        "S-RACE",
        "S-RACE WrongSource",
        "S-RACE SourcePermutation",
        "S-RACE CompatibilityPermutation",
    ]:
        current = results[results["detector"] == detector].set_index(keys)
        for key in current.index.intersection(target.index):
            row = dict(zip(keys, key))
            row["detector"] = detector
            row["delta_recall"] = float(current.loc[key, "recall"] - target.loc[key, "recall"])
            row["delta_FPR"] = float(current.loc[key, "FPR"] - target.loc[key, "FPR"])
            row["delta_AUROC"] = float(current.loc[key, "AUROC"] - target.loc[key, "AUROC"])
            row["delta_AUPRC"] = float(current.loc[key, "AUPRC"] - target.loc[key, "AUPRC"])
            row["delta_success"] = float(current.loc[key, "success"] - target.loc[key, "success"])
            for column in TRANSFERABILITY_COLUMNS:
                row[column] = float(current.loc[key, column])
            rows.append(row)
    return pd.DataFrame(rows)


def _write_figures(
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    equivalence: pd.DataFrame,
    weights: pd.DataFrame,
    score_scatter: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    def save(name: str) -> None:
        path = output_dir / f"{name}.png"
        plt.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(str(path))
        plt.close()

    mean_results = results.groupby(["detector", "N"], as_index=False).agg(recall=("recall", "mean"), FPR=("FPR", "mean"))
    plt.figure(figsize=(6.6, 4.2))
    plt.axhline(RECALL_TARGET, color="black", linewidth=1.0, linestyle="--")
    for detector, group in mean_results.groupby("detector"):
        group = group.sort_values("N")
        plt.plot(group["N"], group["recall"], marker="o", label=detector)
    plt.xlabel("Commissioning healthy cycles")
    plt.ylabel("Recall")
    plt.legend(fontsize=7)
    save("figure1_recall_vs_N")

    plt.figure(figsize=(6.6, 4.2))
    plt.axhline(FALSE_ALERT_BUDGET, color="black", linewidth=1.0, linestyle="--")
    for detector, group in mean_results.groupby("detector"):
        group = group.sort_values("N")
        plt.plot(group["N"], group["FPR"], marker="o", label=detector)
    plt.xlabel("Commissioning healthy cycles")
    plt.ylabel("FPR")
    plt.legend(fontsize=7)
    save("figure2_fpr_vs_N")

    srace_deltas = deltas[deltas["detector"] == "S-RACE"]
    plt.figure(figsize=(6.2, 4.1))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.scatter(srace_deltas["projector_similarity"], srace_deltas["delta_recall"], s=18, alpha=0.7)
    plt.xlabel("Source-target projector similarity")
    plt.ylabel("Recall(S-RACE) - Recall(TargetOnly)")
    save("figure3_transfer_gain_vs_compatibility")

    if not weights.empty:
        plt.figure(figsize=(6.2, 4.1))
        sample = weights[weights["detector"] == "S-RACE"].head(2000)
        plt.scatter(sample["direction"], sample["transfer_weight"], s=10, alpha=0.35)
        plt.xlabel("Target covariance direction")
        plt.ylabel("S-RACE transfer weight")
        save("figure4_per_direction_transfer_weights")

    plt.figure(figsize=(6.2, 4.1))
    scatter = score_scatter[
        (score_scatter["reference_detector"] == "TargetOnly")
        & (score_scatter["candidate_detector"] == "S-RACE")
    ]
    normal = scatter["label"].eq(0)
    plt.scatter(scatter.loc[normal, "reference_score"], scatter.loc[normal, "candidate_score"], s=8, alpha=0.25, label="healthy")
    plt.scatter(scatter.loc[~normal, "reference_score"], scatter.loc[~normal, "candidate_score"], s=8, alpha=0.25, label="anomaly")
    if len(scatter):
        lo = float(np.nanmin([scatter["reference_score"].min(), scatter["candidate_score"].min()]))
        hi = float(np.nanmax([scatter["reference_score"].max(), scatter["candidate_score"].max()]))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("TargetOnly score")
    plt.ylabel("S-RACE score")
    plt.legend(fontsize=8)
    save("figure5_targetonly_vs_srace_equivalence")

    plt.figure(figsize=(6.2, 4.1))
    scatter = score_scatter[
        (score_scatter["reference_detector"] == "S-RACE")
        & (score_scatter["candidate_detector"] == "S-RACE SourcePermutation")
    ]
    normal = scatter["label"].eq(0)
    plt.scatter(scatter.loc[normal, "reference_score"], scatter.loc[normal, "candidate_score"], s=8, alpha=0.25, label="healthy")
    plt.scatter(scatter.loc[~normal, "reference_score"], scatter.loc[~normal, "candidate_score"], s=8, alpha=0.25, label="anomaly")
    if len(scatter):
        lo = float(np.nanmin([scatter["reference_score"].min(), scatter["candidate_score"].min()]))
        hi = float(np.nanmax([scatter["reference_score"].max(), scatter["candidate_score"].max()]))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Real-source S-RACE score")
    plt.ylabel("Source-permutation score")
    plt.legend(fontsize=8)
    save("figure6_real_source_vs_source_permutation")
    return paths


def run_srace(data_path: Path, output_dir: Path, n_values: tuple[int, ...], seeds: tuple[int, ...]) -> dict[str, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = _requested_run_config(data_path, output_dir, n_values, seeds)
    _assert_output_dir_compatible(output_dir, requested)
    rng = np.random.default_rng(GLOBAL_SEED)
    del rng

    print("Loading VORAUS metadata for split construction...", flush=True)
    cycles = load_cycle_metadata(path=data_path)
    rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    score_scatter_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    weight_frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []

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
            selected_ids = _split_episode_ids(split)
            print(
                f"Loading signals/features for N={n}, seed={seed}: "
                f"{len(selected_ids)} episodes...",
                flush=True,
            )
            selected_cycles = load_cycles(
                path=data_path,
                signal_set="measured",
                episode_ids=selected_ids,
            )
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
            wrong_regime = min(
                regimes,
                key=lambda item: float(item.metrics.get("projector_similarity", 0.0)),
            )
            wrong_source = _matrix_from_ids(wrong_regime.source_episode_ids, features_by_episode)
            for regime in regimes:
                source = _matrix_from_ids(regime.source_episode_ids, features_by_episode)
                keys = {
                    "source_pair_id": regime.source_pair_id,
                    "source_group": regime.source_group,
                    "target_group": regime.target_group,
                    "N": n,
                    "seed": seed,
                }
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
                source_rows.append({**keys, **{k: float(v) for k, v in regime.metrics.items()}, "source_size": len(regime.source_episode_ids)})
                payloads: dict[str, dict[str, object]] = {}
                for name in [
                    "TargetOnly",
                    "OriginalRACE",
                    "S-RACE",
                    "S-RACE WrongSource",
                    "S-RACE SourcePermutation",
                    "S-RACE CompatibilityPermutation",
                ]:
                    fit_source = wrong_source if name == "S-RACE WrongSource" else source
                    fit_regime = wrong_regime if name == "S-RACE WrongSource" else regime
                    row, payload, weight_frame = _evaluate_detector(
                        name=name,
                        source=fit_source,
                        target=target,
                        calibration=calibration,
                        healthy_eval=healthy_eval,
                        anomaly_eval=anomaly_eval,
                        n=n,
                        seed=seed,
                        source_pair_id=regime.source_pair_id,
                        source_group=regime.source_group,
                        target_group=regime.target_group,
                        transfer_metrics=fit_regime.metrics,
                        fit_source_pair_id=fit_regime.source_pair_id,
                        fit_source_group=fit_regime.source_group,
                    )
                    rows.append(row)
                    payloads[name] = payload
                    if not weight_frame.empty:
                        weight_frames.append(weight_frame)
                score_rows.extend(_score_equivalence_rows(keys, payloads))
                class_rows.extend(_per_class_rows(keys, split.target_anomaly_evaluation, payloads))
                eval_ids = episode_ids(split.target_normal_evaluation) + episode_ids(split.target_anomaly_evaluation)
                labels = np.r_[
                    np.zeros(len(split.target_normal_evaluation), dtype=np.int64),
                    np.ones(len(split.target_anomaly_evaluation), dtype=np.int64),
                ]
                score_scatter_rows.extend(_score_scatter_rows(keys, eval_ids, labels, payloads))

    results = pd.DataFrame(rows)
    deltas = _paired_deltas(results)
    summary = results.groupby(["detector", "N"], as_index=False).agg(
        recall_mean=("recall", "mean"),
        FPR_mean=("FPR", "mean"),
        AUROC_mean=("AUROC", "mean"),
        AUPRC_mean=("AUPRC", "mean"),
        success_rate=("success", "mean"),
        transferred_dimensions_mean=("transferred_dimensions", "mean"),
        weight_mean=("weight_mean", "mean"),
    )
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    equivalence = pd.DataFrame(score_rows)
    score_scatter = pd.DataFrame(score_scatter_rows)
    mechanism = results[
        [
            "detector",
            "N",
            "seed",
            "source_pair_id",
            "transferred_dimensions",
            "weight_mean",
            "weight_median",
            "weight_max",
            "compatibility_mean",
            "compatibility_median",
            "compatibility_max",
            "structural_compatibility_mean",
            "structural_compatibility_median",
            "structural_compatibility_max",
            "active_structural_compatibility_mean",
            "active_structural_compatibility_median",
            "active_structural_compatibility_max",
            "principal_cos2_mean",
            "principal_cos2_median",
            "principal_cos2_min",
            "principal_cos2_max",
            "pre_gate_weight_mean",
            "pre_gate_weight_median",
            "pre_gate_weight_max",
            "pre_gate_compatibility_mean",
            "pre_gate_compatibility_median",
            "pre_gate_compatibility_max",
            "variance_compatibility_mean",
            "variance_compatibility_median",
            "variance_compatibility_max",
            "location_compatibility_mean",
            "location_compatibility_median",
            "location_compatibility_max",
            "target_uncertainty",
            "shared_rank",
            "private_dimensions",
            "safe_gate_open",
            "safe_gate_margin",
            "fallback",
            "fallback_reason",
            "condition_number",
            "effective_rank",
            "source_shrinkage",
            "target_shrinkage",
        ]
    ].copy()
    if not equivalence.empty:
        mechanism = mechanism.merge(
            equivalence[
                (equivalence["reference_detector"] == "TargetOnly")
                & (equivalence["candidate_detector"].isin(["S-RACE", "OriginalRACE"]))
                & (equivalence["score_split"] == "eval")
            ][["source_pair_id", "N", "seed", "candidate_detector", "affine_r2", "number_changed_predictions", "score_equivalence_flag"]],
            left_on=["source_pair_id", "N", "seed", "detector"],
            right_on=["source_pair_id", "N", "seed", "candidate_detector"],
            how="left",
        )
    figure_paths = _write_figures(results, deltas, equivalence, weights, score_scatter, output_dir / "figures")

    paths = {
        "manifest": output_dir / "srace_manifest.json",
        "seed_results": output_dir / "srace_seed_results.csv",
        "summary": output_dir / "srace_summary.csv",
        "paired_deltas": output_dir / "srace_paired_deltas.csv",
        "per_class_recall": output_dir / "srace_per_class_recall.csv",
        "partition_audit": output_dir / "srace_partition_audit.csv",
        "mechanism_diagnostics": output_dir / "srace_mechanism_diagnostics.csv",
        "transfer_weights": output_dir / "srace_transfer_weights.csv",
        "source_compatibility": output_dir / "srace_source_compatibility.csv",
        "score_equivalence": output_dir / "srace_score_equivalence.csv",
        "score_scatter": output_dir / "srace_score_scatter.csv",
    }
    results.to_csv(paths["seed_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    deltas.to_csv(paths["paired_deltas"], index=False)
    pd.DataFrame(class_rows).to_csv(paths["per_class_recall"], index=False)
    pd.DataFrame(partition_rows).to_csv(paths["partition_audit"], index=False)
    mechanism.to_csv(paths["mechanism_diagnostics"], index=False)
    weights.to_csv(paths["transfer_weights"], index=False)
    pd.DataFrame(source_rows).to_csv(paths["source_compatibility"], index=False)
    equivalence.to_csv(paths["score_equivalence"], index=False)
    score_scatter.to_csv(paths["score_scatter"], index=False)

    manifest = {
        **requested,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "platform": platform.platform(),
        "detectors": [
            "TargetOnly",
            "OriginalRACE",
            "S-RACE",
            "S-RACE WrongSource",
            "S-RACE SourcePermutation",
            "S-RACE CompatibilityPermutation",
        ],
        "predeclared_success_criteria": {
            "source_information_matters": "S-RACE differs measurably from S-RACE SourcePermutation and S-RACE WrongSource",
            "no_trivial_targetonly_equivalence": "TargetOnly vs S-RACE eval affine_r2 below structural-equivalence flag or rankings/predictions change",
            "operational_improvement": "delta_recall > 0 subject to FPR <= 0.01",
            "no_negative_transfer_explosion": "healthy FPR not systematically worsened",
            "compatibility_predicts_benefit": "source-target compatibility relates to transfer usefulness",
            "effective_rank_policy": "shared transferred rank is capped below target commissioning N; remaining dimensions stay target-private",
        },
        "literature_to_design": LITERATURE_TO_DESIGN,
        "outputs": {key: str(value) for key, value in paths.items()},
        "figures": figure_paths,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote S-RACE outputs to {output_dir}")
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
    run_srace(args.data_path, args.output_dir, tuple(args.n), tuple(args.seeds))


if __name__ == "__main__":
    main()
