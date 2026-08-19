"""Run the COLDSTART P0 shared-precision feasibility audit.

This runner is healthy-only. It validates the sparse-precision audit on
synthetic Gaussian graphical models first, then computes source-target healthy
precision-structure overlap for the frozen voraus source-regime definitions.
It never uses target anomalies, target anomaly labels, or evaluation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import extract_feature_batch
from src.feature_extractor import make_feature_names
from src.m3_transfer_regimes import assert_no_episode_leakage, construct_source_regimes, episode_ids
from src.precision_transfer_audit import (
    StabilityConfig,
    compare_precision_structures,
    decide_p0,
    density_matched_graph_null,
    edge_frame,
    estimate_precision_stability,
    feature_permutation_null,
    high_dimensional_synthetic_stress,
    null_summary,
    robust_target_scale,
    synthetic_sanity_checks,
)
from src.reproducibility import file_sha256, reproducibility_metadata
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycle_metadata, load_cycles, select_signal_columns


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p0_precision_feasibility"
PROTOCOL_VERSION = "p0-shared-precision-feasibility-v2"
FROZEN_EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
SOURCE_SUBSET_SIZE = 100
DEFAULT_N_VALUES = (10, 25, 50, 100)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

OUTPUT_SCHEMAS = {
    "per_source_seed": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "source_episodes", "target_commissioning_episodes", "n_features",
        "zero_or_tiny_scale_features", "source_clip_fraction", "target_clip_fraction",
        "source_stable_edges", "target_stable_edges", "shared_stable_edges",
        "union_stable_edges", "source_graph_density", "target_graph_density",
        "stable_jaccard", "weighted_overlap", "stable_differential_edges",
        "differential_ratio", "D_Omega", "observed_jaccard", "null_mean",
        "null_std", "null_q95", "empirical_p", "effect_size",
        "regularization_fragile", "regularization_fragility_reason",
        "density_null_mean", "density_null_std", "density_null_q95",
        "density_empirical_p", "density_effect_size",
        "shared_partial_corr_pearson", "shared_partial_corr_spearman",
        "shared_partial_corr_sign_agreement",
        "shared_partial_corr_median_abs_diff",
        "source_numerically_reliable", "target_numerically_reliable",
        "numerically_reliable",
    ],
    "precision_stability": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "graph", "alpha", "n_samples", "n_features", "subsample_size",
        "resamples", "successful_fits", "successful_fit_fraction",
        "mean_edge_probability", "stability_instability",
        "stable_edges", "stable_density", "fit_failures", "fallback_fits",
        "positive_definite_failures", "warning_fits", "mean_iterations",
        "selected_regularization", "number_of_edges", "graph_density",
        "condition_number", "min_eigenvalue", "max_eigenvalue",
        "positive_definite", "converged", "iterations", "warnings", "row_type",
    ],
    "stable_edges": [
        "N", "seed", "source_pair_id", "feature_i", "feature_j",
        "source_edge_probability", "target_edge_probability",
        "source_stable", "target_stable", "shared_stable",
        "source_partial_correlation", "target_partial_correlation",
    ],
    "shared_edge_summary": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "source_stable_edges", "target_stable_edges", "shared_stable_edges",
        "union_stable_edges", "source_graph_density", "target_graph_density",
        "stable_jaccard", "weighted_overlap", "observed_jaccard",
        "null_mean", "null_std", "null_q95", "empirical_p", "effect_size",
    ],
    "differential_summary": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "stable_differential_edges", "differential_ratio", "D_Omega",
    ],
    "null_overlap": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "null_type", "replicate", "jaccard", "weighted_overlap",
        "permutation_changed_identity",
    ],
    "regularization_sensitivity": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "graph", "alpha", "is_selected_alpha", "stable_edges", "stable_density",
    ],
    "partition_audit": [
        "N", "seed", "commissioning_count", "calibration_count",
        "healthy_eval_count", "anomaly_eval_count",
        "commissioning_calibration_overlap",
        "commissioning_healthy_eval_overlap",
        "commissioning_anomaly_eval_overlap", "target_anomalies_used",
        "target_anomaly_labels_used", "target_evaluation_scores_used",
        "anomaly_outcomes_used_to_select_precision_regularization",
    ],
    "partial_correlation": [
        "N", "seed", "source_pair_id", "source_group", "target_group",
        "shared_partial_corr_pearson", "shared_partial_corr_spearman",
        "shared_partial_corr_sign_agreement",
        "shared_partial_corr_median_abs_diff",
        "source_numerically_reliable", "target_numerically_reliable",
        "numerically_reliable",
    ],
    "summary_by_N_source": [
        "N", "source_pair_id", "source_group", "target_group", "seeds",
        "stable_jaccard_median", "stable_jaccard_iqr",
        "weighted_overlap_median", "differential_ratio_median",
        "effect_size_median", "empirical_null_rejection_fraction",
        "density_empirical_null_rejection_fraction",
        "target_stable_edges_median", "source_stable_edges_median",
        "regularization_fragile_fraction",
        "partial_corr_pearson_median", "partial_corr_spearman_median",
        "partial_corr_sign_agreement_median",
    ],
    "summary_by_N": [
        "N", "rows", "source_regimes", "seeds",
        "stable_jaccard_median", "stable_jaccard_iqr",
        "weighted_overlap_median", "differential_ratio_median",
        "effect_size_median", "empirical_null_rejection_fraction",
        "density_empirical_null_rejection_fraction",
        "target_stable_edges_median", "source_stable_edges_median",
        "regularization_fragile_fraction",
        "partial_corr_pearson_median", "partial_corr_spearman_median",
        "partial_corr_sign_agreement_median",
    ],
    "semantic_summary": [
        "N", "seed", "source_pair_id", "edge_category", "count", "fraction",
    ],
}


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "manifest": output_dir / "p0_manifest.json",
        "per_source_seed": output_dir / "p0_per_source_seed.csv",
        "precision_stability": output_dir / "p0_precision_stability.csv",
        "stable_edges": output_dir / "p0_stable_edges.csv",
        "shared_edge_summary": output_dir / "p0_shared_edge_summary.csv",
        "differential_summary": output_dir / "p0_differential_summary.csv",
        "null_overlap": output_dir / "p0_null_overlap.csv",
        "regularization_sensitivity": output_dir / "p0_regularization_sensitivity.csv",
        "synthetic_sanity": output_dir / "p0_synthetic_sanity.csv",
        "partition_audit": output_dir / "p0_partition_audit.csv",
        "decision": output_dir / "p0_decision.json",
        "partial_correlation": output_dir / "p0_partial_correlation_agreement.csv",
        "synthetic_stress": output_dir / "p0_high_dimensional_synthetic_stress.csv",
        "summary_by_N_source": output_dir / "p0_summary_by_N_source.csv",
        "summary_by_N": output_dir / "p0_summary_by_N.csv",
        "semantic_summary": output_dir / "p0_shared_edge_semantic_summary.csv",
        "completeness": output_dir / "p0_completeness_audit.json",
        "report": output_dir / "P0_PRECISION_FEASIBILITY_REPORT.md",
    }


def run_p0(
    *,
    data_path: Path,
    output_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    config: StabilityConfig,
    null_replicates: int,
    source_subset_size: int,
    synthetic_only: bool = False,
    resume: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    _ensure_output_schemas(paths)

    synthetic = synthetic_sanity_checks(config=_synthetic_config(config), dimension=40)
    synthetic.to_csv(paths["synthetic_sanity"], index=False)
    real_feature_names = _feature_names_from_dataset(data_path) if data_path.exists() else tuple()
    if not bool(synthetic["expectation_pass"].all()):
        decision = decide_p0(pd.DataFrame(), synthetic)
        _write_decision(paths["decision"], decision)
        _write_report(paths["report"], pd.DataFrame(), synthetic, pd.DataFrame(), decision, None)
        _write_manifest(paths["manifest"], data_path, n_values, seeds, source_subset_size, config, null_replicates, paths, real_feature_names)
        raise RuntimeError(
            "P0 synthetic sanity failed; refusing to interpret robotics precision overlap."
        )
    real_feature_dim = len(real_feature_names) if real_feature_names else None
    synthetic_stress = high_dimensional_synthetic_stress(
        dimensions=tuple(v for v in (40, 128, 256, real_feature_dim or 0) if v),
        config=_synthetic_stress_config(config),
        full_dimension=real_feature_dim,
        null_replicates=min(null_replicates, 20),
    )
    synthetic_stress.to_csv(paths["synthetic_stress"], index=False)

    if synthetic_only:
        decision = decide_p0(pd.DataFrame(), synthetic_stress if not synthetic_stress.empty else synthetic)
        _write_decision(paths["decision"], decision)
        _write_report(paths["report"], pd.DataFrame(), synthetic, synthetic_stress, decision, None)
        _write_manifest(paths["manifest"], data_path, n_values, seeds, source_subset_size, config, null_replicates, paths, real_feature_names)
        return paths

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    completed = _completed_keys(paths["per_source_seed"]) if resume else set()
    cycles = load_cycle_metadata(data_path)
    per_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    edge_frames: list[pd.DataFrame] = []
    shared_rows: list[dict[str, object]] = []
    differential_rows: list[dict[str, object]] = []
    null_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    partial_rows: list[dict[str, object]] = []

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
            source_ids = episode_ids(split.source_train)
            target_ids = episode_ids(split.target_commissioning)
            assert_no_episode_leakage(
                {
                    "source": source_ids,
                    "commissioning": target_ids,
                    "calibration": episode_ids(split.target_calibration),
                    "healthy_eval": episode_ids(split.target_normal_evaluation),
                    "anomaly_eval": episode_ids(split.target_anomaly_evaluation),
                }
            )
            partition_rows.append(_partition_audit_row(n, seed, split))
            _assert_p0_leakage_free(partition_rows[-1])
            needed_ids = sorted(set(source_ids) | set(target_ids))
            print(f"P0 loading healthy features for N={n}, seed={seed}: {len(needed_ids)} episodes", flush=True)
            selected_cycles = load_cycles(data_path, signal_set="measured", episode_ids=needed_ids)
            features = _feature_lookup(selected_cycles)
            source_all = _matrix_from_ids(source_ids, features)
            target = _matrix_from_ids(target_ids, features)
            regimes = construct_source_regimes(
                source_episode_ids=source_ids,
                source_features=source_all,
                target_episode_ids=target_ids,
                target_features=target,
                commissioning_size=n,
                seed=seed,
                subset_size=source_subset_size,
            )
            for regime in regimes:
                key = (int(n), int(seed), regime.source_pair_id)
                if key in completed:
                    print(f"P0 skipping completed {key}", flush=True)
                    continue
                source = _matrix_from_ids(regime.source_episode_ids, features)
                source_scaled, target_scaled, scaling = robust_target_scale(source, target)
                seed_base = _stable_seed(n, seed, regime.source_pair_id)
                source_result = estimate_precision_stability(
                    source_scaled,
                    config=config,
                    rng_seed=seed_base,
                    prefix="source",
                )
                target_result = estimate_precision_stability(
                    target_scaled,
                    config=config,
                    rng_seed=seed_base + 1,
                    prefix="target",
                )
                metrics = compare_precision_structures(source_result, target_result)
                null = feature_permutation_null(
                    source_result.edge_probabilities,
                    target_result.edge_probabilities,
                    stable_edge_threshold=config.stable_edge_threshold,
                    replicates=null_replicates,
                    rng_seed=seed_base + 2,
                )
                null_metrics = null_summary(float(metrics["stable_jaccard"]), null)
                density_null = density_matched_graph_null(
                    source_result.stable_edges,
                    target_result.stable_edges,
                    n_features=int(target.shape[1]),
                    replicates=null_replicates,
                    rng_seed=seed_base + 3,
                )
                density_metrics = null_summary(float(metrics["stable_jaccard"]), density_null)
                base = {
                    "N": int(n),
                    "seed": int(seed),
                    "source_pair_id": regime.source_pair_id,
                    "source_group": regime.source_group,
                    "target_group": regime.target_group,
                    "source_episodes": int(len(regime.source_episode_ids)),
                    "target_commissioning_episodes": int(len(target_ids)),
                    "n_features": int(target.shape[1]),
                    **regime.metrics,
                    **scaling,
                }
                row = {
                    **base,
                    **metrics,
                    **null_metrics,
                    "density_null_mean": density_metrics["null_mean"],
                    "density_null_std": density_metrics["null_std"],
                    "density_null_q95": density_metrics["null_q95"],
                    "density_empirical_p": density_metrics["empirical_p"],
                    "density_effect_size": density_metrics["effect_size"],
                    "source_numerically_reliable": bool(source_result.numerical.get("numerically_reliable", False)),
                    "target_numerically_reliable": bool(target_result.numerical.get("numerically_reliable", False)),
                    "numerically_reliable": bool(
                        source_result.numerical.get("numerically_reliable", False)
                        and target_result.numerical.get("numerically_reliable", False)
                    ),
                }
                per_rows.append(row)
                shared_rows.append(
                    {
                        **base,
                        **{k: row[k] for k in [
                            "source_stable_edges",
                            "target_stable_edges",
                            "shared_stable_edges",
                            "union_stable_edges",
                            "source_graph_density",
                            "target_graph_density",
                            "stable_jaccard",
                            "weighted_overlap",
                            "observed_jaccard",
                            "null_mean",
                            "null_std",
                            "null_q95",
                            "empirical_p",
                            "effect_size",
                        ]},
                    }
                )
                differential_rows.append(
                    {
                        **base,
                        "stable_differential_edges": row["stable_differential_edges"],
                        "differential_ratio": row["differential_ratio"],
                        "D_Omega": row["D_Omega"],
                    }
                )
                partial_rows.append(
                    {
                        **base,
                        "shared_partial_corr_pearson": row["shared_partial_corr_pearson"],
                        "shared_partial_corr_spearman": row["shared_partial_corr_spearman"],
                        "shared_partial_corr_sign_agreement": row["shared_partial_corr_sign_agreement"],
                        "shared_partial_corr_median_abs_diff": row["shared_partial_corr_median_abs_diff"],
                    }
                )
                for graph_name, result in [("source", source_result), ("target", target_result)]:
                    stability_rows.extend([{**base, **item, "graph": graph_name} for item in result.stability_rows])
                    sensitivity_rows.extend([{**base, **item, "graph": graph_name} for item in result.sensitivity_rows])
                    stability_rows.append({**base, **result.numerical, "graph": graph_name, "row_type": "final_fit"})
                edge_frames.append(edge_frame(regime.source_pair_id, n, seed, source_result, target_result))
                null_frames.append(pd.concat([null, density_null], ignore_index=True).assign(**base))
                if real_feature_names:
                    _write_semantic_increment(paths["semantic_summary"], _semantic_summary_rows(
                        n, seed, regime.source_pair_id, edge_frames[-1], real_feature_names
                    ))
                _flush(paths, per_rows, stability_rows, edge_frames, shared_rows, differential_rows, null_frames, sensitivity_rows, partition_rows, partial_rows)

    _flush(paths, per_rows, stability_rows, edge_frames, shared_rows, differential_rows, null_frames, sensitivity_rows, partition_rows, partial_rows)
    robotics = pd.read_csv(paths["per_source_seed"]) if paths["per_source_seed"].exists() else pd.DataFrame()
    robotics = _attach_regularization_fragility(robotics, paths["regularization_sensitivity"])
    if not robotics.empty:
        robotics.to_csv(paths["per_source_seed"], index=False)
    completeness = _completeness_audit(data_path, robotics, n_values, seeds, source_subset_size)
    _write_json(paths["completeness"], completeness)
    _write_summary_tables(paths, robotics)
    decision = decide_p0(robotics, synthetic_stress if not synthetic_stress.empty else synthetic, completeness=completeness)
    _write_decision(paths["decision"], decision)
    _write_report(paths["report"], robotics, synthetic, synthetic_stress, decision, completeness)
    _write_manifest(paths["manifest"], data_path, n_values, seeds, source_subset_size, config, null_replicates, paths, real_feature_names)
    return paths


def _flush(
    paths: dict[str, Path],
    per_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    edge_frames: list[pd.DataFrame],
    shared_rows: list[dict[str, object]],
    differential_rows: list[dict[str, object]],
    null_frames: list[pd.DataFrame],
    sensitivity_rows: list[dict[str, object]],
    partition_rows: list[dict[str, object]],
    partial_rows: list[dict[str, object]],
) -> None:
    _append_or_write(paths["per_source_seed"], pd.DataFrame(per_rows), schema_key="per_source_seed")
    _append_or_write(paths["precision_stability"], pd.DataFrame(stability_rows), schema_key="precision_stability")
    _append_or_write(
        paths["stable_edges"],
        pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(),
        schema_key="stable_edges",
    )
    _append_or_write(paths["shared_edge_summary"], pd.DataFrame(shared_rows), schema_key="shared_edge_summary")
    _append_or_write(paths["differential_summary"], pd.DataFrame(differential_rows), schema_key="differential_summary")
    _append_or_write(
        paths["null_overlap"],
        pd.concat(null_frames, ignore_index=True) if null_frames else pd.DataFrame(),
        schema_key="null_overlap",
    )
    _append_or_write(
        paths["regularization_sensitivity"],
        pd.DataFrame(sensitivity_rows),
        schema_key="regularization_sensitivity",
    )
    _append_or_write(paths["partition_audit"], pd.DataFrame(partition_rows), schema_key="partition_audit")
    _append_or_write(paths["partial_correlation"], pd.DataFrame(partial_rows), schema_key="partial_correlation")
    per_rows.clear()
    stability_rows.clear()
    edge_frames.clear()
    shared_rows.clear()
    differential_rows.clear()
    null_frames.clear()
    sensitivity_rows.clear()
    partition_rows.clear()
    partial_rows.clear()


def _append_or_write(path: Path, frame: pd.DataFrame, *, schema_key: str | None = None) -> None:
    if schema_key is not None and schema_key in OUTPUT_SCHEMAS:
        frame = frame.reindex(columns=OUTPUT_SCHEMAS[schema_key])
    if frame.empty:
        if not path.exists():
            frame.to_csv(path, index=False)
        return
    if path.exists() and path.stat().st_size > 0:
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def _ensure_output_schemas(paths: dict[str, Path]) -> None:
    for key, columns in OUTPUT_SCHEMAS.items():
        path = paths[key]
        if not path.exists() or path.stat().st_size == 0:
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def _attach_regularization_fragility(
    robotics: pd.DataFrame,
    sensitivity_path: Path,
) -> pd.DataFrame:
    if robotics.empty or not sensitivity_path.exists() or sensitivity_path.stat().st_size == 0:
        return robotics
    sensitivity = pd.read_csv(sensitivity_path)
    required = {"N", "seed", "source_pair_id", "graph", "is_selected_alpha", "stable_edges"}
    if not required.issubset(sensitivity.columns):
        out = robotics.copy()
        out["regularization_fragile"] = True
        out["regularization_fragility_reason"] = "regularization sensitivity schema missing required columns"
        return out

    rows: list[dict[str, object]] = []
    for key, group in sensitivity.groupby(["N", "seed", "source_pair_id"], dropna=False, sort=True):
        fragile = False
        reasons: list[str] = []
        for graph, graph_group in group.groupby("graph", dropna=False, sort=True):
            selected = graph_group[graph_group["is_selected_alpha"].astype(bool)]
            if selected.empty:
                fragile = True
                reasons.append(f"{graph}:missing_selected_alpha")
                continue
            selected_edges = float(selected["stable_edges"].iloc[0])
            neighbor_edges = graph_group["stable_edges"].astype(float).to_numpy()
            if selected_edges <= 0.0:
                fragile = True
                reasons.append(f"{graph}:selected_has_zero_stable_edges")
            elif np.min(neighbor_edges) <= 0.0:
                fragile = True
                reasons.append(f"{graph}:neighbor_has_zero_stable_edges")
            elif np.max(neighbor_edges) / max(np.min(neighbor_edges), 1.0) > 3.0:
                fragile = True
                reasons.append(f"{graph}:neighbor_edge_count_ratio_gt_3")
        rows.append(
            {
                "N": int(key[0]),
                "seed": int(key[1]),
                "source_pair_id": str(key[2]),
                "regularization_fragile": bool(fragile),
                "regularization_fragility_reason": ";".join(reasons),
            }
        )
    fragility = pd.DataFrame(rows)
    out = robotics.drop(columns=["regularization_fragile", "regularization_fragility_reason"], errors="ignore")
    out = out.merge(fragility, on=["N", "seed", "source_pair_id"], how="left", validate="many_to_one")
    out["regularization_fragile"] = out["regularization_fragile"].fillna(True).astype(bool)
    out["regularization_fragility_reason"] = out["regularization_fragility_reason"].fillna("regularization sensitivity missing")
    return out


def _completed_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=["N", "seed", "source_pair_id"])
    return {(int(row.N), int(row.seed), str(row.source_pair_id)) for row in frame.itertuples(index=False)}


def _feature_lookup(cycles) -> dict[int, np.ndarray]:
    batch = extract_feature_batch(cycles)
    return {int(eid): batch.features[i] for i, eid in enumerate(batch.episode_ids)}


def _matrix_from_ids(ids, features_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([features_by_episode[int(eid)] for eid in ids])


def _partition_audit_row(n: int, seed: int, split) -> dict[str, object]:
    groups = {
        "commissioning": set(episode_ids(split.target_commissioning)),
        "calibration": set(episode_ids(split.target_calibration)),
        "healthy_eval": set(episode_ids(split.target_normal_evaluation)),
        "anomaly_eval": set(episode_ids(split.target_anomaly_evaluation)),
    }
    return {
        "N": int(n),
        "seed": int(seed),
        "commissioning_count": len(groups["commissioning"]),
        "calibration_count": len(groups["calibration"]),
        "healthy_eval_count": len(groups["healthy_eval"]),
        "anomaly_eval_count": len(groups["anomaly_eval"]),
        "commissioning_calibration_overlap": len(groups["commissioning"] & groups["calibration"]),
        "commissioning_healthy_eval_overlap": len(groups["commissioning"] & groups["healthy_eval"]),
        "commissioning_anomaly_eval_overlap": len(groups["commissioning"] & groups["anomaly_eval"]),
        "target_anomalies_used": False,
        "target_anomaly_labels_used": False,
        "target_evaluation_scores_used": False,
        "anomaly_outcomes_used_to_select_precision_regularization": False,
    }


def _assert_p0_leakage_free(row: dict[str, object]) -> None:
    violations = [
        key
        for key in [
            "commissioning_calibration_overlap",
            "commissioning_healthy_eval_overlap",
            "commissioning_anomaly_eval_overlap",
        ]
        if int(row[key]) != 0
    ]
    forbidden_flags = [
        key
        for key in [
            "target_anomalies_used",
            "target_anomaly_labels_used",
            "target_evaluation_scores_used",
            "anomaly_outcomes_used_to_select_precision_regularization",
        ]
        if bool(row[key])
    ]
    if violations or forbidden_flags:
        raise RuntimeError(
            "P0 leakage audit failed: "
            f"overlap violations={violations}; forbidden flags={forbidden_flags}"
        )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_decision(path: Path, decision: dict[str, object]) -> None:
    path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(
    path: Path,
    data_path: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    source_subset_size: int,
    config: StabilityConfig,
    null_replicates: int,
    paths: dict[str, Path],
    feature_names: tuple[str, ...],
) -> None:
    metadata = reproducibility_metadata(
        repo_root=PROJECT_ROOT,
        input_paths={"dataset": data_path},
        artifact_paths={key: value for key, value in paths.items() if key != "manifest"},
    )
    manifest = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "dataset_hash": metadata["input_hashes"]["dataset"]["sha256"],
        "relevant_source_file_hashes": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in [
                PROJECT_ROOT / "src" / "precision_transfer_audit.py",
                PROJECT_ROOT / "experiments" / "run_p0_precision_feasibility.py",
                PROJECT_ROOT / "src" / "m3_transfer_regimes.py",
                PROJECT_ROOT / "src" / "feature_extractor.py",
                PROJECT_ROOT / "src" / "split_generator.py",
                PROJECT_ROOT / "src" / "voraus_loader.py",
            ]
        },
        "feature_dimension": _feature_dimension(paths["per_source_seed"]) or (len(feature_names) if feature_names else None),
        "feature_names": list(feature_names),
        "feature_names_sha256": _sha256_text("\n".join(feature_names)) if feature_names else None,
        "scaling_procedure": "target commissioning robust median/IQR; source transformed with same target healthy scaler",
        "precision_estimator": "GraphicalLasso sparse inverse covariance",
        "regularization_grid": list(config.alpha_grid),
        "stability_method": "StARS-inspired subsampling stability selection; instability=mean_{i<j} 2*pi_ij*(1-pi_ij); alpha=frozen sparse-stable heuristic",
        "GraphicalLasso_configuration": {
            "max_iter": config.max_iter,
            "tol": config.tol,
            "edge_abs_threshold": config.edge_abs_threshold,
            "assume_centered": False,
        },
        "subsample_fraction": config.subsample_fraction,
        "number_of_stability_resamples": config.resamples,
        "stable_edge_threshold": config.stable_edge_threshold,
        "null_definitions": {
            "feature_identity_permutation": "permute target feature identities relative to source and recompute stable-support overlap",
            "density_matched_random_graph": "random undirected graphs with source/target stable edge counts matched",
        },
        "number_of_null_replicates": null_replicates,
        "synthetic_configuration": {
            "cases": [
                "identical sparse graph",
                "partially shared graph",
                "unrelated graphs",
                "N sweep under shared graph",
                "dense target-specific differential structure",
            ],
            "dimension": 40,
        },
        "high_dimensional_synthetic_configuration": {
            "dimensions": _csv_unique_ints(paths["synthetic_stress"], "dimension"),
            "N_values": _csv_unique_ints(paths["synthetic_stress"], "N"),
            "scenarios": _csv_unique_strings(paths["synthetic_stress"], "case"),
            "full_real_dimension_note": "full p is run on a reduced N slice when present to benchmark feasibility under the same estimator",
            "stress_repetition_limitation": "high-dimensional stress uses a reduced four-alpha grid and eight stability resamples; full real p is evaluated at both smallest and largest commissioning N",
        },
        "P0_decision_rule": {
            "source": "src.precision_transfer_audit.decide_p0",
            "decisions": [
                "P0_PASS_SHARED_PRECISION_STRUCTURE",
                "P0_FAIL_TARGET_GRAPH_UNSTABLE",
                "P0_FAIL_SOURCE_TARGET_STRUCTURE_UNRELATED",
                "P0_FAIL_DIFFERENTIAL_NOT_SPARSE",
                "P0_FAIL_REGULARIZATION_FRAGILE",
                "P0_INCONCLUSIVE_MORE_HEALTHY_DATA_REQUIRED",
                "P0_INCOMPLETE_REPLICATION",
            ],
            "pass_requires": "synthetic identifiability, complete real matrix, target stable graph trajectory, feature and density null rejection, sparse differential structure, regularization robustness, partial-correlation sign agreement",
        },
        "commissioning_N_values": list(n_values),
        "seeds": list(seeds),
        "source_pair_definitions": "src.m3_transfer_regimes.construct_source_regimes with near/moderate/high healthy source subsets",
        "source_subset_size": source_subset_size,
        "anomaly_labels_used": False,
        "evaluation_healthy_used_for_selection": False,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(
    path: Path,
    robotics: pd.DataFrame,
    synthetic: pd.DataFrame,
    synthetic_stress: pd.DataFrame,
    decision: dict[str, object],
    completeness: dict[str, object] | None,
) -> None:
    if robotics.empty:
        robotics_summary = "Robotics P0 was not run in this invocation."
        by_n = "No robotics rows."
        by_source = "No robotics rows."
    else:
        robotics_summary = (
            f"Rows: {len(robotics)}; median stable Jaccard: "
            f"{robotics['stable_jaccard'].median():.4f}; median null effect size: "
            f"{robotics['effect_size'].median():.4f}; median differential ratio: "
            f"{robotics['differential_ratio'].median():.4f}."
        )
        by_n = _markdown_table(_summary_by(robotics, ["N"]))
        by_source = _markdown_table(_summary_by(robotics, ["N", "source_pair_id"]))
    stress_summary = (
        "Not run."
        if synthetic_stress.empty
        else (
            f"Rows: {len(synthetic_stress)}; classified correctly: "
            f"{int(synthetic_stress['classified_correctly'].sum())} / {len(synthetic_stress)}; "
            f"dimensions: {sorted(synthetic_stress['dimension'].dropna().astype(int).unique().tolist())}."
        )
    )
    completeness_text = json.dumps(completeness or {}, indent=2, sort_keys=True)
    text = f"""# P0 Precision Feasibility Report

## 1. Scope

This audit tests healthy-only shared sparse precision structure. It does not
fit TransferPrecision, score anomalies, or tune from anomaly outcomes.

## 2. Hypothesis

P0 audits whether the healthy target precision can plausibly be represented as
Omega_T = Omega_shared + Delta_T, where source and target share substantial
conditional-dependence structure and the target-specific correction is sparse.
This is not an anomaly-performance test.

## 3. Leakage Audit

PASS if the partition audit contains zero commissioning overlap and all
forbidden anomaly-use flags remain false. The runner asserts this per N/seed.

## 4. Synthetic Unit Sanity

Synthetic cases passed: {int(synthetic['expectation_pass'].sum())} / {len(synthetic)}.

## 5. High-Dimensional Synthetic Stress Test

{stress_summary}

## 6. Real Robotics Completeness

```json
{completeness_text}
```

## 7. Shared-Support Results

{by_source}

## 8. Differential Sparsity

See `p0_differential_summary.csv` and summary tables. Median differential ratio
is included by N and source regime.

## 9. Null Comparisons

Feature-identity permutation and density-matched random-graph nulls are reported
separately. They are not merged into one statistic.

## 10. Partial-Correlation Agreement

Pearson, Spearman, sign agreement, and median absolute partial-correlation
difference are recorded for shared stable edges.

## 11. Regularization Sensitivity

Each row records selected alpha and neighboring-alpha stable edge counts. Rows
are marked fragile if selected or neighboring supports collapse or vary by more
than the frozen ratio rule.

## 12. Feature-Semantic Edge Audit

Stable shared-edge category counts are written to
`p0_shared_edge_semantic_summary.csv`.

## 13. P0 Final Decision

Robotics summary: {robotics_summary}

Across-source trajectory:

{by_n}

`{decision.get('decision')}`

Reason: {decision.get('reason')}

`do_not_implement_precision_race`: `{decision.get('do_not_implement_precision_race')}`

## Leakage

"""
    path.write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    return "```\n" + frame.to_csv(index=False).strip() + "\n```"


def _summary_by(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_tuple))
        row.update(
            {
                "rows": int(len(group)),
                "stable_jaccard_median": float(group["stable_jaccard"].median()),
                "weighted_overlap_median": float(group["weighted_overlap"].median()),
                "differential_ratio_median": float(group["differential_ratio"].median()),
                "effect_size_median": float(group["effect_size"].median()),
                "density_empirical_p_median": float(group.get("density_empirical_p", pd.Series([np.nan])).median()),
                "partial_sign_median": float(group.get("shared_partial_corr_sign_agreement", pd.Series([np.nan])).median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary_tables(paths: dict[str, Path], robotics: pd.DataFrame) -> None:
    if robotics.empty:
        pd.DataFrame(columns=OUTPUT_SCHEMAS["summary_by_N_source"]).to_csv(paths["summary_by_N_source"], index=False)
        pd.DataFrame(columns=OUTPUT_SCHEMAS["summary_by_N"]).to_csv(paths["summary_by_N"], index=False)
        return
    rows_source = []
    rows_n = []
    for keys, sink in [(["N", "source_pair_id", "source_group", "target_group"], rows_source), (["N"], rows_n)]:
        for key, group in robotics.groupby(keys, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys, key_tuple))
            if keys == ["N"]:
                row["rows"] = int(len(group))
                row["source_regimes"] = int(group["source_pair_id"].nunique())
            row["seeds"] = int(group["seed"].nunique())
            q75, q25 = group["stable_jaccard"].quantile([0.75, 0.25]).to_numpy()
            row.update(
                {
                    "stable_jaccard_median": float(group["stable_jaccard"].median()),
                    "stable_jaccard_iqr": float(q75 - q25),
                    "weighted_overlap_median": float(group["weighted_overlap"].median()),
                    "differential_ratio_median": float(group["differential_ratio"].median()),
                    "effect_size_median": float(group["effect_size"].median()),
                    "empirical_null_rejection_fraction": float(np.mean(group["empirical_p"] <= 0.10)),
                    "density_empirical_null_rejection_fraction": float(np.mean(group["density_empirical_p"] <= 0.10)) if "density_empirical_p" in group else np.nan,
                    "target_stable_edges_median": float(group["target_stable_edges"].median()),
                    "source_stable_edges_median": float(group["source_stable_edges"].median()),
                    "regularization_fragile_fraction": float(np.mean(group.get("regularization_fragile", pd.Series([False] * len(group))).astype(bool))),
                    "partial_corr_pearson_median": float(group.get("shared_partial_corr_pearson", pd.Series([np.nan])).median()),
                    "partial_corr_spearman_median": float(group.get("shared_partial_corr_spearman", pd.Series([np.nan])).median()),
                    "partial_corr_sign_agreement_median": float(group.get("shared_partial_corr_sign_agreement", pd.Series([np.nan])).median()),
                }
            )
            sink.append(row)
    pd.DataFrame(rows_source).reindex(columns=OUTPUT_SCHEMAS["summary_by_N_source"]).to_csv(paths["summary_by_N_source"], index=False)
    pd.DataFrame(rows_n).reindex(columns=OUTPUT_SCHEMAS["summary_by_N"]).to_csv(paths["summary_by_N"], index=False)


def _feature_names_from_dataset(data_path: Path) -> tuple[str, ...]:
    if not data_path.exists():
        return tuple()
    return make_feature_names(select_signal_columns(data_path, "measured"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _csv_unique_ints(path: Path, column: str) -> list[int]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, usecols=[column])
    except Exception:
        return []
    return sorted(int(v) for v in frame[column].dropna().unique())


def _csv_unique_strings(path: Path, column: str) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, usecols=[column])
    except Exception:
        return []
    return sorted(str(v) for v in frame[column].dropna().unique())


def _synthetic_stress_config(config: StabilityConfig) -> StabilityConfig:
    """Reduced but meaningful high-dimensional stability configuration.

    Unlike the earlier pilot, this still uses multiple alphas and multiple
    resamples. A single resample cannot estimate stability probabilities.
    """
    return StabilityConfig(
        alpha_grid=(0.10, 0.20, 0.30, 0.40),
        subsample_fraction=config.subsample_fraction,
        resamples=8,
        stable_edge_threshold=min(config.stable_edge_threshold, 0.60),
        instability_threshold=config.instability_threshold,
        edge_abs_threshold=config.edge_abs_threshold,
        max_iter=min(config.max_iter, 100),
        tol=config.tol,
        min_successful_fit_fraction=config.min_successful_fit_fraction,
    )


def _completeness_audit(
    data_path: Path,
    robotics: pd.DataFrame,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    source_subset_size: int,
) -> dict[str, object]:
    expected: set[tuple[int, int, str]] = set()
    if not data_path.exists():
        return {"complete": False, "reason": "dataset missing", "expected_rows": 0, "observed_rows": int(len(robotics))}
    cycles = load_cycle_metadata(data_path)
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
            source_ids = episode_ids(split.source_train)
            target_ids = episode_ids(split.target_commissioning)
            needed_ids = sorted(set(source_ids) | set(target_ids))
            selected_cycles = load_cycles(data_path, signal_set="measured", episode_ids=needed_ids)
            features = _feature_lookup(selected_cycles)
            regimes = construct_source_regimes(
                source_episode_ids=source_ids,
                source_features=_matrix_from_ids(source_ids, features),
                target_episode_ids=target_ids,
                target_features=_matrix_from_ids(target_ids, features),
                commissioning_size=n,
                seed=seed,
                subset_size=source_subset_size,
            )
            for regime in regimes:
                expected.add((int(n), int(seed), regime.source_pair_id))
    observed_keys = (
        [(int(row.N), int(row.seed), str(row.source_pair_id)) for row in robotics.itertuples(index=False)]
        if not robotics.empty
        else []
    )
    observed = set(observed_keys)
    duplicates = sorted(key for key in observed if observed_keys.count(key) > 1)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "complete": not missing and not duplicates and not extra,
        "expected_rows": int(len(expected)),
        "observed_rows": int(len(observed_keys)),
        "missing_keys": [{"N": k[0], "seed": k[1], "source_pair_id": k[2]} for k in missing],
        "duplicate_keys": [{"N": k[0], "seed": k[1], "source_pair_id": k[2]} for k in duplicates],
        "unexpected_keys": [{"N": k[0], "seed": k[1], "source_pair_id": k[2]} for k in extra],
    }


def _semantic_summary_rows(
    n: int,
    seed: int,
    source_pair_id: str,
    edges: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    shared = edges[edges["shared_stable"].astype(bool)].copy()
    counts: dict[str, int] = {}
    for row in shared.itertuples(index=False):
        category = _edge_semantic_category(feature_names[int(row.feature_i)], feature_names[int(row.feature_j)])
        counts[category] = counts.get(category, 0) + 1
    total = max(1, sum(counts.values()))
    return pd.DataFrame(
        [
            {
                "N": int(n),
                "seed": int(seed),
                "source_pair_id": source_pair_id,
                "edge_category": category,
                "count": int(count),
                "fraction": float(count / total),
            }
            for category, count in sorted(counts.items())
        ]
    )


def _write_semantic_increment(path: Path, frame: pd.DataFrame) -> None:
    _append_or_write(path, frame, schema_key="semantic_summary")


def _edge_semantic_category(left: str, right: str) -> str:
    left_signal, left_stat = _split_feature_name(left)
    right_signal, right_stat = _split_feature_name(right)
    if left_signal == right_signal and left_stat != right_stat:
        return "same_raw_channel_different_statistics"
    if left_stat == right_stat and left_signal != right_signal:
        return "same_statistic_different_channels"
    left_joint = _joint_token(left_signal)
    right_joint = _joint_token(right_signal)
    if left_joint and left_joint == right_joint:
        return "same_joint"
    if left_joint and right_joint and left_joint != right_joint:
        return "cross_joint"
    pair_text = f"{left_signal} {right_signal}".lower()
    if ("pos" in pair_text or "position" in pair_text) and ("vel" in pair_text or "velocity" in pair_text):
        return "position_velocity_coupling"
    if "torque" in pair_text or "tau" in pair_text:
        return "torque_related"
    if _physical_token(left_signal) != _physical_token(right_signal):
        return "cross_sensor_or_physical_variable"
    return "unknown_or_other"


def _split_feature_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        return name, ""
    signal, stat = name.rsplit("__", 1)
    return signal, stat


def _joint_token(signal: str) -> str | None:
    import re

    match = re.search(r"(?:joint|axis|j)[_\- ]?(\d+)", signal.lower())
    return match.group(1) if match else None


def _physical_token(signal: str) -> str:
    text = signal.lower()
    for token in ("position", "pos", "velocity", "vel", "torque", "current", "temperature", "temp", "force"):
        if token in text:
            return token
    return "unknown"


def _feature_dimension(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, usecols=["n_features"])
    except Exception:
        return None
    if frame.empty:
        return None
    return int(frame["n_features"].iloc[0])


def _synthetic_config(config: StabilityConfig) -> StabilityConfig:
    return StabilityConfig(
        alpha_grid=tuple(alpha for alpha in config.alpha_grid if alpha <= 0.40) or (0.05, 0.10, 0.20, 0.40),
        subsample_fraction=config.subsample_fraction,
        resamples=min(config.resamples, 8),
        stable_edge_threshold=min(config.stable_edge_threshold, 0.60),
        instability_threshold=config.instability_threshold,
        edge_abs_threshold=config.edge_abs_threshold,
        max_iter=min(config.max_iter, 100),
        tol=config.tol,
    )


def _stable_seed(n: int, seed: int, label: str) -> int:
    value = f"{PROTOCOL_VERSION}|{n}|{seed}|{label}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--source-subset-size", type=int, default=SOURCE_SUBSET_SIZE)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20, 0.40, 0.80])
    parser.add_argument("--stability-resamples", type=int, default=24)
    parser.add_argument("--subsample-fraction", type=float, default=0.80)
    parser.add_argument("--stable-edge-threshold", type=float, default=0.70)
    parser.add_argument("--null-replicates", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--min-successful-fit-fraction", type=float, default=0.80)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StabilityConfig(
        alpha_grid=tuple(float(v) for v in args.alpha_grid),
        subsample_fraction=float(args.subsample_fraction),
        resamples=int(args.stability_resamples),
        stable_edge_threshold=float(args.stable_edge_threshold),
        max_iter=int(args.max_iter),
        min_successful_fit_fraction=float(args.min_successful_fit_fraction),
    )
    paths = run_p0(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_values=tuple(int(v) for v in args.n_values),
        seeds=tuple(int(v) for v in args.seeds),
        config=config,
        null_replicates=int(args.null_replicates),
        source_subset_size=int(args.source_subset_size),
        synthetic_only=bool(args.synthetic_only),
        resume=not bool(args.no_resume),
    )
    print(f"P0 outputs written to {paths['manifest'].parent}", flush=True)


if __name__ == "__main__":
    main()