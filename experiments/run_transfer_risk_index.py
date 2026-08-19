"""Run the healthy-only Transfer Risk Index diagnostic.

TRI is frozen before joining to anomaly outcomes. This runner reconstructs the
same source/target healthy partitions used by the Original RACE component
ablation, computes covariance-risk proxies from healthy data only, and only
then joins to frozen downstream outcome artifacts.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_original_race_component_ablation import (
    CALIBRATION_SIZE,
    DATASET_PATH,
    FROZEN_EVALUATION_SEED,
    MAXIMUM_COMMISSIONING_SIZE,
    NORMAL_EVALUATION_SIZE,
    SOURCE_SUBSET_SIZE,
    _dataset_hash,
    _feature_lookup,
    _git_commit,
    _git_dirty,
    _matrix,
    _matrix_from_ids,
    _split_episode_ids,
)
from src.m3_transfer_regimes import assert_no_episode_leakage, construct_source_regimes, episode_ids
from src.split_generator import create_frozen_evaluation_split
from src.transfer_risk_index import (
    bootstrap_covariance_estimates,
    compute_transfer_risk_index_from_estimates,
    covariance_bootstrap_uncertainty,
    covariance_jackknife_uncertainty,
    diagonal_gaussian_uncertainty_proxy,
    ledoit_wolf_covariance,
    optimal_bootstrap_blend_weight,
    synthetic_sanity_checks,
)
from src.voraus_loader import load_cycle_metadata, load_cycles


PROTOCOL_VERSION = "transfer-risk-index-v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transfer_risk_index"
DEFAULT_ABLATION_DIR = PROJECT_ROOT / "outputs" / "original_race_component_ablation_seed0_4_aggregate"
DEFAULT_N = (10,)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_RESAMPLES = 200
PRIMARY_COVARIANCE_ESTIMATOR = "LedoitWolf covariance, stabilized at minimum eigenvalue 1e-8"
PRIMARY_DISCREPANCY = "normalized squared Frobenius source_target_covariance_discrepancy_proxy"
PRIMARY_UNCERTAINTY = "normalized bootstrap target_covariance_estimation_uncertainty_proxy"
PRIMARY_INDEX = "TRI = source_target_cov_discrepancy_normalized / max(target_cov_uncertainty_normalized, 1e-12)"
PRIMARY_DOWNSTREAM_TARGET = "CovarianceTransferOnly delta_recall relative to TargetOnly"


LITERATURE_TO_DESIGN = [
    {
        "paper": "Tong et al. (NeurIPS 2021), transferability in multi-source transfer learning",
        "core_theoretical_result": "Transfer benefit depends on both target sample size and source-target discrepancy.",
        "relevance_to_RACE": "Frames covariance borrowing as a hypothesis involving target sample size and source-target mismatch.",
        "what_we_adopt": "Use target covariance uncertainty and source-target covariance discrepancy as healthy-only proxies.",
        "what_we_explicitly_do_not_claim": "TRI is not a theorem for anomaly recall or a complete transferability measure.",
    },
    {
        "paper": "Ben-David et al. domain adaptation bounds and negative-transfer theory",
        "core_theoretical_result": "Target risk can be bounded by source risk, distribution discrepancy, and unavoidable joint error.",
        "relevance_to_RACE": "Healthy source-target mismatch is a bias proxy, not guaranteed downstream fault utility.",
        "what_we_adopt": "Report discrepancy as a proxy and avoid claiming label-free fault identifiability.",
        "what_we_explicitly_do_not_claim": "Healthy covariance alignment proves anomaly separation.",
    },
    {
        "paper": "Ledoit and Wolf covariance shrinkage; Chen, Wiesel, and Hero OAS",
        "core_theoretical_result": "Shrinkage covariance estimators reduce high-dimensional covariance estimation risk for specified shrinkage targets.",
        "relevance_to_RACE": "RACE operates in d >> N commissioning regimes.",
        "what_we_adopt": "Primary covariance estimates use Ledoit-Wolf shrinkage and stabilized eigenvalues.",
        "what_we_explicitly_do_not_claim": "OAS or Ledoit-Wolf directly justify shrinking target covariance toward an arbitrary source covariance.",
    },
    {
        "paper": "Linear covariance shrinkage toward general or structured targets; empirical-Bayes covariance shrinkage; LOOCV shrinkage coefficient selection",
        "core_theoretical_result": "Several estimators study shrinkage toward predefined, structured, or data-selected covariance targets under Frobenius/MSE criteria.",
        "relevance_to_RACE": "These are nearest competitors to using an external source covariance as a reference target.",
        "what_we_adopt": "Treat source covariance as a candidate reference and report mismatch/uncertainty proxies before any anomaly join.",
        "what_we_explicitly_do_not_claim": "TRI is the first covariance-transfer framework or a validated optimal shrinkage estimator.",
    },
    {
        "paper": "Unsupervised domain-adaptation/model-selection limitations",
        "core_theoretical_result": "Unlabeled target data cannot generally select a model that is optimal for unknown labels.",
        "relevance_to_RACE": "Healthy-only TRI can assess normal-model transfer risk, not future fault geometry.",
        "what_we_adopt": "Freeze TRI before joining anomaly outcomes and audit leakage explicitly.",
        "what_we_explicitly_do_not_claim": "TRI is constructed to predict recall.",
    },
    {
        "paper": "M2N2 / new-normal anomaly detection and covariate-shift anomaly-detection work",
        "core_theoretical_result": "Normal-domain shifts can break anomaly detectors even when faults are unchanged.",
        "relevance_to_RACE": "Robotics commissioning is a normal-shift problem under sparse target healthy data.",
        "what_we_adopt": "Interpret TRI as a diagnostic for importing source normal covariance.",
        "what_we_explicitly_do_not_claim": "A low TRI guarantees safety-critical anomaly performance.",
    },
]


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _output_paths(output_dir: Path) -> dict[str, Path]:
    figures = output_dir / "figures"
    return {
        "manifest": output_dir / "tri_manifest.json",
        "per_source_seed": output_dir / "tri_per_source_seed.csv",
        "summary": output_dir / "tri_summary.csv",
        "correlations": output_dir / "tri_correlations.csv",
        "bootstrap_uncertainty": output_dir / "tri_bootstrap_uncertainty.csv",
        "uncertainty_estimators": output_dir / "tri_uncertainty_estimators.csv",
        "covariance_distances": output_dir / "tri_covariance_distances.csv",
        "joined_outcomes": output_dir / "tri_joined_original_race_outcomes.csv",
        "partition_audit": output_dir / "tri_partition_audit.csv",
        "optimal_blend": output_dir / "tri_optimal_blend.csv",
        "synthetic_sanity": output_dir / "tri_synthetic_sanity.csv",
        "figure_1": figures / "figure_1_tri_vs_covariance_delta_recall.png",
        "figure_2": figures / "figure_2_tri_vs_delta_fpr.png",
        "figure_3": figures / "figure_3_target_uncertainty_vs_N.png",
        "figure_4": figures / "figure_4_discrepancy_vs_transfer_gain.png",
        "figure_5": figures / "figure_5_optimal_blend_vs_N.png",
    }


def run_transfer_risk_index(
    *,
    data_path: Path,
    output_dir: Path,
    ablation_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    resamples: int,
) -> dict[str, Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)
    sanity = synthetic_sanity_checks(resamples=min(max(resamples, 8), 40))
    sanity.to_csv(paths["synthetic_sanity"], index=False)
    if not bool(sanity["all_expectations_pass"].all()):
        raise RuntimeError(
            "TRI synthetic sanity checks failed; refusing to compute robotics TRI until the estimator is fixed."
        )

    cycles = load_cycle_metadata(path=data_path)
    tri_rows: list[dict[str, object]] = []
    boot_rows: list[pd.DataFrame] = []
    uncertainty_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    partition_rows: list[dict[str, object]] = []
    blend_rows: list[dict[str, object]] = []

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
            print(f"Loading healthy features for TRI N={n}, seed={seed}: {len(selected_ids)} episodes...", flush=True)
            selected_cycles = load_cycles(path=data_path, signal_set="measured", episode_ids=selected_ids)
            features_by_episode = _feature_lookup(selected_cycles)
            source_all = _matrix(split.source_train, features_by_episode)
            target = _matrix(split.target_commissioning, features_by_episode)
            regimes = construct_source_regimes(
                source_episode_ids=episode_ids(split.source_train),
                source_features=source_all,
                target_episode_ids=episode_ids(split.target_commissioning),
                target_features=target,
                commissioning_size=n,
                seed=seed,
                subset_size=SOURCE_SUBSET_SIZE,
            )
            target_uncertainty, target_uncertainty_norm, target_estimate, bootstrap = covariance_bootstrap_uncertainty(
                target,
                resamples=resamples,
                rng_seed=_stable_seed(n, seed, "target_uncertainty"),
            )
            jackknife_uncertainty, jackknife_uncertainty_norm, jackknife = covariance_jackknife_uncertainty(target)
            diagonal_uncertainty, diagonal_uncertainty_norm = diagonal_gaussian_uncertainty_proxy(target)
            target_keys = {
                "N": int(n),
                "seed": int(seed),
                "target_group": f"target_setting_73_N{n}_seed{seed}",
            }
            boot_rows.append(bootstrap.assign(**target_keys))
            uncertainty_rows.append(
                {
                    **target_keys,
                    "primary_uncertainty_estimator": "bootstrap_ledoit_wolf_covariance_instability",
                    "target_covariance_estimation_uncertainty_proxy": target_uncertainty,
                    "target_covariance_estimation_uncertainty_proxy_normalized": target_uncertainty_norm,
                    "bootstrap_uncertainty": target_uncertainty,
                    "bootstrap_uncertainty_normalized": target_uncertainty_norm,
                    "jackknife_uncertainty": jackknife_uncertainty,
                    "jackknife_uncertainty_normalized": jackknife_uncertainty_norm,
                    "diagonal_gaussian_uncertainty_proxy": diagonal_uncertainty,
                    "diagonal_gaussian_uncertainty_proxy_normalized": diagonal_uncertainty_norm,
                    "jackknife_replicates": int(len(jackknife)),
                    "bootstrap_replicates": int(len(bootstrap)),
                }
            )
            target_boot_covs = bootstrap_covariance_estimates(
                target,
                resamples=resamples,
                rng_seed=_stable_seed(n, seed, "blend"),
            )
            for regime in regimes:
                source = _matrix_from_ids(regime.source_episode_ids, features_by_episode)
                keys = {
                    "N": int(n),
                    "seed": int(seed),
                    "source_pair_id": regime.source_pair_id,
                    "source_group": regime.source_group,
                    "target_group": regime.target_group,
                }
                assert_no_episode_leakage(
                    {
                        "source": regime.source_episode_ids,
                        "commissioning": episode_ids(split.target_commissioning),
                        "calibration": episode_ids(split.target_calibration),
                        "healthy_eval": episode_ids(split.target_normal_evaluation),
                        "anomaly_eval": episode_ids(split.target_anomaly_evaluation),
                    }
                )
                source_estimate = ledoit_wolf_covariance(source)
                result = compute_transfer_risk_index_from_estimates(
                    source_estimate,
                    target_estimate,
                    target_cov_uncertainty=target_uncertainty,
                    target_cov_uncertainty_normalized=target_uncertainty_norm,
                )
                tri_rows.append(
                    _tri_row(
                        keys,
                        result,
                        len(source),
                        target.shape[0],
                        target.shape[1],
                        jackknife_uncertainty=jackknife_uncertainty,
                        jackknife_uncertainty_norm=jackknife_uncertainty_norm,
                        diagonal_uncertainty=diagonal_uncertainty,
                        diagonal_uncertainty_norm=diagonal_uncertainty_norm,
                    )
                )
                distance_rows.append(_distance_row(keys, result))
                partition_rows.append(_partition_row(keys, split, regime))
                w_star, risk = optimal_bootstrap_blend_weight(
                    result.source.covariance,
                    result.target.covariance,
                    target_boot_covs,
                )
                blend_rows.append({**keys, "w_star": w_star, "bootstrap_covariance_risk": risk})

    tri = pd.DataFrame(tri_rows)
    bootstrap_uncertainty = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    uncertainty_estimators = pd.DataFrame(uncertainty_rows)
    distances = pd.DataFrame(distance_rows)
    partition_audit = pd.DataFrame(partition_rows)
    blend = pd.DataFrame(blend_rows)

    tri.to_csv(paths["per_source_seed"], index=False)
    bootstrap_uncertainty.to_csv(paths["bootstrap_uncertainty"], index=False)
    uncertainty_estimators.to_csv(paths["uncertainty_estimators"], index=False)
    distances.to_csv(paths["covariance_distances"], index=False)
    partition_audit.to_csv(paths["partition_audit"], index=False)
    blend.to_csv(paths["optimal_blend"], index=False)

    joined = _join_outcomes(tri, ablation_dir)
    joined.to_csv(paths["joined_outcomes"], index=False)
    _summary(tri, joined).to_csv(paths["summary"], index=False)
    _correlations(joined).to_csv(paths["correlations"], index=False)
    _write_figures(joined, tri, blend, paths)
    _write_manifest(paths, data_path, ablation_dir, n_values, seeds, resamples)
    return paths


def finalize_existing_transfer_risk_outputs(
    *,
    data_path: Path,
    output_dir: Path,
    ablation_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    resamples: int,
) -> dict[str, Path]:
    paths = _output_paths(output_dir)
    tri_path = paths["per_source_seed"]
    blend_path = paths["optimal_blend"]
    if not tri_path.exists():
        raise FileNotFoundError(f"Missing existing TRI table: {tri_path}")
    tri = pd.read_csv(tri_path)
    blend = pd.read_csv(blend_path) if blend_path.exists() else pd.DataFrame()
    joined = _join_outcomes(tri, ablation_dir)
    joined.to_csv(paths["joined_outcomes"], index=False)
    _summary(tri, joined).to_csv(paths["summary"], index=False)
    _correlations(joined).to_csv(paths["correlations"], index=False)
    _write_figures(joined, tri, blend, paths)
    _write_manifest(paths, data_path, ablation_dir, n_values, seeds, resamples)
    return paths


def _stable_seed(n: int, seed: int, label: str) -> int:
    digest = hashlib.sha256(f"tri-v1|{n}|{seed}|{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _tri_row(
    keys: dict[str, object],
    result,
    source_size: int,
    target_size: int,
    n_features: int,
    *,
    jackknife_uncertainty: float = np.nan,
    jackknife_uncertainty_norm: float = np.nan,
    diagonal_uncertainty: float = np.nan,
    diagonal_uncertainty_norm: float = np.nan,
) -> dict[str, object]:
    return {
        **keys,
        "source_size": int(source_size),
        "commissioning_size": int(target_size),
        "n_features": int(n_features),
        "target_cov_uncertainty": result.target_cov_uncertainty,
        "target_cov_uncertainty_normalized": result.target_cov_uncertainty_normalized,
        "target_covariance_estimation_uncertainty_proxy": result.target_cov_uncertainty,
        "target_covariance_estimation_uncertainty_proxy_normalized": result.target_cov_uncertainty_normalized,
        "jackknife_target_cov_uncertainty": float(jackknife_uncertainty),
        "jackknife_target_cov_uncertainty_normalized": float(jackknife_uncertainty_norm),
        "diagonal_gaussian_target_cov_uncertainty_proxy": float(diagonal_uncertainty),
        "diagonal_gaussian_target_cov_uncertainty_proxy_normalized": float(diagonal_uncertainty_norm),
        "source_target_cov_discrepancy": result.source_target_cov_discrepancy,
        "source_target_cov_discrepancy_normalized": result.source_target_cov_discrepancy_normalized,
        "source_target_covariance_discrepancy_proxy": result.source_target_cov_discrepancy,
        "source_target_covariance_discrepancy_proxy_normalized": result.source_target_cov_discrepancy_normalized,
        "TRI": result.tri,
        "target_cov_effective_rank": result.target.effective_rank,
        "source_cov_effective_rank": result.source.effective_rank,
        "target_shrinkage": result.target.shrinkage,
        "source_shrinkage": result.source.shrinkage,
        "target_condition_number": result.target.condition_number,
        "source_condition_number": result.source.condition_number,
        "target_min_eigenvalue": result.target.min_eigenvalue,
        "source_min_eigenvalue": result.source.min_eigenvalue,
        "target_max_eigenvalue": result.target.max_eigenvalue,
        "source_max_eigenvalue": result.source.max_eigenvalue,
        "top_1pct_direction_discrepancy_share": result.top_1pct_direction_discrepancy_share,
        "top_5pct_direction_discrepancy_share": result.top_5pct_direction_discrepancy_share,
        "top_10pct_direction_discrepancy_share": result.top_10pct_direction_discrepancy_share,
    }


def _distance_row(keys: dict[str, object], result) -> dict[str, object]:
    return {
        **keys,
        "primary_metric": PRIMARY_DISCREPANCY,
        "source_target_cov_discrepancy": result.source_target_cov_discrepancy,
        "source_target_cov_discrepancy_normalized": result.source_target_cov_discrepancy_normalized,
        "log_euclidean_distance": result.log_euclidean_distance,
        "bures_wasserstein_distance": result.bures_wasserstein_distance,
        "coral_distance": result.coral_distance,
        "subspace_principal_angle_distance": result.subspace_principal_angle_distance,
        "top_1pct_direction_discrepancy_share": result.top_1pct_direction_discrepancy_share,
        "top_5pct_direction_discrepancy_share": result.top_5pct_direction_discrepancy_share,
        "top_10pct_direction_discrepancy_share": result.top_10pct_direction_discrepancy_share,
    }


def _partition_row(keys: dict[str, object], split, regime) -> dict[str, object]:
    groups = {
        "source": regime.source_episode_ids,
        "commissioning": episode_ids(split.target_commissioning),
        "calibration": episode_ids(split.target_calibration),
        "healthy_eval": episode_ids(split.target_normal_evaluation),
        "anomaly_eval": episode_ids(split.target_anomaly_evaluation),
    }
    row = {**keys, "no_overlap": True}
    for name, ids in groups.items():
        row[f"{name}_count"] = len(ids)
        row[f"{name}_episode_ids"] = ";".join(map(str, ids))
    return row


def _join_outcomes(tri: pd.DataFrame, ablation_dir: Path) -> pd.DataFrame:
    keys = ["source_pair_id", "source_group", "target_group", "N", "seed"]
    deltas = pd.read_csv(ablation_dir / "original_race_paired_deltas_all_seeds.csv")
    scores = pd.read_csv(ablation_dir / "original_race_score_equivalence_all_seeds.csv")
    if "reference_detector" in deltas.columns:
        target_deltas = deltas[deltas["reference_detector"].eq("TargetOnly")].copy()
    else:
        target_deltas = deltas.copy()
    required_delta_columns = set(keys + ["candidate_detector", "delta_recall", "delta_FPR", "delta_AUROC"])
    missing_delta_columns = sorted(required_delta_columns - set(target_deltas.columns))
    if missing_delta_columns:
        raise ValueError(
            "Frozen paired-delta artifact is missing required columns: "
            + ", ".join(missing_delta_columns)
        )
    wide = target_deltas.pivot(index=keys, columns="candidate_detector", values=["delta_recall", "delta_FPR", "delta_AUROC"])
    wide.columns = [f"{metric}_{detector}" for metric, detector in wide.columns]
    wide = wide.reset_index()
    changed = scores[
        scores["reference_detector"].eq("TargetOnly")
        & scores["candidate_detector"].eq("CovarianceTransferOnly")
        & scores["score_split"].eq("eval")
    ][keys + ["number_changed_predictions"]].rename(
        columns={"number_changed_predictions": "changed_predictions_covariance_only"}
    )
    joined = tri.merge(wide, on=keys, how="left", validate="one_to_one")
    joined = joined.merge(changed, on=keys, how="left", validate="one_to_one")
    rename = {
        "delta_recall_OriginalRACE": "delta_recall_original_race",
        "delta_recall_CovarianceTransferOnly": "delta_recall_covariance_only",
        "delta_FPR_CovarianceTransferOnly": "delta_FPR_covariance_only",
        "delta_AUROC_CovarianceTransferOnly": "delta_AUROC_covariance_only",
        "delta_recall_TargetMeanSourceCovariance": "delta_recall_target_mean_source_covariance",
    }
    return joined.rename(columns=rename)


def _summary(tri: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, group in tri.groupby("N", sort=True):
        joined_group = joined[joined["N"].eq(n)]
        rows.append(
            {
                "N": n,
                "rows": len(group),
                "TRI_median": float(group["TRI"].median()),
                "TRI_min": float(group["TRI"].min()),
                "TRI_max": float(group["TRI"].max()),
                "target_cov_uncertainty_normalized_median": float(group["target_cov_uncertainty_normalized"].median()),
                "source_target_cov_discrepancy_normalized_median": float(group["source_target_cov_discrepancy_normalized"].median()),
                "joined_covariance_delta_recall_nonnull": int(joined_group["delta_recall_covariance_only"].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = [
        "delta_recall_covariance_only",
        "delta_FPR_covariance_only",
        "delta_AUROC_covariance_only",
        "changed_predictions_covariance_only",
    ]
    for y in outcomes:
        for x in ["TRI", "source_target_cov_discrepancy_normalized", "target_cov_uncertainty_normalized"]:
            clean = joined[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            ci_low, ci_high = _bootstrap_seed_corr(clean, joined.loc[clean.index, "seed"], x, y)
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "analysis_role": "primary" if x == "TRI" and y == "delta_recall_covariance_only" else "secondary",
                    "n": int(len(clean)),
                    "spearman": _safe_corr(clean, x, y, method="spearman"),
                    "pearson": _safe_corr(clean, x, y, method="pearson"),
                    "bootstrap_seed_level_ci_low": ci_low,
                    "bootstrap_seed_level_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_seed_corr(clean: pd.DataFrame, seeds: pd.Series, x: str, y: str) -> tuple[float, float]:
    unique = np.asarray(sorted(pd.unique(seeds)))
    if len(clean) < 3 or len(unique) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(4242)
    values = []
    for _ in range(1000):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(seeds.to_numpy() == s) for s in sampled])
        sample = clean.iloc[idx]
        corr = _safe_corr(sample, x, y, method="spearman")
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _safe_corr(frame: pd.DataFrame, x: str, y: str, *, method: str) -> float:
    clean = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return float("nan")
    if clean[x].nunique(dropna=True) < 2 or clean[y].nunique(dropna=True) < 2:
        return float("nan")
    return float(clean[x].corr(clean[y], method=method))


def _write_figures(joined: pd.DataFrame, tri: pd.DataFrame, blend: pd.DataFrame, paths: dict[str, Path]) -> None:
    _scatter(joined, "TRI", "delta_recall_covariance_only", paths["figure_1"])
    _scatter(joined, "TRI", "delta_FPR_covariance_only", paths["figure_2"])
    _scatter(tri, "N", "target_cov_uncertainty_normalized", paths["figure_3"])
    _scatter(joined, "source_target_cov_discrepancy_normalized", "delta_recall_covariance_only", paths["figure_4"])
    _scatter(blend, "N", "w_star", paths["figure_5"])


def _scatter(frame: pd.DataFrame, x: str, y: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(clean[x], clean[y], s=42)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_manifest(
    paths: dict[str, Path],
    data_path: Path,
    ablation_dir: Path,
    n_values: tuple[int, ...],
    seeds: tuple[int, ...],
    resamples: int,
) -> None:
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "dataset_path": str(data_path),
        "dataset_hash": _dataset_hash(data_path),
        "ablation_dir": str(ablation_dir),
        "exact_covariance_estimator": PRIMARY_COVARIANCE_ESTIMATOR,
        "bootstrap_method": "Pairs bootstrap over target healthy commissioning rows, recomputing Ledoit-Wolf covariance each replicate.",
        "number_of_resamples": int(resamples),
        "random_seed_policy": "SHA-256 deterministic seed per N/seed/source_pair_id.",
        "commissioning_grid": list(n_values),
        "seeds": list(seeds),
        "primary_covariance_discrepancy": PRIMARY_DISCREPANCY,
        "primary_target_uncertainty": PRIMARY_UNCERTAINTY,
        "primary_tri_definition": PRIMARY_INDEX,
        "tri_threshold_policy": "TRI is continuous only; TRI=1 is not used as a decision boundary.",
        "proxy_terminology": {
            "source_target_covariance_discrepancy_proxy": "Observed distance between source Ledoit-Wolf covariance and target commissioning Ledoit-Wolf covariance.",
            "target_covariance_estimation_uncertainty_proxy": "Resampling instability of target commissioning Ledoit-Wolf covariance, not exact downstream anomaly-risk variance.",
        },
        "uncertainty_estimators": {
            "primary": "bootstrap Ledoit-Wolf covariance instability",
            "secondary": [
                "leave-one-out jackknife Ledoit-Wolf covariance instability",
                "diagonal Gaussian normal-theory covariance uncertainty proxy",
            ],
        },
        "secondary_discrepancy_metrics": [
            "log-Euclidean covariance distance",
            "Bures/Wasserstein covariance distance",
            "CORAL covariance distance",
            "90%-energy subspace principal-angle distance",
            "top-direction discrepancy concentration shares",
        ],
        "primary_correlation_analysis": "Spearman correlation, with seed-level bootstrap CI where possible.",
        "predeclared_downstream_target": PRIMARY_DOWNSTREAM_TARGET,
        "definition_before_outcome_join": True,
        "anomaly_labels_enter_tri": False,
        "frozen_branch_policy": "Original RACE and SS-RACE are not modified or retuned; frozen Original RACE ablation outputs are downstream reference only.",
        "literature_to_design": LITERATURE_TO_DESIGN,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "outputs": {key: str(path) for key, path in paths.items()},
        "synthetic_sanity_required_before_robotics": True,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_N))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Use existing TRI tables in output-dir and only regenerate joined outcomes, summaries, figures, and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "ablation_dir": args.ablation_dir,
        "n_values": tuple(args.n),
        "seeds": tuple(args.seeds),
        "resamples": args.resamples,
    }
    if args.finalize_existing:
        finalize_existing_transfer_risk_outputs(**kwargs)
    else:
        run_transfer_risk_index(**kwargs)


if __name__ == "__main__":
    main()
