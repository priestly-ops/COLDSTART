"""P0.1 healthy-only conditioning and redundancy audit for COLDSTART.

Purpose
-------
Diagnose why independent sparse precision recovery becomes unstable in the
low-target-sample/high-dimensional regime before changing the estimator.

This script uses only healthy source data and healthy target commissioning
executions. It never touches target anomalies, calibration outcomes, or anomaly
scores. The frozen evaluation split is reused only to guarantee that the target
commissioning pool is leakage-safe and comparable with the reviewer-facing P0
protocol.

Outputs
-------
- p01_conditioning_by_seed.csv
- p01_summary_by_N.csv
- p01_manifest.json

The diagnostics are descriptive; no hyperparameter is selected from these
outputs and no anomaly performance is consulted.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import STATISTIC_NAMES, extract_feature_batch
from src.precision_transfer_audit import robust_target_scale
from src.reproducibility import file_sha256, reproducibility_metadata
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycle_metadata, load_cycles


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p01_conditioning_audit"
DEFAULT_N_VALUES = (10, 25, 50, 100)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
FROZEN_EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
SOURCE_SUBSET_SIZE = 100
EPS = 1e-12


def _episode_ids(cycles) -> list[int]:
    return [int(c.episode_id) for c in cycles]


def _matrix_diagnostics(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Expected a >=2-row matrix, got {x.shape}.")

    n, p = x.shape
    xc = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(xc, axis=0, ddof=1)
    nonconstant = std > EPS
    zero_variance_fraction = float(np.mean(~nonconstant))

    singular_values = np.linalg.svd(xc, full_matrices=False, compute_uv=False)
    positive = singular_values[singular_values > EPS]
    numerical_rank = int(len(positive))
    if positive.size:
        covariance_condition_number = float((positive[0] / positive[-1]) ** 2)
        stable_rank = float(np.sum(singular_values**2) / max(singular_values[0] ** 2, EPS))
        energy = singular_values**2
        probs = energy / max(float(np.sum(energy)), EPS)
        effective_rank = float(np.exp(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))))
    else:
        covariance_condition_number = float("inf")
        stable_rank = 0.0
        effective_rank = 0.0

    corr_values = np.asarray([], dtype=np.float64)
    if int(np.sum(nonconstant)) >= 2:
        z = xc[:, nonconstant] / std[nonconstant]
        corr = np.corrcoef(z, rowvar=False)
        corr_values = np.abs(corr[np.triu_indices_from(corr, k=1)])
        corr_values = corr_values[np.isfinite(corr_values)]

    def frac_ge(threshold: float) -> float:
        return float(np.mean(corr_values >= threshold)) if corr_values.size else 0.0

    return {
        "n_samples": int(n),
        "n_features": int(p),
        "p_over_n": float(p / n),
        "numerical_rank": numerical_rank,
        "rank_fraction": float(numerical_rank / p),
        "effective_rank": effective_rank,
        "effective_rank_fraction": float(effective_rank / p),
        "stable_rank": stable_rank,
        "zero_variance_fraction": zero_variance_fraction,
        "raw_covariance_condition_number": covariance_condition_number,
        "median_abs_pair_correlation": float(np.median(corr_values)) if corr_values.size else float("nan"),
        "max_abs_pair_correlation": float(np.max(corr_values)) if corr_values.size else float("nan"),
        "fraction_abs_corr_ge_090": frac_ge(0.90),
        "fraction_abs_corr_ge_095": frac_ge(0.95),
        "fraction_abs_corr_ge_099": frac_ge(0.99),
    }


def _ledoit_wolf_diagnostics(x: np.ndarray) -> dict[str, float]:
    model = LedoitWolf(assume_centered=False).fit(np.asarray(x, dtype=np.float64))
    eig = np.linalg.eigvalsh(model.covariance_)
    return {
        "ledoit_wolf_shrinkage": float(model.shrinkage_),
        "ledoit_wolf_min_eigenvalue": float(np.min(eig)),
        "ledoit_wolf_condition_number": float(np.max(eig) / max(float(np.min(eig)), EPS)),
    }


def _within_signal_redundancy(x: np.ndarray, n_signals: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    n_stats = len(STATISTIC_NAMES)
    if x.shape[1] != n_signals * n_stats:
        raise ValueError("Feature width does not match signal-major statistic schema.")

    cube = x.reshape(x.shape[0], n_signals, n_stats)
    values: list[float] = []
    for signal_idx in range(n_signals):
        block = cube[:, signal_idx, :]
        std = np.std(block, axis=0, ddof=1)
        keep = std > EPS
        if int(np.sum(keep)) < 2:
            continue
        corr = np.corrcoef(block[:, keep], rowvar=False)
        upper = np.abs(corr[np.triu_indices_from(corr, k=1)])
        values.extend(float(v) for v in upper if np.isfinite(v))

    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {
            "within_signal_median_abs_corr": float("nan"),
            "within_signal_fraction_abs_corr_ge_090": 0.0,
            "within_signal_fraction_abs_corr_ge_095": 0.0,
            "within_signal_fraction_abs_corr_ge_099": 0.0,
        }
    return {
        "within_signal_median_abs_corr": float(np.median(arr)),
        "within_signal_fraction_abs_corr_ge_090": float(np.mean(arr >= 0.90)),
        "within_signal_fraction_abs_corr_ge_095": float(np.mean(arr >= 0.95)),
        "within_signal_fraction_abs_corr_ge_099": float(np.mean(arr >= 0.99)),
    }


def _bootstrap_scaler_stability(x: np.ndarray, seed: int, replicates: int) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = []
    scales = []
    for _ in range(replicates):
        idx = rng.choice(x.shape[0], size=x.shape[0], replace=True)
        xb = x[idx]
        center = np.median(xb, axis=0)
        q25 = np.quantile(xb, 0.25, axis=0)
        q75 = np.quantile(xb, 0.75, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std = np.std(xb, axis=0, ddof=1 if len(xb) > 1 else 0)
        scale = np.where(iqr_scale > EPS, iqr_scale, std)
        scale = np.where(scale > EPS, scale, 1.0)
        centers.append(center)
        scales.append(scale)

    centers = np.asarray(centers)
    scales = np.asarray(scales)
    reference_scale = np.median(scales, axis=0)
    center_mad = np.median(np.abs(centers - np.median(centers, axis=0)), axis=0)
    relative_center_jitter = center_mad / np.maximum(reference_scale, EPS)
    log_scale = np.log(np.maximum(scales, EPS))
    scale_log_mad = np.median(np.abs(log_scale - np.median(log_scale, axis=0)), axis=0)
    return {
        "bootstrap_center_jitter_median": float(np.median(relative_center_jitter)),
        "bootstrap_center_jitter_q90": float(np.quantile(relative_center_jitter, 0.90)),
        "bootstrap_log_scale_mad_median": float(np.median(scale_log_mad)),
        "bootstrap_log_scale_mad_q90": float(np.quantile(scale_log_mad, 0.90)),
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "p_over_n", "rank_fraction", "effective_rank_fraction",
        "raw_covariance_condition_number", "ledoit_wolf_condition_number",
        "ledoit_wolf_shrinkage", "fraction_abs_corr_ge_095",
        "within_signal_fraction_abs_corr_ge_095", "source_clip_fraction",
        "target_clip_fraction", "bootstrap_center_jitter_median",
        "bootstrap_log_scale_mad_median",
    ]
    rows = []
    for n, group in frame.groupby("N", sort=True):
        row = {"N": int(n), "seeds": int(group["seed"].nunique()), "rows": int(len(group))}
        for column in numeric:
            vals = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{column}_median"] = float(np.median(vals)) if vals.size else float("nan")
            row[f"{column}_min"] = float(np.min(vals)) if vals.size else float("nan")
            row[f"{column}_max"] = float(np.max(vals)) if vals.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    metadata = load_cycle_metadata(data_path)
    source_ids: set[int] = set()
    target_ids: set[int] = set()
    splits: dict[tuple[int, int], object] = {}

    for n in args.n_values:
        for seed in args.seeds:
            split = create_frozen_evaluation_split(
                metadata,
                commissioning_size=int(n),
                commissioning_seed=int(seed),
                evaluation_seed=FROZEN_EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )
            split.verify_no_overlap()
            splits[(int(n), int(seed))] = split
            source_ids.update(_episode_ids(split.source_train))
            target_ids.update(_episode_ids(split.target_commissioning))

    # Freeze one source subset for all N/seeds so only target commissioning size
    # changes across rows.
    source_all = np.asarray(sorted(source_ids), dtype=np.int64)
    rng = np.random.default_rng(FROZEN_EVALUATION_SEED)
    if len(source_all) > args.source_subset_size:
        source_subset = np.sort(rng.choice(source_all, size=args.source_subset_size, replace=False))
    else:
        source_subset = source_all

    needed = sorted(set(source_subset.tolist()) | target_ids)
    print(f"P0.1 loading {len(needed)} healthy episodes...", flush=True)
    cycles = load_cycles(data_path, episode_ids=needed)
    batch = extract_feature_batch(cycles)
    row_by_id = {int(eid): i for i, eid in enumerate(batch.episode_ids)}

    def matrix_for(ids: list[int] | np.ndarray) -> np.ndarray:
        return np.asarray([batch.features[row_by_id[int(eid)]] for eid in ids], dtype=np.float64)

    source_x = matrix_for(source_subset)
    rows = []
    for n in args.n_values:
        for seed in args.seeds:
            print(f"P0.1 N={n} seed={seed}", flush=True)
            split = splits[(int(n), int(seed))]
            target_episode_ids = _episode_ids(split.target_commissioning)
            target_x = matrix_for(target_episode_ids)
            source_z, target_z, scaling = robust_target_scale(source_x, target_x)

            row = {"N": int(n), "seed": int(seed)}
            row.update({f"target_{k}": v for k, v in _matrix_diagnostics(target_x).items()})
            # Backward-friendly aliases used in the summary table.
            for key, value in _matrix_diagnostics(target_x).items():
                row[key] = value
            row.update(_ledoit_wolf_diagnostics(target_x))
            row.update(_within_signal_redundancy(target_x, len(batch.signal_columns)))
            row.update(_bootstrap_scaler_stability(target_x, 100000 + int(seed) + 1000 * int(n), args.bootstrap_replicates))
            row.update(scaling)
            row["source_subset_size"] = int(len(source_x))
            row["source_transformed_effective_rank_fraction"] = _matrix_diagnostics(source_z)["effective_rank_fraction"]
            row["target_transformed_effective_rank_fraction"] = _matrix_diagnostics(target_z)["effective_rank_fraction"]
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["N", "seed"]).reset_index(drop=True)
    summary = _summary(frame)
    frame.to_csv(output_dir / "p01_conditioning_by_seed.csv", index=False)
    summary.to_csv(output_dir / "p01_summary_by_N.csv", index=False)

    manifest = {
        "protocol": "p01-conditioning-redundancy-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "healthy_only": True,
        "anomaly_outcomes_used": False,
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "n_values": [int(v) for v in args.n_values],
        "seeds": [int(v) for v in args.seeds],
        "source_subset_size": int(len(source_subset)),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "n_signals": int(len(batch.signal_columns)),
        "n_statistics_per_signal": int(len(STATISTIC_NAMES)),
        "n_features": int(batch.features.shape[1]),
        "reproducibility": reproducibility_metadata(),
    }
    (output_dir / "p01_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.1 outputs written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--source-subset-size", type=int, default=SOURCE_SUBSET_SIZE)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
