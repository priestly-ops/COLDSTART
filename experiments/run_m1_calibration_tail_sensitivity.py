"""M1 reviewer-defense experiment: calibration-tail sensitivity.

This experiment intentionally changes ONLY two calibration quantities:

1. healthy calibration-set size M, and
2. nominal false-alert level alpha.

The detector fit, commissioning episodes, and healthy/anomaly evaluation
partitions remain frozen for a given (seed, N). In particular, the historical
voraus-AD protocol used 100 commissioning-pool episodes, 100 calibration
healthy episodes, and 100 healthy evaluation episodes, leaving 19 target
healthy episodes unused. M1 appends those 19 previously-unused episodes to the
calibration pool, which permits M in {50, 100, 119} without moving the frozen
100-episode healthy evaluation set.

Primary output:
    outputs/m1_calibration_tail/m1_seed_results.csv
    outputs/m1_calibration_tail/m1_summary.csv
    outputs/m1_calibration_tail/m1_rank_table.csv
    outputs/m1_calibration_tail/m1_manifest.json
    outputs/m1_calibration_tail/m1_recall_vs_calibration.png
    outputs/m1_calibration_tail/m1_fpr_vs_calibration.png

Run from the repository root:
    python experiments/run_m1_calibration_tail_sensitivity.py

For a quick smoke test:
    python experiments/run_m1_calibration_tail_sensitivity.py --seeds 0 --commissioning 100
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_tail import conformal_threshold_info
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.split_generator import SOURCE_SETTING, TARGET_SETTING
from src.voraus_loader import RobotCycle, load_cycles


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m1_calibration_tail"

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

PROTOCOL_VERSION = "m1-calibration-tail-frozen-eval-v1"
COMMISSIONING_POOL_SIZE = 100
BASE_CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
CALIBRATION_SIZES = (50, 100, 119)
ALPHAS = (0.005, 0.01, 0.02)
COMMISSIONING_GRID = (10, 25, 50, 100)
SEEDS = tuple(range(20))
RECALL_TARGET = 0.90
BOOTSTRAP_SAMPLES = 10_000

CATEGORY_NAMES = {
    1: "damaged_screw",
    2: "extra_component",
    3: "missing_screw",
    4: "damaged_thread",
}


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())


def _episode_ids(cycles: Iterable[RobotCycle]) -> set[int]:
    return {int(cycle.episode_id) for cycle in cycles}


def _verify_disjoint(groups: dict[str, tuple[RobotCycle, ...]]) -> None:
    ids = {name: _episode_ids(cycles) for name, cycles in groups.items()}
    names = list(ids)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = ids[first] & ids[second]
            if overlap:
                raise RuntimeError(
                    f"Leakage between {first} and {second}: {sorted(overlap)[:10]}"
                )


def make_frozen_m1_split(
    cycles: list[RobotCycle],
    commissioning_size: int,
    seed: int,
) -> dict[str, tuple[RobotCycle, ...]]:
    """Reconstruct the frozen original split and expose its 19 unused normals.

    Positions in the seed-specific target-healthy permutation are fixed as:
      [0:100)   commissioning pool
      [100:200) original calibration set
      [200:300) original healthy evaluation set
      [300:319) previously unused healthy set

    M1 calibration pools are prefixes of:
      original_calibration + previously_unused

    Therefore M=100 exactly reproduces the historical calibration/evaluation
    episode identities, while M=119 adds information without changing the
    evaluation set.
    """
    if not 1 <= commissioning_size <= COMMISSIONING_POOL_SIZE:
        raise ValueError(
            f"commissioning_size must be in [1, {COMMISSIONING_POOL_SIZE}]"
        )

    source_healthy = tuple(
        cycle for cycle in cycles if (not cycle.anomaly and cycle.setting == SOURCE_SETTING)
    )
    target_healthy = [
        cycle for cycle in cycles if (not cycle.anomaly and cycle.setting == TARGET_SETTING)
    ]
    anomalies = [cycle for cycle in cycles if cycle.anomaly]

    required = COMMISSIONING_POOL_SIZE + BASE_CALIBRATION_SIZE + HEALTHY_EVALUATION_SIZE
    if len(target_healthy) < required:
        raise ValueError(
            f"Need at least {required} target healthy episodes; found {len(target_healthy)}."
        )

    rng = np.random.default_rng(seed)
    target_order = rng.permutation(len(target_healthy))
    shuffled_target = [target_healthy[int(i)] for i in target_order]
    anomaly_order = rng.permutation(len(anomalies))
    shuffled_anomalies = tuple(anomalies[int(i)] for i in anomaly_order)

    commissioning_pool = tuple(shuffled_target[:COMMISSIONING_POOL_SIZE])
    original_calibration = tuple(shuffled_target[100:200])
    original_evaluation = tuple(shuffled_target[200:300])
    unused_healthy = tuple(shuffled_target[300:])

    extended_calibration = original_calibration + unused_healthy
    max_requested = max(CALIBRATION_SIZES)
    if len(extended_calibration) < max_requested:
        raise ValueError(
            f"Requested M={max_requested}, but only {len(extended_calibration)} frozen-compatible "
            "calibration episodes are available after preserving the original evaluation set."
        )

    split = {
        "source_train": source_healthy,
        "target_commissioning": commissioning_pool[:commissioning_size],
        "calibration_extended": extended_calibration,
        "healthy_evaluation": original_evaluation,
        "anomaly_evaluation": shuffled_anomalies,
    }
    _verify_disjoint(split)
    return split


def _bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    confidence: float = 0.95,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    means = values[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def _strict_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return (scores > threshold).astype(np.int64)


def _evaluate_one_fit(
    detector_name: str,
    detector_factory,
    split: dict[str, tuple[RobotCycle, ...]],
    commissioning_size: int,
    seed: int,
    calibration_sizes: tuple[int, ...],
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_raw, _ = extract_feature_matrix(split["source_train"])
    target_raw, _ = extract_feature_matrix(split["target_commissioning"])
    calibration_raw, calibration_ids = extract_feature_matrix(split["calibration_extended"])
    healthy_raw, healthy_ids = extract_feature_matrix(split["healthy_evaluation"])
    anomaly_raw, anomaly_ids = extract_feature_matrix(split["anomaly_evaluation"])

    detector, preprocessor, _, _ = fit_detector(
        detector_name=detector_name,
        detector_factory=detector_factory,
        source_raw=source_raw,
        target_raw=target_raw,
    )

    calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

    anomaly_categories = np.asarray(
        [cycle.category for cycle in split["anomaly_evaluation"]], dtype=np.int64
    )

    result_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []

    for calibration_size in calibration_sizes:
        if calibration_size > len(calibration_scores):
            continue
        prefix_scores = calibration_scores[:calibration_size]
        prefix_ids = calibration_ids[:calibration_size]

        for alpha in alphas:
            info = conformal_threshold_info(prefix_scores, alpha)
            threshold = info.strict_threshold
            healthy_predictions = _strict_predictions(healthy_scores, threshold)
            anomaly_predictions = _strict_predictions(anomaly_scores, threshold)

            fpr = float(np.mean(healthy_predictions == 1))
            recall = float(np.mean(anomaly_predictions == 1))

            base_row: dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "detector": detector_name,
                "commissioning_size": commissioning_size,
                "seed": seed,
                "calibration_size": calibration_size,
                "alpha": alpha,
                "raw_rank": info.raw_rank,
                "used_rank": info.used_rank,
                "finite_sample_feasible": info.finite_sample_feasible,
                "minimum_attainable_alpha": info.minimum_attainable_alpha,
                "threshold_is_maximum": info.threshold_is_maximum,
                "threshold": threshold,
                "legacy_clipped_threshold": info.legacy_clipped_threshold,
                "false_positive_rate": fpr,
                "recall": recall,
                "success": bool(recall >= RECALL_TARGET and fpr <= alpha),
                "retained_features": int(preprocessor.output_feature_count_),
                "calibration_first_episode_id": int(prefix_ids[0]),
                "calibration_last_episode_id": int(prefix_ids[-1]),
                "healthy_eval_count": int(len(healthy_ids)),
                "anomaly_eval_count": int(len(anomaly_ids)),
            }
            result_rows.append({**base_row, "fault_category": 0, "fault_name": "all_faults"})

            for category in sorted(np.unique(anomaly_categories)):
                mask = anomaly_categories == category
                category_recall = float(np.mean(anomaly_predictions[mask] == 1))
                result_rows.append(
                    {
                        **base_row,
                        "recall": category_recall,
                        "success": bool(category_recall >= RECALL_TARGET and fpr <= alpha),
                        "fault_category": int(category),
                        "fault_name": CATEGORY_NAMES.get(int(category), f"category_{int(category)}"),
                        "anomaly_eval_count": int(mask.sum()),
                    }
                )

            rank_rows.append(
                {
                    "calibration_size": calibration_size,
                    "alpha": alpha,
                    **asdict(info),
                }
            )

    return result_rows, rank_rows


def _make_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = [
        "detector",
        "commissioning_size",
        "calibration_size",
        "alpha",
        "fault_category",
        "fault_name",
        "finite_sample_feasible",
        "raw_rank",
        "threshold_is_maximum",
    ]
    for key, group in results.groupby(group_cols, dropna=False, sort=True):
        recall = group["recall"].to_numpy(np.float64)
        fpr = group["false_positive_rate"].to_numpy(np.float64)
        # Stable deterministic seed generated from the grouping values.
        seed_value = GLOBAL_SEED + sum(ord(ch) for ch in repr(key)) % 1_000_000
        recall_lo, recall_hi = _bootstrap_mean_ci(recall, seed=seed_value)
        fpr_lo, fpr_hi = _bootstrap_mean_ci(fpr, seed=seed_value + 1)

        row = dict(zip(group_cols, key))
        row.update(
            {
                "recall_mean": float(recall.mean()),
                "recall_ci_lower": recall_lo,
                "recall_ci_upper": recall_hi,
                "fpr_mean": float(fpr.mean()),
                "fpr_ci_lower": fpr_lo,
                "fpr_ci_upper": fpr_hi,
                "success_rate": float(group["success"].astype(bool).mean()),
                "threshold_mean": float(group["threshold"].replace([np.inf, -np.inf], np.nan).mean()),
                "number_of_seeds": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _plot_primary(summary: pd.DataFrame, output_dir: Path) -> None:
    primary = summary[
        (summary["fault_category"] == 0)
        & (summary["detector"] == "TargetOnly")
        & (summary["commissioning_size"] == summary["commissioning_size"].max())
    ].copy()
    if primary.empty:
        return

    for metric, ylabel, filename in [
        ("recall_mean", "Recall", "m1_recall_vs_calibration.png"),
        ("fpr_mean", "False-positive rate", "m1_fpr_vs_calibration.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for alpha, group in primary.groupby("alpha", sort=True):
            group = group.sort_values("calibration_size")
            ax.plot(
                group["calibration_size"],
                group[metric],
                marker="o",
                label=f"alpha={alpha:g}",
            )
        if metric == "recall_mean":
            ax.axhline(RECALL_TARGET, linestyle="--", linewidth=1.0, label="Recall target=0.90")
        ax.set_xlabel("Healthy calibration episodes (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"M1 calibration-tail sensitivity — TargetOnly, N={primary['commissioning_size'].iloc[0]}")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=_parse_int_list, default=SEEDS)
    parser.add_argument("--commissioning", type=_parse_int_list, default=COMMISSIONING_GRID)
    parser.add_argument("--calibration-sizes", type=_parse_int_list, default=CALIBRATION_SIZES)
    parser.add_argument("--alphas", type=_parse_float_list, default=ALPHAS)
    parser.add_argument(
        "--detectors",
        type=str,
        default="TargetOnly",
        help="Comma-separated detector names. M1 defaults to TargetOnly to keep the defense focused.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("M1 — CALIBRATION-TAIL SENSITIVITY")
    print("=" * 80)
    print(f"Protocol:           {PROTOCOL_VERSION}")
    print(f"Dataset:            {args.data_path}")
    print(f"Seeds:              {args.seeds}")
    print(f"Commissioning N:    {args.commissioning}")
    print(f"Calibration M:      {args.calibration_sizes}")
    print(f"Alpha:              {args.alphas}")
    print("Frozen healthy eval: positions [200:300) in each seed permutation")
    print("M=119 extension:     appends previously unused positions [300:319)")
    print("=" * 80)

    cycles = load_cycles(args.data_path, signal_set="measured")
    target_healthy_count = sum(
        1 for cycle in cycles if (not cycle.anomaly and cycle.setting == TARGET_SETTING)
    )
    print(f"Loaded {len(cycles)} episodes; target healthy={target_healthy_count}.")

    # Import here so the script remains explicit about which existing detector
    # implementations it reuses.
    from src.evaluation import detector_factories

    factory_map = detector_factories(false_alert_budget=0.01)
    requested_detectors = tuple(name.strip() for name in args.detectors.split(",") if name.strip())
    unknown = sorted(set(requested_detectors) - set(factory_map))
    if unknown:
        raise ValueError(f"Unknown detectors: {unknown}. Available: {sorted(factory_map)}")

    all_rows: list[dict[str, Any]] = []
    all_rank_rows: list[dict[str, Any]] = []

    total_fits = len(args.seeds) * len(args.commissioning) * len(requested_detectors)
    fit_index = 0

    for seed in args.seeds:
        for commissioning_size in args.commissioning:
            split = make_frozen_m1_split(cycles, commissioning_size, seed)

            for detector_name in requested_detectors:
                fit_index += 1
                print(
                    f"Fitting {detector_name}: N={commissioning_size}, seed={seed} "
                    f"({fit_index}/{total_fits})..."
                )
                rows, rank_rows = _evaluate_one_fit(
                    detector_name=detector_name,
                    detector_factory=factory_map[detector_name],
                    split=split,
                    commissioning_size=commissioning_size,
                    seed=seed,
                    calibration_sizes=tuple(args.calibration_sizes),
                    alphas=tuple(args.alphas),
                )
                all_rows.extend(rows)
                all_rank_rows.extend(rank_rows)

    results = pd.DataFrame(all_rows)
    if results.empty:
        raise RuntimeError("M1 produced no results.")

    seed_path = args.output_dir / "m1_seed_results.csv"
    results.to_csv(seed_path, index=False)

    summary = _make_summary(results)
    summary_path = args.output_dir / "m1_summary.csv"
    summary.to_csv(summary_path, index=False)

    rank_table = (
        pd.DataFrame(all_rank_rows)
        .drop_duplicates(subset=["calibration_size", "alpha"])
        .sort_values(["calibration_size", "alpha"])
        .reset_index(drop=True)
    )
    rank_path = args.output_dir / "m1_rank_table.csv"
    rank_table.to_csv(rank_path, index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "dataset_path": str(args.data_path),
        "detectors": requested_detectors,
        "seeds": list(args.seeds),
        "commissioning_sizes": list(args.commissioning),
        "calibration_sizes": list(args.calibration_sizes),
        "alphas": list(args.alphas),
        "recall_target": RECALL_TARGET,
        "commissioning_pool_positions": [0, 100],
        "original_calibration_positions": [100, 200],
        "frozen_healthy_evaluation_positions": [200, 300],
        "extra_calibration_positions": [300, target_healthy_count],
        "target_healthy_count": target_healthy_count,
        "strict_infeasible_policy": "threshold=+inf; report finite_sample_feasible=false",
        "legacy_clip_reported_for_audit_only": True,
    }
    (args.output_dir / "m1_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _plot_primary(summary, args.output_dir)

    print("\nFinite-sample rank table:")
    print(
        rank_table[
            [
                "calibration_size",
                "alpha",
                "raw_rank",
                "finite_sample_feasible",
                "minimum_attainable_alpha",
                "threshold_is_maximum",
            ]
        ].to_string(index=False)
    )

    primary = summary[
        (summary["fault_category"] == 0)
        & (summary["detector"] == "TargetOnly")
    ]
    print("\nPrimary all-fault summary:")
    print(
        primary[
            [
                "commissioning_size",
                "calibration_size",
                "alpha",
                "finite_sample_feasible",
                "recall_mean",
                "recall_ci_lower",
                "recall_ci_upper",
                "fpr_mean",
                "fpr_ci_lower",
                "fpr_ci_upper",
                "success_rate",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nM1 complete.")
    print(f"  Seed results: {seed_path}")
    print(f"  Summary:      {summary_path}")
    print(f"  Rank table:   {rank_path}")
    print(f"  Manifest:     {args.output_dir / 'm1_manifest.json'}")


if __name__ == "__main__":
    main()
