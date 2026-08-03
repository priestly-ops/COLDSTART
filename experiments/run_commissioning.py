from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation import (
    detector_factories,
    evaluate_detector,
)
from src.feature_extractor import extract_feature_matrix
from src.split_generator import create_experiment_split
from src.voraus_loader import load_cycles


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

RAW_RESULTS_PATH = (
    OUTPUT_DIRECTORY
    / "commissioning_seed_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "commissioning_summary.csv"
)

N_STAR_PATH = (
    OUTPUT_DIRECTORY
    / "commissioning_n_star.json"
)


# ---------------------------------------------------------------------------
# Fast validation configuration
# ---------------------------------------------------------------------------

COMMISSIONING_GRID = [10, 25, 50, 100]
SEEDS = list(range(20))

FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90

MAXIMUM_COMMISSIONING_SIZE = 100
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100

BOOTSTRAP_SAMPLES = 10_000
GLOBAL_SEED = 42

# Increment this string whenever the experimental protocol changes.
# It prevents old checkpoint rows from silently mixing with new results.
PROTOCOL_VERSION = "nested-split-conformal-v2"

np.random.seed(GLOBAL_SEED)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_mean_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = GLOBAL_SEED,
) -> tuple[float, float]:
    """Calculate a percentile-bootstrap confidence interval for the mean.

    The seed-level metric values are sampled with replacement. A mean is
    calculated for each bootstrap resample, and the requested percentile
    interval is returned.

    Args:
        values: One metric value per experimental seed.
        confidence: Confidence level, such as 0.95.
        bootstrap_samples: Number of bootstrap resamples.
        seed: Random seed used for deterministic bootstrap sampling.

    Returns:
        Lower and upper confidence limits for the mean.
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            f"Expected a 1D array, received shape {values.shape}."
        )

    if len(values) == 0:
        raise ValueError(
            "Cannot bootstrap an empty array."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Bootstrap values contain NaN or Inf."
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1."
        )

    if bootstrap_samples <= 0:
        raise ValueError(
            "bootstrap_samples must be positive."
        )

    rng = np.random.default_rng(seed)

    sampled_indices = rng.integers(
        low=0,
        high=len(values),
        size=(bootstrap_samples, len(values)),
    )

    bootstrap_means = values[
        sampled_indices
    ].mean(axis=1)

    alpha = 1.0 - confidence

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


# ---------------------------------------------------------------------------
# Checkpoint and resume helpers
# ---------------------------------------------------------------------------

CHECKPOINT_COLUMNS = [
    "protocol_version",
    "detector",
    "commissioning_size",
    "seed",
    "false_positive_rate",
    "recall",
    "success",
    "threshold",
    "retained_features",
    "target_weight",
]


def load_checkpoint() -> pd.DataFrame:
    """Load compatible completed experiment rows.

    An empty DataFrame is returned when no checkpoint exists. Rows produced
    by a different protocol version are rejected to prevent stale results
    from being mixed with the current experiment.
    """
    if not RAW_RESULTS_PATH.exists():
        return pd.DataFrame(
            columns=CHECKPOINT_COLUMNS
        )

    checkpoint = pd.read_csv(
        RAW_RESULTS_PATH
    )

    required_columns = {
        "protocol_version",
        "detector",
        "commissioning_size",
        "seed",
    }

    missing_columns = required_columns.difference(
        checkpoint.columns
    )

    if missing_columns:
        raise ValueError(
            "Existing checkpoint is incompatible with the current "
            f"runner. Missing columns: {sorted(missing_columns)}. "
            "Rename or remove the old result CSV before continuing."
        )

    protocol_values = set(
        checkpoint["protocol_version"]
        .dropna()
        .astype(str)
        .unique()
    )

    if protocol_values and protocol_values != {
        PROTOCOL_VERSION
    }:
        raise ValueError(
            "Existing checkpoint was generated with a different "
            f"protocol: {sorted(protocol_values)}. Current protocol: "
            f"{PROTOCOL_VERSION!r}. Rename or remove the old CSV."
        )

    return checkpoint


def completed_run_keys(
    checkpoint: pd.DataFrame,
) -> set[tuple[str, int, int]]:
    """Return detector, N, and seed keys already in the checkpoint."""
    if checkpoint.empty:
        return set()

    return {
        (
            str(row.detector),
            int(row.commissioning_size),
            int(row.seed),
        )
        for row in checkpoint.itertuples(
            index=False
        )
    }


def save_checkpoint(
    rows: list[dict[str, Any]],
) -> None:
    """Atomically save all completed detector runs.

    The data are first written to a temporary CSV. The temporary file is
    then moved over the checkpoint path, reducing the risk of a partially
    written result file after interruption.
    """
    dataframe = pd.DataFrame(rows)

    for column in CHECKPOINT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    dataframe = dataframe[
        CHECKPOINT_COLUMNS
    ].sort_values(
        [
            "commissioning_size",
            "seed",
            "detector",
        ]
    )

    temporary_path = (
        RAW_RESULTS_PATH.parent
        / f"{RAW_RESULTS_PATH.stem}.tmp.csv"
    )

    dataframe.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(
        RAW_RESULTS_PATH
    )


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def bootstrap_seed_for_group(
    detector_name: str,
    commissioning_size: int,
    metric_offset: int,
) -> int:
    """Create a deterministic bootstrap seed for one result group."""
    detector_value = sum(
        ord(character)
        for character in detector_name
    )

    return (
        GLOBAL_SEED
        + int(commissioning_size) * 100
        + detector_value
        + metric_offset
    )


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate seed-level results into commissioning summaries."""
    summary_rows: list[dict[str, Any]] = []

    grouped = results.groupby(
        [
            "detector",
            "commissioning_size",
        ],
        sort=True,
    )

    for (
        detector_name,
        commissioning_size,
    ), group in grouped:
        recall_values = group[
            "recall"
        ].to_numpy(dtype=np.float64)

        fpr_values = group[
            "false_positive_rate"
        ].to_numpy(dtype=np.float64)

        recall_lower, recall_upper = (
            bootstrap_mean_interval(
                recall_values,
                seed=bootstrap_seed_for_group(
                    detector_name=str(detector_name),
                    commissioning_size=int(
                        commissioning_size
                    ),
                    metric_offset=0,
                ),
            )
        )

        fpr_lower, fpr_upper = (
            bootstrap_mean_interval(
                fpr_values,
                seed=bootstrap_seed_for_group(
                    detector_name=str(detector_name),
                    commissioning_size=int(
                        commissioning_size
                    ),
                    metric_offset=1,
                ),
            )
        )

        summary_rows.append(
            {
                "detector": str(detector_name),
                "commissioning_size": int(
                    commissioning_size
                ),
                "recall_mean": float(
                    recall_values.mean()
                ),
                "recall_ci_lower": recall_lower,
                "recall_ci_upper": recall_upper,
                "fpr_mean": float(
                    fpr_values.mean()
                ),
                "fpr_ci_lower": fpr_lower,
                "fpr_ci_upper": fpr_upper,
                "success_rate": float(
                    group["success"]
                    .astype(bool)
                    .mean()
                ),
                "number_of_seeds": int(
                    len(group)
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    if not summary.empty:
        summary = summary.sort_values(
            [
                "detector",
                "commissioning_size",
            ]
        ).reset_index(drop=True)

    return summary


def estimate_n_star(
    summary: pd.DataFrame,
) -> dict[str, str | int]:
    """Estimate the smallest N satisfying both confidence constraints."""
    estimates: dict[str, str | int] = {}

    for detector_name in sorted(
        summary["detector"].unique()
    ):
        detector_summary = summary[
            summary["detector"]
            == detector_name
        ].sort_values(
            "commissioning_size"
        )

        qualifying_rows = detector_summary[
            (
                detector_summary[
                    "recall_ci_lower"
                ]
                >= RECALL_TARGET
            )
            & (
                detector_summary[
                    "fpr_ci_upper"
                ]
                <= FALSE_ALERT_BUDGET
            )
        ]

        if qualifying_rows.empty:
            estimates[str(detector_name)] = (
                f"Censored (>{max(COMMISSIONING_GRID)})"
            )
        else:
            estimates[str(detector_name)] = int(
                qualifying_rows.iloc[0][
                    "commissioning_size"
                ]
            )

    return estimates


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def verify_source_only_invariance(
    results: pd.DataFrame,
    tolerance: float = 1e-12,
) -> None:
    """Verify SourceOnly is invariant across N within each seed.

    Under the nested split protocol, SourceOnly receives the same source
    training data, target calibration data, and evaluation data at every N.
    Its threshold, FPR, and recall should therefore be identical across
    commissioning sizes for the same seed.
    """
    source_only = results[
        results["detector"] == "SourceOnly"
    ]

    if source_only.empty:
        raise RuntimeError(
            "No SourceOnly rows were found."
        )

    for seed, group in source_only.groupby(
        "seed",
        sort=True,
    ):
        expected_sizes = set(
            COMMISSIONING_GRID
        )

        found_sizes = set(
            group[
                "commissioning_size"
            ].astype(int)
        )

        if found_sizes != expected_sizes:
            # This can occur during an interrupted partial run.
            continue

        for metric in [
            "threshold",
            "false_positive_rate",
            "recall",
        ]:
            values = group[
                metric
            ].to_numpy(dtype=np.float64)

            if not np.allclose(
                values,
                values[0],
                atol=tolerance,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "SourceOnly invariance check failed for "
                    f"seed={seed}, metric={metric}: {values}"
                )

    print(
        "SourceOnly nested-split invariance check: PASS"
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run or resume the commissioning experiment."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    factories = detector_factories(
        false_alert_budget=FALSE_ALERT_BUDGET
    )

    checkpoint = load_checkpoint()

    if checkpoint.empty:
        result_rows: list[
            dict[str, Any]
        ] = []
    else:
        result_rows = checkpoint.to_dict(
            orient="records"
        )

    finished_keys = completed_run_keys(
        checkpoint
    )

    expected_run_keys = {
        (
            detector_name,
            commissioning_size,
            seed,
        )
        for commissioning_size
        in COMMISSIONING_GRID
        for seed in SEEDS
        for detector_name in factories
    }

    finished_current_keys = (
        finished_keys
        & expected_run_keys
    )

    total_runs = len(
        expected_run_keys
    )

    print("=" * 78)
    print("VORAUS-AD COMMISSIONING EXPERIMENT")
    print("=" * 78)
    print(
        f"Protocol:              {PROTOCOL_VERSION}"
    )
    print(
        f"Commissioning grid:    {COMMISSIONING_GRID}"
    )
    print(
        f"Seeds:                 {SEEDS}"
    )
    print(
        f"Calibration cycles:    {CALIBRATION_SIZE}"
    )
    print(
        f"Healthy eval cycles:   {NORMAL_EVALUATION_SIZE}"
    )
    print(
        f"False-alert budget:    {FALSE_ALERT_BUDGET:.3f}"
    )
    print(
        f"Recall target:         {RECALL_TARGET:.3f}"
    )
    print(
        f"Completed runs:        "
        f"{len(finished_current_keys)}/{total_runs}"
    )
    print("=" * 78)

    print("\nLoading voraus-AD cycles...")

    cycles = load_cycles(
        path=DATASET_PATH,
        signal_set="measured",
    )

    print(
        f"Loaded {len(cycles)} total cycles."
    )

    completed_counter = len(
        finished_current_keys
    )

    for seed_index, seed in enumerate(
        SEEDS,
        start=1,
    ):
        print(
            "\n"
            + "=" * 78
        )
        print(
            f"SEED {seed} "
            f"({seed_index}/{len(SEEDS)})"
        )
        print(
            "=" * 78
        )

        for commissioning_size in (
            COMMISSIONING_GRID
        ):
            print(
                f"\nPreparing nested split for "
                f"N={commissioning_size}, seed={seed}..."
            )

            split = create_experiment_split(
                cycles=cycles,
                commissioning_size=(
                    commissioning_size
                ),
                seed=seed,
                calibration_size=(
                    CALIBRATION_SIZE
                ),
                normal_evaluation_size=(
                    NORMAL_EVALUATION_SIZE
                ),
                maximum_commissioning_size=(
                    MAXIMUM_COMMISSIONING_SIZE
                ),
            )

            source_raw, source_ids = (
                extract_feature_matrix(
                    split.source_train
                )
            )

            target_raw, target_ids = (
                extract_feature_matrix(
                    split.target_commissioning
                )
            )

            calibration_raw, calibration_ids = (
                extract_feature_matrix(
                    split.target_calibration
                )
            )

            normal_raw, normal_ids = (
                extract_feature_matrix(
                    split.target_normal_evaluation
                )
            )

            anomaly_raw, anomaly_ids = (
                extract_feature_matrix(
                    split.target_anomaly_evaluation
                )
            )

            all_id_groups = {
                "source": set(
                    source_ids.tolist()
                ),
                "commissioning": set(
                    target_ids.tolist()
                ),
                "calibration": set(
                    calibration_ids.tolist()
                ),
                "normal_evaluation": set(
                    normal_ids.tolist()
                ),
                "anomaly_evaluation": set(
                    anomaly_ids.tolist()
                ),
            }

            group_names = list(
                all_id_groups
            )

            for first_index, first_name in enumerate(
                group_names
            ):
                for second_name in group_names[
                    first_index + 1:
                ]:
                    overlap = (
                        all_id_groups[first_name]
                        & all_id_groups[second_name]
                    )

                    if overlap:
                        raise RuntimeError(
                            "Feature-level leakage detected between "
                            f"{first_name} and {second_name}: "
                            f"{sorted(overlap)[:10]}"
                        )

            print(
                "Feature matrices:"
            )
            print(
                f"  Source:       {source_raw.shape}"
            )
            print(
                f"  Commission:   {target_raw.shape}"
            )
            print(
                f"  Calibration:  {calibration_raw.shape}"
            )
            print(
                f"  Normal eval:  {normal_raw.shape}"
            )
            print(
                f"  Anomaly eval: {anomaly_raw.shape}"
            )

            for (
                detector_name,
                factory,
            ) in factories.items():
                run_key = (
                    detector_name,
                    commissioning_size,
                    seed,
                )

                if run_key in finished_keys:
                    print(
                        "Skipping completed run: "
                        f"N={commissioning_size}, "
                        f"seed={seed}, "
                        f"detector={detector_name}"
                    )
                    continue

                completed_counter += 1

                print(
                    f"Processing N={commissioning_size}, "
                    f"seed={seed}, "
                    f"detector={detector_name} "
                    f"({completed_counter}/{total_runs})"
                )

                result = evaluate_detector(
                    detector_name=detector_name,
                    detector_factory=factory,
                    source_raw=source_raw,
                    target_raw=target_raw,
                    calibration_raw=calibration_raw,
                    normal_evaluation_raw=normal_raw,
                    anomaly_evaluation_raw=anomaly_raw,
                    commissioning_size=(
                        commissioning_size
                    ),
                    seed=seed,
                    false_alert_budget=(
                        FALSE_ALERT_BUDGET
                    ),
                    recall_target=(
                        RECALL_TARGET
                    ),
                )

                row = {
                    "protocol_version": (
                        PROTOCOL_VERSION
                    ),
                    "detector": result.detector,
                    "commissioning_size": (
                        result.commissioning_size
                    ),
                    "seed": result.seed,
                    "false_positive_rate": (
                        result.false_positive_rate
                    ),
                    "recall": result.recall,
                    "success": result.success,
                    "threshold": result.threshold,
                    "retained_features": (
                        result.retained_features
                    ),
                    "target_weight": (
                        result.target_weight
                    ),
                }

                result_rows.append(row)
                finished_keys.add(run_key)

                save_checkpoint(
                    result_rows
                )

                print(
                    f"  threshold={result.threshold:.6f}, "
                    f"FPR={result.false_positive_rate:.4f}, "
                    f"recall={result.recall:.4f}, "
                    f"success={result.success}"
                )

    results = pd.DataFrame(
        result_rows
    )

    if results.empty:
        raise RuntimeError(
            "No experiment results were produced."
        )

    # Keep only rows from the current protocol and configured validation run.
    results = results[
        (
            results["protocol_version"]
            == PROTOCOL_VERSION
        )
        & (
            results["commissioning_size"]
            .astype(int)
            .isin(COMMISSIONING_GRID)
        )
        & (
            results["seed"]
            .astype(int)
            .isin(SEEDS)
        )
        & (
            results["detector"]
            .isin(factories.keys())
        )
    ].copy()

    duplicate_mask = results.duplicated(
        subset=[
            "detector",
            "commissioning_size",
            "seed",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = results.loc[
            duplicate_mask,
            [
                "detector",
                "commissioning_size",
                "seed",
            ],
        ]

        raise RuntimeError(
            "Duplicate detector runs were found:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    expected_results = len(
        expected_run_keys
    )

    if len(results) != expected_results:
        missing_keys = expected_run_keys.difference(
            {
                (
                    str(row.detector),
                    int(row.commissioning_size),
                    int(row.seed),
                )
                for row in results.itertuples(
                    index=False
                )
            }
        )

        raise RuntimeError(
            f"Expected {expected_results} completed runs, "
            f"found {len(results)}. Missing examples: "
            f"{sorted(missing_keys)[:10]}"
        )

    verify_source_only_invariance(
        results
    )

    summary = build_summary(
        results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    n_star = estimate_n_star(
        summary
    )

    N_STAR_PATH.write_text(
        json.dumps(
            n_star,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 110
    )
    print(
        "COMMISSIONING EXPERIMENT COMPLETE"
    )
    print(
        "=" * 110
    )

    display_columns = [
        "detector",
        "commissioning_size",
        "recall_mean",
        "recall_ci_lower",
        "recall_ci_upper",
        "fpr_mean",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "success_rate",
        "number_of_seeds",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print("\nEstimated N*:")

    print(
        json.dumps(
            n_star,
            indent=2,
        )
    )

    print("\nOutput files:")
    print(
        f"  Seed results: {RAW_RESULTS_PATH}"
    )
    print(
        f"  Summary:      {SUMMARY_PATH}"
    )
    print(
        f"  N*:           {N_STAR_PATH}"
    )


if __name__ == "__main__":
    main()