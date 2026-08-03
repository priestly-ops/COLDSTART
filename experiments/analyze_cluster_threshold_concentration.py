#!/usr/bin/env python3
"""
Diagnostic-only analysis of threshold-setting concentration across
exploratory healthy-only clusters.

This script does not fit detectors, change memberships, select clusters,
or tune any hyperparameters. It analyzes frozen per-episode outputs produced by:

    experiments/run_context_stratified_analysis.py
        --exploratory-clusters

Inputs
------
outputs/context_analysis/per_episode_scores.csv

Outputs
-------
outputs/context_analysis/cluster_threshold_concentration.csv
outputs/context_analysis/cluster_score_statistics.csv
outputs/context_analysis/cluster_threshold_tests.csv
outputs/context_analysis/figures/cluster_threshold_frequency.png
outputs/context_analysis/figures/cluster_threshold_rate.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "context_analysis"
FIGURE_DIR = INPUT_DIR / "figures"

SCORES_PATH = INPUT_DIR / "per_episode_scores.csv"

CLUSTER_COLUMN = "exploratory_cluster"
N_COLUMN = "commissioning_size"
THRESHOLD_FLAG_COLUMN = "is_threshold_episode"

EXPECTED_DETECTORS = {
    "SourceOnly",
    "TargetOnly",
    "Pooled",
    "RACE",
}

EXPECTED_GRID = {10, 25, 50, 100}


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    """Raise a useful error when required columns are missing."""
    missing = sorted(required - set(dataframe.columns))

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a defensive ratio without silently adding epsilon."""
    if denominator > 0:
        return float(numerator / denominator)

    if numerator > 0:
        return float(np.inf)

    return float(np.nan)


def summarize_scores(group: pd.DataFrame) -> pd.Series:
    """Compute descriptive calibration-score statistics."""
    values = np.sort(
        group["score"].to_numpy(dtype=np.float64)
    )

    if len(values) == 0:
        raise ValueError("Cannot summarize an empty score group.")

    maximum = float(values[-1])

    if len(values) > 1:
        second_maximum = float(values[-2])
    else:
        second_maximum = float("nan")

    median = float(np.median(values))

    return pd.Series(
        {
            "episode_count": int(len(values)),
            "mean_score": float(np.mean(values)),
            "std_score": (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else float("nan")
            ),
            "median_score": median,
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "second_max_score": second_maximum,
            "max_score": maximum,
            "max_minus_second": (
                maximum - second_maximum
                if np.isfinite(second_maximum)
                else float("nan")
            ),
            "max_to_second_ratio": (
                safe_ratio(maximum, second_maximum)
                if np.isfinite(second_maximum)
                else float("nan")
            ),
            "max_to_median_ratio": safe_ratio(
                maximum,
                median,
            ),
        }
    )


def validate_input(scores: pd.DataFrame) -> None:
    """Validate the saved score table before analysis."""
    required_columns = {
        "seed",
        N_COLUMN,
        "detector",
        "split",
        "episode_id",
        "score",
        "threshold",
        THRESHOLD_FLAG_COLUMN,
        CLUSTER_COLUMN,
    }

    require_columns(
        scores,
        required_columns,
        "per_episode_scores.csv",
    )

    if scores.empty:
        raise ValueError("per_episode_scores.csv is empty.")

    if scores[CLUSTER_COLUMN].isna().any():
        missing_count = int(
            scores[CLUSTER_COLUMN].isna().sum()
        )
        raise ValueError(
            f"{missing_count} rows are missing exploratory cluster "
            "assignments. Run the context analysis again with "
            "--exploratory-clusters."
        )

    detectors = set(
        scores["detector"].dropna().astype(str).unique()
    )
    missing_detectors = EXPECTED_DETECTORS - detectors

    if missing_detectors:
        raise ValueError(
            f"Missing expected detectors: "
            f"{sorted(missing_detectors)}"
        )

    grid = set(
        scores[N_COLUMN].dropna().astype(int).unique()
    )
    missing_grid = EXPECTED_GRID - grid

    if missing_grid:
        raise ValueError(
            f"Missing expected commissioning sizes: "
            f"{sorted(missing_grid)}"
        )

    numeric_scores = pd.to_numeric(
        scores["score"],
        errors="coerce",
    )

    if numeric_scores.isna().any():
        raise ValueError(
            "The score column contains nonnumeric or missing values."
        )

    if not np.isfinite(
        numeric_scores.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "The score column contains NaN or infinite values."
        )

    run_columns = [
        "seed",
        N_COLUMN,
        "detector",
    ]

    calibration = scores[
        scores["split"].eq("calibration")
    ].copy()

    calibration_counts = (
        calibration.groupby(run_columns)
        .size()
    )

    invalid_counts = calibration_counts[
        calibration_counts != 100
    ]

    if not invalid_counts.empty:
        raise AssertionError(
            "Expected exactly 100 calibration episodes per run. "
            f"Invalid groups:\n{invalid_counts.head(20)}"
        )

    threshold_counts = (
        calibration.groupby(run_columns)[
            THRESHOLD_FLAG_COLUMN
        ]
        .sum()
    )

    if (threshold_counts < 1).any():
        bad = threshold_counts[
            threshold_counts < 1
        ]
        raise AssertionError(
            "At least one threshold episode must be recorded "
            f"per run. Invalid groups:\n{bad.head(20)}"
        )


def build_score_statistics(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize calibration scores by detector, N, seed, and cluster."""
    group_columns = [
        "seed",
        N_COLUMN,
        "detector",
        CLUSTER_COLUMN,
    ]

    run_cluster_stats = (
        calibration.groupby(
            group_columns,
            sort=True,
            observed=True,
        )
        .apply(
            summarize_scores,
            include_groups=False,
        )
        .reset_index()
    )

    aggregate_columns = [
        "detector",
        N_COLUMN,
        CLUSTER_COLUMN,
    ]

    aggregate = (
        run_cluster_stats.groupby(
            aggregate_columns,
            sort=True,
            observed=True,
        )
        .agg(
            seed_count=("seed", "nunique"),
            mean_episode_count=(
                "episode_count",
                "mean",
            ),
            mean_score=("mean_score", "mean"),
            mean_median_score=(
                "median_score",
                "mean",
            ),
            mean_q90=("q90", "mean"),
            mean_q95=("q95", "mean"),
            mean_q99=("q99", "mean"),
            mean_max_score=("max_score", "mean"),
            median_max_score=("max_score", "median"),
            mean_second_max_score=(
                "second_max_score",
                "mean",
            ),
            mean_max_minus_second=(
                "max_minus_second",
                "mean",
            ),
            mean_max_to_second_ratio=(
                "max_to_second_ratio",
                "mean",
            ),
            median_max_to_second_ratio=(
                "max_to_second_ratio",
                "median",
            ),
        )
        .reset_index()
    )

    return aggregate


def build_threshold_concentration(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate threshold-setter counts and exposure-adjusted rates."""
    group_columns = [
        "detector",
        N_COLUMN,
        CLUSTER_COLUMN,
    ]

    available = (
        calibration.groupby(
            group_columns,
            sort=True,
            observed=True,
        )
        .size()
        .rename("calibration_episode_exposures")
        .reset_index()
    )

    threshold_rows = calibration[
        calibration[THRESHOLD_FLAG_COLUMN].astype(bool)
    ].copy()

    threshold_counts = (
        threshold_rows.groupby(
            group_columns,
            sort=True,
            observed=True,
        )
        .size()
        .rename("threshold_setter_count")
        .reset_index()
    )

    run_counts = (
        calibration[
            [
                "seed",
                N_COLUMN,
                "detector",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "detector",
                N_COLUMN,
            ],
            sort=True,
        )
        .size()
        .rename("run_count")
        .reset_index()
    )

    concentration = available.merge(
        threshold_counts,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )

    concentration[
        "threshold_setter_count"
    ] = (
        concentration["threshold_setter_count"]
        .fillna(0)
        .astype(int)
    )

    concentration = concentration.merge(
        run_counts,
        on=[
            "detector",
            N_COLUMN,
        ],
        how="left",
        validate="many_to_one",
    )

    concentration["threshold_setter_exposure_rate"] = (
        concentration["threshold_setter_count"]
        / concentration["calibration_episode_exposures"]
    )

    concentration["fraction_of_runs_set_by_cluster"] = (
        concentration["threshold_setter_count"]
        / concentration["run_count"]
    )

    total_setters = (
        concentration.groupby(
            [
                "detector",
                N_COLUMN,
            ]
        )["threshold_setter_count"]
        .transform("sum")
    )

    concentration["share_of_threshold_setters"] = np.where(
        total_setters > 0,
        concentration["threshold_setter_count"]
        / total_setters,
        np.nan,
    )

    return concentration.sort_values(
        [
            "detector",
            N_COLUMN,
            CLUSTER_COLUMN,
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_statistical_tests(
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run descriptive association tests between cluster membership and
    threshold-setter status.

    These tests are diagnostic only. They must not be used to select
    the cluster count, detector, or commissioning size.
    """
    rows: list[dict[str, object]] = []

    test_groups = [
        ("detector_and_N", ["detector", N_COLUMN]),
        ("detector_all_N", ["detector"]),
    ]

    for scope, columns in test_groups:
        grouper: str | list[str]

        if len(columns) == 1:
            grouper = columns[0]
        else:
            grouper = columns

        for keys, group in calibration.groupby(
            grouper,
            sort=True,
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)

            key_values = dict(
                zip(columns, keys, strict=True)
            )

            contingency = pd.crosstab(
                group[CLUSTER_COLUMN],
                group[THRESHOLD_FLAG_COLUMN].astype(bool),
            )

            # Ensure both outcome columns exist without pandas interpreting
            # [False, True] as a Boolean row mask.
            contingency = contingency.reindex(
                columns=[False, True],
                fill_value=0,
            ).sort_index()

            if (
                contingency.shape[0] >= 2
                and contingency.to_numpy().sum() > 0
                and np.all(
                    contingency.sum(axis=1).to_numpy() > 0
                )
            ):
                chi_square, p_value, dof, expected = (
                    chi2_contingency(
                        contingency.to_numpy()
                    )
                )

                minimum_expected = float(
                    np.min(expected)
                )
            else:
                chi_square = np.nan
                p_value = np.nan
                dof = np.nan
                minimum_expected = np.nan

            fisher_p_value = np.nan
            fisher_odds_ratio = np.nan

            # Fisher's exact test is defined directly only for 2x2.
            if contingency.shape == (2, 2):
                fisher_odds_ratio, fisher_p_value = (
                    fisher_exact(
                        contingency.to_numpy()
                    )
                )

            rows.append(
                {
                    "scope": scope,
                    **key_values,
                    "row_count": int(len(group)),
                    "cluster_count": int(
                        contingency.shape[0]
                    ),
                    "chi_square": float(chi_square),
                    "degrees_of_freedom": float(dof),
                    "chi_square_p_value": float(p_value),
                    "minimum_expected_count": (
                        minimum_expected
                    ),
                    "fisher_odds_ratio": float(
                        fisher_odds_ratio
                    ),
                    "fisher_p_value": float(
                        fisher_p_value
                    ),
                    "contingency_json": json.dumps(
                        {
                            str(int(cluster)): {
                                "not_threshold": int(
                                    contingency.loc[cluster, False]
                                ),
                                "threshold": int(
                                    contingency.loc[cluster, True]
                                ),
                            }
                            for cluster in contingency.index
                        },
                        sort_keys=True,
                    ),
                    "diagnostic_only": True,
                    "not_for_method_selection": True,
                }
            )

    return pd.DataFrame(rows)


def plot_threshold_counts(
    concentration: pd.DataFrame,
) -> None:
    """Plot raw threshold-setting run counts by cluster."""
    plot_data = (
        concentration.groupby(
            [
                "detector",
                CLUSTER_COLUMN,
            ],
            sort=True,
        )["threshold_setter_count"]
        .sum()
        .unstack(fill_value=0)
    )

    ax = plot_data.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_xlabel("Detector")
    ax.set_ylabel("Threshold-setting runs")
    ax.set_title(
        "Exploratory cluster membership of threshold-setting episodes"
    )
    ax.legend(
        title="Exploratory cluster"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR
        / "cluster_threshold_frequency.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_threshold_rates(
    concentration: pd.DataFrame,
) -> None:
    """Plot exposure-adjusted threshold-setter rates."""
    plot_data = (
        concentration.groupby(
            [
                "detector",
                CLUSTER_COLUMN,
            ],
            sort=True,
        )
        .agg(
            threshold_setter_count=(
                "threshold_setter_count",
                "sum",
            ),
            calibration_episode_exposures=(
                "calibration_episode_exposures",
                "sum",
            ),
        )
        .reset_index()
    )

    plot_data["rate"] = (
        plot_data["threshold_setter_count"]
        / plot_data["calibration_episode_exposures"]
    )

    pivot = plot_data.pivot(
        index="detector",
        columns=CLUSTER_COLUMN,
        values="rate",
    ).fillna(0.0)

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_xlabel("Detector")
    ax.set_ylabel(
        "Threshold-setter count / calibration exposure"
    )
    ax.set_title(
        "Exposure-adjusted threshold concentration by exploratory cluster"
    )
    ax.legend(
        title="Exploratory cluster"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR
        / "cluster_threshold_rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {SCORES_PATH}\n"
            "Run run_context_stratified_analysis.py with "
            "--exploratory-clusters first."
        )

    scores = pd.read_csv(SCORES_PATH)
    validate_input(scores)

    scores["seed"] = scores["seed"].astype(int)
    scores[N_COLUMN] = scores[N_COLUMN].astype(int)
    scores["episode_id"] = scores[
        "episode_id"
    ].astype(int)
    scores[CLUSTER_COLUMN] = scores[
        CLUSTER_COLUMN
    ].astype(int)
    scores[THRESHOLD_FLAG_COLUMN] = (
        scores[THRESHOLD_FLAG_COLUMN]
        .astype(int)
        .astype(bool)
    )

    calibration = scores[
        scores["split"].eq("calibration")
    ].copy()

    score_statistics = build_score_statistics(
        calibration
    )

    concentration = build_threshold_concentration(
        calibration
    )

    statistical_tests = build_statistical_tests(
        calibration
    )

    concentration_path = (
        INPUT_DIR
        / "cluster_threshold_concentration.csv"
    )
    statistics_path = (
        INPUT_DIR
        / "cluster_score_statistics.csv"
    )
    tests_path = (
        INPUT_DIR
        / "cluster_threshold_tests.csv"
    )

    concentration.to_csv(
        concentration_path,
        index=False,
        float_format="%.12g",
    )
    score_statistics.to_csv(
        statistics_path,
        index=False,
        float_format="%.12g",
    )
    statistical_tests.to_csv(
        tests_path,
        index=False,
        float_format="%.12g",
    )

    plot_threshold_counts(concentration)
    plot_threshold_rates(concentration)

    print("\n=== Cluster Threshold Concentration ===")
    print(concentration.to_string(index=False))

    print("\n=== Cluster Score Statistics ===")
    print(score_statistics.to_string(index=False))

    print("\n=== Diagnostic Association Tests ===")
    print(statistical_tests.to_string(index=False))

    print("\nSaved:")
    print(f"  {concentration_path}")
    print(f"  {statistics_path}")
    print(f"  {tests_path}")
    print(
        "  "
        + str(
            FIGURE_DIR
            / "cluster_threshold_frequency.png"
        )
    )
    print(
        "  "
        + str(
            FIGURE_DIR
            / "cluster_threshold_rate.png"
        )
    )


if __name__ == "__main__":
    main()