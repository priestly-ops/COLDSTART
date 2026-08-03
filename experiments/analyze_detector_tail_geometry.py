#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "outputs" / "context_analysis"
FIGURE_DIR = INPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

N_COLUMN = "commissioning_size"
THRESHOLD_FLAG_COLUMN = "is_threshold_episode"


def tail_statistics(group: pd.DataFrame) -> pd.Series:
    scores = np.sort(
        group["score"].to_numpy(dtype=np.float64)
    )

    maximum = float(scores[-1])
    second = float(scores[-2])
    q95 = float(np.quantile(scores, 0.95))
    q99 = float(np.quantile(scores, 0.99))
    median = float(np.median(scores))

    return pd.Series(
        {
            "count": len(scores),
            "median_score": median,
            "q95": q95,
            "q99": q99,
            "second_max": second,
            "maximum": maximum,
            "max_minus_second": maximum - second,
            "max_to_second_ratio": (
                maximum / second
                if second > 0
                else np.inf
            ),
            "max_to_median_ratio": (
                maximum / median
                if median > 0
                else np.inf
            ),
        }
    )


def main() -> None:
    scores = pd.read_csv(
        INPUT_DIR / "per_episode_scores.csv"
    )

    calibration = scores[
        scores["split"].eq("calibration")
    ].copy()

    run_columns = [
        "seed",
        N_COLUMN,
        "detector",
    ]

    tail = (
        calibration.groupby(run_columns)
        .apply(tail_statistics)
        .reset_index()
    )

    tail.to_csv(
        INPUT_DIR / "detector_tail_geometry.csv",
        index=False,
    )

    detector_summary = (
        tail.groupby(["detector", N_COLUMN])
        .agg(
            mean_max_to_second=(
                "max_to_second_ratio",
                "mean",
            ),
            median_max_to_second=(
                "max_to_second_ratio",
                "median",
            ),
            mean_max_minus_second=(
                "max_minus_second",
                "mean",
            ),
            mean_max_to_median=(
                "max_to_median_ratio",
                "mean",
            ),
        )
        .reset_index()
    )

    detector_summary.to_csv(
        INPUT_DIR / "detector_tail_geometry_summary.csv",
        index=False,
    )

    for metric, ylabel, filename in [
        (
            "max_to_second_ratio",
            "Maximum / second-largest calibration score",
            "tail_ratio_by_detector.png",
        ),
        (
            "max_minus_second",
            "Maximum − second-largest score",
            "tail_gap_by_detector.png",
        ),
    ]:
        plt.figure(figsize=(9, 5))

        positions = []
        values = []
        labels = []

        detectors = sorted(tail["detector"].unique())
        grid = sorted(tail[N_COLUMN].unique())

        position = 0

        for detector in detectors:
            for n in grid:
                subset = tail[
                    tail["detector"].eq(detector)
                    & tail[N_COLUMN].eq(n)
                ][metric].replace(
                    [np.inf, -np.inf],
                    np.nan,
                ).dropna()

                positions.append(position)
                values.append(subset.to_numpy())
                labels.append(f"{detector}\nN={n}")
                position += 1

            position += 0.5

        plt.boxplot(
            values,
            positions=positions,
            widths=0.6,
            showfliers=True,
        )
        plt.xticks(
            positions,
            labels,
            rotation=45,
            ha="right",
        )
        plt.ylabel(ylabel)
        plt.title(
            "Calibration-tail geometry across frozen runs"
        )
        plt.tight_layout()
        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    recurring = (
        calibration[
            calibration[THRESHOLD_FLAG_COLUMN].astype(bool)
        ]
        .groupby(["detector", "episode_id"])
        .size()
        .rename("threshold_setter_count")
        .reset_index()
        .sort_values(
            ["detector", "threshold_setter_count"],
            ascending=[True, False],
        )
    )

    recurring.to_csv(
        INPUT_DIR
        / "detector_threshold_episode_frequency.csv",
        index=False,
    )

    print("\nDetector tail summary:")
    print(detector_summary.to_string(index=False))


if __name__ == "__main__":
    main()