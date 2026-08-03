from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from src.feature_extractor import extract_feature_matrix
from src.split_generator import create_experiment_split
from src.voraus_loader import RobotCycle, load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "outlier_diagnostics"
)

SEEDS_TO_CHECK = [4, 9, 19]
COMMISSIONING_SIZES = [10, 25, 50, 100]

CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100


def robust_z_scores(values: np.ndarray) -> np.ndarray:
    """Return median/MAD-based robust z-scores."""
    values = np.asarray(values, dtype=np.float64)

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad <= 1e-12:
        return np.zeros_like(values)

    return 0.67448975 * (values - median) / mad


def cycle_scalar_statistics(
    cycles: Sequence[RobotCycle],
) -> pd.DataFrame:
    """Calculate simple raw-signal diagnostics per cycle."""
    rows: list[dict[str, float | int]] = []

    for cycle in cycles:
        values = np.asarray(
            cycle.values,
            dtype=np.float64,
        )

        rows.append(
            {
                "episode_id": cycle.episode_id,
                "length": values.shape[0],
                "absolute_max": float(
                    np.max(np.abs(values))
                ),
                "global_mean": float(
                    np.mean(values)
                ),
                "global_std": float(
                    np.std(values)
                ),
                "mean_channel_std": float(
                    np.std(values, axis=0).mean()
                ),
                "maximum_channel_std": float(
                    np.std(values, axis=0).max()
                ),
                "total_variation": float(
                    np.abs(
                        np.diff(values, axis=0)
                    ).sum()
                ),
                "mean_total_variation": float(
                    np.abs(
                        np.diff(values, axis=0)
                    ).sum(axis=0).mean()
                ),
            }
        )

    frame = pd.DataFrame(rows)

    diagnostic_columns = [
        "absolute_max",
        "global_std",
        "mean_channel_std",
        "maximum_channel_std",
        "total_variation",
        "mean_total_variation",
    ]

    for column in diagnostic_columns:
        frame[
            f"{column}_robust_z"
        ] = robust_z_scores(
            frame[column].to_numpy()
        )

    robust_columns = [
        f"{column}_robust_z"
        for column in diagnostic_columns
    ]

    frame["maximum_robust_z"] = (
        frame[robust_columns]
        .abs()
        .max(axis=1)
    )

    return frame.sort_values(
        "maximum_robust_z",
        ascending=False,
    )


def commissioning_mahalanobis_scores(
    cycles: Sequence[RobotCycle],
) -> pd.DataFrame:
    """Score each commissioning cycle against the remaining cycles.

    This is a leave-one-out diagnostic. It helps identify a commissioning
    cycle that is unusually distant from the rest of the healthy target
    commissioning sample.
    """
    raw_features, episode_ids = (
        extract_feature_matrix(cycles)
    )

    scores = np.zeros(
        len(cycles),
        dtype=np.float64,
    )

    for held_out_index in range(
        len(cycles)
    ):
        keep_mask = np.ones(
            len(cycles),
            dtype=bool,
        )
        keep_mask[held_out_index] = False

        training = raw_features[keep_mask]
        held_out = raw_features[
            held_out_index:
            held_out_index + 1
        ]

        variances = np.var(
            training,
            axis=0,
        )

        feature_mask = variances > 1e-12

        training = training[:, feature_mask]
        held_out = held_out[:, feature_mask]

        scaler = StandardScaler()
        training_scaled = scaler.fit_transform(
            training
        )
        held_out_scaled = scaler.transform(
            held_out
        )

        model = LedoitWolf(
            assume_centered=False,
            store_precision=True,
        )
        model.fit(training_scaled)

        centered = (
            held_out_scaled[0]
            - model.location_
        )

        squared_distance = float(
            centered
            @ model.precision_
            @ centered
        )

        scores[held_out_index] = np.sqrt(
            max(squared_distance, 0.0)
        )

    robust_scores = robust_z_scores(scores)

    return pd.DataFrame(
        {
            "episode_id": episode_ids,
            "loo_mahalanobis": scores,
            "loo_mahalanobis_robust_z": (
                robust_scores
            ),
        }
    ).sort_values(
        "loo_mahalanobis",
        ascending=False,
    )


def plot_cycle(
    cycle: RobotCycle,
    seed: int,
    commissioning_size: int,
) -> None:
    """Save a compact raw-signal overview for one cycle."""
    values = np.asarray(
        cycle.values,
        dtype=np.float64,
    )

    # Plot only the 12 channels with the largest variation.
    channel_variation = np.std(
        values,
        axis=0,
    )

    selected_indices = np.argsort(
        channel_variation
    )[-12:]

    figure = plt.figure(
        figsize=(14, 8)
    )

    for index in selected_indices:
        plt.plot(
            values[:, index],
            linewidth=0.8,
            label=cycle.columns[index],
        )

    plt.title(
        f"Seed {seed}, N={commissioning_size}, "
        f"episode {cycle.episode_id}"
    )
    plt.xlabel("Time index")
    plt.ylabel("Raw signal value")
    plt.legend(
        fontsize=7,
        ncol=2,
    )
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / (
            f"seed_{seed}_N_{commissioning_size}"
            f"_episode_{cycle.episode_id}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycles = load_cycles(
        path=DATASET_PATH,
        signal_set="measured",
    )

    all_rows: list[pd.DataFrame] = []
    id_report: dict[str, dict[str, list[int]]] = {}

    for seed in SEEDS_TO_CHECK:
        id_report[str(seed)] = {}

        for commissioning_size in (
            COMMISSIONING_SIZES
        ):
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

            commissioning_cycles = list(
                split.target_commissioning
            )

            episode_ids = [
                cycle.episode_id
                for cycle in commissioning_cycles
            ]

            id_report[str(seed)][
                str(commissioning_size)
            ] = episode_ids

            raw_stats = cycle_scalar_statistics(
                commissioning_cycles
            )

            mahalanobis = (
                commissioning_mahalanobis_scores(
                    commissioning_cycles
                )
            )

            diagnostics = raw_stats.merge(
                mahalanobis,
                on="episode_id",
                how="left",
            )

            diagnostics.insert(
                0,
                "seed",
                seed,
            )

            diagnostics.insert(
                1,
                "commissioning_size",
                commissioning_size,
            )

            diagnostics["suspected_outlier"] = (
                (
                    diagnostics[
                        "maximum_robust_z"
                    ]
                    >= 5.0
                )
                | (
                    diagnostics[
                        "loo_mahalanobis_robust_z"
                    ]
                    >= 5.0
                )
            )

            diagnostics = (
                diagnostics.sort_values(
                    [
                        "suspected_outlier",
                        "loo_mahalanobis",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
            )

            all_rows.append(
                diagnostics
            )

            print(
                "\n"
                + "=" * 78
            )
            print(
                f"Seed={seed}, N={commissioning_size}"
            )
            print(
                "Commissioning episode IDs:"
            )
            print(episode_ids)

            print(
                "\nTop five suspicious cycles:"
            )
            print(
                diagnostics[
                    [
                        "episode_id",
                        "loo_mahalanobis",
                        "loo_mahalanobis_robust_z",
                        "maximum_robust_z",
                        "absolute_max",
                        "global_std",
                        "total_variation",
                        "suspected_outlier",
                    ]
                ]
                .head(5)
                .to_string(
                    index=False
                )
            )

            # Save plots for the top three suspicious cycles.
            top_episode_ids = (
                diagnostics[
                    "episode_id"
                ]
                .head(3)
                .astype(int)
                .tolist()
            )

            cycle_by_id = {
                cycle.episode_id: cycle
                for cycle
                in commissioning_cycles
            }

            for episode_id in top_episode_ids:
                plot_cycle(
                    cycle=cycle_by_id[
                        episode_id
                    ],
                    seed=seed,
                    commissioning_size=(
                        commissioning_size
                    ),
                )

    combined = pd.concat(
        all_rows,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT_DIR
        / "commissioning_outlier_diagnostics.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "commissioning_cycle_ids.json"
    ).write_text(
        json.dumps(
            id_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nDiagnostics written to:"
    )
    print(
        OUTPUT_DIR
        / "commissioning_outlier_diagnostics.csv"
    )
    print(
        OUTPUT_DIR
        / "commissioning_cycle_ids.json"
    )


if __name__ == "__main__":
    main()