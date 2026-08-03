#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.voraus_loader import load_cycles


EPISODE_IDS = [1861, 1771, 1840, 1797, 1962, 1710]

SIGNALS = [
    "robot_current",
    "system_current",
    "robot_voltage",
    "joint_velocity_1",
    "joint_velocity_2",
    "motor_torque_1",
    "motor_torque_2",
]

PHASE_GRID = np.linspace(0.0, 1.0, 200)


def resample(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Signal must contain at least two samples.")

    source_phase = np.linspace(0.0, 1.0, len(values))

    return np.interp(
        PHASE_GRID,
        source_phase,
        values,
    )


def get_signal(cycle, signal: str) -> np.ndarray:
    if hasattr(cycle, "columns") and hasattr(cycle, "values"):
        if signal in cycle.columns:
            signal_index = cycle.columns.index(signal)
            return np.asarray(cycle.values[:, signal_index], dtype=np.float64)

    raise AttributeError(
        f"Could not locate signal {signal!r} in RobotCycle."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "context_analysis"
        / "figures"
        / "episode_fingerprints",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cycles = load_cycles(args.data_path)

    cycle_by_id = {
        int(cycle.episode_id): cycle
        for cycle in cycles
    }

    healthy_target = [
        cycle
        for cycle in cycles
        if int(cycle.setting) == 73
        and not bool(cycle.anomaly)
    ]

    missing = [
        episode_id
        for episode_id in EPISODE_IDS
        if episode_id not in cycle_by_id
    ]

    if missing:
        raise ValueError(
            f"Missing requested episodes: {missing}"
        )

    for signal in SIGNALS:
        healthy_trajectories = []

        for cycle in healthy_target:
            try:
                healthy_trajectories.append(
                    resample(get_signal(cycle, signal))
                )
            except (KeyError, ValueError):
                continue

        if not healthy_trajectories:
            print(f"Skipping unavailable signal: {signal}")
            continue

        healthy_matrix = np.vstack(healthy_trajectories)

        median = np.median(healthy_matrix, axis=0)
        q25 = np.quantile(healthy_matrix, 0.25, axis=0)
        q75 = np.quantile(healthy_matrix, 0.75, axis=0)

        for episode_id in EPISODE_IDS:
            cycle = cycle_by_id[episode_id]
            trajectory = resample(
                get_signal(cycle, signal)
            )

            plt.figure(figsize=(8, 4.5))
            plt.fill_between(
                PHASE_GRID,
                q25,
                q75,
                alpha=0.25,
                label="Healthy target IQR",
            )
            plt.plot(
                PHASE_GRID,
                median,
                linewidth=2,
                label="Healthy target median",
            )
            plt.plot(
                PHASE_GRID,
                trajectory,
                linewidth=1.5,
                label=f"Episode {episode_id}",
            )
            plt.xlabel("Normalized cycle phase")
            plt.ylabel(signal)
            plt.title(
                f"Healthy threshold episode {episode_id}: {signal}"
            )
            plt.legend()
            plt.tight_layout()

            output_path = (
                args.output_dir
                / f"episode_{episode_id}_{signal}.png"
            )

            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    print(
        f"Saved episode fingerprints to {args.output_dir}"
    )


if __name__ == "__main__":
    main()