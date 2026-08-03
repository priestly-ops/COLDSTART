from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.voraus_loader import RobotCycle


SOURCE_SETTING = 72  # PRE_A
TARGET_SETTING = 73  # PRE_B


@dataclass(frozen=True)
class ExperimentSplit:
    source_train: tuple[RobotCycle, ...]
    target_commissioning: tuple[RobotCycle, ...]
    target_calibration: tuple[RobotCycle, ...]
    target_normal_evaluation: tuple[RobotCycle, ...]
    target_anomaly_evaluation: tuple[RobotCycle, ...]

    def verify_no_overlap(self) -> None:
        groups = {
            "source_train": self.source_train,
            "target_commissioning": self.target_commissioning,
            "target_calibration": self.target_calibration,
            "target_normal_evaluation": self.target_normal_evaluation,
            "target_anomaly_evaluation": self.target_anomaly_evaluation,
        }

        id_sets = {
            name: {cycle.episode_id for cycle in cycles}
            for name, cycles in groups.items()
        }

        names = list(id_sets)

        for index, first_name in enumerate(names):
            for second_name in names[index + 1 :]:
                overlap = id_sets[first_name] & id_sets[second_name]

                if overlap:
                    raise RuntimeError(
                        f"Data leakage between {first_name} and "
                        f"{second_name}: {sorted(overlap)[:10]}"
                    )


def _shuffle(
    cycles: Sequence[RobotCycle],
    rng: np.random.Generator,
) -> list[RobotCycle]:
    indices = rng.permutation(len(cycles))
    return [cycles[int(index)] for index in indices]


def create_experiment_split(
    cycles: Sequence[RobotCycle],
    commissioning_size: int,
    seed: int,
    calibration_size: int = 100,
    normal_evaluation_size: int = 100,
    maximum_commissioning_size: int = 100,
) -> ExperimentSplit:
    """Create nested source-target commissioning splits.

    For a given seed, the target cycles are shuffled once. The first
    maximum_commissioning_size cycles form a commissioning pool. Smaller N
    values use prefixes of that same pool. Calibration and evaluation sets
    therefore remain fixed across commissioning sizes.
    """
    if commissioning_size <= 0:
        raise ValueError(
            "commissioning_size must be positive."
        )

    if commissioning_size > maximum_commissioning_size:
        raise ValueError(
            "commissioning_size cannot exceed "
            "maximum_commissioning_size."
        )

    if calibration_size <= 0:
        raise ValueError(
            "calibration_size must be positive."
        )

    if normal_evaluation_size <= 0:
        raise ValueError(
            "normal_evaluation_size must be positive."
        )

    source_healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly
        and cycle.setting == SOURCE_SETTING
    ]

    target_healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly
        and cycle.setting == TARGET_SETTING
    ]

    anomalous = [
        cycle
        for cycle in cycles
        if cycle.anomaly
    ]

    required_target_cycles = (
        maximum_commissioning_size
        + calibration_size
        + normal_evaluation_size
    )

    if len(target_healthy) < required_target_cycles:
        raise ValueError(
            f"Need {required_target_cycles} target healthy cycles, "
            f"but only {len(target_healthy)} are available."
        )

    rng = np.random.default_rng(seed)

    target_indices = rng.permutation(
        len(target_healthy)
    )

    shuffled_target = [
        target_healthy[int(index)]
        for index in target_indices
    ]

    anomaly_indices = rng.permutation(
        len(anomalous)
    )

    shuffled_anomalies = [
        anomalous[int(index)]
        for index in anomaly_indices
    ]

    commissioning_pool_end = (
        maximum_commissioning_size
    )

    calibration_end = (
        commissioning_pool_end
        + calibration_size
    )

    evaluation_end = (
        calibration_end
        + normal_evaluation_size
    )

    commissioning_pool = shuffled_target[
        :commissioning_pool_end
    ]

    split = ExperimentSplit(
        source_train=tuple(source_healthy),

        target_commissioning=tuple(
            commissioning_pool[
                :commissioning_size
            ]
        ),

        target_calibration=tuple(
            shuffled_target[
                commissioning_pool_end:
                calibration_end
            ]
        ),

        target_normal_evaluation=tuple(
            shuffled_target[
                calibration_end:
                evaluation_end
            ]
        ),

        target_anomaly_evaluation=tuple(
            shuffled_anomalies
        ),
    )

    split.verify_no_overlap()
    return split