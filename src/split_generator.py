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
    """Create the legacy per-seed commissioning split.

    NOTE
    ----
    This function is preserved for backward compatibility with existing
    experiments. Because `seed` shuffles the entire target-healthy pool, its
    calibration and normal-evaluation sets change across seeds.

    Reviewer-facing frozen-evaluation experiments should use
    `create_frozen_evaluation_split()` below.
    """
    if commissioning_size <= 0:
        raise ValueError("commissioning_size must be positive.")

    if commissioning_size > maximum_commissioning_size:
        raise ValueError(
            "commissioning_size cannot exceed maximum_commissioning_size."
        )

    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive.")

    if normal_evaluation_size <= 0:
        raise ValueError("normal_evaluation_size must be positive.")

    source_healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly and cycle.setting == SOURCE_SETTING
    ]

    target_healthy = [
        cycle
        for cycle in cycles
        if not cycle.anomaly and cycle.setting == TARGET_SETTING
    ]

    anomalous = [cycle for cycle in cycles if cycle.anomaly]

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

    target_indices = rng.permutation(len(target_healthy))
    shuffled_target = [
        target_healthy[int(index)] for index in target_indices
    ]

    anomaly_indices = rng.permutation(len(anomalous))
    shuffled_anomalies = [
        anomalous[int(index)] for index in anomaly_indices
    ]

    commissioning_pool_end = maximum_commissioning_size
    calibration_end = commissioning_pool_end + calibration_size
    evaluation_end = calibration_end + normal_evaluation_size

    commissioning_pool = shuffled_target[:commissioning_pool_end]

    split = ExperimentSplit(
        source_train=tuple(source_healthy),
        target_commissioning=tuple(
            commissioning_pool[:commissioning_size]
        ),
        target_calibration=tuple(
            shuffled_target[commissioning_pool_end:calibration_end]
        ),
        target_normal_evaluation=tuple(
            shuffled_target[calibration_end:evaluation_end]
        ),
        target_anomaly_evaluation=tuple(shuffled_anomalies),
    )

    split.verify_no_overlap()
    return split


def create_frozen_evaluation_split(
    cycles: Sequence[RobotCycle],
    commissioning_size: int,
    commissioning_seed: int,
    *,
    evaluation_seed: int = 42,
    calibration_size: int = 100,
    normal_evaluation_size: int = 100,
    maximum_commissioning_size: int = 100,
) -> ExperimentSplit:
    """Create a leakage-safe split with calibration/evaluation frozen across seeds.

    Design
    ------
    1. A single `evaluation_seed` deterministically selects the target
       calibration and healthy-evaluation episodes.
    2. Those fixed episodes are removed from the target-healthy pool.
    3. `commissioning_seed` shuffles ONLY the remaining healthy episodes.
       Therefore commissioning composition changes across seeds while
       calibration and evaluation remain identical.
    4. All anomalous episodes are used for anomaly evaluation in a canonical
       episode-ID order. Their set and order therefore remain identical across
       all commissioning seeds and N values.

    This is the split that should be used for M2-v2.1 and other analyses whose
    seed-to-seed estimand is *commissioning-set sensitivity only*.

    Important
    ---------
    With 319 target-healthy episodes and calibration=evaluation=100 each,
    119 target-healthy episodes remain available for commissioning. Thus an
    N=100 commissioning sample can still vary across seeds rather than becoming
    the same fixed 100 episodes every time.
    """
    if commissioning_size <= 0:
        raise ValueError("commissioning_size must be positive.")

    if commissioning_size > maximum_commissioning_size:
        raise ValueError(
            "commissioning_size cannot exceed maximum_commissioning_size."
        )

    if calibration_size <= 0:
        raise ValueError("calibration_size must be positive.")

    if normal_evaluation_size <= 0:
        raise ValueError("normal_evaluation_size must be positive.")

    source_healthy = sorted(
        (
            cycle
            for cycle in cycles
            if not cycle.anomaly and cycle.setting == SOURCE_SETTING
        ),
        key=lambda c: c.episode_id,
    )

    target_healthy = sorted(
        (
            cycle
            for cycle in cycles
            if not cycle.anomaly and cycle.setting == TARGET_SETTING
        ),
        key=lambda c: c.episode_id,
    )

    anomalous = sorted(
        (cycle for cycle in cycles if cycle.anomaly),
        key=lambda c: c.episode_id,
    )

    required_fixed = calibration_size + normal_evaluation_size
    required_total = required_fixed + maximum_commissioning_size

    if len(target_healthy) < required_total:
        raise ValueError(
            f"Need at least {required_total} target healthy cycles, "
            f"but only {len(target_healthy)} are available."
        )

    # Select the fixed calibration/evaluation population once.
    eval_rng = np.random.default_rng(evaluation_seed)
    eval_perm = eval_rng.permutation(len(target_healthy))
    fixed_order = [target_healthy[int(i)] for i in eval_perm]

    fixed_calibration = fixed_order[:calibration_size]
    fixed_healthy_evaluation = fixed_order[
        calibration_size:
        calibration_size + normal_evaluation_size
    ]

    fixed_ids = {
        c.episode_id
        for c in fixed_calibration + fixed_healthy_evaluation
    }

    remaining_healthy = [
        c for c in target_healthy if c.episode_id not in fixed_ids
    ]

    if len(remaining_healthy) < maximum_commissioning_size:
        raise ValueError(
            "Insufficient target-healthy episodes remain after freezing "
            f"calibration/evaluation: need {maximum_commissioning_size}, "
            f"have {len(remaining_healthy)}."
        )

    # Only commissioning composition varies with commissioning_seed.
    commission_rng = np.random.default_rng(commissioning_seed)
    commission_perm = commission_rng.permutation(len(remaining_healthy))
    commissioning_order = [
        remaining_healthy[int(i)] for i in commission_perm
    ]

    split = ExperimentSplit(
        source_train=tuple(source_healthy),
        target_commissioning=tuple(
            commissioning_order[:commissioning_size]
        ),
        target_calibration=tuple(
            sorted(fixed_calibration, key=lambda c: c.episode_id)
        ),
        target_normal_evaluation=tuple(
            sorted(fixed_healthy_evaluation, key=lambda c: c.episode_id)
        ),
        target_anomaly_evaluation=tuple(anomalous),
    )

    split.verify_no_overlap()
    return split