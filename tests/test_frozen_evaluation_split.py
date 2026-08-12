import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import RobotCycle


def _cycle(
    episode_id: int,
    *,
    anomaly: bool,
    setting: int,
    category: int = 12,
) -> RobotCycle:
    return RobotCycle(
        episode_id=episode_id,
        values=np.asarray([[0.0], [1.0]], dtype=np.float64),
        columns=("signal",),
        anomaly=anomaly,
        category=category,
        setting=setting,
    )


def _synthetic_cycles():
    cycles = []

    # Source healthy.
    for i in range(20):
        cycles.append(
            _cycle(
                1000 + i,
                anomaly=False,
                setting=72,
            )
        )

    # Target healthy: enough for fixed calibration/eval plus commission.
    for i in range(40):
        cycles.append(
            _cycle(
                2000 + i,
                anomaly=False,
                setting=73,
            )
        )

    # Anomalies.
    for i in range(10):
        cycles.append(
            _cycle(
                3000 + i,
                anomaly=True,
                setting=73,
                category=i % 2,
            )
        )

    return cycles


def test_frozen_eval_same_across_commissioning_seeds():
    cycles = _synthetic_cycles()

    a = create_frozen_evaluation_split(
        cycles,
        commissioning_size=5,
        commissioning_seed=0,
        evaluation_seed=42,
        calibration_size=10,
        normal_evaluation_size=10,
        maximum_commissioning_size=10,
    )

    b = create_frozen_evaluation_split(
        cycles,
        commissioning_size=5,
        commissioning_seed=1,
        evaluation_seed=42,
        calibration_size=10,
        normal_evaluation_size=10,
        maximum_commissioning_size=10,
    )

    assert {
        c.episode_id for c in a.target_calibration
    } == {
        c.episode_id for c in b.target_calibration
    }

    assert {
        c.episode_id for c in a.target_normal_evaluation
    } == {
        c.episode_id for c in b.target_normal_evaluation
    }

    assert [
        c.episode_id for c in a.target_anomaly_evaluation
    ] == [
        c.episode_id for c in b.target_anomaly_evaluation
    ]

    assert {
        c.episode_id for c in a.target_commissioning
    } != {
        c.episode_id for c in b.target_commissioning
    }


def test_frozen_eval_nested_commissioning_within_seed():
    cycles = _synthetic_cycles()

    small = create_frozen_evaluation_split(
        cycles,
        commissioning_size=5,
        commissioning_seed=7,
        evaluation_seed=42,
        calibration_size=10,
        normal_evaluation_size=10,
        maximum_commissioning_size=10,
    )

    large = create_frozen_evaluation_split(
        cycles,
        commissioning_size=10,
        commissioning_seed=7,
        evaluation_seed=42,
        calibration_size=10,
        normal_evaluation_size=10,
        maximum_commissioning_size=10,
    )

    small_ids = [c.episode_id for c in small.target_commissioning]
    large_ids = [c.episode_id for c in large.target_commissioning]

    assert small_ids == large_ids[:5]


def test_no_overlap_in_frozen_split():
    cycles = _synthetic_cycles()

    split = create_frozen_evaluation_split(
        cycles,
        commissioning_size=10,
        commissioning_seed=3,
        evaluation_seed=42,
        calibration_size=10,
        normal_evaluation_size=10,
        maximum_commissioning_size=10,
    )

    split.verify_no_overlap()