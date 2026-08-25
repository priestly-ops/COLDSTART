from __future__ import annotations

import numpy as np

from src.mvtflow_adapter import (
    classify_e2_bottleneck,
    prepare_mvtflow_data,
)
from src.voraus_loader import RobotCycle


def _cycle(episode_id: int, values: list[list[float]]) -> RobotCycle:
    return RobotCycle(
        episode_id=episode_id,
        values=np.asarray(values, dtype=np.float64),
        columns=("s1", "s2"),
        anomaly=False,
        category=0,
        setting=73,
    )


def test_preprocessing_fits_only_commissioning_and_pads_to_training_max() -> None:
    commissioning = [
        _cycle(1, [[0.0, 0.0], [2.0, 2.0]]),
        _cycle(2, [[4.0, 4.0]]),
    ]
    calibration = [_cycle(3, [[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])]
    healthy = [_cycle(4, [[1.0, 1.0]])]
    anomaly = [_cycle(5, [[5.0, 5.0]])]

    prepared = prepare_mvtflow_data(
        commissioning=commissioning,
        calibration=calibration,
        healthy_eval=healthy,
        anomaly_eval=anomaly,
    )

    assert prepared.target_length == 2
    assert prepared.n_signals == 2
    # Commissioning points are 0, 2, 4 in each signal.
    assert np.allclose(prepared.mean, np.array([2.0, 2.0]))
    assert np.allclose(prepared.scale, np.array([np.sqrt(8.0 / 3.0)] * 2))
    # Calibration is truncated to the commissioning max length.
    assert prepared.calibration.shape == (1, 2, 2)
    # Short healthy cycle is zero-padded after standardization.
    assert np.allclose(prepared.healthy_eval[0, 1], np.zeros(2))


def test_bottleneck_precedence_matches_e1() -> None:
    assert (
        classify_e2_bottleneck(
            oracle_feasible=False,
            empirical_success=False,
            certified_success=False,
        )
        == "representation_limited"
    )
    assert (
        classify_e2_bottleneck(
            oracle_feasible=True,
            empirical_success=False,
            certified_success=False,
        )
        == "calibration_limited"
    )
    assert (
        classify_e2_bottleneck(
            oracle_feasible=True,
            empirical_success=True,
            certified_success=False,
        )
        == "certification_limited"
    )
    assert (
        classify_e2_bottleneck(
            oracle_feasible=True,
            empirical_success=True,
            certified_success=True,
        )
        == "certified"
    )
