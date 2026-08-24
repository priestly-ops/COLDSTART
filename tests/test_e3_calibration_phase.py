from __future__ import annotations

import numpy as np

from src.e3_calibration_phase import evaluate_threshold_from_scores, nested_calibration_order
from src.voraus_loader import RobotCycle


def _cycle(episode_id: int) -> RobotCycle:
    return RobotCycle(
        episode_id=episode_id,
        values=np.array([[float(episode_id)]], dtype=np.float64),
        columns=("x",),
        anomaly=False,
        category=0,
        setting=73,
    )


def test_nested_order_preserves_base_set_at_base_size() -> None:
    base = tuple(_cycle(i) for i in range(100))
    extra = tuple(_cycle(i) for i in range(100, 119))
    ordered = nested_calibration_order(base, extra, ordering_seed=123)

    assert len(ordered) == 119
    assert {c.episode_id for c in ordered[:100]} == set(range(100))
    assert {c.episode_id for c in ordered[100:]} == set(range(100, 119))


def test_nested_order_is_deterministic() -> None:
    base = tuple(_cycle(i) for i in range(100))
    extra = tuple(_cycle(i) for i in range(100, 119))
    first = nested_calibration_order(base, extra, ordering_seed=456)
    second = nested_calibration_order(base, extra, ordering_seed=456)
    assert [c.episode_id for c in first] == [c.episode_id for c in second]


def _eval(m: int):
    calibration = np.linspace(0.0, 1.0, m)
    healthy = np.linspace(0.0, 0.5, 100)
    anomaly = np.linspace(2.0, 3.0, 100)
    return evaluate_threshold_from_scores(
        calibration_scores=calibration,
        healthy_scores=healthy,
        anomaly_scores=anomaly,
        false_alert_budget=0.01,
        recall_target=0.90,
        joint_confidence=0.95,
    )


def test_m50_is_infinite() -> None:
    result = _eval(50)
    assert result.conformal_rank == 51
    assert result.conformal_regime == "infinite"
    assert np.isinf(result.threshold)
    assert result.recall == 0.0
    assert result.false_positive_rate == 0.0


def test_m99_is_first_finite_maximum() -> None:
    result = _eval(99)
    assert result.conformal_rank == 99
    assert result.conformal_regime == "maximum"
    assert np.isfinite(result.threshold)


def test_m198_is_maximum_and_m199_is_submaximum() -> None:
    r198 = _eval(198)
    r199 = _eval(199)
    assert r198.conformal_rank == 198
    assert r198.conformal_regime == "maximum"
    assert r199.conformal_rank == 198
    assert r199.conformal_regime == "submaximum"
