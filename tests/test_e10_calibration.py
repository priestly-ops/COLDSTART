from __future__ import annotations

import numpy as np

from src.e10_calibration import (
    deterministic_split_conformal,
    evt_pot,
    randomized_smoothed_conformal,
    smoothed_conformal_alarm_probabilities,
)


def test_smoothed_conformal_probability_above_max_at_m50_alpha01():
    cal = np.arange(50, dtype=float)
    test = np.array([100.0])
    p_alarm = smoothed_conformal_alarm_probabilities(cal, test, alpha=0.01)
    # For score > max(cal), p = U/(m+1); P[p<=.01] = .01*(51) = .51.
    assert np.isclose(p_alarm[0], 0.51)


def test_smoothed_conformal_probability_is_zero_when_too_many_greater_scores():
    cal = np.arange(100, dtype=float)
    test = np.array([0.0])
    p_alarm = smoothed_conformal_alarm_probabilities(cal, test, alpha=0.01)
    assert p_alarm[0] == 0.0


def test_randomized_conformal_is_reproducible_for_fixed_rng_seed():
    cal = np.arange(50, dtype=float)
    test = np.array([100.0] * 20)
    a = randomized_smoothed_conformal(cal, test, alpha=0.01, rng=np.random.default_rng(123))
    b = randomized_smoothed_conformal(cal, test, alpha=0.01, rng=np.random.default_rng(123))
    assert np.array_equal(a.predictions, b.predictions)
    assert np.allclose(a.expected_alarm_probability, b.expected_alarm_probability)


def test_deterministic_m50_alpha01_is_infinite_and_never_alarms():
    cal = np.arange(50, dtype=float)
    test = np.array([100.0, 200.0])
    result = deterministic_split_conformal(cal, test, alpha=0.01)
    assert result.diagnostic["conformal_regime"] == "infinite"
    assert np.isinf(result.threshold)
    assert not result.predictions.any()


def test_evt_pot_returns_finite_threshold_for_well_behaved_tail():
    rng = np.random.default_rng(7)
    cal = rng.exponential(scale=1.0, size=100)
    test = np.array([0.5, 2.0, 8.0])
    result = evt_pot(cal, test, alpha=0.01, tail_fraction=0.20, minimum_exceedances=10)
    assert result.fit_ok
    assert np.isfinite(result.threshold)
    assert result.predictions.shape == test.shape
    assert result.diagnostic["pot_exceedance_count"] >= 10


def test_evt_pot_fit_failure_is_explicit_not_silent_fallback():
    cal = np.ones(50, dtype=float)
    test = np.array([1.0, 2.0])
    result = evt_pot(cal, test, alpha=0.01, tail_fraction=0.20, minimum_exceedances=10)
    assert not result.fit_ok
    assert "fit_error" in result.diagnostic
