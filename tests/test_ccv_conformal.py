from __future__ import annotations

import numpy as np

from src.ccv_conformal import (
    ccv_feasibility,
    classify_from_pvalues,
    dkwm_adjust_pvalues,
    dkwm_thresholds,
    marginal_conformal_pvalues,
)


def test_marginal_pvalues_are_upper_tail_and_conservative_on_ties() -> None:
    calibration = np.asarray([1.0, 2.0, 3.0, 4.0])
    test = np.asarray([5.0, 4.0, 2.5, 1.0])
    p = marginal_conformal_pvalues(calibration, test)
    np.testing.assert_allclose(p, [0.2, 0.4, 0.6, 1.0])


def test_dkwm_thresholds_are_monotone_and_bounded() -> None:
    b = dkwm_thresholds(100, 0.05)
    assert len(b) == 102
    assert b[0] == 0.0
    assert b[-1] == 1.0
    assert np.all(np.diff(b) >= -1e-15)
    assert np.all((b >= 0.0) & (b <= 1.0))


def test_dkwm_adjustment_is_never_less_than_marginal_for_supported_grid() -> None:
    n = 100
    p = np.arange(1, n + 2, dtype=float) / float(n + 1)
    adjusted = dkwm_adjust_pvalues(p, calibration_size=n, delta=0.05)
    assert np.all(adjusted + 1e-15 >= p)


def test_primary_alpha_is_marginally_but_not_dkwm_feasible_at_m100() -> None:
    result = ccv_feasibility(100, alpha=0.01, delta=0.05)
    assert result.minimum_marginal_pvalue == 1.0 / 101.0
    assert result.marginal_alpha_feasible
    assert result.minimum_dkwm_adjusted_pvalue > 0.01
    assert not result.dkwm_alpha_feasible


def test_classification_uses_less_equal_alpha() -> None:
    p = np.asarray([0.005, 0.01, 0.0100001, 0.2])
    pred = classify_from_pvalues(p, alpha=0.01)
    assert pred.tolist() == [True, True, False, False]
