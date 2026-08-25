from __future__ import annotations

import numpy as np

from src.e8_dependence import (
    autocorrelation,
    block_bootstrap_interval,
    effective_sample_size,
    permutation_pvalue_lag1,
    summarize_dependence,
)


def test_constant_series_has_zero_acf_and_full_ess() -> None:
    x = np.ones(50)
    acf = autocorrelation(x, max_lag=5)
    assert np.allclose(acf, 0.0)
    assert effective_sample_size(x, max_lag=5) == 50.0


def test_strong_ar1_shows_positive_dependence_and_reduced_ess() -> None:
    rng = np.random.default_rng(123)
    x = np.zeros(1000, dtype=float)
    noise = rng.normal(size=len(x))
    for t in range(1, len(x)):
        x[t] = 0.85 * x[t - 1] + noise[t]
    summary = summarize_dependence(x, max_lag=20)
    assert summary.lag1_autocorrelation > 0.70
    assert summary.effective_sample_size < 0.35 * len(x)
    assert summary.ljung_box_pvalue < 1e-6


def test_permutation_test_detects_strong_order_structure() -> None:
    x = np.linspace(-3.0, 3.0, 150)
    p = permutation_pvalue_lag1(x, n_permutations=1000, seed=9)
    assert p < 0.01


def test_block_bootstrap_binary_interval_contains_empirical_mean() -> None:
    x = np.asarray([0.0] * 95 + [1.0] * 5)
    low, high = block_bootstrap_interval(
        x,
        block_size=5,
        confidence=0.95,
        n_bootstrap=1000,
        seed=7,
    )
    assert 0.0 <= low <= 0.05
    assert 0.05 <= high <= 1.0
