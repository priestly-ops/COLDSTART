from __future__ import annotations

import numpy as np

from src.paired_inference import (
    paired_bootstrap_mean_ci,
    paired_signflip_pvalue,
    paired_wilcoxon_pvalue,
    summarize_paired_deltas,
)


def test_bootstrap_ci_is_deterministic() -> None:
    deltas = np.array([-0.2, -0.1, -0.3, -0.4, -0.25], dtype=float)
    first = paired_bootstrap_mean_ci(deltas, repetitions=2000, seed=42)
    second = paired_bootstrap_mean_ci(deltas, repetitions=2000, seed=42)
    assert first == second
    assert first[1] < 0.0


def test_exact_signflip_detects_strong_one_sided_pattern() -> None:
    deltas = -np.arange(1.0, 11.0)
    pvalue, method = paired_signflip_pvalue(deltas, exact_max_nonzero=20)
    assert method == "exact"
    assert pvalue <= 2.0 / (2**10)


def test_all_zero_deltas_are_neutral() -> None:
    deltas = np.zeros(6, dtype=float)
    pvalue, method = paired_signflip_pvalue(deltas)
    assert pvalue == 1.0
    assert method == "all_zero"
    assert paired_wilcoxon_pvalue(deltas) == 1.0


def test_summary_counts_and_interval() -> None:
    deltas = np.array([-0.4, -0.3, -0.2, -0.1, 0.0, 0.05], dtype=float)
    result = summarize_paired_deltas(
        deltas,
        bootstrap_repetitions=3000,
        signflip_repetitions=5000,
        seed=7,
    )
    assert result.n_pairs == 6
    assert result.negative_count == 4
    assert result.positive_count == 1
    assert result.neutral_count == 1
    assert result.bootstrap_ci_lower <= result.mean_delta <= result.bootstrap_ci_upper
    assert 0.0 <= result.signflip_pvalue <= 1.0
    assert 0.0 <= result.wilcoxon_pvalue <= 1.0


def test_invalid_inputs_fail_fast() -> None:
    try:
        summarize_paired_deltas(np.array([1.0]))
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("expected ValueError for a single paired observation")
