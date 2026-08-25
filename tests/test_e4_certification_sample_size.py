from __future__ import annotations

from src.e4_certification_sample_size import (
    minimum_anomaly_sample_size,
    minimum_healthy_sample_size,
)


def test_zero_fp_best_case_boundary_is_368() -> None:
    result = minimum_healthy_sample_size(false_positives=0)
    assert result.minimum_sample_size == 368
    assert result.bound_at_minimum <= 0.01

    previous = minimum_healthy_sample_size(false_positives=0, fpr_budget=0.01, max_n=368)
    assert previous.minimum_sample_size == 368


def test_one_and_two_fp_require_more_healthy_evidence() -> None:
    one = minimum_healthy_sample_size(false_positives=1)
    two = minimum_healthy_sample_size(false_positives=2)
    assert one.minimum_sample_size == 555
    assert two.minimum_sample_size == 720
    assert 368 < one.minimum_sample_size < two.minimum_sample_size


def test_perfect_recall_boundary_is_36() -> None:
    result = minimum_anomaly_sample_size(false_negatives=0)
    assert result.minimum_sample_size == 36
    assert result.bound_at_minimum >= 0.90


def test_one_and_two_misses_require_more_anomaly_evidence() -> None:
    one = minimum_anomaly_sample_size(false_negatives=1)
    two = minimum_anomaly_sample_size(false_negatives=2)
    assert one.minimum_sample_size == 54
    assert two.minimum_sample_size == 70
    assert 36 < one.minimum_sample_size < two.minimum_sample_size
