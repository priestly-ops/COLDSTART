from __future__ import annotations

import numpy as np

from src.transfer_compatibility import compute_compatibility_metrics


def test_identical_distributions_are_more_compatible_than_shifted() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(200, 12))
    target_same = rng.normal(size=(40, 12))
    target_shifted = rng.normal(loc=2.0, scale=1.0, size=(40, 12))

    same = compute_compatibility_metrics(source, target_same, seed=11)
    shifted = compute_compatibility_metrics(source, target_shifted, seed=11)

    assert shifted.mean_shift_rms > same.mean_shift_rms
    assert shifted.rbf_mmd2 > same.rbf_mmd2
    assert shifted.domain_classifier_auc > same.domain_classifier_auc


def test_metrics_are_finite_and_reproducible() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(120, 8))
    target = rng.normal(loc=0.3, size=(25, 8))

    first = compute_compatibility_metrics(source, target, seed=99)
    second = compute_compatibility_metrics(source, target, seed=99)

    for name in (
        "mean_shift_rms",
        "covariance_correlation_frobenius",
        "rbf_mmd2",
        "domain_classifier_auc",
    ):
        a = float(getattr(first, name))
        b = float(getattr(second, name))
        assert np.isfinite(a)
        assert a == b


def test_domain_auc_is_folded_to_half_or_above() -> None:
    rng = np.random.default_rng(21)
    source = rng.normal(size=(100, 5))
    target = rng.normal(size=(20, 5))
    result = compute_compatibility_metrics(source, target, seed=3)
    assert 0.5 <= result.domain_classifier_auc <= 1.0
