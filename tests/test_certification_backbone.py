import math

import numpy as np

from src.base_detector import BaseDetector
from src.certification import (
    certify_operating_point,
    exact_one_sided_fpr_upper,
    exact_one_sided_recall_lower,
)


def test_zero_false_positive_upper_bound_joint_95():
    delta = 0.025
    n = 368

    bound = exact_one_sided_fpr_upper(
        fp=0,
        tn=n,
        delta=delta,
    )

    expected = 1.0 - delta ** (1.0 / n)

    assert np.isclose(bound, expected)
    assert bound <= 0.01


def test_367_zero_false_positives_do_not_certify_one_percent():
    bound = exact_one_sided_fpr_upper(
        fp=0,
        tn=367,
        delta=0.025,
    )

    assert bound > 0.01


def test_all_false_positive_boundary():
    assert (
        exact_one_sided_fpr_upper(
            fp=100,
            tn=0,
            delta=0.025,
        )
        == 1.0
    )


def test_zero_true_positive_boundary():
    assert (
        exact_one_sided_recall_lower(
            tp=0,
            fn=100,
            delta=0.025,
        )
        == 0.0
    )


def test_all_anomalies_detected_joint_95_best_case():
    delta = 0.025
    n = 36

    bound = exact_one_sided_recall_lower(
        tp=n,
        fn=0,
        delta=delta,
    )

    expected = delta ** (1.0 / n)

    assert np.isclose(bound, expected)
    assert bound >= 0.90


def test_35_all_detected_do_not_certify_90_percent_joint_95():
    bound = exact_one_sided_recall_lower(
        tp=35,
        fn=0,
        delta=0.025,
    )

    assert bound < 0.90


def test_joint_certification_uses_bonferroni_error_split():
    result = certify_operating_point(
        tp=36,
        fn=0,
        fp=0,
        tn=368,
        recall_target=0.90,
        fpr_budget=0.01,
        joint_confidence=0.95,
    )

    assert np.isclose(result.delta_recall, 0.025)
    assert np.isclose(result.delta_fpr, 0.025)
    assert result.certified


def test_conformal_infinite_regime():
    scores = np.arange(50, dtype=float)

    threshold = BaseDetector.conformal_quantile(
        scores,
        alpha=0.01,
    )

    assert math.isinf(threshold)
    assert threshold > 0.0


def test_conformal_maximum_regime_at_100():
    scores = np.arange(100, dtype=float)

    threshold = BaseDetector.conformal_quantile(
        scores,
        alpha=0.01,
    )

    assert threshold == 99.0


def test_conformal_second_largest_transition_at_199():
    scores = np.arange(199, dtype=float)

    threshold = BaseDetector.conformal_quantile(
        scores,
        alpha=0.01,
    )

    assert threshold == 197.0
