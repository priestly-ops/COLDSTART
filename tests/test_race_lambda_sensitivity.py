from __future__ import annotations

import math

import numpy as np

from src.detectors import RACEDetector
from src.race_lambda_sensitivity import (
    RACELambdaSensitivityDetector,
    lambda_label,
    target_weight_for_lambda,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    source = rng.normal(loc=0.0, scale=1.0, size=(80, 6))
    target = rng.normal(loc=0.2, scale=1.1, size=(20, 6))
    test = rng.normal(loc=0.1, scale=1.0, size=(15, 6))
    return source, target, test


def test_target_weight_endpoints_and_reference() -> None:
    assert target_weight_for_lambda(100, 0.0) == 1.0
    assert target_weight_for_lambda(100, math.inf) == 0.0
    assert target_weight_for_lambda(100, 60.0) == 0.625


def test_lambda_labels() -> None:
    assert lambda_label(0.0) == "0"
    assert lambda_label(60.0) == "60"
    assert lambda_label(math.inf) == "inf"


def test_lambda60_matches_frozen_race() -> None:
    source, target, test = _toy_data()
    frozen = RACEDetector(lambda_reg=60.0).fit(source, target)
    sensitivity = RACELambdaSensitivityDetector(lambda_reg=60.0).fit(source, target)

    np.testing.assert_allclose(frozen.location_, sensitivity.location_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(frozen.precision_, sensitivity.precision_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        frozen.score_samples(test),
        sensitivity.score_samples(test),
        rtol=1e-10,
        atol=1e-12,
    )
    assert frozen.target_weight_ == sensitivity.target_weight_


def test_zero_and_infinity_are_valid_family_endpoints() -> None:
    source, target, test = _toy_data()
    target_endpoint = RACELambdaSensitivityDetector(lambda_reg=0.0).fit(source, target)
    source_endpoint = RACELambdaSensitivityDetector(lambda_reg=math.inf).fit(source, target)

    assert target_endpoint.target_weight_ == 1.0
    assert source_endpoint.target_weight_ == 0.0
    assert np.isfinite(target_endpoint.score_samples(test)).all()
    assert np.isfinite(source_endpoint.score_samples(test)).all()


def test_negative_lambda_rejected() -> None:
    try:
        RACELambdaSensitivityDetector(lambda_reg=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative lambda should be rejected")
