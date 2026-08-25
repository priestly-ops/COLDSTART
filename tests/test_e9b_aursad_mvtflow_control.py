from __future__ import annotations

import numpy as np

from src.base_detector import deterministic_conformal_threshold
from src.certification import certify_operating_point
from src.mvtflow_adapter import OFFICIAL_CONFIG, classify_e2_bottleneck


def test_mvtflow_official_training_schedule_is_frozen() -> None:
    assert OFFICIAL_CONFIG["epochs"] == 70
    assert OFFICIAL_CONFIG["batch_size"] == 32
    assert OFFICIAL_CONFIG["n_coupling_blocks"] == 4
    assert OFFICIAL_CONFIG["learning_rate"] == 8e-4


def test_primary_mcal_199_is_submaximum_at_one_percent() -> None:
    scores = np.arange(199, dtype=np.float64)
    threshold, rank, regime = deterministic_conformal_threshold(scores, alpha=0.01)
    assert rank == 198
    assert regime == "submaximum"
    assert threshold == 197.0


def test_368_zero_fp_crosses_exact_fpr_certification_boundary() -> None:
    result = certify_operating_point(
        tp=625,
        fn=0,
        fp=0,
        tn=368,
        recall_target=0.90,
        fpr_budget=0.01,
        joint_confidence=0.95,
    )
    assert result.fpr_upper <= 0.01
    assert result.recall_lower >= 0.90
    assert result.certified


def test_367_zero_fp_does_not_cross_boundary() -> None:
    result = certify_operating_point(
        tp=625,
        fn=0,
        fp=0,
        tn=367,
        recall_target=0.90,
        fpr_budget=0.01,
        joint_confidence=0.95,
    )
    assert result.fpr_upper > 0.01
    assert not result.certified


def test_representation_precedence_is_preserved() -> None:
    label = classify_e2_bottleneck(
        oracle_feasible=False,
        empirical_success=False,
        certified_success=False,
    )
    assert label == "representation_limited"
