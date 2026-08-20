from __future__ import annotations

import numpy as np

from experiments.run_p02a_transfer_specificity import (
    _precision_chain,
    _source_truths,
    _truth_similarity,
)


def test_source_regimes_are_spd_and_distinct() -> None:
    target = _precision_chain(20)
    regimes = _source_truths(target)

    assert set(regimes) == {
        "identical",
        "mild",
        "moderate",
        "disjoint",
        "sign_reversed",
        "adversarial",
        "diagonal",
        "permuted",
    }

    for matrix in regimes.values():
        eig = np.linalg.eigvalsh(matrix)
        assert np.min(eig) > 0.0

    assert np.allclose(regimes["identical"], target)
    assert not np.allclose(regimes["adversarial"], target)
    assert not np.allclose(regimes["permuted"], target)


def test_similarity_controls_have_expected_extremes() -> None:
    target = _precision_chain(20)
    regimes = _source_truths(target)

    identical = _truth_similarity(regimes["identical"], target)
    disjoint = _truth_similarity(regimes["disjoint"], target)
    sign_reversed = _truth_similarity(regimes["sign_reversed"], target)

    assert identical["source_target_truth_relative_frobenius"] == 0.0
    assert identical["source_target_truth_support_jaccard"] == 1.0
    assert identical["source_target_truth_sign_agreement"] == 1.0

    assert disjoint["source_target_truth_support_jaccard"] == 0.0
    assert sign_reversed["source_target_truth_support_jaccard"] == 1.0
    assert sign_reversed["source_target_truth_sign_agreement"] == 0.0
