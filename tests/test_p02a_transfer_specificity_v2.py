from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.run_p02a_transfer_specificity_v2 import (
    _crossfit_target_clime_candidate,
    _fold_indices,
    _paired_summary,
)


def test_fold_indices_are_deterministic_and_cover_model_pool() -> None:
    a = _fold_indices(20, 5, 42)
    b = _fold_indices(20, 5, 42)
    assert len(a) == len(b) == 5
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    joined = np.concatenate(a)
    assert sorted(joined.tolist()) == list(range(20))


def test_crossfit_target_clime_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 6))
    a = _crossfit_target_clime_candidate(x, lam=0.5, n_folds=5, seed=123)
    b = _crossfit_target_clime_candidate(x, lam=0.5, n_folds=5, seed=123)
    assert a.method == "CrossfitTargetCLIME"
    assert np.allclose(a.matrix, b.matrix)
    assert np.min(np.linalg.eigvalsh(a.matrix)) > 0.0


def test_paired_summary_uses_method_matched_baselines() -> None:
    rows = []
    for source_kind, distance in [("identical", 0.0), ("adversarial", 0.5)]:
        for rep in range(3):
            common = {
                "p": 20,
                "target_n": 25,
                "replication": rep,
                "source_kind": source_kind,
                "source_target_truth_relative_frobenius": distance,
                "source_target_truth_support_jaccard": 1.0 if source_kind == "identical" else 0.0,
                "source_target_truth_sign_agreement": 1.0 if source_kind == "identical" else 0.0,
            }
            rows.extend([
                {**common, "method": "ReferenceTargetCLIME", "relative_frobenius_spd": 0.50},
                {**common, "method": "CrossfitTargetCLIME", "relative_frobenius_spd": 0.40},
                {**common, "method": "BestMatchedTargetOnly", "relative_frobenius_spd": 0.35},
                {**common, "method": "ReferenceTransCLIME", "relative_frobenius_spd": 0.45},
                {**common, "method": "CrossfitTransCLIME", "relative_frobenius_spd": 0.30},
            ])

    summary = _paired_summary(pd.DataFrame(rows), n_boot=100)
    assert len(summary) == 4

    ref = summary[(summary.method == "ReferenceTransCLIME") & (summary.source_kind == "identical")].iloc[0]
    cf = summary[(summary.method == "CrossfitTransCLIME") & (summary.source_kind == "identical")].iloc[0]

    assert np.isclose(ref.median_gain_vs_matched_target, 0.10)
    assert np.isclose(cf.median_gain_vs_matched_target, 0.25)
    assert np.isclose(cf.median_gain_vs_best_matched_target, (0.35 - 0.30) / 0.35)


def test_many_transfer_rows_merge_to_single_baselines_without_duplication() -> None:
    rows = []
    for rep in range(2):
        common = {
            "p": 20,
            "target_n": 25,
            "replication": rep,
            "source_kind": "identical",
            "source_target_truth_relative_frobenius": 0.0,
            "source_target_truth_support_jaccard": 1.0,
            "source_target_truth_sign_agreement": 1.0,
        }
        rows.extend([
            {**common, "method": "ReferenceTargetCLIME", "relative_frobenius_spd": 0.5},
            {**common, "method": "CrossfitTargetCLIME", "relative_frobenius_spd": 0.4},
            {**common, "method": "BestMatchedTargetOnly", "relative_frobenius_spd": 0.35},
            {**common, "method": "ReferenceTransCLIME", "relative_frobenius_spd": 0.45},
            {**common, "method": "CrossfitTransCLIME", "relative_frobenius_spd": 0.30},
        ])
    summary = _paired_summary(pd.DataFrame(rows), n_boot=50)
    assert set(summary.method) == {"ReferenceTransCLIME", "CrossfitTransCLIME"}
    assert (summary.replications == 2).all()
