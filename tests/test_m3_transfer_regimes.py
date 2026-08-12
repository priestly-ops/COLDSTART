from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src.m3_transfer_regimes import (
    TRANSFERABILITY_COLUMNS,
    assert_no_episode_leakage,
    construct_source_regimes,
    paired_deltas,
    transfer_weight_diagnostics,
)


@dataclass(frozen=True)
class DummyCycle:
    episode_id: int


def _base_results() -> pd.DataFrame:
    rows = []
    transfer = {column: 0.1 for column in TRANSFERABILITY_COLUMNS}
    for detector, recall, fpr, weight in [
        ("TargetOnly", 0.60, 0.01, 0.0),
        ("RACE", 0.70, 0.01, 0.5),
        ("SourcePermutation", 0.55, 0.02, 0.5),
        ("WeightPermutation", 0.58, 0.02, 0.5),
    ]:
        rows.append(
            {
                "source_pair_id": "near_shift_N10_seed0",
                "source_group": "near_shift_source_setting_72",
                "target_group": "target_setting_73_N10_seed0",
                "commissioning_size": 10,
                "seed": 0,
                "detector": detector,
                "recall": recall,
                "fpr": fpr,
                "auroc": recall + 0.1,
                "auprc": recall + 0.05,
                "success": float(recall >= 0.9 and fpr <= 0.01),
                "transfer_weight": weight,
                **transfer,
            }
        )
    return pd.DataFrame(rows)


def test_construct_source_regimes_is_deterministic_and_disjoint():
    rng = np.random.default_rng(3)
    source_features = rng.normal(size=(30, 4))
    target_features = rng.normal(loc=0.2, size=(10, 4))
    source_ids = list(range(100, 130))
    target_ids = list(range(200, 210))

    first = construct_source_regimes(
        source_episode_ids=source_ids,
        source_features=source_features,
        target_episode_ids=target_ids,
        target_features=target_features,
        commissioning_size=10,
        seed=0,
        subset_size=5,
    )
    second = construct_source_regimes(
        source_episode_ids=source_ids,
        source_features=source_features,
        target_episode_ids=target_ids,
        target_features=target_features,
        commissioning_size=10,
        seed=0,
        subset_size=5,
    )

    assert [r.source_episode_ids for r in first] == [r.source_episode_ids for r in second]
    assert [r.source_group for r in first] == [
        "near_shift_source_setting_72",
        "moderate_shift_source_setting_72",
        "high_shift_source_setting_72",
    ]
    for regime in first:
        assert set(regime.source_episode_ids).isdisjoint(target_ids)
        assert set(TRANSFERABILITY_COLUMNS).issubset(regime.metrics)
        assert all(np.isfinite(regime.metrics[column]) for column in TRANSFERABILITY_COLUMNS)


def test_no_leakage_raises_on_episode_overlap():
    with pytest.raises(RuntimeError, match="overlaps"):
        assert_no_episode_leakage({"source": [1, 2], "target": [2, 3]})


def test_paired_deltas_compute_expected_values_and_controls():
    deltas = paired_deltas(_base_results())

    assert len(deltas) == 1
    row = deltas.iloc[0]
    assert row["delta_recall"] == pytest.approx(0.10)
    assert row["delta_fpr"] == pytest.approx(0.0)
    assert row["delta_auroc"] == pytest.approx(0.10)
    assert row["source_permutation_recall"] == pytest.approx(0.55)
    assert row["weight_permutation_recall"] == pytest.approx(0.58)


def test_transfer_weight_diagnostics_schema():
    results = _base_results()
    deltas = paired_deltas(results)
    diagnostics = transfer_weight_diagnostics(results, deltas)

    assert {
        "level",
        "transferability_metric",
        "benefit_metric",
        "n",
        "pearson",
        "spearman",
    }.issubset(diagnostics.columns)


def test_expected_m3_seed_result_schema_subset():
    expected = {
        "source_pair_id",
        "source_group",
        "target_group",
        "commissioning_size",
        "seed",
        "detector",
        "recall",
        "fpr",
        "auroc",
        "auprc",
        "success",
        "threshold",
        "calibration_size",
        "transfer_weight",
        *TRANSFERABILITY_COLUMNS,
    }
    available = set(_base_results().columns) | {"threshold", "calibration_size"}
    assert expected.issubset(available)
