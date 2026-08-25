from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.e6_replication import (
    EXPECTED_SEEDS,
    bootstrap_mean_summary,
    metric_summary_table,
    paired_adaptation_table,
    validate_e6r1_design,
)


def _synthetic_full_design() -> pd.DataFrame:
    rows = []
    for aggregation in ("q99", "mean"):
        agg_offset = 0.0 if aggregation == "q99" else 0.1
        for seed in EXPECTED_SEEDS:
            base = 0.50 + 0.002 * seed + agg_offset
            rows.append({
                "cycle_score_aggregation": aggregation,
                "variant": "M2N2-Offline",
                "commissioning_size": 100,
                "seed": seed,
                "recall": base,
                "false_positive_rate": 0.01,
                "auroc": base + 0.20,
                "oracle_recall_at_fpr_budget": base + 0.05,
            })
            rows.append({
                "cycle_score_aggregation": aggregation,
                "variant": "M2N2-CommissioningAdapt",
                "commissioning_size": 100,
                "seed": seed,
                "recall": base + 0.03,
                "false_positive_rate": 0.015,
                "auroc": base + 0.24,
                "oracle_recall_at_fpr_budget": base + 0.07,
            })
    return pd.DataFrame(rows)


def test_bootstrap_summary_is_deterministic() -> None:
    values = np.arange(1.0, 21.0)
    first = bootstrap_mean_summary(values, repetitions=1000, seed=7)
    second = bootstrap_mean_summary(values, repetitions=1000, seed=7)
    assert first == second
    assert first.n == 20
    assert first.ci_lower < first.mean < first.ci_upper


def test_full_design_validation_and_metric_summary() -> None:
    df = _synthetic_full_design()
    validate_e6r1_design(df, strict=True)
    summary = metric_summary_table(df, bootstrap_repetitions=500, seed=11)
    assert len(summary) == 2 * 2 * 4
    assert set(summary["n_runs"]) == {20}
    assert np.isfinite(summary["bootstrap_ci_lower"]).all()
    assert np.isfinite(summary["bootstrap_ci_upper"]).all()


def test_paired_adaptation_table_recovers_positive_deltas() -> None:
    df = _synthetic_full_design()
    paired = paired_adaptation_table(
        df,
        bootstrap_repetitions=500,
        signflip_repetitions=1000,
        seed=13,
    )
    auroc = paired[paired.metric == "auroc"]
    assert len(auroc) == 2
    assert np.allclose(auroc.mean_delta, 0.04)
    assert (auroc.positive_count == 20).all()
    assert (auroc.negative_count == 0).all()


def test_strict_validation_rejects_missing_seed() -> None:
    df = _synthetic_full_design()
    bad = df[~((df.cycle_score_aggregation == "q99") & (df.variant == "M2N2-Offline") & (df.seed == 19))]
    with pytest.raises(ValueError, match="expected seeds"):
        validate_e6r1_design(bad, strict=True)
