from __future__ import annotations

import numpy as np

from src.oracle_feasibility import empirical_oracle_feasibility
from src.oracle_stability import (
    bootstrap_oracle_recall,
    oracle_recall_at_fpr_budget_fast,
)


def test_fast_oracle_matches_reference_random_scores() -> None:
    rng = np.random.default_rng(7)
    for n_h in (20, 100):
        healthy = rng.normal(size=n_h)
        anomaly = rng.normal(loc=1.0, size=75)
        fast, _, _ = oracle_recall_at_fpr_budget_fast(
            healthy,
            anomaly,
            false_alert_budget=0.01,
        )
        reference = empirical_oracle_feasibility(
            healthy,
            anomaly,
            false_alert_budget=0.01,
            recall_target=0.90,
        )
        assert fast == reference.max_recall_at_fpr_budget


def test_fast_oracle_matches_reference_with_ties() -> None:
    healthy = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    anomaly = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    for budget in (0.0, 0.1, 0.2, 0.5):
        fast, threshold, allowed = oracle_recall_at_fpr_budget_fast(
            healthy,
            anomaly,
            false_alert_budget=budget,
        )
        reference = empirical_oracle_feasibility(
            healthy,
            anomaly,
            false_alert_budget=budget,
            recall_target=0.90,
        )
        assert fast == reference.max_recall_at_fpr_budget
        assert np.mean(healthy > threshold) <= budget + 1e-15
        assert allowed == int(np.floor(budget * len(healthy) + 1e-12))


def test_bootstrap_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(11)
    healthy = rng.normal(size=100)
    anomaly = rng.normal(loc=0.5, size=120)
    first = bootstrap_oracle_recall(
        healthy,
        anomaly,
        repetitions=300,
        seed=99,
        chunk_size=37,
    )
    second = bootstrap_oracle_recall(
        healthy,
        anomaly,
        repetitions=300,
        seed=99,
        chunk_size=37,
    )
    assert first == second
    assert 0.0 <= first.bootstrap_interval_lower <= 1.0
    assert 0.0 <= first.bootstrap_interval_upper <= 1.0
    assert first.bootstrap_interval_lower <= first.bootstrap_interval_upper
    assert 0.0 <= first.bootstrap_feasibility_rate <= 1.0


def test_clear_separation_produces_stable_feasible_oracle() -> None:
    healthy = np.linspace(0.0, 1.0, 100)
    anomaly = np.linspace(2.0, 3.0, 80)
    result = bootstrap_oracle_recall(
        healthy,
        anomaly,
        false_alert_budget=0.01,
        recall_target=0.90,
        repetitions=250,
        seed=3,
    )
    assert result.observed_oracle_recall == 1.0
    assert result.bootstrap_interval_lower == 1.0
    assert result.bootstrap_interval_upper == 1.0
    assert result.bootstrap_feasibility_rate == 1.0
