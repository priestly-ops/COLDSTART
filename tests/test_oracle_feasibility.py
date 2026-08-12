import subprocess
import sys
from pathlib import Path

import numpy as np

from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m2_script_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "experiments/run_m2_oracle_feasibility.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--seeds" in result.stdout


def test_oracle_feasible_when_scores_are_separated():
    healthy = np.arange(100, dtype=float)
    anomaly = np.arange(200, 300, dtype=float)
    result = empirical_oracle_feasibility(healthy, anomaly, 0.01, 0.90)
    assert result.empirically_feasible
    assert result.max_recall_at_fpr_budget == 1.0
    assert result.min_fpr_at_recall_target == 0.0
    assert result.allowed_false_positives == 1


def test_oracle_infeasible_when_scores_overlap_badly():
    healthy = np.arange(100, dtype=float)
    anomaly = np.arange(100, dtype=float)
    result = empirical_oracle_feasibility(healthy, anomaly, 0.01, 0.90)
    assert not result.empirically_feasible
    assert result.max_recall_at_fpr_budget <= 0.02
    assert result.min_fpr_at_recall_target >= 0.89


def test_ties_use_strict_greater_than_detector_convention():
    healthy = np.array([0.0, 1.0, 1.0, 2.0])
    anomaly = np.array([1.0, 1.0, 2.0, 3.0])
    result = empirical_oracle_feasibility(healthy, anomaly, 0.25, 0.50)
    assert result.empirically_feasible
    # threshold 1.0 => one healthy score >1, two anomaly scores >1.
    assert result.max_recall_at_fpr_budget >= 0.5


def test_probability_of_superiority_handles_ties():
    healthy = np.array([0.0, 1.0])
    anomaly = np.array([1.0, 2.0])
    # Pairs: (1>0), (1=1)/2, (2>0), (2>1) => 3.5 / 4 = .875
    assert np.isclose(probability_of_superiority(healthy, anomaly), 0.875)
