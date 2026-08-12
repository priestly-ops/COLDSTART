
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.m2_reviewer_audit import (
    independent_bruteforce_oracle,
    count_based_sensitivity,
    score_ordering_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_script_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "experiments/run_m2_reviewer_defense.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--seeds" in result.stdout


def test_independent_clear_separation():
    h = np.array([0.0, 1.0, 2.0, 3.0])
    a = np.array([10.0, 11.0, 12.0, 13.0])
    r = independent_bruteforce_oracle(
        h, a, false_alert_budget=0.0, recall_target=0.9
    )
    assert r.feasible
    assert r.max_recall_at_budget == 1.0
    assert r.min_fp_count_at_target == 0


def test_independent_overlap_infeasible():
    h = np.arange(100, dtype=float)
    a = np.arange(50, 60, dtype=float)
    r = independent_bruteforce_oracle(
        h, a, false_alert_budget=0.01, recall_target=0.9
    )
    assert not r.feasible


def test_count_sensitivity_has_9_cells():
    h = np.arange(100, dtype=float)
    a = np.arange(200, 220, dtype=float)
    rows = count_based_sensitivity(h, a)
    assert len(rows) == 9
    assert {r["allowed_fp_count_requested"] for r in rows} == {0, 1, 2}
    assert {r["recall_target_requested"] for r in rows} == {0.8, 0.9, 0.95}


def test_score_ordering_audit():
    h = np.array([0.0, 1.0, 2.0])
    a = np.array([3.0, 4.0])
    d = score_ordering_audit(h, a)
    assert d["strict_complete_separation"]
    assert d["healthy_above_anomaly_min"] == 0
