import subprocess
import sys
from pathlib import Path

import numpy as np

from src.calibration_tail import conformal_threshold_info


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_script_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "experiments/run_m1_calibration_tail_sensitivity.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--seeds" in result.stdout


def test_m100_alpha001_is_feasible_and_uses_maximum():
    info = conformal_threshold_info(np.arange(100, dtype=float), alpha=0.01)
    assert info.raw_rank == 100
    assert info.finite_sample_feasible
    assert info.threshold_is_maximum
    assert info.strict_threshold == 99.0


def test_m50_alpha001_is_infeasible_under_strict_finite_sample_rule():
    info = conformal_threshold_info(np.arange(50, dtype=float), alpha=0.01)
    assert info.raw_rank == 51
    assert not info.finite_sample_feasible
    assert np.isposinf(info.strict_threshold)
    assert info.legacy_clipped_threshold == 49.0


def test_m100_alpha002_uses_99th_order_statistic():
    info = conformal_threshold_info(np.arange(100, dtype=float), alpha=0.02)
    assert info.raw_rank == 99
    assert info.finite_sample_feasible
    assert not info.threshold_is_maximum
    assert info.strict_threshold == 98.0


def test_m119_alpha001_is_feasible_but_still_maximum():
    info = conformal_threshold_info(np.arange(119, dtype=float), alpha=0.01)
    assert info.raw_rank == 119
    assert info.finite_sample_feasible
    assert info.threshold_is_maximum
    assert info.strict_threshold == 118.0
