from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_p10_calibration_margin_theory import (
    _landmarks,
    _make_rank_table,
    _paired_drift,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_script_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "experiments/run_p10_calibration_margin_theory.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--p09-seed-diagnostics" in result.stdout


def test_one_percent_landmarks_are_exact():
    table = _make_rank_table(250, (0.01,))
    landmarks = _landmarks(table)
    row = landmarks.iloc[0]
    assert int(row.minimum_calibration_for_finite_threshold) == 100
    assert int(row.minimum_calibration_for_nonmax_threshold) == 199
    assert int(row.largest_calibration_still_forced_to_maximum) == 198


def test_half_percent_landmarks_are_exact():
    table = _make_rank_table(500, (0.005,))
    landmarks = _landmarks(table)
    row = landmarks.iloc[0]
    assert int(row.minimum_calibration_for_finite_threshold) == 200
    assert int(row.minimum_calibration_for_nonmax_threshold) == 399
    assert int(row.largest_calibration_still_forced_to_maximum) == 398


def test_paired_drift_recovers_threshold_margin_identity(tmp_path: Path):
    rows = []
    for seed in (0, 1):
        rows.append(
            {
                "budget": 224,
                "seed": seed,
                "method": "RACECovSafeCV",
                "threshold_robust_z": 5.0,
                "recall": 0.2,
                "auroc": 0.7,
                "cat1_median_robust_z": 1.0,
                "cat1_median_margin_z": -4.0,
                "cat1_auroc": 0.6,
            }
        )
        rows.append(
            {
                "budget": 400,
                "seed": seed,
                "method": "RACECovSafeCV",
                "threshold_robust_z": 7.0,
                "recall": 0.1,
                "auroc": 0.72,
                "cat1_median_robust_z": 1.5,
                "cat1_median_margin_z": -5.5,
                "cat1_auroc": 0.63,
            }
        )
    path = tmp_path / "p09.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    drift, ranking = _paired_drift(path, 224, 400)
    row = drift[(drift.method == "RACECovSafeCV") & (drift.category == 1)].iloc[0]
    assert np.isclose(row.delta_threshold_robust_z_mean, 2.0)
    assert np.isclose(row.delta_anomaly_median_robust_z_mean, 0.5)
    assert np.isclose(row.delta_margin_z_mean, -1.5)
    rank = ranking.iloc[0]
    assert np.isclose(rank.delta_auroc_mean, 0.02)
    assert np.isclose(rank.delta_recall_mean, -0.1)
    assert np.isclose(rank.fraction_auroc_up_recall_down, 1.0)
