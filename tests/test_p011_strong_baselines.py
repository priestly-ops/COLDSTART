from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from src.strong_baselines import (
    ConformalKNNBaseline,
    IsolationForestBaseline,
    RawCycleKNNBaseline,
    _align_to_reference,
    _linear_resample,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runner_help_from_repo_root():
    result = subprocess.run(
        [sys.executable, "experiments/run_p011_strong_baselines_voraus.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FeatureConformalKNN" in result.stdout
    assert "PAKCT" in result.stdout


def test_feature_knn_scores_shifted_points_higher():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(30, 4))
    model = ConformalKNNBaseline(k=5).fit(train)
    near = rng.normal(size=(10, 4))
    far = rng.normal(loc=8.0, size=(10, 4))
    assert model.score_samples(far).mean() > model.score_samples(near).mean()


def test_isolation_forest_scores_shifted_points_higher():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(80, 5))
    model = IsolationForestBaseline(random_state=42).fit(train)
    near = rng.normal(size=(20, 5))
    far = rng.normal(loc=7.0, size=(20, 5))
    assert model.score_samples(far).mean() > model.score_samples(near).mean()


def test_linear_resample_preserves_shape_contract():
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = _linear_resample(x, 15)
    assert y.shape == (15, 3)
    assert np.allclose(y[0], x[0])
    assert np.allclose(y[-1], x[-1])


def test_fastdtw_alignment_returns_reference_grid_shape():
    t1 = np.linspace(0, 1, 20)
    t2 = np.linspace(0, 1, 27)
    ref = np.column_stack([np.sin(4 * np.pi * t1), np.cos(4 * np.pi * t1)])
    cur = np.column_stack([np.sin(4 * np.pi * t2), np.cos(4 * np.pi * t2)])
    aligned = _align_to_reference(cur, ref)
    assert aligned.shape == ref.shape
    assert np.isfinite(aligned).all()


def test_raw_knn_both_modes_score():
    rng = np.random.default_rng(3)
    fit = [rng.normal(size=(20 + i % 3, 2)) for i in range(12)]
    query = [rng.normal(size=(21, 2)), rng.normal(loc=5.0, size=(19, 2))]
    raw = RawCycleKNNBaseline(k=3, phase_align=False).fit(fit)
    aligned = RawCycleKNNBaseline(k=3, phase_align=True).fit(fit)
    s_raw = raw.score_cycles(query)
    s_aligned = aligned.score_cycles(query)
    assert s_raw.shape == (2,)
    assert s_aligned.shape == (2,)
    assert np.isfinite(s_raw).all()
    assert np.isfinite(s_aligned).all()
