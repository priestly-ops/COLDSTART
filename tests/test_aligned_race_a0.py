from __future__ import annotations

import numpy as np

from src.aligned_race_a0 import AlignedRACEA0Detector


def make_transferable(seed: int = 7):
    rng = np.random.default_rng(seed)
    d = 24
    k_true = 4
    q, _ = np.linalg.qr(rng.normal(size=(d, k_true)))

    def sample(n: int, shift: float):
        z = rng.normal(size=(n, k_true)) * np.array([3.0, 2.0, 1.5, 1.0])
        residual = 0.25 * rng.normal(size=(n, d))
        return z @ q.T + residual + shift

    Xs = sample(300, 2.0)
    Xt = sample(20, -1.0)
    Xh = sample(100, -1.0)

    Xa = sample(100, -1.0)
    Xa[:, :3] += 4.0
    return Xs, Xt, Xh, Xa


def test_principal_vector_identity():
    rng = np.random.default_rng(0)
    q1, _ = np.linalg.qr(rng.normal(size=(30, 6)))
    q2, _ = np.linalg.qr(rng.normal(size=(30, 6)))

    vt, vs, s = AlignedRACEA0Detector._principal_vectors(q1, q2)
    cross = vs.T @ vt

    assert np.allclose(cross, np.diag(s), atol=1e-10)
    assert np.all((s >= 0.0) & (s <= 1.0))


def test_target_only_fallback_is_finite():
    Xs, Xt, Xh, _ = make_transferable()
    model = AlignedRACEA0Detector(mode="target_only", k_max=8)
    model.fit(Xs, Xt)
    scores = model.score_samples(Xh)

    assert model.fallback_
    assert model.diagnostics_ is not None
    assert model.diagnostics_.n_shared_directions == 0
    assert scores.shape == (len(Xh),)
    assert np.isfinite(scores).all()


def test_aligned_model_has_consistent_component_shapes():
    Xs, Xt, Xh, _ = make_transferable()
    model = AlignedRACEA0Detector(
        mode="aligned",
        k_max=8,
        direction_min_cos2=0.0,
        global_alignment_min=0.0,
    )
    model.fit(Xs, Xt)
    total, shared, residual = model.score_components(Xh)

    assert total.shape == shared.shape == residual.shape == (len(Xh),)
    assert np.isfinite(total).all()
    assert model.diagnostics_ is not None
    assert 0.0 <= model.diagnostics_.alignment_mean_cos2 <= 1.0


def test_conformal_calibration_uses_healthy_scores_only():
    Xs, Xt, Xh, Xa = make_transferable()
    model = AlignedRACEA0Detector(
        mode="aligned",
        k_max=8,
        direction_min_cos2=0.0,
        global_alignment_min=0.0,
        false_alert_budget=0.01,
    )
    model.fit(Xs, Xt)
    model.calibrate(Xh)
    pred = model.predict(Xa)

    assert model.threshold_ is not None
    assert pred.shape == (len(Xa),)
    assert set(np.unique(pred)).issubset({0, 1})
