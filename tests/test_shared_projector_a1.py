import numpy as np

from src.shared_projector_a1 import SharedProjectorA1Detector


def _pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mixing = np.array(
        [
            [2.5, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.8, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.1, 0.2, 0.0],
        ]
    )
    source = rng.normal(size=(80, 3)) @ mixing + 0.05 * rng.normal(size=(80, 6))
    target = rng.normal(size=(30, 3)) @ mixing + 0.05 * rng.normal(size=(30, 6))
    return source, target


def test_gamma_zero_equals_target_projector_basis():
    source, target = _pair()

    model = SharedProjectorA1Detector(k_max=3, gamma=0.0).fit(source, target)

    target_projector = model.target_pca_basis_ @ model.target_pca_basis_.T
    shared_projector = model.shared_basis_ @ model.shared_basis_.T
    np.testing.assert_allclose(shared_projector, target_projector, atol=1e-8)


def test_projector_blend_eigenvalues_are_finite_bounded_and_sorted():
    source, target = _pair(seed=2)

    model = SharedProjectorA1Detector(k_max=3, gamma=0.2).fit(source, target)

    assert np.all(np.isfinite(model.projector_eigenvalues_))
    assert np.all(model.projector_eigenvalues_ <= 1.0 + 1e-10)
    assert np.all(model.projector_eigenvalues_ >= -1e-10)
    assert np.all(np.diff(model.projector_eigenvalues_) <= 1e-10)
    np.testing.assert_allclose(model.shared_basis_.T @ model.shared_basis_, np.eye(3), atol=1e-10)


def test_gamma_selection_is_deterministic_and_from_grid():
    source, target = _pair(seed=3)

    first = SharedProjectorA1Detector(k_max=3, gamma=None, random_state=9).fit(source, target)
    second = SharedProjectorA1Detector(k_max=3, gamma=None, random_state=123).fit(source, target)

    assert first.selected_gamma_ in {0.0, 0.05, 0.10, 0.20, 0.40}
    assert first.selected_gamma_ == second.selected_gamma_
    assert set(first.gamma_risks_) == {0.0, 0.05, 0.10, 0.20, 0.40}


def test_scores_and_conformal_calibration_are_finite():
    source, target = _pair(seed=4)
    calibration = target[:20]
    evaluation = target[20:]

    model = SharedProjectorA1Detector(k_max=3, gamma=0.1).fit(source, target)
    model.calibrate(calibration)
    scores = model.score_samples(evaluation)

    assert model.is_calibrated_
    assert np.isfinite(model.threshold_)
    assert np.all(np.isfinite(scores))


def test_anomaly_values_do_not_affect_fitted_state():
    source, target = _pair(seed=5)
    anomaly_a = target + 10.0
    anomaly_b = target - 10.0

    first = SharedProjectorA1Detector(k_max=3, gamma=None).fit(source, target)
    first_scores = first.score_samples(anomaly_a)
    second = SharedProjectorA1Detector(k_max=3, gamma=None).fit(source, target)
    second_scores = second.score_samples(anomaly_b)

    np.testing.assert_allclose(first.shared_basis_ @ first.shared_basis_.T, second.shared_basis_ @ second.shared_basis_.T)
    assert first.selected_gamma_ == second.selected_gamma_
    assert not np.allclose(first_scores, second_scores)


def test_k_rule_matches_target_sample_limit():
    rng = np.random.default_rng(6)
    source = rng.normal(size=(80, 12))
    target = rng.normal(size=(30, 12))

    model = SharedProjectorA1Detector(k_max=16, gamma=0.0).fit(source, target[:10])

    assert model.diagnostics_.k_effective == 8
