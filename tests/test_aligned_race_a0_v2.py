from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.aligned_race_a0 import AlignedRACEA0Detector


def _structured_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = np.column_stack(
        [
            rng.normal(0.0, 3.0, 80),
            rng.normal(0.0, 2.0, 80),
            np.zeros(80),
            np.zeros(80),
        ]
    )
    source = np.column_stack(
        [
            rng.normal(0.0, 3.0, 100),
            np.zeros(100),
            rng.normal(0.0, 2.5, 100),
            np.zeros(100),
        ]
    )
    return source, target


def _fit(mode: str = "aligned", *, beta: float = 0.5) -> AlignedRACEA0Detector:
    source, target = _structured_data()
    return AlignedRACEA0Detector(
        k_max=2,
        beta=beta,
        lambda_weight=0.25,
        mode=mode,  # type: ignore[arg-type]
        random_state=7,
    ).fit(source, target)


def test_robust_scaling_is_finite_with_constant_features() -> None:
    rng = np.random.default_rng(1)
    target = np.column_stack([np.ones(30), rng.normal(size=(30, 3))])
    source = np.column_stack([np.ones(40), rng.normal(size=(40, 3))])
    model = AlignedRACEA0Detector(k_max=2).fit(source, target)
    scores = model.score_samples(target[:5])

    assert np.isfinite(model.target_scale_).all()
    assert np.all(model.target_scale_ > 0)
    assert np.isfinite(scores).all()


def test_k_rule_and_principal_vector_geometry() -> None:
    source, target = _structured_data()
    model = AlignedRACEA0Detector(k_max=16).fit(source, target[:10])
    diag = model.diagnostics_

    assert diag is not None
    assert diag.k_effective <= 8
    assert np.allclose(model.target_pca_basis_.T @ model.target_pca_basis_, np.eye(diag.k_effective))
    assert np.allclose(model.source_pca_basis_.T @ model.source_pca_basis_, np.eye(diag.k_effective))
    assert np.all((model.singular_values_ >= -1e-12) & (model.singular_values_ <= 1 + 1e-12))
    assert np.allclose(
        model.target_principal_vectors_.T @ model.target_principal_vectors_,
        np.eye(diag.k_effective),
    )
    assert np.allclose(
        model.source_principal_vectors_.T @ model.source_principal_vectors_,
        np.eye(diag.k_effective),
    )
    overlap = model.source_principal_vectors_.T @ model.target_principal_vectors_
    assert np.allclose(overlap, np.diag(np.diag(overlap)), atol=1e-8)
    assert np.allclose(np.diag(overlap), model.singular_values_, atol=1e-8)


def test_principal_angle_weight_ordering_is_preserved() -> None:
    model = _fit("aligned")
    raw = model.raw_cos2_weights_
    weights = model.effective_weights_

    assert np.all(np.diff(raw) <= 1e-12)
    assert np.all(np.diff(weights) <= 1e-12)


def test_weight_permuted_race_changes_only_weight_assignment() -> None:
    source, target = _structured_data()
    real = AlignedRACEA0Detector(k_max=2, mode="aligned", random_state=11).fit(source, target)
    permuted = AlignedRACEA0Detector(k_max=2, mode="weight_permuted", random_state=11).fit(source, target)

    assert not np.array_equal(permuted.weight_permutation_, np.arange(2))
    assert np.allclose(real.target_center_, permuted.target_center_)
    assert np.allclose(real.target_scale_, permuted.target_scale_)
    assert np.allclose(real.target_pca_basis_, permuted.target_pca_basis_)
    assert np.allclose(real.target_principal_vectors_, permuted.target_principal_vectors_)
    assert np.allclose(real.mode_center_, permuted.mode_center_)
    assert np.allclose(real.mode_variance_, permuted.mode_variance_)
    assert np.allclose(real.residual_center_, permuted.residual_center_)
    assert np.allclose(real.residual_variance_, permuted.residual_variance_)
    assert np.allclose(
        np.sort(real.effective_weights_),
        np.sort(permuted.effective_weights_),
    )


def test_weight_permutation_changes_shared_scores_when_modes_are_excited() -> None:
    source, target = _structured_data()
    probe = target[:12].copy()
    probe[:, 0] += np.linspace(0.0, 4.0, len(probe))
    probe[:, 1] += np.linspace(4.0, 0.0, len(probe))
    real = AlignedRACEA0Detector(k_max=2, beta=1.0, mode="aligned", random_state=13).fit(source, target)
    permuted = AlignedRACEA0Detector(
        k_max=2, beta=1.0, mode="weight_permuted", random_state=13
    ).fit(source, target)

    real_shared = real.score_components(probe)["shared_score"]
    permuted_shared = permuted.score_components(probe)["shared_score"]

    assert not np.allclose(real_shared, permuted_shared, rtol=1e-10, atol=1e-12)


def test_synthetic_positive_control_correct_mapping_beats_permuted() -> None:
    source, target = _structured_data()
    rng = np.random.default_rng(3)
    healthy_eval = np.column_stack(
        [
            rng.normal(0.0, 3.0, 80),
            rng.normal(0.0, 2.0, 80),
            np.zeros(80),
            np.zeros(80),
        ]
    )
    real = AlignedRACEA0Detector(k_max=2, beta=1.0, mode="aligned", random_state=5).fit(source, target)
    permuted = AlignedRACEA0Detector(
        k_max=2, beta=1.0, mode="weight_permuted", random_state=5
    ).fit(source, target)
    anomaly_eval = healthy_eval.copy()
    high_compatibility_direction = real.target_principal_vectors_[:, 0] * real.target_scale_
    anomaly_eval += 7.0 * high_compatibility_direction
    y = np.r_[np.zeros(len(healthy_eval)), np.ones(len(anomaly_eval))]
    real_auc = roc_auc_score(y, np.r_[real.score_samples(healthy_eval), real.score_samples(anomaly_eval)])
    permuted_auc = roc_auc_score(
        y,
        np.r_[permuted.score_samples(healthy_eval), permuted.score_samples(anomaly_eval)],
    )

    assert real_auc - permuted_auc > 0.10


def test_negative_control_does_not_create_large_artificial_effect() -> None:
    rng = np.random.default_rng(9)
    target = rng.normal(size=(80, 6))
    source = rng.normal(size=(90, 6))
    probe = rng.normal(size=(30, 6))
    real = AlignedRACEA0Detector(k_max=2, beta=1.0, mode="aligned", random_state=2).fit(source, target)
    permuted = AlignedRACEA0Detector(
        k_max=2, beta=1.0, mode="weight_permuted", random_state=2
    ).fit(source, target)

    diff = np.mean(np.abs(real.score_samples(probe) - permuted.score_samples(probe)))
    assert diff < 0.75


def test_beta_extremes_select_expected_branches() -> None:
    model0 = _fit("aligned", beta=0.0)
    model1 = _fit("aligned", beta=1.0)
    _, target = _structured_data()

    comp0 = model0.score_components(target[:10])
    comp1 = model1.score_components(target[:10])

    assert np.allclose(comp0["final_score"], comp0["target_specific_score"])
    assert np.allclose(comp1["final_score"], comp1["shared_score"])


def test_degenerate_weight_masses_are_finite() -> None:
    source, target = _structured_data()
    high = AlignedRACEA0Detector(k_max=2, lambda_weight=1e-12).fit(source, target)
    low = AlignedRACEA0Detector(k_max=2, lambda_weight=1e12).fit(source, target)

    assert np.isfinite(high.score_samples(target[:10])).all()
    assert np.isfinite(low.score_samples(target[:10])).all()


def test_exact_target_only_fallback_is_finite_and_deterministic() -> None:
    source, target = _structured_data()
    first = AlignedRACEA0Detector(k_max=2, mode="target_only", random_state=4).fit(source, target)
    second = AlignedRACEA0Detector(k_max=2, mode="target_only", random_state=4).fit(source, target)

    assert first.diagnostics_.fallback is True
    assert first.diagnostics_.fallback_reason == "target_only_mode"
    assert np.isfinite(first.score_samples(target[:10])).all()
    assert np.allclose(first.score_samples(target[:10]), second.score_samples(target[:10]))


def test_partition_id_disjointness_check_shape() -> None:
    source = {"s1", "s2"}
    commission = {"t1", "t2"}
    calibration = {"c1"}
    healthy_eval = {"h1"}
    anomaly_eval = {"a1"}
    groups = [source, commission, calibration, healthy_eval, anomaly_eval]

    for i, left in enumerate(groups):
        for right in groups[i + 1 :]:
            assert left.isdisjoint(right)
