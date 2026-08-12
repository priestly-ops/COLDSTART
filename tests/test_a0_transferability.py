import numpy as np

from src.a0_transferability import (
    audit_pair,
    healthy_transferability_metrics,
    principal_angle_cos2,
    projector_similarity,
    robust_center_scale,
)


def test_principal_angle_cos2_identity_and_orthogonal():
    basis = np.eye(3)[:, :2]
    orthogonal = np.eye(3)[:, 1:]

    np.testing.assert_allclose(principal_angle_cos2(basis, basis), np.ones(2))
    cos2 = principal_angle_cos2(basis, orthogonal)
    assert np.all((0.0 <= cos2) & (cos2 <= 1.0))
    assert cos2[-1] == 0.0


def test_projector_similarity_is_bounded():
    basis = np.eye(4)[:, :2]
    rotated = basis @ np.array([[0.0, 1.0], [1.0, 0.0]])

    assert projector_similarity(basis, rotated) == 1.0


def test_robust_center_scale_handles_constant_features():
    x = np.column_stack([np.ones(10), np.arange(10, dtype=float)])

    center, scale = robust_center_scale(x)

    assert np.all(np.isfinite(center))
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0.0)


def test_audit_pair_outputs_finite_healthy_only_metrics():
    rng = np.random.default_rng(5)
    source = rng.normal(size=(40, 6))
    target = rng.normal(loc=0.2, scale=1.1, size=(12, 6))

    audit, cos2 = audit_pair(
        source,
        target,
        dataset="synthetic",
        source_domain="source",
        target_domain="target",
        n_target=12,
        seed=0,
        k_max=4,
        bootstrap_resamples=10,
    )

    assert audit.dataset == "synthetic"
    assert audit.k == 4
    assert len(cos2) == 4
    assert np.all(np.isfinite(cos2))
    assert 0.0 <= audit.alignment_mean_cos2 <= 1.0
    assert np.isfinite(audit.standardized_mean_distance)


def test_healthy_transferability_metrics_are_finite_and_directional():
    rng = np.random.default_rng(11)
    target = rng.normal(size=(30, 5))
    near = target + rng.normal(scale=0.01, size=target.shape)
    far = rng.normal(loc=3.0, size=(30, 5))

    near_metrics = healthy_transferability_metrics(near, target, k_max=3)
    far_metrics = healthy_transferability_metrics(far, target, k_max=3)

    expected = {
        "mean_shift_distance",
        "standardized_mean_shift",
        "covariance_discrepancy",
        "projector_discrepancy",
        "projector_similarity",
        "mmd_rbf",
        "wasserstein_diag",
    }
    assert expected.issubset(near_metrics)
    assert all(np.isfinite(value) for value in near_metrics.values())
    assert far_metrics["mean_shift_distance"] > near_metrics["mean_shift_distance"]
