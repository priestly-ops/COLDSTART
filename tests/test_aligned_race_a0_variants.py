import numpy as np

from src.aligned_race_a0 import AlignedRACEA0Detector
from src.aligned_race_a0_variants import (
    FeaturePermutedSourceRACEA0Detector,
    StabilityAwareRACEA0Detector,
    VarianceAwareRACEA0Detector,
    bootstrap_mode_stability,
    variance_agreement,
)


def _low_rank_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent_t = rng.normal(size=(80, 2))
    latent_s = rng.normal(size=(90, 2))
    mixing = np.array(
        [
            [3.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.2, 0.3, 0.0, 0.0, 0.0],
        ]
    )
    target = latent_t @ mixing + 0.03 * rng.normal(size=(80, 6))
    source = latent_s @ mixing + 0.03 * rng.normal(size=(90, 6))
    return source, target


def test_variance_agreement_is_bounded_and_equal_variance_near_one():
    source = np.array([1.0, 2.0, 4.0])
    target = np.array([1.0, 2.0, 4.0])

    agreement = variance_agreement(source, target, tau=1.0)

    assert np.all(np.isfinite(agreement))
    assert np.all((0.0 <= agreement) & (agreement <= 1.0))
    np.testing.assert_allclose(agreement, np.ones_like(agreement), atol=1e-8)


def test_increasing_variance_mismatch_lowers_agreement():
    reference = np.array([1.0])
    mild = variance_agreement(reference, np.array([2.0]), tau=1.0)
    severe = variance_agreement(reference, np.array([16.0]), tau=1.0)

    assert severe[0] < mild[0] < 1.0


def test_variance_aware_weights_are_finite_bounded_and_deterministic():
    source, target = _low_rank_pair()

    first = VarianceAwareRACEA0Detector(k_max=2, random_state=7).fit(source, target)
    second = VarianceAwareRACEA0Detector(k_max=2, random_state=7).fit(source, target)

    assert first.variance_tau_ in {0.5, 1.0, 2.0}
    assert np.all(np.isfinite(first.variance_agreement_))
    assert np.all((0.0 <= first.effective_weights_) & (first.effective_weights_ <= 1.0))
    np.testing.assert_allclose(first.effective_weights_, second.effective_weights_)


def test_variance_mismatch_suppresses_angle_only_transfer():
    source, target = _low_rank_pair()
    mismatched_source = source.copy()
    mismatched_source[:, 0] *= 12.0

    angle_only = AlignedRACEA0Detector(k_max=2, random_state=3).fit(mismatched_source, target)
    aware = VarianceAwareRACEA0Detector(k_max=2, random_state=3, variance_tau=0.5).fit(
        mismatched_source, target
    )

    assert np.mean(aware.effective_weights_) < np.mean(angle_only.effective_weights_)


def test_bootstrap_stability_is_bounded_and_separates_stable_from_random_reference():
    _, target = _low_rank_pair(seed=2)
    base = AlignedRACEA0Detector(k_max=2, random_state=11).fit(target, target)
    stable = bootstrap_mode_stability(
        target,
        base.target_principal_vectors_,
        center=base.target_center_,
        scale=base.target_scale_,
        k=2,
        resamples=30,
        random_state=11,
    )
    rng = np.random.default_rng(11)
    random_basis, _ = np.linalg.qr(rng.normal(size=base.target_principal_vectors_.shape))
    random_stability = bootstrap_mode_stability(
        target,
        random_basis,
        center=base.target_center_,
        scale=base.target_scale_,
        k=2,
        resamples=30,
        random_state=11,
    )

    assert np.all(np.isfinite(stable))
    assert np.all((0.0 <= stable) & (stable <= 1.0))
    assert float(np.median(stable)) > float(np.median(random_stability))


def test_stability_aware_weights_are_finite_bounded_and_deterministic():
    source, target = _low_rank_pair(seed=4)

    first = StabilityAwareRACEA0Detector(k_max=2, random_state=5, bootstrap_resamples=25).fit(
        source, target
    )
    second = StabilityAwareRACEA0Detector(k_max=2, random_state=5, bootstrap_resamples=25).fit(
        source, target
    )

    assert np.all(np.isfinite(first.stability_factors_))
    assert np.all((0.0 <= first.effective_weights_) & (first.effective_weights_ <= 1.0))
    np.testing.assert_allclose(first.effective_weights_, second.effective_weights_)


def test_source_feature_permutation_changes_source_target_geometry():
    source, target = _low_rank_pair(seed=6)

    real = AlignedRACEA0Detector(k_max=2, random_state=1).fit(source, target)
    permuted = FeaturePermutedSourceRACEA0Detector(k_max=2, random_state=1).fit(source, target)

    assert not np.array_equal(permuted.source_feature_permutation_, np.arange(source.shape[1]))
    assert not np.allclose(real.singular_values_, permuted.singular_values_)

