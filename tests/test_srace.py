import numpy as np

from src.srace import SelectiveRACEDetector, score_equivalence_stats


def _source_target_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(32, 8))
    transform = np.diag([1.1, 0.9, 1.2, 0.8, 1.0, 1.0, 0.7, 1.3])
    source = rng.normal(size=(80, 8)) @ transform
    source[:, 0] += 0.15
    return source, target


def test_srace_keeps_target_location_anchor():
    source, target = _source_target_pair()

    model = SelectiveRACEDetector(random_state=3).fit(source, target)

    np.testing.assert_allclose(model.location_, np.mean(target, axis=0), atol=1e-10)


def test_srace_scores_are_finite_and_weights_are_bounded():
    source, target = _source_target_pair(seed=2)

    model = SelectiveRACEDetector(random_state=7).fit(source, target)
    scores = model.score_samples(target[:5])

    assert np.all(np.isfinite(scores))
    assert model.transfer_weights_ is not None
    assert np.all((0.0 <= model.transfer_weights_) & (model.transfer_weights_ <= 1.0))
    assert model.compatibility_ is not None
    assert np.all((0.0 <= model.compatibility_) & (model.compatibility_ <= 1.0))
    assert model.structural_compatibility_ is not None
    assert np.all((0.0 <= model.structural_compatibility_) & (model.structural_compatibility_ <= 1.0))
    assert model.principal_cos2_ is not None
    assert np.all((0.0 <= model.principal_cos2_) & (model.principal_cos2_ <= 1.0))
    assert model.variance_compatibility_ is not None
    assert model.location_compatibility_ is not None
    assert model.pre_gate_compatibility_ is not None
    assert model.pre_gate_transfer_weights_ is not None
    assert model.pre_gate_transfer_weights_.shape == model.transfer_weights_.shape


def test_srace_source_permutation_changes_internal_transfer_geometry():
    source, target = _source_target_pair(seed=4)

    real = SelectiveRACEDetector(random_state=11).fit(source, target)
    permuted = SelectiveRACEDetector(mode="source_permutation", random_state=11).fit(source, target)

    assert real.source_projected_variance_ is not None
    assert permuted.source_projected_variance_ is not None
    assert not np.allclose(real.source_projected_variance_, permuted.source_projected_variance_)


def test_srace_uses_conservative_shared_rank_and_private_target_structure():
    source, target = _source_target_pair(seed=5)
    small_target = target[:10]

    model = SelectiveRACEDetector(safe_gate_tolerance=1e9, random_state=12).fit(source, small_target)

    assert model.diagnostics_ is not None
    assert model.diagnostics_.shared_rank < small_target.shape[0]
    assert model.diagnostics_.private_dimensions == small_target.shape[1] - model.diagnostics_.shared_rank
    assert model.transfer_weights_ is not None
    assert np.allclose(model.transfer_weights_[model.diagnostics_.shared_rank :], 0.0)
    assert model.principal_cos2_ is not None
    assert len(model.principal_cos2_) == model.diagnostics_.shared_rank
    assert model.diagnostics_.active_structural_compatibility_mean >= model.diagnostics_.structural_compatibility_mean


def test_srace_compatibility_permutation_changes_weights_when_transfer_active():
    source, target = _source_target_pair(seed=6)

    real = SelectiveRACEDetector(
        safe_gate_tolerance=1e9,
        random_state=13,
    ).fit(source, target)
    permuted = SelectiveRACEDetector(
        mode="compatibility_permutation",
        safe_gate_tolerance=1e9,
        random_state=13,
    ).fit(source, target)

    assert real.compatibility_ is not None
    assert permuted.compatibility_ is not None
    assert not np.allclose(real.compatibility_, permuted.compatibility_)


def test_srace_gate_can_fall_back_to_target_only():
    source, target = _source_target_pair(seed=8)
    shifted_source = source + 100.0

    model = SelectiveRACEDetector(random_state=17).fit(shifted_source, target)

    assert model.diagnostics_ is not None
    if model.diagnostics_.fallback:
        assert model.diagnostics_.fallback_reason == "healthy_loo_gate_closed"
        assert np.allclose(model.transfer_weights_, 0.0)
        assert model.pre_gate_transfer_weights_ is not None
        assert np.any(model.pre_gate_transfer_weights_ >= 0.0)


def test_srace_records_safe_gate_margin_and_pre_gate_terms():
    source, target = _source_target_pair(seed=10)

    model = SelectiveRACEDetector(random_state=19).fit(source, target)

    assert model.diagnostics_ is not None
    assert np.isfinite(model.diagnostics_.safe_gate_margin) or model.diagnostics_.safe_gate_margin == float("-inf")
    assert np.isfinite(model.diagnostics_.pre_gate_weight_mean)
    assert np.isfinite(model.diagnostics_.variance_compatibility_mean)
    assert np.isfinite(model.diagnostics_.location_compatibility_mean)


def test_score_equivalence_flags_affine_rescaling():
    x = np.linspace(1.0, 10.0, 20)
    y = 0.5 * x + 3.0

    stats = score_equivalence_stats(x, y)

    assert stats["structural_score_equivalence"]
    assert stats["score_equivalence_flag"] == "STRUCTURAL_SCORE_EQUIVALENCE"
