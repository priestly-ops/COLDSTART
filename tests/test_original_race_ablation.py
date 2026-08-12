import numpy as np

from src.detectors import RACEDetector
from src.original_race_ablation import (
    FROZEN_ORIGINAL_RACE_ABLATIONS,
    OriginalRaceComponentDetector,
    build_original_race_component,
    covariance_condition,
    directional_original_race_audit,
    fit_source_target_gaussians,
)


def _source_target(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source = rng.normal(size=(80, 6))
    target = rng.normal(size=(16, 6))
    source = source @ np.diag([1.8, 1.1, 0.6, 1.4, 0.8, 1.2])
    target = target @ np.diag([0.7, 1.5, 1.0, 0.9, 1.3, 0.5])
    source[:, 0] += 0.4
    target[:, 1] -= 0.2
    return source, target


def test_frozen_original_race_ablation_components_are_psd():
    source, target = _source_target()
    fit = fit_source_target_gaussians(source, target)

    for variant in FROZEN_ORIGINAL_RACE_ABLATIONS:
        mean, covariance = build_original_race_component(fit, variant)
        min_eigenvalue, max_eigenvalue, condition = covariance_condition(covariance)

        assert mean.shape == (source.shape[1],)
        assert covariance.shape == (source.shape[1], source.shape[1])
        assert np.isfinite(covariance).all()
        assert min_eigenvalue > 0.0
        assert max_eigenvalue >= min_eigenvalue
        assert np.isfinite(condition)


def test_component_detectors_score_and_calibrate_finitely():
    source, target = _source_target(seed=2)
    calibration = target[:10]

    for variant in FROZEN_ORIGINAL_RACE_ABLATIONS:
        model = OriginalRaceComponentDetector(variant=variant).fit(source, target)
        scores = model.score_samples(calibration)
        model.calibrate_from_scores(scores)

        assert np.isfinite(scores).all()
        assert model.threshold_ is not None
        assert np.isfinite(model.threshold_)


def test_directional_audit_can_run_without_anomaly_labels():
    source, target = _source_target(seed=4)
    fit = fit_source_target_gaussians(source, target)
    rows = directional_original_race_audit(
        fit,
        calibration=target[:8],
        healthy_eval=target[8:],
    )

    assert len(rows) == target.shape[1]
    assert "posthoc_target_direction_separation" not in rows[0]
    assert all(0.0 <= row["healthy_compatibility"] <= 1.0 for row in rows)


def test_directional_audit_marks_anomaly_terms_as_posthoc():
    source, target = _source_target(seed=5)
    anomaly = target + 3.0
    fit = fit_source_target_gaussians(source, target)
    rows = directional_original_race_audit(
        fit,
        calibration=target[:8],
        healthy_eval=target[8:],
        anomaly_eval=anomaly,
    )

    assert "posthoc_target_direction_separation" in rows[0]
    assert "posthoc_race_direction_separation" in rows[0]
    assert "posthoc_direction_separation_change" in rows[0]


def test_original_race_ablation_matches_historical_race_scores():
    source, target = _source_target(seed=7)
    probe = np.vstack([target[:5], source[:5]])

    historical = RACEDetector().fit(source, target)
    ablation = OriginalRaceComponentDetector(variant="OriginalRACE").fit(source, target)

    np.testing.assert_allclose(ablation.location_, historical.location_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(ablation.score_samples(probe), historical.score_samples(probe), rtol=1e-8, atol=1e-8)
