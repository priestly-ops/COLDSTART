"""Principled RACE-A0 compatibility variants and source-specific controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.aligned_race_a0 import AlignedRACEA0Detector


@dataclass(frozen=True)
class CompatibilityDiagnostics:
    variant: str
    variance_tau: float | None
    bootstrap_resamples: int | None
    angle_mean: float
    variance_agreement_mean: float | None
    stability_mean: float | None


def variance_agreement(
    source_variance: np.ndarray,
    target_variance: np.ndarray,
    *,
    tau: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Healthy-only variance agreement in paired principal-vector modes."""
    if tau <= 0:
        raise ValueError("tau must be positive.")
    source = np.asarray(source_variance, dtype=np.float64)
    target = np.asarray(target_variance, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source and target variances must have the same shape.")
    mismatch = np.abs(np.log(target + eps) - np.log(source + eps))
    agreement = np.exp(-mismatch / tau)
    return np.clip(agreement, 0.0, 1.0)


def fit_tau_by_healthy_risk(
    source_variance: np.ndarray,
    target_variance: np.ndarray,
    grid: Iterable[float] = (0.5, 1.0, 2.0),
    *,
    eps: float = 1e-8,
) -> float:
    """Select tau from healthy-only variance predictive risk."""
    source = np.asarray(source_variance, dtype=np.float64)
    target = np.asarray(target_variance, dtype=np.float64)
    best_tau: float | None = None
    best_risk = np.inf
    for tau in grid:
        agreement = variance_agreement(source, target, tau=float(tau), eps=eps)
        predicted = agreement * source + (1.0 - agreement) * target
        risk = float(np.mean((np.log(predicted + eps) - np.log(target + eps)) ** 2))
        if risk < best_risk:
            best_risk = risk
            best_tau = float(tau)
    if best_tau is None:
        raise ValueError("tau grid cannot be empty.")
    return best_tau


def bootstrap_mode_stability(
    target_features: np.ndarray,
    reference_vectors: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    k: int,
    resamples: int = 50,
    random_state: int = 42,
    clip: float = 8.0,
) -> np.ndarray:
    """Estimate target principal-vector stability from healthy resampling."""
    target = np.asarray(target_features, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if target.ndim != 2 or reference.ndim != 2:
        raise ValueError("target_features and reference_vectors must be matrices.")
    if k <= 0:
        return np.zeros(0, dtype=np.float64)
    if resamples <= 0:
        raise ValueError("resamples must be positive.")

    y = np.clip((target - center) / scale, -clip, clip)
    rng = np.random.default_rng(random_state)
    scores: list[np.ndarray] = []
    for _ in range(resamples):
        indices = rng.integers(0, y.shape[0], size=y.shape[0])
        sample = y[indices]
        _, _, vh = np.linalg.svd(sample - np.mean(sample, axis=0), full_matrices=False)
        basis = vh[:k].T
        overlaps = np.clip(basis.T @ reference, -1.0, 1.0) ** 2
        scores.append(np.max(overlaps, axis=0))
    return np.clip(np.median(np.vstack(scores), axis=0), 0.0, 1.0)


class VarianceAwareRACEA0Detector(AlignedRACEA0Detector):
    """A0 with angle compatibility gated by healthy variance agreement."""

    def __init__(self, *args, variance_tau: float | None = None, tau_grid=(0.5, 1.0, 2.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.variance_tau = variance_tau
        self.tau_grid = tuple(float(value) for value in tau_grid)
        self.variance_agreement_: np.ndarray | None = None
        self.compatibility_diagnostics_: CompatibilityDiagnostics | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "VarianceAwareRACEA0Detector":
        super().fit(source_features, target_features)
        if self.mode in {"target_only", "target_pca"} or self.target_principal_vectors_ is None:
            return self

        ys = self._source_transform_common_metric(source_features)
        yt = self._target_transform(target_features)
        vt = self.target_principal_vectors_
        vs = self.source_principal_vectors_
        var_t = self._safe_variance(yt @ vt, axis=0, floor=self.score_floor)
        var_s = self._safe_variance(ys @ vs, axis=0, floor=self.score_floor)
        tau = self.variance_tau
        if tau is None:
            tau = fit_tau_by_healthy_risk(var_s, var_t, self.tau_grid, eps=self.score_floor)
        agreement = variance_agreement(var_s, var_t, tau=float(tau), eps=self.score_floor)
        self.variance_tau_ = float(tau)
        self.variance_agreement_ = agreement
        self.effective_weights_ = np.clip(self.raw_cos2_weights_ * agreement, 0.0, 1.0)
        self.compatibility_diagnostics_ = CompatibilityDiagnostics(
            variant="variance_aware",
            variance_tau=float(tau),
            bootstrap_resamples=None,
            angle_mean=float(np.mean(self.raw_cos2_weights_)),
            variance_agreement_mean=float(np.mean(agreement)),
            stability_mean=None,
        )
        return self


class StabilityAwareRACEA0Detector(AlignedRACEA0Detector):
    """A0 with angle compatibility gated by target bootstrap stability."""

    def __init__(self, *args, bootstrap_resamples: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.bootstrap_resamples = int(bootstrap_resamples)
        self.stability_factors_: np.ndarray | None = None
        self.compatibility_diagnostics_: CompatibilityDiagnostics | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "StabilityAwareRACEA0Detector":
        super().fit(source_features, target_features)
        if self.mode in {"target_only", "target_pca"} or self.target_principal_vectors_ is None:
            return self

        k = self.target_principal_vectors_.shape[1]
        stability = bootstrap_mode_stability(
            target_features,
            self.target_principal_vectors_,
            center=self.target_center_,
            scale=self.target_scale_,
            k=k,
            resamples=self.bootstrap_resamples,
            random_state=self.random_state,
            clip=self.clip,
        )
        self.stability_factors_ = stability
        self.effective_weights_ = np.clip(self.raw_cos2_weights_ * stability, 0.0, 1.0)
        self.compatibility_diagnostics_ = CompatibilityDiagnostics(
            variant="stability_aware",
            variance_tau=None,
            bootstrap_resamples=self.bootstrap_resamples,
            angle_mean=float(np.mean(self.raw_cos2_weights_)),
            variance_agreement_mean=None,
            stability_mean=float(np.mean(stability)),
        )
        return self


class FeaturePermutedSourceRACEA0Detector(AlignedRACEA0Detector):
    """Control that destroys source feature semantics before source geometry."""

    def __init__(self, *args, feature_permutation_seed: int = 13_337, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_permutation_seed = int(feature_permutation_seed)
        self.source_feature_permutation_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "FeaturePermutedSourceRACEA0Detector":
        xs = np.asarray(source_features, dtype=np.float64)
        rng = np.random.default_rng(self.feature_permutation_seed)
        permutation = rng.permutation(xs.shape[1])
        if xs.shape[1] > 1 and np.array_equal(permutation, np.arange(xs.shape[1])):
            permutation = np.roll(permutation, 1)
        self.source_feature_permutation_ = permutation
        return super().fit(xs[:, permutation], target_features)
