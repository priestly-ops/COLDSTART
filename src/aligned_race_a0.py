from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.base_detector import BaseDetector


A0Mode = Literal[
    "aligned",
    "target_only",
    "target_pca",
    "raw_source_pca",
    "random_subspace",
    "feature_permuted",
    "weight_permuted",
]


@dataclass(frozen=True)
class A0Diagnostics:
    mode: str
    n_target: int
    n_source: int
    n_features: int
    k_requested: int
    k_effective: int
    n_shared_directions: int
    alignment_mean_cos2: float
    alignment_min_cos2: float
    alignment_max_cos2: float
    angle_distance: float
    global_gate_open: bool
    fallback: bool
    fallback_reason: str
    singular_values: tuple[float, ...]
    raw_cos2_weights: tuple[float, ...]
    effective_weights: tuple[float, ...]


class ScoreComponents(dict[str, np.ndarray]):
    """Mapping API for v2 fields plus legacy tuple unpacking."""

    def __iter__(self):
        return iter((self["score"], self["shared_score"], self["target_specific_score"]))


class AlignedRACEA0Detector(BaseDetector):
    """Aligned RACE-A0 detector.

    Source data may only influence reliability weights assigned to target
    principal-vector coordinates. Target location, scale, PCA coordinates,
    residual statistics, and calibration remain target-only.
    """

    def __init__(
        self,
        k_max: int = 16,
        beta: float = 0.50,
        lambda_weight: float = 0.25,
        direction_min_cos2: float = 0.20,
        global_alignment_min: float = 0.20,
        clip: float = 8.0,
        scale_floor_relative: float = 1e-8,
        score_floor: float = 1e-8,
        mode: A0Mode = "aligned",
        false_alert_budget: float = 0.01,
        random_state: int = 42,
    ) -> None:
        super().__init__(false_alert_budget=false_alert_budget)
        if k_max < 1:
            raise ValueError("k_max must be positive.")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        if lambda_weight <= 0.0:
            raise ValueError("lambda_weight must be positive.")
        if clip <= 0.0:
            raise ValueError("clip must be positive.")
        if mode not in A0Mode.__args__:
            raise ValueError(f"Unknown A0 mode: {mode}")

        self.k_max = int(k_max)
        self.beta = float(beta)
        self.lambda_weight = float(lambda_weight)
        self.direction_min_cos2 = float(direction_min_cos2)
        self.global_alignment_min = float(global_alignment_min)
        self.clip = float(clip)
        self.scale_floor_relative = float(scale_floor_relative)
        self.score_floor = float(score_floor)
        self.mode = mode
        self.random_state = int(random_state)

        self.target_center_: np.ndarray | None = None
        self.target_scale_: np.ndarray | None = None
        self.source_center_: np.ndarray | None = None
        self.target_pca_basis_: np.ndarray | None = None
        self.source_pca_basis_: np.ndarray | None = None
        self.target_principal_vectors_: np.ndarray | None = None
        self.source_principal_vectors_: np.ndarray | None = None
        self.weight_permutation_: np.ndarray | None = None
        self.mode_center_: np.ndarray | None = None
        self.mode_variance_: np.ndarray | None = None
        self.residual_center_: np.ndarray | None = None
        self.residual_variance_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.raw_cos2_weights_: np.ndarray | None = None
        self.effective_weights_: np.ndarray | None = None
        self.fallback_: bool = False
        self.fallback_reason_: str = ""
        self.diagnostics_: A0Diagnostics | None = None

    @staticmethod
    def _median_center(x: np.ndarray) -> np.ndarray:
        return np.median(x, axis=0)

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        center = self._median_center(x)
        mad = np.median(np.abs(x - center), axis=0)
        scale = 1.4826 * mad
        finite = scale[np.isfinite(scale) & (scale > 0.0)]
        relative = self.scale_floor_relative * (
            float(np.median(finite)) if finite.size else 1.0
        )
        floor = max(1e-6, relative, self.score_floor)
        return np.maximum(scale, floor)

    def _target_transform(self, x: np.ndarray) -> np.ndarray:
        if self.target_center_ is None or self.target_scale_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        return np.clip((x - self.target_center_) / self.target_scale_, -self.clip, self.clip)

    def _source_transform_common_metric(self, x: np.ndarray) -> np.ndarray:
        if self.source_center_ is None or self.target_scale_ is None:
            raise RuntimeError("Detector must be fitted before source transform.")
        return np.clip((x - self.source_center_) / self.target_scale_, -self.clip, self.clip)

    @staticmethod
    def _pca_basis(x: np.ndarray, k: int) -> np.ndarray:
        centered = x - np.mean(x, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:k].T
        if basis.shape != (x.shape[1], k):
            raise RuntimeError("Unexpected PCA basis shape.")
        if not np.allclose(basis.T @ basis, np.eye(k), atol=1e-10):
            raise RuntimeError("PCA basis is not orthonormal.")
        return basis

    def _effective_k(self, n_target: int, n_source: int, n_features: int) -> int:
        return min(self.k_max, n_target - 2, n_features, n_source - 1)

    @staticmethod
    def _principal_vectors(
        source_basis: np.ndarray,
        target_basis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, singular_values, q_t = np.linalg.svd(source_basis.T @ target_basis, full_matrices=False)
        q = q_t.T
        return target_basis @ q, source_basis @ p, np.clip(singular_values, 0.0, 1.0)

    @staticmethod
    def _non_identity_permutation(k: int, seed: int) -> np.ndarray:
        permutation = np.arange(k)
        if k <= 1:
            return permutation
        rng = np.random.default_rng(seed)
        for _ in range(64):
            candidate = rng.permutation(k)
            if not np.array_equal(candidate, permutation):
                return candidate
        return np.roll(permutation, 1)

    @staticmethod
    def _safe_variance(x: np.ndarray, axis: int = 0, floor: float = 1e-8) -> np.ndarray:
        var = np.var(x, axis=axis, ddof=1 if x.shape[axis] > 1 else 0)
        finite = var[np.isfinite(var) & (var > 0.0)]
        adaptive = 0.01 * float(np.median(finite)) if finite.size else floor
        return np.maximum(var, max(floor, adaptive))

    def fit(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
    ) -> "AlignedRACEA0Detector":
        xs = self._validate_features(source_features)
        xt = self._validate_features(target_features)
        if xs.shape[1] != xt.shape[1]:
            raise ValueError("Source and target feature counts must match.")

        n_target, n_features = xt.shape
        n_source = xs.shape[0]
        self.target_center_ = self._median_center(xt)
        self.target_scale_ = self._robust_scale(xt)
        self.source_center_ = self._median_center(xs)

        yt = self._target_transform(xt)
        ys = self._source_transform_common_metric(xs)
        if self.mode == "feature_permuted":
            ys = ys[:, self._non_identity_permutation(n_features, self.random_state + 17)]

        k = self._effective_k(n_target, n_source, n_features)
        fallback = False
        fallback_reason = ""
        if k < 1:
            k = 1
            fallback = True
            fallback_reason = "insufficient_samples_for_alignment"

        ut = self._pca_basis(yt, k)
        us = self._pca_basis(ys, k)
        self.target_pca_basis_ = ut
        self.source_pca_basis_ = us

        if self.mode == "target_only":
            vt = ut
            vs = np.zeros_like(ut)
            singular_values = np.zeros(k)
            raw_weights = np.zeros(k)
            weights = np.zeros(k)
            fallback = True
            fallback_reason = fallback_reason or "target_only_mode"
        elif self.mode == "target_pca":
            vt = ut
            vs = ut
            singular_values = np.ones(k)
            raw_weights = np.ones(k)
            weights = np.ones(k)
        elif self.mode == "raw_source_pca":
            vt = us
            vs = us
            singular_values = np.ones(k)
            raw_weights = np.ones(k)
            weights = np.ones(k)
        elif self.mode == "random_subspace":
            random_matrix = np.random.default_rng(self.random_state).normal(size=(n_features, k))
            vt, _ = np.linalg.qr(random_matrix)
            vt = vt[:, :k]
            vs = vt
            singular_values = np.ones(k)
            raw_weights = np.ones(k)
            weights = np.ones(k)
        else:
            p, singular_values, q_t = np.linalg.svd(us.T @ ut, full_matrices=False)
            q = q_t.T
            vt = ut @ q
            vs = us @ p
            singular_values = np.clip(singular_values, 0.0, 1.0)
            raw_weights = singular_values**2
            weights = raw_weights / (raw_weights + self.lambda_weight)
            if not np.isfinite(weights).all():
                fallback = True
                fallback_reason = "nonfinite_alignment_weights"
                weights = np.zeros(k)
            if self.mode == "weight_permuted":
                self.weight_permutation_ = self._non_identity_permutation(k, self.random_state)
                weights = weights[self.weight_permutation_]
            else:
                self.weight_permutation_ = np.arange(k)

            diagonal = vs.T @ vt
            if not np.allclose(diagonal, np.diag(np.diag(diagonal)), atol=1e-8):
                raise RuntimeError("Principal-vector alignment is not diagonal.")
            if not np.allclose(np.diag(diagonal), singular_values, atol=1e-8):
                raise RuntimeError("Principal-vector singular values are inconsistent.")

        if not np.allclose(vt.T @ vt, np.eye(k), atol=1e-10):
            raise RuntimeError("Target principal vectors are not orthonormal.")
        if self.mode not in {"target_only"} and not np.allclose(vs.T @ vs, np.eye(k), atol=1e-10):
            raise RuntimeError("Source principal vectors are not orthonormal.")

        self.target_principal_vectors_ = vt
        self.source_principal_vectors_ = vs
        self.singular_values_ = singular_values
        self.raw_cos2_weights_ = raw_weights
        self.effective_weights_ = weights
        self.fallback_ = bool(fallback)
        self.fallback_reason_ = fallback_reason

        zt = yt @ vt
        self.mode_center_ = np.median(zt, axis=0)
        self.mode_variance_ = self._safe_variance(zt, axis=0, floor=self.score_floor)
        residual = yt - zt @ vt.T
        self.residual_center_ = np.median(residual, axis=0)
        self.residual_variance_ = self._safe_variance(
            residual, axis=0, floor=self.score_floor
        )

        mean_cos2 = float(np.mean(raw_weights)) if raw_weights.size else 0.0
        self.diagnostics_ = A0Diagnostics(
            mode=self.mode,
            n_target=n_target,
            n_source=n_source,
            n_features=n_features,
            k_requested=self.k_max,
            k_effective=k,
            n_shared_directions=int(np.sum(raw_weights >= self.direction_min_cos2)),
            alignment_mean_cos2=mean_cos2,
            alignment_min_cos2=float(np.min(raw_weights)) if raw_weights.size else 0.0,
            alignment_max_cos2=float(np.max(raw_weights)) if raw_weights.size else 0.0,
            angle_distance=float(np.mean(1.0 - raw_weights)) if raw_weights.size else 1.0,
            global_gate_open=bool(mean_cos2 >= self.global_alignment_min),
            fallback=fallback,
            fallback_reason=fallback_reason,
            singular_values=tuple(float(v) for v in singular_values),
            raw_cos2_weights=tuple(float(v) for v in raw_weights),
            effective_weights=tuple(float(v) for v in weights),
        )
        self.is_fitted_ = True
        self.is_calibrated_ = False
        self.threshold_ = None
        return self

    def score_components(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring.")
        if (
            self.target_principal_vectors_ is None
            or self.mode_center_ is None
            or self.mode_variance_ is None
            or self.residual_center_ is None
            or self.residual_variance_ is None
            or self.effective_weights_ is None
        ):
            raise RuntimeError("Detector state is incomplete.")

        x = self._validate_features(features)
        y = self._target_transform(x)
        vt = self.target_principal_vectors_
        z = y @ vt
        mode_energy = (z - self.mode_center_) ** 2 / (self.mode_variance_ + self.score_floor)
        weights = self.effective_weights_

        shared_mass = float(np.sum(weights))
        shared_contrib = mode_energy * weights
        if shared_mass > self.score_floor:
            shared_score = np.sum(shared_contrib, axis=1) / (shared_mass + self.score_floor)
        else:
            shared_score = np.zeros(x.shape[0], dtype=np.float64)

        target_weights = 1.0 - weights
        target_mass = float(np.sum(target_weights))
        target_mode_contrib = mode_energy * target_weights
        if target_mass > self.score_floor:
            target_modes_score = np.sum(target_mode_contrib, axis=1) / (
                target_mass + self.score_floor
            )
        else:
            target_modes_score = np.zeros(x.shape[0], dtype=np.float64)

        residual = y - z @ vt.T
        residual_energy = (residual - self.residual_center_) ** 2 / (
            self.residual_variance_ + self.score_floor
        )
        orthogonal_score = np.mean(residual_energy, axis=1)
        target_specific_score = 0.5 * target_modes_score + 0.5 * orthogonal_score

        if self.mode == "target_only":
            final_score = 0.5 * np.mean(mode_energy, axis=1) + 0.5 * orthogonal_score
        else:
            final_score = self.beta * shared_score + (1.0 - self.beta) * target_specific_score

        return ScoreComponents({
            "score": np.asarray(final_score, dtype=np.float64),
            "final_score": np.asarray(final_score, dtype=np.float64),
            "shared_score": np.asarray(shared_score, dtype=np.float64),
            "target_modes_score": np.asarray(target_modes_score, dtype=np.float64),
            "orthogonal_score": np.asarray(orthogonal_score, dtype=np.float64),
            "target_specific_score": np.asarray(target_specific_score, dtype=np.float64),
            "mode_energy": np.asarray(mode_energy, dtype=np.float64),
            "shared_contributions": np.asarray(shared_contrib, dtype=np.float64),
            "target_only_contributions": np.asarray(target_mode_contrib, dtype=np.float64),
        })

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        scores = self.score_components(features)["score"]
        return self._validate_scores(scores)

    def get_params(self) -> dict[str, object]:
        params = super().get_params()
        params.update(
            {
                "k_max": self.k_max,
                "beta": self.beta,
                "lambda_weight": self.lambda_weight,
                "direction_min_cos2": self.direction_min_cos2,
                "global_alignment_min": self.global_alignment_min,
                "clip": self.clip,
                "scale_floor_relative": self.scale_floor_relative,
                "score_floor": self.score_floor,
                "mode": self.mode,
                "random_state": self.random_state,
            }
        )
        return params
