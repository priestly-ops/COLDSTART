"""Shared-projector A1 detector for target-anchored transfer diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.base_detector import BaseDetector


@dataclass(frozen=True)
class A1Diagnostics:
    n_target: int
    n_source: int
    n_features: int
    k_effective: int
    selected_gamma: float
    gamma_risks: dict[float, float]
    gamma_risk_deltas_vs_0: dict[float, float]
    projector_alignment: float
    target_projector_mass: float
    source_projector_mass: float


class SharedProjectorA1Detector(BaseDetector):
    """A1 detector with a target-owned metric and a source-target shared projector.

    Source information enters only through the source healthy PCA projector. Target
    center, scale, scoring statistics, residual statistics, and calibration remain
    target-only.
    """

    def __init__(
        self,
        *,
        k_max: int = 16,
        gamma: float | None = None,
        gamma_grid: Iterable[float] = (0.0, 0.05, 0.10, 0.20, 0.40),
        beta: float = 0.5,
        clip: float = 8.0,
        scale_floor_relative: float = 1e-8,
        score_floor: float = 1e-8,
        false_alert_budget: float = 0.01,
        random_state: int = 42,
    ) -> None:
        super().__init__(false_alert_budget=false_alert_budget)
        if k_max < 1:
            raise ValueError("k_max must be positive.")
        if gamma is not None and not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        self.k_max = int(k_max)
        self.gamma = None if gamma is None else float(gamma)
        self.gamma_grid = tuple(float(value) for value in gamma_grid)
        self.beta = float(beta)
        self.clip = float(clip)
        self.scale_floor_relative = float(scale_floor_relative)
        self.score_floor = float(score_floor)
        self.random_state = int(random_state)

        self.target_center_: np.ndarray | None = None
        self.target_scale_: np.ndarray | None = None
        self.source_center_: np.ndarray | None = None
        self.target_pca_basis_: np.ndarray | None = None
        self.source_pca_basis_: np.ndarray | None = None
        self.shared_basis_: np.ndarray | None = None
        self.projector_eigenvalues_: np.ndarray | None = None
        self.mode_center_: np.ndarray | None = None
        self.mode_variance_: np.ndarray | None = None
        self.residual_center_: np.ndarray | None = None
        self.residual_variance_: np.ndarray | None = None
        self.selected_gamma_: float | None = None
        self.gamma_risks_: dict[float, float] | None = None
        self.gamma_risk_deltas_vs_0_: dict[float, float] | None = None
        self.diagnostics_: A1Diagnostics | None = None

    @staticmethod
    def _median_center(x: np.ndarray) -> np.ndarray:
        return np.median(x, axis=0)

    def _robust_scale(self, x: np.ndarray) -> np.ndarray:
        center = self._median_center(x)
        scale = 1.4826 * np.median(np.abs(x - center), axis=0)
        finite = scale[np.isfinite(scale) & (scale > 0)]
        fallback = float(np.median(finite)) if finite.size else 1.0
        floor = max(self.scale_floor_relative * fallback, 1e-6)
        return np.where(np.isfinite(scale) & (scale > floor), scale, max(fallback, floor))

    def _target_transform(self, x: np.ndarray) -> np.ndarray:
        if self.target_center_ is None or self.target_scale_ is None:
            raise RuntimeError("Detector must be fitted before target transform.")
        return np.clip((x - self.target_center_) / self.target_scale_, -self.clip, self.clip)

    def _source_transform_common_metric(self, x: np.ndarray) -> np.ndarray:
        if self.source_center_ is None or self.target_scale_ is None:
            raise RuntimeError("Detector must be fitted before source transform.")
        return np.clip((x - self.source_center_) / self.target_scale_, -self.clip, self.clip)

    @staticmethod
    def _pca_basis(x: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return np.zeros((x.shape[1], 0), dtype=np.float64)
        _, _, vh = np.linalg.svd(x - np.mean(x, axis=0), full_matrices=False)
        return vh[:k].T

    def _effective_k(self, n_target: int, n_source: int, n_features: int) -> int:
        return min(self.k_max, n_target - 2, n_source - 1, n_features)

    @staticmethod
    def _projector(basis: np.ndarray) -> np.ndarray:
        return basis @ basis.T

    @staticmethod
    def _shared_basis_from_projectors(
        target_projector: np.ndarray,
        source_projector: np.ndarray,
        gamma: float,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blended = (1.0 - gamma) * target_projector + gamma * source_projector
        eigenvalues, eigenvectors = np.linalg.eigh(blended)
        order = np.argsort(eigenvalues)[::-1]
        basis = eigenvectors[:, order[:k]]
        values = eigenvalues[order[:k]]
        return basis, values, blended

    def _safe_variance(self, x: np.ndarray) -> np.ndarray:
        variance = np.var(x, axis=0, ddof=1 if x.shape[0] > 1 else 0)
        finite = variance[np.isfinite(variance) & (variance > self.score_floor)]
        fallback = float(np.median(finite)) if finite.size else 1.0
        return np.where(
            np.isfinite(variance) & (variance > self.score_floor),
            variance,
            max(fallback, self.score_floor),
        )

    def _loo_gamma_risks(self, ys: np.ndarray, yt: np.ndarray, k: int) -> dict[float, float]:
        if self.gamma_grid == ():
            raise ValueError("gamma_grid cannot be empty.")
        source_basis = self._pca_basis(ys, k)
        source_projector = self._projector(source_basis)
        risks: dict[float, float] = {}
        for gamma in self.gamma_grid:
            residuals: list[float] = []
            for heldout in range(yt.shape[0]):
                mask = np.ones(yt.shape[0], dtype=bool)
                mask[heldout] = False
                target_basis = self._pca_basis(yt[mask], k)
                target_projector = self._projector(target_basis)
                basis, _, _ = self._shared_basis_from_projectors(
                    target_projector,
                    source_projector,
                    gamma,
                    k,
                )
                y = yt[heldout]
                residual = y - (y @ basis) @ basis.T
                residuals.append(float(np.mean(residual**2)))
            risks[float(gamma)] = float(np.mean(residuals))
        return risks

    @staticmethod
    def _select_gamma_one_se(risks: dict[float, float]) -> float:
        min_risk = min(risks.values())
        best = [gamma for gamma, risk in risks.items() if risk <= min_risk + 1e-12]
        return float(min(best))

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "SharedProjectorA1Detector":
        xs = self._validate_features(source_features)
        xt = self._validate_features(target_features)
        if xs.shape[1] != xt.shape[1]:
            raise ValueError("Source and target feature counts must match.")
        n_target, n_features = xt.shape
        n_source = xs.shape[0]
        k = self._effective_k(n_target, n_source, n_features)
        if k <= 0:
            raise ValueError("Not enough healthy target/source samples for A1.")

        self.target_center_ = self._median_center(xt)
        self.target_scale_ = self._robust_scale(xt)
        self.source_center_ = self._median_center(xs)
        yt = self._target_transform(xt)
        ys = self._source_transform_common_metric(xs)
        ut = self._pca_basis(yt, k)
        us = self._pca_basis(ys, k)
        pt = self._projector(ut)
        ps = self._projector(us)

        risks = self._loo_gamma_risks(ys, yt, k) if self.gamma is None else {float(self.gamma): np.nan}
        selected_gamma = self._select_gamma_one_se(risks) if self.gamma is None else float(self.gamma)
        basis, eigenvalues, blended = self._shared_basis_from_projectors(pt, ps, selected_gamma, k)
        zt = yt @ basis
        residual = yt - zt @ basis.T

        self.target_pca_basis_ = ut
        self.source_pca_basis_ = us
        self.shared_basis_ = basis
        self.projector_eigenvalues_ = eigenvalues
        self.mode_center_ = np.median(zt, axis=0)
        self.mode_variance_ = self._safe_variance(zt)
        self.residual_center_ = np.median(residual, axis=0)
        self.residual_variance_ = self._safe_variance(residual)
        self.selected_gamma_ = selected_gamma
        self.gamma_risks_ = risks
        base_risk = risks.get(0.0, np.nan)
        self.gamma_risk_deltas_vs_0_ = {
            gamma: float(risk - base_risk) if np.isfinite(base_risk) and np.isfinite(risk) else np.nan
            for gamma, risk in risks.items()
        }
        alignment = float(np.trace(pt @ ps) / k)
        self.diagnostics_ = A1Diagnostics(
            n_target=n_target,
            n_source=n_source,
            n_features=n_features,
            k_effective=k,
            selected_gamma=selected_gamma,
            gamma_risks=risks,
            gamma_risk_deltas_vs_0=self.gamma_risk_deltas_vs_0_,
            projector_alignment=alignment,
            target_projector_mass=float(np.trace(blended @ pt) / k),
            source_projector_mass=float(np.trace(blended @ ps) / k),
        )
        self.is_fitted_ = True
        return self

    def score_components(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if (
            self.shared_basis_ is None
            or self.mode_center_ is None
            or self.mode_variance_ is None
            or self.residual_center_ is None
            or self.residual_variance_ is None
        ):
            raise RuntimeError("Detector must be fitted before scoring.")
        x = self._validate_features(features)
        y = self._target_transform(x)
        z = y @ self.shared_basis_
        mode_energy = ((z - self.mode_center_) ** 2) / (self.mode_variance_ + self.score_floor)
        subspace_score = np.mean(mode_energy, axis=1)
        residual = y - z @ self.shared_basis_.T
        residual_energy = ((residual - self.residual_center_) ** 2) / (
            self.residual_variance_ + self.score_floor
        )
        residual_score = np.mean(residual_energy, axis=1)
        final_score = self.beta * subspace_score + (1.0 - self.beta) * residual_score
        return {
            "final_score": np.asarray(final_score, dtype=np.float64),
            "subspace_score": np.asarray(subspace_score, dtype=np.float64),
            "residual_score": np.asarray(residual_score, dtype=np.float64),
            "mode_energy": np.asarray(mode_energy, dtype=np.float64),
        }

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        return self.score_components(features)["final_score"]

    def get_params(self) -> dict[str, object]:
        return {
            "k_max": self.k_max,
            "gamma": self.gamma,
            "gamma_grid": self.gamma_grid,
            "beta": self.beta,
            "clip": self.clip,
            "scale_floor_relative": self.scale_floor_relative,
            "score_floor": self.score_floor,
            "false_alert_budget": self.false_alert_budget,
            "random_state": self.random_state,
        }
