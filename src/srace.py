from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf

from src.base_detector import BaseDetector


SRACEMode = Literal[
    "srace",
    "source_permutation",
    "compatibility_permutation",
    "target_only",
]


@dataclass(frozen=True)
class SRACEDiagnostics:
    mode: str
    n_source: int
    n_target: int
    n_features: int
    source_prior_strength: float
    safe_gate_open: bool
    safe_gate_margin: float
    safe_gate_metric: str
    fallback: bool
    fallback_reason: str
    transferred_dimensions: int
    weight_mean: float
    weight_median: float
    weight_max: float
    compatibility_mean: float
    compatibility_median: float
    compatibility_max: float
    structural_compatibility_mean: float
    structural_compatibility_median: float
    structural_compatibility_max: float
    active_structural_compatibility_mean: float
    active_structural_compatibility_median: float
    active_structural_compatibility_max: float
    principal_cos2_mean: float
    principal_cos2_median: float
    principal_cos2_min: float
    principal_cos2_max: float
    pre_gate_weight_mean: float
    pre_gate_weight_median: float
    pre_gate_weight_max: float
    pre_gate_compatibility_mean: float
    pre_gate_compatibility_median: float
    pre_gate_compatibility_max: float
    variance_compatibility_mean: float
    variance_compatibility_median: float
    variance_compatibility_max: float
    location_compatibility_mean: float
    location_compatibility_median: float
    location_compatibility_max: float
    target_uncertainty: float
    shared_rank: int
    private_dimensions: int
    condition_number: float
    effective_rank: float
    min_eigenvalue: float
    max_eigenvalue: float
    source_shrinkage: float
    target_shrinkage: float


def _as_symmetric(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (values + values.T)


def _safe_eigh(covariance: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(_as_symmetric(covariance))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], floor)
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors


def _covariance_diagnostics(eigenvalues: np.ndarray) -> dict[str, float]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        effective_rank = 0.0
    else:
        proportions = values / total
        effective_rank = float(np.exp(-np.sum(proportions * np.log(proportions + 1e-12))))
    return {
        "condition_number": float(np.max(values) / max(float(np.min(values)), 1e-12)),
        "effective_rank": effective_rank,
        "min_eigenvalue": float(np.min(values)),
        "max_eigenvalue": float(np.max(values)),
    }


def _rank_cap(eigenvalues: np.ndarray, n_samples: int, explained_variance: float = 0.95) -> int:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return 0
    finite = np.where(np.isfinite(values), values, 0.0)
    finite = np.maximum(finite, 0.0)
    sample_cap = max(1, min(values.size, (int(n_samples) - 1) // 2))
    total = float(np.sum(finite))
    if total <= 0.0:
        return sample_cap
    cumulative = np.cumsum(finite) / total
    variance_rank = int(np.searchsorted(cumulative, explained_variance, side="left") + 1)
    return max(1, min(sample_cap, variance_rank, values.size))


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _non_identity_permutation(length: int, seed: int) -> np.ndarray:
    base = np.arange(length)
    if length <= 1:
        return base
    rng = np.random.default_rng(seed)
    for _ in range(64):
        candidate = rng.permutation(length)
        if not np.array_equal(candidate, base):
            return candidate
    return np.roll(base, 1)


def score_equivalence_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool | str]:
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Score vectors must have the same shape.")

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rank_x = np.asarray(np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort"), dtype=np.float64)
    rank_y = np.asarray(np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort"), dtype=np.float64)
    concordant = 0
    discordant = 0
    n = len(x)
    for i in range(max(0, n - 1)):
        product = (x[i + 1 :] - x[i]) * (y[i + 1 :] - y[i])
        concordant += int(np.sum(product > 0.0))
        discordant += int(np.sum(product < 0.0))
    denom = n * (n - 1) / 2
    kendall = float((concordant - discordant) / denom) if denom else float("nan")

    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ratios = np.divide(y, x, out=np.full_like(y, np.nan), where=np.abs(x) > 1e-12)
    ratios = ratios[np.isfinite(ratios)]
    affine_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    structural_equivalence = bool(
        np.isfinite(affine_r2)
        and affine_r2 >= 0.9999
        and corr(rank_x, rank_y) >= 0.999
        and discordant <= max(1, int(0.001 * max(1, denom)))
    )
    return {
        "pearson_score_corr": corr(x, y),
        "spearman_score_corr": corr(rank_x, rank_y),
        "kendall_tau": kendall,
        "best_affine_slope": float(slope),
        "best_affine_intercept": float(intercept),
        "affine_r2": affine_r2,
        "median_score_ratio": float(np.median(ratios)) if ratios.size else float("nan"),
        "number_changed_rankings": int(discordant),
        "score_equivalence_flag": "STRUCTURAL_SCORE_EQUIVALENCE" if structural_equivalence else "",
        "structural_score_equivalence": structural_equivalence,
    }


class SelectiveRACEDetector(BaseDetector):
    """Selective Robust Adaptation for Cold-start Estimation.

    S-RACE keeps target location fixed and transfers only source covariance
    information that is compatible with target healthy commissioning samples.
    Transfer is per target covariance direction and is disabled by a
    healthy-only leave-one-out predictive-likelihood gate when adaptation
    worsens target-normal modeling.
    """

    def __init__(
        self,
        *,
        source_prior_strength: float = 20.0,
        compatibility_log_tau: float = 1.0,
        median_shift_tau: float = 2.0,
        compatibility_floor: float = 0.05,
        min_eigenvalue: float = 1e-8,
        safe_gate_tolerance: float = 0.01,
        mode: SRACEMode = "srace",
        random_state: int = 42,
        false_alert_budget: float = 0.01,
    ) -> None:
        super().__init__(false_alert_budget=false_alert_budget)
        if source_prior_strength <= 0.0:
            raise ValueError("source_prior_strength must be positive.")
        if compatibility_log_tau <= 0.0 or median_shift_tau <= 0.0:
            raise ValueError("compatibility temperatures must be positive.")
        if not 0.0 <= compatibility_floor <= 1.0:
            raise ValueError("compatibility_floor must be in [0, 1].")
        if min_eigenvalue <= 0.0:
            raise ValueError("min_eigenvalue must be positive.")
        if mode not in SRACEMode.__args__:
            raise ValueError(f"Unknown S-RACE mode: {mode}")

        self.source_prior_strength = float(source_prior_strength)
        self.compatibility_log_tau = float(compatibility_log_tau)
        self.median_shift_tau = float(median_shift_tau)
        self.compatibility_floor = float(compatibility_floor)
        self.min_eigenvalue = float(min_eigenvalue)
        self.safe_gate_tolerance = float(safe_gate_tolerance)
        self.mode = mode
        self.random_state = int(random_state)

        self.location_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.target_covariance_: np.ndarray | None = None
        self.source_covariance_: np.ndarray | None = None
        self.target_eigenvectors_: np.ndarray | None = None
        self.target_variance_: np.ndarray | None = None
        self.source_projected_variance_: np.ndarray | None = None
        self.structural_compatibility_: np.ndarray | None = None
        self.principal_cos2_: np.ndarray | None = None
        self.variance_compatibility_: np.ndarray | None = None
        self.location_compatibility_: np.ndarray | None = None
        self.pre_gate_compatibility_: np.ndarray | None = None
        self.pre_gate_transfer_weights_: np.ndarray | None = None
        self.compatibility_: np.ndarray | None = None
        self.transfer_weights_: np.ndarray | None = None
        self.adapted_variance_: np.ndarray | None = None
        self.diagnostics_: SRACEDiagnostics | None = None

    @staticmethod
    def _fit_lw(features: np.ndarray) -> LedoitWolf:
        estimator = LedoitWolf(assume_centered=False, store_precision=False)
        estimator.fit(features)
        return estimator

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "SelectiveRACEDetector":
        xs = self._validate_features(source_features)
        xt = self._validate_features(target_features)
        if xs.shape[1] != xt.shape[1]:
            raise ValueError("Source and target feature dimensions differ.")
        if xs.shape[0] < 2 or xt.shape[0] < 2:
            raise ValueError("At least two source and target cycles are required.")

        if self.mode == "source_permutation":
            permutation = _non_identity_permutation(xs.shape[1], self.random_state + 17)
            xs = xs[:, permutation]

        source_estimator = self._fit_lw(xs)
        target_estimator = self._fit_lw(xt)
        mu_t = np.asarray(target_estimator.location_, dtype=np.float64)
        cov_s = _as_symmetric(np.asarray(source_estimator.covariance_, dtype=np.float64))
        cov_t = _as_symmetric(np.asarray(target_estimator.covariance_, dtype=np.float64))

        target_values, target_vectors = _safe_eigh(cov_t, self.min_eigenvalue)
        source_values, source_vectors = _safe_eigh(cov_s, self.min_eigenvalue)
        shared_rank = min(
            _rank_cap(target_values, xt.shape[0]),
            _rank_cap(source_values, xs.shape[0]),
            target_vectors.shape[1],
        )
        source_shared = source_vectors[:, :shared_rank]
        target_shared = target_vectors[:, :shared_rank]
        principal_cos2 = np.linalg.svd(target_shared.T @ source_shared, compute_uv=False) ** 2
        principal_cos2 = np.clip(principal_cos2, 0.0, 1.0)
        structural_compatibility = np.sum((target_vectors.T @ source_shared) ** 2, axis=1)
        structural_compatibility = np.clip(structural_compatibility, 0.0, 1.0)
        structural_compatibility[shared_rank:] = 0.0
        source_projected = np.maximum(
            np.einsum("ij,jk,ik->i", target_vectors.T, cov_s, target_vectors.T, optimize=True),
            self.min_eigenvalue,
        )
        target_projected = np.maximum(target_values, self.min_eigenvalue)

        zt = (xt - mu_t) @ target_vectors
        zs = (xs - np.asarray(source_estimator.location_, dtype=np.float64)) @ target_vectors
        var_agreement = np.exp(
            -np.abs(np.log(source_projected) - np.log(target_projected)) / self.compatibility_log_tau
        )
        median_shift = np.abs(np.median(zs, axis=0) - np.median(zt, axis=0)) / np.sqrt(target_projected)
        location_agreement = np.exp(-median_shift / self.median_shift_tau)
        compatibility = np.clip(
            structural_compatibility * var_agreement * location_agreement,
            0.0,
            1.0,
        )
        compatibility = np.where(compatibility >= self.compatibility_floor, compatibility, 0.0)

        if self.mode == "compatibility_permutation":
            compatibility = compatibility[_non_identity_permutation(len(compatibility), self.random_state + 31)]
        if self.mode == "target_only":
            compatibility = np.zeros_like(compatibility)

        uncertainty = self.source_prior_strength / (xt.shape[0] + self.source_prior_strength)
        pre_gate_compatibility = compatibility.copy()
        pre_gate_weights = np.clip(uncertainty * pre_gate_compatibility, 0.0, 1.0)
        weights = pre_gate_weights.copy()
        adapted_variance = (
            xt.shape[0] * target_projected
            + self.source_prior_strength * pre_gate_compatibility * source_projected
        ) / np.maximum(xt.shape[0] + self.source_prior_strength * pre_gate_compatibility, 1e-12)
        adapted_variance = np.maximum(adapted_variance, self.min_eigenvalue)

        gate_open, gate_margin = self._safe_transfer_gate(xs, xt, weights, compatibility)
        fallback = False
        fallback_reason = ""
        if self.mode == "target_only":
            gate_open = False
            fallback = True
            fallback_reason = "target_only_mode"
        elif not gate_open:
            weights = np.zeros_like(weights)
            compatibility = np.zeros_like(compatibility)
            adapted_variance = target_projected.copy()
            fallback = True
            fallback_reason = "healthy_loo_gate_closed"

        covariance = _as_symmetric(target_vectors @ np.diag(adapted_variance) @ target_vectors.T)
        precision = _as_symmetric(target_vectors @ np.diag(1.0 / adapted_variance) @ target_vectors.T)
        if not np.isfinite(precision).all():
            raise RuntimeError("S-RACE precision contains non-finite values.")

        cov_diag = _covariance_diagnostics(adapted_variance)
        self.location_ = mu_t
        self.precision_ = precision
        self.covariance_ = covariance
        self.target_covariance_ = cov_t
        self.source_covariance_ = cov_s
        self.target_eigenvectors_ = target_vectors
        self.target_variance_ = target_projected
        self.source_projected_variance_ = source_projected
        self.structural_compatibility_ = structural_compatibility
        self.principal_cos2_ = principal_cos2
        self.variance_compatibility_ = var_agreement
        self.location_compatibility_ = location_agreement
        self.pre_gate_compatibility_ = pre_gate_compatibility
        self.pre_gate_transfer_weights_ = pre_gate_weights
        self.compatibility_ = compatibility
        self.transfer_weights_ = weights
        self.adapted_variance_ = adapted_variance
        active_structural = structural_compatibility[:shared_rank]
        active_structural_summary = _summary(active_structural)
        principal_summary = _summary(principal_cos2)
        pre_gate_weight_summary = _summary(pre_gate_weights)
        pre_gate_compat_summary = _summary(pre_gate_compatibility)
        variance_summary = _summary(var_agreement)
        location_summary = _summary(location_agreement)
        self.diagnostics_ = SRACEDiagnostics(
            mode=self.mode,
            n_source=xs.shape[0],
            n_target=xt.shape[0],
            n_features=xt.shape[1],
            source_prior_strength=self.source_prior_strength,
            safe_gate_open=bool(gate_open),
            safe_gate_margin=float(gate_margin),
            safe_gate_metric="leave_one_out_mean_predictive_log_likelihood",
            fallback=fallback,
            fallback_reason=fallback_reason,
            transferred_dimensions=int(np.sum(weights > 0.0)),
            weight_mean=float(np.mean(weights)),
            weight_median=float(np.median(weights)),
            weight_max=float(np.max(weights)),
            compatibility_mean=float(np.mean(compatibility)),
            compatibility_median=float(np.median(compatibility)),
            compatibility_max=float(np.max(compatibility)),
            structural_compatibility_mean=float(np.mean(structural_compatibility)),
            structural_compatibility_median=float(np.median(structural_compatibility)),
            structural_compatibility_max=float(np.max(structural_compatibility)),
            active_structural_compatibility_mean=active_structural_summary["mean"],
            active_structural_compatibility_median=active_structural_summary["median"],
            active_structural_compatibility_max=active_structural_summary["max"],
            principal_cos2_mean=principal_summary["mean"],
            principal_cos2_median=principal_summary["median"],
            principal_cos2_min=principal_summary["min"],
            principal_cos2_max=principal_summary["max"],
            pre_gate_weight_mean=pre_gate_weight_summary["mean"],
            pre_gate_weight_median=pre_gate_weight_summary["median"],
            pre_gate_weight_max=pre_gate_weight_summary["max"],
            pre_gate_compatibility_mean=pre_gate_compat_summary["mean"],
            pre_gate_compatibility_median=pre_gate_compat_summary["median"],
            pre_gate_compatibility_max=pre_gate_compat_summary["max"],
            variance_compatibility_mean=variance_summary["mean"],
            variance_compatibility_median=variance_summary["median"],
            variance_compatibility_max=variance_summary["max"],
            location_compatibility_mean=location_summary["mean"],
            location_compatibility_median=location_summary["median"],
            location_compatibility_max=location_summary["max"],
            target_uncertainty=float(uncertainty),
            shared_rank=int(shared_rank),
            private_dimensions=int(xt.shape[1] - shared_rank),
            source_shrinkage=float(getattr(source_estimator, "shrinkage_", np.nan)),
            target_shrinkage=float(getattr(target_estimator, "shrinkage_", np.nan)),
            **cov_diag,
        )
        self.is_fitted_ = True
        self.is_calibrated_ = False
        self.threshold_ = None
        return self

    def _safe_transfer_gate(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        weights: np.ndarray,
        compatibility: np.ndarray,
    ) -> tuple[bool, float]:
        if not np.any(weights > 0.0):
            return False, float("-inf")
        if target_features.shape[0] < 4:
            return False, float("-inf")
        target_scores: list[float] = []
        adapted_scores: list[float] = []
        for holdout in range(target_features.shape[0]):
            train = np.delete(target_features, holdout, axis=0)
            x = target_features[holdout : holdout + 1]
            target_model = SelectiveRACEDetector(
                source_prior_strength=self.source_prior_strength,
                compatibility_log_tau=self.compatibility_log_tau,
                median_shift_tau=self.median_shift_tau,
                compatibility_floor=self.compatibility_floor,
                min_eigenvalue=self.min_eigenvalue,
                safe_gate_tolerance=self.safe_gate_tolerance,
                mode="target_only",
                random_state=self.random_state,
                false_alert_budget=self.false_alert_budget,
            )._fit_without_gate(source_features, train, force_target_only=True)
            adapted_model = SelectiveRACEDetector(
                source_prior_strength=self.source_prior_strength,
                compatibility_log_tau=self.compatibility_log_tau,
                median_shift_tau=self.median_shift_tau,
                compatibility_floor=self.compatibility_floor,
                min_eigenvalue=self.min_eigenvalue,
                safe_gate_tolerance=self.safe_gate_tolerance,
                mode=self.mode,
                random_state=self.random_state,
                false_alert_budget=self.false_alert_budget,
            )._fit_without_gate(source_features, train, force_target_only=False)
            target_scores.append(target_model._gaussian_log_likelihood(x)[0])
            adapted_scores.append(adapted_model._gaussian_log_likelihood(x)[0])
        margin = float(np.mean(adapted_scores) - np.mean(target_scores))
        return bool(margin >= -abs(self.safe_gate_tolerance)), margin

    def _fit_without_gate(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        *,
        force_target_only: bool,
    ) -> "SelectiveRACEDetector":
        xs = self._validate_features(source_features)
        xt = self._validate_features(target_features)
        source_estimator = self._fit_lw(xs)
        target_estimator = self._fit_lw(xt)
        mu_t = np.asarray(target_estimator.location_, dtype=np.float64)
        cov_s = _as_symmetric(np.asarray(source_estimator.covariance_, dtype=np.float64))
        cov_t = _as_symmetric(np.asarray(target_estimator.covariance_, dtype=np.float64))
        target_values, target_vectors = _safe_eigh(cov_t, self.min_eigenvalue)
        source_values, source_vectors = _safe_eigh(cov_s, self.min_eigenvalue)
        shared_rank = min(
            _rank_cap(target_values, xt.shape[0]),
            _rank_cap(source_values, xs.shape[0]),
            target_vectors.shape[1],
        )
        source_projected = np.maximum(
            np.einsum("ij,jk,ik->i", target_vectors.T, cov_s, target_vectors.T, optimize=True),
            self.min_eigenvalue,
        )
        target_projected = np.maximum(target_values, self.min_eigenvalue)
        if force_target_only:
            adapted_variance = target_projected
        else:
            zt = (xt - mu_t) @ target_vectors
            zs = (xs - np.asarray(source_estimator.location_, dtype=np.float64)) @ target_vectors
            var_agreement = np.exp(
                -np.abs(np.log(source_projected) - np.log(target_projected)) / self.compatibility_log_tau
            )
            median_shift = np.abs(np.median(zs, axis=0) - np.median(zt, axis=0)) / np.sqrt(target_projected)
            structural_compatibility = np.sum(
                (target_vectors.T @ source_vectors[:, :shared_rank]) ** 2,
                axis=1,
            )
            structural_compatibility = np.clip(structural_compatibility, 0.0, 1.0)
            structural_compatibility[shared_rank:] = 0.0
            compatibility = np.clip(
                structural_compatibility * var_agreement * np.exp(-median_shift / self.median_shift_tau),
                0.0,
                1.0,
            )
            compatibility = np.where(compatibility >= self.compatibility_floor, compatibility, 0.0)
            if self.mode == "compatibility_permutation":
                compatibility = compatibility[
                    _non_identity_permutation(len(compatibility), self.random_state + 31)
                ]
            adapted_variance = (
                xt.shape[0] * target_projected
                + self.source_prior_strength * compatibility * source_projected
            ) / np.maximum(xt.shape[0] + self.source_prior_strength * compatibility, 1e-12)
        adapted_variance = np.maximum(adapted_variance, self.min_eigenvalue)
        self.location_ = mu_t
        self.precision_ = _as_symmetric(target_vectors @ np.diag(1.0 / adapted_variance) @ target_vectors.T)
        self.covariance_ = _as_symmetric(target_vectors @ np.diag(adapted_variance) @ target_vectors.T)
        self.is_fitted_ = True
        return self

    def _gaussian_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        if self.location_ is None or self.precision_ is None or self.covariance_ is None:
            raise RuntimeError("Detector must be fitted before likelihood scoring.")
        x = self._validate_features(features)
        centered = x - self.location_
        mahal = np.einsum("ij,jk,ik->i", centered, self.precision_, centered, optimize=True)
        sign, logdet = np.linalg.slogdet(self.covariance_)
        if sign <= 0:
            return np.full(x.shape[0], -np.inf)
        return -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + logdet + mahal)

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted_ or self.location_ is None or self.precision_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        x = self._validate_features(features)
        centered = x - self.location_
        squared = np.einsum("ij,jk,ik->i", centered, self.precision_, centered, optimize=True)
        return self._validate_scores(np.sqrt(np.maximum(squared, 0.0)))

    def get_params(self) -> dict[str, object]:
        params = super().get_params()
        params.update(
            {
                "source_prior_strength": self.source_prior_strength,
                "compatibility_log_tau": self.compatibility_log_tau,
                "median_shift_tau": self.median_shift_tau,
                "compatibility_floor": self.compatibility_floor,
                "min_eigenvalue": self.min_eigenvalue,
                "safe_gate_tolerance": self.safe_gate_tolerance,
                "mode": self.mode,
                "random_state": self.random_state,
            }
        )
        return params
