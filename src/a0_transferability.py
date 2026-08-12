"""Healthy-only source-target transferability diagnostics for RACE-A0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransferabilityAudit:
    dataset: str
    source_domain: str
    target_domain: str
    n_target: int
    seed: int
    k: int
    alignment_mean_cos2: float
    projector_similarity: float
    bootstrap_alignment_mean: float
    bootstrap_alignment_std: float
    variance_agreement_mean: float
    scale_ratio_median: float
    standardized_mean_distance: float
    covariance_frobenius: float
    mean_shift_distance: float = 0.0
    covariance_discrepancy: float = 0.0
    projector_discrepancy: float = 0.0
    mmd_rbf: float = 0.0
    wasserstein_diag: float = 0.0


def robust_center_scale(x: np.ndarray, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    scale = 1.4826 * mad
    finite = scale[np.isfinite(scale) & (scale > floor)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, max(fallback, floor))
    return center, scale


def pca_basis(x: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    if k <= 0:
        return np.zeros((values.shape[1], 0), dtype=np.float64)
    _, _, vh = np.linalg.svd(values - np.mean(values, axis=0), full_matrices=False)
    return vh[:k].T


def principal_angle_cos2(source_basis: np.ndarray, target_basis: np.ndarray) -> np.ndarray:
    if source_basis.shape != target_basis.shape:
        raise ValueError("source and target bases must have the same shape.")
    if source_basis.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    singular = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    return np.clip(singular, 0.0, 1.0) ** 2


def projector_similarity(source_basis: np.ndarray, target_basis: np.ndarray) -> float:
    k = source_basis.shape[1]
    if k == 0:
        return 0.0
    source_projector = source_basis @ source_basis.T
    target_projector = target_basis @ target_basis.T
    return float(np.trace(source_projector @ target_projector) / k)


def bootstrap_target_stability(
    target_common: np.ndarray,
    target_basis: np.ndarray,
    *,
    resamples: int = 50,
    random_state: int = 42,
) -> tuple[float, float]:
    k = target_basis.shape[1]
    if k == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(random_state)
    similarities: list[float] = []
    for _ in range(resamples):
        idx = rng.integers(0, target_common.shape[0], size=target_common.shape[0])
        basis = pca_basis(target_common[idx], k)
        similarities.append(projector_similarity(basis, target_basis))
    values = np.asarray(similarities, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values))


def common_metric_discrepancy(source_common: np.ndarray, target_common: np.ndarray) -> tuple[float, float]:
    mean_delta = np.mean(source_common, axis=0) - np.mean(target_common, axis=0)
    standardized_mean_distance = float(np.linalg.norm(mean_delta) / np.sqrt(source_common.shape[1]))
    cov_source = np.cov(source_common, rowvar=False)
    cov_target = np.cov(target_common, rowvar=False)
    covariance_frobenius = float(
        np.linalg.norm(cov_source - cov_target, ord="fro") / max(1.0, source_common.shape[1])
    )
    return standardized_mean_distance, covariance_frobenius


def _median_pairwise_sq_distance(x: np.ndarray, max_points: int = 300) -> float:
    values = np.asarray(x, dtype=np.float64)
    if values.shape[0] > max_points:
        idx = np.linspace(0, values.shape[0] - 1, max_points).round().astype(int)
        values = values[idx]
    diffs = values[:, None, :] - values[None, :, :]
    sq = np.sum(diffs * diffs, axis=2)
    upper = sq[np.triu_indices_from(sq, k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0.0)]
    return float(np.median(finite)) if finite.size else 1.0


def rbf_mmd(source_common: np.ndarray, target_common: np.ndarray) -> float:
    """Biased RBF MMD using a deterministic median-distance bandwidth."""
    source = np.asarray(source_common, dtype=np.float64)
    target = np.asarray(target_common, dtype=np.float64)
    pooled = np.vstack([source, target])
    gamma = 1.0 / max(2.0 * _median_pairwise_sq_distance(pooled), 1e-12)

    def kernel_mean(a: np.ndarray, b: np.ndarray) -> float:
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return float(np.mean(np.exp(-gamma * sq)))

    value = kernel_mean(source, source) + kernel_mean(target, target) - 2.0 * kernel_mean(source, target)
    return float(max(value, 0.0))


def diagonal_wasserstein(source_common: np.ndarray, target_common: np.ndarray) -> float:
    """Diagonal Gaussian 2-Wasserstein approximation in the common robust metric."""
    source = np.asarray(source_common, dtype=np.float64)
    target = np.asarray(target_common, dtype=np.float64)
    mean_term = np.sum((np.mean(source, axis=0) - np.mean(target, axis=0)) ** 2)
    std_s = np.sqrt(np.maximum(np.var(source, axis=0), 0.0))
    std_t = np.sqrt(np.maximum(np.var(target, axis=0), 0.0))
    scale_term = np.sum((std_s - std_t) ** 2)
    return float(np.sqrt(max(mean_term + scale_term, 0.0)))


def healthy_transferability_metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k_max: int = 16,
) -> dict[str, float]:
    """Compute healthy-only source-target geometry metrics.

    The target commissioning sample defines the robust metric. Source location is
    centered independently, matching the existing A0/A1 source transform.
    """
    source_raw = np.asarray(source, dtype=np.float64)
    target_raw = np.asarray(target, dtype=np.float64)
    center_t, scale_t = robust_center_scale(target_raw)
    center_s, _ = robust_center_scale(source_raw)
    target_common = np.clip((target_raw - center_t) / scale_t, -8.0, 8.0)
    source_common = np.clip((source_raw - center_s) / scale_t, -8.0, 8.0)
    mean_delta_raw = np.mean(source_raw, axis=0) - np.mean(target_raw, axis=0)
    mean_delta_common = np.mean(source_common, axis=0) - np.mean(target_common, axis=0)
    cov_source = np.cov(source_common, rowvar=False)
    cov_target = np.cov(target_common, rowvar=False)
    k = min(k_max, source_common.shape[0] - 1, target_common.shape[0] - 1, source_common.shape[1])
    if k > 0:
        source_basis = pca_basis(source_common, k)
        target_basis = pca_basis(target_common, k)
        projector_gap = np.linalg.norm(
            source_basis @ source_basis.T - target_basis @ target_basis.T,
            ord="fro",
        ) / np.sqrt(2.0 * k)
        projector_sim = projector_similarity(source_basis, target_basis)
    else:
        projector_gap = 0.0
        projector_sim = 0.0
    return {
        "mean_shift_distance": float(np.linalg.norm(mean_delta_raw)),
        "standardized_mean_shift": float(np.linalg.norm(mean_delta_common) / np.sqrt(source_common.shape[1])),
        "covariance_discrepancy": float(np.linalg.norm(cov_source - cov_target, ord="fro") / max(1.0, source_common.shape[1])),
        "projector_discrepancy": float(projector_gap),
        "projector_similarity": float(projector_sim),
        "mmd_rbf": rbf_mmd(source_common, target_common),
        "wasserstein_diag": diagonal_wasserstein(source_common, target_common),
    }


def audit_pair(
    source: np.ndarray,
    target: np.ndarray,
    *,
    dataset: str,
    source_domain: str,
    target_domain: str,
    n_target: int,
    seed: int,
    k_max: int = 16,
    bootstrap_resamples: int = 50,
) -> tuple[TransferabilityAudit, np.ndarray]:
    source_raw = np.asarray(source, dtype=np.float64)
    target_raw = np.asarray(target, dtype=np.float64)
    k = min(k_max, source_raw.shape[0] - 1, target_raw.shape[0] - 1, source_raw.shape[1])
    center_t, scale_t = robust_center_scale(target_raw)
    center_s, scale_s_native = robust_center_scale(source_raw)
    target_common = np.clip((target_raw - center_t) / scale_t, -8.0, 8.0)
    source_common = np.clip((source_raw - center_s) / scale_t, -8.0, 8.0)

    target_basis = pca_basis(target_common, k)
    source_basis = pca_basis(source_common, k)
    cos2 = principal_angle_cos2(source_basis, target_basis)
    target_scores = target_common @ target_basis
    source_scores = source_common @ source_basis
    var_t = np.var(target_scores, axis=0, ddof=1)
    var_s = np.var(source_scores, axis=0, ddof=1)
    variance_ratio = np.minimum(var_s, var_t) / np.maximum(var_s, var_t)
    stability_mean, stability_std = bootstrap_target_stability(
        target_common,
        target_basis,
        resamples=bootstrap_resamples,
        random_state=seed,
    )
    mean_distance, cov_frob = common_metric_discrepancy(source_common, target_common)
    extra = healthy_transferability_metrics(source_raw, target_raw, k_max=k_max)
    scale_ratio = scale_s_native / scale_t

    audit = TransferabilityAudit(
        dataset=dataset,
        source_domain=source_domain,
        target_domain=target_domain,
        n_target=n_target,
        seed=seed,
        k=k,
        alignment_mean_cos2=float(np.mean(cos2)) if cos2.size else 0.0,
        projector_similarity=projector_similarity(source_basis, target_basis),
        bootstrap_alignment_mean=stability_mean,
        bootstrap_alignment_std=stability_std,
        variance_agreement_mean=float(np.mean(variance_ratio)) if variance_ratio.size else 0.0,
        scale_ratio_median=float(np.median(scale_ratio)),
        standardized_mean_distance=mean_distance,
        covariance_frobenius=cov_frob,
        mean_shift_distance=extra["mean_shift_distance"],
        covariance_discrepancy=extra["covariance_discrepancy"],
        projector_discrepancy=extra["projector_discrepancy"],
        mmd_rbf=extra["mmd_rbf"],
        wasserstein_diag=extra["wasserstein_diag"],
    )
    return audit, cos2
