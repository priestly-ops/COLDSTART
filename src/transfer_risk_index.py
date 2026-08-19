from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


EPS = 1e-12


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    shrinkage: float
    effective_rank: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float


@dataclass(frozen=True)
class TransferRiskResult:
    target_cov_uncertainty: float
    target_cov_uncertainty_normalized: float
    source_target_cov_discrepancy: float
    source_target_cov_discrepancy_normalized: float
    tri: float
    target: CovarianceEstimate
    source: CovarianceEstimate
    log_euclidean_distance: float
    bures_wasserstein_distance: float
    coral_distance: float
    subspace_principal_angle_distance: float
    top_1pct_direction_discrepancy_share: float
    top_5pct_direction_discrepancy_share: float
    top_10pct_direction_discrepancy_share: float


def validate_feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("Expected a 2D feature matrix with at least two rows.")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or Inf.")
    return matrix


def ledoit_wolf_covariance(features: np.ndarray) -> CovarianceEstimate:
    matrix = validate_feature_matrix(features)
    estimator = LedoitWolf(assume_centered=False, store_precision=False)
    estimator.fit(matrix)
    covariance = stabilize_covariance(estimator.covariance_)
    values = np.linalg.eigvalsh(covariance)
    total = float(np.sum(values))
    effective_rank = float((total * total) / max(float(np.sum(values * values)), EPS))
    min_eig = float(np.min(values))
    max_eig = float(np.max(values))
    return CovarianceEstimate(
        covariance=covariance,
        shrinkage=float(getattr(estimator, "shrinkage_", np.nan)),
        effective_rank=effective_rank,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        condition_number=float(max_eig / max(min_eig, EPS)),
    )


def stabilize_covariance(covariance: np.ndarray, *, minimum_eigenvalue: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square covariance matrix.")
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, minimum_eigenvalue)
    out = vectors @ np.diag(values) @ vectors.T
    return 0.5 * (out + out.T)


def covariance_bootstrap_uncertainty(
    target_features: np.ndarray,
    *,
    resamples: int = 200,
    rng_seed: int = 42,
) -> tuple[float, float, CovarianceEstimate, pd.DataFrame]:
    target = validate_feature_matrix(target_features)
    if resamples < 2:
        raise ValueError("resamples must be at least 2.")
    reference = ledoit_wolf_covariance(target)
    denom = float(np.linalg.norm(reference.covariance, ord="fro") ** 2 + EPS)
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, float | int]] = []
    distances: list[float] = []
    for b in range(resamples):
        idx = rng.integers(0, target.shape[0], size=target.shape[0])
        estimate = ledoit_wolf_covariance(target[idx])
        dist = frobenius_squared(estimate.covariance, reference.covariance)
        distances.append(dist)
        rows.append(
            {
                "bootstrap_index": b,
                "bootstrap_frobenius_squared": dist,
                "bootstrap_frobenius_squared_normalized": float(dist / denom),
                "bootstrap_shrinkage": estimate.shrinkage,
                "bootstrap_effective_rank": estimate.effective_rank,
                "bootstrap_condition_number": estimate.condition_number,
            }
        )
    uncertainty = float(np.mean(distances))
    return uncertainty, float(uncertainty / denom), reference, pd.DataFrame(rows)


def covariance_jackknife_uncertainty(target_features: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    target = validate_feature_matrix(target_features)
    reference = ledoit_wolf_covariance(target)
    denom = float(np.linalg.norm(reference.covariance, ord="fro") ** 2 + EPS)
    rows: list[dict[str, float | int]] = []
    distances: list[float] = []
    for i in range(target.shape[0]):
        estimate = ledoit_wolf_covariance(np.delete(target, i, axis=0))
        dist = frobenius_squared(estimate.covariance, reference.covariance)
        distances.append(dist)
        rows.append(
            {
                "jackknife_index": i,
                "jackknife_frobenius_squared": dist,
                "jackknife_frobenius_squared_normalized": float(dist / denom),
                "jackknife_shrinkage": estimate.shrinkage,
                "jackknife_effective_rank": estimate.effective_rank,
                "jackknife_condition_number": estimate.condition_number,
            }
        )
    uncertainty = float(np.mean(distances))
    return uncertainty, float(uncertainty / denom), pd.DataFrame(rows)


def diagonal_gaussian_uncertainty_proxy(target_features: np.ndarray) -> tuple[float, float]:
    """Narrow diagonal normal-theory covariance uncertainty proxy.

    This is not the primary TRI estimator and is not a full shrinkage-risk
    estimate. It is a lower-dimensional sensitivity check for d >> N regimes.
    """
    target = validate_feature_matrix(target_features)
    variances = np.var(target, axis=0, ddof=1)
    diagonal_variance_sum = float(np.sum(2.0 * variances * variances / max(target.shape[0] - 1, 1)))
    reference = ledoit_wolf_covariance(target)
    denom = float(np.linalg.norm(reference.covariance, ord="fro") ** 2 + EPS)
    return diagonal_variance_sum, float(diagonal_variance_sum / denom)


def bootstrap_covariance_estimates(
    target_features: np.ndarray,
    *,
    resamples: int = 200,
    rng_seed: int = 42,
) -> list[np.ndarray]:
    target = validate_feature_matrix(target_features)
    rng = np.random.default_rng(rng_seed)
    covariances: list[np.ndarray] = []
    for _ in range(resamples):
        idx = rng.integers(0, target.shape[0], size=target.shape[0])
        covariances.append(ledoit_wolf_covariance(target[idx]).covariance)
    return covariances


def compute_transfer_risk_index(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    resamples: int = 200,
    rng_seed: int = 42,
) -> tuple[TransferRiskResult, pd.DataFrame]:
    source = ledoit_wolf_covariance(source_features)
    uncertainty, uncertainty_norm, target, bootstrap = covariance_bootstrap_uncertainty(
        target_features,
        resamples=resamples,
        rng_seed=rng_seed,
    )
    result = compute_transfer_risk_index_from_estimates(
        source,
        target,
        target_cov_uncertainty=uncertainty,
        target_cov_uncertainty_normalized=uncertainty_norm,
    )
    return result, bootstrap


def compute_transfer_risk_index_from_estimates(
    source: CovarianceEstimate,
    target: CovarianceEstimate,
    *,
    target_cov_uncertainty: float,
    target_cov_uncertainty_normalized: float,
) -> TransferRiskResult:
    discrepancy = frobenius_squared(source.covariance, target.covariance)
    denom = float(np.linalg.norm(target.covariance, ord="fro") ** 2 + EPS)
    discrepancy_norm = float(discrepancy / denom)
    tri = float(discrepancy_norm / max(target_cov_uncertainty_normalized, EPS))
    return TransferRiskResult(
        target_cov_uncertainty=float(target_cov_uncertainty),
        target_cov_uncertainty_normalized=float(target_cov_uncertainty_normalized),
        source_target_cov_discrepancy=discrepancy,
        source_target_cov_discrepancy_normalized=discrepancy_norm,
        tri=tri,
        target=target,
        source=source,
        log_euclidean_distance=log_euclidean_distance(source.covariance, target.covariance),
        bures_wasserstein_distance=bures_wasserstein_distance(source.covariance, target.covariance),
        coral_distance=coral_distance(source.covariance, target.covariance, n_features=target.covariance.shape[0]),
        subspace_principal_angle_distance=subspace_principal_angle_distance(source.covariance, target.covariance),
        top_1pct_direction_discrepancy_share=directional_discrepancy_share(source.covariance, target.covariance, top_fraction=0.01),
        top_5pct_direction_discrepancy_share=directional_discrepancy_share(source.covariance, target.covariance, top_fraction=0.05),
        top_10pct_direction_discrepancy_share=directional_discrepancy_share(source.covariance, target.covariance, top_fraction=0.10),
    )


def frobenius_squared(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(delta, ord="fro") ** 2)


def log_euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(_matrix_log(left) - _matrix_log(right), ord="fro"))


def bures_wasserstein_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = stabilize_covariance(left)
    b = stabilize_covariance(right)
    sqrt_a = _matrix_sqrt(a)
    middle = _matrix_sqrt(sqrt_a @ b @ sqrt_a)
    value = float(np.trace(a) + np.trace(b) - 2.0 * np.trace(middle))
    return float(np.sqrt(max(value, 0.0)))


def coral_distance(left: np.ndarray, right: np.ndarray, *, n_features: int) -> float:
    return float(frobenius_squared(left, right) / (4.0 * n_features * n_features))


def subspace_principal_angle_distance(left: np.ndarray, right: np.ndarray, *, energy: float = 0.9) -> float:
    _, left_vectors = _top_energy_subspace(left, energy=energy)
    _, right_vectors = _top_energy_subspace(right, energy=energy)
    k = min(left_vectors.shape[1], right_vectors.shape[1])
    singular = np.linalg.svd(left_vectors[:, :k].T @ right_vectors[:, :k], compute_uv=False)
    angles = np.arccos(np.clip(singular, -1.0, 1.0))
    return float(np.linalg.norm(angles))


def directional_discrepancy_share(left: np.ndarray, right: np.ndarray, *, top_fraction: float) -> float:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    delta = stabilize_covariance(left) - stabilize_covariance(right)
    _, target_vectors = np.linalg.eigh(stabilize_covariance(right))
    projected = np.einsum("ij,jk,ik->i", target_vectors.T, delta, target_vectors.T, optimize=True)
    contributions = np.sort(projected * projected)[::-1]
    total = float(np.sum(contributions))
    if total <= EPS:
        return 0.0
    k = max(1, int(np.ceil(top_fraction * contributions.size)))
    return float(np.sum(contributions[:k]) / total)


def optimal_bootstrap_blend_weight(
    source_covariance: np.ndarray,
    target_covariance: np.ndarray,
    bootstrap_covariances: list[np.ndarray],
    *,
    grid_size: int = 101,
) -> tuple[float, float]:
    if not bootstrap_covariances:
        return float("nan"), float("nan")
    weights = np.linspace(0.0, 1.0, grid_size)
    risks = []
    for w in weights:
        blended = (1.0 - w) * target_covariance + w * source_covariance
        risk = float(np.mean([frobenius_squared(blended, cov) for cov in bootstrap_covariances]))
        risks.append(risk)
    idx = int(np.argmin(risks))
    return float(weights[idx]), float(risks[idx])


def synthetic_sanity_checks(*, resamples: int = 40, rng_seed: int = 20260812) -> pd.DataFrame:
    """Pre-robotics checks for proxy behavior, not anomaly-transfer claims."""
    rng = np.random.default_rng(rng_seed)
    base_covariance = np.diag([1.0, 1.4, 1.8, 2.2, 2.6, 3.0])
    mismatch_covariance = np.diag([1.0, 3.0, 6.0, 9.0, 12.0, 15.0])

    case_a_target = rng.multivariate_normal(np.zeros(6), base_covariance, size=10)
    case_b_target = rng.multivariate_normal(np.zeros(6), base_covariance, size=200)
    case_d_small_target = rng.multivariate_normal(np.zeros(6), base_covariance, size=10)
    case_d_large_target = rng.multivariate_normal(np.zeros(6), base_covariance, size=200)
    case_d_source = rng.multivariate_normal(np.zeros(6), mismatch_covariance, size=400)

    case_a, _ = compute_transfer_risk_index(case_a_target, case_a_target, resamples=resamples, rng_seed=rng_seed + 1)
    case_b, _ = compute_transfer_risk_index(case_b_target, case_b_target, resamples=resamples, rng_seed=rng_seed + 2)
    case_d_small, _ = compute_transfer_risk_index(
        case_d_source,
        case_d_small_target,
        resamples=resamples,
        rng_seed=rng_seed + 4,
    )
    case_d_large, _ = compute_transfer_risk_index(
        case_d_source,
        case_d_large_target,
        resamples=resamples,
        rng_seed=rng_seed + 5,
    )

    rows = [
        _sanity_row("A_same_covariance_small_N", case_a),
        _sanity_row("B_same_covariance_large_N", case_b),
        _sanity_row("D_mismatch_small_N", case_d_small),
        _sanity_row("D_mismatch_large_N", case_d_large),
    ]
    case_c_target = rng.multivariate_normal(np.zeros(6), base_covariance, size=40)
    case_c_results: list[TransferRiskResult] = []
    for scale in [0.0, 0.25, 0.50, 1.0, 2.0]:
        perturbation = np.diag([0.0, 0.4, 0.8, 1.2, 1.6, 2.0]) * scale
        source = rng.multivariate_normal(np.zeros(6), base_covariance + perturbation, size=800)
        result, _ = compute_transfer_risk_index(
            source,
            case_c_target,
            resamples=resamples,
            rng_seed=rng_seed + 100 + int(scale * 100),
        )
        row = _sanity_row(f"C_covariance_perturbation_scale_{scale:g}", result)
        row["perturbation_scale"] = scale
        rows.append(row)
        case_c_results.append(result)

    out = pd.DataFrame(rows)
    case_c_discrepancies = [
        result.source_target_cov_discrepancy_normalized for result in case_c_results
    ]
    expectations = {
        "A_discrepancy_near_zero": case_a.source_target_cov_discrepancy_normalized < 0.05,
        "B_uncertainty_less_than_A": case_b.target_cov_uncertainty_normalized
        < case_a.target_cov_uncertainty_normalized,
        "C_discrepancy_monotone_non_decreasing": bool(np.all(np.diff(case_c_discrepancies) >= -1e-10)),
        "D_large_N_uncertainty_less_than_small_N": case_d_large.target_cov_uncertainty_normalized
        < case_d_small.target_cov_uncertainty_normalized,
    }
    out["all_expectations_pass"] = bool(all(expectations.values()))
    for name, value in expectations.items():
        out[name] = bool(value)
    return out


def _sanity_row(case: str, result: TransferRiskResult) -> dict[str, float | str]:
    return {
        "case": case,
        "target_cov_uncertainty_normalized": result.target_cov_uncertainty_normalized,
        "source_target_cov_discrepancy_normalized": result.source_target_cov_discrepancy_normalized,
        "TRI": result.tri,
        "target_cov_effective_rank": result.target.effective_rank,
        "source_cov_effective_rank": result.source.effective_rank,
    }


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(stabilize_covariance(matrix))
    return vectors @ np.diag(np.sqrt(np.maximum(values, EPS))) @ vectors.T


def _matrix_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(stabilize_covariance(matrix))
    return vectors @ np.diag(np.log(np.maximum(values, EPS))) @ vectors.T


def _top_energy_subspace(matrix: np.ndarray, *, energy: float) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(stabilize_covariance(matrix))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    cumulative = np.cumsum(values) / max(float(np.sum(values)), EPS)
    k = int(np.searchsorted(cumulative, energy, side="left") + 1)
    k = min(max(k, 1), vectors.shape[1])
    return values[:k], vectors[:, :k]
