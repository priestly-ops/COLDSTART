"""Healthy-only precision-transfer estimators for COLDSTART P0.2.

This module implements a reviewer-facing Python reference for:

- CLIME via column-wise linear programs;
- a reference-style Trans-CLIME update following Li, Cai & Li;
- deterministic SPD projection diagnostics for downstream quadratic geometry;
- cross-fitted positive-transfer aggregation as an explicitly labeled extension;
- leakage-safe scaling helpers whose target statistics are fit only on the
  training fold supplied by the caller.

Important
---------
The published Trans-CLIME reference implementation is in R/fastclime.  The
functions below preserve the same constrained optimization structure but are a
clean-room Python implementation using SciPy HiGHS.  The cross-fitted variant is
*not* the published method and must be labeled as a COLDSTART extension.

No anomaly labels, anomaly scores, or deployment thresholds are used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import linprog


EPS = 1e-12


@dataclass(frozen=True)
class LPDiagnostics:
    feasible_columns: int
    total_columns: int
    success_fraction: float
    max_constraint_violation: float


@dataclass(frozen=True)
class SPDProjectionResult:
    matrix: np.ndarray
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    relative_frobenius_change: float


@dataclass(frozen=True)
class PrecisionEstimate:
    raw: np.ndarray
    symmetric: np.ndarray
    spd: np.ndarray
    lp_diagnostics: LPDiagnostics
    spd_projection: SPDProjectionResult
    metadata: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray
    source_weight: float
    mode: str

    def transform(self, x: np.ndarray, clip: float | None = 12.0) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.center) / self.scale
        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))
        return z


def _as_matrix(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got {arr.shape}.")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} has invalid shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def empirical_covariance(x: np.ndarray) -> np.ndarray:
    x = _as_matrix(x, "x")
    return np.atleast_2d(np.cov(x, rowvar=False, ddof=1)).astype(np.float64)


def symmetrize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a + a.T)


def spd_project(
    matrix: np.ndarray,
    *,
    eigen_floor: float = 1e-6,
) -> SPDProjectionResult:
    """Project a symmetric matrix to the SPD cone by eigenvalue flooring.

    This projection is deterministic and intentionally simple.  Its magnitude
    is always reported because a large correction is evidence that the raw
    precision estimate is unsuitable for Mahalanobis-style deployment geometry.
    """
    if eigen_floor <= 0:
        raise ValueError("eigen_floor must be positive.")
    a = symmetrize(np.asarray(matrix, dtype=np.float64))
    values, vectors = np.linalg.eigh(a)
    min_before = float(np.min(values))
    clipped = np.maximum(values, float(eigen_floor))
    projected = (vectors * clipped) @ vectors.T
    projected = symmetrize(projected)
    denom = max(float(np.linalg.norm(a, ord="fro")), EPS)
    relative = float(np.linalg.norm(projected - a, ord="fro") / denom)
    return SPDProjectionResult(
        matrix=projected,
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=float(np.min(np.linalg.eigvalsh(projected))),
        relative_frobenius_change=relative,
    )


def _cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = symmetrize(np.asarray(cov, dtype=np.float64))
    diag = np.diag(cov)
    if np.any(diag <= EPS):
        raise ValueError("Covariance contains zero/tiny diagonal entries.")
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    corr = symmetrize(corr)
    return corr, scale


def _solve_clime_columns(
    sigma: np.ndarray,
    bmat: np.ndarray,
    *,
    lam: float,
) -> tuple[np.ndarray, LPDiagnostics]:
    """Solve min ||theta_j||_1 s.t. ||Sigma theta_j - b_j||_inf <= lam.

    Uses the standard split-variable linear-program form theta = u-v with
    u,v >= 0 and SciPy's HiGHS backend.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    bmat = np.asarray(bmat, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be square.")
    p = sigma.shape[0]
    if bmat.shape != (p, p):
        raise ValueError(f"bmat must have shape {(p, p)}, got {bmat.shape}.")
    if lam <= 0:
        raise ValueError("lam must be positive.")

    objective = np.ones(2 * p, dtype=np.float64)
    # Sigma(u-v)-b <= lam and -(Sigma(u-v)-b) <= lam
    a_ub = np.vstack(
        (
            np.hstack((sigma, -sigma)),
            np.hstack((-sigma, sigma)),
        )
    )
    bounds = [(0.0, None)] * (2 * p)
    theta = np.zeros((p, p), dtype=np.float64)
    feasible = 0
    max_violation = 0.0

    for j in range(p):
        bj = bmat[:, j]
        b_ub = np.concatenate((bj + lam, -bj + lam))
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        if not result.success or result.x is None:
            continue
        col = result.x[:p] - result.x[p:]
        violation = float(np.max(np.abs(sigma @ col - bj)) - lam)
        max_violation = max(max_violation, max(0.0, violation))
        theta[:, j] = col
        feasible += 1

    return theta, LPDiagnostics(
        feasible_columns=int(feasible),
        total_columns=int(p),
        success_fraction=float(feasible / p),
        max_constraint_violation=float(max_violation),
    )


def clime_from_covariance(
    covariance: np.ndarray,
    *,
    lam: float,
    bmat: np.ndarray | None = None,
    correlation_scale: bool = True,
    reference_column_rescale: bool = True,
    eigen_floor: float = 1e-6,
) -> PrecisionEstimate:
    """CLIME-style constrained inverse estimator from a covariance matrix.

    ``correlation_scale=True`` mirrors the scale-normalization used by the
    Trans-CLIME authors' public R helper.  ``reference_column_rescale`` applies
    their post-LP column normalization when the corresponding denominator is
    numerically safe.
    """
    cov = symmetrize(np.asarray(covariance, dtype=np.float64))
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square.")
    p = cov.shape[0]
    rhs = np.eye(p, dtype=np.float64) if bmat is None else np.asarray(bmat, dtype=np.float64)

    if correlation_scale:
        work, std = _cov_to_corr(cov)
    else:
        work = cov.copy()
        std = np.ones(p, dtype=np.float64)

    raw, diag = _solve_clime_columns(work, rhs, lam=float(lam))

    if correlation_scale:
        raw = raw / np.outer(std, std)

    if reference_column_rescale:
        for j in range(p):
            denom = float(cov[j, :] @ raw[:, j])
            target = float(rhs[j, j])
            if abs(denom) > EPS and abs(target) > EPS:
                raw[:, j] *= target / denom

    symmetric = symmetrize(raw)
    projection = spd_project(symmetric, eigen_floor=eigen_floor)
    return PrecisionEstimate(
        raw=raw,
        symmetric=symmetric,
        spd=projection.matrix,
        lp_diagnostics=diag,
        spd_projection=projection,
        metadata={
            "estimator": "CLIME",
            "lambda": float(lam),
            "correlation_scale": bool(correlation_scale),
            "reference_column_rescale": bool(reference_column_rescale),
        },
    )


def clime(
    x: np.ndarray,
    *,
    lam: float,
    eigen_floor: float = 1e-6,
) -> PrecisionEstimate:
    return clime_from_covariance(
        empirical_covariance(x),
        lam=lam,
        eigen_floor=eigen_floor,
    )


def fit_robust_scaler(
    target_train: np.ndarray,
    *,
    source_train: np.ndarray | None = None,
    mode: Literal["target", "source", "shrink"] = "target",
    lambda_reg: float = 60.0,
) -> RobustScaler:
    """Fit leakage-safe robust scaling on supplied training rows only.

    ``source`` freezes source median/IQR statistics.
    ``target`` uses target-training statistics only.
    ``shrink`` uses the original RACE-style N/(N+lambda) interpolation for
    centers and log-scales.  No validation rows are accepted by this API.
    """
    target = _as_matrix(target_train, "target_train")

    def robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.median(x, axis=0)
        q25 = np.quantile(x, 0.25, axis=0)
        q75 = np.quantile(x, 0.75, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        std = np.std(x, axis=0, ddof=1 if len(x) > 1 else 0)
        scale = np.where(iqr_scale > EPS, iqr_scale, std)
        scale = np.where(scale > EPS, scale, 1.0)
        return center.astype(np.float64), scale.astype(np.float64)

    target_center, target_scale = robust_stats(target)
    if mode == "target":
        return RobustScaler(target_center, target_scale, 0.0, mode)

    if source_train is None:
        raise ValueError(f"source_train is required for mode={mode!r}.")
    source = _as_matrix(source_train, "source_train")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_train and target_train must have same feature width.")
    source_center, source_scale = robust_stats(source)

    if mode == "source":
        return RobustScaler(source_center, source_scale, 1.0, mode)
    if mode != "shrink":
        raise ValueError(f"Unknown scaler mode {mode!r}.")

    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive.")
    target_weight = float(len(target) / (len(target) + float(lambda_reg)))
    source_weight = 1.0 - target_weight
    center = source_weight * source_center + target_weight * target_center
    # Geometric interpolation avoids negative scales and behaves naturally for
    # multiplicative scale differences.
    log_scale = source_weight * np.log(np.maximum(source_scale, EPS)) + target_weight * np.log(
        np.maximum(target_scale, EPS)
    )
    return RobustScaler(center, np.exp(log_scale), source_weight, mode)


def _trans_clime_core(
    target_fit: np.ndarray,
    source_fit: np.ndarray,
    *,
    target_clime: PrecisionEstimate,
    delta_lambda: float | None,
    transfer_lambda_const: float,
    eigen_floor: float,
) -> tuple[PrecisionEstimate, dict[str, float | int]]:
    target_fit = _as_matrix(target_fit, "target_fit")
    source_fit = _as_matrix(source_fit, "source_fit")
    if target_fit.shape[1] != source_fit.shape[1]:
        raise ValueError("target/source feature widths differ.")
    p = target_fit.shape[1]
    n0 = target_fit.shape[0]
    n_source = source_fit.shape[0]
    sigma0 = empirical_covariance(target_fit)
    sigma_a = empirical_covariance(source_fit)

    omega_l1 = float(np.mean(np.sum(np.abs(target_clime.raw), axis=0)))
    if delta_lambda is None:
        delta_lambda = omega_l1 * np.sqrt(np.log(max(p, 2)) / n0)
    divergence_rhs = np.eye(p) - target_clime.raw.T @ sigma_a
    delta, delta_diag = _solve_clime_columns(
        np.eye(p),
        divergence_rhs,
        lam=float(delta_lambda),
    )

    transfer_rhs = np.eye(p) - delta.T
    transfer_lambda = float(2.0 * transfer_lambda_const * np.sqrt(np.log(max(p, 2)) / n_source))
    transferred = clime_from_covariance(
        sigma_a,
        lam=transfer_lambda,
        bmat=transfer_rhs,
        correlation_scale=True,
        reference_column_rescale=True,
        eigen_floor=eigen_floor,
    )
    metadata = {
        "delta_lambda": float(delta_lambda),
        "transfer_lambda": transfer_lambda,
        "delta_lp_success_fraction": float(delta_diag.success_fraction),
        "source_samples": int(n_source),
        "target_fit_samples": int(n0),
    }
    return transferred, metadata


def _columnwise_positive_transfer_aggregate(
    validation: np.ndarray,
    target_candidate: np.ndarray,
    transfer_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference-style two-candidate column aggregation on held-out target data."""
    x = _as_matrix(validation, "validation")
    p = x.shape[1]
    if target_candidate.shape != (p, p) or transfer_candidate.shape != (p, p):
        raise ValueError("candidate precision shapes do not match validation width.")
    selected = np.zeros((p, p), dtype=np.float64)
    choices = np.zeros(p, dtype=np.int64)
    z0 = x @ target_candidate
    z1 = x @ transfer_candidate
    for j in range(p):
        if len(x) < 2:
            risks = [float("inf"), float("inf")]
        else:
            v0 = float(np.var(z0[:, j], ddof=1) - 2.0 * target_candidate[j, j])
            v1 = float(np.var(z1[:, j], ddof=1) - 2.0 * transfer_candidate[j, j])
            risks = [v0, v1]
        choice = int(np.argmin(risks))
        choices[j] = choice
        selected[:, j] = target_candidate[:, j] if choice == 0 else transfer_candidate[:, j]
    return selected, choices


def reference_trans_clime(
    target_fit: np.ndarray,
    target_validation: np.ndarray,
    source_fit: np.ndarray,
    *,
    target_lambda: float,
    transfer_lambda_const: float = 1.0,
    delta_lambda: float | None = None,
    eigen_floor: float = 1e-6,
) -> PrecisionEstimate:
    """Reference-style Trans-CLIME with an explicit held-out target split."""
    target_fit = _as_matrix(target_fit, "target_fit")
    target_validation = _as_matrix(target_validation, "target_validation")
    source_fit = _as_matrix(source_fit, "source_fit")
    p = target_fit.shape[1]
    if target_validation.shape[1] != p or source_fit.shape[1] != p:
        raise ValueError("feature widths must agree.")

    base = clime(target_fit, lam=target_lambda, eigen_floor=eigen_floor)
    transferred, meta = _trans_clime_core(
        target_fit,
        source_fit,
        target_clime=base,
        delta_lambda=delta_lambda,
        transfer_lambda_const=transfer_lambda_const,
        eigen_floor=eigen_floor,
    )
    raw, choices = _columnwise_positive_transfer_aggregate(
        target_validation,
        base.raw,
        transferred.raw,
    )
    symmetric = symmetrize(raw)
    projection = spd_project(symmetric, eigen_floor=eigen_floor)
    combined_success = min(
        base.lp_diagnostics.success_fraction,
        transferred.lp_diagnostics.success_fraction,
        float(meta["delta_lp_success_fraction"]),
    )
    return PrecisionEstimate(
        raw=raw,
        symmetric=symmetric,
        spd=projection.matrix,
        lp_diagnostics=LPDiagnostics(
            feasible_columns=int(round(combined_success * p)),
            total_columns=p,
            success_fraction=float(combined_success),
            max_constraint_violation=max(
                base.lp_diagnostics.max_constraint_violation,
                transferred.lp_diagnostics.max_constraint_violation,
            ),
        ),
        spd_projection=projection,
        metadata={
            "estimator": "Trans-CLIME-reference-style",
            "target_lambda": float(target_lambda),
            "transfer_lambda_const": float(transfer_lambda_const),
            "validation_samples": int(len(target_validation)),
            "transfer_selected_fraction": float(np.mean(choices == 1)),
            **meta,
        },
    )


def crossfit_trans_clime(
    target: np.ndarray,
    source: np.ndarray,
    *,
    target_lambda: float,
    n_folds: int = 5,
    transfer_lambda_const: float = 1.0,
    delta_lambda: float | None = None,
    seed: int = 42,
    eigen_floor: float = 1e-6,
) -> PrecisionEstimate:
    """COLDSTART extension: deterministic cross-fitted Trans-CLIME aggregation.

    Each fold fits all target-dependent quantities on the complement and uses
    only the held-out fold for positive-transfer aggregation.  Fold estimates
    are averaged.  This avoids a single two-row validation split at N=10, but it
    is an extension and must not be described as the published estimator.
    """
    target = _as_matrix(target, "target")
    source = _as_matrix(source, "source")
    if target.shape[1] != source.shape[1]:
        raise ValueError("target/source feature widths differ.")
    if n_folds < 2 or n_folds > len(target):
        raise ValueError("n_folds must be in [2, len(target)].")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    folds = [np.asarray(v, dtype=np.int64) for v in np.array_split(order, n_folds) if len(v)]
    estimates: list[np.ndarray] = []
    transfer_fractions: list[float] = []
    success: list[float] = []
    violations: list[float] = []

    all_idx = np.arange(len(target))
    for validation_idx in folds:
        fit_idx = np.setdiff1d(all_idx, validation_idx, assume_unique=True)
        if len(fit_idx) < 2 or len(validation_idx) < 2:
            # The reference aggregation variance is not meaningful for a one-row
            # validation fold.  Skip rather than fabricating a risk estimate.
            continue
        est = reference_trans_clime(
            target[fit_idx],
            target[validation_idx],
            source,
            target_lambda=target_lambda,
            transfer_lambda_const=transfer_lambda_const,
            delta_lambda=delta_lambda,
            eigen_floor=eigen_floor,
        )
        estimates.append(est.raw)
        transfer_fractions.append(float(est.metadata["transfer_selected_fraction"]))
        success.append(est.lp_diagnostics.success_fraction)
        violations.append(est.lp_diagnostics.max_constraint_violation)

    if not estimates:
        raise ValueError("No valid cross-fitting folds; use fewer folds.")
    raw = np.mean(np.stack(estimates, axis=0), axis=0)
    symmetric = symmetrize(raw)
    projection = spd_project(symmetric, eigen_floor=eigen_floor)
    p = target.shape[1]
    success_fraction = float(min(success))
    return PrecisionEstimate(
        raw=raw,
        symmetric=symmetric,
        spd=projection.matrix,
        lp_diagnostics=LPDiagnostics(
            feasible_columns=int(round(success_fraction * p)),
            total_columns=p,
            success_fraction=success_fraction,
            max_constraint_violation=float(max(violations)),
        ),
        spd_projection=projection,
        metadata={
            "estimator": "Trans-CLIME-crossfit-COLDSTART-extension",
            "target_lambda": float(target_lambda),
            "transfer_lambda_const": float(transfer_lambda_const),
            "requested_folds": int(n_folds),
            "used_folds": int(len(estimates)),
            "mean_transfer_selected_fraction": float(np.mean(transfer_fractions)),
            "seed": int(seed),
        },
    )


def gaussian_precision_risk(x: np.ndarray, precision: np.ndarray) -> float:
    """Per-feature Gaussian negative log-likelihood up to additive constants."""
    x = _as_matrix(x, "x")
    omega = symmetrize(np.asarray(precision, dtype=np.float64))
    sign, logdet = np.linalg.slogdet(omega)
    if sign <= 0 or not np.isfinite(logdet):
        return float("inf")
    cov = empirical_covariance(x)
    return float((np.trace(cov @ omega) - logdet) / x.shape[1])


def relative_frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    return float(np.linalg.norm(estimate - truth, ord="fro") / max(np.linalg.norm(truth, ord="fro"), EPS))


def max_abs_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(estimate) - np.asarray(truth))))


def support_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float = 1e-6,
) -> dict[str, float]:
    e = np.abs(np.asarray(estimate, dtype=np.float64)) > threshold
    t = np.abs(np.asarray(truth, dtype=np.float64)) > threshold
    np.fill_diagonal(e, False)
    np.fill_diagonal(t, False)
    upper = np.triu(np.ones_like(e, dtype=bool), k=1)
    e = e & upper
    t = t & upper
    tp = int(np.sum(e & t))
    fp = int(np.sum(e & ~t))
    fn = int(np.sum(~e & t))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    union = int(np.sum(e | t))
    jaccard = tp / union if union else 1.0
    return {
        "support_precision": float(precision),
        "support_recall": float(recall),
        "support_f1": float(f1),
        "support_jaccard": float(jaccard),
    }
