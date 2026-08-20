"""Covariance-shrinkage transfer estimators for COLDSTART P0.3.

This module keeps the P0.3 question deliberately narrow: can source-assisted
covariance shrinkage improve on strong target-only covariance regularization
when healthy target commissioning data are scarce?

All estimators are healthy-only.  Synthetic truth is never accepted by any fit
or tuning API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.covariance import LedoitWolf


EPS = 1e-12


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    precision: np.ndarray
    method: str
    metadata: dict[str, float | int | str | bool]


def _as_matrix(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} has invalid shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _sym(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return 0.5 * (a + a.T)


def _spd_floor(covariance: np.ndarray, eigen_floor: float = 1e-8) -> tuple[np.ndarray, float]:
    if eigen_floor <= 0:
        raise ValueError("eigen_floor must be positive")
    cov = _sym(covariance)
    vals, vecs = np.linalg.eigh(cov)
    before = float(np.min(vals))
    vals = np.maximum(vals, float(eigen_floor))
    out = _sym((vecs * vals) @ vecs.T)
    return out, before


def _estimate_from_covariance(
    covariance: np.ndarray,
    method: str,
    metadata: dict[str, float | int | str | bool] | None = None,
    *,
    eigen_floor: float = 1e-8,
) -> CovarianceEstimate:
    cov, min_before = _spd_floor(covariance, eigen_floor=eigen_floor)
    precision = np.linalg.inv(cov)
    meta = dict(metadata or {})
    meta["min_eigenvalue_before_floor"] = min_before
    meta["condition_number"] = float(np.linalg.cond(cov))
    return CovarianceEstimate(covariance=cov, precision=precision, method=method, metadata=meta)


def ledoit_wolf_covariance(
    x: np.ndarray,
    *,
    method: str = "LedoitWolf",
    eigen_floor: float = 1e-8,
) -> CovarianceEstimate:
    x = _as_matrix(x, "x")
    fit = LedoitWolf(assume_centered=False, store_precision=True).fit(x)
    return _estimate_from_covariance(
        np.asarray(fit.covariance_, dtype=np.float64),
        method,
        {
            "n_samples": int(len(x)),
            "n_features": int(x.shape[1]),
            "ledoit_wolf_shrinkage": float(fit.shrinkage_),
        },
        eigen_floor=eigen_floor,
    )


def ridge_covariance(
    x: np.ndarray,
    *,
    gamma: float,
    method: str = "TargetRidge",
    eigen_floor: float = 1e-8,
) -> CovarianceEstimate:
    x = _as_matrix(x, "x")
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    empirical = np.atleast_2d(np.cov(x, rowvar=False, ddof=1)).astype(np.float64)
    empirical = _sym(empirical)
    p = empirical.shape[0]
    mu = max(float(np.trace(empirical) / p), EPS)
    cov = (1.0 - gamma) * empirical + gamma * mu * np.eye(p)
    return _estimate_from_covariance(
        cov,
        method,
        {"gamma": gamma, "n_samples": int(len(x)), "n_features": int(p)},
        eigen_floor=eigen_floor,
    )


def pooled_ledoit_wolf(
    target_x: np.ndarray,
    source_x: np.ndarray,
    *,
    eigen_floor: float = 1e-8,
) -> CovarianceEstimate:
    target_x = _as_matrix(target_x, "target_x")
    source_x = _as_matrix(source_x, "source_x")
    if target_x.shape[1] != source_x.shape[1]:
        raise ValueError("target/source feature widths differ")
    pooled = np.vstack((target_x, source_x))
    est = ledoit_wolf_covariance(pooled, method="PooledLedoitWolf", eigen_floor=eigen_floor)
    meta = dict(est.metadata)
    meta.update({"target_samples": int(len(target_x)), "source_samples": int(len(source_x))})
    return CovarianceEstimate(est.covariance, est.precision, est.method, meta)


def race_covariance(
    target_x: np.ndarray,
    source_x: np.ndarray,
    *,
    lambda_reg: float = 60.0,
    eigen_floor: float = 1e-8,
    method: str = "RACECov",
) -> CovarianceEstimate:
    """Blend target/source Ledoit-Wolf covariances using the original RACE weight.

    target_weight = N / (N + lambda_reg)
    source_weight = 1 - target_weight

    lambda_reg=0 is intentionally supported: it is exactly the target-only
    Ledoit-Wolf estimator and allows healthy-only adaptive selection to reject
    source transfer without a separate special case.
    """
    target_x = _as_matrix(target_x, "target_x")
    source_x = _as_matrix(source_x, "source_x")
    if target_x.shape[1] != source_x.shape[1]:
        raise ValueError("target/source feature widths differ")
    lambda_reg = float(lambda_reg)
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be non-negative")

    target = ledoit_wolf_covariance(target_x, method="TargetLedoitWolf", eigen_floor=eigen_floor)
    source = ledoit_wolf_covariance(source_x, method="SourceLedoitWolf", eigen_floor=eigen_floor)
    n = len(target_x)
    target_weight = 1.0 if lambda_reg == 0.0 else float(n / (n + lambda_reg))
    source_weight = 1.0 - target_weight
    covariance = target_weight * target.covariance + source_weight * source.covariance
    return _estimate_from_covariance(
        covariance,
        method,
        {
            "lambda_reg": lambda_reg,
            "target_weight": target_weight,
            "source_weight": source_weight,
            "target_samples": int(n),
            "source_samples": int(len(source_x)),
            "target_lw_shrinkage": float(target.metadata["ledoit_wolf_shrinkage"]),
            "source_lw_shrinkage": float(source.metadata["ledoit_wolf_shrinkage"]),
        },
        eigen_floor=eigen_floor,
    )


def choose_by_healthy_risk(
    estimates: Iterable[CovarianceEstimate],
    tune_x: np.ndarray,
) -> tuple[CovarianceEstimate, list[dict[str, float | str]]]:
    """Select an estimator using held-out healthy Gaussian negative log-risk."""
    from src.precision_transfer_estimators import gaussian_precision_risk

    tune_x = _as_matrix(tune_x, "tune_x")
    rows: list[dict[str, float | str]] = []
    best: tuple[float, str, CovarianceEstimate] | None = None
    for est in estimates:
        risk = float(gaussian_precision_risk(tune_x, est.precision))
        rows.append({"method": est.method, "healthy_risk": risk})
        key = (risk, est.method, est)
        if best is None or key[0] < best[0] or (key[0] == best[0] and key[1] < best[1]):
            best = key
    if best is None:
        raise ValueError("No covariance candidates supplied")
    return best[2], rows
