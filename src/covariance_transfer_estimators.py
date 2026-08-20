"""Covariance-shrinkage transfer estimators for COLDSTART P0.3/P0.3b.

The module keeps the scientific question narrow: can source-assisted covariance
shrinkage improve on strong target-only covariance regularization when healthy
target commissioning data are scarce?

All fit/tuning APIs are healthy-only. Synthetic truth and anomaly labels are
never accepted by estimator-selection functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold


EPS = 1e-12


@dataclass(frozen=True)
class CovarianceEstimate:
    covariance: np.ndarray
    precision: np.ndarray
    method: str
    metadata: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class SafeCVSelection:
    """Result of conservative healthy-only RACE source-weight selection."""

    estimate: CovarianceEstimate
    selected_lambda: float
    selected_source_weight: float
    accepted_transfer: bool
    cv_rows: tuple[dict[str, float | int | bool], ...]


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

    lambda_reg=0 is exactly target-only Ledoit-Wolf and is deliberately
    supported so a safe selector can reject source transfer.
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


def _validated_cv_splits(n_samples: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    if n_folds > n_samples:
        raise ValueError("n_folds cannot exceed n_samples")
    splitter = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    return [(train.astype(int), test.astype(int)) for train, test in splitter.split(np.arange(n_samples))]


def safe_cv_race_covariance(
    target_x: np.ndarray,
    source_x: np.ndarray,
    *,
    lambdas: Sequence[float],
    n_folds: int = 5,
    seed: int = 42,
    se_multiplier: float = 1.0,
    eigen_floor: float = 1e-8,
    method: str = "RACECovSafeCV",
) -> SafeCVSelection:
    """Conservatively select source borrowing with paired healthy CV risk.

    For each candidate lambda, the exact same target folds are used. Lambda=0
    is the no-source Ledoit-Wolf fallback. For lambda>0, define paired fold
    improvement d_k = risk(lambda=0)_k - risk(lambda)_k. Transfer is eligible
    only if mean(d) - se_multiplier * SE(d) > 0. Among eligible candidates,
    choose the largest conservative lower improvement; ties prefer smaller
    lambda (less source borrowing). If none qualify, select lambda=0.

    This is a one-standard-error-style safety rule, not a hypothesis test. It
    uses healthy target observations only and never sees anomaly labels/truth.
    The final selected estimator is refit on all target commissioning samples.
    """
    from src.precision_transfer_estimators import gaussian_precision_risk

    target_x = _as_matrix(target_x, "target_x")
    source_x = _as_matrix(source_x, "source_x")
    if target_x.shape[1] != source_x.shape[1]:
        raise ValueError("target/source feature widths differ")
    if se_multiplier < 0:
        raise ValueError("se_multiplier must be non-negative")

    grid = sorted({float(v) for v in lambdas})
    if not grid or any(v < 0 for v in grid):
        raise ValueError("lambdas must be non-empty and non-negative")
    if 0.0 not in grid:
        raise ValueError("lambdas must include 0 for safe source rejection")

    splits = _validated_cv_splits(len(target_x), int(n_folds), int(seed))
    fold_risks: dict[float, list[float]] = {lam: [] for lam in grid}
    for train_idx, val_idx in splits:
        train_x = target_x[train_idx]
        val_x = target_x[val_idx]
        for lam in grid:
            est = race_covariance(
                train_x,
                source_x,
                lambda_reg=lam,
                eigen_floor=eigen_floor,
                method=f"{method}[lambda={lam:g}]",
            )
            fold_risks[lam].append(float(gaussian_precision_risk(val_x, est.precision)))

    baseline = np.asarray(fold_risks[0.0], dtype=np.float64)
    rows: list[dict[str, float | int | bool]] = []
    eligible: list[tuple[float, float]] = []
    for lam in grid:
        risks = np.asarray(fold_risks[lam], dtype=np.float64)
        diff = baseline - risks
        mean_risk = float(np.mean(risks))
        mean_improvement = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.inf
        lower = mean_improvement - float(se_multiplier) * se
        accepted = bool(lam > 0.0 and lower > 0.0)
        rows.append({
            "lambda_reg": lam,
            "mean_cv_risk": mean_risk,
            "mean_improvement_vs_no_source": mean_improvement,
            "se_improvement_vs_no_source": se,
            "conservative_lower_improvement": lower,
            "accepted_transfer_candidate": accepted,
            "n_folds": int(n_folds),
        })
        if accepted:
            eligible.append((lower, lam))

    if eligible:
        # Highest conservative evidence for benefit; tie -> smaller lambda.
        eligible.sort(key=lambda item: (-item[0], item[1]))
        selected_lambda = float(eligible[0][1])
        accepted_transfer = True
    else:
        selected_lambda = 0.0
        accepted_transfer = False

    final = race_covariance(
        target_x,
        source_x,
        lambda_reg=selected_lambda,
        eigen_floor=eigen_floor,
        method=method,
    )
    meta = dict(final.metadata)
    meta.update({
        "accepted_transfer": accepted_transfer,
        "cv_folds": int(n_folds),
        "cv_seed": int(seed),
        "se_multiplier": float(se_multiplier),
    })
    final = CovarianceEstimate(final.covariance, final.precision, method, meta)
    return SafeCVSelection(
        estimate=final,
        selected_lambda=selected_lambda,
        selected_source_weight=float(final.metadata["source_weight"]),
        accepted_transfer=accepted_transfer,
        cv_rows=tuple(rows),
    )


def safe_cv_target_only(
    target_x: np.ndarray,
    *,
    ridge_gammas: Sequence[float],
    n_folds: int = 5,
    seed: int = 42,
    eigen_floor: float = 1e-8,
    method: str = "BestTargetOnlySafeCV",
) -> tuple[CovarianceEstimate, list[dict[str, float | str]]]:
    """Target-only covariance selection with the same K-fold data budget.

    Candidates are Ledoit-Wolf and ridge shrinkage. Mean held-out healthy
    Gaussian risk across identical folds selects the family/hyperparameter;
    the winner is then refit on all commissioning observations.
    """
    from src.precision_transfer_estimators import gaussian_precision_risk

    target_x = _as_matrix(target_x, "target_x")
    gammas = tuple(float(g) for g in ridge_gammas)
    if not gammas or any(g <= 0 or g > 1 for g in gammas):
        raise ValueError("ridge_gammas must be in (0,1]")

    splits = _validated_cv_splits(len(target_x), int(n_folds), int(seed))
    specs: list[tuple[str, float | None]] = [("LedoitWolf", None)] + [("Ridge", g) for g in gammas]
    rows: list[dict[str, float | str]] = []
    scored: list[tuple[float, str, float | None]] = []
    for family, value in specs:
        risks: list[float] = []
        for train_idx, val_idx in splits:
            train_x = target_x[train_idx]
            val_x = target_x[val_idx]
            if family == "LedoitWolf":
                est = ledoit_wolf_covariance(train_x, method="TargetLedoitWolfCV", eigen_floor=eigen_floor)
            else:
                assert value is not None
                est = ridge_covariance(train_x, gamma=value, method="TargetRidgeCV", eigen_floor=eigen_floor)
            risks.append(float(gaussian_precision_risk(val_x, est.precision)))
        mean_risk = float(np.mean(risks))
        label = family if value is None else f"{family}[gamma={value:g}]"
        rows.append({"candidate": label, "mean_cv_risk": mean_risk})
        scored.append((mean_risk, label, value))

    scored.sort(key=lambda item: (item[0], item[1]))
    _, label, value = scored[0]
    if label == "LedoitWolf":
        final = ledoit_wolf_covariance(target_x, method=method, eigen_floor=eigen_floor)
        selected_family = "LedoitWolf"
    else:
        assert value is not None
        final = ridge_covariance(target_x, gamma=value, method=method, eigen_floor=eigen_floor)
        selected_family = "Ridge"
    meta = dict(final.metadata)
    meta.update({"selected_family": selected_family, "selected_candidate": label, "cv_folds": int(n_folds), "cv_seed": int(seed)})
    return CovarianceEstimate(final.covariance, final.precision, method, meta), rows
