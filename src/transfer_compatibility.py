from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


@dataclass(frozen=True)
class CompatibilityMetrics:
    mean_shift_rms: float
    covariance_correlation_frobenius: float
    rbf_mmd2: float
    domain_classifier_auc: float
    source_count: int
    target_count: int
    retained_features: int


def _validate_matrix(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError(f"invalid {name} matrix shape: {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return x


def _source_standardize(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = _validate_matrix(source, "source")
    target = _validate_matrix(target, "target")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source/target feature dimensions differ")
    mu = source.mean(axis=0)
    sd = source.std(axis=0, ddof=1)
    keep = sd > 1e-12
    if not np.any(keep):
        raise ValueError("all source features are constant")
    return (
        (source[:, keep] - mu[keep]) / sd[keep],
        (target[:, keep] - mu[keep]) / sd[keep],
    )


def _correlation(covariance: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    corr = covariance / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def covariance_correlation_discrepancy(source_z: np.ndarray, target_z: np.ndarray) -> float:
    source_cov = LedoitWolf().fit(source_z).covariance_
    target_cov = LedoitWolf().fit(target_z).covariance_
    diff = _correlation(source_cov) - _correlation(target_cov)
    p = diff.shape[0]
    return float(np.linalg.norm(diff, ord="fro") / math.sqrt(p * p))


def rbf_mmd2_balanced(source_z: np.ndarray, target_z: np.ndarray, *, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    n = target_z.shape[0]
    if source_z.shape[0] < n:
        raise ValueError("source must have at least as many rows as target")
    x = source_z[rng.choice(source_z.shape[0], size=n, replace=False)]
    y = target_z
    pooled = np.vstack([x, y])
    distances = pdist(pooled, metric="euclidean")
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1)[:, None]
        b2 = np.sum(b * b, axis=1)[None, :]
        d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
        return np.exp(-gamma * d2)

    value = float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())
    return max(value, 0.0)


def domain_classifier_auc_balanced(source_z: np.ndarray, target_z: np.ndarray, *, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    n = target_z.shape[0]
    x = np.vstack([source_z[rng.choice(source_z.shape[0], size=n, replace=False)], target_z])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    cv = StratifiedKFold(n_splits=min(5, n), shuffle=True, random_state=int(seed))
    model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=int(seed))
    probabilities = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, probabilities))
    return max(auc, 1.0 - auc)


def compute_compatibility_metrics(source_features: np.ndarray, target_features: np.ndarray, *, seed: int) -> CompatibilityMetrics:
    source_z, target_z = _source_standardize(source_features, target_features)
    mean_shift = float(np.sqrt(np.mean(np.square(target_z.mean(axis=0) - source_z.mean(axis=0)))))
    return CompatibilityMetrics(
        mean_shift_rms=mean_shift,
        covariance_correlation_frobenius=covariance_correlation_discrepancy(source_z, target_z),
        rbf_mmd2=rbf_mmd2_balanced(source_z, target_z, seed=seed),
        domain_classifier_auc=domain_classifier_auc_balanced(source_z, target_z, seed=seed),
        source_count=int(source_z.shape[0]),
        target_count=int(target_z.shape[0]),
        retained_features=int(source_z.shape[1]),
    )
