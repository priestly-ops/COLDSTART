from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class DependenceSummary:
    n: int
    lag1_autocorrelation: float
    max_abs_autocorrelation: float
    ljung_box_q: float
    ljung_box_pvalue: float
    effective_sample_size: float


def _as_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 3:
        raise ValueError("At least three observations are required.")
    if not np.isfinite(x).all():
        raise ValueError("values contain NaN or Inf.")
    return x


def autocorrelation(values: np.ndarray, max_lag: int = 20) -> np.ndarray:
    x = _as_1d(values)
    if max_lag < 1:
        raise ValueError("max_lag must be positive.")
    max_lag = min(int(max_lag), x.size - 2)
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return np.zeros(max_lag, dtype=np.float64)
    return np.asarray(
        [float(np.dot(centered[:-lag], centered[lag:]) / denom) for lag in range(1, max_lag + 1)],
        dtype=np.float64,
    )


def ljung_box(values: np.ndarray, max_lag: int = 20) -> tuple[float, float]:
    x = _as_1d(values)
    acf = autocorrelation(x, max_lag=max_lag)
    lags = np.arange(1, len(acf) + 1, dtype=np.float64)
    n = float(x.size)
    q = float(n * (n + 2.0) * np.sum((acf ** 2) / (n - lags)))
    p = float(chi2.sf(q, df=len(acf)))
    return q, p


def effective_sample_size(values: np.ndarray, max_lag: int = 20) -> float:
    """Initial-positive-sequence ESS estimate for a scalar series."""
    x = _as_1d(values)
    acf = autocorrelation(x, max_lag=max_lag)
    positive_sum = 0.0
    for rho in acf:
        if rho <= 0.0:
            break
        positive_sum += float(rho)
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    ess = float(x.size / tau)
    return float(np.clip(ess, 1.0, float(x.size)))


def summarize_dependence(values: np.ndarray, max_lag: int = 20) -> DependenceSummary:
    x = _as_1d(values)
    acf = autocorrelation(x, max_lag=max_lag)
    q, p = ljung_box(x, max_lag=max_lag)
    return DependenceSummary(
        n=int(x.size),
        lag1_autocorrelation=float(acf[0]),
        max_abs_autocorrelation=float(np.max(np.abs(acf))),
        ljung_box_q=float(q),
        ljung_box_pvalue=float(p),
        effective_sample_size=float(effective_sample_size(x, max_lag=max_lag)),
    )


def permutation_pvalue_lag1(
    values: np.ndarray,
    *,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    x = _as_1d(values)
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")
    observed = abs(float(autocorrelation(x, max_lag=1)[0]))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_permutations)):
        shuffled = rng.permutation(x)
        value = abs(float(autocorrelation(shuffled, max_lag=1)[0]))
        exceed += int(value >= observed - 1e-15)
    return float((exceed + 1) / (int(n_permutations) + 1))


def moving_block_bootstrap_means(
    values: np.ndarray,
    *,
    block_size: int,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Circular moving-block bootstrap for a scalar ordered sequence."""
    x = _as_1d(values)
    n = x.size
    block_size = int(block_size)
    if block_size <= 0 or block_size > n:
        raise ValueError("block_size must lie in [1, n].")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    rng = np.random.default_rng(seed)
    blocks_per_draw = int(np.ceil(n / block_size))
    means = np.empty(int(n_bootstrap), dtype=np.float64)
    offsets = np.arange(block_size, dtype=np.int64)

    for b in range(int(n_bootstrap)):
        starts = rng.integers(0, n, size=blocks_per_draw)
        indices = (starts[:, None] + offsets[None, :]) % n
        sample = x[indices.reshape(-1)[:n]]
        means[b] = float(np.mean(sample))
    return means


def block_bootstrap_interval(
    values: np.ndarray,
    *,
    block_size: int,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    means = moving_block_bootstrap_means(
        values,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )
