from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class PairedInferenceResult:
    n_pairs: int
    mean_delta: float
    median_delta: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    wilcoxon_pvalue: float
    signflip_pvalue: float
    signflip_method: str
    positive_count: int
    negative_count: int
    neutral_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_deltas(deltas: np.ndarray) -> np.ndarray:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"paired deltas must be one-dimensional, got shape {values.shape}")
    if values.size < 2:
        raise ValueError("paired inference requires at least two paired observations")
    if not np.isfinite(values).all():
        raise ValueError("paired deltas contain NaN or Inf")
    return values


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired mean difference.

    Resampling is performed over complete pairs, equivalently over the precomputed
    paired differences. This preserves the TargetOnly/RACE pairing within each seed.
    """
    values = _validate_deltas(deltas)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    rng = np.random.default_rng(int(seed))
    n = values.size
    indices = rng.integers(0, n, size=(int(repetitions), n))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def paired_signflip_pvalue(
    deltas: np.ndarray,
    *,
    exact_max_nonzero: int = 20,
    monte_carlo_repetitions: int = 100_000,
    seed: int = 42,
) -> tuple[float, str]:
    """Two-sided paired randomization/sign-flip p-value for the mean difference.

    For at most ``exact_max_nonzero`` non-zero pairs, all 2^k sign assignments are
    enumerated exactly. For larger samples, a deterministic Monte Carlo estimate
    with the +1 correction is used.
    """
    values = _validate_deltas(deltas)
    nonzero = values[np.abs(values) > 0.0]
    k = int(nonzero.size)
    if k == 0:
        return 1.0, "all_zero"

    magnitudes = np.abs(nonzero)
    observed = abs(float(nonzero.sum()))
    tolerance = 1e-15

    if k <= int(exact_max_nonzero):
        total = 1 << k
        extreme = 0
        for bits in range(total):
            signed_sum = 0.0
            for j, magnitude in enumerate(magnitudes):
                signed_sum += magnitude if (bits >> j) & 1 else -magnitude
            if abs(signed_sum) >= observed - tolerance:
                extreme += 1
        return float(extreme / total), "exact"

    if monte_carlo_repetitions < 1:
        raise ValueError("monte_carlo_repetitions must be >= 1")
    rng = np.random.default_rng(int(seed))
    extreme = 0
    for _ in range(int(monte_carlo_repetitions)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=k)
        permuted = float(np.dot(signs, magnitudes))
        if abs(permuted) >= observed - tolerance:
            extreme += 1
    pvalue = (extreme + 1.0) / (int(monte_carlo_repetitions) + 1.0)
    return float(pvalue), "monte_carlo"


def paired_wilcoxon_pvalue(deltas: np.ndarray) -> float:
    values = _validate_deltas(deltas)
    if np.all(values == 0.0):
        return 1.0
    result = wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return float(result.pvalue)


def summarize_paired_deltas(
    deltas: np.ndarray,
    *,
    confidence: float = 0.95,
    bootstrap_repetitions: int = 10_000,
    signflip_repetitions: int = 100_000,
    seed: int = 42,
    neutral_tolerance: float = 1e-12,
) -> PairedInferenceResult:
    values = _validate_deltas(deltas)
    lower, upper = paired_bootstrap_mean_ci(
        values,
        confidence=confidence,
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    signflip_p, signflip_method = paired_signflip_pvalue(
        values,
        monte_carlo_repetitions=signflip_repetitions,
        seed=seed,
    )
    wilcoxon_p = paired_wilcoxon_pvalue(values)

    positive = int(np.sum(values > neutral_tolerance))
    negative = int(np.sum(values < -neutral_tolerance))
    neutral = int(values.size - positive - negative)

    return PairedInferenceResult(
        n_pairs=int(values.size),
        mean_delta=float(values.mean()),
        median_delta=float(np.median(values)),
        bootstrap_ci_lower=lower,
        bootstrap_ci_upper=upper,
        wilcoxon_pvalue=wilcoxon_p,
        signflip_pvalue=signflip_p,
        signflip_method=signflip_method,
        positive_count=positive,
        negative_count=negative,
        neutral_count=neutral,
    )
