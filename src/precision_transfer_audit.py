"""Healthy-only precision-structure feasibility audit for COLDSTART P0.

The module tests whether source and target healthy executions share stable
conditional-dependency structure. It intentionally does not score anomalies,
tune from anomalies, or implement a deployment detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


EPSILON = 1e-12
P0_DECISIONS = (
    "P0_PASS_SHARED_PRECISION_STRUCTURE",
    "P0_FAIL_SYNTHETIC_IDENTIFIABILITY",
    "P0_FAIL_SOURCE_TARGET_STRUCTURE_UNRELATED",
    "P0_FAIL_DIFFERENTIAL_NOT_SPARSE",
    "P0_FAIL_REGULARIZATION_FRAGILE",
    "P0_INCONCLUSIVE_MORE_HEALTHY_DATA_REQUIRED",
    "P0_INCOMPLETE_REPLICATION",
)


@dataclass(frozen=True)
class StabilityConfig:
    """Configuration for StARS-inspired stability selection.

    The implemented instability score is the StARS edge-instability functional
    mean_{i<j} 2*pi_ij*(1-pi_ij) over subsample edge-selection probabilities.
    Alpha selection is a frozen sparse-stable heuristic: choose the largest
    alpha with nonempty stable support under the instability cap, otherwise the
    nonempty alpha with minimum instability. This is not the full bounded StARS
    path rule and is therefore described as StARS-inspired in reports.
    """

    alpha_grid: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
    subsample_fraction: float = 0.80
    resamples: int = 24
    stable_edge_threshold: float = 0.70
    instability_threshold: float = 0.05
    edge_abs_threshold: float = 1e-8
    max_iter: int = 200
    tol: float = 1e-4
    min_successful_fit_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not self.alpha_grid:
            raise ValueError("alpha_grid cannot be empty.")
        if any(alpha <= 0.0 for alpha in self.alpha_grid):
            raise ValueError("alpha_grid values must be positive.")
        if not 0.0 < self.subsample_fraction <= 1.0:
            raise ValueError("subsample_fraction must be in (0, 1].")
        if self.resamples <= 0:
            raise ValueError("resamples must be positive.")
        if not 0.0 <= self.stable_edge_threshold <= 1.0:
            raise ValueError("stable_edge_threshold must be in [0, 1].")
        if self.instability_threshold < 0.0:
            raise ValueError("instability_threshold cannot be negative.")
        if not 0.0 < self.min_successful_fit_fraction <= 1.0:
            raise ValueError("min_successful_fit_fraction must be in (0, 1].")


@dataclass(frozen=True)
class PrecisionStabilityResult:
    selected_alpha: float
    edge_probabilities: np.ndarray
    precision: np.ndarray
    stable_edges: frozenset[tuple[int, int]]
    stability_rows: tuple[dict[str, object], ...]
    sensitivity_rows: tuple[dict[str, object], ...]
    numerical: dict[str, object]


def robust_target_scale(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Scale source and target with target-commissioning robust statistics."""

    source = _validate_matrix(source, "source")
    target = _validate_matrix(target, "target")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension.")

    center = np.median(target, axis=0)
    q25 = np.quantile(target, 0.25, axis=0)
    q75 = np.quantile(target, 0.75, axis=0)
    iqr = q75 - q25
    target_std = np.std(target, axis=0, ddof=1 if len(target) > 1 else 0)
    scale = np.where(iqr > EPSILON, iqr / 1.349, target_std)
    scale = np.where(scale > EPSILON, scale, 1.0)
    source_z = (source - center) / scale
    target_z = (target - center) / scale
    source_clip_fraction = float(np.mean(np.abs(source_z) > 12.0))
    target_clip_fraction = float(np.mean(np.abs(target_z) > 12.0))

    return (
        np.clip(source_z, -12.0, 12.0),
        np.clip(target_z, -12.0, 12.0),
        {
            "scaling_procedure": "target commissioning robust median/IQR; source transformed with same target healthy scaler",
            "zero_or_tiny_scale_features": int(np.sum(scale <= EPSILON)),
            "source_clip_fraction": source_clip_fraction,
            "target_clip_fraction": target_clip_fraction,
        },
    )


def estimate_precision_stability(
    features: np.ndarray,
    *,
    config: StabilityConfig,
    rng_seed: int,
    prefix: str,
) -> PrecisionStabilityResult:
    """Estimate sparse precision edge probabilities by stability resampling.

    Important numerical rule:
    failed/non-PD GraphicalLasso fits do NOT contribute fallback edges to the
    stability counts. Edge probabilities are normalized by the number of
    successful fits for that alpha. This prevents numerical failures from
    appearing as reproducible dense graph structure.
    """

    x = _validate_matrix(features, prefix)
    n_samples, n_features = x.shape
    resample_size = max(2, min(n_samples, int(np.ceil(config.subsample_fraction * n_samples))))
    edge_counts_by_alpha: dict[float, np.ndarray] = {}
    stability_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(rng_seed)

    # Freeze the exact same resamples for all alpha values so regularization
    # comparisons are not confounded by different random subsets.
    resample_indices = [
        rng.choice(n_samples, size=resample_size, replace=(n_samples < resample_size))
        for _ in range(config.resamples)
    ]

    for alpha in config.alpha_grid:
        counts = np.zeros((n_features, n_features), dtype=np.float64)
        fit_rows: list[dict[str, object]] = []
        successful_fits = 0

        for idx in resample_indices:
            fit = fit_graphical_lasso(
                x[idx],
                alpha=float(alpha),
                edge_abs_threshold=config.edge_abs_threshold,
                max_iter=config.max_iter,
                tol=config.tol,
            )
            usable = bool(fit["converged"]) and bool(fit["positive_definite"]) and not bool(fit.get("used_fallback", False))
            if usable:
                counts += fit["edge_mask"].astype(np.float64)
                successful_fits += 1

            fit_rows.append(
                {
                    "usable": usable,
                    "converged": bool(fit["converged"]),
                    "iterations": int(fit["iterations"]),
                    "edges": int(fit["edge_count"]),
                    "positive_definite": bool(fit["positive_definite"]),
                    "condition_number": float(fit["condition_number"]),
                    "warnings": str(fit.get("warnings", "")),
                    "used_fallback": bool(fit.get("used_fallback", False)),
                }
            )

        success_fraction = successful_fits / float(config.resamples)
        if successful_fits > 0:
            probabilities = counts / float(successful_fits)
        else:
            probabilities = np.zeros_like(counts)

        edge_counts_by_alpha[float(alpha)] = probabilities
        upper = _upper_values(probabilities)
        instability = float(np.mean(2.0 * upper * (1.0 - upper))) if upper.size else 0.0
        stable_edges = edge_set(probabilities, config.stable_edge_threshold)

        stability_rows.append(
            {
                "graph": prefix,
                "alpha": float(alpha),
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "subsample_size": int(resample_size),
                "resamples": int(config.resamples),
                "successful_fits": int(successful_fits),
                "successful_fit_fraction": float(success_fraction),
                "mean_edge_probability": float(np.mean(upper)) if upper.size else 0.0,
                "stability_instability": instability,
                "stable_edges": int(len(stable_edges)),
                "stable_density": graph_density(len(stable_edges), n_features),
                "fit_failures": int(sum(not row["usable"] for row in fit_rows)),
                "positive_definite_failures": int(sum(not row["positive_definite"] for row in fit_rows)),
                "fallback_fits": int(sum(row["used_fallback"] for row in fit_rows)),
                "warning_fits": int(sum(bool(row["warnings"]) for row in fit_rows)),
                "mean_iterations": float(np.mean([row["iterations"] for row in fit_rows])),
            }
        )

    selected_alpha = select_alpha(stability_rows, config)
    selected_probabilities = edge_counts_by_alpha[selected_alpha]
    selected_row = next(row for row in stability_rows if float(row["alpha"]) == float(selected_alpha))

    final_fit = fit_graphical_lasso(
        x,
        alpha=selected_alpha,
        edge_abs_threshold=config.edge_abs_threshold,
        max_iter=config.max_iter,
        tol=config.tol,
    )

    stable_edges = edge_set(selected_probabilities, config.stable_edge_threshold)
    sensitivity_rows = regularization_sensitivity(
        edge_counts_by_alpha=edge_counts_by_alpha,
        selected_alpha=selected_alpha,
        config=config,
        prefix=prefix,
    )

    numerically_reliable = bool(
        float(selected_row["successful_fit_fraction"]) >= config.min_successful_fit_fraction
        and bool(final_fit["converged"])
        and bool(final_fit["positive_definite"])
        and not bool(final_fit.get("used_fallback", False))
    )

    numerical = {
        "graph": prefix,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "selected_regularization": float(selected_alpha),
        "selected_successful_fit_fraction": float(selected_row["successful_fit_fraction"]),
        "numerically_reliable": numerically_reliable,
        "number_of_edges": int(final_fit["edge_count"]),
        "graph_density": graph_density(int(final_fit["edge_count"]), n_features),
        "condition_number": float(final_fit["condition_number"]),
        "min_eigenvalue": float(final_fit["min_eigenvalue"]),
        "max_eigenvalue": float(final_fit["max_eigenvalue"]),
        "positive_definite": bool(final_fit["positive_definite"]),
        "converged": bool(final_fit["converged"]),
        "used_fallback": bool(final_fit.get("used_fallback", False)),
        "iterations": int(final_fit["iterations"]),
        "warnings": str(final_fit.get("warnings", "")),
    }

    return PrecisionStabilityResult(
        selected_alpha=selected_alpha,
        edge_probabilities=selected_probabilities,
        precision=np.asarray(final_fit["precision"], dtype=np.float64),
        stable_edges=frozenset(stable_edges),
        stability_rows=tuple(stability_rows),
        sensitivity_rows=tuple(sensitivity_rows),
        numerical=numerical,
    )


def fit_graphical_lasso(
    features: np.ndarray,
    *,
    alpha: float,
    edge_abs_threshold: float = 1e-8,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> dict[str, object]:
    """Fit Graphical Lasso and return numerical diagnostics."""

    x = _validate_matrix(features, "features")
    model = GraphicalLasso(alpha=float(alpha), max_iter=max_iter, tol=tol, assume_centered=False)
    converged = True
    used_fallback = False
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x)
        warning_messages = [str(item.message) for item in caught]
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            converged = False
        precision = np.asarray(model.precision_, dtype=np.float64)
        iterations = int(getattr(model, "n_iter_", -1))
    except FloatingPointError as exc:
        converged = False
        used_fallback = True
        warning_messages = [f"GraphicalLasso FloatingPointError fallback: {exc}"]
        covariance = np.cov(x, rowvar=False)
        covariance = np.atleast_2d(covariance) + float(alpha) * np.eye(x.shape[1])
        precision = np.linalg.pinv(covariance)
        iterations = -1
    except Exception as exc:
        converged = False
        used_fallback = True
        warning_messages = [f"GraphicalLasso exception fallback: {type(exc).__name__}: {exc}"]
        covariance = np.cov(x, rowvar=False)
        covariance = np.atleast_2d(covariance) + float(alpha) * np.eye(x.shape[1])
        precision = np.linalg.pinv(covariance)
        iterations = -1

    precision = _symmetrize(precision)
    eigvals = np.linalg.eigvalsh(precision)
    min_eig = float(np.min(eigvals))
    max_eig = float(np.max(eigvals))
    positive = bool(min_eig > 0.0 and np.isfinite(eigvals).all())
    condition = float(max_eig / max(min_eig, EPSILON)) if np.isfinite(max_eig) else float("inf")
    mask = precision_edge_mask(precision, edge_abs_threshold=edge_abs_threshold)
    return {
        "precision": precision,
        "edge_mask": mask,
        "edge_count": int(np.sum(np.triu(mask, k=1))),
        "condition_number": condition,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "positive_definite": positive,
        "converged": converged,
        "used_fallback": used_fallback,
        "iterations": iterations,
        "warnings": "; ".join(warning_messages[:3]),
    }


def select_alpha(
    stability_rows: Sequence[dict[str, object]],
    config: StabilityConfig,
) -> float:
    """Select the sparsest stable alpha under the configured instability cap."""

    rows = sorted(stability_rows, key=lambda row: float(row["alpha"]))
    eligible = [
        row
        for row in rows
        if float(row["stability_instability"]) <= config.instability_threshold
        and int(row["stable_edges"]) > 0
        and float(row.get("successful_fit_fraction", 1.0)) >= config.min_successful_fit_fraction
    ]
    if eligible:
        return float(max(eligible, key=lambda row: float(row["alpha"]))["alpha"])
    nonempty = [
        row for row in rows
        if int(row["stable_edges"]) > 0
        and float(row.get("successful_fit_fraction", 1.0)) >= config.min_successful_fit_fraction
    ]
    if nonempty:
        return float(min(nonempty, key=lambda row: float(row["stability_instability"]))["alpha"])
    return float(max(rows, key=lambda row: float(row["alpha"]))["alpha"])


def regularization_sensitivity(
    *,
    edge_counts_by_alpha: dict[float, np.ndarray],
    selected_alpha: float,
    config: StabilityConfig,
    prefix: str,
) -> list[dict[str, object]]:
    alphas = sorted(edge_counts_by_alpha)
    selected_index = alphas.index(selected_alpha)
    neighbor_indices = sorted(set([max(0, selected_index - 1), selected_index, min(len(alphas) - 1, selected_index + 1)]))
    rows: list[dict[str, object]] = []
    for index in neighbor_indices:
        alpha = alphas[index]
        probs = edge_counts_by_alpha[alpha]
        stable = edge_set(probs, config.stable_edge_threshold)
        rows.append(
            {
                "graph": prefix,
                "alpha": float(alpha),
                "is_selected_alpha": bool(alpha == selected_alpha),
                "stable_edges": int(len(stable)),
                "stable_density": graph_density(len(stable), probs.shape[0]),
            }
        )
    return rows


def compare_precision_structures(
    source_result: PrecisionStabilityResult,
    target_result: PrecisionStabilityResult,
    *,
    differential_partial_corr_threshold: float = 0.10,
) -> dict[str, object]:
    """Compute P0 shared-edge, differential, and partial-correlation metrics."""

    source_edges = set(source_result.stable_edges)
    target_edges = set(target_result.stable_edges)
    shared = source_edges & target_edges
    union = source_edges | target_edges
    p = source_result.edge_probabilities.shape[0]
    weighted_overlap = float(
        np.sum(np.minimum(_upper_values(source_result.edge_probabilities), _upper_values(target_result.edge_probabilities)))
        / (np.sum(np.maximum(_upper_values(source_result.edge_probabilities), _upper_values(target_result.edge_probabilities))) + EPSILON)
    )
    diff_edges = stable_differential_edges(
        source_result.precision,
        target_result.precision,
        source_edges,
        target_edges,
        partial_corr_abs_diff_threshold=differential_partial_corr_threshold,
    )
    frob = float(
        np.linalg.norm(target_result.precision - source_result.precision, ord="fro")
        / (np.linalg.norm(target_result.precision, ord="fro") + EPSILON)
    )
    partial = partial_correlation_agreement(
        source_result.precision,
        target_result.precision,
        shared,
    )
    return {
        "source_stable_edges": int(len(source_edges)),
        "target_stable_edges": int(len(target_edges)),
        "shared_stable_edges": int(len(shared)),
        "union_stable_edges": int(len(union)),
        "source_graph_density": graph_density(len(source_edges), p),
        "target_graph_density": graph_density(len(target_edges), p),
        "stable_jaccard": jaccard(source_edges, target_edges),
        "weighted_overlap": weighted_overlap,
        "stable_differential_edges": int(len(diff_edges)),
        "differential_ratio": float(0.0 if not union else len(diff_edges) / len(union)),
        "D_Omega": frob,
        **partial,
    }


def stable_differential_edges(
    source_precision: np.ndarray,
    target_precision: np.ndarray,
    source_edges: Iterable[tuple[int, int]],
    target_edges: Iterable[tuple[int, int]],
    *,
    partial_corr_abs_diff_threshold: float = 0.10,
) -> frozenset[tuple[int, int]]:
    """Return the stable target-specific differential edge set.

    Every stable support disagreement is automatically differential:
        E_support_delta = E_source symmetric_difference E_target

    For edges stable in both graphs, a sufficiently large partial-correlation
    change is also differential. This avoids under-counting Delta_T merely
    because a support-disagreement edge has a small final-fit coefficient.
    """

    source_edges = set(source_edges)
    target_edges = set(target_edges)
    support_delta = source_edges ^ target_edges
    shared = source_edges & target_edges
    threshold = float(partial_corr_abs_diff_threshold)

    weight_delta = {
        (i, j)
        for i, j in shared
        if abs(
            partial_correlation(target_precision, i, j)
            - partial_correlation(source_precision, i, j)
        ) >= threshold
    }
    return frozenset(support_delta | weight_delta)


def feature_permutation_null(
    source_probs: np.ndarray,
    target_probs: np.ndarray,
    *,
    stable_edge_threshold: float,
    replicates: int,
    rng_seed: int,
) -> pd.DataFrame:
    """Permute feature identity in the target graph and recompute overlap."""

    source_edges = edge_set(source_probs, stable_edge_threshold)
    target_p = np.asarray(target_probs, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    for replicate in range(replicates):
        perm = rng.permutation(target_p.shape[0])
        permuted = target_p[np.ix_(perm, perm)]
        target_edges = edge_set(permuted, stable_edge_threshold)
        rows.append(
            {
                "null_type": "feature_identity_permutation",
                "replicate": int(replicate),
                "permutation_changed_identity": bool(np.any(perm != np.arange(target_p.shape[0]))),
                "jaccard": jaccard(source_edges, target_edges),
                "weighted_overlap": float(
                    np.sum(np.minimum(_upper_values(source_probs), _upper_values(permuted)))
                    / (np.sum(np.maximum(_upper_values(source_probs), _upper_values(permuted))) + EPSILON)
                ),
            }
        )
    return pd.DataFrame(rows)


def density_matched_graph_null(
    source_edges: Iterable[tuple[int, int]],
    target_edges: Iterable[tuple[int, int]],
    *,
    n_features: int,
    replicates: int,
    rng_seed: int,
) -> pd.DataFrame:
    """Null Jaccard for random undirected graphs with matched edge counts.

    This controls the trivial fact that two dense graphs overlap more often
    than two sparse graphs, without using feature identities or outcomes.
    """

    source_edges = set(source_edges)
    target_edges = set(target_edges)
    all_edges = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    source_count = min(len(source_edges), len(all_edges))
    target_count = min(len(target_edges), len(all_edges))
    for replicate in range(replicates):
        source_idx = rng.choice(len(all_edges), size=source_count, replace=False) if source_count else []
        target_idx = rng.choice(len(all_edges), size=target_count, replace=False) if target_count else []
        random_source = {all_edges[int(i)] for i in np.atleast_1d(source_idx)}
        random_target = {all_edges[int(i)] for i in np.atleast_1d(target_idx)}
        rows.append(
            {
                "null_type": "density_matched_random_graph",
                "replicate": int(replicate),
                "jaccard": jaccard(random_source, random_target),
                "weighted_overlap": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def null_summary(observed_jaccard: float, null: pd.DataFrame) -> dict[str, object]:
    values = null["jaccard"].to_numpy(dtype=np.float64)
    mean = float(np.mean(values)) if len(values) else float("nan")
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    q95 = float(np.quantile(values, 0.95)) if len(values) else float("nan")
    empirical_p = float((1.0 + np.sum(values >= observed_jaccard)) / (len(values) + 1.0)) if len(values) else float("nan")
    effect = float((observed_jaccard - mean) / (std + EPSILON)) if np.isfinite(std) else float("nan")
    return {
        "observed_jaccard": float(observed_jaccard),
        "null_mean": mean,
        "null_std": std,
        "null_q95": q95,
        "empirical_p": empirical_p,
        "effect_size": effect,
    }


def partial_correlation_agreement(
    source_precision: np.ndarray,
    target_precision: np.ndarray,
    shared_edges: Iterable[tuple[int, int]],
) -> dict[str, object]:
    edges = list(shared_edges)
    if not edges:
        return {
            "shared_partial_corr_pearson": float("nan"),
            "shared_partial_corr_spearman": float("nan"),
            "shared_partial_corr_sign_agreement": float("nan"),
            "shared_partial_corr_median_abs_diff": float("nan"),
        }
    source_rho = np.asarray([partial_correlation(source_precision, i, j) for i, j in edges], dtype=np.float64)
    target_rho = np.asarray([partial_correlation(target_precision, i, j) for i, j in edges], dtype=np.float64)
    if len(edges) < 2 or np.std(source_rho) <= EPSILON or np.std(target_rho) <= EPSILON:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(source_rho, target_rho)[0, 1])
        spearman = float(stats.spearmanr(source_rho, target_rho).statistic)
    signs = np.sign(source_rho) == np.sign(target_rho)
    return {
        "shared_partial_corr_pearson": pearson,
        "shared_partial_corr_spearman": spearman,
        "shared_partial_corr_sign_agreement": float(np.mean(signs)),
        "shared_partial_corr_median_abs_diff": float(np.median(np.abs(source_rho - target_rho))),
    }


def partial_correlation(precision: np.ndarray, i: int, j: int) -> float:
    denom = np.sqrt(max(float(precision[i, i] * precision[j, j]), EPSILON))
    return float(-precision[i, j] / denom)


def precision_edge_mask(
    precision: np.ndarray,
    *,
    edge_abs_threshold: float,
) -> np.ndarray:
    values = np.abs(np.asarray(precision, dtype=np.float64)) > float(edge_abs_threshold)
    np.fill_diagonal(values, False)
    return values


def edge_set(probabilities: np.ndarray, threshold: float) -> frozenset[tuple[int, int]]:
    probs = np.asarray(probabilities, dtype=np.float64)
    edges = [
        (int(i), int(j))
        for i in range(probs.shape[0])
        for j in range(i + 1, probs.shape[1])
        if probs[i, j] >= threshold
    ]
    return frozenset(edges)


def edge_frame(
    source_pair_id: str,
    n: int,
    seed: int,
    source_result: PrecisionStabilityResult,
    target_result: PrecisionStabilityResult,
) -> pd.DataFrame:
    source_edges = set(source_result.stable_edges)
    target_edges = set(target_result.stable_edges)
    rows = []
    for i, j in sorted(source_edges | target_edges):
        rows.append(
            {
                "N": int(n),
                "seed": int(seed),
                "source_pair_id": source_pair_id,
                "feature_i": int(i),
                "feature_j": int(j),
                "source_edge_probability": float(source_result.edge_probabilities[i, j]),
                "target_edge_probability": float(target_result.edge_probabilities[i, j]),
                "source_stable": bool((i, j) in source_edges),
                "target_stable": bool((i, j) in target_edges),
                "shared_stable": bool((i, j) in source_edges and (i, j) in target_edges),
                "source_partial_correlation": partial_correlation(source_result.precision, i, j),
                "target_partial_correlation": partial_correlation(target_result.precision, i, j),
            }
        )
    return pd.DataFrame(rows)


def decide_p0(
    robotics_summary: pd.DataFrame,
    synthetic_sanity: pd.DataFrame,
    *,
    completeness: dict[str, object] | None = None,
    min_target_stable_edges: int = 5,
    min_effect_size: float = 1.0,
    max_empirical_p: float = 0.10,
    max_density_empirical_p: float = 0.10,
    max_differential_ratio: float = 0.50,
    max_fragile_regularization_fraction: float = 0.25,
    min_synthetic_pass_rate: float = 0.80,
    min_partial_sign_agreement: float = 0.60,
    min_row_consistency_fraction: float = 0.80,
    min_consistent_n_values: int = 2,
) -> dict[str, object]:
    """Prospective P0 rule.

    A final PASS requires:
      * complete replication;
      * synthetic identifiability;
      * numerically nontrivial target graph;
      * overlap above both nulls;
      * sparse differential structure;
      * regularization robustness;
      * partial-correlation sign agreement;
      * row-level consistency across multiple commissioning N values.

    The consistency rule prevents a strong global median from hiding failure in
    many seeds/source regimes.
    """

    if completeness is not None and not bool(completeness.get("complete", False)):
        return {
            "decision": "P0_INCOMPLETE_REPLICATION",
            "reason": "the full N/seed/source-regime replication matrix is incomplete",
            "completeness": completeness,
            "do_not_implement_precision_race": True,
        }

    if robotics_summary.empty:
        return {
            "decision": "P0_INCONCLUSIVE_MORE_HEALTHY_DATA_REQUIRED",
            "reason": "no robotics P0 rows were available",
            "do_not_implement_precision_race": True,
        }

    if "classified_correctly" in synthetic_sanity.columns:
        pass_rate = float(np.mean(synthetic_sanity["classified_correctly"].astype(bool)))
        synthetic_metric = "classified_correctly"
    elif "expectation_pass" in synthetic_sanity.columns:
        pass_rate = float(np.mean(synthetic_sanity["expectation_pass"].astype(bool)))
        synthetic_metric = "expectation_pass"
    else:
        pass_rate = 0.0
        synthetic_metric = "missing"

    if pass_rate < min_synthetic_pass_rate:
        return {
            "decision": "P0_FAIL_SYNTHETIC_IDENTIFIABILITY",
            "reason": "synthetic calibration did not distinguish the required graph-sharing scenarios",
            "synthetic_pass_rate": pass_rate,
            "synthetic_metric": synthetic_metric,
            "do_not_implement_precision_race": True,
        }

    target_edges_median = float(robotics_summary["target_stable_edges"].median())
    if target_edges_median < min_target_stable_edges:
        return {
            "decision": "P0_INCONCLUSIVE_MORE_HEALTHY_DATA_REQUIRED",
            "reason": "target stable graph has too few stable edges for reliable structural interpretation",
            "target_stable_edges_median": target_edges_median,
            "do_not_implement_precision_race": True,
        }

    # Numerical reliability, if available, is a hard prerequisite.
    if "numerically_reliable" in robotics_summary.columns:
        reliable_fraction = float(np.mean(robotics_summary["numerically_reliable"].astype(bool)))
        if reliable_fraction < min_row_consistency_fraction:
            return {
                "decision": "P0_FAIL_REGULARIZATION_FRAGILE",
                "reason": "too many rows are numerically unreliable under the selected precision regularization",
                "numerically_reliable_fraction": reliable_fraction,
                "do_not_implement_precision_race": True,
            }

    effect_median = float(robotics_summary["effect_size"].median())
    p_median = float(robotics_summary["empirical_p"].median())
    density_p_median = float(robotics_summary["density_empirical_p"].median()) if "density_empirical_p" in robotics_summary else np.nan

    differential_median = float(robotics_summary["differential_ratio"].median())
    if differential_median > max_differential_ratio:
        return {
            "decision": "P0_FAIL_DIFFERENTIAL_NOT_SPARSE",
            "reason": "stable differential structure is too large for a shared-plus-sparse-difference assumption",
            "differential_ratio_median": differential_median,
            "do_not_implement_precision_race": True,
        }

    if "regularization_fragile" in robotics_summary:
        fragile_fraction = float(np.mean(robotics_summary["regularization_fragile"].astype(bool)))
        if fragile_fraction > max_fragile_regularization_fraction:
            return {
                "decision": "P0_FAIL_REGULARIZATION_FRAGILE",
                "reason": "shared-precision evidence is too sensitive to the frozen regularization neighborhood",
                "regularization_fragile_fraction": fragile_fraction,
                "do_not_implement_precision_race": True,
            }

    if "shared_partial_corr_sign_agreement" in robotics_summary:
        sign_agreement = float(robotics_summary["shared_partial_corr_sign_agreement"].median())
        if np.isfinite(sign_agreement) and sign_agreement < min_partial_sign_agreement:
            return {
                "decision": "P0_FAIL_SOURCE_TARGET_STRUCTURE_UNRELATED",
                "reason": "shared stable edges do not have sufficient partial-correlation sign agreement",
                "shared_partial_corr_sign_agreement_median": sign_agreement,
                "do_not_implement_precision_race": True,
            }
    else:
        sign_agreement = float("nan")

    # Per-row rule.
    row_pass = (
        (robotics_summary["target_stable_edges"].astype(float) >= min_target_stable_edges)
        & (robotics_summary["effect_size"].astype(float) >= min_effect_size)
        & (robotics_summary["empirical_p"].astype(float) <= max_empirical_p)
        & (robotics_summary["differential_ratio"].astype(float) <= max_differential_ratio)
    )
    if "density_empirical_p" in robotics_summary.columns:
        row_pass &= robotics_summary["density_empirical_p"].astype(float) <= max_density_empirical_p
    if "regularization_fragile" in robotics_summary.columns:
        row_pass &= ~robotics_summary["regularization_fragile"].astype(bool)
    if "shared_partial_corr_sign_agreement" in robotics_summary.columns:
        sign_ok = (
            ~np.isfinite(robotics_summary["shared_partial_corr_sign_agreement"].astype(float))
            | (robotics_summary["shared_partial_corr_sign_agreement"].astype(float) >= min_partial_sign_agreement)
        )
        row_pass &= sign_ok

    audit = robotics_summary.copy()
    audit["_row_pass"] = row_pass.astype(bool)

    per_n_consistency: dict[str, float] = {}
    consistent_n = 0
    if "N" in audit.columns:
        for n_value, group in audit.groupby("N", sort=True):
            frac = float(np.mean(group["_row_pass"].astype(bool)))
            per_n_consistency[str(int(n_value))] = frac
            if frac >= min_row_consistency_fraction:
                consistent_n += 1
    else:
        frac = float(np.mean(audit["_row_pass"].astype(bool)))
        per_n_consistency["all"] = frac
        consistent_n = int(frac >= min_row_consistency_fraction)

    if consistent_n < min_consistent_n_values:
        return {
            "decision": "P0_FAIL_SOURCE_TARGET_STRUCTURE_UNRELATED",
            "reason": "shared-precision evidence is not consistent across enough commissioning N values/seeds/source regimes",
            "per_N_row_pass_fraction": per_n_consistency,
            "required_fraction": min_row_consistency_fraction,
            "required_consistent_N_values": min_consistent_n_values,
            "do_not_implement_precision_race": True,
        }

    above_null = bool(
        effect_median >= min_effect_size
        and p_median <= max_empirical_p
        and (not np.isfinite(density_p_median) or density_p_median <= max_density_empirical_p)
    )
    if not above_null:
        return {
            "decision": "P0_FAIL_SOURCE_TARGET_STRUCTURE_UNRELATED",
            "reason": "observed source-target graph overlap does not clear the frozen null-control rule",
            "effect_size_median": effect_median,
            "empirical_p_median": p_median,
            "density_empirical_p_median": density_p_median,
            "do_not_implement_precision_race": True,
        }

    return {
        "decision": "P0_PASS_SHARED_PRECISION_STRUCTURE",
        "reason": "complete healthy-only replication shows numerically stable, null-separated, sparse-differential shared precision structure with multi-N consistency",
        "synthetic_pass_rate": pass_rate,
        "synthetic_metric": synthetic_metric,
        "target_stable_edges_median": target_edges_median,
        "effect_size_median": effect_median,
        "empirical_p_median": p_median,
        "density_empirical_p_median": density_p_median,
        "differential_ratio_median": differential_median,
        "shared_partial_corr_sign_agreement_median": sign_agreement,
        "per_N_row_pass_fraction": per_n_consistency,
        "consistent_N_values": consistent_n,
        "do_not_implement_precision_race": False,
    }


def synthetic_sanity_checks(
    *,
    n_values: Sequence[int] = (10, 25, 50, 100),
    dimension: int = 40,
    source_samples: int = 120,
    config: StabilityConfig | None = None,
    rng_seed: int = 123,
) -> pd.DataFrame:
    """Run lightweight Gaussian graphical-model cases before robotics P0."""

    cfg = config or StabilityConfig(
        alpha_grid=(0.05, 0.10, 0.20, 0.40),
        resamples=8,
        stable_edge_threshold=0.60,
        max_iter=100,
    )
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    base = make_chain_precision(dimension, weight=0.30)
    partial = alter_precision_edges(base, fraction=0.25, rng=rng)
    unrelated = make_random_sparse_precision(dimension, edge_probability=0.04, rng=rng)
    dense_diff = alter_precision_edges(base, fraction=0.70, rng=rng)
    cases = [
        ("A_identical_sparse_graph", base, base, "high_overlap_consistent_partial_correlation"),
        ("B_partially_shared_graph", base, partial, "intermediate_overlap"),
        ("C_unrelated_graphs", base, unrelated, "near_null_overlap"),
        ("E_dense_target_specific_differential", base, dense_diff, "high_differential"),
    ]
    for case_name, source_precision, target_precision, expectation in cases:
        n = max(n_values)
        rows.append(
            _synthetic_case_row(case_name, source_precision, target_precision, n, source_samples, cfg, rng, expectation)
        )
    for n in n_values:
        rows.append(
            _synthetic_case_row(
                f"D_identical_graph_N{n}",
                base,
                base,
                int(n),
                source_samples,
                cfg,
                rng,
                "target_stability_improves_with_N",
            )
        )
    frame = pd.DataFrame(rows)
    d_rows = frame[frame["case"].str.startswith("D_identical_graph_N")].sort_values("N")
    if len(d_rows) >= 2:
        first_overlap = float(d_rows.iloc[0]["stable_jaccard"])
        last_overlap = float(d_rows.iloc[-1]["stable_jaccard"])
        frame.loc[frame["case"].str.startswith("D_identical_graph_N"), "expectation_pass"] = last_overlap >= first_overlap
    return frame


def high_dimensional_synthetic_stress(
    *,
    dimensions: Sequence[int],
    n_values: Sequence[int] = (10, 25, 50, 100),
    source_samples: int = 120,
    config: StabilityConfig | None = None,
    rng_seed: int = 456,
    null_replicates: int = 20,
    full_dimension: int | None = None,
    full_grid_max_dimension: int = 128,
) -> pd.DataFrame:
    """Run P0-identifiability stress tests across realistic p/N regimes."""

    cfg = config or StabilityConfig(
        alpha_grid=(0.05, 0.10, 0.20, 0.40),
        resamples=6,
        stable_edge_threshold=0.60,
        max_iter=100,
    )
    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, object]] = []
    dims = tuple(dict.fromkeys(int(v) for v in dimensions if int(v) >= 2))
    for p in dims:
        base = make_random_sparse_precision(p, edge_probability=min(0.04, 4.0 / max(p, 2)), rng=rng)
        high_shared = alter_precision_edges(base, fraction=0.10, rng=rng)
        partial = alter_precision_edges(base, fraction=0.45, rng=rng)
        unrelated = make_random_sparse_precision(p, edge_probability=true_edge_density(base), rng=rng)
        dense_diff = alter_precision_edges(base, fraction=0.80, rng=rng)
        weight_perturbed = perturb_precision_weights(base, scale=0.35, rng=rng)
        density_null = make_random_sparse_precision(p, edge_probability=true_edge_density(base), rng=rng)
        scenarios = [
            ("identical_sparse_precision_graphs", base, base, "shared"),
            ("highly_shared_sparse_differential", base, high_shared, "shared"),
            ("partially_shared_graph", base, partial, "partial"),
            ("unrelated_density_comparable_graphs", base, unrelated, "unrelated"),
            ("shared_base_dense_target_differential", base, dense_diff, "partial"),
            ("same_support_perturbed_weights", base, weight_perturbed, "shared"),
            ("graph_density_matched_null_case", base, density_null, "unrelated"),
        ]
        for n in n_values:
            # Keep at least one full-p stress test; avoid silently launching an
            # intractable full Cartesian product under the same estimator.
            if p > int(full_grid_max_dimension) and int(n) not in (min(n_values), max(n_values)):
                continue
            for case_name, source_precision, target_precision, expected_class in scenarios:
                row = _synthetic_case_row(
                    case_name,
                    source_precision,
                    target_precision,
                    int(n),
                    source_samples,
                    cfg,
                    rng,
                    expected_class,
                    null_replicates=null_replicates,
                )
                true_metrics = true_graph_relationship(source_precision, target_precision)
                row.update(
                    {
                        "dimension": int(p),
                        "true_shared_edge_fraction": true_metrics["true_shared_edge_fraction"],
                        "true_differential_fraction": true_metrics["true_differential_fraction"],
                        "expected_class": expected_class,
                        "classified_class": classify_synthetic_result(row),
                    }
                )
                row["classified_correctly"] = bool(row["classified_class"] == expected_class)
                rows.append(row)
    return pd.DataFrame(rows)


def make_chain_precision(dimension: int, *, weight: float) -> np.ndarray:
    precision = np.eye(dimension)
    for i in range(dimension - 1):
        precision[i, i + 1] = weight
        precision[i + 1, i] = weight
    return _make_positive_definite(precision)


def make_random_sparse_precision(
    dimension: int,
    *,
    edge_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    precision = np.eye(dimension)
    for i, j in combinations(range(dimension), 2):
        if rng.random() < edge_probability:
            value = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.35)
            precision[i, j] = value
            precision[j, i] = value
    return _make_positive_definite(precision)


def alter_precision_edges(
    precision: np.ndarray,
    *,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(precision, dtype=np.float64, copy=True)
    p = out.shape[0]
    existing = [(i, j) for i, j in combinations(range(p), 2) if abs(out[i, j]) > EPSILON]
    change_count = max(1, int(round(len(existing) * fraction)))
    chosen = rng.choice(len(existing), size=min(change_count, len(existing)), replace=False)
    for edge_index in np.atleast_1d(chosen):
        i, j = existing[int(edge_index)]
        out[i, j] = 0.0
        out[j, i] = 0.0
    while change_count > 0:
        i, j = sorted(rng.choice(p, size=2, replace=False))
        if abs(out[i, j]) <= EPSILON:
            value = rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.35)
            out[i, j] = value
            out[j, i] = value
            change_count -= 1
    return _make_positive_definite(out)


def perturb_precision_weights(
    precision: np.ndarray,
    *,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.array(precision, dtype=np.float64, copy=True)
    for i, j in combinations(range(out.shape[0]), 2):
        if abs(out[i, j]) > EPSILON:
            out[i, j] *= float(1.0 + rng.normal(0.0, scale))
            out[j, i] = out[i, j]
    return _make_positive_definite(out)


def true_edge_density(precision: np.ndarray) -> float:
    return graph_density(len(edge_set(np.abs(precision) > EPSILON, 0.5)), precision.shape[0])


def true_graph_relationship(source_precision: np.ndarray, target_precision: np.ndarray) -> dict[str, float]:
    source_edges = edge_set(np.abs(source_precision) > EPSILON, 0.5)
    target_edges = edge_set(np.abs(target_precision) > EPSILON, 0.5)
    union = source_edges | target_edges
    return {
        "true_shared_edge_fraction": float(1.0 if not union else len(source_edges & target_edges) / len(union)),
        "true_differential_fraction": float(0.0 if not union else len(union - (source_edges & target_edges)) / len(union)),
    }


def classify_synthetic_result(metrics: dict[str, object]) -> str:
    j = float(metrics.get("stable_jaccard", np.nan))
    p = float(metrics.get("empirical_p", np.nan))
    diff = float(metrics.get("differential_ratio", np.nan))
    effect = float(metrics.get("effect_size", np.nan))
    if np.isfinite(j) and np.isfinite(p) and np.isfinite(effect) and p <= 0.10 and effect >= 1.0 and diff <= 0.50:
        return "shared"
    if np.isfinite(p) and p > 0.10:
        return "unrelated"
    return "partial"


def graph_density(edge_count: int, n_features: int) -> float:
    possible = n_features * (n_features - 1) / 2.0
    return float(edge_count / possible) if possible > 0 else 0.0


def jaccard(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> float:
    """Jaccard index |A intersection B| / |A union B| for undirected edges.

    Empty-vs-empty is defined as 1.0 because two empty stable supports are
    exactly equal. Callers that require nonempty graph evidence must gate on
    edge counts separately.
    """
    a = set(left)
    b = set(right)
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _synthetic_case_row(
    case_name: str,
    source_precision: np.ndarray,
    target_precision: np.ndarray,
    n: int,
    source_samples: int,
    config: StabilityConfig,
    rng: np.random.Generator,
    expectation: str,
    null_replicates: int = 20,
) -> dict[str, object]:
    source_cov = np.linalg.inv(source_precision)
    target_cov = np.linalg.inv(target_precision)
    source = rng.multivariate_normal(np.zeros(source_precision.shape[0]), source_cov, size=source_samples)
    target = rng.multivariate_normal(np.zeros(target_precision.shape[0]), target_cov, size=n)
    source_scaled, target_scaled, _ = robust_target_scale(source, target)
    seed_base = int(rng.integers(0, 2**31 - 1))
    source_result = estimate_precision_stability(source_scaled, config=config, rng_seed=seed_base, prefix="synthetic_source")
    target_result = estimate_precision_stability(target_scaled, config=config, rng_seed=seed_base + 1, prefix="synthetic_target")
    metrics = compare_precision_structures(source_result, target_result)
    metrics["source_graph_stability"] = float(np.mean(_upper_values(source_result.edge_probabilities)))
    metrics["target_graph_stability"] = float(np.mean(_upper_values(target_result.edge_probabilities)))
    null = feature_permutation_null(
        source_result.edge_probabilities,
        target_result.edge_probabilities,
        stable_edge_threshold=config.stable_edge_threshold,
        replicates=null_replicates,
        rng_seed=seed_base + 2,
    )
    metrics.update(null_summary(float(metrics["stable_jaccard"]), null))
    expectation_pass = _synthetic_expectation_pass(expectation, metrics)
    return {
        "case": case_name,
        "N": int(n),
        "expectation": expectation,
        "expectation_pass": bool(expectation_pass),
        **metrics,
    }


def _synthetic_expectation_pass(expectation: str, metrics: dict[str, object]) -> bool:
    j = float(metrics["stable_jaccard"])
    effect = float(metrics["effect_size"])
    diff = float(metrics["differential_ratio"])
    empirical_p = float(metrics.get("empirical_p", np.nan))
    if expectation == "shared":
        return j >= 0.20 and diff <= 0.50 and (not np.isfinite(empirical_p) or empirical_p <= 0.25)
    if expectation == "partial":
        return 0.01 < j < 0.95
    if expectation == "unrelated":
        return j <= float(metrics["null_q95"]) + EPSILON or (np.isfinite(empirical_p) and empirical_p > 0.10)
    if expectation == "high_overlap_consistent_partial_correlation":
        sign_agreement = float(metrics.get("shared_partial_corr_sign_agreement", np.nan))
        median_abs_diff = float(metrics.get("shared_partial_corr_median_abs_diff", np.nan))
        empirical_p = float(metrics.get("empirical_p", np.nan))
        significant_overlap = effect > 0.0 and (not np.isfinite(empirical_p) or empirical_p <= 0.25)
        consistent_shared_weights = (
            np.isfinite(sign_agreement)
            and sign_agreement >= 0.70
            and np.isfinite(median_abs_diff)
            and median_abs_diff <= 0.10
        )
        return j > 0.20 and significant_overlap and consistent_shared_weights
    if expectation == "high_overlap_low_differential":
        return j > 0.20 and diff <= 0.50 and effect > 0.0
    if expectation == "intermediate_overlap":
        return 0.02 < j < 0.95
    if expectation == "near_null_overlap":
        return j <= float(metrics["null_q95"]) + EPSILON
    if expectation == "high_differential":
        return diff >= 0.10
    if expectation == "target_stability_improves_with_N":
        return True
    return False


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _validate_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if matrix.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two features.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return matrix


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2.0


def _make_positive_definite(matrix: np.ndarray) -> np.ndarray:
    out = _symmetrize(matrix)
    eigvals = np.linalg.eigvalsh(out)
    min_eig = float(np.min(eigvals))
    if min_eig <= 0.05:
        out += np.eye(out.shape[0]) * (0.05 - min_eig)
    return out
