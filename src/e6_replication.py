from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.paired_inference import summarize_paired_deltas


REQUIRED_VARIANTS = ("M2N2-Offline", "M2N2-CommissioningAdapt")
REQUIRED_AGGREGATIONS = ("q99", "mean")
EXPECTED_SEEDS = tuple(range(20))
EXPECTED_N = 100
PRIMARY_METRICS = (
    "recall",
    "false_positive_rate",
    "auroc",
    "oracle_recall_at_fpr_budget",
)


@dataclass(frozen=True)
class BootstrapSummary:
    n: int
    mean: float
    std: float
    ci_lower: float
    ci_upper: float


def bootstrap_mean_summary(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 42,
) -> BootstrapSummary:
    x = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("bootstrap summary requires at least two one-dimensional observations")
    if not np.isfinite(x).all():
        raise ValueError("bootstrap values contain NaN or Inf")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, x.size, size=(int(repetitions), x.size))
    means = x[idx].mean(axis=1)
    alpha = 1.0 - float(confidence)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapSummary(
        n=int(x.size),
        mean=float(x.mean()),
        std=float(x.std(ddof=1)),
        ci_lower=float(lo),
        ci_upper=float(hi),
    )


def validate_e6r1_design(df: pd.DataFrame, *, strict: bool = True) -> None:
    required = {
        "cycle_score_aggregation",
        "variant",
        "commissioning_size",
        "seed",
        *PRIMARY_METRICS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"E6-R1 results missing required columns: {missing}")
    if df.duplicated(["cycle_score_aggregation", "variant", "commissioning_size", "seed"]).any():
        raise ValueError("duplicate E6-R1 aggregation/variant/N/seed rows detected")
    for col in PRIMARY_METRICS:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{col} contains NaN or Inf")

    if not strict:
        return

    actual_aggs = tuple(sorted(df["cycle_score_aggregation"].astype(str).unique()))
    if actual_aggs != tuple(sorted(REQUIRED_AGGREGATIONS)):
        raise ValueError(f"expected aggregations {REQUIRED_AGGREGATIONS}, found {actual_aggs}")
    actual_n = tuple(sorted(int(v) for v in df["commissioning_size"].unique()))
    if actual_n != (EXPECTED_N,):
        raise ValueError(f"expected commissioning size {EXPECTED_N}, found {actual_n}")

    for aggregation in REQUIRED_AGGREGATIONS:
        for variant in REQUIRED_VARIANTS:
            g = df[(df.cycle_score_aggregation == aggregation) & (df.variant == variant)]
            seeds = tuple(sorted(int(v) for v in g.seed.unique()))
            if seeds != EXPECTED_SEEDS:
                raise ValueError(
                    f"{aggregation}/{variant} expected seeds 0..19, found {seeds}"
                )


def metric_summary_table(
    df: pd.DataFrame,
    *,
    confidence: float = 0.95,
    bootstrap_repetitions: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["cycle_score_aggregation", "variant", "commissioning_size"], sort=True)
    for group_index, ((aggregation, variant, n_value), g) in enumerate(grouped):
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            summary = bootstrap_mean_summary(
                g[metric].to_numpy(dtype=float),
                confidence=confidence,
                repetitions=bootstrap_repetitions,
                seed=int(seed) + 100 * group_index + metric_index,
            )
            rows.append({
                "cycle_score_aggregation": str(aggregation),
                "variant": str(variant),
                "commissioning_size": int(n_value),
                "metric": metric,
                "n_runs": summary.n,
                "mean": summary.mean,
                "std": summary.std,
                "bootstrap_ci_lower": summary.ci_lower,
                "bootstrap_ci_upper": summary.ci_upper,
            })
    return pd.DataFrame(rows)


def paired_adaptation_table(
    df: pd.DataFrame,
    *,
    confidence: float = 0.95,
    bootstrap_repetitions: int = 10_000,
    signflip_repetitions: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for agg_index, aggregation in enumerate(sorted(df.cycle_score_aggregation.astype(str).unique())):
        g = df[df.cycle_score_aggregation == aggregation]
        offline = g[g.variant == "M2N2-Offline"].set_index("seed")
        adapted = g[g.variant == "M2N2-CommissioningAdapt"].set_index("seed")
        common = sorted(set(offline.index) & set(adapted.index))
        if len(common) < 2:
            raise ValueError(f"insufficient paired seeds for aggregation {aggregation}")
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            delta = (
                adapted.loc[common, metric].to_numpy(dtype=float)
                - offline.loc[common, metric].to_numpy(dtype=float)
            )
            result = summarize_paired_deltas(
                delta,
                confidence=confidence,
                bootstrap_repetitions=bootstrap_repetitions,
                signflip_repetitions=signflip_repetitions,
                seed=int(seed) + 100 * agg_index + metric_index,
            )
            row = {
                "cycle_score_aggregation": aggregation,
                "commissioning_size": EXPECTED_N,
                "metric": metric,
                "delta_definition": "M2N2-CommissioningAdapt minus M2N2-Offline",
                **result.to_dict(),
                "positive_rate": result.positive_count / result.n_pairs,
                "negative_rate": result.negative_count / result.n_pairs,
                "neutral_rate": result.neutral_count / result.n_pairs,
            }
            rows.append(row)
    return pd.DataFrame(rows)
