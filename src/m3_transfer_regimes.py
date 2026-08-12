"""M3 transferability-regime utilities.

This module is intentionally detector-light: it defines source regimes and
paired analysis without tuning detector hyperparameters on anomaly outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.a0_transferability import healthy_transferability_metrics, robust_center_scale


TRANSFERABILITY_COLUMNS = (
    "mean_shift_distance",
    "standardized_mean_shift",
    "covariance_discrepancy",
    "projector_discrepancy",
    "projector_similarity",
    "mmd_rbf",
    "wasserstein_diag",
)


@dataclass(frozen=True)
class SourceRegime:
    source_pair_id: str
    source_group: str
    target_group: str
    source_episode_ids: tuple[int, ...]
    target_episode_ids: tuple[int, ...]
    metrics: dict[str, float]


def episode_ids(cycles: Sequence[object]) -> tuple[int, ...]:
    return tuple(int(getattr(cycle, "episode_id")) for cycle in cycles)


def assert_no_episode_leakage(groups: dict[str, Iterable[int]]) -> None:
    sets = {name: set(int(v) for v in values) for name, values in groups.items()}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise RuntimeError(
                    f"M3 leakage: {left} overlaps {right}: {sorted(overlap)[:10]}"
                )


def _robust_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    center, scale = robust_center_scale(target)
    z = np.clip((np.asarray(source, dtype=np.float64) - center) / scale, -8.0, 8.0)
    return np.linalg.norm(z, axis=1) / np.sqrt(z.shape[1])


def construct_source_regimes(
    *,
    source_episode_ids: Sequence[int],
    source_features: np.ndarray,
    target_episode_ids: Sequence[int],
    target_features: np.ndarray,
    commissioning_size: int,
    seed: int,
    subset_size: int = 100,
) -> list[SourceRegime]:
    """Construct near/moderate/high source regimes from healthy geometry only."""
    ids = np.asarray(source_episode_ids, dtype=np.int64)
    source = np.asarray(source_features, dtype=np.float64)
    target = np.asarray(target_features, dtype=np.float64)
    if source.shape[0] != ids.shape[0]:
        raise ValueError("source_episode_ids and source_features length mismatch.")
    if source.shape[0] < 3:
        raise ValueError("At least three source episodes are required.")

    distances = _robust_distances(source, target)
    order = np.argsort(distances, kind="mergesort")
    n = len(order)
    size = min(int(subset_size), max(1, n // 3))
    windows = {
        "near": order[:size],
        "moderate": order[max(0, (n - size) // 2) : max(0, (n - size) // 2) + size],
        "high": order[-size:],
    }
    regimes: list[SourceRegime] = []
    target_ids = tuple(int(v) for v in target_episode_ids)
    target_group = f"target_setting_73_N{commissioning_size}_seed{seed}"
    for name, idx in windows.items():
        selected_ids = tuple(int(v) for v in ids[idx])
        selected_features = source[idx]
        metrics = healthy_transferability_metrics(selected_features, target)
        regimes.append(
            SourceRegime(
                source_pair_id=f"{name}_shift_N{commissioning_size}_seed{seed}",
                source_group=f"{name}_shift_source_setting_72",
                target_group=target_group,
                source_episode_ids=selected_ids,
                target_episode_ids=target_ids,
                metrics=metrics,
            )
        )
    return regimes


def bootstrap_ci(
    values: Sequence[float],
    *,
    rng_seed: int = 42,
    resamples: int = 2000,
    statistic: str = "mean",
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(rng_seed)
    samples = rng.choice(arr, size=(resamples, arr.size), replace=True)
    if statistic == "median":
        stats = np.median(samples, axis=1)
    else:
        stats = np.mean(samples, axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def add_transferability_regimes(
    frame: pd.DataFrame,
    *,
    metric: str = "standardized_mean_shift",
) -> pd.DataFrame:
    out = frame.copy()
    pairs = out[["source_pair_id", metric]].drop_duplicates()
    quantiles = pairs[metric].quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()

    def label(value: float) -> str:
        if value <= quantiles[0]:
            return "low_shift"
        if value <= quantiles[1]:
            return "moderate_shift"
        return "high_shift"

    mapping = {row.source_pair_id: label(float(getattr(row, metric))) for row in pairs.itertuples()}
    out["transferability_regime"] = out["source_pair_id"].map(mapping)
    return out


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["source_pair_id", "source_group", "target_group", "commissioning_size", "seed"]
    metrics = ["recall", "fpr", "auroc", "auprc", "success"]
    race = results[results["detector"] == "RACE"].set_index(keys)
    target = results[results["detector"] == "TargetOnly"].set_index(keys)
    rows: list[dict[str, object]] = []
    common_index = race.index.intersection(target.index)
    for key in common_index:
        r = race.loc[key]
        t = target.loc[key]
        row = dict(zip(keys, key))
        for metric in metrics:
            row[f"delta_{metric}"] = float(r[metric] - t[metric])
        for column in TRANSFERABILITY_COLUMNS:
            row[column] = float(r[column])
        row["transfer_weight"] = float(r.get("transfer_weight", np.nan))
        row["source_permutation_recall"] = _control_value(results, key, "SourcePermutation", "recall", keys)
        row["weight_permutation_recall"] = _control_value(results, key, "WeightPermutation", "recall", keys)
        rows.append(row)
    return pd.DataFrame(rows)


def _control_value(
    results: pd.DataFrame,
    key: tuple[object, ...],
    detector: str,
    metric: str,
    keys: list[str],
) -> float:
    mask = results["detector"].eq(detector)
    for column, value in zip(keys, key):
        mask &= results[column].eq(value)
    rows = results.loc[mask, metric]
    return float(rows.iloc[0]) if len(rows) else float("nan")


def detector_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings = [
        (["detector", "commissioning_size"], "all"),
        (["detector", "commissioning_size", "transferability_regime"], None),
    ]
    for group_cols, forced_regime in groupings:
        grouped = results.groupby(group_cols, dropna=False, sort=True)
        for key, group in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(group_cols, key_tuple))
            if forced_regime is not None:
                row["transferability_regime"] = forced_regime
            row["valid_runs"] = int(len(group))
            for metric in ["recall", "fpr", "auroc", "auprc"]:
                values = group[metric].to_numpy(dtype=np.float64)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                lo, hi = bootstrap_ci(values)
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
            success = group["success"].to_numpy(dtype=np.float64)
            row["success_rate"] = float(np.nanmean(success))
            lo, hi = bootstrap_ci(success)
            row["success_ci_low"] = lo
            row["success_ci_high"] = hi
            rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings = [
        (["commissioning_size"], "all"),
        (["commissioning_size", "transferability_regime"], None),
    ]
    for group_cols, forced_regime in groupings:
        for key, group in deltas.groupby(group_cols, dropna=False, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = dict(zip(group_cols, key_tuple))
            if forced_regime is not None:
                row["transferability_regime"] = forced_regime
            row["valid_pairs"] = int(len(group))
            for metric in ["delta_recall", "delta_fpr", "delta_auroc", "delta_auprc", "delta_success"]:
                values = group[metric].to_numpy(dtype=np.float64)
                row[f"{metric}_mean"] = float(np.nanmean(values))
                row[f"{metric}_median"] = float(np.nanmedian(values))
                lo, hi = bootstrap_ci(values)
                row[f"{metric}_ci_low"] = lo
                row[f"{metric}_ci_high"] = hi
                row[f"{metric}_prop_positive"] = float(np.nanmean(values > 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for transfer_metric in TRANSFERABILITY_COLUMNS:
        for benefit_metric in ["delta_recall", "delta_fpr", "delta_auroc", "delta_auprc", "delta_success"]:
            rows.append(_correlation_row(deltas, transfer_metric, benefit_metric, level="run"))
            pair_level = (
                deltas.groupby(["source_pair_id", "commissioning_size"], as_index=False)[
                    [transfer_metric, benefit_metric]
                ]
                .mean(numeric_only=True)
            )
            rows.append(_correlation_row(pair_level, transfer_metric, benefit_metric, level="pair_N"))
    return pd.DataFrame(rows)


def _correlation_row(frame: pd.DataFrame, x_col: str, y_col: str, *, level: str) -> dict[str, object]:
    clean = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    row: dict[str, object] = {
        "level": level,
        "transferability_metric": x_col,
        "benefit_metric": y_col,
        "n": int(len(clean)),
    }
    if len(clean) < 3 or clean[x_col].nunique() < 2 or clean[y_col].nunique() < 2:
        row.update({"pearson": np.nan, "spearman": np.nan, "pearson_ci_low": np.nan, "pearson_ci_high": np.nan})
        return row
    x = clean[x_col].to_numpy(dtype=np.float64)
    y = clean[y_col].to_numpy(dtype=np.float64)
    row["pearson"] = float(np.corrcoef(x, y)[0, 1])
    row["spearman"] = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    rng = np.random.default_rng(42)
    boot: list[float] = []
    for _ in range(1000):
        idx = rng.integers(0, len(clean), len(clean))
        if np.std(x[idx]) > 0.0 and np.std(y[idx]) > 0.0:
            boot.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    if boot:
        lo, hi = np.percentile(boot, [2.5, 97.5])
        row["pearson_ci_low"] = float(lo)
        row["pearson_ci_high"] = float(hi)
    else:
        row["pearson_ci_low"] = np.nan
        row["pearson_ci_high"] = np.nan
    return row


def transfer_weight_diagnostics(results: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    race = results[results["detector"] == "RACE"].copy()
    rows = []
    for metric in TRANSFERABILITY_COLUMNS:
        rows.append(_correlation_row(race, metric, "transfer_weight", level="run"))
    rows.append(_correlation_row(deltas, "transfer_weight", "delta_recall", level="run"))
    rows.append(_correlation_row(deltas, "transfer_weight", "delta_success", level="run"))
    return pd.DataFrame(rows)


def scientific_decision(deltas: pd.DataFrame) -> dict[str, object]:
    if deltas.empty:
        return {"decision": "INCONCLUSIVE", "reason": "no paired RACE/TargetOnly deltas"}
    by_regime = deltas.groupby("transferability_regime").agg(
        delta_recall_mean=("delta_recall", "mean"),
        delta_fpr_mean=("delta_fpr", "mean"),
        delta_success_mean=("delta_success", "mean"),
        n=("delta_recall", "size"),
    )
    best_recall = float(by_regime["delta_recall_mean"].max())
    worst_recall = float(by_regime["delta_recall_mean"].min())
    best_regime = str(by_regime["delta_recall_mean"].idxmax())
    high_harm = float(by_regime.loc["high_shift", "delta_recall_mean"]) if "high_shift" in by_regime.index else np.nan
    fpr_at_best = float(by_regime.loc[best_regime, "delta_fpr_mean"])
    mean_delta = float(deltas["delta_recall"].mean())
    if best_recall >= 0.05 and fpr_at_best <= 0.005:
        decision = "SUPPORTS_TRANSFER_REGIME"
        reason = "preidentified regime has positive paired recall benefit without material FPR worsening"
    elif mean_delta < -0.02 or (np.isfinite(high_harm) and high_harm < -0.03):
        decision = "NEGATIVE_TRANSFER"
        reason = "paired recall is harmed overall or under high healthy shift"
    elif abs(mean_delta) <= 0.03 and best_recall < 0.05:
        decision = "NO_MEANINGFUL_TRANSFER_BENEFIT"
        reason = "RACE is essentially tied with TargetOnly across transferability regimes"
    else:
        decision = "INCONCLUSIVE"
        reason = "effects are small or unstable relative to the run count"
    return {
        "decision": decision,
        "reason": reason,
        "mean_delta_recall": mean_delta,
        "best_regime": best_regime,
        "best_regime_delta_recall_mean": best_recall,
        "worst_regime_delta_recall_mean": worst_recall,
        "best_regime_delta_fpr_mean": fpr_at_best,
        "regime_table": by_regime.reset_index().to_dict(orient="records"),
    }
