"""
M2-v2.2 Extreme Healthy-Tail Forensic Audit
===========================================

This is a POST-PROCESSING / FORENSIC audit for the corrected M2-v2.1 run.

It does NOT:
    - refit TargetOnly,
    - change the frozen evaluation split,
    - introduce a new calibration method,
    - replace the primary M2 result,
    - remove any episode from the primary analysis.

It addresses the remaining reviewer attack:

    "Your 0-FP -> 1-FP oracle transition may be driven by a corrupted,
     mislabeled, duplicated, or otherwise invalid healthy episode."

The audit performs five checks:

1. Dominance / recurrence
   Identify which healthy episode is the maximum-score episode for every
   (N, commissioning_seed) run and how often the same episode recurs.

2. Dataset metadata integrity
   Reload candidate episodes directly from the parquet and verify:
       - anomaly label,
       - category,
       - setting,
       - signal schema,
       - finite raw values,
       - execution length.

3. Raw-signal / feature integrity
   Compare candidate episodes against the same frozen healthy-evaluation
   population using:
       - cycle length,
       - per-signal min/max/std/TV,
       - extracted statistical features,
       - robust median/MAD feature deviations.

   This is diagnostic only. A large feature deviation is not automatically
   data corruption.

4. Score persistence
   Quantify the candidate's score/rank across N and commissioning seeds.
   If the same physical healthy episode is extreme under many independently
   sampled commissioning sets, that argues against a one-seed numerical accident.

5. Sensitivity-only counterfactuals
   Recompute the oracle after:
       A. removing episode 1710, if present,
       B. removing the per-run maximum healthy episode.

   These are explicitly NOT primary results. They answer only:
       "How much of the empirical oracle geometry is controlled by the
        extreme healthy episode?"

Reviewer-safe interpretation
----------------------------
Passing this audit supports:

    "The extreme-tail episode is a legitimate labeled healthy execution in
     the released dataset and consistently dominates the tested score geometry."

It does NOT support:

    "The episode is representative of all future healthy operation."

That population question belongs to M5 / external validation.

Usage
-----

Run from project root:

    python experiments/run_m2_extreme_tail_audit.py

Optional explicit paths:

    python experiments/run_m2_extreme_tail_audit.py `
        --raw-scores outputs/m2_oracle_reviewer_defense/m2_v2_1_raw_eval_scores.csv `
        --data-path data/raw/voraus-ad-dataset-100hz.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math

import numpy as np
import pandas as pd

from src.feature_extractor import (
    extract_cycle_features,
    make_feature_names,
)
from src.voraus_loader import load_cycles


PROTOCOL_VERSION = "m2-extreme-tail-forensic-v2.2"

DEFAULT_RAW_SCORES = (
    PROJECT_ROOT
    / "outputs"
    / "m2_oracle_reviewer_defense"
    / "m2_v2_1_raw_eval_scores.csv"
)

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "m2_extreme_tail_audit"
)

PRIMARY_EPISODE_OF_INTEREST = 1710

RECALL_TARGET = 0.90
PRIMARY_ALLOWED_FP = 1


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return math.nan
    return float(a / b)


def _median_abs_deviation(x: np.ndarray, axis=0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis)
    return np.median(np.abs(x - med), axis=axis)


def _robust_z(
    row: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Median/MAD robust z-score.

    Uses 1.4826*MAD as a robust scale estimate.
    Features with zero MAD are reported as NaN rather than creating infinities.
    """
    row = np.asarray(row, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    med = np.median(reference, axis=0)
    mad = _median_abs_deviation(reference, axis=0)
    scale = 1.4826 * mad

    z = np.full(row.shape, np.nan, dtype=np.float64)
    valid = scale > 0
    z[valid] = (row[valid] - med[valid]) / scale[valid]
    return z


def _oracle_counts(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    allowed_fp: int,
    recall_target: float,
) -> dict:
    """Independent count-based retrospective oracle.

    Higher score = more anomalous.

    Enumerates all intervals between observed scores using candidate thresholds
    from {-inf, unique scores, midpoints, +inf}.
    """
    h = np.asarray(healthy_scores, dtype=np.float64).reshape(-1)
    a = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)

    if len(h) == 0 or len(a) == 0:
        raise ValueError("Oracle requires non-empty healthy and anomaly scores.")

    if not np.isfinite(h).all() or not np.isfinite(a).all():
        raise ValueError("Oracle received NaN/Inf.")

    unique = np.unique(np.concatenate([h, a]))

    candidates = [-np.inf]

    if len(unique):
        candidates.append(float(unique[0]))

        if len(unique) > 1:
            mids = unique[:-1] + (unique[1:] - unique[:-1]) / 2.0
            candidates.extend(float(x) for x in mids)

        candidates.append(float(unique[-1]))

    candidates.append(np.inf)

    required_tp = int(math.ceil(recall_target * len(a) - 1e-12))

    records = []

    for threshold in candidates:
        fp = int(np.sum(h > threshold))
        tp = int(np.sum(a > threshold))

        records.append(
            {
                "threshold": float(threshold),
                "fp": fp,
                "tp": tp,
                "fpr": fp / len(h),
                "recall": tp / len(a),
            }
        )

    feasible_budget = [
        r
        for r in records
        if r["fp"] <= allowed_fp
    ]

    if not feasible_budget:
        raise RuntimeError("No threshold satisfies the requested FP count.")

    best_tp = max(r["tp"] for r in feasible_budget)

    best_budget = [
        r
        for r in feasible_budget
        if r["tp"] == best_tp
    ]

    # Prefer lower FP among equal-recall solutions.
    min_fp = min(r["fp"] for r in best_budget)

    best_budget = [
        r
        for r in best_budget
        if r["fp"] == min_fp
    ]

    # Threshold value itself is secondary; choose the highest for determinism.
    selected = max(best_budget, key=lambda r: r["threshold"])

    target_records = [
        r
        for r in records
        if r["tp"] >= required_tp
    ]

    if target_records:
        target_min_fp = min(r["fp"] for r in target_records)
    else:
        target_min_fp = len(h) + 1

    return {
        "healthy_count": int(len(h)),
        "anomaly_count": int(len(a)),
        "allowed_fp": int(allowed_fp),
        "required_tp": int(required_tp),
        "max_tp_at_budget": int(selected["tp"]),
        "max_recall_at_budget": float(selected["recall"]),
        "fpr_at_selected": float(selected["fpr"]),
        "selected_threshold": float(selected["threshold"]),
        "min_fp_for_target_recall": int(target_min_fp),
        "empirically_feasible": bool(
            selected["tp"] >= required_tp
        ),
    }


# ---------------------------------------------------------------------------
# Raw-score provenance audit
# ---------------------------------------------------------------------------


def _validate_raw_score_file(raw: pd.DataFrame) -> None:
    required = {
        "commissioning_size",
        "commissioning_seed",
        "partition",
        "episode_id",
        "score",
    }

    missing = required - set(raw.columns)

    if missing:
        raise ValueError(
            f"Raw-score file is missing columns: {sorted(missing)}"
        )

    if raw["score"].isna().any():
        raise ValueError("Raw-score file contains NaN scores.")

    if not np.isfinite(raw["score"].to_numpy(dtype=float)).all():
        raise ValueError("Raw-score file contains Inf scores.")

    # Every run must contain one row per episode, not duplicated exports.
    duplicate_mask = raw.duplicated(
        subset=[
            "commissioning_size",
            "commissioning_seed",
            "partition",
            "episode_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        examples = raw.loc[
            duplicate_mask,
            [
                "commissioning_size",
                "commissioning_seed",
                "partition",
                "episode_id",
            ],
        ].head(10)

        raise RuntimeError(
            "Duplicate episode rows exist in raw-score artifact.\n"
            f"{examples.to_string(index=False)}"
        )


def _extreme_episode_table(raw: pd.DataFrame) -> pd.DataFrame:
    healthy = raw[
        raw["partition"] == "healthy_eval"
    ].copy()

    rows = []

    for (n, seed), group in healthy.groupby(
        ["commissioning_size", "commissioning_seed"],
        sort=True,
    ):
        group = group.sort_values(
            ["score", "episode_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        max_row = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None

        median_score = float(group["score"].median())
        max_score = float(max_row["score"])
        second_score = (
            float(second["score"])
            if second is not None
            else math.nan
        )

        rows.append(
            {
                "commissioning_size": int(n),
                "commissioning_seed": int(seed),
                "max_healthy_episode_id": int(max_row["episode_id"]),
                "max_healthy_score": max_score,
                "second_max_healthy_episode_id": (
                    int(second["episode_id"])
                    if second is not None
                    else -1
                ),
                "second_max_healthy_score": second_score,
                "healthy_median_score": median_score,
                "max_to_median_ratio": _safe_div(
                    max_score,
                    median_score,
                ),
                "max_to_second_max_ratio": _safe_div(
                    max_score,
                    second_score,
                ),
            }
        )

    return pd.DataFrame(rows)


def _episode_recurrence(extreme_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    total_runs = len(extreme_df)

    for episode_id, group in extreme_df.groupby(
        "max_healthy_episode_id",
        sort=False,
    ):
        rows.append(
            {
                "episode_id": int(episode_id),
                "times_maximum": int(len(group)),
                "fraction_of_all_exported_runs": float(
                    len(group) / total_runs
                ),
                "N_values_where_maximum": ",".join(
                    str(int(x))
                    for x in sorted(group["commissioning_size"].unique())
                ),
                "seed_values_where_maximum": ",".join(
                    str(int(x))
                    for x in sorted(group["commissioning_seed"].unique())
                ),
                "median_max_score_when_maximum": float(
                    group["max_healthy_score"].median()
                ),
                "minimum_max_score_when_maximum": float(
                    group["max_healthy_score"].min()
                ),
                "maximum_max_score_when_maximum": float(
                    group["max_healthy_score"].max()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["times_maximum", "episode_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Dataset / feature forensic audit
# ---------------------------------------------------------------------------


def _load_frozen_healthy_cycles(
    data_path: Path,
    raw: pd.DataFrame,
):
    healthy_ids = sorted(
        raw.loc[
            raw["partition"] == "healthy_eval",
            "episode_id",
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    cycles = load_cycles(
        data_path,
        episode_ids=healthy_ids,
    )

    by_id = {
        int(c.episode_id): c
        for c in cycles
    }

    missing = set(healthy_ids) - set(by_id)

    if missing:
        raise RuntimeError(
            f"Dataset reload is missing healthy IDs: {sorted(missing)}"
        )

    return healthy_ids, by_id


def _cycle_basic_audit(
    healthy_ids: list[int],
    cycles_by_id: dict,
) -> pd.DataFrame:
    lengths = np.asarray(
        [
            len(cycles_by_id[eid].values)
            for eid in healthy_ids
        ],
        dtype=np.float64,
    )

    length_median = float(np.median(lengths))
    length_q25 = float(np.quantile(lengths, 0.25))
    length_q75 = float(np.quantile(lengths, 0.75))
    length_iqr = length_q75 - length_q25

    rows = []

    for episode_id in healthy_ids:
        cycle = cycles_by_id[episode_id]
        values = np.asarray(cycle.values, dtype=np.float64)

        rows.append(
            {
                "episode_id": int(episode_id),
                "anomaly_label": bool(cycle.anomaly),
                "category": int(cycle.category),
                "setting": int(cycle.setting),
                "n_timesteps": int(values.shape[0]),
                "n_signals": int(values.shape[1]),
                "all_raw_values_finite": bool(np.isfinite(values).all()),
                "raw_abs_max": float(np.max(np.abs(values))),
                "raw_global_mean": float(np.mean(values)),
                "raw_global_std": float(np.std(values, ddof=1)),
                "raw_global_total_variation": float(
                    np.sum(np.abs(np.diff(values, axis=0)))
                ),
                "healthy_length_median": length_median,
                "healthy_length_q25": length_q25,
                "healthy_length_q75": length_q75,
                "healthy_length_iqr": float(length_iqr),
                "length_outside_1_5_iqr": bool(
                    (
                        values.shape[0]
                        < length_q25 - 1.5 * length_iqr
                    )
                    or
                    (
                        values.shape[0]
                        > length_q75 + 1.5 * length_iqr
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def _feature_forensic_tables(
    healthy_ids: list[int],
    cycles_by_id: dict,
    candidate_ids: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_features = np.vstack(
        [
            extract_cycle_features(cycles_by_id[eid])
            for eid in healthy_ids
        ]
    )

    feature_names = make_feature_names(
        cycles_by_id[healthy_ids[0]].columns
    )

    summary_rows = []
    top_rows = []

    for candidate_id in candidate_ids:
        if candidate_id not in cycles_by_id:
            continue

        candidate_features = extract_cycle_features(
            cycles_by_id[candidate_id]
        )

        z = _robust_z(
            candidate_features,
            reference_features,
        )

        finite_z = z[np.isfinite(z)]

        abs_z = np.abs(z)
        valid_indices = np.where(np.isfinite(abs_z))[0]

        ordered = valid_indices[
            np.argsort(abs_z[valid_indices])[::-1]
        ]

        summary_rows.append(
            {
                "episode_id": int(candidate_id),
                "feature_count": int(len(candidate_features)),
                "features_with_nonzero_MAD": int(len(finite_z)),
                "features_abs_robust_z_gt_3": int(
                    np.sum(np.abs(finite_z) > 3)
                ),
                "features_abs_robust_z_gt_5": int(
                    np.sum(np.abs(finite_z) > 5)
                ),
                "features_abs_robust_z_gt_10": int(
                    np.sum(np.abs(finite_z) > 10)
                ),
                "median_abs_robust_z": float(
                    np.median(np.abs(finite_z))
                )
                if len(finite_z)
                else math.nan,
                "max_abs_robust_z": float(
                    np.max(np.abs(finite_z))
                )
                if len(finite_z)
                else math.nan,
            }
        )

        for rank, idx in enumerate(ordered[:25], start=1):
            ref_col = reference_features[:, idx]

            top_rows.append(
                {
                    "episode_id": int(candidate_id),
                    "rank": int(rank),
                    "feature_index": int(idx),
                    "feature_name": str(feature_names[idx]),
                    "candidate_value": float(candidate_features[idx]),
                    "healthy_median": float(np.median(ref_col)),
                    "healthy_q25": float(np.quantile(ref_col, 0.25)),
                    "healthy_q75": float(np.quantile(ref_col, 0.75)),
                    "robust_z": float(z[idx]),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(top_rows)


def _signal_forensics(
    healthy_ids: list[int],
    cycles_by_id: dict,
    candidate_ids: list[int],
) -> pd.DataFrame:
    """Per-signal diagnostics for candidate episodes.

    Each candidate signal is compared with the distribution of the same raw
    signal summary across all frozen healthy evaluation episodes.
    """
    if not healthy_ids:
        return pd.DataFrame()

    columns = cycles_by_id[healthy_ids[0]].columns

    reference = {}

    for signal_index, signal_name in enumerate(columns):
        rows = []

        for eid in healthy_ids:
            x = np.asarray(
                cycles_by_id[eid].values[:, signal_index],
                dtype=np.float64,
            )

            rows.append(
                [
                    np.mean(x),
                    np.std(x, ddof=1),
                    np.min(x),
                    np.max(x),
                    np.sum(np.abs(np.diff(x))),
                ]
            )

        reference[signal_index] = np.asarray(
            rows,
            dtype=np.float64,
        )

    stat_names = [
        "mean",
        "std",
        "min",
        "max",
        "total_variation",
    ]

    output = []

    for episode_id in candidate_ids:
        if episode_id not in cycles_by_id:
            continue

        cycle = cycles_by_id[episode_id]

        for signal_index, signal_name in enumerate(columns):
            x = np.asarray(
                cycle.values[:, signal_index],
                dtype=np.float64,
            )

            candidate_stats = np.asarray(
                [
                    np.mean(x),
                    np.std(x, ddof=1),
                    np.min(x),
                    np.max(x),
                    np.sum(np.abs(np.diff(x))),
                ],
                dtype=np.float64,
            )

            ref = reference[signal_index]

            for stat_index, stat_name in enumerate(stat_names):
                ref_values = ref[:, stat_index]
                med = np.median(ref_values)
                mad = np.median(np.abs(ref_values - med))
                scale = 1.4826 * mad

                robust_z = (
                    (candidate_stats[stat_index] - med) / scale
                    if scale > 0
                    else math.nan
                )

                output.append(
                    {
                        "episode_id": int(episode_id),
                        "signal_index": int(signal_index),
                        "signal_name": str(signal_name),
                        "statistic": stat_name,
                        "candidate_value": float(
                            candidate_stats[stat_index]
                        ),
                        "healthy_median": float(med),
                        "healthy_q25": float(
                            np.quantile(ref_values, 0.25)
                        ),
                        "healthy_q75": float(
                            np.quantile(ref_values, 0.75)
                        ),
                        "robust_z": float(robust_z)
                        if np.isfinite(robust_z)
                        else math.nan,
                    }
                )

    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# Score persistence and sensitivity-only counterfactuals
# ---------------------------------------------------------------------------


def _candidate_score_persistence(
    raw: pd.DataFrame,
    candidate_ids: list[int],
) -> pd.DataFrame:
    healthy = raw[
        raw["partition"] == "healthy_eval"
    ].copy()

    rows = []

    for (n, seed), group in healthy.groupby(
        ["commissioning_size", "commissioning_seed"],
        sort=True,
    ):
        ranked = group.sort_values(
            ["score", "episode_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        rank_map = {
            int(row.episode_id): int(index + 1)
            for index, row in ranked.iterrows()
        }

        score_map = {
            int(row.episode_id): float(row.score)
            for _, row in ranked.iterrows()
        }

        median_score = float(ranked["score"].median())
        second_max_score = (
            float(ranked.iloc[1]["score"])
            if len(ranked) >= 2
            else math.nan
        )

        for candidate_id in candidate_ids:
            if candidate_id not in score_map:
                continue

            score = score_map[candidate_id]

            rows.append(
                {
                    "commissioning_size": int(n),
                    "commissioning_seed": int(seed),
                    "episode_id": int(candidate_id),
                    "healthy_rank_descending": int(
                        rank_map[candidate_id]
                    ),
                    "score": float(score),
                    "healthy_median_score": median_score,
                    "score_to_median_ratio": _safe_div(
                        score,
                        median_score,
                    ),
                    "score_to_second_max_ratio": _safe_div(
                        score,
                        second_max_score,
                    ),
                    "is_maximum_healthy_score": bool(
                        rank_map[candidate_id] == 1
                    ),
                }
            )

    return pd.DataFrame(rows)


def _counterfactual_sensitivity(
    raw: pd.DataFrame,
    fixed_episode_id: int,
) -> pd.DataFrame:
    rows = []

    for (n, seed), run in raw.groupby(
        ["commissioning_size", "commissioning_seed"],
        sort=True,
    ):
        healthy = run[
            run["partition"] == "healthy_eval"
        ].copy()

        anomaly = run[
            run["partition"] == "anomaly_eval"
        ].copy()

        h = healthy["score"].to_numpy(dtype=float)
        a = anomaly["score"].to_numpy(dtype=float)

        primary = _oracle_counts(
            h,
            a,
            allowed_fp=PRIMARY_ALLOWED_FP,
            recall_target=RECALL_TARGET,
        )

        zero_fp = _oracle_counts(
            h,
            a,
            allowed_fp=0,
            recall_target=RECALL_TARGET,
        )

        # Counterfactual A: remove the named episode if it is in frozen eval.
        named = healthy[
            healthy["episode_id"].astype(int)
            != int(fixed_episode_id)
        ]

        named_removed = _oracle_counts(
            named["score"].to_numpy(dtype=float),
            a,
            allowed_fp=0,
            recall_target=RECALL_TARGET,
        )

        # Counterfactual B: remove each run's actual maximum healthy episode.
        max_idx = healthy["score"].idxmax()
        max_episode_id = int(
            healthy.loc[max_idx, "episode_id"]
        )

        max_removed_healthy = healthy.drop(index=max_idx)

        max_removed = _oracle_counts(
            max_removed_healthy["score"].to_numpy(dtype=float),
            a,
            allowed_fp=0,
            recall_target=RECALL_TARGET,
        )

        rows.append(
            {
                "commissioning_size": int(n),
                "commissioning_seed": int(seed),

                "primary_healthy_count": int(len(healthy)),
                "anomaly_count": int(len(anomaly)),

                "primary_1fp_max_recall": float(
                    primary["max_recall_at_budget"]
                ),
                "primary_1fp_feasible": bool(
                    primary["empirically_feasible"]
                ),

                "primary_0fp_max_recall": float(
                    zero_fp["max_recall_at_budget"]
                ),
                "primary_0fp_feasible": bool(
                    zero_fp["empirically_feasible"]
                ),

                "named_episode_id": int(fixed_episode_id),
                "named_episode_present": bool(
                    int(fixed_episode_id)
                    in healthy["episode_id"].astype(int).values
                ),
                "named_episode_score": (
                    float(
                        healthy.loc[
                            healthy["episode_id"].astype(int)
                            == int(fixed_episode_id),
                            "score",
                        ].iloc[0]
                    )
                    if int(fixed_episode_id)
                    in healthy["episode_id"].astype(int).values
                    else math.nan
                ),

                "after_named_episode_removal_healthy_count": int(
                    len(named)
                ),
                "after_named_episode_removal_0fp_max_recall": float(
                    named_removed["max_recall_at_budget"]
                ),
                "after_named_episode_removal_0fp_feasible": bool(
                    named_removed["empirically_feasible"]
                ),

                "per_run_max_episode_id": int(max_episode_id),
                "per_run_max_score": float(
                    healthy.loc[max_idx, "score"]
                ),
                "after_per_run_max_removal_healthy_count": int(
                    len(max_removed_healthy)
                ),
                "after_per_run_max_removal_0fp_max_recall": float(
                    max_removed["max_recall_at_budget"]
                ),
                "after_per_run_max_removal_0fp_feasible": bool(
                    max_removed["empirically_feasible"]
                ),

                "sensitivity_only_not_primary": True,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--raw-scores",
        type=Path,
        default=DEFAULT_RAW_SCORES,
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--episode-of-interest",
        type=int,
        default=PRIMARY_EPISODE_OF_INTEREST,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.raw_scores.exists():
        raise FileNotFoundError(
            f"M2-v2.1 raw scores not found: {args.raw_scores}"
        )

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"voraus-AD parquet not found: {args.data_path}"
        )

    raw = pd.read_csv(args.raw_scores)

    _validate_raw_score_file(raw)

    # ------------------------------------------------------------------
    # 1. Extreme episode recurrence
    # ------------------------------------------------------------------

    extreme_df = _extreme_episode_table(raw)

    recurrence_df = _episode_recurrence(extreme_df)

    extreme_df.to_csv(
        args.output_dir / "m2_v2_2_extreme_episode_by_run.csv",
        index=False,
    )

    recurrence_df.to_csv(
        args.output_dir / "m2_v2_2_extreme_episode_recurrence.csv",
        index=False,
    )

    candidate_ids = recurrence_df[
        "episode_id"
    ].astype(int).tolist()

    if args.episode_of_interest not in candidate_ids:
        candidate_ids.append(
            int(args.episode_of_interest)
        )

    # ------------------------------------------------------------------
    # 2. Reload exactly the frozen healthy population from the parquet
    # ------------------------------------------------------------------

    healthy_ids, cycles_by_id = _load_frozen_healthy_cycles(
        args.data_path,
        raw,
    )

    basic_df = _cycle_basic_audit(
        healthy_ids,
        cycles_by_id,
    )

    basic_df.to_csv(
        args.output_dir / "m2_v2_2_healthy_episode_integrity.csv",
        index=False,
    )

    # Candidate-only dataset integrity summary.
    candidate_integrity = basic_df[
        basic_df["episode_id"].isin(candidate_ids)
    ].copy()

    candidate_integrity.to_csv(
        args.output_dir / "m2_v2_2_candidate_integrity.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Feature / raw signal forensic analysis
    # ------------------------------------------------------------------

    feature_summary_df, top_feature_df = _feature_forensic_tables(
        healthy_ids,
        cycles_by_id,
        candidate_ids,
    )

    feature_summary_df.to_csv(
        args.output_dir / "m2_v2_2_candidate_feature_summary.csv",
        index=False,
    )

    top_feature_df.to_csv(
        args.output_dir / "m2_v2_2_candidate_top_features.csv",
        index=False,
    )

    signal_df = _signal_forensics(
        healthy_ids,
        cycles_by_id,
        candidate_ids,
    )

    signal_df.to_csv(
        args.output_dir / "m2_v2_2_candidate_signal_forensics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Score persistence
    # ------------------------------------------------------------------

    persistence_df = _candidate_score_persistence(
        raw,
        candidate_ids,
    )

    persistence_df.to_csv(
        args.output_dir / "m2_v2_2_candidate_score_persistence.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. Sensitivity-only removal analysis
    # ------------------------------------------------------------------

    counterfactual_df = _counterfactual_sensitivity(
        raw,
        fixed_episode_id=int(args.episode_of_interest),
    )

    counterfactual_df.to_csv(
        args.output_dir / "m2_v2_2_removal_sensitivity.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Reviewer-facing condensed summary
    # ------------------------------------------------------------------

    top = recurrence_df.iloc[0]

    episode_interest_rows = persistence_df[
        persistence_df["episode_id"]
        == int(args.episode_of_interest)
    ]

    named_sensitivity = (
        counterfactual_df
        .groupby("commissioning_size")
        .agg(
            runs=("commissioning_seed", "nunique"),
            primary_1fp_feasible_fraction=(
                "primary_1fp_feasible",
                "mean",
            ),
            primary_0fp_feasible_fraction=(
                "primary_0fp_feasible",
                "mean",
            ),
            after_named_removal_0fp_feasible_fraction=(
                "after_named_episode_removal_0fp_feasible",
                "mean",
            ),
            after_per_run_max_removal_0fp_feasible_fraction=(
                "after_per_run_max_removal_0fp_feasible",
                "mean",
            ),
            mean_primary_0fp_recall=(
                "primary_0fp_max_recall",
                "mean",
            ),
            mean_after_named_removal_0fp_recall=(
                "after_named_episode_removal_0fp_max_recall",
                "mean",
            ),
            mean_after_per_run_max_removal_0fp_recall=(
                "after_per_run_max_removal_0fp_max_recall",
                "mean",
            ),
        )
        .reset_index()
    )

    named_sensitivity.to_csv(
        args.output_dir / "m2_v2_2_reviewer_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Guardrails / automated conclusions
    # ------------------------------------------------------------------

    candidate_row = candidate_integrity[
        candidate_integrity["episode_id"]
        == int(args.episode_of_interest)
    ]

    episode_found = not candidate_row.empty

    if episode_found:
        row = candidate_row.iloc[0]
        episode_labeled_healthy = not bool(row["anomaly_label"])
        episode_finite = bool(row["all_raw_values_finite"])
        episode_length_outlier = bool(row["length_outside_1_5_iqr"])
    else:
        episode_labeled_healthy = False
        episode_finite = False
        episode_length_outlier = False

    episode_interest_max_fraction = (
        float(
            episode_interest_rows[
                "is_maximum_healthy_score"
            ].mean()
        )
        if len(episode_interest_rows)
        else math.nan
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "input_raw_scores": str(args.raw_scores),
        "dataset_path": str(args.data_path),
        "episode_of_interest": int(args.episode_of_interest),
        "audit_scope": {
            "detector_refit": False,
            "primary_result_changed": False,
            "episode_removed_from_primary_analysis": False,
            "removal_analysis": "sensitivity-only",
        },
        "reviewer_questions": {
            "same_episode_recurrence": (
                "Which healthy episode controls the maximum score across "
                "commissioning seeds?"
            ),
            "label_integrity": (
                "Is the controlling episode actually labeled healthy in "
                "the released parquet?"
            ),
            "raw_integrity": (
                "Does it contain finite raw data and a plausible execution length?"
            ),
            "feature_integrity": (
                "Which measured-signal statistics make it unusual relative to "
                "the same frozen healthy evaluation population?"
            ),
            "persistence": (
                "Does the episode remain extreme under independently sampled "
                "commissioning sets?"
            ),
            "removal_sensitivity": (
                "How much does the oracle geometry change if the extreme healthy "
                "episode is removed? Sensitivity only."
            ),
        },
        "automated_checks": {
            "episode_found_in_frozen_healthy_eval": bool(episode_found),
            "episode_is_labeled_healthy": bool(episode_labeled_healthy),
            "episode_raw_values_all_finite": bool(episode_finite),
            "episode_length_outside_1_5_iqr": bool(episode_length_outlier),
            "episode_fraction_of_exported_runs_as_maximum": (
                float(episode_interest_max_fraction)
                if np.isfinite(episode_interest_max_fraction)
                else None
            ),
            "most_frequent_max_episode_id": int(top["episode_id"]),
            "most_frequent_max_episode_times": int(top["times_maximum"]),
            "number_of_exported_runs": int(len(extreme_df)),
        },
        "interpretation_guardrails": [
            "A large healthy score is not itself proof of data corruption.",
            "A valid healthy label does not prove representativeness of future healthy operation.",
            "Episode-removal results are sensitivity analyses and must not replace the primary frozen result.",
            "If the same episode dominates across many seeds, that supports score-tail persistence but does not identify its physical cause.",
            "Physical-cause claims require inspection of the raw signal/feature audit, not score magnitude alone.",
        ],
    }

    with open(
        args.output_dir / "m2_v2_2_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    print("=" * 78)
    print("M2-v2.2 EXTREME HEALTHY-TAIL FORENSIC AUDIT")
    print("=" * 78)

    print("\nMost frequent maximum healthy episodes:")
    print(
        recurrence_df.head(10).to_string(index=False)
    )

    print(
        f"\nEpisode of interest: {args.episode_of_interest}"
    )
    print(
        f"  present in frozen healthy eval: {episode_found}"
    )
    print(
        f"  labeled healthy: {episode_labeled_healthy}"
    )
    print(
        f"  all raw values finite: {episode_finite}"
    )
    print(
        f"  execution-length outlier: {episode_length_outlier}"
    )

    if np.isfinite(episode_interest_max_fraction):
        print(
            "  fraction of exported N=50/100 runs where it is "
            f"the maximum healthy score: "
            f"{episode_interest_max_fraction:.3f}"
        )

    print("\nSensitivity-only oracle summary:")
    print(
        named_sensitivity.to_string(index=False)
    )

    print(
        f"\nSaved outputs to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()