#!/usr/bin/env python3
"""
experiments/analyze_aursad_score_diagnostics.py

Formal analysis-only diagnostic for the frozen AURSAD commissioning benchmark.

Consumes:
    outputs/aursad/score_diagnostics/aursad_episode_scores.csv
    outputs/aursad/score_diagnostics/aursad_score_run_verification.csv
    outputs/aursad/score_diagnostics/aursad_score_export_manifest.json

Produces:
    outputs/aursad/score_diagnostics/formal_analysis/
        01_run_level_metrics.csv
        02_healthy_shift_by_run.csv
        03_healthy_shift_summary.csv
        04_calibration_tail_by_run.csv
        05_calibration_tail_summary.csv
        06_fault_separation_by_run.csv
        07_fault_separation_summary.csv
        08_score_threshold_ratio_summary.csv
        09_fault_operating_point_summary.csv
        10_mechanism_evidence_table.csv
        formal_diagnostic_report.json

This script:
- does not refit any detector;
- does not change thresholds;
- does not import Matplotlib;
- checks that score-export reruns were verified against frozen benchmark results;
- uses deterministic bootstrap confidence intervals across the predeclared diagnostic seeds;
- keeps descriptive/inferential claims scoped to this diagnostic subset.

Scientific questions
--------------------
1. Healthy shift:
   Are held-out healthy evaluation scores systematically different from calibration healthy scores?

2. Calibration-tail conservatism:
   Is the frozen conformal threshold an extreme calibration-tail statistic, and how far is it
   above typical healthy scores?

3. Fault separability:
   Does each fault class actually rank above held-out healthy cycles?

4. Operating-point feasibility:
   At the frozen 1% false-alert threshold, which fault classes exceed the threshold?

5. Mechanism classification:
   Does evidence support:
       - healthy distribution shift,
       - calibration-tail conservatism,
       - weak fault/healthy separation,
       - fault-specific difficulty?

Default diagnostic subset was predeclared in the export stage:
    detectors = TargetOnly, Euclidean conformal k-NN
    N = 100, 500
    seeds = 0, 4, 9, 13, 19
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "score_diagnostics"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "formal_analysis"

GLOBAL_SEED = 42
BOOTSTRAP_REPS = 10_000
CONFIDENCE = 0.95

EXPECTED_PARTITIONS = {
    "calibration_healthy",
    "evaluation_healthy",
    "anomaly_evaluation",
}

FAULT_NAME_HINTS = [
    "damaged_screw",
    "extra_component",
    "missing_screw",
    "damaged_thread",
]

VERSION = "aursad-formal-score-diagnostic-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Formal analysis of verified AURSAD episode-level diagnostic scores."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    p.add_argument("--confidence", type=float, default=CONFIDENCE)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [json_safe(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    confidence: float,
    *,
    seed_offset: int = 0,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(GLOBAL_SEED + seed_offset)
    n = len(x)
    means = np.empty(reps, dtype=np.float64)

    chunk = 1000
    start = 0
    while start < reps:
        size = min(chunk, reps - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start:start + size] = x[idx].mean(axis=1)
        start += size

    alpha = 1.0 - confidence
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def empirical_probability_superiority(anomaly: np.ndarray, healthy: np.ndarray) -> float:
    """
    P(A > H) + 0.5 P(A == H), equivalent to normalized Mann-Whitney U / AUROC
    when comparing one anomaly class against healthy scores.
    """
    a = np.asarray(anomaly, dtype=np.float64)
    h = np.asarray(healthy, dtype=np.float64)
    a = a[np.isfinite(a)]
    h = h[np.isfinite(h)]
    if len(a) == 0 or len(h) == 0:
        return np.nan
    u = mannwhitneyu(a, h, alternative="two-sided").statistic
    return float(u / (len(a) * len(h)))


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-15:
        return np.nan
    return float(a / b)


def validate_inputs(
    scores: pd.DataFrame,
    verification: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    required = {
        "detector",
        "commissioning_size",
        "seed",
        "episode_id",
        "partition",
        "label",
        "label_name",
        "score",
        "threshold",
        "prediction",
        "threshold_margin",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Episode score CSV missing columns: {missing}")

    if scores.empty:
        raise ValueError("Episode score CSV is empty.")

    # Frozen reproduction verification is non-negotiable.
    if "status" not in verification.columns:
        raise ValueError("Verification CSV has no status column.")
    bad = verification[verification["status"].astype(str).str.upper().ne("PASS")]
    if not bad.empty:
        raise RuntimeError(
            "Formal analysis blocked: diagnostic score export contains failed "
            "frozen-result verification rows."
        )

    verification_summary = manifest.get("verification", {})
    if verification_summary.get("all_reproduced_runs_match_frozen_results") is not True:
        raise RuntimeError(
            "Formal analysis blocked: manifest does not certify frozen-result reproduction."
        )

    partitions = set(scores["partition"].astype(str).unique())
    if partitions != EXPECTED_PARTITIONS:
        raise ValueError(
            f"Unexpected partitions: {sorted(partitions)}; expected {sorted(EXPECTED_PARTITIONS)}."
        )

    key = ["detector", "commissioning_size", "seed", "partition", "episode_id"]
    if scores.duplicated(key).any():
        raise ValueError("Duplicate detector/N/seed/partition/episode_id rows detected.")

    for c in ["commissioning_size", "seed", "label", "score", "threshold"]:
        scores[c] = pd.to_numeric(scores[c], errors="raise")

    if not np.isfinite(scores["score"].to_numpy(float)).all():
        raise ValueError("Scores contain NaN/Inf.")
    if not np.isfinite(scores["threshold"].to_numpy(float)).all():
        raise ValueError("Thresholds contain NaN/Inf.")

    # One threshold per detector/N/seed.
    n_thresh = (
        scores.groupby(["detector", "commissioning_size", "seed"])["threshold"]
        .nunique()
    )
    if (n_thresh != 1).any():
        raise ValueError("A run contains multiple thresholds.")

    # Calibration and healthy evaluation must truly be healthy.
    for partition in ["calibration_healthy", "evaluation_healthy"]:
        x = scores[scores["partition"].eq(partition)]
        if not x["label"].eq(0).all():
            raise ValueError(f"{partition} contains nonzero labels.")

    anomaly = scores[scores["partition"].eq("anomaly_evaluation")]
    if anomaly["label"].eq(0).any():
        raise ValueError("Anomaly evaluation contains healthy labels.")


def run_level_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (detector, n, seed), g in scores.groupby(
        ["detector", "commissioning_size", "seed"], sort=True
    ):
        threshold = float(g["threshold"].iloc[0])
        cal = g[g["partition"].eq("calibration_healthy")]["score"].to_numpy(float)
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)
        anomaly = g[g["partition"].eq("anomaly_evaluation")]["score"].to_numpy(float)

        rows.append(
            {
                "detector": detector,
                "commissioning_size": int(n),
                "seed": int(seed),
                "n_calibration": int(len(cal)),
                "n_healthy_eval": int(len(healthy)),
                "n_anomaly_eval": int(len(anomaly)),
                "threshold": threshold,
                "healthy_eval_fpr": float(np.mean(healthy > threshold)),
                "anomaly_recall": float(np.mean(anomaly > threshold)),
                "calibration_exceedance_rate": float(np.mean(cal > threshold)),
                "calibration_tie_rate": float(np.mean(np.isclose(cal, threshold))),
                "healthy_median_over_threshold": safe_ratio(float(np.median(healthy)), threshold),
                "anomaly_median_over_threshold": safe_ratio(float(np.median(anomaly)), threshold),
            }
        )

    return pd.DataFrame(rows)


def healthy_shift_by_run(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (detector, n, seed), g in scores.groupby(
        ["detector", "commissioning_size", "seed"], sort=True
    ):
        cal = g[g["partition"].eq("calibration_healthy")]["score"].to_numpy(float)
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)
        threshold = float(g["threshold"].iloc[0])

        ks = ks_2samp(cal, healthy, alternative="two-sided", method="auto")
        mw = mannwhitneyu(healthy, cal, alternative="two-sided")

        rows.append(
            {
                "detector": detector,
                "commissioning_size": int(n),
                "seed": int(seed),
                "threshold": threshold,
                "calibration_count": int(len(cal)),
                "healthy_eval_count": int(len(healthy)),
                "calibration_mean": float(np.mean(cal)),
                "healthy_eval_mean": float(np.mean(healthy)),
                "mean_shift_eval_minus_cal": float(np.mean(healthy) - np.mean(cal)),
                "mean_shift_normalized_by_threshold": safe_ratio(
                    float(np.mean(healthy) - np.mean(cal)), threshold
                ),
                "calibration_median": float(np.median(cal)),
                "healthy_eval_median": float(np.median(healthy)),
                "median_shift_eval_minus_cal": float(np.median(healthy) - np.median(cal)),
                "calibration_q95": float(np.quantile(cal, 0.95)),
                "healthy_eval_q95": float(np.quantile(healthy, 0.95)),
                "q95_shift_eval_minus_cal": float(
                    np.quantile(healthy, 0.95) - np.quantile(cal, 0.95)
                ),
                "calibration_q99": float(np.quantile(cal, 0.99)),
                "healthy_eval_q99": float(np.quantile(healthy, 0.99)),
                "q99_shift_eval_minus_cal": float(
                    np.quantile(healthy, 0.99) - np.quantile(cal, 0.99)
                ),
                "calibration_max": float(np.max(cal)),
                "healthy_eval_max": float(np.max(healthy)),
                "max_shift_eval_minus_cal": float(np.max(healthy) - np.max(cal)),
                "wasserstein_distance": float(wasserstein_distance(cal, healthy)),
                "wasserstein_normalized_by_threshold": safe_ratio(
                    float(wasserstein_distance(cal, healthy)), threshold
                ),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "mannwhitney_u": float(mw.statistic),
                "mannwhitney_pvalue": float(mw.pvalue),
                "probability_healthy_eval_gt_calibration": empirical_probability_superiority(
                    healthy, cal
                ),
            }
        )

    return pd.DataFrame(rows)


def aggregate_healthy_shift(
    by_run: pd.DataFrame, reps: int, confidence: float
) -> pd.DataFrame:
    metrics = [
        "mean_shift_normalized_by_threshold",
        "q99_shift_eval_minus_cal",
        "wasserstein_normalized_by_threshold",
        "ks_statistic",
        "probability_healthy_eval_gt_calibration",
    ]
    rows = []
    for (detector, n), g in by_run.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        row: dict[str, Any] = {
            "detector": detector,
            "commissioning_size": int(n),
            "number_of_seeds": int(len(g)),
        }
        for i, metric in enumerate(metrics):
            mean, lo, hi = bootstrap_mean_ci(
                g[metric].to_numpy(float),
                reps,
                confidence,
                seed_offset=i + int(n),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lo
            row[f"{metric}_ci_upper"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_tail_by_run(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (detector, n, seed), g in scores.groupby(
        ["detector", "commissioning_size", "seed"], sort=True
    ):
        threshold = float(g["threshold"].iloc[0])
        cal = np.sort(
            g[g["partition"].eq("calibration_healthy")]["score"].to_numpy(float)
        )
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)

        maximum = float(cal[-1])
        second = float(cal[-2]) if len(cal) >= 2 else np.nan
        q95 = float(np.quantile(cal, 0.95))
        q99 = float(np.quantile(cal, 0.99))
        median = float(np.median(cal))

        rows.append(
            {
                "detector": detector,
                "commissioning_size": int(n),
                "seed": int(seed),
                "n_calibration": int(len(cal)),
                "threshold": threshold,
                "calibration_median": median,
                "calibration_q95": q95,
                "calibration_q99": q99,
                "calibration_second_max": second,
                "calibration_max": maximum,
                "threshold_equals_max": bool(np.isclose(threshold, maximum)),
                "threshold_rank_from_bottom": int(
                    np.searchsorted(cal, threshold, side="right")
                ),
                "threshold_over_calibration_median": safe_ratio(threshold, median),
                "threshold_over_calibration_q95": safe_ratio(threshold, q95),
                "threshold_over_calibration_q99": safe_ratio(threshold, q99),
                "max_over_second_max": safe_ratio(maximum, second),
                "healthy_median_over_threshold": safe_ratio(
                    float(np.median(healthy)), threshold
                ),
                "healthy_q95_over_threshold": safe_ratio(
                    float(np.quantile(healthy, 0.95)), threshold
                ),
                "healthy_q99_over_threshold": safe_ratio(
                    float(np.quantile(healthy, 0.99)), threshold
                ),
                "healthy_exceedance_rate": float(np.mean(healthy > threshold)),
            }
        )

    return pd.DataFrame(rows)


def aggregate_calibration_tail(
    by_run: pd.DataFrame, reps: int, confidence: float
) -> pd.DataFrame:
    metrics = [
        "threshold_over_calibration_median",
        "threshold_over_calibration_q95",
        "threshold_over_calibration_q99",
        "max_over_second_max",
        "healthy_median_over_threshold",
        "healthy_q95_over_threshold",
        "healthy_q99_over_threshold",
        "healthy_exceedance_rate",
    ]

    rows = []
    for (detector, n), g in by_run.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        row: dict[str, Any] = {
            "detector": detector,
            "commissioning_size": int(n),
            "number_of_seeds": int(len(g)),
            "fraction_threshold_equals_calibration_max": float(
                np.mean(g["threshold_equals_max"].astype(float))
            ),
        }
        for i, metric in enumerate(metrics):
            mean, lo, hi = bootstrap_mean_ci(
                g[metric].to_numpy(float),
                reps,
                confidence,
                seed_offset=100 + i + int(n),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lo
            row[f"{metric}_ci_upper"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def fault_separation_by_run(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (detector, n, seed), g in scores.groupby(
        ["detector", "commissioning_size", "seed"], sort=True
    ):
        threshold = float(g["threshold"].iloc[0])
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)

        anomalies = g[g["partition"].eq("anomaly_evaluation")]

        for (label, label_name), fg in anomalies.groupby(
            ["label", "label_name"], sort=True
        ):
            fault = fg["score"].to_numpy(float)
            mw = mannwhitneyu(fault, healthy, alternative="two-sided")

            rows.append(
                {
                    "detector": detector,
                    "commissioning_size": int(n),
                    "seed": int(seed),
                    "label": int(label),
                    "label_name": str(label_name),
                    "healthy_count": int(len(healthy)),
                    "fault_count": int(len(fault)),
                    "threshold": threshold,
                    "healthy_mean": float(np.mean(healthy)),
                    "fault_mean": float(np.mean(fault)),
                    "mean_margin_fault_minus_healthy": float(
                        np.mean(fault) - np.mean(healthy)
                    ),
                    "healthy_median": float(np.median(healthy)),
                    "fault_median": float(np.median(fault)),
                    "median_margin_fault_minus_healthy": float(
                        np.median(fault) - np.median(healthy)
                    ),
                    "healthy_q95": float(np.quantile(healthy, 0.95)),
                    "fault_q25": float(np.quantile(fault, 0.25)),
                    "q25_fault_minus_q95_healthy": float(
                        np.quantile(fault, 0.25) - np.quantile(healthy, 0.95)
                    ),
                    "healthy_median_over_threshold": safe_ratio(
                        float(np.median(healthy)), threshold
                    ),
                    "fault_median_over_threshold": safe_ratio(
                        float(np.median(fault)), threshold
                    ),
                    "fault_q25_over_threshold": safe_ratio(
                        float(np.quantile(fault, 0.25)), threshold
                    ),
                    "fault_q75_over_threshold": safe_ratio(
                        float(np.quantile(fault, 0.75)), threshold
                    ),
                    "fault_threshold_exceedance_rate": float(
                        np.mean(fault > threshold)
                    ),
                    "probability_fault_score_gt_healthy": empirical_probability_superiority(
                        fault, healthy
                    ),
                    "mannwhitney_u": float(mw.statistic),
                    "mannwhitney_pvalue": float(mw.pvalue),
                    "wasserstein_distance": float(
                        wasserstein_distance(fault, healthy)
                    ),
                    "wasserstein_normalized_by_threshold": safe_ratio(
                        float(wasserstein_distance(fault, healthy)), threshold
                    ),
                }
            )

    return pd.DataFrame(rows)


def aggregate_fault_separation(
    by_run: pd.DataFrame, reps: int, confidence: float
) -> pd.DataFrame:
    metrics = [
        "fault_threshold_exceedance_rate",
        "probability_fault_score_gt_healthy",
        "fault_median_over_threshold",
        "fault_q25_over_threshold",
        "fault_q75_over_threshold",
        "q25_fault_minus_q95_healthy",
        "wasserstein_normalized_by_threshold",
    ]

    rows = []
    for (detector, n, label, label_name), g in by_run.groupby(
        ["detector", "commissioning_size", "label", "label_name"], sort=True
    ):
        row: dict[str, Any] = {
            "detector": detector,
            "commissioning_size": int(n),
            "label": int(label),
            "label_name": str(label_name),
            "number_of_seeds": int(len(g)),
        }
        for i, metric in enumerate(metrics):
            mean, lo, hi = bootstrap_mean_ci(
                g[metric].to_numpy(float),
                reps,
                confidence,
                seed_offset=200 + i + int(n) + int(label),
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lower"] = lo
            row[f"{metric}_ci_upper"] = hi
        rows.append(row)

    return pd.DataFrame(rows)


def score_threshold_ratio_summary(scores: pd.DataFrame) -> pd.DataFrame:
    x = scores.copy()
    x["score_over_threshold"] = x["score"] / x["threshold"]

    def group_name(row: pd.Series) -> str:
        if row["partition"] == "calibration_healthy":
            return "Calibration healthy"
        if row["partition"] == "evaluation_healthy":
            return "Evaluation healthy"
        return str(row["label_name"])

    x["score_group"] = x.apply(group_name, axis=1)

    rows = []
    for (detector, n, group), g in x.groupby(
        ["detector", "commissioning_size", "score_group"], sort=True
    ):
        vals = g["score_over_threshold"].to_numpy(float)
        rows.append(
            {
                "detector": detector,
                "commissioning_size": int(n),
                "score_group": str(group),
                "n_episode_scores": int(len(vals)),
                "mean_score_over_threshold": float(np.mean(vals)),
                "median_score_over_threshold": float(np.median(vals)),
                "q10_score_over_threshold": float(np.quantile(vals, 0.10)),
                "q25_score_over_threshold": float(np.quantile(vals, 0.25)),
                "q75_score_over_threshold": float(np.quantile(vals, 0.75)),
                "q90_score_over_threshold": float(np.quantile(vals, 0.90)),
                "q95_score_over_threshold": float(np.quantile(vals, 0.95)),
                "q99_score_over_threshold": float(np.quantile(vals, 0.99)),
                "threshold_exceedance_rate": float(np.mean(vals > 1.0)),
            }
        )
    return pd.DataFrame(rows)


def operating_point_summary(
    fault_summary: pd.DataFrame,
    healthy_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    healthy_key = healthy_summary[
        healthy_summary["score_group"].eq("Evaluation healthy")
    ][
        [
            "detector",
            "commissioning_size",
            "threshold_exceedance_rate",
            "median_score_over_threshold",
            "q99_score_over_threshold",
        ]
    ].rename(
        columns={
            "threshold_exceedance_rate": "healthy_fpr",
            "median_score_over_threshold": "healthy_median_over_threshold",
            "q99_score_over_threshold": "healthy_q99_over_threshold",
        }
    )

    for _, r in fault_summary.iterrows():
        h = healthy_key[
            healthy_key["detector"].eq(r["detector"])
            & healthy_key["commissioning_size"].eq(r["commissioning_size"])
        ]
        if len(h) != 1:
            raise RuntimeError("Could not match healthy operating-point summary.")
        hv = h.iloc[0]

        recall = float(r["fault_threshold_exceedance_rate_mean"])
        separation = float(r["probability_fault_score_gt_healthy_mean"])
        median_ratio = float(r["fault_median_over_threshold_mean"])
        fpr = float(hv["healthy_fpr"])

        if separation < 0.65 and recall < 0.20:
            interpretation = "weak_score_separation"
        elif separation >= 0.90 and recall < 0.90:
            interpretation = "strong_ranking_but_operating_threshold_limits_recall"
        elif recall >= 0.90 and fpr <= 0.01:
            interpretation = "commissioning_ready_for_fault_class"
        elif recall < 0.20:
            interpretation = "low_operating_point_recall"
        else:
            interpretation = "intermediate"

        rows.append(
            {
                "detector": r["detector"],
                "commissioning_size": int(r["commissioning_size"]),
                "label": int(r["label"]),
                "label_name": str(r["label_name"]),
                "healthy_fpr": fpr,
                "fault_recall": recall,
                "probability_fault_score_gt_healthy": separation,
                "fault_median_over_threshold": median_ratio,
                "healthy_median_over_threshold": float(
                    hv["healthy_median_over_threshold"]
                ),
                "healthy_q99_over_threshold": float(
                    hv["healthy_q99_over_threshold"]
                ),
                "mechanism_interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def mechanism_evidence_table(
    healthy_shift_summary: pd.DataFrame,
    calibration_tail_summary: pd.DataFrame,
    fault_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (detector, n), hs in healthy_shift_summary.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        hs = hs.iloc[0]
        ct = calibration_tail_summary[
            calibration_tail_summary["detector"].eq(detector)
            & calibration_tail_summary["commissioning_size"].eq(n)
        ].iloc[0]

        fs = fault_summary[
            fault_summary["detector"].eq(detector)
            & fault_summary["commissioning_size"].eq(n)
        ]

        # These are deliberately descriptive, conservative evidence rules.
        # They do NOT claim causal proof.
        ks = float(hs["ks_statistic_mean"])
        wnorm = float(hs["wasserstein_normalized_by_threshold_mean"])
        pshift = float(hs["probability_healthy_eval_gt_calibration_mean"])

        healthy_shift_evidence = (
            "weak"
            if ks < 0.10 and wnorm < 0.10 and 0.40 <= pshift <= 0.60
            else "moderate_or_stronger"
        )

        fraction_max = float(ct["fraction_threshold_equals_calibration_max"])
        threshold_q99_ratio = float(ct["threshold_over_calibration_q99_mean"])
        calibration_tail_evidence = (
            "strong"
            if fraction_max >= 0.80
            else "moderate"
            if fraction_max >= 0.40 or threshold_q99_ratio >= 1.0
            else "weak"
        )

        probs = fs["probability_fault_score_gt_healthy_mean"].to_numpy(float)
        recalls = fs["fault_threshold_exceedance_rate_mean"].to_numpy(float)

        weak_classes = int(np.sum((probs < 0.65) & (recalls < 0.20)))
        strong_rank_classes = int(np.sum(probs >= 0.90))

        if weak_classes >= max(1, len(fs) // 2):
            separation_evidence = "strong_fault_specific_overlap"
        elif weak_classes >= 1:
            separation_evidence = "mixed_fault_specific_overlap"
        else:
            separation_evidence = "limited_overlap_evidence"

        rows.append(
            {
                "detector": detector,
                "commissioning_size": int(n),
                "healthy_shift_evidence": healthy_shift_evidence,
                "calibration_tail_evidence": calibration_tail_evidence,
                "fault_separation_evidence": separation_evidence,
                "number_of_fault_classes": int(len(fs)),
                "weakly_separated_fault_classes": weak_classes,
                "strongly_ranked_fault_classes": strong_rank_classes,
                "mean_healthy_ks_statistic": ks,
                "mean_healthy_wasserstein_over_threshold": wnorm,
                "mean_probability_healthy_eval_gt_calibration": pshift,
                "fraction_threshold_equals_calibration_max": fraction_max,
                "mean_threshold_over_calibration_q99": threshold_q99_ratio,
            }
        )

    return pd.DataFrame(rows)


def build_report(
    *,
    manifest: dict[str, Any],
    run_metrics: pd.DataFrame,
    healthy_summary: pd.DataFrame,
    tail_summary: pd.DataFrame,
    fault_summary: pd.DataFrame,
    operating_summary: pd.DataFrame,
    mechanism_table: pd.DataFrame,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    reps: int,
    confidence: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "AURSAD",
            "analysis_type": "formal_diagnostic_analysis",
            "primary_benchmark_modified": False,
            "diagnostic_subset_only": True,
            "source_score_export_selection": manifest.get("selection", {}),
            "bootstrap_replicates": reps,
            "confidence": confidence,
        },
        "input_integrity": {
            "frozen_result_reproduction_verified": True,
            "inputs": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            },
        },
        "headline_findings": [],
        "detector_findings": {},
        "limitations": [
            "This formal diagnostic uses the predeclared representative subset rather than all 20 seeds.",
            "Only TargetOnly and Euclidean conformal k-NN are included in this score-level diagnostic.",
            "Mechanism labels are evidence classifications, not causal proofs.",
            "P-values are descriptive diagnostics and are not used to redefine the frozen primary endpoint.",
            "The primary commissioning results remain the full 20-seed frozen benchmark.",
        ],
        "outputs": {
            name: str(path) for name, path in outputs.items()
        },
    }

    # Build evidence-backed textual findings from tables.
    for (detector, n), mech in mechanism_table.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        m = mech.iloc[0]
        faults = operating_summary[
            operating_summary["detector"].eq(detector)
            & operating_summary["commissioning_size"].eq(n)
        ].sort_values("fault_recall")

        detector_key = f"{detector}|N={int(n)}"
        report["detector_findings"][detector_key] = {
            "healthy_shift_evidence": m["healthy_shift_evidence"],
            "calibration_tail_evidence": m["calibration_tail_evidence"],
            "fault_separation_evidence": m["fault_separation_evidence"],
            "faults": [
                {
                    "label_name": r["label_name"],
                    "fault_recall": float(r["fault_recall"]),
                    "probability_fault_score_gt_healthy": float(
                        r["probability_fault_score_gt_healthy"]
                    ),
                    "fault_median_over_threshold": float(
                        r["fault_median_over_threshold"]
                    ),
                    "mechanism_interpretation": r["mechanism_interpretation"],
                }
                for _, r in faults.iterrows()
            ],
        }

    # Headline statements are phrased conservatively.
    if not mechanism_table.empty:
        if (
            mechanism_table["healthy_shift_evidence"]
            .eq("weak")
            .all()
        ):
            report["headline_findings"].append(
                "Across the diagnostic subset, calibration-healthy and held-out healthy "
                "score distributions show weak evidence of systematic healthy distribution shift."
            )

        if (
            mechanism_table["fault_separation_evidence"]
            .isin(["strong_fault_specific_overlap", "mixed_fault_specific_overlap"])
            .any()
        ):
            report["headline_findings"].append(
                "Fault detectability is strongly class-dependent: at least one detector/N "
                "condition shows fault classes with weak score separation from held-out healthy cycles."
            )

        if (
            mechanism_table["calibration_tail_evidence"]
            .isin(["strong", "moderate"])
            .any()
        ):
            report["headline_findings"].append(
                "Calibration-tail conservatism is present in at least part of the diagnostic subset, "
                "but it should be interpreted jointly with fault/healthy score separation."
            )

    return report


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive.")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must be between 0 and 1.")

    scores_path = input_dir / "aursad_episode_scores.csv"
    verification_path = input_dir / "aursad_score_run_verification.csv"
    manifest_path = input_dir / "aursad_score_export_manifest.json"

    require_file(scores_path, "Episode score export")
    require_file(verification_path, "Frozen reproduction verification")
    require_file(manifest_path, "Score export manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains files. Use --overwrite to regenerate."
        )

    scores = pd.read_csv(scores_path)
    verification = pd.read_csv(verification_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validate_inputs(scores, verification, manifest)

    print("=" * 78)
    print("AURSAD FORMAL SCORE DIAGNOSTIC ANALYSIS")
    print("=" * 78)
    print(f"Scores:      {scores_path}")
    print(f"Rows:        {len(scores):,}")
    print(f"Detectors:   {sorted(scores['detector'].unique().tolist())}")
    print(f"N values:    {sorted(scores['commissioning_size'].unique().tolist())}")
    print(f"Seeds:       {sorted(scores['seed'].unique().tolist())}")
    print(f"Bootstrap:   {args.bootstrap_reps:,}")
    print(f"Confidence:  {args.confidence:.3f}")
    print()

    run_metrics = run_level_metrics(scores)
    healthy_by_run = healthy_shift_by_run(scores)
    healthy_summary = aggregate_healthy_shift(
        healthy_by_run, args.bootstrap_reps, args.confidence
    )
    tail_by_run = calibration_tail_by_run(scores)
    tail_summary = aggregate_calibration_tail(
        tail_by_run, args.bootstrap_reps, args.confidence
    )
    fault_by_run = fault_separation_by_run(scores)
    fault_summary = aggregate_fault_separation(
        fault_by_run, args.bootstrap_reps, args.confidence
    )
    ratio_summary = score_threshold_ratio_summary(scores)
    operating_summary = operating_point_summary(fault_summary, ratio_summary)
    mechanism_table = mechanism_evidence_table(
        healthy_summary, tail_summary, fault_summary
    )

    outputs = {
        "run_level_metrics": output_dir / "01_run_level_metrics.csv",
        "healthy_shift_by_run": output_dir / "02_healthy_shift_by_run.csv",
        "healthy_shift_summary": output_dir / "03_healthy_shift_summary.csv",
        "calibration_tail_by_run": output_dir / "04_calibration_tail_by_run.csv",
        "calibration_tail_summary": output_dir / "05_calibration_tail_summary.csv",
        "fault_separation_by_run": output_dir / "06_fault_separation_by_run.csv",
        "fault_separation_summary": output_dir / "07_fault_separation_summary.csv",
        "score_threshold_ratio_summary": output_dir / "08_score_threshold_ratio_summary.csv",
        "fault_operating_point_summary": output_dir / "09_fault_operating_point_summary.csv",
        "mechanism_evidence_table": output_dir / "10_mechanism_evidence_table.csv",
        "formal_diagnostic_report": output_dir / "formal_diagnostic_report.json",
    }

    atomic_csv(run_metrics, outputs["run_level_metrics"])
    atomic_csv(healthy_by_run, outputs["healthy_shift_by_run"])
    atomic_csv(healthy_summary, outputs["healthy_shift_summary"])
    atomic_csv(tail_by_run, outputs["calibration_tail_by_run"])
    atomic_csv(tail_summary, outputs["calibration_tail_summary"])
    atomic_csv(fault_by_run, outputs["fault_separation_by_run"])
    atomic_csv(fault_summary, outputs["fault_separation_summary"])
    atomic_csv(ratio_summary, outputs["score_threshold_ratio_summary"])
    atomic_csv(operating_summary, outputs["fault_operating_point_summary"])
    atomic_csv(mechanism_table, outputs["mechanism_evidence_table"])

    report = build_report(
        manifest=manifest,
        run_metrics=run_metrics,
        healthy_summary=healthy_summary,
        tail_summary=tail_summary,
        fault_summary=fault_summary,
        operating_summary=operating_summary,
        mechanism_table=mechanism_table,
        inputs={
            "episode_scores": scores_path,
            "verification": verification_path,
            "score_export_manifest": manifest_path,
        },
        outputs=outputs,
        reps=args.bootstrap_reps,
        confidence=args.confidence,
    )

    outputs["formal_diagnostic_report"].write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    print("Formal analysis complete.")
    print()
    print("Key evidence table:")
    print(mechanism_table.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()