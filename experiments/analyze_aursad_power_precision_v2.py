#!/usr/bin/env python3
"""
experiments/analyze_aursad_power_precision.py

Power / precision analysis for the AURSAD score-level diagnostic.

This is a SECONDARY diagnostic analysis. It does not change the frozen primary
commissioning benchmark, thresholds, detector configurations, or N* estimates.

It answers two reviewer-facing questions:

1) Fault recall precision
   - How many anomaly executions are available per fault class?
   - How wide are the per-run binomial confidence intervals?
   - Across the 20 frozen diagnostic seeds, how precise is mean fault recall?

2) Healthy-shift diagnostic power
   - Given n_calibration=600 and n_healthy_eval=300, how large a standardized
     location shift would the two-sample KS diagnostic reliably detect?
   - We estimate empirical power by resampling the observed healthy scores,
     injecting a controlled location shift measured in pooled-standard-deviation
     units, and re-running the KS test.

Important interpretation
------------------------
The KS power analysis quantifies sensitivity to a *location-shift alternative*.
It does not establish power for every possible distributional change (e.g. pure
tail-shape changes, variance-only changes, multimodality changes). The report
states this explicitly.

Expected input
--------------
Run the score exporter first for all 20 seeds:

python experiments/export_aursad_diagnostic_scores.py \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --n-values 100,500 \
  --detectors targetonly,euclidean_knn \
  --overwrite

Then run this script.

Outputs
-------
outputs/aursad/score_diagnostics/power_precision/
├── 01_fault_counts.csv
├── 02_fault_recall_precision_by_run.csv
├── 03_fault_recall_precision_summary.csv
├── 04_healthy_shift_observed_by_run.csv
├── 05_ks_power_curve_by_run.csv
├── 06_ks_power_summary.csv
├── 07_detectable_shift_summary.csv
└── power_precision_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "score_diagnostics"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "power_precision"

GLOBAL_SEED = 42
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_REPS = 10_000
DEFAULT_POWER_REPS = 2_000
DEFAULT_TEST_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
DEFAULT_SHIFT_GRID = tuple(np.round(np.arange(0.0, 1.01, 0.05), 2))

VERSION = "aursad-power-precision-v2"


def parse_float_csv(value: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("At least one float required.")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError("Shift values must be >= 0.")
    return tuple(sorted(set(vals)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Power and precision analysis for AURSAD diagnostic scores."
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    p.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    p.add_argument("--power-reps", type=int, default=DEFAULT_POWER_REPS)
    p.add_argument("--test-alpha", type=float, default=DEFAULT_TEST_ALPHA)
    p.add_argument("--target-power", type=float, default=DEFAULT_TARGET_POWER)
    p.add_argument(
        "--shift-grid",
        type=parse_float_csv,
        default=DEFAULT_SHIFT_GRID,
        help="Injected location shifts in pooled-SD units.",
    )
    p.add_argument(
        "--require-20-seeds",
        action="store_true",
        default=True,
        help="Require exactly seeds 0..19 for each detector/N condition (default true).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


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


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float,
) -> tuple[float, float]:
    if trials <= 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denom
    half = (
        z
        * math.sqrt(
            phat * (1.0 - phat) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    confidence: float,
    seed_offset: int,
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(GLOBAL_SEED + seed_offset)
    n = len(x)
    means = np.empty(reps, dtype=np.float64)
    chunk = 1000
    pos = 0
    while pos < reps:
        k = min(chunk, reps - pos)
        idx = rng.integers(0, n, size=(k, n))
        means[pos:pos+k] = x[idx].mean(axis=1)
        pos += k

    alpha = 1.0 - confidence
    return (
        mean,
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def validate(scores: pd.DataFrame, verification: pd.DataFrame, require_20: bool) -> None:
    required = {
        "detector", "commissioning_size", "seed", "episode_id", "partition",
        "label", "label_name", "score", "threshold", "prediction"
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Scores missing required columns: {missing}")

    if "status" not in verification.columns:
        raise ValueError("Verification CSV missing status column.")
    bad = verification[verification["status"].astype(str).str.upper().ne("PASS")]
    if not bad.empty:
        raise RuntimeError("At least one exported run failed frozen-result verification.")

    if require_20:
        expected = set(range(20))
        for (detector, n), g in scores.groupby(["detector", "commissioning_size"]):
            observed = set(pd.to_numeric(g["seed"]).astype(int).unique().tolist())
            if observed != expected:
                raise RuntimeError(
                    f"{detector}, N={n}: expected seeds 0..19, found {sorted(observed)}."
                )


def fault_counts(scores: pd.DataFrame) -> pd.DataFrame:
    anomaly = scores[scores["partition"].eq("anomaly_evaluation")].copy()

    # Episode membership is fixed across seeds/N, so count unique episode IDs once.
    rows = []
    unique = anomaly[["episode_id", "label", "label_name"]].drop_duplicates()
    for (label, label_name), g in unique.groupby(["label", "label_name"], sort=True):
        rows.append({
            "label": int(label),
            "label_name": str(label_name),
            "unique_fault_executions": int(g["episode_id"].nunique()),
        })
    return pd.DataFrame(rows)


def fault_recall_precision_by_run(
    scores: pd.DataFrame,
    confidence: float,
) -> pd.DataFrame:
    rows = []
    anomaly = scores[scores["partition"].eq("anomaly_evaluation")].copy()

    for (detector, n, seed, label, label_name), g in anomaly.groupby(
        ["detector", "commissioning_size", "seed", "label", "label_name"],
        sort=True,
    ):
        pred = g["prediction"].astype(bool).to_numpy()
        k = int(pred.sum())
        total = int(len(pred))
        recall = k / total if total else np.nan
        lo, hi = wilson_interval(k, total, confidence)

        rows.append({
            "detector": detector,
            "commissioning_size": int(n),
            "seed": int(seed),
            "label": int(label),
            "label_name": str(label_name),
            "fault_count": total,
            "detected_count": k,
            "recall": float(recall),
            "wilson_ci_lower": float(lo),
            "wilson_ci_upper": float(hi),
            "wilson_ci_width": float(hi - lo),
        })

    return pd.DataFrame(rows)


def fault_recall_precision_summary(
    by_run: pd.DataFrame,
    reps: int,
    confidence: float,
) -> pd.DataFrame:
    rows = []

    for (detector, n, label, label_name), g in by_run.groupby(
        ["detector", "commissioning_size", "label", "label_name"], sort=True
    ):
        mean, lo, hi = bootstrap_mean_ci(
            g["recall"].to_numpy(float),
            reps,
            confidence,
            seed_offset=int(n) + int(label) * 100,
        )
        rows.append({
            "detector": detector,
            "commissioning_size": int(n),
            "label": int(label),
            "label_name": str(label_name),
            "number_of_seeds": int(len(g)),
            "fault_count_per_seed": int(g["fault_count"].iloc[0]),
            "mean_recall_across_seeds": mean,
            "bootstrap_ci_lower": lo,
            "bootstrap_ci_upper": hi,
            "bootstrap_ci_width": hi - lo,
            "mean_within_run_wilson_ci_width": float(g["wilson_ci_width"].mean()),
            "min_detected_count": int(g["detected_count"].min()),
            "max_detected_count": int(g["detected_count"].max()),
        })

    return pd.DataFrame(rows)


def healthy_shift_observed_by_run(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (detector, n, seed), g in scores.groupby(
        ["detector", "commissioning_size", "seed"], sort=True
    ):
        cal = g[g["partition"].eq("calibration_healthy")]["score"].to_numpy(float)
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)

        pooled = np.concatenate([cal, healthy])
        pooled_sd = float(np.std(pooled, ddof=1))
        pooled_mean = float(np.mean(pooled))
        if not np.isfinite(pooled_sd) or pooled_sd <= 0:
            raise RuntimeError(f"Degenerate pooled healthy SD for {detector}, N={n}, seed={seed}")

        cal_z = (cal - pooled_mean) / pooled_sd
        healthy_z = (healthy - pooled_mean) / pooled_sd
        ks = ks_2samp(cal_z, healthy_z, alternative="two-sided", method="auto")

        rows.append({
            "detector": detector,
            "commissioning_size": int(n),
            "seed": int(seed),
            "n_calibration": int(len(cal)),
            "n_healthy_eval": int(len(healthy)),
            "pooled_score_sd": pooled_sd,
            "standardized_mean_shift_eval_minus_cal": float(
                np.mean(healthy_z) - np.mean(cal_z)
            ),
            "standardized_median_shift_eval_minus_cal": float(
                np.median(healthy_z) - np.median(cal_z)
            ),
            "observed_ks_statistic": float(ks.statistic),
            "observed_ks_pvalue": float(ks.pvalue),
        })

    return pd.DataFrame(rows)


def ks_power_for_run(
    cal: np.ndarray,
    healthy: np.ndarray,
    shifts: tuple[float, ...],
    reps: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """
    Empirical two-sample KS power under a controlled location-shift alternative.

    IMPORTANT NULL CONSTRUCTION
    ---------------------------
    Under delta=0 both samples MUST come from the same distribution. Therefore
    calibration and healthy-evaluation observations are first pooled, standardized,
    and BOTH simulated groups are resampled from that pooled empirical distribution.

    For each replicate:
      1. sample n_cal observations with replacement from pooled healthy scores;
      2. sample n_eval observations independently from the SAME pooled distribution;
      3. add delta pooled-SD units only to the evaluation sample;
      4. run the two-sided KS test;
      5. record rejection at the requested alpha.

    This makes delta=0 a genuine empirical null. A null rejection rate materially
    different from alpha is treated as a failed calibration check.

    The alternative is specifically a location shift. It does not characterize
    power against variance-only, tail-only, multimodal, or other shape changes.
    """
    cal = np.asarray(cal, dtype=float)
    healthy = np.asarray(healthy, dtype=float)

    pooled = np.concatenate([cal, healthy])
    mean = float(pooled.mean())
    sd = float(pooled.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        raise RuntimeError("Cannot standardize degenerate healthy score distribution.")

    pooled_z = (pooled - mean) / sd
    n_cal = len(cal)
    n_eval = len(healthy)
    n_pool = len(pooled_z)
    results = []

    for delta in shifts:
        rejections = 0
        for _ in range(reps):
            c = pooled_z[rng.integers(0, n_pool, size=n_cal)]
            h = pooled_z[rng.integers(0, n_pool, size=n_eval)] + float(delta)
            p = ks_2samp(c, h, alternative="two-sided", method="auto").pvalue
            if p < alpha:
                rejections += 1

        power = rejections / reps
        mc_se = math.sqrt(power * (1.0 - power) / reps) if reps > 0 else np.nan
        results.append({
            "standardized_location_shift": float(delta),
            "estimated_power": float(power),
            "monte_carlo_se": float(mc_se),
        })

    return results


def ks_power_curve(
    scores: pd.DataFrame,
    shifts: tuple[float, ...],
    reps: int,
    alpha: float,
) -> pd.DataFrame:
    rows = []

    run_idx = 0
    groups = list(scores.groupby(["detector", "commissioning_size", "seed"], sort=True))
    total = len(groups)

    for (detector, n, seed), g in groups:
        run_idx += 1
        print(f"  KS power [{run_idx:02d}/{total}] {detector} N={n} seed={seed}")
        cal = g[g["partition"].eq("calibration_healthy")]["score"].to_numpy(float)
        healthy = g[g["partition"].eq("evaluation_healthy")]["score"].to_numpy(float)
        rng = np.random.default_rng(
            GLOBAL_SEED + int(n) * 1000 + int(seed) + int(hashlib.sha256(detector.encode("utf-8")).hexdigest()[:8], 16) % 100000
        )
        run_results = ks_power_for_run(cal, healthy, shifts, reps, alpha, rng)
        for r in run_results:
            rows.append({
                "detector": detector,
                "commissioning_size": int(n),
                "seed": int(seed),
                "n_calibration": int(len(cal)),
                "n_healthy_eval": int(len(healthy)),
                **r,
            })

    return pd.DataFrame(rows)


def ks_power_summary(
    curve: pd.DataFrame,
    reps: int,
    confidence: float,
) -> pd.DataFrame:
    rows = []

    for (detector, n, delta), g in curve.groupby(
        ["detector", "commissioning_size", "standardized_location_shift"], sort=True
    ):
        mean, lo, hi = bootstrap_mean_ci(
            g["estimated_power"].to_numpy(float),
            reps,
            confidence,
            seed_offset=int(n) + int(round(delta * 1000)),
        )
        rows.append({
            "detector": detector,
            "commissioning_size": int(n),
            "standardized_location_shift": float(delta),
            "number_of_seeds": int(len(g)),
            "mean_estimated_power": mean,
            "bootstrap_ci_lower": lo,
            "bootstrap_ci_upper": hi,
            "mean_monte_carlo_se": float(g["monte_carlo_se"].mean()),
        })

    return pd.DataFrame(rows)


def detectable_shift_summary(
    power_summary: pd.DataFrame,
    target_power: float,
) -> pd.DataFrame:
    rows = []

    for (detector, n), g in power_summary.groupby(
        ["detector", "commissioning_size"], sort=True
    ):
        g = g.sort_values("standardized_location_shift")
        passing = g[g["mean_estimated_power"] >= target_power]
        conservative = g[g["bootstrap_ci_lower"] >= target_power]

        rows.append({
            "detector": detector,
            "commissioning_size": int(n),
            "target_power": float(target_power),
            "smallest_shift_mean_power_ge_target": (
                float(passing.iloc[0]["standardized_location_shift"])
                if not passing.empty else np.nan
            ),
            "smallest_shift_lower_ci_ge_target": (
                float(conservative.iloc[0]["standardized_location_shift"])
                if not conservative.empty else np.nan
            ),
            "null_shift_mean_rejection_rate": float(
                g.iloc[
                    np.argmin(
                        np.abs(g["standardized_location_shift"].to_numpy(float) - 0.0)
                    )
                ]["mean_estimated_power"]
            ),
            "max_tested_shift": float(g["standardized_location_shift"].max()),
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1.")
    if args.bootstrap_reps <= 0 or args.power_reps <= 0:
        raise ValueError("Replication counts must be positive.")
    if not 0 < args.test_alpha < 1:
        raise ValueError("--test-alpha must be between 0 and 1.")
    if not 0 < args.target_power < 1:
        raise ValueError("--target-power must be between 0 and 1.")

    scores_path = input_dir / "aursad_episode_scores.csv"
    verification_path = input_dir / "aursad_score_run_verification.csv"
    manifest_path = input_dir / "aursad_score_export_manifest.json"

    require_file(scores_path, "Episode score export")
    require_file(verification_path, "Score run verification")
    require_file(manifest_path, "Score export manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} contains files. Use --overwrite.")

    scores = pd.read_csv(scores_path)
    verification = pd.read_csv(verification_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validate(scores, verification, args.require_20_seeds)

    print("=" * 78)
    print("AURSAD POWER / PRECISION ANALYSIS")
    print("=" * 78)
    print(f"Rows:        {len(scores):,}")
    print(f"Detectors:   {sorted(scores['detector'].unique().tolist())}")
    print(f"N values:    {sorted(scores['commissioning_size'].unique().tolist())}")
    print(f"Seeds:       {sorted(scores['seed'].unique().tolist())}")
    print(f"Power reps:  {args.power_reps:,}")
    print(f"Shift grid:  {list(args.shift_grid)}")
    print()

    counts = fault_counts(scores)
    recall_by_run = fault_recall_precision_by_run(scores, args.confidence)
    recall_summary = fault_recall_precision_summary(
        recall_by_run, args.bootstrap_reps, args.confidence
    )
    observed_shift = healthy_shift_observed_by_run(scores)

    print("Estimating KS power curves...")
    power_curve = ks_power_curve(
        scores,
        args.shift_grid,
        args.power_reps,
        args.test_alpha,
    )
    power_summary = ks_power_summary(
        power_curve,
        args.bootstrap_reps,
        args.confidence,
    )
    detectable = detectable_shift_summary(power_summary, args.target_power)

    # Sanity check: delta=0 is now a genuine null. With finite Monte Carlo error,
    # allow a conservative tolerance around the requested alpha. A large deviation
    # indicates the simulation is not calibrated and detectable-shift claims must
    # not be reported.
    null_rows = power_summary[
        np.isclose(power_summary["standardized_location_shift"], 0.0)
    ]
    null_tolerance = max(
        0.02,
        4.0 * math.sqrt(
            args.test_alpha * (1.0 - args.test_alpha) / args.power_reps
        ),
    )
    bad_null = null_rows[
        (null_rows["mean_estimated_power"] - args.test_alpha).abs() > null_tolerance
    ]
    if not bad_null.empty:
        raise RuntimeError(
            "KS power simulation failed null-calibration check. "
            f"Expected rejection rate near alpha={args.test_alpha:.3f} within "
            f"±{null_tolerance:.3f}. Problem rows:\n"
            + bad_null.to_string(index=False)
        )

    outputs = {
        "fault_counts": output_dir / "01_fault_counts.csv",
        "fault_recall_precision_by_run": output_dir / "02_fault_recall_precision_by_run.csv",
        "fault_recall_precision_summary": output_dir / "03_fault_recall_precision_summary.csv",
        "healthy_shift_observed_by_run": output_dir / "04_healthy_shift_observed_by_run.csv",
        "ks_power_curve_by_run": output_dir / "05_ks_power_curve_by_run.csv",
        "ks_power_summary": output_dir / "06_ks_power_summary.csv",
        "detectable_shift_summary": output_dir / "07_detectable_shift_summary.csv",
        "report": output_dir / "power_precision_report.json",
    }

    atomic_csv(counts, outputs["fault_counts"])
    atomic_csv(recall_by_run, outputs["fault_recall_precision_by_run"])
    atomic_csv(recall_summary, outputs["fault_recall_precision_summary"])
    atomic_csv(observed_shift, outputs["healthy_shift_observed_by_run"])
    atomic_csv(power_curve, outputs["ks_power_curve_by_run"])
    atomic_csv(power_summary, outputs["ks_power_summary"])
    atomic_csv(detectable, outputs["detectable_shift_summary"])

    report = {
        "version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset": "AURSAD",
            "analysis_type": "secondary_power_precision_diagnostic",
            "primary_benchmark_modified": False,
            "required_seed_set": list(range(20)) if args.require_20_seeds else None,
            "confidence": args.confidence,
            "bootstrap_replicates": args.bootstrap_reps,
            "ks_test_alpha": args.test_alpha,
            "ks_power_replicates_per_run_shift": args.power_reps,
            "target_power": args.target_power,
            "standardized_location_shift_grid": list(args.shift_grid),
        },
        "input_integrity": {
            "episode_scores_path": str(scores_path),
            "episode_scores_sha256": sha256_file(scores_path),
            "verification_path": str(verification_path),
            "verification_sha256": sha256_file(verification_path),
            "all_verification_rows_pass": True,
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
        },
        "fault_counts": counts.to_dict(orient="records"),
        "detectable_shift_summary": detectable.to_dict(orient="records"),
        "interpretation": {
            "fault_precision": (
                "Per-run Wilson intervals quantify finite anomaly-count precision. "
                "Across-seed bootstrap intervals quantify variability of mean recall "
                "over the 20 frozen commissioning seeds."
            ),
            "healthy_shift_power": (
                "KS power is estimated from a calibrated empirical null in which both simulated "
                "groups are resampled from the same pooled healthy distribution; a controlled "
                "location shift is then injected into the evaluation group. This does not "
                "measure power for every possible distributional shift shape."
            ),
        },
        "limitations": [
            "The KS power result is specific to location-shift alternatives.",
            "The same fixed anomaly evaluation episodes are scored under multiple seeds, so across-seed recall values are not independent draws of new fault episodes.",
            "Wilson intervals describe within-run binomial precision conditional on the fixed anomaly evaluation episodes.",
            "Bootstrap intervals across seeds describe commissioning-model variability, not new-dataset sampling uncertainty.",
            "This diagnostic remains secondary to the frozen 20-seed primary commissioning benchmark.",
        ],
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    outputs["report"].write_text(
        json.dumps(json_safe(report), indent=2),
        encoding="utf-8",
    )

    print()
    print("Fault counts:")
    print(counts.to_string(index=False))
    print()
    print("Detectable standardized healthy location shifts:")
    print(detectable.to_string(index=False))
    print()
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()