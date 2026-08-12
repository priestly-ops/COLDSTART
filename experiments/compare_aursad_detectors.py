#!/usr/bin/env python3
"""Compare frozen AURSAD commissioning results across detectors.

This script is intentionally read-only with respect to detector result folders.
It discovers each detector's seed-level and summary CSV files, validates the
seed/N grid, recomputes aggregate metrics and confidence intervals, verifies
reported summaries when possible, and writes one publication-ready comparison.

Primary detectors:
    - TargetOnly
    - Isolation Forest
    - Euclidean conformal k-NN
    - PAKCT

Typical usage (from the repository root):

    python experiments/compare_aursad_detectors.py

Allow a still-running PAKCT experiment while comparing completed methods:

    python experiments/compare_aursad_detectors.py --allow-incomplete

Fail unless every detector has the full 6 x 20 grid:

    python experiments/compare_aursad_detectors.py --require-complete

The script never edits files under outputs/aursad/<detector>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("compare_aursad_detectors")
GLOBAL_SEED = 42
DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))


@dataclass(frozen=True)
class DetectorSpec:
    key: str
    display_name: str
    directory_candidates: tuple[str, ...]
    seed_file_candidates: tuple[str, ...]
    summary_file_candidates: tuple[str, ...]
    n_star_file_candidates: tuple[str, ...]
    manifest_file_candidates: tuple[str, ...]


DETECTORS: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        key="targetonly",
        display_name="TargetOnly",
        directory_candidates=("targetonly", "target_only"),
        seed_file_candidates=(
            "targetonly_seed_results.csv",
            "target_only_seed_results.csv",
            "seed_results.csv",
        ),
        summary_file_candidates=(
            "targetonly_summary.csv",
            "target_only_summary.csv",
            "summary.csv",
        ),
        n_star_file_candidates=(
            "targetonly_n_star.json",
            "target_only_n_star.json",
            "n_star.json",
        ),
        manifest_file_candidates=(
            "targetonly_run_manifest.json",
            "target_only_run_manifest.json",
            "run_manifest.json",
        ),
    ),
    DetectorSpec(
        key="isolation_forest",
        display_name="Isolation Forest",
        directory_candidates=("isolation_forest", "isolationforest", "iforest"),
        seed_file_candidates=(
            "isolation_forest_seed_results.csv",
            "isolationforest_seed_results.csv",
            "seed_results.csv",
        ),
        summary_file_candidates=(
            "isolation_forest_summary.csv",
            "isolationforest_summary.csv",
            "summary.csv",
        ),
        n_star_file_candidates=(
            "isolation_forest_n_star.json",
            "isolationforest_n_star.json",
            "n_star.json",
        ),
        manifest_file_candidates=(
            "isolation_forest_run_manifest.json",
            "isolationforest_run_manifest.json",
            "run_manifest.json",
        ),
    ),
    DetectorSpec(
        key="euclidean_conformal_knn",
        display_name="Euclidean conformal k-NN",
        directory_candidates=(
            "euclidean_conformal_knn",
            "euclidean_knn",
            "unaligned_conformal_knn",
        ),
        seed_file_candidates=(
            "euclidean_knn_seed_results.csv",
            "euclidean_conformal_knn_seed_results.csv",
            "seed_results.csv",
        ),
        summary_file_candidates=(
            "euclidean_knn_summary.csv",
            "euclidean_conformal_knn_summary.csv",
            "summary.csv",
        ),
        n_star_file_candidates=(
            "euclidean_knn_n_star.json",
            "euclidean_conformal_knn_n_star.json",
            "n_star.json",
        ),
        manifest_file_candidates=(
            "euclidean_knn_run_manifest.json",
            "euclidean_conformal_knn_run_manifest.json",
            "run_manifest.json",
        ),
    ),
    DetectorSpec(
        key="pakct",
        display_name="PAKCT",
        directory_candidates=("pakct",),
        seed_file_candidates=("pakct_seed_results.csv", "seed_results.csv"),
        summary_file_candidates=("pakct_summary.csv", "summary.csv"),
        n_star_file_candidates=("pakct_n_star.json", "n_star.json"),
        manifest_file_candidates=("pakct_run_manifest.json", "run_manifest.json"),
    ),
)


COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "commissioning_size": ("commissioning_size", "n", "N", "commissioning_n"),
    "seed": ("seed", "random_seed", "split_seed"),
    "recall": ("recall", "anomaly_recall", "tpr", "true_positive_rate"),
    "fpr": ("fpr", "false_positive_rate", "false_alarm_rate"),
    "auroc": ("auroc", "roc_auc", "auc", "auc_roc"),
    "success": ("success", "joint_success", "meets_constraints"),
    "runtime_seconds": (
        "runtime_seconds",
        "elapsed_seconds",
        "duration_seconds",
        "wall_time_seconds",
        "runtime_sec",
    ),
    "threshold": ("threshold", "anomaly_threshold", "calibration_threshold"),
}

SUMMARY_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "commissioning_size": COLUMN_ALIASES["commissioning_size"],
    "recall_mean": ("recall_mean", "mean_recall"),
    "recall_ci_lower": ("recall_ci_lower", "recall_lower", "recall_ci_low"),
    "recall_ci_upper": ("recall_ci_upper", "recall_upper", "recall_ci_high"),
    "fpr_mean": ("fpr_mean", "mean_fpr", "false_positive_rate_mean"),
    "fpr_ci_lower": ("fpr_ci_lower", "fpr_lower", "fpr_ci_low"),
    "fpr_ci_upper": ("fpr_ci_upper", "fpr_upper", "fpr_ci_high"),
    "auroc_mean": ("auroc_mean", "mean_auroc", "roc_auc_mean", "auc_mean"),
    "auroc_ci_lower": ("auroc_ci_lower", "auroc_lower", "auroc_ci_low"),
    "auroc_ci_upper": ("auroc_ci_upper", "auroc_upper", "auroc_ci_high"),
    "success_rate": ("success_rate", "joint_success_rate"),
    "number_of_seeds": ("number_of_seeds", "n_seeds", "seed_count"),
}


@dataclass
class DetectorAudit:
    detector: str
    status: str
    result_directory: str | None
    seed_results_file: str | None
    reported_summary_file: str | None
    reported_n_star_file: str | None
    manifest_file: str | None
    observed_rows: int
    observed_commissioning_sizes: list[int]
    observed_seeds: list[int]
    expected_rows: int
    missing_pairs: list[str]
    unexpected_pairs: list[str]
    duplicate_pairs: list[str]
    complete: bool
    summary_matches: bool | None
    summary_mismatches: list[str]
    reported_n_star: str | None
    recomputed_n_star: str | None
    runtime_total_seconds: float | None
    runtime_median_seconds: float | None
    seed_results_sha256: str | None
    warnings: list[str]


class ComparisonError(RuntimeError):
    """Raised when result validation cannot continue safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/aursad"),
        help="Root containing one result folder per detector (default: outputs/aursad).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aursad/comparison"),
        help="Directory for comparison artifacts.",
    )
    parser.add_argument(
        "--commissioning-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_GRID),
        help="Expected commissioning sizes.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Expected random seeds.",
    )
    parser.add_argument("--recall-target", type=float, default=0.90)
    parser.add_argument("--fpr-budget", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Include available rows for unfinished detectors and mark them partial.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every discovered detector has the full expected grid.",
    )
    parser.add_argument(
        "--strict-summary-check",
        action="store_true",
        help="Fail when a detector's saved summary differs from recomputation.",
    )
    parser.add_argument(
        "--summary-tolerance",
        type=float,
        default=5e-4,
        help="Absolute tolerance for checking saved summaries (default: 5e-4).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()

    if args.allow_incomplete and args.require_complete:
        parser.error("--allow-incomplete and --require-complete are mutually exclusive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must lie strictly between 0 and 1")
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    if not args.commissioning_grid:
        parser.error("--commissioning-grid cannot be empty")
    if not args.seeds:
        parser.error("--seeds cannot be empty")
    return args


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_first_existing(directory: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    return None


def resolve_detector_directory(results_root: Path, spec: DetectorSpec) -> Path | None:
    for name in spec.directory_candidates:
        candidate = results_root / name
        if candidate.is_dir():
            return candidate
    return None


def resolve_column(df: pd.DataFrame, canonical: str, aliases: Mapping[str, tuple[str, ...]]) -> str | None:
    for candidate in aliases[canonical]:
        if candidate in df.columns:
            return candidate
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for candidate in aliases[canonical]:
        original = lower_to_original.get(candidate.lower())
        if original is not None:
            return original
    return None


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).ne(0.0)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "pass": True,
        "fail": False,
    }
    unknown = sorted(set(normalized.dropna()) - set(mapping))
    if unknown:
        raise ComparisonError(f"Unrecognized boolean values: {unknown[:10]}")
    return normalized.map(mapping).astype(bool)


def canonicalize_seed_results(
    raw: pd.DataFrame,
    detector_name: str,
    recall_target: float,
    fpr_budget: float,
) -> pd.DataFrame:
    if raw.empty:
        raise ComparisonError(f"{detector_name}: seed results are empty")

    required = ("commissioning_size", "seed", "recall", "fpr")
    found = {name: resolve_column(raw, name, COLUMN_ALIASES) for name in COLUMN_ALIASES}
    missing = [name for name in required if found[name] is None]
    if missing:
        raise ComparisonError(
            f"{detector_name}: seed results lack required columns {missing}; "
            f"available={list(raw.columns)}"
        )

    frame = pd.DataFrame(index=raw.index)
    frame["detector"] = detector_name
    frame["commissioning_size"] = pd.to_numeric(raw[found["commissioning_size"]], errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(raw[found["seed"]], errors="raise").astype(int)
    frame["recall"] = pd.to_numeric(raw[found["recall"]], errors="raise").astype(float)
    frame["fpr"] = pd.to_numeric(raw[found["fpr"]], errors="raise").astype(float)

    for metric in ("auroc", "runtime_seconds", "threshold"):
        column = found[metric]
        frame[metric] = (
            pd.to_numeric(raw[column], errors="coerce").astype(float)
            if column is not None
            else np.nan
        )

    success_column = found["success"]
    recomputed_success = (frame["recall"] >= recall_target) & (frame["fpr"] <= fpr_budget)
    if success_column is None:
        frame["success"] = recomputed_success
    else:
        reported = coerce_bool(raw[success_column])
        mismatch = reported.ne(recomputed_success)
        if mismatch.any():
            examples = frame.loc[mismatch, ["commissioning_size", "seed", "recall", "fpr"]].head(5)
            raise ComparisonError(
                f"{detector_name}: {int(mismatch.sum())} success flags disagree with the frozen "
                f"criterion recall>={recall_target}, fpr<={fpr_budget}. Examples:\n{examples}"
            )
        frame["success"] = reported

    finite_required = np.isfinite(frame[["recall", "fpr"]].to_numpy()).all(axis=1)
    if not finite_required.all():
        bad = frame.loc[~finite_required, ["commissioning_size", "seed", "recall", "fpr"]]
        raise ComparisonError(f"{detector_name}: non-finite required metrics:\n{bad.head(10)}")

    for metric in ("recall", "fpr"):
        invalid = ~frame[metric].between(0.0, 1.0, inclusive="both")
        if invalid.any():
            raise ComparisonError(f"{detector_name}: {metric} values outside [0,1]")
    finite_auc = frame["auroc"].notna()
    if finite_auc.any() and not frame.loc[finite_auc, "auroc"].between(0.0, 1.0).all():
        raise ComparisonError(f"{detector_name}: AUROC values outside [0,1]")

    return frame.sort_values(["commissioning_size", "seed"], kind="stable").reset_index(drop=True)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    confidence: float,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, mean, mean

    # Chunking prevents excessive memory use if the seed count or replicate
    # count is increased in future experiments.
    bootstrap_means: list[np.ndarray] = []
    remaining = replicates
    max_chunk = 20_000
    while remaining > 0:
        chunk = min(max_chunk, remaining)
        indices = rng.integers(0, array.size, size=(chunk, array.size))
        bootstrap_means.append(array[indices].mean(axis=1))
        remaining -= chunk
    samples = np.concatenate(bootstrap_means)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return mean, float(lower), float(upper)


def aggregate_seed_results(
    frame: pd.DataFrame,
    confidence: float,
    bootstrap_replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for commissioning_size, group in frame.groupby("commissioning_size", sort=True):
        # A stable group-specific seed makes output independent of detector
        # discovery order.
        seed_material = f"{frame['detector'].iloc[0]}:{int(commissioning_size)}:{random_seed}".encode()
        stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        rng = np.random.default_rng(stable_seed)

        recall_mean, recall_low, recall_high = bootstrap_mean_ci(
            group["recall"], confidence, bootstrap_replicates, rng
        )
        fpr_mean, fpr_low, fpr_high = bootstrap_mean_ci(
            group["fpr"], confidence, bootstrap_replicates, rng
        )
        auroc_mean, auroc_low, auroc_high = bootstrap_mean_ci(
            group["auroc"], confidence, bootstrap_replicates, rng
        )
        runtime_mean, runtime_low, runtime_high = bootstrap_mean_ci(
            group["runtime_seconds"], confidence, bootstrap_replicates, rng
        )

        rows.append(
            {
                "detector": group["detector"].iloc[0],
                "commissioning_size": int(commissioning_size),
                "recall_mean": recall_mean,
                "recall_ci_lower": recall_low,
                "recall_ci_upper": recall_high,
                "fpr_mean": fpr_mean,
                "fpr_ci_lower": fpr_low,
                "fpr_ci_upper": fpr_high,
                "auroc_mean": auroc_mean,
                "auroc_ci_lower": auroc_low,
                "auroc_ci_upper": auroc_high,
                "success_rate": float(group["success"].mean()),
                "runtime_seconds_mean": runtime_mean,
                "runtime_seconds_ci_lower": runtime_low,
                "runtime_seconds_ci_upper": runtime_high,
                "number_of_seeds": int(group["seed"].nunique()),
                "number_of_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def format_n_star(value: int | None, max_requested_n: int) -> str:
    return str(int(value)) if value is not None else f"Censored (>{max_requested_n})"


def compute_n_star(
    summary: pd.DataFrame,
    recall_target: float,
    fpr_budget: float,
    max_requested_n: int,
) -> str:
    eligible = summary.loc[
        (summary["recall_ci_lower"] >= recall_target)
        & (summary["fpr_ci_upper"] <= fpr_budget)
    ]
    n_star = None if eligible.empty else int(eligible["commissioning_size"].min())
    return format_n_star(n_star, max_requested_n)


def canonicalize_reported_summary(raw: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=raw.index)
    for canonical in SUMMARY_COLUMN_ALIASES:
        column = resolve_column(raw, canonical, SUMMARY_COLUMN_ALIASES)
        if column is not None:
            output[canonical] = pd.to_numeric(raw[column], errors="coerce")
    if "commissioning_size" not in output.columns:
        raise ComparisonError("Saved summary lacks a commissioning-size column")
    output["commissioning_size"] = output["commissioning_size"].astype(int)
    return output


def compare_reported_summary(
    recomputed: pd.DataFrame,
    reported_path: Path | None,
    tolerance: float,
) -> tuple[bool | None, list[str]]:
    if reported_path is None:
        return None, ["No saved summary file found"]
    try:
        reported = canonicalize_reported_summary(pd.read_csv(reported_path))
    except Exception as exc:  # defensive: audit should report, not hide, malformed files
        return False, [f"Could not parse saved summary: {exc}"]

    joined = recomputed.merge(reported, on="commissioning_size", how="outer", suffixes=("_new", "_saved"), indicator=True)
    mismatches: list[str] = []
    missing_rows = joined.loc[joined["_merge"] != "both", ["commissioning_size", "_merge"]]
    for row in missing_rows.itertuples(index=False):
        mismatches.append(f"N={row.commissioning_size}: row present in {row._1}")

    comparable_metrics = (
        "recall_mean",
        "recall_ci_lower",
        "recall_ci_upper",
        "fpr_mean",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "auroc_mean",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "success_rate",
        "number_of_seeds",
    )
    both = joined.loc[joined["_merge"] == "both"]
    for metric in comparable_metrics:
        new_column = f"{metric}_new"
        saved_column = f"{metric}_saved"
        if new_column not in both.columns or saved_column not in both.columns:
            continue
        for row in both[["commissioning_size", new_column, saved_column]].itertuples(index=False, name=None):
            n, new_value, saved_value = row
            if pd.isna(new_value) and pd.isna(saved_value):
                continue
            if pd.isna(new_value) != pd.isna(saved_value) or not np.isclose(
                float(new_value), float(saved_value), atol=tolerance, rtol=0.0
            ):
                mismatches.append(
                    f"N={int(n)} {metric}: recomputed={new_value!r}, saved={saved_value!r}"
                )
    return len(mismatches) == 0, mismatches


def extract_n_star_from_json(path: Path | None, max_requested_n: int) -> str | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not parse %s: %s", path, exc)
        return None

    candidates: list[Any] = []
    if isinstance(payload, Mapping):
        for key in ("n_star", "N_star", "n*", "estimate", "value"):
            if key in payload:
                candidates.append(payload[key])
        # Some runners store one mapping per detector.
        for value in payload.values():
            if isinstance(value, Mapping):
                for key in ("n_star", "N_star", "estimate", "value"):
                    if key in value:
                        candidates.append(value[key])
    else:
        candidates.append(payload)

    for value in candidates:
        if value is None:
            return f"Censored (>{max_requested_n})"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, float) and np.isfinite(value):
            return str(int(value))
        text = str(value).strip()
        if not text:
            continue
        if re.search(r"censor|>\s*\d+|unmet|none", text, flags=re.IGNORECASE):
            return f"Censored (>{max_requested_n})"
        match = re.search(r"\d+", text)
        if match:
            return str(int(match.group()))
    return None


def validate_grid(
    frame: pd.DataFrame,
    expected_grid: Sequence[int],
    expected_seeds: Sequence[int],
) -> tuple[list[str], list[str], list[str]]:
    observed_pairs = list(zip(frame["commissioning_size"], frame["seed"]))
    counts = pd.Series(observed_pairs).value_counts()
    duplicates = [f"N={n},seed={seed},count={int(count)}" for (n, seed), count in counts.items() if count > 1]

    expected = {(int(n), int(seed)) for n in expected_grid for seed in expected_seeds}
    observed = set(observed_pairs)
    missing = [f"N={n},seed={seed}" for n, seed in sorted(expected - observed)]
    unexpected = [f"N={n},seed={seed}" for n, seed in sorted(observed - expected)]
    return missing, unexpected, duplicates


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_write_json(payload: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    return value


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    generated = (
        "aursad_detector_comparison_by_n.csv",
        "aursad_detector_comparison_overall.csv",
        "aursad_n_star_comparison.csv",
        "aursad_combined_seed_results.csv",
        "aursad_comparison_audit.json",
    )
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        names = ", ".join(str(item) for item in existing)
        raise ComparisonError(f"Output files already exist: {names}. Pass --overwrite to replace them.")


def build_overall_table(
    by_n: pd.DataFrame,
    audits: Sequence[DetectorAudit],
) -> pd.DataFrame:
    audit_by_name = {audit.detector: audit for audit in audits}
    rows: list[dict[str, Any]] = []
    for detector, group in by_n.groupby("detector", sort=False):
        audit = audit_by_name[detector]
        # A compact detector-level row uses the largest completed N, which is
        # the most informative single commissioning point. Full curves remain
        # available in the by-N table.
        largest_n = int(group["commissioning_size"].max())
        row = group.loc[group["commissioning_size"] == largest_n].iloc[0]
        rows.append(
            {
                "detector": detector,
                "evaluation_status": audit.status,
                "largest_completed_n": largest_n,
                "mean_recall": row["recall_mean"],
                "recall_ci_lower": row["recall_ci_lower"],
                "recall_ci_upper": row["recall_ci_upper"],
                "mean_fpr": row["fpr_mean"],
                "fpr_ci_lower": row["fpr_ci_lower"],
                "fpr_ci_upper": row["fpr_ci_upper"],
                "mean_auroc": row["auroc_mean"],
                "auroc_ci_lower": row["auroc_ci_lower"],
                "auroc_ci_upper": row["auroc_ci_upper"],
                "seed_success_rate": row["success_rate"],
                "n_star": audit.recomputed_n_star,
                "completed_seed_n_runs": audit.observed_rows,
                "expected_seed_n_runs": audit.expected_rows,
                "runtime_total_seconds": audit.runtime_total_seconds,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    np.random.seed(GLOBAL_SEED)
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    expected_grid = sorted(set(int(value) for value in args.commissioning_grid))
    expected_seeds = sorted(set(int(value) for value in args.seeds))
    expected_rows = len(expected_grid) * len(expected_seeds)
    max_requested_n = max(expected_grid)

    if not results_root.is_dir():
        raise ComparisonError(f"Results root does not exist: {results_root}")
    prepare_output_directory(output_dir, args.overwrite)

    combined_seed_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    audits: list[DetectorAudit] = []

    for spec in DETECTORS:
        LOGGER.info("Checking %s...", spec.display_name)
        directory = resolve_detector_directory(results_root, spec)
        if directory is None:
            warning = f"No result directory found under {results_root}"
            LOGGER.warning("%s: %s", spec.display_name, warning)
            audits.append(
                DetectorAudit(
                    detector=spec.display_name,
                    status="missing",
                    result_directory=None,
                    seed_results_file=None,
                    reported_summary_file=None,
                    reported_n_star_file=None,
                    manifest_file=None,
                    observed_rows=0,
                    observed_commissioning_sizes=[],
                    observed_seeds=[],
                    expected_rows=expected_rows,
                    missing_pairs=[f"N={n},seed={seed}" for n in expected_grid for seed in expected_seeds],
                    unexpected_pairs=[],
                    duplicate_pairs=[],
                    complete=False,
                    summary_matches=None,
                    summary_mismatches=[],
                    reported_n_star=None,
                    recomputed_n_star=None,
                    runtime_total_seconds=None,
                    runtime_median_seconds=None,
                    seed_results_sha256=None,
                    warnings=[warning],
                )
            )
            continue

        seed_path = find_first_existing(directory, spec.seed_file_candidates)
        summary_path = find_first_existing(directory, spec.summary_file_candidates)
        n_star_path = find_first_existing(directory, spec.n_star_file_candidates)
        manifest_path = find_first_existing(directory, spec.manifest_file_candidates)
        if seed_path is None:
            warning = f"Result directory exists but no seed-level CSV was found: {directory}"
            LOGGER.warning("%s", warning)
            audits.append(
                DetectorAudit(
                    detector=spec.display_name,
                    status="missing_seed_results",
                    result_directory=str(directory),
                    seed_results_file=None,
                    reported_summary_file=str(summary_path) if summary_path else None,
                    reported_n_star_file=str(n_star_path) if n_star_path else None,
                    manifest_file=str(manifest_path) if manifest_path else None,
                    observed_rows=0,
                    observed_commissioning_sizes=[],
                    observed_seeds=[],
                    expected_rows=expected_rows,
                    missing_pairs=[f"N={n},seed={seed}" for n in expected_grid for seed in expected_seeds],
                    unexpected_pairs=[],
                    duplicate_pairs=[],
                    complete=False,
                    summary_matches=None,
                    summary_mismatches=[],
                    reported_n_star=extract_n_star_from_json(n_star_path, max_requested_n),
                    recomputed_n_star=None,
                    runtime_total_seconds=None,
                    runtime_median_seconds=None,
                    seed_results_sha256=None,
                    warnings=[warning],
                )
            )
            continue

        frame = canonicalize_seed_results(
            pd.read_csv(seed_path),
            spec.display_name,
            args.recall_target,
            args.fpr_budget,
        )
        missing, unexpected, duplicates = validate_grid(frame, expected_grid, expected_seeds)
        complete = not missing and not unexpected and not duplicates and len(frame) == expected_rows
        status = "complete" if complete else "partial"
        warnings: list[str] = []
        if duplicates:
            # Duplicates make aggregate statistics ambiguous and can indicate
            # mixed smoke/full runs; never silently deduplicate them.
            raise ComparisonError(
                f"{spec.display_name}: duplicate (N, seed) rows detected. "
                f"Possible mixed smoke/full output: {duplicates[:10]}"
            )
        if unexpected:
            warnings.append(f"Unexpected grid pairs: {len(unexpected)}")
        if missing:
            warnings.append(f"Missing expected grid pairs: {len(missing)}")

        if not complete and not args.allow_incomplete:
            LOGGER.warning(
                "%s is partial (%d/%d expected rows); excluding it. Pass --allow-incomplete to include checkpoints.",
                spec.display_name,
                len(frame),
                expected_rows,
            )
            audits.append(
                DetectorAudit(
                    detector=spec.display_name,
                    status="partial_excluded",
                    result_directory=str(directory),
                    seed_results_file=str(seed_path),
                    reported_summary_file=str(summary_path) if summary_path else None,
                    reported_n_star_file=str(n_star_path) if n_star_path else None,
                    manifest_file=str(manifest_path) if manifest_path else None,
                    observed_rows=int(len(frame)),
                    observed_commissioning_sizes=sorted(frame["commissioning_size"].unique().astype(int).tolist()),
                    observed_seeds=sorted(frame["seed"].unique().astype(int).tolist()),
                    expected_rows=expected_rows,
                    missing_pairs=missing,
                    unexpected_pairs=unexpected,
                    duplicate_pairs=duplicates,
                    complete=False,
                    summary_matches=None,
                    summary_mismatches=[],
                    reported_n_star=extract_n_star_from_json(n_star_path, max_requested_n),
                    recomputed_n_star=None,
                    runtime_total_seconds=float(frame["runtime_seconds"].sum()) if frame["runtime_seconds"].notna().any() else None,
                    runtime_median_seconds=float(frame["runtime_seconds"].median()) if frame["runtime_seconds"].notna().any() else None,
                    seed_results_sha256=sha256_file(seed_path),
                    warnings=warnings,
                )
            )
            continue

        summary = aggregate_seed_results(
            frame,
            args.confidence,
            args.bootstrap_replicates,
            GLOBAL_SEED,
        )
        summary["evaluation_status"] = status
        n_star = compute_n_star(summary, args.recall_target, args.fpr_budget, max_requested_n)
        reported_n_star = extract_n_star_from_json(n_star_path, max_requested_n)
        summary_matches, summary_mismatches = compare_reported_summary(
            summary,
            summary_path,
            args.summary_tolerance,
        )
        if summary_matches is False:
            warnings.append(f"Saved summary mismatch count: {len(summary_mismatches)}")
            LOGGER.warning("%s: saved summary differs from recomputation", spec.display_name)
            if args.strict_summary_check:
                raise ComparisonError(
                    f"{spec.display_name}: saved summary mismatch:\n" + "\n".join(summary_mismatches[:20])
                )
        if reported_n_star is not None and reported_n_star != n_star:
            warnings.append(f"Reported N*={reported_n_star}, recomputed N*={n_star}")
            LOGGER.warning(
                "%s: reported N* (%s) differs from recomputed N* (%s)",
                spec.display_name,
                reported_n_star,
                n_star,
            )

        runtime_values = frame["runtime_seconds"].dropna()
        audit = DetectorAudit(
            detector=spec.display_name,
            status=status,
            result_directory=str(directory),
            seed_results_file=str(seed_path),
            reported_summary_file=str(summary_path) if summary_path else None,
            reported_n_star_file=str(n_star_path) if n_star_path else None,
            manifest_file=str(manifest_path) if manifest_path else None,
            observed_rows=int(len(frame)),
            observed_commissioning_sizes=sorted(frame["commissioning_size"].unique().astype(int).tolist()),
            observed_seeds=sorted(frame["seed"].unique().astype(int).tolist()),
            expected_rows=expected_rows,
            missing_pairs=missing,
            unexpected_pairs=unexpected,
            duplicate_pairs=duplicates,
            complete=complete,
            summary_matches=summary_matches,
            summary_mismatches=summary_mismatches,
            reported_n_star=reported_n_star,
            recomputed_n_star=n_star,
            runtime_total_seconds=float(runtime_values.sum()) if not runtime_values.empty else None,
            runtime_median_seconds=float(runtime_values.median()) if not runtime_values.empty else None,
            seed_results_sha256=sha256_file(seed_path),
            warnings=warnings,
        )
        audits.append(audit)
        combined_seed_frames.append(frame)
        summary_frames.append(summary)
        LOGGER.info(
            "%s: %s, rows=%d/%d, recomputed N*=%s",
            spec.display_name,
            status,
            len(frame),
            expected_rows,
            n_star,
        )

    included = [audit for audit in audits if audit.status in {"complete", "partial"}]
    if not included:
        raise ComparisonError("No detector had usable seed-level results")
    if args.require_complete:
        incomplete = [audit.detector for audit in audits if not audit.complete]
        if incomplete:
            raise ComparisonError("Incomplete or missing detectors: " + ", ".join(incomplete))

    combined_seed_results = pd.concat(combined_seed_frames, ignore_index=True)
    by_n = pd.concat(summary_frames, ignore_index=True)
    detector_order = [spec.display_name for spec in DETECTORS]
    by_n["detector"] = pd.Categorical(by_n["detector"], detector_order, ordered=True)
    by_n = by_n.sort_values(["detector", "commissioning_size"]).reset_index(drop=True)
    by_n["detector"] = by_n["detector"].astype(str)

    overall = build_overall_table(by_n, audits)
    order_map = {name: index for index, name in enumerate(detector_order)}
    overall["_order"] = overall["detector"].map(order_map)
    overall = overall.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    n_star_table = pd.DataFrame(
        [
            {
                "detector": audit.detector,
                "evaluation_status": audit.status,
                "reported_n_star": audit.reported_n_star,
                "recomputed_n_star": audit.recomputed_n_star,
                "complete": audit.complete,
            }
            for audit in audits
        ]
    )

    atomic_write_csv(combined_seed_results, output_dir / "aursad_combined_seed_results.csv")
    atomic_write_csv(by_n, output_dir / "aursad_detector_comparison_by_n.csv")
    atomic_write_csv(overall, output_dir / "aursad_detector_comparison_overall.csv")
    atomic_write_csv(n_star_table, output_dir / "aursad_n_star_comparison.csv")

    audit_payload = clean_json_value(
        {
            "schema_version": "aursad-detector-comparison-v1",
            "global_seed": GLOBAL_SEED,
            "results_root": str(results_root),
            "output_directory": str(output_dir),
            "commissioning_grid": expected_grid,
            "seeds": expected_seeds,
            "recall_target": args.recall_target,
            "fpr_budget": args.fpr_budget,
            "confidence": args.confidence,
            "bootstrap_replicates": args.bootstrap_replicates,
            "allow_incomplete": args.allow_incomplete,
            "detectors": [asdict(audit) for audit in audits],
        }
    )
    atomic_write_json(audit_payload, output_dir / "aursad_comparison_audit.json")

    print("\nAURSAD detector comparison (largest completed N per detector)\n")
    printable = overall.copy()
    for column in (
        "mean_recall",
        "recall_ci_lower",
        "recall_ci_upper",
        "mean_fpr",
        "fpr_ci_lower",
        "fpr_ci_upper",
        "mean_auroc",
        "auroc_ci_lower",
        "auroc_ci_upper",
        "seed_success_rate",
    ):
        printable[column] = printable[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    print(printable.to_string(index=False))
    print(f"\nWrote comparison artifacts to: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    try:
        return run(args)
    except ComparisonError as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception:
        LOGGER.exception("Unexpected failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())