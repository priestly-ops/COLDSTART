from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:  # scipy should normally exist in this project
    ks_2samp = None
    wasserstein_distance = None


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

DEFAULT_GRID = [10, 25, 50, 100, 250, 500]

DETECTOR_SPECS = {
    "targetonly": {
        "display": "TargetOnly",
        "folder": "targetonly",
        "prefixes": ["targetonly"],
    },
    "isolation_forest": {
        "display": "Isolation Forest",
        "folder": "isolation_forest",
        "prefixes": ["isolation_forest", "iforest", "isolationforest"],
    },
    "euclidean_conformal_knn": {
        "display": "Euclidean conformal k-NN",
        "folder": "euclidean_conformal_knn",
        "prefixes": ["euclidean_knn", "euclidean_conformal_knn", "knn"],
    },
    "pakct": {
        "display": "PAKCT",
        "folder": "pakct",
        "prefixes": ["pakct"],
    },
}

# Canonical AURSAD fault labels used only for ordering/pretty-printing.
# Unknown labels are preserved rather than dropped.
FAULT_ORDER_HINTS = [
    "damaged screw",
    "extra component",
    "missing screw",
    "damaged thread",
]

COLUMN_ALIASES = {
    "n": [
        "n", "N", "commissioning_n", "commissioning_size",
        "n_commissioning", "num_commissioning", "train_size",
    ],
    "seed": ["seed", "random_seed", "rng_seed"],
    "recall": ["recall", "anomaly_recall", "tpr", "sensitivity"],
    "fpr": ["fpr", "false_positive_rate", "false_alarm_rate", "false_alarm"],
    "auroc": ["auroc", "auc", "roc_auc", "roc_auc_score"],
    "auprc": ["auprc", "pr_auc", "average_precision"],
    "success": ["success", "joint_success", "passed", "meets_constraints"],
    "threshold": [
        "threshold", "conformal_threshold", "anomaly_threshold",
        "score_threshold", "decision_threshold",
    ],
    "runtime_seconds": [
        "runtime_seconds", "elapsed_seconds", "duration_seconds",
        "runtime_s", "elapsed_s",
    ],
    "class_label": [
        "fault_class", "anomaly_class", "class", "label_name", "fault",
        "anomaly_type", "class_label",
    ],
    "score": [
        "score", "anomaly_score", "nonconformity_score", "distance",
        "mahalanobis_score", "knn_score",
    ],
    "partition": [
        "partition", "split", "dataset_role", "role", "subset", "set",
    ],
    "episode_id": [
        "episode_id", "sample_nr", "execution_id", "cycle_id", "id",
    ],
    "is_anomaly": [
        "is_anomaly", "anomaly", "y_true", "target", "binary_label",
    ],
}


@dataclass
class DetectorAudit:
    detector: str
    result_directory: str
    seed_file: Optional[str]
    per_class_file: Optional[str]
    score_files: list[str]
    raw_score_mode: bool
    seed_rows: int
    score_rows: int
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose low recall / calibration behavior in frozen AURSAD results."
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root containing detector output folders. Default: <repo>/outputs/aursad",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostic tables and figures. Default: <repo>/outputs/aursad/diagnostics",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root. Default: parent of the experiments directory containing this script.",
    )
    p.add_argument(
        "--detectors",
        nargs="+",
        default=list(DETECTOR_SPECS.keys()),
        choices=list(DETECTOR_SPECS.keys()),
        help="Detector keys to analyze.",
    )
    p.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=DEFAULT_GRID,
        help="Commissioning sizes to include.",
    )
    p.add_argument(
        "--recall-target",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--fpr-budget",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--require-score-files",
        action="store_true",
        help="Fail if any requested detector lacks episode-level score files.",
    )
    p.add_argument(
        "--max-score-files",
        type=int,
        default=10000,
        help="Safety cap on recursively discovered score CSV files per detector.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing diagnostic outputs.",
    )
    return p.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        # Read-only inputs; outputs can be safely regenerated, but require explicit intent.
        existing = list(path.glob("*"))
        if existing:
            raise FileExistsError(
                f"{path} already contains diagnostic outputs. "
                "Use --overwrite to regenerate them."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)


def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, canonical: str, required: bool = False) -> Optional[str]:
    aliases = COLUMN_ALIASES[canonical]
    normalized = {normalize_name(c): c for c in df.columns}
    for alias in aliases:
        hit = normalized.get(normalize_name(alias))
        if hit is not None:
            return hit
    if required:
        raise KeyError(
            f"Could not locate required '{canonical}' column. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def canonicalize_columns(df: pd.DataFrame, wanted: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for canonical in wanted:
        col = find_column(out, canonical, required=False)
        if col is not None and col != canonical:
            rename[col] = canonical
    out = out.rename(columns=rename)
    return out


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def discover_seed_file(detector_dir: Path, prefixes: list[str]) -> Optional[Path]:
    candidates = []
    for prefix in prefixes:
        candidates.extend([
            detector_dir / f"{prefix}_seed_results.csv",
            detector_dir / f"{prefix}_results.csv",
        ])
    candidates.extend([
        detector_dir / "seed_results.csv",
        detector_dir / "results.csv",
    ])
    p = first_existing(candidates)
    if p:
        return p

    # Conservative fallback: exact-ish "seed_results" filename.
    files = sorted(detector_dir.glob("*seed*result*.csv"))
    return files[0] if files else None


def discover_per_class_file(detector_dir: Path, prefixes: list[str]) -> Optional[Path]:
    candidates = []
    for prefix in prefixes:
        candidates.extend([
            detector_dir / f"{prefix}_per_class_seed_results.csv",
            detector_dir / f"{prefix}_per_class_recall.csv",
        ])
    candidates.extend([
        detector_dir / "per_class_seed_results.csv",
        detector_dir / "per_class_recall.csv",
    ])
    p = first_existing(candidates)
    if p:
        return p
    files = sorted(detector_dir.glob("*per*class*.csv"))
    return files[0] if files else None


def discover_score_files(detector_dir: Path, max_files: int) -> list[Path]:
    """
    Look for true episode-level score artifacts.

    We deliberately exclude aggregate summary/seed files that happen to contain a
    threshold or score-like column.
    """
    hits = []
    for p in detector_dir.rglob("*.csv"):
        name = normalize_name(p.name)
        if "score" not in name:
            continue
        if any(bad in name for bad in ["summary", "seed_results", "per_class_recall"]):
            continue
        hits.append(p)
        if len(hits) >= max_files:
            break
    return sorted(set(hits))


def load_seed_results(path: Path, detector_name: str, n_values: set[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = canonicalize_columns(
        df,
        ["n", "seed", "recall", "fpr", "auroc", "auprc", "success",
         "threshold", "runtime_seconds"],
    )

    for required in ["n", "seed", "recall", "fpr"]:
        if required not in df.columns:
            raise KeyError(
                f"{path}: missing required seed-result column '{required}'. "
                f"Available: {list(df.columns)}"
            )

    for c in ["n", "seed", "recall", "fpr", "auroc", "auprc", "threshold", "runtime_seconds"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["n"].isin(n_values)].copy()
    df["detector"] = detector_name

    # Recompute success independently if absent.
    if "success" not in df.columns:
        df["success"] = np.nan

    return df


def load_per_class(path: Path, detector_name: str, n_values: set[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = canonicalize_columns(df, ["n", "seed", "class_label", "recall"])
    if "class_label" not in df.columns:
        # Some aggregate per-class files can have one class per column.
        id_cols = [c for c in ["n", "seed"] if c in df.columns]
        metric_cols = [
            c for c in df.columns
            if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        if metric_cols:
            df = df.melt(
                id_vars=id_cols,
                value_vars=metric_cols,
                var_name="class_label",
                value_name="recall",
            )

    needed = {"n", "class_label", "recall"}
    if not needed.issubset(df.columns):
        raise KeyError(
            f"{path}: cannot standardize per-class results. "
            f"Need {sorted(needed)}, got {list(df.columns)}."
        )

    for c in ["n", "seed", "recall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["n"].isin(n_values)].copy()
    df["detector"] = detector_name
    df["class_label"] = df["class_label"].astype(str)
    return df


def infer_partition(value: str) -> str:
    x = normalize_name(value)
    if "cal" in x and ("healthy" in x or "normal" in x or x in {"cal", "calibration"}):
        return "calibration_healthy"
    if "calibration" in x:
        return "calibration_healthy"
    if any(k in x for k in ["healthy_eval", "normal_eval", "evaluation_healthy", "eval_healthy"]):
        return "evaluation_healthy"
    if x in {"healthy", "normal"}:
        return "evaluation_healthy"
    if any(k in x for k in ["anomaly", "fault", "abnormal"]):
        return "anomaly_evaluation"
    return x


def load_score_file(
    path: Path,
    detector_name: str,
    n_values: set[int],
) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    df = canonicalize_columns(
        df,
        ["n", "seed", "score", "partition", "episode_id", "class_label",
         "is_anomaly", "threshold"],
    )
    if "score" not in df.columns:
        return None

    # N/seed may be encoded in path/filename.
    if "n" not in df.columns:
        text = str(path)
        m = re.search(r"(?:^|[^a-zA-Z])N[_=-]?(\d+)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"(?:^|[_-])n[_-]?(\d+)", path.name, re.IGNORECASE)
        if m:
            df["n"] = int(m.group(1))

    if "seed" not in df.columns:
        text = str(path)
        m = re.search(r"seed[_=-]?(\d+)", text, re.IGNORECASE)
        if m:
            df["seed"] = int(m.group(1))

    if "n" not in df.columns or "seed" not in df.columns:
        # Without N and seed, score distributions cannot be tied back to the frozen run.
        return None

    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["n"].isin(n_values)].copy()
    df = df[np.isfinite(df["score"])].copy()
    if df.empty:
        return None

    if "partition" in df.columns:
        df["partition"] = df["partition"].astype(str).map(infer_partition)
    else:
        df["partition"] = "unknown"

    if "class_label" not in df.columns:
        df["class_label"] = np.nan

    # If binary label exists, make anomaly role more explicit.
    if "is_anomaly" in df.columns:
        tmp = pd.to_numeric(df["is_anomaly"], errors="coerce")
        mask = tmp == 1
        df.loc[mask & (df["partition"] == "unknown"), "partition"] = "anomaly_evaluation"

    df["detector"] = detector_name
    df["source_file"] = str(path)
    return df


def load_all_scores(
    score_files: list[Path],
    detector_name: str,
    n_values: set[int],
) -> pd.DataFrame:
    frames = []
    for p in score_files:
        f = load_score_file(p, detector_name, n_values)
        if f is not None and not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Deduplicate exact repeated episode scores if the same artifact is mirrored.
    dedup_cols = [
        c for c in ["detector", "n", "seed", "episode_id", "partition", "class_label", "score"]
        if c in out.columns
    ]
    if dedup_cols:
        out = out.drop_duplicates(subset=dedup_cols)
    return out


def bootstrap_ci(values: np.ndarray, reps: int = 10000, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(GLOBAL_SEED)
    n = len(values)
    # Vectorized in chunks to avoid huge memory for large reps/n.
    means = np.empty(reps, dtype=float)
    chunk = 1000
    pos = 0
    while pos < reps:
        k = min(chunk, reps - pos)
        idx = rng.integers(0, n, size=(k, n))
        means[pos:pos+k] = values[idx].mean(axis=1)
        pos += k
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def summarize_seed_results(seed_df: pd.DataFrame, recall_target: float, fpr_budget: float) -> pd.DataFrame:
    rows = []
    if seed_df.empty:
        return pd.DataFrame()

    for (detector, n), g in seed_df.groupby(["detector", "n"], sort=True):
        recall = g["recall"].to_numpy(float)
        fpr = g["fpr"].to_numpy(float)
        rec_lo, rec_hi = bootstrap_ci(recall)
        fpr_lo, fpr_hi = bootstrap_ci(fpr)

        independent_success = (g["recall"] >= recall_target) & (g["fpr"] <= fpr_budget)
        row = {
            "detector": detector,
            "n": int(n),
            "number_of_seeds": int(len(g)),
            "recall_mean": float(np.nanmean(recall)),
            "recall_ci_lower": rec_lo,
            "recall_ci_upper": rec_hi,
            "fpr_mean": float(np.nanmean(fpr)),
            "fpr_ci_lower": fpr_lo,
            "fpr_ci_upper": fpr_hi,
            "joint_seed_success_rate": float(np.nanmean(independent_success.astype(float))),
        }
        if "auroc" in g.columns:
            row["auroc_mean"] = float(np.nanmean(g["auroc"]))
        if "auprc" in g.columns:
            row["auprc_mean"] = float(np.nanmean(g["auprc"]))
        if "threshold" in g.columns:
            vals = g["threshold"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row["threshold_mean"] = float(vals.mean()) if len(vals) else np.nan
            row["threshold_median"] = float(np.median(vals)) if len(vals) else np.nan
            row["threshold_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if "runtime_seconds" in g.columns:
            vals = g["runtime_seconds"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row["runtime_median_seconds"] = float(np.median(vals)) if len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_per_class(per_class_df: pd.DataFrame) -> pd.DataFrame:
    if per_class_df.empty:
        return pd.DataFrame()

    rows = []
    for (detector, n, cls), g in per_class_df.groupby(
        ["detector", "n", "class_label"], dropna=False, sort=True
    ):
        vals = pd.to_numeric(g["recall"], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "detector": detector,
            "n": int(n),
            "class_label": str(cls),
            "number_of_seeds": int(len(vals)),
            "recall_mean": float(vals.mean()),
            "recall_ci_lower": lo,
            "recall_ci_upper": hi,
            "recall_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def merge_thresholds_into_scores(score_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty or "threshold" not in seed_df.columns:
        return score_df
    threshold = seed_df[["detector", "n", "seed", "threshold"]].copy()
    threshold = threshold.dropna(subset=["threshold"])
    if threshold.empty:
        return score_df
    out = score_df.drop(columns=["threshold"], errors="ignore").merge(
        threshold,
        on=["detector", "n", "seed"],
        how="left",
        validate="many_to_one",
    )
    return out


def label_score_groups(score_df: pd.DataFrame) -> pd.DataFrame:
    out = score_df.copy()

    def row_group(row) -> str:
        partition = str(row.get("partition", "unknown"))
        cls = row.get("class_label", np.nan)
        cls_text = "" if pd.isna(cls) else str(cls).strip()

        if partition == "calibration_healthy":
            return "Calibration healthy"
        if partition == "evaluation_healthy":
            return "Evaluation healthy"
        if partition == "anomaly_evaluation":
            if cls_text:
                return cls_text
            return "Anomaly evaluation"
        # If class is populated and looks non-healthy, use it.
        if cls_text and normalize_name(cls_text) not in {"healthy", "normal", "0", "nan"}:
            return cls_text
        return partition

    out["score_group"] = out.apply(row_group, axis=1)
    return out


def summarize_score_distributions(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    score_df = label_score_groups(score_df)
    rows = []

    for (detector, n, group), g in score_df.groupby(
        ["detector", "n", "score_group"], sort=True
    ):
        vals = g["score"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        row = {
            "detector": detector,
            "n": int(n),
            "score_group": group,
            "n_scores": int(len(vals)),
            "score_mean": float(np.mean(vals)),
            "score_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "score_median": float(np.median(vals)),
            "score_q25": float(np.quantile(vals, 0.25)),
            "score_q75": float(np.quantile(vals, 0.75)),
            "score_q90": float(np.quantile(vals, 0.90)),
            "score_q95": float(np.quantile(vals, 0.95)),
            "score_q99": float(np.quantile(vals, 0.99)),
            "score_max": float(np.max(vals)),
        }
        if "threshold" in g.columns:
            th = g["threshold"].to_numpy(float)
            th = th[np.isfinite(th)]
            row["threshold_mean"] = float(th.mean()) if len(th) else np.nan
            if len(th):
                # Episode-level exceedance under each episode's own seed/N threshold.
                valid = np.isfinite(g["threshold"].to_numpy(float))
                row["threshold_exceedance_rate"] = float(
                    np.mean(
                        g.loc[valid, "score"].to_numpy(float)
                        > g.loc[valid, "threshold"].to_numpy(float)
                    )
                )
        rows.append(row)
    return pd.DataFrame(rows)


def healthy_shift_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()

    rows = []
    for (detector, n, seed), g in score_df.groupby(["detector", "n", "seed"], sort=True):
        cal = g.loc[g["partition"] == "calibration_healthy", "score"].to_numpy(float)
        healthy = g.loc[g["partition"] == "evaluation_healthy", "score"].to_numpy(float)
        cal = cal[np.isfinite(cal)]
        healthy = healthy[np.isfinite(healthy)]
        if len(cal) < 2 or len(healthy) < 2:
            continue

        row = {
            "detector": detector,
            "n": int(n),
            "seed": int(seed),
            "calibration_count": int(len(cal)),
            "healthy_eval_count": int(len(healthy)),
            "calibration_mean": float(np.mean(cal)),
            "healthy_eval_mean": float(np.mean(healthy)),
            "mean_shift": float(np.mean(healthy) - np.mean(cal)),
            "calibration_q99": float(np.quantile(cal, 0.99)),
            "healthy_eval_q99": float(np.quantile(healthy, 0.99)),
            "q99_shift": float(np.quantile(healthy, 0.99) - np.quantile(cal, 0.99)),
            "calibration_max": float(np.max(cal)),
            "healthy_eval_max": float(np.max(healthy)),
            "max_shift": float(np.max(healthy) - np.max(cal)),
        }
        if ks_2samp is not None:
            ks = ks_2samp(cal, healthy)
            row["ks_statistic"] = float(ks.statistic)
            row["ks_pvalue"] = float(ks.pvalue)
        if wasserstein_distance is not None:
            row["wasserstein_distance"] = float(wasserstein_distance(cal, healthy))
        rows.append(row)

    return pd.DataFrame(rows)


def anomaly_separation_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()

    score_df = label_score_groups(score_df)
    rows = []

    for (detector, n, seed), g in score_df.groupby(["detector", "n", "seed"], sort=True):
        healthy = g.loc[g["partition"] == "evaluation_healthy", "score"].to_numpy(float)
        healthy = healthy[np.isfinite(healthy)]
        if len(healthy) < 1:
            continue

        anomaly_groups = g[
            ~g["score_group"].isin(["Calibration healthy", "Evaluation healthy", "unknown"])
        ]
        for cls, cg in anomaly_groups.groupby("score_group", sort=True):
            anomaly = cg["score"].to_numpy(float)
            anomaly = anomaly[np.isfinite(anomaly)]
            if len(anomaly) == 0:
                continue

            # Pairwise probability P(anomaly_score > healthy_score) via ranks can be
            # expensive if done as full outer product, so use sorted search.
            hs = np.sort(healthy)
            greater_counts = np.searchsorted(hs, anomaly, side="left")
            probability_superiority = float(np.mean(greater_counts / len(hs)))

            row = {
                "detector": detector,
                "n": int(n),
                "seed": int(seed),
                "class_label": cls,
                "healthy_count": int(len(healthy)),
                "anomaly_count": int(len(anomaly)),
                "healthy_median": float(np.median(healthy)),
                "anomaly_median": float(np.median(anomaly)),
                "median_margin": float(np.median(anomaly) - np.median(healthy)),
                "healthy_q95": float(np.quantile(healthy, 0.95)),
                "anomaly_q25": float(np.quantile(anomaly, 0.25)),
                "q25_anomaly_minus_q95_healthy": float(
                    np.quantile(anomaly, 0.25) - np.quantile(healthy, 0.95)
                ),
                "probability_anomaly_score_gt_healthy": probability_superiority,
            }
            if wasserstein_distance is not None:
                row["wasserstein_distance"] = float(
                    wasserstein_distance(healthy, anomaly)
                )
            rows.append(row)

    return pd.DataFrame(rows)


def threshold_tail_summary(score_df: pd.DataFrame, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if not score_df.empty:
        for (detector, n, seed), g in score_df.groupby(["detector", "n", "seed"], sort=True):
            cal = g.loc[g["partition"] == "calibration_healthy", "score"].to_numpy(float)
            cal = cal[np.isfinite(cal)]
            if len(cal) == 0:
                continue

            threshold = np.nan
            if "threshold" in g.columns:
                th = g["threshold"].to_numpy(float)
                th = th[np.isfinite(th)]
                if len(th):
                    threshold = float(np.median(th))

            sorted_cal = np.sort(cal)
            max_score = float(sorted_cal[-1])
            second = float(sorted_cal[-2]) if len(sorted_cal) >= 2 else np.nan
            ratio = (
                max_score / max(abs(second), 1e-12)
                if np.isfinite(second) else np.nan
            )
            row = {
                "detector": detector,
                "n": int(n),
                "seed": int(seed),
                "n_calibration_scores": int(len(cal)),
                "threshold": threshold,
                "calibration_max": max_score,
                "calibration_second_max": second,
                "max_over_second_max": ratio,
                "threshold_minus_q99": (
                    threshold - float(np.quantile(cal, 0.99))
                    if np.isfinite(threshold) else np.nan
                ),
                "threshold_equals_calibration_max": bool(
                    np.isfinite(threshold) and np.isclose(threshold, max_score)
                ),
            }
            rows.append(row)

    # If no raw cal scores exist, preserve threshold trends from seed results.
    if not rows and "threshold" in seed_df.columns:
        for _, r in seed_df.iterrows():
            if not np.isfinite(r.get("threshold", np.nan)):
                continue
            rows.append({
                "detector": r["detector"],
                "n": int(r["n"]),
                "seed": int(r["seed"]),
                "n_calibration_scores": np.nan,
                "threshold": float(r["threshold"]),
                "calibration_max": np.nan,
                "calibration_second_max": np.nan,
                "max_over_second_max": np.nan,
                "threshold_minus_q99": np.nan,
                "threshold_equals_calibration_max": np.nan,
            })

    return pd.DataFrame(rows)


def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df.to_csv(path, index=False)


def plot_metric_vs_n(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    out: Path,
    reference: Optional[float] = None,
) -> None:
    if summary.empty or metric not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for detector, g in summary.groupby("detector", sort=False):
        g = g.sort_values("n")
        ax.plot(g["n"], g[metric], marker="o", label=detector)
    if reference is not None:
        ax.axhline(reference, linestyle="--", linewidth=1.2, label=f"Reference = {reference:g}")
    ax.set_xlabel("Commissioning size N")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["n"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recall_fpr(summary: pd.DataFrame, out: Path, recall_target: float, fpr_budget: float) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for detector, g in summary.groupby("detector", sort=False):
        g = g.sort_values("n")
        ax.plot(g["n"], g["recall_mean"], marker="o", label=f"{detector} recall")
        ax.plot(g["n"], g["fpr_mean"], marker="x", linestyle="--", label=f"{detector} FPR")
    ax.axhline(recall_target, linestyle=":", linewidth=1.2, label=f"Recall target {recall_target:.2f}")
    ax.axhline(fpr_budget, linestyle="-.", linewidth=1.2, label=f"FPR budget {fpr_budget:.3f}")
    ax.set_xlabel("Commissioning size N")
    ax.set_ylabel("Rate")
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["n"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, out: Path) -> None:
    if per_class.empty:
        return
    # Plot one line per detector/class; intentionally descriptive.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for (detector, cls), g in per_class.groupby(["detector", "class_label"], sort=False):
        g = g.sort_values("n")
        ax.plot(g["n"], g["recall_mean"], marker="o", label=f"{detector} — {cls}")
    ax.set_xlabel("Commissioning size N")
    ax.set_ylabel("Per-fault recall")
    ax.set_xscale("log")
    ax.set_xticks(sorted(per_class["n"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(score_df: pd.DataFrame, figures_dir: Path) -> None:
    if score_df.empty:
        return
    labeled = label_score_groups(score_df)

    for (detector, n), g in labeled.groupby(["detector", "n"], sort=True):
        groups = []
        for group_name, gg in g.groupby("score_group", sort=True):
            vals = gg["score"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                groups.append((group_name, vals))

        if len(groups) < 2:
            continue

        # Use histogram density with a common range.
        all_vals = np.concatenate([v for _, v in groups])
        lo, hi = np.quantile(all_vals, [0.001, 0.999])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        bins = np.linspace(lo, hi, 50) if hi > lo else 20

        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for group_name, vals in groups:
            clipped = vals[(vals >= lo) & (vals <= hi)]
            if len(clipped):
                ax.hist(
                    clipped,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    label=f"{group_name} (n={len(vals)})",
                )

        # Median threshold across seeds for this detector/N.
        if "threshold" in g.columns:
            th = g["threshold"].to_numpy(float)
            th = th[np.isfinite(th)]
            if len(th):
                ax.axvline(
                    np.median(th),
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Median threshold = {np.median(th):.3g}",
                )
        ax.set_title(f"{detector}: score distributions at N={int(n)}")
        ax.set_xlabel("Anomaly / nonconformity score")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = normalize_name(detector)
        fig.savefig(
            figures_dir / f"score_distributions_{safe}_N{int(n)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_calibration_vs_healthy(score_df: pd.DataFrame, figures_dir: Path) -> None:
    if score_df.empty:
        return

    for (detector, n), g in score_df.groupby(["detector", "n"], sort=True):
        cal = g.loc[g["partition"] == "calibration_healthy", "score"].to_numpy(float)
        healthy = g.loc[g["partition"] == "evaluation_healthy", "score"].to_numpy(float)
        cal = cal[np.isfinite(cal)]
        healthy = healthy[np.isfinite(healthy)]
        if len(cal) == 0 or len(healthy) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.6, 5.5))
        cal_sorted = np.sort(cal)
        healthy_sorted = np.sort(healthy)
        cal_y = np.arange(1, len(cal_sorted) + 1) / len(cal_sorted)
        healthy_y = np.arange(1, len(healthy_sorted) + 1) / len(healthy_sorted)
        ax.step(cal_sorted, cal_y, where="post", label=f"Calibration healthy (n={len(cal)})")
        ax.step(healthy_sorted, healthy_y, where="post", label=f"Evaluation healthy (n={len(healthy)})")
        ax.set_title(f"{detector}: calibration vs healthy evaluation, N={int(n)}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        safe = normalize_name(detector)
        fig.savefig(
            figures_dir / f"calibration_vs_healthy_{safe}_N{int(n)}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def classify_findings(
    seed_summary: pd.DataFrame,
    healthy_shift: pd.DataFrame,
    separation: pd.DataFrame,
    score_summary: pd.DataFrame,
    threshold_tail: pd.DataFrame,
    recall_target: float,
    fpr_budget: float,
) -> dict:
    findings = {
        "interpretation_rules": {
            "weak_anomaly_separation": (
                "Suggested when anomaly-vs-healthy probability superiority is near 0.5 "
                "and score quantiles substantially overlap."
            ),
            "healthy_distribution_shift": (
                "Suggested when evaluation-healthy scores are systematically above "
                "calibration-healthy scores (positive mean/q99 shift, large KS/Wasserstein)."
            ),
            "calibration_tail_conservatism": (
                "Suggested when the conformal threshold equals or nearly equals the "
                "maximum calibration score and anomaly recall is low despite moderate AUROC."
            ),
            "fault_specific_difficulty": (
                "Suggested when some fault classes have materially lower recall/separation "
                "than others."
            ),
        },
        "detectors": {},
    }

    detectors = sorted(set(seed_summary["detector"])) if not seed_summary.empty else []
    for detector in detectors:
        d = {"notes": []}
        s = seed_summary[seed_summary["detector"] == detector].sort_values("n")
        if not s.empty:
            last = s.iloc[-1]
            d["largest_N"] = int(last["n"])
            d["largest_N_recall_mean"] = float(last["recall_mean"])
            d["largest_N_fpr_mean"] = float(last["fpr_mean"])
            d["largest_N_joint_seed_success_rate"] = float(last["joint_seed_success_rate"])
            if "auroc_mean" in last and pd.notna(last["auroc_mean"]):
                d["largest_N_auroc_mean"] = float(last["auroc_mean"])

            if last["recall_mean"] < recall_target:
                d["notes"].append("Recall remains below the commissioning target at the largest analyzed N.")
            if last["fpr_mean"] > fpr_budget:
                d["notes"].append("Mean FPR exceeds the commissioning budget at the largest analyzed N.")

        ht = healthy_shift[healthy_shift["detector"] == detector] if not healthy_shift.empty else pd.DataFrame()
        if not ht.empty:
            d["healthy_mean_shift_average"] = float(ht["mean_shift"].mean())
            d["healthy_q99_shift_average"] = float(ht["q99_shift"].mean())
            if ht["q99_shift"].mean() > 0:
                d["notes"].append(
                    "Evaluation-healthy upper-tail scores exceed calibration-healthy scores on average; "
                    "this is evidence consistent with healthy distribution shift."
                )

        sep = separation[separation["detector"] == detector] if not separation.empty else pd.DataFrame()
        if not sep.empty:
            by_class = (
                sep.groupby("class_label")["probability_anomaly_score_gt_healthy"]
                .mean()
                .sort_values()
            )
            d["probability_superiority_by_class"] = {
                str(k): float(v) for k, v in by_class.items()
            }
            if len(by_class) and float(by_class.iloc[0]) < 0.65:
                d["notes"].append(
                    "At least one fault class has weak anomaly-vs-healthy score separation."
                )

        tt = threshold_tail[threshold_tail["detector"] == detector] if not threshold_tail.empty else pd.DataFrame()
        if not tt.empty and "threshold_equals_calibration_max" in tt.columns:
            x = tt["threshold_equals_calibration_max"].dropna()
            if len(x):
                rate = float(np.mean(x.astype(bool)))
                d["fraction_threshold_equals_calibration_max"] = rate
                if rate >= 0.5:
                    d["notes"].append(
                        "The conformal threshold equals the calibration maximum in many runs, "
                        "supporting calibration-tail conservatism as a mechanism."
                    )

        findings["detectors"][detector] = d

    return findings


def main() -> int:
    args = parse_args()

    # Resolve paths from the repository location, not the caller's current directory.
    script_path = Path(__file__).resolve()
    inferred_repo_root = script_path.parent.parent if script_path.parent.name == "experiments" else Path.cwd().resolve()
    repo_root = (args.repo_root or inferred_repo_root).resolve()
    args.results_root = (args.results_root or (repo_root / "outputs" / "aursad")).resolve()
    args.output_dir = (args.output_dir or (repo_root / "outputs" / "aursad" / "diagnostics")).resolve()

    n_values = set(args.n_values)
    ensure_output_dir(args.output_dir, args.overwrite)
    figures_dir = args.output_dir / "figures"

    all_seed = []
    all_per_class = []
    all_scores = []
    detector_audits: list[DetectorAudit] = []

    print("=" * 78)
    print("AURSAD score diagnostics")
    print("=" * 78)
    print(f"Results root : {args.results_root}")
    print(f"Output dir   : {args.output_dir}")
    print(f"N values     : {sorted(n_values)}")
    print()

    for key in args.detectors:
        spec = DETECTOR_SPECS[key]
        detector_name = spec["display"]
        detector_dir = args.results_root / spec["folder"]
        warnings = []

        print(f"[{detector_name}]")
        if not detector_dir.exists():
            msg = f"Result directory does not exist: {detector_dir}"
            if args.require_score_files:
                raise FileNotFoundError(msg)
            warnings.append(msg)
            print(f"  WARNING: {msg}")
            detector_audits.append(
                DetectorAudit(
                    detector=detector_name,
                    result_directory=str(detector_dir),
                    seed_file=None,
                    per_class_file=None,
                    score_files=[],
                    raw_score_mode=False,
                    seed_rows=0,
                    score_rows=0,
                    warnings=warnings,
                )
            )
            continue

        seed_file = discover_seed_file(detector_dir, spec["prefixes"])
        per_class_file = discover_per_class_file(detector_dir, spec["prefixes"])
        score_files = discover_score_files(detector_dir, args.max_score_files)

        seed_df = pd.DataFrame()
        if seed_file is not None:
            seed_df = load_seed_results(seed_file, detector_name, n_values)
            all_seed.append(seed_df)
            print(f"  seed results : {seed_file.name} ({len(seed_df)} rows)")
        else:
            warnings.append("No seed-results CSV found.")
            print("  WARNING: no seed-results CSV found.")

        if per_class_file is not None:
            try:
                per_class_df = load_per_class(per_class_file, detector_name, n_values)
                all_per_class.append(per_class_df)
                print(f"  per-class    : {per_class_file.name} ({len(per_class_df)} rows)")
            except Exception as e:
                warnings.append(f"Could not parse per-class file {per_class_file}: {e}")
                print(f"  WARNING: per-class parse failed: {e}")

        score_df = load_all_scores(score_files, detector_name, n_values)
        if not score_df.empty:
            all_scores.append(score_df)
            print(f"  score mode   : FULL ({len(score_df)} episode-score rows)")
        else:
            msg = (
                "No usable episode-level score artifacts found. "
                "Falling back to seed/per-class diagnostics."
            )
            warnings.append(msg)
            print(f"  score mode   : FALLBACK")
            if args.require_score_files:
                raise FileNotFoundError(
                    f"{detector_name}: {msg} Searched under {detector_dir}"
                )

        detector_audits.append(
            DetectorAudit(
                detector=detector_name,
                result_directory=str(detector_dir),
                seed_file=str(seed_file) if seed_file else None,
                per_class_file=str(per_class_file) if per_class_file else None,
                score_files=[str(p) for p in score_files],
                raw_score_mode=not score_df.empty,
                seed_rows=int(len(seed_df)),
                score_rows=int(len(score_df)),
                warnings=warnings,
            )
        )
        print()

    seed_df = pd.concat(all_seed, ignore_index=True) if all_seed else pd.DataFrame()
    per_class_df = (
        pd.concat(all_per_class, ignore_index=True)
        if all_per_class else pd.DataFrame()
    )
    score_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()

    if seed_df.empty:
        raise RuntimeError(
            "No usable seed-level result files were found for any requested detector."
        )

    if not score_df.empty:
        score_df = merge_thresholds_into_scores(score_df, seed_df)

    seed_summary = summarize_seed_results(
        seed_df, args.recall_target, args.fpr_budget
    )
    per_class_summary = summarize_per_class(per_class_df)
    score_summary = summarize_score_distributions(score_df)
    healthy_shift = healthy_shift_summary(score_df)
    separation = anomaly_separation_summary(score_df)
    tail_summary = threshold_tail_summary(score_df, seed_df)

    safe_save_csv(seed_summary, args.output_dir / "diagnostic_seed_summary.csv")
    safe_save_csv(per_class_summary, args.output_dir / "diagnostic_per_class_summary.csv")
    safe_save_csv(score_summary, args.output_dir / "score_distribution_summary.csv")
    safe_save_csv(tail_summary, args.output_dir / "threshold_tail_summary.csv")
    safe_save_csv(healthy_shift, args.output_dir / "healthy_shift_summary.csv")
    safe_save_csv(separation, args.output_dir / "anomaly_separation_summary.csv")

    # Figures available in both modes.
    plot_metric_vs_n(
        seed_summary,
        "threshold_mean",
        "Mean conformal threshold",
        figures_dir / "threshold_vs_n.png",
    )
    plot_recall_fpr(
        seed_summary,
        figures_dir / "recall_fpr_vs_n.png",
        args.recall_target,
        args.fpr_budget,
    )
    plot_metric_vs_n(
        seed_summary,
        "auroc_mean",
        "Mean AUROC",
        figures_dir / "auroc_vs_n.png",
    )
    plot_per_class(
        per_class_summary,
        figures_dir / "per_class_recall.png",
    )

    # Full-score figures.
    plot_score_distributions(score_df, figures_dir)
    plot_calibration_vs_healthy(score_df, figures_dir)

    findings = classify_findings(
        seed_summary=seed_summary,
        healthy_shift=healthy_shift,
        separation=separation,
        score_summary=score_summary,
        threshold_tail=tail_summary,
        recall_target=args.recall_target,
        fpr_budget=args.fpr_budget,
    )

    findings["configuration"] = {
        "global_seed": GLOBAL_SEED,
        "commissioning_sizes": sorted(n_values),
        "recall_target": args.recall_target,
        "fpr_budget": args.fpr_budget,
        "requested_detectors": args.detectors,
        "raw_score_diagnostics_available_for": [
            a.detector for a in detector_audits if a.raw_score_mode
        ],
        "fallback_only_detectors": [
            a.detector for a in detector_audits if not a.raw_score_mode
        ],
    }
    findings["audit"] = [asdict(a) for a in detector_audits]

    if any(not a.raw_score_mode for a in detector_audits):
        findings["important_limitation"] = (
            "At least one detector lacked usable episode-level score artifacts. "
            "For those detectors, conclusions about calibration-vs-evaluation score shift "
            "and class-conditional score overlap are unavailable. Seed-level performance "
            "and per-class recall diagnostics remain valid."
        )

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(args.output_dir / "diagnostic_findings.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(findings), f, indent=2, allow_nan=False)

    print("=" * 78)
    print("Diagnostic outputs written")
    print("=" * 78)
    print(args.output_dir)
    print()
    print("Raw score diagnostics available for:")
    full = [a.detector for a in detector_audits if a.raw_score_mode]
    print("  " + (", ".join(full) if full else "none"))
    fallback = [a.detector for a in detector_audits if not a.raw_score_mode]
    if fallback:
        print("Fallback-only detectors:")
        print("  " + ", ".join(fallback))
        print()
        print(
            "If you want true score-distribution plots for these detectors, update their "
            "runner to persist one row per episode with at least: "
            "N, seed, episode_id, partition, class_label, score."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())