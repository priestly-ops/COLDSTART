#!/usr/bin/env python3
"""
experiments/export_aursad_diagnostic_scores.py

Lightweight score-export diagnostic for the frozen AURSAD commissioning benchmark.

This script DOES NOT modify or replace the frozen primary results. It refits only a
predeclared representative subset of already-frozen (N, seed) runs using the exact
TargetOnly and Euclidean conformal k-NN scoring logic, exports episode-level scores,
and verifies that each reproduced run matches the existing frozen result row.

Default diagnostic subset
-------------------------
Commissioning sizes:
    N = 100, 500

Seeds:
    0, 4, 9, 13, 19

Detectors:
    TargetOnly
    Euclidean conformal k-NN

Per reproduced run, scores are exported for:
    calibration healthy   (600 executions)
    healthy evaluation    (300 executions)
    anomaly evaluation    (625 executions, with fault class)

Outputs
-------
outputs/aursad/score_diagnostics/
├── aursad_episode_scores.csv
├── aursad_score_run_verification.csv
└── aursad_score_export_manifest.json

Important scientific safeguard
------------------------------
A diagnostic run is accepted only if its recomputed threshold, recall, FPR, and AUROC
match the corresponding frozen benchmark row within strict numerical tolerances.

The feature cache and frozen protocol are read only. Primary result folders are read only.

Example
-------
From repository root:

    python experiments/export_aursad_diagnostic_scores.py

Recreate outputs:

    python experiments/export_aursad_diagnostic_scores.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors import TargetOnlyDetector
from src.feature_extractor import (
    FeatureBatch,
    FeaturePreprocessor,
    load_feature_batch,
)

DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "outputs" / "aursad" / "feature_cache" / "aursad_features.npz"
)
DEFAULT_PROTOCOL_DIR = PROJECT_ROOT / "reports" / "aursad" / "protocol"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "aursad"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "score_diagnostics"

DEFAULT_N_VALUES = (100, 500)
DEFAULT_SEEDS = (0, 4, 9, 13, 19)
DEFAULT_DETECTORS = ("targetonly", "euclidean_knn")
DEFAULT_K = 10
DEFAULT_ALPHA = 0.01

SCORE_EXPORT_VERSION = "aursad-score-diagnostics-v1"

DETECTOR_INFO = {
    "targetonly": {
        "display": "TargetOnly",
        "frozen_csv": "targetonly/targetonly_seed_results.csv",
    },
    "euclidean_knn": {
        "display": "Euclidean conformal k-NN",
        "frozen_csv": "euclidean_conformal_knn/euclidean_knn_seed_results.csv",
    },
}

SCORE_COLUMNS = (
    "score_export_version",
    "detector",
    "commissioning_size",
    "seed",
    "episode_id",
    "partition",
    "label",
    "label_name",
    "is_anomaly",
    "score",
    "threshold",
    "prediction",
    "threshold_margin",
    "retained_features",
    "requested_k",
    "effective_k",
)

VERIFY_COLUMNS = (
    "detector",
    "commissioning_size",
    "seed",
    "recomputed_threshold",
    "frozen_threshold",
    "threshold_abs_error",
    "recomputed_fpr",
    "frozen_fpr",
    "fpr_abs_error",
    "recomputed_recall",
    "frozen_recall",
    "recall_abs_error",
    "recomputed_auroc",
    "frozen_auroc",
    "auroc_abs_error",
    "retained_features",
    "frozen_retained_features",
    "requested_k",
    "effective_k",
    "status",
)


def parse_int_csv(value: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    if len(set(vals)) != len(vals):
        raise argparse.ArgumentTypeError("Duplicate integers are not allowed.")
    return vals


def parse_str_csv(value: str) -> tuple[str, ...]:
    vals = tuple(x.strip().lower() for x in value.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("At least one detector is required.")
    unknown = sorted(set(vals) - set(DETECTOR_INFO))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown detector(s): {unknown}. Expected {sorted(DETECTOR_INFO)}."
        )
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export episode-level scores for representative frozen AURSAD runs."
    )
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-values", type=parse_int_csv, default=DEFAULT_N_VALUES)
    p.add_argument("--seeds", type=parse_int_csv, default=DEFAULT_SEEDS)
    p.add_argument("--detectors", type=parse_str_csv, default=DEFAULT_DETECTORS)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument(
        "--verification-atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for matching frozen scalar metrics.",
    )
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


def load_protocol_csv(path: Path, expected_partition: str) -> pd.DataFrame:
    require_file(path, f"{expected_partition} protocol CSV")
    df = pd.read_csv(path)

    required = {"sample_nr", "partition", "label", "label_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df.copy()
    df["sample_nr"] = pd.to_numeric(df["sample_nr"], errors="raise").astype(np.int64)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(np.int64)
    df["partition"] = df["partition"].astype(str).str.strip()

    unexpected = sorted(set(df["partition"].unique()) - {expected_partition})
    if unexpected:
        raise ValueError(
            f"{path.name}: unexpected partitions {unexpected}; "
            f"expected only {expected_partition!r}."
        )

    if expected_partition == "commissioning":
        extra = {"seed", "commissioning_n", "selection_rank"}
        missing = sorted(extra - set(df.columns))
        if missing:
            raise ValueError(f"{path.name} missing commissioning columns: {missing}")
        for c in extra:
            df[c] = pd.to_numeric(df[c], errors="raise").astype(np.int64)

    return df


def load_protocol(protocol_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "commissioning": load_protocol_csv(
            protocol_dir / "commissioning_ids.csv", "commissioning"
        ),
        "calibration": load_protocol_csv(
            protocol_dir / "calibration_ids.csv", "calibration"
        ),
        "healthy_eval": load_protocol_csv(
            protocol_dir / "healthy_eval_ids.csv", "healthy_eval"
        ),
        "anomaly_eval": load_protocol_csv(
            protocol_dir / "anomaly_eval_ids.csv", "anomaly_eval"
        ),
    }


def unique_ids(df: pd.DataFrame) -> np.ndarray:
    ids = df["sample_nr"].to_numpy(np.int64)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Fixed protocol partition contains duplicate sample_nr.")
    return ids


def commissioning_ids(
    commissioning: pd.DataFrame, seed: int, n_value: int
) -> np.ndarray:
    x = commissioning[
        commissioning["seed"].eq(seed)
        & commissioning["commissioning_n"].eq(n_value)
    ].sort_values("selection_rank")
    ids = x["sample_nr"].to_numpy(np.int64)
    if len(ids) != n_value:
        raise ValueError(
            f"Seed {seed}, N={n_value}: expected {n_value} commissioning IDs, "
            f"found {len(ids)}."
        )
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Seed {seed}, N={n_value}: duplicate commissioning IDs.")
    return ids


def validate_protocol(
    protocol: dict[str, pd.DataFrame],
    seeds: tuple[int, ...],
    n_values: tuple[int, ...],
) -> None:
    if not protocol["commissioning"]["label"].eq(0).all():
        raise RuntimeError("Commissioning contains anomaly labels.")
    if not protocol["calibration"]["label"].eq(0).all():
        raise RuntimeError("Calibration contains anomaly labels.")
    if not protocol["healthy_eval"]["label"].eq(0).all():
        raise RuntimeError("Healthy evaluation contains anomaly labels.")
    if not protocol["anomaly_eval"]["label"].isin([1, 2, 3, 4]).all():
        raise RuntimeError("Anomaly evaluation contains unsupported labels.")

    fixed = {
        k: set(unique_ids(protocol[k]).tolist())
        for k in ("calibration", "healthy_eval", "anomaly_eval")
    }

    names = list(fixed)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = fixed[a] & fixed[b]
            if overlap:
                raise RuntimeError(
                    f"Protocol leakage between {a} and {b}: {sorted(overlap)[:10]}"
                )

    for seed in seeds:
        previous: set[int] = set()
        for n in sorted(n_values):
            ids = commissioning_ids(protocol["commissioning"], seed, n)
            current = set(ids.tolist())
            if previous and not previous.issubset(current):
                raise RuntimeError(
                    f"Commissioning diagnostic subsets are not nested for seed={seed}, N={n}."
                )
            for role, fixed_ids in fixed.items():
                overlap = current & fixed_ids
                if overlap:
                    raise RuntimeError(
                        f"Seed={seed}, N={n} commissioning overlaps {role}: "
                        f"{sorted(overlap)[:10]}"
                    )
            previous = current


def subset(cache: FeatureBatch, ids: Iterable[int]) -> FeatureBatch:
    return cache.select_episode_ids(
        [int(x) for x in ids],
        preserve_requested_order=True,
        require_all=True,
    )


def split_conformal_threshold(
    scores: np.ndarray, alpha: float
) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Calibration scores must be a non-empty 1D array.")
    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores contain NaN/Inf.")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(rank, len(scores))
    return float(np.sort(scores)[rank - 1]), rank


def kth_scores(
    model: NearestNeighbors, features: np.ndarray, effective_k: int
) -> np.ndarray:
    d, _ = model.kneighbors(
        features,
        n_neighbors=effective_k,
        return_distance=True,
    )
    scores = np.asarray(d[:, effective_k - 1], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("k-NN produced NaN/Inf scores.")
    return scores


def make_rows(
    *,
    detector: str,
    n_value: int,
    seed: int,
    partition: str,
    batch: FeatureBatch,
    table: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    retained_features: int,
    requested_k: int | None,
    effective_k: int | None,
) -> list[dict[str, Any]]:
    if len(batch.episode_ids) != len(scores):
        raise RuntimeError("Score count does not match episode count.")

    metadata = table.set_index("sample_nr")[["label", "label_name"]].to_dict("index")
    rows = []

    for ep_id, score in zip(batch.episode_ids.astype(int), scores):
        meta = metadata[int(ep_id)]
        prediction = bool(float(score) > float(threshold))
        rows.append(
            {
                "score_export_version": SCORE_EXPORT_VERSION,
                "detector": detector,
                "commissioning_size": int(n_value),
                "seed": int(seed),
                "episode_id": int(ep_id),
                "partition": str(partition),
                "label": int(meta["label"]),
                "label_name": str(meta["label_name"]),
                "is_anomaly": bool(int(meta["label"]) != 0),
                "score": float(score),
                "threshold": float(threshold),
                "prediction": prediction,
                "threshold_margin": float(score - threshold),
                "retained_features": int(retained_features),
                "requested_k": (
                    np.nan if requested_k is None else int(requested_k)
                ),
                "effective_k": (
                    np.nan if effective_k is None else int(effective_k)
                ),
            }
        )
    return rows


def frozen_row(
    frozen_df: pd.DataFrame, n_value: int, seed: int
) -> pd.Series:
    x = frozen_df[
        frozen_df["commissioning_size"].eq(n_value)
        & frozen_df["seed"].eq(seed)
    ]
    if len(x) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen result for N={n_value}, seed={seed}; "
            f"found {len(x)}."
        )
    return x.iloc[0]


def verify_reproduction(
    *,
    detector: str,
    n_value: int,
    seed: int,
    threshold: float,
    fpr: float,
    recall: float,
    auroc: float,
    retained_features: int,
    requested_k: int | None,
    effective_k: int | None,
    frozen: pd.Series,
    atol: float,
) -> dict[str, Any]:
    frozen_fpr_col = (
        "false_positive_rate"
        if "false_positive_rate" in frozen.index
        else "fpr"
    )

    checks = {
        "threshold": (threshold, float(frozen["threshold"])),
        "fpr": (fpr, float(frozen[frozen_fpr_col])),
        "recall": (recall, float(frozen["recall"])),
        "auroc": (auroc, float(frozen["auroc"])),
    }

    errors = {name: abs(a - b) for name, (a, b) in checks.items()}
    status = "PASS" if all(err <= atol for err in errors.values()) else "FAIL"

    frozen_retained = (
        int(frozen["retained_features"])
        if "retained_features" in frozen.index
        else retained_features
    )
    if frozen_retained != int(retained_features):
        status = "FAIL"

    if detector == "Euclidean conformal k-NN":
        if "requested_k" in frozen.index and int(frozen["requested_k"]) != int(requested_k):
            status = "FAIL"
        if "effective_k" in frozen.index and int(frozen["effective_k"]) != int(effective_k):
            status = "FAIL"

    return {
        "detector": detector,
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "recomputed_threshold": float(threshold),
        "frozen_threshold": float(frozen["threshold"]),
        "threshold_abs_error": float(errors["threshold"]),
        "recomputed_fpr": float(fpr),
        "frozen_fpr": float(frozen[frozen_fpr_col]),
        "fpr_abs_error": float(errors["fpr"]),
        "recomputed_recall": float(recall),
        "frozen_recall": float(frozen["recall"]),
        "recall_abs_error": float(errors["recall"]),
        "recomputed_auroc": float(auroc),
        "frozen_auroc": float(frozen["auroc"]),
        "auroc_abs_error": float(errors["auroc"]),
        "retained_features": int(retained_features),
        "frozen_retained_features": int(frozen_retained),
        "requested_k": np.nan if requested_k is None else int(requested_k),
        "effective_k": np.nan if effective_k is None else int(effective_k),
        "status": status,
    }


def run_targetonly(
    *,
    cache: FeatureBatch,
    protocol: dict[str, pd.DataFrame],
    n_value: int,
    seed: int,
    alpha: float,
    frozen: pd.Series,
    atol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_ids = commissioning_ids(protocol["commissioning"], seed, n_value)

    train = subset(cache, train_ids)
    cal = subset(cache, unique_ids(protocol["calibration"]))
    healthy = subset(cache, unique_ids(protocol["healthy_eval"]))
    anomaly = subset(cache, unique_ids(protocol["anomaly_eval"]))

    pre = FeaturePreprocessor(variance_threshold=1e-12)
    x_train = pre.fit_transform(train.features)
    x_cal = pre.transform(cal.features)
    x_healthy = pre.transform(healthy.features)
    x_anomaly = pre.transform(anomaly.features)

    detector = TargetOnlyDetector(false_alert_budget=alpha)
    detector.fit(source_features=x_train, target_features=x_train)

    # Capture scores explicitly, then calibrate exactly as the frozen runner did.
    cal_scores = detector.score_samples(x_cal)
    detector.calibrate(x_cal)
    if detector.threshold_ is None:
        raise RuntimeError("TargetOnly did not produce threshold.")
    threshold = float(detector.threshold_)

    # Independently check finite-sample threshold math.
    manual_threshold, _ = split_conformal_threshold(cal_scores, alpha)
    if not np.isclose(threshold, manual_threshold, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "TargetOnly detector calibration disagrees with finite-sample threshold."
        )

    healthy_scores = detector.score_samples(x_healthy)
    anomaly_scores = detector.score_samples(x_anomaly)

    fpr = float(np.mean(healthy_scores > threshold))
    recall = float(np.mean(anomaly_scores > threshold))
    y = np.concatenate(
        [np.zeros(len(healthy_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]
    )
    s = np.concatenate([healthy_scores, anomaly_scores])
    auroc = float(roc_auc_score(y, s))

    retained = int(pre.output_feature_count_)

    rows = []
    rows += make_rows(
        detector="TargetOnly",
        n_value=n_value,
        seed=seed,
        partition="calibration_healthy",
        batch=cal,
        table=protocol["calibration"],
        scores=cal_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=None,
        effective_k=None,
    )
    rows += make_rows(
        detector="TargetOnly",
        n_value=n_value,
        seed=seed,
        partition="evaluation_healthy",
        batch=healthy,
        table=protocol["healthy_eval"],
        scores=healthy_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=None,
        effective_k=None,
    )
    rows += make_rows(
        detector="TargetOnly",
        n_value=n_value,
        seed=seed,
        partition="anomaly_evaluation",
        batch=anomaly,
        table=protocol["anomaly_eval"],
        scores=anomaly_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=None,
        effective_k=None,
    )

    verify = verify_reproduction(
        detector="TargetOnly",
        n_value=n_value,
        seed=seed,
        threshold=threshold,
        fpr=fpr,
        recall=recall,
        auroc=auroc,
        retained_features=retained,
        requested_k=None,
        effective_k=None,
        frozen=frozen,
        atol=atol,
    )
    return rows, verify


def run_euclidean_knn(
    *,
    cache: FeatureBatch,
    protocol: dict[str, pd.DataFrame],
    n_value: int,
    seed: int,
    requested_k: int,
    alpha: float,
    frozen: pd.Series,
    atol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_ids = commissioning_ids(protocol["commissioning"], seed, n_value)

    train = subset(cache, train_ids)
    cal = subset(cache, unique_ids(protocol["calibration"]))
    healthy = subset(cache, unique_ids(protocol["healthy_eval"]))
    anomaly = subset(cache, unique_ids(protocol["anomaly_eval"]))

    pre = FeaturePreprocessor(variance_threshold=1e-12)
    x_train = pre.fit_transform(train.features)
    x_cal = pre.transform(cal.features)
    x_healthy = pre.transform(healthy.features)
    x_anomaly = pre.transform(anomaly.features)

    effective_k = min(int(requested_k), len(x_train))
    if effective_k <= 0:
        raise RuntimeError("effective_k must be positive.")

    model = NearestNeighbors(
        n_neighbors=effective_k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    model.fit(x_train)

    cal_scores = kth_scores(model, x_cal, effective_k)
    threshold, _ = split_conformal_threshold(cal_scores, alpha)
    healthy_scores = kth_scores(model, x_healthy, effective_k)
    anomaly_scores = kth_scores(model, x_anomaly, effective_k)

    fpr = float(np.mean(healthy_scores > threshold))
    recall = float(np.mean(anomaly_scores > threshold))
    y = np.concatenate(
        [np.zeros(len(healthy_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]
    )
    s = np.concatenate([healthy_scores, anomaly_scores])
    auroc = float(roc_auc_score(y, s))
    retained = int(pre.output_feature_count_)

    rows = []
    rows += make_rows(
        detector="Euclidean conformal k-NN",
        n_value=n_value,
        seed=seed,
        partition="calibration_healthy",
        batch=cal,
        table=protocol["calibration"],
        scores=cal_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=requested_k,
        effective_k=effective_k,
    )
    rows += make_rows(
        detector="Euclidean conformal k-NN",
        n_value=n_value,
        seed=seed,
        partition="evaluation_healthy",
        batch=healthy,
        table=protocol["healthy_eval"],
        scores=healthy_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=requested_k,
        effective_k=effective_k,
    )
    rows += make_rows(
        detector="Euclidean conformal k-NN",
        n_value=n_value,
        seed=seed,
        partition="anomaly_evaluation",
        batch=anomaly,
        table=protocol["anomaly_eval"],
        scores=anomaly_scores,
        threshold=threshold,
        retained_features=retained,
        requested_k=requested_k,
        effective_k=effective_k,
    )

    verify = verify_reproduction(
        detector="Euclidean conformal k-NN",
        n_value=n_value,
        seed=seed,
        threshold=threshold,
        fpr=fpr,
        recall=recall,
        auroc=auroc,
        retained_features=retained,
        requested_k=requested_k,
        effective_k=effective_k,
        frozen=frozen,
        atol=atol,
    )
    return rows, verify


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    args = parse_args()

    cache_path = args.cache_path.expanduser().resolve()
    protocol_dir = args.protocol_dir.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    n_values = tuple(sorted(int(x) for x in args.n_values))
    seeds = tuple(int(x) for x in args.seeds)
    detectors = tuple(args.detectors)

    if any(n <= 0 for n in n_values):
        raise ValueError("--n-values must be positive.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must be non-negative.")
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.verification_atol < 0:
        raise ValueError("--verification-atol cannot be negative.")

    require_file(cache_path, "AURSAD feature cache")
    if not protocol_dir.is_dir():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "aursad_episode_scores.csv"
    verify_path = output_dir / "aursad_score_run_verification.csv"
    manifest_path = output_dir / "aursad_score_export_manifest.json"

    existing = [p for p in (score_path, verify_path, manifest_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Diagnostic outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(f"  - {p}" for p in existing)
        )

    print("=" * 78)
    print("AURSAD REPRESENTATIVE SCORE EXPORT")
    print("=" * 78)
    print(f"Cache:       {cache_path}")
    print(f"Protocol:    {protocol_dir}")
    print(f"Results:     {results_root}")
    print(f"Output:      {output_dir}")
    print(f"N values:    {list(n_values)}")
    print(f"Seeds:       {list(seeds)}")
    print(f"Detectors:   {[DETECTOR_INFO[d]['display'] for d in detectors]}")
    print(f"Alpha:       {args.alpha}")
    print()

    started = time.perf_counter()

    cache = load_feature_batch(cache_path)
    if cache.features.shape[1] != 288:
        raise ValueError(
            f"Expected 288 AURSAD cached features, found {cache.features.shape[1]}."
        )

    protocol = load_protocol(protocol_dir)
    validate_protocol(protocol, seeds, n_values)

    all_required = (
        set(unique_ids(protocol["calibration"]).tolist())
        | set(unique_ids(protocol["healthy_eval"]).tolist())
        | set(unique_ids(protocol["anomaly_eval"]).tolist())
    )
    for seed in seeds:
        for n in n_values:
            all_required |= set(
                commissioning_ids(protocol["commissioning"], seed, n).tolist()
            )

    cache_ids = set(cache.episode_ids.astype(int).tolist())
    missing = sorted(all_required - cache_ids)
    if missing:
        raise RuntimeError(f"Feature cache missing protocol IDs: {missing[:20]}")

    frozen_tables: dict[str, pd.DataFrame] = {}
    frozen_hashes: dict[str, str] = {}
    for key in detectors:
        p = results_root / DETECTOR_INFO[key]["frozen_csv"]
        require_file(p, f"{DETECTOR_INFO[key]['display']} frozen seed results")
        df = pd.read_csv(p)
        for c in ("commissioning_size", "seed"):
            if c not in df.columns:
                raise ValueError(f"{p} missing required column {c!r}")
            df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
        frozen_tables[key] = df
        frozen_hashes[key] = sha256_file(p)

    score_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []

    total = len(detectors) * len(n_values) * len(seeds)
    run_idx = 0

    for key in detectors:
        display = DETECTOR_INFO[key]["display"]
        for n in n_values:
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx:02d}/{total}] {display} N={n} seed={seed}")

                frozen = frozen_row(frozen_tables[key], n, seed)

                if key == "targetonly":
                    rows, verify = run_targetonly(
                        cache=cache,
                        protocol=protocol,
                        n_value=n,
                        seed=seed,
                        alpha=args.alpha,
                        frozen=frozen,
                        atol=args.verification_atol,
                    )
                elif key == "euclidean_knn":
                    rows, verify = run_euclidean_knn(
                        cache=cache,
                        protocol=protocol,
                        n_value=n,
                        seed=seed,
                        requested_k=args.k,
                        alpha=args.alpha,
                        frozen=frozen,
                        atol=args.verification_atol,
                    )
                else:
                    raise AssertionError(key)

                print(
                    "   "
                    f"threshold={verify['recomputed_threshold']:.6g} "
                    f"recall={verify['recomputed_recall']:.4f} "
                    f"FPR={verify['recomputed_fpr']:.4f} "
                    f"AUROC={verify['recomputed_auroc']:.4f} "
                    f"verify={verify['status']}"
                )

                score_rows.extend(rows)
                verification_rows.append(verify)

    verify_df = pd.DataFrame(verification_rows, columns=VERIFY_COLUMNS)

    failures = verify_df[verify_df["status"].ne("PASS")]
    if not failures.empty:
        # Write the verification table for debugging, but NEVER accept/export scores
        # from a run that does not reproduce the frozen benchmark.
        atomic_csv(verify_df, verify_path)
        raise RuntimeError(
            "One or more diagnostic reruns failed frozen-result verification. "
            f"See {verify_path}. Failed rows:\n"
            + failures.to_string(index=False)
        )

    score_df = pd.DataFrame(score_rows, columns=SCORE_COLUMNS)

    # Strong structural checks on the exported episode-score artifact.
    expected_per_run = (
        len(protocol["calibration"])
        + len(protocol["healthy_eval"])
        + len(protocol["anomaly_eval"])
    )
    expected_score_rows = expected_per_run * total

    if len(score_df) != expected_score_rows:
        raise RuntimeError(
            f"Expected {expected_score_rows} score rows, found {len(score_df)}."
        )

    key_cols = ["detector", "commissioning_size", "seed", "partition", "episode_id"]
    if score_df.duplicated(key_cols).any():
        dup = score_df.loc[score_df.duplicated(key_cols, keep=False), key_cols].head(20)
        raise RuntimeError("Duplicate exported score keys:\n" + dup.to_string(index=False))

    if not np.isfinite(score_df["score"].to_numpy(float)).all():
        raise RuntimeError("Export contains non-finite scores.")
    if not np.isfinite(score_df["threshold"].to_numpy(float)).all():
        raise RuntimeError("Export contains non-finite thresholds.")

    atomic_csv(score_df, score_path)
    atomic_csv(verify_df, verify_path)

    elapsed = time.perf_counter() - started

    manifest = {
        "score_export_version": SCORE_EXPORT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "purpose": (
            "Diagnostic-only episode-level score export from a predeclared subset of "
            "already-frozen AURSAD commissioning runs. Does not replace primary results."
        ),
        "dataset": "AURSAD",
        "selection": {
            "detectors": [DETECTOR_INFO[d]["display"] for d in detectors],
            "commissioning_sizes": list(n_values),
            "seeds": list(seeds),
            "alpha": float(args.alpha),
            "requested_k": int(args.k),
            "selection_was_predeclared": True,
        },
        "inputs": {
            "feature_cache": {
                "path": str(cache_path),
                "sha256": sha256_file(cache_path),
                "shape": list(cache.features.shape),
            },
            "protocol_files": {
                name: {
                    "path": str(protocol_dir / f"{name}_ids.csv"),
                    "sha256": sha256_file(protocol_dir / f"{name}_ids.csv"),
                }
                for name in ("commissioning", "calibration", "healthy_eval", "anomaly_eval")
            },
            "frozen_seed_results_sha256": frozen_hashes,
        },
        "verification": {
            "all_reproduced_runs_match_frozen_results": True,
            "tolerance_absolute": float(args.verification_atol),
            "run_count": int(len(verify_df)),
            "failed_run_count": 0,
        },
        "outputs": {
            "episode_scores": str(score_path),
            "episode_scores_sha256": sha256_file(score_path),
            "verification": str(verify_path),
            "verification_sha256": sha256_file(verify_path),
            "episode_score_rows": int(len(score_df)),
        },
        "timing_seconds": {
            "total": float(elapsed),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "This is a diagnostic subset, not a new primary benchmark run.",
            "Only TargetOnly and Euclidean conformal k-NN are exported initially.",
            "Only seeds 0,4,9,13,19 and N=100,500 are selected by default.",
            "No PAKCT rerun is required for this diagnostic stage.",
        ],
    }

    manifest_path.write_text(
        json.dumps(json_safe(manifest), indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("SCORE EXPORT COMPLETE")
    print("=" * 78)
    print(f"Verified runs:      {len(verify_df)}/{total} PASS")
    print(f"Episode score rows: {len(score_df):,}")
    print(f"Scores:             {score_path}")
    print(f"Verification:       {verify_path}")
    print(f"Manifest:           {manifest_path}")
    print(f"Elapsed:            {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()