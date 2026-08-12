#!/usr/bin/env python3
"""Run the target-only Isolation Forest commissioning baseline on AURSAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extractor import FeaturePreprocessor, load_feature_batch
CACHE = ROOT / "outputs/aursad/feature_cache/aursad_features.npz"
PROTOCOL = ROOT / "reports/aursad/protocol"
OUTPUT = ROOT / "outputs/aursad/isolation_forest"
GRID = (10, 25, 50, 100, 250, 500)
SEEDS = tuple(range(20))
VERSION = "aursad-isolation-forest-split-conformal-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Use unique comma-separated integers.")
    return values


def load_split(path: Path, partition: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"sample_nr", "partition", "label", "label_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    df = df.copy()
    df["sample_nr"] = pd.to_numeric(df["sample_nr"], errors="raise").astype(np.int64)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(np.int64)
    if set(df["partition"].astype(str)) != {partition}:
        raise ValueError(f"{path.name} contains unexpected partitions.")
    return df


def load_protocol(protocol_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "commissioning": load_split(protocol_dir / "commissioning_ids.csv", "commissioning"),
        "calibration": load_split(protocol_dir / "calibration_ids.csv", "calibration"),
        "healthy_eval": load_split(protocol_dir / "healthy_eval_ids.csv", "healthy_eval"),
        "anomaly_eval": load_split(protocol_dir / "anomaly_eval_ids.csv", "anomaly_eval"),
    }
    c = tables["commissioning"]
    for col in ("seed", "commissioning_n", "selection_rank"):
        if col not in c.columns:
            raise ValueError(f"commissioning_ids.csv missing {col}")
        c[col] = pd.to_numeric(c[col], errors="raise").astype(np.int64)
    return tables


def unique_ids(df: pd.DataFrame) -> np.ndarray:
    ids = df["sample_nr"].to_numpy(np.int64)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("Fixed split contains duplicate sample_nr values.")
    return ids


def validate_protocol(t: dict[str, pd.DataFrame], grid: tuple[int, ...], seeds: tuple[int, ...]) -> None:
    if not t["commissioning"]["label"].eq(0).all():
        raise ValueError("Commissioning must be normal only.")
    if not t["calibration"]["label"].eq(0).all():
        raise ValueError("Calibration must be normal only.")
    if not t["healthy_eval"]["label"].eq(0).all():
        raise ValueError("Healthy evaluation must be normal only.")
    if not t["anomaly_eval"]["label"].isin([1, 2, 3, 4]).all():
        raise ValueError("Anomaly evaluation contains unsupported labels.")

    fixed = {
        "calibration": set(unique_ids(t["calibration"]).tolist()),
        "healthy_eval": set(unique_ids(t["healthy_eval"]).tolist()),
        "anomaly_eval": set(unique_ids(t["anomaly_eval"]).tolist()),
    }
    names = list(fixed)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if fixed[left] & fixed[right]:
                raise RuntimeError(f"Leakage between {left} and {right}.")

    c = t["commissioning"]
    for seed in seeds:
        previous: set[int] = set()
        for n in grid:
            rows = c[(c.seed == seed) & (c.commissioning_n == n)].sort_values("selection_rank")
            ids = rows.sample_nr.astype(int).tolist()
            if len(ids) != n or len(set(ids)) != n:
                raise ValueError(f"Invalid commissioning membership for seed={seed}, N={n}.")
            current = set(ids)
            if previous and not previous.issubset(current):
                raise ValueError(f"Non-nested commissioning sets for seed={seed}, N={n}.")
            for name, ids_fixed in fixed.items():
                if current & ids_fixed:
                    raise RuntimeError(f"Leakage with {name} for seed={seed}, N={n}.")
            previous = current


def commissioning_ids(df: pd.DataFrame, seed: int, n: int) -> np.ndarray:
    rows = df[(df.seed == seed) & (df.commissioning_n == n)].sort_values("selection_rank")
    ids = rows.sample_nr.to_numpy(np.int64)
    if len(ids) != n:
        raise ValueError(f"Expected {n} commissioning IDs, found {len(ids)}.")
    return ids


def conformal_threshold(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores.")
    rank = min(int(np.ceil((len(scores) + 1) * (1.0 - alpha))), len(scores))
    return float(np.sort(scores)[rank - 1]), rank


def anomaly_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    scores = -model.decision_function(x)
    if not np.isfinite(scores).all():
        raise RuntimeError("Isolation Forest produced non-finite scores.")
    return np.asarray(scores, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, samples: int, confidence: float, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    a = 1.0 - confidence
    return float(np.quantile(means, a / 2)), float(np.quantile(means, 1 - a / 2))


def run_one(cache, protocol, seed: int, n: int, args) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c_ids = commissioning_ids(protocol["commissioning"], seed, n)
    cal_ids = unique_ids(protocol["calibration"])
    h_ids = unique_ids(protocol["healthy_eval"])
    a_ids = unique_ids(protocol["anomaly_eval"])

    train = cache.select_episode_ids(c_ids.tolist())
    cal = cache.select_episode_ids(cal_ids.tolist())
    healthy = cache.select_episode_ids(h_ids.tolist())
    anomaly = cache.select_episode_ids(a_ids.tolist())

    if train.anomaly_labels.any() or cal.anomaly_labels.any() or healthy.anomaly_labels.any():
        raise RuntimeError("Normal-only split contains anomalous rows.")
    if not anomaly.anomaly_labels.all():
        raise RuntimeError("Anomaly split contains healthy rows.")

    prep = FeaturePreprocessor(variance_threshold=1e-12)
    x_train = prep.fit_transform(train.features)
    x_cal = prep.transform(cal.features)
    x_h = prep.transform(healthy.features)
    x_a = prep.transform(anomaly.features)

    model_seed = args.global_seed + seed * 10000 + n
    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        max_features=args.max_features,
        contamination="auto",
        bootstrap=False,
        n_jobs=-1,
        random_state=model_seed,
    )
    model.fit(x_train)

    threshold, rank = conformal_threshold(anomaly_scores(model, x_cal), args.false_alert_budget)
    h_scores = anomaly_scores(model, x_h)
    a_scores = anomaly_scores(model, x_a)
    h_pred = h_scores > threshold
    a_pred = a_scores > threshold

    fpr = float(h_pred.mean())
    recall = float(a_pred.mean())
    y = np.concatenate([np.zeros(len(h_scores), dtype=int), np.ones(len(a_scores), dtype=int)])
    s = np.concatenate([h_scores, a_scores])
    auroc = float(roc_auc_score(y, s))

    row = {
        "protocol_version": VERSION,
        "detector": "IsolationForest",
        "commissioning_size": n,
        "seed": seed,
        "commissioning_count": len(c_ids),
        "calibration_count": len(cal_ids),
        "healthy_eval_count": len(h_ids),
        "anomaly_eval_count": len(a_ids),
        "retained_features": int(prep.output_feature_count_),
        "conformal_rank": rank,
        "threshold": threshold,
        "false_positive_rate": fpr,
        "recall": recall,
        "auroc": auroc,
        "success": bool(recall >= args.recall_target and fpr <= args.false_alert_budget),
    }

    pred_by_id = dict(zip(anomaly.episode_ids.astype(int).tolist(), a_pred.astype(bool).tolist()))
    class_rows: list[dict[str, Any]] = []
    for (label, label_name), group in protocol["anomaly_eval"].groupby(["label", "label_name"], sort=True):
        ids = group.sample_nr.astype(int).tolist()
        values = np.asarray([pred_by_id[i] for i in ids], dtype=bool)
        class_rows.append({
            "protocol_version": VERSION,
            "detector": "IsolationForest",
            "commissioning_size": n,
            "seed": seed,
            "label": int(label),
            "label_name": str(label_name),
            "execution_count": len(ids),
            "recall": float(values.mean()),
        })
    return row, class_rows


def make_summary(results: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for n, g in results.groupby("commissioning_size", sort=True):
        rec = g.recall.to_numpy(float)
        fpr = g.false_positive_rate.to_numpy(float)
        auc = g.auroc.to_numpy(float)
        rl, ru = bootstrap_ci(rec, args.bootstrap_samples, args.confidence, args.global_seed + int(n) * 100)
        fl, fu = bootstrap_ci(fpr, args.bootstrap_samples, args.confidence, args.global_seed + int(n) * 100 + 1)
        al, au = bootstrap_ci(auc, args.bootstrap_samples, args.confidence, args.global_seed + int(n) * 100 + 2)
        rows.append({
            "protocol_version": VERSION,
            "detector": "IsolationForest",
            "commissioning_size": int(n),
            "seed_count": len(g),
            "mean_recall": float(rec.mean()),
            "recall_ci_lower": rl,
            "recall_ci_upper": ru,
            "mean_false_positive_rate": float(fpr.mean()),
            "fpr_ci_lower": fl,
            "fpr_ci_upper": fu,
            "mean_auroc": float(auc.mean()),
            "auroc_ci_lower": al,
            "auroc_ci_upper": au,
            "success_rate": float(g.success.astype(bool).mean()),
            "mean_retained_features": float(g.retained_features.mean()),
            "meets_joint_ci_criterion": bool(rl >= args.recall_target and fu <= args.false_alert_budget),
        })
    return pd.DataFrame(rows).sort_values("commissioning_size").reset_index(drop=True)


def make_class_summary(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []
    for (n, label, label_name), g in df.groupby(["commissioning_size", "label", "label_name"], sort=True):
        values = g.recall.to_numpy(float)
        lo, hi = bootstrap_ci(values, args.bootstrap_samples, args.confidence,
                              args.global_seed + int(n) * 100 + int(label) * 10 + 5)
        rows.append({
            "protocol_version": VERSION,
            "detector": "IsolationForest",
            "commissioning_size": int(n),
            "label": int(label),
            "label_name": str(label_name),
            "execution_count_per_seed": int(g.execution_count.iloc[0]),
            "seed_count": len(g),
            "mean_recall": float(values.mean()),
            "recall_ci_lower": lo,
            "recall_ci_upper": hi,
        })
    return pd.DataFrame(rows).sort_values(["commissioning_size", "label"]).reset_index(drop=True)


def n_star(summary: pd.DataFrame) -> dict[str, Any]:
    good = summary[summary.meets_joint_ci_criterion.astype(bool)].sort_values("commissioning_size")
    maximum = int(summary.commissioning_size.max())
    if good.empty:
        return {"status": "censored", "n_star": None, "display": f"Censored (>{maximum})", "maximum_tested_n": maximum}
    value = int(good.iloc[0].commissioning_size)
    return {"status": "observed", "n_star": value, "display": str(value), "maximum_tested_n": maximum}


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-path", type=Path, default=CACHE)
    p.add_argument("--protocol-dir", type=Path, default=PROTOCOL)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--grid", type=parse_csv_ints, default=GRID)
    p.add_argument("--seeds", type=parse_csv_ints, default=SEEDS)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-samples", default="auto")
    p.add_argument("--max-features", type=float, default=1.0)
    p.add_argument("--false-alert-budget", type=float, default=0.01)
    p.add_argument("--recall-target", type=float, default=0.90)
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--global-seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_path = args.cache_path.expanduser().resolve()
    args.protocol_dir = args.protocol_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.grid = tuple(int(x) for x in args.grid)
    args.seeds = tuple(int(x) for x in args.seeds)

    if not args.cache_path.exists():
        raise FileNotFoundError(args.cache_path)
    if args.n_estimators <= 0 or not 0 < args.max_features <= 1:
        raise ValueError("Invalid Isolation Forest parameters.")
    if not 0 < args.false_alert_budget < 1 or not 0 < args.recall_target <= 1:
        raise ValueError("Invalid evaluation targets.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "isolation_forest_seed_results.csv"
    class_seed_path = args.output_dir / "isolation_forest_per_class_seed_results.csv"
    summary_path = args.output_dir / "isolation_forest_summary.csv"
    class_summary_path = args.output_dir / "isolation_forest_per_class_recall.csv"
    nstar_path = args.output_dir / "isolation_forest_n_star.json"
    manifest_path = args.output_dir / "isolation_forest_run_manifest.json"

    if args.overwrite:
        for path in (seed_path, class_seed_path, summary_path, class_summary_path, nstar_path, manifest_path):
            if path.exists():
                path.unlink()

    print("=" * 72)
    print("AURSAD ISOLATION FOREST")
    print("=" * 72)
    print(f"Cache: {args.cache_path}")
    print(f"Grid: {list(args.grid)}")
    print(f"Seeds: {list(args.seeds)}")

    start = time.perf_counter()
    cache = load_feature_batch(args.cache_path)
    if cache.features.shape[1] != 288:
        raise ValueError(f"Expected 288 features, found {cache.features.shape[1]}.")

    protocol = load_protocol(args.protocol_dir)
    validate_protocol(protocol, args.grid, args.seeds)

    existing = pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    existing_class = pd.read_csv(class_seed_path) if class_seed_path.exists() else pd.DataFrame()
    done = set()
    if not existing.empty:
        done = {(int(r.commissioning_size), int(r.seed)) for r in existing.itertuples(index=False)}

    result_rows = existing.to_dict("records") if not existing.empty else []
    class_rows = existing_class.to_dict("records") if not existing_class.empty else []
    total = len(args.grid) * len(args.seeds)
    counter = len(done)

    for n in args.grid:
        for seed in args.seeds:
            if (n, seed) in done:
                continue
            counter += 1
            print(f"Processing N={n} seed={seed} ({counter}/{total})...")
            row, rows_class = run_one(cache, protocol, seed, n, args)
            result_rows.append(row)
            class_rows.extend(rows_class)
            atomic_csv(pd.DataFrame(result_rows).sort_values(["commissioning_size", "seed"]), seed_path)
            atomic_csv(pd.DataFrame(class_rows).sort_values(["commissioning_size", "seed", "label"]), class_seed_path)
            print(f"  recall={row['recall']:.4f} FPR={row['false_positive_rate']:.4f} AUROC={row['auroc']:.4f} success={row['success']}")

    results = pd.read_csv(seed_path)
    per_class = pd.read_csv(class_seed_path)
    summary = make_summary(results, args)
    class_summary = make_class_summary(per_class, args)
    estimate = n_star(summary)

    summary.to_csv(summary_path, index=False)
    class_summary.to_csv(class_summary_path, index=False)
    nstar_path.write_text(json.dumps({
        "protocol_version": VERSION,
        "detector": "IsolationForest",
        "criterion": {
            "recall_target": args.recall_target,
            "false_alert_budget": args.false_alert_budget,
            "confidence": args.confidence,
        },
        "estimate": estimate,
    }, indent=2), encoding="utf-8")

    manifest = {
        "run_version": VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": "AURSAD",
        "detector": {
            "name": "IsolationForest",
            "implementation": "sklearn.ensemble.IsolationForest",
            "n_estimators": args.n_estimators,
            "max_samples": str(args.max_samples),
            "max_features": args.max_features,
            "score_direction": "negative decision_function; larger is more anomalous",
        },
        "protocol": {
            "grid": list(args.grid),
            "seeds": list(args.seeds),
            "false_alert_budget": args.false_alert_budget,
            "recall_target": args.recall_target,
            "confidence": args.confidence,
            "bootstrap_samples": args.bootstrap_samples,
            "calibration_count": int(len(unique_ids(protocol["calibration"]))),
            "healthy_eval_count": int(len(unique_ids(protocol["healthy_eval"]))),
            "anomaly_eval_count": int(len(unique_ids(protocol["anomaly_eval"]))),
        },
        "input": {
            "cache_path": str(args.cache_path),
            "cache_sha256": sha256(args.cache_path),
            "cache_shape": list(cache.features.shape),
        },
        "result": {
            "run_count": int(len(results)),
            "n_star": estimate,
        },
        "outputs": {
            "seed_results": str(seed_path),
            "summary": str(summary_path),
            "per_class_seed_results": str(class_seed_path),
            "per_class_summary": str(class_summary_path),
            "n_star": str(nstar_path),
        },
        "timing_seconds": {"total": time.perf_counter() - start},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "validation": {
            "training_only_preprocessing": True,
            "normal_only_training": True,
            "fixed_split_conformal_calibration": True,
            "zero_partition_overlap": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("ISOLATION FOREST COMPLETE")
    print("=" * 72)
    print(summary[[
        "commissioning_size", "mean_recall", "recall_ci_lower", "recall_ci_upper",
        "mean_false_positive_rate", "fpr_ci_lower", "fpr_ci_upper",
        "mean_auroc", "success_rate", "meets_joint_ci_criterion",
    ]].to_string(index=False))
    print(f"\nEstimated N*: {estimate['display']}")
    print("\nArtifacts:")
    for p in (seed_path, summary_path, class_seed_path, class_summary_path, nstar_path, manifest_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()