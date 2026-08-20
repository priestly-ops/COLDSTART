"""P0.5: frozen anomaly commissioning evaluation on voraus-AD.

This is the first anomaly-label evaluation of the covariance-transfer branch
that survived P0.3c synthetic stress and P0.4 real-healthy validation.

Scientific constraints
----------------------
* The healthy-data method is frozen before this experiment.
* RACECovSafeCV uses the P0.3b/P0.4 healthy-only selector unchanged.
* N < 25 uses the predeclared target-only fallback.
* Calibration uses healthy target cycles only and the finite-sample split-
  conformal order statistic already used by the project.
* A newly frozen healthy calibration/evaluation partition is used here so
  P0.5 does not reuse P0.4's healthy evaluation partition.
* No anomaly labels are used for fitting, preprocessing, source gating, or
  threshold calibration. They are used only for final evaluation metrics.

Primary endpoint
----------------
For each detector and commissioning N, across commissioning seeds:
  - recall lower 95% percentile-bootstrap CI >= 0.90, and
  - FPR upper 95% percentile-bootstrap CI <= 0.01.
N* is the smallest tested N satisfying both. If no N qualifies, it is censored
above the largest feasible N tested on this dataset.

Detectors
---------
1. BestTargetOnlySafeCV: strong target-only covariance baseline.
2. SourceOnly: source-only Ledoit-Wolf Gaussian with source-only preprocessing.
3. Pooled: pooled source+target Ledoit-Wolf with pooled preprocessing.
4. RACECov60: fixed covariance-only RACE lambda=60; target mean is retained.
5. RACECovSafeCV: frozen safe covariance transfer; target mean is retained.

The two RACE variants transfer covariance only. This matches the healthy-data
branch isolated by P0.3/P0.4 and avoids reintroducing mean transfer after the
healthy estimator was frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector
from src.covariance_transfer_estimators import (
    CovarianceEstimate,
    ledoit_wolf_covariance,
    pooled_ledoit_wolf,
    race_covariance,
    safe_cv_race_covariance,
    safe_cv_target_only,
)
from src.feature_extractor import extract_feature_batch, load_feature_batch, save_feature_batch
from src.precision_transfer_estimators import fit_robust_scaler
from src.split_generator import SOURCE_SETTING, TARGET_SETTING, create_frozen_evaluation_split
from src.voraus_loader import load_cycle_metadata, load_cycles

PROTOCOL_VERSION = "p05-voraus-anomaly-commissioning-v1"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p05_anomaly_commissioning"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "cache" / "voraus_measured_all_features.npz"
DEFAULT_NS = (10, 25, 50, 100)
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_CALIBRATION_SIZE = 100
DEFAULT_EVAL_SIZE = 100
# Deliberately different from P0.4's evaluation_seed=42.
DEFAULT_EVALUATION_SEED = 20260820
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
DEFAULT_CV_FOLDS = 5
DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90
MIN_TRANSFER_N = 25
METHODS = (
    "BestTargetOnlySafeCV",
    "SourceOnly",
    "Pooled",
    "RACECov60",
    "RACECovSafeCV",
)


def _sha256_ints(values: list[int]) -> str:
    arr = np.asarray(sorted(int(v) for v in values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _ensure_feature_cache(dataset: Path, cache: Path, signal_set: str):
    if cache.exists():
        return load_feature_batch(cache)
    print(f"Building full feature cache at {cache} ...", flush=True)
    cycles = load_cycles(dataset, signal_set=signal_set)
    batch = extract_feature_batch(cycles)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_feature_batch(
        batch,
        cache,
        metadata={
            "protocol": PROTOCOL_VERSION,
            "dataset": str(dataset),
            "signal_set": signal_set,
            "contains_anomalies": True,
        },
    )
    return batch


def _rows(batch, ids: list[int]) -> np.ndarray:
    return np.asarray(
        batch.select_episode_ids(ids, preserve_requested_order=True, require_all=True).features,
        dtype=np.float64,
    )


def _fit_method_scaler(method: str, source_raw: np.ndarray, target_raw: np.ndarray):
    if method == "SourceOnly":
        return fit_robust_scaler(source_raw, mode="target")
    if method == "Pooled":
        return fit_robust_scaler(np.vstack((source_raw, target_raw)), mode="target")
    # TargetOnly and both covariance-transfer methods share target-only
    # preprocessing, ensuring source information cannot improve TargetOnly.
    return fit_robust_scaler(target_raw, mode="target")


def _fit_estimate(
    method: str,
    source_x: np.ndarray,
    target_x: np.ndarray,
    *,
    n: int,
    cv_seed: int,
    args: argparse.Namespace,
) -> tuple[CovarianceEstimate, np.ndarray, dict[str, object]]:
    """Return covariance estimate, location, and selector diagnostics."""
    if method == "BestTargetOnlySafeCV":
        est, _ = safe_cv_target_only(
            target_x,
            ridge_gammas=tuple(args.ridge_gammas),
            n_folds=min(int(args.cv_folds), len(target_x)),
            seed=cv_seed,
            method="BestTargetOnlySafeCV",
        )
        return est, np.mean(target_x, axis=0), {
            "selected_lambda": np.nan,
            "source_weight": 0.0,
            "accepted_transfer": False,
            "policy_fallback": False,
        }

    if method == "SourceOnly":
        est = ledoit_wolf_covariance(source_x, method="SourceOnly")
        return est, np.mean(source_x, axis=0), {
            "selected_lambda": np.nan,
            "source_weight": 1.0,
            "accepted_transfer": False,
            "policy_fallback": False,
        }

    if method == "Pooled":
        est = pooled_ledoit_wolf(target_x, source_x)
        est = CovarianceEstimate(est.covariance, est.precision, "Pooled", est.metadata)
        return est, np.mean(np.vstack((source_x, target_x)), axis=0), {
            "selected_lambda": np.nan,
            "source_weight": float(len(source_x) / (len(source_x) + len(target_x))),
            "accepted_transfer": False,
            "policy_fallback": False,
        }

    if method == "RACECov60":
        est = race_covariance(target_x, source_x, lambda_reg=60.0, method="RACECov60")
        return est, np.mean(target_x, axis=0), {
            "selected_lambda": 60.0,
            "source_weight": float(est.metadata["source_weight"]),
            "accepted_transfer": True,
            "policy_fallback": False,
        }

    if method == "RACECovSafeCV":
        if int(n) < MIN_TRANSFER_N:
            est, _ = safe_cv_target_only(
                target_x,
                ridge_gammas=tuple(args.ridge_gammas),
                n_folds=min(int(args.cv_folds), len(target_x)),
                seed=cv_seed,
                method="RACECovSafeCV",
            )
            return est, np.mean(target_x, axis=0), {
                "selected_lambda": 0.0,
                "source_weight": 0.0,
                "accepted_transfer": False,
                "policy_fallback": True,
            }
        safe = safe_cv_race_covariance(
            target_x,
            source_x,
            lambdas=tuple(args.race_lambdas),
            n_folds=min(int(args.cv_folds), len(target_x)),
            seed=cv_seed,
            se_multiplier=float(args.se_multiplier),
            method="RACECovSafeCV",
        )
        return safe.estimate, np.mean(target_x, axis=0), {
            "selected_lambda": float(safe.selected_lambda),
            "source_weight": float(safe.selected_source_weight),
            "accepted_transfer": bool(safe.accepted_transfer),
            "policy_fallback": False,
        }

    raise ValueError(f"Unknown method {method}")


def _scores(x: np.ndarray, location: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = np.asarray(x, dtype=np.float64) - np.asarray(location, dtype=np.float64)
    sq = np.einsum("ni,ij,nj->n", centered, precision, centered, optimize=True)
    return np.sqrt(np.maximum(sq, 0.0))


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boot[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.mean(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _append_row(path: Path, row: dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=(not path.exists()) or path.stat().st_size == 0,
        index=False,
    )


def _summarize(results: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gi, ((n, method), g) in enumerate(results.groupby(["N", "method"], sort=True)):
        recall_mean, recall_lo, recall_hi = _bootstrap_mean(
            g.recall.to_numpy(), args.bootstrap_replicates, 70_000 + gi
        )
        fpr_mean, fpr_lo, fpr_hi = _bootstrap_mean(
            g.false_positive_rate.to_numpy(), args.bootstrap_replicates, 80_000 + gi
        )
        auroc_mean, auroc_lo, auroc_hi = _bootstrap_mean(
            g.auroc.to_numpy(), args.bootstrap_replicates, 90_000 + gi
        )
        auprc_mean, auprc_lo, auprc_hi = _bootstrap_mean(
            g.auprc.to_numpy(), args.bootstrap_replicates, 100_000 + gi
        )
        rows.append({
            "N": int(n),
            "method": str(method),
            "seeds": int(g.seed.nunique()),
            "recall_mean": recall_mean,
            "recall_ci_lower": recall_lo,
            "recall_ci_upper": recall_hi,
            "fpr_mean": fpr_mean,
            "fpr_ci_lower": fpr_lo,
            "fpr_ci_upper": fpr_hi,
            "success_rate": float(g.success.mean()),
            "auroc_mean": auroc_mean,
            "auroc_ci_lower": auroc_lo,
            "auroc_ci_upper": auroc_hi,
            "auprc_mean": auprc_mean,
            "auprc_ci_lower": auprc_lo,
            "auprc_ci_upper": auprc_hi,
            "median_threshold": float(g.threshold.median()),
            "median_source_weight": float(g.source_weight.median()),
            "fraction_transfer_accepted": float(g.accepted_transfer.mean()),
            "fraction_policy_fallback": float(g.policy_fallback.mean()),
        })
    return pd.DataFrame(rows)


def _n_star(summary: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for method in METHODS:
        g = summary[summary.method == method].sort_values("N")
        qualified = g[
            (g.recall_ci_lower >= float(args.recall_target))
            & (g.fpr_ci_upper <= float(args.false_alert_budget))
        ]
        if qualified.empty:
            out[method] = f"Censored (>{max(args.ns)})"
        else:
            out[method] = int(qualified.iloc[0].N)
    return out


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    cache = Path(args.feature_cache).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "p05_seed_results.csv"

    if args.no_resume and raw_path.exists():
        raw_path.unlink()

    metadata = load_cycle_metadata(dataset)
    batch = _ensure_feature_cache(dataset, cache, args.signal_set)

    existing = pd.read_csv(raw_path) if raw_path.exists() and raw_path.stat().st_size else pd.DataFrame()
    completed = set()
    if not existing.empty:
        required = {"protocol_version", "N", "seed", "method"}
        if not required.issubset(existing.columns):
            raise RuntimeError("Existing P0.5 checkpoint is incompatible")
        protocols = set(existing.protocol_version.astype(str).unique())
        if protocols != {PROTOCOL_VERSION}:
            raise RuntimeError(f"Checkpoint protocol mismatch: {protocols}")
        completed = {(int(r.N), int(r.seed), str(r.method)) for r in existing.itertuples()}

    maximum_n = max(int(v) for v in args.ns)
    for n in args.ns:
        for si, seed in enumerate(args.seeds, start=1):
            print(f"P0.5 N={n} seed={seed} ({si}/{len(args.seeds)})", flush=True)
            split = create_frozen_evaluation_split(
                metadata,
                commissioning_size=int(n),
                commissioning_seed=int(seed),
                evaluation_seed=int(args.evaluation_seed),
                calibration_size=int(args.calibration_size),
                normal_evaluation_size=int(args.eval_size),
                maximum_commissioning_size=maximum_n,
            )
            split.verify_no_overlap()

            source_ids = [int(c.episode_id) for c in split.source_train]
            target_ids = [int(c.episode_id) for c in split.target_commissioning]
            calibration_ids = [int(c.episode_id) for c in split.target_calibration]
            normal_ids = [int(c.episode_id) for c in split.target_normal_evaluation]
            anomaly_ids = [int(c.episode_id) for c in split.target_anomaly_evaluation]

            source_raw = _rows(batch, source_ids)
            target_raw = _rows(batch, target_ids)
            calibration_raw = _rows(batch, calibration_ids)
            normal_raw = _rows(batch, normal_ids)
            anomaly_raw = _rows(batch, anomaly_ids)
            anomaly_categories = batch.select_episode_ids(anomaly_ids).categories

            cv_seed = 5_200_000 + int(n) * 1000 + int(seed)

            for method in METHODS:
                if (int(n), int(seed), method) in completed:
                    continue
                scaler = _fit_method_scaler(method, source_raw, target_raw)
                source_x = scaler.transform(source_raw)
                target_x = scaler.transform(target_raw)
                calibration_x = scaler.transform(calibration_raw)
                normal_x = scaler.transform(normal_raw)
                anomaly_x = scaler.transform(anomaly_raw)

                est, location, diag = _fit_estimate(
                    method,
                    source_x,
                    target_x,
                    n=int(n),
                    cv_seed=cv_seed,
                    args=args,
                )
                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)
                threshold = BaseDetector.conformal_quantile(cal_scores, alpha=float(args.false_alert_budget))
                normal_pred = normal_scores > threshold
                anomaly_pred = anomaly_scores > threshold
                fpr = float(np.mean(normal_pred))
                recall = float(np.mean(anomaly_pred))

                y = np.concatenate((np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)))
                s = np.concatenate((normal_scores, anomaly_scores))
                auroc = float(roc_auc_score(y, s))
                auprc = float(average_precision_score(y, s))

                category_recalls = {}
                for cat in sorted(set(int(v) for v in anomaly_categories.tolist())):
                    mask = anomaly_categories == cat
                    category_recalls[str(cat)] = float(np.mean(anomaly_pred[mask]))

                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "N": int(n),
                    "seed": int(seed),
                    "method": method,
                    "false_positive_rate": fpr,
                    "recall": recall,
                    "success": bool(recall >= args.recall_target and fpr <= args.false_alert_budget),
                    "auroc": auroc,
                    "auprc": auprc,
                    "threshold": float(threshold),
                    "calibration_max_score": float(np.max(cal_scores)),
                    "calibration_threshold_is_max": bool(np.isclose(threshold, np.max(cal_scores))),
                    "source_weight": float(diag["source_weight"]),
                    "selected_lambda": float(diag["selected_lambda"]) if np.isfinite(diag["selected_lambda"]) else np.nan,
                    "accepted_transfer": bool(diag["accepted_transfer"]),
                    "policy_fallback": bool(diag["policy_fallback"]),
                    "feature_count": int(target_x.shape[1]),
                    "source_n": int(len(source_x)),
                    "target_n": int(len(target_x)),
                    "calibration_n": int(len(calibration_x)),
                    "normal_eval_n": int(len(normal_x)),
                    "anomaly_eval_n": int(len(anomaly_x)),
                    "source_ids_sha256": _sha256_ints(source_ids),
                    "commissioning_ids_sha256": _sha256_ints(target_ids),
                    "calibration_ids_sha256": _sha256_ints(calibration_ids),
                    "normal_eval_ids_sha256": _sha256_ints(normal_ids),
                    "anomaly_eval_ids_sha256": _sha256_ints(anomaly_ids),
                    "category_recalls_json": json.dumps(category_recalls, sort_keys=True),
                }
                _append_row(raw_path, row)
                completed.add((int(n), int(seed), method))

    results = pd.read_csv(raw_path)
    summary = _summarize(results, args)
    summary.to_csv(output / "p05_summary.csv", index=False)

    nstar = _n_star(summary, args)
    (output / "p05_n_star.json").write_text(json.dumps(nstar, indent=2), encoding="utf-8")

    split_audit = results.groupby(["N", "seed"], sort=True).agg(
        source_hashes=("source_ids_sha256", "nunique"),
        commissioning_hashes=("commissioning_ids_sha256", "nunique"),
        calibration_hashes=("calibration_ids_sha256", "nunique"),
        normal_eval_hashes=("normal_eval_ids_sha256", "nunique"),
        anomaly_eval_hashes=("anomaly_eval_ids_sha256", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()
    split_audit.to_csv(output / "p05_split_audit.csv", index=False)
    if not (
        (split_audit.source_hashes == 1)
        & (split_audit.commissioning_hashes == 1)
        & (split_audit.calibration_hashes == 1)
        & (split_audit.normal_eval_hashes == 1)
        & (split_audit.anomaly_eval_hashes == 1)
        & (split_audit.methods == len(METHODS))
    ).all():
        raise RuntimeError("P0.5 split/method pairing audit failed")

    safe = results[results.method == "RACECovSafeCV"].groupby("N", sort=True).agg(
        seeds=("seed", "nunique"),
        median_lambda=("selected_lambda", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_zero_source_weight=("source_weight", lambda x: float(np.mean(np.isclose(x, 0.0)))),
        fraction_transfer_accepted=("accepted_transfer", "mean"),
        fraction_policy_fallback=("policy_fallback", "mean"),
    ).reset_index()
    safe.to_csv(output / "p05_source_weight_audit.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "feature_cache": str(cache),
        "source_setting": int(SOURCE_SETTING),
        "target_setting": int(TARGET_SETTING),
        "evaluation_seed": int(args.evaluation_seed),
        "calibration_size": int(args.calibration_size),
        "normal_eval_size": int(args.eval_size),
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
        "min_transfer_n": MIN_TRANSFER_N,
        "ns": [int(v) for v in args.ns],
        "seeds": [int(v) for v in args.seeds],
        "methods": list(METHODS),
        "anomaly_labels_used_for_fit_or_selection": False,
        "healthy_method_frozen_before_p05": True,
        "p04_evaluation_partition_reused": False,
    }
    (output / "p05_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nP0.5 summary\n", summary.to_string(index=False), flush=True)
    print("\nN*\n", json.dumps(nstar, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--signal-set", default="measured", choices=["measured", "machine"])
    ap.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    ap.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    ap.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    ap.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if not args.ns or any(int(n) < 3 for n in args.ns):
        ap.error("All N values must be >=3")
    if max(args.ns) > 119:
        ap.error("With 100 calibration + 100 healthy-eval cycles, frozen voraus protocol supports max N=119")
    if 25 not in args.ns or 50 not in args.ns:
        print("Warning: final commissioning curve normally includes N=25 and N=50", file=sys.stderr)
    if not (0.0 < args.false_alert_budget < 1.0):
        ap.error("false-alert-budget must be in (0,1)")
    if not (0.0 < args.recall_target <= 1.0):
        ap.error("recall-target must be in (0,1]")
    return args


if __name__ == "__main__":
    run(parse_args())
