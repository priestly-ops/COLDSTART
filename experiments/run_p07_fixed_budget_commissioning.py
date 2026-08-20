"""P0.7: fixed total commissioning-budget evaluation on voraus-AD.

Scientific question
-------------------
How many *target healthy commissioning cycles in total* are required before a
method meets the deployment criterion?  Unlike P0.5, the budget here counts
both model-fitting and calibration cycles:

    B = N_fit + N_calibration

The same target-data allocation is used by every detector at a given budget.
No detector receives a method-specific allocation.

Frozen data protocol
--------------------
* PRE_A (setting 72): source healthy training population.
* PRE_B (setting 73): target healthy population.
* A fixed 70-cycle PRE_B holdout is used only for same-domain FPR evaluation.
* The remaining PRE_B healthy cycles form the commissioning pool.
* For each commissioning seed, one permutation of that pool is generated.
  Fit and calibration sets are disjoint slices from the same permutation.
* All anomalous episodes are used only for final recall/AUROC/AUPRC evaluation.
* BETWEEN_A/B/C are not used for the primary FPR endpoint; P0.6b showed that
  they are strongly shifted and are therefore robustness-only populations.

Frozen candidate allocations
----------------------------
  B=175: fit=25, cal=150
  B=200: fit=25, cal=175
  B=224: fit=25, cal=199  (first 1% conformal non-max threshold)
  B=225: fit=25, cal=200
  B=249: fit=50, cal=199

Primary endpoint
----------------
For each method and B, across commissioning seeds:
  recall lower 95% bootstrap CI >= 0.90
  FPR upper 95% bootstrap CI <= 0.01

B* is the smallest tested total commissioning budget satisfying both.
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
from src.split_generator import SOURCE_SETTING, TARGET_SETTING
from src.voraus_loader import load_cycle_metadata

# Reuse the already-frozen P0.5 detector fitting/scoring implementation rather
# than creating a new algorithm variant after seeing anomaly results.
from experiments.run_p05_anomaly_commissioning import (
    METHODS,
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    DEFAULT_RACE_LAMBDAS,
    DEFAULT_RIDGE_GAMMAS,
    DEFAULT_CV_FOLDS,
    _bootstrap_mean,
    _ensure_feature_cache,
    _fit_estimate,
    _fit_method_scaler,
    _rows,
    _scores,
)

PROTOCOL_VERSION = "p07-fixed-budget-preb-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p07_fixed_budget_commissioning"
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_EVALUATION_SEED = 20260821
DEFAULT_EVAL_SIZE = 70
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90

# Predeclared before running P0.7 anomaly evaluation.
DEFAULT_ALLOCATIONS: tuple[tuple[int, int, int], ...] = (
    (175, 25, 150),
    (200, 25, 175),
    (224, 25, 199),
    (225, 25, 200),
    (249, 50, 199),
)


def _sha256_ints(values: list[int]) -> str:
    arr = np.asarray(sorted(int(v) for v in values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _append_row(path: Path, row: dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=(not path.exists()) or path.stat().st_size == 0,
        index=False,
    )


def _allocation_map() -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for budget, fit_n, cal_n in DEFAULT_ALLOCATIONS:
        if fit_n + cal_n != budget:
            raise RuntimeError("Invalid frozen allocation")
        out[int(budget)] = (int(fit_n), int(cal_n))
    return out


def _frozen_preb_partitions(metadata, *, seed: int, eval_seed: int, eval_size: int):
    source = sorted(
        [c for c in metadata if (not c.anomaly) and int(c.setting) == int(SOURCE_SETTING)],
        key=lambda c: c.episode_id,
    )
    target = sorted(
        [c for c in metadata if (not c.anomaly) and int(c.setting) == int(TARGET_SETTING)],
        key=lambda c: c.episode_id,
    )
    anomalies = sorted([c for c in metadata if c.anomaly], key=lambda c: c.episode_id)

    if len(target) < eval_size + max(b for b, _, _ in DEFAULT_ALLOCATIONS):
        raise ValueError(
            f"Need {eval_size + max(b for b, _, _ in DEFAULT_ALLOCATIONS)} PRE_B healthy cycles; "
            f"found {len(target)}"
        )

    eval_rng = np.random.default_rng(int(eval_seed))
    perm = eval_rng.permutation(len(target))
    eval_cycles = [target[int(i)] for i in perm[:eval_size]]
    eval_ids = {int(c.episode_id) for c in eval_cycles}
    commissioning_pool = [c for c in target if int(c.episode_id) not in eval_ids]

    seed_rng = np.random.default_rng(int(seed))
    seed_perm = seed_rng.permutation(len(commissioning_pool))
    ordered_pool = [commissioning_pool[int(i)] for i in seed_perm]

    return source, ordered_pool, sorted(eval_cycles, key=lambda c: c.episode_id), anomalies


def _summarize(results: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gi, ((budget, method), g) in enumerate(results.groupby(["budget", "method"], sort=True)):
        recall_mean, recall_lo, recall_hi = _bootstrap_mean(
            g.recall.to_numpy(), args.bootstrap_replicates, 110_000 + gi
        )
        fpr_mean, fpr_lo, fpr_hi = _bootstrap_mean(
            g.false_positive_rate.to_numpy(), args.bootstrap_replicates, 120_000 + gi
        )
        auroc_mean, auroc_lo, auroc_hi = _bootstrap_mean(
            g.auroc.to_numpy(), args.bootstrap_replicates, 130_000 + gi
        )
        auprc_mean, auprc_lo, auprc_hi = _bootstrap_mean(
            g.auprc.to_numpy(), args.bootstrap_replicates, 140_000 + gi
        )
        rows.append({
            "budget": int(budget),
            "fit_n": int(g.fit_n.iloc[0]),
            "calibration_n": int(g.calibration_n.iloc[0]),
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
            "fraction_threshold_is_calibration_max": float(g.calibration_threshold_is_max.mean()),
            "median_threshold": float(g.threshold.median()),
            "median_source_weight": float(g.source_weight.median()),
            "fraction_transfer_accepted": float(g.accepted_transfer.mean()),
        })
    return pd.DataFrame(rows)


def _b_star(summary: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    max_budget = int(summary.budget.max())
    for method in METHODS:
        g = summary[summary.method == method].sort_values("budget")
        qualified = g[
            (g.recall_ci_lower >= float(args.recall_target))
            & (g.fpr_ci_upper <= float(args.false_alert_budget))
        ]
        out[method] = int(qualified.iloc[0].budget) if not qualified.empty else f"Censored (>{max_budget})"
    return out


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    dataset = Path(args.dataset).resolve()
    cache = Path(args.feature_cache).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "p07_seed_results.csv"

    if args.no_resume and raw_path.exists():
        raw_path.unlink()

    allocations = _allocation_map()
    requested_budgets = [int(v) for v in args.budgets]
    unknown = sorted(set(requested_budgets) - set(allocations))
    if unknown:
        raise ValueError(f"Budgets not in frozen allocation set: {unknown}")

    metadata = load_cycle_metadata(dataset)
    batch = _ensure_feature_cache(dataset, cache, args.signal_set)

    existing = pd.read_csv(raw_path) if raw_path.exists() and raw_path.stat().st_size else pd.DataFrame()
    completed: set[tuple[int, int, str]] = set()
    if not existing.empty:
        if set(existing.protocol_version.astype(str).unique()) != {PROTOCOL_VERSION}:
            raise RuntimeError("Checkpoint protocol mismatch")
        completed = {
            (int(r.budget), int(r.seed), str(r.method))
            for r in existing.itertuples()
        }

    for budget in requested_budgets:
        fit_n, cal_n = allocations[budget]
        for si, seed in enumerate(args.seeds, start=1):
            print(
                f"P0.7 B={budget} fit={fit_n} cal={cal_n} seed={seed} "
                f"({si}/{len(args.seeds)})",
                flush=True,
            )
            source_cycles, pool, eval_cycles, anomaly_cycles = _frozen_preb_partitions(
                metadata,
                seed=int(seed),
                eval_seed=int(args.evaluation_seed),
                eval_size=int(args.eval_size),
            )
            if fit_n + cal_n > len(pool):
                raise RuntimeError("Frozen allocation exceeds commissioning pool")

            fit_cycles = pool[:fit_n]
            cal_cycles = pool[fit_n:fit_n + cal_n]

            source_ids = [int(c.episode_id) for c in source_cycles]
            fit_ids = [int(c.episode_id) for c in fit_cycles]
            cal_ids = [int(c.episode_id) for c in cal_cycles]
            eval_ids = [int(c.episode_id) for c in eval_cycles]
            anomaly_ids = [int(c.episode_id) for c in anomaly_cycles]

            groups = [set(source_ids), set(fit_ids), set(cal_ids), set(eval_ids), set(anomaly_ids)]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        raise RuntimeError("P0.7 data leakage detected")

            source_raw = _rows(batch, source_ids)
            target_raw = _rows(batch, fit_ids)
            calibration_raw = _rows(batch, cal_ids)
            normal_raw = _rows(batch, eval_ids)
            anomaly_raw = _rows(batch, anomaly_ids)

            cv_seed = 6_200_000 + int(budget) * 1000 + int(seed)

            for method in METHODS:
                key = (budget, int(seed), method)
                if key in completed:
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
                    n=int(fit_n),
                    cv_seed=cv_seed,
                    args=args,
                )

                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)

                threshold = BaseDetector.conformal_quantile(
                    cal_scores, alpha=float(args.false_alert_budget)
                )
                normal_pred = normal_scores > threshold
                anomaly_pred = anomaly_scores > threshold
                fpr = float(np.mean(normal_pred))
                recall = float(np.mean(anomaly_pred))

                y = np.concatenate((
                    np.zeros(len(normal_scores), dtype=int),
                    np.ones(len(anomaly_scores), dtype=int),
                ))
                s = np.concatenate((normal_scores, anomaly_scores))
                auroc = float(roc_auc_score(y, s))
                auprc = float(average_precision_score(y, s))

                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "budget": int(budget),
                    "fit_n": int(fit_n),
                    "calibration_n": int(cal_n),
                    "seed": int(seed),
                    "method": method,
                    "false_positive_rate": fpr,
                    "recall": recall,
                    "success": bool(
                        recall >= float(args.recall_target)
                        and fpr <= float(args.false_alert_budget)
                    ),
                    "auroc": auroc,
                    "auprc": auprc,
                    "threshold": float(threshold),
                    "calibration_max_score": float(np.max(cal_scores)),
                    "calibration_threshold_is_max": bool(
                        np.isclose(threshold, np.max(cal_scores))
                    ),
                    "source_weight": float(diag["source_weight"]),
                    "selected_lambda": (
                        float(diag["selected_lambda"])
                        if np.isfinite(diag["selected_lambda"])
                        else np.nan
                    ),
                    "accepted_transfer": bool(diag["accepted_transfer"]),
                    "policy_fallback": bool(diag["policy_fallback"]),
                    "source_n": int(len(source_x)),
                    "normal_eval_n": int(len(normal_x)),
                    "anomaly_eval_n": int(len(anomaly_x)),
                    "source_ids_sha256": _sha256_ints(source_ids),
                    "fit_ids_sha256": _sha256_ints(fit_ids),
                    "calibration_ids_sha256": _sha256_ints(cal_ids),
                    "normal_eval_ids_sha256": _sha256_ints(eval_ids),
                    "anomaly_eval_ids_sha256": _sha256_ints(anomaly_ids),
                }
                _append_row(raw_path, row)
                completed.add(key)

    results = pd.read_csv(raw_path)
    results = results[results.budget.isin(requested_budgets)].copy()
    summary = _summarize(results, args)
    summary.to_csv(output / "p07_summary.csv", index=False)

    bstar = _b_star(summary, args)
    (output / "p07_b_star.json").write_text(
        json.dumps(bstar, indent=2), encoding="utf-8"
    )

    split_audit = results.groupby(["budget", "seed"], sort=True).agg(
        source_hashes=("source_ids_sha256", "nunique"),
        fit_hashes=("fit_ids_sha256", "nunique"),
        calibration_hashes=("calibration_ids_sha256", "nunique"),
        normal_eval_hashes=("normal_eval_ids_sha256", "nunique"),
        anomaly_eval_hashes=("anomaly_eval_ids_sha256", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()
    split_audit.to_csv(output / "p07_split_audit.csv", index=False)
    if not (
        (split_audit.source_hashes == 1)
        & (split_audit.fit_hashes == 1)
        & (split_audit.calibration_hashes == 1)
        & (split_audit.normal_eval_hashes == 1)
        & (split_audit.anomaly_eval_hashes == 1)
        & (split_audit.methods == len(METHODS))
    ).all():
        raise RuntimeError("P0.7 method-pairing audit failed")

    global_audit = {
        "source_hashes_global": int(results.source_ids_sha256.nunique()),
        "normal_eval_hashes_global": int(results.normal_eval_ids_sha256.nunique()),
        "anomaly_eval_hashes_global": int(results.anomaly_eval_ids_sha256.nunique()),
    }
    if global_audit != {
        "source_hashes_global": 1,
        "normal_eval_hashes_global": 1,
        "anomaly_eval_hashes_global": 1,
    }:
        raise RuntimeError(f"P0.7 global frozen-evaluation audit failed: {global_audit}")

    safe = results[results.method == "RACECovSafeCV"].groupby("budget", sort=True).agg(
        seeds=("seed", "nunique"),
        fit_n=("fit_n", "first"),
        calibration_n=("calibration_n", "first"),
        median_lambda=("selected_lambda", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_transfer_accepted=("accepted_transfer", "mean"),
        fraction_threshold_is_calibration_max=("calibration_threshold_is_max", "mean"),
    ).reset_index()
    safe.to_csv(output / "p07_source_weight_audit.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "feature_cache": str(cache),
        "source_setting": int(SOURCE_SETTING),
        "target_setting": int(TARGET_SETTING),
        "evaluation_seed": int(args.evaluation_seed),
        "normal_eval_size": int(args.eval_size),
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
        "allocations": [
            {"budget": b, "fit_n": allocations[b][0], "calibration_n": allocations[b][1]}
            for b in requested_budgets
        ],
        "seeds": [int(v) for v in args.seeds],
        "methods": list(METHODS),
        "anomaly_labels_used_for_fit_selection_or_calibration": False,
        "same_allocation_for_all_methods_within_budget": True,
        "between_settings_used_for_primary_fpr": False,
        "global_split_audit": global_audit,
    }
    (output / "p07_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nP0.7 summary\n", summary.to_string(index=False), flush=True)
    print("\nB*\n", json.dumps(bstar, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--signal-set", default="measured", choices=["measured", "machine"])
    ap.add_argument("--budgets", type=int, nargs="+", default=[b for b, _, _ in DEFAULT_ALLOCATIONS])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    ap.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    ap.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.eval_size != DEFAULT_EVAL_SIZE:
        ap.error("P0.7 primary protocol freezes PRE_B holdout size at 70")
    if not (0.0 < args.false_alert_budget < 1.0):
        ap.error("false-alert-budget must be in (0,1)")
    if not (0.0 < args.recall_target <= 1.0):
        ap.error("recall-target must be in (0,1]")
    return args


if __name__ == "__main__":
    run(parse_args())
