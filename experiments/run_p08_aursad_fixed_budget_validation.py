"""P0.8: fixed total commissioning-budget validation on AURSAD.

Frozen healthy-domain split from the acquisition-block audit:
- source healthy: segments 0--9, equivalently sample_nr <= 2336
- target healthy: segments 10--18, equivalently sample_nr >= 2337

Primary anomaly evaluation uses all tightening anomalies (labels 1--4).
Label 5 (loosening/picking) is excluded.

Deployment commissioning budget is B = N_fit + N_calibration. A fixed
100-cycle target-healthy holdout is used only for FPR evaluation. Every method
receives identical episode IDs for source, fit, calibration, normal evaluation,
and anomaly evaluation at each budget/seed. No anomaly labels are used for
fitting, scaling, transfer selection, or calibration.
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

from src.aursad_loader import DEFAULT_DATA_PATH, DEFAULT_INVENTORY_PATH, load_episode_inventory
from src.base_detector import BaseDetector
from src.feature_extractor import load_feature_batch
from experiments.run_p05_anomaly_commissioning import (
    METHODS,
    DEFAULT_RACE_LAMBDAS,
    DEFAULT_RIDGE_GAMMAS,
    DEFAULT_CV_FOLDS,
    _bootstrap_mean,
    _fit_estimate,
    _fit_method_scaler,
    _rows,
    _scores,
)

PROTOCOL_VERSION = "p08-aursad-fixed-budget-block-transfer-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p08_aursad_fixed_budget"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "aursad" / "feature_cache" / "aursad_features.npz"
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_EVALUATION_SEED = 20260822
DEFAULT_EVAL_SIZE = 100
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_FALSE_ALERT_BUDGET = 0.01
DEFAULT_RECALL_TARGET = 0.90

SOURCE_LAST_SAMPLE_NR = 2336
TARGET_FIRST_SAMPLE_NR = 2337
NORMAL_LABEL = 0
ANOMALY_LABELS = (1, 2, 3, 4)
CATEGORY_NAMES = {
    1: "damaged_screw",
    2: "extra_component",
    3: "missing_screw",
    4: "damaged_thread",
}

DEFAULT_ALLOCATIONS = (
    (175, 25, 150),
    (224, 25, 199),
    (249, 50, 199),
    (300, 100, 200),
    (400, 150, 250),
)


def _sha256_ints(values):
    arr = np.asarray(sorted(int(v) for v in values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _append_row(path: Path, row: dict[str, object]) -> None:
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=(not path.exists()) or path.stat().st_size == 0,
        index=False,
    )


def _allocation_map():
    out = {}
    for budget, fit_n, cal_n in DEFAULT_ALLOCATIONS:
        if fit_n + cal_n != budget:
            raise RuntimeError("Invalid frozen P0.8 allocation")
        out[int(budget)] = (int(fit_n), int(cal_n))
    return out


def _inventory_partitions(inventory, *, seed: int, eval_seed: int, eval_size: int):
    inv = inventory.copy()
    inv["sample_nr"] = pd.to_numeric(inv["sample_nr"], errors="raise").astype(np.int64)
    inv["label"] = pd.to_numeric(inv["label"], errors="raise").astype(np.int64)

    source_ids = (
        inv[inv.label.eq(0) & inv.sample_nr.le(SOURCE_LAST_SAMPLE_NR)]
        .sort_values("sample_nr").sample_nr.astype(int).tolist()
    )
    target_ids = (
        inv[inv.label.eq(0) & inv.sample_nr.ge(TARGET_FIRST_SAMPLE_NR)]
        .sort_values("sample_nr").sample_nr.astype(int).tolist()
    )
    anomaly_frame = inv[inv.label.isin(ANOMALY_LABELS)].sort_values(["label", "sample_nr"])
    anomaly_ids = anomaly_frame.sample_nr.astype(int).tolist()
    target_anomaly_ids = (
        anomaly_frame[anomaly_frame.sample_nr.ge(TARGET_FIRST_SAMPLE_NR)]
        .sample_nr.astype(int).tolist()
    )

    if len(source_ids) != 806:
        raise RuntimeError(f"Expected 806 source healthy executions; found {len(source_ids)}")
    if len(target_ids) != 614:
        raise RuntimeError(f"Expected 614 target healthy executions; found {len(target_ids)}")
    if len(anomaly_ids) != 625:
        raise RuntimeError(f"Expected 625 tightening anomalies; found {len(anomaly_ids)}")

    eval_rng = np.random.default_rng(int(eval_seed))
    perm = eval_rng.permutation(len(target_ids))
    eval_ids = [target_ids[int(i)] for i in perm[:eval_size]]
    eval_set = set(eval_ids)
    pool = [v for v in target_ids if v not in eval_set]

    seed_rng = np.random.default_rng(int(seed))
    seed_perm = seed_rng.permutation(len(pool))
    pool = [pool[int(i)] for i in seed_perm]

    return source_ids, pool, sorted(eval_ids), anomaly_ids, target_anomaly_ids


def _require_cache_coverage(batch, required_ids):
    available = set(int(v) for v in batch.episode_ids.tolist())
    missing = sorted(set(required_ids) - available)
    if missing:
        raise RuntimeError(
            "AURSAD feature cache does not cover P0.8. "
            f"Missing {len(missing)} executions, e.g. {missing[:20]}. "
            "Rebuild with experiments/build_aursad_feature_cache.py."
        )


def _summarize(results, args):
    rows = []
    for gi, ((budget, method), g) in enumerate(results.groupby(["budget", "method"], sort=True)):
        recall_mean, recall_lo, recall_hi = _bootstrap_mean(g.recall.to_numpy(), args.bootstrap_replicates, 210000 + gi)
        fpr_mean, fpr_lo, fpr_hi = _bootstrap_mean(g.false_positive_rate.to_numpy(), args.bootstrap_replicates, 220000 + gi)
        auroc_mean, auroc_lo, auroc_hi = _bootstrap_mean(g.auroc.to_numpy(), args.bootstrap_replicates, 230000 + gi)
        auprc_mean, auprc_lo, auprc_hi = _bootstrap_mean(g.auprc.to_numpy(), args.bootstrap_replicates, 240000 + gi)
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
            "target_block_recall_mean": float(g.target_block_recall.mean()),
        })
    return pd.DataFrame(rows)


def _b_star(summary, args):
    out = {}
    max_budget = int(summary.budget.max())
    for method in METHODS:
        g = summary[summary.method == method].sort_values("budget")
        ok = g[
            (g.recall_ci_lower >= float(args.recall_target))
            & (g.fpr_ci_upper <= float(args.false_alert_budget) + 1e-12)
        ]
        out[method] = int(ok.iloc[0].budget) if not ok.empty else f"Censored (>{max_budget})"
    return out


def _category_summary(category_rows, args):
    rows = []
    grouped = category_rows.groupby(["budget", "method", "category", "category_name"], sort=True)
    for gi, ((budget, method, category, category_name), g) in enumerate(grouped):
        mean, lo, hi = _bootstrap_mean(g.recall.to_numpy(), args.bootstrap_replicates, 250000 + gi)
        rows.append({
            "budget": int(budget),
            "method": str(method),
            "category": int(category),
            "category_name": str(category_name),
            "anomaly_n": int(g.anomaly_n.iloc[0]),
            "seeds": int(g.seed.nunique()),
            "recall_mean": mean,
            "recall_ci_lower": lo,
            "recall_ci_upper": hi,
        })
    return pd.DataFrame(rows)


def run(args):
    output = Path(args.output_dir).resolve()
    dataset = Path(args.dataset).resolve()
    inventory_path = Path(args.inventory).resolve()
    cache = Path(args.feature_cache).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw_path = output / "p08_seed_results.csv"
    cat_raw_path = output / "p08_category_seed_results.csv"
    if args.no_resume:
        for p in (raw_path, cat_raw_path):
            if p.exists():
                p.unlink()

    for p, label in ((dataset, "dataset"), (inventory_path, "episode inventory"), (cache, "feature cache")):
        if not p.is_file():
            raise FileNotFoundError(f"AURSAD {label} not found: {p}")

    allocations = _allocation_map()
    budgets = [int(v) for v in args.budgets]
    unknown = sorted(set(budgets) - set(allocations))
    if unknown:
        raise ValueError(f"Budgets not in frozen P0.8 set: {unknown}")

    inventory = load_episode_inventory(inventory_path)
    batch = load_feature_batch(cache)
    if batch.features.shape[1] != 288:
        raise RuntimeError(f"Expected frozen 288-feature AURSAD cache; found {batch.features.shape[1]}")

    source0, pool0, eval0, anomalies0, target_anomalies0 = _inventory_partitions(
        inventory, seed=0, eval_seed=args.evaluation_seed, eval_size=args.eval_size
    )
    _require_cache_coverage(batch, source0 + pool0 + eval0 + anomalies0)
    if max(budgets) > len(pool0):
        raise RuntimeError(f"Largest budget {max(budgets)} exceeds target commissioning pool {len(pool0)}")

    existing = pd.read_csv(raw_path) if raw_path.exists() and raw_path.stat().st_size else pd.DataFrame()
    completed = set()
    if not existing.empty:
        if set(existing.protocol_version.astype(str).unique()) != {PROTOCOL_VERSION}:
            raise RuntimeError("P0.8 checkpoint protocol mismatch")
        completed = {(int(r.budget), int(r.seed), str(r.method)) for r in existing.itertuples()}

    category_by_id = {
        int(r.sample_nr): int(r.label)
        for r in inventory[inventory.label.isin(ANOMALY_LABELS)].itertuples(index=False)
    }

    for budget in budgets:
        fit_n, cal_n = allocations[budget]
        for si, seed in enumerate(args.seeds, start=1):
            print(f"P0.8 B={budget} fit={fit_n} cal={cal_n} seed={seed} ({si}/{len(args.seeds)})", flush=True)
            source_ids, pool, eval_ids, anomaly_ids, target_anomaly_ids = _inventory_partitions(
                inventory, seed=int(seed), eval_seed=args.evaluation_seed, eval_size=args.eval_size
            )
            fit_ids = pool[:fit_n]
            cal_ids = pool[fit_n:fit_n + cal_n]

            groups = [set(source_ids), set(fit_ids), set(cal_ids), set(eval_ids), set(anomaly_ids)]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        raise RuntimeError("P0.8 data leakage detected")

            if any(v <= SOURCE_LAST_SAMPLE_NR for v in fit_ids + cal_ids + eval_ids):
                raise RuntimeError("Target healthy partition crossed into source blocks")
            if any(v >= TARGET_FIRST_SAMPLE_NR for v in source_ids):
                raise RuntimeError("Source healthy partition crossed into target blocks")

            source_raw = _rows(batch, source_ids)
            target_raw = _rows(batch, fit_ids)
            calibration_raw = _rows(batch, cal_ids)
            normal_raw = _rows(batch, eval_ids)
            anomaly_raw = _rows(batch, anomaly_ids)
            target_anomaly_set = set(target_anomaly_ids)
            target_positions = [i for i, eid in enumerate(anomaly_ids) if eid in target_anomaly_set]

            cv_seed = 8200000 + int(budget) * 1000 + int(seed)

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
                    method, source_x, target_x, n=fit_n, cv_seed=cv_seed, args=args
                )
                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)

                threshold = BaseDetector.conformal_quantile(cal_scores, alpha=args.false_alert_budget)
                normal_pred = normal_scores > threshold
                anomaly_pred = anomaly_scores > threshold
                fpr = float(np.mean(normal_pred))
                recall = float(np.mean(anomaly_pred))
                target_block_recall = float(np.mean(anomaly_pred[target_positions])) if target_positions else np.nan

                y = np.concatenate([np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)])
                scores = np.concatenate([normal_scores, anomaly_scores])
                auroc = float(roc_auc_score(y, scores))
                auprc = float(average_precision_score(y, scores))

                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "budget": budget,
                    "fit_n": fit_n,
                    "calibration_n": cal_n,
                    "seed": int(seed),
                    "method": method,
                    "false_positive_rate": fpr,
                    "recall": recall,
                    "target_block_recall": target_block_recall,
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
                    "source_n": len(source_ids),
                    "normal_eval_n": len(eval_ids),
                    "anomaly_eval_n": len(anomaly_ids),
                    "target_block_anomaly_eval_n": len(target_anomaly_ids),
                    "source_ids_sha256": _sha256_ints(source_ids),
                    "fit_ids_sha256": _sha256_ints(fit_ids),
                    "calibration_ids_sha256": _sha256_ints(cal_ids),
                    "normal_eval_ids_sha256": _sha256_ints(eval_ids),
                    "anomaly_eval_ids_sha256": _sha256_ints(anomaly_ids),
                }
                _append_row(raw_path, row)

                for category in ANOMALY_LABELS:
                    pos = [i for i, eid in enumerate(anomaly_ids) if category_by_id[eid] == category]
                    _append_row(cat_raw_path, {
                        "protocol_version": PROTOCOL_VERSION,
                        "budget": budget,
                        "seed": int(seed),
                        "method": method,
                        "category": category,
                        "category_name": CATEGORY_NAMES[category],
                        "anomaly_n": len(pos),
                        "recall": float(np.mean(anomaly_pred[pos])) if pos else np.nan,
                    })
                completed.add(key)

    results = pd.read_csv(raw_path)
    results = results[results.budget.isin(budgets)].copy()
    summary = _summarize(results, args)
    summary.to_csv(output / "p08_summary.csv", index=False)

    bstar = _b_star(summary, args)
    (output / "p08_b_star.json").write_text(json.dumps(bstar, indent=2), encoding="utf-8")

    split_audit = results.groupby(["budget", "seed"], sort=True).agg(
        source_hashes=("source_ids_sha256", "nunique"),
        fit_hashes=("fit_ids_sha256", "nunique"),
        calibration_hashes=("calibration_ids_sha256", "nunique"),
        normal_eval_hashes=("normal_eval_ids_sha256", "nunique"),
        anomaly_eval_hashes=("anomaly_eval_ids_sha256", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()
    split_audit.to_csv(output / "p08_split_audit.csv", index=False)
    if not ((split_audit.source_hashes == 1) & (split_audit.fit_hashes == 1) &
            (split_audit.calibration_hashes == 1) & (split_audit.normal_eval_hashes == 1) &
            (split_audit.anomaly_eval_hashes == 1) & (split_audit.methods == len(METHODS))).all():
        raise RuntimeError("P0.8 method-pairing audit failed")

    global_audit = {
        "source_hashes_global": int(results.source_ids_sha256.nunique()),
        "normal_eval_hashes_global": int(results.normal_eval_ids_sha256.nunique()),
        "anomaly_eval_hashes_global": int(results.anomaly_eval_ids_sha256.nunique()),
    }
    if global_audit != {"source_hashes_global": 1, "normal_eval_hashes_global": 1, "anomaly_eval_hashes_global": 1}:
        raise RuntimeError(f"P0.8 global frozen-evaluation audit failed: {global_audit}")

    safe = results[results.method == "RACECovSafeCV"].groupby("budget", sort=True).agg(
        seeds=("seed", "nunique"),
        fit_n=("fit_n", "first"),
        calibration_n=("calibration_n", "first"),
        median_lambda=("selected_lambda", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_transfer_accepted=("accepted_transfer", "mean"),
        fraction_threshold_is_calibration_max=("calibration_threshold_is_max", "mean"),
    ).reset_index()
    safe.to_csv(output / "p08_source_weight_audit.csv", index=False)

    cat_rows = pd.read_csv(cat_raw_path)
    cat_rows = cat_rows[cat_rows.budget.isin(budgets)].copy()
    _category_summary(cat_rows, args).to_csv(output / "p08_category_recall.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "inventory": str(inventory_path),
        "feature_cache": str(cache),
        "feature_count": int(batch.features.shape[1]),
        "source_definition": "normal label 0, inferred acquisition segments 0-9, sample_nr <= 2336",
        "target_definition": "normal label 0, inferred acquisition segments 10-18, sample_nr >= 2337",
        "source_healthy_n": 806,
        "target_healthy_n": 614,
        "primary_anomaly_definition": "all tightening anomaly labels 1-4",
        "primary_anomaly_n": 625,
        "supplementary_label5_used": False,
        "evaluation_seed": int(args.evaluation_seed),
        "normal_eval_size": int(args.eval_size),
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
        "allocations": [{"budget": b, "fit_n": allocations[b][0], "calibration_n": allocations[b][1]} for b in budgets],
        "seeds": [int(v) for v in args.seeds],
        "methods": list(METHODS),
        "anomaly_labels_used_for_fit_selection_or_calibration": False,
        "same_allocation_for_all_methods_within_budget": True,
        "healthy_source_target_acquisition_blocks_disjoint": True,
        "global_split_audit": global_audit,
    }
    (output / "p08_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nP0.8 summary\n", summary.to_string(index=False), flush=True)
    print("\nB*\n", json.dumps(bstar, indent=2), flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
        ap.error("P0.8 freezes target healthy holdout size at 100")
    if not (0.0 < args.false_alert_budget < 1.0):
        ap.error("false-alert-budget must be in (0,1)")
    if not (0.0 < args.recall_target <= 1.0):
        ap.error("recall-target must be in (0,1]")
    return args


if __name__ == "__main__":
    run(parse_args())
