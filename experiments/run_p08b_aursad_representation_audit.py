"""P0.8b: predeclared AURSAD representation sensitivity audit.

This diagnostic does NOT tune RACE. It holds the P0.8 acquisition-block split,
commissioning allocations, detector implementations, seeds, calibration rule,
and anomaly evaluation fixed while varying only the input sensor representation.

Representations
---------------
M  : frozen 48 measured robot signals (288 statistical features)
S  : 4 continuous screwdriver process registers (24 statistical features)
MS : measured + screwdriver signals (52 signals, 312 statistical features)

The six statistics are the frozen shared feature extractor statistics:
mean, std, median, q25, q75, total variation.

All three representations are reported. No representation is selected or hidden
based on anomaly results.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aursad_loader import (
    AURSAD_MEASURED_SIGNAL_COLUMNS,
    DEFAULT_DATA_PATH,
    DEFAULT_INVENTORY_PATH,
    load_cycles,
    load_episode_inventory,
)
from src.base_detector import BaseDetector
from src.feature_extractor import extract_feature_batch
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
from experiments.run_p08_aursad_fixed_budget_validation import (
    ANOMALY_LABELS,
    CATEGORY_NAMES,
    DEFAULT_ALLOCATIONS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_FALSE_ALERT_BUDGET,
    DEFAULT_RECALL_TARGET,
    SOURCE_LAST_SAMPLE_NR,
    TARGET_FIRST_SAMPLE_NR,
    _inventory_partitions,
)

PROTOCOL_VERSION = "p08b-aursad-representation-audit-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p08b_aursad_representation_audit"
DEFAULT_SEEDS = (0, 1)
DEFAULT_BUDGETS = (224, 249, 400)
DEFAULT_BOOTSTRAPS = 500
DEFAULT_EVAL_SIZE = 100

SCREWDRIVER_CONTINUOUS_COLUMNS = (
    "output_double_register_24",
    "output_double_register_25",
    "output_double_register_26",
    "output_double_register_27",
)

REPRESENTATIONS = {
    "M": tuple(AURSAD_MEASURED_SIGNAL_COLUMNS),
    "S": SCREWDRIVER_CONTINUOUS_COLUMNS,
    "MS": tuple(AURSAD_MEASURED_SIGNAL_COLUMNS) + SCREWDRIVER_CONTINUOUS_COLUMNS,
}


def _allocation_map() -> dict[int, tuple[int, int]]:
    return {int(b): (int(f), int(c)) for b, f, c in DEFAULT_ALLOCATIONS}


def _subset_cycles(cycles, columns):
    index = {name: i for i, name in enumerate(cycles[0].columns)}
    positions = [index[name] for name in columns]
    out = []
    for cycle in cycles:
        out.append(
            replace(
                cycle,
                values=np.asarray(cycle.values[:, positions], dtype=np.float64),
                columns=tuple(columns),
            )
        )
    return out


def _build_batches(dataset: Path, inventory: pd.DataFrame):
    primary = inventory[inventory.label.isin((0, *ANOMALY_LABELS))]
    ids = primary.sample_nr.astype(int).sort_values().tolist()
    combined = REPRESENTATIONS["MS"]
    print(
        f"Loading {len(ids)} primary AURSAD executions once with "
        f"{len(combined)} combined signals...",
        flush=True,
    )
    cycles = load_cycles(
        dataset,
        episode_ids=ids,
        inventory_path=DEFAULT_INVENTORY_PATH,
        signal_columns=combined,
    )
    batches = {}
    for name, columns in REPRESENTATIONS.items():
        print(
            f"Extracting representation {name}: {len(columns)} signals, "
            f"{len(columns) * 6} features",
            flush=True,
        )
        batches[name] = extract_feature_batch(_subset_cycles(cycles, columns))
    return batches


def _summary(raw: pd.DataFrame, bootstraps: int) -> pd.DataFrame:
    rows = []
    grouped = raw.groupby(["representation", "budget", "method"], sort=True)
    for gi, ((rep, budget, method), g) in enumerate(grouped):
        rec = _bootstrap_mean(g.recall.to_numpy(), bootstraps, 310000 + gi)
        fpr = _bootstrap_mean(g.false_positive_rate.to_numpy(), bootstraps, 320000 + gi)
        auc = _bootstrap_mean(g.auroc.to_numpy(), bootstraps, 330000 + gi)
        ap = _bootstrap_mean(g.auprc.to_numpy(), bootstraps, 340000 + gi)
        rows.append({
            "representation": rep,
            "signal_count": int(g.signal_count.iloc[0]),
            "feature_count": int(g.feature_count.iloc[0]),
            "budget": int(budget),
            "fit_n": int(g.fit_n.iloc[0]),
            "calibration_n": int(g.calibration_n.iloc[0]),
            "method": method,
            "seeds": int(g.seed.nunique()),
            "recall_mean": rec[0],
            "recall_ci_lower": rec[1],
            "recall_ci_upper": rec[2],
            "fpr_mean": fpr[0],
            "fpr_ci_lower": fpr[1],
            "fpr_ci_upper": fpr[2],
            "auroc_mean": auc[0],
            "auroc_ci_lower": auc[1],
            "auroc_ci_upper": auc[2],
            "auprc_mean": ap[0],
            "auprc_ci_lower": ap[1],
            "auprc_ci_upper": ap[2],
            "success_rate": float(g.success.mean()),
            "fraction_threshold_is_calibration_max": float(g.threshold_is_calibration_max.mean()),
            "median_source_weight": float(g.source_weight.median()),
            "fraction_transfer_accepted": float(g.accepted_transfer.mean()),
        })
    return pd.DataFrame(rows)


def _category_summary(raw: pd.DataFrame, bootstraps: int) -> pd.DataFrame:
    rows = []
    grouped = raw.groupby(
        ["representation", "budget", "method", "category", "category_name"],
        sort=True,
    )
    for gi, ((rep, budget, method, category, name), g) in enumerate(grouped):
        rec = _bootstrap_mean(g.recall.to_numpy(), bootstraps, 350000 + gi)
        rows.append({
            "representation": rep,
            "budget": int(budget),
            "method": method,
            "category": int(category),
            "category_name": name,
            "anomaly_n": int(g.anomaly_n.iloc[0]),
            "seeds": int(g.seed.nunique()),
            "recall_mean": rec[0],
            "recall_ci_lower": rec[1],
            "recall_ci_upper": rec[2],
        })
    return pd.DataFrame(rows)


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset).resolve()
    inventory_path = Path(args.inventory).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)

    allocations = _allocation_map()
    budgets = [int(v) for v in args.budgets]
    unknown = sorted(set(budgets) - set(allocations))
    if unknown:
        raise ValueError(f"Budgets not in frozen P0.8 allocation set: {unknown}")

    inventory = load_episode_inventory(inventory_path)
    batches = _build_batches(dataset, inventory)

    # Fixed partitions are identical across representations and methods.
    source0, pool0, eval0, anomalies0, _ = _inventory_partitions(
        inventory,
        seed=0,
        eval_seed=args.evaluation_seed,
        eval_size=args.eval_size,
    )
    required_ids = set(source0 + pool0 + eval0 + anomalies0)
    for rep, batch in batches.items():
        missing = required_ids - set(int(v) for v in batch.episode_ids.tolist())
        if missing:
            raise RuntimeError(f"Representation {rep} missing {len(missing)} required episodes")

    raw_rows = []
    category_rows = []
    category_by_id = {
        int(r.sample_nr): int(r.label)
        for r in inventory[inventory.label.isin(ANOMALY_LABELS)].itertuples(index=False)
    }

    for rep, batch in batches.items():
        for budget in budgets:
            fit_n, cal_n = allocations[budget]
            for si, seed in enumerate(args.seeds, 1):
                print(
                    f"P0.8b rep={rep} B={budget} fit={fit_n} cal={cal_n} "
                    f"seed={seed} ({si}/{len(args.seeds)})",
                    flush=True,
                )
                source_ids, pool, eval_ids, anomaly_ids, _ = _inventory_partitions(
                    inventory,
                    seed=int(seed),
                    eval_seed=args.evaluation_seed,
                    eval_size=args.eval_size,
                )
                fit_ids = pool[:fit_n]
                cal_ids = pool[fit_n:fit_n + cal_n]

                groups = [set(source_ids), set(fit_ids), set(cal_ids), set(eval_ids), set(anomaly_ids)]
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        if groups[i] & groups[j]:
                            raise RuntimeError("P0.8b data leakage detected")
                if any(v <= SOURCE_LAST_SAMPLE_NR for v in fit_ids + cal_ids + eval_ids):
                    raise RuntimeError("Target healthy partition crossed into source blocks")
                if any(v >= TARGET_FIRST_SAMPLE_NR for v in source_ids):
                    raise RuntimeError("Source healthy partition crossed into target blocks")

                source_raw = _rows(batch, source_ids)
                target_raw = _rows(batch, fit_ids)
                calibration_raw = _rows(batch, cal_ids)
                normal_raw = _rows(batch, eval_ids)
                anomaly_raw = _rows(batch, anomaly_ids)

                cv_seed = 8_300_000 + int(budget) * 1000 + int(seed)
                for method in METHODS:
                    scaler = _fit_method_scaler(method, source_raw, target_raw)
                    sx = scaler.transform(source_raw)
                    tx = scaler.transform(target_raw)
                    cx = scaler.transform(calibration_raw)
                    nx = scaler.transform(normal_raw)
                    ax = scaler.transform(anomaly_raw)

                    est, location, diag = _fit_estimate(
                        method,
                        sx,
                        tx,
                        n=fit_n,
                        cv_seed=cv_seed,
                        args=args,
                    )
                    cal_scores = _scores(cx, location, est.precision)
                    normal_scores = _scores(nx, location, est.precision)
                    anomaly_scores = _scores(ax, location, est.precision)
                    threshold = BaseDetector.conformal_quantile(
                        cal_scores,
                        alpha=args.false_alert_budget,
                    )
                    npred = normal_scores > threshold
                    apred = anomaly_scores > threshold
                    fpr = float(np.mean(npred))
                    recall = float(np.mean(apred))
                    y = np.concatenate([
                        np.zeros(len(normal_scores), dtype=int),
                        np.ones(len(anomaly_scores), dtype=int),
                    ])
                    score = np.concatenate([normal_scores, anomaly_scores])
                    auroc = float(roc_auc_score(y, score))
                    auprc = float(average_precision_score(y, score))

                    raw_rows.append({
                        "protocol_version": PROTOCOL_VERSION,
                        "representation": rep,
                        "signal_count": len(REPRESENTATIONS[rep]),
                        "feature_count": batch.features.shape[1],
                        "budget": budget,
                        "fit_n": fit_n,
                        "calibration_n": cal_n,
                        "seed": int(seed),
                        "method": method,
                        "false_positive_rate": fpr,
                        "recall": recall,
                        "success": bool(recall >= args.recall_target and fpr <= args.false_alert_budget),
                        "auroc": auroc,
                        "auprc": auprc,
                        "threshold": float(threshold),
                        "threshold_is_calibration_max": bool(np.isclose(threshold, np.max(cal_scores))),
                        "source_weight": float(diag["source_weight"]),
                        "selected_lambda": float(diag["selected_lambda"]) if np.isfinite(diag["selected_lambda"]) else np.nan,
                        "accepted_transfer": bool(diag["accepted_transfer"]),
                    })

                    anomaly_categories = np.asarray([category_by_id[eid] for eid in anomaly_ids], dtype=int)
                    for category in ANOMALY_LABELS:
                        mask = anomaly_categories == int(category)
                        category_rows.append({
                            "representation": rep,
                            "budget": budget,
                            "seed": int(seed),
                            "method": method,
                            "category": int(category),
                            "category_name": CATEGORY_NAMES[int(category)],
                            "anomaly_n": int(mask.sum()),
                            "recall": float(np.mean(apred[mask])) if mask.any() else np.nan,
                        })

    raw = pd.DataFrame(raw_rows)
    cat_raw = pd.DataFrame(category_rows)
    raw.to_csv(output / "p08b_seed_results.csv", index=False)
    cat_raw.to_csv(output / "p08b_category_seed_results.csv", index=False)
    summary = _summary(raw, args.bootstrap_replicates)
    cat_summary = _category_summary(cat_raw, args.bootstrap_replicates)
    summary.to_csv(output / "p08b_representation_summary.csv", index=False)
    cat_summary.to_csv(output / "p08b_category_recall.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "inventory": str(inventory_path),
        "representations": {
            key: {
                "signals": list(value),
                "signal_count": len(value),
                "feature_count": len(value) * 6,
            }
            for key, value in REPRESENTATIONS.items()
        },
        "source_definition": "normal label 0, acquisition segments 0-9 / sample_nr <= 2336",
        "target_definition": "normal label 0, acquisition segments 10-18 / sample_nr >= 2337",
        "primary_anomaly_definition": "all labels 1-4; used only for evaluation",
        "representation_selected_using_anomaly_results": False,
        "all_representations_reported": True,
        "detector_hyperparameters_changed_from_p08": False,
        "budgets": budgets,
        "seeds": [int(v) for v in args.seeds],
        "methods": list(METHODS),
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
    }
    (output / "p08b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nP0.8b representation summary\n", summary.to_string(index=False), flush=True)
    print(f"\nOutputs written to {output}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
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
    args = ap.parse_args()
    if args.eval_size != DEFAULT_EVAL_SIZE:
        ap.error("P0.8b freezes target healthy evaluation size at 100")
    return args


if __name__ == "__main__":
    run(parse_args())
