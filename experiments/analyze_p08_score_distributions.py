"""Post-freeze score-distribution diagnostic for P0.8 AURSAD.

Purpose
-------
Explain *why* recall changes with commissioning budget without changing the
frozen P0.8 detector, split, calibration rule, or primary endpoint.

This diagnostic reuses the exact P0.8 healthy partitions and the frozen P0.5
fitting/scoring implementation.  Target anomaly labels are used only after
scores are produced, for evaluation and per-fault stratification.

Primary questions
-----------------
1. Does healthy-score geometry move with B?
2. Do anomaly scores move inward relative to the calibrated threshold?
3. Is the effect concentrated in particular fault classes?
4. Is the failure threshold/calibration-limited or ranking/representation-limited?

Outputs
-------
- p08_score_rows.csv
    One row per scored cycle for calibration, healthy evaluation, and anomaly
    evaluation, including score/threshold ratio.
- p08_score_distribution_summary.csv
    Budget/method/population quantiles and threshold exceedance rates.
- p08_fault_separation_summary.csv
    Per-fault recall, AUROC vs frozen healthy evaluation, and normalized score
    separation.
- p08_budget_shift_summary.csv
    Changes relative to the smallest analyzed budget.
- p08_score_replay_audit.csv
    Recomputed threshold/FPR/recall/AUROC compared with frozen P0.8 outputs.
- figures/*.png
    ECDF plots of score/threshold ratios for reviewer-facing diagnostics.

This is an explanatory analysis only.  It must not be used to retune RACE or
change the frozen primary endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
    _fit_estimate,
    _fit_method_scaler,
    _rows,
    _scores,
)
from experiments.run_p08_aursad_fixed_budget_validation import (
    PROTOCOL_VERSION as P08_PROTOCOL_VERSION,
    DEFAULT_CACHE,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVAL_SIZE,
    DEFAULT_FALSE_ALERT_BUDGET,
    DEFAULT_ALLOCATIONS,
    ANOMALY_LABELS,
    CATEGORY_NAMES,
    _allocation_map,
    _inventory_partitions,
    _require_cache_coverage,
)

DIAGNOSTIC_VERSION = "p08-score-distribution-audit-v1"
DEFAULT_P08_OUTPUT = PROJECT_ROOT / "outputs" / "p08_aursad_fixed_budget"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p08_score_distribution_audit"
DEFAULT_BUDGETS = (224, 249, 400)
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_METHODS = ("BestTargetOnlySafeCV", "RACECov60", "RACECovSafeCV")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / float(len(x))
    return x, y


def _quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "iqr": float(q[3] - q[1]),
    }


def _append_score_rows(
    rows: list[dict[str, object]],
    *,
    budget: int,
    seed: int,
    method: str,
    threshold: float,
    population: str,
    ids: list[int],
    scores: np.ndarray,
    category_by_id: dict[int, int] | None = None,
) -> None:
    for episode_id, score in zip(ids, np.asarray(scores, dtype=np.float64), strict=True):
        category = None
        category_name = None
        if category_by_id is not None and int(episode_id) in category_by_id:
            category = int(category_by_id[int(episode_id)])
            category_name = CATEGORY_NAMES.get(category, f"label_{category}")
        rows.append({
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "p08_protocol_version": P08_PROTOCOL_VERSION,
            "budget": int(budget),
            "seed": int(seed),
            "method": str(method),
            "population": str(population),
            "episode_id": int(episode_id),
            "category": category,
            "category_name": category_name,
            "score": float(score),
            "threshold": float(threshold),
            "score_over_threshold": float(score / threshold) if threshold > 0 else np.nan,
            "above_threshold": bool(score > threshold),
        })


def _distribution_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for keys, g in score_df.groupby(["budget", "method", "population"], sort=True):
        budget, method, population = keys
        raw = _quantiles(g.score.to_numpy())
        norm = _quantiles(g.score_over_threshold.to_numpy())
        out.append({
            "budget": int(budget),
            "method": str(method),
            "population": str(population),
            "rows": int(len(g)),
            "seeds": int(g.seed.nunique()),
            **{f"score_{k}": v for k, v in raw.items()},
            **{f"normalized_{k}": v for k, v in norm.items()},
            "fraction_above_threshold": float(g.above_threshold.mean()),
        })
    return pd.DataFrame(out)


def _fault_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    healthy = score_df[score_df.population == "healthy_eval"]
    anomaly = score_df[score_df.population == "anomaly"]
    rows: list[dict[str, object]] = []
    for (budget, method, seed, category, category_name), ag in anomaly.groupby(
        ["budget", "method", "seed", "category", "category_name"], sort=True
    ):
        hg = healthy[
            (healthy.budget == budget)
            & (healthy.method == method)
            & (healthy.seed == seed)
        ]
        if hg.empty:
            raise RuntimeError("Missing paired healthy scores")
        y = np.concatenate([
            np.zeros(len(hg), dtype=int),
            np.ones(len(ag), dtype=int),
        ])
        s = np.concatenate([hg.score.to_numpy(), ag.score.to_numpy()])
        healthy_iqr = float(np.subtract(*np.quantile(hg.score.to_numpy(), [0.75, 0.25])))
        median_gap = float(np.median(ag.score) - np.median(hg.score))
        rows.append({
            "budget": int(budget),
            "method": str(method),
            "seed": int(seed),
            "category": int(category),
            "category_name": str(category_name),
            "anomaly_n": int(len(ag)),
            "healthy_n": int(len(hg)),
            "recall": float(ag.above_threshold.mean()),
            "auroc_vs_healthy": float(roc_auc_score(y, s)),
            "median_anomaly_score_over_threshold": float(np.median(ag.score_over_threshold)),
            "median_healthy_score_over_threshold": float(np.median(hg.score_over_threshold)),
            "median_raw_gap": median_gap,
            "median_gap_over_healthy_iqr": (
                float(median_gap / healthy_iqr) if healthy_iqr > 0 else np.nan
            ),
        })
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby(
        ["budget", "method", "category", "category_name"], sort=True
    ).agg(
        seeds=("seed", "nunique"),
        anomaly_n=("anomaly_n", "first"),
        recall_mean=("recall", "mean"),
        recall_sd=("recall", "std"),
        auroc_mean=("auroc_vs_healthy", "mean"),
        auroc_sd=("auroc_vs_healthy", "std"),
        anomaly_norm_median=("median_anomaly_score_over_threshold", "median"),
        healthy_norm_median=("median_healthy_score_over_threshold", "median"),
        median_gap_over_healthy_iqr=("median_gap_over_healthy_iqr", "median"),
    ).reset_index()
    return summary


def _budget_shift_summary(distribution: pd.DataFrame, base_budget: int) -> pd.DataFrame:
    core = distribution[distribution.population.isin(["healthy_eval", "anomaly"])].copy()
    rows: list[dict[str, object]] = []
    for (method, population), g in core.groupby(["method", "population"], sort=True):
        base = g[g.budget == base_budget]
        if base.empty:
            continue
        base_raw = float(base.score_median.iloc[0])
        base_norm = float(base.normalized_median.iloc[0])
        for r in g.sort_values("budget").itertuples(index=False):
            rows.append({
                "method": str(method),
                "population": str(population),
                "base_budget": int(base_budget),
                "budget": int(r.budget),
                "median_raw_score": float(r.score_median),
                "median_normalized_score": float(r.normalized_median),
                "delta_raw_median_vs_base": float(r.score_median - base_raw),
                "delta_normalized_median_vs_base": float(r.normalized_median - base_norm),
                "relative_raw_median_vs_base": (
                    float(r.score_median / base_raw - 1.0) if base_raw != 0 else np.nan
                ),
                "relative_normalized_median_vs_base": (
                    float(r.normalized_median / base_norm - 1.0) if base_norm != 0 else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _make_figures(score_df: pd.DataFrame, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figures", flush=True)
        return

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    pooled = score_df.groupby(
        ["budget", "method", "population", "episode_id"], sort=False, as_index=False
    ).agg(score_over_threshold=("score_over_threshold", "median"))

    for method in sorted(pooled.method.unique()):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        sub = pooled[pooled.method == method]
        for budget in sorted(sub.budget.unique()):
            for population in ("healthy_eval", "anomaly"):
                vals = sub[(sub.budget == budget) & (sub.population == population)].score_over_threshold.to_numpy()
                x, y = _ecdf(vals)
                ax.plot(x, y, label=f"B={budget} {population}")
        ax.axvline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Anomaly score / conformal threshold")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"AURSAD P0.8 score separation: {method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure_dir / f"p08_ecdf_{method}.png", dpi=180)
        plt.close(fig)

    for method in sorted(pooled.method.unique()):
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        sub = score_df[(score_df.method == method) & (score_df.population == "anomaly")].copy()
        grouped = sub.groupby(["budget", "category_name"], sort=True).score_over_threshold.median().reset_index()
        for category_name, g in grouped.groupby("category_name", sort=True):
            ax.plot(g.budget.to_numpy(), g.score_over_threshold.to_numpy(), marker="o", label=category_name)
        ax.axhline(1.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Total commissioning budget B")
        ax.set_ylabel("Median anomaly score / threshold")
        ax.set_title(f"Per-fault normalized score trajectory: {method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(figure_dir / f"p08_fault_trajectory_{method}.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory = load_episode_inventory(Path(args.inventory))
    batch = load_feature_batch(Path(args.feature_cache))
    if batch.features.shape[1] != 288:
        raise RuntimeError(f"Expected frozen 288-feature AURSAD cache; found {batch.features.shape[1]}")

    allocations = _allocation_map()
    budgets = [int(v) for v in args.budgets]
    unknown_budgets = sorted(set(budgets) - set(allocations))
    if unknown_budgets:
        raise ValueError(f"Budgets not in frozen P0.8 allocation set: {unknown_budgets}")

    methods = [str(v) for v in args.methods]
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown frozen P0.8 methods: {unknown_methods}")

    source0, pool0, eval0, anomalies0, _ = _inventory_partitions(
        inventory,
        seed=0,
        eval_seed=int(args.evaluation_seed),
        eval_size=int(args.eval_size),
    )
    _require_cache_coverage(batch, source0 + pool0 + eval0 + anomalies0)

    inv = inventory.copy()
    inv["sample_nr"] = pd.to_numeric(inv["sample_nr"], errors="raise").astype(np.int64)
    inv["label"] = pd.to_numeric(inv["label"], errors="raise").astype(np.int64)
    category_by_id = {
        int(r.sample_nr): int(r.label)
        for r in inv[inv.label.isin(ANOMALY_LABELS)].itertuples(index=False)
    }

    frozen_path = Path(args.p08_output_dir) / "p08_seed_results.csv"
    frozen = pd.read_csv(frozen_path) if frozen_path.is_file() else pd.DataFrame()
    if not frozen.empty and set(frozen.protocol_version.astype(str).unique()) != {P08_PROTOCOL_VERSION}:
        raise RuntimeError("Frozen P0.8 result protocol mismatch")

    score_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []

    for budget in budgets:
        fit_n, cal_n = allocations[budget]
        for si, seed in enumerate(args.seeds, start=1):
            print(f"P0.8 score audit B={budget} seed={seed} ({si}/{len(args.seeds)})", flush=True)
            source_ids, pool, eval_ids, anomaly_ids, _ = _inventory_partitions(
                inventory,
                seed=int(seed),
                eval_seed=int(args.evaluation_seed),
                eval_size=int(args.eval_size),
            )
            fit_ids = pool[:fit_n]
            cal_ids = pool[fit_n:fit_n + cal_n]

            source_raw = _rows(batch, source_ids)
            target_raw = _rows(batch, fit_ids)
            calibration_raw = _rows(batch, cal_ids)
            normal_raw = _rows(batch, eval_ids)
            anomaly_raw = _rows(batch, anomaly_ids)
            cv_seed = 8200000 + int(budget) * 1000 + int(seed)

            for method in methods:
                scaler = _fit_method_scaler(method, source_raw, target_raw)
                source_x = scaler.transform(source_raw)
                target_x = scaler.transform(target_raw)
                calibration_x = scaler.transform(calibration_raw)
                normal_x = scaler.transform(normal_raw)
                anomaly_x = scaler.transform(anomaly_raw)

                est, location, _ = _fit_estimate(
                    method,
                    source_x,
                    target_x,
                    n=fit_n,
                    cv_seed=cv_seed,
                    args=args,
                )
                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)
                threshold = BaseDetector.conformal_quantile(
                    cal_scores,
                    alpha=float(args.false_alert_budget),
                )

                fpr = float(np.mean(normal_scores > threshold))
                recall = float(np.mean(anomaly_scores > threshold))
                y = np.concatenate([
                    np.zeros(len(normal_scores), dtype=int),
                    np.ones(len(anomaly_scores), dtype=int),
                ])
                s = np.concatenate([normal_scores, anomaly_scores])
                auroc = float(roc_auc_score(y, s))

                replay = {
                    "budget": int(budget),
                    "seed": int(seed),
                    "method": str(method),
                    "replay_threshold": float(threshold),
                    "replay_fpr": fpr,
                    "replay_recall": recall,
                    "replay_auroc": auroc,
                }
                if not frozen.empty:
                    fr = frozen[
                        (frozen.budget == budget)
                        & (frozen.seed == int(seed))
                        & (frozen.method == method)
                    ]
                    if len(fr) != 1:
                        raise RuntimeError(
                            f"Expected one frozen P0.8 row for B={budget}, seed={seed}, method={method}; found {len(fr)}"
                        )
                    r = fr.iloc[0]
                    replay.update({
                        "frozen_threshold": float(r.threshold),
                        "frozen_fpr": float(r.false_positive_rate),
                        "frozen_recall": float(r.recall),
                        "frozen_auroc": float(r.auroc),
                        "abs_threshold_error": abs(float(threshold) - float(r.threshold)),
                        "abs_fpr_error": abs(fpr - float(r.false_positive_rate)),
                        "abs_recall_error": abs(recall - float(r.recall)),
                        "abs_auroc_error": abs(auroc - float(r.auroc)),
                    })
                replay_rows.append(replay)

                _append_score_rows(
                    score_rows,
                    budget=budget,
                    seed=int(seed),
                    method=method,
                    threshold=threshold,
                    population="calibration",
                    ids=cal_ids,
                    scores=cal_scores,
                )
                _append_score_rows(
                    score_rows,
                    budget=budget,
                    seed=int(seed),
                    method=method,
                    threshold=threshold,
                    population="healthy_eval",
                    ids=eval_ids,
                    scores=normal_scores,
                )
                _append_score_rows(
                    score_rows,
                    budget=budget,
                    seed=int(seed),
                    method=method,
                    threshold=threshold,
                    population="anomaly",
                    ids=anomaly_ids,
                    scores=anomaly_scores,
                    category_by_id=category_by_id,
                )

    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_csv(output / "p08_score_replay_audit.csv", index=False)
    error_cols = [c for c in replay_df.columns if c.startswith("abs_")]
    if error_cols:
        max_error = float(replay_df[error_cols].to_numpy(dtype=float).max())
        if max_error > float(args.replay_tolerance):
            raise RuntimeError(
                f"Score diagnostic does not replay frozen P0.8 exactly enough: max abs error={max_error:.3e}"
            )

    score_df = pd.DataFrame(score_rows)
    score_df.to_csv(output / "p08_score_rows.csv", index=False)

    distribution = _distribution_summary(score_df)
    distribution.to_csv(output / "p08_score_distribution_summary.csv", index=False)

    fault = _fault_summary(score_df)
    fault.to_csv(output / "p08_fault_separation_summary.csv", index=False)

    shift = _budget_shift_summary(distribution, base_budget=min(budgets))
    shift.to_csv(output / "p08_budget_shift_summary.csv", index=False)

    _make_figures(score_df, output)

    manifest = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "p08_protocol_version": P08_PROTOCOL_VERSION,
        "budgets": budgets,
        "seeds": [int(v) for v in args.seeds],
        "methods": methods,
        "false_alert_budget": float(args.false_alert_budget),
        "evaluation_seed": int(args.evaluation_seed),
        "evaluation_size": int(args.eval_size),
        "uses_frozen_p08_partitions": True,
        "uses_frozen_p05_fit_and_score_code": True,
        "anomaly_labels_used_for_model_selection": False,
        "anomaly_labels_used_for_posthoc_fault_stratification_only": True,
        "primary_endpoint_changed": False,
        "retuning_allowed": False,
        "interpretation": (
            "If anomaly score/threshold ratios and per-fault AUROC fall with B while healthy "
            "FPR remains controlled and thresholds are non-max, evidence supports a ranking/" 
            "representation-limited mechanism rather than a conformal-granularity mechanism."
        ),
    }
    (output / "p08_score_distribution_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nDistribution summary\n", distribution.to_string(index=False), flush=True)
    print("\nPer-fault separation\n", fault.to_string(index=False), flush=True)
    print("\nBudget shifts\n", shift.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--p08-output-dir", type=Path, default=DEFAULT_P08_OUTPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--methods", type=str, nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    ap.add_argument("--replay-tolerance", type=float, default=1e-10)
    args = ap.parse_args()
    if args.eval_size != DEFAULT_EVAL_SIZE:
        ap.error(f"P0.8 freezes eval size at {DEFAULT_EVAL_SIZE}")
    if not np.isclose(args.false_alert_budget, DEFAULT_FALSE_ALERT_BUDGET):
        ap.error("This mechanism audit must replay the frozen P0.8 alpha=0.01 endpoint")
    return args


if __name__ == "__main__":
    run(parse_args())
