"""P1.0: cross-dataset tail-geometry comparison for frozen voraus P0.7 and AURSAD P0.9.

This diagnostic does not alter any frozen detector, representation, split, or
threshold rule. It recomputes the requested voraus P0.7 budgets using the exact
P0.7 machinery, converts scores to the same robust calibration-relative scale
used by P0.9, and then compares those results with the already-generated P0.9
AURSAD diagnostics.

Primary scientific question:
    Are the voraus and AURSAD commissioning bottlenecks mechanistically
    different when viewed through the same normalized tail-geometry quantities?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector
from src.voraus_loader import load_cycle_metadata
from experiments.run_p05_anomaly_commissioning import (
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    DEFAULT_CV_FOLDS,
    DEFAULT_RACE_LAMBDAS,
    DEFAULT_RIDGE_GAMMAS,
    _ensure_feature_cache,
    _fit_estimate,
    _fit_method_scaler,
    _rows,
    _scores,
)
from experiments.run_p07_fixed_budget_commissioning import (
    DEFAULT_ALLOCATIONS as P07_ALLOCATIONS,
    DEFAULT_EVALUATION_SEED as P07_EVALUATION_SEED,
    DEFAULT_EVAL_SIZE as P07_EVAL_SIZE,
    DEFAULT_FALSE_ALERT_BUDGET,
    _frozen_preb_partitions,
)

PROTOCOL_VERSION = "p10-cross-dataset-tail-geometry-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p10_cross_dataset_tail_geometry"
DEFAULT_P09 = PROJECT_ROOT / "outputs" / "p09_aursad_score_geometry"
DEFAULT_METHODS = ("BestTargetOnlySafeCV", "RACECov60", "RACECovSafeCV")
DEFAULT_VORAUS_BUDGETS = (175, 224, 249)
DEFAULT_SEEDS = tuple(range(20))


def _allocation_map():
    return {int(b): (int(f), int(c)) for b, f, c in P07_ALLOCATIONS}


def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-12:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
        scale = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return med, scale


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def _paired_bootstrap_delta(x: np.ndarray, y: np.ndarray, *, seed: int, reps: int = 10000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def _plot_metric(combined: pd.DataFrame, metric: str, output: Path) -> None:
    if metric not in combined.columns:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for (dataset, method), g in combined.groupby(["dataset", "method"], sort=True):
        g = g.sort_values("budget")
        ax.plot(g.budget, g[metric], marker="o", label=f"{dataset}: {method}")
    ax.set_xlabel("total commissioning budget B")
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " "))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f"p10_{metric}.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    p09_dir = Path(args.p09_dir).resolve()
    p09_seed_path = p09_dir / "p09_seed_diagnostics.csv"
    p09_summary_path = p09_dir / "p09_budget_summary.csv"
    if not p09_seed_path.is_file() or not p09_summary_path.is_file():
        raise FileNotFoundError(
            "P0.9 AURSAD diagnostics not found. Run "
            "experiments/run_p09_aursad_score_geometry_diagnostic.py first."
        )

    dataset = Path(args.dataset).resolve()
    cache = Path(args.feature_cache).resolve()
    metadata = load_cycle_metadata(dataset)
    batch = _ensure_feature_cache(dataset, cache, args.signal_set)

    allocations = _allocation_map()
    budgets = [int(v) for v in args.voraus_budgets]
    unknown = sorted(set(budgets) - set(allocations))
    if unknown:
        raise ValueError(f"voraus budgets not in frozen P0.7 allocation set: {unknown}")

    category_by_id = {int(c.episode_id): int(c.category) for c in metadata if c.anomaly}
    rows: list[dict[str, object]] = []

    for budget in budgets:
        fit_n, cal_n = allocations[budget]
        for si, seed in enumerate(args.seeds, start=1):
            print(f"P1.0 voraus B={budget} seed={seed} ({si}/{len(args.seeds)})", flush=True)
            source_cycles, pool, eval_cycles, anomaly_cycles = _frozen_preb_partitions(
                metadata,
                seed=int(seed),
                eval_seed=int(P07_EVALUATION_SEED),
                eval_size=int(P07_EVAL_SIZE),
            )
            fit_cycles = pool[:fit_n]
            cal_cycles = pool[fit_n:fit_n + cal_n]

            source_ids = [int(c.episode_id) for c in source_cycles]
            fit_ids = [int(c.episode_id) for c in fit_cycles]
            cal_ids = [int(c.episode_id) for c in cal_cycles]
            eval_ids = [int(c.episode_id) for c in eval_cycles]
            anomaly_ids = [int(c.episode_id) for c in anomaly_cycles]

            source_raw = _rows(batch, source_ids)
            target_raw = _rows(batch, fit_ids)
            calibration_raw = _rows(batch, cal_ids)
            normal_raw = _rows(batch, eval_ids)
            anomaly_raw = _rows(batch, anomaly_ids)
            cv_seed = 6_200_000 + int(budget) * 1000 + int(seed)

            for method in args.methods:
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
                    n=int(fit_n),
                    cv_seed=int(cv_seed),
                    args=args,
                )
                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)
                threshold = float(BaseDetector.conformal_quantile(cal_scores, alpha=args.false_alert_budget))

                cal_med, cal_scale = _robust_center_scale(cal_scores)
                threshold_z = (threshold - cal_med) / cal_scale
                normal_z = (normal_scores - cal_med) / cal_scale
                anomaly_z = (anomaly_scores - cal_med) / cal_scale

                row: dict[str, object] = {
                    "dataset": "voraus",
                    "protocol_version": "p07-fixed-budget-preb-v1",
                    "budget": int(budget),
                    "fit_n": int(fit_n),
                    "calibration_n": int(cal_n),
                    "seed": int(seed),
                    "method": str(method),
                    "threshold_robust_z": float(threshold_z),
                    "normal_eval_median_robust_z": float(np.median(normal_z)),
                    "normal_eval_q99_robust_z": float(np.quantile(normal_z, 0.99)),
                    "fpr": float(np.mean(normal_scores > threshold)),
                    "recall": float(np.mean(anomaly_scores > threshold)),
                    "auroc": float(roc_auc_score(
                        np.concatenate([np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]),
                        np.concatenate([normal_scores, anomaly_scores]),
                    )),
                    "anomaly_median_robust_z": float(np.median(anomaly_z)),
                    "anomaly_median_margin_z": float(np.median(anomaly_z - threshold_z)),
                }

                for cat in sorted(set(category_by_id.values())):
                    pos = [i for i, eid in enumerate(anomaly_ids) if category_by_id[eid] == cat]
                    if not pos:
                        continue
                    cs = anomaly_scores[pos]
                    cz = anomaly_z[pos]
                    row[f"cat{cat}_recall"] = float(np.mean(cs > threshold))
                    row[f"cat{cat}_median_robust_z"] = float(np.median(cz))
                    row[f"cat{cat}_median_margin_z"] = float(np.median(cz - threshold_z))
                    row[f"cat{cat}_auroc"] = float(roc_auc_score(
                        np.concatenate([np.zeros(len(normal_scores), dtype=int), np.ones(len(cs), dtype=int)]),
                        np.concatenate([normal_scores, cs]),
                    ))
                rows.append(row)

    voraus_seed = pd.DataFrame(rows)
    voraus_seed.to_csv(output / "p10_voraus_seed_geometry.csv", index=False)

    # Collapse voraus to budget/method means on the same columns used by P0.9.
    id_cols = {"dataset", "protocol_version", "budget", "fit_n", "calibration_n", "seed", "method"}
    metric_cols = [c for c in voraus_seed.columns if c not in id_cols]
    v_summary_rows = []
    for (budget, method), g in voraus_seed.groupby(["budget", "method"], sort=True):
        rr = {
            "dataset": "voraus", "budget": int(budget), "method": str(method),
            "seeds": int(g.seed.nunique()), "fit_n": int(g.fit_n.iloc[0]),
            "calibration_n": int(g.calibration_n.iloc[0]),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            rr[col] = float(vals.mean())
        v_summary_rows.append(rr)
    v_summary = pd.DataFrame(v_summary_rows)

    a_seed = pd.read_csv(p09_seed_path)
    a_seed.insert(0, "dataset", "AURSAD")
    a_summary_raw = pd.read_csv(p09_summary_path)
    a_summary = pd.DataFrame({
        "dataset": "AURSAD",
        "budget": a_summary_raw["budget"],
        "method": a_summary_raw["method"],
        "seeds": a_summary_raw["seeds"],
        "fit_n": a_summary_raw["fit_n"],
        "calibration_n": a_summary_raw["calibration_n"],
        "threshold_robust_z": a_summary_raw["threshold_robust_z_mean"],
        "normal_eval_median_robust_z": a_summary_raw["normal_eval_median_robust_z_mean"],
        "normal_eval_q99_robust_z": a_summary_raw["normal_eval_q99_robust_z_mean"],
        "fpr": a_summary_raw["fpr_mean"],
        "recall": a_summary_raw["recall_mean"],
        "auroc": a_summary_raw["auroc_mean"],
    })
    for cat in (1, 2, 3, 4):
        for stem in ("recall", "median_robust_z", "median_margin_z", "auroc"):
            src = f"cat{cat}_{stem}_mean"
            if src in a_summary_raw.columns:
                a_summary[f"cat{cat}_{stem}"] = a_summary_raw[src]

    combined = pd.concat([v_summary, a_summary], ignore_index=True, sort=False)
    combined.to_csv(output / "p10_cross_dataset_budget_summary.csv", index=False)

    # Matched-budget direct differences, AURSAD minus voraus, on shared budgets.
    shared_budgets = sorted(set(v_summary.budget) & set(a_summary.budget))
    comparison_rows = []
    metrics = [
        "threshold_robust_z", "normal_eval_median_robust_z", "normal_eval_q99_robust_z",
        "fpr", "recall", "auroc",
        "cat1_median_margin_z", "cat2_median_margin_z", "cat3_median_margin_z",
        "cat1_auroc", "cat2_auroc", "cat3_auroc",
    ]
    for budget in shared_budgets:
        for method in args.methods:
            vg = v_summary[(v_summary.budget == budget) & (v_summary.method == method)]
            ag = a_summary[(a_summary.budget == budget) & (a_summary.method == method)]
            if vg.empty or ag.empty:
                continue
            for metric in metrics:
                if metric not in vg.columns or metric not in ag.columns:
                    continue
                vv = float(vg.iloc[0][metric])
                av = float(ag.iloc[0][metric])
                if np.isfinite(vv) and np.isfinite(av):
                    comparison_rows.append({
                        "budget": int(budget), "method": str(method), "metric": metric,
                        "voraus": vv, "AURSAD": av, "AURSAD_minus_voraus": av - vv,
                    })
    pd.DataFrame(comparison_rows).to_csv(output / "p10_matched_budget_contrast.csv", index=False)

    # Within-dataset paired budget changes to separate threshold movement from ranking movement.
    change_rows = []
    for dataset_name, df in (("voraus", voraus_seed), ("AURSAD", a_seed)):
        for method in args.methods:
            mg = df[df.method == method].copy()
            if dataset_name == "voraus":
                b0, b1 = min(budgets), max(budgets)
            else:
                available = sorted(set(int(v) for v in mg.budget.unique()))
                b0, b1 = min(available), max(available)
            g0 = mg[mg.budget == b0].set_index("seed")
            g1 = mg[mg.budget == b1].set_index("seed")
            common = sorted(set(g0.index) & set(g1.index))
            if not common:
                continue
            for metric in ["threshold_robust_z", "normal_eval_median_robust_z", "recall", "auroc",
                           "cat1_median_margin_z", "cat2_median_margin_z", "cat3_median_margin_z"]:
                if metric not in g0.columns or metric not in g1.columns:
                    continue
                delta, lo, hi = _paired_bootstrap_delta(
                    g1.loc[common, metric].to_numpy(float),
                    g0.loc[common, metric].to_numpy(float),
                    seed=910000 + len(change_rows),
                )
                change_rows.append({
                    "dataset": dataset_name, "method": str(method), "metric": metric,
                    "budget_start": int(b0), "budget_end": int(b1), "paired_seeds": int(len(common)),
                    "mean_delta_end_minus_start": delta, "delta_ci_lower": lo, "delta_ci_upper": hi,
                })
    pd.DataFrame(change_rows).to_csv(output / "p10_within_dataset_budget_change.csv", index=False)

    for metric in ("threshold_robust_z", "normal_eval_q99_robust_z", "recall", "auroc"):
        _plot_metric(combined, metric, output)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "voraus_protocol": "p07-fixed-budget-preb-v1",
        "aursad_protocol": "p09-aursad-score-geometry-v1 over frozen P0.8",
        "voraus_budgets": budgets,
        "aursad_source": str(p09_dir),
        "seeds": [int(v) for v in args.seeds],
        "methods": [str(v) for v in args.methods],
        "normalization": "per-seed calibration median and IQR; diagnostic only",
        "predictions_or_thresholds_modified": False,
        "anomaly_labels_used_for_fit_selection_or_calibration": False,
        "primary_question": "Do voraus and AURSAD exhibit different commissioning bottlenecks under matched tail-geometry diagnostics?",
    }
    (output / "p10_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nP1.0 cross-dataset summary\n", combined.to_string(index=False), flush=True)
    print(f"\nWrote outputs to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--p09-dir", type=Path, default=DEFAULT_P09)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--voraus-budgets", type=int, nargs="+", default=list(DEFAULT_VORAUS_BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--signal-set", choices=("measured", "machine"), default="measured")
    ap.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
