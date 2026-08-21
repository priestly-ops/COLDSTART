"""P0.9: mechanistic score-geometry diagnostic for frozen AURSAD P0.8.

Purpose
-------
Explain *why* recall falls or remains low as the target healthy commissioning
budget grows. This is a diagnostic only: it does not change the frozen P0.8
representation, detector family, transfer hyperparameters, split logic, or
operational threshold rule.

For each requested budget/seed/method, the script recomputes the exact P0.8
model and exports episode-level scores for:
  * calibration healthy,
  * fixed target healthy evaluation,
  * each anomaly category.

It then reports quantities that distinguish two mechanisms:
  1. threshold movement: the conformal threshold moves outward relative to the
     healthy calibration distribution;
  2. anomaly contraction: anomaly scores move inward relative to healthy
     geometry as the target covariance estimate changes.

All diagnostic normalization is post-hoc and evaluation-only. It is never used
for fitting, transfer selection, calibration, or thresholding.
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

from src.aursad_loader import DEFAULT_DATA_PATH, DEFAULT_INVENTORY_PATH, load_episode_inventory
from src.base_detector import BaseDetector
from src.feature_extractor import load_feature_batch
from experiments.run_p05_anomaly_commissioning import (
    DEFAULT_CV_FOLDS,
    DEFAULT_RACE_LAMBDAS,
    DEFAULT_RIDGE_GAMMAS,
    _fit_estimate,
    _fit_method_scaler,
    _rows,
    _scores,
)
from experiments.run_p08_aursad_fixed_budget_validation import (
    ANOMALY_LABELS,
    CATEGORY_NAMES,
    DEFAULT_ALLOCATIONS,
    DEFAULT_CACHE,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_FALSE_ALERT_BUDGET,
    DEFAULT_RECALL_TARGET,
    _inventory_partitions,
    _require_cache_coverage,
)

PROTOCOL_VERSION = "p09-aursad-score-geometry-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p09_aursad_score_geometry"
DEFAULT_METHODS = ("BestTargetOnlySafeCV", "RACECov60", "RACECovSafeCV")
DEFAULT_BUDGETS = (224, 249, 400)
DEFAULT_SEEDS = tuple(range(20))
DEFAULT_EVAL_SIZE = 100


def _allocation_map() -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for budget, fit_n, cal_n in DEFAULT_ALLOCATIONS:
        out[int(budget)] = (int(fit_n), int(cal_n))
    return out


def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
        iqr = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    return med, iqr


def _quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q90": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
    }


def _append_records(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    pd.DataFrame(records).to_csv(
        path,
        mode="a",
        header=(not path.exists()) or path.stat().st_size == 0,
        index=False,
    )


def _plot_budget_panels(score_df: pd.DataFrame, output: Path, method: str) -> None:
    """Create one standalone figure per method/budget using threshold-normalized scores."""
    for budget in sorted(score_df.budget.unique()):
        g = score_df[(score_df.method == method) & (score_df.budget == budget)].copy()
        if g.empty:
            continue

        # Aggregate across seeds only for visualization. Each record is already
        # normalized by the threshold from its own seed/model.
        groups: list[tuple[str, np.ndarray]] = []
        normal = g[g.partition == "normal_eval"].score_over_threshold.to_numpy(float)
        groups.append(("healthy_eval", normal))
        for cat in ANOMALY_LABELS:
            vals = g[(g.partition == "anomaly") & (g.category == cat)].score_over_threshold.to_numpy(float)
            if vals.size:
                groups.append((CATEGORY_NAMES[int(cat)], vals))

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, vals in groups:
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, label=label)
        ax.axvline(1.0, linestyle="--", linewidth=1.5, label="threshold")
        ax.set_xlabel("score / conformal threshold")
        ax.set_ylabel("empirical CDF")
        ax.set_title(f"AURSAD score geometry: {method}, B={int(budget)}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        safe_method = method.lower().replace("/", "_").replace(" ", "_")
        fig.savefig(output / f"p09_ecdf_{safe_method}_B{int(budget)}.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "p09_episode_scores.csv"
    seed_path = output / "p09_seed_diagnostics.csv"

    if args.no_resume:
        for p in (score_path, seed_path):
            if p.exists():
                p.unlink()

    dataset = Path(args.dataset).resolve()
    inventory_path = Path(args.inventory).resolve()
    cache = Path(args.feature_cache).resolve()
    for p, label in ((dataset, "dataset"), (inventory_path, "inventory"), (cache, "feature cache")):
        if not p.is_file():
            raise FileNotFoundError(f"AURSAD {label} not found: {p}")

    allocations = _allocation_map()
    budgets = [int(v) for v in args.budgets]
    unknown = sorted(set(budgets) - set(allocations))
    if unknown:
        raise ValueError(f"Budgets not in frozen P0.8 allocation set: {unknown}")

    inventory = load_episode_inventory(inventory_path)
    batch = load_feature_batch(cache)
    if batch.features.shape[1] != 288:
        raise RuntimeError(
            f"P0.9 must use frozen 288-feature representation; found {batch.features.shape[1]}"
        )

    source0, pool0, eval0, anomalies0, _ = _inventory_partitions(
        inventory,
        seed=0,
        eval_seed=int(args.evaluation_seed),
        eval_size=int(args.eval_size),
    )
    _require_cache_coverage(batch, source0 + pool0 + eval0 + anomalies0)

    category_by_id = {
        int(r.sample_nr): int(r.label)
        for r in inventory[inventory.label.isin(ANOMALY_LABELS)].itertuples(index=False)
    }

    completed: set[tuple[int, int, str]] = set()
    if seed_path.exists() and seed_path.stat().st_size:
        prev = pd.read_csv(seed_path)
        if not prev.empty:
            if set(prev.protocol_version.astype(str).unique()) != {PROTOCOL_VERSION}:
                raise RuntimeError("P0.9 checkpoint protocol mismatch")
            completed = {
                (int(r.budget), int(r.seed), str(r.method)) for r in prev.itertuples()
            }

    for budget in budgets:
        fit_n, cal_n = allocations[budget]
        for si, seed in enumerate(args.seeds, start=1):
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

            cv_seed = 8_200_000 + int(budget) * 1000 + int(seed)
            print(
                f"P0.9 B={budget} fit={fit_n} cal={cal_n} seed={seed} "
                f"({si}/{len(args.seeds)})",
                flush=True,
            )

            for method in args.methods:
                key = (int(budget), int(seed), str(method))
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
                    cv_seed=int(cv_seed),
                    args=args,
                )

                cal_scores = _scores(calibration_x, location, est.precision)
                normal_scores = _scores(normal_x, location, est.precision)
                anomaly_scores = _scores(anomaly_x, location, est.precision)
                threshold = float(
                    BaseDetector.conformal_quantile(
                        cal_scores,
                        alpha=float(args.false_alert_budget),
                    )
                )

                cal_med, cal_scale = _robust_center_scale(cal_scores)
                threshold_z = (threshold - cal_med) / cal_scale
                normal_z = (normal_scores - cal_med) / cal_scale
                anomaly_z = (anomaly_scores - cal_med) / cal_scale

                normal_pred = normal_scores > threshold
                anomaly_pred = anomaly_scores > threshold
                overall_recall = float(np.mean(anomaly_pred))
                fpr = float(np.mean(normal_pred))

                y = np.concatenate([
                    np.zeros(len(normal_scores), dtype=int),
                    np.ones(len(anomaly_scores), dtype=int),
                ])
                overall_auc = float(roc_auc_score(y, np.concatenate([normal_scores, anomaly_scores])))

                seed_row: dict[str, object] = {
                    "protocol_version": PROTOCOL_VERSION,
                    "budget": int(budget),
                    "fit_n": int(fit_n),
                    "calibration_n": int(cal_n),
                    "seed": int(seed),
                    "method": str(method),
                    "threshold": threshold,
                    "calibration_median": cal_med,
                    "calibration_iqr_scale": cal_scale,
                    "threshold_robust_z": float(threshold_z),
                    "normal_eval_median_robust_z": float(np.median(normal_z)),
                    "normal_eval_q99_robust_z": float(np.quantile(normal_z, 0.99)),
                    "fpr": fpr,
                    "recall": overall_recall,
                    "auroc": overall_auc,
                    "source_weight": float(diag["source_weight"]),
                    "selected_lambda": (
                        float(diag["selected_lambda"])
                        if np.isfinite(diag["selected_lambda"])
                        else np.nan
                    ),
                    "accepted_transfer": bool(diag["accepted_transfer"]),
                }

                episode_rows: list[dict[str, object]] = []
                for eid, score, z in zip(cal_ids, cal_scores, (cal_scores - cal_med) / cal_scale):
                    episode_rows.append({
                        "protocol_version": PROTOCOL_VERSION,
                        "budget": int(budget), "seed": int(seed), "method": str(method),
                        "partition": "calibration", "episode_id": int(eid), "category": 0,
                        "category_name": "healthy_calibration", "score": float(score),
                        "score_robust_z": float(z), "score_minus_threshold": float(score - threshold),
                        "score_over_threshold": float(score / threshold) if threshold != 0 else np.nan,
                        "threshold": threshold,
                    })
                for eid, score, z in zip(eval_ids, normal_scores, normal_z):
                    episode_rows.append({
                        "protocol_version": PROTOCOL_VERSION,
                        "budget": int(budget), "seed": int(seed), "method": str(method),
                        "partition": "normal_eval", "episode_id": int(eid), "category": 0,
                        "category_name": "healthy_eval", "score": float(score),
                        "score_robust_z": float(z), "score_minus_threshold": float(score - threshold),
                        "score_over_threshold": float(score / threshold) if threshold != 0 else np.nan,
                        "threshold": threshold,
                    })

                for cat in ANOMALY_LABELS:
                    positions = [i for i, eid in enumerate(anomaly_ids) if category_by_id[int(eid)] == int(cat)]
                    cat_scores = anomaly_scores[positions]
                    cat_z = anomaly_z[positions]
                    if len(cat_scores):
                        cat_pred = cat_scores > threshold
                        cat_auc = float(
                            roc_auc_score(
                                np.concatenate([
                                    np.zeros(len(normal_scores), dtype=int),
                                    np.ones(len(cat_scores), dtype=int),
                                ]),
                                np.concatenate([normal_scores, cat_scores]),
                            )
                        )
                        seed_row[f"cat{cat}_recall"] = float(np.mean(cat_pred))
                        seed_row[f"cat{cat}_auroc"] = cat_auc
                        seed_row[f"cat{cat}_median_robust_z"] = float(np.median(cat_z))
                        seed_row[f"cat{cat}_median_margin_z"] = float(np.median(cat_z - threshold_z))
                        seed_row[f"cat{cat}_q90_margin_z"] = float(np.quantile(cat_z - threshold_z, 0.90))

                        for pos, score, z in zip(positions, cat_scores, cat_z):
                            eid = int(anomaly_ids[pos])
                            episode_rows.append({
                                "protocol_version": PROTOCOL_VERSION,
                                "budget": int(budget), "seed": int(seed), "method": str(method),
                                "partition": "anomaly", "episode_id": eid, "category": int(cat),
                                "category_name": CATEGORY_NAMES[int(cat)], "score": float(score),
                                "score_robust_z": float(z), "score_minus_threshold": float(score - threshold),
                                "score_over_threshold": float(score / threshold) if threshold != 0 else np.nan,
                                "threshold": threshold,
                            })

                _append_records(score_path, episode_rows)
                _append_records(seed_path, [seed_row])
                completed.add(key)

    scores = pd.read_csv(score_path)
    seeds = pd.read_csv(seed_path)

    # Distribution summaries retain seed-level pairing.
    dist_rows: list[dict[str, object]] = []
    for keys, g in scores.groupby(
        ["budget", "seed", "method", "partition", "category", "category_name"],
        sort=True,
    ):
        budget, seed, method, partition, category, category_name = keys
        row = {
            "budget": int(budget), "seed": int(seed), "method": str(method),
            "partition": str(partition), "category": int(category),
            "category_name": str(category_name), "n": int(len(g)),
            **{f"score_{k}": v for k, v in _quantiles(g.score.to_numpy(float)).items()},
            **{f"z_{k}": v for k, v in _quantiles(g.score_robust_z.to_numpy(float)).items()},
            "fraction_above_threshold": float(np.mean(g.score_minus_threshold.to_numpy(float) > 0)),
        }
        dist_rows.append(row)
    pd.DataFrame(dist_rows).to_csv(output / "p09_distribution_summary.csv", index=False)

    # Across-seed summary of the mechanistic quantities.
    numeric_cols = [
        c for c in seeds.columns
        if c not in {"protocol_version", "method"}
        and pd.api.types.is_numeric_dtype(seeds[c])
    ]
    agg_cols = [c for c in numeric_cols if c not in {"budget", "fit_n", "calibration_n", "seed"}]
    summary_rows: list[dict[str, object]] = []
    for (budget, method), g in seeds.groupby(["budget", "method"], sort=True):
        row: dict[str, object] = {
            "budget": int(budget),
            "method": str(method),
            "seeds": int(g.seed.nunique()),
            "fit_n": int(g.fit_n.iloc[0]),
            "calibration_n": int(g.calibration_n.iloc[0]),
        }
        for col in agg_cols:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_median"] = float(np.median(vals))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "p09_budget_summary.csv", index=False)

    # Explicit B=224 -> B=400 change table to answer the scientific question.
    change_rows: list[dict[str, object]] = []
    for method in args.methods:
        g224 = seeds[(seeds.method == method) & (seeds.budget == 224)].set_index("seed")
        g400 = seeds[(seeds.method == method) & (seeds.budget == 400)].set_index("seed")
        common = sorted(set(g224.index) & set(g400.index))
        if not common:
            continue
        cols = [
            "threshold_robust_z", "normal_eval_median_robust_z", "recall", "auroc",
            "cat1_median_robust_z", "cat2_median_robust_z", "cat3_median_robust_z",
            "cat1_median_margin_z", "cat2_median_margin_z", "cat3_median_margin_z",
            "cat1_auroc", "cat2_auroc", "cat3_auroc",
        ]
        for col in cols:
            if col not in g224.columns or col not in g400.columns:
                continue
            delta = g400.loc[common, col].to_numpy(float) - g224.loc[common, col].to_numpy(float)
            change_rows.append({
                "method": str(method),
                "metric": col,
                "paired_seeds": int(len(common)),
                "mean_delta_B400_minus_B224": float(np.mean(delta)),
                "median_delta_B400_minus_B224": float(np.median(delta)),
                "fraction_delta_negative": float(np.mean(delta < 0)),
            })
    pd.DataFrame(change_rows).to_csv(output / "p09_budget_change_224_to_400.csv", index=False)

    for method in args.methods:
        _plot_budget_panels(scores, output, method)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "inventory": str(inventory_path),
        "feature_cache": str(cache),
        "feature_count": int(batch.features.shape[1]),
        "representation": "frozen P0.8 measured 48-signal / 288-feature representation",
        "budgets": budgets,
        "seeds": [int(v) for v in args.seeds],
        "methods": [str(v) for v in args.methods],
        "evaluation_seed": int(args.evaluation_seed),
        "normal_eval_size": int(args.eval_size),
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
        "anomaly_labels_used_for_fit_selection_or_calibration": False,
        "diagnostic_normalization_used_for_model_or_threshold": False,
        "scientific_question": (
            "Does recall decline because the conformal threshold moves outward, "
            "because anomaly scores contract toward healthy geometry, or both?"
        ),
    }
    (output / "p09_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nP0.9 budget summary\n", summary.to_string(index=False), flush=True)
    print(f"\nWrote diagnostics to {output}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    ap.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    if args.eval_size != DEFAULT_EVAL_SIZE:
        ap.error("P0.9 must preserve the frozen P0.8 100-cycle normal holdout")
    return args


if __name__ == "__main__":
    run(parse_args())
