"""P0.11: strong baseline completion under the frozen P0.7 voraus protocol.

This runner adds CPU-compatible baselines without changing any P0.7 split,
commissioning allocation, anomaly endpoint, or deployment criterion.

Methods
-------
IsolationForest      target-only statistical-feature Isolation Forest
FeatureConformalKNN  target-only Euclidean k-NN on frozen cycle features
RawConformalKNN      target-only raw-cycle Euclidean k-NN after linear resampling
PAKCT                target-only FastDTW phase-aligned raw-cycle k-NN

M2N2 is deliberately not approximated here. Its official AAAI-2024 method is a
trainable time-series test-time-adaptation pipeline, not a drop-in sklearn
cycle scorer. A companion exporter is provided separately so the official
implementation can be run on exactly the same frozen episode identities.

Primary protocol is identical to P0.7:
  B=175: fit=25, cal=150
  B=200: fit=25, cal=175
  B=224: fit=25, cal=199
  B=225: fit=25, cal=200
  B=249: fit=50, cal=199

Run a quick smoke test first:
  .venv/bin/python experiments/run_p011_strong_baselines_voraus.py \
      --budgets 175 --seeds 0 --methods IsolationForest FeatureConformalKNN

Then the feature baselines:
  .venv/bin/python experiments/run_p011_strong_baselines_voraus.py \
      --methods IsolationForest FeatureConformalKNN

RawConformalKNN and PAKCT are substantially more expensive; run them separately
with checkpoint/resume enabled by default.
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
from src.strong_baselines import (
    ConformalKNNBaseline,
    IsolationForestBaseline,
    RawCycleKNNBaseline,
)
from src.voraus_loader import load_cycle_metadata, load_cycles
from experiments.run_p05_anomaly_commissioning import (
    DEFAULT_CACHE,
    DEFAULT_DATASET,
    _bootstrap_mean,
    _ensure_feature_cache,
    _rows,
)
from experiments.run_p07_fixed_budget_commissioning import (
    DEFAULT_ALLOCATIONS,
    DEFAULT_BOOTSTRAPS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVAL_SIZE,
    DEFAULT_FALSE_ALERT_BUDGET,
    DEFAULT_RECALL_TARGET,
    DEFAULT_SEEDS,
    _frozen_preb_partitions,
)

PROTOCOL_VERSION = "p011-voraus-strong-baselines-frozen-p07-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p011_strong_baselines_voraus"
ALL_METHODS = (
    "IsolationForest",
    "FeatureConformalKNN",
    "RawConformalKNN",
    "PAKCT",
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
            raise RuntimeError("invalid frozen P0.7 allocation")
        out[int(budget)] = (int(fit_n), int(cal_n))
    return out


def _fit_feature_method(method: str, fit_x: np.ndarray, seed: int):
    if method == "IsolationForest":
        return IsolationForestBaseline(random_state=42 + int(seed)).fit(fit_x)
    if method == "FeatureConformalKNN":
        return ConformalKNNBaseline(k=10).fit(fit_x)
    raise ValueError(method)


def _fit_raw_method(method: str, fit_cycles: list[np.ndarray]):
    if method == "RawConformalKNN":
        return RawCycleKNNBaseline(k=10, phase_align=False).fit(fit_cycles)
    if method == "PAKCT":
        return RawCycleKNNBaseline(k=10, phase_align=True).fit(fit_cycles)
    raise ValueError(method)


def _summarize(results: pd.DataFrame, bootstrap_replicates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gi, ((budget, method), g) in enumerate(results.groupby(["budget", "method"], sort=True)):
        recall_mean, recall_lo, recall_hi = _bootstrap_mean(
            g.recall.to_numpy(float), bootstrap_replicates, 410_000 + gi
        )
        fpr_mean, fpr_lo, fpr_hi = _bootstrap_mean(
            g.false_positive_rate.to_numpy(float), bootstrap_replicates, 420_000 + gi
        )
        auroc_mean, auroc_lo, auroc_hi = _bootstrap_mean(
            g.auroc.to_numpy(float), bootstrap_replicates, 430_000 + gi
        )
        auprc_mean, auprc_lo, auprc_hi = _bootstrap_mean(
            g.auprc.to_numpy(float), bootstrap_replicates, 440_000 + gi
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
        })
    return pd.DataFrame(rows)


def _b_star(summary: pd.DataFrame, recall_target: float, fpr_target: float) -> dict[str, object]:
    out: dict[str, object] = {}
    max_budget = int(summary.budget.max())
    for method in sorted(summary.method.unique()):
        g = summary[summary.method.eq(method)].sort_values("budget")
        ok = g[(g.recall_ci_lower >= recall_target) & (g.fpr_ci_upper <= fpr_target + 1e-12)]
        out[str(method)] = int(ok.iloc[0].budget) if not ok.empty else f"Censored (>{max_budget})"
    return out


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "p011_seed_results.csv"
    if args.no_resume and raw_path.exists():
        raw_path.unlink()

    methods = tuple(args.methods)
    unknown_methods = sorted(set(methods) - set(ALL_METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")

    allocation_map = _allocation_map()
    budgets = [int(v) for v in args.budgets]
    unknown_budgets = sorted(set(budgets) - set(allocation_map))
    if unknown_budgets:
        raise ValueError(f"budgets not in frozen P0.7 grid: {unknown_budgets}")

    dataset = Path(args.dataset).resolve()
    cache = Path(args.feature_cache).resolve()
    metadata = load_cycle_metadata(dataset)
    batch = _ensure_feature_cache(dataset, cache, args.signal_set)

    raw_methods = [m for m in methods if m in {"RawConformalKNN", "PAKCT"}]
    raw_cycle_by_id: dict[int, np.ndarray] = {}
    if raw_methods:
        # Load only PRE_B healthy + anomalies used by the frozen P0.7 protocol.
        needed = [c.episode_id for c in metadata if (int(c.setting) == 73 and not c.anomaly) or c.anomaly]
        print(f"Loading raw measured cycles for {len(needed)} episodes...", flush=True)
        loaded = load_cycles(dataset, signal_set=args.signal_set, episode_ids=needed)
        raw_cycle_by_id = {int(c.episode_id): np.asarray(c.values, dtype=np.float64) for c in loaded}

    existing = pd.read_csv(raw_path) if raw_path.exists() and raw_path.stat().st_size else pd.DataFrame()
    completed: set[tuple[int, int, str]] = set()
    if not existing.empty:
        if set(existing.protocol_version.astype(str).unique()) != {PROTOCOL_VERSION}:
            raise RuntimeError("P0.11 checkpoint protocol mismatch")
        completed = {(int(r.budget), int(r.seed), str(r.method)) for r in existing.itertuples()}

    for budget in budgets:
        fit_n, cal_n = allocation_map[budget]
        for si, seed in enumerate(args.seeds, start=1):
            print(
                f"P0.11 B={budget} fit={fit_n} cal={cal_n} seed={seed} "
                f"({si}/{len(args.seeds)})",
                flush=True,
            )
            _, pool, eval_cycles, anomaly_cycles = _frozen_preb_partitions(
                metadata,
                seed=int(seed),
                eval_seed=int(args.evaluation_seed),
                eval_size=int(args.eval_size),
            )
            fit_cycles_meta = pool[:fit_n]
            cal_cycles_meta = pool[fit_n:fit_n + cal_n]
            fit_ids = [int(c.episode_id) for c in fit_cycles_meta]
            cal_ids = [int(c.episode_id) for c in cal_cycles_meta]
            eval_ids = [int(c.episode_id) for c in eval_cycles]
            anomaly_ids = [int(c.episode_id) for c in anomaly_cycles]

            groups = [set(fit_ids), set(cal_ids), set(eval_ids), set(anomaly_ids)]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        raise RuntimeError("P0.11 leakage detected")

            fit_x = _rows(batch, fit_ids)
            cal_x = _rows(batch, cal_ids)
            eval_x = _rows(batch, eval_ids)
            anomaly_x = _rows(batch, anomaly_ids)

            for method in methods:
                key = (budget, int(seed), str(method))
                if key in completed:
                    continue
                print(f"  {method}...", flush=True)

                if method in {"IsolationForest", "FeatureConformalKNN"}:
                    detector = _fit_feature_method(method, fit_x, int(seed))
                    cal_scores = detector.score_samples(cal_x)
                    normal_scores = detector.score_samples(eval_x)
                    anomaly_scores = detector.score_samples(anomaly_x)
                else:
                    fit_raw = [raw_cycle_by_id[i] for i in fit_ids]
                    cal_raw = [raw_cycle_by_id[i] for i in cal_ids]
                    eval_raw = [raw_cycle_by_id[i] for i in eval_ids]
                    anomaly_raw = [raw_cycle_by_id[i] for i in anomaly_ids]
                    detector = _fit_raw_method(method, fit_raw)
                    cal_scores = detector.score_cycles(cal_raw)
                    normal_scores = detector.score_cycles(eval_raw)
                    anomaly_scores = detector.score_cycles(anomaly_raw)

                threshold = BaseDetector.conformal_quantile(
                    cal_scores, alpha=float(args.false_alert_budget)
                )
                normal_pred = normal_scores > threshold
                anomaly_pred = anomaly_scores > threshold
                fpr = float(np.mean(normal_pred))
                recall = float(np.mean(anomaly_pred))
                y = np.concatenate([
                    np.zeros(len(normal_scores), dtype=int),
                    np.ones(len(anomaly_scores), dtype=int),
                ])
                s = np.concatenate([normal_scores, anomaly_scores])
                auroc = float(roc_auc_score(y, s))
                auprc = float(average_precision_score(y, s))

                row = {
                    "protocol_version": PROTOCOL_VERSION,
                    "budget": int(budget),
                    "fit_n": int(fit_n),
                    "calibration_n": int(cal_n),
                    "seed": int(seed),
                    "method": str(method),
                    "false_positive_rate": fpr,
                    "recall": recall,
                    "success": bool(recall >= args.recall_target and fpr <= args.false_alert_budget),
                    "auroc": auroc,
                    "auprc": auprc,
                    "threshold": float(threshold),
                    "calibration_max_score": float(np.max(cal_scores)),
                    "calibration_threshold_is_max": bool(np.isclose(threshold, np.max(cal_scores))),
                    "fit_ids_sha256": _sha256_ints(fit_ids),
                    "calibration_ids_sha256": _sha256_ints(cal_ids),
                    "normal_eval_ids_sha256": _sha256_ints(eval_ids),
                    "anomaly_eval_ids_sha256": _sha256_ints(anomaly_ids),
                    "normal_eval_n": len(eval_ids),
                    "anomaly_eval_n": len(anomaly_ids),
                }
                _append_row(raw_path, row)
                completed.add(key)

    results = pd.read_csv(raw_path)
    results = results[results.budget.isin(budgets) & results.method.isin(methods)].copy()
    summary = _summarize(results, int(args.bootstrap_replicates))
    summary.to_csv(output / "p011_summary.csv", index=False)
    bstar = _b_star(summary, float(args.recall_target), float(args.false_alert_budget))
    (output / "p011_b_star.json").write_text(json.dumps(bstar, indent=2), encoding="utf-8")

    audit = results.groupby(["budget", "seed"], sort=True).agg(
        fit_hashes=("fit_ids_sha256", "nunique"),
        calibration_hashes=("calibration_ids_sha256", "nunique"),
        normal_eval_hashes=("normal_eval_ids_sha256", "nunique"),
        anomaly_eval_hashes=("anomaly_eval_ids_sha256", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()
    audit.to_csv(output / "p011_split_audit.csv", index=False)
    if not ((audit.fit_hashes == 1) & (audit.calibration_hashes == 1) &
            (audit.normal_eval_hashes == 1) & (audit.anomaly_eval_hashes == 1)).all():
        raise RuntimeError("P0.11 split identity mismatch across methods")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol": "p07-fixed-budget-preb-v1",
        "methods": list(methods),
        "budgets": budgets,
        "seeds": [int(v) for v in args.seeds],
        "false_alert_budget": float(args.false_alert_budget),
        "recall_target": float(args.recall_target),
        "k": 10,
        "isolation_forest_n_estimators": 300,
        "m2n2_included": False,
        "m2n2_reason": "official method requires separate trainable time-series TTA pipeline; do not approximate",
    }
    (output / "p011_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nP0.11 summary")
    print(summary.to_string(index=False))
    print("\nB*", json.dumps(bstar, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-set", type=str, default="measured")
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[b for b, _, _ in DEFAULT_ALLOCATIONS])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--false-alert-budget", type=float, default=DEFAULT_FALSE_ALERT_BUDGET)
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
