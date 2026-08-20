"""P0.4: real healthy voraus-AD covariance-transfer feasibility audit.

This is the first post-synthetic test of the frozen RACE-Cov Safe-CV policy.
It uses only healthy source/target robot executions. No anomaly labels, anomaly
scores, or deployment thresholds enter fitting, tuning, or model selection.

Protocol
--------
Source domain: setting 72 (PRE_A), healthy only.
Target domain: setting 73 (PRE_B), healthy only.
Feature schema: frozen cycle statistics from src.feature_extractor.
Target evaluation/calibration episode IDs are frozen across commissioning seeds
via create_frozen_evaluation_split(). Only commissioning composition varies.

Preprocessing is target-only: a robust scaler is fit on each target
commissioning set and then applied unchanged to source, target calibration, and
target evaluation features. This avoids source information entering the
TargetOnly baseline through preprocessing.

Primary metric: held-out healthy Gaussian precision risk on the fixed target
normal-evaluation set (lower is better). The primary paired estimand is
risk_gain = (TargetOnlyRisk - MethodRisk) / |TargetOnlyRisk|.

The fixed target calibration partition is used only to construct a reporting
reference covariance/precision; it is never used for Safe-CV source selection.

Frozen advancement criteria for RACECovSafeCV:
- N=25: lower 95% bootstrap CI of median held-out risk gain > 0.
- N=50: median held-out risk gain >= 0.
- every tested N: fraction with risk_gain < -0.10 <= 0.20.
- N=25: median covariance-distance gain to the fixed calibration reference >= 0.

These criteria are frozen before looking at P0.4 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.covariance_transfer_estimators import (
    CovarianceEstimate,
    ledoit_wolf_covariance,
    pooled_ledoit_wolf,
    race_covariance,
    safe_cv_race_covariance,
    safe_cv_target_only,
)
from src.feature_extractor import (
    extract_feature_batch,
    load_feature_batch,
    save_feature_batch,
)
from src.precision_transfer_estimators import (
    fit_robust_scaler,
    gaussian_precision_risk,
    relative_frobenius_error,
)
from src.split_generator import (
    SOURCE_SETTING,
    TARGET_SETTING,
    create_frozen_evaluation_split,
)
from src.voraus_loader import load_cycle_metadata, load_cycles

PROTOCOL_VERSION = "p04-real-healthy-voraus-covariance-v1"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p04_real_healthy_covariance"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "cache" / "voraus_measured_healthy_features.npz"
DEFAULT_NS = (25, 50, 100)
DEFAULT_SEEDS = tuple(range(30))
DEFAULT_CALIBRATION_SIZE = 100
DEFAULT_EVAL_SIZE = 100
DEFAULT_EVALUATION_SEED = 42
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
DEFAULT_CV_FOLDS = 5
NEGATIVE_TRANSFER_THRESHOLD = -0.10
MAX_NEGATIVE_TRANSFER_FRACTION = 0.20


def _sha256_ints(values: list[int]) -> str:
    arr = np.asarray(sorted(values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _bootstrap_median(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty array")
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boot[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return (
        float(np.median(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _score_correlation(estimate: CovarianceEstimate, reference: CovarianceEstimate, eval_x: np.ndarray) -> float:
    a = np.einsum("ni,ij,nj->n", eval_x, estimate.precision, eval_x)
    b = np.einsum("ni,ij,nj->n", eval_x, reference.precision, eval_x)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ensure_feature_cache(dataset: Path, cache: Path, signal_set: str) -> object:
    if cache.exists():
        batch = load_feature_batch(cache)
        # Defensive check: cache must contain only healthy PRE_A/PRE_B rows.
        allowed = {int(SOURCE_SETTING), int(TARGET_SETTING)}
        if bool(np.any(batch.anomaly_labels)):
            raise RuntimeError("P0.4 feature cache unexpectedly contains anomalies")
        if not set(batch.settings.tolist()).issubset(allowed):
            raise RuntimeError("P0.4 feature cache contains settings outside PRE_A/PRE_B")
        return batch

    metadata = load_cycle_metadata(dataset)
    healthy_ids = [
        int(c.episode_id)
        for c in metadata
        if (not c.anomaly) and int(c.setting) in {int(SOURCE_SETTING), int(TARGET_SETTING)}
    ]
    cycles = load_cycles(dataset, signal_set=signal_set, episode_ids=healthy_ids)
    batch = extract_feature_batch(cycles)
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_feature_batch(
        batch,
        cache,
        metadata={
            "protocol": PROTOCOL_VERSION,
            "dataset": str(dataset),
            "signal_set": signal_set,
            "healthy_only": True,
            "settings": [int(SOURCE_SETTING), int(TARGET_SETTING)],
        },
    )
    return batch


def _feature_rows(batch: object, ids: list[int]) -> np.ndarray:
    subset = batch.select_episode_ids(ids, preserve_requested_order=True, require_all=True)
    return np.asarray(subset.features, dtype=np.float64)


def _run_one(
    *,
    batch: object,
    metadata: list[object],
    n: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    split = create_frozen_evaluation_split(
        metadata,
        commissioning_size=int(n),
        commissioning_seed=int(seed),
        evaluation_seed=int(args.evaluation_seed),
        calibration_size=int(args.calibration_size),
        normal_evaluation_size=int(args.eval_size),
        maximum_commissioning_size=max(int(v) for v in args.ns),
    )
    split.verify_no_overlap()

    source_ids = [int(c.episode_id) for c in split.source_train]
    target_ids = [int(c.episode_id) for c in split.target_commissioning]
    calibration_ids = [int(c.episode_id) for c in split.target_calibration]
    eval_ids = [int(c.episode_id) for c in split.target_normal_evaluation]

    source_raw = _feature_rows(batch, source_ids)
    target_raw = _feature_rows(batch, target_ids)
    calibration_raw = _feature_rows(batch, calibration_ids)
    eval_raw = _feature_rows(batch, eval_ids)

    # Target-only preprocessing: source information never influences scaling.
    scaler = fit_robust_scaler(target_raw, mode="target")
    source_x = scaler.transform(source_raw)
    target_x = scaler.transform(target_raw)
    calibration_x = scaler.transform(calibration_raw)
    eval_x = scaler.transform(eval_raw)

    folds = min(int(args.cv_folds), int(n))
    cv_seed = 4_200_000 + int(n) * 1000 + int(seed)

    best_target, target_cv_rows = safe_cv_target_only(
        target_x,
        ridge_gammas=tuple(float(x) for x in args.ridge_gammas),
        n_folds=folds,
        seed=cv_seed,
        method="BestTargetOnlySafeCV",
    )
    safe = safe_cv_race_covariance(
        target_x,
        source_x,
        lambdas=tuple(float(x) for x in args.race_lambdas),
        n_folds=folds,
        seed=cv_seed,
        se_multiplier=float(args.se_multiplier),
        method="RACECovSafeCV",
    )
    fixed = race_covariance(target_x, source_x, lambda_reg=60.0, method="RACECov60Full")
    source_only = ledoit_wolf_covariance(source_x, method="SourceLedoitWolf")
    pooled = pooled_ledoit_wolf(target_x, source_x)
    reference = ledoit_wolf_covariance(calibration_x, method="TargetCalibrationReference")

    estimates = [best_target, fixed, safe.estimate, source_only, pooled]
    rows: list[dict[str, object]] = []
    for est in estimates:
        risk = float(gaussian_precision_risk(eval_x, est.precision))
        ref_cov_error = float(relative_frobenius_error(est.covariance, reference.covariance))
        ref_precision_error = float(relative_frobenius_error(est.precision, reference.precision))
        rows.append({
            "N": int(n),
            "seed": int(seed),
            "method": est.method,
            "source_setting": int(SOURCE_SETTING),
            "target_setting": int(TARGET_SETTING),
            "feature_count": int(target_x.shape[1]),
            "source_n": int(len(source_x)),
            "target_n": int(len(target_x)),
            "calibration_n": int(len(calibration_x)),
            "eval_n": int(len(eval_x)),
            "cv_seed": int(cv_seed),
            "cv_folds": int(folds),
            "source_ids_sha256": _sha256_ints(source_ids),
            "commissioning_ids_sha256": _sha256_ints(target_ids),
            "calibration_ids_sha256": _sha256_ints(calibration_ids),
            "eval_ids_sha256": _sha256_ints(eval_ids),
            "heldout_gaussian_risk": risk,
            "reference_covariance_relative_frobenius": ref_cov_error,
            "reference_precision_relative_frobenius": ref_precision_error,
            "reference_mahalanobis_score_correlation": _score_correlation(est, reference, eval_x),
            "condition_number": float(np.linalg.cond(est.covariance)),
            "selected_lambda": float(est.metadata.get("lambda_reg", np.nan)),
            "source_weight": float(est.metadata.get("source_weight", np.nan)),
            "accepted_transfer": bool(est.metadata.get("accepted_transfer", False)),
            "selected_target_family": str(est.metadata.get("selected_family", "")),
            "selected_target_candidate": str(est.metadata.get("selected_candidate", "")),
        })

    tuning_rows: list[dict[str, object]] = []
    for row in target_cv_rows:
        tuning_rows.append({
            "N": int(n), "seed": int(seed), "selector": "BestTargetOnlySafeCV", **row
        })
    for row in safe.cv_rows:
        tuning_rows.append({
            "N": int(n), "seed": int(seed), "selector": "RACECovSafeCV", **row
        })
    return rows, tuning_rows


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["N", "seed"]
    base = results[results.method == "BestTargetOnlySafeCV"][
        keys + ["heldout_gaussian_risk", "reference_covariance_relative_frobenius"]
    ].rename(columns={
        "heldout_gaussian_risk": "base_risk",
        "reference_covariance_relative_frobenius": "base_ref_cov_error",
    })
    compared = results[results.method != "BestTargetOnlySafeCV"].merge(base, on=keys, validate="many_to_one")
    compared["risk_gain_vs_best_target"] = (
        compared.base_risk - compared.heldout_gaussian_risk
    ) / np.maximum(np.abs(compared.base_risk), 1e-12)
    compared["reference_covariance_gain_vs_best_target"] = (
        compared.base_ref_cov_error - compared.reference_covariance_relative_frobenius
    ) / np.maximum(compared.base_ref_cov_error, 1e-12)
    compared["meaningful_negative_transfer"] = (
        compared.risk_gain_vs_best_target < NEGATIVE_TRANSFER_THRESHOLD
    )

    rows: list[dict[str, object]] = []
    for gi, ((n, method), g) in enumerate(compared.groupby(["N", "method"], sort=True)):
        med, lo, hi = _bootstrap_median(g.risk_gain_vs_best_target.to_numpy(), n_boot, 51_000 + gi)
        cov_med, cov_lo, cov_hi = _bootstrap_median(
            g.reference_covariance_gain_vs_best_target.to_numpy(), n_boot, 61_000 + gi
        )
        rows.append({
            "N": int(n),
            "method": str(method),
            "seeds": int(g.seed.nunique()),
            "median_risk_gain_vs_best_target": med,
            "risk_gain_ci_lower": lo,
            "risk_gain_ci_upper": hi,
            "fraction_risk_better_than_best_target": float(np.mean(g.risk_gain_vs_best_target > 0)),
            "meaningful_negative_transfer_fraction": float(np.mean(g.meaningful_negative_transfer)),
            "median_reference_covariance_gain_vs_best_target": cov_med,
            "reference_covariance_gain_ci_lower": cov_lo,
            "reference_covariance_gain_ci_upper": cov_hi,
            "median_heldout_gaussian_risk": float(g.heldout_gaussian_risk.median()),
            "median_reference_score_correlation": float(g.reference_mahalanobis_score_correlation.median()),
            "median_source_weight": float(g.source_weight.dropna().median()) if g.source_weight.notna().any() else np.nan,
            "fraction_transfer_accepted": float(g.accepted_transfer.mean()),
        })
    return pd.DataFrame(rows)


def _gate(summary: pd.DataFrame, ns: list[int]) -> dict[str, object]:
    safe = summary[summary.method == "RACECovSafeCV"].set_index("N")
    required = {25, 50}
    missing = required - set(int(v) for v in safe.index)
    if missing:
        raise RuntimeError(f"P0.4 gate requires N=25 and N=50; missing {sorted(missing)}")

    checks: dict[str, bool] = {
        "N25_risk_gain_ci_lower_gt_zero": bool(safe.loc[25, "risk_gain_ci_lower"] > 0.0),
        "N50_median_risk_gain_ge_zero": bool(safe.loc[50, "median_risk_gain_vs_best_target"] >= 0.0),
        "N25_reference_covariance_gain_ge_zero": bool(
            safe.loc[25, "median_reference_covariance_gain_vs_best_target"] >= 0.0
        ),
    }
    for n in ns:
        if int(n) in safe.index:
            checks[f"N{int(n)}_negative_transfer_fraction_le_0_20"] = bool(
                safe.loc[int(n), "meaningful_negative_transfer_fraction"] <= MAX_NEGATIVE_TRANSFER_FRACTION
            )
    primary_pass = all(checks.values())
    return {
        "primary_gate_pass": bool(primary_pass),
        "decision": "P0.4_PASS_ADVANCE_ANOMALY_EVAL" if primary_pass else "P0.4_HOLD",
        "checks": checks,
        "frozen_thresholds": {
            "meaningful_negative_transfer_gain_threshold": NEGATIVE_TRANSFER_THRESHOLD,
            "max_negative_transfer_fraction": MAX_NEGATIVE_TRANSFER_FRACTION,
            "N25_min_risk_gain_ci_lower": 0.0,
            "N50_min_median_risk_gain": 0.0,
            "N25_min_reference_covariance_gain": 0.0,
        },
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    cache = Path(args.feature_cache)

    metadata = load_cycle_metadata(dataset)
    metadata = [
        c for c in metadata
        if (not c.anomaly) and int(c.setting) in {int(SOURCE_SETTING), int(TARGET_SETTING)}
    ]
    batch = _ensure_feature_cache(dataset, cache, args.signal_set)

    rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    total = len(args.ns) * len(args.seeds)
    done = 0
    for n in args.ns:
        for seed in args.seeds:
            done += 1
            print(f"[P0.4] N={n} seed={seed} ({done}/{total})", flush=True)
            one_rows, one_tuning = _run_one(
                batch=batch,
                metadata=metadata,
                n=int(n),
                seed=int(seed),
                args=args,
            )
            rows.extend(one_rows)
            tuning_rows.extend(one_tuning)

    results = pd.DataFrame(rows)
    results.to_csv(out / "p04_results.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "p04_cv_tuning.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p04_paired_summary.csv", index=False)

    weight = (
        results[results.method == "RACECovSafeCV"]
        .groupby("N", sort=True)
        .agg(
            seeds=("seed", "nunique"),
            median_lambda=("selected_lambda", "median"),
            median_source_weight=("source_weight", "median"),
            fraction_zero_source_weight=("source_weight", lambda s: float(np.mean(np.isclose(s, 0.0)))),
            fraction_transfer_accepted=("accepted_transfer", "mean"),
        )
        .reset_index()
    )
    weight.to_csv(out / "p04_source_weight_audit.csv", index=False)

    split_audit = (
        results.groupby(["N", "seed"], sort=True)
        .agg(
            source_hashes=("source_ids_sha256", "nunique"),
            commissioning_hashes=("commissioning_ids_sha256", "nunique"),
            calibration_hashes=("calibration_ids_sha256", "nunique"),
            eval_hashes=("eval_ids_sha256", "nunique"),
            methods=("method", "nunique"),
        )
        .reset_index()
    )
    split_audit.to_csv(out / "p04_split_audit.csv", index=False)

    decision = _gate(summary, [int(n) for n in args.ns])
    decision.update({
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "feature_cache": str(cache),
        "source_setting": int(SOURCE_SETTING),
        "target_setting": int(TARGET_SETTING),
        "evaluation_seed": int(args.evaluation_seed),
        "anomaly_labels_used": False,
        "preprocessing": "target-commissioning robust scaler only",
    })
    (out / "p04_gate_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--signal-set", choices=["measured", "machine"], default="measured")
    ap.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE)
    ap.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    ap.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    ap.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    ap.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    ap.add_argument("--se-multiplier", type=float, default=1.0)
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    ap.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    args = ap.parse_args()
    if any(int(n) < 25 for n in args.ns):
        ap.error("P0.4 frozen RACE transfer policy is defined only for N>=25")
    if max(int(n) for n in args.ns) > 119:
        ap.error("voraus frozen split leaves 119 target healthy episodes for commissioning")
    if 25 not in args.ns or 50 not in args.ns:
        ap.error("P0.4 gate requires --ns to include 25 and 50")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds must be unique")
    return args


if __name__ == "__main__":
    run(parse_args())
