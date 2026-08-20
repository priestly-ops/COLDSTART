"""P0.3b: conservative healthy-only source gating for RACE-Cov.

This experiment is a targeted hardening step after P0.3. It reuses the same
synthetic source regimes and target seeds, but removes the tiny holdout tuning
instability by using paired K-fold healthy risk. The primary comparison is
RACECovSafeCV vs BestTargetOnlySafeCV under the same target-data budget.

Predeclared advancement criteria (frozen before the confirmatory run):
- identical: lower 95% bootstrap CI on median gain > 0
- mild: lower 95% bootstrap CI on median gain > 0
- moderate: median gain >= 0
- adversarial: P(gain < -0.10) <= 0.20
- identical median gain >= 0.15 (retain >=~75% of prior ~0.20 adaptive gain)
- source-similarity correlation > 0.5 is diagnostic support, not sole gate

Synthetic truth is used only for reporting. Selection sees healthy target data
only and never sees anomaly labels or truth matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.covariance_transfer_estimators import (
    CovarianceEstimate,
    ledoit_wolf_covariance,
    race_covariance,
    safe_cv_race_covariance,
    safe_cv_target_only,
)
from src.precision_transfer_estimators import gaussian_precision_risk, relative_frobenius_error
from src.reproducibility import reproducibility_metadata

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p03b_safe_cv_gate"
PROTOCOL_VERSION = "p03b-safe-cv-gate-v1"
DEFAULT_DIMENSIONS = (20,)
DEFAULT_TARGET_NS = (25,)
DEFAULT_SOURCE_N = 100
DEFAULT_REPLICATIONS = 30
DEFAULT_EVAL_N = 500
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
DEFAULT_CV_FOLDS = 5
DEFAULT_SE_MULTIPLIER = 1.0

NEGATIVE_TRANSFER_THRESHOLD = -0.10
MAX_ADVERSARIAL_NEGATIVE_TRANSFER_FRACTION = 0.20
MIN_IDENTICAL_GAIN = 0.15
SIMILARITY_CORRELATION_DIAGNOSTIC = 0.50


def _precision_chain(p: int, edge: float = 0.22) -> np.ndarray:
    omega = np.eye(p, dtype=np.float64)
    for i in range(p - 1):
        omega[i, i + 1] = omega[i + 1, i] = -float(edge)
    return omega


def _assert_spd(name: str, omega: np.ndarray) -> None:
    minimum = float(np.min(np.linalg.eigvalsh(0.5 * (omega + omega.T))))
    if minimum <= 1e-8:
        raise ValueError(f"{name} is not SPD; min eigenvalue={minimum}")


def _source_truths(target_precision: np.ndarray) -> dict[str, np.ndarray]:
    p = target_precision.shape[0]
    regimes: dict[str, np.ndarray] = {"identical": target_precision.copy()}

    mild = np.eye(p)
    for i in range(p - 1):
        edge = 0.18 if i % 2 == 0 else 0.25
        mild[i, i + 1] = mild[i + 1, i] = -edge
    regimes["mild"] = mild

    moderate = np.eye(p)
    for i in range(0, p - 1, 2):
        moderate[i, i + 1] = moderate[i + 1, i] = -0.22
    for i in range(1, p - 2, 4):
        moderate[i, i + 2] = moderate[i + 2, i] = -0.22
    regimes["moderate"] = moderate

    disjoint = np.eye(p)
    for i in range(0, p - 2, 2):
        disjoint[i, i + 2] = disjoint[i + 2, i] = -0.22
    regimes["disjoint"] = disjoint

    adversarial = np.eye(p)
    for i in range(0, p - 2, 2):
        adversarial[i, i + 2] = adversarial[i + 2, i] = -0.38
    regimes["adversarial"] = adversarial

    for name, omega in regimes.items():
        _assert_spd(name, omega)
    return regimes


def _sample(rng: np.random.Generator, precision: np.ndarray, n: int) -> np.ndarray:
    covariance = np.linalg.inv(precision)
    return rng.multivariate_normal(np.zeros(precision.shape[0]), covariance, size=int(n))


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _truth_similarity(source_precision: np.ndarray, target_precision: np.ndarray) -> dict[str, float]:
    return {
        "source_target_precision_relative_frobenius": float(relative_frobenius_error(source_precision, target_precision)),
        "source_target_covariance_relative_frobenius": float(
            relative_frobenius_error(np.linalg.inv(source_precision), np.linalg.inv(target_precision))
        ),
    }


def _metrics(estimate: CovarianceEstimate, truth_precision: np.ndarray, eval_x: np.ndarray) -> dict[str, float]:
    truth_covariance = np.linalg.inv(truth_precision)
    est_scores = np.einsum("ni,ij,nj->n", eval_x, estimate.precision, eval_x)
    oracle_scores = np.einsum("ni,ij,nj->n", eval_x, truth_precision, eval_x)
    est_norm = est_scores / max(float(np.median(est_scores)), 1e-12)
    oracle_norm = oracle_scores / max(float(np.median(oracle_scores)), 1e-12)
    corr = float(np.corrcoef(est_scores, oracle_scores)[0, 1]) if np.std(est_scores) > 0 and np.std(oracle_scores) > 0 else np.nan
    return {
        "covariance_relative_frobenius": float(relative_frobenius_error(estimate.covariance, truth_covariance)),
        "precision_relative_frobenius": float(relative_frobenius_error(estimate.precision, truth_precision)),
        "heldout_gaussian_risk": float(gaussian_precision_risk(eval_x, estimate.precision)),
        "condition_number": float(np.linalg.cond(estimate.covariance)),
        "mahalanobis_score_correlation": corr,
        "mahalanobis_normalized_median_abs_error": float(np.median(np.abs(est_norm - oracle_norm))),
    }


def _bootstrap_median(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boot[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return float(np.median(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    base = results[results.method == "BestTargetOnlySafeCV"][keys + ["covariance_relative_frobenius"]].rename(
        columns={"covariance_relative_frobenius": "best_target_error"}
    )
    if base.duplicated(keys).any():
        raise RuntimeError("Duplicate BestTargetOnlySafeCV rows")
    compared = results[results.method.isin(["RACECov60Full", "RACECovSafeCV"])].merge(
        base, on=keys, validate="many_to_one"
    )
    compared["gain_vs_best_target"] = (
        compared.best_target_error - compared.covariance_relative_frobenius
    ) / compared.best_target_error
    compared["meaningful_negative_transfer"] = compared.gain_vs_best_target < NEGATIVE_TRANSFER_THRESHOLD

    rows: list[dict[str, object]] = []
    for gi, (key, g) in enumerate(compared.groupby(["p", "target_n", "source_kind", "method"], sort=True)):
        p, n, source_kind, method = key
        med, lo, hi = _bootstrap_median(g.gain_vs_best_target.to_numpy(), n_boot, 12000 + gi)
        rows.append({
            "p": int(p), "N": int(n), "source_kind": str(source_kind), "method": str(method),
            "replications": int(len(g)),
            "median_gain_vs_best_target": med,
            "gain_ci_lower": lo,
            "gain_ci_upper": hi,
            "fraction_better_than_best_target": float(np.mean(g.gain_vs_best_target > 0)),
            "meaningful_negative_transfer_fraction": float(np.mean(g.meaningful_negative_transfer)),
            "median_covariance_error": float(g.covariance_relative_frobenius.median()),
            "median_precision_error": float(g.precision_relative_frobenius.median()),
            "median_heldout_gaussian_risk": float(g.heldout_gaussian_risk.median()),
            "median_mahalanobis_score_correlation": float(g.mahalanobis_score_correlation.median()),
            "median_mahalanobis_normalized_abs_error": float(g.mahalanobis_normalized_median_abs_error.median()),
            "source_target_precision_relative_frobenius": float(g.source_target_precision_relative_frobenius.iloc[0]),
            "source_target_covariance_relative_frobenius": float(g.source_target_covariance_relative_frobenius.iloc[0]),
        })
    return pd.DataFrame(rows)


def _gate_decision(summary: pd.DataFrame, similarity_corr: float) -> dict[str, object]:
    safe = summary[summary.method == "RACECovSafeCV"].set_index("source_kind")
    required = {"identical", "mild", "moderate", "adversarial"}
    missing = required - set(safe.index)
    if missing:
        raise RuntimeError(f"Missing gate rows: {sorted(missing)}")

    checks = {
        "identical_ci_lower_gt_zero": bool(safe.loc["identical", "gain_ci_lower"] > 0.0),
        "mild_ci_lower_gt_zero": bool(safe.loc["mild", "gain_ci_lower"] > 0.0),
        "moderate_median_gain_ge_zero": bool(safe.loc["moderate", "median_gain_vs_best_target"] >= 0.0),
        "adversarial_negative_transfer_fraction_le_0_20": bool(
            safe.loc["adversarial", "meaningful_negative_transfer_fraction"] <= MAX_ADVERSARIAL_NEGATIVE_TRANSFER_FRACTION
        ),
        "identical_median_gain_ge_0_15": bool(safe.loc["identical", "median_gain_vs_best_target"] >= MIN_IDENTICAL_GAIN),
        "similarity_correlation_gt_0_5_diagnostic": bool(similarity_corr > SIMILARITY_CORRELATION_DIAGNOSTIC),
    }
    primary_names = [k for k in checks if not k.endswith("_diagnostic")]
    primary_pass = all(checks[k] for k in primary_names)
    return {
        "primary_gate_pass": primary_pass,
        "decision": "P0.3B_PASS_ADVANCE" if primary_pass else "P0.3B_HOLD",
        "checks": checks,
        "frozen_thresholds": {
            "meaningful_negative_transfer_gain_threshold": NEGATIVE_TRANSFER_THRESHOLD,
            "max_adversarial_negative_transfer_fraction": MAX_ADVERSARIAL_NEGATIVE_TRANSFER_FRACTION,
            "min_identical_median_gain": MIN_IDENTICAL_GAIN,
            "similarity_correlation_diagnostic": SIMILARITY_CORRELATION_DIAGNOSTIC,
        },
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "p03b_results.csv"
    tuning_path = out / "p03b_cv_tuning.csv"
    if args.no_resume:
        for path in (results_path, tuning_path):
            if path.exists():
                path.unlink()

    completed: set[tuple[int, int, int, str, str]] = set()
    if results_path.exists() and results_path.stat().st_size:
        old = pd.read_csv(results_path)
        completed = {
            (int(r.p), int(r.target_n), int(r.replication), str(r.source_kind), str(r.method))
            for r in old.itertuples(index=False)
        }

    for p in args.dimensions:
        target_truth = _precision_chain(int(p))
        source_truths = _source_truths(target_truth)
        for target_n in args.target_ns:
            folds = min(int(args.cv_folds), int(target_n))
            if folds < 2:
                raise ValueError("Need at least 2 CV folds")
            for rep in range(args.replications):
                # Same target seeds as P0.3 for paired historical comparison.
                target_seed = 8_000_000 + int(p) * 100_000 + int(target_n) * 1_000 + rep
                target_x = _sample(np.random.default_rng(target_seed), target_truth, int(target_n))
                eval_x = _sample(np.random.default_rng(target_seed + 90_000_000), target_truth, int(args.eval_n))
                target_hash = _sha256(target_x)
                cv_seed = target_seed + 12345

                best_target, target_cv_rows = safe_cv_target_only(
                    target_x,
                    ridge_gammas=tuple(args.ridge_gammas),
                    n_folds=folds,
                    seed=cv_seed,
                    method="BestTargetOnlySafeCV",
                )

                for source_index, (source_kind, source_truth) in enumerate(source_truths.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source_x = _sample(np.random.default_rng(source_seed), source_truth, int(args.source_n))
                    similarity = _truth_similarity(source_truth, target_truth)
                    safe = safe_cv_race_covariance(
                        target_x,
                        source_x,
                        lambdas=tuple(args.race_lambdas),
                        n_folds=folds,
                        seed=cv_seed,
                        se_multiplier=float(args.se_multiplier),
                        method="RACECovSafeCV",
                    )
                    fixed = race_covariance(target_x, source_x, lambda_reg=60.0, method="RACECov60Full")
                    target_lw = ledoit_wolf_covariance(target_x, method="TargetLedoitWolfFull")

                    estimates = [target_lw, best_target, fixed, safe.estimate]
                    for est in estimates:
                        key = (int(p), int(target_n), int(rep), source_kind, est.method)
                        if key in completed:
                            continue
                        row = {
                            "p": int(p), "target_n": int(target_n), "replication": int(rep),
                            "source_kind": source_kind, "method": est.method,
                            "target_seed": int(target_seed), "source_seed": int(source_seed),
                            "target_sample_sha256": target_hash,
                            "target_data_size": int(len(target_x)),
                            "cv_seed": int(cv_seed), "cv_folds": int(folds),
                            "selected_base_method": str(est.metadata.get("selected_family", "")),
                            "selected_candidate": str(est.metadata.get("selected_candidate", "")),
                            "lambda_reg": float(est.metadata.get("lambda_reg", np.nan)),
                            "source_weight": float(est.metadata.get("source_weight", np.nan)),
                            "accepted_transfer": bool(est.metadata.get("accepted_transfer", False)),
                            **similarity,
                            **_metrics(est, target_truth, eval_x),
                        }
                        pd.DataFrame([row]).to_csv(
                            results_path, mode="a",
                            header=not results_path.exists() or results_path.stat().st_size == 0,
                            index=False,
                        )
                        completed.add(key)

                    target_frame = pd.DataFrame(target_cv_rows)
                    target_frame.insert(0, "method_family", "BestTargetOnlySafeCV")
                    target_frame.insert(0, "source_kind", source_kind)
                    target_frame.insert(0, "replication", rep)
                    target_frame.insert(0, "target_n", target_n)
                    target_frame.insert(0, "p", p)
                    target_frame.to_csv(tuning_path, mode="a", header=not tuning_path.exists() or tuning_path.stat().st_size == 0, index=False)

                    safe_frame = pd.DataFrame(list(safe.cv_rows))
                    safe_frame.insert(0, "method_family", "RACECovSafeCV")
                    safe_frame.insert(0, "source_kind", source_kind)
                    safe_frame.insert(0, "replication", rep)
                    safe_frame.insert(0, "target_n", target_n)
                    safe_frame.insert(0, "p", p)
                    safe_frame.to_csv(tuning_path, mode="a", header=not tuning_path.exists() or tuning_path.stat().st_size == 0, index=False)

                    print(f"P0.3b p={p} N={target_n} rep={rep+1}/{args.replications} source={source_kind}", flush=True)

    results = pd.read_csv(results_path)
    pairing = results.groupby(["p", "target_n", "replication"]).agg(
        source_kinds=("source_kind", "nunique"),
        target_seeds=("target_seed", "nunique"),
        target_hashes=("target_sample_sha256", "nunique"),
        target_sizes=("target_data_size", "nunique"),
        cv_seeds=("cv_seed", "nunique"),
        cv_folds=("cv_folds", "nunique"),
    ).reset_index()
    expected_sources = len(_source_truths(_precision_chain(int(args.dimensions[0]))))
    if not (
        (pairing.source_kinds == expected_sources).all()
        and (pairing.target_seeds == 1).all()
        and (pairing.target_hashes == 1).all()
        and (pairing.target_sizes == 1).all()
        and (pairing.cv_seeds == 1).all()
        and (pairing.cv_folds == 1).all()
    ):
        raise RuntimeError("P0.3b pairing/data-budget audit failed")
    pairing.to_csv(out / "p03b_pairing_audit.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p03b_paired_summary.csv", index=False)

    safe = results[results.method == "RACECovSafeCV"].copy()
    base = results[results.method == "BestTargetOnlySafeCV"][
        ["p", "target_n", "replication", "source_kind", "covariance_relative_frobenius"]
    ].rename(columns={"covariance_relative_frobenius": "base_error"})
    safe = safe.merge(base, on=["p", "target_n", "replication", "source_kind"], validate="one_to_one")
    safe["gain"] = (safe.base_error - safe.covariance_relative_frobenius) / safe.base_error

    trend_rows: list[dict[str, float | int]] = []
    for (p, n), g in safe.groupby(["p", "target_n"], sort=True):
        sim = -g.source_target_covariance_relative_frobenius.to_numpy(float)
        gain = g.gain.to_numpy(float)
        corr = float(np.corrcoef(sim, gain)[0, 1]) if np.std(sim) > 0 and np.std(gain) > 0 else np.nan
        trend_rows.append({"p": int(p), "N": int(n), "pearson_gain_vs_negative_covariance_truth_distance": corr})
    trend = pd.DataFrame(trend_rows)
    trend.to_csv(out / "p03b_similarity_trend.csv", index=False)

    usage = results[results.method == "RACECovSafeCV"].groupby(["p", "target_n", "source_kind"], sort=True).agg(
        replications=("replication", "nunique"),
        median_lambda=("lambda_reg", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_zero_source_weight=("source_weight", lambda s: float(np.mean(np.asarray(s) <= 1e-12))),
        fraction_transfer_accepted=("accepted_transfer", "mean"),
    ).reset_index()
    usage.to_csv(out / "p03b_source_weight_audit.csv", index=False)

    # Current confirmation setup has one p/N; write one decision per p/N.
    decisions = []
    for (p, n), block in summary.groupby(["p", "N"], sort=True):
        corr_row = trend[(trend.p == p) & (trend.N == n)]
        corr = float(corr_row.iloc[0].pearson_gain_vs_negative_covariance_truth_distance)
        decisions.append({"p": int(p), "N": int(n), **_gate_decision(block, corr)})
    (out / "p03b_gate_decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions],
        "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n),
        "replications": int(args.replications),
        "eval_n": int(args.eval_n),
        "cv_folds": int(args.cv_folds),
        "se_multiplier": float(args.se_multiplier),
        "race_lambda_grid": [float(v) for v in args.race_lambdas],
        "ridge_gamma_grid": [float(v) for v in args.ridge_gammas],
        "source_regimes": list(_source_truths(_precision_chain(int(args.dimensions[0]))).keys()),
        "primary_comparison": "RACECovSafeCV vs BestTargetOnlySafeCV on covariance relative Frobenius error",
        "selection_rule": "paired K-fold healthy risk; transfer only if mean improvement - se_multiplier*SE > 0",
        "truth_used_for_tuning": False,
        "anomaly_labels_used_for_tuning": False,
        "negative_transfer_definition": "gain_vs_BestTargetOnlySafeCV < -0.10",
        "frozen_gate_thresholds": {
            "identical_ci_lower_gt_zero": True,
            "mild_ci_lower_gt_zero": True,
            "moderate_median_gain_ge_zero": True,
            "max_adversarial_negative_transfer_fraction": MAX_ADVERSARIAL_NEGATIVE_TRANSFER_FRACTION,
            "min_identical_median_gain": MIN_IDENTICAL_GAIN,
            "similarity_correlation_gt": SIMILARITY_CORRELATION_DIAGNOSTIC,
        },
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (out / "p03b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.3b outputs written to {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--target-ns", type=int, nargs="+", default=list(DEFAULT_TARGET_NS))
    parser.add_argument("--source-n", type=int, default=DEFAULT_SOURCE_N)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    parser.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--se-multiplier", type=float, default=DEFAULT_SE_MULTIPLIER)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if any(v < 2 for v in args.dimensions): parser.error("dimensions must be >=2")
    if any(v < 10 for v in args.target_ns): parser.error("target-ns must be >=10")
    if args.source_n < 10: parser.error("source-n must be >=10")
    if args.replications <= 0: parser.error("replications must be positive")
    if args.eval_n < 50: parser.error("eval-n must be >=50")
    if args.cv_folds < 2: parser.error("cv-folds must be >=2")
    if args.se_multiplier < 0: parser.error("se-multiplier must be non-negative")
    if any(v <= 0 or v > 1 for v in args.ridge_gammas): parser.error("ridge gammas must be in (0,1]")
    if any(v < 0 for v in args.race_lambdas): parser.error("race lambdas must be non-negative")
    if 0.0 not in [float(v) for v in args.race_lambdas]: parser.error("race lambda grid must include 0")
    return args


if __name__ == "__main__":
    run(parse_args())
