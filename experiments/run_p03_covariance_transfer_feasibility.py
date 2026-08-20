"""P0.3: covariance-transfer feasibility for COLDSTART/RACE.

Goal
----
Test whether source-assisted covariance shrinkage can beat strong, target-only
regularization under a strictly matched target-data budget.

Primary scientific comparison
-----------------------------
RACECovAdaptive vs BestTargetOnly, especially for identical/mild sources.

Synthetic truth is used only for reporting error and source similarity.  All
hyperparameter/model selection uses held-out healthy target data only.
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
    choose_by_healthy_risk,
    ledoit_wolf_covariance,
    pooled_ledoit_wolf,
    race_covariance,
    ridge_covariance,
)
from src.precision_transfer_estimators import gaussian_precision_risk, relative_frobenius_error
from src.reproducibility import reproducibility_metadata

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p03_covariance_transfer_feasibility"
PROTOCOL_VERSION = "p03-covariance-transfer-v1"
DEFAULT_DIMENSIONS = (20, 40)
DEFAULT_TARGET_NS = (10, 25, 50)
DEFAULT_SOURCE_N = 200
DEFAULT_REPLICATIONS = 50
DEFAULT_EVAL_N = 500
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)


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


def _split_target(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    tune_n = max(3, int(round(0.20 * n)))
    model_n = n - tune_n
    if model_n < 4:
        raise ValueError(f"N={n} leaves too few model rows after healthy tuning split")
    return x[:model_n], x[model_n:]


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _truth_similarity(source_precision: np.ndarray, target_precision: np.ndarray) -> dict[str, float]:
    return {
        "source_target_precision_relative_frobenius": float(
            relative_frobenius_error(source_precision, target_precision)
        ),
        "source_target_covariance_relative_frobenius": float(
            relative_frobenius_error(np.linalg.inv(source_precision), np.linalg.inv(target_precision))
        ),
    }


def _select_target_ridge(
    model_x: np.ndarray,
    tune_x: np.ndarray,
    gammas: tuple[float, ...],
) -> tuple[CovarianceEstimate, list[dict[str, float | str]]]:
    candidates = [
        ridge_covariance(model_x, gamma=g, method=f"TargetRidge[g={g:g}]")
        for g in gammas
    ]
    return choose_by_healthy_risk(candidates, tune_x)


def _select_race(
    model_x: np.ndarray,
    source_x: np.ndarray,
    tune_x: np.ndarray,
    lambdas: tuple[float, ...],
) -> tuple[CovarianceEstimate, list[dict[str, float | str]]]:
    candidates = [
        race_covariance(
            model_x,
            source_x,
            lambda_reg=lam,
            method=f"RACECovAdaptive[lambda={lam:g}]",
        )
        for lam in lambdas
    ]
    return choose_by_healthy_risk(candidates, tune_x)


def _metrics(
    estimate: CovarianceEstimate,
    truth_precision: np.ndarray,
    eval_x: np.ndarray,
) -> dict[str, float]:
    truth_covariance = np.linalg.inv(truth_precision)
    est_cov = estimate.covariance
    est_prec = estimate.precision

    est_scores = np.einsum("ni,ij,nj->n", eval_x, est_prec, eval_x)
    oracle_scores = np.einsum("ni,ij,nj->n", eval_x, truth_precision, eval_x)
    est_norm = est_scores / max(float(np.median(est_scores)), 1e-12)
    oracle_norm = oracle_scores / max(float(np.median(oracle_scores)), 1e-12)
    if np.std(est_scores) > 0 and np.std(oracle_scores) > 0:
        score_corr = float(np.corrcoef(est_scores, oracle_scores)[0, 1])
    else:
        score_corr = np.nan

    return {
        "covariance_relative_frobenius": float(relative_frobenius_error(est_cov, truth_covariance)),
        "precision_relative_frobenius": float(relative_frobenius_error(est_prec, truth_precision)),
        "heldout_gaussian_risk": float(gaussian_precision_risk(eval_x, est_prec)),
        "condition_number": float(np.linalg.cond(est_cov)),
        "mahalanobis_score_correlation": score_corr,
        "mahalanobis_normalized_median_abs_error": float(np.median(np.abs(est_norm - oracle_norm))),
    }


def _bootstrap_median(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boot[i] = np.median(rng.choice(values, size=len(values), replace=True))
    return (
        float(np.median(values)),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    base = results[results.method == "BestTargetOnly"][keys + ["covariance_relative_frobenius"]].rename(
        columns={"covariance_relative_frobenius": "best_target_error"}
    )
    if base.duplicated(keys).any():
        raise RuntimeError("Duplicate BestTargetOnly rows")

    compared = results[results.method.isin([
        "SourceLedoitWolf",
        "PooledLedoitWolf",
        "RACECov60",
        "RACECovAdaptive",
    ])].merge(base, on=keys, validate="many_to_one")
    compared["gain_vs_best_target"] = (
        compared.best_target_error - compared.covariance_relative_frobenius
    ) / compared.best_target_error
    compared["meaningful_negative_transfer"] = compared.gain_vs_best_target < -0.10

    rows: list[dict[str, object]] = []
    group_cols = ["p", "target_n", "source_kind", "method"]
    for gi, (key, g) in enumerate(compared.groupby(group_cols, sort=True)):
        p, n, source_kind, method = key
        med, lo, hi = _bootstrap_median(g.gain_vs_best_target.to_numpy(), n_boot, 9000 + gi)
        rows.append({
            "p": int(p),
            "N": int(n),
            "source_kind": str(source_kind),
            "method": str(method),
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


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "p03_results.csv"
    tuning_path = out / "p03_tuning.csv"
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
        sources = _source_truths(target_truth)
        for target_n in args.target_ns:
            for rep in range(args.replications):
                target_seed = 8_000_000 + int(p) * 100_000 + int(target_n) * 1_000 + rep
                rng = np.random.default_rng(target_seed)
                target = _sample(rng, target_truth, int(target_n))
                model_x, tune_x = _split_target(target)
                eval_x = _sample(np.random.default_rng(target_seed + 90_000_000), target_truth, int(args.eval_n))

                target_lw = ledoit_wolf_covariance(model_x, method="TargetLedoitWolf")
                target_ridge, ridge_tuning = _select_target_ridge(model_x, tune_x, tuple(args.ridge_gammas))
                best_target, best_target_tuning = choose_by_healthy_risk([target_lw, target_ridge], tune_x)
                best_target = CovarianceEstimate(
                    best_target.covariance,
                    best_target.precision,
                    "BestTargetOnly",
                    {**best_target.metadata, "selected_base_method": best_target.method},
                )

                target_hash = _sha256(target)
                model_hash = _sha256(model_x)
                tune_hash = _sha256(tune_x)

                for source_index, (source_kind, source_truth) in enumerate(sources.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source_x = _sample(np.random.default_rng(source_seed), source_truth, int(args.source_n))
                    similarity = _truth_similarity(source_truth, target_truth)

                    source_lw = ledoit_wolf_covariance(source_x, method="SourceLedoitWolf")
                    pooled = pooled_ledoit_wolf(model_x, source_x)
                    fixed = race_covariance(model_x, source_x, lambda_reg=60.0, method="RACECov60")
                    adaptive, race_tuning = _select_race(model_x, source_x, tune_x, tuple(args.race_lambdas))
                    adaptive = CovarianceEstimate(
                        adaptive.covariance,
                        adaptive.precision,
                        "RACECovAdaptive",
                        {**adaptive.metadata, "selected_candidate": adaptive.method},
                    )

                    estimates = [target_lw, target_ridge, best_target, source_lw, pooled, fixed, adaptive]
                    for est in estimates:
                        method = est.method
                        key = (int(p), int(target_n), int(rep), source_kind, method)
                        if key in completed:
                            continue
                        row = {
                            "p": int(p),
                            "target_n": int(target_n),
                            "replication": int(rep),
                            "source_kind": source_kind,
                            "method": method,
                            "target_seed": int(target_seed),
                            "source_seed": int(source_seed),
                            "target_sample_sha256": target_hash,
                            "model_pool_sha256": model_hash,
                            "tune_sha256": tune_hash,
                            "model_pool_size": int(len(model_x)),
                            "tune_size": int(len(tune_x)),
                            "selected_base_method": str(est.metadata.get("selected_base_method", "")),
                            "lambda_reg": float(est.metadata.get("lambda_reg", np.nan)),
                            "source_weight": float(est.metadata.get("source_weight", np.nan)),
                            **similarity,
                            **_metrics(est, target_truth, eval_x),
                        }
                        pd.DataFrame([row]).to_csv(
                            results_path,
                            mode="a",
                            header=not results_path.exists() or results_path.stat().st_size == 0,
                            index=False,
                        )
                        completed.add(key)

                    for label, tuning_rows in (
                        ("TargetRidge", ridge_tuning),
                        ("BestTargetOnly", best_target_tuning),
                        ("RACECovAdaptive", race_tuning),
                    ):
                        frame = pd.DataFrame(tuning_rows)
                        frame.insert(0, "method_family", label)
                        frame.insert(0, "source_kind", source_kind)
                        frame.insert(0, "replication", rep)
                        frame.insert(0, "target_n", target_n)
                        frame.insert(0, "p", p)
                        frame.to_csv(
                            tuning_path,
                            mode="a",
                            header=not tuning_path.exists() or tuning_path.stat().st_size == 0,
                            index=False,
                        )

                    print(
                        f"P0.3 p={p} N={target_n} rep={rep+1}/{args.replications} source={source_kind}",
                        flush=True,
                    )

    results = pd.read_csv(results_path)
    pairing = results.groupby(["p", "target_n", "replication"]).agg(
        source_kinds=("source_kind", "nunique"),
        target_seeds=("target_seed", "nunique"),
        target_hashes=("target_sample_sha256", "nunique"),
        model_pool_hashes=("model_pool_sha256", "nunique"),
        tune_hashes=("tune_sha256", "nunique"),
        model_pool_sizes=("model_pool_size", "nunique"),
    ).reset_index()
    expected_sources = len(_source_truths(_precision_chain(int(args.dimensions[0]))))
    if not (
        (pairing.source_kinds == expected_sources).all()
        and (pairing.target_seeds == 1).all()
        and (pairing.target_hashes == 1).all()
        and (pairing.model_pool_hashes == 1).all()
        and (pairing.tune_hashes == 1).all()
        and (pairing.model_pool_sizes == 1).all()
    ):
        raise RuntimeError("P0.3 pairing/data-budget audit failed")
    pairing.to_csv(out / "p03_pairing_audit.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p03_paired_summary.csv", index=False)

    # Similarity trend for adaptive RACE-Cov only.
    adaptive = results[results.method == "RACECovAdaptive"].copy()
    base = results[results.method == "BestTargetOnly"][
        ["p", "target_n", "replication", "source_kind", "covariance_relative_frobenius"]
    ].rename(columns={"covariance_relative_frobenius": "base_error"})
    adaptive = adaptive.merge(
        base,
        on=["p", "target_n", "replication", "source_kind"],
        validate="one_to_one",
    )
    adaptive["gain"] = (
        adaptive.base_error - adaptive.covariance_relative_frobenius
    ) / adaptive.base_error
    trend_rows = []
    for (p, n), g in adaptive.groupby(["p", "target_n"], sort=True):
        sim = -g.source_target_covariance_relative_frobenius.to_numpy(dtype=float)
        gain = g.gain.to_numpy(dtype=float)
        corr = float(np.corrcoef(sim, gain)[0, 1]) if np.std(sim) > 0 and np.std(gain) > 0 else np.nan
        trend_rows.append({
            "p": int(p),
            "N": int(n),
            "pearson_gain_vs_negative_covariance_truth_distance": corr,
        })
    pd.DataFrame(trend_rows).to_csv(out / "p03_similarity_trend.csv", index=False)

    # Healthy-only adaptive source usage audit.
    usage = results[results.method == "RACECovAdaptive"].groupby(
        ["p", "target_n", "source_kind"], sort=True
    ).agg(
        replications=("replication", "nunique"),
        median_lambda=("lambda_reg", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_zero_source_weight=("source_weight", lambda s: float(np.mean(np.asarray(s) <= 1e-12))),
    ).reset_index()
    usage.to_csv(out / "p03_source_weight_audit.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions],
        "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n),
        "replications": int(args.replications),
        "eval_n": int(args.eval_n),
        "race_lambda_grid": [float(v) for v in args.race_lambdas],
        "ridge_gamma_grid": [float(v) for v in args.ridge_gammas],
        "source_regimes": list(_source_truths(_precision_chain(int(args.dimensions[0]))).keys()),
        "primary_comparison": "RACECovAdaptive vs BestTargetOnly on covariance relative Frobenius error",
        "truth_used_for_tuning": False,
        "negative_transfer_definition": "gain_vs_BestTargetOnly < -0.10",
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (out / "p03_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.3 outputs written to {out}", flush=True)


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
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if any(v < 2 for v in args.dimensions): parser.error("dimensions must be >=2")
    if any(v < 10 for v in args.target_ns): parser.error("target-ns must be >=10")
    if args.source_n < 10: parser.error("source-n must be >=10")
    if args.replications <= 0: parser.error("replications must be positive")
    if args.eval_n < 50: parser.error("eval-n must be >=50")
    if any(v <= 0 or v > 1 for v in args.ridge_gammas): parser.error("ridge gammas must be in (0,1]")
    if any(v < 0 for v in args.race_lambdas): parser.error("race lambdas must be non-negative")
    if 0.0 not in [float(v) for v in args.race_lambdas]: parser.error("race lambda grid must include 0 for source rejection")
    return args


if __name__ == "__main__":
    run(parse_args())
