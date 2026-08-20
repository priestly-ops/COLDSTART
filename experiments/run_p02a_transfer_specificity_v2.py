"""P0.2A+ v2: source-specific precision transfer with matched target-data budgets.

This runner fixes the key confound discovered in v1: CrossfitTransCLIME used
(target_fit + target_aggregate) while target-only controls used target_fit only.
Here every matched comparison receives the same target model pool and the same
external healthy tuning/evaluation data.

Primary comparisons
-------------------
1. ReferenceTransCLIME vs ReferenceTargetCLIME (published-style split fidelity).
2. CrossfitTransCLIME vs CrossfitTargetCLIME (same target pool/folds).
3. CrossfitTransCLIME vs BestMatchedTargetOnly (strongest deployable target-only
   control selected from matched CLIME/crossfit-CLIME/ridge/Ledoit-Wolf using
   healthy tuning data only).

Synthetic truth is used only after fitting for evaluation/source-similarity
metrics; never for tuning or method selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_p02a_transfer_specificity import (
    MatrixCandidate,
    _bootstrap_median,
    _lambda_grid,
    _ledoit_candidate,
    _matrix_metrics,
    _precision_chain,
    _precision_metrics,
    _ridge_candidates,
    _sample,
    _select_matrix,
    _select_precision,
    _sha256,
    _source_truths,
    _split_target,
    _truth_similarity,
)
from src.precision_transfer_estimators import (
    clime,
    crossfit_trans_clime,
    reference_trans_clime,
    spd_project,
    symmetrize,
)
from src.reproducibility import reproducibility_metadata

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p02a_transfer_specificity_v2"
PROTOCOL_VERSION = "p02a-transfer-specificity-v2-matched-target-budget"
DEFAULT_DIMENSIONS = (20,)
DEFAULT_TARGET_NS = (25, 50)
DEFAULT_SOURCE_N = 200
DEFAULT_REPLICATIONS = 50
DEFAULT_LAMBDA_MULTIPLIERS = (0.3, 0.5, 0.8, 1.0, 1.2)
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_BOOTSTRAPS = 10000


def _fold_indices(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds < 2 or n_folds > n:
        raise ValueError("n_folds must be in [2, n]")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [np.asarray(v, dtype=np.int64) for v in np.array_split(order, n_folds) if len(v) >= 2]


def _crossfit_target_clime_candidate(
    model_pool: np.ndarray,
    *,
    lam: float,
    n_folds: int,
    seed: int,
) -> MatrixCandidate:
    """Target-only crossfit control using the same target pool/folds as transfer.

    Each fold fits CLIME on the complement. Fold raw precision estimates are
    averaged, symmetrized, and SPD-projected. No source or anomaly information
    is used.
    """
    folds = _fold_indices(len(model_pool), n_folds, seed)
    all_idx = np.arange(len(model_pool))
    raw_estimates: list[np.ndarray] = []
    for validation_idx in folds:
        fit_idx = np.setdiff1d(all_idx, validation_idx, assume_unique=True)
        if len(fit_idx) < 2:
            continue
        raw_estimates.append(clime(model_pool[fit_idx], lam=float(lam)).raw)
    if not raw_estimates:
        raise RuntimeError("No valid target-only crossfit folds")
    raw = np.mean(np.stack(raw_estimates, axis=0), axis=0)
    matrix = spd_project(symmetrize(raw)).matrix
    return MatrixCandidate(
        matrix=matrix,
        method="CrossfitTargetCLIME",
        tuning_value=float(lam),
        metadata={"lambda": float(lam), "n_folds": int(len(raw_estimates))},
    )


def _as_matrix_candidate(estimate, method: str, tuning_value: float) -> MatrixCandidate:
    return MatrixCandidate(
        matrix=np.asarray(estimate.spd, dtype=np.float64),
        method=method,
        tuning_value=float(tuning_value),
        metadata={"tuning_value": float(tuning_value)},
    )


def _baseline_maps(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["p", "target_n", "replication", "source_kind"]
    def one(method: str, column: str) -> pd.DataFrame:
        x = results[results.method == method][keys + ["relative_frobenius_spd"]].rename(
            columns={"relative_frobenius_spd": column}
        )
        if x.duplicated(keys).any():
            raise RuntimeError(f"Duplicate {method} rows detected")
        return x
    return (
        one("ReferenceTargetCLIME", "reference_target_error"),
        one("CrossfitTargetCLIME", "crossfit_target_error"),
        one("BestMatchedTargetOnly", "best_matched_target_error"),
    )


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    ref_target, cf_target, best_target = _baseline_maps(results)
    transfer = results[results.method.isin(["ReferenceTransCLIME", "CrossfitTransCLIME"])].copy()
    paired = (
        transfer
        .merge(ref_target, on=keys, validate="many_to_one")
        .merge(cf_target, on=keys, validate="many_to_one")
        .merge(best_target, on=keys, validate="many_to_one")
    )
    if len(paired) != len(transfer):
        raise RuntimeError("Pairing changed transfer row count")

    is_ref = paired.method == "ReferenceTransCLIME"
    paired["matched_target_error"] = np.where(
        is_ref, paired.reference_target_error, paired.crossfit_target_error
    )
    paired["gain_vs_matched_target"] = (
        paired.matched_target_error - paired.relative_frobenius_spd
    ) / paired.matched_target_error
    paired["gain_vs_best_matched_target"] = (
        paired.best_matched_target_error - paired.relative_frobenius_spd
    ) / paired.best_matched_target_error
    paired["meaningful_negative_transfer"] = paired.gain_vs_matched_target < -0.10

    rows: list[dict[str, object]] = []
    group_cols = ["p", "target_n", "source_kind", "method"]
    for gi, (key, g) in enumerate(paired.groupby(group_cols, sort=True)):
        p, n, source_kind, method = key
        mt, mt_lo, mt_hi = _bootstrap_median(g.gain_vs_matched_target.to_numpy(), n_boot, 1000 + gi)
        bm, bm_lo, bm_hi = _bootstrap_median(g.gain_vs_best_matched_target.to_numpy(), n_boot, 5000 + gi)
        rows.append({
            "p": int(p), "N": int(n), "source_kind": str(source_kind), "method": str(method),
            "replications": int(len(g)),
            "median_gain_vs_matched_target": mt,
            "gain_vs_matched_target_ci_lower": mt_lo,
            "gain_vs_matched_target_ci_upper": mt_hi,
            "fraction_better_than_matched_target": float(np.mean(g.gain_vs_matched_target > 0)),
            "meaningful_negative_transfer_fraction": float(np.mean(g.meaningful_negative_transfer)),
            "median_gain_vs_best_matched_target": bm,
            "gain_vs_best_matched_target_ci_lower": bm_lo,
            "gain_vs_best_matched_target_ci_upper": bm_hi,
            "fraction_better_than_best_matched_target": float(np.mean(g.gain_vs_best_matched_target > 0)),
            "source_target_truth_relative_frobenius": float(g.source_target_truth_relative_frobenius.iloc[0]),
            "source_target_truth_support_jaccard": float(g.source_target_truth_support_jaccard.iloc[0]),
            "source_target_truth_sign_agreement": float(g.source_target_truth_sign_agreement.iloc[0]),
        })
    return pd.DataFrame(rows)


def _similarity_trend(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    cf_target = results[results.method == "CrossfitTargetCLIME"][keys + ["relative_frobenius_spd"]].rename(
        columns={"relative_frobenius_spd": "crossfit_target_error"}
    )
    cf = results[results.method == "CrossfitTransCLIME"].merge(cf_target, on=keys, validate="one_to_one")
    cf["gain"] = (cf.crossfit_target_error - cf.relative_frobenius_spd) / cf.crossfit_target_error
    rows = []
    for (p, n), g in cf.groupby(["p", "target_n"], sort=True):
        sim = -g.source_target_truth_relative_frobenius.to_numpy(dtype=float)
        gain = g.gain.to_numpy(dtype=float)
        pearson = float(np.corrcoef(sim, gain)[0, 1]) if np.std(sim) > 0 and np.std(gain) > 0 else np.nan
        rows.append({"p": int(p), "N": int(n), "pearson_gain_vs_negative_truth_distance": pearson})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "p02a_specificity_v2_results.csv"
    tuning_path = out / "p02a_specificity_v2_tuning.csv"
    if args.no_resume:
        for path in (results_path, tuning_path):
            if path.exists(): path.unlink()

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
            for rep in range(args.replications):
                target_seed = 17_000_000 + int(p) * 100_000 + int(target_n) * 1_000 + rep
                rng = np.random.default_rng(target_seed)
                target = _sample(rng, target_truth, int(target_n))
                target_fit, target_agg, target_tune = _split_target(target)
                model_pool = np.vstack((target_fit, target_agg))
                eval_rng = np.random.default_rng(target_seed + 91_000_000)
                target_eval = _sample(eval_rng, target_truth, int(args.eval_n))
                lambdas_ref = _lambda_grid(int(p), len(target_fit), tuple(args.lambda_multipliers))
                lambdas_matched = _lambda_grid(int(p), len(model_pool), tuple(args.lambda_multipliers))
                folds = max(2, min(5, len(model_pool) // 2))

                # Published-style target comparator uses target_fit only.
                ref_target_candidates = [(lam, clime(target_fit, lam=lam)) for lam in lambdas_ref]
                ref_target_lambda, ref_target_est, ref_target_tuning = _select_precision(ref_target_candidates, target_tune)

                # Matched controls use exactly the same target model pool as crossfit transfer.
                matched_clime_candidates = [(lam, clime(model_pool, lam=lam)) for lam in lambdas_matched]
                matched_clime_lambda, matched_clime_est, matched_clime_tuning = _select_precision(matched_clime_candidates, target_tune)

                cf_target_candidates = [
                    _crossfit_target_clime_candidate(model_pool, lam=lam, n_folds=folds, seed=target_seed)
                    for lam in lambdas_matched
                ]
                cf_target, cf_target_tuning = _select_matrix(cf_target_candidates, target_tune)

                matched_ridge, matched_ridge_tuning = _select_matrix(
                    _ridge_candidates(model_pool, tuple(args.ridge_gammas)), target_tune
                )
                matched_lw = _ledoit_candidate(model_pool)

                best_family_candidates = [
                    *[_as_matrix_candidate(est, "MatchedTargetCLIME", lam) for lam, est in matched_clime_candidates],
                    *cf_target_candidates,
                    *_ridge_candidates(model_pool, tuple(args.ridge_gammas)),
                    matched_lw,
                ]
                best_matched, best_matched_tuning = _select_matrix(best_family_candidates, target_tune)

                target_hash = _sha256(target)
                model_pool_hash = _sha256(model_pool)
                tune_hash = _sha256(target_tune)

                for source_index, (source_kind, source_truth) in enumerate(source_truths.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source = _sample(np.random.default_rng(source_seed), source_truth, int(args.source_n))
                    similarity = _truth_similarity(source_truth, target_truth)

                    ref_candidates = [
                        (lam, reference_trans_clime(target_fit, target_agg, source, target_lambda=lam, transfer_lambda_const=1.0))
                        for lam in lambdas_ref
                    ]
                    ref_lambda, ref_est, ref_tuning = _select_precision(ref_candidates, target_tune)

                    cf_candidates = [
                        (lam, crossfit_trans_clime(model_pool, source, target_lambda=lam, n_folds=folds, seed=target_seed))
                        for lam in lambdas_matched
                    ]
                    cf_lambda, cf_est, cf_tuning = _select_precision(cf_candidates, target_tune)

                    rows = [
                        ("ReferenceTargetCLIME", ref_target_lambda, _precision_metrics(ref_target_est, target_truth, target_eval), "target_fit"),
                        ("ReferenceTransCLIME", ref_lambda, _precision_metrics(ref_est, target_truth, target_eval), "target_fit+aggregate"),
                        ("MatchedTargetCLIME", matched_clime_lambda, _precision_metrics(matched_clime_est, target_truth, target_eval), "model_pool"),
                        ("CrossfitTargetCLIME", cf_target.tuning_value, _matrix_metrics(cf_target.matrix, target_truth, target_eval), "model_pool"),
                        ("MatchedTargetRidge", matched_ridge.tuning_value, _matrix_metrics(matched_ridge.matrix, target_truth, target_eval), "model_pool"),
                        ("MatchedTargetLedoitWolf", matched_lw.tuning_value, _matrix_metrics(matched_lw.matrix, target_truth, target_eval), "model_pool"),
                        ("BestMatchedTargetOnly", best_matched.tuning_value, _matrix_metrics(best_matched.matrix, target_truth, target_eval), "model_pool"),
                        ("CrossfitTransCLIME", cf_lambda, _precision_metrics(cf_est, target_truth, target_eval), "model_pool"),
                    ]
                    for method, tuning_value, metrics, budget in rows:
                        key = (int(p), int(target_n), int(rep), source_kind, method)
                        if key in completed: continue
                        row = {
                            "p": int(p), "target_n": int(target_n), "replication": int(rep),
                            "source_kind": source_kind, "method": method,
                            "target_seed": int(target_seed), "source_seed": int(source_seed),
                            "target_sample_sha256": target_hash,
                            "model_pool_sha256": model_pool_hash,
                            "tune_sha256": tune_hash,
                            "target_fit_n": int(len(target_fit)),
                            "target_aggregate_n": int(len(target_agg)),
                            "model_pool_n": int(len(model_pool)),
                            "target_tune_n": int(len(target_tune)),
                            "target_budget_label": budget,
                            "selected_tuning_value": float(tuning_value),
                            "best_matched_selected_family": str(best_matched.method),
                            **similarity, **metrics,
                        }
                        pd.DataFrame([row]).to_csv(
                            results_path, mode="a",
                            header=not results_path.exists() or results_path.stat().st_size == 0,
                            index=False,
                        )
                        completed.add(key)

                    tuning_sets = {
                        "ReferenceTargetCLIME": ref_target_tuning,
                        "ReferenceTransCLIME": ref_tuning,
                        "MatchedTargetCLIME": matched_clime_tuning,
                        "CrossfitTargetCLIME": cf_target_tuning,
                        "MatchedTargetRidge": matched_ridge_tuning,
                        "BestMatchedTargetOnly": best_matched_tuning,
                        "CrossfitTransCLIME": cf_tuning,
                    }
                    for method, tuning_rows in tuning_sets.items():
                        frame = pd.DataFrame(tuning_rows)
                        frame.insert(0, "method", method)
                        frame.insert(0, "source_kind", source_kind)
                        frame.insert(0, "replication", rep)
                        frame.insert(0, "target_n", target_n)
                        frame.insert(0, "p", p)
                        frame.to_csv(tuning_path, mode="a", header=not tuning_path.exists() or tuning_path.stat().st_size == 0, index=False)

                    print(f"P0.2A+v2 p={p} N={target_n} rep={rep+1}/{args.replications} source={source_kind}", flush=True)

    results = pd.read_csv(results_path)
    pairing = results.groupby(["p", "target_n", "replication"]).agg(
        source_kinds=("source_kind", "nunique"),
        target_seeds=("target_seed", "nunique"),
        target_hashes=("target_sample_sha256", "nunique"),
        model_pool_hashes=("model_pool_sha256", "nunique"),
        tune_hashes=("tune_sha256", "nunique"),
        model_pool_sizes=("model_pool_n", "nunique"),
    ).reset_index()
    expected_sources = len(_source_truths(_precision_chain(int(args.dimensions[0]))))
    ok = (
        (pairing.source_kinds == expected_sources).all()
        and (pairing.target_seeds == 1).all()
        and (pairing.target_hashes == 1).all()
        and (pairing.model_pool_hashes == 1).all()
        and (pairing.tune_hashes == 1).all()
        and (pairing.model_pool_sizes == 1).all()
    )
    if not ok:
        raise RuntimeError("Matched target-budget pairing audit failed")
    pairing.to_csv(out / "p02a_specificity_v2_pairing_audit.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p02a_specificity_v2_paired_summary.csv", index=False)
    trend = _similarity_trend(results)
    trend.to_csv(out / "p02a_specificity_v2_similarity_trend.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions],
        "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n),
        "replications": int(args.replications),
        "eval_n": int(args.eval_n),
        "truth_used_for_tuning": False,
        "primary_crossfit_comparison": "CrossfitTransCLIME vs CrossfitTargetCLIME with identical model_pool/folds",
        "strong_comparison": "CrossfitTransCLIME vs BestMatchedTargetOnly",
        "negative_transfer_definition": "gain_vs_matched_target < -0.10",
        "source_regimes": list(_source_truths(_precision_chain(int(args.dimensions[0]))),),
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (out / "p02a_specificity_v2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.2A+ v2 outputs written to {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--target-ns", type=int, nargs="+", default=list(DEFAULT_TARGET_NS))
    parser.add_argument("--source-n", type=int, default=DEFAULT_SOURCE_N)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--eval-n", type=int, default=500)
    parser.add_argument("--lambda-multipliers", type=float, nargs="+", default=list(DEFAULT_LAMBDA_MULTIPLIERS))
    parser.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if any(v < 2 for v in args.dimensions): parser.error("dimensions must be >=2")
    if any(v < 10 for v in args.target_ns): parser.error("target-ns must be >=10")
    if args.source_n < 10: parser.error("source-n must be >=10")
    if args.replications <= 0: parser.error("replications must be positive")
    if args.eval_n < 50: parser.error("eval-n must be >=50")
    if any(v <= 0 for v in args.lambda_multipliers): parser.error("lambda multipliers must be positive")
    if any(v <= 0 or v > 1 for v in args.ridge_gammas): parser.error("ridge gammas must be in (0,1]")
    return args


if __name__ == "__main__":
    run(parse_args())
