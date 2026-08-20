"""P0.2A+: distinguish genuine precision transfer from generic regularization.

This healthy/synthetic-only benchmark is intentionally small and paired.  It
uses the same target sample, target split, tuning set, evaluation set, lambda
grid, and cross-fit seed across all source regimes within each
(p, target_n, replication).  Only the source precision/data change.

Questions
---------
1. Does transfer beat TargetCLIME when the source is genuinely related?
2. Does it beat strong no-source regularization controls?
3. Does benefit decrease as source truth becomes less compatible?
4. Is harmful negative transfer controlled for adversarial/permuted sources?

Synthetic truth is NEVER used to choose deployable hyperparameters.  It is used
only after fitting to report estimation/structural error and source similarity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precision_transfer_estimators import (
    clime,
    crossfit_trans_clime,
    gaussian_precision_risk,
    reference_trans_clime,
    relative_frobenius_error,
    spd_project,
    support_metrics,
)
from src.reproducibility import reproducibility_metadata

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p02a_transfer_specificity"
PROTOCOL_VERSION = "p02a-transfer-specificity-v1"
DEFAULT_DIMENSIONS = (20,)
DEFAULT_TARGET_NS = (25, 50)
DEFAULT_SOURCE_N = 200
DEFAULT_REPLICATIONS = 50
DEFAULT_LAMBDA_MULTIPLIERS = (0.3, 0.5, 0.8, 1.0, 1.2)
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_BOOTSTRAPS = 10000


@dataclass(frozen=True)
class MatrixCandidate:
    matrix: np.ndarray
    method: str
    tuning_value: float
    metadata: dict[str, float | int | str]


def _precision_chain(p: int, edge: float = 0.22) -> np.ndarray:
    omega = np.eye(p, dtype=np.float64)
    for i in range(p - 1):
        omega[i, i + 1] = omega[i + 1, i] = -float(edge)
    return omega


def _assert_spd(name: str, omega: np.ndarray) -> None:
    eig = np.linalg.eigvalsh(0.5 * (omega + omega.T))
    if float(np.min(eig)) <= 1e-8:
        raise ValueError(f"Source regime {name!r} is not SPD; min eigenvalue={eig.min():.6g}")


def _source_truths(target: np.ndarray, seed: int = 20260819) -> dict[str, np.ndarray]:
    """Create predeclared source regimes spanning similarity and harmful shift."""
    p = target.shape[0]
    regimes: dict[str, np.ndarray] = {}
    regimes["identical"] = target.copy()

    mild = np.eye(p, dtype=np.float64)
    for i in range(p - 1):
        # Same support, modestly perturbed weights.
        edge = 0.18 if i % 2 == 0 else 0.25
        mild[i, i + 1] = mild[i + 1, i] = -edge
    regimes["mild"] = mild

    moderate = np.eye(p, dtype=np.float64)
    # Retain alternating target edges and add disjoint skip-one edges.
    for i in range(0, p - 1, 2):
        moderate[i, i + 1] = moderate[i + 1, i] = -0.22
    for i in range(1, p - 2, 4):
        moderate[i, i + 2] = moderate[i + 2, i] = -0.22
    regimes["moderate"] = moderate

    disjoint = np.eye(p, dtype=np.float64)
    for i in range(0, p - 2, 2):
        disjoint[i, i + 2] = disjoint[i + 2, i] = -0.22
    regimes["disjoint"] = disjoint

    # Same support as target but opposite conditional-dependence signs.
    sign_reversed = np.eye(p, dtype=np.float64)
    for i in range(p - 1):
        sign_reversed[i, i + 1] = sign_reversed[i + 1, i] = +0.22
    regimes["sign_reversed"] = sign_reversed

    # Strong wrong edges with no target support.  Degree <=2 and |edge|=.38,
    # so unit diagonal remains strictly diagonally dominant.
    adversarial = np.eye(p, dtype=np.float64)
    for i in range(0, p - 2, 2):
        adversarial[i, i + 2] = adversarial[i + 2, i] = -0.38
    regimes["adversarial"] = adversarial

    regimes["diagonal"] = np.eye(p, dtype=np.float64)

    # Fixed nontrivial permutation of the genuinely related target graph.
    rng = np.random.default_rng(seed + p)
    perm = rng.permutation(p)
    if np.all(perm == np.arange(p)):
        perm = np.roll(perm, 1)
    regimes["permuted"] = target[np.ix_(perm, perm)].copy()

    for name, omega in regimes.items():
        _assert_spd(name, omega)
    return regimes


def _sample(rng: np.random.Generator, omega: np.ndarray, n: int) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(omega.shape[0]), np.linalg.inv(omega), size=int(n))


def _split_target(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    tune = max(3, int(round(0.20 * n)))
    aggregate = max(3, int(round(0.20 * n)))
    fit_end = n - tune - aggregate
    if fit_end < 4:
        raise ValueError(f"Target N={n} too small for P0.2A+ split")
    return x[:fit_end], x[fit_end:fit_end + aggregate], x[fit_end + aggregate:]


def _lambda_grid(p: int, n_fit: int, multipliers: tuple[float, ...]) -> tuple[float, ...]:
    base = 2.0 * np.sqrt(np.log(max(p, 2)) / max(n_fit, 2))
    return tuple(float(m * base) for m in multipliers)


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _truth_similarity(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    rel = relative_frobenius_error(source, target)
    structural = support_metrics(source, target, threshold=1e-8)
    target_edges = np.triu(np.abs(target) > 1e-8, k=1)
    idx = np.where(target_edges)
    if len(idx[0]) == 0:
        sign_agreement = 1.0
    else:
        sign_agreement = float(np.mean(np.sign(source[idx]) == np.sign(target[idx])))
    return {
        "source_target_truth_relative_frobenius": float(rel),
        "source_target_truth_support_jaccard": float(structural["support_jaccard"]),
        "source_target_truth_sign_agreement": sign_agreement,
    }


def _ridge_candidates(target_fit: np.ndarray, gammas: tuple[float, ...]) -> list[MatrixCandidate]:
    cov = np.cov(target_fit, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)
    p = cov.shape[0]
    mu = max(float(np.trace(cov) / p), 1e-8)
    eye = np.eye(p)
    out: list[MatrixCandidate] = []
    for gamma in gammas:
        g = float(gamma)
        shrunk = (1.0 - g) * cov + g * mu * eye
        precision = np.linalg.inv(shrunk)
        out.append(MatrixCandidate(precision, "TargetRidgePrecision", g, {"ridge_gamma": g}))
    return out


def _ledoit_candidate(target_fit: np.ndarray) -> MatrixCandidate:
    fit = LedoitWolf(assume_centered=False).fit(target_fit)
    return MatrixCandidate(
        matrix=np.asarray(fit.precision_, dtype=np.float64),
        method="TargetLedoitWolfPrecision",
        tuning_value=float(fit.shrinkage_),
        metadata={"ledoit_wolf_shrinkage": float(fit.shrinkage_)},
    )


def _select_matrix(candidates: list[MatrixCandidate], tune: np.ndarray) -> tuple[MatrixCandidate, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best: tuple[float, float, MatrixCandidate] | None = None
    for candidate in candidates:
        projected = spd_project(candidate.matrix).matrix
        risk = gaussian_precision_risk(tune, projected)
        rows.append({"tuning_value": float(candidate.tuning_value), "healthy_risk": float(risk)})
        key = (float(risk), float(candidate.tuning_value), candidate)
        if best is None or key[0] < best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    assert best is not None
    return best[2], rows


def _select_precision(candidates, tune: np.ndarray):
    rows = []
    best = None
    for lam, estimate in candidates:
        risk = gaussian_precision_risk(tune, estimate.spd)
        rows.append({"tuning_value": float(lam), "healthy_risk": float(risk)})
        key = (float(risk), float(lam), estimate)
        if best is None or key[0] < best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    assert best is not None
    return float(best[1]), best[2], rows


def _matrix_metrics(matrix: np.ndarray, truth: np.ndarray, eval_x: np.ndarray) -> dict[str, float]:
    projection = spd_project(matrix)
    support = support_metrics(projection.matrix, truth, threshold=1e-6)
    return {
        "relative_frobenius_spd": relative_frobenius_error(projection.matrix, truth),
        "heldout_gaussian_risk_spd": gaussian_precision_risk(eval_x, projection.matrix),
        "spd_projection_relative_change": float(projection.relative_frobenius_change),
        "lp_success_fraction": 1.0,
        "spd_support_f1": float(support["support_f1"]),
        "spd_support_jaccard": float(support["support_jaccard"]),
    }


def _precision_metrics(estimate, truth: np.ndarray, eval_x: np.ndarray) -> dict[str, float]:
    support = support_metrics(estimate.spd, truth, threshold=1e-6)
    return {
        "relative_frobenius_spd": relative_frobenius_error(estimate.spd, truth),
        "heldout_gaussian_risk_spd": gaussian_precision_risk(eval_x, estimate.spd),
        "spd_projection_relative_change": float(estimate.spd_projection.relative_frobenius_change),
        "lp_success_fraction": float(estimate.lp_diagnostics.success_fraction),
        "spd_support_f1": float(support["support_f1"]),
        "spd_support_jaccard": float(support["support_jaccard"]),
    }


def _bootstrap_median(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = np.median(rng.choice(x, size=len(x), replace=True))
    return float(np.median(x)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    target = results[results.method == "TargetCLIME"][keys + ["relative_frobenius_spd"]].rename(
        columns={"relative_frobenius_spd": "target_error"}
    )
    no_source = results[results.method == "BestNoSource"][keys + ["relative_frobenius_spd"]].rename(
        columns={"relative_frobenius_spd": "best_no_source_error"}
    )
    transfer = results[results.method.isin(["ReferenceTransCLIME", "CrossfitTransCLIME"])].copy()
    paired = transfer.merge(target, on=keys, validate="one_to_one").merge(no_source, on=keys, validate="one_to_one")
    paired["gain_vs_target"] = (paired.target_error - paired.relative_frobenius_spd) / paired.target_error
    paired["gain_vs_best_no_source"] = (
        paired.best_no_source_error - paired.relative_frobenius_spd
    ) / paired.best_no_source_error
    paired["meaningful_negative_transfer"] = paired.gain_vs_target < -0.10

    rows = []
    group_cols = ["p", "target_n", "source_kind", "method"]
    for gi, (key, g) in enumerate(paired.groupby(group_cols, sort=True)):
        p, n, source_kind, method = key
        med_t, lo_t, hi_t = _bootstrap_median(g.gain_vs_target.to_numpy(), n_boot, 42 + gi)
        med_n, lo_n, hi_n = _bootstrap_median(g.gain_vs_best_no_source.to_numpy(), n_boot, 4200 + gi)
        rows.append({
            "p": int(p),
            "N": int(n),
            "source_kind": str(source_kind),
            "method": str(method),
            "replications": int(len(g)),
            "median_gain_vs_target": med_t,
            "gain_vs_target_ci_lower": lo_t,
            "gain_vs_target_ci_upper": hi_t,
            "fraction_better_than_target": float(np.mean(g.gain_vs_target > 0)),
            "meaningful_negative_transfer_fraction": float(np.mean(g.meaningful_negative_transfer)),
            "median_gain_vs_best_no_source": med_n,
            "gain_vs_best_no_source_ci_lower": lo_n,
            "gain_vs_best_no_source_ci_upper": hi_n,
            "fraction_better_than_best_no_source": float(np.mean(g.gain_vs_best_no_source > 0)),
            "source_target_truth_relative_frobenius": float(g.source_target_truth_relative_frobenius.iloc[0]),
            "source_target_truth_support_jaccard": float(g.source_target_truth_support_jaccard.iloc[0]),
            "source_target_truth_sign_agreement": float(g.source_target_truth_sign_agreement.iloc[0]),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "p02a_specificity_results.csv"
    tuning_path = out / "p02a_specificity_tuning.csv"
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
            for rep in range(args.replications):
                target_seed = 7_000_000 + int(p) * 100_000 + int(target_n) * 1_000 + rep
                target_rng = np.random.default_rng(target_seed)
                target = _sample(target_rng, target_truth, int(target_n))
                target_fit, target_agg, target_tune = _split_target(target)
                eval_rng = np.random.default_rng(target_seed + 91_000_000)
                target_eval = _sample(eval_rng, target_truth, int(args.eval_n))
                lambdas = _lambda_grid(int(p), len(target_fit), tuple(args.lambda_multipliers))
                target_hash = _sha256(target)

                target_candidates = [(lam, clime(target_fit, lam=lam)) for lam in lambdas]
                target_lambda, target_est, target_tuning = _select_precision(target_candidates, target_tune)

                ridge, ridge_tuning = _select_matrix(
                    _ridge_candidates(target_fit, tuple(args.ridge_gammas)), target_tune
                )
                lw = _ledoit_candidate(target_fit)
                no_source_candidates = [ridge, lw]
                best_no_source, no_source_tuning = _select_matrix(no_source_candidates, target_tune)

                for source_index, (source_kind, source_truth) in enumerate(source_truths.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source_rng = np.random.default_rng(source_seed)
                    source = _sample(source_rng, source_truth, int(args.source_n))
                    similarity = _truth_similarity(source_truth, target_truth)
                    combined = np.vstack((target_fit, target_agg))
                    folds = max(2, min(5, len(combined) // 2))

                    trans_ref = []
                    trans_cf = []
                    for lam in lambdas:
                        trans_ref.append((lam, reference_trans_clime(
                            target_fit, target_agg, source, target_lambda=lam, transfer_lambda_const=1.0
                        )))
                        trans_cf.append((lam, crossfit_trans_clime(
                            combined, source, target_lambda=lam, n_folds=folds, seed=target_seed
                        )))
                    ref_lambda, ref_est, ref_tuning = _select_precision(trans_ref, target_tune)
                    cf_lambda, cf_est, cf_tuning = _select_precision(trans_cf, target_tune)

                    rows = [
                        ("TargetCLIME", target_est, target_lambda, _precision_metrics(target_est, target_truth, target_eval)),
                        ("TargetRidgePrecision", ridge, ridge.tuning_value, _matrix_metrics(ridge.matrix, target_truth, target_eval)),
                        ("TargetLedoitWolfPrecision", lw, lw.tuning_value, _matrix_metrics(lw.matrix, target_truth, target_eval)),
                        ("BestNoSource", best_no_source, best_no_source.tuning_value, _matrix_metrics(best_no_source.matrix, target_truth, target_eval)),
                        ("ReferenceTransCLIME", ref_est, ref_lambda, _precision_metrics(ref_est, target_truth, target_eval)),
                        ("CrossfitTransCLIME", cf_est, cf_lambda, _precision_metrics(cf_est, target_truth, target_eval)),
                    ]
                    for method, obj, tuning_value, metrics in rows:
                        key = (int(p), int(target_n), int(rep), source_kind, method)
                        if key in completed:
                            continue
                        row = {
                            "p": int(p), "target_n": int(target_n), "replication": int(rep),
                            "source_kind": source_kind, "method": method,
                            "target_seed": int(target_seed), "source_seed": int(source_seed),
                            "target_sample_sha256": target_hash,
                            "selected_tuning_value": float(tuning_value),
                            **similarity, **metrics,
                        }
                        pd.DataFrame([row]).to_csv(results_path, mode="a", header=not results_path.exists() or results_path.stat().st_size == 0, index=False)
                        completed.add(key)

                    tuning_sets = {
                        "TargetCLIME": target_tuning,
                        "TargetRidgePrecision": ridge_tuning,
                        "BestNoSource": no_source_tuning,
                        "ReferenceTransCLIME": ref_tuning,
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

                    print(f"P0.2A+ p={p} N={target_n} rep={rep+1}/{args.replications} source={source_kind}", flush=True)

    results = pd.read_csv(results_path)

    # Pairing audit: every source regime for a target replication must share one target hash/seed.
    pairing = results.groupby(["p", "target_n", "replication"]).agg(
        source_kinds=("source_kind", "nunique"),
        target_seeds=("target_seed", "nunique"),
        target_hashes=("target_sample_sha256", "nunique"),
    ).reset_index()
    expected_sources = len(_source_truths(_precision_chain(int(args.dimensions[0]))))
    if not ((pairing.target_seeds == 1).all() and (pairing.target_hashes == 1).all() and (pairing.source_kinds == expected_sources).all()):
        raise RuntimeError("Pairing audit failed; refusing to summarize transfer specificity")
    pairing.to_csv(out / "p02a_specificity_pairing_audit.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p02a_specificity_paired_summary.csv", index=False)

    # Source-similarity monotonicity diagnostic using per-replication cross-fit gains.
    keys = ["p", "target_n", "replication", "source_kind"]
    target = results[results.method == "TargetCLIME"][keys + ["relative_frobenius_spd"]].rename(columns={"relative_frobenius_spd": "target_error"})
    cf = results[results.method == "CrossfitTransCLIME"].merge(target, on=keys, validate="one_to_one")
    cf["gain"] = (cf.target_error - cf.relative_frobenius_spd) / cf.target_error
    mono_rows = []
    for (p, n), g in cf.groupby(["p", "target_n"], sort=True):
        # Similarity increases as truth distance decreases.
        sim = -g.source_target_truth_relative_frobenius.to_numpy(dtype=float)
        gain = g.gain.to_numpy(dtype=float)
        pearson = float(np.corrcoef(sim, gain)[0, 1]) if np.std(sim) > 0 and np.std(gain) > 0 else np.nan
        mono_rows.append({"p": int(p), "N": int(n), "pearson_gain_vs_negative_truth_distance": pearson})
    pd.DataFrame(mono_rows).to_csv(out / "p02a_specificity_similarity_trend.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions],
        "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n),
        "replications": int(args.replications),
        "eval_n": int(args.eval_n),
        "source_regimes": list(_source_truths(_precision_chain(int(args.dimensions[0]))).keys()),
        "methods": ["TargetCLIME", "TargetRidgePrecision", "TargetLedoitWolfPrecision", "BestNoSource", "ReferenceTransCLIME", "CrossfitTransCLIME"],
        "truth_used_for_tuning": False,
        "negative_transfer_definition": "gain_vs_TargetCLIME < -0.10",
        "crossfit_label": "COLDSTART extension; not published Trans-CLIME",
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (out / "p02a_specificity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.2A+ outputs written to {out}", flush=True)


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
