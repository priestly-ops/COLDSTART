"""P0.3c: robotics-shaped stress test for the frozen RACE-Cov Safe-CV policy.

This stage deliberately DOES NOT redesign or retune the estimator.  It asks
whether the P0.3b result survives a harder healthy-data model that resembles
our robot-cycle statistical feature geometry:

* signal-major groups of six statistics per signal,
* strong within-signal correlation,
* block/factor dependence across signals,
* low effective rank plus idiosyncratic noise,
* heterogeneous feature scales,
* mildly heavy-tailed (multivariate-t) healthy observations,
* related and mismatched source covariance regimes,
* p >> N at p in {128, 256}, N in {25, 50}.

The production safety policy is frozen from P0.3b:
    N < 25  -> target-only fallback (no source transfer)
    N >= 25 -> RACECovSafeCV using healthy target CV only.

Synthetic truth is used only for reporting.  No truth/anomaly information is
used for fitting, source gating, or hyperparameter selection.
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
    race_covariance,
    safe_cv_race_covariance,
    safe_cv_target_only,
)
from src.precision_transfer_estimators import gaussian_precision_risk, relative_frobenius_error
from src.reproducibility import reproducibility_metadata

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p03c_robotics_covariance_stress"
PROTOCOL_VERSION = "p03c-robotics-covariance-stress-v1"
DEFAULT_DIMENSIONS = (128, 256)
DEFAULT_TARGET_NS = (25, 50)
DEFAULT_SOURCE_N = 400
DEFAULT_REPLICATIONS = 30
DEFAULT_EVAL_N = 500
DEFAULT_BOOTSTRAPS = 10000
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
STATISTICS_PER_SIGNAL = 6
MIN_TRANSFER_N = 25
T_DF = 8.0


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return cov / np.outer(d, d)


def _make_robotics_covariance(p: int, seed: int = 20260819) -> np.ndarray:
    """Construct a deterministic signal-major, low-effective-rank covariance."""
    if p < 12:
        raise ValueError("p must be >= 12")
    rng = np.random.default_rng(seed + p)
    n_signals = int(np.ceil(p / STATISTICS_PER_SIGNAL))

    # Six statistical features for one robot signal are highly dependent.
    stat_corr = np.array([
        [1.00, 0.46, 0.88, 0.79, 0.79, 0.24],
        [0.46, 1.00, 0.42, 0.36, 0.48, 0.61],
        [0.88, 0.42, 1.00, 0.86, 0.86, 0.20],
        [0.79, 0.36, 0.86, 1.00, 0.73, 0.16],
        [0.79, 0.48, 0.86, 0.73, 1.00, 0.24],
        [0.24, 0.61, 0.20, 0.16, 0.24, 1.00],
    ], dtype=float)
    vals = np.linalg.eigvalsh(stat_corr)
    if vals.min() <= 0:
        raise RuntimeError("statistic correlation template is not SPD")

    # Base block covariance, truncated for p not divisible by six.
    block = np.kron(np.eye(n_signals), stat_corr)[:p, :p]

    # Signals are organized into latent robot subsystems.  Each subsystem factor
    # affects many signal-statistic features and induces low effective rank.
    n_subsystems = max(4, min(12, n_signals // 3))
    loadings = np.zeros((p, n_subsystems + 2), dtype=float)
    for j in range(p):
        signal = j // STATISTICS_PER_SIGNAL
        stat = j % STATISTICS_PER_SIGNAL
        subsystem = signal % n_subsystems
        stat_strength = np.array([0.50, 0.34, 0.48, 0.44, 0.44, 0.28])[stat]
        loadings[j, subsystem] = stat_strength * (0.9 + 0.2 * rng.random())
        loadings[j, -2] = 0.14 * (1.0 + 0.25 * np.sin(signal / 3.0))
        loadings[j, -1] = 0.08 * (1.0 if stat < 5 else -0.7)

    # Feature scales intentionally vary, as robot channels/statistics do.
    signal_scale = 0.75 + 0.55 * (1.0 + np.sin(np.arange(n_signals) * 0.37)) / 2.0
    stat_scale = np.array([1.0, 0.75, 1.0, 0.92, 1.08, 1.35])
    scales = np.array([
        signal_scale[j // STATISTICS_PER_SIGNAL] * stat_scale[j % STATISTICS_PER_SIGNAL]
        for j in range(p)
    ])

    latent = loadings @ loadings.T
    cov = 0.46 * block + latent + np.diag(0.20 + 0.08 * rng.random(p))
    cov = np.diag(scales) @ cov @ np.diag(scales)
    cov = _sym(cov)

    # Normalize median variance near one but preserve heteroscedasticity.
    cov /= float(np.median(np.diag(cov)))
    eig = np.linalg.eigvalsh(cov)
    if eig.min() <= 1e-8:
        raise RuntimeError(f"robotics covariance not SPD: min eig={eig.min()}")
    return cov


def _source_covariances(target_cov: np.ndarray, seed: int = 20260819) -> dict[str, np.ndarray]:
    """Predeclared source regimes from compatible to strongly mismatched."""
    p = target_cov.shape[0]
    rng = np.random.default_rng(seed + 31 * p)
    out: dict[str, np.ndarray] = {"identical": target_cov.copy()}

    # Mild: small per-signal scale drift while retaining correlation structure.
    n_signals = int(np.ceil(p / STATISTICS_PER_SIGNAL))
    sig = 0.96 + 0.08 * rng.random(n_signals)
    scales = np.array([sig[j // STATISTICS_PER_SIGNAL] for j in range(p)])
    out["mild"] = _sym(np.diag(scales) @ target_cov @ np.diag(scales))

    # Moderate: preserve most geometry but attenuate cross-subsystem correlation.
    corr = _cov_to_corr(target_cov)
    moderate_corr = corr.copy()
    for i in range(p):
        si = (i // STATISTICS_PER_SIGNAL) % 4
        for j in range(i + 1, p):
            sj = (j // STATISTICS_PER_SIGNAL) % 4
            if si != sj:
                moderate_corr[i, j] *= 0.60
                moderate_corr[j, i] = moderate_corr[i, j]
    d = np.sqrt(np.diag(target_cov))
    out["moderate"] = _sym(np.diag(d) @ moderate_corr @ np.diag(d))

    # Block mismatch: permute complete six-statistic signal blocks.  Marginal
    # variances remain plausible while signal identities are wrong.
    order = np.arange(n_signals)
    rng.shuffle(order)
    idx: list[int] = []
    for s in order:
        idx.extend(range(s * STATISTICS_PER_SIGNAL, min((s + 1) * STATISTICS_PER_SIGNAL, p)))
    idx = idx[:p]
    if len(idx) != p or len(set(idx)) != p:
        # p may truncate the final signal; use a simple full-feature permutation.
        idx = rng.permutation(p).tolist()
    out["block_mismatch"] = target_cov[np.ix_(idx, idx)].copy()

    # Adversarial: flip signs of a large subset of correlations while retaining
    # the same eigenvalues via D Sigma D, with alternating signal signs.
    signs = np.array([
        -1.0 if (j // STATISTICS_PER_SIGNAL) % 2 else 1.0
        for j in range(p)
    ])
    D = np.diag(signs)
    out["adversarial"] = _sym(D @ target_cov @ D)

    for name, cov in out.items():
        eig = np.linalg.eigvalsh(cov)
        if eig.min() <= 1e-8:
            raise RuntimeError(f"source covariance {name} not SPD: {eig.min()}")
    return out


def _sample_multivariate_t(rng: np.random.Generator, covariance: np.ndarray, n: int, df: float) -> np.ndarray:
    """Sample zero-mean multivariate t with exactly the requested covariance."""
    if df <= 2:
        raise ValueError("df must exceed 2 for finite covariance")
    # If z~N(0, scale) and u~chi2(df), z/sqrt(u/df) has covariance
    # scale*df/(df-2).  Choose scale=cov*(df-2)/df.
    scale = covariance * ((df - 2.0) / df)
    z = rng.multivariate_normal(np.zeros(covariance.shape[0]), scale, size=int(n))
    u = rng.chisquare(df, size=int(n))
    return z / np.sqrt(u / df)[:, None]


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _metrics(est: CovarianceEstimate, truth_cov: np.ndarray, eval_x: np.ndarray) -> dict[str, float]:
    truth_prec = np.linalg.inv(truth_cov)
    est_scores = np.einsum("ni,ij,nj->n", eval_x, est.precision, eval_x)
    oracle_scores = np.einsum("ni,ij,nj->n", eval_x, truth_prec, eval_x)
    est_norm = est_scores / max(float(np.median(est_scores)), 1e-12)
    oracle_norm = oracle_scores / max(float(np.median(oracle_scores)), 1e-12)
    corr = float(np.corrcoef(est_scores, oracle_scores)[0, 1]) if np.std(est_scores) > 0 else np.nan
    return {
        "covariance_relative_frobenius": float(relative_frobenius_error(est.covariance, truth_cov)),
        "precision_relative_frobenius": float(relative_frobenius_error(est.precision, truth_prec)),
        "heldout_gaussian_risk": float(gaussian_precision_risk(eval_x, est.precision)),
        "condition_number": float(np.linalg.cond(est.covariance)),
        "mahalanobis_score_correlation": corr,
        "mahalanobis_normalized_median_abs_error": float(np.median(np.abs(est_norm - oracle_norm))),
    }


def _bootstrap_median(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        vals[i] = np.median(rng.choice(x, size=len(x), replace=True))
    return float(np.median(x)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _paired_summary(results: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    keys = ["p", "target_n", "replication", "source_kind"]
    base = results[results.method == "BestTargetOnlySafeCV"][keys + ["covariance_relative_frobenius"]].rename(
        columns={"covariance_relative_frobenius": "base_error"}
    )
    if base.duplicated(keys).any():
        raise RuntimeError("duplicate target-only baseline rows")
    compared = results[results.method.isin(["RACECov60Full", "RACECovSafeCV"])].merge(
        base, on=keys, validate="many_to_one"
    )
    compared["gain"] = (compared.base_error - compared.covariance_relative_frobenius) / compared.base_error
    compared["meaningful_negative_transfer"] = compared.gain < -0.10
    rows = []
    for gi, ((p, n, source_kind, method), g) in enumerate(
        compared.groupby(["p", "target_n", "source_kind", "method"], sort=True)
    ):
        med, lo, hi = _bootstrap_median(g.gain.to_numpy(), n_boot, 31000 + gi)
        rows.append({
            "p": int(p), "N": int(n), "source_kind": source_kind, "method": method,
            "replications": int(len(g)), "median_gain_vs_best_target": med,
            "gain_ci_lower": lo, "gain_ci_upper": hi,
            "fraction_better_than_best_target": float(np.mean(g.gain > 0)),
            "meaningful_negative_transfer_fraction": float(np.mean(g.meaningful_negative_transfer)),
            "median_covariance_error": float(g.covariance_relative_frobenius.median()),
            "median_precision_error": float(g.precision_relative_frobenius.median()),
            "median_heldout_gaussian_risk": float(g.heldout_gaussian_risk.median()),
            "median_mahalanobis_score_correlation": float(g.mahalanobis_score_correlation.median()),
            "median_mahalanobis_normalized_abs_error": float(g.mahalanobis_normalized_median_abs_error.median()),
            "source_target_covariance_relative_frobenius": float(g.source_target_covariance_relative_frobenius.iloc[0]),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "p03c_results.csv"
    if args.no_resume and results_path.exists():
        results_path.unlink()

    completed: set[tuple[int, int, int, str, str]] = set()
    if results_path.exists() and results_path.stat().st_size:
        old = pd.read_csv(results_path)
        completed = {
            (int(r.p), int(r.target_n), int(r.replication), str(r.source_kind), str(r.method))
            for r in old.itertuples(index=False)
        }

    for p in args.dimensions:
        truth_cov = _make_robotics_covariance(int(p))
        sources = _source_covariances(truth_cov)
        for n in args.target_ns:
            for rep in range(args.replications):
                target_seed = 9_000_000 + int(p) * 100_000 + int(n) * 1_000 + rep
                target = _sample_multivariate_t(np.random.default_rng(target_seed), truth_cov, int(n), args.t_df)
                eval_x = _sample_multivariate_t(
                    np.random.default_rng(target_seed + 91_000_000), truth_cov, int(args.eval_n), args.t_df
                )
                cv_folds = min(5, int(n))
                cv_seed = target_seed + 711
                best_target, _ = safe_cv_target_only(
                    target, ridge_gammas=tuple(args.ridge_gammas), n_folds=cv_folds, seed=cv_seed
                )
                target_hash = _sha256(target)

                for source_index, (source_kind, source_cov) in enumerate(sources.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source = _sample_multivariate_t(
                        np.random.default_rng(source_seed), source_cov, int(args.source_n), args.t_df
                    )
                    fixed = race_covariance(target, source, lambda_reg=60.0, method="RACECov60Full")
                    if int(n) < MIN_TRANSFER_N:
                        safe = CovarianceEstimate(
                            best_target.covariance, best_target.precision, "RACECovSafeCV",
                            {**best_target.metadata, "accepted_transfer": False, "lambda_reg": 0.0,
                             "source_weight": 0.0, "min_transfer_n_fallback": True},
                        )
                    else:
                        sel = safe_cv_race_covariance(
                            target, source, lambdas=tuple(args.race_lambdas), n_folds=cv_folds,
                            seed=cv_seed, se_multiplier=float(args.se_multiplier)
                        )
                        safe = sel.estimate

                    source_distance = float(relative_frobenius_error(source_cov, truth_cov))
                    for est in (best_target, fixed, safe):
                        method = est.method
                        key = (int(p), int(n), int(rep), source_kind, method)
                        if key in completed:
                            continue
                        row = {
                            "p": int(p), "target_n": int(n), "replication": int(rep),
                            "source_kind": source_kind, "method": method,
                            "target_seed": int(target_seed), "source_seed": int(source_seed),
                            "target_sample_sha256": target_hash, "target_size": int(len(target)),
                            "cv_seed": int(cv_seed), "cv_folds": int(cv_folds),
                            "source_target_covariance_relative_frobenius": source_distance,
                            "lambda_reg": float(est.metadata.get("lambda_reg", np.nan)),
                            "source_weight": float(est.metadata.get("source_weight", np.nan)),
                            "accepted_transfer": bool(est.metadata.get("accepted_transfer", False)),
                            **_metrics(est, truth_cov, eval_x),
                        }
                        pd.DataFrame([row]).to_csv(
                            results_path, mode="a",
                            header=not results_path.exists() or results_path.stat().st_size == 0,
                            index=False,
                        )
                        completed.add(key)
                    print(f"P0.3c p={p} N={n} rep={rep+1}/{args.replications} source={source_kind}", flush=True)

    results = pd.read_csv(results_path)
    pairing = results.groupby(["p", "target_n", "replication"]).agg(
        source_kinds=("source_kind", "nunique"), target_seeds=("target_seed", "nunique"),
        target_hashes=("target_sample_sha256", "nunique"), target_sizes=("target_size", "nunique"),
        cv_seeds=("cv_seed", "nunique"), cv_folds=("cv_folds", "nunique"),
    ).reset_index()
    if not ((pairing.source_kinds == 5).all() and (pairing.target_seeds == 1).all()
            and (pairing.target_hashes == 1).all() and (pairing.target_sizes == 1).all()
            and (pairing.cv_seeds == 1).all() and (pairing.cv_folds == 1).all()):
        raise RuntimeError("P0.3c pairing/data-budget audit failed")
    pairing.to_csv(out / "p03c_pairing_audit.csv", index=False)

    summary = _paired_summary(results, int(args.bootstrap_replicates))
    summary.to_csv(out / "p03c_paired_summary.csv", index=False)

    usage = results[results.method == "RACECovSafeCV"].groupby(
        ["p", "target_n", "source_kind"], sort=True
    ).agg(
        replications=("replication", "nunique"), median_lambda=("lambda_reg", "median"),
        median_source_weight=("source_weight", "median"),
        fraction_zero_source_weight=("source_weight", lambda s: float(np.mean(np.asarray(s) <= 1e-12))),
        fraction_transfer_accepted=("accepted_transfer", "mean"),
    ).reset_index()
    usage.to_csv(out / "p03c_source_weight_audit.csv", index=False)

    # Similarity diagnostic: source truth similarity should predict Safe-CV gain.
    keys = ["p", "target_n", "replication", "source_kind"]
    base = results[results.method == "BestTargetOnlySafeCV"][keys + ["covariance_relative_frobenius"]].rename(
        columns={"covariance_relative_frobenius": "base_error"}
    )
    safe = results[results.method == "RACECovSafeCV"].merge(base, on=keys, validate="one_to_one")
    safe["gain"] = (safe.base_error - safe.covariance_relative_frobenius) / safe.base_error
    trends = []
    for (p, n), g in safe.groupby(["p", "target_n"], sort=True):
        sim = -g.source_target_covariance_relative_frobenius.to_numpy(float)
        gain = g.gain.to_numpy(float)
        corr = float(np.corrcoef(sim, gain)[0, 1]) if np.std(sim) > 0 and np.std(gain) > 0 else np.nan
        trends.append({"p": int(p), "N": int(n), "pearson_gain_vs_negative_covariance_truth_distance": corr})
    trend_df = pd.DataFrame(trends)
    trend_df.to_csv(out / "p03c_similarity_trend.csv", index=False)

    # Keep the same frozen gates used in P0.3b scale-up.
    decisions = []
    for (p, n), g in summary.groupby(["p", "N"], sort=True):
        safe_g = g[g.method == "RACECovSafeCV"].set_index("source_kind")
        corr = float(trend_df[(trend_df.p == p) & (trend_df.N == n)].pearson_gain_vs_negative_covariance_truth_distance.iloc[0])
        checks = {
            "identical_ci_lower_gt_zero": bool(safe_g.loc["identical", "gain_ci_lower"] > 0),
            "mild_ci_lower_gt_zero": bool(safe_g.loc["mild", "gain_ci_lower"] > 0),
            "moderate_median_gain_ge_zero": bool(safe_g.loc["moderate", "median_gain_vs_best_target"] >= 0),
            "adversarial_negative_transfer_fraction_le_0_20": bool(safe_g.loc["adversarial", "meaningful_negative_transfer_fraction"] <= 0.20),
            "identical_median_gain_ge_0_15": bool(safe_g.loc["identical", "median_gain_vs_best_target"] >= 0.15),
            "similarity_correlation_gt_0_5_diagnostic": bool(corr > 0.5),
        }
        primary = all(v for k, v in checks.items() if not k.endswith("_diagnostic"))
        decisions.append({
            "p": int(p), "N": int(n), "primary_gate_pass": primary,
            "decision": "P0.3C_PASS_ADVANCE_REAL_HEALTHY" if primary else "P0.3C_HOLD",
            "checks": checks,
            "frozen_thresholds": {
                "meaningful_negative_transfer_gain_threshold": -0.10,
                "max_adversarial_negative_transfer_fraction": 0.20,
                "min_identical_median_gain": 0.15,
                "similarity_correlation_diagnostic": 0.5,
                "min_transfer_n": MIN_TRANSFER_N,
            },
        })
    (out / "p03c_gate_decision.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions], "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n), "replications": int(args.replications), "eval_n": int(args.eval_n),
        "t_df": float(args.t_df), "statistics_per_signal": STATISTICS_PER_SIGNAL,
        "min_transfer_n": MIN_TRANSFER_N, "race_lambda_grid": [float(v) for v in args.race_lambdas],
        "ridge_gamma_grid": [float(v) for v in args.ridge_gammas], "se_multiplier": float(args.se_multiplier),
        "source_regimes": ["identical", "mild", "moderate", "block_mismatch", "adversarial"],
        "truth_used_for_tuning": False,
        "primary_comparison": "RACECovSafeCV vs BestTargetOnlySafeCV on covariance relative Frobenius error",
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (out / "p03c_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.3c outputs written to {out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    p.add_argument("--target-ns", type=int, nargs="+", default=list(DEFAULT_TARGET_NS))
    p.add_argument("--source-n", type=int, default=DEFAULT_SOURCE_N)
    p.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    p.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N)
    p.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    p.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    p.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    p.add_argument("--se-multiplier", type=float, default=1.0)
    p.add_argument("--t-df", type=float, default=T_DF)
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    if any(v < 12 for v in a.dimensions): p.error("dimensions must be >=12")
    if any(v < 10 for v in a.target_ns): p.error("target-ns must be >=10")
    if a.source_n < 10: p.error("source-n must be >=10")
    if a.replications <= 0: p.error("replications must be positive")
    if a.eval_n < 50: p.error("eval-n must be >=50")
    if a.t_df <= 2: p.error("t-df must exceed 2")
    if 0.0 not in [float(v) for v in a.race_lambdas]: p.error("race lambda grid must include 0")
    return a


if __name__ == "__main__":
    run(parse_args())
