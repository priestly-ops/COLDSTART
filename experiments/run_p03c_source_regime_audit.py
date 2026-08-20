"""Audit whether P0.3c source regimes are truly beneficial or harmful.

This is a diagnostic for the *benchmark*, not a new estimator.  It uses the
synthetic population covariance only after the frozen estimator has been fit
logic-wise, and therefore must never be used for deployable source selection.

Why this exists
---------------
In p >> n, even a structurally distant source can reduce covariance estimation
variance enough to improve target error.  Calling such a source "adversarial"
would make a negative-transfer safety check misleading.  This audit therefore
quantifies two oracle notions for each source regime:

1. Population blend bias: error of (1-w) Sigma_T + w Sigma_S relative to Sigma_T.
   This is zero only at w=0 unless the source covariance equals the target.
2. Finite-sample oracle transferability: over paired synthetic replications,
   the best lambda *chosen with truth* from the frozen RACE lambda grid.  This
   is reporting only.  If the so-called adversarial source often has a positive
   oracle gain, then it is not a valid harmful-source stress regime for the
   finite-sample benchmark.

No anomaly labels are involved.  Truth is used only for this diagnostic audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_p03c_robotics_covariance_stress import (
    _make_robotics_covariance,
    _sample_multivariate_t,
    _source_covariances,
)
from src.covariance_transfer_estimators import (
    race_covariance,
    safe_cv_target_only,
)
from src.precision_transfer_estimators import relative_frobenius_error

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p03c_source_regime_audit"
DEFAULT_DIMENSIONS = (128, 256)
DEFAULT_TARGET_NS = (25, 50)
DEFAULT_SOURCE_N = 400
DEFAULT_REPLICATIONS = 30
DEFAULT_RACE_LAMBDAS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
DEFAULT_RIDGE_GAMMAS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.0)
DEFAULT_T_DF = 8.0


def _population_blend_rows(target_cov: np.ndarray, source_cov: np.ndarray) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for source_weight in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0):
        blend = (1.0 - source_weight) * target_cov + source_weight * source_cov
        rows.append({
            "source_weight": float(source_weight),
            "population_covariance_relative_frobenius": float(
                relative_frobenius_error(blend, target_cov)
            ),
        })
    return rows


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    population_rows: list[dict[str, object]] = []
    finite_rows: list[dict[str, object]] = []

    for p in args.dimensions:
        truth_cov = _make_robotics_covariance(int(p))
        source_covs = _source_covariances(truth_cov)

        for source_kind, source_cov in source_covs.items():
            source_distance = float(relative_frobenius_error(source_cov, truth_cov))
            for row in _population_blend_rows(truth_cov, source_cov):
                population_rows.append({
                    "p": int(p),
                    "source_kind": source_kind,
                    "source_target_covariance_relative_frobenius": source_distance,
                    **row,
                })

        for n in args.target_ns:
            if n < 25:
                raise ValueError("P0.3c audit is only defined for N>=25 frozen transfer policy")
            for rep in range(args.replications):
                target_seed = 9_000_000 + int(p) * 100_000 + int(n) * 1_000 + rep
                target = _sample_multivariate_t(
                    np.random.default_rng(target_seed), truth_cov, int(n), float(args.t_df)
                )
                cv_seed = target_seed + 711
                target_only, _ = safe_cv_target_only(
                    target,
                    ridge_gammas=tuple(args.ridge_gammas),
                    n_folds=min(5, int(n)),
                    seed=cv_seed,
                )
                base_error = float(relative_frobenius_error(target_only.covariance, truth_cov))

                for source_index, (source_kind, source_cov) in enumerate(source_covs.items()):
                    source_seed = target_seed + 10_000_000 + source_index * 100_000
                    source = _sample_multivariate_t(
                        np.random.default_rng(source_seed),
                        source_cov,
                        int(args.source_n),
                        float(args.t_df),
                    )

                    candidates: list[tuple[float, float, float]] = []
                    for lam in args.race_lambdas:
                        est = race_covariance(
                            target,
                            source,
                            lambda_reg=float(lam),
                            method=f"OracleAudit[lambda={float(lam):g}]",
                        )
                        err = float(relative_frobenius_error(est.covariance, truth_cov))
                        gain = (base_error - err) / max(base_error, 1e-12)
                        candidates.append((err, float(lam), float(gain)))

                    candidates.sort(key=lambda x: (x[0], x[1]))
                    oracle_error, oracle_lambda, oracle_gain = candidates[0]
                    no_source_error = next(err for err, lam, _ in candidates if lam == 0.0)
                    no_source_gain = (base_error - no_source_error) / max(base_error, 1e-12)

                    finite_rows.append({
                        "p": int(p),
                        "N": int(n),
                        "replication": int(rep),
                        "source_kind": source_kind,
                        "source_target_covariance_relative_frobenius": float(
                            relative_frobenius_error(source_cov, truth_cov)
                        ),
                        "best_target_only_error": base_error,
                        "lambda0_error": no_source_error,
                        "lambda0_gain_vs_best_target": float(no_source_gain),
                        "oracle_best_lambda": oracle_lambda,
                        "oracle_best_error": oracle_error,
                        "oracle_best_gain_vs_best_target": oracle_gain,
                        "oracle_prefers_source": bool(oracle_lambda > 0.0),
                        "oracle_meaningful_negative_even_at_best": bool(oracle_gain < -0.10),
                    })

    population = pd.DataFrame(population_rows)
    finite = pd.DataFrame(finite_rows)
    population.to_csv(out / "p03c_population_blend_bias.csv", index=False)
    finite.to_csv(out / "p03c_finite_sample_oracle.csv", index=False)

    summary = finite.groupby(["p", "N", "source_kind"], sort=True).agg(
        replications=("replication", "nunique"),
        source_target_covariance_relative_frobenius=(
            "source_target_covariance_relative_frobenius", "first"
        ),
        median_oracle_best_lambda=("oracle_best_lambda", "median"),
        fraction_oracle_prefers_source=("oracle_prefers_source", "mean"),
        median_oracle_best_gain_vs_best_target=("oracle_best_gain_vs_best_target", "median"),
        q25_oracle_best_gain_vs_best_target=("oracle_best_gain_vs_best_target", lambda s: float(np.quantile(s, 0.25))),
        q75_oracle_best_gain_vs_best_target=("oracle_best_gain_vs_best_target", lambda s: float(np.quantile(s, 0.75))),
    ).reset_index()
    summary["valid_harmful_stress"] = (
        (summary["fraction_oracle_prefers_source"] <= 0.20)
        & (summary["median_oracle_best_gain_vs_best_target"] <= 0.0)
    )
    summary.to_csv(out / "p03c_source_regime_oracle_summary.csv", index=False)

    decisions: list[dict[str, object]] = []
    for (p, n), g in summary.groupby(["p", "N"], sort=True):
        adv = g[g.source_kind == "adversarial"]
        if len(adv) != 1:
            raise RuntimeError("Expected exactly one adversarial summary row")
        row = adv.iloc[0]
        valid = bool(row.valid_harmful_stress)
        decisions.append({
            "p": int(p),
            "N": int(n),
            "adversarial_is_valid_harmful_stress": valid,
            "fraction_oracle_prefers_source": float(row.fraction_oracle_prefers_source),
            "median_oracle_best_gain_vs_best_target": float(row.median_oracle_best_gain_vs_best_target),
            "decision": "KEEP_P03C_ADVERSARIAL" if valid else "REDESIGN_BENCHMARK_ADVERSARIAL_ONLY",
        })

    (out / "p03c_source_regime_audit_decision.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"P0.3c source-regime audit written to {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--target-ns", type=int, nargs="+", default=list(DEFAULT_TARGET_NS))
    parser.add_argument("--source-n", type=int, default=DEFAULT_SOURCE_N)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--race-lambdas", type=float, nargs="+", default=list(DEFAULT_RACE_LAMBDAS))
    parser.add_argument("--ridge-gammas", type=float, nargs="+", default=list(DEFAULT_RIDGE_GAMMAS))
    parser.add_argument("--t-df", type=float, default=DEFAULT_T_DF)
    args = parser.parse_args()
    if any(int(v) < 12 for v in args.dimensions): parser.error("dimensions must be >=12")
    if any(int(v) < 25 for v in args.target_ns): parser.error("target-ns must be >=25")
    if args.source_n < 10: parser.error("source-n must be >=10")
    if args.replications <= 0: parser.error("replications must be positive")
    if 0.0 not in [float(v) for v in args.race_lambdas]: parser.error("race-lambdas must include 0")
    return args


if __name__ == "__main__":
    run(parse_args())
