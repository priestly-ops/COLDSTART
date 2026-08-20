"""Calibrate a genuinely harmful P0.3c source regime on development seeds.

This script does NOT tune RACE. It audits a predeclared, ordered family of
synthetic source-mismatch regimes and selects the *least severe* regime that is
actually harmful in finite samples against the strong target-only baseline.

Why this exists
---------------
The original P0.3c sign-flip source is far from the target in population
covariance, yet at p >> N its extra samples can still reduce estimation
variance enough to help. A valid negative-transfer stress therefore has to be
verified at the estimator/risk level, not by population distance alone.

Calibration/evaluation separation
----------------------------------
Only development seeds in the 95,000,000 range are used here. The final P0.3c
stress experiment must use different seeds. Synthetic truth is used only for
this benchmark-design audit and never by RACE/Safe-CV.
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
    STATISTICS_PER_SIGNAL,
    T_DF,
    _make_robotics_covariance,
    _sample_multivariate_t,
)
from src.covariance_transfer_estimators import race_covariance, safe_cv_target_only
from src.precision_transfer_estimators import relative_frobenius_error

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p03c_harmful_regime_calibration"
DEFAULT_LAMBDAS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0)


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _signal_permutation(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_signals = int(np.ceil(p / STATISTICS_PER_SIGNAL))
    order = np.arange(n_signals)
    rng.shuffle(order)
    idx: list[int] = []
    for s in order:
        idx.extend(range(s * STATISTICS_PER_SIGNAL, min((s + 1) * STATISTICS_PER_SIGNAL, p)))
    idx = idx[:p]
    if len(idx) != p or len(set(idx)) != p:
        return rng.permutation(p)
    return np.asarray(idx, dtype=int)


def _congruence_with_signal_gains(
    cov: np.ndarray,
    low: float,
    high: float,
    *,
    sign_flip: bool,
) -> np.ndarray:
    p = cov.shape[0]
    diag = np.empty(p, dtype=float)
    for j in range(p):
        signal = j // STATISTICS_PER_SIGNAL
        gain = low if signal % 2 == 0 else high
        sign = -1.0 if sign_flip and signal % 3 == 1 else 1.0
        diag[j] = gain * sign
    D = np.diag(diag)
    return _sym(D @ cov @ D)


def _candidate_regimes(target_cov: np.ndarray, seed: int) -> list[tuple[str, np.ndarray, dict[str, object]]]:
    """Predeclared severity order; first valid harmful regime will be frozen."""
    p = target_cov.shape[0]

    signs = np.array([
        -1.0 if (j // STATISTICS_PER_SIGNAL) % 2 else 1.0
        for j in range(p)
    ])
    Dsign = np.diag(signs)
    sign_flip = _sym(Dsign @ target_cov @ Dsign)

    gain_2x = _congruence_with_signal_gains(
        target_cov, 0.50, 2.00, sign_flip=True
    )
    gain_4x = _congruence_with_signal_gains(
        target_cov, 0.25, 4.00, sign_flip=True
    )

    idx = _signal_permutation(p, seed + 17)
    permuted = target_cov[np.ix_(idx, idx)]
    perm_gain_4x = _congruence_with_signal_gains(
        permuted, 0.25, 4.00, sign_flip=True
    )
    perm_gain_6x = _congruence_with_signal_gains(
        permuted, 1.0 / 6.0, 6.0, sign_flip=True
    )

    candidates = [
        ("sign_flip", sign_flip, {"family": "sign_flip", "severity_rank": 1}),
        ("gain_shift_2x", gain_2x, {"family": "alternating_signal_gain", "low": 0.50, "high": 2.00, "sign_flip": True, "severity_rank": 2}),
        ("gain_shift_4x", gain_4x, {"family": "alternating_signal_gain", "low": 0.25, "high": 4.00, "sign_flip": True, "severity_rank": 3}),
        ("permuted_gain_4x", perm_gain_4x, {"family": "signal_permutation_plus_gain", "low": 0.25, "high": 4.00, "sign_flip": True, "severity_rank": 4, "permutation_seed": seed + 17}),
        ("permuted_gain_6x", perm_gain_6x, {"family": "signal_permutation_plus_gain", "low": 1.0 / 6.0, "high": 6.00, "sign_flip": True, "severity_rank": 5, "permutation_seed": seed + 17}),
    ]

    for name, cov, _ in candidates:
        eig = np.linalg.eigvalsh(cov)
        if eig.min() <= 1e-10:
            raise RuntimeError(f"candidate {name} is not SPD: min eig={eig.min()}")
    return candidates


def run(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for p in args.dimensions:
        target_cov = _make_robotics_covariance(int(p))
        candidates = _candidate_regimes(target_cov, seed=20260820 + int(p))

        for n in args.target_ns:
            for rep in range(args.replications):
                # Development-only seed range; deliberately disjoint from P0.3c.
                target_seed = 95_000_000 + int(p) * 100_000 + int(n) * 1_000 + rep
                target = _sample_multivariate_t(
                    np.random.default_rng(target_seed), target_cov, int(n), args.t_df
                )
                best_target, _ = safe_cv_target_only(
                    target,
                    ridge_gammas=tuple(args.ridge_gammas),
                    n_folds=min(5, int(n)),
                    seed=target_seed + 711,
                )
                base_error = float(relative_frobenius_error(best_target.covariance, target_cov))

                for ci, (name, source_cov, meta) in enumerate(candidates):
                    source_seed = target_seed + 20_000_000 + ci * 100_000
                    source = _sample_multivariate_t(
                        np.random.default_rng(source_seed), source_cov, int(args.source_n), args.t_df
                    )
                    candidate_rows = []
                    for lam in args.lambdas:
                        est = race_covariance(target, source, lambda_reg=float(lam), method="OracleAudit")
                        err = float(relative_frobenius_error(est.covariance, target_cov))
                        gain = (base_error - err) / base_error
                        candidate_rows.append((gain, float(lam), err))
                    # Oracle chooses the best positive source weight. This is an
                    # intentionally generous test for the source regime.
                    best_gain, best_lam, best_err = max(candidate_rows, key=lambda x: x[0])
                    rows.append({
                        "p": int(p),
                        "N": int(n),
                        "replication": int(rep),
                        "candidate": name,
                        "severity_rank": int(meta["severity_rank"]),
                        "target_error": base_error,
                        "oracle_best_lambda": best_lam,
                        "oracle_best_error": best_err,
                        "oracle_best_gain_vs_target": best_gain,
                        "oracle_prefers_source": bool(best_gain > 0.0),
                        "source_target_covariance_relative_frobenius": float(relative_frobenius_error(source_cov, target_cov)),
                    })

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "p03c_harmful_regime_calibration_results.csv", index=False)

    summary = raw.groupby(["p", "N", "candidate", "severity_rank"], sort=True).agg(
        replications=("replication", "nunique"),
        source_target_covariance_relative_frobenius=("source_target_covariance_relative_frobenius", "first"),
        median_oracle_best_lambda=("oracle_best_lambda", "median"),
        fraction_oracle_prefers_source=("oracle_prefers_source", "mean"),
        median_oracle_best_gain_vs_target=("oracle_best_gain_vs_target", "median"),
        q25_oracle_best_gain_vs_target=("oracle_best_gain_vs_target", lambda s: float(np.quantile(s, 0.25))),
        q75_oracle_best_gain_vs_target=("oracle_best_gain_vs_target", lambda s: float(np.quantile(s, 0.75))),
    ).reset_index()
    summary["valid_harmful_stress"] = (
        (summary.fraction_oracle_prefers_source <= float(args.max_oracle_prefer_fraction))
        & (summary.median_oracle_best_gain_vs_target <= 0.0)
    )
    summary.to_csv(out / "p03c_harmful_regime_calibration_summary.csv", index=False)

    # Freeze the least severe candidate that passes for every requested p,N.
    candidate_names = (
        summary[["candidate", "severity_rank"]]
        .drop_duplicates()
        .sort_values("severity_rank")
    )
    selected: str | None = None
    selected_rank: int | None = None
    for r in candidate_names.itertuples(index=False):
        g = summary[summary.candidate == r.candidate]
        if len(g) == len(args.dimensions) * len(args.target_ns) and bool(g.valid_harmful_stress.all()):
            selected = str(r.candidate)
            selected_rank = int(r.severity_rank)
            break

    decision = {
        "decision": "P03C_HARMFUL_REGIME_FROZEN" if selected else "P03C_NO_VALID_HARMFUL_REGIME",
        "selected_candidate": selected,
        "selected_severity_rank": selected_rank,
        "selection_rule": "least severe predeclared candidate passing all requested p,N cells",
        "valid_harmful_rule": {
            "fraction_oracle_prefers_source_le": float(args.max_oracle_prefer_fraction),
            "median_oracle_best_gain_vs_target_le": 0.0,
        },
        "calibration_seed_namespace": "95,000,000 development-only; final evaluation must use disjoint seeds",
        "race_tuned_in_this_script": False,
    }
    (out / "p03c_harmful_regime_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dimensions", type=int, nargs="+", default=[128])
    ap.add_argument("--target-ns", type=int, nargs="+", default=[25])
    ap.add_argument("--source-n", type=int, default=200)
    ap.add_argument("--replications", type=int, default=10)
    ap.add_argument("--t-df", type=float, default=T_DF)
    ap.add_argument("--max-oracle-prefer-fraction", type=float, default=0.20)
    ap.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    ap.add_argument("--ridge-gammas", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.70, 1.0])
    args = ap.parse_args()
    if any(n < 25 for n in args.target_ns):
        ap.error("P0.3c transfer stress is only defined for N>=25")
    if not 0.0 <= args.max_oracle_prefer_fraction <= 1.0:
        ap.error("max-oracle-prefer-fraction must be in [0,1]")
    if any(lam <= 0 for lam in args.lambdas):
        ap.error("oracle audit lambdas must be strictly positive")
    return args


if __name__ == "__main__":
    run(parse_args())
