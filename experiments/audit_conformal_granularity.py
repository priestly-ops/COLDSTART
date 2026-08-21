"""Exact finite-sample granularity audit for the project's split-conformal rule.

For n healthy calibration scores and false-alert level alpha, COLDSTART uses

    k = ceil((n + 1) * (1 - alpha))

and thresholds at the k-th ordered score, clamping k to n when k > n.

Two distinct sample-size boundaries matter:

1. Requested rank is representable without clamping:
       ceil((n+1)(1-alpha)) <= n
   which is equivalent to
       n >= ceil(1/alpha - 1)   (with exact integer handling below).

2. The deployed threshold can fall strictly below the maximum score:
       ceil((n+1)(1-alpha)) <= n - 1
   which is equivalent to
       n >= ceil(2/alpha - 1).

At alpha=0.01 these boundaries are n=99 and n=199 respectively.  Thus
100--119 calibration cycles do not suffer an infeasible requested coverage
rank, but they *are mathematically forced to use the maximum calibration score*.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def requested_rank(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    # A tiny downward tolerance prevents binary floating-point representations
    # such as 99.00000000000001 from spuriously advancing an integer rank.
    x = (n + 1) * (1.0 - alpha)
    return int(math.ceil(x - 1e-12))


def deployed_rank(n: int, alpha: float) -> int:
    return min(requested_rank(n, alpha), n)


def first_n_without_clamp(alpha: float, search_limit: int = 1_000_000) -> int:
    for n in range(1, search_limit + 1):
        if requested_rank(n, alpha) <= n:
            return n
    raise RuntimeError("search_limit too small")


def first_n_below_max(alpha: float, search_limit: int = 1_000_000) -> int:
    for n in range(1, search_limit + 1):
        if requested_rank(n, alpha) <= n - 1:
            return n
    raise RuntimeError("search_limit too small")


def analytic_without_clamp(alpha: float) -> int:
    # Need alpha*(n+1) >= 1.  Search around the algebraic boundary to avoid
    # relying on floating-point ceil at exact reciprocal values.
    candidate = max(1, int(math.floor(1.0 / alpha - 1.0)) - 2)
    while requested_rank(candidate, alpha) > candidate:
        candidate += 1
    return candidate


def analytic_below_max(alpha: float) -> int:
    # Need alpha*(n+1) >= 2.
    candidate = max(1, int(math.floor(2.0 / alpha - 1.0)) - 2)
    while requested_rank(candidate, alpha) > candidate - 1:
        candidate += 1
    return candidate


def build_table(alphas: list[float], n_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha in alphas:
        first_feasible = analytic_without_clamp(alpha)
        first_nonmax = analytic_below_max(alpha)
        # Defensive brute-force agreement check.
        if first_feasible != first_n_without_clamp(alpha):
            raise RuntimeError("Analytic/no-clamp boundary disagrees with brute force")
        if first_nonmax != first_n_below_max(alpha):
            raise RuntimeError("Analytic/non-max boundary disagrees with brute force")

        for n in n_values:
            k_req = requested_rank(n, alpha)
            k_dep = min(k_req, n)
            rows.append(
                {
                    "alpha": float(alpha),
                    "calibration_n": int(n),
                    "requested_rank": int(k_req),
                    "deployed_rank": int(k_dep),
                    "requires_clamp": bool(k_req > n),
                    "threshold_is_forced_max": bool(k_dep == n),
                    "first_n_without_clamp": int(first_feasible),
                    "first_n_strictly_below_max": int(first_nonmax),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02],
    )
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=[50, 99, 100, 119, 150, 175, 198, 199, 200, 250, 399, 400],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/calibration_granularity"),
    )
    args = parser.parse_args()

    if any(not 0.0 < a < 1.0 for a in args.alphas):
        raise ValueError("all alphas must lie in (0,1)")
    if any(n <= 0 for n in args.n_values):
        raise ValueError("all n-values must be positive")

    table = build_table(list(args.alphas), list(args.n_values))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "conformal_rank_granularity.csv", index=False)

    boundaries = []
    for alpha in args.alphas:
        no_clamp = analytic_without_clamp(alpha)
        nonmax = analytic_below_max(alpha)
        boundaries.append(
            {
                "alpha": float(alpha),
                "first_n_without_clamp": int(no_clamp),
                "first_n_strictly_below_max": int(nonmax),
                "interpretation": (
                    f"At alpha={alpha:g}, n<{no_clamp} requires clamping to the "
                    f"largest calibration score; {no_clamp}<=n<{nonmax} is "
                    f"representable but still forced to the largest score; n>={nonmax} "
                    f"permits a threshold below the maximum."
                ),
            }
        )

    payload = {
        "audit_version": "split-conformal-granularity-v1",
        "rank_rule": "ceil((n+1)*(1-alpha)), clamped to n",
        "propositions": {
            "rank_representability": (
                "The requested finite-sample rank is <= n iff alpha*(n+1) >= 1."
            ),
            "non_maximum_threshold": (
                "The requested finite-sample rank is <= n-1 iff alpha*(n+1) >= 2."
            ),
        },
        "boundaries": boundaries,
        "primary_alpha_0_01": {
            "first_n_without_clamp": analytic_without_clamp(0.01),
            "first_n_strictly_below_max": analytic_below_max(0.01),
            "paper_safe_statement": (
                "For the frozen split-conformal rule at alpha=0.01, calibration "
                "sizes from 99 through 198 are representable without an infeasible "
                "rank once n>=99, yet the deployed order statistic remains the "
                "maximum healthy calibration score. A threshold strictly below the "
                "maximum first becomes possible at n=199."
            ),
        },
    }
    (out / "conformal_granularity_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(table.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
