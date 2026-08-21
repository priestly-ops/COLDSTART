"""Audit the reviewer-facing right-censored commissioning bound from P0.7.

This script does not refit a detector and does not touch anomaly scores.  It
consumes the frozen P0.7 outputs and verifies that the manuscript claim

    RACECov: observed B*=175
    TargetOnly: B*>249 (right-censored)

is supported by the prespecified joint CI endpoint and by exhaustion of the
available PRE_B commissioning population under the P0.7 protocol.

The audit deliberately distinguishes the protocol estimator B* from an
unobserved population-level sample complexity.  It therefore never interpolates
or extrapolates a censored method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL_VERSION = "p07-fixed-budget-preb-v1"
FROZEN_BUDGETS = (175, 200, 224, 225, 249)
DEFAULT_TARGET_HEALTHY = 319
DEFAULT_HEALTHY_EVAL = 70
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_FPR_BUDGET = 0.01
DEFAULT_SEEDS = 20
DEFAULT_TARGET_METHOD = "BestTargetOnlySafeCV"
DEFAULT_RACE_METHODS = ("RACECov60", "RACECovSafeCV")


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{name} missing required columns: {sorted(missing)}")


def _is_close(a: float, b: float, atol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol))


def _passes(row: pd.Series, recall_target: float, fpr_budget: float) -> bool:
    return bool(
        float(row["recall_ci_lower"]) >= float(recall_target)
        and float(row["fpr_ci_upper"]) <= float(fpr_budget)
    )


def _method_first_pass(
    summary: pd.DataFrame,
    method: str,
    recall_target: float,
    fpr_budget: float,
) -> int | None:
    g = summary[summary["method"].astype(str) == method].sort_values("budget")
    passing = [
        int(r.budget)
        for r in g.itertuples(index=False)
        if float(r.recall_ci_lower) >= recall_target
        and float(r.fpr_ci_upper) <= fpr_budget
    ]
    return min(passing) if passing else None


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return data


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "p07_summary.csv"
    seed_path = output_dir / "p07_seed_results.csv"
    split_path = output_dir / "p07_split_audit.csv"
    bstar_path = output_dir / "p07_b_star.json"

    for path in (summary_path, seed_path, split_path, bstar_path):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = pd.read_csv(summary_path)
    seeds = pd.read_csv(seed_path)
    split = pd.read_csv(split_path)
    bstar = _load_json(bstar_path)

    _require_columns(
        summary,
        {
            "budget", "fit_n", "calibration_n", "method", "seeds",
            "recall_mean", "recall_ci_lower", "recall_ci_upper",
            "fpr_mean", "fpr_ci_lower", "fpr_ci_upper", "success_rate",
        },
        "p07_summary.csv",
    )
    _require_columns(
        seeds,
        {
            "protocol_version", "budget", "fit_n", "calibration_n", "seed",
            "method", "false_positive_rate", "recall",
            "source_ids_sha256", "fit_ids_sha256", "calibration_ids_sha256",
            "normal_eval_ids_sha256", "anomaly_eval_ids_sha256",
        },
        "p07_seed_results.csv",
    )
    _require_columns(
        split,
        {
            "budget", "seed", "source_hashes", "fit_hashes",
            "calibration_hashes", "normal_eval_hashes", "anomaly_eval_hashes",
            "methods",
        },
        "p07_split_audit.csv",
    )

    protocol_versions = set(seeds["protocol_version"].astype(str).unique())
    if protocol_versions != {PROTOCOL_VERSION}:
        raise RuntimeError(
            f"Protocol mismatch: expected {PROTOCOL_VERSION}, got {sorted(protocol_versions)}"
        )

    observed_budgets = tuple(sorted(int(v) for v in summary["budget"].unique()))
    if observed_budgets != FROZEN_BUDGETS:
        raise RuntimeError(
            f"Frozen budget grid mismatch: expected {FROZEN_BUDGETS}, got {observed_budgets}"
        )

    # Every summary allocation must consume exactly the stated total budget.
    if not np.all(
        summary["fit_n"].to_numpy(dtype=int)
        + summary["calibration_n"].to_numpy(dtype=int)
        == summary["budget"].to_numpy(dtype=int)
    ):
        raise RuntimeError("At least one P0.7 allocation does not sum to its budget")

    # P0.7 maximum is imposed by PRE_B exhaustion, not an arbitrary experiment stop.
    maximum_feasible = int(args.target_healthy_total) - int(args.healthy_eval_size)
    if maximum_feasible != max(FROZEN_BUDGETS):
        raise RuntimeError(
            "Dataset-exhaustion boundary changed: "
            f"target_healthy_total={args.target_healthy_total}, "
            f"healthy_eval_size={args.healthy_eval_size}, "
            f"maximum_feasible={maximum_feasible}, expected 249"
        )

    # Reviewer-facing seed count and per-method completeness.
    group_counts = (
        seeds.groupby(["budget", "method"], sort=True)["seed"]
        .nunique()
        .reset_index(name="seed_count")
    )
    bad_counts = group_counts[group_counts["seed_count"] != int(args.expected_seeds)]
    if not bad_counts.empty:
        raise RuntimeError(
            "Incomplete P0.7 seed coverage:\n" + bad_counts.to_string(index=False)
        )

    # All methods at a budget/seed must use exactly the same partition hashes.
    hash_cols = (
        "source_hashes", "fit_hashes", "calibration_hashes",
        "normal_eval_hashes", "anomaly_eval_hashes",
    )
    if not all(bool((split[col] == 1).all()) for col in hash_cols):
        raise RuntimeError("Split audit shows method-dependent memberships")

    target_method = str(args.target_method)
    if target_method not in set(summary["method"].astype(str)):
        raise RuntimeError(f"Target method not found: {target_method}")

    target_first_pass = _method_first_pass(
        summary, target_method, args.recall_target, args.false_alert_budget
    )
    if target_first_pass is not None:
        raise RuntimeError(
            f"TargetOnly is not censored: it first passes at budget {target_first_pass}"
        )

    target_249 = summary[
        (summary["method"].astype(str) == target_method)
        & (summary["budget"].astype(int) == maximum_feasible)
    ]
    if len(target_249) != 1:
        raise RuntimeError("Expected exactly one TargetOnly summary row at budget 249")
    target_249 = target_249.iloc[0]

    if _passes(target_249, args.recall_target, args.false_alert_budget):
        raise RuntimeError("TargetOnly unexpectedly satisfies the joint CI criterion at 249")

    limiting_constraints: list[str] = []
    if float(target_249["recall_ci_lower"]) < args.recall_target:
        limiting_constraints.append("recall_ci")
    if float(target_249["fpr_ci_upper"]) > args.false_alert_budget:
        limiting_constraints.append("fpr_ci")
    if not limiting_constraints:
        raise RuntimeError("Could not identify why TargetOnly fails at 249")

    race_rows: list[dict[str, Any]] = []
    observed_race_bstar: list[int] = []
    for method in args.race_methods:
        method = str(method)
        if method not in set(summary["method"].astype(str)):
            raise RuntimeError(f"RACE method not found: {method}")
        first_pass = _method_first_pass(
            summary, method, args.recall_target, args.false_alert_budget
        )
        race_rows.append({"method": method, "first_passing_budget": first_pass})
        if first_pass is not None:
            observed_race_bstar.append(int(first_pass))

    if not observed_race_bstar:
        raise RuntimeError("No declared RACE method satisfies the joint CI criterion")
    race_bstar = min(observed_race_bstar)
    if race_bstar != 175:
        raise RuntimeError(f"Expected observed RACE B*=175, got {race_bstar}")

    # Cross-check the runner's own B* artifact rather than trusting either file alone.
    target_bstar_text = str(bstar.get(target_method, ""))
    if "249" not in target_bstar_text or "Censored" not in target_bstar_text:
        raise RuntimeError(
            f"p07_b_star.json does not encode TargetOnly censoring at 249: {target_bstar_text!r}"
        )
    for method in args.race_methods:
        if bstar.get(str(method)) != 175:
            raise RuntimeError(
                f"p07_b_star.json disagrees for {method}: {bstar.get(str(method))!r}"
            )

    absolute_lower_bound = maximum_feasible - race_bstar
    relative_lower_bound = 1.0 - (race_bstar / maximum_feasible)

    audit = {
        "audit_version": "p07-censoring-bound-audit-v1",
        "protocol_version": PROTOCOL_VERSION,
        "estimand": "protocol-specific total healthy target commissioning budget B*",
        "criterion": {
            "recall_lower_95_ci_gte": float(args.recall_target),
            "fpr_upper_95_ci_lte": float(args.false_alert_budget),
        },
        "dataset_exhaustion": {
            "target_preb_healthy_total": int(args.target_healthy_total),
            "reserved_healthy_evaluation": int(args.healthy_eval_size),
            "maximum_feasible_commissioning_budget": int(maximum_feasible),
            "reason": "PRE_B healthy population exhausted under frozen P0.7 protocol",
        },
        "race": {
            "observed_b_star": int(race_bstar),
            "methods": race_rows,
        },
        "target_only": {
            "method": target_method,
            "status": "right-censored",
            "display": f">{maximum_feasible}",
            "budget_249": {
                "recall_mean": float(target_249["recall_mean"]),
                "recall_ci_lower": float(target_249["recall_ci_lower"]),
                "recall_ci_upper": float(target_249["recall_ci_upper"]),
                "fpr_mean": float(target_249["fpr_mean"]),
                "fpr_ci_lower": float(target_249["fpr_ci_lower"]),
                "fpr_ci_upper": float(target_249["fpr_ci_upper"]),
                "limiting_constraints": limiting_constraints,
            },
        },
        "reviewer_facing_bound": {
            "absolute_cycles": f">{absolute_lower_bound}",
            "relative_reduction_fraction": f">{relative_lower_bound:.12f}",
            "relative_reduction_percent": f">{100.0 * relative_lower_bound:.6f}",
            "recommended_text": (
                f"Under the frozen P0.7 voraus-AD protocol, RACE satisfies the joint CI "
                f"criterion at a total healthy target budget of {race_bstar} cycles, whereas "
                f"TargetOnly remains unsatisfied at the maximum feasible budget of "
                f"{maximum_feasible} cycles and is therefore right-censored above that "
                f"boundary. This implies a commissioning-data reduction greater than "
                f"{absolute_lower_bound} cycles, or greater than "
                f"{100.0 * relative_lower_bound:.1f}%, relative to the smallest TargetOnly "
                f"requirement consistent with the frozen observations."
            ),
        },
        "prohibitions": [
            "Do not interpolate or extrapolate a numerical TargetOnly B* beyond 249.",
            "Do not describe >249 as an arbitrary largest-tested-point censoring event.",
            "Do not generalize the >29.7% lower bound beyond the frozen P0.7 voraus protocol.",
        ],
    }

    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "p07_censoring_bound_audit.json"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    table = pd.DataFrame(
        [
            {
                "method": str(method),
                "B_star": 175,
                "status": "observed",
                "maximum_feasible_budget": maximum_feasible,
            }
            for method in args.race_methods
        ]
        + [
            {
                "method": target_method,
                "B_star": f">{maximum_feasible}",
                "status": "right-censored",
                "maximum_feasible_budget": maximum_feasible,
            }
        ]
    )
    table.to_csv(audit_dir / "p07_censoring_table.csv", index=False)

    print(json.dumps(audit, indent=2))
    print(f"PASS: wrote {json_path}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/p07_fixed_budget_commissioning"),
    )
    parser.add_argument("--target-healthy-total", type=int, default=DEFAULT_TARGET_HEALTHY)
    parser.add_argument("--healthy-eval-size", type=int, default=DEFAULT_HEALTHY_EVAL)
    parser.add_argument("--expected-seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument("--false-alert-budget", type=float, default=DEFAULT_FPR_BUDGET)
    parser.add_argument("--target-method", default=DEFAULT_TARGET_METHOD)
    parser.add_argument("--race-methods", nargs="+", default=list(DEFAULT_RACE_METHODS))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
