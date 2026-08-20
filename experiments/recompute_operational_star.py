"""Recompute P0.5 N* or P0.7 B* with boundary-safe comparisons.

Why this exists
---------------
Bootstrap quantiles are floating-point values. A mathematically exact boundary
such as FPR=1/100=0.01 can be represented as 0.010000000000000002 and fail a
naive ``<= 0.01`` comparison. This script applies only a tiny numerical
comparison tolerance (default 1e-12); it does not change any metric, threshold,
bootstrap sample, detector, or scientific criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _geq_boundary(value: float, target: float, atol: float) -> bool:
    return bool(value > target or np.isclose(value, target, rtol=0.0, atol=atol))


def _leq_boundary(value: float, target: float, atol: float) -> bool:
    return bool(value < target or np.isclose(value, target, rtol=0.0, atol=atol))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--index-column", choices=["N", "budget"], required=True)
    ap.add_argument("--recall-target", type=float, default=0.90)
    ap.add_argument("--fpr-budget", type=float, default=0.01)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    required = {
        args.index_column,
        "method",
        "recall_ci_lower",
        "fpr_ci_upper",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out: dict[str, object] = {}
    audit_rows: list[dict[str, object]] = []
    max_index = int(df[args.index_column].max())

    for method in sorted(df.method.astype(str).unique()):
        g = df[df.method.astype(str) == method].sort_values(args.index_column)
        first = None
        for r in g.itertuples(index=False):
            idx = int(getattr(r, args.index_column))
            recall_lo = float(r.recall_ci_lower)
            fpr_hi = float(r.fpr_ci_upper)
            recall_pass = _geq_boundary(recall_lo, args.recall_target, args.atol)
            fpr_pass = _leq_boundary(fpr_hi, args.fpr_budget, args.atol)
            audit_rows.append({
                "method": method,
                args.index_column: idx,
                "recall_ci_lower": recall_lo,
                "fpr_ci_upper": fpr_hi,
                "recall_pass": recall_pass,
                "fpr_pass": fpr_pass,
                "joint_pass": bool(recall_pass and fpr_pass),
                "fpr_minus_budget": fpr_hi - args.fpr_budget,
            })
            if first is None and recall_pass and fpr_pass:
                first = idx
        out[method] = first if first is not None else f"Censored (>{max_index})"

    output = args.output
    if output is None:
        output = args.summary.with_name(
            "boundary_safe_star.json"
        )
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    audit_path = output.with_name(output.stem + "_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print(json.dumps(out, indent=2))
    print(f"\nWrote: {output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
