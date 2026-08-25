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

from src.paired_inference import summarize_paired_deltas


INPUT_PATH = PROJECT_ROOT / "outputs" / "e7_transfer_compatibility" / "e7_paired_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "e7_transfer_compatibility" / "e7_paired_inference.csv"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "e7_transfer_compatibility" / "e7_paired_inference_manifest.json"

PROTOCOL_VERSION = "coldstart-e7-paired-inference-v1"
PRIMARY_DELTA_COLUMN = "delta_oracle_recall_at_fpr_budget"
EXPECTED_N_VALUES = (10, 25, 50, 100)
EXPECTED_SEEDS = tuple(range(20))
GLOBAL_SEED = 42


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"E7 paired results not found: {path}")
    df = pd.read_csv(path)
    required = {"commissioning_size", "seed", PRIMARY_DELTA_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"E7 paired results missing required columns: {missing}")

    if df.duplicated(["commissioning_size", "seed"]).any():
        duplicates = df.loc[
            df.duplicated(["commissioning_size", "seed"], keep=False),
            ["commissioning_size", "seed"],
        ]
        raise ValueError(f"duplicate E7 N/seed pairs detected:\n{duplicates.to_string(index=False)}")

    values = df[PRIMARY_DELTA_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{PRIMARY_DELTA_COLUMN} contains NaN or Inf")
    return df


def _validate_expected_design(df: pd.DataFrame, *, strict: bool) -> None:
    if not strict:
        return
    actual_n = tuple(sorted(int(v) for v in df["commissioning_size"].unique()))
    if actual_n != EXPECTED_N_VALUES:
        raise ValueError(f"expected commissioning sizes {EXPECTED_N_VALUES}, found {actual_n}")

    for n_value, group in df.groupby("commissioning_size", sort=True):
        actual_seeds = tuple(sorted(int(v) for v in group["seed"].unique()))
        if actual_seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"N={int(n_value)} expected seeds 0..19, found {actual_seeds}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process frozen E7 paired results with paired uncertainty and hypothesis tests."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--signflip-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a subset of the frozen N/seed design for smoke tests only.",
    )
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    df = _load_and_validate(args.input)
    _validate_expected_design(df, strict=not args.allow_partial)

    rows: list[dict[str, object]] = []
    for n_value, group in df.groupby("commissioning_size", sort=True):
        deltas = group[PRIMARY_DELTA_COLUMN].to_numpy(dtype=float)
        result = summarize_paired_deltas(
            deltas,
            confidence=0.95,
            bootstrap_repetitions=int(args.bootstrap_repetitions),
            signflip_repetitions=int(args.signflip_repetitions),
            seed=int(args.seed) + int(n_value),
        )
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "commissioning_size": int(n_value),
            "primary_outcome": PRIMARY_DELTA_COLUMN,
            **result.to_dict(),
            "positive_rate": result.positive_count / result.n_pairs,
            "negative_rate": result.negative_count / result.n_pairs,
            "neutral_rate": result.neutral_count / result.n_pairs,
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    _atomic_csv(output, args.output)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "paired inferential post-processing of frozen E7 RACE-vs-TargetOnly transfer outcomes",
        "input": str(args.input.relative_to(PROJECT_ROOT)) if args.input.is_relative_to(PROJECT_ROOT) else str(args.input),
        "output": str(args.output.relative_to(PROJECT_ROOT)) if args.output.is_relative_to(PROJECT_ROOT) else str(args.output),
        "primary_outcome": PRIMARY_DELTA_COLUMN,
        "delta_definition": "RACE minus TargetOnly empirical oracle recall at FPR <= 0.01",
        "grouping": "paired within commissioning size over identical commissioning seeds",
        "confidence": 0.95,
        "bootstrap": {
            "method": "percentile bootstrap over paired seed differences",
            "repetitions": int(args.bootstrap_repetitions),
            "seed": int(args.seed),
        },
        "signflip_test": {
            "alternative": "two-sided",
            "statistic": "absolute paired mean difference",
            "exact_when_nonzero_pairs_leq": 20,
            "monte_carlo_repetitions_if_needed": int(args.signflip_repetitions),
        },
        "wilcoxon": {
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "method": "auto",
        },
        "neutral_tolerance": 1e-12,
        "expected_commissioning_sizes": list(EXPECTED_N_VALUES),
        "expected_seeds": list(EXPECTED_SEEDS),
        "strict_design_validation": not args.allow_partial,
        "interpretation_guard": "This post-processing quantifies paired uncertainty; it does not change the frozen E7 detector outputs or compatibility metrics.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(output.to_string(index=False))
    print(f"Wrote paired inference to {args.output}")
    print(f"Wrote paired inference manifest to {args.manifest}")


if __name__ == "__main__":
    main()
