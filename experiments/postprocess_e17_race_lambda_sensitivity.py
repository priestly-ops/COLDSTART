from __future__ import annotations

"""Rebuild E17 summary/inference from checkpointed seed results.

This postprocessor exists so a completed/resumed E17 sweep never needs to rerun
model evaluations merely because pandas inferred ``lambda_label`` values such as
"inf" as floating-point values when reloading the checkpoint CSV.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.run_e17_race_lambda_sensitivity import (
    MANIFEST_PATH,
    N_GRID,
    PAIRED_PATH,
    PROTOCOL_VERSION,
    RACE_LAMBDA_GRID,
    SEEDS,
    SEED_RESULTS_PATH,
    SUMMARY_PATH,
    _atomic_csv,
    _paired_inference,
    _summary,
)
from src.race_lambda_sensitivity import FROZEN_RACE_LAMBDA, lambda_label


def _load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"E17 checkpoint not found: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "lambda_label": "string",
            "protocol_version": "string",
        },
    )
    required = {"commissioning_size", "seed", "lambda_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"E17 checkpoint missing required columns: {missing}")

    df["lambda_label"] = df["lambda_label"].astype(str).str.strip().str.lower()
    # Defensive normalization if an earlier pandas load/write cycle produced
    # labels like "0.0" or "60.0".
    canonical = {}
    for value in RACE_LAMBDA_GRID:
        label = lambda_label(value)
        canonical[label] = label
        if label != "inf":
            canonical[str(float(value))] = label
    df["lambda_label"] = df["lambda_label"].map(lambda x: canonical.get(x, x))

    if df.duplicated(["commissioning_size", "seed", "lambda_label"]).any():
        dup = df.loc[
            df.duplicated(["commissioning_size", "seed", "lambda_label"], keep=False),
            ["commissioning_size", "seed", "lambda_label"],
        ]
        raise RuntimeError(
            "Duplicate E17 checkpoint cells detected:\n" + dup.to_string(index=False)
        )
    return df


def _validate_full_design(df: pd.DataFrame) -> None:
    expected_labels = {lambda_label(v) for v in RACE_LAMBDA_GRID}
    expected_cells = {
        (int(n), int(seed), label)
        for n in N_GRID
        for seed in SEEDS
        for label in expected_labels
    }
    actual_cells = {
        (int(r.commissioning_size), int(r.seed), str(r.lambda_label))
        for r in df.itertuples(index=False)
        if int(r.commissioning_size) in N_GRID and int(r.seed) in SEEDS
    }
    missing = sorted(expected_cells - actual_cells)
    extra = sorted(actual_cells - expected_cells)
    if missing or extra:
        raise RuntimeError(
            f"E17 full design is incomplete: missing={len(missing)} extra={len(extra)}. "
            f"First missing={missing[:5]} first extra={extra[:5]}"
        )


def main() -> None:
    df = _load_checkpoint(SEED_RESULTS_PATH)
    _validate_full_design(df)

    expected_labels = [lambda_label(v) for v in RACE_LAMBDA_GRID]
    selected = df[
        df.commissioning_size.isin(N_GRID)
        & df.seed.isin(SEEDS)
        & df.lambda_label.isin(expected_labels)
    ].copy()

    _atomic_csv(_summary(selected), SUMMARY_PATH)
    _atomic_csv(_paired_inference(selected), PAIRED_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "robustness audit of the frozen RACE shrinkage strength; not hyperparameter tuning",
        "dataset": "voraus-AD",
        "commissioning_grid": list(N_GRID),
        "commissioning_seeds": list(SEEDS),
        "lambda_grid": expected_labels,
        "frozen_race_lambda": lambda_label(FROZEN_RACE_LAMBDA),
        "postprocessed_from_checkpoint": True,
        "checkpoint_rows": int(len(selected)),
        "lambda_label_read_policy": "force string dtype and canonicalize labels before completeness checks",
        "interpretation_guard": "lambda=60 remains the original frozen method. No lambda selected from this sweep may be used to redefine RACE or replace the primary result.",
        "full_frozen_design_complete": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(_summary(selected).to_string(index=False))
    print(f"Rebuilt E17 summary: {SUMMARY_PATH}")
    print(f"Rebuilt E17 paired inference: {PAIRED_PATH}")
    print(f"Rebuilt E17 manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
