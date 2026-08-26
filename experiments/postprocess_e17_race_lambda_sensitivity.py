from __future__ import annotations

"""Rebuild E17 summary/inference from checkpointed seed results.

This postprocessor exists so a completed/resumed E17 sweep never needs to rerun
model evaluations merely because pandas inferred ``lambda_label`` values such as
"inf" as floating-point values when reloading the checkpoint CSV.

It also repairs duplicate checkpoint cells produced by an earlier resume bug,
but only when the duplicate rows are numerically/string-wise identical. Any
conflicting duplicate remains a hard error so we never silently choose between
inconsistent experimental results.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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


KEY_COLUMNS = ["commissioning_size", "seed", "lambda_label"]


def _canonicalize_lambda_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lambda_label"] = out["lambda_label"].astype(str).str.strip().str.lower()
    canonical: dict[str, str] = {}
    for value in RACE_LAMBDA_GRID:
        label = lambda_label(value)
        canonical[label] = label
        if label != "inf":
            canonical[str(float(value))] = label
    out["lambda_label"] = out["lambda_label"].map(lambda x: canonical.get(x, x))
    return out


def _series_values_equal(a: pd.Series, b: pd.Series) -> bool:
    """Compare two checkpoint rows without hiding genuine disagreements."""
    for column in a.index:
        av = a[column]
        bv = b[column]
        if pd.isna(av) and pd.isna(bv):
            continue
        if pd.isna(av) != pd.isna(bv):
            return False

        # Preserve exact semantics for booleans/strings while tolerating only
        # tiny CSV round-trip differences for numeric values.
        try:
            af = float(av)
            bf = float(bv)
            if np.isfinite(af) and np.isfinite(bf):
                if not np.isclose(af, bf, rtol=1e-12, atol=1e-12):
                    return False
                continue
            if np.isinf(af) or np.isinf(bf):
                if af != bf:
                    return False
                continue
        except (TypeError, ValueError):
            pass

        if str(av) != str(bv):
            return False
    return True


def _deduplicate_identical_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_mask = df.duplicated(KEY_COLUMNS, keep=False)
    if not duplicate_mask.any():
        return df, 0

    conflict_keys: list[tuple[int, int, str]] = []
    duplicate_rows_removed = 0

    for key, group in df.loc[duplicate_mask].groupby(KEY_COLUMNS, sort=True, dropna=False):
        first = group.iloc[0]
        for i in range(1, len(group)):
            if not _series_values_equal(first, group.iloc[i]):
                conflict_keys.append((int(key[0]), int(key[1]), str(key[2])))
                break
        duplicate_rows_removed += len(group) - 1

    if conflict_keys:
        raise RuntimeError(
            "Conflicting duplicate E17 checkpoint cells detected; refusing to "
            f"choose a result silently. First conflicts={conflict_keys[:10]}"
        )

    cleaned = df.drop_duplicates(KEY_COLUMNS, keep="first").reset_index(drop=True)
    return cleaned, duplicate_rows_removed


def _load_checkpoint(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"E17 checkpoint not found: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "lambda_label": "string",
            "protocol_version": "string",
        },
    )
    required = set(KEY_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"E17 checkpoint missing required columns: {missing}")

    df = _canonicalize_lambda_labels(df)
    df, duplicates_removed = _deduplicate_identical_cells(df)
    return df, duplicates_removed


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
    df, duplicates_removed = _load_checkpoint(SEED_RESULTS_PATH)
    _validate_full_design(df)

    expected_labels = [lambda_label(v) for v in RACE_LAMBDA_GRID]
    selected = df[
        df.commissioning_size.isin(N_GRID)
        & df.seed.isin(SEEDS)
        & df.lambda_label.isin(expected_labels)
    ].copy()

    # Rewrite the checkpoint itself only after strict validation. This repairs
    # the previous duplicate/resume artifact and leaves one canonical row per
    # experimental cell for future reproducibility checks.
    _atomic_csv(selected, SEED_RESULTS_PATH)
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
        "duplicate_checkpoint_rows_removed": int(duplicates_removed),
        "duplicate_policy": "remove only duplicate N/seed/lambda cells whose complete rows agree; conflicting duplicates are fatal",
        "lambda_label_read_policy": "force string dtype and canonicalize labels before completeness checks",
        "interpretation_guard": "lambda=60 remains the original frozen method. No lambda selected from this sweep may be used to redefine RACE or replace the primary result.",
        "full_frozen_design_complete": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Removed {duplicates_removed} identical duplicate checkpoint row(s).")
    print(f"Validated {len(selected)} unique E17 cells (expected 560).")
    print(_summary(selected).to_string(index=False))
    print(f"Rebuilt E17 summary: {SUMMARY_PATH}")
    print(f"Rebuilt E17 paired inference: {PAIRED_PATH}")
    print(f"Rebuilt E17 manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
