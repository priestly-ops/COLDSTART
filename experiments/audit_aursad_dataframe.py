#!/usr/bin/env python3
"""Memory-safe schema and column audit for the raw AURSAD HDF5 table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "aursad"

ID_TERMS = ("id", "sample", "run", "cycle", "episode", "execution", "sequence", "trial")
LABEL_TERMS = ("label", "class", "anomaly", "fault", "failure", "condition", "result")
TIME_TERMS = ("time", "timestamp", "date")
OPERATION_TERMS = ("operation", "action", "task", "process", "stage", "phase", "tighten", "loosen", "pick")


def decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_terms(name: str, terms: tuple[str, ...]) -> str:
    normalized = normalize(name)
    return ",".join(term for term in terms if term in normalized)


def json_safe(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return decode(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Stats:
    def __init__(self, name: str, dtype: np.dtype) -> None:
        self.name = name
        self.dtype = np.dtype(dtype)
        self.count = 0
        self.missing = 0
        self.finite = 0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.total = 0.0
        self.total_sq = 0.0
        self.unique = Counter()
        self.unique_truncated = False
        self.preview: list[Any] = []

    @property
    def numeric(self) -> bool:
        return np.issubdtype(self.dtype, np.number)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values).reshape(-1)
        self.count += values.size

        if len(self.preview) < 20:
            remaining = 20 - len(self.preview)
            self.preview.extend(json_safe(values[:remaining]))

        if self.numeric:
            numeric = values.astype(np.float64, copy=False)
            mask = np.isfinite(numeric)
            valid = numeric[mask]
            self.missing += int((~mask).sum())
            self.finite += int(valid.size)
            if valid.size:
                local_min = float(valid.min())
                local_max = float(valid.max())
                self.minimum = local_min if self.minimum is None else min(self.minimum, local_min)
                self.maximum = local_max if self.maximum is None else max(self.maximum, local_max)
                self.total += float(valid.sum(dtype=np.float64))
                self.total_sq += float(np.square(valid).sum(dtype=np.float64))
                self._update_unique(values[mask])
        else:
            decoded = np.asarray([decode(v) for v in values], dtype=object)
            mask = ~pd.isna(decoded)
            self.missing += int((~mask).sum())
            self.finite += int(mask.sum())
            self._update_unique(decoded[mask])

    def _update_unique(self, values: np.ndarray) -> None:
        if self.unique_truncated or values.size == 0:
            return
        unique, counts = np.unique(values, return_counts=True)
        for value, count in zip(unique, counts):
            self.unique[str(decode(value))] += int(count)
            if len(self.unique) > 50_000:
                self.unique.clear()
                self.unique_truncated = True
                break

    def record(self) -> dict[str, Any]:
        mean = None
        std = None
        if self.numeric and self.finite:
            mean = self.total / self.finite
            variance = max(0.0, self.total_sq / self.finite - mean * mean)
            std = math.sqrt(variance)

        return {
            "column": self.name,
            "dtype": str(self.dtype),
            "row_count": self.count,
            "missing_count": self.missing,
            "missing_fraction": self.missing / self.count if self.count else None,
            "finite_count": self.finite,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "std": std,
            "unique_count_exact": None if self.unique_truncated else len(self.unique),
            "unique_count_truncated": self.unique_truncated,
            "is_constant": False if self.unique_truncated else len(self.unique) <= 1,
            "preview_json": json.dumps(self.preview, ensure_ascii=False),
            "top_values_json": json.dumps(
                [{"value": k, "count": v} for k, v in self.unique.most_common(20)],
                ensure_ascii=False,
            ),
        }


def inspect(path: Path, chunk_rows: int):
    with h5py.File(path, "r") as handle:
        group = handle["/complete_data"]
        block_items = sorted(name for name in group if re.fullmatch(r"block\d+_items", name))
        locations = []

        for item_name in block_items:
            block = item_name.split("_")[0]
            value_name = f"{block}_values"
            names = [str(decode(v)) for v in group[item_name][:]]
            values = group[value_name]
            if len(names) != values.shape[1]:
                raise ValueError(f"Column mismatch for {block}")
            for index, name in enumerate(names):
                locations.append((name, value_name, index, values.dtype))

        row_counts = {group[value_name].shape[0] for _, value_name, _, _ in locations}
        if len(row_counts) != 1:
            raise ValueError(f"Inconsistent row counts: {row_counts}")
        row_count = row_counts.pop()

        stats = {name: Stats(name, dtype) for name, _, _, dtype in locations}
        by_block: dict[str, list[tuple[str, int]]] = {}
        for name, value_name, index, _ in locations:
            by_block.setdefault(value_name, []).append((name, index))

        for start in range(0, row_count, chunk_rows):
            stop = min(start + chunk_rows, row_count)
            print(f"Rows {start:,}:{stop:,} ({100 * stop / row_count:.1f}%)")
            for value_name, columns in by_block.items():
                chunk = group[value_name][start:stop, :]
                for name, index in columns:
                    stats[name].update(chunk[:, index])

        schema = pd.DataFrame([
            {
                "column": name,
                "normalized_column": normalize(name),
                "dtype": str(dtype),
                "block_name": value_name,
                "block_column_index": index,
                "id_terms": find_terms(name, ID_TERMS),
                "label_terms": find_terms(name, LABEL_TERMS),
                "time_terms": find_terms(name, TIME_TERMS),
                "operation_terms": find_terms(name, OPERATION_TERMS),
            }
            for name, value_name, index, dtype in locations
        ])

        statistics = pd.DataFrame([stats[name].record() for name, *_ in locations])
        candidate_mask = (
            schema["id_terms"].ne("")
            | schema["label_terms"].ne("")
            | schema["time_terms"].ne("")
            | schema["operation_terms"].ne("")
        )
        candidates = schema.loc[candidate_mask].merge(
            statistics[[
                "column", "unique_count_exact", "unique_count_truncated",
                "minimum", "maximum", "missing_fraction", "preview_json",
                "top_values_json",
            ]],
            on="column",
            how="left",
            validate="one_to_one",
        )
        constants = statistics[statistics["is_constant"]].copy()

        summary = {
            "row_count": row_count,
            "column_count": len(locations),
            "block_shapes": {name: list(group[name].shape) for name in by_block},
            "candidate_column_count": len(candidates),
            "constant_column_count": len(constants),
        }

    return schema, statistics, candidates, constants, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    data_path = args.data_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    dataset_hash = None if args.skip_hash else sha256_file(data_path)
    schema, statistics, candidates, constants, summary = inspect(data_path, args.chunk_rows)

    outputs = {
        "schema": output_dir / "aursad_dataframe_schema.csv",
        "statistics": output_dir / "aursad_column_statistics.csv",
        "candidates": output_dir / "aursad_episode_label_candidates.csv",
        "constants": output_dir / "aursad_constant_columns.csv",
        "manifest": output_dir / "aursad_dataframe_audit.json",
    }

    schema.to_csv(outputs["schema"], index=False)
    statistics.to_csv(outputs["statistics"], index=False)
    candidates.to_csv(outputs["candidates"], index=False)
    constants.to_csv(outputs["constants"], index=False)

    manifest = {
        "audit_version": "aursad-dataframe-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": {
            "path": str(data_path),
            "size_bytes": data_path.stat().st_size,
            "sha256": dataset_hash,
        },
        "audit": summary,
        "artifacts": {key: str(value) for key, value in outputs.items() if key != "manifest"},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "limitations": [
            "No execution identifier has yet been declared.",
            "No leakage-safe protocol or detector split has been created.",
            "Candidate column semantics must be verified from raw counts and documentation.",
        ],
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nLikely episode/label/time/operation columns:")
    if candidates.empty:
        print("None detected by name.")
    else:
        print(candidates[[
            "column", "dtype", "id_terms", "label_terms", "time_terms",
            "operation_terms", "unique_count_exact", "minimum", "maximum",
            "missing_fraction",
        ]].to_string(index=False))

    print("\nSaved:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()