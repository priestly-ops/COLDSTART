#!/usr/bin/env python3
"""
Inspect the raw AURSAD HDF5 file without making assumptions about its schema.

This script recursively inventories groups, datasets, shapes, dtypes,
attributes, representative values, and likely episode/label structures.

It does not train models, alter the dataset, or construct experimental splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "aursad"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 without loading the full 6+ GB file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert HDF5/NumPy metadata into JSON-safe values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        if value.size <= 50:
            return [json_safe(item) for item in value.tolist()]

        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "preview": [
                json_safe(item)
                for item in value.reshape(-1)[:10].tolist()
            ],
        }

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return {
        "__h5py_type__": type(value).__name__,
        "repr": repr(value),
    }


def json_dumps(value: Any) -> str:
    """Serialize values to JSON while handling unusual objects safely."""
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        default=str,
    )


def preview_dataset(
    dataset: h5py.Dataset,
    maximum_values: int = 20,
) -> dict[str, Any]:
    """
    Read only a tiny preview from a dataset.

    This avoids accidentally loading a very large HDF5 dataset.
    """
    result: dict[str, Any] = {
        "preview_available": False,
        "preview": None,
        "minimum": None,
        "maximum": None,
        "unique_preview": None,
    }

    if dataset.size == 0:
        return result

    try:
        if dataset.ndim == 0:
            values = np.asarray([dataset[()]])
        elif dataset.ndim == 1:
            values = np.asarray(
                dataset[: min(dataset.shape[0], maximum_values)]
            )
        else:
            selection = tuple(
                slice(0, min(size, 2))
                for size in dataset.shape
            )
            values = np.asarray(dataset[selection]).reshape(-1)
            values = values[:maximum_values]

        result["preview_available"] = True
        result["preview"] = json_safe(values)

        if np.issubdtype(values.dtype, np.number):
            finite = values[np.isfinite(values)]

            if finite.size:
                result["minimum"] = float(np.min(finite))
                result["maximum"] = float(np.max(finite))

        try:
            unique = np.unique(values)

            if unique.size <= 20:
                result["unique_preview"] = json_safe(unique)
            else:
                result["unique_preview"] = json_safe(unique[:20])
        except (TypeError, ValueError):
            pass

    except Exception as exc:
        result["preview_error"] = repr(exc)

    return result


def inspect_hdf5(
    input_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Recursively inspect the HDF5 tree."""
    object_records: list[dict[str, Any]] = []
    attribute_records: list[dict[str, Any]] = []

    with h5py.File(input_path, "r") as handle:
        root_attributes = {
            str(key): json_safe(value)
            for key, value in handle.attrs.items()
        }

        for key, value in handle.attrs.items():
            attribute_records.append(
                {
                    "object_path": "/",
                    "object_type": "file",
                    "attribute_name": str(key),
                    "attribute_value": json_dumps(value),
                }
            )

        object_records.append(
            {
                "path": "/",
                "name": "/",
                "object_type": "file",
                "parent_path": "",
                "shape": "",
                "ndim": "",
                "dtype": "",
                "size": "",
                "compression": "",
                "chunks": "",
                "attribute_count": len(handle.attrs),
                "preview": "",
                "minimum_preview": "",
                "maximum_preview": "",
                "unique_preview": "",
                "preview_error": "",
            }
        )

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            full_path = f"/{name}"
            parent_path = str(Path(full_path).parent).replace("\\", "/")

            if parent_path == ".":
                parent_path = "/"

            if isinstance(obj, h5py.Group):
                object_type = "group"
                shape = ""
                ndim = ""
                dtype = ""
                size = ""
                compression = ""
                chunks = ""
                preview = {}

            elif isinstance(obj, h5py.Dataset):
                object_type = "dataset"
                shape = json.dumps(list(obj.shape))
                ndim = int(obj.ndim)
                dtype = str(obj.dtype)
                size = int(obj.size)
                compression = obj.compression or ""
                chunks = (
                    json.dumps(list(obj.chunks))
                    if obj.chunks is not None
                    else ""
                )
                preview = preview_dataset(obj)

            else:
                object_type = type(obj).__name__
                shape = ""
                ndim = ""
                dtype = ""
                size = ""
                compression = ""
                chunks = ""
                preview = {}

            object_records.append(
                {
                    "path": full_path,
                    "name": Path(full_path).name,
                    "object_type": object_type,
                    "parent_path": parent_path,
                    "shape": shape,
                    "ndim": ndim,
                    "dtype": dtype,
                    "size": size,
                    "compression": compression,
                    "chunks": chunks,
                    "attribute_count": len(obj.attrs),
                    "preview": json_dumps(
                        preview.get("preview")
                    )
                    if preview
                    else "",
                    "minimum_preview": preview.get("minimum", "")
                    if preview
                    else "",
                    "maximum_preview": preview.get("maximum", "")
                    if preview
                    else "",
                    "unique_preview": json_dumps(
                        preview.get("unique_preview")
                    )
                    if preview
                    else "",
                    "preview_error": preview.get(
                        "preview_error",
                        "",
                    )
                    if preview
                    else "",
                }
            )

            for attribute_name, attribute_value in obj.attrs.items():
                attribute_records.append(
                    {
                        "object_path": full_path,
                        "object_type": object_type,
                        "attribute_name": str(attribute_name),
                        "attribute_value": json_dumps(attribute_value),
                    }
                )

        handle.visititems(visitor)

    summary = {
        "root_attributes": root_attributes,
        "object_count": len(object_records),
        "group_count": sum(
            record["object_type"] == "group"
            for record in object_records
        ),
        "dataset_count": sum(
            record["object_type"] == "dataset"
            for record in object_records
        ),
    }

    return object_records, attribute_records, summary


def find_schema_candidates(
    objects: pd.DataFrame,
) -> pd.DataFrame:
    """Flag paths likely related to episodes, labels, samples, or signals."""
    candidate_terms = (
        "label",
        "class",
        "anomaly",
        "fault",
        "sample",
        "episode",
        "execution",
        "sequence",
        "time",
        "timestamp",
        "robot",
        "screw",
        "sensor",
        "data",
    )

    path_text = objects["path"].astype(str).str.lower()

    mask = np.zeros(len(objects), dtype=bool)

    matched_terms: list[str] = []

    for index, path in enumerate(path_text):
        terms = [term for term in candidate_terms if term in path]
        matched_terms.append(",".join(terms))
        mask[index] = bool(terms)

    candidates = objects.loc[mask].copy()
    candidates["matched_terms"] = np.asarray(matched_terms)[mask]

    return candidates


def build_tree_text(objects: pd.DataFrame) -> str:
    """Create a readable text tree for quick inspection."""
    lines: list[str] = []

    for record in objects.itertuples(index=False):
        path = str(record.path)

        if path == "/":
            depth = 0
        else:
            depth = path.strip("/").count("/") + 1

        indentation = "  " * depth

        if record.object_type == "dataset":
            descriptor = (
                f"dataset shape={record.shape} "
                f"dtype={record.dtype}"
            )
        else:
            descriptor = record.object_type

        lines.append(
            f"{indentation}{path} [{descriptor}]"
        )

    return "\n".join(lines) + "\n"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the raw AURSAD HDF5 structure."
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to AURSAD.h5.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit artifacts.",
    )

    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA-256 calculation for faster inspection.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.data_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"AURSAD file does not exist: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"AURSAD path is not a file: {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AURSAD RAW HDF5 INSPECTION")
    print("=" * 72)
    print(f"Input: {input_path}")
    print(
        f"Size: {input_path.stat().st_size / (1024 ** 3):.3f} GiB"
    )

    if args.skip_hash:
        file_hash = None
        print("SHA-256: skipped")
    else:
        print("Computing SHA-256...")
        file_hash = sha256_file(input_path)
        print(f"SHA-256: {file_hash}")

    print("Inspecting HDF5 hierarchy...")

    object_records, attribute_records, structure_summary = inspect_hdf5(
        input_path
    )

    objects = pd.DataFrame(object_records)
    attributes = pd.DataFrame(attribute_records)
    candidates = find_schema_candidates(objects)

    objects_path = output_dir / "aursad_hdf5_objects.csv"
    attributes_path = output_dir / "aursad_hdf5_attributes.csv"
    candidates_path = output_dir / "aursad_schema_candidates.csv"
    tree_path = output_dir / "aursad_hdf5_tree.txt"
    manifest_path = output_dir / "aursad_raw_inspection.json"

    objects.to_csv(objects_path, index=False)
    attributes.to_csv(attributes_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    tree_path.write_text(
        build_tree_text(objects),
        encoding="utf-8",
    )

    manifest = {
        "inspection_version": "aursad-raw-inspection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "dataset": {
            "path": str(input_path),
            "filename": input_path.name,
            "size_bytes": input_path.stat().st_size,
            "size_gib": input_path.stat().st_size / (1024 ** 3),
            "sha256": file_hash,
        },
        "hdf5": structure_summary,
        "artifacts": {
            "objects_csv": str(objects_path),
            "attributes_csv": str(attributes_path),
            "schema_candidates_csv": str(candidates_path),
            "tree_txt": str(tree_path),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h5py": h5py.__version__,
        },
        "official_reference_expectations_not_yet_verified": {
            "sampling_frequency_hz": 100,
            "tightening_execution_count": 2045,
            "normal_execution_count": 1420,
            "damaged_screw_count": 221,
            "extra_component_count": 183,
            "missing_screw_count": 218,
            "damaged_thread_count": 3,
            "supplementary_loosening_count": 2049,
        },
        "limitations": [
            "This file inventories the raw HDF5 schema only.",
            "Official published counts are recorded as expectations and "
            "must be independently verified against the raw file.",
            "No experimental split has been constructed.",
            "No samples, windows, or rows have been treated as independent "
            "episodes without schema confirmation.",
        ],
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nInspection complete.")
    print(f"Objects: {len(objects)}")
    print(
        f"Groups: "
        f"{(objects['object_type'] == 'group').sum()}"
    )
    print(
        f"Datasets: "
        f"{(objects['object_type'] == 'dataset').sum()}"
    )
    print(f"Schema candidates: {len(candidates)}")

    print("\nArtifacts:")
    print(f"  {objects_path}")
    print(f"  {attributes_path}")
    print(f"  {candidates_path}")
    print(f"  {tree_path}")
    print(f"  {manifest_path}")

    print("\nLikely schema candidates:")
    display_columns = [
        "path",
        "object_type",
        "shape",
        "dtype",
        "matched_terms",
    ]

    if candidates.empty:
        print("  No candidate paths identified by name.")
    else:
        print(
            candidates[display_columns]
            .head(100)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()