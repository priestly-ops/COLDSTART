"""Run healthy-only source-target transferability audit for RACE-A0."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.a0_transferability import audit_pair
from src.feature_extractor import extract_feature_batch
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


DEFAULT_VORAUS = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_AURSAD_INVENTORY = PROJECT_ROOT / "reports" / "aursad" / "aursad_episode_inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
N_VALUES = (10, 25)
SEEDS = (0, 1, 2, 3, 4)


def _episode_feature_map(cycles) -> dict[int, np.ndarray]:
    batch = extract_feature_batch(cycles)
    return {
        int(episode_id): batch.features[index]
        for index, episode_id in enumerate(batch.episode_ids)
    }


def _matrix_for(cycles, features_by_episode: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([features_by_episode[int(cycle.episode_id)] for cycle in cycles])


def _aursad_domain_note(inventory_path: Path) -> dict[str, object]:
    if not inventory_path.exists():
        return {
            "supports_defensible_split": False,
            "reason": f"AURSAD inventory not found at {inventory_path}.",
            "metadata_columns": [],
        }
    inventory = pd.read_csv(inventory_path, nrows=20)
    columns = list(inventory.columns)
    candidate_domain_columns = [
        name
        for name in columns
        if name.lower() in {"robot", "unit", "operation", "tool", "load", "session", "trajectory"}
    ]
    return {
        "supports_defensible_split": bool(candidate_domain_columns),
        "reason": (
            "Candidate domain metadata columns found."
            if candidate_domain_columns
            else "Inventory exposes execution/sample audit columns but no robot/unit/session/tool/load domain field."
        ),
        "metadata_columns": columns,
        "candidate_domain_columns": candidate_domain_columns,
    }


def run_audit(
    *,
    voraus_path: Path = DEFAULT_VORAUS,
    output_dir: Path = OUTPUT_DIR,
    bootstrap_resamples: int = 50,
) -> dict[str, Path]:
    cycles = load_cycles(voraus_path)
    features_by_episode = _episode_feature_map(cycles)
    rows: list[dict[str, object]] = []
    angle_rows: list[dict[str, object]] = []

    for n_target in N_VALUES:
        for seed in SEEDS:
            split = create_frozen_evaluation_split(
                cycles,
                commissioning_size=n_target,
                commissioning_seed=seed,
            )
            source = _matrix_for(split.source_train, features_by_episode)
            target = _matrix_for(split.target_commissioning, features_by_episode)
            audit, cos2 = audit_pair(
                source,
                target,
                dataset="voraus-ad",
                source_domain="source_train_protocol",
                target_domain=f"target_commissioning_N{n_target}_seed{seed}",
                n_target=n_target,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            rows.append(asdict(audit))
            for index, value in enumerate(cos2):
                angle_rows.append(
                    {
                        "dataset": audit.dataset,
                        "source_domain": audit.source_domain,
                        "target_domain": audit.target_domain,
                        "n_target": n_target,
                        "seed": seed,
                        "mode_index": index,
                        "cos2": float(value),
                        "principal_angle_degrees": float(
                            np.degrees(np.arccos(np.sqrt(np.clip(value, 0.0, 1.0))))
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "a0_transferability_audit.csv"
    angles_path = output_dir / "a0_transferability_principal_angles.csv"
    manifest_path = output_dir / "a0_transferability_manifest.json"
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    pd.DataFrame(angle_rows).to_csv(angles_path, index=False)
    manifest = {
        "configuration": {
            "datasets": ["voraus-ad"],
            "n_values": list(N_VALUES),
            "seeds": list(SEEDS),
            "bootstrap_resamples": bootstrap_resamples,
            "k_max": 16,
            "selection_rule": "healthy-only frozen source_train vs target_commissioning protocol",
        },
        "aursad_domain_investigation": _aursad_domain_note(DEFAULT_AURSAD_INVENTORY),
        "outputs": {
            "audit_csv": str(audit_path),
            "principal_angles_csv": str(angles_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "audit": audit_path,
        "principal_angles": angles_path,
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voraus-path", type=Path, default=DEFAULT_VORAUS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=50)
    args = parser.parse_args()
    outputs = run_audit(
        voraus_path=args.voraus_path,
        output_dir=args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
