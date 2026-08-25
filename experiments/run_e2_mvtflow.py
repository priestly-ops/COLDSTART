from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mvtflow_adapter import OFFICIAL_CONFIG, run_mvtflow_replicate
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e2_mvtflow"
SEED_RESULTS_PATH = OUTPUT_DIR / "e2_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e2_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e2_manifest.json"

COMMISSIONING_GRID = [10, 25, 50, 100]
SEEDS = list(range(20))
EVALUATION_SEED = 42
GLOBAL_SEED = 42
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
PROTOCOL_VERSION = "coldstart-e2-mvtflow-v1"

np.random.seed(GLOBAL_SEED)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_csv(SEED_RESULTS_PATH)
    if "protocol_version" not in frame.columns:
        raise ValueError("Incompatible E2 checkpoint: protocol_version missing.")
    versions = set(frame["protocol_version"].dropna().astype(str).unique())
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(f"Incompatible E2 protocol versions: {sorted(versions)}")
    return frame


def _completed_keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
    if frame.empty:
        return set()
    return {
        (int(row.commissioning_size), int(row.seed))
        for row in frame.itertuples(index=False)
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = [
        "representation_limited",
        "calibration_limited",
        "certification_limited",
        "certified",
    ]
    for n_value, group in results.groupby("commissioning_size", sort=True):
        row: dict[str, Any] = {
            "detector": "TargetOnly-MVTFlow",
            "commissioning_size": int(n_value),
            "number_of_seeds": int(len(group)),
            "mean_recall": float(group["recall"].mean()),
            "mean_fpr": float(group["false_positive_rate"].mean()),
            "mean_auroc": float(group["auroc"].mean()),
            "mean_oracle_recall_at_fpr_budget": float(
                group["oracle_recall_at_fpr_budget"].mean()
            ),
            "oracle_feasibility_rate": float(
                group["oracle_empirically_feasible"].astype(bool).mean()
            ),
            "empirical_success_rate": float(
                group["empirical_success"].astype(bool).mean()
            ),
            "certified_success_rate": float(
                group["certified_success"].astype(bool).mean()
            ),
            "mean_seedwise_recall_lower": float(group["recall_lower"].mean()),
            "mean_seedwise_fpr_upper": float(group["fpr_upper"].mean()),
            "maximum_calibration_rate": float(
                (group["conformal_regime"] == "maximum").mean()
            ),
            "infinite_calibration_rate": float(
                (group["conformal_regime"] == "infinite").mean()
            ),
        }
        for label in labels:
            row[f"{label}_rate"] = float(
                (group["bottleneck_label"] == label).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("commissioning_size").reset_index(drop=True)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MVT-Flow uses the official machine-signal group, not the handcrafted
    # measured-signal feature representation used by TargetOnly/RACE.
    cycles = load_cycles(path=DATASET_PATH, signal_set="machine")

    checkpoint = _load_checkpoint()
    rows = checkpoint.to_dict(orient="records") if not checkpoint.empty else []
    completed = _completed_keys(checkpoint)
    expected = {(n, seed) for n in COMMISSIONING_GRID for seed in SEEDS}

    fixed_calibration_ids: tuple[int, ...] | None = None
    fixed_healthy_eval_ids: tuple[int, ...] | None = None
    fixed_anomaly_eval_ids: tuple[int, ...] | None = None

    print("=" * 78)
    print("COLDSTART E2: TARGET-ONLY MVT-FLOW")
    print("=" * 78)
    print(f"Protocol:             {PROTOCOL_VERSION}")
    print(f"Grid:                 {COMMISSIONING_GRID}")
    print(f"Seeds:                {SEEDS}")
    print(f"Calibration size:     {CALIBRATION_SIZE}")
    print(f"Healthy eval size:    {NORMAL_EVALUATION_SIZE}")
    print(f"Recall target:        {RECALL_TARGET}")
    print(f"FPR budget / alpha:   {FALSE_ALERT_BUDGET}")
    print(f"Joint confidence:     {JOINT_CONFIDENCE}")
    print(f"Official epochs:      {OFFICIAL_CONFIG['epochs']}")
    print(f"Completed:            {len(completed & expected)}/{len(expected)}")
    print("=" * 78)

    counter = len(completed & expected)
    for seed in SEEDS:
        for n_value in COMMISSIONING_GRID:
            key = (n_value, seed)
            if key in completed:
                continue

            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n_value,
                commissioning_seed=seed,
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )

            calibration_ids = tuple(c.episode_id for c in split.target_calibration)
            healthy_ids = tuple(c.episode_id for c in split.target_normal_evaluation)
            anomaly_ids = tuple(c.episode_id for c in split.target_anomaly_evaluation)
            if fixed_calibration_ids is None:
                fixed_calibration_ids = calibration_ids
                fixed_healthy_eval_ids = healthy_ids
                fixed_anomaly_eval_ids = anomaly_ids
            else:
                if calibration_ids != fixed_calibration_ids:
                    raise RuntimeError("E2 calibration IDs changed across runs.")
                if healthy_ids != fixed_healthy_eval_ids:
                    raise RuntimeError("E2 healthy evaluation IDs changed across runs.")
                if anomaly_ids != fixed_anomaly_eval_ids:
                    raise RuntimeError("E2 anomaly evaluation IDs changed across runs.")

            counter += 1
            print(
                f"E2 {counter}/{len(expected)} | N={n_value} seed={seed} "
                f"train_cycles={len(split.target_commissioning)}"
            )

            result = run_mvtflow_replicate(
                commissioning=split.target_commissioning,
                calibration=split.target_calibration,
                healthy_eval=split.target_normal_evaluation,
                anomaly_eval=split.target_anomaly_evaluation,
                project_root=PROJECT_ROOT,
                commissioning_size=n_value,
                seed=seed,
                false_alert_budget=FALSE_ALERT_BUDGET,
                recall_target=RECALL_TARGET,
                joint_confidence=JOINT_CONFIDENCE,
            )
            row = asdict(result)
            row["protocol_version"] = PROTOCOL_VERSION
            rows.append(row)
            completed.add(key)
            _atomic_write_csv(
                pd.DataFrame(rows).sort_values(["commissioning_size", "seed"]),
                SEED_RESULTS_PATH,
            )

    results = pd.DataFrame(rows)
    summary = _build_summary(results)
    _atomic_write_csv(summary, SUMMARY_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": "TargetOnly-MVTFlow",
        "method_source": "official vorausrobotik/voraus-ad-dataset MVT-Flow architecture",
        "training_budget_adaptation": "train only on N target commissioning healthy cycles",
        "commissioning_grid": COMMISSIONING_GRID,
        "commissioning_seeds": SEEDS,
        "evaluation_seed": EVALUATION_SEED,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "calibration_alpha_mapping": "alpha_equals_B",
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": NORMAL_EVALUATION_SIZE,
        "official_hyperparameters": OFFICIAL_CONFIG,
        "normalization": "per-signal mean/std fitted on target commissioning time points only",
        "padding": "target length=max commissioning cycle; truncate then right-pad zeros",
        "score": "official get_loss_per_sample: 0.5*sum(z^2)-jacobian",
        "oracle_decision_family": "same_scalar_score_strict_greater_than_threshold",
        "oracle_is_diagnostic_only": True,
        "frozen_calibration_ids": list(fixed_calibration_ids or ()),
        "frozen_healthy_eval_ids": list(fixed_healthy_eval_ids or ()),
        "frozen_anomaly_eval_ids": list(fixed_anomaly_eval_ids or ()),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nE2 complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
