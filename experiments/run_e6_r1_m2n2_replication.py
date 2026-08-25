from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from experiments.run_e6_m2n2_modern_adaptation import _evaluate_frozen
from src.e6_replication import (
    EXPECTED_N,
    EXPECTED_SEEDS,
    metric_summary_table,
    paired_adaptation_table,
    validate_e6r1_design,
)
from src.m2n2_adapter import ChannelStandardizer, M2N2Adapter, M2N2Config
from src.split_generator import SOURCE_SETTING, create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROTOCOL_VERSION = "coldstart-e6-r1-m2n2-20seed-v1"
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e6_r1_m2n2_20seed"
B = 0.01
R0 = 0.90
MODEL_SEED = 42
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
AGGREGATIONS = ("q99", "mean")


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _completed_seeds(df: pd.DataFrame, aggregation: str) -> set[int]:
    if df.empty:
        return set()
    required = {"cycle_score_aggregation", "variant", "commissioning_size", "seed"}
    if not required.issubset(df.columns):
        return set()
    done: set[int] = set()
    for seed, g in df[df.cycle_score_aggregation == aggregation].groupby("seed"):
        variants = set(g.variant.astype(str))
        if {"M2N2-Offline", "M2N2-CommissioningAdapt"}.issubset(variants):
            done.add(int(seed))
    return done


def _run_aggregation(
    *,
    aggregation: str,
    cycles,
    source_cycles,
    seeds: tuple[int, ...],
    device: str,
    resume: bool,
) -> pd.DataFrame:
    out_dir = OUTPUT_DIR / aggregation
    seed_path = out_dir / "e6r1_seed_results.csv"
    existing = _load_existing(seed_path) if resume else pd.DataFrame()
    done = _completed_seeds(existing, aggregation) if resume else set()

    cycle_score_quantile = 0.99 if aggregation == "q99" else None
    standardizer = ChannelStandardizer.fit(source_cycles)
    cfg = M2N2Config(seed=MODEL_SEED, cycle_score_quantile=cycle_score_quantile)

    print(
        f"E6-R1 [{aggregation}] training frozen source model on "
        f"{len(source_cycles)} healthy source cycles using device={device}"
    )
    source_model = M2N2Adapter(
        num_channels=len(source_cycles[0].columns),
        standardizer=standardizer,
        config=cfg,
        device=device,
    ).fit_source(source_cycles)

    rows = [] if existing.empty else existing.to_dict("records")
    total = len(seeds)
    for index, seed in enumerate(seeds, start=1):
        if seed in done:
            print(f"E6-R1 [{aggregation}] {index}/{total}: seed={seed} already complete; skipping")
            continue

        split = create_frozen_evaluation_split(
            cycles=cycles,
            commissioning_size=EXPECTED_N,
            commissioning_seed=seed,
            evaluation_seed=EVALUATION_SEED,
            calibration_size=CALIBRATION_SIZE,
            normal_evaluation_size=HEALTHY_EVAL_SIZE,
            maximum_commissioning_size=MAX_COMMISSIONING_SIZE,
        )

        offline = source_model.clone()
        offline_row = _evaluate_frozen(
            variant="M2N2-Offline",
            model=offline,
            calibration_cycles=split.target_calibration,
            healthy_cycles=split.target_normal_evaluation,
            anomaly_cycles=split.target_anomaly_evaluation,
            n_value=EXPECTED_N,
            seed=seed,
            aggregation=aggregation,
        )
        offline_row["protocol_version"] = PROTOCOL_VERSION
        rows.append(offline_row)

        adapted = source_model.clone()
        adapt_info = adapted.adapt_cycles(split.target_commissioning)
        adapted_row = _evaluate_frozen(
            variant="M2N2-CommissioningAdapt",
            model=adapted,
            calibration_cycles=split.target_calibration,
            healthy_cycles=split.target_normal_evaluation,
            anomaly_cycles=split.target_anomaly_evaluation,
            n_value=EXPECTED_N,
            seed=seed,
            aggregation=aggregation,
            adaptation_info=adapt_info,
        )
        adapted_row["protocol_version"] = PROTOCOL_VERSION
        rows.append(adapted_row)

        current = pd.DataFrame(rows).sort_values(["seed", "variant"]).reset_index(drop=True)
        _atomic_csv(current, seed_path)
        print(
            f"E6-R1 [{aggregation}] {index}/{total}: seed={seed} "
            f"offline AUROC={offline_row['auroc']:.4f} oracle={offline_row['oracle_recall_at_fpr_budget']:.4f}; "
            f"adapt AUROC={adapted_row['auroc']:.4f} oracle={adapted_row['oracle_recall_at_fpr_budget']:.4f}"
        )

    result = pd.DataFrame(rows).sort_values(["seed", "variant"]).reset_index(drop=True)
    _atomic_csv(result, seed_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E6-R1: frozen 20-seed replication of M2N2 at N=100 using q99 primary and mean sensitivity."
    )
    parser.add_argument("--seeds", default=",".join(map(str, EXPECTED_SEEDS)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--aggregation", choices=("both", "q99", "mean"), default="both")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing E6-R1 seed CSVs and rerun requested seeds.")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--signflip-repetitions", type=int, default=100_000)
    args = parser.parse_args()

    seeds = _parse_ints(args.seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not set(seeds).issubset(EXPECTED_SEEDS):
        raise ValueError("seeds must be a subset of 0..19")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    np.random.seed(MODEL_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    source_cycles = sorted(
        [c for c in cycles if (not c.anomaly and c.setting == SOURCE_SETTING)],
        key=lambda c: int(c.episode_id),
    )
    if not source_cycles:
        raise RuntimeError("No healthy source cycles found")

    requested_aggs = AGGREGATIONS if args.aggregation == "both" else (args.aggregation,)
    result_frames: list[pd.DataFrame] = []
    for aggregation in requested_aggs:
        result_frames.append(
            _run_aggregation(
                aggregation=aggregation,
                cycles=cycles,
                source_cycles=source_cycles,
                seeds=seeds,
                device=args.device,
                resume=not args.no_resume,
            )
        )

    combined = pd.concat(result_frames, ignore_index=True)
    _atomic_csv(combined, OUTPUT_DIR / "e6r1_seed_results_combined.csv")

    full_design = (
        set(seeds) == set(EXPECTED_SEEDS)
        and set(requested_aggs) == set(AGGREGATIONS)
    )
    validate_e6r1_design(combined, strict=full_design)

    metric_summary = metric_summary_table(
        combined,
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        seed=MODEL_SEED,
    )
    paired = paired_adaptation_table(
        combined,
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        signflip_repetitions=int(args.signflip_repetitions),
        seed=MODEL_SEED,
    )
    _atomic_csv(metric_summary, OUTPUT_DIR / "e6r1_metric_summary.csv")
    _atomic_csv(paired, OUTPUT_DIR / "e6r1_paired_inference.csv")

    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_protocol": "coldstart-e6-m2n2-v1",
        "purpose": "20-seed replication of frozen M2N2 primary and predeclared aggregation sensitivity at N=100",
        "dataset": "voraus-AD",
        "source_setting": 72,
        "target_setting": 73,
        "commissioning_size": EXPECTED_N,
        "commissioning_seeds": list(seeds),
        "model_seed": MODEL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "false_alert_budget": B,
        "recall_target": R0,
        "device": args.device,
        "aggregations": {
            "q99": "primary execution-level aggregation retained from frozen E6 protocol",
            "mean": "predeclared sensitivity retained from frozen E6 protocol",
        },
        "variants": ["M2N2-Offline", "M2N2-CommissioningAdapt"],
        "online_variant_excluded": "M2N2-Online changes the detector during evaluation and is therefore kept diagnostic-only outside this fixed-detector replication.",
        "uncertainty": {
            "metric_summary": "95% percentile bootstrap over commissioning draws",
            "paired_delta": "CommissioningAdapt minus Offline on identical commissioning seeds",
            "bootstrap_repetitions": int(args.bootstrap_repetitions),
            "signflip_repetitions": int(args.signflip_repetitions),
            "seed": MODEL_SEED,
        },
        "resume_policy": "completed aggregation/seed pairs are skipped when both frozen variants are present; use --no-resume to force rerun",
        "interpretation_guard": "Commissioning seeds quantify sensitivity to target commissioning draws under the fixed evaluation partition; they are not independent robot deployments.",
        "full_frozen_design_complete": bool(full_design),
    }
    (OUTPUT_DIR / "e6r1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nE6-R1 metric summary")
    print(metric_summary.to_string(index=False))
    print("\nE6-R1 paired adaptation inference")
    print(paired.to_string(index=False))
    print(f"\nWrote E6-R1 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
