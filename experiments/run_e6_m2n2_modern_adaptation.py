from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bottleneck_decomposition import classify_bottleneck
from src.calibration_tail import conformal_threshold_info
from src.certification import certify_operating_point
from src.m2n2_adapter import ChannelStandardizer, M2N2Adapter, M2N2Config
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.split_generator import SOURCE_SETTING, create_frozen_evaluation_split
from src.voraus_loader import load_cycles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
BASE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e6_m2n2"

PROTOCOL_VERSION = "coldstart-e6-m2n2-v1"
SEEDS = tuple(range(20))
COMMISSIONING_GRID = (10, 25, 50, 100)
ONLINE_DIAGNOSTIC_N = 100
ONLINE_DIAGNOSTIC_SEEDS = (0, 1, 2)
EVALUATION_SEED = 42
CALIBRATION_SIZE = 100
HEALTHY_EVAL_SIZE = 100
MAX_COMMISSIONING_SIZE = 100
B = 0.01
R0 = 0.90
CONFIDENCE = 0.95
MODEL_SEED = 42


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _evaluate_frozen(
    *,
    variant: str,
    model: M2N2Adapter,
    calibration_cycles,
    healthy_cycles,
    anomaly_cycles,
    n_value: int,
    seed: int,
    aggregation: str,
    adaptation_info: dict[str, float] | None = None,
) -> dict[str, Any]:
    calibration_scores = model.score_cycles(calibration_cycles)
    healthy_scores = model.score_cycles(healthy_cycles)
    anomaly_scores = model.score_cycles(anomaly_cycles)
    info = conformal_threshold_info(calibration_scores, alpha=B)
    threshold = float(info.strict_threshold)
    healthy_pred = healthy_scores > threshold
    anomaly_pred = anomaly_scores > threshold
    fp = int(healthy_pred.sum())
    tn = int(len(healthy_pred) - fp)
    tp = int(anomaly_pred.sum())
    fn = int(len(anomaly_pred) - tp)
    recall = float(tp / (tp + fn))
    fpr = float(fp / (fp + tn))
    bounds = certify_operating_point(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall_target=R0,
        fpr_budget=B,
        joint_confidence=CONFIDENCE,
    )
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
    )
    bottleneck = classify_bottleneck(
        oracle=oracle,
        deployed_recall=recall,
        deployed_fpr=fpr,
        recall_lower=bounds.recall_lower,
        fpr_upper=bounds.fpr_upper,
        recall_target=R0,
        fpr_budget=B,
    )
    row: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "cycle_score_aggregation": aggregation,
        "variant": variant,
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "model_seed": MODEL_SEED,
        "calibration_size": len(calibration_scores),
        "healthy_eval_count": len(healthy_scores),
        "anomaly_eval_count": len(anomaly_scores),
        "threshold": threshold,
        "conformal_rank": int(info.raw_rank),
        "conformal_regime": (
            "infinite" if not info.finite_sample_feasible
            else "maximum" if info.threshold_is_maximum
            else "submaximum"
        ),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "false_positive_rate": fpr,
        "recall_lower": float(bounds.recall_lower),
        "fpr_upper": float(bounds.fpr_upper),
        "empirical_success": bool(recall >= R0 and fpr <= B),
        "certified_success": bool(bounds.certified),
        "certification_valid": True,
        "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": str(bottleneck.bottleneck_label),
    }
    if adaptation_info:
        row.update(adaptation_info)
    return row


def _evaluate_online(
    *,
    model: M2N2Adapter,
    threshold: float,
    healthy_cycles,
    anomaly_cycles,
    n_value: int,
    seed: int,
    aggregation: str,
) -> dict[str, Any]:
    stream = sorted(
        list(healthy_cycles) + list(anomaly_cycles),
        key=lambda c: int(c.episode_id),
    )
    ids, scores, online_info = model.online_score_and_adapt(stream)
    score_by_id = {int(i): float(s) for i, s in zip(ids, scores)}
    healthy_scores = np.asarray([score_by_id[int(c.episode_id)] for c in healthy_cycles])
    anomaly_scores = np.asarray([score_by_id[int(c.episode_id)] for c in anomaly_cycles])
    fp = int(np.sum(healthy_scores > threshold))
    tn = int(len(healthy_scores) - fp)
    tp = int(np.sum(anomaly_scores > threshold))
    fn = int(len(anomaly_scores) - tp)
    recall = float(tp / (tp + fn))
    fpr = float(fp / (fp + tn))
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=B,
        recall_target=R0,
    )
    label = (
        "representation_limited"
        if not oracle.empirically_feasible
        else "online_operating_point_limited"
        if not (recall >= R0 and fpr <= B)
        else "empirically_successful_uncertified_online"
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "cycle_score_aggregation": aggregation,
        "variant": "M2N2-Online",
        "commissioning_size": int(n_value),
        "seed": int(seed),
        "model_seed": MODEL_SEED,
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_count": len(healthy_scores),
        "anomaly_eval_count": len(anomaly_scores),
        "threshold": float(threshold),
        "conformal_rank": 100,
        "conformal_regime": "maximum",
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "false_positive_rate": fpr,
        "recall_lower": np.nan,
        "fpr_upper": np.nan,
        "empirical_success": bool(recall >= R0 and fpr <= B),
        "certified_success": False,
        "certification_valid": False,
        "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": label,
        **online_info,
    }


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (aggregation, variant, n_value), g in df.groupby(
        ["cycle_score_aggregation", "variant", "commissioning_size"],
        sort=True,
    ):
        rows.append({
            "cycle_score_aggregation": aggregation,
            "variant": variant,
            "commissioning_size": int(n_value),
            "n_runs": int(len(g)),
            "mean_recall": float(g.recall.mean()),
            "mean_fpr": float(g.false_positive_rate.mean()),
            "mean_auroc": float(g.auroc.mean()),
            "mean_oracle_recall_at_fpr_budget": float(g.oracle_recall_at_fpr_budget.mean()),
            "oracle_feasibility_rate": float(g.oracle_empirically_feasible.astype(bool).mean()),
            "empirical_success_rate": float(g.empirical_success.astype(bool).mean()),
            "certified_success_rate": float(g.certified_success.astype(bool).mean()),
            "certification_valid_rate": float(g.certification_valid.astype(bool).mean()),
            "representation_limited_rate": float((g.bottleneck_label == "representation_limited").mean()),
            "calibration_limited_rate": float((g.bottleneck_label == "calibration_limited").mean()),
            "certification_limited_rate": float((g.bottleneck_label == "certification_limited").mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["cycle_score_aggregation", "variant", "commissioning_size"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", default=",".join(map(str, COMMISSIONING_GRID)))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--skip-online", action="store_true")
    parser.add_argument(
        "--cycle-score-aggregation",
        choices=("q99", "mean"),
        default="q99",
        help="Execution-level aggregation of M2N2 timestep scores. q99 is primary; mean is the predeclared sensitivity.",
    )
    args = parser.parse_args()

    n_values = _parse_ints(args.n_values)
    seeds = _parse_ints(args.seeds)
    if not set(n_values).issubset(COMMISSIONING_GRID):
        raise ValueError(f"n-values must be a subset of {COMMISSIONING_GRID}")
    if not set(seeds).issubset(SEEDS):
        raise ValueError("seeds must be a subset of 0..19")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    aggregation = str(args.cycle_score_aggregation)
    cycle_score_quantile = 0.99 if aggregation == "q99" else None

    # Keep the primary q99 outputs in the historical location and write the
    # mean sensitivity separately so a sensitivity run cannot overwrite the
    # already-completed primary E6 results.
    output_dir = (
        BASE_OUTPUT_DIR
        if aggregation == "q99"
        else PROJECT_ROOT / "outputs" / "e6_m2n2_mean_sensitivity"
    )
    seed_results_path = output_dir / "e6_seed_results.csv"
    summary_path = output_dir / "e6_summary.csv"
    manifest_path = output_dir / "e6_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    source_cycles = sorted(
        [c for c in cycles if (not c.anomaly and c.setting == SOURCE_SETTING)],
        key=lambda c: c.episode_id,
    )
    if not source_cycles:
        raise RuntimeError("No healthy source cycles found")

    standardizer = ChannelStandardizer.fit(source_cycles)
    cfg = M2N2Config(
        seed=MODEL_SEED,
        cycle_score_quantile=cycle_score_quantile,
    )
    print(
        f"E6: aggregation={aggregation}; training one source M2N2 MLP on "
        f"{len(source_cycles)} healthy source cycles ({args.device})"
    )
    source_model = M2N2Adapter(
        num_channels=len(source_cycles[0].columns),
        standardizer=standardizer,
        config=cfg,
        device=args.device,
    ).fit_source(source_cycles)
    print(f"E6: source pseudo-normal mask threshold={source_model.source_mask_threshold_:.8g}")

    rows: list[dict[str, Any]] = []
    total = len(n_values) * len(seeds)
    done = 0
    for n_value in n_values:
        for seed in seeds:
            done += 1
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=n_value,
                commissioning_seed=seed,
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=HEALTHY_EVAL_SIZE,
                maximum_commissioning_size=MAX_COMMISSIONING_SIZE,
            )

            offline = source_model.clone()
            rows.append(_evaluate_frozen(
                variant="M2N2-Offline",
                model=offline,
                calibration_cycles=split.target_calibration,
                healthy_cycles=split.target_normal_evaluation,
                anomaly_cycles=split.target_anomaly_evaluation,
                n_value=n_value,
                seed=seed,
                aggregation=aggregation,
            ))

            adapted = source_model.clone()
            adapt_info = adapted.adapt_cycles(split.target_commissioning)
            adapted_row = _evaluate_frozen(
                variant="M2N2-CommissioningAdapt",
                model=adapted,
                calibration_cycles=split.target_calibration,
                healthy_cycles=split.target_normal_evaluation,
                anomaly_cycles=split.target_anomaly_evaluation,
                n_value=n_value,
                seed=seed,
                aggregation=aggregation,
                adaptation_info=adapt_info,
            )
            rows.append(adapted_row)

            if (
                not args.skip_online
                and n_value == ONLINE_DIAGNOSTIC_N
                and seed in ONLINE_DIAGNOSTIC_SEEDS
            ):
                online = adapted.clone()
                rows.append(_evaluate_online(
                    model=online,
                    threshold=float(adapted_row["threshold"]),
                    healthy_cycles=split.target_normal_evaluation,
                    anomaly_cycles=split.target_anomaly_evaluation,
                    n_value=n_value,
                    seed=seed,
                    aggregation=aggregation,
                ))

            _atomic_csv(pd.DataFrame(rows), seed_results_path)
            print(
                f"E6 {done}/{total}: aggregation={aggregation} N={n_value} seed={seed} "
                f"adapt recall={adapted_row['recall']:.3f} "
                f"FPR={adapted_row['false_positive_rate']:.3f} "
                f"AUROC={adapted_row['auroc']:.3f}"
            )

    results = pd.DataFrame(rows)
    summary = _summary(results)
    _atomic_csv(summary, summary_path)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "paper": "Kim, Park & Choo, When Model Meets New Normals, AAAI 2024",
        "official_repository": "carrtesy/M2N2",
        "purpose": "modern adaptation baseline under COLDSTART commissioning protocol",
        "source_setting": 72,
        "target_setting": 73,
        "commissioning_grid": list(n_values),
        "commissioning_seeds": list(seeds),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_eval_size": HEALTHY_EVAL_SIZE,
        "false_alert_budget": B,
        "recall_target": R0,
        "joint_confidence": CONFIDENCE,
        "model_seed": MODEL_SEED,
        "device": args.device,
        "cycle_score_aggregation": aggregation,
        "cycle_score_quantile": cycle_score_quantile,
        "m2n2_config": cfg.__dict__,
        "variants": {
            "M2N2-Offline": "source-trained MLP control; target calibration only; frozen at evaluation",
            "M2N2-CommissioningAdapt": "M2N2 masked self-training on target commissioning executions; frozen before target calibration/evaluation",
            "M2N2-Online": "continued label-blind adaptation during evaluation; empirical/oracle diagnostic only; no exact fixed-detector certification claim",
        },
        "aggregation_policy": {
            "q99": "primary: 99th percentile of channel-mean timestep reconstruction errors",
            "mean": "predeclared sensitivity: arithmetic mean of channel-mean timestep reconstruction errors",
        },
        "online_diagnostic": {
            "N": ONLINE_DIAGNOSTIC_N,
            "seeds": list(ONLINE_DIAGNOSTIC_SEEDS),
        },
        "no_leakage": "anomaly labels never used for training/adaptation/calibration; online labels used only after scoring for metrics",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote E6 outputs to {output_dir}")


if __name__ == "__main__":
    main()
