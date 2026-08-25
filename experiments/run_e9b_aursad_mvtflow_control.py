from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.aursad_loader import (
    AURSAD_MEASURED_SIGNAL_COLUMNS,
    DEFAULT_DATA_PATH,
    DEFAULT_INVENTORY_PATH,
    load_cycles,
)
from src.base_detector import deterministic_conformal_threshold
from src.certification import certify_operating_point
from src.mvtflow_adapter import (
    OFFICIAL_CONFIG,
    classify_e2_bottleneck,
    prepare_mvtflow_data,
    score_mvtflow,
    train_mvtflow,
)
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = PROJECT_ROOT / "reports" / "aursad" / "coldstart_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "e9b_mvtflow"
SEED_RESULTS_PATH = OUTPUT_DIR / "e9b_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e9b_summary.csv"
PER_CLASS_PATH = OUTPUT_DIR / "e9b_per_class_recall.csv"
NSTAR_PATH = OUTPUT_DIR / "e9b_nstar.csv"
MANIFEST_PATH = OUTPUT_DIR / "e9b_manifest.json"

PROTOCOL_VERSION = "coldstart-e9b-aursad-mvtflow-v1"
DEFAULT_GRID = (10, 25, 50, 100, 250, 500)
DEFAULT_SEEDS = tuple(range(20))
CALIBRATION_SIZE = 199
HEALTHY_EVAL_SIZE = 368
ANOMALY_EVAL_SIZE = 625
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
Q0 = 0.80
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not vals or len(vals) != len(set(vals)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated integers")
    return vals


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_partition(name: str, partition: str) -> pd.DataFrame:
    path = PROTOCOL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing E9 protocol file: {path}")
    df = pd.read_csv(path)
    required = {"sample_nr", "partition", "label", "label_name", "selection_rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if set(df["partition"].astype(str)) != {partition}:
        raise ValueError(f"{name} contains unexpected partition labels")
    for col in ("sample_nr", "label", "selection_rank"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(np.int64)
    return df


def _load_protocol() -> dict[str, pd.DataFrame]:
    tables = {
        "commissioning": _load_partition("commissioning_ids.csv", "commissioning"),
        "calibration": _load_partition("calibration_ids.csv", "calibration"),
        "healthy": _load_partition("healthy_eval_ids.csv", "healthy_eval"),
        "anomaly": _load_partition("anomaly_eval_ids.csv", "anomaly_eval"),
    }
    com = tables["commissioning"]
    for col in ("seed", "commissioning_n"):
        if col not in com.columns:
            raise ValueError(f"commissioning_ids.csv missing {col}")
        com[col] = pd.to_numeric(com[col], errors="raise").astype(np.int64)
    tables["calibration"] = tables["calibration"].sort_values("selection_rank").reset_index(drop=True)
    tables["healthy"] = tables["healthy"].sort_values("selection_rank").reset_index(drop=True)
    tables["anomaly"] = tables["anomaly"].sort_values("sample_nr").reset_index(drop=True)
    if len(tables["calibration"]) < CALIBRATION_SIZE:
        raise ValueError("E9B requires at least 199 calibration executions")
    if len(tables["healthy"]) != HEALTHY_EVAL_SIZE:
        raise ValueError(f"Expected {HEALTHY_EVAL_SIZE} healthy eval executions")
    if len(tables["anomaly"]) != ANOMALY_EVAL_SIZE:
        raise ValueError(f"Expected {ANOMALY_EVAL_SIZE} anomaly executions")
    fixed = [
        set(tables["calibration"]["sample_nr"].astype(int)),
        set(tables["healthy"]["sample_nr"].astype(int)),
        set(tables["anomaly"]["sample_nr"].astype(int)),
    ]
    for i in range(len(fixed)):
        for j in range(i + 1, len(fixed)):
            if fixed[i] & fixed[j]:
                raise RuntimeError("Fixed AURSAD partitions overlap")
    return tables


def _commissioning_ids(df: pd.DataFrame, seed: int, n_value: int) -> list[int]:
    rows = df[(df["seed"] == seed) & (df["commissioning_n"] == n_value)].sort_values("selection_rank")
    ids = rows["sample_nr"].astype(int).tolist()
    if len(ids) != n_value or len(set(ids)) != n_value:
        raise ValueError(f"seed={seed}, N={n_value}: invalid commissioning membership")
    return ids


def _load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SEED_RESULTS_PATH)
    versions = set(df.get("protocol_version", pd.Series(dtype=str)).dropna().astype(str))
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(f"Incompatible checkpoint versions: {sorted(versions)}")
    return df


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_value, g in df.groupby("commissioning_size", sort=True):
        rows.append({
            "detector": "TargetOnly-MVTFlow-AURSAD",
            "commissioning_size": int(n_value),
            "n_seeds": int(len(g)),
            "mean_recall": float(g.recall.mean()),
            "mean_fpr": float(g.false_positive_rate.mean()),
            "mean_auroc": float(g.auroc.mean()),
            "mean_oracle_recall_at_fpr_budget": float(g.oracle_recall_at_fpr_budget.mean()),
            "oracle_feasibility_rate": float(g.oracle_empirically_feasible.astype(bool).mean()),
            "empirical_success_rate": float(g.empirical_success.astype(bool).mean()),
            "certified_success_rate": float(g.certified_success.astype(bool).mean()),
            "representation_limited_rate": float((g.bottleneck_label == "representation_limited").mean()),
            "calibration_limited_rate": float((g.bottleneck_label == "calibration_limited").mean()),
            "certification_limited_rate": float((g.bottleneck_label == "certification_limited").mean()),
            "certified_rate": float((g.bottleneck_label == "certified").mean()),
            "mean_seedwise_recall_lower": float(g.recall_lower.mean()),
            "mean_seedwise_fpr_upper": float(g.fpr_upper.mean()),
        })
    return pd.DataFrame(rows).sort_values("commissioning_size").reset_index(drop=True)


def _nstar(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("oracle_feasibility_rate", "empirical_success_rate", "certified_success_rate"):
        passing = summary[summary[metric] >= Q0].sort_values("commissioning_size")
        rows.append({
            "criterion": metric,
            "q0": Q0,
            "n_star": int(passing.iloc[0].commissioning_size) if not passing.empty else np.nan,
            "censored": bool(passing.empty),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values", type=_parse_int_csv, default=DEFAULT_GRID)
    parser.add_argument("--seeds", type=_parse_int_csv, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    grid = tuple(int(v) for v in args.n_values)
    seeds = tuple(int(v) for v in args.seeds)
    if any(v not in DEFAULT_GRID for v in grid):
        raise ValueError(f"n-values must be subset of {DEFAULT_GRID}")
    if any(v not in DEFAULT_SEEDS for v in seeds):
        raise ValueError("seeds must be in 0..19")
    if not DEFAULT_DATA_PATH.exists():
        raise FileNotFoundError(f"AURSAD HDF5 missing: {DEFAULT_DATA_PATH}")
    if not DEFAULT_INVENTORY_PATH.exists():
        raise FileNotFoundError(f"AURSAD inventory missing: {DEFAULT_INVENTORY_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for p in (SEED_RESULTS_PATH, SUMMARY_PATH, PER_CLASS_PATH, NSTAR_PATH, MANIFEST_PATH):
            if p.exists():
                p.unlink()

    tables = _load_protocol()
    cal_ids = tables["calibration"].iloc[:CALIBRATION_SIZE]["sample_nr"].astype(int).tolist()
    healthy_ids = tables["healthy"]["sample_nr"].astype(int).tolist()
    anomaly_ids = tables["anomaly"]["sample_nr"].astype(int).tolist()

    required_ids = set(cal_ids) | set(healthy_ids) | set(anomaly_ids)
    for seed in seeds:
        for n_value in grid:
            required_ids.update(_commissioning_ids(tables["commissioning"], seed, n_value))

    print(f"Loading {len(required_ids)} AURSAD executions with {len(AURSAD_MEASURED_SIGNAL_COLUMNS)} measured signals...")
    cycles = load_cycles(
        DEFAULT_DATA_PATH,
        episode_ids=sorted(required_ids),
        inventory_path=DEFAULT_INVENTORY_PATH,
        signal_columns=AURSAD_MEASURED_SIGNAL_COLUMNS,
    )
    by_id = {int(c.episode_id): c for c in cycles}
    if len(by_id) != len(required_ids):
        raise RuntimeError("AURSAD raw-cycle loading did not return every requested execution")

    checkpoint = _load_checkpoint()
    rows = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed = {
        (int(r.commissioning_size), int(r.seed))
        for r in checkpoint.itertuples(index=False)
    } if not checkpoint.empty else set()
    class_df = pd.read_csv(PER_CLASS_PATH) if PER_CLASS_PATH.exists() and not args.overwrite else pd.DataFrame()
    class_rows = class_df.to_dict("records") if not class_df.empty else []

    total = len(grid) * len(seeds)
    counter = sum((n, s) in completed for s in seeds for n in grid)
    for seed in seeds:
        for n_value in grid:
            key = (n_value, seed)
            if key in completed:
                continue
            counter += 1
            print(f"E9B {counter}/{total} | N={n_value} seed={seed} | 70 epochs", flush=True)
            commission_ids = _commissioning_ids(tables["commissioning"], seed, n_value)
            groups = [set(commission_ids), set(cal_ids), set(healthy_ids), set(anomaly_ids)]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        raise RuntimeError(f"Leakage detected for N={n_value}, seed={seed}")

            commissioning = [by_id[i] for i in commission_ids]
            calibration = [by_id[i] for i in cal_ids]
            healthy = [by_id[i] for i in healthy_ids]
            anomaly = [by_id[i] for i in anomaly_ids]
            prepared = prepare_mvtflow_data(commissioning, calibration, healthy, anomaly)
            model_seed = 42_000 + seed * 1_000 + n_value
            model, nf_module, device = train_mvtflow(
                prepared,
                project_root=PROJECT_ROOT,
                seed=model_seed,
                device_name=args.device,
            )
            calibration_scores = score_mvtflow(model, nf_module, device, prepared.calibration)
            healthy_scores = score_mvtflow(model, nf_module, device, prepared.healthy_eval)
            anomaly_scores = score_mvtflow(model, nf_module, device, prepared.anomaly_eval)

            threshold, rank, regime = deterministic_conformal_threshold(calibration_scores, alpha=FALSE_ALERT_BUDGET)
            hp = healthy_scores > threshold
            ap = anomaly_scores > threshold
            fp, tn = int(hp.sum()), int((~hp).sum())
            tp, fn = int(ap.sum()), int((~ap).sum())
            cert = certify_operating_point(
                tp=tp, fn=fn, fp=fp, tn=tn,
                recall_target=RECALL_TARGET,
                fpr_budget=FALSE_ALERT_BUDGET,
                joint_confidence=JOINT_CONFIDENCE,
            )
            oracle = empirical_oracle_feasibility(
                healthy_scores, anomaly_scores,
                false_alert_budget=FALSE_ALERT_BUDGET,
                recall_target=RECALL_TARGET,
            )
            empirical_success = bool(cert.recall >= RECALL_TARGET and cert.fpr <= FALSE_ALERT_BUDGET)
            bottleneck = classify_e2_bottleneck(
                oracle_feasible=bool(oracle.empirically_feasible),
                empirical_success=empirical_success,
                certified_success=bool(cert.certified),
            )
            row = {
                "protocol_version": PROTOCOL_VERSION,
                "detector": "TargetOnly-MVTFlow-AURSAD",
                "commissioning_size": n_value,
                "seed": seed,
                "model_seed": model_seed,
                "device": str(device),
                "calibration_size": CALIBRATION_SIZE,
                "healthy_eval_count": HEALTHY_EVAL_SIZE,
                "anomaly_eval_count": ANOMALY_EVAL_SIZE,
                "n_signals": int(prepared.n_signals),
                "target_length": int(prepared.target_length),
                "threshold": float(threshold),
                "conformal_rank": int(rank),
                "conformal_regime": str(regime),
                "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "recall": float(cert.recall),
                "false_positive_rate": float(cert.fpr),
                "recall_lower": float(cert.recall_lower),
                "fpr_upper": float(cert.fpr_upper),
                "empirical_success": empirical_success,
                "certified_success": bool(cert.certified),
                "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
                "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
                "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
                "oracle_empirically_feasible": bool(oracle.empirically_feasible),
                "bottleneck_label": bottleneck,
            }
            rows.append(row)
            completed.add(key)

            pred_by_id = {int(i): bool(p) for i, p in zip(anomaly_ids, ap)}
            for (label, label_name), g in tables["anomaly"].groupby(["label", "label_name"], sort=True):
                ids = g["sample_nr"].astype(int).tolist()
                preds = np.asarray([pred_by_id[i] for i in ids], dtype=bool)
                class_rows.append({
                    "protocol_version": PROTOCOL_VERSION,
                    "detector": "TargetOnly-MVTFlow-AURSAD",
                    "commissioning_size": n_value,
                    "seed": seed,
                    "label": int(label),
                    "label_name": str(label_name),
                    "execution_count": int(len(ids)),
                    "recall": float(preds.mean()),
                    "descriptive_only": bool(int(label) == 4),
                })

            _atomic_csv(pd.DataFrame(rows).sort_values(["commissioning_size", "seed"]), SEED_RESULTS_PATH)
            _atomic_csv(pd.DataFrame(class_rows).sort_values(["commissioning_size", "seed", "label"]), PER_CLASS_PATH)

            del model, prepared, calibration_scores, healthy_scores, anomaly_scores
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    results = pd.DataFrame(rows)
    summary = _summary(results)
    _atomic_csv(summary, SUMMARY_PATH)
    _atomic_csv(_nstar(summary), NSTAR_PATH)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "strong healthy-only raw-time-series representation control for AURSAD E9",
        "detector": "TargetOnly-MVTFlow-AURSAD",
        "method_source": "official vorausrobotik MVT-Flow architecture reused through src.mvtflow_adapter",
        "fairness_constraint": "training uses only target healthy commissioning executions; no anomaly labels used for training, model selection, or calibration",
        "signal_policy": "48 AURSAD measured robot signals; excludes labels, timestamps, runtime-state, and hard-coded control variables",
        "signal_columns": list(AURSAD_MEASURED_SIGNAL_COLUMNS),
        "commissioning_grid": list(DEFAULT_GRID),
        "commissioning_seeds": list(DEFAULT_SEEDS),
        "calibration_size": CALIBRATION_SIZE,
        "healthy_evaluation_size": HEALTHY_EVAL_SIZE,
        "anomaly_evaluation_size": ANOMALY_EVAL_SIZE,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "q0": Q0,
        "official_hyperparameters": OFFICIAL_CONFIG,
        "normalization": "per-signal mean/std fitted on target commissioning timepoints only",
        "padding": "target length equals maximum commissioning-cycle length; truncate and right-pad zeros",
        "score": "MVT-Flow negative log-likelihood score: 0.5*sum(z^2)-jacobian",
        "calibration_rule": "strict deterministic split conformal",
        "alarm_rule": "score > threshold",
        "oracle_is_diagnostic_only": True,
        "damaged_thread_label_4_descriptive_only": True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("\nE9B complete")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
