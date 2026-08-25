from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.bottleneck_decomposition import classify_bottleneck
from src.certification import certify_operating_point
from src.detectors import TargetOnlyDetector
from src.e10_calibration import (
    deterministic_split_conformal,
    evt_pot,
    randomized_smoothed_conformal,
)
from src.evaluation import fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import empirical_oracle_feasibility, probability_of_superiority
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
E0_RESULTS_PATH = PROJECT_ROOT / "outputs" / "e0_certification" / "e0_seed_results.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "e10_calibration_robustness"
SCORE_DIR = OUTPUT_DIR / "scores"
SEED_RESULTS_PATH = OUTPUT_DIR / "e10_seed_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "e10_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "e10_manifest.json"

PROTOCOL_VERSION = "coldstart-e10-calibration-robustness-v1"
E0_PROTOCOL_VERSION = "coldstart-e0-certification-v1"
GLOBAL_SEED = 42
EVALUATION_SEED = 42
CALIBRATION_PREFIX_SEED = 271828
RANDOMIZATION_BASE_SEED = 910000
SEEDS = tuple(range(20))
COMMISSIONING_GRID = (25, 50, 100)
CALIBRATION_POOL_SIZE = 100
CALIBRATION_PREFIX_SIZES = (50, 99, 100)
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
JOINT_CONFIDENCE = 0.95
POT_TAIL_FRACTION = 0.20
POT_MIN_EXCEEDANCES = 10
METHODS = ("deterministic_conformal", "randomized_conformal", "evt_pot")

np.random.seed(GLOBAL_SEED)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _load_e0() -> pd.DataFrame:
    if not E0_RESULTS_PATH.exists():
        raise FileNotFoundError(f"E10 requires frozen E0 results: {E0_RESULTS_PATH}")
    df = pd.read_csv(E0_RESULTS_PATH)
    if "protocol_version" not in df.columns:
        raise ValueError("E0 results missing protocol_version")
    versions = set(df.protocol_version.dropna().astype(str))
    if versions != {E0_PROTOCOL_VERSION}:
        raise ValueError(f"Unexpected E0 protocol versions: {sorted(versions)}")
    return df[df.detector.astype(str) == "TargetOnly"].copy()


def _e0_row(e0: pd.DataFrame, n_value: int, seed: int) -> pd.Series:
    g = e0[(e0.commissioning_size == n_value) & (e0.seed == seed)]
    if len(g) != 1:
        raise RuntimeError(f"Expected one E0 TargetOnly row for N={n_value}, seed={seed}; got {len(g)}")
    return g.iloc[0]


def _score_path(n_value: int, seed: int) -> Path:
    return SCORE_DIR / f"targetonly_N{n_value}_seed{seed}.npz"


def _fit_or_load_scores(cycles, n_value: int, seed: int) -> dict[str, np.ndarray]:
    path = _score_path(n_value, seed)
    if path.exists():
        with np.load(path) as z:
            return {k: np.asarray(z[k]) for k in z.files}

    split = create_frozen_evaluation_split(
        cycles=cycles,
        commissioning_size=n_value,
        commissioning_seed=seed,
        evaluation_seed=EVALUATION_SEED,
        calibration_size=CALIBRATION_POOL_SIZE,
        normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
        maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
    )
    source_raw, _ = extract_feature_matrix(split.source_train)
    target_raw, _ = extract_feature_matrix(split.target_commissioning)
    calibration_raw, _ = extract_feature_matrix(split.target_calibration)
    healthy_raw, _ = extract_feature_matrix(split.target_normal_evaluation)
    anomaly_raw, _ = extract_feature_matrix(split.target_anomaly_evaluation)

    detector, preprocessor, _, _ = fit_detector(
        detector_name="TargetOnly",
        detector_factory=lambda: TargetOnlyDetector(false_alert_budget=FALSE_ALERT_BUDGET),
        source_raw=source_raw,
        target_raw=target_raw,
    )
    calibration_scores = detector.score_samples(preprocessor.transform(calibration_raw))
    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))

    # Fixed random permutation of the frozen 100-cycle calibration pool. This
    # makes M=50 and M=99 true nested prefixes while M=100 remains exactly E0.
    prefix_rng = np.random.default_rng(CALIBRATION_PREFIX_SEED)
    order = prefix_rng.permutation(CALIBRATION_POOL_SIZE)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)[order]

    payload = {
        "calibration_scores": calibration_scores,
        "healthy_scores": np.asarray(healthy_scores, dtype=np.float64),
        "anomaly_scores": np.asarray(anomaly_scores, dtype=np.float64),
        "calibration_episode_ids": np.asarray([c.episode_id for c in split.target_calibration], dtype=np.int64)[order],
        "healthy_episode_ids": np.asarray([c.episode_id for c in split.target_normal_evaluation], dtype=np.int64),
        "anomaly_episode_ids": np.asarray([c.episode_id for c in split.target_anomaly_evaluation], dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def _evaluate_method(
    *,
    method: str,
    calibration_scores: np.ndarray,
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    randomization_seed: int,
):
    if method == "deterministic_conformal":
        h = deterministic_split_conformal(calibration_scores, healthy_scores, alpha=FALSE_ALERT_BUDGET)
        a = deterministic_split_conformal(calibration_scores, anomaly_scores, alpha=FALSE_ALERT_BUDGET)
    elif method == "randomized_conformal":
        # One predeclared reproducible deployment realization. Healthy and anomaly
        # random variates are independent, neither depends on labels/scores.
        h = randomized_smoothed_conformal(
            calibration_scores, healthy_scores, alpha=FALSE_ALERT_BUDGET,
            rng=np.random.default_rng(randomization_seed),
        )
        a = randomized_smoothed_conformal(
            calibration_scores, anomaly_scores, alpha=FALSE_ALERT_BUDGET,
            rng=np.random.default_rng(randomization_seed + 1),
        )
    elif method == "evt_pot":
        h = evt_pot(
            calibration_scores, healthy_scores, alpha=FALSE_ALERT_BUDGET,
            tail_fraction=POT_TAIL_FRACTION, minimum_exceedances=POT_MIN_EXCEEDANCES,
        )
        a = evt_pot(
            calibration_scores, anomaly_scores, alpha=FALSE_ALERT_BUDGET,
            tail_fraction=POT_TAIL_FRACTION, minimum_exceedances=POT_MIN_EXCEEDANCES,
        )
    else:
        raise ValueError(method)

    fit_ok = bool(h.fit_ok and a.fit_ok)
    hp = np.asarray(h.predictions, dtype=bool)
    ap = np.asarray(a.predictions, dtype=bool)
    fp, tn = int(hp.sum()), int((~hp).sum())
    tp, fn = int(ap.sum()), int((~ap).sum())
    bounds = certify_operating_point(
        tp=tp, fn=fn, fp=fp, tn=tn,
        recall_target=RECALL_TARGET,
        fpr_budget=FALSE_ALERT_BUDGET,
        joint_confidence=JOINT_CONFIDENCE,
    )
    oracle = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )
    empirical_success = bool(bounds.recall >= RECALL_TARGET and bounds.fpr <= FALSE_ALERT_BUDGET)
    if not fit_ok:
        bottleneck_label = "calibration_fit_failed"
    else:
        b = classify_bottleneck(
            oracle=oracle,
            deployed_recall=bounds.recall,
            deployed_fpr=bounds.fpr,
            recall_lower=bounds.recall_lower,
            fpr_upper=bounds.fpr_upper,
            recall_target=RECALL_TARGET,
            fpr_budget=FALSE_ALERT_BUDGET,
        )
        bottleneck_label = str(b.bottleneck_label)

    threshold = float(h.threshold)
    return {
        "fit_ok": fit_ok,
        "threshold": threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": float(bounds.recall),
        "fpr": float(bounds.fpr),
        "recall_lower": float(bounds.recall_lower),
        "fpr_upper": float(bounds.fpr_upper),
        "empirical_success": empirical_success,
        "certified_success": bool(bounds.certified),
        "auroc": float(probability_of_superiority(healthy_scores, anomaly_scores)),
        "oracle_recall_at_fpr_budget": float(oracle.max_recall_at_fpr_budget),
        "oracle_fpr_at_max_recall": float(oracle.fpr_at_max_recall),
        "oracle_empirically_feasible": bool(oracle.empirically_feasible),
        "bottleneck_label": bottleneck_label,
        "expected_recall_over_randomization": float(np.mean(a.expected_alarm_probability)),
        "expected_fpr_over_randomization": float(np.mean(h.expected_alarm_probability)),
        "diagnostic": {**h.diagnostic, **{f"anomaly_{k}": v for k, v in a.diagnostic.items()}},
    }


def _assert_e0_reproduction(result: dict[str, Any], e0_row: pd.Series) -> None:
    for key, e0_key in (("recall", "recall"), ("fpr", "false_positive_rate")):
        if not np.isclose(float(result[key]), float(e0_row[e0_key]), atol=1e-12, rtol=0.0):
            raise RuntimeError(f"E10 M=100 deterministic does not reproduce E0 {key}")
    for key in ("tp", "fn", "fp", "tn"):
        if int(result[key]) != int(e0_row[key]):
            raise RuntimeError(f"E10 M=100 deterministic does not reproduce E0 {key}")


def _load_checkpoint() -> pd.DataFrame:
    if not SEED_RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SEED_RESULTS_PATH)
    versions = set(df.protocol_version.dropna().astype(str)) if "protocol_version" in df else set()
    if versions and versions != {PROTOCOL_VERSION}:
        raise ValueError(f"Incompatible E10 checkpoint versions: {sorted(versions)}")
    return df


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["method", "calibration_size", "commissioning_size"], sort=True)
    for (method, m, n), g in grouped:
        rows.append({
            "method": method,
            "calibration_size": int(m),
            "commissioning_size": int(n),
            "n_seeds": int(g.seed.nunique()),
            "fit_ok_rate": float(g.fit_ok.astype(bool).mean()),
            "mean_recall": float(g.recall.mean()),
            "mean_fpr": float(g.fpr.mean()),
            "mean_expected_recall_over_randomization": float(g.expected_recall_over_randomization.mean()),
            "mean_expected_fpr_over_randomization": float(g.expected_fpr_over_randomization.mean()),
            "mean_auroc": float(g.auroc.mean()),
            "mean_oracle_recall_at_fpr_budget": float(g.oracle_recall_at_fpr_budget.mean()),
            "oracle_feasibility_rate": float(g.oracle_empirically_feasible.astype(bool).mean()),
            "empirical_success_rate": float(g.empirical_success.astype(bool).mean()),
            "certified_success_rate": float(g.certified_success.astype(bool).mean()),
            "representation_limited_rate": float((g.bottleneck_label == "representation_limited").mean()),
            "calibration_limited_rate": float((g.bottleneck_label == "calibration_limited").mean()),
            "certification_limited_rate": float((g.bottleneck_label == "certification_limited").mean()),
            "calibration_fit_failed_rate": float((g.bottleneck_label == "calibration_fit_failed").mean()),
        })
    return pd.DataFrame(rows).sort_values(["calibration_size", "commissioning_size", "method"]).reset_index(drop=True)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    e0 = _load_e0()
    cycles = load_cycles(path=DATASET_PATH, signal_set="measured")
    checkpoint = _load_checkpoint()
    rows = checkpoint.to_dict("records") if not checkpoint.empty else []
    completed = {
        (str(r.method), int(r.calibration_size), int(r.commissioning_size), int(r.seed))
        for r in checkpoint.itertuples(index=False)
    } if not checkpoint.empty else set()

    total = len(METHODS) * len(CALIBRATION_PREFIX_SIZES) * len(COMMISSIONING_GRID) * len(SEEDS)
    counter = len(completed)
    print("=" * 86)
    print("COLDSTART E10: CALIBRATION ROBUSTNESS ON FROZEN TARGETONLY SCORES")
    print("=" * 86)
    print(f"N: {COMMISSIONING_GRID}; calibration prefixes: {CALIBRATION_PREFIX_SIZES}")
    print(f"Methods: {METHODS}")
    print("M=100 deterministic conformal must reproduce frozen E0 exactly.")
    print("EVT/POT is a parametric robustness analysis, not a conformal guarantee.")
    print("=" * 86)

    for n_value in COMMISSIONING_GRID:
        for seed in SEEDS:
            scores = _fit_or_load_scores(cycles, n_value, seed)
            healthy_scores = np.asarray(scores["healthy_scores"], dtype=np.float64)
            anomaly_scores = np.asarray(scores["anomaly_scores"], dtype=np.float64)
            full_cal = np.asarray(scores["calibration_scores"], dtype=np.float64)

            for m in CALIBRATION_PREFIX_SIZES:
                cal = full_cal[:m]
                for method_index, method in enumerate(METHODS):
                    key = (method, m, n_value, seed)
                    if key in completed:
                        continue
                    randomization_seed = (
                        RANDOMIZATION_BASE_SEED
                        + n_value * 10000 + seed * 100 + m * 3 + method_index
                    )
                    result = _evaluate_method(
                        method=method,
                        calibration_scores=cal,
                        healthy_scores=healthy_scores,
                        anomaly_scores=anomaly_scores,
                        randomization_seed=randomization_seed,
                    )
                    if method == "deterministic_conformal" and m == 100:
                        _assert_e0_reproduction(result, _e0_row(e0, n_value, seed))

                    diagnostic = result.pop("diagnostic")
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "detector": "TargetOnly",
                        "method": method,
                        "commissioning_size": n_value,
                        "seed": seed,
                        "calibration_size": m,
                        "false_alert_budget": FALSE_ALERT_BUDGET,
                        "recall_target": RECALL_TARGET,
                        "healthy_eval_count": len(healthy_scores),
                        "anomaly_eval_count": len(anomaly_scores),
                        "randomization_seed": randomization_seed if method == "randomized_conformal" else np.nan,
                        **result,
                        "diagnostic_json": json.dumps(diagnostic, sort_keys=True),
                    }
                    rows.append(row)
                    completed.add(key)
                    counter += 1
                    _atomic_csv(
                        pd.DataFrame(rows).sort_values(["calibration_size", "commissioning_size", "seed", "method"]),
                        SEED_RESULTS_PATH,
                    )
                    print(f"E10 {counter}/{total} | N={n_value} seed={seed} M={m} method={method}", flush=True)

    results = pd.DataFrame(rows)
    summary = _summary(results)
    _atomic_csv(summary, SUMMARY_PATH)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "test whether the observed calibration bottleneck is specific to deterministic conformal granularity",
        "detector": "TargetOnly only",
        "score_policy": "fit once per N/commissioning seed; identical cached score vectors reused across all calibration methods and M",
        "commissioning_grid": list(COMMISSIONING_GRID),
        "commissioning_seeds": list(SEEDS),
        "calibration_pool_size": CALIBRATION_POOL_SIZE,
        "calibration_prefix_sizes": list(CALIBRATION_PREFIX_SIZES),
        "calibration_prefix_seed": CALIBRATION_PREFIX_SEED,
        "healthy_evaluation_size": HEALTHY_EVALUATION_SIZE,
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "joint_confidence": JOINT_CONFIDENCE,
        "methods": {
            "deterministic_conformal": "frozen strict split-conformal order statistic",
            "randomized_conformal": "smoothed split-conformal randomized p-values with fixed independent U draws; expected metrics also reported",
            "evt_pot": {
                "description": "GPD peaks-over-threshold extrapolation; parametric diagnostic, not finite-sample conformal",
                "tail_fraction": POT_TAIL_FRACTION,
                "minimum_exceedances": POT_MIN_EXCEEDANCES,
            },
        },
        "primary_safeguard": "M=100 deterministic rows must exactly reproduce E0 TargetOnly counts/recall/FPR",
        "oracle": "diagnostic only; never used to choose thresholds or tune calibration methods",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("\nE10 complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
