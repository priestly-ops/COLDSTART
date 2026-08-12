"""Five-seed, N={10,25} mechanism pilot for Aligned RACE A0 on voraus-AD.

Run from project root on Windows PowerShell:

    python experiments/run_race_a0_pilot.py

Optional overrides:

    python experiments/run_race_a0_pilot.py --seeds 0 1 2 3 4 --n 10 25

Outputs:
    outputs/race_a0_pilot_seed_results.csv
    outputs/race_a0_pilot_summary.csv
    outputs/race_a0_principal_angles.csv

This is deliberately a mechanism pilot, NOT the final 20-seed commissioning table.
No anomaly label is used during detector fitting, source compatibility estimation,
or conformal calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligned_race_a0 import AlignedRACEA0Detector
from src.feature_extractor import extract_feature_matrix
from src.split_generator import create_experiment_split
from src.voraus_loader import load_cycles


GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FALSE_ALERT_BUDGET = 0.01
CALIBRATION_SIZE = 100
NORMAL_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100

# Frozen A0 probe defaults. Change only by declaring a new protocol version.
PROTOCOL_VERSION = "race-a0-v2-principal-vector-soft-weights"
K_MAX = 16
BETA = 0.50
BETA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
LAMBDA_WEIGHT = 0.25
DIRECTION_MIN_COS2 = 0.20
GLOBAL_ALIGNMENT_MIN = 0.20


MODEL_MODES = {
    "A0TargetOnly": "target_only",
    "TargetPCA": "target_pca",
    "RawSourcePCA": "raw_source_pca",
    "RandomSubspace": "random_subspace",
    "FeaturePermutedSource": "feature_permuted",
    "WeightPermutedRACE": "weight_permuted",
    "RACE-A0": "aligned",
}


def _conformal_rank(n: int, alpha: float) -> int:
    return min(n, int(np.ceil((n + 1) * (1.0 - alpha))))


def _oracle_threshold_at_fpr(normal_scores: np.ndarray, alpha: float) -> float:
    """Lowest observed healthy threshold giving <= alpha empirical FPR.

    With strict prediction `score > threshold`, allowing m false positives means
    thresholding at the (m+1)-th largest healthy score. This is an ORACLE
    diagnostic only because the held-out healthy evaluation set sets the cutoff.
    """
    s = np.sort(np.asarray(normal_scores, dtype=np.float64))
    allowed_fp = int(np.floor(alpha * len(s) + 1e-12))
    index = max(0, len(s) - allowed_fp - 1)
    return float(s[index])


def _ranking_metrics(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float]:
    y = np.concatenate(
        [
            np.zeros(len(normal_scores), dtype=np.int64),
            np.ones(len(anomaly_scores), dtype=np.int64),
        ]
    )
    scores = np.concatenate([normal_scores, anomaly_scores])

    auroc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))
    # sklearn's max_fpr returns standardized partial AUC. Name it explicitly.
    pauroc_001_std = float(roc_auc_score(y, scores, max_fpr=0.01))

    oracle_threshold = _oracle_threshold_at_fpr(normal_scores, 0.01)
    oracle_fpr = float(np.mean(normal_scores > oracle_threshold))
    oracle_recall = float(np.mean(anomaly_scores > oracle_threshold))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "pauroc_0_01_standardized": pauroc_001_std,
        "oracle_threshold_1pct": oracle_threshold,
        "oracle_fpr_1pct": oracle_fpr,
        "oracle_recall_1pct": oracle_recall,
    }


def _build_detector_with_beta(mode: str, seed: int, beta: float) -> AlignedRACEA0Detector:
    return AlignedRACEA0Detector(
        k_max=K_MAX,
        beta=beta,
        lambda_weight=LAMBDA_WEIGHT,
        direction_min_cos2=DIRECTION_MIN_COS2,
        global_alignment_min=GLOBAL_ALIGNMENT_MIN,
        mode=mode,  # type: ignore[arg-type]
        false_alert_budget=FALSE_ALERT_BUDGET,
        random_state=10_000 + seed,
    )


def _build_detector(mode: str, seed: int) -> AlignedRACEA0Detector:
    return _build_detector_with_beta(mode, seed, BETA)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _assert_disjoint(split) -> None:
    split.verify_no_overlap()
    groups = {
        "source": split.source_train,
        "commission": split.target_commissioning,
        "calibration": split.target_calibration,
        "healthy_eval": split.target_normal_evaluation,
        "anomaly_eval": split.target_anomaly_evaluation,
    }
    sets = {name: {c.episode_id for c in rows} for name, rows in groups.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise RuntimeError(f"Leakage: {a} overlaps {b}: {sorted(overlap)[:5]}")


def run_one(
    *,
    detector_name: str,
    mode: str,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    calibration_raw: np.ndarray,
    normal_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    n: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model = _build_detector(mode, seed)
    model.fit(source_raw, target_raw)

    # Ranking / low-FPR geometry uses frozen model before calibration.
    normal_components = model.score_components(normal_raw)
    anomaly_components = model.score_components(anomaly_raw)
    normal_scores = normal_components["final_score"]
    anomaly_scores = anomaly_components["final_score"]
    ranking = _ranking_metrics(normal_scores, anomaly_scores)

    # Strict deployment layer: disjoint healthy calibration set.
    calibration_scores = model.score_samples(calibration_raw)
    model.calibrate_from_scores(calibration_scores)
    if model.threshold_ is None:
        raise RuntimeError("Calibration failed to produce threshold.")

    conformal_fpr = float(np.mean(normal_scores > model.threshold_))
    conformal_recall = float(np.mean(anomaly_scores > model.threshold_))
    conformal_success = bool(
        conformal_recall >= 0.90 and conformal_fpr <= FALSE_ALERT_BUDGET
    )

    diag = model.diagnostics_
    if diag is None:
        raise RuntimeError("A0 diagnostics are missing.")

    row: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "detector": detector_name,
        "mode": mode,
        "commissioning_size": n,
        "seed": seed,
        "n_source": diag.n_source,
        "n_features": diag.n_features,
        "k_effective": diag.k_effective,
        "n_shared_directions": diag.n_shared_directions,
        "alignment_mean_cos2": diag.alignment_mean_cos2,
        "alignment_min_cos2": diag.alignment_min_cos2,
        "alignment_max_cos2": diag.alignment_max_cos2,
        "angle_distance": diag.angle_distance,
        "global_gate_open": diag.global_gate_open,
        "fallback": diag.fallback,
        "fallback_reason": diag.fallback_reason,
        "beta": BETA,
        "lambda_weight": LAMBDA_WEIGHT,
        "direction_min_cos2": DIRECTION_MIN_COS2,
        "global_alignment_min": GLOBAL_ALIGNMENT_MIN,
        **ranking,
        "calibration_size": len(calibration_scores),
        "conformal_rank": _conformal_rank(len(calibration_scores), FALSE_ALERT_BUDGET),
        "conformal_threshold": float(model.threshold_),
        "conformal_fpr": conformal_fpr,
        "conformal_recall": conformal_recall,
        "conformal_success": conformal_success,
        "shared_score_mean": float(np.mean(np.r_[normal_components["shared_score"], anomaly_components["shared_score"]])),
        "shared_score_std": float(np.std(np.r_[normal_components["shared_score"], anomaly_components["shared_score"]])),
        "target_specific_score_mean": float(np.mean(np.r_[normal_components["target_specific_score"], anomaly_components["target_specific_score"]])),
        "target_specific_score_std": float(np.std(np.r_[normal_components["target_specific_score"], anomaly_components["target_specific_score"]])),
        "final_score_mean": float(np.mean(np.r_[normal_scores, anomaly_scores])),
        "final_score_std": float(np.std(np.r_[normal_scores, anomaly_scores])),
        "effective_weight_mass": float(np.sum(diag.effective_weights)),
    }

    angle_rows: list[dict[str, object]] = []
    for j, (s, raw_w, eff_w) in enumerate(
        zip(diag.singular_values, diag.raw_cos2_weights, diag.effective_weights),
        start=1,
    ):
        angle_rows.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "detector": detector_name,
                "commissioning_size": n,
                "seed": seed,
                "principal_index": j,
                "singular_value": s,
                "principal_angle_degrees": float(np.degrees(np.arccos(np.clip(s, 0, 1)))),
                "cos2_weight_raw": raw_w,
                "weight_effective": eff_w,
            }
        )

    return row, angle_rows


def score_sensitivity_rows(
    *,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    normal_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    n: int,
    seed: int,
    beta: float = 1.0,
) -> list[dict[str, object]]:
    real = _build_detector_with_beta("aligned", seed, beta).fit(source_raw, target_raw)
    permuted = _build_detector_with_beta("weight_permuted", seed, beta).fit(source_raw, target_raw)
    eval_raw = np.vstack([normal_raw, anomaly_raw])
    real_components = real.score_components(eval_raw)
    permuted_components = permuted.score_components(eval_raw)
    shared_diff = np.abs(real_components["shared_score"] - permuted_components["shared_score"])
    final_diff = np.abs(real_components["final_score"] - permuted_components["final_score"])
    return [
        {
            "protocol_version": PROTOCOL_VERSION,
            "commissioning_size": n,
            "seed": seed,
            "beta": beta,
            "mean_abs_shared_score_difference": float(np.mean(shared_diff)),
            "median_abs_shared_score_difference": float(np.median(shared_diff)),
            "max_abs_shared_score_difference": float(np.max(shared_diff)),
            "shared_score_spearman": _spearman(
                real_components["shared_score"], permuted_components["shared_score"]
            ),
            "final_score_spearman": _spearman(
                real_components["final_score"], permuted_components["final_score"]
            ),
            "allclose_shared": bool(
                np.allclose(
                    real_components["shared_score"],
                    permuted_components["shared_score"],
                    rtol=1e-10,
                    atol=1e-12,
                )
            ),
            "allclose_final": bool(
                np.allclose(
                    real_components["final_score"],
                    permuted_components["final_score"],
                    rtol=1e-10,
                    atol=1e-12,
                )
            ),
        }
    ]


def beta_sensitivity_rows(
    *,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    normal_raw: np.ndarray,
    anomaly_raw: np.ndarray,
    n: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = np.r_[np.zeros(len(normal_raw), dtype=int), np.ones(len(anomaly_raw), dtype=int)]
    for beta in BETA_GRID:
        baseline_scores: np.ndarray | None = None
        for detector_name, mode in [
            ("RACE-A0", "aligned"),
            ("WeightPermutedRACE", "weight_permuted"),
        ]:
            model = _build_detector_with_beta(mode, seed, beta).fit(source_raw, target_raw)
            normal_scores = model.score_samples(normal_raw)
            anomaly_scores = model.score_samples(anomaly_raw)
            scores = np.r_[normal_scores, anomaly_scores]
            metrics = _ranking_metrics(normal_scores, anomaly_scores)
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "detector": detector_name,
                    "commissioning_size": n,
                    "seed": seed,
                    "beta": beta,
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "pauroc_1pct": metrics["pauroc_0_01_standardized"],
                    "oracle_recall_1pct": metrics["oracle_recall_1pct"],
                    "mean_absolute_score_difference": float(np.nan)
                    if baseline_scores is None
                    else float(np.mean(np.abs(scores - baseline_scores))),
                    "score_correlation": float(np.nan)
                    if baseline_scores is None
                    else float(np.corrcoef(scores, baseline_scores)[0, 1]),
                    "label_count": int(len(labels)),
                }
            )
            if baseline_scores is None:
                baseline_scores = scores
    return rows


def directional_contribution_rows(
    *,
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    eval_raw: np.ndarray,
    n: int,
    seed: int,
    max_rows: int = 25,
) -> list[dict[str, object]]:
    model = _build_detector("aligned", seed).fit(source_raw, target_raw)
    components = model.score_components(eval_raw[:max_rows])
    rows: list[dict[str, object]] = []
    for row_idx in range(components["mode_energy"].shape[0]):
        for j in range(components["mode_energy"].shape[1]):
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "commissioning_size": n,
                    "seed": seed,
                    "eval_row_index": row_idx,
                    "principal_index": j + 1,
                    "mode_energy": float(components["mode_energy"][row_idx, j]),
                    "contribution": float(components["shared_contributions"][row_idx, j]),
                    "target_only_contribution": float(
                        components["target_only_contributions"][row_idx, j]
                    ),
                    "weight": float(model.effective_weights_[j]),
                    "raw_cos2_weight": float(model.raw_cos2_weights_[j]),
                }
            )
    return rows


def _dataset_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "auroc",
        "auprc",
        "pauroc_0_01_standardized",
        "oracle_recall_1pct",
        "oracle_fpr_1pct",
        "conformal_recall",
        "conformal_fpr",
        "alignment_mean_cos2",
        "n_shared_directions",
    ]
    summary = (
        results.groupby(["detector", "commissioning_size"], sort=True)[metric_columns]
        .agg(["mean", "std", "median"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    success = (
        results.groupby(["detector", "commissioning_size"], sort=True)["conformal_success"]
        .mean()
        .rename("conformal_success_rate")
        .reset_index()
    )
    return summary.merge(success, on=["detector", "commissioning_size"], how="left")


def add_paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Add RACE-A0 paired deltas against TargetPCA and A0TargetOnly."""
    out = results.copy()
    keys = ["commissioning_size", "seed"]
    metrics = ["auroc", "pauroc_0_01_standardized", "oracle_recall_1pct", "conformal_recall"]

    race = out[out["detector"] == "RACE-A0"][keys + metrics].copy()
    for baseline_name in ["TargetPCA", "A0TargetOnly", "FeaturePermutedSource", "WeightPermutedRACE"]:
        base = out[out["detector"] == baseline_name][keys + metrics].copy()
        merged = race.merge(base, on=keys, suffixes=("_race", "_base"))
        for metric in metrics:
            delta_map = {
                (int(r.commissioning_size), int(r.seed)): float(getattr(r, f"{metric}_race") - getattr(r, f"{metric}_base"))
                for r in merged.itertuples(index=False)
            }
            col = f"paired_delta_{metric}_vs_{baseline_name}"
            out[col] = [
                delta_map.get((int(n), int(seed)), np.nan)
                if det == "RACE-A0" else np.nan
                for det, n, seed in zip(out["detector"], out["commissioning_size"], out["seed"])
            ]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n", type=int, nargs="+", default=[10, 25])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}\n"
            "Expected the existing project parquet or pass --data-path."
        )

    print("=" * 80)
    print("ALIGNED RACE A0 MECHANISM PILOT")
    print("=" * 80)
    print(f"Protocol:       {PROTOCOL_VERSION}")
    print(f"Data:           {args.data_path}")
    print(f"N:              {args.n}")
    print(f"Seeds:          {args.seeds}")
    print(f"k_max:          {K_MAX}")
    print(f"beta:           {BETA}")
    print(f"lambda_weight:  {LAMBDA_WEIGHT}")
    print(f"direction gate: cos^2 >= {DIRECTION_MIN_COS2}")
    print(f"global gate:    mean cos^2 >= {GLOBAL_ALIGNMENT_MIN}")
    print("=" * 80)

    print("Loading voraus-AD cycles once...")
    cycles = load_cycles(path=args.data_path, signal_set="measured")
    print(f"Loaded {len(cycles)} cycles.")

    result_rows: list[dict[str, object]] = []
    angle_rows: list[dict[str, object]] = []
    beta_rows: list[dict[str, object]] = []
    score_sensitivity: list[dict[str, object]] = []
    directional_rows: list[dict[str, object]] = []
    total = len(args.seeds) * len(args.n) * len(MODEL_MODES)
    counter = 0

    # Cache source features because source setting is fixed; however split generation
    # is still called for each seed/N to preserve and verify the established protocol.
    for seed in args.seeds:
        for n in args.n:
            if n <= 0 or n > MAXIMUM_COMMISSIONING_SIZE:
                raise ValueError(f"Unsupported N={n}; must be 1..{MAXIMUM_COMMISSIONING_SIZE}.")

            split = create_experiment_split(
                cycles=cycles,
                commissioning_size=n,
                seed=seed,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=NORMAL_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )
            _assert_disjoint(split)

            source_raw, _ = extract_feature_matrix(split.source_train)
            target_raw, _ = extract_feature_matrix(split.target_commissioning)
            calibration_raw, _ = extract_feature_matrix(split.target_calibration)
            normal_raw, _ = extract_feature_matrix(split.target_normal_evaluation)
            anomaly_raw, _ = extract_feature_matrix(split.target_anomaly_evaluation)
            eval_raw = np.vstack([normal_raw, anomaly_raw])

            score_sensitivity.extend(
                score_sensitivity_rows(
                    source_raw=source_raw,
                    target_raw=target_raw,
                    normal_raw=normal_raw,
                    anomaly_raw=anomaly_raw,
                    n=n,
                    seed=seed,
                )
            )
            beta_rows.extend(
                beta_sensitivity_rows(
                    source_raw=source_raw,
                    target_raw=target_raw,
                    normal_raw=normal_raw,
                    anomaly_raw=anomaly_raw,
                    n=n,
                    seed=seed,
                )
            )
            directional_rows.extend(
                directional_contribution_rows(
                    source_raw=source_raw,
                    target_raw=target_raw,
                    eval_raw=eval_raw,
                    n=n,
                    seed=seed,
                )
            )

            for detector_name, mode in MODEL_MODES.items():
                counter += 1
                print(
                    f"[{counter:03d}/{total:03d}] N={n:3d} seed={seed:2d} "
                    f"detector={detector_name}"
                )
                row, angles = run_one(
                    detector_name=detector_name,
                    mode=mode,
                    source_raw=source_raw,
                    target_raw=target_raw,
                    calibration_raw=calibration_raw,
                    normal_raw=normal_raw,
                    anomaly_raw=anomaly_raw,
                    n=n,
                    seed=seed,
                )
                result_rows.append(row)
                angle_rows.extend(angles)
                print(
                    "    "
                    f"align={row['alignment_mean_cos2']:.4f} "
                    f"shared={row['n_shared_directions']} "
                    f"AUROC={row['auroc']:.4f} "
                    f"pAUC01={row['pauroc_0_01_standardized']:.4f} "
                    f"oracleR@1%={row['oracle_recall_1pct']:.4f} "
                    f"confR={row['conformal_recall']:.4f} "
                    f"confFPR={row['conformal_fpr']:.4f}"
                )

    results = pd.DataFrame(result_rows)
    results = add_paired_deltas(results)
    angles = pd.DataFrame(angle_rows)
    beta_sensitivity = pd.DataFrame(beta_rows)
    score_sensitivity_df = pd.DataFrame(score_sensitivity)
    directional = pd.DataFrame(directional_rows)
    summary = summarize(results)

    result_path = OUTPUT_DIR / "race_a0_v2_seed_results.csv"
    summary_path = OUTPUT_DIR / "race_a0_v2_summary.csv"
    angles_path = OUTPUT_DIR / "race_a0_v2_principal_angles.csv"
    beta_path = OUTPUT_DIR / "race_a0_v2_beta_sensitivity.csv"
    score_sensitivity_path = OUTPUT_DIR / "race_a0_v2_score_sensitivity.csv"
    directional_path = OUTPUT_DIR / "race_a0_v2_directional_contributions.csv"
    manifest_path = OUTPUT_DIR / "race_a0_v2_manifest.json"

    results.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    angles.to_csv(angles_path, index=False)
    beta_sensitivity.to_csv(beta_path, index=False)
    score_sensitivity_df.to_csv(score_sensitivity_path, index=False)
    directional.to_csv(directional_path, index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.data_path),
        "dataset_hash_sha256": _dataset_hash(args.data_path),
        "feature_count": int(results["n_features"].max()) if not results.empty else None,
        "source_episode_count": int(results["n_source"].max()) if not results.empty else None,
        "target_episode_counts": sorted(int(v) for v in results["commissioning_size"].unique()),
        "commissioning_sizes": args.n,
        "seeds": args.seeds,
        "k_rule": "min(k_max, N - 2, d, n_source - 1)",
        "beta": BETA,
        "beta_grid": BETA_GRID,
        "lambda_weight": LAMBDA_WEIGHT,
        "scale_floor": "max(1e-6, scale_floor_relative * median_positive_scale, score_floor)",
        "clip_value": 8.0,
        "calibration_alpha": FALSE_ALERT_BUDGET,
        "calibration_size": CALIBRATION_SIZE,
        "git_commit": _git_commit(),
        "output_files": [
            str(result_path),
            str(summary_path),
            str(angles_path),
            str(beta_path),
            str(score_sensitivity_path),
            str(directional_path),
            str(manifest_path),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("PILOT COMPLETE")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved: {result_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {angles_path}")
    print(f"Saved: {beta_path}")
    print(f"Saved: {score_sensitivity_path}")
    print(f"Saved: {directional_path}")
    print(f"Saved: {manifest_path}")
    print("\nPrimary go/no-go checks:")
    print("  1) RACE-A0 > TargetPCA on oracle Recall@1% and pAUROC@1%.")
    print("  2) RACE-A0 > FeaturePermutedSource.")
    print("  3) RACE-A0 > WeightPermutedRACE.")
    print("  4) Positive deltas should repeat across seeds, not come from one outlier.")
    print("  5) Only after mechanism evidence: expand to A1 / 20-seed evaluation.")


if __name__ == "__main__":
    main()
