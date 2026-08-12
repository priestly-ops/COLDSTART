"""Numerical audit: A1(gamma=0) vs A0 TargetPCA at N=10, seed=0.

Reproduces both runners' exact fit/score math and traces the 12 diagnostic
steps, then computes counterfactual A1 scores under A0 choices without
modifying either runner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aligned_race_a0 import AlignedRACEA0Detector
from src.shared_projector_a1 import SharedProjectorA1Detector
from src.feature_extractor import extract_feature_batch
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles

DEFAULT_VORAUS = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit_a1_vs_targetpca"
N_TARGET = 10
SEED = 0
ALPHA = 0.01
K_MAX = 16
BETA = 0.5
SCORE_FLOOR = 1e-8


def _episode_feature_map(cycles):
    batch = extract_feature_batch(cycles)
    return {int(e): batch.features[i] for i, e in enumerate(batch.episode_ids)}


def _matrix_for(cycles, features_by_episode):
    return np.vstack([features_by_episode[int(c.episode_id)] for c in cycles])


def _oracle_threshold_at_fpr(normal_scores, alpha):
    rank = int(np.ceil((1.0 - alpha) * len(normal_scores))) - 1
    rank = int(np.clip(rank, 0, len(normal_scores) - 1))
    return float(np.sort(normal_scores)[rank])


def _metrics(normal, anomaly, calibration, threshold):
    labels = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomaly), dtype=int)]
    scores = np.r_[normal, anomaly]
    oracle_threshold = _oracle_threshold_at_fpr(normal, ALPHA)
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "pauroc_0_01_standardized": float(roc_auc_score(labels, scores, max_fpr=ALPHA)),
        "oracle_recall_1pct": float(np.mean(anomaly > oracle_threshold)),
        "oracle_fpr_1pct": float(np.mean(normal > oracle_threshold)),
        "conformal_recall": float(np.mean(anomaly > threshold)),
        "conformal_fpr": float(np.mean(normal > threshold)),
        "threshold": float(threshold),
        "threshold_rank": int(np.sum(calibration <= threshold)),
    }


def _safe_variance_a0(x, axis=0, floor=1e-8):
    var = np.var(x, axis=axis, ddof=1 if x.shape[axis] > 1 else 0)
    finite = var[np.isfinite(var) & (var > 0.0)]
    adaptive = 0.01 * float(np.median(finite)) if finite.size else floor
    return np.maximum(var, max(floor, adaptive))


def _safe_variance_a1(x, floor=1e-8):
    variance = np.var(x, axis=0, ddof=1 if x.shape[0] > 1 else 0)
    finite = variance[np.isfinite(variance) & (variance > floor)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    return np.where(
        np.isfinite(variance) & (variance > floor),
        variance,
        max(fallback, floor),
    )


def _score_with(yt, basis, variance_fn, weighting, floor=SCORE_FLOOR):
    """Score transformed features in `basis` with `variance_fn` and `weighting`.

    weighting "A0" -> 0.5*subspace + 0.25*residual (TargetPCA weights=1,
    target_modes=0). weighting "A1" -> 0.5*subspace + 0.5*residual.
    """
    z = yt @ basis
    mode_center = np.median(z, axis=0)
    mode_variance = variance_fn(z)
    mode_energy = (z - mode_center) ** 2 / (mode_variance + floor)
    subspace = np.mean(mode_energy, axis=1)

    residual = yt - z @ basis.T
    residual_center = np.median(residual, axis=0)
    residual_variance = variance_fn(residual)
    residual_energy = (residual - residual_center) ** 2 / (residual_variance + floor)
    residual_score = np.mean(residual_energy, axis=1)

    if weighting == "A0":
        final = 0.5 * subspace + 0.25 * residual_score
    else:
        final = 0.5 * subspace + 0.5 * residual_score
    return {
        "subspace": subspace,
        "residual": residual_score,
        "final": final,
    }


def _fmt_diff(name, a, b):
    d = np.abs(a - b)
    return {
        "name": name,
        "max_abs": float(np.max(d)) if d.size else 0.0,
        "mean_abs": float(np.mean(d)) if d.size else 0.0,
        "rms": float(np.sqrt(np.mean(d**2))) if d.size else 0.0,
        "n_differ_gt_0": int(np.sum(d > 0)),
        "n_differ_gt_1e-8": int(np.sum(d > 1e-8)),
        "n_differ_gt_1e-4": int(np.sum(d > 1e-4)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/9] loading cycles...", flush=True)
    cycles = load_cycles(DEFAULT_VORAUS)
    print(f"  loaded {len(cycles)} cycles", flush=True)

    print("[2/9] extracting features...", flush=True)
    features_by_episode = _episode_feature_map(cycles)
    print(f"  features for {len(features_by_episode)} episodes", flush=True)

    print("[3/9] building N=10 seed=0 split...", flush=True)
    split = create_frozen_evaluation_split(cycles, N_TARGET, SEED)
    source = _matrix_for(split.source_train, features_by_episode)
    target = _matrix_for(split.target_commissioning, features_by_episode)
    calibration = _matrix_for(split.target_calibration, features_by_episode)
    normal = _matrix_for(split.target_normal_evaluation, features_by_episode)
    anomaly = _matrix_for(split.target_anomaly_evaluation, features_by_episode)
    print(
        f"  source={source.shape} target={target.shape} cal={calibration.shape} "
        f"normal={normal.shape} anomaly={anomaly.shape}",
        flush=True,
    )

    print("[4/9] fitting A0 TargetPCA and A1(gamma=0)...", flush=True)
    common = {"k_max": K_MAX, "beta": BETA, "random_state": SEED}
    a0 = AlignedRACEA0Detector(mode="target_pca", **common)
    a1 = SharedProjectorA1Detector(gamma=0.0, **common)
    a0.fit(source, target)
    a1.fit(source, target)

    report: dict[str, object] = {}
    report["setup"] = {
        "n_target": N_TARGET,
        "seed": SEED,
        "k_A0": a0.target_principal_vectors_.shape[1],
        "k_A1": a1.shared_basis_.shape[1],
    }

    # ---- Step 1: episode IDs ----
    def ids(cycles_tuple):
        return [int(c.episode_id) for c in cycles_tuple]

    step1 = {
        "source_count": len(ids(split.source_train)),
        "target_count": len(ids(split.target_commissioning)),
        "calibration_count": len(ids(split.target_calibration)),
        "normal_count": len(ids(split.target_normal_evaluation)),
        "anomaly_count": len(ids(split.target_anomaly_evaluation)),
        "source_ids": ids(split.source_train)[:8],
        "target_ids": ids(split.target_commissioning),
        "calibration_ids": ids(split.target_calibration)[:8],
        "normal_ids": ids(split.target_normal_evaluation)[:8],
        "anomaly_ids": ids(split.target_anomaly_evaluation)[:8],
    }
    report["step1_episode_ids"] = step1
    print("[5/9] step1 episode IDs:", step1, flush=True)

    # ---- Step 2: feature matrices identical by construction ----
    report["step2_feature_matrices"] = {
        "identical_source_target_cal_normal_anomaly": True,
        "n_features": int(source.shape[1]),
    }

    # ---- Step 3: mu_T and scale_T ----
    mu_diff = _fmt_diff("mu_T", a0.target_center_, a1.target_center_)
    scale_diff = _fmt_diff("scale_T", a0.target_scale_, a1.target_scale_)
    report["step3_center_scale"] = {"mu_T": mu_diff, "scale_T": scale_diff}
    print("[6/9] step3 mu/scale:", json.dumps({"mu_T": mu_diff, "scale_T": scale_diff}), flush=True)

    yt_a0 = a0._target_transform(target)
    yt_a1 = a1._target_transform(target)
    yt_diff = _fmt_diff("yt", yt_a0, yt_a1)
    report["step3b_transforms"] = {"yt": yt_diff}

    # ---- Step 4: projector P_T ----
    ut_a0 = a0.target_pca_basis_
    ut_a1 = a1.target_pca_basis_
    shared_basis = a1.shared_basis_
    pt_a0 = ut_a0 @ ut_a0.T
    pt_a1_ut = ut_a1 @ ut_a1.T
    pt_a1_eigh = shared_basis @ shared_basis.T
    projector_diff = {
        "P_A0_minus_P_A1_eigh_fro": float(np.linalg.norm(pt_a0 - pt_a1_eigh, ord="fro")),
        "P_A0_minus_P_A1_ut_fro": float(np.linalg.norm(pt_a0 - pt_a1_ut, ord="fro")),
        "P_A0_minus_P_A1_eigh_max": float(np.max(np.abs(pt_a0 - pt_a1_eigh))),
    }
    basis_diff = {
        "ut_A0_minus_ut_A1_fro": float(np.linalg.norm(ut_a0 - ut_a1, ord="fro")),
        "ut_A0_minus_eigh_fro": float(np.linalg.norm(ut_a0 - shared_basis, ord="fro")),
    }
    report["step4_projector"] = {"projector_diffs": projector_diff, "basis_diffs": basis_diff}
    print("[7/9] step4 projector:", json.dumps(projector_diff), flush=True)

    # ---- Step 5: basis coordinates ----
    z_a0 = yt_a0 @ ut_a0
    z_a1 = yt_a1 @ shared_basis
    coord_diff = _fmt_diff("coordinates", z_a0, z_a1)
    rotation = ut_a0.T @ shared_basis
    report["step5_coordinates"] = {
        "coord_diff": coord_diff,
        "rotation_offdiag_max": float(np.max(np.abs(rotation - np.diag(np.diag(rotation))))),
        "rotation_orthonormality_err": float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[1]), ord="fro")),
    }

    # ---- Steps 6-7: variances ----
    var0 = _fmt_diff("mode_variance", a0.mode_variance_, a1.mode_variance_)
    var1 = _fmt_diff("residual_variance", a0.residual_variance_, a1.residual_variance_)
    report["step6_7_variances"] = {"mode_variance": var0, "residual_variance": var1}
    print("[8/9] step6/7 variances:", json.dumps({"mode": var0, "residual": var1}), flush=True)

    # ---- Steps 8-10: scores ----
    comp_a0 = a0.score_components(calibration)
    comp_a1 = a1.score_components(calibration)
    sub_diff = _fmt_diff("subspace_score_cal", comp_a0["shared_score"], comp_a1["subspace_score"])
    res_diff = _fmt_diff("residual_score_cal", comp_a0["orthogonal_score"], comp_a1["residual_score"])
    final_diff = _fmt_diff("final_score_cal", comp_a0["final_score"], comp_a1["final_score"])
    report["steps8_10_scores"] = {
        "subspace_score_cal": sub_diff,
        "residual_score_cal": res_diff,
        "final_score_cal": final_diff,
    }
    print("[9/9] step8-10 scores:", json.dumps({"sub": sub_diff, "res": res_diff, "final": final_diff}), flush=True)

    # ---- Step 11: calibration threshold ----
    cal_scores_a0 = a0.score_samples(calibration)
    cal_scores_a1 = a1.score_samples(calibration)
    a0.calibrate_from_scores(cal_scores_a0)
    a1.calibrate_from_scores(cal_scores_a1)
    normal_scores_a0 = a0.score_samples(normal)
    anomaly_scores_a0 = a0.score_samples(anomaly)
    normal_scores_a1 = a1.score_samples(normal)
    anomaly_scores_a1 = a1.score_samples(anomaly)
    report["step11_threshold"] = {
        "A0_threshold": float(a0.threshold_),
        "A1_threshold": float(a1.threshold_),
        "ratio_A1_over_A0": float(a1.threshold_ / a0.threshold_),
        "A0_max_cal_score": float(np.max(cal_scores_a0)),
        "A1_max_cal_score": float(np.max(cal_scores_a1)),
    }

    # ---- Step 12: metrics ----
    metrics_a0 = _metrics(normal_scores_a0, anomaly_scores_a0, cal_scores_a0, a0.threshold_)
    metrics_a1 = _metrics(normal_scores_a1, anomaly_scores_a1, cal_scores_a1, a1.threshold_)
    report["step12_metrics"] = {"A0_TargetPCA": metrics_a0, "A1_gamma0": metrics_a1}

    # ---- Key diagnostic: |P diff| vs |S diff| ----
    final_diff_norm = float(np.sqrt(np.mean((comp_a0["final_score"] - comp_a1["final_score"]) ** 2)))
    report["diagnostic"] = {
        "P_fro_norm": projector_diff["P_A0_minus_P_A1_eigh_fro"],
        "score_rms_cal": final_diff_norm,
        "score_mean_abs_cal": final_diff["mean_abs"],
    }

    # ---- Counterfactuals ----
    # Proper counterfactual scoring: fit statistics are computed on TARGET only.
    def cf_scores(yt_fit, yt_eval, basis, variance_fn, weighting):
        z_fit = yt_fit @ basis
        z_eval = yt_eval @ basis
        mode_center = np.median(z_fit, axis=0)
        mode_variance = variance_fn(z_fit)
        mode_energy = (z_eval - mode_center) ** 2 / (mode_variance + SCORE_FLOOR)
        subspace = np.mean(mode_energy, axis=1)

        residual_fit = yt_fit - z_fit @ basis.T
        residual_eval = yt_eval - z_eval @ basis.T
        residual_center = np.median(residual_fit, axis=0)
        residual_variance = variance_fn(residual_fit)
        residual_energy = (residual_eval - residual_center) ** 2 / (residual_variance + SCORE_FLOOR)
        residual_score = np.mean(residual_energy, axis=1)

        if weighting == "A0":
            final = 0.5 * subspace + 0.25 * residual_score
        else:
            final = 0.5 * subspace + 0.5 * residual_score
        return final

    yt_fit = {"A0": yt_a0, "A1": yt_a1}
    yt_eval = {
        "A0_cal": a0._target_transform(calibration),
        "A1_cal": a1._target_transform(calibration),
        "A0_norm": a0._target_transform(normal),
        "A1_norm": a1._target_transform(normal),
        "A0_anom": a0._target_transform(anomaly),
        "A1_anom": a1._target_transform(anomaly),
    }

    cf_defs = {
        "A1_native": ("eigh", "A1", "A1", "A1"),
        "A0_native": ("ut", "A0", "A0", "A0"),
        "CF_ut_basis_A1_rest": ("ut", "A1", "A1", "A1"),
        "CF_ut_basis_A0_var_A1_weight": ("ut", "A1", "A0", "A1"),
        "CF_ut_basis_A1_var_A0_weight": ("ut", "A1", "A1", "A0"),
        "CF_ut_basis_A0_var_A0_weight": ("ut", "A1", "A0", "A0"),
        "CF_eigh_A0_var_A0_weight": ("eigh", "A1", "A0", "A0"),
        "CF_eigh_A1_var_A0_weight": ("eigh", "A1", "A1", "A0"),
        "CF_A0_scale_ut_A0_floor_A0_weight": ("ut", "A0", "A0", "A0"),
    }
    # user counterfactual: A1 projector but A0 PCA basis, A0 floor, A0 weight ->
    # basis=ut (A0 PCA), transform=A0 (same target-only transform), var=A0, weight=A0
    # which is exactly A0_native; keep both names for the record.

    for name, (basis_name, transform_name, var_name, weight_name) in cf_defs.items():
        basis = bases[basis_name]
        variance_fn = variance_fns[var_name]
        fit_yt = yt_fit[transform_name]
        cal_scores = cf_scores(fit_yt, yt_eval[f"{transform_name}_cal"], basis, variance_fn, weight_name)
        norm_scores = cf_scores(fit_yt, yt_eval[f"{transform_name}_norm"], basis, variance_fn, weight_name)
        anom_scores = cf_scores(fit_yt, yt_eval[f"{transform_name}_anom"], basis, variance_fn, weight_name)
        threshold = np.sort(cal_scores)[
            int(np.clip(int(np.ceil((len(cal_scores) + 1) * (1.0 - ALPHA))) - 1, 0, len(cal_scores) - 1))
        ]
        metrics = _metrics(norm_scores, anom_scores, cal_scores, float(threshold))
        cf_rows.append(
            {
                "variant": name,
                "basis": basis_name,
                "transform": transform_name,
                "variance": var_name,
                "weighting": weight_name,
                **metrics,
            }
        )
    report["counterfactuals"] = cf_rows

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    cf_df = pd.DataFrame(cf_rows)
    cf_path = OUTPUT_DIR / "counterfactual_metrics.csv"
    cf_df.to_csv(cf_path, index=False)

    print("\n===== REPORT =====", flush=True)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f"\nwrote {report_path}", flush=True)
    print(f"wrote {cf_path}", flush=True)


if __name__ == "__main__":
    main()

