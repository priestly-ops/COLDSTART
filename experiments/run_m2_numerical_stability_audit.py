"""
M2-v2.3 Numerical-Stability and Score-Attribution Audit
=======================================================

FINAL M2 technical audit for the frozen-evaluation TargetOnly experiment.

Reviewer attack addressed
-------------------------
"The extreme healthy episode and the 0-FP -> 1-FP discontinuity may be a
numerical artifact caused by near-zero-variance engineered features or an
ill-conditioned Mahalanobis covariance estimate."

This script is intentionally bounded. It does NOT add another detector family,
another dataset, another calibration method, or another evaluation split.

It performs:

A. PRIMARY-MODEL DIAGNOSTICS
   - reproduces the frozen M2-v2.1 TargetOnly fit,
   - records raw training feature variances,
   - records which features survive the primary 1e-12 variance filter,
   - records StandardScaler scales,
   - records Ledoit-Wolf precision eigenvalues / condition number,
   - records episode 1710's transformed feature magnitudes,
   - decomposes its squared Mahalanobis score into signed feature contributions.

B. PRE-SPECIFIED NUMERICAL SENSITIVITY
   Refit the SAME TargetOnly + Ledoit-Wolf model under only four variance
   thresholds:

       1e-12  (primary)
       1e-10
       1e-8
       1e-6

   For each N in {50,100} and each commissioning seed in {0,...,19}, recompute:
   - oracle feasibility at 1 allowed healthy FP,
   - oracle feasibility at 0 allowed healthy FP,
   - probability-of-superiority/AUROC,
   - episode 1710 healthy rank,
   - episode 1710 score / healthy median score,
   - maximum healthy episode identity,
   - retained feature count,
   - covariance condition number.

Interpretation
--------------
This is a sensitivity audit only.

If the main geometry persists after increasingly aggressive removal of
low-variance training features, then the reviewer cannot reasonably attribute
M2 solely to the primary variance-filter floor.

If the geometry disappears, M2 must be rewritten as a representation /
conditioning phenomenon rather than a pure calibration-tail phenomenon.

No sensitivity variant replaces the primary detector.

Usage
-----

From the repository root:

    python experiments/run_m2_numerical_stability_audit.py

Optional smoke:

    python experiments/run_m2_numerical_stability_audit.py `
        --seeds 0 `
        --commissioning 100
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.feature_extractor import extract_feature_matrix, make_feature_names
from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import load_cycles


PROTOCOL_VERSION = "m2-numerical-stability-v2.3"

GLOBAL_SEED = 42
EVALUATION_SEED = 42

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "m2_numerical_stability_audit"
)

SEEDS = tuple(range(20))
COMMISSIONING_GRID = (50, 100)

PRIMARY_VARIANCE_THRESHOLD = 1e-12

# Small, pre-specified numerical sensitivity only.
VARIANCE_THRESHOLDS = (
    1e-12,
    1e-10,
    1e-8,
    1e-6,
)

CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100

RECALL_TARGET = 0.90
FALSE_ALERT_BUDGET = 0.01
PRIMARY_ALLOWED_FP = 1

EPISODE_OF_INTEREST = 1710

np.random.seed(GLOBAL_SEED)


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    )


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(
        float(x.strip())
        for x in value.split(",")
        if x.strip()
    )


def _probability_of_superiority(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> float:
    """Empirical P(A>H)+0.5P(A=H), computed independently."""
    h = np.asarray(healthy_scores, dtype=np.float64)
    a = np.asarray(anomaly_scores, dtype=np.float64)

    # Rank-based equivalent; pairwise formulation is clear and exact here.
    greater = (a[:, None] > h[None, :]).sum()
    ties = (a[:, None] == h[None, :]).sum()

    return float(
        (greater + 0.5 * ties)
        / (len(a) * len(h))
    )


def _oracle_counts(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    allowed_fp: int,
    recall_target: float,
) -> dict:
    """Exact empirical retrospective oracle in COUNT space."""
    h = np.asarray(healthy_scores, dtype=np.float64).reshape(-1)
    a = np.asarray(anomaly_scores, dtype=np.float64).reshape(-1)

    if len(h) == 0 or len(a) == 0:
        raise ValueError("Oracle requires non-empty score arrays.")

    if not np.isfinite(h).all() or not np.isfinite(a).all():
        raise ValueError("Oracle received NaN/Inf.")

    unique = np.unique(
        np.concatenate([h, a])
    )

    thresholds = [-np.inf]

    if len(unique):
        thresholds.append(float(unique[0]))

        if len(unique) > 1:
            mids = (
                unique[:-1]
                + (unique[1:] - unique[:-1]) / 2.0
            )
            thresholds.extend(
                float(x) for x in mids
            )

        thresholds.append(float(unique[-1]))

    thresholds.append(np.inf)

    required_tp = int(
        math.ceil(
            recall_target * len(a) - 1e-12
        )
    )

    rows = []

    for t in thresholds:
        fp = int(np.sum(h > t))
        tp = int(np.sum(a > t))

        rows.append(
            {
                "threshold": float(t),
                "fp": fp,
                "tp": tp,
                "fpr": fp / len(h),
                "recall": tp / len(a),
            }
        )

    budget_rows = [
        r
        for r in rows
        if r["fp"] <= allowed_fp
    ]

    best_tp = max(
        r["tp"] for r in budget_rows
    )

    best = [
        r
        for r in budget_rows
        if r["tp"] == best_tp
    ]

    min_fp = min(
        r["fp"] for r in best
    )

    best = [
        r
        for r in best
        if r["fp"] == min_fp
    ]

    selected = max(
        best,
        key=lambda r: r["threshold"],
    )

    target_rows = [
        r
        for r in rows
        if r["tp"] >= required_tp
    ]

    min_fp_for_target = (
        min(r["fp"] for r in target_rows)
        if target_rows
        else len(h) + 1
    )

    return {
        "healthy_count": int(len(h)),
        "anomaly_count": int(len(a)),
        "allowed_fp": int(allowed_fp),
        "required_tp": int(required_tp),
        "max_tp_at_budget": int(best_tp),
        "max_recall_at_budget": float(
            selected["recall"]
        ),
        "fpr_at_selected": float(
            selected["fpr"]
        ),
        "selected_threshold": float(
            selected["threshold"]
        ),
        "min_fp_for_target_recall": int(
            min_fp_for_target
        ),
        "empirically_feasible": bool(
            best_tp >= required_tp
        ),
    }


def _fit_targetonly(
    target_raw: np.ndarray,
    *,
    variance_threshold: float,
) -> dict:
    """Fit the exact TargetOnly preprocessing + LedoitWolf pipeline."""
    target_raw = np.asarray(
        target_raw,
        dtype=np.float64,
    )

    raw_variances = np.var(
        target_raw,
        axis=0,
        ddof=0,
    )

    variance_filter = VarianceThreshold(
        threshold=variance_threshold
    )

    filtered = variance_filter.fit_transform(
        target_raw
    )

    if filtered.shape[1] == 0:
        raise ValueError(
            f"All features removed at threshold={variance_threshold:g}"
        )

    scaler = StandardScaler()
    target_scaled = scaler.fit_transform(
        filtered
    )

    estimator = LedoitWolf(
        assume_centered=False,
        store_precision=True,
    )
    estimator.fit(target_scaled)

    precision = np.asarray(
        estimator.precision_,
        dtype=np.float64,
    )

    covariance = np.asarray(
        estimator.covariance_,
        dtype=np.float64,
    )

    location = np.asarray(
        estimator.location_,
        dtype=np.float64,
    )

    eig_cov = np.linalg.eigvalsh(
        (covariance + covariance.T) / 2.0
    )

    eig_precision = np.linalg.eigvalsh(
        (precision + precision.T) / 2.0
    )

    positive_cov = eig_cov[
        eig_cov > np.finfo(np.float64).eps
    ]

    condition_number = (
        float(
            positive_cov.max()
            / positive_cov.min()
        )
        if len(positive_cov)
        else math.inf
    )

    return {
        "variance_threshold": float(
            variance_threshold
        ),
        "raw_variances": raw_variances,
        "support_mask": variance_filter.get_support(),
        "retained_indices": np.where(
            variance_filter.get_support()
        )[0],
        "scaler": scaler,
        "variance_filter": variance_filter,
        "location": location,
        "covariance": covariance,
        "precision": precision,
        "covariance_eigen_min": float(
            eig_cov.min()
        ),
        "covariance_eigen_max": float(
            eig_cov.max()
        ),
        "precision_eigen_min": float(
            eig_precision.min()
        ),
        "precision_eigen_max": float(
            eig_precision.max()
        ),
        "covariance_condition_number": float(
            condition_number
        ),
        "ledoitwolf_shrinkage": float(
            estimator.shrinkage_
        ),
        "retained_feature_count": int(
            filtered.shape[1]
        ),
        "training_scaled": target_scaled,
    }


def _transform(
    model: dict,
    raw: np.ndarray,
) -> np.ndarray:
    filtered = model[
        "variance_filter"
    ].transform(raw)

    return model[
        "scaler"
    ].transform(filtered)


def _score(
    model: dict,
    scaled: np.ndarray,
) -> np.ndarray:
    centered = (
        scaled
        - model["location"]
    )

    squared = np.einsum(
        "ij,jk,ik->i",
        centered,
        model["precision"],
        centered,
        optimize=True,
    )

    squared = np.maximum(
        squared,
        0.0,
    )

    return np.sqrt(squared)


def _feature_contributions(
    model: dict,
    scaled_row: np.ndarray,
) -> np.ndarray:
    """Signed additive decomposition of squared Mahalanobis distance.

    c_i = delta_i * (P delta)_i
    and sum_i c_i = delta^T P delta.

    Individual c_i may be negative due to covariance cross-terms. This is an
    attribution diagnostic, not an independent per-feature anomaly score.
    """
    delta = (
        np.asarray(
            scaled_row,
            dtype=np.float64,
        )
        - model["location"]
    )

    projected = (
        model["precision"]
        @ delta
    )

    return (
        delta
        * projected
    )


def _rank_descending(
    scores: np.ndarray,
    ids: np.ndarray,
    episode_id: int,
) -> tuple[int | None, float | None]:
    ids = np.asarray(ids).astype(int)
    scores = np.asarray(scores, dtype=np.float64)

    matches = np.where(
        ids == int(episode_id)
    )[0]

    if len(matches) != 1:
        return None, None

    idx = int(matches[0])

    order = np.lexsort(
        (
            ids,
            -scores,
        )
    )

    rank = int(
        np.where(order == idx)[0][0]
        + 1
    )

    return rank, float(scores[idx])


def _pairwise_disjoint(parts: dict[str, tuple]) -> None:
    id_sets = {
        name: {
            int(c.episode_id)
            for c in values
        }
        for name, values in parts.items()
    }

    names = list(id_sets)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = (
                id_sets[a]
                & id_sets[b]
            )

            if overlap:
                raise RuntimeError(
                    f"Leakage between {a} and {b}: "
                    f"{sorted(overlap)[:5]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=SEEDS,
    )

    parser.add_argument(
        "--commissioning",
        type=_parse_int_list,
        default=COMMISSIONING_GRID,
    )

    parser.add_argument(
        "--variance-thresholds",
        type=_parse_float_list,
        default=VARIANCE_THRESHOLDS,
    )

    parser.add_argument(
        "--episode-of-interest",
        type=int,
        default=EPISODE_OF_INTEREST,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycles = load_cycles(
        args.data_path
    )

    # Names are needed for attribution.
    if not cycles:
        raise RuntimeError("Dataset is empty.")

    all_feature_names = make_feature_names(
        cycles[0].columns
    )

    run_rows = []
    feature_rows = []
    contribution_rows = []

    total = (
        len(args.commissioning)
        * len(args.seeds)
        * len(args.variance_thresholds)
    )

    counter = 0

    for n in args.commissioning:
        for seed in args.seeds:

            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=int(n),
                commissioning_seed=int(seed),
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )

            split.verify_no_overlap()

            _pairwise_disjoint(
                {
                    "source": split.source_train,
                    "commissioning": split.target_commissioning,
                    "calibration": split.target_calibration,
                    "healthy_eval": split.target_normal_evaluation,
                    "anomaly_eval": split.target_anomaly_evaluation,
                }
            )

            target_raw, target_ids = extract_feature_matrix(
                split.target_commissioning
            )

            healthy_raw, healthy_ids = extract_feature_matrix(
                split.target_normal_evaluation
            )

            anomaly_raw, anomaly_ids = extract_feature_matrix(
                split.target_anomaly_evaluation
            )

            # Frozen evaluation ID set invariants are recorded later.
            for variance_threshold in args.variance_thresholds:
                counter += 1

                print(
                    f"M2-v2.3 N={n} seed={seed} "
                    f"variance={variance_threshold:g} "
                    f"({counter}/{total})..."
                )

                model = _fit_targetonly(
                    target_raw,
                    variance_threshold=float(
                        variance_threshold
                    ),
                )

                healthy_scaled = _transform(
                    model,
                    healthy_raw,
                )

                anomaly_scaled = _transform(
                    model,
                    anomaly_raw,
                )

                healthy_scores = _score(
                    model,
                    healthy_scaled,
                )

                anomaly_scores = _score(
                    model,
                    anomaly_scaled,
                )

                oracle_1fp = _oracle_counts(
                    healthy_scores,
                    anomaly_scores,
                    allowed_fp=1,
                    recall_target=RECALL_TARGET,
                )

                oracle_0fp = _oracle_counts(
                    healthy_scores,
                    anomaly_scores,
                    allowed_fp=0,
                    recall_target=RECALL_TARGET,
                )

                auc_pairwise = _probability_of_superiority(
                    healthy_scores,
                    anomaly_scores,
                )

                y = np.concatenate(
                    [
                        np.zeros(
                            len(healthy_scores),
                            dtype=int,
                        ),
                        np.ones(
                            len(anomaly_scores),
                            dtype=int,
                        ),
                    ]
                )

                combined_scores = np.concatenate(
                    [
                        healthy_scores,
                        anomaly_scores,
                    ]
                )

                auc_sklearn = float(
                    roc_auc_score(
                        y,
                        combined_scores,
                    )
                )

                if not np.isclose(
                    auc_pairwise,
                    auc_sklearn,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        "Independent AUROC implementations disagree."
                    )

                rank_1710, score_1710 = _rank_descending(
                    healthy_scores,
                    healthy_ids,
                    int(args.episode_of_interest),
                )

                max_idx = int(
                    np.argmax(
                        healthy_scores
                    )
                )

                max_episode_id = int(
                    healthy_ids[max_idx]
                )

                max_score = float(
                    healthy_scores[max_idx]
                )

                healthy_median = float(
                    np.median(
                        healthy_scores
                    )
                )

                raw_variances = model[
                    "raw_variances"
                ]

                retained = model[
                    "retained_indices"
                ]

                retained_variances = (
                    raw_variances[
                        retained
                    ]
                )

                run_rows.append(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "commissioning_size": int(n),
                        "commissioning_seed": int(seed),
                        "evaluation_seed": EVALUATION_SEED,
                        "variance_threshold": float(
                            variance_threshold
                        ),
                        "is_primary_variance_threshold": bool(
                            variance_threshold
                            == PRIMARY_VARIANCE_THRESHOLD
                        ),
                        "retained_feature_count": int(
                            model[
                                "retained_feature_count"
                            ]
                        ),
                        "retained_min_raw_variance": float(
                            retained_variances.min()
                        ),
                        "retained_median_raw_variance": float(
                            np.median(
                                retained_variances
                            )
                        ),
                        "retained_max_raw_variance": float(
                            retained_variances.max()
                        ),
                        "ledoitwolf_shrinkage": float(
                            model[
                                "ledoitwolf_shrinkage"
                            ]
                        ),
                        "covariance_eigen_min": float(
                            model[
                                "covariance_eigen_min"
                            ]
                        ),
                        "covariance_eigen_max": float(
                            model[
                                "covariance_eigen_max"
                            ]
                        ),
                        "covariance_condition_number": float(
                            model[
                                "covariance_condition_number"
                            ]
                        ),
                        "precision_eigen_min": float(
                            model[
                                "precision_eigen_min"
                            ]
                        ),
                        "precision_eigen_max": float(
                            model[
                                "precision_eigen_max"
                            ]
                        ),
                        "auc_probability_superiority": float(
                            auc_pairwise
                        ),
                        "auc_sklearn": float(
                            auc_sklearn
                        ),
                        "oracle_1fp_feasible": bool(
                            oracle_1fp[
                                "empirically_feasible"
                            ]
                        ),
                        "oracle_1fp_max_recall": float(
                            oracle_1fp[
                                "max_recall_at_budget"
                            ]
                        ),
                        "oracle_1fp_min_fp_for_90recall": int(
                            oracle_1fp[
                                "min_fp_for_target_recall"
                            ]
                        ),
                        "oracle_0fp_feasible": bool(
                            oracle_0fp[
                                "empirically_feasible"
                            ]
                        ),
                        "oracle_0fp_max_recall": float(
                            oracle_0fp[
                                "max_recall_at_budget"
                            ]
                        ),
                        "max_healthy_episode_id": int(
                            max_episode_id
                        ),
                        "max_healthy_score": float(
                            max_score
                        ),
                        "healthy_median_score": float(
                            healthy_median
                        ),
                        "max_to_median_ratio": float(
                            max_score
                            / healthy_median
                        )
                        if healthy_median != 0
                        else math.inf,
                        "episode_of_interest": int(
                            args.episode_of_interest
                        ),
                        "episode_of_interest_present": bool(
                            rank_1710 is not None
                        ),
                        "episode_of_interest_healthy_rank_desc": (
                            int(rank_1710)
                            if rank_1710 is not None
                            else -1
                        ),
                        "episode_of_interest_score": (
                            float(score_1710)
                            if score_1710 is not None
                            else math.nan
                        ),
                        "episode_of_interest_score_to_healthy_median": (
                            float(
                                score_1710
                                / healthy_median
                            )
                            if (
                                score_1710 is not None
                                and healthy_median != 0
                            )
                            else math.nan
                        ),
                        "sensitivity_only_not_primary": bool(
                            variance_threshold
                            != PRIMARY_VARIANCE_THRESHOLD
                        ),
                    }
                )

                # ----------------------------------------------------------
                # Primary-model feature diagnostics only.
                # ----------------------------------------------------------
                if variance_threshold == PRIMARY_VARIANCE_THRESHOLD:
                    support = model[
                        "support_mask"
                    ]

                    scales = np.full(
                        len(all_feature_names),
                        np.nan,
                        dtype=np.float64,
                    )

                    scales[
                        model["retained_indices"]
                    ] = model[
                        "scaler"
                    ].scale_

                    for feature_index, feature_name in enumerate(
                        all_feature_names
                    ):
                        feature_rows.append(
                            {
                                "commissioning_size": int(n),
                                "commissioning_seed": int(seed),
                                "feature_index": int(
                                    feature_index
                                ),
                                "feature_name": str(
                                    feature_name
                                ),
                                "training_raw_variance": float(
                                    raw_variances[
                                        feature_index
                                    ]
                                ),
                                "retained_primary": bool(
                                    support[
                                        feature_index
                                    ]
                                ),
                                "standard_scaler_scale_if_retained": (
                                    float(
                                        scales[
                                            feature_index
                                        ]
                                    )
                                    if np.isfinite(
                                        scales[
                                            feature_index
                                        ]
                                    )
                                    else math.nan
                                ),
                                "variance_threshold": float(
                                    PRIMARY_VARIANCE_THRESHOLD
                                ),
                            }
                        )

                    if rank_1710 is not None:
                        idx = int(
                            np.where(
                                healthy_ids.astype(int)
                                == int(
                                    args.episode_of_interest
                                )
                            )[0][0]
                        )

                        row = healthy_scaled[
                            idx
                        ]

                        contributions = _feature_contributions(
                            model,
                            row,
                        )

                        retained_names = np.asarray(
                            all_feature_names,
                            dtype=object,
                        )[
                            model[
                                "retained_indices"
                            ]
                        ]

                        centered = (
                            row
                            - model[
                                "location"
                            ]
                        )

                        squared_distance = float(
                            np.sum(
                                contributions
                            )
                        )

                        if not np.isclose(
                            squared_distance,
                            float(
                                score_1710
                                ** 2
                            ),
                            rtol=1e-8,
                            atol=1e-8,
                        ):
                            raise RuntimeError(
                                "Mahalanobis contribution decomposition "
                                "does not sum to squared score."
                            )

                        order = np.argsort(
                            np.abs(
                                contributions
                            )
                        )[::-1]

                        for rank, local_index in enumerate(
                            order[:50],
                            start=1,
                        ):
                            contribution_rows.append(
                                {
                                    "commissioning_size": int(n),
                                    "commissioning_seed": int(seed),
                                    "episode_id": int(
                                        args.episode_of_interest
                                    ),
                                    "contribution_rank": int(
                                        rank
                                    ),
                                    "retained_local_feature_index": int(
                                        local_index
                                    ),
                                    "original_feature_index": int(
                                        model[
                                            "retained_indices"
                                        ][
                                            local_index
                                        ]
                                    ),
                                    "feature_name": str(
                                        retained_names[
                                            local_index
                                        ]
                                    ),
                                    "training_raw_variance": float(
                                        retained_variances[
                                            local_index
                                        ]
                                    ),
                                    "standardized_feature_value": float(
                                        row[
                                            local_index
                                        ]
                                    ),
                                    "centered_standardized_value": float(
                                        centered[
                                            local_index
                                        ]
                                    ),
                                    "signed_squared_mahalanobis_contribution": float(
                                        contributions[
                                            local_index
                                        ]
                                    ),
                                    "absolute_contribution": float(
                                        abs(
                                            contributions[
                                                local_index
                                            ]
                                        )
                                    ),
                                    "fraction_of_abs_top50_contribution": math.nan,
                                    "interpretation_note": (
                                        "Signed additive contribution to "
                                        "squared Mahalanobis distance; may be "
                                        "negative because of covariance "
                                        "cross-terms."
                                    ),
                                }
                            )

    run_df = pd.DataFrame(
        run_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    contribution_df = pd.DataFrame(
        contribution_rows
    )

    # ----------------------------------------------------------------------
    # Frozen evaluation audit: IDs must be invariant across all requested
    # commissioning seeds/N. We reconstruct hashes directly.
    # ----------------------------------------------------------------------
    frozen_rows = []

    for n in args.commissioning:
        for seed in args.seeds:
            split = create_frozen_evaluation_split(
                cycles=cycles,
                commissioning_size=int(n),
                commissioning_seed=int(seed),
                evaluation_seed=EVALUATION_SEED,
                calibration_size=CALIBRATION_SIZE,
                normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
                maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
            )

            frozen_rows.append(
                {
                    "commissioning_size": int(n),
                    "commissioning_seed": int(seed),
                    "healthy_eval_ids": "|".join(
                        str(int(c.episode_id))
                        for c in split.target_normal_evaluation
                    ),
                    "anomaly_eval_ids": "|".join(
                        str(int(c.episode_id))
                        for c in split.target_anomaly_evaluation
                    ),
                    "calibration_ids": "|".join(
                        str(int(c.episode_id))
                        for c in split.target_calibration
                    ),
                }
            )

    frozen_df = pd.DataFrame(
        frozen_rows
    )

    if frozen_df["healthy_eval_ids"].nunique() != 1:
        raise RuntimeError(
            "Healthy evaluation set is not frozen."
        )

    if frozen_df["anomaly_eval_ids"].nunique() != 1:
        raise RuntimeError(
            "Anomaly evaluation set is not frozen."
        )

    if frozen_df["calibration_ids"].nunique() != 1:
        raise RuntimeError(
            "Calibration set is not frozen."
        )

    # ----------------------------------------------------------------------
    # Summaries
    # ----------------------------------------------------------------------
    summary = (
        run_df
        .groupby(
            [
                "commissioning_size",
                "variance_threshold",
            ],
            sort=True,
        )
        .agg(
            runs=(
                "commissioning_seed",
                "nunique",
            ),
            retained_features_median=(
                "retained_feature_count",
                "median",
            ),
            retained_features_min=(
                "retained_feature_count",
                "min",
            ),
            retained_features_max=(
                "retained_feature_count",
                "max",
            ),
            covariance_condition_median=(
                "covariance_condition_number",
                "median",
            ),
            covariance_condition_max=(
                "covariance_condition_number",
                "max",
            ),
            auc_mean=(
                "auc_sklearn",
                "mean",
            ),
            oracle_1fp_feasible_fraction=(
                "oracle_1fp_feasible",
                "mean",
            ),
            oracle_1fp_mean_max_recall=(
                "oracle_1fp_max_recall",
                "mean",
            ),
            oracle_0fp_feasible_fraction=(
                "oracle_0fp_feasible",
                "mean",
            ),
            oracle_0fp_mean_max_recall=(
                "oracle_0fp_max_recall",
                "mean",
            ),
            episode_interest_maximum_fraction=(
                "episode_of_interest_healthy_rank_desc",
                lambda x: float(
                    np.mean(
                        np.asarray(x)
                        == 1
                    )
                ),
            ),
            episode_interest_median_rank=(
                "episode_of_interest_healthy_rank_desc",
                "median",
            ),
            episode_interest_median_score_to_healthy_median=(
                "episode_of_interest_score_to_healthy_median",
                "median",
            ),
        )
        .reset_index()
    )

    # Compare every sensitivity threshold directly against primary within
    # the same N/seed.
    primary = run_df[
        run_df["variance_threshold"] == PRIMARY_VARIANCE_THRESHOLD
    ][
        [
            "commissioning_size",
            "commissioning_seed",
            "oracle_1fp_feasible",
            "oracle_1fp_max_recall",
            "oracle_0fp_max_recall",
            "auc_sklearn",
            "episode_of_interest_healthy_rank_desc",
            "episode_of_interest_score_to_healthy_median",
        ]
    ].copy()

    primary = primary.rename(
        columns={
            "oracle_1fp_feasible":
                "primary_oracle_1fp_feasible",
            "oracle_1fp_max_recall":
                "primary_oracle_1fp_max_recall",
            "oracle_0fp_max_recall":
                "primary_oracle_0fp_max_recall",
            "auc_sklearn":
                "primary_auc",
            "episode_of_interest_healthy_rank_desc":
                "primary_episode_interest_rank",
            "episode_of_interest_score_to_healthy_median":
                "primary_episode_interest_score_to_median",
        }
    )

    comparison = run_df.merge(
        primary,
        on=[
            "commissioning_size",
            "commissioning_seed",
        ],
        how="left",
        validate="many_to_one",
    )

    comparison[
        "one_fp_feasibility_changed_vs_primary"
    ] = (
        comparison[
            "oracle_1fp_feasible"
        ]
        != comparison[
            "primary_oracle_1fp_feasible"
        ]
    )

    comparison[
        "one_fp_recall_delta_vs_primary"
    ] = (
        comparison[
            "oracle_1fp_max_recall"
        ]
        - comparison[
            "primary_oracle_1fp_max_recall"
        ]
    )

    comparison[
        "zero_fp_recall_delta_vs_primary"
    ] = (
        comparison[
            "oracle_0fp_max_recall"
        ]
        - comparison[
            "primary_oracle_0fp_max_recall"
        ]
    )

    comparison[
        "auc_delta_vs_primary"
    ] = (
        comparison[
            "auc_sklearn"
        ]
        - comparison[
            "primary_auc"
        ]
    )

    # ----------------------------------------------------------------------
    # Primary feature-conditioning summary.
    # ----------------------------------------------------------------------
    retained_primary = feature_df[
        feature_df[
            "retained_primary"
        ]
    ].copy()

    feature_conditioning_summary = (
        retained_primary
        .groupby(
            [
                "commissioning_size",
                "commissioning_seed",
            ]
        )
        .agg(
            retained_feature_count=(
                "feature_index",
                "count",
            ),
            retained_variance_min=(
                "training_raw_variance",
                "min",
            ),
            retained_variance_q01=(
                "training_raw_variance",
                lambda x: float(
                    np.quantile(
                        np.asarray(x),
                        0.01,
                    )
                ),
            ),
            retained_variance_median=(
                "training_raw_variance",
                "median",
            ),
            scaler_scale_min=(
                "standard_scaler_scale_if_retained",
                "min",
            ),
            scaler_scale_q01=(
                "standard_scaler_scale_if_retained",
                lambda x: float(
                    np.quantile(
                        np.asarray(x),
                        0.01,
                    )
                ),
            ),
            count_retained_variance_lt_1e_10=(
                "training_raw_variance",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        < 1e-10
                    )
                ),
            ),
            count_retained_variance_lt_1e_8=(
                "training_raw_variance",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        < 1e-8
                    )
                ),
            ),
            count_retained_variance_lt_1e_6=(
                "training_raw_variance",
                lambda x: int(
                    np.sum(
                        np.asarray(x)
                        < 1e-6
                    )
                ),
            ),
        )
        .reset_index()
    )

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    run_df.to_csv(
        args.output_dir
        / "m2_v2_3_seed_results.csv",
        index=False,
    )

    summary.to_csv(
        args.output_dir
        / "m2_v2_3_summary.csv",
        index=False,
    )

    comparison.to_csv(
        args.output_dir
        / "m2_v2_3_primary_comparison.csv",
        index=False,
    )

    feature_df.to_csv(
        args.output_dir
        / "m2_v2_3_primary_feature_variances.csv",
        index=False,
    )

    feature_conditioning_summary.to_csv(
        args.output_dir
        / "m2_v2_3_feature_conditioning_summary.csv",
        index=False,
    )

    contribution_df.to_csv(
        args.output_dir
        / "m2_v2_3_episode1710_mahalanobis_contributions.csv",
        index=False,
    )

    frozen_df.to_csv(
        args.output_dir
        / "m2_v2_3_frozen_partition_audit.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # Automatic reviewer-facing decision criteria.
    # These do NOT declare scientific truth; they summarize whether the
    # specific numerical-artifact attack survives this audit.
    # ----------------------------------------------------------------------
    nonprimary = comparison[
        comparison["variance_threshold"] != PRIMARY_VARIANCE_THRESHOLD
    ].copy()

    # Strongest sensitivity threshold is the largest requested threshold.
    max_threshold = float(
        max(
            args.variance_thresholds
        )
    )

    strongest = run_df[
        np.isclose(
            run_df[
                "variance_threshold"
            ],
            max_threshold,
        )
    ].copy()

    decision_by_n = []

    for n in sorted(
        strongest[
            "commissioning_size"
        ].unique()
    ):
        group = strongest[
            strongest[
                "commissioning_size"
            ]
            == n
        ]

        decision_by_n.append(
            {
                "commissioning_size": int(n),
                "strongest_variance_threshold": max_threshold,
                "oracle_1fp_feasible_fraction": float(
                    group[
                        "oracle_1fp_feasible"
                    ].mean()
                ),
                "episode1710_maximum_fraction": float(
                    np.mean(
                        group[
                            "episode_of_interest_healthy_rank_desc"
                        ].to_numpy()
                        == 1
                    )
                ),
                "median_auc": float(
                    group[
                        "auc_sklearn"
                    ].median()
                ),
                "median_condition_number": float(
                    group[
                        "covariance_condition_number"
                    ].median()
                ),
            }
        )

    decision_df = pd.DataFrame(
        decision_by_n
    )

    decision_df.to_csv(
        args.output_dir
        / "m2_v2_3_reviewer_decision_table.csv",
        index=False,
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "voraus-AD 100Hz",
        "detector": "TargetOnly Gaussian Mahalanobis with Ledoit-Wolf",
        "evaluation_seed": EVALUATION_SEED,
        "commissioning_seeds": list(
            args.seeds
        ),
        "commissioning_grid": list(
            args.commissioning
        ),
        "episode_of_interest": int(
            args.episode_of_interest
        ),
        "primary_variance_threshold": float(
            PRIMARY_VARIANCE_THRESHOLD
        ),
        "sensitivity_variance_thresholds": [
            float(x)
            for x in args.variance_thresholds
        ],
        "audit_question": (
            "Does the M2 extreme healthy-tail / 0FP-to-1FP geometry depend "
            "solely on features with near-zero commissioning variance or on "
            "an ill-conditioned Ledoit-Wolf covariance estimate?"
        ),
        "primary_model_diagnostics": [
            "raw commissioning-feature variance",
            "retained feature identities",
            "StandardScaler scales",
            "Ledoit-Wolf shrinkage",
            "covariance and precision eigenvalues",
            "covariance condition number",
            "episode-1710 signed squared-Mahalanobis contributions",
        ],
        "sensitivity_scope": {
            "new_detector_family": False,
            "new_evaluation_split": False,
            "new_calibration_method": False,
            "new_dataset": False,
            "primary_result_replaced": False,
            "only_change": (
                "training-only variance-filter threshold before the same "
                "StandardScaler + LedoitWolf TargetOnly score"
            ),
        },
        "interpretation_guardrails": [
            "Sensitivity variants are diagnostic and do not replace the primary detector.",
            "Stable oracle geometry across thresholds argues against a sole near-zero-variance explanation.",
            "Instability across thresholds means the paper must acknowledge representation conditioning as part of the mechanism.",
            "A finite covariance condition number does not by itself prove statistical adequacy.",
            "Mahalanobis feature contributions are signed because covariance cross-terms are present and must not be interpreted as independent causal sensor effects.",
            "M2 remains retrospective oracle analysis; M3 evaluates deployable tail calibration.",
            "Evaluation-population uncertainty remains deferred to M5.",
        ],
    }

    with open(
        args.output_dir
        / "m2_v2_3_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 88)
    print("M2-v2.3 NUMERICAL-STABILITY SUMMARY")
    print("=" * 88)
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Strongest variance-filter sensitivity:")
    print(
        decision_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Frozen healthy-eval sets:",
        frozen_df[
            "healthy_eval_ids"
        ].nunique(),
    )
    print(
        "Frozen anomaly-eval sets:",
        frozen_df[
            "anomaly_eval_ids"
        ].nunique(),
    )
    print(
        "Frozen calibration sets:",
        frozen_df[
            "calibration_ids"
        ].nunique(),
    )

    print()
    print(
        f"Saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()