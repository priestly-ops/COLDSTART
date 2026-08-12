"""
M2-v2.1 Frozen-Evaluation Reviewer-Defense Audit
=================================================

Purpose
-------
This experiment audits the M2 oracle-feasibility result while fixing the
protocol issue discovered in M2-v2:

    - Calibration episodes are frozen across commissioning seeds.
    - Healthy evaluation episodes are frozen across commissioning seeds.
    - Anomaly evaluation episodes are frozen across commissioning seeds.
    - ONLY the commissioning subset changes with commissioning seed.
    - Raw healthy/anomaly scores are exported exactly once per run.
    - Independent oracle and AUROC implementations cross-check the result.

Reviewer attacks addressed
--------------------------
1. "Your oracle implementation may be wrong."
2. "The perfect N=50/N=100 result may come from data leakage."
3. "Evaluation data changes across seeds, so seed variation is confounded."
4. "1% FPR with only 100 healthy episodes is one false positive."
5. "The category-level results look suspiciously identical."
6. "The oracle uses evaluation labels."
7. "Small anomaly categories have coarse recall resolution."
8. "90% recall / 1% FPR might be cherry-picked."
9. "Your raw-score artifact cannot independently reproduce the results."

Important interpretation
------------------------
This is a RETROSPECTIVE ORACLE diagnostic.

It answers:

    "Does a threshold satisfying the empirical operating requirement exist
    in the observed score geometry?"

It DOES NOT answer:

    "Can a deployable calibration procedure identify that threshold?"

That latter question belongs to M3.
"""

from __future__ import annotations


# ============================================================================
# Make `src` importable when this script is executed directly:
#
#     python experiments/run_m2_reviewer_defense.py
#
# Without this, Python normally adds only experiments/ to sys.path.
# ============================================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Standard library
# ============================================================================

import argparse
import json
from typing import Any


# ============================================================================
# Third-party
# ============================================================================

import numpy as np
import pandas as pd


# ============================================================================
# COLDSTART imports
# ============================================================================

from src.evaluation import detector_factories, fit_detector
from src.feature_extractor import extract_feature_matrix

from src.m2_reviewer_audit import (
    count_based_sensitivity,
    independent_bruteforce_oracle,
    score_ordering_audit,
    sha256_array,
    sha256_strings,
    sklearn_auc_check,
)

from src.oracle_feasibility import (
    empirical_oracle_feasibility,
    probability_of_superiority,
)

from src.split_generator import create_frozen_evaluation_split
from src.voraus_loader import RobotCycle, load_cycles


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "voraus-ad-dataset-100hz.parquet"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "m2_oracle_reviewer_defense"
)

PROTOCOL_VERSION = "m2-oracle-reviewer-defense-v2.1-frozen-eval"

GLOBAL_SEED = 42

# This seed defines the fixed calibration/evaluation populations.
# It must NOT change across commissioning seeds.
EVALUATION_SEED = 42

SEEDS = tuple(range(20))

COMMISSIONING_GRID = (
    10,
    25,
    50,
    100,
)

FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90

CALIBRATION_SIZE = 100
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100


# ============================================================================
# Official voraus-AD category mapping
# ============================================================================

VORAUS_CATEGORY_NAMES = {
    0: "axis_friction",
    1: "axis_weight",
    2: "collision_foam",
    3: "collision_cable",
    4: "collision_carton",
    5: "miss_can",
    6: "lose_can",
    7: "can_weight",
    8: "entangled",
    9: "invalid_position",
    10: "motor_commutation",
    11: "wobbling_station",
    12: "normal_operation",
}


np.random.seed(GLOBAL_SEED)


# ============================================================================
# CLI helpers
# ============================================================================


def _parse_int_list(value: str) -> tuple[int, ...]:
    """
    Parse comma-separated integers.

    Examples
    --------
    "0" -> (0,)
    "0,1,2" -> (0, 1, 2)
    "10,25,50,100" -> (10, 25, 50, 100)
    """
    return tuple(
        int(x.strip())
        for x in value.split(",")
        if x.strip()
    )


# ============================================================================
# Hashing / provenance helpers
# ============================================================================


def _canonical_id_hash(ids) -> str:
    """
    Hash episode SET identity rather than incidental ordering.

    M2-v2 used order-sensitive hashes, which made the anomaly set appear to
    change across seeds even though only its ordering changed.

    Sorting before hashing makes this represent actual set membership.
    """
    return sha256_strings(
        sorted(str(x) for x in ids)
    )


def _pairwise_overlap_counts(
    parts: dict[str, tuple[RobotCycle, ...]],
) -> dict[str, int]:
    """
    Independently verify pairwise partition disjointness.

    This supplements ExperimentSplit.verify_no_overlap().
    """
    ids = {
        name: {
            int(cycle.episode_id)
            for cycle in cycles
        }
        for name, cycles in parts.items()
    }

    names = list(ids)
    output: dict[str, int] = {}

    for index, first_name in enumerate(names):

        for second_name in names[index + 1 :]:

            overlap = (
                ids[first_name]
                & ids[second_name]
            )

            key = f"{first_name}__{second_name}"

            output[key] = len(overlap)

            if overlap:

                raise RuntimeError(
                    f"DATA LEAKAGE detected between "
                    f"{first_name} and {second_name}. "
                    f"Overlap count={len(overlap)}, "
                    f"examples={sorted(overlap)[:5]}"
                )

    return output


# ============================================================================
# Score generation
# ============================================================================


def _run_scores(
    cycles: list[RobotCycle],
    commissioning_size: int,
    commissioning_seed: int,
) -> dict[str, Any]:
    """
    Fit TargetOnly using a commissioning sample while holding calibration
    and evaluation populations fixed.

    ONLY commissioning composition may change with commissioning_seed.
    """

    split = create_frozen_evaluation_split(
        cycles=cycles,
        commissioning_size=commissioning_size,
        commissioning_seed=commissioning_seed,

        # Fixed across ALL runs:
        evaluation_seed=EVALUATION_SEED,

        calibration_size=CALIBRATION_SIZE,
        normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
        maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
    )

    # Existing leakage check.
    split.verify_no_overlap()

    parts = {
        "source": split.source_train,
        "commissioning": split.target_commissioning,
        "calibration": split.target_calibration,
        "healthy_eval": split.target_normal_evaluation,
        "anomaly_eval": split.target_anomaly_evaluation,
    }

    # Second independent leakage check.
    overlaps = _pairwise_overlap_counts(parts)

    # ----------------------------------------------------------------------
    # Extract feature matrices
    # ----------------------------------------------------------------------

    source_raw, source_ids = extract_feature_matrix(
        split.source_train
    )

    target_raw, commissioning_ids = extract_feature_matrix(
        split.target_commissioning
    )

    calibration_raw, calibration_ids = extract_feature_matrix(
        split.target_calibration
    )

    healthy_raw, healthy_ids = extract_feature_matrix(
        split.target_normal_evaluation
    )

    anomaly_raw, anomaly_ids = extract_feature_matrix(
        split.target_anomaly_evaluation
    )

    # ----------------------------------------------------------------------
    # Fit TargetOnly
    # ----------------------------------------------------------------------

    factory = detector_factories(
        FALSE_ALERT_BUDGET
    )["TargetOnly"]

    detector, preprocessor, _, _ = fit_detector(
        detector_name="TargetOnly",
        detector_factory=factory,
        source_raw=source_raw,
        target_raw=target_raw,
    )

    # ----------------------------------------------------------------------
    # Score frozen evaluation populations
    # ----------------------------------------------------------------------

    healthy_features = preprocessor.transform(
        healthy_raw
    )

    anomaly_features = preprocessor.transform(
        anomaly_raw
    )

    healthy_scores = np.asarray(
        detector.score_samples(
            healthy_features
        ),
        dtype=np.float64,
    )

    anomaly_scores = np.asarray(
        detector.score_samples(
            anomaly_features
        ),
        dtype=np.float64,
    )

    anomaly_categories = np.asarray(
        [
            cycle.category
            for cycle in split.target_anomaly_evaluation
        ],
        dtype=np.int64,
    )

    # ----------------------------------------------------------------------
    # Defensive identity checks
    # ----------------------------------------------------------------------

    if len(np.unique(healthy_ids)) != len(healthy_ids):

        raise RuntimeError(
            "Duplicate healthy evaluation episode IDs."
        )

    if len(np.unique(anomaly_ids)) != len(anomaly_ids):

        raise RuntimeError(
            "Duplicate anomaly evaluation episode IDs."
        )

    if (
        set(map(int, healthy_ids))
        & set(map(int, anomaly_ids))
    ):

        raise RuntimeError(
            "Healthy and anomaly evaluation IDs overlap."
        )

    if len(healthy_scores) != len(healthy_ids):

        raise RuntimeError(
            "Healthy score count does not match "
            "healthy episode count."
        )

    if len(anomaly_scores) != len(anomaly_ids):

        raise RuntimeError(
            "Anomaly score count does not match "
            "anomaly episode count."
        )

    return {
        "split": split,

        "overlaps": overlaps,

        "source_ids": np.asarray(
            source_ids
        ),

        "commissioning_ids": np.asarray(
            commissioning_ids
        ),

        "calibration_ids": np.asarray(
            calibration_ids
        ),

        "healthy_ids": np.asarray(
            healthy_ids
        ),

        "anomaly_ids": np.asarray(
            anomaly_ids
        ),

        "healthy_scores": healthy_scores,

        "anomaly_scores": anomaly_scores,

        "anomaly_categories": anomaly_categories,

        "retained_features": int(
            preprocessor.output_feature_count_
        ),
    }


# ============================================================================
# Independent oracle cross-check
# ============================================================================


def _primary_vs_independent(
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Compute M2 oracle using TWO independent implementations.

    Primary:
        src.oracle_feasibility.empirical_oracle_feasibility()

    Independent:
        src.m2_reviewer_audit.independent_bruteforce_oracle()

    The experiment aborts if they disagree.
    """

    primary = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )

    independent = independent_bruteforce_oracle(
        healthy_scores,
        anomaly_scores,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )

    # ----------------------------------------------------------------------
    # Independent ranking check
    # ----------------------------------------------------------------------

    auc_probability_superiority = (
        probability_of_superiority(
            healthy_scores,
            anomaly_scores,
        )
    )

    auc_sklearn = sklearn_auc_check(
        healthy_scores,
        anomaly_scores,
    )

    # ----------------------------------------------------------------------
    # Agreement checks
    # ----------------------------------------------------------------------

    checks = {
        "feasible_equal": bool(
            primary.empirically_feasible
            == independent.feasible
        ),

        "max_recall_equal": bool(
            np.isclose(
                primary.max_recall_at_fpr_budget,
                independent.max_recall_at_budget,
                atol=1e-12,
            )
        ),

        "fpr_equal": bool(
            np.isclose(
                primary.fpr_at_max_recall,
                independent.fpr_at_max_recall,
                atol=1e-12,
            )
        ),

        "min_fpr_equal": bool(
            np.isclose(
                primary.min_fpr_at_recall_target,
                independent.min_fpr_at_target,
                atol=1e-12,
                equal_nan=True,
            )
        ),

        "auc_equal": (
            True
            if not np.isfinite(
                auc_sklearn
            )
            else bool(
                np.isclose(
                    auc_probability_superiority,
                    auc_sklearn,
                    atol=1e-12,
                )
            )
        ),
    }

    if not all(checks.values()):

        raise RuntimeError(
            "Independent M2 oracle audit disagreement: "
            f"{checks}"
        )

    # ----------------------------------------------------------------------
    # Return publication/audit values
    # ----------------------------------------------------------------------

    return {
        "primary_max_recall_at_1fp":
            float(
                primary.max_recall_at_fpr_budget
            ),

        "primary_fpr_at_max_recall":
            float(
                primary.fpr_at_max_recall
            ),

        "primary_min_fpr_at_90recall":
            float(
                primary.min_fpr_at_recall_target
            ),

        "primary_feasible":
            bool(
                primary.empirically_feasible
            ),

        "independent_max_recall_at_1fp":
            float(
                independent.max_recall_at_budget
            ),

        "independent_fpr_at_max_recall":
            float(
                independent.fpr_at_max_recall
            ),

        "independent_min_fpr_at_90recall":
            float(
                independent.min_fpr_at_target
            ),

        "independent_feasible":
            bool(
                independent.feasible
            ),

        "auc_probability_superiority":
            float(
                auc_probability_superiority
            ),

        "auc_sklearn":
            float(
                auc_sklearn
            ),

        # Count-level diagnostics.
        "max_recall_count_at_budget":
            int(
                independent.max_recall_count_at_budget
            ),

        "required_recall_count":
            int(
                independent.required_recall_count
            ),

        "recall_count_margin":
            int(
                independent.max_recall_count_at_budget
                - independent.required_recall_count
            ),

        "min_fp_count_at_90recall":
            int(
                independent.min_fp_count_at_target
            ),

        "allowed_fp_count":
            int(
                independent.allowed_fp_count
            ),

        "fp_count_margin":
            int(
                independent.allowed_fp_count
                - independent.min_fp_count_at_target
            ),

        **checks,
    }


# ============================================================================
# Main experiment
# ============================================================================


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
        help=(
            "Commissioning seeds, e.g. "
            "--seeds 0 or --seeds 0,1,2"
        ),
    )

    parser.add_argument(
        "--commissioning",
        type=_parse_int_list,
        default=COMMISSIONING_GRID,
        help=(
            "Commissioning N values, e.g. "
            "--commissioning 100"
        ),
    )

    parser.add_argument(
        "--raw-score-n",
        type=_parse_int_list,
        default=(50, 100),
        help=(
            "N values for which full raw evaluation "
            "scores are exported."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("M2-v2.1 FROZEN-EVALUATION REVIEWER-DEFENSE AUDIT")
    print("=" * 78)

    print(
        f"Dataset: {args.data_path}"
    )

    print(
        f"Output:  {args.output_dir}"
    )

    print(
        f"Commissioning seeds: {args.seeds}"
    )

    print(
        f"Commissioning N: {args.commissioning}"
    )

    print(
        f"Fixed evaluation seed: {EVALUATION_SEED}"
    )

    print()


    # ----------------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------------

    cycles = load_cycles(
        args.data_path
    )

    anomalous_cycles = [
        cycle
        for cycle in cycles
        if cycle.anomaly
    ]

    print(
        f"Loaded cycles: {len(cycles)}"
    )

    print(
        f"Anomalous cycles: {len(anomalous_cycles)}"
    )

    print()


    # ----------------------------------------------------------------------
    # Category counts
    # ----------------------------------------------------------------------

    category_counts: dict[int, int] = {}

    for cycle in anomalous_cycles:

        category_id = int(
            cycle.category
        )

        category_counts[category_id] = (
            category_counts.get(
                category_id,
                0,
            )
            + 1
        )


    # ----------------------------------------------------------------------
    # Output collections
    # ----------------------------------------------------------------------

    audit_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    ordering_rows: list[dict] = []
    raw_rows: list[dict] = []
    split_rows: list[dict] = []


    total_runs = (
        len(args.seeds)
        * len(args.commissioning)
    )

    run_index = 0


    # ======================================================================
    # Experiment loop
    # ======================================================================

    for commissioning_size in args.commissioning:

        for commissioning_seed in args.seeds:

            run_index += 1

            print(
                f"Processing "
                f"N={commissioning_size} "
                f"seed={commissioning_seed} "
                f"({run_index}/{total_runs})..."
            )

            result = _run_scores(
                cycles,
                int(commissioning_size),
                int(commissioning_seed),
            )

            healthy_scores = (
                result["healthy_scores"]
            )

            anomaly_scores_all = (
                result["anomaly_scores"]
            )

            anomaly_categories = (
                result["anomaly_categories"]
            )


            # ==============================================================
            # Split provenance row
            # ==============================================================

            split_row = {
                "protocol_version":
                    PROTOCOL_VERSION,

                "evaluation_seed":
                    EVALUATION_SEED,

                "commissioning_size":
                    int(
                        commissioning_size
                    ),

                "commissioning_seed":
                    int(
                        commissioning_seed
                    ),

                "retained_features":
                    int(
                        result[
                            "retained_features"
                        ]
                    ),

                "commissioning_count":
                    int(
                        len(
                            result[
                                "commissioning_ids"
                            ]
                        )
                    ),

                "calibration_count":
                    int(
                        len(
                            result[
                                "calibration_ids"
                            ]
                        )
                    ),

                "healthy_eval_count":
                    int(
                        len(
                            healthy_scores
                        )
                    ),

                "anomaly_eval_count":
                    int(
                        len(
                            anomaly_scores_all
                        )
                    ),

                # Canonical set hashes.
                "commissioning_ids_sha256":
                    _canonical_id_hash(
                        result[
                            "commissioning_ids"
                        ]
                    ),

                "calibration_ids_sha256":
                    _canonical_id_hash(
                        result[
                            "calibration_ids"
                        ]
                    ),

                "healthy_ids_sha256":
                    _canonical_id_hash(
                        result[
                            "healthy_ids"
                        ]
                    ),

                "anomaly_ids_sha256":
                    _canonical_id_hash(
                        result[
                            "anomaly_ids"
                        ]
                    ),

                # Score hashes intentionally remain order-sensitive because
                # arrays should be generated in deterministic episode order.
                "healthy_scores_sha256":
                    sha256_array(
                        healthy_scores
                    ),

                "anomaly_scores_sha256":
                    sha256_array(
                        anomaly_scores_all
                    ),

                "all_pairwise_partition_overlaps_zero":
                    bool(
                        all(
                            value == 0
                            for value
                            in result[
                                "overlaps"
                            ].values()
                        )
                    ),
            }

            split_row.update(
                {
                    f"overlap_{key}":
                        int(value)

                    for key, value
                    in result[
                        "overlaps"
                    ].items()
                }
            )

            split_rows.append(
                split_row
            )


            # ==============================================================
            # RAW SCORE EXPORT
            #
            # IMPORTANT:
            # Export healthy and anomaly rows ONCE per run.
            #
            # M2-v2 mistakenly exported healthy rows inside the category loop,
            # producing duplicated healthy records.
            # ==============================================================

            if (
                int(commissioning_size)
                in set(args.raw_score_n)
            ):

                for (
                    episode_id,
                    score,
                ) in zip(
                    result["healthy_ids"],
                    healthy_scores,
                ):

                    raw_rows.append(
                        {
                            "protocol_version":
                                PROTOCOL_VERSION,

                            "commissioning_size":
                                int(
                                    commissioning_size
                                ),

                            "commissioning_seed":
                                int(
                                    commissioning_seed
                                ),

                            "partition":
                                "healthy_eval",

                            "episode_id":
                                str(
                                    int(
                                        episode_id
                                    )
                                ),

                            "category_id":
                                12,

                            "category_name":
                                "normal_operation",

                            "score":
                                float(
                                    score
                                ),
                        }
                    )


                for (
                    episode_id,
                    score,
                    category,
                ) in zip(
                    result[
                        "anomaly_ids"
                    ],

                    anomaly_scores_all,

                    anomaly_categories,
                ):

                    category_id = int(
                        category
                    )

                    raw_rows.append(
                        {
                            "protocol_version":
                                PROTOCOL_VERSION,

                            "commissioning_size":
                                int(
                                    commissioning_size
                                ),

                            "commissioning_seed":
                                int(
                                    commissioning_seed
                                ),

                            "partition":
                                "anomaly_eval",

                            "episode_id":
                                str(
                                    int(
                                        episode_id
                                    )
                                ),

                            "category_id":
                                category_id,

                            "category_name":
                                VORAUS_CATEGORY_NAMES.get(
                                    category_id,
                                    f"unknown_{category_id}",
                                ),

                            "score":
                                float(
                                    score
                                ),
                        }
                    )


            # ==============================================================
            # Pooled + category groups
            # ==============================================================

            groups: list[
                tuple[
                    int,
                    str,
                    np.ndarray,
                ]
            ] = [
                (
                    -1,

                    "all_anomaly_categories_pooled",

                    np.ones(
                        len(
                            anomaly_scores_all
                        ),
                        dtype=bool,
                    ),
                )
            ]


            for category_id in sorted(
                np.unique(
                    anomaly_categories
                )
            ):

                category_id = int(
                    category_id
                )

                groups.append(
                    (
                        category_id,

                        VORAUS_CATEGORY_NAMES.get(
                            category_id,
                            f"unknown_{category_id}",
                        ),

                        anomaly_categories
                        == category_id,
                    )
                )


            # ==============================================================
            # Oracle analysis
            # ==============================================================

            for (
                category_id,
                category_name,
                category_mask,
            ) in groups:

                anomaly_scores = (
                    anomaly_scores_all[
                        category_mask
                    ]
                )

                anomaly_count = int(
                    len(
                        anomaly_scores
                    )
                )

                base = {
                    "protocol_version":
                        PROTOCOL_VERSION,

                    "evaluation_seed":
                        EVALUATION_SEED,

                    "commissioning_size":
                        int(
                            commissioning_size
                        ),

                    "commissioning_seed":
                        int(
                            commissioning_seed
                        ),

                    "category_id":
                        int(
                            category_id
                        ),

                    "category_name":
                        str(
                            category_name
                        ),

                    "is_pooled":
                        bool(
                            category_id == -1
                        ),

                    "healthy_count":
                        int(
                            len(
                                healthy_scores
                            )
                        ),

                    "anomaly_count":
                        anomaly_count,

                    "recall_resolution":
                        float(
                            1.0
                            / anomaly_count
                        ),

                    "coarse_recall_resolution_gt_5pct":
                        bool(
                            (
                                1.0
                                / anomaly_count
                            )
                            > 0.05
                        ),

                    "retained_features":
                        int(
                            result[
                                "retained_features"
                            ]
                        ),

                    # Explicit scope guard.
                    "retrospective_oracle_only":
                        True,

                    "deployable_claim_allowed":
                        False,
                }


                # ----------------------------------------------------------
                # Independent oracle verification
                # ----------------------------------------------------------

                cross_check = (
                    _primary_vs_independent(
                        healthy_scores,
                        anomaly_scores,
                    )
                )

                audit_rows.append(
                    {
                        **base,
                        **cross_check,
                    }
                )


                # ----------------------------------------------------------
                # Score geometry / ordering audit
                # ----------------------------------------------------------

                ordering_rows.append(
                    {
                        **base,

                        **score_ordering_audit(
                            healthy_scores,
                            anomaly_scores,
                        ),
                    }
                )


                # ----------------------------------------------------------
                # Count-based criterion sensitivity
                # ----------------------------------------------------------

                for sensitivity in (
                    count_based_sensitivity(
                        healthy_scores,
                        anomaly_scores,
                    )
                ):

                    sensitivity_rows.append(
                        {
                            **base,
                            **sensitivity,
                        }
                    )


    # ======================================================================
    # Convert to DataFrames
    # ======================================================================

    audit_df = pd.DataFrame(
        audit_rows
    )

    ordering_df = pd.DataFrame(
        ordering_rows
    )

    sensitivity_df = pd.DataFrame(
        sensitivity_rows
    )

    split_df = pd.DataFrame(
        split_rows
    )

    raw_df = pd.DataFrame(
        raw_rows
    )


    # ======================================================================
    # Submission-critical frozen evaluation assertions
    # ======================================================================

    healthy_hash_count = int(
        split_df[
            "healthy_ids_sha256"
        ].nunique()
    )

    anomaly_hash_count = int(
        split_df[
            "anomaly_ids_sha256"
        ].nunique()
    )

    calibration_hash_count = int(
        split_df[
            "calibration_ids_sha256"
        ].nunique()
    )


    if healthy_hash_count != 1:

        raise RuntimeError(
            "FROZEN EVALUATION FAILURE: "
            f"healthy evaluation IDs have "
            f"{healthy_hash_count} unique hashes. "
            "Expected exactly 1."
        )


    if anomaly_hash_count != 1:

        raise RuntimeError(
            "FROZEN EVALUATION FAILURE: "
            f"anomaly evaluation IDs have "
            f"{anomaly_hash_count} unique hashes. "
            "Expected exactly 1."
        )


    if calibration_hash_count != 1:

        raise RuntimeError(
            "FROZEN EVALUATION FAILURE: "
            f"calibration IDs have "
            f"{calibration_hash_count} unique hashes. "
            "Expected exactly 1."
        )


    if not split_df[
        "all_pairwise_partition_overlaps_zero"
    ].all():

        raise RuntimeError(
            "At least one leakage audit failed."
        )


    # ======================================================================
    # Verify commissioning actually changes across seeds
    # ======================================================================

    commissioning_variation = (
        split_df
        .groupby(
            "commissioning_size"
        )[
            "commissioning_ids_sha256"
        ]
        .nunique()
    )


    if len(args.seeds) > 1:

        for commissioning_size in (
            commissioning_variation.index
        ):

            unique_count = int(
                commissioning_variation.loc[
                    commissioning_size
                ]
            )

            if unique_count <= 1:

                raise RuntimeError(
                    "Commissioning set failed to vary "
                    f"across seeds at "
                    f"N={commissioning_size}."
                )


    # ======================================================================
    # Save primary audit tables
    # ======================================================================

    audit_df.to_csv(
        args.output_dir
        / "m2_v2_1_independent_oracle_audit.csv",
        index=False,
    )

    ordering_df.to_csv(
        args.output_dir
        / "m2_v2_1_score_ordering_audit.csv",
        index=False,
    )

    sensitivity_df.to_csv(
        args.output_dir
        / "m2_v2_1_count_sensitivity.csv",
        index=False,
    )

    split_df.to_csv(
        args.output_dir
        / "m2_v2_1_split_hash_audit.csv",
        index=False,
    )


    # ======================================================================
    # Raw-score export validation
    # ======================================================================

    if not raw_df.empty:

        raw_path = (
            args.output_dir
            / "m2_v2_1_raw_eval_scores.csv"
        )

        raw_df.to_csv(
            raw_path,
            index=False,
        )


        # Expected anomaly count is fixed because all anomalies are used.
        expected_rows_per_run = (
            HEALTHY_EVALUATION_SIZE
            + len(anomalous_cycles)
        )


        raw_counts = (
            raw_df
            .groupby(
                [
                    "commissioning_size",
                    "commissioning_seed",
                ]
            )
            .size()
        )


        if not (
            raw_counts
            == expected_rows_per_run
        ).all():

            raise RuntimeError(
                "Raw-score export row count failure. "
                f"Expected "
                f"{expected_rows_per_run} rows "
                f"per exported run."
            )


        healthy_raw_counts = (
            raw_df[
                raw_df["partition"]
                == "healthy_eval"
            ]
            .groupby(
                [
                    "commissioning_size",
                    "commissioning_seed",
                ]
            )
            .size()
        )


        if not (
            healthy_raw_counts
            == HEALTHY_EVALUATION_SIZE
        ).all():

            raise RuntimeError(
                "Healthy raw-score export "
                "is duplicated or incomplete."
            )


        anomaly_raw_counts = (
            raw_df[
                raw_df["partition"]
                == "anomaly_eval"
            ]
            .groupby(
                [
                    "commissioning_size",
                    "commissioning_seed",
                ]
            )
            .size()
        )


        if not (
            anomaly_raw_counts
            == len(anomalous_cycles)
        ).all():

            raise RuntimeError(
                "Anomaly raw-score export "
                "is duplicated or incomplete."
            )


    # ======================================================================
    # Category provenance / resolution table
    # ======================================================================

    category_rows = []


    for (
        category_id,
        count,
    ) in sorted(
        category_counts.items()
    ):

        recall_resolution = (
            1.0
            / count
        )

        category_rows.append(
            {
                "category_id":
                    int(
                        category_id
                    ),

                "official_category_name":
                    VORAUS_CATEGORY_NAMES.get(
                        category_id,
                        f"unknown_{category_id}",
                    ),

                "anomaly_episode_count":
                    int(
                        count
                    ),

                "recall_resolution":
                    float(
                        recall_resolution
                    ),

                "coarse_recall_resolution_gt_5pct":
                    bool(
                        recall_resolution
                        > 0.05
                    ),

                "mapping_known":
                    bool(
                        category_id
                        in VORAUS_CATEGORY_NAMES
                    ),
            }
        )


    category_df = pd.DataFrame(
        category_rows
    )


    category_df.to_csv(
        args.output_dir
        / "m2_v2_1_category_resolution_audit.csv",
        index=False,
    )


    # ======================================================================
    # Reviewer-facing pooled summary
    # ======================================================================

    pooled = audit_df[
        audit_df["is_pooled"]
    ].copy()


    summary = (
        pooled
        .groupby(
            "commissioning_size",
            sort=True,
        )
        .agg(
            seeds=(
                "commissioning_seed",
                "nunique",
            ),

            oracle_feasible_fraction=(
                "primary_feasible",
                "mean",
            ),

            mean_max_recall_at_1fp=(
                "primary_max_recall_at_1fp",
                "mean",
            ),

            minimum_max_recall_at_1fp=(
                "primary_max_recall_at_1fp",
                "min",
            ),

            mean_min_fpr_at_90recall=(
                "primary_min_fpr_at_90recall",
                "mean",
            ),

            all_independent_checks_pass=(
                "feasible_equal",
                lambda x: bool(
                    x.all()
                ),
            ),
        )
        .reset_index()
    )


    # ======================================================================
    # Add 0-FP / 1-FP / 2-FP sensitivity
    # ======================================================================

    pooled_sensitivity = (
        sensitivity_df[
            sensitivity_df["is_pooled"]
            & (
                sensitivity_df[
                    "recall_target_requested"
                ]
                == RECALL_TARGET
            )
        ]
        .copy()
    )


    for allowed_fp_count in (
        0,
        1,
        2,
    ):

        condition = (
            pooled_sensitivity[
                pooled_sensitivity[
                    "allowed_fp_count_requested"
                ]
                == allowed_fp_count
            ]
        )


        partial_summary = (
            condition
            .groupby(
                "commissioning_size"
            )
            .agg(
                **{
                    (
                        f"feasible_fraction_"
                        f"90recall_"
                        f"{allowed_fp_count}fp"
                    ): (
                        "feasible",
                        "mean",
                    ),

                    (
                        f"mean_max_recall_"
                        f"{allowed_fp_count}fp"
                    ): (
                        "max_recall_at_budget",
                        "mean",
                    ),
                }
            )
            .reset_index()
        )


        summary = summary.merge(
            partial_summary,
            on="commissioning_size",
            how="left",
        )


    # Frozen-partition audit goes directly into summary.
    summary[
        "frozen_healthy_hash_count"
    ] = healthy_hash_count

    summary[
        "frozen_anomaly_hash_count"
    ] = anomaly_hash_count

    summary[
        "frozen_calibration_hash_count"
    ] = calibration_hash_count


    summary.to_csv(
        args.output_dir
        / "m2_v2_1_reviewer_summary.csv",
        index=False,
    )


    # ======================================================================
    # Manifest
    # ======================================================================

    manifest = {
        "protocol_version":
            PROTOCOL_VERSION,

        "dataset":
            "voraus-AD 100Hz",

        "detector":
            "TargetOnly",

        "global_seed":
            GLOBAL_SEED,

        "evaluation_seed":
            EVALUATION_SEED,

        "commissioning_seeds":
            list(
                args.seeds
            ),

        "commissioning_grid":
            list(
                args.commissioning
            ),

        "primary_benchmark": {
            "recall_target":
                RECALL_TARGET,

            "false_alert_budget":
                FALSE_ALERT_BUDGET,

            "healthy_eval_count":
                HEALTHY_EVALUATION_SIZE,

            "allowed_false_positives":
                1,
        },

        "frozen_partition_design": {

            "calibration":
                (
                    "Selected once using evaluation_seed "
                    "and reused across every commissioning "
                    "seed and N."
                ),

            "healthy_evaluation":
                (
                    "Selected once using evaluation_seed "
                    "and reused across every commissioning "
                    "seed and N."
                ),

            "anomaly_evaluation":
                (
                    "All anomalous episodes in canonical "
                    "episode-ID order; identical set and "
                    "ordering in every run."
                ),

            "commissioning":
                (
                    "Sampled only from remaining target-healthy "
                    "episodes after fixed calibration/evaluation "
                    "removal. commissioning_seed changes only "
                    "this subset."
                ),
        },

        "reviewer_defenses": {

            "oracle_implementation":
                (
                    "Cross-checked against a separately "
                    "implemented midpoint brute-force oracle. "
                    "The experiment aborts on disagreement."
                ),

            "ranking_implementation":
                (
                    "Probability-of-superiority is independently "
                    "cross-checked against sklearn roc_auc_score."
                ),

            "leakage":
                (
                    "ExperimentSplit.verify_no_overlap plus "
                    "explicit pairwise episode-ID overlap checks."
                ),

            "fixed_evaluation":
                (
                    "Submission-critical assertions require "
                    "exactly one healthy evaluation ID hash, "
                    "one anomaly evaluation ID hash, and one "
                    "calibration ID hash across all runs."
                ),

            "canonical_hashes":
                (
                    "Episode IDs are sorted before hashing, "
                    "so hashes represent set identity rather "
                    "than incidental ordering."
                ),

            "commissioning_variation":
                (
                    "Commissioning hashes are required to vary "
                    "across commissioning seeds."
                ),

            "raw_score_reproducibility":
                (
                    "Raw evaluation scores are exported exactly "
                    "once per episode for N=50 and N=100 by "
                    "default. Per-run counts are asserted."
                ),

            "fpr_granularity":
                (
                    "With n_H=100, empirical FPR resolution "
                    "is 1%. Results are therefore also reported "
                    "as 0/1/2 allowed healthy false-positive "
                    "counts."
                ),

            "evaluation_label_selection":
                (
                    "Oracle analysis is explicitly retrospective "
                    "and non-deployable."
                ),

            "category_support":
                (
                    "Official category sample counts, recall "
                    "resolution, and coarse-resolution flags "
                    "are exported."
                ),

            "criterion_choice":
                (
                    "Primary benchmark remains 90% recall / 1% FPR. "
                    "Descriptive sensitivity additionally evaluates "
                    "80/90/95% recall and 0/1/2 healthy FP."
                ),
        },

        "scope_limits": [

            (
                "Oracle success establishes empirical threshold "
                "existence only, not deployable calibration."
            ),

            (
                "Oracle failure is conditional on the tested "
                "TargetOnly score representation."
            ),

            (
                "Population uncertainty for new evaluation "
                "episodes is not estimated here. M5 handles "
                "cluster-aware uncertainty."
            ),

            (
                "Pooled anomaly performance depends on the "
                "dataset's anomaly-category mixture."
            ),

            (
                "Small categories with coarse recall resolution "
                "must remain descriptive."
            ),

            (
                "M2 does not evaluate RACE/ST-1 efficacy."
            ),

            (
                "M2 does not establish cross-dataset generality."
            ),
        ],

        "raw_score_export_N":
            list(
                args.raw_score_n
            ),
    }


    manifest_path = (
        args.output_dir
        / "m2_v2_1_manifest.json"
    )


    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )


    # ======================================================================
    # Console report
    # ======================================================================

    print()
    print("=" * 78)
    print("FROZEN PARTITION AUDIT")
    print("=" * 78)

    print(
        "healthy eval hashes:",
        healthy_hash_count,
    )

    print(
        "anomaly eval hashes:",
        anomaly_hash_count,
    )

    print(
        "calibration hashes:",
        calibration_hash_count,
    )

    print()

    print(
        "commissioning unique hashes by N:"
    )

    print(
        commissioning_variation.to_string()
    )


    print()
    print("=" * 78)
    print("M2-v2.1 REVIEWER SUMMARY")
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )


    print()
    print(
        f"Saved outputs to:"
    )

    print(
        args.output_dir
    )


    print()
    print("Expected critical invariants:")

    print(
        "  healthy eval hashes = 1"
    )

    print(
        "  anomaly eval hashes = 1"
    )

    print(
        "  calibration hashes = 1"
    )

    print(
        "  commissioning hashes > 1 per N "
        "for a multi-seed run"
    )


# ============================================================================
# Entrypoint
# ============================================================================


if __name__ == "__main__":
    main()