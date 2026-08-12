"""M2: retrospective oracle operating-point feasibility on voraus-AD.

Reviewer attack addressed
-------------------------
"You attribute low deployment recall to calibration, but perhaps the requested
90%-recall / 1%-FPR operating point does not exist in the detector's score
representation at all."

M2 intentionally uses the frozen *evaluation* scores themselves to select the
best empirical threshold. It is therefore an optimistic diagnostic, NOT a
calibration method and NOT a deployable performance estimate. If even this
oracle cannot satisfy the joint criterion, the observed score geometry is
empirically incompatible with the requested operating point. If it can satisfy
it, M2 only establishes empirical existence; M3 must test whether a legitimate
tail-calibration method can recover that point without using evaluation labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import detector_factories, fit_detector
from src.feature_extractor import extract_feature_matrix
from src.oracle_feasibility import (
    empirical_oracle_feasibility,
    probability_of_superiority,
)
from src.split_generator import create_experiment_split
from src.voraus_loader import RobotCycle, load_cycles


DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "m2_oracle_feasibility"

PROTOCOL_VERSION = "m2-oracle-feasibility-v1"
GLOBAL_SEED = 42
SEEDS = tuple(range(20))
COMMISSIONING_GRID = (10, 25, 50, 100)
FALSE_ALERT_BUDGET = 0.01
RECALL_TARGET = 0.90
CALIBRATION_SIZE = 100  # reserved only to reconstruct the frozen split
HEALTHY_EVALUATION_SIZE = 100
MAXIMUM_COMMISSIONING_SIZE = 100

# Official voraus-AD Category enum from the dataset authors' voraus_ad.py.
# IMPORTANT: These are NOT the AURSAD-style damaged/missing-screw labels that
# appeared in an earlier exploratory M1 helper. M2 corrects that provenance.
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


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _verify_no_overlap(split: Any) -> None:
    split.verify_no_overlap()


def _scores_for_run(
    cycles: list[RobotCycle],
    commissioning_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    split = create_experiment_split(
        cycles=cycles,
        commissioning_size=commissioning_size,
        seed=seed,
        calibration_size=CALIBRATION_SIZE,
        normal_evaluation_size=HEALTHY_EVALUATION_SIZE,
        maximum_commissioning_size=MAXIMUM_COMMISSIONING_SIZE,
    )
    _verify_no_overlap(split)

    source_raw, _ = extract_feature_matrix(split.source_train)
    target_raw, _ = extract_feature_matrix(split.target_commissioning)
    healthy_raw, healthy_ids = extract_feature_matrix(split.target_normal_evaluation)
    anomaly_raw, anomaly_ids = extract_feature_matrix(split.target_anomaly_evaluation)

    factory = detector_factories(FALSE_ALERT_BUDGET)["TargetOnly"]
    detector, preprocessor, _, _ = fit_detector(
        detector_name="TargetOnly",
        detector_factory=factory,
        source_raw=source_raw,
        target_raw=target_raw,
    )

    healthy_scores = detector.score_samples(preprocessor.transform(healthy_raw))
    anomaly_scores = detector.score_samples(preprocessor.transform(anomaly_raw))
    anomaly_categories = np.asarray(
        [cycle.category for cycle in split.target_anomaly_evaluation], dtype=np.int64
    )

    if set(healthy_ids.tolist()) & set(anomaly_ids.tolist()):
        raise RuntimeError("Healthy and anomaly evaluation episode IDs overlap.")

    return (
        healthy_scores,
        anomaly_scores,
        anomaly_categories,
        anomaly_ids,
        int(preprocessor.output_feature_count_),
    )


def _diagnostic_row(
    *,
    commissioning_size: int,
    seed: int,
    category_id: int | None,
    category_name: str,
    healthy_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    retained_features: int,
) -> dict[str, Any]:
    result = empirical_oracle_feasibility(
        healthy_scores=healthy_scores,
        anomaly_scores=anomaly_scores,
        false_alert_budget=FALSE_ALERT_BUDGET,
        recall_target=RECALL_TARGET,
    )

    # This is a descriptive score-ranking statistic. It is intentionally
    # reported beside the operating-point diagnostic to demonstrate that
    # ranking and low-FPR feasibility are distinct questions.
    superiority = probability_of_superiority(healthy_scores, anomaly_scores)

    if result.empirically_feasible:
        interpretation = "oracle_empirically_feasible_only"
    else:
        interpretation = "oracle_empirically_infeasible_on_observed_scores"

    return {
        "protocol_version": PROTOCOL_VERSION,
        "detector": "TargetOnly",
        "commissioning_size": int(commissioning_size),
        "seed": int(seed),
        "category_id": -1 if category_id is None else int(category_id),
        "category_name": category_name,
        "is_pooled_all_anomalies": bool(category_id is None),
        "healthy_eval_count": result.healthy_count,
        "anomaly_eval_count": result.anomaly_count,
        "false_alert_budget": result.false_alert_budget,
        "recall_target": result.recall_target,
        "allowed_false_positives": result.allowed_false_positives,
        "fpr_resolution": result.fpr_resolution,
        "recall_resolution": result.recall_resolution,
        "probability_of_superiority": superiority,
        "max_recall_at_fpr_budget": result.max_recall_at_fpr_budget,
        "fpr_at_max_recall": result.fpr_at_max_recall,
        "threshold_at_fpr_budget": result.threshold_at_fpr_budget,
        "min_fpr_at_recall_target": result.min_fpr_at_recall_target,
        "recall_at_min_fpr": result.recall_at_min_fpr,
        "threshold_at_recall_target": result.threshold_at_recall_target,
        "empirically_feasible": result.empirically_feasible,
        "recall_slack": result.recall_slack,
        "fpr_slack": result.fpr_slack,
        "retained_features": int(retained_features),
        "interpretation": interpretation,
    }


def _summarize(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["commissioning_size", "category_id", "category_name", "is_pooled_all_anomalies"]
    for keys, group in seed_df.groupby(group_columns, sort=True, dropna=False):
        n, cid, cname, pooled = keys
        rows.append(
            {
                "commissioning_size": int(n),
                "category_id": int(cid),
                "category_name": str(cname),
                "is_pooled_all_anomalies": bool(pooled),
                "seeds": int(len(group)),
                "anomaly_eval_count": int(group["anomaly_eval_count"].iloc[0]),
                "recall_resolution": float(group["recall_resolution"].iloc[0]),
                "mean_probability_of_superiority": float(group["probability_of_superiority"].mean()),
                "median_probability_of_superiority": float(group["probability_of_superiority"].median()),
                "mean_max_recall_at_1pct_fpr": float(group["max_recall_at_fpr_budget"].mean()),
                "median_max_recall_at_1pct_fpr": float(group["max_recall_at_fpr_budget"].median()),
                "minimum_max_recall_at_1pct_fpr": float(group["max_recall_at_fpr_budget"].min()),
                "mean_min_fpr_at_90pct_recall": float(group["min_fpr_at_recall_target"].mean()),
                "median_min_fpr_at_90pct_recall": float(group["min_fpr_at_recall_target"].median()),
                "oracle_feasible_seed_count": int(group["empirically_feasible"].sum()),
                "oracle_feasible_seed_fraction": float(group["empirically_feasible"].mean()),
                "mean_recall_slack": float(group["recall_slack"].mean()),
                "mean_fpr_slack": float(group["fpr_slack"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _save_category_audit(cycles: list[RobotCycle], output_dir: Path) -> pd.DataFrame:
    anomalous = [cycle for cycle in cycles if cycle.anomaly]
    rows = []
    for category in sorted({cycle.category for cycle in anomalous}):
        count = sum(cycle.category == category for cycle in anomalous)
        rows.append(
            {
                "category_id": int(category),
                "official_category_name": VORAUS_CATEGORY_NAMES.get(category, f"unknown_{category}"),
                "anomalous_episode_count": int(count),
                "recall_resolution": float(1.0 / count),
                "official_mapping_known": bool(category in VORAUS_CATEGORY_NAMES),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "m2_category_provenance_audit.csv", index=False)
    unknown = audit.loc[~audit["official_mapping_known"]]
    if not unknown.empty:
        raise RuntimeError(
            "Found anomaly categories absent from the official voraus-AD map: "
            f"{unknown['category_id'].tolist()}"
        )
    return audit


def _plot(seed_df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    # Plot A: category-level empirical oracle feasibility at each N. We show
    # fractions of seeds, not a binary pooled claim, because detector geometry
    # changes with commissioning sample.
    cat = seed_df[~seed_df["is_pooled_all_anomalies"]].copy()
    pivot = (
        cat.groupby(["category_name", "commissioning_size"])["empirically_feasible"]
        .mean()
        .unstack("commissioning_size")
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns])
    ax.set_xlabel("Commissioning episodes (N)")
    ax.set_ylabel("Official voraus-AD anomaly category")
    ax.set_title("M2: fraction of seeds with an empirical 90%-recall / 1%-FPR oracle operating point")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Oracle-feasible seed fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "m2_oracle_feasible_fraction_by_category.png", dpi=180)
    plt.close(fig)

    # Plot B: ranking vs strict operating-point feasibility. This is the direct
    # reviewer-defense figure: strong ranking can coexist with poor feasible
    # recall at 1% FPR.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, sorted(cat["commissioning_size"].unique())):
        group = cat[cat["commissioning_size"] == n]
        for cname, g in group.groupby("category_name"):
            ax.scatter(
                g["probability_of_superiority"],
                g["max_recall_at_fpr_budget"],
                alpha=0.65,
                label=cname,
            )
        ax.axhline(RECALL_TARGET, linestyle="--", linewidth=1)
        ax.set_title(f"TargetOnly, N={int(n)}")
        ax.set_xlabel("Probability of superiority / empirical AUROC")
        ax.set_ylabel("Max empirical recall with FPR ≤ 1%")
        ax.set_xlim(0.0, 1.01)
        ax.set_ylim(0.0, 1.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # A single legend is enough; categories are identical across panels.
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.suptitle("M2: ranking quality versus low-FPR empirical oracle feasibility", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "m2_ranking_vs_oracle_recall.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=_parse_int_list, default=SEEDS)
    parser.add_argument("--commissioning", type=_parse_int_list, default=COMMISSIONING_GRID)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cycles = load_cycles(args.data_path)
    category_audit = _save_category_audit(cycles, args.output_dir)

    detector_names = detector_factories(FALSE_ALERT_BUDGET)
    if "TargetOnly" not in detector_names:
        raise RuntimeError("TargetOnly detector factory is unavailable.")

    rows: list[dict[str, Any]] = []
    total = len(args.seeds) * len(args.commissioning)
    run_index = 0
    for n in args.commissioning:
        for seed in args.seeds:
            run_index += 1
            print(f"M2 TargetOnly N={n} seed={seed} ({run_index}/{total})...")
            (
                healthy_scores,
                anomaly_scores,
                anomaly_categories,
                anomaly_ids,
                retained_features,
            ) = _scores_for_run(cycles, n, seed)

            # Pooled result is retained only as a descriptive overall mixture.
            rows.append(
                _diagnostic_row(
                    commissioning_size=n,
                    seed=seed,
                    category_id=None,
                    category_name="all_anomaly_categories_pooled",
                    healthy_scores=healthy_scores,
                    anomaly_scores=anomaly_scores,
                    retained_features=retained_features,
                )
            )

            for category_id in sorted(np.unique(anomaly_categories)):
                mask = anomaly_categories == category_id
                rows.append(
                    _diagnostic_row(
                        commissioning_size=n,
                        seed=seed,
                        category_id=int(category_id),
                        category_name=VORAUS_CATEGORY_NAMES[int(category_id)],
                        healthy_scores=healthy_scores,
                        anomaly_scores=anomaly_scores[mask],
                        retained_features=retained_features,
                    )
                )

    seed_df = pd.DataFrame(rows)
    summary_df = _summarize(seed_df)
    seed_df.to_csv(args.output_dir / "m2_seed_results.csv", index=False)
    summary_df.to_csv(args.output_dir / "m2_summary.csv", index=False)
    _plot(seed_df, args.output_dir)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "voraus-AD 100Hz",
        "detector": "TargetOnly",
        "seeds": list(args.seeds),
        "commissioning_sizes": list(args.commissioning),
        "false_alert_budget": FALSE_ALERT_BUDGET,
        "recall_target": RECALL_TARGET,
        "healthy_evaluation_size": HEALTHY_EVALUATION_SIZE,
        "allowed_false_positives_at_budget": int(np.floor(FALSE_ALERT_BUDGET * HEALTHY_EVALUATION_SIZE)),
        "oracle_is_retrospective": True,
        "oracle_uses_evaluation_labels": True,
        "oracle_is_deployable": False,
        "positive_interpretation": "An empirical operating point exists in the observed evaluation scores; this is not a calibration guarantee.",
        "negative_interpretation": "Even the optimistic empirical oracle cannot satisfy the joint criterion on the observed score geometry.",
        "pooled_anomaly_warning": "The pooled all-anomaly row depends on the dataset category mixture and is not a category-generalization claim.",
        "category_mapping": VORAUS_CATEGORY_NAMES,
        "category_mapping_source": "official vorausrobotik/voraus-ad-dataset voraus_ad.py Category enum",
        "small_category_warning": "Category-level recall granularity is 1/n; interpret categories with small n descriptively and report n alongside results.",
        "no_inferential_ci_on_oracle_selected_metrics": True,
        "ci_reason": "Threshold is selected on the same evaluation scores, so ordinary binomial/Wilson intervals would ignore selection and overstate inferential validity.",
    }
    with open(args.output_dir / "m2_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nCategory provenance audit:")
    print(category_audit.to_string(index=False))
    print("\nM2 summary (N=100):")
    print(summary_df[summary_df["commissioning_size"] == max(args.commissioning)].to_string(index=False))
    print(f"\nSaved M2 outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
