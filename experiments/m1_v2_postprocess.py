"""M1-v2 post-processing for COLDSTART calibration-tail sensitivity.

This script does NOT refit any detector. It consumes the outputs from the
original M1 run and produces publication-oriented diagnostics that:

1. exclude finite-sample-infeasible cells from performance summaries;
2. explicitly separate the conformal feasibility floor from the interiority
   floor (the point where the selected rank stops being the sample maximum);
3. show all commissioning seeds rather than relying on smooth mean curves;
4. expose extreme-threshold / near-zero-recall seed behavior;
5. retain mean/median summaries only as secondary descriptive statistics.

Expected input files in --input-dir:
    m1_seed_results.csv
    m1_rank_table.csv
    m1_manifest.json

Default output directory:
    outputs/m1_calibration_tail_v2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTOCOL_VERSION = "m1-calibration-tail-v2-postprocess"
RECALL_TARGET = 0.90
RECALL_COLLAPSE_THRESHOLD = 0.10
THRESHOLD_EXTREME_MULTIPLIER = 10.0
GLOBAL_SEED = 42


def feasibility_floor(alpha: float) -> int:
    """Smallest M for which a finite deterministic split-conformal threshold exists.

    With rank k = ceil((M+1)(1-alpha)), feasibility requires k <= M.
    This is equivalent to M >= ceil(1/alpha) - 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return int(math.ceil(1.0 / alpha) - 1)


def interiority_floor(alpha: float) -> int:
    """Smallest M for which the conformal rank is strictly below the sample maximum.

    Interiority means k <= M-1, where k = ceil((M+1)(1-alpha)).
    This is equivalent to M >= ceil(2/alpha - 1), with integer care handled
    by direct upward rounding.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    # Need M >= 2/alpha - 1. Ceil gives the smallest integer satisfying it.
    return int(math.ceil((2.0 / alpha) - 1.0))


def _rank(M: int, alpha: float) -> int:
    return int(math.ceil((M + 1) * (1.0 - alpha)))


def verify_floor_formula(alpha: float) -> tuple[bool, bool]:
    """Verify both closed-form floors by direct conformal-rank arithmetic."""
    f = feasibility_floor(alpha)
    i = interiority_floor(alpha)
    feasibility_ok = _rank(f, alpha) <= f and (f == 0 or _rank(f - 1, alpha) > f - 1)
    interiority_ok = _rank(i, alpha) <= i - 1 and (i == 0 or _rank(i - 1, alpha) > i - 2)
    return feasibility_ok, interiority_ok


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    seed_path = input_dir / "m1_seed_results.csv"
    rank_path = input_dir / "m1_rank_table.csv"
    manifest_path = input_dir / "m1_manifest.json"
    for path in (seed_path, rank_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required M1 file: {path}")
    seed_df = pd.read_csv(seed_path)
    rank_df = pd.read_csv(rank_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return seed_df, rank_df, manifest


def make_alpha_floor_table(alphas: Iterable[float]) -> pd.DataFrame:
    rows = []
    for alpha in sorted(set(float(a) for a in alphas)):
        feasibility_verified, interiority_verified = verify_floor_formula(alpha)
        rows.append(
            {
                "alpha": alpha,
                "feasibility_floor_M": feasibility_floor(alpha),
                "interiority_floor_M": interiority_floor(alpha),
                "feasibility_floor_verified_by_rank": feasibility_verified,
                "interiority_floor_verified_by_rank": interiority_verified,
                "interpretation": (
                    "finite threshold possible at/above feasibility floor; "
                    "rank stops being sample maximum only at/above interiority floor"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_tested_feasibility_table(rank_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "calibration_size",
        "alpha",
        "raw_rank",
        "finite_sample_feasible",
        "minimum_attainable_alpha",
        "threshold_is_maximum",
    ]
    out = rank_df[cols].copy().drop_duplicates().sort_values(["calibration_size", "alpha"])
    out["feasibility_floor_M"] = out["alpha"].map(feasibility_floor)
    out["interiority_floor_M"] = out["alpha"].map(interiority_floor)
    out["interior_threshold"] = out["finite_sample_feasible"] & (~out["threshold_is_maximum"])
    return out


def make_seed_outcomes(seed_df: pd.DataFrame) -> pd.DataFrame:
    # M1-v2 performance analysis is deliberately restricted to the primary
    # TargetOnly, all-fault rows. Fault-level separability is deferred to M2/M4.
    primary = seed_df[
        (seed_df["detector"] == "TargetOnly")
        & (seed_df["fault_name"] == "all_faults")
        & (seed_df["finite_sample_feasible"].astype(bool))
    ].copy()
    if primary.empty:
        raise RuntimeError("No feasible TargetOnly/all_faults rows found.")

    cell = ["commissioning_size", "calibration_size", "alpha"]
    primary["cell_median_threshold"] = primary.groupby(cell)["threshold"].transform("median")
    primary["threshold_ratio_to_cell_median"] = (
        primary["threshold"] / primary["cell_median_threshold"]
    )
    primary["recall_collapse_lt_0p10"] = primary["recall"] < RECALL_COLLAPSE_THRESHOLD
    primary["threshold_extreme_gt_10x_cell_median"] = (
        primary["threshold_ratio_to_cell_median"] > THRESHOLD_EXTREME_MULTIPLIER
    )
    primary["descriptive_instability_flag"] = (
        primary["recall_collapse_lt_0p10"]
        | primary["threshold_extreme_gt_10x_cell_median"]
    )
    primary["condition"] = primary.apply(
        lambda r: f"M={int(r['calibration_size'])}, α={r['alpha']:g}", axis=1
    )
    return primary.sort_values(cell + ["seed"]).reset_index(drop=True)


def make_group_summary(seed_outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cell = ["commissioning_size", "calibration_size", "alpha"]
    for key, group in seed_outcomes.groupby(cell, sort=True):
        recall = group["recall"].to_numpy(float)
        fpr = group["false_positive_rate"].to_numpy(float)
        threshold = group["threshold"].to_numpy(float)
        stable_seed = GLOBAL_SEED + int(sum(key) * 1000) % 100000
        recall_lo, recall_hi = bootstrap_mean_ci(recall, seed=stable_seed)
        fpr_lo, fpr_hi = bootstrap_mean_ci(fpr, seed=stable_seed + 1)
        rows.append(
            {
                "commissioning_size": key[0],
                "calibration_size": key[1],
                "alpha": key[2],
                "number_of_seeds": len(group),
                "recall_mean": float(np.mean(recall)),
                "recall_median": float(np.median(recall)),
                "recall_min": float(np.min(recall)),
                "recall_max": float(np.max(recall)),
                "recall_mean_ci_lower": recall_lo,
                "recall_mean_ci_upper": recall_hi,
                "fpr_mean": float(np.mean(fpr)),
                "fpr_median": float(np.median(fpr)),
                "fpr_mean_ci_lower": fpr_lo,
                "fpr_mean_ci_upper": fpr_hi,
                "threshold_mean": float(np.mean(threshold)),
                "threshold_median": float(np.median(threshold)),
                "threshold_min": float(np.min(threshold)),
                "threshold_max": float(np.max(threshold)),
                "seed_success_rate": float(group["success"].astype(bool).mean()),
                "recall_collapse_lt_0p10_count": int(group["recall_collapse_lt_0p10"].sum()),
                "threshold_extreme_gt_10x_cell_median_count": int(
                    group["threshold_extreme_gt_10x_cell_median"].sum()
                ),
                "descriptive_instability_flag_count": int(group["descriptive_instability_flag"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(cell).reset_index(drop=True)


def make_extreme_audit(seed_outcomes: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "commissioning_size",
        "calibration_size",
        "alpha",
        "seed",
        "threshold",
        "cell_median_threshold",
        "threshold_ratio_to_cell_median",
        "recall",
        "false_positive_rate",
        "recall_collapse_lt_0p10",
        "threshold_extreme_gt_10x_cell_median",
        "descriptive_instability_flag",
        "calibration_first_episode_id",
        "calibration_last_episode_id",
    ]
    return seed_outcomes.loc[seed_outcomes["descriptive_instability_flag"], cols].copy()


def plot_threshold_vs_recall(seed_outcomes: pd.DataFrame, output_path: Path) -> None:
    Ns = sorted(seed_outcomes["commissioning_size"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    markers = ["o", "s", "^", "D", "P", "X"]
    conditions = (
        seed_outcomes[["calibration_size", "alpha", "condition"]]
        .drop_duplicates()
        .sort_values(["calibration_size", "alpha"])
        .reset_index(drop=True)
    )
    marker_map = {
        row.condition: markers[i % len(markers)] for i, row in conditions.iterrows()
    }

    for ax, N in zip(axes, Ns):
        panel = seed_outcomes[seed_outcomes["commissioning_size"] == N]
        for condition, group in panel.groupby("condition", sort=False):
            ax.scatter(
                group["threshold"],
                group["recall"],
                marker=marker_map[condition],
                alpha=0.72,
                label=condition,
            )
            flagged = group[group["descriptive_instability_flag"]]
            for _, row in flagged.iterrows():
                ax.annotate(
                    str(int(row["seed"])),
                    (row["threshold"], row["recall"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set_xscale("log")
        ax.axhline(RECALL_TARGET, linestyle="--", linewidth=1.0)
        ax.set_title(f"TargetOnly, N={N}")
        ax.set_xlabel("Calibration threshold (log scale)")
        ax.set_ylabel("Recall")
        ax.grid(True, alpha=0.2)

    for ax in axes[len(Ns):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=True)
    fig.suptitle(
        "M1-v2: seed-level threshold vs recall (feasible settings only)\n"
        "Annotated seed IDs meet a descriptive instability flag",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.885])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_seed_recall_by_condition(seed_outcomes: pd.DataFrame, output_path: Path) -> None:
    Ns = sorted(seed_outcomes["commissioning_size"].unique())
    conditions = (
        seed_outcomes[["calibration_size", "alpha", "condition"]]
        .drop_duplicates()
        .sort_values(["calibration_size", "alpha"])["condition"]
        .tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(GLOBAL_SEED)

    for ax, N in zip(axes, Ns):
        panel = seed_outcomes[seed_outcomes["commissioning_size"] == N]
        for x, condition in enumerate(conditions):
            group = panel[panel["condition"] == condition].sort_values("seed")
            if group.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(group))
            ax.scatter(np.full(len(group), x) + jitter, group["recall"], alpha=0.72)
            flagged = group[group["descriptive_instability_flag"]]
            for _, row in flagged.iterrows():
                # Seed location is approximated to the category center to avoid
                # implying precise x-position meaning beyond jitter.
                ax.annotate(
                    str(int(row["seed"])),
                    (x, row["recall"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.axhline(RECALL_TARGET, linestyle="--", linewidth=1.0)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=35, ha="right")
        ax.set_title(f"TargetOnly, N={N}")
        ax.set_ylabel("Recall")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[len(Ns):]:
        ax.axis("off")
    fig.suptitle(
        "M1-v2: all 20 seed recalls by feasible calibration condition\n"
        "Points are individual commissioning seeds; no smooth mean curve is implied",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_manifest(output_dir: Path, source_manifest: dict, seed_outcomes: pd.DataFrame) -> None:
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_protocol_version": source_manifest.get("protocol_version"),
        "postprocess_only": True,
        "detector_refit": False,
        "performance_scope": "TargetOnly / all_faults / finite-sample-feasible cells only",
        "source_seeds": source_manifest.get("seeds"),
        "source_commissioning_sizes": source_manifest.get("commissioning_sizes"),
        "source_calibration_sizes": source_manifest.get("calibration_sizes"),
        "source_alphas": source_manifest.get("alphas"),
        "feasibility_definition": "k=ceil((M+1)(1-alpha)) <= M",
        "feasibility_floor_formula": "M >= ceil(1/alpha) - 1",
        "interiority_definition": "k=ceil((M+1)(1-alpha)) <= M-1",
        "interiority_floor_formula": "M >= ceil(2/alpha - 1)",
        "descriptive_flags": {
            "recall_collapse": f"recall < {RECALL_COLLAPSE_THRESHOLD}",
            "threshold_extreme": (
                f"threshold > {THRESHOLD_EXTREME_MULTIPLIER}x median threshold within the same (N,M,alpha) cell"
            ),
            "instability_flag": "logical OR of the two descriptive flags; not a formal endpoint",
        },
        "publication_interpretation": {
            "infeasible_cells": "audit/feasibility only; excluded from recall/FPR performance summaries and plots",
            "fault_level_claims": "not made in M1-v2; deferred to M2/M4",
            "mean_curves": "not used as primary visualization",
            "large_M_asymptotics": "not identifiable with frozen voraus-AD healthy pool (M <= 119 compatible with frozen eval)",
        },
        "feasible_primary_rows": int(len(seed_outcomes)),
    }
    (output_dir / "m1_v2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/m1_calibration_tail"),
        help="Directory containing M1-v1 CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m1_calibration_tail_v2"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_df, rank_df, manifest = load_inputs(args.input_dir)

    alpha_floors = make_alpha_floor_table(manifest.get("alphas", rank_df["alpha"].unique()))
    tested_feasibility = make_tested_feasibility_table(rank_df)
    seed_outcomes = make_seed_outcomes(seed_df)
    group_summary = make_group_summary(seed_outcomes)
    extreme_audit = make_extreme_audit(seed_outcomes)

    alpha_floors.to_csv(args.output_dir / "m1_v2_alpha_floors.csv", index=False)
    tested_feasibility.to_csv(args.output_dir / "m1_v2_tested_feasibility.csv", index=False)
    seed_outcomes.to_csv(args.output_dir / "m1_v2_seed_outcomes.csv", index=False)
    group_summary.to_csv(args.output_dir / "m1_v2_group_summary.csv", index=False)
    extreme_audit.to_csv(args.output_dir / "m1_v2_extreme_seed_audit.csv", index=False)

    plot_threshold_vs_recall(seed_outcomes, args.output_dir / "m1_v2_threshold_vs_recall_by_N.png")
    plot_seed_recall_by_condition(seed_outcomes, args.output_dir / "m1_v2_seed_recall_by_condition.png")
    write_manifest(args.output_dir, manifest, seed_outcomes)

    print("M1-v2 post-processing complete. No detector refits were performed.")
    print("\nAlpha floors:")
    print(alpha_floors.to_string(index=False))
    print("\nTested feasibility:")
    print(tested_feasibility.to_string(index=False))
    print("\nFeasible all-fault group summary:")
    print(group_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nDescriptive instability rows: {len(extreme_audit)}")
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
