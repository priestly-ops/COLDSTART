"""P0.10: finite-sample conformal calibration + threshold-margin theory audit.

This script does not refit or retune any detector. It combines:
  1) exact split-conformal order-statistic arithmetic,
  2) frozen P0.7 voraus commissioning summaries, and
  3) frozen P0.9 AURSAD seed-level score-geometry diagnostics.

The goal is to make two paper-level claims auditable:

A. Finite calibration size discretizes the attainable false-alert level. For
   calibration size m and target alpha, the standard split-conformal rank is

       k = ceil((m + 1) * (1 - alpha)).

   If k > m, no finite observed calibration score can provide the requested
   deterministic split-conformal guarantee; strict calibration uses +inf.
   If k == m, the threshold is necessarily the maximum calibration score.

B. Operational recall depends on anomaly-to-threshold margin, not AUROC alone.
   For anomaly class c and commissioning budget B, define the diagnostic

       M_c(B) = median_z(anomaly_c; B) - threshold_z(B),

   where z uses the calibration median and IQR from the frozen P0.9 run.
   Negative drift Delta M_c means the operational decision boundary moves away
   from the anomaly class even if ranking/AUROC improves.

References/positioning
----------------------
The conformal rank formula is the standard finite-sample split-conformal order
statistic. The repository already implements strict feasibility diagnostics in
src/calibration_tail.py and exercises them in M1. P0.10 reuses those semantics
rather than inventing a new threshold rule after seeing anomaly labels.

Outputs
-------
  p10_conformal_rank_table.csv
  p10_calibration_landmarks.csv
  p10_voraus_commissioning_bound.csv          (when P0.7 inputs exist)
  p10_aursad_threshold_margin_drift.csv       (when P0.9 inputs exist)
  p10_aursad_auroc_recall_drift.csv           (when P0.9 inputs exist)
  p10_manifest.json
  p10_manuscript_claims.md
  p10_conformal_tail_resolution.png
  p10_aursad_auroc_vs_recall_drift.png        (when P0.9 inputs exist)

Run from repository root:

    .venv/bin/python experiments/run_p10_calibration_margin_theory.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROTOCOL_VERSION = "p10-calibration-margin-theory-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p10_calibration_margin_theory"
DEFAULT_P07_SUMMARY = PROJECT_ROOT / "outputs" / "p07_fixed_budget_commissioning" / "p07_summary.csv"
DEFAULT_P07_BSTAR = PROJECT_ROOT / "outputs" / "p07_fixed_budget_commissioning" / "p07_b_star.json"
DEFAULT_P09_SEEDS = PROJECT_ROOT / "outputs" / "p09_aursad_score_geometry" / "p09_seed_diagnostics.csv"
DEFAULT_ALPHAS = (0.005, 0.01, 0.02)
DEFAULT_MAX_CALIBRATION = 500
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_FPR_TARGET = 0.01
DEFAULT_DRIFT_START = 224
DEFAULT_DRIFT_END = 400
CATEGORY_NAMES = {
    1: "damaged_screw",
    2: "extra_component",
    3: "missing_screw",
    4: "damaged_thread",
}


def _parse_float_list(values: list[str]) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _rank_info(m: int, alpha: float) -> dict[str, object]:
    if m < 1:
        raise ValueError("calibration size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    raw_rank = int(math.ceil((m + 1) * (1.0 - alpha)))
    feasible = raw_rank <= m
    threshold_is_maximum = bool(feasible and raw_rank == m)
    nonmax = bool(feasible and raw_rank < m)
    # If threshold is the k-th order statistic and prediction is score > q,
    # there are m-k observed calibration scores strictly above q at most
    # (ignoring ties). This is descriptive rank geometry, not a new guarantee.
    exceedance_slots = int(m - raw_rank) if feasible else 0
    return {
        "calibration_n": int(m),
        "alpha": float(alpha),
        "raw_rank": int(raw_rank),
        "finite_sample_feasible": bool(feasible),
        "threshold_is_maximum": threshold_is_maximum,
        "threshold_is_nonmax": nonmax,
        "minimum_attainable_alpha": 1.0 / float(m + 1),
        "rank_exceedance_slots": exceedance_slots,
    }


def _make_rank_table(max_m: int, alphas: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for alpha in alphas:
        for m in range(1, max_m + 1):
            rows.append(_rank_info(m, alpha))
    return pd.DataFrame(rows)


def _landmarks(rank_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha, g in rank_table.groupby("alpha", sort=True):
        g = g.sort_values("calibration_n")
        feasible = g[g.finite_sample_feasible]
        nonmax = g[g.threshold_is_nonmax]
        max_rows = g[g.threshold_is_maximum]
        rows.append({
            "alpha": float(alpha),
            "minimum_calibration_for_finite_threshold": (
                int(feasible.calibration_n.min()) if not feasible.empty else np.nan
            ),
            "minimum_calibration_for_nonmax_threshold": (
                int(nonmax.calibration_n.min()) if not nonmax.empty else np.nan
            ),
            "largest_calibration_still_forced_to_maximum": (
                int(max_rows.calibration_n.max()) if not max_rows.empty else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _plot_tail_resolution(rank_table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for alpha, g in rank_table.groupby("alpha", sort=True):
        g = g.sort_values("calibration_n")
        ax.plot(
            g.calibration_n,
            g.minimum_attainable_alpha,
            label=f"target alpha={alpha:g}",
        )
        ax.axhline(float(alpha), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Healthy calibration cycles m")
    ax.set_ylabel("1 / (m + 1)")
    ax.set_yscale("log")
    ax.set_title("Finite-sample calibration tail resolution")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "p10_conformal_tail_resolution.png", dpi=220)
    plt.close(fig)


def _voraus_bound(
    summary_path: Path,
    bstar_path: Path,
    recall_target: float,
    fpr_target: float,
) -> pd.DataFrame:
    if not summary_path.is_file():
        return pd.DataFrame()
    summary = pd.read_csv(summary_path)
    required = {"budget", "method", "recall_ci_lower", "fpr_ci_upper"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"P0.7 summary missing columns: {sorted(missing)}")

    max_budget = int(summary.budget.max())
    rows: list[dict[str, object]] = []
    bstar_json: dict[str, object] = {}
    if bstar_path.is_file():
        bstar_json = json.loads(bstar_path.read_text(encoding="utf-8"))

    for method, g in summary.groupby("method", sort=True):
        g = g.sort_values("budget")
        ok = g[
            (g.recall_ci_lower >= float(recall_target))
            & (g.fpr_ci_upper <= float(fpr_target) + 1e-12)
        ]
        if ok.empty:
            estimate = f">{max_budget}"
            status = "right_censored"
            observed = np.nan
        else:
            observed = int(ok.iloc[0].budget)
            estimate = str(observed)
            status = "observed"
        rows.append({
            "method": str(method),
            "N_star": estimate,
            "status": status,
            "observed_N_star": observed,
            "max_evaluated_budget": max_budget,
            "bstar_file_value": bstar_json.get(str(method)),
        })

    out = pd.DataFrame(rows)
    target_names = [m for m in out.method if "TargetOnly" in m]
    race_names = [m for m in out.method if "RACECovSafeCV" in m]
    if target_names and race_names:
        t = out[out.method == target_names[0]].iloc[0]
        r = out[out.method == race_names[0]].iloc[0]
        if t.status == "right_censored" and r.status == "observed":
            race_n = int(r.observed_N_star)
            lower_abs = max_budget - race_n
            lower_rel = 1.0 - race_n / float(max_budget)
            out["race_vs_target_min_absolute_saving"] = np.nan
            out["race_vs_target_min_relative_saving"] = np.nan
            out.loc[out.method == race_names[0], "race_vs_target_min_absolute_saving"] = lower_abs
            out.loc[out.method == race_names[0], "race_vs_target_min_relative_saving"] = lower_rel
    return out


def _paired_drift(p09_path: Path, start_budget: int, end_budget: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not p09_path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(p09_path)
    required = {"budget", "seed", "method", "threshold_robust_z", "recall", "auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"P0.9 seed diagnostics missing columns: {sorted(missing)}")

    start = df[df.budget.eq(start_budget)].copy()
    end = df[df.budget.eq(end_budget)].copy()
    paired = start.merge(end, on=["seed", "method"], suffixes=("_start", "_end"), validate="one_to_one")
    if paired.empty:
        raise ValueError(f"No paired P0.9 rows for B={start_budget} and B={end_budget}")

    drift_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for method, g in paired.groupby("method", sort=True):
        delta_threshold = g.threshold_robust_z_end - g.threshold_robust_z_start
        delta_recall = g.recall_end - g.recall_start
        delta_auroc = g.auroc_end - g.auroc_start
        rank_rows.append({
            "method": method,
            "start_budget": start_budget,
            "end_budget": end_budget,
            "paired_seeds": int(len(g)),
            "delta_threshold_robust_z_mean": float(delta_threshold.mean()),
            "delta_recall_mean": float(delta_recall.mean()),
            "delta_auroc_mean": float(delta_auroc.mean()),
            "fraction_auroc_up_recall_down": float(np.mean((delta_auroc > 0) & (delta_recall < 0))),
        })

        for cat, name in CATEGORY_NAMES.items():
            z_col = f"cat{cat}_median_robust_z"
            m_col = f"cat{cat}_median_margin_z"
            a_col = f"cat{cat}_auroc"
            if f"{z_col}_start" not in g or f"{m_col}_start" not in g:
                continue
            delta_anomaly = g[f"{z_col}_end"] - g[f"{z_col}_start"]
            delta_margin = g[f"{m_col}_end"] - g[f"{m_col}_start"]
            row = {
                "method": method,
                "category": int(cat),
                "category_name": name,
                "start_budget": start_budget,
                "end_budget": end_budget,
                "paired_seeds": int(len(g)),
                "delta_threshold_robust_z_mean": float(delta_threshold.mean()),
                "delta_anomaly_median_robust_z_mean": float(delta_anomaly.mean()),
                "delta_margin_z_mean": float(delta_margin.mean()),
                "delta_margin_z_median": float(delta_margin.median()),
                "fraction_margin_worsened": float(np.mean(delta_margin < 0)),
            }
            if f"{a_col}_start" in g:
                row["delta_category_auroc_mean"] = float((g[f"{a_col}_end"] - g[f"{a_col}_start"]).mean())
            drift_rows.append(row)
    return pd.DataFrame(drift_rows), pd.DataFrame(rank_rows)


def _plot_auroc_recall_drift(p09_path: Path, output: Path, start_budget: int, end_budget: int) -> None:
    if not p09_path.is_file():
        return
    df = pd.read_csv(p09_path)
    start = df[df.budget.eq(start_budget)]
    end = df[df.budget.eq(end_budget)]
    paired = start.merge(end, on=["seed", "method"], suffixes=("_start", "_end"))
    if paired.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for method, g in paired.groupby("method", sort=True):
        x = g.auroc_end - g.auroc_start
        y = g.recall_end - g.recall_start
        ax.scatter(x, y, label=str(method), alpha=0.75)
    ax.axvline(0.0, linewidth=1.0)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(f"Delta AUROC (B={end_budget} - B={start_budget})")
    ax.set_ylabel(f"Delta operational recall (B={end_budget} - B={start_budget})")
    ax.set_title("Ranking can improve while operational recall worsens")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "p10_aursad_auroc_vs_recall_drift.png", dpi=220)
    plt.close(fig)


def _write_claims(output: Path, landmarks: pd.DataFrame, voraus: pd.DataFrame, drift: pd.DataFrame, ranking: pd.DataFrame) -> None:
    lines = [
        "# P0.10 manuscript-ready claims",
        "",
        "## Finite-sample calibration",
        "",
        "For split conformal with calibration size m and false-alert target alpha, the threshold rank is",
        "`k = ceil((m + 1) * (1 - alpha))`. If `k > m`, no finite observed calibration score can",
        "provide the requested deterministic finite-sample guarantee; if `k = m`, the threshold is",
        "forced to the maximum calibration score.",
        "",
    ]
    for r in landmarks.itertuples(index=False):
        lines.append(
            f"- alpha={r.alpha:g}: first finite threshold m={int(r.minimum_calibration_for_finite_threshold)}, "
            f"first non-maximum threshold m={int(r.minimum_calibration_for_nonmax_threshold)}."
        )
    if not voraus.empty:
        lines += ["", "## voraus commissioning bound", ""]
        race = voraus[voraus.method.astype(str).str.contains("RACECovSafeCV")]
        target = voraus[voraus.method.astype(str).str.contains("TargetOnly")]
        if not race.empty and not target.empty:
            rr, tt = race.iloc[0], target.iloc[0]
            lines.append(f"- RACE SafeCV N* = {rr.N_star}; TargetOnly N* = {tt.N_star}.")
            if "race_vs_target_min_absolute_saving" in voraus and pd.notna(rr.race_vs_target_min_absolute_saving):
                lines.append(
                    f"- This implies >{int(rr.race_vs_target_min_absolute_saving)} fewer healthy target cycles "
                    f"and >{100.0 * float(rr.race_vs_target_min_relative_saving):.1f}% relative reduction "
                    "with respect to the smallest TargetOnly requirement consistent with the censoring bound."
                )
    if not ranking.empty:
        lines += ["", "## AURSAD threshold-margin drift", ""]
        for r in ranking.itertuples(index=False):
            lines.append(
                f"- {r.method}: mean Delta threshold-z={r.delta_threshold_robust_z_mean:.3f}, "
                f"Delta recall={r.delta_recall_mean:.3f}, Delta AUROC={r.delta_auroc_mean:.3f}, "
                f"P(AUROC up & recall down)={r.fraction_auroc_up_recall_down:.2f}."
            )
        lines.append(
            "- Interpretation: AUROC is threshold-free, whereas deployment recall is determined by the "
            "anomaly-to-threshold margin. A detector can rank anomalies better while becoming worse at the "
            "fixed false-alert operating point."
        )
    output.joinpath("p10_manuscript_claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    alphas = tuple(float(a) for a in args.alphas)

    rank_table = _make_rank_table(int(args.max_calibration), alphas)
    landmarks = _landmarks(rank_table)
    rank_table.to_csv(output / "p10_conformal_rank_table.csv", index=False)
    landmarks.to_csv(output / "p10_calibration_landmarks.csv", index=False)
    _plot_tail_resolution(rank_table, output)

    voraus = _voraus_bound(
        Path(args.p07_summary),
        Path(args.p07_bstar),
        float(args.recall_target),
        float(args.fpr_target),
    )
    if not voraus.empty:
        voraus.to_csv(output / "p10_voraus_commissioning_bound.csv", index=False)

    drift, ranking = _paired_drift(
        Path(args.p09_seed_diagnostics),
        int(args.drift_start_budget),
        int(args.drift_end_budget),
    )
    if not drift.empty:
        drift.to_csv(output / "p10_aursad_threshold_margin_drift.csv", index=False)
    if not ranking.empty:
        ranking.to_csv(output / "p10_aursad_auroc_recall_drift.csv", index=False)
    _plot_auroc_recall_drift(
        Path(args.p09_seed_diagnostics),
        output,
        int(args.drift_start_budget),
        int(args.drift_end_budget),
    )
    _write_claims(output, landmarks, voraus, drift, ranking)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_only": True,
        "detectors_refit_or_retuned": False,
        "alphas": list(alphas),
        "max_calibration": int(args.max_calibration),
        "conformal_rank_formula": "ceil((m + 1) * (1 - alpha))",
        "strict_infeasible_threshold": "+inf",
        "p07_summary": str(Path(args.p07_summary).resolve()),
        "p07_summary_exists": Path(args.p07_summary).is_file(),
        "p09_seed_diagnostics": str(Path(args.p09_seed_diagnostics).resolve()),
        "p09_seed_diagnostics_exists": Path(args.p09_seed_diagnostics).is_file(),
        "drift_start_budget": int(args.drift_start_budget),
        "drift_end_budget": int(args.drift_end_budget),
        "threshold_margin_definition": "median anomaly robust-z minus threshold robust-z",
        "anomaly_labels_used_for_fit_selection_or_calibration": False,
    }
    (output / "p10_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("P0.10 complete")
    print("\nCalibration landmarks")
    print(landmarks.to_string(index=False))
    if not voraus.empty:
        print("\nvoraus commissioning bound")
        print(voraus.to_string(index=False))
    if not ranking.empty:
        print("\nAURSAD AUROC/recall drift")
        print(ranking.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--p07-summary", type=Path, default=DEFAULT_P07_SUMMARY)
    parser.add_argument("--p07-bstar", type=Path, default=DEFAULT_P07_BSTAR)
    parser.add_argument("--p09-seed-diagnostics", type=Path, default=DEFAULT_P09_SEEDS)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--max-calibration", type=int, default=DEFAULT_MAX_CALIBRATION)
    parser.add_argument("--recall-target", type=float, default=DEFAULT_RECALL_TARGET)
    parser.add_argument("--fpr-target", type=float, default=DEFAULT_FPR_TARGET)
    parser.add_argument("--drift-start-budget", type=int, default=DEFAULT_DRIFT_START)
    parser.add_argument("--drift-end-budget", type=int, default=DEFAULT_DRIFT_END)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
