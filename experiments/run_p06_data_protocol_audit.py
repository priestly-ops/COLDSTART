"""P0.6 pre-redesign audit: data-budget and 1% calibration feasibility.

This script does not fit or tune any detector. It audits the voraus-AD metadata
and the arithmetic imposed by a 1% false-alert target before we change the
commissioning protocol again.

Outputs
-------
- p06_setting_counts.csv: episode counts by setting/anomaly flag.
- p06_normal_setting_counts.csv: normal episode counts by setting.
- p06_calibration_feasibility.csv: split-conformal rank behavior versus
  calibration size, including whether the threshold is forced to the maximum.
- p06_budget_feasibility.csv: feasible train/calibration/evaluation allocations
  under the observed PRE_B healthy pool.
- p06_manifest.json: immutable audit metadata.

The key goal is to distinguish three different sample counts that were mixed in
P0.5: target model-fitting cycles, target calibration cycles, and held-out test
cycles. Only the first two are deployment/commissioning data. Held-out test
cycles are an evaluation resource, not a deployment resource.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Support direct execution via:
#   .venv/bin/python experiments/run_p06_data_protocol_audit.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_generator import SOURCE_SETTING, TARGET_SETTING
from src.voraus_loader import load_cycle_metadata

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "voraus-ad-dataset-100hz.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p06_data_protocol_audit"
PROTOCOL_VERSION = "p06-data-protocol-audit-v1"
ALPHA = 0.01


def _conformal_rank(n_cal: int, alpha: float) -> int:
    return min(int(np.ceil((int(n_cal) + 1) * (1.0 - float(alpha)))), int(n_cal))


def _raw_conformal_rank(n_cal: int, alpha: float) -> int:
    return int(np.ceil((int(n_cal) + 1) * (1.0 - float(alpha))))


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = load_cycle_metadata(dataset)
    rows = []
    for c in metadata:
        rows.append(
            {
                "episode_id": int(c.episode_id),
                "setting": int(c.setting),
                "anomaly": bool(c.anomaly),
                "category": int(c.category),
            }
        )
    df = pd.DataFrame(rows)

    setting_counts = (
        df.groupby(["setting", "anomaly"], sort=True)
        .size()
        .rename("episodes")
        .reset_index()
    )
    setting_counts.to_csv(output / "p06_setting_counts.csv", index=False)

    normal_counts = (
        df.loc[~df.anomaly]
        .groupby("setting", sort=True)
        .size()
        .rename("healthy_episodes")
        .reset_index()
    )
    normal_counts.to_csv(output / "p06_normal_setting_counts.csv", index=False)

    target_healthy = int(((df.setting == int(TARGET_SETTING)) & (~df.anomaly)).sum())
    source_healthy = int(((df.setting == int(SOURCE_SETTING)) & (~df.anomaly)).sum())
    non_source_healthy = int(((df.setting != int(SOURCE_SETTING)) & (~df.anomaly)).sum())
    all_healthy = int((~df.anomaly).sum())
    anomalies = int(df.anomaly.sum())

    cal_rows = []
    for m in sorted(set(list(range(25, 121, 5)) + [125, 150, 175, 190, 195, 198, 199, 200, 225, 250, 275, 299, 300])):
        raw_rank = _raw_conformal_rank(m, args.alpha)
        used_rank = _conformal_rank(m, args.alpha)
        exceedances_above_threshold = max(0, int(m) - int(used_rank))
        cal_rows.append(
            {
                "calibration_n": int(m),
                "alpha": float(args.alpha),
                "raw_rank": int(raw_rank),
                "used_rank": int(used_rank),
                "threshold_forced_to_max": bool(used_rank == int(m)),
                "calibration_scores_strictly_above_threshold": int(exceedances_above_threshold),
                "minimum_attainable_nonzero_pvalue": float(1.0 / (int(m) + 1)),
            }
        )
    pd.DataFrame(cal_rows).to_csv(output / "p06_calibration_feasibility.csv", index=False)

    budget_rows = []
    eval_sizes = sorted(set([0, 50, 100] + [int(v) for v in args.eval_sizes]))
    fit_sizes = sorted(set([10, 25, 50, 75, 100, 119] + [int(v) for v in args.fit_sizes]))
    cal_sizes = sorted(set([50, 100, 150, 175, 199, 200, 225, 250] + [int(v) for v in args.calibration_sizes]))
    for eval_n in eval_sizes:
        for fit_n in fit_sizes:
            for cal_n in cal_sizes:
                deployment_budget = fit_n + cal_n
                total_target_required = deployment_budget + eval_n
                budget_rows.append(
                    {
                        "target_healthy_available": target_healthy,
                        "fit_n": fit_n,
                        "calibration_n": cal_n,
                        "heldout_eval_n": eval_n,
                        "deployment_commissioning_budget": deployment_budget,
                        "total_target_healthy_required_for_experiment": total_target_required,
                        "feasible_with_PRE_B_only": bool(total_target_required <= target_healthy),
                        "threshold_forced_to_max_at_alpha": bool(_conformal_rank(cal_n, args.alpha) == cal_n),
                    }
                )
    pd.DataFrame(budget_rows).to_csv(output / "p06_budget_feasibility.csv", index=False)

    non_source_normals = normal_counts[normal_counts.setting != int(SOURCE_SETTING)].copy()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(dataset),
        "alpha": float(args.alpha),
        "source_setting": int(SOURCE_SETTING),
        "target_setting": int(TARGET_SETTING),
        "source_healthy_episodes": source_healthy,
        "target_PRE_B_healthy_episodes": target_healthy,
        "all_non_PRE_A_healthy_episodes": non_source_healthy,
        "all_healthy_episodes": all_healthy,
        "anomaly_episodes": anomalies,
        "non_PRE_A_normal_settings": {
            str(int(r.setting)): int(r.healthy_episodes) for r in non_source_normals.itertuples(index=False)
        },
        "important_note": (
            "Deployment commissioning budget should count target fitting + target calibration cycles. "
            "Held-out evaluation cycles are test resources and should not be counted as deployment data."
        ),
    }
    (output / "p06_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print("\nCalibration rows near the 1% rank transition:")
    cal_df = pd.DataFrame(cal_rows)
    print(cal_df[cal_df.calibration_n.between(190, 200)].to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--eval-sizes", type=int, nargs="*", default=[])
    ap.add_argument("--fit-sizes", type=int, nargs="*", default=[])
    ap.add_argument("--calibration-sizes", type=int, nargs="*", default=[])
    args = ap.parse_args()
    if not (0.0 < args.alpha < 1.0):
        ap.error("alpha must be in (0,1)")
    return args


if __name__ == "__main__":
    run(parse_args())
