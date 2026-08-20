"""P0.6b healthy-only audit of PRE_B vs BETWEEN_A/B/C normal domains.

Purpose
-------
Determine whether settings 74/75/76 can serve as the primary held-out healthy
FPR evaluation population for a redesigned commissioning-budget experiment.
No anomaly labels are used anywhere in this audit.

Primary compatibility check
---------------------------
Across 20 PRE_B-only fit/calibration splits (fit=50, calibration=199 by
default), compare:
  * same-domain PRE_B holdout FPR, and
  * pooled BETWEEN_A/B/C FPR.

The pooled BETWEEN set is declared compatible for primary FPR evaluation only
if BOTH predeclared checks pass:
  1. upper 95% bootstrap CI of pooled BETWEEN mean FPR <= 0.02; and
  2. upper 95% bootstrap CI of (BETWEEN FPR - PRE_B holdout FPR) <= 0.01.

Individual BETWEEN settings are reported diagnostically because their sample
sizes are small (37/44/19). Distribution-shift diagnostics are descriptive and
are not used to tune RACE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.base_detector import BaseDetector
from src.feature_extractor import load_feature_batch
from src.precision_transfer_estimators import fit_robust_scaler

PROTOCOL_VERSION = "p06b-between-domain-audit-v1"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "cache" / "voraus_measured_all_features.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p06b_between_domain_audit"
PRE_B = 73
BETWEEN_SETTINGS = (74, 75, 76)
ALPHA = 0.01


def _scores(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    d = x - mean
    q = np.einsum("ni,ij,nj->n", d, precision, d, optimize=True)
    return np.sqrt(np.maximum(q, 0.0))


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    return float(np.mean(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired vectors must have equal length")
    d = a - b
    return _bootstrap_mean(d, n_boot, seed)


def _relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro") / max(np.linalg.norm(a, ord="fro"), 1e-12))


def _healthy_rows(batch, setting: int) -> np.ndarray:
    mask = (~batch.anomaly_labels) & (batch.settings == int(setting))
    return np.asarray(batch.features[mask], dtype=np.float64)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    batch = load_feature_batch(Path(args.feature_cache).resolve())

    pre_b_raw = _healthy_rows(batch, PRE_B)
    between_raw = {s: _healthy_rows(batch, s) for s in BETWEEN_SETTINGS}
    between_pooled_raw = np.vstack([between_raw[s] for s in BETWEEN_SETTINGS])

    required = int(args.fit_n) + int(args.calibration_n) + int(args.min_holdout_n)
    if len(pre_b_raw) < required:
        raise ValueError(
            f"Need at least {required} PRE_B healthy cycles for fit/calibration/holdout; have {len(pre_b_raw)}"
        )

    seed_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(pre_b_raw))
        fit_idx = order[: args.fit_n]
        cal_idx = order[args.fit_n : args.fit_n + args.calibration_n]
        hold_idx = order[args.fit_n + args.calibration_n :]

        fit_raw = pre_b_raw[fit_idx]
        cal_raw = pre_b_raw[cal_idx]
        hold_raw = pre_b_raw[hold_idx]

        scaler = fit_robust_scaler(fit_raw, mode="target")
        fit_x = scaler.transform(fit_raw)
        cal_x = scaler.transform(cal_raw)
        hold_x = scaler.transform(hold_raw)
        between_x = {s: scaler.transform(between_raw[s]) for s in BETWEEN_SETTINGS}
        pooled_x = scaler.transform(between_pooled_raw)

        lw = LedoitWolf(store_precision=True, assume_centered=False).fit(fit_x)
        mean = np.asarray(lw.location_, dtype=np.float64)
        precision = np.asarray(lw.precision_, dtype=np.float64)

        cal_scores = _scores(cal_x, mean, precision)
        threshold = BaseDetector.conformal_quantile(cal_scores, alpha=float(args.alpha))
        pre_b_scores = _scores(hold_x, mean, precision)
        pooled_scores = _scores(pooled_x, mean, precision)

        row: dict[str, object] = {
            "seed": int(seed),
            "fit_n": int(len(fit_x)),
            "calibration_n": int(len(cal_x)),
            "pre_b_holdout_n": int(len(hold_x)),
            "threshold": float(threshold),
            "threshold_is_calibration_max": bool(np.isclose(threshold, np.max(cal_scores))),
            "pre_b_holdout_fpr": float(np.mean(pre_b_scores > threshold)),
            "between_pooled_fpr": float(np.mean(pooled_scores > threshold)),
            "between_pooled_ks": float(ks_2samp(pre_b_scores, pooled_scores).statistic),
            "between_pooled_wasserstein": float(wasserstein_distance(pre_b_scores, pooled_scores)),
        }
        for s in BETWEEN_SETTINGS:
            sc = _scores(between_x[s], mean, precision)
            row[f"setting_{s}_n"] = int(len(sc))
            row[f"setting_{s}_fpr"] = float(np.mean(sc > threshold))
            row[f"setting_{s}_ks"] = float(ks_2samp(pre_b_scores, sc).statistic)
            row[f"setting_{s}_wasserstein"] = float(wasserstein_distance(pre_b_scores, sc))
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(output / "p06b_seed_results.csv", index=False)

    pb_mean, pb_lo, pb_hi = _bootstrap_mean(
        seed_df.pre_b_holdout_fpr.to_numpy(), args.bootstrap_replicates, 61_001
    )
    bt_mean, bt_lo, bt_hi = _bootstrap_mean(
        seed_df.between_pooled_fpr.to_numpy(), args.bootstrap_replicates, 61_002
    )
    diff_mean, diff_lo, diff_hi = _bootstrap_diff(
        seed_df.between_pooled_fpr.to_numpy(),
        seed_df.pre_b_holdout_fpr.to_numpy(),
        args.bootstrap_replicates,
        61_003,
    )

    summary_rows = [
        {
            "domain": "PRE_B_holdout",
            "episodes_per_seed": float(seed_df.pre_b_holdout_n.mean()),
            "mean_fpr": pb_mean,
            "fpr_ci_lower": pb_lo,
            "fpr_ci_upper": pb_hi,
            "mean_delta_fpr_vs_PRE_B": 0.0,
            "delta_ci_lower": 0.0,
            "delta_ci_upper": 0.0,
        },
        {
            "domain": "BETWEEN_pooled",
            "episodes_per_seed": int(sum(len(between_raw[s]) for s in BETWEEN_SETTINGS)),
            "mean_fpr": bt_mean,
            "fpr_ci_lower": bt_lo,
            "fpr_ci_upper": bt_hi,
            "mean_delta_fpr_vs_PRE_B": diff_mean,
            "delta_ci_lower": diff_lo,
            "delta_ci_upper": diff_hi,
        },
    ]
    for j, s in enumerate(BETWEEN_SETTINGS):
        vals = seed_df[f"setting_{s}_fpr"].to_numpy()
        m, lo, hi = _bootstrap_mean(vals, args.bootstrap_replicates, 61_100 + j)
        dm, dlo, dhi = _bootstrap_diff(vals, seed_df.pre_b_holdout_fpr.to_numpy(), args.bootstrap_replicates, 61_200 + j)
        summary_rows.append({
            "domain": f"setting_{s}",
            "episodes_per_seed": int(len(between_raw[s])),
            "mean_fpr": m,
            "fpr_ci_lower": lo,
            "fpr_ci_upper": hi,
            "mean_delta_fpr_vs_PRE_B": dm,
            "delta_ci_lower": dlo,
            "delta_ci_upper": dhi,
        })
    pd.DataFrame(summary_rows).to_csv(output / "p06b_fpr_summary.csv", index=False)

    # Full-domain descriptive geometry, standardized using PRE_B only.
    full_scaler = fit_robust_scaler(pre_b_raw, mode="target")
    pb = full_scaler.transform(pre_b_raw)
    pb_mean_vec = np.mean(pb, axis=0)
    pb_cov = LedoitWolf(store_precision=False).fit(pb).covariance_
    geometry = []
    for name, raw in [("BETWEEN_pooled", between_pooled_raw)] + [(f"setting_{s}", between_raw[s]) for s in BETWEEN_SETTINGS]:
        x = full_scaler.transform(raw)
        x_cov = LedoitWolf(store_precision=False).fit(x).covariance_
        geometry.append({
            "domain": name,
            "n": int(len(x)),
            "standardized_mean_l2": float(np.linalg.norm(np.mean(x, axis=0) - pb_mean_vec)),
            "relative_covariance_frobenius": _relative_frobenius(pb_cov, x_cov),
        })
    pd.DataFrame(geometry).to_csv(output / "p06b_geometry_summary.csv", index=False)

    checks = {
        "between_pooled_fpr_ci_upper_le_0_02": bool(bt_hi <= 0.02),
        "between_minus_preb_fpr_ci_upper_le_0_01": bool(diff_hi <= 0.01),
    }
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "decision": "BETWEEN_PRIMARY_FPR_COMPATIBLE" if all(checks.values()) else "BETWEEN_ROBUSTNESS_ONLY",
        "primary_gate_pass": bool(all(checks.values())),
        "checks": checks,
        "frozen_thresholds": {
            "max_between_pooled_fpr_ci_upper": 0.02,
            "max_between_minus_preb_fpr_ci_upper": 0.01,
        },
        "fit_n": int(args.fit_n),
        "calibration_n": int(args.calibration_n),
        "alpha": float(args.alpha),
        "seeds": [int(s) for s in args.seeds],
        "anomaly_labels_used": False,
    }
    (output / "p06b_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print("\nFPR summary\n", pd.DataFrame(summary_rows).to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--fit-n", type=int, default=50)
    ap.add_argument("--calibration-n", type=int, default=199)
    ap.add_argument("--min-holdout-n", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = ap.parse_args()
    if args.fit_n < 2 or args.calibration_n < 2 or args.min_holdout_n < 1:
        ap.error("fit-n>=2, calibration-n>=2, min-holdout-n>=1 required")
    if not (0.0 < args.alpha < 1.0):
        ap.error("alpha must be in (0,1)")
    return args


if __name__ == "__main__":
    run(parse_args())
