"""P0.2A: implementation-validation benchmark for precision transfer.

This stage is intentionally small. It validates the Python CLIME/Trans-CLIME
implementation on known Gaussian precision matrices before any p=128/256/564
experiment or real robotics transfer analysis is allowed.

Design safeguards
-----------------
* Synthetic truth is used only for reporting estimation error, never for the
  deployable tuning choice.
* Tuning is selected by a target-healthy tuning split.
* Reported Gaussian risk is evaluated on a separate, independently generated
  target-healthy evaluation set that is never used for tuning.
* Related and unrelated source conditions use the exact same target sample,
  target split, target evaluation set, lambda grid, and cross-fitting folds
  within each (p, N, replication). This makes negative-transfer comparisons
  paired and removes target-sampling noise as a confound.
* Reference-style Trans-CLIME and the COLDSTART cross-fitted extension are
  labeled separately.
* Source truth distance/support overlap are recorded so the negative control is
  auditable rather than inferred from a name.
* Raw and SPD-projected estimates are both audited.
* Results are checkpointed one method at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precision_transfer_estimators import (
    clime,
    crossfit_trans_clime,
    gaussian_precision_risk,
    max_abs_error,
    reference_trans_clime,
    relative_frobenius_error,
    support_metrics,
)
from src.reproducibility import reproducibility_metadata


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "p02a_reference_validation"
PROTOCOL_VERSION = "p02a-reference-validation-v2-paired"
DEFAULT_DIMENSIONS = (10, 20)
DEFAULT_TARGET_NS = (25, 50)
DEFAULT_REPLICATIONS = 10
DEFAULT_SOURCE_N = 200
DEFAULT_EVAL_N = 1000
DEFAULT_LAMBDA_MULTIPLIERS = (0.5, 1.0, 1.5)


def _precision_chain(p: int, edge: float = 0.22) -> np.ndarray:
    omega = np.eye(p, dtype=np.float64)
    for i in range(p - 1):
        omega[i, i + 1] = omega[i + 1, i] = -float(edge)
    return omega


def _precision_unrelated(p: int, edge: float = 0.22) -> np.ndarray:
    """Density-comparable graph with no chain edges in common for p >= 3."""
    omega = np.eye(p, dtype=np.float64)
    for i in range(0, p - 2, 2):
        omega[i, i + 2] = omega[i + 2, i] = -float(edge)
    return omega


def _sample(rng: np.random.Generator, omega: np.ndarray, n: int) -> np.ndarray:
    covariance = np.linalg.inv(omega)
    return rng.multivariate_normal(np.zeros(omega.shape[0]), covariance, size=int(n))


def _split_target(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    tune = max(3, int(round(0.20 * n)))
    aggregate = max(3, int(round(0.20 * n)))
    fit_end = n - tune - aggregate
    if fit_end < 4:
        raise ValueError(f"Target N={n} is too small for P0.2A reference split.")
    return x[:fit_end], x[fit_end : fit_end + aggregate], x[fit_end + aggregate :]


def _lambda_grid(p: int, n_fit: int, multipliers: tuple[float, ...]) -> tuple[float, ...]:
    base = 2.0 * np.sqrt(np.log(max(p, 2)) / max(n_fit, 2))
    return tuple(float(m * base) for m in multipliers)


def _target_seed(p: int, target_n: int, replication: int) -> int:
    return 100000 * int(p) + 1000 * int(target_n) + 10 * int(replication)


def _evaluation_seed(p: int, target_n: int, replication: int) -> int:
    return 700000000 + _target_seed(p, target_n, replication)


def _source_seed(p: int, target_n: int, replication: int, source_kind: str) -> int:
    offset = {"related": 1, "unrelated": 2}[source_kind]
    return 900000000 + _target_seed(p, target_n, replication) + offset


def _matrix_hash(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _truth_relationship(source_truth: np.ndarray, target_truth: np.ndarray) -> dict[str, float]:
    support = support_metrics(source_truth, target_truth, threshold=1e-12)
    return {
        "source_target_truth_relative_frobenius": relative_frobenius_error(source_truth, target_truth),
        "source_target_truth_support_jaccard": float(support["support_jaccard"]),
        "source_target_truth_support_f1": float(support["support_f1"]),
    }


def _estimate_metrics(
    *,
    method: str,
    estimate,
    truth: np.ndarray,
    independent_eval: np.ndarray,
    p: int,
    target_n: int,
    source_kind: str,
    replication: int,
    tuning_lambda: float,
    selected_by: str,
    target_seed: int,
    target_hash: str,
    source_truth: np.ndarray,
) -> dict[str, object]:
    raw_support = support_metrics(estimate.symmetric, truth, threshold=1e-6)
    spd_support = support_metrics(estimate.spd, truth, threshold=1e-6)
    relationship = _truth_relationship(source_truth, truth)
    return {
        "p": int(p),
        "target_n": int(target_n),
        "source_kind": source_kind,
        "replication": int(replication),
        "target_seed": int(target_seed),
        "target_sample_sha256": target_hash,
        "method": method,
        "selected_lambda": float(tuning_lambda),
        "selected_by": selected_by,
        "relative_frobenius_raw": relative_frobenius_error(estimate.symmetric, truth),
        "relative_frobenius_spd": relative_frobenius_error(estimate.spd, truth),
        "max_abs_error_raw": max_abs_error(estimate.symmetric, truth),
        "heldout_gaussian_risk_spd": gaussian_precision_risk(independent_eval, estimate.spd),
        "independent_evaluation_samples": int(len(independent_eval)),
        "lp_success_fraction": float(estimate.lp_diagnostics.success_fraction),
        "lp_max_constraint_violation": float(estimate.lp_diagnostics.max_constraint_violation),
        "spd_projection_relative_change": float(estimate.spd_projection.relative_frobenius_change),
        "min_eigenvalue_raw_symmetric": float(estimate.spd_projection.min_eigenvalue_before),
        "transfer_selected_fraction": float(
            estimate.metadata.get(
                "transfer_selected_fraction",
                estimate.metadata.get("mean_transfer_selected_fraction", np.nan),
            )
        ),
        **relationship,
        **{f"raw_{k}": v for k, v in raw_support.items()},
        **{f"spd_{k}": v for k, v in spd_support.items()},
    }


def _fit_candidates(
    target_fit: np.ndarray,
    target_aggregate: np.ndarray,
    source: np.ndarray,
    lambdas: tuple[float, ...],
    *,
    crossfit_seed: int,
) -> dict[str, list[tuple[float, object]]]:
    methods: dict[str, list[tuple[float, object]]] = {
        "TargetCLIME": [],
        "ReferenceTransCLIME": [],
        "CrossfitTransCLIME": [],
    }
    for lam in lambdas:
        methods["TargetCLIME"].append((lam, clime(target_fit, lam=lam)))
        methods["ReferenceTransCLIME"].append(
            (
                lam,
                reference_trans_clime(
                    target_fit,
                    target_aggregate,
                    source,
                    target_lambda=lam,
                    transfer_lambda_const=1.0,
                ),
            )
        )
        combined = np.vstack((target_fit, target_aggregate))
        folds = max(2, min(5, len(combined) // 2))
        methods["CrossfitTransCLIME"].append(
            (
                lam,
                crossfit_trans_clime(
                    combined,
                    source,
                    target_lambda=lam,
                    n_folds=folds,
                    seed=crossfit_seed,
                ),
            )
        )
    return methods


def _select_by_healthy_risk(
    candidates: list[tuple[float, object]],
    tune: np.ndarray,
) -> tuple[float, object, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best: tuple[float, float, object] | None = None
    for lam, estimate in candidates:
        risk = gaussian_precision_risk(tune, estimate.spd)
        rows.append({"lambda": float(lam), "healthy_risk": float(risk)})
        key = (float(risk), float(lam), estimate)
        if best is None or key[0] < best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    assert best is not None
    return best[1], best[2], rows


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["p", "target_n", "source_kind", "method"]
    for key, group in results.groupby(group_cols, sort=True):
        p, target_n, source_kind, method = key
        rows.append(
            {
                "p": int(p),
                "target_n": int(target_n),
                "source_kind": str(source_kind),
                "method": str(method),
                "replications": int(group["replication"].nunique()),
                "relative_frobenius_spd_median": float(group["relative_frobenius_spd"].median()),
                "relative_frobenius_spd_q25": float(group["relative_frobenius_spd"].quantile(0.25)),
                "relative_frobenius_spd_q75": float(group["relative_frobenius_spd"].quantile(0.75)),
                "heldout_gaussian_risk_spd_median": float(group["heldout_gaussian_risk_spd"].median()),
                "support_f1_spd_median": float(group["spd_support_f1"].median()),
                "lp_success_fraction_min": float(group["lp_success_fraction"].min()),
                "spd_projection_relative_change_median": float(group["spd_projection_relative_change"].median()),
                "source_target_truth_relative_frobenius": float(group["source_target_truth_relative_frobenius"].iloc[0]),
                "source_target_truth_support_jaccard": float(group["source_target_truth_support_jaccard"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _assert_paired_target_rows(results: pd.DataFrame) -> None:
    """Fail if source conditions were evaluated on different target samples."""
    keys = ["p", "target_n", "replication"]
    for _, group in results.groupby(keys, sort=False):
        if group["source_kind"].nunique() < 2:
            continue
        if group["target_seed"].nunique() != 1 or group["target_sample_sha256"].nunique() != 1:
            raise RuntimeError("P0.2A pairing violation: source kinds saw different target samples.")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "p02a_results.csv"
    tuning_path = output_dir / "p02a_tuning_audit.csv"

    completed: set[tuple[int, int, str, int, str]] = set()
    if args.resume and results_path.exists() and results_path.stat().st_size > 0:
        existing = pd.read_csv(results_path)
        required_v2 = {"target_seed", "target_sample_sha256", "independent_evaluation_samples"}
        if not required_v2.issubset(existing.columns):
            raise RuntimeError(
                "Existing P0.2A output is from an older unpaired protocol. "
                "Use a fresh output directory or remove the old P0.2A outputs."
            )
        if not existing.empty:
            completed = {
                (int(r.p), int(r.target_n), str(r.source_kind), int(r.replication), str(r.method))
                for r in existing.itertuples(index=False)
            }

    for p in args.dimensions:
        target_truth = _precision_chain(int(p))
        source_truths = {
            "related": target_truth.copy(),
            "unrelated": _precision_unrelated(int(p)),
        }
        for target_n in args.target_ns:
            for replication in range(args.replications):
                t_seed = _target_seed(int(p), int(target_n), int(replication))
                e_seed = _evaluation_seed(int(p), int(target_n), int(replication))
                target_rng = np.random.default_rng(t_seed)
                eval_rng = np.random.default_rng(e_seed)
                target = _sample(target_rng, target_truth, int(target_n))
                independent_eval = _sample(eval_rng, target_truth, int(args.eval_n))
                target_fit, target_aggregate, target_tune = _split_target(target)
                lambdas = _lambda_grid(int(p), len(target_fit), tuple(args.lambda_multipliers))
                target_hash = _matrix_hash(target)

                for source_kind, source_truth in source_truths.items():
                    expected_methods = ("TargetCLIME", "ReferenceTransCLIME", "CrossfitTransCLIME")
                    if all((p, target_n, source_kind, replication, m) in completed for m in expected_methods):
                        continue

                    s_seed = _source_seed(int(p), int(target_n), int(replication), source_kind)
                    source_rng = np.random.default_rng(s_seed)
                    source = _sample(source_rng, source_truth, int(args.source_n))
                    candidates = _fit_candidates(
                        target_fit,
                        target_aggregate,
                        source,
                        lambdas,
                        crossfit_seed=t_seed,
                    )
                    print(
                        f"P0.2A p={p} N={target_n} rep={replication + 1}/{args.replications} source={source_kind}",
                        flush=True,
                    )
                    for method, method_candidates in candidates.items():
                        key = (p, target_n, source_kind, replication, method)
                        if key in completed:
                            continue
                        selected_lambda, estimate, tuning_rows = _select_by_healthy_risk(method_candidates, target_tune)
                        row = _estimate_metrics(
                            method=method,
                            estimate=estimate,
                            truth=target_truth,
                            independent_eval=independent_eval,
                            p=p,
                            target_n=target_n,
                            source_kind=source_kind,
                            replication=replication,
                            tuning_lambda=selected_lambda,
                            selected_by="target_healthy_tuning_split_gaussian_risk",
                            target_seed=t_seed,
                            target_hash=target_hash,
                            source_truth=source_truth,
                        )
                        pd.DataFrame([row]).to_csv(
                            results_path,
                            mode="a",
                            header=not results_path.exists() or results_path.stat().st_size == 0,
                            index=False,
                        )
                        tune_frame = pd.DataFrame(tuning_rows)
                        tune_frame.insert(0, "method", method)
                        tune_frame.insert(0, "replication", replication)
                        tune_frame.insert(0, "source_kind", source_kind)
                        tune_frame.insert(0, "target_n", target_n)
                        tune_frame.insert(0, "p", p)
                        tune_frame.insert(5, "target_seed", t_seed)
                        tune_frame.to_csv(
                            tuning_path,
                            mode="a",
                            header=not tuning_path.exists() or tuning_path.stat().st_size == 0,
                            index=False,
                        )
                        completed.add(key)

    results = pd.read_csv(results_path)
    _assert_paired_target_rows(results)
    summary = _summary(results)
    summary.to_csv(output_dir / "p02a_summary.csv", index=False)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": [int(v) for v in args.dimensions],
        "target_ns": [int(v) for v in args.target_ns],
        "source_n": int(args.source_n),
        "eval_n": int(args.eval_n),
        "replications": int(args.replications),
        "lambda_multipliers": [float(v) for v in args.lambda_multipliers],
        "truth_used_for_tuning": False,
        "paired_target_across_source_conditions": True,
        "independent_target_evaluation_used_for_reported_risk": True,
        "methods": ["TargetCLIME", "ReferenceTransCLIME", "CrossfitTransCLIME"],
        "crossfit_label": "COLDSTART extension; not published Trans-CLIME",
        "reproducibility": reproducibility_metadata(repo_root=PROJECT_ROOT),
    }
    (output_dir / "p02a_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"P0.2A outputs written to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dimensions", type=int, nargs="+", default=list(DEFAULT_DIMENSIONS))
    parser.add_argument("--target-ns", type=int, nargs="+", default=list(DEFAULT_TARGET_NS))
    parser.add_argument("--source-n", type=int, default=DEFAULT_SOURCE_N)
    parser.add_argument("--eval-n", type=int, default=DEFAULT_EVAL_N)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    parser.add_argument("--lambda-multipliers", type=float, nargs="+", default=list(DEFAULT_LAMBDA_MULTIPLIERS))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    args.resume = not args.no_resume
    if any(v < 2 for v in args.dimensions):
        parser.error("dimensions must be >=2")
    if any(v < 10 for v in args.target_ns):
        parser.error("target-ns must be >=10")
    if args.source_n < 10:
        parser.error("source-n must be >=10")
    if args.eval_n < 20:
        parser.error("eval-n must be >=20")
    if args.replications <= 0:
        parser.error("replications must be positive")
    if any(v <= 0 for v in args.lambda_multipliers):
        parser.error("lambda-multipliers must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
