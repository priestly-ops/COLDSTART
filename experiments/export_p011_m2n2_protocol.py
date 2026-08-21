"""Export frozen P0.7 episode partitions for the official M2N2 baseline.

Why export instead of reimplementing M2N2 here?
----------------------------------------------
M2N2 (AAAI 2024) is a trainable time-series anomaly-detection / test-time-
adaptation pipeline with trend estimation and self-supervised adaptation. It is
not scientifically equivalent to a cycle-level sklearn wrapper. This exporter
therefore preserves the exact frozen P0.7 episode identities and raw measured
signals so the official implementation can be run without changing our split.

For each (budget, seed), one directory is created containing:
  fit.npz         target healthy fit cycles
  calibration.npz target healthy calibration cycles
  normal_eval.npz fixed PRE_B healthy evaluation cycles
  anomaly_eval.npz all frozen anomaly evaluation cycles
  manifest.json   episode IDs, labels, settings, and split hashes

Arrays are stored as object arrays because cycle lengths can vary. No anomaly
labels are used for fitting or calibration; labels are included only in the
held-out anomaly file for final evaluation bookkeeping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.voraus_loader import load_cycle_metadata, load_cycles
from experiments.run_p05_anomaly_commissioning import DEFAULT_DATASET
from experiments.run_p07_fixed_budget_commissioning import (
    DEFAULT_ALLOCATIONS,
    DEFAULT_EVALUATION_SEED,
    DEFAULT_EVAL_SIZE,
    DEFAULT_SEEDS,
    _frozen_preb_partitions,
)

PROTOCOL_VERSION = "p011-m2n2-export-frozen-p07-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "p011_m2n2_export"


def _sha(values: list[int]) -> str:
    arr = np.asarray(sorted(int(v) for v in values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _save_partition(path: Path, cycles, cycle_by_id) -> None:
    ids = [int(c.episode_id) for c in cycles]
    values = np.empty(len(ids), dtype=object)
    for i, eid in enumerate(ids):
        values[i] = np.asarray(cycle_by_id[eid].values, dtype=np.float32)
    np.savez_compressed(
        path,
        episode_ids=np.asarray(ids, dtype=np.int64),
        values=values,
        anomaly=np.asarray([bool(c.anomaly) for c in cycles], dtype=bool),
        category=np.asarray([int(c.category) for c in cycles], dtype=np.int64),
        setting=np.asarray([int(c.setting) for c in cycles], dtype=np.int64),
    )


def run(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_cycle_metadata(dataset)

    # Load all target PRE_B healthy cycles and all anomalies once.
    needed_ids = [
        int(c.episode_id)
        for c in metadata
        if ((not c.anomaly and int(c.setting) == 73) or c.anomaly)
    ]
    loaded = load_cycles(dataset, signal_set=args.signal_set, episode_ids=needed_ids)
    cycle_by_id = {int(c.episode_id): c for c in loaded}

    alloc = {int(b): (int(f), int(c)) for b, f, c in DEFAULT_ALLOCATIONS}
    for budget in args.budgets:
        budget = int(budget)
        if budget not in alloc:
            raise ValueError(f"budget {budget} not in frozen P0.7 grid")
        fit_n, cal_n = alloc[budget]
        for seed in args.seeds:
            _, pool, normal_eval, anomaly_eval = _frozen_preb_partitions(
                metadata,
                seed=int(seed),
                eval_seed=int(args.evaluation_seed),
                eval_size=int(args.eval_size),
            )
            fit = pool[:fit_n]
            calibration = pool[fit_n:fit_n + cal_n]

            groups = [
                {int(c.episode_id) for c in fit},
                {int(c.episode_id) for c in calibration},
                {int(c.episode_id) for c in normal_eval},
                {int(c.episode_id) for c in anomaly_eval},
            ]
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if groups[i] & groups[j]:
                        raise RuntimeError("M2N2 export leakage detected")

            out = output / f"B{budget}" / f"seed{int(seed):02d}"
            out.mkdir(parents=True, exist_ok=True)
            _save_partition(out / "fit.npz", fit, cycle_by_id)
            _save_partition(out / "calibration.npz", calibration, cycle_by_id)
            _save_partition(out / "normal_eval.npz", normal_eval, cycle_by_id)
            _save_partition(out / "anomaly_eval.npz", anomaly_eval, cycle_by_id)

            manifest = {
                "protocol_version": PROTOCOL_VERSION,
                "parent_protocol": "p07-fixed-budget-preb-v1",
                "budget": budget,
                "fit_n": fit_n,
                "calibration_n": cal_n,
                "seed": int(seed),
                "signal_set": args.signal_set,
                "fit_ids_sha256": _sha([int(c.episode_id) for c in fit]),
                "calibration_ids_sha256": _sha([int(c.episode_id) for c in calibration]),
                "normal_eval_ids_sha256": _sha([int(c.episode_id) for c in normal_eval]),
                "anomaly_eval_ids_sha256": _sha([int(c.episode_id) for c in anomaly_eval]),
                "m2n2_official_repo": "https://github.com/carrtesy/M2N2",
                "aggregation_rule_status": "must_be_predeclared_before_scoring",
            }
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"exported B={budget} seed={seed}: {out}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-set", type=str, default="measured")
    parser.add_argument("--budgets", nargs="+", type=int, default=[b for b, _, _ in DEFAULT_ALLOCATIONS])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
    parser.add_argument("--eval-size", type=int, default=DEFAULT_EVAL_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
