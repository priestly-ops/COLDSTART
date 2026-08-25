from __future__ import annotations

"""E9C: high-data AURSAD MVT-Flow stress test.

This is a thin, deliberate wrapper around the audited E9B runner. It changes
only the protocol directory, allowed commissioning sizes/seeds, output paths,
and protocol version. The detector implementation, raw-signal policy,
calibration rule, operating point, certification logic, and MVT-Flow training
configuration remain exactly those used by E9B.

Frozen diagnostic design
------------------------
Commissioning N: 500, 650, 750, 850
Seeds: 0, 1, 2
Calibration: 199 independent healthy executions
Healthy certification/evaluation: 368 independent healthy executions
Anomaly evaluation: 625 executions
Recall target: 0.90
FPR budget: 0.01
Joint confidence: 0.95

The protocol must already exist at:
    reports/aursad/coldstart_highdata

Example:
    python -m experiments.run_e9c_aursad_mvtflow_highdata \
      --n-values 500,650,750,850 --seeds 0,1,2 --device cuda
"""

from pathlib import Path

import experiments.run_e9b_aursad_mvtflow_control as e9b


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Freeze the high-data diagnostic protocol without duplicating the audited E9B
# implementation. e9b.main() reads these globals at runtime, so replacing them
# here changes only experiment bookkeeping/membership, not model semantics.
e9b.PROTOCOL_DIR = PROJECT_ROOT / "reports" / "aursad" / "coldstart_highdata"
e9b.OUTPUT_DIR = PROJECT_ROOT / "outputs" / "aursad" / "e9c_mvtflow_highdata"
e9b.SEED_RESULTS_PATH = e9b.OUTPUT_DIR / "e9c_seed_results.csv"
e9b.SUMMARY_PATH = e9b.OUTPUT_DIR / "e9c_summary.csv"
e9b.PER_CLASS_PATH = e9b.OUTPUT_DIR / "e9c_per_class_recall.csv"
e9b.NSTAR_PATH = e9b.OUTPUT_DIR / "e9c_nstar.csv"
e9b.MANIFEST_PATH = e9b.OUTPUT_DIR / "e9c_manifest.json"

e9b.PROTOCOL_VERSION = "coldstart-e9c-aursad-mvtflow-highdata-v1"
e9b.DEFAULT_GRID = (500, 650, 750, 850)
e9b.DEFAULT_SEEDS = (0, 1, 2)

# These are intentionally unchanged from E9B and repeated as assertions so a
# future E9B edit cannot silently alter the frozen E9C diagnostic.
assert e9b.CALIBRATION_SIZE == 199
assert e9b.HEALTHY_EVAL_SIZE == 368
assert e9b.ANOMALY_EVAL_SIZE == 625
assert e9b.FALSE_ALERT_BUDGET == 0.01
assert e9b.RECALL_TARGET == 0.90
assert e9b.JOINT_CONFIDENCE == 0.95


if __name__ == "__main__":
    e9b.main()
