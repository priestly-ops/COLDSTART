import importlib.util
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "run_m2_numerical_stability_audit.py"
)

if not SCRIPT_PATH.exists():
    raise FileNotFoundError(
        f"Audit script not found: {SCRIPT_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "run_m2_numerical_stability_audit",
    SCRIPT_PATH,
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Could not load audit script from {SCRIPT_PATH}"
    )


audit_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(audit_module)


_oracle_counts = audit_module._oracle_counts

_probability_of_superiority = (
    audit_module._probability_of_superiority
)


def test_oracle_clear_separation_zero_fp():
    healthy = np.array(
        [0.0, 1.0, 2.0]
    )

    anomaly = np.array(
        [10.0, 11.0, 12.0]
    )

    result = _oracle_counts(
        healthy,
        anomaly,
        allowed_fp=0,
        recall_target=0.90,
    )

    assert result["empirically_feasible"] is True

    assert (
        result["max_recall_at_budget"]
        == 1.0
    )

    assert (
        result["min_fp_for_target_recall"]
        == 0
    )


def test_one_extreme_healthy_causes_zero_to_one_fp_discontinuity():
    healthy = np.array(
        [
            0.0,
            1.0,
            100.0,
        ]
    )

    anomaly = np.array(
        [
            10.0,
            11.0,
            12.0,
            13.0,
        ]
    )

    zero_fp = _oracle_counts(
        healthy,
        anomaly,
        allowed_fp=0,
        recall_target=0.90,
    )

    one_fp = _oracle_counts(
        healthy,
        anomaly,
        allowed_fp=1,
        recall_target=0.90,
    )

    assert (
        zero_fp["empirically_feasible"]
        is False
    )

    assert (
        zero_fp["max_recall_at_budget"]
        == 0.0
    )

    assert (
        one_fp["empirically_feasible"]
        is True
    )

    assert (
        one_fp["max_recall_at_budget"]
        == 1.0
    )


def test_probability_of_superiority_with_ties():
    healthy = np.array(
        [
            0.0,
            1.0,
        ]
    )

    anomaly = np.array(
        [
            1.0,
            2.0,
        ]
    )

    auc = _probability_of_superiority(
        healthy,
        anomaly,
    )

    assert math.isclose(
        auc,
        0.875,
        abs_tol=1e-12,
    )