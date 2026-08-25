from __future__ import annotations

import numpy as np

from src.calibration_tail import conformal_threshold_info
from src.certification import exact_one_sided_fpr_upper


def test_e9_calibration_regimes_at_one_percent() -> None:
    rng = np.random.default_rng(42)
    scores = rng.normal(size=500)

    expected = {
        50: (51, False, False),
        99: (99, True, True),
        100: (100, True, True),
        198: (198, True, True),
        199: (198, True, False),
        250: (249, True, False),
        500: (496, True, False),
    }

    for mcal, (rank, feasible, maximum) in expected.items():
        info = conformal_threshold_info(scores[:mcal], alpha=0.01)
        assert info.raw_rank == rank
        assert info.finite_sample_feasible is feasible
        assert info.threshold_is_maximum is maximum


def test_e9_healthy_certification_boundary_is_368() -> None:
    delta = 0.025
    assert exact_one_sided_fpr_upper(fp=0, tn=367, delta=delta) > 0.01
    assert exact_one_sided_fpr_upper(fp=0, tn=368, delta=delta) <= 0.01


def test_e9_frozen_allocation_fits_aursad_healthy_pool() -> None:
    healthy_total = 1420
    calibration = 500
    healthy_eval = 368
    reservoir = healthy_total - calibration - healthy_eval
    assert reservoir == 552
    assert reservoir >= 500
