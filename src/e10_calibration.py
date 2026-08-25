from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class CalibrationDecision:
    method: str
    threshold: float
    predictions: np.ndarray
    expected_alarm_probability: np.ndarray
    fit_ok: bool
    diagnostic: dict[str, float | int | str | bool]


def _as_scores(values: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return x


def deterministic_split_conformal(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float,
) -> CalibrationDecision:
    """Strict deterministic split-conformal rule used by the frozen paper protocol."""
    from src.calibration_tail import conformal_threshold_info

    cal = _as_scores(calibration_scores, "calibration_scores")
    test = _as_scores(test_scores, "test_scores")
    info = conformal_threshold_info(cal, alpha=alpha)
    threshold = float(info.strict_threshold)
    pred = test > threshold
    expected = pred.astype(np.float64)
    if not info.finite_sample_feasible:
        regime = "infinite"
    elif info.threshold_is_maximum:
        regime = "maximum"
    else:
        regime = "submaximum"
    return CalibrationDecision(
        method="deterministic_conformal",
        threshold=threshold,
        predictions=pred,
        expected_alarm_probability=expected,
        fit_ok=True,
        diagnostic={
            "conformal_rank": int(info.raw_rank),
            "conformal_regime": regime,
            "minimum_attainable_alpha": float(info.minimum_attainable_alpha),
        },
    )


def smoothed_conformal_alarm_probabilities(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Expected alarm probability for randomized/smoothed split conformal.

    For a test score s and m calibration scores, the randomized p-value is

        p = (#{cal > s} + U * (#{cal = s} + 1)) / (m + 1),  U~Unif(0,1).

    The returned value is P(p <= alpha | scores), i.e. expectation over U.
    This removes the deterministic grid barrier while retaining the standard
    exchangeability-based randomized conformal construction.
    """
    cal = _as_scores(calibration_scores, "calibration_scores")
    test = _as_scores(test_scores, "test_scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")

    target = alpha * (cal.size + 1)
    out = np.empty(test.size, dtype=np.float64)
    for i, score in enumerate(test):
        gt = int(np.sum(cal > score))
        eq_plus_test = int(np.sum(cal == score)) + 1
        if target <= gt:
            prob = 0.0
        elif target >= gt + eq_plus_test:
            prob = 1.0
        else:
            prob = (target - gt) / eq_plus_test
        out[i] = float(np.clip(prob, 0.0, 1.0))
    return out


def randomized_smoothed_conformal(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float,
    rng: np.random.Generator,
) -> CalibrationDecision:
    """One reproducible deployment realization of smoothed conformal."""
    probs = smoothed_conformal_alarm_probabilities(
        calibration_scores, test_scores, alpha=alpha
    )
    uniforms = rng.random(probs.size)
    pred = uniforms <= probs
    return CalibrationDecision(
        method="randomized_conformal",
        threshold=float("nan"),
        predictions=pred,
        expected_alarm_probability=probs,
        fit_ok=True,
        diagnostic={
            "randomized": True,
            "mean_expected_alarm_probability": float(probs.mean()),
        },
    )


def _pot_threshold(
    calibration_scores: np.ndarray,
    *,
    alpha: float,
    tail_fraction: float = 0.20,
    minimum_exceedances: int = 10,
) -> tuple[float, dict[str, float | int | str | bool]]:
    """Fit a GPD to a predeclared upper calibration tail and extrapolate P(S>t)=alpha.

    This is a parametric robustness analysis, not a finite-sample conformal
    guarantee. The tail size is max(minimum_exceedances, ceil(tail_fraction*m)).
    """
    cal = _as_scores(calibration_scores, "calibration_scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0,1)")

    m = cal.size
    tail_count = max(int(minimum_exceedances), int(np.ceil(tail_fraction * m)))
    if tail_count >= m:
        raise ValueError("Not enough calibration scores for POT tail fit")

    ordered = np.sort(cal)
    u = float(ordered[m - tail_count - 1])
    exceedances = cal[cal > u] - u
    if exceedances.size < minimum_exceedances:
        raise ValueError(
            f"POT has only {exceedances.size} strict exceedances; "
            f"requires {minimum_exceedances}"
        )
    if np.allclose(exceedances, 0.0):
        raise ValueError("POT exceedances are degenerate")

    shape, loc, scale = genpareto.fit(exceedances, floc=0.0)
    if loc != 0.0 or not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Invalid GPD fit")

    p_u = float(exceedances.size / m)
    conditional_tail = float(alpha / p_u)
    if not 0.0 < conditional_tail < 1.0:
        raise ValueError("Requested alpha is outside fitted POT tail")

    excess_quantile = float(
        genpareto.ppf(1.0 - conditional_tail, c=shape, loc=0.0, scale=scale)
    )
    threshold = float(u + excess_quantile)
    if not np.isfinite(threshold):
        raise ValueError("POT threshold is non-finite")

    return threshold, {
        "pot_tail_fraction": float(tail_fraction),
        "pot_tail_count_requested": int(tail_count),
        "pot_exceedance_count": int(exceedances.size),
        "pot_u": u,
        "pot_shape": float(shape),
        "pot_scale": float(scale),
        "pot_empirical_tail_mass": p_u,
        "pot_conditional_tail_target": conditional_tail,
    }


def evt_pot(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    *,
    alpha: float,
    tail_fraction: float = 0.20,
    minimum_exceedances: int = 10,
) -> CalibrationDecision:
    test = _as_scores(test_scores, "test_scores")
    try:
        threshold, diagnostic = _pot_threshold(
            calibration_scores,
            alpha=alpha,
            tail_fraction=tail_fraction,
            minimum_exceedances=minimum_exceedances,
        )
        pred = test > threshold
        return CalibrationDecision(
            method="evt_pot",
            threshold=threshold,
            predictions=pred,
            expected_alarm_probability=pred.astype(np.float64),
            fit_ok=True,
            diagnostic=diagnostic,
        )
    except Exception as exc:
        return CalibrationDecision(
            method="evt_pot",
            threshold=float("nan"),
            predictions=np.zeros(test.size, dtype=bool),
            expected_alarm_probability=np.zeros(test.size, dtype=np.float64),
            fit_ok=False,
            diagnostic={"fit_error": str(exc)},
        )
