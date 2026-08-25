from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import beta


@dataclass(frozen=True)
class CertificationBounds:
    """Exact one-sided binomial certification for a frozen detector."""

    recall: float
    fpr: float
    recall_lower: float
    fpr_upper: float
    certified: bool
    joint_confidence: float
    delta_recall: float
    delta_fpr: float


def exact_one_sided_recall_lower(
    tp: int,
    fn: int,
    delta: float,
) -> float:
    """Exact one-sided Clopper-Pearson lower bound for recall."""
    tp = int(tp)
    fn = int(fn)

    if tp < 0 or fn < 0:
        raise ValueError("TP and FN must be non-negative.")

    n = tp + fn
    if n <= 0:
        raise ValueError(
            "At least one anomalous evaluation episode is required."
        )

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1.")

    if tp == 0:
        return 0.0

    return float(
        beta.ppf(
            delta,
            tp,
            fn + 1,
        )
    )


def exact_one_sided_fpr_upper(
    fp: int,
    tn: int,
    delta: float,
) -> float:
    """Exact one-sided Clopper-Pearson upper bound for healthy FPR."""
    fp = int(fp)
    tn = int(tn)

    if fp < 0 or tn < 0:
        raise ValueError("FP and TN must be non-negative.")

    n = fp + tn
    if n <= 0:
        raise ValueError(
            "At least one healthy evaluation episode is required."
        )

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1.")

    if fp == n:
        return 1.0

    return float(
        beta.ppf(
            1.0 - delta,
            fp + 1,
            tn,
        )
    )


def certify_operating_point(
    *,
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    recall_target: float = 0.90,
    fpr_budget: float = 0.01,
    joint_confidence: float = 0.95,
) -> CertificationBounds:
    """Bonferroni simultaneous certification of recall and healthy FPR."""
    if not 0.0 < joint_confidence < 1.0:
        raise ValueError(
            "joint_confidence must lie strictly between 0 and 1."
        )

    if not 0.0 <= recall_target <= 1.0:
        raise ValueError("recall_target must lie in [0, 1].")

    if not 0.0 <= fpr_budget <= 1.0:
        raise ValueError("fpr_budget must lie in [0, 1].")

    tp = int(tp)
    fn = int(fn)
    fp = int(fp)
    tn = int(tn)

    n_anom = tp + fn
    n_healthy = fp + tn

    if n_anom <= 0:
        raise ValueError("No anomalous evaluation episodes.")
    if n_healthy <= 0:
        raise ValueError("No healthy evaluation episodes.")

    familywise_delta = 1.0 - joint_confidence
    delta_recall = familywise_delta / 2.0
    delta_fpr = familywise_delta / 2.0

    recall = tp / n_anom
    fpr = fp / n_healthy

    recall_lower = exact_one_sided_recall_lower(
        tp=tp,
        fn=fn,
        delta=delta_recall,
    )
    fpr_upper = exact_one_sided_fpr_upper(
        fp=fp,
        tn=tn,
        delta=delta_fpr,
    )

    certified = bool(
        recall_lower >= recall_target
        and fpr_upper <= fpr_budget
    )

    return CertificationBounds(
        recall=float(recall),
        fpr=float(fpr),
        recall_lower=float(recall_lower),
        fpr_upper=float(fpr_upper),
        certified=certified,
        joint_confidence=float(joint_confidence),
        delta_recall=float(delta_recall),
        delta_fpr=float(delta_fpr),
    )
