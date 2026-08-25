from __future__ import annotations

from dataclasses import dataclass

from src.certification import (
    exact_one_sided_fpr_upper,
    exact_one_sided_recall_lower,
)


@dataclass(frozen=True)
class MinimumEvidenceResult:
    event_count: int
    minimum_sample_size: int
    bound_at_minimum: float
    target: float
    delta: float


def minimum_healthy_sample_size(
    *,
    false_positives: int,
    fpr_budget: float = 0.01,
    delta_fpr: float = 0.025,
    max_n: int = 100_000,
) -> MinimumEvidenceResult:
    """Smallest healthy evaluation size that certifies an FPR budget.

    The detector and threshold are assumed frozen before evaluation.  The
    returned sample size is therefore an evidence requirement, not a model
    training or calibration requirement.
    """
    fp = int(false_positives)
    if fp < 0:
        raise ValueError("false_positives must be non-negative")
    if not 0.0 < fpr_budget < 1.0:
        raise ValueError("fpr_budget must lie strictly between 0 and 1")
    if not 0.0 < delta_fpr < 1.0:
        raise ValueError("delta_fpr must lie strictly between 0 and 1")
    if max_n <= fp:
        raise ValueError("max_n must exceed false_positives")

    for n in range(max(1, fp + 1), int(max_n) + 1):
        bound = exact_one_sided_fpr_upper(
            fp=fp,
            tn=n - fp,
            delta=delta_fpr,
        )
        if bound <= fpr_budget:
            return MinimumEvidenceResult(
                event_count=fp,
                minimum_sample_size=n,
                bound_at_minimum=float(bound),
                target=float(fpr_budget),
                delta=float(delta_fpr),
            )

    raise RuntimeError(
        f"No healthy sample size <= {max_n} certifies FP={fp} "
        f"at FPR budget {fpr_budget}."
    )


def minimum_anomaly_sample_size(
    *,
    false_negatives: int,
    recall_target: float = 0.90,
    delta_recall: float = 0.025,
    max_n: int = 100_000,
) -> MinimumEvidenceResult:
    """Smallest anomaly evaluation size that certifies a recall target."""
    fn = int(false_negatives)
    if fn < 0:
        raise ValueError("false_negatives must be non-negative")
    if not 0.0 < recall_target < 1.0:
        raise ValueError("recall_target must lie strictly between 0 and 1")
    if not 0.0 < delta_recall < 1.0:
        raise ValueError("delta_recall must lie strictly between 0 and 1")
    if max_n <= fn:
        raise ValueError("max_n must exceed false_negatives")

    for n in range(max(1, fn + 1), int(max_n) + 1):
        tp = n - fn
        bound = exact_one_sided_recall_lower(
            tp=tp,
            fn=fn,
            delta=delta_recall,
        )
        if bound >= recall_target:
            return MinimumEvidenceResult(
                event_count=fn,
                minimum_sample_size=n,
                bound_at_minimum=float(bound),
                target=float(recall_target),
                delta=float(delta_recall),
            )

    raise RuntimeError(
        f"No anomaly sample size <= {max_n} certifies FN={fn} "
        f"at recall target {recall_target}."
    )


def healthy_bound_curve(
    *,
    false_positives: int,
    sample_sizes: list[int],
    delta_fpr: float = 0.025,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for n in sample_sizes:
        n = int(n)
        if n <= false_positives:
            continue
        upper = exact_one_sided_fpr_upper(
            fp=false_positives,
            tn=n - false_positives,
            delta=delta_fpr,
        )
        rows.append(
            {
                "false_positives": int(false_positives),
                "healthy_sample_size": n,
                "empirical_fpr": float(false_positives / n),
                "fpr_upper": float(upper),
            }
        )
    return rows


def recall_bound_curve(
    *,
    false_negatives: int,
    sample_sizes: list[int],
    delta_recall: float = 0.025,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for n in sample_sizes:
        n = int(n)
        if n <= false_negatives:
            continue
        tp = n - false_negatives
        lower = exact_one_sided_recall_lower(
            tp=tp,
            fn=false_negatives,
            delta=delta_recall,
        )
        rows.append(
            {
                "false_negatives": int(false_negatives),
                "anomaly_sample_size": n,
                "empirical_recall": float(tp / n),
                "recall_lower": float(lower),
            }
        )
    return rows
