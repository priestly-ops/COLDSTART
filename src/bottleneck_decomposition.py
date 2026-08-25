from __future__ import annotations

from dataclasses import dataclass

from src.oracle_feasibility import OracleFeasibilityResult


REPRESENTATION_LIMITED = "representation_limited"
CALIBRATION_LIMITED = "calibration_limited"
CERTIFICATION_LIMITED = "certification_limited"
CERTIFIED = "certified"

VALID_BOTTLENECK_LABELS = {
    REPRESENTATION_LIMITED,
    CALIBRATION_LIMITED,
    CERTIFICATION_LIMITED,
    CERTIFIED,
}


@dataclass(frozen=True)
class BottleneckDecompositionResult:
    """Deterministic E1 attribution for one frozen detector replicate.

    The attribution is hierarchical and uses only predeclared quantities:

    1. Representation/score limitation is diagnosed with the retrospective
       empirical oracle operating on the same scalar score and the same
       monotone `score > threshold` decision family as deployment.
    2. If the oracle is feasible but the deployed empirical operating point
       fails, the replicate is calibration-limited.
    3. If the deployed empirical operating point passes but exact joint
       certification fails, the replicate is certification-limited.
    4. Otherwise the replicate is certified.

    Oracle quantities are diagnostic only and are never used for deployment,
    model fitting, threshold calibration, or certification.
    """

    bottleneck_label: str
    oracle_empirically_feasible: bool
    deployed_empirical_success: bool
    deployed_certified_success: bool
    oracle_recall_at_fpr_budget: float
    oracle_fpr_at_max_recall: float
    deployed_recall: float
    deployed_fpr: float
    recall_lower: float
    fpr_upper: float
    oracle_minus_deployed_recall: float
    deployed_recall_deficit: float
    deployed_fpr_excess: float


def classify_bottleneck(
    *,
    oracle: OracleFeasibilityResult,
    deployed_recall: float,
    deployed_fpr: float,
    recall_lower: float,
    fpr_upper: float,
    recall_target: float,
    fpr_budget: float,
    tolerance: float = 1e-12,
) -> BottleneckDecompositionResult:
    """Classify one run into a reproducible COLDSTART bottleneck regime.

    Parameters
    ----------
    oracle:
        Retrospective empirical threshold oracle evaluated on the same score
        family as the deployed detector.
    deployed_recall, deployed_fpr:
        Empirical metrics at the predeclared deployed calibration threshold.
    recall_lower, fpr_upper:
        Exact one-sided seed-level certification bounds from E0.
    recall_target, fpr_budget:
        Operating specification used by both deployment and oracle analysis.
    tolerance:
        Numerical tolerance for comparisons at exact operating boundaries.
    """
    for name, value in {
        "deployed_recall": deployed_recall,
        "deployed_fpr": deployed_fpr,
        "recall_lower": recall_lower,
        "fpr_upper": fpr_upper,
        "recall_target": recall_target,
        "fpr_budget": fpr_budget,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")

    if abs(oracle.recall_target - recall_target) > tolerance:
        raise ValueError(
            "Oracle recall target does not match decomposition target: "
            f"{oracle.recall_target} vs {recall_target}."
        )
    if abs(oracle.false_alert_budget - fpr_budget) > tolerance:
        raise ValueError(
            "Oracle FPR budget does not match decomposition budget: "
            f"{oracle.false_alert_budget} vs {fpr_budget}."
        )

    empirical_success = bool(
        deployed_recall >= recall_target - tolerance
        and deployed_fpr <= fpr_budget + tolerance
    )
    certified_success = bool(
        recall_lower >= recall_target - tolerance
        and fpr_upper <= fpr_budget + tolerance
    )

    # If the deployed threshold empirically satisfies the operating point,
    # the exhaustive oracle over the same threshold family must also declare
    # the score empirically feasible. Any disagreement is an implementation
    # or tie-handling error and should fail loudly rather than be classified.
    if empirical_success and not oracle.empirically_feasible:
        raise RuntimeError(
            "Deployed empirical success contradicts oracle infeasibility. "
            "Check score direction, threshold convention, and tie handling."
        )

    # Exact certification implies empirical success for the same thresholds
    # under the predeclared criterion. This check protects against accidental
    # mixing of bounds, operating points, or evaluation sets.
    if certified_success and not empirical_success:
        raise RuntimeError(
            "Certified success without empirical success is inconsistent. "
            "Check evaluation counts and operating-point definitions."
        )

    if not oracle.empirically_feasible:
        label = REPRESENTATION_LIMITED
    elif not empirical_success:
        label = CALIBRATION_LIMITED
    elif not certified_success:
        label = CERTIFICATION_LIMITED
    else:
        label = CERTIFIED

    return BottleneckDecompositionResult(
        bottleneck_label=label,
        oracle_empirically_feasible=bool(oracle.empirically_feasible),
        deployed_empirical_success=empirical_success,
        deployed_certified_success=certified_success,
        oracle_recall_at_fpr_budget=float(oracle.max_recall_at_fpr_budget),
        oracle_fpr_at_max_recall=float(oracle.fpr_at_max_recall),
        deployed_recall=float(deployed_recall),
        deployed_fpr=float(deployed_fpr),
        recall_lower=float(recall_lower),
        fpr_upper=float(fpr_upper),
        oracle_minus_deployed_recall=float(
            oracle.max_recall_at_fpr_budget - deployed_recall
        ),
        deployed_recall_deficit=float(
            max(0.0, recall_target - deployed_recall)
        ),
        deployed_fpr_excess=float(
            max(0.0, deployed_fpr - fpr_budget)
        ),
    )
