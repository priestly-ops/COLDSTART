Yes. Here is the exact version I would now freeze for the appendix and implementation so there is no remaining ambiguity between theory, notation, and code.

### Conformal calibration: exact \(\alpha\)-to-\(B\) mapping

For the **primary deterministic, nonrandomized split-conformal calibration**, set

\[
\boxed{\alpha=B}
\]

where \(B\) is the deployment false-alarm budget.

Thus, for the primary operating point,

\[
B=0.01
\quad\Longrightarrow\quad
\alpha=0.01.
\]

Given healthy calibration scores

\[
A_1,\ldots,A_m
\]

sorted as

\[
A_{(1)}\le\cdots\le A_{(m)},
\]

define

\[
\boxed{
k_\alpha=\left\lceil (m+1)(1-\alpha)\right\rceil.
}
\]

The deployed threshold is

\[
\boxed{
\tau_\alpha=
\begin{cases}
A_{(k_\alpha)}, & k_\alpha\le m,\\[4pt]
+\infty, & k_\alpha=m+1.
\end{cases}
}
\]

The detector then uses

\[
\boxed{
\hat y(X)=\mathbf 1[s(X)>\tau_\alpha].
}
\]

So there are three operational regimes:

- \(k_\alpha=m+1\): threshold is \(+\infty\), hence no finite score can trigger an alarm and recall is exactly 0.
- \(k_\alpha=m\): threshold is the largest calibration score.
- \(k_\alpha\le m-1\): threshold is below the maximum calibration score.

For \(B=\alpha=0.01\), this produces the important transitions:

\[
m<99
\]

can enter the \(+\infty\) regime depending on the exact rank;

\[
m=99
\]

is the first point where a finite deterministic 1% threshold is available;

and

\[
m=199
\]

is the first point where the selected threshold moves below the calibration maximum.

This should be the exact rule implemented in code. If we later add randomized conformal, it must be a separately named calibration method rather than silently changing this rule.

---

## Exact binomial certification

After training and calibration, the detector is frozen.

For healthy evaluation episodes,

\[
X_F=FP\sim\mathrm{Binomial}(n_H,p_F),
\]

where

\[
n_H=FP+TN.
\]

For anomaly episodes,

\[
X_R=TP\sim\mathrm{Binomial}(n_A,p_R),
\]

where

\[
n_A=TP+FN.
\]

The primary joint-certification confidence is

\[
\gamma=0.95.
\]

Using Bonferroni,

\[
\boxed{
\delta_F=\delta_R=\frac{1-\gamma}{2}=0.025.
}
\]

Therefore each component uses a one-sided **97.5%** exact interval.

The SciPy `beta.ppf` operation is appropriate because it returns the inverse Beta CDF. SciPy's binomial documentation gives the binomial CDF in incomplete-Beta form, which is the identity underlying the Clopper–Pearson inversion. ([docs.scipy.org](https://docs.scipy.org/doc/scipy-1.8.1/scipy-ref-1.8.1.pdf?utm_source=chatgpt.com))

### FPR upper bound

For

\[
0\le FP<n_H,
\]

use

\[
\boxed{
U_F
=
\mathrm{Beta}^{-1}
\left(
1-\delta_F;
FP+1,\,
n_H-FP
\right).
}
\]

Since

\[
n_H-FP=TN,
\]

this is equivalently

\[
\boxed{
U_F
=
\mathrm{Beta}^{-1}
\left(
0.975;
FP+1,\,
TN
\right).
}
\]

Boundary case:

\[
\boxed{
FP=n_H
\Rightarrow
U_F=1.
}
\]

For zero false positives,

\[
FP=0,
\]

the expression reduces to

\[
\boxed{
U_F
=
1-\delta_F^{1/n_H}.
}
\]

At our Bonferroni setting,

\[
\delta_F=0.025.
\]

---

## Recall lower bound

For

\[
0<TP\le n_A,
\]

use

\[
\boxed{
L_R
=
\mathrm{Beta}^{-1}
\left(
\delta_R;
TP,\,
n_A-TP+1
\right).
}
\]

Since

\[
n_A-TP=FN,
\]

this becomes

\[
\boxed{
L_R
=
\mathrm{Beta}^{-1}
\left(
0.025;
TP,\,
FN+1
\right).
}
\]

Boundary case:

\[
\boxed{
TP=0
\Rightarrow
L_R=0.
}
\]

When all anomalies are detected,

\[
TP=n_A,
\]

this simplifies to

\[
\boxed{
L_R=\delta_R^{1/n_A}.
}
\]

---

# Exact Python implementation

This is the implementation I would put into the project and unit-test directly.

```python id="xura7e"
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class CertificationBounds:
    recall: float
    fpr: float
    recall_lower: float
    fpr_upper: float
    certified: bool


def exact_one_sided_recall_lower(
    tp: int,
    fn: int,
    delta: float,
) -> float:
    """
    Exact one-sided Clopper-Pearson lower confidence bound
    for recall p_R.

    Confidence level = 1 - delta.
    """
    tp = int(tp)
    fn = int(fn)

    if tp < 0 or fn < 0:
        raise ValueError("TP and FN must be non-negative.")

    n = tp + fn
    if n <= 0:
        raise ValueError("At least one anomalous evaluation episode is required.")

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
    """
    Exact one-sided Clopper-Pearson upper confidence bound
    for healthy false-positive probability p_F.

    Confidence level = 1 - delta.
    """
    fp = int(fp)
    tn = int(tn)

    if fp < 0 or tn < 0:
        raise ValueError("FP and TN must be non-negative.")

    n = fp + tn
    if n <= 0:
        raise ValueError("At least one healthy evaluation episode is required.")

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
    """
    Bonferroni simultaneous certification of:

        recall >= recall_target
        FPR    <= fpr_budget

    joint_confidence=0.95 allocates 0.025 error probability
    to each one-sided bound.
    """
    if not 0.0 < joint_confidence < 1.0:
        raise ValueError("joint_confidence must lie strictly between 0 and 1.")

    if not 0.0 <= recall_target <= 1.0:
        raise ValueError("recall_target must lie in [0, 1].")

    if not 0.0 <= fpr_budget <= 1.0:
        raise ValueError("fpr_budget must lie in [0, 1].")

    n_anom = tp + fn
    n_healthy = fp + tn

    if n_anom == 0:
        raise ValueError("No anomalous evaluation episodes.")
    if n_healthy == 0:
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
    )
```

---

# Exact conformal implementation

```python id="ly32rl"
def deterministic_conformal_threshold(
    calibration_scores: np.ndarray,
    alpha: float,
) -> tuple[float, int, str]:
    """
    Deterministic nonrandomized one-sided split-conformal
    threshold for anomaly scores where larger = more anomalous.

    Returns
    -------
    threshold
        Finite calibration order statistic or +inf.
    rank
        1-indexed conformal order-statistic rank.
    regime
        One of:
        - "infinite"
        - "maximum"
        - "submaximum"
    """
    scores = np.asarray(
        calibration_scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError("calibration_scores must be one-dimensional.")

    if scores.size == 0:
        raise ValueError("At least one calibration score is required.")

    if not np.isfinite(scores).all():
        raise ValueError("Calibration scores must all be finite.")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    m = scores.size

    rank = int(
        math.ceil(
            (m + 1) * (1.0 - alpha)
        )
    )

    # Conformal augmented order statistic A_(m+1) = +infinity.
    if rank > m:
        return (
            float("inf"),
            rank,
            "infinite",
        )

    ordered = np.sort(scores)

    # rank is 1-indexed; NumPy is 0-indexed.
    threshold = float(
        ordered[rank - 1]
    )

    if rank == m:
        regime = "maximum"
    else:
        regime = "submaximum"

    return (
        threshold,
        rank,
        regime,
    )
```

For the primary protocol, the caller should use:

```python id="fi5eyp"
alpha = fpr_budget
```

not an independently tuned alpha.

---

# Boundary behavior should be explicitly tested

These tests are manuscript-critical because they ensure the implementation matches the derivation.

```python id="ua7q9w"
def test_zero_false_positive_upper_bound():
    # Joint 95% Bonferroni => delta_F = 0.025
    delta = 0.025
    n = 368

    bound = exact_one_sided_fpr_upper(
        fp=0,
        tn=n,
        delta=delta,
    )

    expected = 1.0 - delta ** (1.0 / n)

    assert np.isclose(bound, expected)
    assert bound <= 0.01


def test_all_false_positive_boundary():
    assert (
        exact_one_sided_fpr_upper(
            fp=100,
            tn=0,
            delta=0.025,
        )
        == 1.0
    )


def test_zero_true_positive_boundary():
    assert (
        exact_one_sided_recall_lower(
            tp=0,
            fn=100,
            delta=0.025,
        )
        == 0.0
    )


def test_all_anomalies_detected_closed_form():
    delta = 0.025
    n = 36

    bound = exact_one_sided_recall_lower(
        tp=n,
        fn=0,
        delta=delta,
    )

    expected = delta ** (1.0 / n)

    assert np.isclose(bound, expected)
    assert bound >= 0.90


def test_conformal_infinite_regime():
    scores = np.arange(50, dtype=float)

    threshold, rank, regime = deterministic_conformal_threshold(
        scores,
        alpha=0.01,
    )

    assert rank == 51
    assert math.isinf(threshold)
    assert regime == "infinite"


def test_conformal_maximum_regime():
    scores = np.arange(100, dtype=float)

    threshold, rank, regime = deterministic_conformal_threshold(
        scores,
        alpha=0.01,
    )

    assert rank == 100
    assert threshold == 99.0
    assert regime == "maximum"


def test_conformal_second_largest_transition():
    scores = np.arange(199, dtype=float)

    threshold, rank, regime = deterministic_conformal_threshold(
        scores,
        alpha=0.01,
    )

    assert rank == 198
    assert threshold == 197.0
    assert regime == "submaximum"
```

The first two conformal transition points are therefore reproducibly encoded:

\[
m=50,\alpha=.01
\Rightarrow
k=51
\Rightarrow
\tau=+\infty
\]

and

\[
m=199,\alpha=.01
\Rightarrow
k=198
\Rightarrow
\tau=A_{(198)}.
\]

---

## Appendix-ready wording

I would put this paragraph almost verbatim into the paper:

> **Confidence semantics.** For each commissioning replicate, detector fitting and threshold calibration are completed before evaluation. Conditional on the resulting frozen detector, healthy and anomalous evaluation episodes are treated as Bernoulli trials at the episode level. We compute exact one-sided Clopper–Pearson limits by inversion of the binomial distribution. For a target simultaneous confidence \(\gamma=0.95\), Bonferroni allocation sets \(\delta_F=\delta_R=(1-\gamma)/2=0.025\). The false-positive requirement is certified when the one-sided upper bound \(U_F=\mathrm{Beta}^{-1}(1-\delta_F;FP+1,TN)\le B\), with \(U_F=1\) when \(FP=n_H\). The recall requirement is certified when the one-sided lower bound \(L_R=\mathrm{Beta}^{-1}(\delta_R;TP,FN+1)\ge R_0\), with \(L_R=0\) when \(TP=0\). Certification therefore refers to the stated episode-level binomial model and simultaneous confidence procedure; repeated commissioning seeds are used separately to quantify commissioning-draw variability and are not interpreted as independent physical robot replications.

That now aligns the math, boundary cases, multiplicity rule, and implementation.

One final thing I would lock in the code: save the fields `delta_recall`, `delta_fpr`, `joint_confidence`, `calibration_alpha`, `conformal_rank`, and `conformal_regime` into every result row or manifest. That way a reviewer—or we ourselves six months later—can reconstruct exactly why a run was or was not certified.