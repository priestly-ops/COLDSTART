# P0 redesign: literature-grounded precision-transfer feasibility protocol

## Why the original P0 gate is being revised

The original P0 audit independently estimated source and target sparse precision
matrices and then compared their recovered stable supports. The high-dimensional
synthetic stress test showed that this gate is not sufficiently identifiable in
the low-target-sample regime: even when the *true* source and target precision
matrices share support, independent estimators can recover very different edge
sets. Therefore an observed source-target support mismatch cannot yet be
interpreted as evidence that transferable structure is absent.

This is a methodological diagnosis, not a favorable reinterpretation of the
robotics result. The original P0 outputs must remain frozen as an audit trail.

## Literature basis

The redesign is motivated by four established lines of work:

1. **High-dimensional sparse precision estimation.** Graphical Lasso and CLIME
   can estimate sparse inverse covariance matrices when p exceeds n, but
   support recovery still requires sparsity, sufficient signal strength, and
   sample size scaling with graph complexity and log(p). A numerically valid
   optimizer is not equivalent to reliable edge recovery.

2. **Stability selection / StARS.** Stability-based tuning reduces sensitivity
   to a single fitted graph, but it does not manufacture information in an
   extreme p >> n regime. Hard support thresholding can amplify small
   probability differences into large Jaccard differences.

3. **Joint graphical models.** Fused and group graphical lasso estimate related
   networks jointly and explicitly borrow strength across classes rather than
   requiring each class to recover its graph independently.

4. **Transfer precision estimation.** Trans-CLIME and Trans-Glasso explicitly
   target the setting where a small target study can borrow information from
   related source studies while allowing a sparse source-target differential
   structure. This matches the statistical question underlying RACE more
   closely than independent source/target support recovery.

Key references to cite in the paper/protocol:

- Friedman, Hastie & Tibshirani, *Sparse inverse covariance estimation with the
  graphical lasso*, Biostatistics, 2008.
- Ravikumar et al., *High-dimensional covariance estimation by minimizing
  l1-penalized log-determinant divergence*, Electronic Journal of Statistics,
  2011.
- Cai, Liu & Luo, *A constrained l1 minimization approach to sparse precision
  matrix estimation*, JASA, 2011 (CLIME).
- Liu, Roeder & Wasserman, *Stability Approach to Regularization Selection
  (StARS) for high dimensional graphical models*, NIPS, 2010.
- Danaher, Wang & Witten, *The joint graphical lasso for inverse covariance
  estimation across multiple classes*, JRSS-B, 2014.
- Li, Cai & Li, *Transfer Learning in Large-scale Gaussian Graphical Models with
  False Discovery Rate Control* (Trans-CLIME), JASA.
- Zhao, Ma & Kolar, *Trans-Glasso: A Transfer Learning Approach to Precision
  Matrix Estimation*.

## Revised sequence

### P0.1 — conditioning and redundancy audit

**Question:** Is the target feature geometry sufficiently conditioned for
independent precision support recovery at N = 10, 25, 50, 100?

Healthy-only diagnostics:

- p / N
- centered numerical rank and rank fraction
- entropy effective rank and stable rank
- raw covariance condition number
- Ledoit-Wolf shrinkage and conditioned covariance condition number
- fraction of feature pairs with |r| >= .90, .95, .99
- redundancy among the six statistics emitted for each physical signal
- bootstrap instability of the target-derived median/IQR scaler
- source/target clipping induced by the target-derived robust scaler

No anomaly outcomes are used and no estimator hyperparameter is selected.

**Interpretation:** If raw rank/effective rank is extremely small relative to p,
within-signal redundancy is high, and target scaling is unstable at small N,
then support-recovery failure is expected and the representation/estimator must
be changed before another robotics transfer claim is attempted.

### P0.2 — synthetic estimator-recovery benchmark

**Question:** Can a transfer-aware estimator recover the target precision matrix
more accurately than target-only estimators when the source is genuinely
related, without strong negative transfer when it is unrelated?

Synthetic truth cases:

- identical sparse precision matrices
- same support with perturbed weights
- highly shared support with sparse differential edges
- partially shared structure
- unrelated / density-matched structure

Dimensions are evaluated sequentially rather than all at once:

- Stage A: p = 40, 128
- Stage B (only if Stage A is valid): p = 256
- Stage C (only if Stage B is valid): real feature dimension

Target N: 10, 25, 50, 100. Source n should remain large enough to represent a
commissioned source system.

Primary metrics use known synthetic truth:

- relative Frobenius error of target precision
- elementwise max error
- support precision / recall / F1
- support Jaccard to the *true target graph*

Methods:

- Target Graphical Lasso
- Target CLIME
- Joint/Fused Graphical Lasso
- Trans-CLIME
- Trans-Glasso when a reproducible implementation is available

The benchmark must not tune against desired RACE outcomes. Tuning grids and
selection rules are frozen before real anomaly evaluation.

### P0.3 — representation sensitivity

Only after P0.1 establishes substantial redundancy/conditioning problems.
Compare a small predeclared set of healthy-only representations:

1. frozen 6-statistic representation (current 564-D reference),
2. physically grouped/channel-level representation,
3. PCA or another unsupervised projection only as a sensitivity control.

The primary representation should preserve physical interpretability where
possible. PCA changes the node semantics of a precision graph and therefore
must not silently replace the original representation.

### P0.4 — real healthy transfer feasibility

Freeze one estimator and one representation based only on synthetic recovery
and healthy conditioning diagnostics. Then test real source-target healthy data
across the original source-shift regimes and commissioning seeds.

Required negative controls:

- unrelated source regime,
- source feature permutation,
- source-label/episode permutation where scientifically meaningful.

The method must show evidence of useful transfer on related regimes and limited
negative transfer on unrelated controls before anomaly outcomes are inspected.

### P1 — deployment performance

Only after P0.1–P0.4 are frozen may the transfer estimator enter the original
commissioning evaluation:

- Recall >= 0.90
- FPR <= 0.01
- confidence-interval-based N* estimator
- TargetOnly comparison
- frozen calibration protocol

## Reviewer-facing claim discipline

Until the revised gate passes, do **not** claim that the real robot lacks shared
precision structure. The supported statement is narrower:

> Independent high-dimensional precision-support recovery was not identifiable
> enough under the original low-N protocol to adjudicate the transfer
> hypothesis.

Likewise, a successful transfer-aware estimator would not by itself establish
anomaly-detection benefit. It would only justify advancing to the frozen P1
commissioning experiment.
