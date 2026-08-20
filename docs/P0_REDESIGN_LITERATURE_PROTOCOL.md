# P0 redesign: hardened literature-grounded precision-transfer feasibility protocol

## Status

This document supersedes the earlier informal P0 redesign. The original P0
outputs remain frozen and must not be overwritten. The purpose of this redesign
is to determine whether precision transfer is statistically viable for COLDSTART
without tuning on anomaly outcomes.

## Why the original P0 gate is insufficient

The original P0 audit independently estimated source and target sparse precision
matrices and compared recovered stable supports. In the low-target-sample,
high-dimensional regime, disagreement between two estimated supports can reflect
estimation noise rather than true source-target structural mismatch. The current
voraus feature representation has p=564 while target commissioning N is only
10, 25, 50, or 100. P0.1 confirmed severe underdetermination across that entire
range.

Therefore the original result supports only the following statement:

> Independent full-dimensional precision-support recovery was not identifiable
> enough under the original low-N protocol to adjudicate the transfer hypothesis.

It does not establish that transferable source-target precision structure is
absent.

## Literature basis

The redesign is grounded in five established lines of work:

1. Sparse precision estimation (Graphical Lasso and CLIME): valid optimization
   at p>n does not imply reliable support recovery; performance depends on
   sparsity, signal strength, conditioning, and sample size relative to graph
   complexity.
2. Stability selection / StARS: resampling can stabilize tuning but cannot
   recover information that is not present in an extreme p>>n target sample.
3. Joint graphical models: Fused/Group Graphical Lasso borrow strength across
   related classes but require the amount and type of cross-class similarity to
   be tuned.
4. Trans-CLIME: transfer gains require sufficiently informative auxiliary
   studies and rely on a sparse source-target divergence. Its positive-transfer
   aggregation uses a held-out target split.
5. Trans-Glasso: transfer is modeled as shared structure plus differential
   refinement and is most appropriate when source and target precision matrices
   share a substantial fraction of entries.

Primary references:

- Friedman, Hastie & Tibshirani (2008), graphical lasso.
- Ravikumar et al. (2011), high-dimensional log-determinant estimation.
- Cai, Liu & Luo (2011), CLIME.
- Liu, Roeder & Wasserman (2010), StARS.
- Danaher, Wang & Witten (2014), Joint Graphical Lasso.
- Li, Cai & Li, Trans-CLIME.
- Zhao, Ma & Kolar, Trans-Glasso.
- He et al., Trans-Copula-CLIME, for non-Gaussian robustness.

## P0.1 finding

The conditioning audit is treated as a diagnostic result, not a model-selection
step. At p=564, p/N ranges from 56.4 at N=10 to 5.64 at N=100. The centered
sample rank is bounded by N-1 and the observed effective rank is a small fraction
of the nominal feature dimension. Bootstrap instability of target-derived
location/scale estimates is also largest at small N.

Decision: proceed to transfer-aware estimation, but do not assume that a
full-dimensional estimator will be viable at N=10.

## Critical limitations that must be addressed before P0.2

### L1. Existing Trans-CLIME theory does not directly cover our most extreme regime

Published Trans-CLIME results assume sparsity/similarity conditions and target
sample-size growth sufficient for the stated rates. The paper's numerical
experiments use substantially larger target samples than N=10. Consequently,
its theoretical guarantees cannot be cited as guarantees for p=564, N=10.

Mitigation: treat p=564,N=10 as an empirical stress regime. Validate the method
on synthetic truth at progressively harder p/N ratios before real-data use.

### L2. Target sample splitting is a severe problem at N=10

The reference Trans-CLIME procedure splits the target sample so that most target
samples fit the initial estimators and a held-out fraction performs positive-
transfer aggregation. With N=10, an 80/20 split leaves only approximately eight
fit samples and two aggregation samples, making the aggregation weights highly
variable.

Mitigation: evaluate two predeclared variants:

- Reference Trans-CLIME with the published split, reported for fidelity.
- Cross-fitted Trans-CLIME, where aggregation is repeated across deterministic
  folds and predictions/estimators are aggregated. This is labeled an extension,
  not the published method.

If cross-fitting is used downstream, it must be frozen before anomaly testing.

### L3. CLIME estimates need not be positive definite

A symmetric CLIME estimate is not automatically positive definite, while RACE
uses precision geometry for Mahalanobis-type scoring and held-out Gaussian
likelihood requires a valid positive-definite precision matrix.

Mitigation: report both the raw estimator and a deterministic SPD projection.
Record the projection magnitude. Estimation comparisons use the raw estimator
when mathematically valid; deployment geometry uses the projected estimator.
A method that requires a large projection is flagged as numerically unsuitable.

### L4. Gaussianity is unverified for robot-cycle statistical features

The engineered feature vector contains means, quantiles, standard deviations,
and total variation; Gaussianity should not be assumed from construction.

Mitigation: add healthy-only marginal/tail diagnostics and include
Trans-Copula-CLIME as a predeclared robustness sensitivity if Gaussian departure
is substantial. Do not select between Gaussian and copula variants using anomaly
labels.

### L5. Source-target similarity is unknown and negative transfer is possible

Transfer-precision methods improve estimation only when auxiliary domains are
sufficiently informative. An unrelated source can bias the target estimator.

Mitigation: synthetic scenarios must include identical, mildly shifted,
moderately shifted, and unrelated sources. Real P0.4 must include the existing
near/moderate/far source regimes and source-permutation controls. A transfer
method is not allowed into P1 if it has severe negative transfer on unrelated
controls.

### L6. Full source pooling can create a mixture distribution

Using every healthy source episode is beneficial only if those episodes represent
one coherent source operating distribution. Pooling distinct source conditions
can produce a covariance matrix that represents a mixture rather than a valid
single-domain GGM.

Mitigation: report source composition explicitly. Primary transfer uses one
predeclared coherent source regime. Full-source pooling is a sensitivity analysis
unless homogeneity is verified using healthy-only metadata and diagnostics.

### L7. Scaling can leak information into held-out estimator selection

Using statistics computed from all target commissioning samples before a
held-out likelihood or aggregation step contaminates that validation objective,
even though anomaly labels are absent.

Mitigation: within any target train/validation split, fit target-dependent
scaling on the training fold only. A source-frozen scaler is also evaluated as a
predeclared baseline. Any RACE-style source-to-target scale shrinkage is an
explicit method variant and is frozen on healthy/synthetic criteria only.

### L8. Synthetic benchmarks can be too clean and produce optimistic conclusions

Sparse Gaussian graphs alone do not mimic the low effective rank, heavy tails,
feature blocks, and possible mixtures seen in robot features.

Mitigation: use a two-tier synthetic suite:

Tier 1 -- canonical theory-aligned Gaussian sparse graphs.
Tier 2 -- robotics-stress synthetic data with low-rank factors, correlated
feature blocks, non-Gaussian monotone transforms, and moderate contamination.

A method must succeed in Tier 1 before Tier 2 is interpreted.

### L9. Tuning can accidentally use oracle information

Synthetic truth is available and can tempt tuning directly on Frobenius error,
which would overstate deployable performance.

Mitigation: distinguish two evaluations:

- Oracle-capability analysis: best grid point with access to truth, clearly
  labeled as an upper bound.
- Deployable analysis: tuning chosen only from training/healthy validation data
  using a predeclared criterion.

Only the deployable estimator may advance to real robotics P0.4/P1.

### L10. Support recovery is not the same objective as anomaly scoring

A precision estimator can have imperfect edge support yet still estimate the
quadratic geometry well enough for anomaly ranking.

Mitigation: make relative Frobenius error and held-out healthy likelihood the
primary synthetic estimation metrics. Support precision/recall/F1/Jaccard are
secondary structural metrics. Downstream anomaly benefit remains a separate P1
question.

### L11. Better precision estimation does not solve the calibration bottleneck

Previous COLDSTART experiments showed that low-FPR calibration itself can be
operationally limiting. Therefore even a successful P0 transfer estimator may
fail to reduce commissioning N* under Recall>=0.90 and FPR<=0.01.

Mitigation: preserve the existing frozen calibration audit as an independent
axis. P0 success is necessary evidence for a precision-transfer RACE variant,
not evidence that the final commissioning target will be met.

### L12. Cycles may not be iid

Sequential robot executions can share drift, batches, warm-up state, or temporal
dependence, while standard GGM and conformal arguments often rely on iid or
exchangeable observations.

Mitigation: add episode-order/batch diagnostics where metadata permits. Use
blocked sensitivity splits and cluster/block bootstrap intervals if dependence
is detected. Do not randomize away an observed batch effect in the primary
analysis.

### L13. Computational cost at p=564 can be large

CLIME-type optimization decomposes by columns but still requires hundreds of
high-dimensional constrained optimizations per tuning value. Multi-method,
multi-seed synthetic sweeps can become expensive.

Mitigation: stage dimensions sequentially, cache generated data and covariance
matrices, parallelize across independent replications rather than oversubscribing
BLAS within one fit, and stop escalating dimension after a predefined failure.

### L14. Novelty risk

If RACE simply reimplements Trans-CLIME or Trans-Glasso, the method contribution
is not novel.

Mitigation: treat transfer-GGM estimators as literature baselines or statistical
building blocks. RACE's contribution must remain distinct: commissioning-aware
source gating/shrinkage, robotics-specific adaptation, operational low-FPR
calibration, or another clearly new mechanism validated by ablation.

## Hardened P0.2 design

### Stage A -- estimator correctness tests

Before experiments, unit-test each estimator on small known matrices:

- symmetry and dimensions,
- objective/constraint feasibility,
- deterministic output for fixed seed,
- recovery on identity precision,
- behavior with duplicated/constant features,
- SPD projection magnitude,
- agreement with official/reference implementation on a small reproducible case.

No large benchmark proceeds until these tests pass.

### Stage B -- canonical synthetic recovery

Dimensions: p in {40, 128} initially.
Target N: {10, 25, 50, 100} where computationally feasible.
Source sizes: use predeclared feasible sizes that are actually available in the
real source system, plus a larger synthetic source-size sensitivity.
Replications: 20 smoke, 50 development/freeze, 200 final only for the frozen
small set of primary comparisons.

Structures:

1. identical precision,
2. same support with perturbed weights,
3. highly related sparse differential,
4. moderately related differential,
5. unrelated source.

Methods:

- Target Graphical Lasso,
- Target CLIME,
- reference Trans-CLIME,
- cross-fitted Trans-CLIME extension,
- Joint/Fused Graphical Lasso,
- Trans-Glasso if reference implementation can be reproduced.

Primary metrics:

- relative Frobenius error to true target precision,
- held-out target Gaussian negative log-likelihood after valid SPD handling,
- negative-transfer regret relative to the matched target-only estimator.

Secondary metrics:

- elementwise max error,
- spectral error,
- support precision, recall, F1, and Jaccard,
- SPD projection magnitude,
- convergence/failure rate and runtime.

### Stage C -- robotics-stress synthetic recovery

Only methods that are valid in Stage B advance. Add:

- low-rank factor structure calibrated to P0.1 effective rank,
- correlated feature blocks,
- monotone non-Gaussian transforms,
- source-target scale/location shifts,
- mild contamination/outliers.

### Stage D -- dimensional escalation

Advance p=128 -> 256 -> 564 only if the frozen deployable estimator beats the
matched target-only baseline in related-source cases without material negative
transfer in unrelated-source cases.

A stage failure is a scientific result and stops further dimension escalation
for that estimator.

## Predeclared decision criteria

For each related-source scenario, define paired relative improvement

    G = (E_target_only - E_transfer) / E_target_only

where E is the primary precision-estimation error.

A method advances if all of the following hold on the frozen synthetic suite:

1. Median G > 0 in the majority of related-source scenarios.
2. The paired 95% bootstrap interval for G excludes a practically important
   negative effect in the primary related scenario.
3. In unrelated-source scenarios, median negative-transfer regret is below a
   predeclared tolerance and catastrophic failures are absent.
4. Numerical/convergence failure rate is <=5% in the primary stage.
5. SPD projection is not routinely large enough to dominate the original
   estimate.
6. The deployable tuning rule, not the oracle tuning result, meets criteria.

A 10% improvement may be reported as an engineering effect-size target, but is
not used as a universal statistical law or tuned post hoc.

## P0.3 representation sensitivity

P0.3 is triggered if all full-dimensional transfer estimators fail or are
numerically unstable.

Representations are frozen before real anomaly evaluation:

1. original 564-D statistical features,
2. physically grouped/channel-level representation,
3. source-fitted PCA at a small predeclared set of dimensions as sensitivity
   analysis only.

PCA must be fit on source healthy data only (or within training folds for a
strict target-only comparator) and applied unchanged to held-out target data.
Precision-graph semantics in PCA space must not be described as original sensor
conditional-independence edges.

## P0.4 real healthy transfer feasibility

One estimator and one representation are frozen from synthetic/healthy-only
criteria. The real healthy experiment then evaluates:

- near/moderate/far source regimes,
- original commissioning grid and seeds,
- source-permutation/unrelated-source controls,
- estimator numerical reliability,
- held-out healthy likelihood/geometry stability,
- source-size sensitivity where scientifically coherent.

No anomaly outcomes are inspected during selection.

## P1 deployment performance remains independent

Only after P0.1-P0.4 are frozen may the estimator enter anomaly evaluation:

- Recall >= 0.90,
- FPR <= 0.01,
- confidence-interval-based N*,
- TargetOnly comparison,
- frozen calibration protocol,
- calibration-size/alpha feasibility audit retained independently.

A P0 success is not interpreted as a P1 success.

## Stop rules

Stop precision-transfer development if any of the following occurs:

- reference methods cannot be reproduced on small canonical examples,
- no transfer estimator improves synthetic target precision estimation under
  genuinely related sources,
- improvements exist only with oracle tuning,
- negative transfer remains substantial under unrelated sources,
- full-dimensional methods require representation changes so severe that the
  resulting object no longer matches the intended RACE precision geometry,
- downstream gains disappear under the frozen calibration protocol.

These stop rules protect the paper from post-hoc method chasing.
