# P0 precision-transfer limitation resolution matrix

This document records which limitations can be removed by implementation, which
can only be mitigated experimentally, and which are irreducible facts of the
commissioning problem.  It prevents later result-driven changes from being
presented as if they were part of the original protocol.

## Status vocabulary

- **RESOLVED IN CODE**: the avoidable implementation failure mode has a concrete
  safeguard and test.
- **MITIGATED BY EXPERIMENT**: no algorithm can remove the underlying assumption;
  the protocol measures sensitivity and can stop the method.
- **IRREDUCIBLE**: this is a property of the deployment regime, not a software
  defect.  The paper must report it rather than claim it has been solved.

| ID | Limitation | Status | Concrete resolution / gate |
|---|---|---|---|
| L1 | Published transfer-GGM theory does not guarantee p=564,N=10 | IRREDUCIBLE | Treat 564/10 only as an empirical stress regime; staged p=10/20 -> 40/128 -> 256 -> 564 escalation; never claim a theorem covers the robotics endpoint. |
| L2 | Reference Trans-CLIME target split is unstable at N=10 | MITIGATED BY EXPERIMENT | Keep reference split for fidelity and add separately named cross-fitted COLDSTART extension. Cross-fit is not called published Trans-CLIME. |
| L3 | CLIME/Trans-CLIME need not be SPD | RESOLVED IN CODE | `src/precision_transfer_estimators.py` retains raw/symmetric estimates, applies deterministic eigen-floor SPD projection, and reports relative projection magnitude. Large projection triggers a numerical-suitability failure. |
| L4 | Gaussianity of cycle features is unverified | MITIGATED BY EXPERIMENT | Add healthy-only distribution diagnostics before real-data estimator freeze. If substantial departure persists, Trans-Copula-CLIME is the predeclared robustness comparator; anomaly labels cannot choose between them. |
| L5 | Negative transfer from unrelated sources | MITIGATED BY EXPERIMENT | Every synthetic stage includes related and unrelated source truth. Real P0.4 keeps near/moderate/far/permuted-source controls. A method with practically important unrelated-source regret cannot enter P1. |
| L6 | Full source pooling can create a mixture | MITIGATED BY EXPERIMENT | Primary source must be one coherent predeclared regime. Full pooling is sensitivity-only until healthy metadata/distribution diagnostics support homogeneity. |
| L7 | Scaling leakage into validation/aggregation | RESOLVED IN CODE | Scaling API accepts training rows only. Any fold-dependent target scaling is fit inside that fold. Source-frozen and source->target shrink scalers are explicit variants. |
| L8 | Clean Gaussian simulation can be unrealistically easy | MITIGATED BY EXPERIMENT | Two-tier simulation: theory-aligned canonical graphs first, then robotics-stress low-rank/block/non-Gaussian/contaminated data. Tier 2 is not interpreted unless Tier 1 implementation validation passes. |
| L9 | Synthetic truth can leak into tuning | RESOLVED IN PROTOCOL/CODE | P0.2A selects lambda by held-out healthy Gaussian risk. Truth is used only for reporting error. Any oracle sweep is labeled upper-bound and cannot advance to real data. |
| L10 | Support recovery is not the same as RACE geometry | RESOLVED IN PROTOCOL | Primary estimator metrics are relative Frobenius error and held-out healthy risk; support F1/Jaccard are secondary. P1 remains the only anomaly-performance test. |
| L11 | Better precision does not solve 1% FPR calibration | IRREDUCIBLE / SEPARATE AXIS | Preserve the existing frozen calibration audit. Precision-transfer success is necessary evidence only; N* is decided exclusively in P1 under the original calibration protocol. |
| L12 | Robot executions may not be iid/exchangeable | MITIGATED BY EXPERIMENT | Before P0.4, audit episode order/batch effects. If dependence is detected, add blocked/time-ordered sensitivity and block/bootstrap inference instead of randomizing it away. |
| L13 | p=564 CLIME sweeps can be expensive | RESOLVED IN PROTOCOL | Sequential dimension gates, resume/checkpoint files, covariance/data caching, and parallelism across replications rather than nested BLAS oversubscription. |
| L14 | Applying Trans-CLIME directly may erase RACE novelty | RESOLVED IN CLAIMS | Trans-CLIME, JGL and Trans-Glasso are baselines/foundations. RACE must retain a distinct commissioning-specific contribution (source gating/weighting, target-size-dependent transfer, calibration integration, or lightweight robotics-specific adaptation) and must be ablated against these baselines. |
| L15 | Hyperparameter multiplicity / post-hoc tuning | RESOLVED IN PROTOCOL | Small grids and selection criteria are frozen before anomaly evaluation; method escalation follows deterministic gates. Report all attempted predeclared variants, not only the winner. |
| L16 | Multiple synthetic comparisons can create false discoveries | MITIGATED BY EXPERIMENT | Use paired replication-level comparisons, bootstrap confidence intervals, effect sizes, and family-aware interpretation. Primary endpoint/method contrast is predeclared; secondary contrasts are labeled exploratory. |
| L17 | Source-size advantage can be confounded with estimator family | MITIGATED BY EXPERIMENT | Compare TargetOnly and transfer estimators at matched target N and report source-n sensitivity separately. Do not attribute a gain to a new estimator if it is solely caused by increasing source N. |
| L18 | SPD projection can artificially improve held-out likelihood | MITIGATED BY REPORTING | Report raw-estimator truth error, SPD-estimator risk, and projection magnitude together. A method cannot pass solely because a large projection repaired it. |
| L19 | Cross-fitting can become a hidden new method | RESOLVED IN LABELING | Reference Trans-CLIME and CrossfitTransCLIME have separate method names, outputs and manuscript labels. The extension requires its own ablation. |
| L20 | Real-source similarity selection could leak anomaly outcomes | RESOLVED IN PROTOCOL | Source selection/gating uses healthy data and metadata only. Anomaly labels/scores are unavailable until the estimator and representation are frozen for P1. |

## Current implementation state

Implemented now:

1. P0.1 conditioning/redundancy audit.
2. Reference-style CLIME constrained optimization in Python using SciPy HiGHS.
3. Reference-style Trans-CLIME three-stage update with explicit held-out target
   aggregation.
4. Separately labeled cross-fitted Trans-CLIME COLDSTART extension.
5. Raw/symmetric/SPD outputs and projection-magnitude diagnostics.
6. Leakage-safe robust scaling API with target-only, source-frozen and
   source->target shrink modes.
7. P0.2A synthetic implementation-validation runner with known truth, related
   and unrelated sources, held-out healthy tuning, deterministic seeds,
   checkpoint/resume and manifest metadata.
8. Unit tests for LP feasibility, SPD projection, determinism and scaler
   leakage.

Not yet allowed to be claimed as solved:

- p=564,N=10 statistical identifiability;
- Gaussianity of real robot features;
- source homogeneity;
- episode independence/exchangeability;
- final 1% FPR commissioning calibration;
- RACE publication novelty.

These are empirical or conceptual questions and require the staged gates below.

## Hard gates

### Gate A -- reference implementation sanity

Run unit tests and P0.2A at p=10,20.  Do not proceed if LP success is poor,
constraint violations are non-negligible, SPD projection is routinely large, or
related-source transfer cannot outperform TargetCLIME in any controlled setting.

### Gate B -- canonical high-dimensional synthetic

Proceed p=40 -> 128.  Primary comparison: relative Frobenius error using the
**deployably tuned** estimator. Related-source benefit must be reproducible;
unrelated-source regret must remain limited.

### Gate C -- robotics-stress synthetic

Add low effective rank, feature blocks, source/target scale shift, monotone
non-Gaussian transforms and mild contamination.  If ordinary Trans-CLIME loses
robustness, evaluate predeclared Trans-Copula-CLIME rather than inventing a
result-driven robust estimator.

### Gate D -- p=256 / p=564 stress

Only methods passing B and C escalate.  Failure at p=564,N=10 is a reportable
boundary, not a reason to tune until success.

### Gate E -- real healthy P0.4

Freeze estimator, representation, scaler and source-selection rule using only
synthetic/healthy criteria.  Check source homogeneity and episode-order effects.
Use negative controls.  No anomalies.

### Gate F -- P1 commissioning

Only after E.  Restore the original deployment endpoint: Recall >= .90 and FPR
<= .01 with the frozen calibration procedure and CI-based N*.  P0 improvement
does not imply P1 success.

## Reviewer-proof claim boundary

At the current stage the strongest defensible statement is:

> The original independent support-overlap gate was statistically
> underidentified in the full-dimensional low-N regime.  We therefore validate
> transfer-aware precision estimators against synthetic truth and healthy-only
> controls before permitting any anomaly-performance claim.

Do not state that source-target precision transfer works on the robot until
Gates A-E pass, and do not state that commissioning sample complexity is reduced
until Gate F passes.
