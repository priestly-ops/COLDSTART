# E4 — Certification Sample-Size Experiment

## Question

After a detector and its deployment threshold are frozen, how much **independent evaluation evidence** is required to certify the paper's operating point?

Primary operating specification:

- recall target: 0.90
- healthy false-positive-rate budget: 0.01
- simultaneous confidence: 0.95
- Bonferroni allocation: delta_recall = delta_fpr = 0.025

E4 is deliberately separate from model fitting and threshold calibration. It measures certification evidence, not commissioning/model-training evidence.

## Exact certification rule

Healthy evaluation episodes are Bernoulli trials with FP count X_F. The one-sided exact Clopper-Pearson upper bound is used for FPR.

Anomaly evaluation episodes are Bernoulli trials with TP/FN counts. The one-sided exact Clopper-Pearson lower bound is used for recall.

The operating point is certified only when both

- recall lower bound >= 0.90, and
- FPR upper bound <= 0.01.

## Primary structural experiment

For healthy evidence, compute exact FPR upper-bound curves over increasing n_H for observed FP counts 0, 1, 2, and 3, and solve for the smallest n_H that meets the 1% budget.

For anomaly evidence, compute exact recall lower-bound curves over increasing n_A for observed FN counts 0, 1, 2, and 3, and solve for the smallest n_A that meets the 90% target.

Expected key boundaries under the frozen simultaneous-confidence rule:

- FP=0: n_H = 368
- FP=1: n_H = 555
- FP=2: n_H = 720
- FN=0: n_A = 36
- FN=1: n_A = 54
- FN=2: n_A = 70

These are exact finite-sample evidence requirements under the specified binomial interpretation; they are not universal constants and change with the operating target and confidence allocation.

## Dataset-feasibility audit

voraus-AD contains 319 target-healthy episodes under the frozen target setting. Therefore the dataset cannot supply the 368 independent healthy certification episodes required even in the best-case zero-FP scenario.

Under the E3 primary allocation (N=100 commissioning, Mcal=119), only 100 target-healthy episodes remain for independent healthy evaluation.

This shortage must not be 'fixed' by bootstrap duplication or repeated reuse of the same healthy episodes. Resampled copies are not new independent certification evidence.

The anomaly side is much less restrictive globally because 755 anomaly evaluation episodes are available, though very small fault subtypes must still be treated separately and descriptively where appropriate.

## Interpretation

A failure to certify with n_H=100 does not imply that the detector's true FPR exceeds 1%. It means the available independent evaluation evidence is insufficient to establish FPR <= 1% at the predeclared confidence level.

E4 therefore completes the third stage of the COLDSTART decomposition:

1. score/representation adequacy,
2. threshold calibration adequacy,
3. finite-sample certification adequacy.
