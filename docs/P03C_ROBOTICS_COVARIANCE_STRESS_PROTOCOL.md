# P0.3c Robotics-Shaped Covariance Stress Protocol

## Purpose

P0.3c is a stress-validation stage for the already-frozen RACE-Cov Safe-CV mechanism. It is not a new estimator search. The question is whether the P0.3b source-specific transfer result survives a harder healthy-data geometry before any real voraus-AD evaluation.

## Frozen production policy

- If target commissioning `N < 25`, do not transfer; use the target-only covariance selector.
- If `N >= 25`, use `RACECovSafeCV` with the P0.3b lambda grid and one-standard-error-style paired healthy-CV rule.
- No anomaly labels or synthetic truth may enter fitting or source-weight selection.

The `N=25` activation boundary is frozen from P0.3b scale-up because the Safe-CV selector was unsafe at `N=10` but passed the predeclared adversarial negative-transfer gate at `N>=25` across `p=20,40`.

## Why this stress model is robotics-shaped

The frozen robot cycle feature extractor emits six statistics per signal in signal-major order: mean, standard deviation, median, q25, q75, and total variation. The P0.3c covariance generator mirrors that structure with six-feature signal blocks, strong within-signal dependence, subsystem latent factors shared across signals, heterogeneous feature scales, and idiosyncratic noise.

Healthy observations are sampled from a multivariate Student-t distribution with finite covariance (`df=8`) rather than a pure Gaussian. This deliberately challenges Gaussian-risk selection without changing the true covariance target.

## Dimensions and sample sizes

Primary stress grid:

- `p = {128, 256}`
- `N = {25, 50}`
- source healthy samples = 400
- replications = 30
- independent healthy evaluation samples = 500

The purpose is to test `p >> N` before moving to the full 564-dimensional real feature space.

## Source regimes

1. `identical`: exact target covariance.
2. `mild`: small signal-level scale drift with preserved correlation structure.
3. `moderate`: attenuated cross-subsystem correlations while keeping marginal variances plausible.
4. `block_mismatch`: signal-block permutation, preserving locally plausible six-statistic blocks but assigning them to the wrong signal identity.
5. `adversarial`: sign-flipped cross-signal dependence via an SPD-preserving diagonal sign transform.

## Baselines and method

- `BestTargetOnlySafeCV`: Ledoit-Wolf versus ridge, selected by the same target-only K-fold healthy risk budget.
- `RACECov60Full`: fixed original RACE-style covariance interpolation, included as an unsafe-transfer reference.
- `RACECovSafeCV`: frozen conservative source-borrowing mechanism.

## Metrics

Primary metric:

- covariance relative Frobenius error versus synthetic truth.

Secondary diagnostics:

- precision relative Frobenius error,
- held-out healthy Gaussian risk,
- covariance condition number,
- correlation of estimated versus oracle Mahalanobis scores,
- normalized Mahalanobis median absolute error,
- selected source weight,
- source acceptance frequency,
- source-similarity versus transfer-gain correlation.

## Frozen gates

The P0.3b gates are reused without relaxation:

- identical-source 95% bootstrap lower CI for gain versus best target-only > 0,
- mild-source 95% bootstrap lower CI > 0,
- moderate-source median gain >= 0,
- adversarial meaningful-negative-transfer fraction <= 0.20,
- identical-source median gain >= 0.15,
- source-similarity correlation > 0.5 is a supporting diagnostic.

Meaningful negative transfer remains defined as relative covariance-error gain `< -0.10` versus `BestTargetOnlySafeCV`.

No threshold will be changed after viewing P0.3c results.

## Advancement rule

If both `p=128` and `p=256` pass at `N=25` and `N=50`, advance directly to real healthy voraus-AD source-target transfer. If failures are isolated, diagnose them but do not redesign the estimator unless there is a reproducible implementation or protocol defect. If the method broadly fails, stop before anomaly-label evaluation and report the synthetic stress limitation.

## Literature motivation

The protocol is consistent with prior work showing that covariance shrinkage helps when dimensionality is large relative to sample size, while transfer across heterogeneous populations can introduce bias or negative transfer. Recent work on covariance-aware multi-source shrinkage likewise emphasizes related-source gains and heterogeneity risk. Robotics anomaly-detection transfer work has also used source-similarity weighting to mitigate negative transfer under limited target data. P0.3c uses those ideas only as motivation; its implementation and pass/fail criteria are frozen from our own preceding stages rather than tuned to those papers.
