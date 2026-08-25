# E10 — Calibration robustness on frozen TargetOnly scores

## Question

Does the calibration-limited diagnosis on voraus-AD survive alternatives to the frozen deterministic split-conformal threshold?

E10 changes **only the calibration rule**. It does not change the detector, preprocessing, commissioning split, healthy evaluation set, anomaly evaluation set, operating point, or score representation.

## Frozen operating point

- Recall target: `R0 = 0.90`
- False-alert budget: `B = 0.01`
- Joint confidence: `gamma = 0.95`
- Detector: `TargetOnly`
- Dataset: voraus-AD target setting 73
- Commissioning sizes: `N = {25, 50, 100}`
- Commissioning seeds: `0..19`
- Frozen calibration pool: 100 healthy target executions
- Frozen healthy evaluation: 100 healthy target executions
- Anomaly evaluation: all frozen anomalous executions from the existing E0 protocol

## Calibration prefixes

A fixed random permutation of the E0 100-cycle calibration pool is generated with seed `271828`. The nested prefixes are:

- `M = 50`
- `M = 99`
- `M = 100`

At `alpha = 0.01`, these deliberately probe the deterministic conformal granularity regimes:

- `M=50`: infinite threshold under the strict deterministic rule
- `M=99`: maximum-score threshold
- `M=100`: maximum-score threshold and exact reproduction of frozen E0

The experiment intentionally does **not** force `M=198/199` into this same allocation because that would change the frozen healthy-evaluation or commissioning allocation on voraus-AD.

## Calibration methods

### 1. Deterministic split conformal — primary reference

The frozen strict order-statistic calibration used throughout the paper. Alarm iff `score > threshold`.

### 2. Randomized/smoothed split conformal — validity stress test

For test score `s` and `m` calibration scores, define

`p = (# {cal > s} + U * (# {cal = s} + 1)) / (m + 1)`, with `U ~ Uniform(0,1)` independent.

Alarm iff `p <= alpha`.

One reproducible deployment realization is generated using a predeclared seed derived only from `N`, commissioning seed, `M`, and method index. Expected recall/FPR over the auxiliary randomization are also reported. No anomaly label influences the random numbers.

This method is included specifically to test whether deterministic conformal discreteness is responsible for the observed calibration bottleneck.

### 3. EVT/POT-GPD — parametric tail-model stress test

A generalized Pareto distribution is fit to the upper calibration tail using a predeclared tail size:

`max(10, ceil(0.20 * M))` exceedances.

The GPD threshold extrapolates to estimated tail probability `0.01`.

This is a **parametric robustness diagnostic**. It does not inherit the finite-sample distribution-free validity of conformal calibration. Fit failures are recorded explicitly as `calibration_fit_failed`; there is no silent fallback to another threshold.

## Frozen-score safeguard

TargetOnly is fit once per `(N, commissioning seed)`. Calibration, healthy-evaluation, and anomaly scores are cached and reused by all methods and calibration-prefix sizes.

For `M=100`, deterministic conformal must exactly reproduce frozen E0 TargetOnly TP/FN/FP/TN, recall, and FPR. The runner aborts on any mismatch.

## Reported quantities

For every `(N, seed, M, method)`:

- threshold where applicable
- TP/FN/FP/TN
- recall and FPR
- exact one-sided Clopper-Pearson recall lower and FPR upper bounds with Bonferroni simultaneous 95% confidence
- empirical success
- certified success
- AUROC
- oracle recall at 1% FPR (diagnostic only)
- bottleneck class
- for randomized conformal: expected recall and FPR over auxiliary randomization
- for EVT/POT: fitted threshold and GPD diagnostics / explicit fit failure

## Interpretation rules

E10 is not a threshold-search exercise and must not select a method after viewing anomaly results.

- If randomized conformal materially improves joint operation, deterministic grid granularity is part of the calibration bottleneck.
- If EVT/POT improves operation but randomized conformal does not, parametric tail modeling may improve efficiency, but the gain comes with stronger assumptions.
- If neither alternative materially restores Recall >= 0.90 and FPR <= 0.01 when the oracle is feasible, the calibration-limited conclusion is not specific to the frozen deterministic order statistic.
- Certification remains structurally impossible on the frozen 100 healthy evaluation executions at B=0.01 even with FP=0; E10 therefore primarily addresses empirical calibration robustness, not the E4 certification-evidence boundary.
