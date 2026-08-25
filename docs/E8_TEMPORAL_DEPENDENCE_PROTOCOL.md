# E8: Temporal Dependence / Exchangeability Audit

## Question

Do nearby robot executions exhibit enough sample-order dependence to materially challenge the exchangeability/independence assumptions used by split-conformal calibration and exact binomial certification?

E8 is an **assumption audit**, not a detector benchmark and not a replacement certification method.

## Frozen dataset and ordering

- Dataset: voraus-AD, 100 Hz.
- Target domain: PRE_B / setting 73.
- Healthy execution is the unit of analysis.
- Ordering: ascending `sample` / episode ID.
- Important limitation: sample ID is treated as an acquisition-order **proxy**. We do not claim the repository documents it as a wall-clock timestamp.

## Two complementary audits

### A. Feature-level dependence

Use all 319 target-healthy executions, ordered by episode ID.

1. Extract the frozen statistical features used by TargetOnly.
2. Standardize non-constant features for this diagnostic only.
3. Report the distribution of per-feature lag-1 autocorrelations.
4. Fit PCA(1) only as a scalar diagnostic summary and report:
   - lag-1 autocorrelation,
   - ACF through lag 20,
   - Ljung-Box statistic/p-value,
   - initial-positive-sequence effective sample size (ESS),
   - permutation p-value for absolute lag-1 autocorrelation.

This block asks whether the healthy robot data themselves exhibit ordered structure. It does not use anomaly labels and does not alter the detector.

### B. Score / alarm dependence under the frozen TargetOnly protocol

For each commissioning seed 0..19:

- N = 100 TargetOnly healthy commissioning executions.
- Calibration = frozen 100 healthy executions.
- Healthy evaluation = frozen 100 healthy executions.
- FPR budget = 0.01.
- Detector/preprocessor fitted only on the commissioning set.
- Calibration/evaluation episodes remain exactly those of the frozen E0 split.

For the fixed 200 calibration + healthy-evaluation episodes, ordered by episode ID, report:

- lag-1 score autocorrelation,
- maximum absolute ACF through lag 20,
- Ljung-Box p-value,
- ESS and ESS fraction,
- permutation p-value for lag-1 dependence.

For the 100 healthy evaluation alarm indicators at the frozen deterministic conformal threshold, report:

- empirical FPR,
- exact IID Clopper-Pearson upper bound used by the paper,
- alarm-sequence dependence when non-degenerate,
- circular moving-block-bootstrap FPR intervals for block sizes 1, 2, 5, 10, 20.

The moving-block bootstrap is **sensitivity analysis only**. It does not inherit the paper's exact finite-sample binomial guarantee.

## Interpretation guardrails

E8 must not claim that a non-significant dependence test proves independence or exchangeability.

Use wording such as:

- `No material dependence was detected by the predeclared diagnostics.`
- `The observed dependence sensitivity was modest/large relative to the IID analysis.`
- `Results are robust/not robust to block-preserving resampling under the sample-ID ordering proxy.`

If strong dependence is found, the manuscript must narrow the exact IID/exchangeability guarantees and report the block sensitivity prominently.

If weak dependence is found, the manuscript may state that the available diagnostics did not reveal material sample-order dependence, while retaining the ordering-proxy limitation.

## Literature motivation

Standard conformal finite-sample validity is based on exchangeability. Time-series dependence can violate that assumption, motivating block/randomization or mixing-process analyses. E8 therefore audits dependence empirically rather than silently assuming episode-level IID behavior.

## Outputs

`outputs/e8_temporal_dependence/`

- `e8_seed_results.csv`
- `e8_block_bootstrap.csv`
- `e8_feature_dependence.csv`
- `e8_summary.csv`
- `e8_manifest.json`
