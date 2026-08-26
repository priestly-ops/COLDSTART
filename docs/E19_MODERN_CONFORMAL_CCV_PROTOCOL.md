# E19 — Modern conformal comparator: CAD with calibration-conditional validity

## Goal

Add a recent industrial conformal anomaly-detection comparator without changing the underlying anomaly-score model. E19 isolates the effect of calibration by applying recent conformal p-value methodology to the frozen TargetOnly Ledoit–Wolf Mahalanobis score.

## Literature basis

Primary application reference:

- O. Kundacina, V. Vincan, G. Gojic, V. Ninkovic, and D. Miskovic, **Conformal Anomaly Detection for Predictive Maintenance in Thermal Power Plants**, IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3546451.

Statistical basis:

- S. Bates, E. Candes, L. Lei, Y. Romano, and M. Sesia, **Testing for outliers with conformal p-values**, Annals of Statistics, 2023.

Kundacina et al. formulate anomaly detection as hypothesis testing with marginal conformal p-values and study calibration-conditional validity adjustments to strengthen false-positive-rate control when many test points share a fixed calibration set. E19 implements the DKWM adjustment because its finite-sample formula is explicit and reproducible.

## Frozen design

Dataset: voraus-AD

- source: PRE_A / setting 72
- target: PRE_B / setting 73
- commissioning sizes: `10, 25, 50, 100`
- commissioning seeds: `0..19`
- calibration size: `100`
- healthy evaluation size: `100`
- anomaly evaluation set: frozen target anomaly evaluation partition
- false-alert budget / p-value significance: `alpha = 0.01`
- recall target: `0.90`
- CCV tolerance: `delta = 0.05`

The execution/cycle is the statistical unit. No 100 Hz rows are treated as independent calibration or evaluation units.

## Base anomaly score

Both conformal variants use exactly the same frozen TargetOnly detector:

1. fit feature preprocessing on the target commissioning healthy executions only;
2. fit the TargetOnly Ledoit–Wolf Gaussian detector;
3. score the independent target calibration and evaluation executions;
4. do not refit the score model using calibration or evaluation labels.

Larger COLDSTART scores indicate greater abnormality.

## Marginal conformal p-values

Kundacina et al. write the conformal formula using conformity scores that decrease with abnormality. E19 uses the equivalent upper-tail form for anomaly scores that increase with abnormality:

`p(x) = (1 + #{s_cal >= s(x)}) / (n_cal + 1)`.

A point is anomalous when `p <= alpha`.

## Calibration-conditional DKWM adjustment

For calibration size `n_cal` and tolerance `delta`, define

`b_i = min(i / n_cal + sqrt(log(2/delta) / (2 n_cal)), 1)`

for `i=1,...,n_cal`, with `b_0=0` and `b_(n_cal+1)=1`.

The adjusted p-value is

`p_ccv = b_ceil((n_cal+1) p_marg)`.

A point is anomalous when `p_ccv <= alpha`.

## Variants

- `CAD-Marginal`: standard marginal conformal p-value comparator.
- `CAD-CCV-DKWM`: DKWM calibration-conditional adjustment applied to the same marginal p-values.

This pairing isolates calibration methodology from representation quality.

## Predeclared calibration-granularity diagnostic

At `n_cal=100`, the minimum marginal conformal p-value is `1/101 ≈ 0.00990099`, so the primary `alpha=0.01` is only just attainable marginally.

The DKWM adjustment is more conservative. E19 reports the minimum attainable adjusted p-value and whether `alpha=0.01` is attainable before any anomaly labels are examined.

If the adjusted minimum p-value exceeds `0.01`, zero alarms are mathematically forced by calibration granularity. This must be interpreted as a calibration-limit result, not as evidence that the underlying anomaly score has no ranking ability.

## Metrics

For every variant, N, and commissioning seed:

- recall
- false-positive rate
- AUROC of the unchanged base anomaly score
- retrospective empirical oracle recall at FPR <= 0.01
- empirical operating-point success
- exact post-freeze certification status
- bottleneck label
- minimum attainable p-value
- alpha-feasibility indicator

## Certification

Both variants are frozen before evaluation. The base score, fixed calibration set, and p-value mapping are all determined before the independent evaluation partition is scored. Therefore the existing exact post-freeze binomial certification audit remains applicable as a separate deployment-certification statement.

The calibration-conditional guarantee from the comparator literature and COLDSTART's independent post-freeze certification are distinct statistical claims and must not be conflated.

## Interpretation constraints

1. Do not call E19 an exact reproduction of the thermal-power-plant experiment. It is a literature-faithful implementation of the CAD marginal p-value construction and DKWM CCV adjustment mapped onto the COLDSTART score model and robot-execution protocol.
2. Do not tune `delta`, `alpha`, the calibration size, or the underlying TargetOnly score model using anomaly labels.
3. Do not replace the frozen COLDSTART primary calibration rule using whichever E19 variant looks best after evaluation.
4. If DKWM produces no alarms because its minimum attainable adjusted p-value exceeds `alpha`, describe the result as finite-calibration conservatism at the strict 1% false-alert operating point.
