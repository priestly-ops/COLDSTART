# E6 — M2N2 modern adaptation baseline

## Purpose

E6 tests whether a recent unsupervised test-time adaptation method changes the
COLDSTART commissioning conclusion. The baseline is based on Kim, Park & Choo,
**When Model Meets New Normals: Test-Time Adaptation for Unsupervised
Time-Series Anomaly Detection** (AAAI 2024), using the authors' official
`carrtesy/M2N2` repository as the implementation reference.

This is **not claimed to be an official reproduction on voraus-AD**. M2N2 was
introduced for timestep-level continuous TSAD, whereas COLDSTART evaluates
whole robot executions. E6 therefore preserves the M2N2 adaptation mechanism
while mapping it to execution-level evaluation.

## Frozen robotics adaptation

Official M2N2 ingredients retained:

- MLP reconstruction backbone;
- standard scaling from source healthy training data;
- 12-step non-overlapping windows;
- latent dimension 128;
- AdamW source training, lr = 1e-3, 10 epochs;
- EMA detrending;
- SGD test-time learning, lr = 0.005;
- gamma = 0.99999;
- a high-quantile source-normal score threshold (q99.6) for pseudo-normal masks;
- reconstruction loss applied only where the current score is predicted normal.

The q99.6 / ttlr / gamma values follow the official SWaT online MLP script.
Window/batch/training defaults follow the official training config.

### Execution-level score

For each execution, the time series is partitioned into windows. Channel-wise
squared reconstruction error is averaged over channels, producing timestep
anomaly scores. The **primary cycle score is the 99th percentile** of all
resulting timestep scores in that execution. This tail-preserving aggregation
is predeclared so localized robotic faults are not averaged away, and is not
tuned using anomaly labels. A mean-score sensitivity can be run by setting
`M2N2Config(cycle_score_quantile=None)`.

## Dataset/protocol

Primary dataset: voraus-AD.

- source healthy: setting 72 / PRE_A;
- target: setting 73 / PRE_B;
- commissioning N: 10, 25, 50, 100;
- commissioning seeds: 0..19;
- fixed target calibration: 100 healthy executions;
- fixed healthy evaluation: 100 executions;
- anomaly evaluation: the same frozen anomaly population used by E0;
- false-alert budget: 0.01;
- recall target: 0.90;
- simultaneous confidence: 0.95.

Episode IDs remain disjoint through `create_frozen_evaluation_split`.

## Variants

### M2N2-Offline

Source-trained MLP with no target model adaptation. Target healthy calibration
is used only to choose the strict split-conformal execution threshold. This is
the architecture control for M2N2 adaptation.

### M2N2-CommissioningAdapt — primary E6 baseline

Start from the same source-trained MLP. Process target commissioning executions
in ascending episode/sample ID (acquisition-order proxy). For each target batch:

1. update EMA detrending statistics;
2. reconstruct;
3. compute channel-mean timestep squared error;
4. mark locations below the source q99.6 threshold as pseudo-normal;
5. take one SGD step on masked reconstruction loss.

After commissioning, **freeze the model**. Only then score the independent target
calibration and evaluation sets. This is the E6 variant eligible for the same
conformal threshold and exact post-freeze certification machinery as other
COLDSTART detectors.

### M2N2-Online — diagnostic only

For N=100, seeds 0,1,2, start from the commissioning-adapted model and continue
M2N2 adaptation through the evaluation stream without using labels. This is
closer to the original online-test setting.

Because model parameters and normalization statistics change during evaluation,
standard fixed-detector Clopper-Pearson certification is **not claimed** for
this variant. It reports empirical metrics and retrospective oracle geometry
only.

## No-leakage rule

Anomaly labels are never used for source training, target adaptation,
calibration, pseudo-normal selection, or threshold selection. In M2N2-Online,
labels are consulted only after the full label-blind stream has been scored, to
compute evaluation metrics.

## Interpretation

The primary adaptation comparison is:

`M2N2-Offline` vs `M2N2-CommissioningAdapt`

with the same backbone, source training state, target split, calibration rule,
and evaluation populations. This isolates whether the M2N2 adaptation mechanism
improves commissioning efficiency.

Do not describe E6 as an official M2N2 reproduction. Use:

> an M2N2-style robotics adaptation based on the authors' official AAAI 2024
> implementation.

Do not compare certified-success rates for M2N2-Online against frozen detectors,
because exact fixed-detector certification is intentionally undefined there.
