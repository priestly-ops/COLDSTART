# E6-R1 — M2N2 20-Seed Replication Protocol

## Purpose

E6-R1 closes the reviewer-sensitive limitation that the existing E6 M2N2 result was based on only three commissioning seeds. The replication keeps the frozen E6 modeling choices and increases commissioning-draw coverage to seeds 0–19 at N=100.

## Frozen design

- Dataset: voraus-AD.
- Source setting: PRE_A / setting 72.
- Target setting: PRE_B / setting 73.
- Commissioning size: N=100 only.
- Commissioning seeds: 0–19.
- Calibration size: 100 target-healthy executions.
- Healthy evaluation size: 100 target-healthy executions.
- Anomaly evaluation set: frozen E6 target anomaly evaluation set.
- Evaluation seed: 42.
- Model seed: 42.
- Recall target: 0.90.
- False-alert budget: 0.01.

## Aggregation policy

Two execution-level score mappings are retained exactly from E6:

1. `q99` — **primary** aggregation, the 99th percentile of channel-mean timestep reconstruction error.
2. `mean` — **predeclared sensitivity** aggregation, the arithmetic mean of channel-mean timestep reconstruction error.

The mean sensitivity must not replace q99 as the primary result based on favorable outcomes.

## Variants

Primary fixed-detector comparison:

- `M2N2-Offline`: source-trained model, target calibration only, frozen during evaluation.
- `M2N2-CommissioningAdapt`: source-trained model adapted using target commissioning healthy executions, then frozen before calibration/evaluation.

`M2N2-Online` is excluded from E6-R1 fixed-detector certification because it changes the model during evaluation. Existing online results remain diagnostic only.

## Required outputs

For each aggregation and variant, report mean, sample standard deviation, and 95% percentile-bootstrap interval over commissioning draws for:

- deployed recall,
- deployed false-positive rate,
- AUROC,
- empirical oracle recall at FPR <= 0.01.

For paired `CommissioningAdapt - Offline` differences on identical commissioning seeds, report:

- mean difference,
- median difference,
- paired 95% percentile-bootstrap interval,
- Wilcoxon signed-rank p-value,
- paired sign-flip/randomization p-value,
- positive / negative / neutral seed counts.

Bootstrap repetitions: 10,000. Sign-flip Monte Carlo fallback: 100,000 repetitions. With 20 nonzero pairs or fewer, the shared paired-inference implementation performs exact enumeration.

## Interpretation guard

The 20 seeds represent sensitivity to different target commissioning draws under a fixed evaluation partition. They are not 20 independent robots, deployments, or independent population-level test sets. Bootstrap intervals therefore summarize commissioning-draw variability conditional on the frozen evaluation design and must not be presented as population-level certification intervals.

## No-selection rule

No architecture, hyperparameter, aggregation, or source-target pairing may be changed after observing E6-R1 outcomes. q99 remains primary even if mean produces stronger results.
