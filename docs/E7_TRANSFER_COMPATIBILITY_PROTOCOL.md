# E7 — Transfer compatibility / negative-transfer audit

## Question

When does source transfer help or hurt RACE relative to TargetOnly, and can that
behavior be explained by source-target compatibility measured from healthy data
alone?

This is an audit, not a new detector and not a tuned transfer gate.

## Motivation

Negative transfer is the case where using source knowledge reduces target-domain
performance. Prior transfer-learning work treats source-target discrepancy and
domain-similarity estimation as central diagnostics for negative-transfer risk.
E7 therefore measures compatibility without anomaly labels, then compares those
measurements with retrospective paired RACE-vs-TargetOnly transfer outcomes.

## Frozen protocol

Dataset: voraus-AD.

- source healthy: setting 72 / PRE_A;
- target: setting 73 / PRE_B;
- commissioning N: 10, 25, 50, 100;
- commissioning seeds: 0..19;
- fixed target calibration: 100 healthy executions;
- fixed healthy evaluation: 100 executions;
- anomaly evaluation: same frozen anomaly population used by E0;
- false-alert budget: 0.01;
- recall target: 0.90.

All detector comparisons are paired on the same commissioning seed and frozen
calibration/evaluation sets.

## Healthy-only compatibility metrics

All compatibility metrics are computed from source healthy statistical features
and the target healthy commissioning sample only.

1. **Mean-shift RMS** — RMS difference between target and source means after
   source standardization.
2. **Covariance/correlation discrepancy** — normalized Frobenius distance
   between Ledoit-Wolf shrinkage correlation matrices.
3. **RBF MMD^2** — balanced source/target MMD with median-distance bandwidth.
4. **Domain-classifier AUC** — cross-validated logistic AUC for discriminating
   balanced source and target healthy samples. The label orientation is folded
   so the reported value is at least 0.5; 0.5 means indistinguishable and 1.0
   means maximally separable.

Larger values always mean greater discrepancy / lower compatibility.

## Primary transfer outcome

Primary outcome:

`delta_oracle_recall_at_fpr_budget = RACE oracle recall@1% FPR - TargetOnly oracle recall@1% FPR`

This is preferred over deployed recall because it asks whether source transfer
changes the score geometry at the operational tail without conflating the result
with calibration threshold error.

Secondary paired outcomes:

- delta AUROC;
- delta deployed recall;
- delta deployed FPR.

Negative transfer is defined operationally as:

`delta_oracle_recall_at_fpr_budget < 0`.

Positive transfer uses `> 0`; exact ties are neutral.

## Association analysis

For each compatibility metric and transfer outcome:

- report Spearman correlation separately at each N;
- report a pooled within-N rank correlation, so between-N sample-size effects do
  not masquerade as compatibility effects;
- obtain the pooled p-value from 5,000 permutations that shuffle outcomes only
  within the same N group.

These are exploratory explanatory associations, not a learned transfer-gating
rule and not a causal claim.

## Interpretation rules

A consistent negative association between discrepancy and RACE benefit supports
"compatibility-limited transfer": more mismatched healthy source/target domains
are associated with weaker or negative transfer.

No association means E7 does not support using these healthy-only discrepancies
as an explanation or pre-transfer gate.

A positive association would contradict the simple similarity hypothesis and
must be reported rather than tuned away.

Do not select thresholds on these compatibility metrics after seeing anomaly
performance. Any future safe-transfer gate would require a separately frozen
validation protocol.
