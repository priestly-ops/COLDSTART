# Frozen E9B/E9C Interpretation — AURSAD MVT-Flow

Status: **Frozen evidence note for manuscript drafting**

This note records the interpretation that should be used for E9B/E9C unless new evidence directly contradicts it. It is intentionally narrower than a universal statement about MVT-Flow or AURSAD.

## Experimental role

E9B/E9C are representation controls for COLDSTART. They test whether a validated deep temporal anomaly detector removes the AURSAD representation bottleneck under a healthy-only commissioning protocol.

The detector is TargetOnly-MVTFlow-AURSAD using the existing audited MVT-Flow implementation and 48 measured AURSAD robot signals. The operating point is fixed at recall target R0=0.90 and false-alert budget B=0.01. Calibration uses M_cal=199 independent healthy executions; certification uses n_H=368 independent healthy executions and n_A=625 fixed anomaly executions. E9C extends the healthy commissioning set to N in {500,650,750,850} while preserving the independent calibration and certification partitions.

## Native-benchmark validation

Before interpreting AURSAD failure, the implementation was validated on its native voraus-AD benchmark using the official 70-epoch configuration. The final mean AUROC was 0.943, exceeding the repository sanity threshold of 0.90. The official loss direction was also verified: higher get_loss_per_sample() corresponds to more anomalous samples.

Therefore, the AURSAD result should not be attributed to an obviously broken implementation, reversed score direction, or nonfunctional GPU/training pipeline.

## E9C high-data result

Across 3 seeds per commissioning size:

| N | Mean recall | Mean FPR | Mean AUROC | Mean oracle recall @ 1% FPR | Representation-limited rate |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.0064 | 0.001812 | 0.455999 | 0.012267 | 1.00 |
| 650 | 0.005333 | 0.008152 | 0.454422 | 0.011733 | 1.00 |
| 750 | 0.0048 | 0.006341 | 0.456068 | 0.010667 | 1.00 |
| 850 | 0.0048 | 0.003623 | 0.454899 | 0.010133 | 1.00 |

No oracle, empirical, or certified N* is observed by N=850; all are right-censored at the maximum tested commissioning size.

## Interpretation

The primary manuscript claim is:

> **A validated deep temporal anomaly detector remained severely representation-limited on AURSAD even when trained with up to 850 healthy target executions. Increasing the healthy commissioning set from 500 to 850 produced no systematic improvement in AUROC or extreme-tail oracle recall under the 1% false-alert budget.**

The stronger but still careful interpretation is:

> **Together with successful native-benchmark validation on voraus-AD, the flat high-data AURSAD response is consistent with a domain/representation mismatch rather than merely insufficient low-N training data within the experimentally accessible healthy-data range.**

Do **not** write that domain mismatch has been universally proven. Do **not** write that MVT-Flow cannot model AURSAD under every possible architecture adaptation, preprocessing policy, or training regime.

## Why this is representation-limited rather than calibration-limited

The deployed recall is near zero, but the decisive diagnostic is the oracle threshold analysis. Even the best retrospective threshold satisfying the empirical 1% FPR budget attains only about 1.0–1.2% mean recall across N=500–850, far below the 90% target. Hence the principal failure occurs upstream of threshold calibration.

At some seeds the healthy certification side itself is statistically adequate (for example FP=0 with n_H=368 gives an exact one-sided FPR upper bound below 0.01), yet recall remains near zero. This cleanly separates certification evidence from detector adequacy.

## Fault-class behavior

At N=850, mean deployed recall remains approximately 0.45% for damaged screw, about 1.1% for extra component, and 0% for missing screw. Damaged-thread has only 3 executions and remains descriptive only.

Therefore, the aggregate failure is not explained by a single dominant fault class; the major AURSAD anomaly classes are all poorly detected by this representation at the deployed threshold.

## Relation to TargetOnly-Stat

The statistical TargetOnly control previously achieved substantially stronger AURSAD ranking and extreme-tail oracle recall than MVT-Flow at N=500. The manuscript-safe conclusion is not that simple models are universally better than deep models. It is:

> **Under this AURSAD healthy-only commissioning protocol, the higher-capacity temporal model did not provide a more commissioning-efficient anomaly representation than the statistical baseline.**

This supports the broader COLDSTART message that benchmark strength or model capacity does not by itself guarantee deployability at a strict false-alert operating point.

## Figure interpretation

The diagnostic figure should emphasize three facts:

1. mean AUROC stays approximately 0.455 across N=500–850, i.e. near chance and flat with additional healthy data;
2. mean oracle recall under the 1% FPR budget stays near 1%, versus the 90% operational target;
3. deployed recall remains near zero across the major fault classes at N=850.

The current archived E9C artifacts do **not** contain per-execution healthy/anomaly score vectors, so a literal score histogram/KDE cannot be reconstructed from the frozen CSV outputs without rerunning and saving raw scores. Do not fabricate a score-distribution plot from summary statistics.

## Manuscript-safe summary sentence

> **MVT-Flow passed its native voraus-AD sanity benchmark (mean AUROC 0.943), yet under the AURSAD healthy-only commissioning protocol it remained representation-limited through N=850: AUROC stayed near 0.455 and oracle recall under a 1% false-alert budget stayed near 1%, with no evidence of a late high-data recovery.**
