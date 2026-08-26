# COLDSTART Final Manuscript Structure

Date: 2026-08-25

This document fixes the recommended structure for the final COLDSTART manuscript. It is governed by `reports/final_paper_claim_to_evidence_matrix.md`. The paper must be written as an operational commissioning and evidence-allocation study, not as a positive RACE sample-complexity paper.

## Working title

**COLDSTART: Operational Commissioning of Robotic Anomaly Detectors Under Recall and False-Alert Constraints**

Alternative shorter title if needed:

**COLDSTART: Commissioning Robotic Anomaly Detectors Under Strict False-Alert Constraints**

## One-sentence thesis

COLDSTART studies how representation quality, adaptation, calibration, and post-freeze certification jointly determine how much healthy robot data are needed before an anomaly detector can be trusted at a specified recall and false-alert operating point.

## Abstract structure

The abstract should contain exactly five moves:

1. **Problem.** Robot anomaly detectors are usually reported with ranking metrics, but deployment requires a detector to satisfy a concrete operating point such as recall >= 90% and FPR <= 1%.
2. **Methodological contribution.** Introduce COLDSTART as a commissioning framework that separates representation-limited, calibration-limited, and certification-limited failure modes.
3. **Finite-sample result.** State that low-FPR calibration and exact post-freeze certification impose discrete and sometimes large healthy-sample requirements; at 95% joint confidence, zero observed false positives requires 368 healthy evaluation executions to certify FPR <= 1% under the frozen binomial model.
4. **Empirical result.** Across voraus-AD and AURSAD, high ranking quality does not guarantee commissioning success; in the frozen PRE_A->PRE_B case, RACE shows statistically supported negative transfer, and neither M2N2 nor the modern calibration-conditional conformal comparator resolves the primary 1% FPR criterion.
5. **Conclusion.** Deployment readiness should be evaluated as an evidence-allocation problem rather than inferred from AUROC or adaptation success alone.

Do not include claims that RACE reduces commissioning data, that transfer is generally harmful, that lambda=60 is optimal, or that the experiments constitute prospective production-robot validation.

## 1. Introduction

### 1.1 Deployment question

Open with the commissioning question:

> When a new robot is installed, how many healthy executions must be observed before an anomaly detector can be trusted to catch at least 90% of faults while producing at most 1% false alerts?

Explain why AUROC/AUPRC alone do not answer this question. Emphasize that a detector can rank anomalies well while still failing after threshold selection or while lacking enough independent post-freeze evidence for certification.

### 1.2 Gap

Frame the gap as the absence of an operational commissioning protocol that distinguishes:

- score/representation quality,
- adaptation/transfer effects,
- finite-sample calibration,
- independent post-freeze certification.

### 1.3 Contributions

Use four contributions only:

1. **Operational commissioning formulation.** Define commissioning success jointly by recall and false-alert constraints and estimate the smallest commissioning size satisfying the criterion.
2. **Failure-mode decomposition.** Separate representation-limited, calibration-limited, and certification-limited outcomes using retrospective oracle geometry, deployed calibration, and exact post-freeze bounds.
3. **Finite-sample evidence analysis.** Derive and empirically demonstrate low-alpha conformal granularity and exact certification sample requirements.
4. **Robotic benchmark study.** Evaluate statistical, transfer/adaptation, and modern conformal baselines on voraus-AD with external validation on AURSAD, including a negative-transfer case study and reviewer-oriented robustness audits.

Do not list RACE as the central positive contribution.

## 2. Related Work

Keep this section compact and organized around the paper's actual thesis.

### 2.1 Robotic and industrial anomaly detection

Cover execution-level anomaly detection, time-series anomaly scoring, and predictive-maintenance settings. Distinguish ranking-oriented evaluation from deployment-threshold evaluation.

### 2.2 Low-data adaptation and test-time adaptation

Motivate transfer from related healthy systems and include M2N2 as a modern adaptation reference. State that the paper tests whether adaptation survives a strict operational criterion rather than assuming adaptation must help.

### 2.3 Conformal anomaly detection and low-FPR calibration

Discuss split conformal, conformal p-values, and calibration-conditional corrections. Connect this literature directly to the finite-sample tail problem at alpha=0.01.

### 2.4 Reliability demonstration and post-freeze certification

Connect the exact binomial confidence-bound calculation to reliability-demonstration testing. Explicitly distinguish calibration from independent certification.

## 3. Problem Formulation: Commissioning Under an Operating Constraint

### 3.1 Statistical unit

The robot execution/cycle is the unit of analysis. Raw 100 Hz rows are not treated as independent observations.

### 3.2 Data partitions

Define four logically distinct pools:

1. source healthy data,
2. target commissioning healthy data,
3. target calibration healthy data,
4. independent target evaluation data containing held-out healthy and anomalous executions.

State the no-overlap requirement by execution ID.

### 3.3 Operating criterion

Primary criterion:

- recall target R0 = 0.90,
- FPR budget B = 0.01,
- joint confidence = 0.95.

Commissioning grid is generic {10,25,50,100,250,500}, subject to dataset capacity; the frozen voraus-AD target split supports only N in {10,25,50,100}, so any N* censoring there is >100 rather than >500.

### 3.4 Exact post-freeze certification

Use one-sided Clopper-Pearson bounds with Bonferroni allocation delta_R = delta_F = 0.025:

- certify recall if L_R >= R0,
- certify FPR if U_F <= B.

State the exact finite-sample counts used later, including 368 healthy executions for FP=0 and 36 anomalous executions for perfect observed recall at the primary criterion.

### 3.5 Commissioning estimator

Define N* as the smallest admissible N satisfying the joint certification requirement; otherwise report right-censoring at the largest supported N.

## 4. COLDSTART Failure-Mode Decomposition

### 4.1 Representation-limited

Retrospective empirical oracle recall at FPR <= B is below R0. The score geometry itself does not support the observed operating point.

### 4.2 Calibration-limited

The oracle can support the operating point, but the frozen deployed threshold fails the empirical recall/FPR criterion.

### 4.3 Certification-limited

The frozen detector passes the empirical operating point but exact independent bounds do not yet certify it.

### 4.4 Certified

Both empirical operation and exact post-freeze bounds satisfy the criterion.

Clarify that the oracle is label-dependent and retrospective. It is a diagnostic, not a deployable threshold and not a population guarantee.

## 5. Methods and Baselines

### 5.1 TargetOnly

Ledoit-Wolf Gaussian model with Mahalanobis score using target commissioning healthy data.

### 5.2 RACE transfer case study

Present the frozen RACE method only as the predeclared transfer estimator used to test the transfer hypothesis:

w = N / (N + 60).

Describe mean/covariance shrinkage from source toward target. Do not present lambda=60 as optimal.

### 5.3 Other statistical baselines

Include SourceOnly, Pooled, Euclidean conformal k-NN, PAKCT, and Isolation Forest where used in the frozen benchmark tables.

### 5.4 M2N2

Describe the robotics adaptation of M2N2, q99 as the frozen primary execution-level aggregation, and mean as predeclared sensitivity. Keep online adaptation diagnostic-only because the detector changes during evaluation.

### 5.5 Modern conformal comparator

Describe CAD-Marginal and CAD-CCV-DKWM using the same frozen TargetOnly score. Stress that the comparator isolates calibration effects by holding representation fixed.

## 6. Experimental Protocol

### 6.1 Datasets

- voraus-AD: primary benchmark, PRE_A setting 72 as source and PRE_B setting 73 as target.
- AURSAD: external cross-task validation of commissioning difficulty.

### 6.2 Seeds and resampling

Use 20 commissioning seeds for primary repeated subsampling analyses. Clarify that these quantify sensitivity to commissioning draws conditional on the fixed evaluation partition; they are not 20 independent robot deployments.

### 6.3 Metrics

Report:

- deployed recall,
- deployed FPR,
- AUROC,
- empirical oracle recall at FPR <= 1%,
- empirical success rate,
- certified success rate,
- N* or right-censoring.

### 6.4 Uncertainty

Separate two uncertainty sources:

- commissioning-draw variability: bootstrap across the 20 commissioning seeds,
- post-freeze certification uncertainty: exact one-sided binomial bounds on the independent evaluation executions.

Do not merge these into one confidence interval.

## 7. Results

The Results section should be ordered by the paper thesis, not by experiment number.

### 7.1 Ranking quality is not deployment readiness

Lead with the primary TargetOnly/RACE commissioning curves and the gap between AUROC/oracle geometry and deployed recall/FPR. Introduce the three-way bottleneck classification here.

Primary figure: commissioning curves plus operating constraint.

Primary table: main commissioning benchmark with N*, empirical success, and certification outcome.

### 7.2 Finite-sample calibration and certification create distinct evidence burdens

Show the strict conformal rank granularity at alpha=0.01 and exact certification sample requirements. This subsection should contain the 368-healthy and 36-anomaly counts.

Primary figure: calibration-size phase diagram / conformal regime plot.

Primary table: calibration and certification minimum sample-size table.

### 7.3 Transfer can hurt even when source-target similarity looks plausible

Present E7 paired RACE-vs-TargetOnly negative-transfer results. Include the paired mean delta oracle recall and 95% bootstrap interval for each N, plus exact sign-flip p-values. Mention that all 20 draws favor TargetOnly at N=25,50,100.

Follow immediately with E17 lambda sensitivity: shrinkage strength matters, but no tested lambda reaches the required low-FPR feasibility. Do not retune RACE post hoc.

Primary figure: paired RACE-vs-TargetOnly oracle-recall differences across N.

Supporting figure/table: lambda sensitivity.

### 7.4 Modern adaptation does not trivially solve the operating-point problem

Present E6-R1. Under q99, M2N2 CommissioningAdapt gives no low-FPR oracle-recall benefit. Under mean aggregation, ranking and recall improve consistently but the 1% FPR operating criterion remains unmet.

Primary table: M2N2 q99 primary and mean sensitivity side by side.

### 7.5 Stronger conformal validity can be operationally too conservative

Present E19 with the same TargetOnly score for CAD-Marginal and CAD-CCV-DKWM. Show:

- minimum marginal p-value = 1/101 ~= 0.00990,
- minimum DKWM-adjusted p-value ~= 0.14581,
- at alpha=0.01, marginal conformal can alarm but overshoots the false-alert budget,
- DKWM cannot alarm at all with m=100.

This subsection should explicitly support the distinction between representation and calibration.

### 7.6 Oracle conclusions are themselves finite-sample diagnostics

Present E15 briefly. Emphasize that TargetOnly's observed oracle success can be sample-sensitive at a 1% FPR budget, while RACE remains substantially weaker under the bootstrap perturbation. Keep this as a cautionary robustness result rather than a new deployment guarantee.

### 7.7 External validation on AURSAD

Report that evaluated detectors remain censored beyond N=500 under the joint criterion. Use this only to support cross-task commissioning difficulty, not RACE generalization.

## 8. Robustness and Reviewer-Defense Analyses

This can be a short main-text section or mostly appendix material depending on page limits.

Include:

- E5 operating-point sensitivity,
- E8 temporal-dependence diagnostics,
- E15 oracle bootstrap if not already shown in main text,
- E17 lambda sensitivity details,
- M2 numerical-stability audit,
- E7 compatibility/discrepancy association analysis.

Main-text takeaway: none of these audits overturns the central operational conclusion, but several constrain its interpretation.

## 9. Discussion

### 9.1 Why AUROC can be misleading operationally

A detector can rank faults well while failing because the low-FPR threshold is statistically difficult to estimate.

### 9.2 Evidence allocation across model, calibration, and certification

Argue that commissioning budgets should be allocated explicitly among:

- learning/adapting the score,
- estimating the healthy tail for thresholding,
- collecting independent post-freeze healthy and anomaly evidence for certification.

### 9.3 Negative transfer is a deployment risk, not a universal law

Use the RACE case study to motivate compatibility checks and conservative transfer policies. Do not generalize beyond the tested source-target pair.

### 9.4 Stronger statistical validity can cost operational power

Use E19 to show that a calibration rule can provide stronger conditional protection yet become unusable at a stringent alpha with small m.

### 9.5 Practical commissioning implication

The commissioning bottleneck may move as N increases: early failures may be representation-limited, later failures calibration-limited, and finally certification-limited. This is the operational value of the decomposition.

## 10. Limitations

State explicitly:

- public datasets rather than prospective physical production commissioning,
- fixed source-target pair for the RACE negative-transfer result,
- commissioning seeds are resampled draws from a fixed target pool, not independent deployed robots,
- exchangeability/independence assumptions are not proven by E8,
- AURSAD supports commissioning difficulty, not successful transfer generalization,
- clean-room reproduction must be completed before claiming full reproducibility.

## 11. Conclusion

End on the operational message:

> Deployment readiness cannot be inferred from ranking quality alone. For robotic anomaly detection, the amount of evidence needed to learn a score, calibrate a stringent false-alert threshold, and certify post-freeze performance are distinct quantities. COLDSTART makes these quantities explicit and exposes where commissioning actually fails.

Do not end with a positive RACE claim.

## Appendix plan

Recommended appendix order:

A. Exact certification derivations and boundary cases.

B. Split-conformal rank/granularity derivation.

C. Full detector hyperparameters and data partitions.

D. Full E7 paired inference and lambda sensitivity.

E. M2N2 implementation and 20-seed replication details.

F. Modern conformal comparator derivation and p-value floor.

G. Operating-point sensitivity.

H. Temporal-dependence diagnostics and moving-block-bootstrap sensitivity.

I. Oracle stability bootstrap.

J. AURSAD per-fault descriptive results.

K. Reproducibility manifest, environment lock, artifact hashes, and clean-room rerun record.

## Drafting guardrail

Before any paragraph is accepted into the manuscript, map each empirical statement to an ID in `reports/final_paper_claim_to_evidence_matrix.md`. If a sentence has no supported claim ID, either narrow it, move it to a clearly labeled hypothesis/limitation, or remove it.
