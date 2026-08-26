# Final Results Figure and Table Inventory

Date: 2026-08-25

This inventory defines the paper-facing figures and tables that should be regenerated from frozen artifacts. It is intentionally selective: exploratory plots, smoke-run summaries, and mechanism-only diagnostics do not belong in the main paper unless needed for a specific reviewer defense.

## Main-paper figures

| ID | Figure | Purpose | Primary source artifacts | Required visual content | Claim IDs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | Commissioning outcome decomposition on voraus-AD | Establish the core gap between ranking quality and deployable operating-point success | Primary E0/E1 TargetOnly and RACE outputs; bottleneck labels | Panels over N showing deployed recall, deployed FPR, oracle recall at B=0.01, and/or bottleneck category; mark R0=0.90 and B=0.01 | C1, C2, C5, C17 | MUST REGENERATE |
| F2 | Low-alpha calibration phase diagram | Show finite-sample conformal granularity | E3/E10/M1 calibration outputs and analytical rank formulas | Calibration size m on x-axis; regions for infeasible, maximum-score, sub-maximum threshold at alpha=0.01; optionally overlay sensitivity alpha values | C3, C4 | MUST REGENERATE |
| F3 | RACE negative-transfer paired differences | Show the primary transfer result without hiding seedwise pairing | E7 paired results and paired inference | Delta oracle recall (RACE - TargetOnly) by N, seed points/paired distribution, 95% bootstrap mean CI, horizontal zero line | C6 | MUST REGENERATE |
| F4 | Calibration rule comparison with identical score model | Isolate calibration effects | E19 seed results and summary | CAD-Marginal vs CAD-CCV-DKWM: recall/FPR across N; annotate min p-value 0.00990 vs 0.14581 at m=100 | C10, C11 | MUST REGENERATE |
| F5 | External commissioning result on AURSAD | Show cross-task commissioning difficulty | AURSAD detector comparison and N-star outputs | Joint success/N-star or recall/FPR curves for selected detectors; clearly label >500 censoring | C16 | MUST REGENERATE |

### Optional main-paper figure if space permits

| ID | Figure | Purpose | Source | Claim IDs |
| --- | --- | --- | --- | --- |
| F6 | Evidence bottleneck schematic | Conceptual summary of representation -> calibration -> certification | Derived from protocol, not a data result | C2, C17 |

If F6 is used, it should be a simple conceptual diagram and must not imply causal identifiability beyond the operational taxonomy.

## Main-paper tables

| ID | Table | Purpose | Required rows/columns | Primary evidence | Claim IDs | Status |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Dataset and protocol summary | Make the frozen evaluation design easy to audit | Dataset, task, source/target definition, normal/fault counts, N grid, calibration size, healthy eval size, anomaly eval size, seeds, primary R/B | Split manifests, dataset loaders, E9 artifacts | C17, C20 | MUST BUILD |
| T2 | Primary commissioning benchmark | Main operational comparison | Detector; N or N*; mean recall; mean FPR; AUROC; oracle recall@1%; empirical success rate; certified success rate; right-censoring | Frozen primary voraus outputs | C1, C2, C5, C17 | MUST BUILD |
| T3 | Exact finite-sample evidence requirements | State the core sample-complexity arithmetic | Condition; observed errors; required healthy/anomaly eval count; exact one-sided confidence rule | E4 + statistical appendix | C3, C4 | MUST BUILD |
| T4 | RACE paired negative-transfer inference | Reviewer-facing paired statistics | N; TargetOnly oracle; RACE oracle; mean delta; 95% bootstrap CI; exact sign-flip p; negative/positive/neutral counts | E7 paired inference | C6 | MUST BUILD |
| T5 | Modern baseline comparison | Prevent baseline weakness attack | Baseline/config; aggregation/calibration rule; recall; FPR; AUROC; oracle recall; operating-point status | E6-R1 and E19 | C9, C10, C11 | MUST BUILD |
| T6 | External validation summary | Summarize AURSAD without overclaiming transfer | Detector; max N; recall/FPR or success; N* censoring; fault-class scope | AURSAD comparison outputs | C16 | MUST BUILD |

## Appendix figures

| ID | Figure | Primary evidence | Reviewer question answered | Claim IDs |
| --- | --- | --- | --- | --- |
| AF1 | RACE lambda sensitivity heatmap/curves | E17 summary | “Is negative transfer just lambda=60?” | C7 |
| AF2 | E7 compatibility vs transfer scatter | E7 compatibility association outputs | “Can healthy-only discrepancy predict transfer?” | C8, C18 |
| AF3 | M2N2 q99 vs mean sensitivity | E6-R1 | “Does aggregation hide adaptation benefit?” | C9 |
| AF4 | Oracle stability bootstrap intervals | E15 | “Is the empirical oracle stable?” | C12 |
| AF5 | Operating-point sensitivity grid | E5 | “Is the conclusion specific to R90/B01?” | C13 |
| AF6 | Temporal dependence diagnostics | E8 | “Are execution-level samples obviously serially dependent?” | C15 |
| AF7 | Moving-block bootstrap sensitivity of FPR upper behavior | E8 | “What if weak local dependence exists?” | C15 |
| AF8 | Numerical conditioning/variance-threshold sensitivity | M2 | “Is the result a bad covariance matrix?” | C14 |
| AF9 | AURSAD per-fault descriptive behavior | E9/AURSAD outputs | “Are results driven by one fault class?” | C16 |

## Appendix tables

| ID | Table | Evidence | Purpose |
| --- | --- | --- | --- |
| AT1 | Full operating-point sensitivity | E5 | All predeclared R/B endpoints and N* outcomes |
| AT2 | Full lambda sensitivity | E17 | All N x lambda summaries and paired deltas vs 60 |
| AT3 | M2N2 20-seed inference | E6-R1 | q99 primary and mean sensitivity with uncertainty |
| AT4 | Oracle bootstrap summary | E15 | Observed oracle, bootstrap mean, interval class, feasibility fraction |
| AT5 | Temporal dependence statistics | E8 | Lag-1, Ljung-Box, ESS, MBB summaries |
| AT6 | Numerical stability audit | M2 | Conditioning and variance-filter sensitivity |
| AT7 | AURSAD per-fault counts and outcomes | AURSAD artifacts | Descriptive fault-specific context, with damaged-thread n=3 flagged as descriptive only |
| AT8 | Full hyperparameter/configuration table | manifests and source configs | Reproducibility |
| AT9 | Data partition audit | split manifests | Verify no episode overlap |
| AT10 | Artifact provenance and hashes | final freeze manifest | Submission reproducibility record |

## Numbers that must appear consistently

The following numbers are high-risk for inconsistency and should be generated from one canonical source when the final tables are built:

- Primary operating point: Recall >= 0.90, FPR <= 0.01.
- Joint confidence: 0.95, with one-sided Bonferroni allocation delta_R = delta_F = 0.025.
- voraus commissioning grid: N = {10,25,50,100}; censoring is >100.
- Generic/AURSAD grid where supported: {10,25,50,100,250,500}; AURSAD censoring may be >500.
- Calibration size in the frozen primary analyses: 100.
- Zero-FP healthy certification requirement at B=0.01: 368 executions.
- Perfect-recall anomaly certification requirement at R0=0.90: 36 executions.
- Split conformal alpha=0.01: infeasible for m<=98, maximum-score for m=99..198, first sub-maximum threshold at m=199.
- RACE frozen lambda: 60.
- E19 minimum marginal p-value at m=100: 1/101 ~= 0.00990099.
- E19 minimum DKWM-adjusted p-value at m=100, delta=0.05: ~= 0.14581015.

## Figure/table drafting rules

1. Every main-text result must map to one or more supported claim IDs from `reports/final_paper_claim_to_evidence_matrix.md`.
2. Do not use exploratory smoke-run artifacts when a frozen 20-seed result exists.
3. Never label bootstrap-over-seeds intervals as deployment confidence intervals.
4. Never present empirical oracle thresholds as deployable thresholds.
5. Use execution/cycle counts, not 100 Hz row counts, in all sample-size labels.
6. Preserve right-censoring explicitly (`>100` on voraus, `>500` on AURSAD where applicable); do not silently substitute a generic maximum N.
7. RACE lambda sensitivity must be labeled robustness/sensitivity, not tuning.
8. E19 must say the representation is held fixed between calibration methods.
9. AURSAD must be described as external validation of commissioning difficulty, not transfer-success generalization.
10. Every final figure/table should be regenerated from frozen CSV/JSON artifacts by script, then hashed and visually inspected.

## Recommended main-paper order

A compact paper can usually fit the following sequence without becoming an experiment catalog:

1. **F6** conceptual bottleneck schematic (optional but useful early).
2. **F1** primary commissioning/decomposition result.
3. **T3** exact finite-sample calibration/certification requirements.
4. **F3 + T4** RACE paired negative-transfer result.
5. **T5** M2N2 + modern conformal comparator.
6. **F4** calibration wrapper contrast.
7. **F5/T6** AURSAD external validation.

Move lambda sensitivity, oracle bootstrap, operating-point sensitivity, temporal-dependence diagnostics, and numerical-stability evidence to the appendix unless page space is generous.

## Publication-readiness gate

A figure/table becomes `FINAL` only after all of the following are true:

- source artifact is from a frozen supported experiment,
- generating script is committed,
- generated file hash is recorded,
- labels and units are reviewed,
- right-censoring is correct,
- uncertainty language is correct,
- no claim exceeds the evidence matrix,
- visual QA passes at single-column and double-column sizes,
- the same number appears identically in the manuscript text, table, and appendix.

Until that gate is passed, mark the item `DRAFT` rather than `FINAL`.
