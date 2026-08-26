# Final Paper Claim-to-Evidence Matrix

Date: 2026-08-25

This is the reviewer-facing evidence ledger for the final COLDSTART manuscript. It supersedes older exploratory claim ledgers for manuscript drafting purposes but does not delete or rewrite historical experiment records.

The governing rule is conservative: **a manuscript claim is allowed only at the strength supported by the frozen experiment and its counter-audits.** Diagnostic/oracle analyses are not deployment guarantees; sensitivity analyses are not post-hoc tuning; fixed commissioning seeds are not independent robot deployments.

## Claim matrix

| ID | Manuscript-level claim | Primary evidence | Counter-test / audit | Status | Allowed wording | Do not claim | Suggested placement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Operational commissioning should be evaluated jointly in terms of anomaly recall and false-alert rate rather than ranking metrics alone. | `outputs/e0_*` primary TargetOnly/RACE commissioning results; `outputs/e1_*` oracle bottleneck analysis; `outputs/e19_modern_conformal_ccv/e19_summary.csv` | Compare AUROC/oracle recall against deployed recall/FPR and exact certification | **Supported** | “High ranking quality does not by itself establish deployability at a specified recall/FPR operating point.” | “AUROC is useless.” | Abstract, Introduction, Results |
| C2 | COLDSTART separates three distinct failure modes: representation-limited, calibration-limited, and certification-limited. | `src/bottleneck_decomposition.py`; E1/E3/E4/E19 outputs | Reciprocal oracle checks; deployed-vs-oracle comparison; exact post-freeze binomial bounds | **Supported** | “The protocol distinguishes whether failure originates in score geometry, calibration, or finite-sample certification.” | “These categories are causal mechanisms in every deployment.” | Methods, Results |
| C3 | Exact low-FPR certification is sample-hungry even after a detector is frozen. | `docs/E4_CERTIFICATION_SAMPLE_SIZE_PROTOCOL.md`; `docs/manuscript/COLDSTART_STATISTICAL_CERTIFICATION_APPENDIX.md`; E4 outputs | Exact one-sided Clopper–Pearson calculations with Bonferroni split of joint error | **Supported** | “At 95% joint confidence, zero observed false positives requires 368 healthy evaluation executions to certify FPR <= 1%, while perfect anomaly detection requires 36 anomaly executions to certify recall >= 90% under the frozen binomial model.” | “368 samples always guarantee 1% FPR in any dependent deployment.” | Methods, Appendix, Discussion |
| C4 | Strict conformal calibration has discrete low-alpha granularity that can make small calibration sets operationally unusable. | `docs/E3_CALIBRATION_PHASE_PROTOCOL.md`; `docs/E10_CALIBRATION_ROBUSTNESS_PROTOCOL.md`; M1/E3/E10 outputs; E19 manifest | Exact split-conformal rank derivation and calibration-size sensitivity | **Supported** | “For alpha=0.01, strict split conformal is infeasible for m<=98, uses the maximum calibration score for m=99..198, and first permits a sub-maximum threshold at m=199.” | “Conformal prediction fails in general.” | Methods, Results, Appendix |
| C5 | On voraus-AD PRE_B, TargetOnly can have excellent score ranking while still missing the joint operating requirement after calibration. | Primary TargetOnly outputs; E1; `outputs/e15_oracle_stability/e15_summary.csv`; `outputs/e19_modern_conformal_ccv/e19_summary.csv` | Oracle feasibility, calibration wrapper comparison, bootstrap stability audit | **Supported** | “At larger N, TargetOnly score geometry can support very high empirical oracle recall, yet the deployed calibrated detector can still violate the 1% false-alert budget.” | “TargetOnly is a poor detector.” | Results |
| C6 | Under the frozen PRE_A->PRE_B transfer setting, RACE exhibits negative transfer relative to TargetOnly in low-FPR empirical oracle recall. | `outputs/e7_transfer_compatibility/e7_paired_results.csv`; `outputs/e7_transfer_compatibility/e7_paired_inference.csv`; E7 manifest | Paired bootstrap, exact sign-flip test, Wilcoxon; identical commissioning seeds | **Strongly supported for this source-target pair** | “RACE shows statistically supported negative transfer in the frozen PRE_A->PRE_B setting; at N=25,50,100 all 20 paired commissioning draws favor TargetOnly in empirical oracle recall at FPR<=1%.” | “Transfer is harmful in robotics generally.” | Results, Discussion |
| C7 | The E7 negative-transfer conclusion is not explained solely by the original choice lambda=60. | `outputs/e17_race_lambda_sensitivity/e17_summary.csv`; `outputs/e17_race_lambda_sensitivity/e17_paired_inference_vs_lambda60.csv`; E17 manifest | Lambda sweep {0,10,30,60,100,300,inf} over all N and 20 seeds | **Supported as robustness analysis** | “Shrinkage strength materially changes performance, but no tested lambda attains the required low-FPR feasibility, empirical success, or certification.” | “lambda=60 is optimal.” / “lambda=300 should replace the frozen method.” | Results, Appendix |
| C8 | Healthy-only source-target discrepancy is detectable, but the tested discrepancy measures do not reliably predict seedwise low-FPR transfer benefit. | E7 compatibility metrics and association outputs | Per-N Spearman tests and pooled within-N permutation association | **Supported, narrowly** | “Source-target discrepancy is clearly detectable, but the tested healthy-only metrics are not reliable seedwise predictors of the primary transfer endpoint.” | “Compatibility can never predict transfer.” | Results, Discussion |
| C9 | A modern adaptation baseline (M2N2) does not rescue the primary 1% FPR commissioning criterion in this setting. | `outputs/e6_r1_m2n2_20seed/e6r1_seed_results_combined.csv`; `e6r1_metric_summary.csv`; `e6r1_paired_inference.csv`; E6-R1 manifest | 20 commissioning seeds; q99 frozen primary; mean predeclared sensitivity; Offline vs CommissioningAdapt | **Supported** | “Under q99, commissioning adaptation does not improve low-FPR oracle recall; under mean aggregation it improves ranking/recall consistently, but both variants remain infeasible at the 1% FPR criterion.” | “M2N2 does not work.” | Results, Baselines |
| C10 | A calibration-conditional conformal correction can be statistically stronger yet operationally unusable with only 100 calibration cycles at alpha=0.01. | `outputs/e19_modern_conformal_ccv/e19_summary.csv`; `e19_manifest.json` | Same frozen TargetOnly score for CAD-Marginal and CAD-CCV-DKWM; pre-anomaly-label minimum attainable p-value check | **Supported** | “With m=100, the DKWM-adjusted minimum attainable p-value is about 0.1458, so no alarm can be issued at alpha=0.01 despite informative anomaly scores.” | “DKWM is a bad method.” / “The underlying score failed.” | Results, Discussion |
| C11 | Marginal conformal can remain permissive enough to alarm while still exceeding the operational false-alert budget. | E19 seed results and summary | Same score model and evaluation split as DKWM comparator | **Supported** | “Marginal conformal remains operationally permissive but overshoots the 1% false-alert budget in the tested setting.” | “Marginal conformal has no validity.” | Results |
| C12 | The empirical oracle is a useful diagnostic of score geometry but is not itself a population guarantee. | `outputs/e15_oracle_stability/e15_seed_oracle_stability.csv`; `e15_summary.csv`; E15 manifest; `src/oracle_feasibility.py` | 5,000 execution-level bootstrap resamples per detector/N/seed; observed-fast-oracle guard against exact implementation | **Strongly supported** | “Oracle feasibility is sample-sensitive at a 1% FPR budget; it is therefore used descriptively rather than as a deployment guarantee.” | “Oracle recall is the true achievable population recall.” | Methods, Discussion |
| C13 | The main conclusions are not an artifact of a single exact recall/FPR endpoint. | `outputs/e5_operating_point_sensitivity/*`; E5 manifest | Predeclared endpoint grid: R90/B01 primary, R80/B01, R90/B005, R90/B02, R95/B01 | **Supported** | “The commissioning conclusions are qualitatively stable across the predeclared operating-point sensitivity grid.” | “The results hold for all operating points.” | Appendix, Discussion |
| C14 | The reported phenomenon is not explained purely by numerical covariance instability. | M2 numerical-stability outputs; conditioning summary; frozen partition audit | Variance-threshold sensitivity, finite condition-number checks, oracle feasibility under filtered features | **Supported as robustness audit** | “Numerical conditioning is not sufficient to explain the observed commissioning failures.” | “Covariance estimation plays no role.” | Appendix |
| C15 | The frozen TargetOnly score sequence shows no detected serial dependence under the E8 tests, although raw feature channels can be temporally dependent. | `outputs/e8_temporal_dependence/*`; E8 manifest | Lag-1 permutation test, Ljung–Box, ESS, moving-block bootstrap sensitivity | **Supported with caution** | “E8 found no significant serial dependence in the frozen TargetOnly score sequences under the tested diagnostics, while raw channels showed non-negligible autocorrelation.” | “The robot executions are independent.” / “Exchangeability is proven.” | Appendix, Limitations |
| C16 | AURSAD supports the broader claim that stringent commissioning can remain difficult on another robotic manipulation task. | `outputs/aursad/comparison/aursad_detector_comparison_overall.csv`; `outputs/aursad/comparison/aursad_n_star_comparison.csv`; E9/AURSAD closeout artifacts | Multiple detectors to N=500; per-fault descriptive checks; leakage-safe split audit | **Supported as external validation of commissioning difficulty** | “On AURSAD, the evaluated detectors remain censored beyond N=500 under the joint criterion, supporting cross-task commissioning difficulty.” | “RACE generalizes across AURSAD.” / “All robotic tasks behave this way.” | Results, Discussion |
| C17 | The central contribution is an operational evidence-allocation framework, not a positive RACE sample-complexity result. | E0–E19 consolidated evidence; `reports/race_mechanism_closeout.md`; this ledger | RACE negative transfer, M2N2, modern conformal comparator, certification/sample-size analysis | **Supported and should govern paper framing** | “COLDSTART studies how score quality, adaptation, calibration, and post-freeze certification jointly determine commissioning evidence requirements.” | “RACE reduces commissioning data by >40%.” | Title, Abstract, Introduction, Conclusion |
| C18 | The current evidence does not support a universal healthy-only transferability predictor. | E7 compatibility associations; S-RACE negative controls; P0 precision-transfer audits | Negative controls, source permutation, compatibility permutation, null/stability analyses | **Not established / negative evidence** | “We did not find a reliable healthy-only predictor of seedwise transfer benefit in the tested analyses.” | “We can predict negative transfer before deployment.” | Discussion, Limitations |
| C19 | Full clean-room reproducibility is already established. | Current manifests, tracked reproducibility utilities, environment-lock work | Dirty-tree checks, dependency lock, clean checkout rerun, artifact hashes | **Not yet supported** | “Code and protocol artifacts are versioned; final clean-room reproduction is pending completion.” | “Fully reproducible from a clean checkout” until actually demonstrated. | Reproducibility statement |
| C20 | Results establish behavior on physical production robots. | None | None | **Unsupported** | “Experiments use public robotic manipulation datasets.” | “Validated in production/real commissioning deployments.” | Limitations |

## Abstract-safe claims

The following claims are strong enough for the abstract if written narrowly:

1. **Operational gap:** high anomaly ranking can coexist with failure of a joint recall/FPR commissioning requirement.
2. **Three-way decomposition:** COLDSTART separates representation, calibration, and certification bottlenecks.
3. **Finite-sample evidence burden:** low-FPR calibration and exact certification can require substantially more healthy executions than ranking metrics suggest.
4. **Negative-transfer case study:** in the frozen voraus-AD PRE_A->PRE_B setting, RACE exhibits statistically supported negative transfer relative to TargetOnly.
5. **Modern baselines do not trivially solve the issue:** M2N2 and a modern calibration-conditional conformal comparator do not satisfy the primary 1% FPR commissioning criterion under the tested protocol.
6. **External validation:** AURSAD shows that stringent commissioning difficulty is not unique to the primary pick-and-place benchmark.

## Claims that must stay out of the abstract

- Any statement that RACE reduces commissioning sample complexity.
- Any universal statement that transfer is harmful.
- Any statement that lambda=60 is optimal.
- Any statement that oracle recall is a population guarantee.
- Any claim that E8 proves independence or exchangeability.
- Any claim of production-robot or prospective field validation.
- Any claim of clean-room reproducibility until the final clean freeze is actually completed.

## Final evidence hierarchy

### Tier 1 — Primary paper evidence

- Frozen TargetOnly/RACE commissioning curves and joint operating criterion.
- E1 oracle bottleneck decomposition.
- E3/E4 low-FPR calibration and certification sample-size analysis.
- E7 20-seed paired RACE negative-transfer inference.
- E6-R1 20-seed M2N2 replication.
- E19 modern conformal comparator.
- AURSAD external validation.

### Tier 2 — Reviewer-defense robustness evidence

- E5 operating-point sensitivity.
- E8 temporal-dependence diagnostics.
- E15 oracle stability bootstrap.
- E17 lambda sensitivity.
- M2 numerical-stability audit.
- E7 compatibility/negative-transfer diagnostics.

### Tier 3 — Diagnostic/history only

- Early smoke runs and seed-0 mechanism investigations.
- S-RACE redesign attempts and negative controls.
- P0 precision-transfer feasibility audits unless explicitly needed in the appendix.
- Any dirty-tree exploratory result that has not been cleanly regenerated.

## Submission stoplights

| Area | Status | Submission interpretation |
| --- | --- | --- |
| Primary operating criterion | GREEN | Frozen and consistently used |
| TargetOnly primary benchmark | GREEN | Complete |
| RACE negative-transfer inference | GREEN | 20-seed paired inference complete |
| Lambda sensitivity | GREEN | Full 560-cell sweep complete; no post-hoc retuning allowed |
| Modern adaptation baseline | GREEN | E6-R1 20-seed replication complete |
| Modern conformal comparator | GREEN | E19 complete |
| Oracle robustness | GREEN | E15 bootstrap complete |
| Operating-point sensitivity | GREEN | E5 complete |
| Temporal-dependence audit | GREEN | E8 complete, with cautious interpretation |
| AURSAD external validation | GREEN/YELLOW | Strong for commissioning difficulty; not a transfer-success claim |
| Clean-room reproduction | RED until rerun | Must be completed before submission claim |
| Physical/prospective robot validation | YELLOW | Valuable but not required for the current dataset-based claim; must be explicit limitation |
| Final figure/table QA | YELLOW | Regenerate from frozen artifacts, hash, and visually inspect |

## Recommended paper thesis

> **COLDSTART is an operational commissioning framework for robotic anomaly detectors under explicit recall and false-alert constraints. Across two robotic manipulation datasets and multiple statistical/adaptation/calibration baselines, the experiments show that excellent score ranking does not imply deployability: representation quality, finite-sample calibration, and independent post-freeze certification create distinct evidence bottlenecks. A source-transfer case study further shows statistically supported negative transfer under one frozen robot-setting shift, motivating explicit compatibility and certification audits rather than assuming transfer reduces commissioning data.**

This thesis is supported by the current evidence. It is intentionally narrower than the original positive-transfer RACE hypothesis and should be used to police the abstract, introduction, results, and conclusion against overclaiming.
