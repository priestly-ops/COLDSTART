# Adversarial Reviewer Audit

Date: 2026-08-12

## Purpose

This audit converts the current evidence into reviewer-facing risk language. It
does not declare the project submission-ready. Its job is to identify which
reviewer attacks are currently defended, which are only partially defended, and
which remain open.

## Current Decision

The current working-tree decision remains:

`STOP_MODIFYING_RACE_FOR_POSITIVE_TRANSFER_ON_CURRENT_EVIDENCE`

The strongest manuscript path is a rigorous commissioning/negative-transfer
story unless a new predeclared clean experiment establishes conditional
positive transfer without method changes.

## Reviewer Attack Matrix

| Reviewer attack | Current defense | Evidence | Status | Remaining gap |
|---|---|---|---|---|
| "You tuned RACE variants after seeing anomaly results until something worked." | Current decision explicitly stops additional RACE/SS-RACE retuning for a positive-transfer claim. Original RACE component ablation is diagnostic only. | `reports/race_mechanism_closeout.md`; `reports/srace_seed0_stop_go.md`; `reports/claim_to_evidence_ledger.md` | Partially defended | Need clean git freeze and rerun of any cited validation before manuscript use. |
| "Compatibility is tautological because source quality is defined by anomaly recall." | SS-RACE compatibility and source regimes use source healthy plus target commissioning healthy data only; anomaly separation is post-hoc. | `src/srace.py`; `experiments/run_srace.py`; `src/m3_transfer_regimes.py`; `outputs/srace_small_validation/srace_source_compatibility.csv`; `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_aggregate_manifest.json` | Design defended; predictive claim not established | Healthy-only transferability has not been shown to predict benefit. Do not claim it yet. |
| "Your target test set influenced adaptation." | Frozen split policy separates source healthy, target commissioning, target calibration, healthy evaluation, and anomaly evaluation; partition audits assert disjoint episode IDs. | `src/split_generator.py`; `outputs/srace_small_validation/srace_partition_audit.csv`; `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_partition_audit_all_seeds.csv` | Defended for current working-tree runs | Need clean rerun and manifest freeze for final evidence. |
| "Excellent AUROC is being cherry-picked because the operational claim failed." | Ledger and reports keep recall/FPR joint success primary; AUROC is secondary. Multiple artifacts record success rate 0.0 despite nontrivial AUROC. | `reports/claim_to_evidence_ledger.md`; `reports/srace_seed0_stop_go.md`; `reports/original_race_component_ablation_seed0_4_replication.md`; AURSAD comparison outputs | Defended as a framing principle | Final manuscript tables must present joint success and N-star before AUROC. |
| "Conformal calibration cancels your detector changes." | M3 structural-equivalence audit explicitly measures Pearson/Spearman/Kendall, affine R2, threshold ratio, changed rankings, and prediction changes. OriginalRACE component ablation also reports when this is not the explanation. | `reports/m3_structural_score_equivalence_audit.md`; `outputs/m3_n10_seeds0_4_equivalence/m3_score_equivalence_audit.csv`; `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_score_equivalence_summary.csv` | Defended for audited branches | Need decide how much goes main paper vs supplementary to avoid overloading narrative. |
| "The Original RACE gain is actually mean transfer." | Frozen component ablation falsifies mean transfer as the driver: `MeanTransferOnly` is negative while `CovarianceTransferOnly` matches OriginalRACE. Direction-level audit shows sparse covariance/whitening effects, not high healthy compatibility alone, explain the largest post-hoc separation gains. | `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_delta_summary.csv`; `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_direction_audit_all_seeds.csv`; `reports/original_race_targeted_mechanism_investigation.md`; `reports/race_mechanism_closeout.md` | Defended as diagnostic mechanism claim | Do not convert this into an SS-RACE tuning rule; it is post-hoc explanatory evidence. |
| "Your covariance estimates are meaningless in d >> N." | Methods use Ledoit-Wolf shrinkage, rank caps in SS-RACE, eigenvalue floors, condition numbers, and M2 numerical robustness audit. | `src/srace.py`; `outputs/m2_numerical_stability_audit/m2_v2_3_summary.csv`; `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_component_summary.csv`; `reports/ss_race_freeze_status.md` | Partially defended | Need final clean numerical summary for every main-table method and N. |
| "Negative controls are too weak." | S-RACE includes WrongSource, SourcePermutation, and CompatibilityPermutation controls. These controls do not support a positive source-transfer claim. | `outputs/srace_seed0_diagnostics_v2/srace_seed_results.csv`; `outputs/srace_seed0_diagnostics_v2/srace_paired_deltas.csv`; `reports/srace_seed0_stop_go.md` | Defended for seed-0 mechanism validation | If SS-RACE is not claimed positive, these are mainly falsification evidence. Full-grid controls are not justified unless a new predeclared positive result appears. |
| "The paper overclaims RACE reduces commissioning sample complexity." | Claim-to-evidence ledger explicitly marks positive RACE/SS-RACE sample-complexity claims as unsupported or contradicted. | `reports/claim_to_evidence_ledger.md`; `reports/race_mechanism_closeout.md` | Defended if manuscript obeys ledger | Abstract/introduction must avoid >40% reduction or reliable-transfer claims. |
| "This is just one dataset." | AURSAD outputs partially support external commissioning difficulty for selected detectors. They do not support SS-RACE transfer. | `outputs/aursad/comparison/aursad_detector_comparison_overall.csv`; `outputs/aursad/comparison/aursad_n_star_comparison.csv`; `reports/claim_to_evidence_ledger.md` | Partially defended | External validation remains incomplete for final transfer story; current defensible claim is broader commissioning difficulty, not transfer success. |
| "There are too many methods and exploratory analyses." | Current closeout recommends a negative/commissioning-focused narrative and relegating exploratory history to supplement. | `reports/race_mechanism_closeout.md`; `reports/claim_to_evidence_ledger.md` | Open manuscript risk | Need final figure/table inventory and a concise main-paper story. |
| "The repository is not reproducible." | Manifests record commit/dirty state and artifacts are linked, but dirty-tree status remains. | Manifests in M1/M2/M3/S-RACE/AURSAD/OriginalRACE outputs; `reports/claim_to_evidence_ledger.md` | Not defended for submission | Clean git state, clean reruns, dependency lock, and clean-room reproduction remain required. |

## Claims Allowed By Current Evidence

The following are defensible as working-tree claims, subject to clean rerun
before manuscript use:

- High anomaly ranking does not imply operational commissioning success under
  stringent false-alert constraints.
- Finite calibration size creates a low-FPR tail-estimation bottleneck.
- Original/global transfer can change score geometry without improving
  operational decisions.
- When Original RACE improves recall in the N=10 diagnostic branch, the driver
  is covariance/whitening geometry rather than mean transfer.
- The current frozen SS-RACE seed-0 validation does not improve over TargetOnly
  and does not beat negative controls.

## Claims Not Allowed

Do not claim:

- RACE reduces commissioning sample complexity by more than 40%.
- SS-RACE reduces N-star.
- Source transfer reliably improves cold-start anomaly detection.
- Healthy-only transferability predicts benefit.
- The current evidence is a clean reproducibility freeze.

## Next Required Evidence For Submission

1. Clean code freeze with `git_dirty: false` manifests.
2. Clean rerun of any experiment cited in the main manuscript.
3. Final bootstrap intervals and N-star tables for all main claims.
4. Final per-fault analysis.
5. AURSAD external-validation closeout scoped to commissioning difficulty.
6. Figure/table inventory with regenerated hashes or tolerances.
7. Clean-room reproduction from a fresh checkout/environment.
