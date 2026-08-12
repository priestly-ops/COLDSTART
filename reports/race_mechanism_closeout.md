# RACE Mechanism Closeout

Date: 2026-08-12

## Purpose

This note closes the current RACE mechanism audit at the working-tree evidence
level. It separates three findings that should not be conflated:

1. Original/global RACE can change score geometry.
2. In one N=10 diagnostic branch, OriginalRACE's recall gain is explained by
   covariance/whitening geometry rather than mean transfer.
3. SS-RACE does not currently provide a positive selective-transfer result and
   should not be retuned using post-hoc anomaly evidence.

This is not a final manuscript reproducibility freeze. Several manifests still
record a dirty worktree, and the repository is not in a clean release state.

## Evidence Base

Primary artifacts:

- `reports/m3_structural_score_equivalence_audit.md`
- `reports/original_race_component_ablation_seed0_4_replication.md`
- `outputs/original_race_component_ablation_seed0_4_aggregate/`
- `reports/srace_seed0_stop_go.md`
- `reports/srace_compatibility_scale_audit.md`
- `reports/claim_to_evidence_ledger.md`

## Finding 1: Original RACE Can Be Structurally Equivalent To TargetOnly

The corrected M3 structural-equivalence audit found that TargetOnly and
OriginalRACE can produce near-identical operational decisions despite large
internal score/threshold changes.

Key interpretation:

- In that M3 branch, RACE changes score magnitude and threshold scale.
- Conformal calibration largely cancels the transformation.
- The result is no meaningful transfer benefit at the decision level.

This supports the claim that internal transfer weights or score magnitudes are
insufficient evidence of useful source transfer.

## Finding 2: When Original RACE Improves Recall, Covariance Transfer Explains It

The five-seed Original RACE component ablation at `N=10` compared:

- `MeanTransferOnly`
- `CovarianceTransferOnly`
- `EigenvectorsOnly`
- `EigenvaluesOnly`
- `TargetMeanSourceCovariance`
- `SourceMeanTargetCovariance`

Aggregate across 15 source-regime evaluations:

| detector | delta recall vs TargetOnly | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.2792 | -0.0007 | +0.0921 |
| CovarianceTransferOnly | +0.2828 | -0.0027 | +0.1031 |
| TargetMeanSourceCovariance | +0.2468 | -0.0027 | +0.0980 |
| MeanTransferOnly | -0.0237 | -0.0020 | -0.0081 |

Interpretation:

- The recall gain is not caused by source mean transfer.
- The gain is reproduced by covariance transfer.
- Target mean plus source covariance recovers most of the gain.
- Eigenvalues alone are weak.
- Eigenvectors alone alter decisions but exceed the 1% FPR operating budget in
  aggregate.

The score-equivalence aggregate also rejects the explanation that this branch is
only an affine/monotone TargetOnly score rescaling:

- Mean TargetOnly-vs-OriginalRACE eval affine R2: `0.5782`
- Mean Spearman correlation: `0.7395`
- Mean changed predictions: `251.7`
- Structural-equivalence flags: `0 / 15`

## Finding 3: The OriginalRACE Gain Is Not An Operational Success Claim

All component ablations have success rate `0.0` under the frozen commissioning
criterion:

`Recall >= 0.90 and FPR <= 0.01`

Therefore the correct claim is mechanistic, not positive-transfer:

> In this diagnostic branch, OriginalRACE's recall gain comes from covariance
> whitening geometry, but it still does not satisfy the operational
> commissioning criterion.

The project should not claim that OriginalRACE reduces commissioning sample
complexity from this evidence.

## Finding 4: SS-RACE Does Not Escape TargetOnly Equivalence In The Seed-0 Validation

The predeclared S-RACE seed-0 validation showed:

| Detector | Recall | FPR | AUROC | Success |
|---|---:|---:|---:|---:|
| TargetOnly | 0.2119 | 0.0000 | 0.8395 | 0.0 |
| S-RACE | 0.2115 | 0.0000 | 0.8395 | 0.0 |
| S-RACE SourcePermutation | 0.2119 | 0.0000 | 0.8395 | 0.0 |
| S-RACE CompatibilityPermutation | 0.2159 | 0.0000 | 0.8416 | 0.0 |

The score-equivalence diagnostics flag TargetOnly-vs-S-RACE structural
equivalence across the source regimes. The safe-gate and compatibility audit
also clarified that the tiny all-dimension weight means are primarily a
rank/dimension reporting effect plus healthy-only gate behavior, not proof of
near-zero active subspace overlap.

## Scientific Decision

`STOP_MODIFYING_RACE_FOR_POSITIVE_TRANSFER_ON_CURRENT_EVIDENCE`

Do not loosen SS-RACE thresholds, compatibility floors, effective-rank choices,
or safe-gate tolerances using the OriginalRACE component-ablation anomaly
separation results.

The defensible interpretation is:

- Original RACE sometimes changes decisions through covariance geometry.
- Selective healthy-only transfer as currently frozen does not produce a
  positive operational result in the seed-0 validation.
- The next manuscript path should emphasize commissioning difficulty,
  calibration-tail limitations, numerical robustness, mechanism controls, and
  negative/conditional transfer evidence unless a new predeclared experiment
  establishes otherwise.

## Remaining Before Manuscript Use

Before any of this becomes final manuscript evidence:

- Freeze a clean repository state.
- Rerun cited experiments from that clean state.
- Preserve manifests with `git_dirty: false`.
- Recompute final bootstrap intervals and N-star tables for claims used in the
  manuscript.
- Run clean-room reproduction or mark the evidence as exploratory/supplementary
  only.
