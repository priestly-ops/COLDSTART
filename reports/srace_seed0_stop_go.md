# S-RACE Seed-0 Stop/Go Decision

Date: 2026-08-12

## Scope

This note summarizes the completed S-RACE seed-0 validation in:

`outputs/srace_small_validation/`

and the diagnostic rerun with persisted compatibility/gate diagnostics in:

`outputs/srace_seed0_diagnostics_v2/`

This is a small predeclared mechanism validation, not the full commissioning grid. The manifest records `git_dirty: true`, so this is valid stop/go evidence for the current working tree but not a clean reproducibility freeze.

## Artifact Coverage

All required validation artifacts are present:

- `srace_manifest.json`
- `srace_seed_results.csv`
- `srace_summary.csv`
- `srace_paired_deltas.csv`
- `srace_per_class_recall.csv`
- `srace_partition_audit.csv`
- `srace_mechanism_diagnostics.csv`
- `srace_transfer_weights.csv`
- `srace_source_compatibility.csv`
- `srace_score_equivalence.csv`
- `srace_score_scatter.csv`

The partition audit has 3 rows and 0 overlap failures.

## Protocol

- Dataset hash: `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704`
- Commissioning grid: `N={10}`
- Seeds: `0`
- Calibration healthy size: `100`
- Healthy evaluation size: `100`
- False-alert budget: `0.01`
- Recall target: `0.90`
- Detectors: `TargetOnly`, `OriginalRACE`, `S-RACE`, `S-RACE WrongSource`, `S-RACE SourcePermutation`, `S-RACE CompatibilityPermutation`

## Operational Results

Mean results across the three source regimes:

| Detector | Recall | FPR | AUROC | AUPRC | Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| TargetOnly | 0.2119 | 0.0000 | 0.8395 | 0.9747 | 0.0 |
| OriginalRACE | 0.4468 | 0.0033 | 0.9124 | 0.9873 | 0.0 |
| S-RACE | 0.2115 | 0.0000 | 0.8395 | 0.9747 | 0.0 |
| S-RACE WrongSource | 0.2106 | 0.0000 | 0.8396 | 0.9747 | 0.0 |
| S-RACE SourcePermutation | 0.2119 | 0.0000 | 0.8395 | 0.9747 | 0.0 |
| S-RACE CompatibilityPermutation | 0.2159 | 0.0000 | 0.8416 | 0.9750 | 0.0 |

No detector satisfies the operational criterion `Recall >= 0.90 and FPR <= 0.01`.

## Paired S-RACE Deltas vs TargetOnly

Mean paired deltas:

| Detector | Delta Recall | Delta FPR | Delta AUROC | Delta AUPRC | Delta Success |
| --- | ---: | ---: | ---: | ---: | ---: |
| S-RACE | -0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| S-RACE WrongSource | -0.0013 | 0.0000 | 0.0001 | 0.0000 | 0.0 |
| S-RACE SourcePermutation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| S-RACE CompatibilityPermutation | 0.0040 | 0.0000 | 0.0021 | 0.0004 | 0.0 |

Interpretation: S-RACE does not produce a meaningful decision-level improvement over TargetOnly and does not beat negative controls.

## Mechanism Evidence

For `S-RACE`:

- Mean transferred dimensions: `1.33`
- Mean transfer weight: `0.00057`
- Mean compatibility: `0.00085`
- Mean structural compatibility: `0.00426`
- Mean active structural compatibility: near `0.597`, moderate `0.593`, high `0.611`
- Mean canonical principal-angle `cos^2`: near `0.597`, moderate `0.593`, high `0.611`
- Pre-gate mean transfer weight: near `0.001355`, moderate `0.001391`, high `0.001702`
- Shared rank: `4`
- Private dimensions: `560`

For near and moderate source regimes, the healthy-only leave-one-out gate closes and S-RACE falls back to TargetOnly:

- near `safe_gate_margin = -0.042890`
- moderate `safe_gate_margin = -0.029024`

For the high-shift regime, the gate opens:

- high `safe_gate_margin = 0.017760`

but the resulting prediction change is negligible and not beneficial.

## Score-Equivalence Evidence

TargetOnly vs S-RACE on the combined evaluation split:

- Near shift: exact structural score equivalence; 0 changed predictions.
- Moderate shift: exact structural score equivalence; 0 changed predictions.
- High shift: `affine_r2 = 0.999996`, `threshold_ratio = 0.999111`, structural score equivalence flagged, 1 changed prediction.

Interpretation: S-RACE is operationally and structurally almost identical to TargetOnly in this seed-0 validation.

## Decision

`STOP_MODIFYING_RACE_FOR_POSITIVE_TRANSFER_ON_THIS_EVIDENCE`

This seed-0 validation does not justify proceeding to a full 20-seed grid for a positive SS-RACE claim. The scientifically defensible path is to freeze this as evidence that the selective transfer attempt did not escape TargetOnly-equivalence in the small validation, then decide whether the paper should proceed as a rigorously controlled negative/commissioning-bottleneck result.

The diagnostic rerun sharpens the mechanism: tiny all-dimension compatibility/weight means are primarily a rank/dimension reporting effect plus post-gate fallback, not evidence of near-zero active source-target subspace overlap.

Because the run was produced from a dirty worktree, the next reproducibility step is to freeze the exact code state and rerun this validation once from a clean manifest before using it as final manuscript evidence.
