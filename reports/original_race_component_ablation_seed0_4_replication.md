# Original RACE Component Ablation: Seeds 0-4 Replication

## Scope

This report summarizes the frozen Original RACE component ablation across five
commissioning seeds at `N=10`. The run is diagnostic: no anomaly-based tuning,
threshold loosening, compatibility-rule adjustment, or post-hoc method change
was introduced.

Artifacts:

- `outputs/original_race_component_ablation_seed0/`
- `outputs/original_race_component_ablation_seed1/`
- `outputs/original_race_component_ablation_seed2/`
- `outputs/original_race_component_ablation_seed3/`
- `outputs/original_race_component_ablation_seed4/`

Each seed evaluates the same frozen ablation set over near/moderate/high source
regimes using source healthy data, target commissioning healthy data, target
healthy calibration, held-out healthy evaluation, and post-hoc anomaly
evaluation.

## Aggregate Across 15 Source-Regime Evaluations

Five seeds times near/moderate/high source regimes.

| detector | recall | FPR | AUROC |
|---|---:|---:|---:|
| TargetOnly | 0.2040 | 0.0080 | 0.8197 |
| OriginalRACE | 0.4832 | 0.0073 | 0.9117 |
| CovarianceTransferOnly | 0.4868 | 0.0053 | 0.9227 |
| TargetMeanSourceCovariance | 0.4508 | 0.0053 | 0.9177 |
| EigenvectorsOnly | 0.3059 | 0.0160 | 0.8432 |
| EigenvaluesOnly | 0.2283 | 0.0060 | 0.8234 |
| MeanTransferOnly | 0.1803 | 0.0060 | 0.8116 |
| SourceMeanTargetCovariance | 0.1774 | 0.0053 | 0.8066 |

## Mean Delta vs TargetOnly

| detector | delta recall | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.2792 | -0.0007 | +0.0921 |
| CovarianceTransferOnly | +0.2828 | -0.0027 | +0.1031 |
| TargetMeanSourceCovariance | +0.2468 | -0.0027 | +0.0980 |
| EigenvectorsOnly | +0.1019 | +0.0080 | +0.0236 |
| EigenvaluesOnly | +0.0244 | -0.0020 | +0.0038 |
| MeanTransferOnly | -0.0237 | -0.0020 | -0.0081 |
| SourceMeanTargetCovariance | -0.0266 | -0.0027 | -0.0131 |

## Mechanism Conclusion

Across five seeds, Original RACE's recall benefit is explained by
covariance/whitening geometry:

- `CovarianceTransferOnly` slightly exceeds Original RACE's mean recall gain.
- `TargetMeanSourceCovariance` recovers most of the gain, so source mean
  transfer is not necessary.
- `MeanTransferOnly` is worse than TargetOnly on mean recall and AUROC.
- `SourceMeanTargetCovariance` is also worse than TargetOnly.
- `EigenvaluesOnly` is weak.
- `EigenvectorsOnly` changes decisions, but its FPR is above the 1% operating
  budget in aggregate.

The result does not support a positive operational-transfer claim: every
ablation has success rate `0.0` under the frozen `recall >= 0.90` and
`FPR <= 0.01` commissioning criterion.

## Score-Equivalence Check

Seed 4 again shows TargetOnly vs OriginalRACE is not structural score
equivalence:

| source | affine R2 | Spearman | changed predictions |
|---|---:|---:|---:|
| near | 0.6107 | 0.7888 | 112 |
| moderate | 0.6242 | 0.7595 | 241 |
| high | 0.4844 | 0.6111 | 205 |

Thus the Original RACE gain in this branch is not merely conformal cancellation
of an affine or monotone score rescaling.

## Boundary For SS-RACE

This evidence should be used to explain Original RACE, not to tune SS-RACE. The
post-hoc direction and anomaly-separation diagnostics show how source covariance
can improve anomaly separation in this small-N setting, but those diagnostics
cannot become compatibility thresholds, safe-gate rules, or transfer-weight
updates.
