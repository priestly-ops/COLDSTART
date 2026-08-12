# Original RACE Component Ablation: Seeds 0-3 Replication

## Scope

This report summarizes the checkpointed Original RACE component ablation across
four commissioning seeds at `N=10`. The ablation definitions remain frozen and
no anomaly-based tuning was introduced.

Artifacts:

- `outputs/original_race_component_ablation_seed0/`
- `outputs/original_race_component_ablation_seed1/`
- `outputs/original_race_component_ablation_seed2/`
- `outputs/original_race_component_ablation_seed3/`

Each run uses the same frozen split policy, near/moderate/high source-regime
construction, healthy-only fit/calibration protocol, and post-hoc anomaly
evaluation.

## Aggregate Across 12 Source-Regime Evaluations

Four seeds times near/moderate/high source regimes.

| detector | recall | FPR | AUROC |
|---|---:|---:|---:|
| TargetOnly | 0.1924 | 0.0075 | 0.8226 |
| OriginalRACE | 0.5115 | 0.0092 | 0.9170 |
| CovarianceTransferOnly | 0.5071 | 0.0058 | 0.9304 |
| TargetMeanSourceCovariance | 0.4833 | 0.0058 | 0.9281 |
| EigenvectorsOnly | 0.3049 | 0.0142 | 0.8535 |
| EigenvaluesOnly | 0.2118 | 0.0050 | 0.8253 |
| MeanTransferOnly | 0.1724 | 0.0042 | 0.8110 |
| SourceMeanTargetCovariance | 0.1708 | 0.0033 | 0.8056 |

## Mean Delta vs TargetOnly

| detector | delta recall | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.3191 | +0.0017 | +0.0944 |
| CovarianceTransferOnly | +0.3147 | -0.0017 | +0.1078 |
| TargetMeanSourceCovariance | +0.2909 | -0.0017 | +0.1055 |
| EigenvectorsOnly | +0.1125 | +0.0067 | +0.0309 |
| EigenvaluesOnly | +0.0194 | -0.0025 | +0.0027 |
| MeanTransferOnly | -0.0200 | -0.0033 | -0.0116 |
| SourceMeanTargetCovariance | -0.0216 | -0.0042 | -0.0170 |

## Mechanism Conclusion

The four-seed replication preserves the same explanation:

- `CovarianceTransferOnly` nearly reproduces Original RACE's recall gain.
- `TargetMeanSourceCovariance` recovers most of the gain.
- `MeanTransferOnly` and `SourceMeanTargetCovariance` are worse than
  TargetOnly on mean recall.
- `EigenvaluesOnly` is weak.
- `EigenvectorsOnly` changes decisions but increases FPR above the operating
  budget in aggregate.

Thus, when Original RACE improves recall in this diagnostic setting, the driver
is covariance/whitening geometry, not source mean transfer.

This still does not create an operational success claim. Every detector has
success rate `0.0` under the frozen `recall >= 0.90` and `FPR <= 0.01`
criterion.

## Score-Equivalence Check

Seed 3 again shows TargetOnly vs OriginalRACE is not structural score
equivalence:

| source | affine R2 | Spearman | changed predictions |
|---|---:|---:|---:|
| near | 0.5060 | 0.7691 | 279 |
| moderate | 0.5426 | 0.7346 | 268 |
| high | 0.5088 | 0.7113 | 456 |

So the observed gain is not explained by conformal cancellation of a monotone
score rescaling.

## Scientific Boundary

This audit explains Original RACE behavior. It should not be used to tune
SS-RACE compatibility thresholds, safe gates, or transfer weights. The
post-hoc anomaly separation evidence remains diagnostic only.
