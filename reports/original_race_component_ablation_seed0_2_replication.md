# Original RACE Component Ablation: Seeds 0-2 Replication

## Scope

This report extends the frozen Original RACE component ablation to three
commissioning seeds at `N=10`. Each seed uses the same frozen evaluation split
policy, source-regime construction, healthy-only calibration, and predeclared
ablation definitions.

Artifacts:

- `outputs/original_race_component_ablation_seed0/`
- `outputs/original_race_component_ablation_seed1/`
- `outputs/original_race_component_ablation_seed2/`

No anomaly-based tuning was introduced. Fault labels are used only after model
fit and healthy calibration to compute recall/AUROC/AUPRC and post-hoc
directional separation.

## Aggregate Across 9 Source-Regime Evaluations

Three seeds times near/moderate/high source regimes.

| detector | recall | FPR | AUROC |
|---|---:|---:|---:|
| TargetOnly | 0.2141 | 0.0100 | 0.8258 |
| OriginalRACE | 0.4977 | 0.0044 | 0.9183 |
| CovarianceTransferOnly | 0.5021 | 0.0044 | 0.9286 |
| TargetMeanSourceCovariance | 0.4708 | 0.0022 | 0.9248 |
| EigenvectorsOnly | 0.3216 | 0.0144 | 0.8524 |
| EigenvaluesOnly | 0.2349 | 0.0067 | 0.8332 |
| MeanTransferOnly | 0.1770 | 0.0044 | 0.8204 |
| SourceMeanTargetCovariance | 0.1725 | 0.0033 | 0.8161 |

## Mean Delta vs TargetOnly

| detector | delta recall | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.2836 | -0.0056 | +0.0925 |
| CovarianceTransferOnly | +0.2880 | -0.0056 | +0.1028 |
| TargetMeanSourceCovariance | +0.2567 | -0.0078 | +0.0990 |
| EigenvectorsOnly | +0.1074 | +0.0044 | +0.0266 |
| EigenvaluesOnly | +0.0208 | -0.0033 | +0.0074 |
| MeanTransferOnly | -0.0371 | -0.0056 | -0.0054 |
| SourceMeanTargetCovariance | -0.0416 | -0.0067 | -0.0097 |

## Mechanism Conclusion

Across three seeds, Original RACE's apparent N=10 benefit is reproduced almost
exactly by `CovarianceTransferOnly`. It is also largely reproduced by using the
target mean with the source covariance. Mean transfer alone is consistently not
helpful.

This points to covariance/whitening geometry, not source mean transfer, as the
dominant mechanism in this diagnostic setting.

The result is still not an operational success claim: success rate is zero for
all ablations under the predeclared `recall >= 0.90` and `FPR <= 0.01`
criterion. It is a mechanism audit explaining how Original RACE changes scores
and decisions when it improves recall.

## Score-Equivalence Check

Seed 2 again shows TargetOnly vs OriginalRACE is not a trivial affine or
rank-preserving rescaling:

| source | affine R2 | Spearman | changed predictions |
|---|---:|---:|---:|
| near | 0.6331 | 0.7967 | 153 |
| moderate | 0.6804 | 0.7865 | 167 |
| high | 0.6099 | 0.7044 | 190 |

Thus this branch of Original RACE behavior is not explained by conformal
calibration canceling a monotone score transform.

## Boundary

These findings should not be used to loosen SS-RACE compatibility thresholds or
safe gates. The scientifically defensible use is explanatory: Original RACE can
benefit from covariance import in directions that may not look compatible under
healthy-only selective-transfer criteria, but that observation was made
post-hoc and cannot become a tuning rule.
