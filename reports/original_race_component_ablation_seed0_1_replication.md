# Original RACE Component Ablation: Seeds 0-1 Replication

## Purpose

This extends the seed-0 Original RACE component ablation with a second frozen
commissioning seed. No thresholds, compatibility gates, or detector definitions
were changed based on anomaly outcomes.

The seed-1 run used the checkpointed runner:

`outputs/original_race_component_ablation_seed1/`

The prior seed-0 run remains:

`outputs/original_race_component_ablation_seed0/`

## Runner Robustness Update

`experiments/run_original_race_component_ablation.py` now flushes all CSV
outputs after each completed source regime. This is a reproducibility safeguard
after the interrupted seeds-0-through-4 run produced an empty output folder.
It does not change any ablation definition or scoring method.

## Aggregate Result Across Seeds 0 and 1

Mean across 6 source-regime evaluations: 2 commissioning seeds times
near/moderate/high source regimes.

| detector | recall | FPR | AUROC |
|---|---:|---:|---:|
| TargetOnly | 0.1901 | 0.0050 | 0.8197 |
| OriginalRACE | 0.5391 | 0.0067 | 0.9220 |
| CovarianceTransferOnly | 0.5446 | 0.0067 | 0.9296 |
| TargetMeanSourceCovariance | 0.5168 | 0.0033 | 0.9280 |
| EigenvectorsOnly | 0.3369 | 0.0167 | 0.8563 |
| EigenvaluesOnly | 0.2300 | 0.0100 | 0.8341 |
| MeanTransferOnly | 0.1850 | 0.0033 | 0.8208 |
| SourceMeanTargetCovariance | 0.1845 | 0.0033 | 0.8191 |

## Mean Delta vs TargetOnly

| detector | delta recall | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.3490 | +0.0017 | +0.1023 |
| CovarianceTransferOnly | +0.3545 | +0.0017 | +0.1099 |
| TargetMeanSourceCovariance | +0.3267 | -0.0017 | +0.1083 |
| EigenvectorsOnly | +0.1468 | +0.0117 | +0.0366 |
| EigenvaluesOnly | +0.0400 | +0.0050 | +0.0144 |
| MeanTransferOnly | -0.0051 | -0.0017 | +0.0011 |
| SourceMeanTargetCovariance | -0.0055 | -0.0017 | -0.0006 |

## Scientific Read

The second seed supports the seed-0 conclusion:

- Original RACE's benefit is primarily covariance/whitening geometry.
- Mean transfer is not a plausible explanation for the benefit.
- Source covariance with target mean recovers most of the gain, which argues
  against source mean transfer being necessary.
- Eigenvalues alone are weak.
- Eigenvectors alone can alter decisions, but it also raises FPR above the
  operating budget in aggregate, so it is not a clean operational mechanism.

This remains diagnostic evidence about Original RACE. It does not license
loosening SS-RACE thresholds or using anomaly separation to choose compatibility
parameters.
