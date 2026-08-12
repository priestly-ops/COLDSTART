# Original RACE Targeted Mechanism Investigation

Date: 2026-08-12

## Scope

This is a targeted diagnostic investigation only. It uses no anomaly-based
tuning and makes no changes to SS-RACE thresholds, compatibility weights,
effective-rank policy, or safe-gate logic.

The goal is to compare three quantities for each Original RACE
source/direction:

1. healthy compatibility;
2. contribution to Original RACE covariance/score geometry;
3. eventual anomaly separation as post-hoc diagnostic evidence only.

Primary artifacts:

- `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_direction_audit_all_seeds.csv`
- `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_component_summary.csv`
- `outputs/original_race_component_ablation_seed0_4_aggregate/original_race_delta_summary.csv`
- `reports/original_race_component_ablation_seed0_4_replication.md`

The ablation set was frozen before interpretation:

- `MeanTransferOnly`
- `CovarianceTransferOnly`
- `EigenvectorsOnly`
- `EigenvaluesOnly`
- `TargetMeanSourceCovariance`
- `SourceMeanTargetCovariance`

## Direction-Level Audit

The direction audit contains `8460` direction rows:

- 5 seeds;
- 3 source regimes;
- 564 target directions per source/seed pair.

Relevant healthy-only quantities:

- `healthy_compatibility`
- `source_target_eigenvector_alignment_max_cos2`
- `variance_compatibility`
- `target_eigenvalue`
- `source_projected_variance`

Relevant covariance/score contribution quantities:

- `covariance_delta_projected`
- `calibration_score_contribution_change_mean`
- `healthy_score_contribution_change_mean`
- `healthy_score_contribution_change_abs_mean`

Post-hoc anomaly diagnostic quantity:

- `posthoc_direction_separation_change`

This last quantity is used only to explain what happened after evaluation. It
must not be converted into transfer weights, gates, thresholds, rank choices, or
compatibility definitions.

## Source-Level Means

Mean over all directions:

| source | healthy compatibility | covariance delta projected | healthy score change abs | post-hoc separation change |
|---|---:|---:|---:|---:|
| near | 0.0133 | -0.7063 | 1.0330 | -0.6244 |
| moderate | 0.0137 | +4.2349 | 1.0283 | -0.7382 |
| high | 0.0155 | +11.8669 | 1.0688 | -0.8186 |

The all-direction means are dominated by the 564-dimensional basis. They are
therefore useful for aggregate accounting, but not sufficient to identify the
few directions that drive score changes.

Among the top 10 post-hoc separation-improving directions per source/seed:

| source | healthy compatibility | covariance delta projected | healthy score change abs | post-hoc separation change |
|---|---:|---:|---:|---:|
| near | 0.0230 | -100.9042 | 0.6324 | +1.1746 |
| moderate | 0.0200 | -27.2077 | 0.5727 | +0.7599 |
| high | 0.0253 | -2.4130 | 0.8892 | +0.6955 |

This indicates that the most anomaly-separating directions are sparse and are
not simply the largest healthy-compatibility directions.

## Correlation Checks

Across all source/direction rows:

| quantity pair | Pearson | Spearman |
|---|---:|---:|
| healthy compatibility vs post-hoc separation change | -0.0012 | -0.2656 |
| healthy compatibility vs healthy score change abs | +0.0020 | +0.3102 |
| covariance delta projected vs post-hoc separation change | -0.2675 | -0.4204 |
| healthy score change abs vs post-hoc separation change | -0.7445 | -0.3088 |
| eigenvector alignment vs post-hoc separation change | -0.2258 | -0.3002 |
| variance compatibility vs post-hoc separation change | +0.3600 | +0.1004 |
| target eigenvalue vs post-hoc separation change | +0.0119 | -0.0042 |
| source projected variance vs post-hoc separation change | -0.0974 | -0.4111 |

The key result is negative: healthy compatibility is not a simple proxy for
eventual anomaly separation. The directions that explain post-hoc anomaly
separation do not justify retrofitting the compatibility metric.

## Top-Direction Examples

The largest post-hoc separation gain appears repeatedly at seed 0, direction
28:

| seed | source | direction | healthy compatibility | covariance delta | score change abs | post-hoc separation change |
|---:|---|---:|---:|---:|---:|---:|
| 0 | high | 28 | 0.0111 | +101.3156 | 16.4831 | +2.4221 |
| 0 | near | 28 | 0.0172 | +68.2899 | 15.7632 | +2.1334 |
| 0 | moderate | 28 | 0.0124 | +101.6090 | 16.3718 | +2.1241 |

This is a covariance/whitening geometry signature, not a high healthy
compatibility signature.

In contrast, the highest healthy-compatibility directions are mostly dominant
low-rank covariance directions, but they do not reliably improve anomaly
separation:

| seed | source | direction | healthy compatibility | covariance delta | score change abs | post-hoc separation change |
|---:|---|---:|---:|---:|---:|---:|
| 2 | high | 0 | 0.7346 | +265.2187 | 1.1929 | -1.1106 |
| 4 | moderate | 0 | 0.7044 | +712.2480 | 0.0994 | -0.1815 |
| 3 | near | 2 | 0.6952 | -94.7471 | 0.0564 | +0.0604 |
| 3 | near | 0 | 0.6845 | -167.1514 | 0.1594 | +0.3449 |

This argues against interpreting healthy subspace overlap alone as sufficient
evidence of useful transfer.

## Component Ablation Result

Aggregate mean delta versus TargetOnly over 15 source-regime evaluations:

| detector | delta recall | delta FPR | delta AUROC |
|---|---:|---:|---:|
| OriginalRACE | +0.2792 | -0.0007 | +0.0921 |
| CovarianceTransferOnly | +0.2828 | -0.0027 | +0.1031 |
| TargetMeanSourceCovariance | +0.2468 | -0.0027 | +0.0980 |
| EigenvectorsOnly | +0.1019 | +0.0080 | +0.0236 |
| EigenvaluesOnly | +0.0244 | -0.0020 | +0.0038 |
| MeanTransferOnly | -0.0237 | -0.0020 | -0.0081 |
| SourceMeanTargetCovariance | -0.0266 | -0.0027 | -0.0131 |

All methods have success rate `0.0` under the frozen commissioning criterion:

`recall >= 0.90 and FPR <= 0.01`.

## Mechanism Interpretation

The Original RACE high-shift recall benefit is best explained by covariance
transfer and whitening geometry.

Supported:

- `CovarianceTransferOnly` reproduces and slightly exceeds OriginalRACE's recall
  and AUROC gain.
- `TargetMeanSourceCovariance` recovers most of the effect.
- `MeanTransferOnly` and `SourceMeanTargetCovariance` are worse than TargetOnly.
- `EigenvaluesOnly` is weak.
- `EigenvectorsOnly` changes decisions, but the aggregate FPR exceeds the 1%
  operating budget.

Not supported:

- source mean transfer as the driver;
- source eigenvalues alone as the driver;
- healthy compatibility alone as a sufficient predictor of anomaly-separating
  directions;
- operational success under the commissioning criterion.

Plausible mechanism:

Original RACE changes the Mahalanobis whitening geometry through source
covariance structure. A small number of covariance directions can increase
post-hoc anomaly separation even when their healthy compatibility is not large.
This can improve recall at the fixed conformal operating point, but in the
current evidence it still does not produce a successful commissioned detector.

## Boundary

Do not use this report to tune SS-RACE.

The only legitimate use is explanatory:

- Original RACE's observed recall bump comes from covariance geometry.
- The bump is not enough for operational commissioning success.
- Post-hoc anomaly-separating directions expose a mechanism but cannot define a
  healthy-only transfer rule.
