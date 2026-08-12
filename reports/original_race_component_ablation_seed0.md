# Original RACE Component Ablation: Seed 0, N=10

## Purpose

This is a targeted diagnostic of why Original RACE improved recall in
`outputs/srace_small_validation/` without adding anomaly-based tuning or
loosening thresholds.

The ablation definitions were frozen before interpretation:

- `MeanTransferOnly`
- `CovarianceTransferOnly`
- `EigenvectorsOnly`
- `EigenvaluesOnly`
- `TargetMeanSourceCovariance`
- `SourceMeanTargetCovariance`

Anomaly outcomes are used only after fitting and healthy calibration to report
recall, AUROC/AUPRC, and post-hoc directional separation.

## Artifacts

Output folder:

`outputs/original_race_component_ablation_seed0/`

Key files:

- `original_race_manifest.json`
- `original_race_component_ablation.csv`
- `original_race_summary.csv`
- `original_race_paired_deltas.csv`
- `original_race_direction_audit.csv`
- `original_race_score_equivalence.csv`
- `original_race_source_compatibility.csv`
- `original_race_partition_audit.csv`

## Main Result

Original RACE's seed-0 benefit is primarily a covariance effect, not a mean
transfer effect.

Mean recall across near/moderate/high source regimes:

| detector | recall | FPR | AUROC |
|---|---:|---:|---:|
| TargetOnly | 0.2119 | 0.0000 | 0.8395 |
| OriginalRACE | 0.4468 | 0.0033 | 0.9124 |
| MeanTransferOnly | 0.2026 | 0.0000 | 0.8356 |
| CovarianceTransferOnly | 0.4693 | 0.0033 | 0.9162 |
| TargetMeanSourceCovariance | 0.4221 | 0.0033 | 0.9121 |
| EigenvectorsOnly | 0.3426 | 0.0200 | 0.8419 |
| EigenvaluesOnly | 0.2238 | 0.0000 | 0.8486 |
| SourceMeanTargetCovariance | 0.2000 | 0.0000 | 0.8334 |

## Per-Source Recall Delta vs TargetOnly

| ablation | near | moderate | high |
|---|---:|---:|---:|
| OriginalRACE | +0.0715 | +0.2013 | +0.4318 |
| MeanTransferOnly | +0.0079 | -0.0053 | -0.0305 |
| CovarianceTransferOnly | +0.1629 | +0.1960 | +0.4132 |
| EigenvectorsOnly | +0.2450 | +0.0093 | +0.1377 |
| EigenvaluesOnly | +0.0119 | +0.0119 | +0.0119 |
| TargetMeanSourceCovariance | +0.1364 | +0.1563 | +0.3377 |
| SourceMeanTargetCovariance | +0.0079 | -0.0079 | -0.0358 |

Interpretation:

- Mean transfer is not the driver.
- Covariance transfer explains almost all of Original RACE's moderate/high
  recall gain.
- Source covariance with the target mean also recovers much of the gain,
  implying whitening/covariance geometry dominates.
- Eigenvalues alone do little.
- Eigenvectors alone can help near shift, but with FPR inflation, so this is
  not an operational success.

## Score-Geometry Result

TargetOnly vs OriginalRACE is not a trivial affine rescaling in this run.

Eval-score affine R2 and changed predictions:

| source | affine R2 | Spearman | changed predictions |
|---|---:|---:|---:|
| near | 0.6274 | 0.7638 | 120 |
| moderate | 0.5916 | 0.7505 | 184 |
| high | 0.6122 | 0.7438 | 333 |

Thus the seed-0 Original RACE gain is not explained by conformal cancellation
of a monotone score rescaling.

## Direction Audit

The largest post-hoc separation increases are concentrated in a small number of
target-eigenbasis directions, especially direction 28 across all three source
regimes. These directions have low healthy compatibility by the diagnostic
scale, so they should not be used to tune SS-RACE thresholds. They are evidence
about how Original RACE changed whitening geometry after the method was fixed.

Top post-hoc direction examples:

| source | direction | healthy compatibility | covariance delta | healthy score change | post-hoc separation change |
|---|---:|---:|---:|---:|---:|
| high | 28 | 0.0111 | 101.3156 | -16.2286 | +2.4221 |
| near | 28 | 0.0172 | 68.2899 | -15.6611 | +2.1334 |
| moderate | 28 | 0.0124 | 101.6090 | -16.2334 | +2.1241 |
| near | 58 | 0.0111 | -4.7973 | +0.3637 | +1.4182 |
| near | 268 | 0.0080 | -7.6167 | +0.1909 | +1.2677 |

## Current Scientific Read

For this seed, Original RACE's high-shift benefit appears to come from global
covariance/whitening geometry and source covariance structure, not from source
mean transfer and not from source eigenvalues alone.

This does not justify loosening SS-RACE gates. It indicates that Original RACE
sometimes gains anomaly separation by importing covariance directions that look
weak under healthy-only compatibility diagnostics. That is useful mechanism
evidence, but it is not a legitimate input to redesign transfer thresholds.
