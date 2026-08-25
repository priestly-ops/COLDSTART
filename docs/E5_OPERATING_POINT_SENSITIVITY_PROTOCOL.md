# E5 — Operating-Point Sensitivity

## Question

Are the COLDSTART bottleneck conclusions specific to the frozen primary
operating point `(recall >= 0.90, FPR <= 0.01)`, or do they persist when the
requested deployment specification is made easier or harder?

E5 is a sensitivity analysis, not a model-selection experiment. No endpoint is
chosen after observing final anomaly performance.

## Frozen sensitivity grid

| Endpoint | Recall target | FPR budget | Purpose |
|---|---:|---:|---|
| `primary_R90_B01` | 0.90 | 0.01 | frozen primary endpoint |
| `lower_recall_R80_B01` | 0.80 | 0.01 | easier recall requirement |
| `strict_fpr_R90_B005` | 0.90 | 0.005 | stricter false-alert budget |
| `loose_fpr_R90_B02` | 0.90 | 0.02 | looser false-alert budget |
| `strict_recall_R95_B01` | 0.95 | 0.01 | stricter recall requirement |

The commissioning grid remains `N={10,25,50,100}` with seeds `0..19`.
Calibration size remains 100 healthy target episodes. Healthy evaluation
remains 100 target episodes. All frozen anomaly-evaluation episodes are used.
Joint confidence remains 95% with the same Bonferroni allocation as E0–E4.

## Detectors

E5 evaluates the two core statistical methods used to define the original
transfer question:

- `TargetOnly`: target commissioning data only.
- `RACE`: source+target shrinkage with `lambda_reg=60.0`.

MVT-Flow is not retrained for the full endpoint sweep in E5. E2 already serves
as the strong-representation control. E5 isolates sensitivity of the core
commissioning/transfer conclusions without adding a new expensive model sweep.

## Score reuse and anti-tuning rule

For each `(detector, N, seed)`, the model is fitted exactly once. The identical
calibration, healthy-evaluation, and anomaly-evaluation score vectors are then
reused for all five operating points.

Changing the FPR budget changes only the deterministic split-conformal
threshold through the frozen mapping `alpha = B`. Changing only the recall
target does not change the deployed threshold.

The primary endpoint must reproduce the frozen E0 threshold, conformal regime,
TP/FN/FP/TN counts, recall, and FPR exactly. Any mismatch aborts the run.

## Important finite-sample calibration consequence

With `M_cal=100`:

- `B=0.005`: requested rank is 101, so the strict threshold is `+inf`.
- `B=0.01`: requested rank is 100, so the threshold is the maximum calibration score.
- `B=0.02`: requested rank is 99, so the threshold is submaximum.

Thus E5 tests both endpoint sensitivity and the finite-sample tail-resolution
consequence induced by the requested false-alert budget.

## Certification feasibility

With 95% simultaneous confidence, the best-case healthy-evaluation requirement
when zero false positives are observed is:

- `B=0.005`: 736 independent healthy episodes.
- `B=0.01`: 368 independent healthy episodes.
- `B=0.02`: 183 independent healthy episodes.

The frozen E5 healthy evaluation size is only 100. Therefore no tested endpoint
can be certified on the healthy-FPR side even under zero observed false
positives. This is structural and should not be interpreted as detector
failure.

For perfect anomaly detection, the best-case anomaly-evaluation requirement is:

- `R0=0.80`: 17 anomaly episodes.
- `R0=0.90`: 36 anomaly episodes.
- `R0=0.95`: 72 anomaly episodes.

## Outputs

- `outputs/e5_operating_point_sensitivity/e5_seed_results.csv`
- `outputs/e5_operating_point_sensitivity/e5_summary.csv`
- `outputs/e5_operating_point_sensitivity/e5_nstar_sensitivity.csv`
- `outputs/e5_operating_point_sensitivity/e5_manifest.json`

`e5_nstar_sensitivity.csv` reports the first tested commissioning size where at
least 80% of commissioning draws pass empirically, certify exactly, or are
oracle-feasible. If no tested N passes, the estimate is right-censored.

## Interpretation rule

The oracle remains retrospective and diagnostic only. It is never used to fit,
calibrate, tune, or certify a detector. Claims must remain protocol-specific.
