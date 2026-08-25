# E1 — Oracle-vs-Deployed Bottleneck Decomposition

## Status

This experiment is downstream of the frozen E0 certification backbone. It does not alter detector fitting, calibration, certification confidence semantics, the `alpha = B` mapping, or the deployed `score > threshold` decision rule.

## Primary operating point

- Recall target: `R0 = 0.90`
- False-alert budget: `B = 0.01`
- Joint certification confidence inherited from E0: `0.95`
- Detectors: `TargetOnly`, `RACE`
- Commissioning sizes: `N in {10, 25, 50, 100}`
- Commissioning seeds: `0..19`
- Frozen evaluation seed: `42`
- Calibration size: `100`
- Healthy evaluation size: `100`

## Oracle definition

For each frozen detector replicate, E1 uses the same scalar anomaly score and the same strict monotone decision family as deployment:

`alarm = 1[score > threshold]`.

The retrospective empirical oracle is

`R_oracle(B) = max_threshold Recall(threshold)` subject to `empirical FPR(threshold) <= B`.

The oracle exhausts all distinct empirical threshold decisions. Evaluation labels are used only for this retrospective diagnostic. The oracle is never used to fit the detector, select the deployed threshold, or certify performance.

## Deterministic bottleneck classification

Classification is hierarchical and mutually exclusive:

1. `representation_limited`: the empirical oracle cannot achieve `Recall >= R0` while satisfying `FPR <= B`.
2. `calibration_limited`: the oracle is empirically feasible, but the deployed calibrated operating point fails the empirical joint requirement.
3. `certification_limited`: the deployed empirical operating point passes, but the exact simultaneous certification bounds fail.
4. `certified`: the exact simultaneous certification bounds pass.

If a deployed threshold empirically succeeds while the exhaustive oracle reports infeasibility, E1 raises an error instead of assigning a label.

## E0 reconstruction guard

Before any oracle result is accepted, E1 reconstructs the frozen E0 detector and verifies agreement with E0 on:

- deployed threshold,
- recall,
- FPR,
- TP/FN/FP/TN,
- conformal rank,
- conformal regime,
- calibration size.

Any mismatch aborts the experiment.

## Outputs

E1 writes only to `outputs/e1_oracle_decomposition/`:

- `e1_seed_results.csv`
- `e1_summary.csv`
- `e1_manifest.json`

The summary reports per detector and commissioning size:

- empirical AUROC,
- deployed recall/FPR,
- oracle recall at the FPR budget,
- oracle feasibility rate,
- oracle-minus-deployed recall,
- empirical and certified success rates,
- counts and rates for each bottleneck class.

## Interpretation boundary

Oracle success is evidence only that the requested operating point exists in the observed evaluation score geometry for the same score-threshold family. It is not a deployable calibration method and does not provide future-sample FPR or recall guarantees.
