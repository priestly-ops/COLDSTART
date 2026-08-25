# E3 — Calibration-size phase transition

## Goal

E3 tests whether the TargetOnly failure mode identified by E1 at larger commissioning size is caused by finite calibration-tail granularity rather than by score representation.

The deployed rule remains frozen:

- anomaly iff `score > threshold`
- alpha = B = 0.01
- rank `k = ceil((Mcal + 1) * (1 - alpha))`
- if k > Mcal, threshold = +inf
- if k = Mcal, threshold = maximum healthy calibration score
- if k <= Mcal - 1, threshold is submaximum

No anomaly labels are used for calibration.

## Dataset constraint

voraus-AD provides 319 target-healthy cycles in the target setting. Under the frozen COLDSTART protocol, retaining 100 independent healthy evaluation cycles and N=100 commissioning cycles leaves only 119 independent cycles for calibration. Therefore a leakage-safe N=100 experiment cannot reach the first submaximum transition at Mcal=199.

E3 does not reuse commissioning or evaluation episodes to force a larger calibration set.

## Arm A: primary calibration-limited experiment

- detector: TargetOnly-Stat
- N = 100
- seeds = 0..19
- fixed healthy evaluation size = 100
- Mcal = {25, 50, 75, 99, 100, 119}

This arm directly tests the N=100 setting where E1 found all TargetOnly replicates calibration-limited.

The exact E0 calibration set is preserved at Mcal=100. Smaller sizes are deterministic nested subsets of that same frozen set. Mcal=119 adds only previously unused independent target-healthy cycles.

## Arm B: structural order-statistic transition

- detector: TargetOnly-Stat
- N = 10
- seeds = 0..19
- fixed healthy evaluation size = 100
- Mcal = {150, 198, 199, 209}

N=10 leaves enough independent healthy cycles to observe the deterministic conformal transition:

- Mcal=198 -> rank 198 -> maximum calibration score
- Mcal=199 -> rank 198 -> first submaximum threshold

Because E1 found N=10 to be largely representation-limited, this arm is used to verify threshold mechanics, not to claim that larger calibration alone fixes detector performance.

## Outputs

`outputs/e3_calibration_phase/`

- `e3_seed_results.csv`
- `e3_summary.csv`
- `e3_manifest.json`

Each row includes the threshold regime, empirical recall/FPR, exact certification bounds, oracle feasibility, and COLDSTART bottleneck label.

## Interpretation

The primary evidence is whether increasing Mcal changes TargetOnly's empirical operating point while its fitted score and frozen evaluation set remain unchanged for a given seed. The oracle is constant across Mcal within a seed and serves as a guard that only calibration changed.
