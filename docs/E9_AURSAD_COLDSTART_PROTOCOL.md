# E9 — AURSAD COLDSTART external validation

## Purpose

E9 tests whether the representation → calibration → certification decomposition observed on voraus-AD persists on AURSAD, while taking advantage of AURSAD's larger pool of independent healthy tightening executions.

## Frozen operating specification

- Recall target: `R0 = 0.90`
- False-alert budget: `B = 0.01`
- Joint confidence: `gamma = 0.95`
- Commissioning-draw success target: `q0 = 0.80`
- Seeds: `0..19`

## Statistical unit

The unit is one complete AURSAD `sample_nr` execution. Time samples inside an execution are never treated as independent certification observations.

## Data allocation

Build a new protocol in `reports/aursad/coldstart_v2/` using the already-audited episode inventory:

- 500 fixed healthy calibration executions
- 368 fixed healthy certification/evaluation executions
- remaining 552 healthy executions as the commissioning reservoir
- all 625 eligible anomaly tightening executions as anomaly evaluation
- supplementary-operation label 5 excluded

Commissioning grid:

`N = [10, 25, 50, 100, 250, 500]`

Calibration-prefix grid:

`Mcal = [50, 99, 100, 198, 199, 250, 500]`

The 198→199 pair is the predeclared 1% split-conformal maximum-to-submaximum transition.

## Protocol construction

Run:

```powershell
python -m experiments.build_aursad_protocol `
  --output-dir reports/aursad/coldstart_v2 `
  --calibration-count 500 `
  --healthy-eval-count 368 `
  --grid 10,25,50,100,250,500 `
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 `
  --master-seed 42 `
  --overwrite
```

The builder must report pairwise overlap count zero and preserve fixed calibration/evaluation membership across seeds with nested commissioning sets.

## Detector

Primary external-validation detector: `TargetOnly-Stat`.

For each N/seed:

1. Fit the feature preprocessor on commissioning executions only.
2. Fit TargetOnly Ledoit-Wolf Gaussian on those commissioning features only.
3. Score the full 500-cycle fixed healthy calibration pool once.
4. Score the fixed 368 healthy evaluation executions once.
5. Score all fixed anomaly executions once.
6. Reuse those identical scores across all calibration prefixes.

No anomaly labels are used for training or threshold calibration.

## Calibration

Use the frozen deterministic split-conformal rule:

`k = ceil((Mcal + 1) * (1 - B))`

If `k > Mcal`, threshold is `+inf`.

Expected 1% regimes:

- M=50: infinite
- M=99: maximum
- M=100: maximum
- M=198: maximum
- M=199: submaximum
- M=250: submaximum
- M=500: submaximum

Alarm iff `score > threshold`.

## Oracle bottleneck decomposition

For every N/seed, evaluate the retrospective empirical oracle over the same scalar score and strict `score > threshold` family.

Classification precedence:

1. representation-limited if oracle cannot meet R0/B
2. calibration-limited if oracle is feasible but deployed empirical threshold fails
3. certification-limited if deployed empirical endpoint passes but exact bounds fail
4. certified otherwise

The oracle is diagnostic only and never used for deployment.

## Certification

Use the same exact one-sided Clopper-Pearson bounds as E0–E5, with Bonferroni allocation of the 5% familywise error.

The fixed healthy evaluation size is deliberately `n_H=368`, because under zero observed false positives this is the first sample size that can certify FPR≤1% at simultaneous 95% confidence.

## Fault-specific reporting

Report aggregate anomaly recall and class-specific recall for labels 1–4. Damaged-thread (`label=4`) has only three executions and must remain descriptive only; no strong class-specific certification claim is permitted for that class.

## Outputs

`outputs/aursad/e9_coldstart/`

- `e9_seed_results.csv`
- `e9_summary.csv`
- `e9_per_class_recall.csv`
- `e9_nstar.csv`
- `e9_manifest.json`

## Interpretation guardrails

- AURSAD is external task/domain validation, not proof of cross-robot transfer unless an explicit source→target transfer experiment is run.
- Certification observations are complete executions, not 100 Hz rows.
- Bootstrap duplicates do not count as independent certification evidence.
- Calibration-prefix sensitivity reuses a single frozen score vector and is not hyperparameter tuning.
