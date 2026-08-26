# E17 — RACE Lambda Sensitivity Protocol

## Purpose

E17 is a reviewer-facing robustness audit of the frozen RACE shrinkage strength. It does **not** retune RACE and does **not** redefine the primary method after observing results.

The original frozen RACE setting remains `lambda_reg = 60`.

## Dataset and frozen operating point

- Dataset: voraus-AD
- Source: PRE_A / setting 72
- Target: PRE_B / setting 73
- Commissioning sizes: `N = {10, 25, 50, 100}`
- Commissioning seeds: `0..19`
- Calibration size: 100 healthy target executions
- Healthy evaluation size: 100 target executions
- Recall target: 0.90
- False-alert budget: 0.01
- Joint certification confidence: 0.95
- Evaluation seed: 42

The same leakage-safe frozen split generator used by the primary experiments is retained.

## Sensitivity grid

The predeclared robustness grid is:

`lambda = {0, 10, 30, 60, 100, 300, +infinity}`

For target commissioning size `N`, the RACE target weight is

`w_T = N / (N + lambda)`

with the algebraic limits:

- `lambda = 0`: `w_T = 1`, the target-Gaussian endpoint of the RACE family.
- `lambda = +infinity`: `w_T = 0`, the source-Gaussian endpoint of the RACE family.

The endpoint runs retain the pooled RACE feature preprocessing. They therefore should be described as **RACE-family endpoints**, not silently substituted for the standalone TargetOnly or SourceOnly baseline pipelines.

## Frozen-method equivalence guard

`src/race_lambda_sensitivity.py` implements the sensitivity family separately so that the frozen `RACEDetector` code is not modified. A regression test requires `lambda=60` to reproduce the frozen RACE location, precision, target weight, and anomaly scores on identical inputs.

## Metrics

For every `(N, seed, lambda)` cell, record:

- target weight
- deployed recall
- deployed FPR
- AUROC
- empirical oracle recall at FPR <= 0.01
- empirical oracle feasibility
- exact post-freeze certification bounds
- empirical success
- certified success
- bottleneck classification

## Paired inference

For every `N`, each lambda is compared with the frozen `lambda=60` reference on identical commissioning seeds.

For each of the following outcomes:

- oracle recall at FPR <= 0.01
- AUROC
- deployed recall
- deployed FPR

report:

- mean paired delta
- median paired delta
- 95% percentile bootstrap CI over paired commissioning-draw differences
- Wilcoxon signed-rank p-value
- exact sign-flip p-value when there are at most 20 non-zero pairs
- positive / negative / neutral pair counts

Bootstrap repetitions: 10,000. Monte Carlo sign-flip fallback: 100,000. Global reproducibility seed: 42.

## Interpretation guard

This sweep answers only whether the qualitative RACE conclusion is sensitive to the originally fixed shrinkage strength. It must not be used to choose a post-hoc best lambda and then present that selected value as the primary RACE method.

If another lambda performs better, it may be reported as sensitivity evidence. The paper's primary RACE result remains `lambda=60`.

## Outputs

The runner writes to `outputs/e17_race_lambda_sensitivity/`:

- `e17_seed_results.csv`
- `e17_summary.csv`
- `e17_paired_inference_vs_lambda60.csv`
- `e17_manifest.json`

The seed-results file is checkpointed incrementally. Re-running the same command resumes completed `(N, seed, lambda)` cells unless `--no-resume` is supplied.

## Commands

Smoke test:

```powershell
python experiments/run_e17_race_lambda_sensitivity.py --n-values 10 --seeds 0,1 --lambdas 0,60,inf
```

Full frozen run:

```powershell
python experiments/run_e17_race_lambda_sensitivity.py
```
