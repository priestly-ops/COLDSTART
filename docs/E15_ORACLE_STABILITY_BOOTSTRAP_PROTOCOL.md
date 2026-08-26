# E15 — Oracle Stability Bootstrap

## Purpose

E15 asks a narrow reviewer-facing question: **are the empirical low-FPR oracle conclusions stable to the finite held-out evaluation sample?**

The experiment is a retrospective sample-stability audit. It does **not** tune detectors, choose thresholds for deployment, redefine the frozen commissioning result, or replace the paper's exact post-freeze certification procedure.

## Frozen setting

- Dataset: voraus-AD
- Source: setting 72
- Target: setting 73
- Commissioning sizes: `N = {10, 25, 50, 100}`
- Commissioning seeds: `0..19`
- Fixed evaluation seed: `42`
- Healthy evaluation size: `100` executions
- Recall target: `0.90`
- False-alert budget: `0.01`
- Detectors:
  - TargetOnly
  - RACE with the original frozen `lambda = 60`

The statistical unit is the robot execution/cycle, never individual 100 Hz rows.

## Oracle quantity

For held-out healthy scores `H` and anomaly scores `A`, E15 uses the already-frozen retrospective oracle quantity

`R_oracle(B) = max_t Recall(t) subject to empirical FPR(t) <= B`.

For the repository-wide rule `score > threshold`, the maximum-recall threshold at budget `B` is determined by the healthy-score order statistic corresponding to `floor(B * n_H)` allowed false positives. The E15 implementation checks every observed fast-oracle value against `src.oracle_feasibility.empirical_oracle_feasibility` before reporting bootstrap results.

## Bootstrap

For every `(detector, N, commissioning seed)` cell:

1. Fit the frozen detector exactly as in the main protocol.
2. Score the fixed held-out healthy and anomalous evaluation executions.
3. Independently resample healthy evaluation executions with replacement at size `n_H`.
4. Independently resample anomalous evaluation executions with replacement at size `n_A`.
5. Recompute `R_oracle(0.01)` for each bootstrap replicate.
6. Repeat `5,000` times with deterministic seeds.

Reported per-cell diagnostics include:

- observed empirical oracle recall;
- bootstrap mean, standard deviation, and median;
- 95% percentile bootstrap sensitivity interval;
- bootstrap fraction with oracle recall at least `0.90`;
- interval relation to the target: entirely below, straddling, or entirely above `0.90`.

## Interpretation guard

The bootstrap interval is **not** a population confidence interval and is **not** a post-freeze certification bound. It quantifies sensitivity to the finite fixed evaluation sample under iid-style resampling of the observed evaluation executions. It does not establish the sampling assumptions required for exact deployment certification.

In particular:

- E15 does not replace the Clopper–Pearson certification analysis;
- E15 does not alter the primary `N*` result;
- E15 does not justify retuning RACE;
- E15 does not convert empirical oracle feasibility into a deployable thresholding guarantee;
- E15 should be described as evaluation-sample stability / bootstrap sensitivity.

## Outputs

The runner writes:

- `outputs/e15_oracle_stability/e15_seed_oracle_stability.csv`
- `outputs/e15_oracle_stability/e15_summary.csv`
- `outputs/e15_oracle_stability/e15_manifest.json`

## Commands

Smoke test:

```powershell
python experiments/run_e15_oracle_stability_bootstrap.py --n-values 100 --seeds 0,1 --bootstrap-repetitions 500
```

Full frozen run:

```powershell
python experiments/run_e15_oracle_stability_bootstrap.py
```

The runner checkpoints each completed `(detector, N, seed, bootstrap_repetitions)` cell and resumes by default.
