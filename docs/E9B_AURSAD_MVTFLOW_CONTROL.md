# E9B — AURSAD MVT-Flow Representation Control

## Question

Does the AURSAD representation bottleneck observed with TargetOnly-Stat persist when the six-statistic cycle representation is replaced by a stronger raw multivariate time-series normalizing-flow score?

## Why MVT-Flow

The original AURSAD use-case paper reports strong supervised 1D ResNet performance, but that setup trains with labeled anomalies and therefore does not match COLDSTART's healthy-only commissioning premise. E9B instead reuses the audited MVT-Flow implementation already used in E2. MVT-Flow trains only on healthy target commissioning cycles and scores complete multivariate time-series executions.

This is a representation control, not a reproduction of the original AURSAD supervised benchmark and not a reproduction of the original MVT-Flow voraus-AD evaluation protocol.

## Frozen data protocol

E9B reuses `reports/aursad/coldstart_v2` exactly:

- commissioning grid: N = {10, 25, 50, 100, 250, 500}
- commissioning seeds: 0..19
- calibration: first 199 executions from the fixed E9 calibration pool
- independent healthy certification evaluation: 368 executions
- fixed anomaly evaluation: 625 executions
- damaged-thread label 4: descriptive only

No episode may appear in more than one partition.

## Signal policy

E9B uses the 48 measured AURSAD robot signals from `AURSAD_MEASURED_SIGNAL_COLUMNS`.

Excluded from model input:

- `sample_nr`
- label
- timestamp
- runtime state
- hard-coded control/reference variables

This avoids label/control leakage and follows the AURSAD paper's recommendation that annotation-oriented control variables not be used as ML features.

## Model

E9B reuses `src/mvtflow_adapter.py` and its frozen official-style configuration:

- 70 epochs
- batch size 32
- 4 coupling blocks
- clamp 1.2
- learning rate 8e-4
- scale 2
- first temporal kernel 13 with dilation 2
- milestones 11 and 61
- gamma 0.1

Per-signal normalization is fit using only the target commissioning cycles. Sequence target length is the maximum commissioning-cycle length for the replicate. Other cycles are truncated to this length and right-padded with zeros.

Score:

`0.5 * sum(z^2) - jacobian`

## Deployment and certification

Primary operating point:

- recall target: 0.90
- FPR budget: 0.01
- simultaneous confidence: 0.95
- split-conformal alpha: 0.01
- alarm rule: score > threshold

At Mcal=199, the strict deterministic conformal rule enters the submaximum regime.

Exact one-sided Clopper-Pearson certification uses the same COLDSTART backbone as E0-E9. With 368 independent healthy executions, zero observed false positives is sufficient for the FPR side of the 1% specification.

The empirical oracle is diagnostic only and may not be used to choose the deployed threshold.

## Predeclared interpretation

- oracle recall < 0.90 at FPR <= 0.01: representation-limited
- oracle feasible but deployed empirical point fails: calibration-limited
- deployed empirical point succeeds but exact bounds fail: certification-limited
- exact bounds pass: certified

The purpose is to compare E9 TargetOnly-Stat with E9B TargetOnly-MVTFlow under the same data allocation. No anomaly labels are used for training, hyperparameter selection, or calibration.

## Execution

Install MVT-Flow dependencies and initialize the official voraus submodule if needed:

```powershell
pip install -r requirements-mvtflow.txt
git submodule update --init --recursive
```

Protocol tests:

```powershell
python -m pytest tests/test_e9b_aursad_mvtflow_control.py tests/test_mvtflow_adapter.py tests/test_certification_backbone.py tests/test_oracle_feasibility.py -v
```

Small execution sanity run (not for reporting or model selection):

```powershell
python -m experiments.run_e9b_aursad_mvtflow_control --n-values 100,500 --seeds 0
```

Full predeclared run:

```powershell
python -m experiments.run_e9b_aursad_mvtflow_control
```

The runner checkpoints every completed N/seed replicate and resumes from the existing checkpoint.

Outputs are written to `outputs/aursad/e9b_mvtflow/`.
