# E2 — MVT-Flow under the frozen COLDSTART protocol

## Purpose

E2 tests whether a stronger, dataset-specialized multivariate time-series representation changes the commissioning bottleneck pattern observed in E0/E1.

MVT-Flow is the normalizing-flow detector released with voraus-AD. The official implementation trains on normal robot time series, standardizes using training data only, pads cycles to the maximum training length, and scores each cycle with the flow negative log-likelihood. The paper configuration uses 70 epochs, batch size 32, four coupling blocks, clamp 1.2, Adam learning rate 8e-4, and the machine-signal group.

## Frozen COLDSTART rules

E2 does not modify the statistical protocol frozen in E0/E1:

- commissioning grid: N = {10, 25, 50, 100}
- commissioning seeds: 0..19
- evaluation seed: 42
- calibration size: 100 healthy target cycles
- healthy evaluation size: 100 target healthy cycles
- all frozen anomalous evaluation cycles
- recall target R0 = 0.90
- false-alert budget B = 0.01
- deterministic split-conformal calibration with alpha = B
- simultaneous confidence gamma = 0.95 with Bonferroni delta_R = delta_F = 0.025
- oracle uses the same scalar MVT-Flow score and the same strict score > threshold decision family
- oracle is diagnostic only

## Training-budget interpretation

The official voraus-AD paper trains MVT-Flow on the source PRE_A normal training partition. That is not the COLDSTART commissioning question. In E2, the main representation baseline is therefore **TargetOnly-MVTFlow**: the architecture and training recipe follow the official MVT-Flow implementation, but the training set is restricted to the N target commissioning cycles for each seed.

This distinction must be stated in the manuscript. E2 is not a claim that we reproduced the paper's original train/test AUROC protocol; it is a faithful architecture/optimization adaptation to the frozen commissioning protocol.

## Time-series preprocessing

For each seed/N:

1. Load machine signals, matching the official MVT-Flow signal group.
2. Fit per-signal mean and standard deviation using only target commissioning time points.
3. Standardize commissioning, calibration, healthy evaluation, and anomaly evaluation using those commissioning statistics.
4. Set sequence length to the maximum commissioning-cycle length.
5. Truncate longer non-training cycles to that length and right-pad shorter cycles with zeros after standardization, matching the behavior of the official loader.

No calibration or evaluation labels or values are used to fit normalization or the flow.

## Score

The MVT-Flow cycle anomaly score is the official per-sample flow loss:

score(x) = 0.5 * sum(z^2) - log|det J|.

Larger scores are more anomalous.

## Calibration and E1 decomposition

After training, score the 100 healthy calibration cycles and apply the already-frozen deterministic conformal threshold. Then score the frozen healthy and anomaly evaluation sets, compute exact certification, and run the E1 empirical oracle decomposition over the same scalar scores.

## Reproducibility

Each model run uses a deterministic seed derived from the global seed, commissioning seed, and N. CPU execution is supported. GPU execution is optional; deterministic algorithms are requested where supported.

Outputs are versioned under `outputs/e2_mvtflow/` and never overwrite E0/E1 artifacts.
