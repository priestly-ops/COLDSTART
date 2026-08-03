# Same-N Outlier Remove-and-Replace Ablation

Protocol: `outlier-ablation-v1.1.0-coldstart-api`. Each intervention preserves N=100 and holds calibration and evaluation memberships fixed.

| Seed | Detector | Condition | Removed | Replacement | Recall | Δ Recall | FPR | Δ FPR | Success |
|---:|---|---|---:|---:|---:|---:|---:|---:|:---:|
| 4 | RACE | ABLATE_SUSPECT | 1840 | 1987 | 0.8437 | +0.0344 | 0.0000 | +0.0000 | No |
| 4 | RACE | ORIGINAL | — | — | 0.8093 | +0.0000 | 0.0000 | +0.0000 | No |
| 4 | RACE | RANDOM_CONTROL | 1709 | 1799 | 0.8119 | +0.0026 | 0.0000 | +0.0000 | No |
| 4 | TargetOnly | ABLATE_SUSPECT | 1840 | 1987 | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |
| 4 | TargetOnly | ORIGINAL | — | — | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |
| 4 | TargetOnly | RANDOM_CONTROL | 1709 | 1799 | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |
| 19 | RACE | ABLATE_SUSPECT | 1962 | 1736 | 0.7801 | +0.0026 | 0.0000 | +0.0000 | No |
| 19 | RACE | ORIGINAL | — | — | 0.7775 | +0.0000 | 0.0000 | +0.0000 | No |
| 19 | RACE | RANDOM_CONTROL | 1728 | 1840 | 0.7550 | -0.0225 | 0.0000 | +0.0000 | No |
| 19 | TargetOnly | ABLATE_SUSPECT | 1962 | 1736 | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |
| 19 | TargetOnly | ORIGINAL | — | — | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |
| 19 | TargetOnly | RANDOM_CONTROL | 1728 | 1840 | 0.0026 | +0.0000 | 0.0000 | +0.0000 | No |

## Matched-control interpretation

- Seed 4, TargetOnly: both removals produced the same recall change (difference +0.0000).
- Seed 4, RACE: suspect removal improved recall more than random control (difference +0.0318).
- Seed 19, TargetOnly: both removals produced the same recall change (difference +0.0000).
- Seed 19, RACE: suspect removal improved recall more than random control (difference +0.0252).

These are controlled seed-level mechanism tests, not proof of a general causal effect.
