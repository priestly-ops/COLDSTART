# SS-RACE Freeze Status

Date: 2026-08-12

## Status

SS-RACE is now frozen as a code-level method specification and has one completed seed-0 mechanism validation in `outputs/srace_small_validation/`. That validation is negative: it does not support a positive SS-RACE transfer claim. Do not make performance claims until the exact code state is cleanly frozen and rerun.

## Frozen Design Elements

Implementation: `src/srace.py`

- Target location is anchored to target commissioning healthy data: `mu_R = mu_T`.
- Source and target covariances are estimated with Ledoit-Wolf shrinkage.
- Transfer is direction-specific, not governed by a single global scalar.
- Shared directions are selected with a conservative effective-rank policy: transferred rank is capped below target commissioning `N`.
- Source-target structural compatibility is calculated from the squared projection of target covariance directions onto the source shared subspace.
- Distributional compatibility suppresses transfer when source and target projected variances disagree.
- Direction weights combine target uncertainty, structural compatibility, and distributional compatibility.
- Target-private directions keep target covariance structure by forcing zero transfer weight outside the shared rank.
- A healthy-only leave-one-out gate can fall back to TargetOnly without using anomaly labels.
- Diagnostics record transferred dimensions, compatibility, structural compatibility, shared rank, private dimensions, condition number, eigenvalue range, and shrinkage coefficients.

## Frozen Negative Controls

Runner: `experiments/run_srace.py`

The validation runner now includes:

- `TargetOnly`
- `OriginalRACE`
- `S-RACE`
- `S-RACE WrongSource`
- `S-RACE SourcePermutation`
- `S-RACE CompatibilityPermutation`

`WrongSource` uses the least compatible source according to healthy-only projector similarity among the predeclared source regimes for the same target split. Rows keep the primary source/target key for paired comparison and record `fit_source_pair_id` / `fit_source_group` for the actual source used.

## Required Output Artifacts

A completed S-RACE validation must produce:

- `srace_manifest.json`
- `srace_seed_results.csv`
- `srace_summary.csv`
- `srace_paired_deltas.csv`
- `srace_per_class_recall.csv`
- `srace_partition_audit.csv`
- `srace_mechanism_diagnostics.csv`
- `srace_transfer_weights.csv`
- `srace_source_compatibility.csv`
- `srace_score_equivalence.csv`
- `srace_score_scatter.csv`

The current `outputs/srace_small_validation/` directory contains these artifacts. See `reports/srace_seed0_stop_go.md` for the seed-0 stop/go decision.

## Verification

Focused tests passed:

`.\.venv311\Scripts\python.exe -m pytest tests\test_srace.py tests\test_m3_transfer_regimes.py -q`

Result: `12 passed`

Compile check passed:

`.\.venv311\Scripts\python.exe -m py_compile src\srace.py experiments\run_srace.py`

## Stop/Go Rule

After the first completed predeclared validation:

- If `S-RACE` improves decision-level commissioning outcomes over `TargetOnly` and beats `WrongSource` / permutation controls, proceed to the full grid.
- If `S-RACE` is structurally score-equivalent to `TargetOnly`, fails to beat controls, or only improves AUROC without satisfying the recall/FPR operating criterion, stop modifying RACE and freeze the negative scientific story.

Seed-0 validation result: the second condition holds. S-RACE is essentially TargetOnly-equivalent and does not beat negative controls under the operating criterion.
