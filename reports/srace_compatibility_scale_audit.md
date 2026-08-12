# S-RACE Compatibility Scale Audit

Date: 2026-08-12

## Question

Why does the S-RACE seed-0 validation report:

- `structural_compatibility_mean ~= 0.004`
- final mean transfer weights around `1e-3`

and is that a real source-target incompatibility, a dimensional/reporting artifact, an overly aggressive product of compatibility terms, safe-gate over-conservatism, or an implementation bug?

## Literature Anchor

Principal-angle compatibility has a specific mathematical scale. For two orthonormal subspace bases `U_s` and `U_t`, the cosines of the principal angles are the singular values of:

`U_s.T @ U_t`

Thus a canonical `cos^2(theta_j)` compatibility is a rank-level quantity in `[0, 1]` for paired principal directions. This is the standard Grassmann/principal-angle definition used by subspace alignment and geodesic-flow domain-adaptation work.

Relevant sources:

- Fernando et al., ICCV 2013, "Unsupervised Visual Domain Adaptation Using Subspace Alignment": source and target domains are represented as eigenvector subspaces and aligned by a mapping between those subspaces. https://openaccess.thecvf.com/content_iccv_2013/html/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.html
- Miao and Ben-Israel, Linear Algebra and its Applications 1992, "On principal angles between subspaces in Rn": principal angles define relative subspace position. https://www.sciencedirect.com/science/article/pii/0024379592902515
- Grassmann/geodesic-flow domain-adaptation descriptions compute principal-angle cosines from SVD of basis cross-products and use them as overlap geometry. Example: https://www.mdpi.com/2072-4292/8/3/234

## Current Implementation

Implementation: `src/srace.py`

The current implementation does **not** compute paired canonical `cos^2(theta_j)` via SVD of `U_s_shared.T @ U_t_shared`.

It computes a per-target-direction projection mass:

`structural_compatibility[j] = || source_shared.T @ target_vector_j ||^2`

Then it forces target-private directions to zero:

`structural_compatibility[shared_rank:] = 0`

For seed-0, `shared_rank = 4` and `n_features = 564`, so 560 of 564 directions are intentionally zeroed.

This projection statistic is not meaningless, but it should not be described as the principal-angle vector `cos^2(theta_j)`. It is better described as target-direction membership in the source shared subspace.

## Observed Scale

Artifacts:

- Original stop/go run: `outputs/srace_small_validation/`
- Diagnostic rerun with persisted pre-gate and principal-angle fields: `outputs/srace_seed0_diagnostics_v2/`

For `S-RACE` across the three source regimes:

- All rows: `1692` directions = `3 source regimes * 564 features`
- Directions with `structural_compatibility > 0`: `12`
- Directions with final `transfer_weight > 0`: `4`

All-dimension mean in the diagnostic rerun:

- `structural_compatibility_mean = 0.004258`
- `pre_gate_compatibility_mean = 0.002224`
- `pre_gate_weight_mean = 0.001483`
- `post_gate compatibility_mean = 0.000851`
- `post_gate transfer_weight_mean = 0.000567`

Active structural directions only in the diagnostic rerun:

- mean `structural_compatibility = 0.600440` across all three regimes
- mean canonical `principal_cos2 = 0.597069`, `0.592882`, and `0.611370` for near/moderate/high
- min/max `structural_compatibility = 0.191453 / 0.872350`

Active pre-gate transferred directions:

- mean `structural_compatibility = 0.611370`
- mean `pre_gate_compatibility = 0.360055` for the high-shift active directions
- active `pre_gate_transfer_weight` range `0.055069 / 0.434378`
- high-shift active weights survive the gate unchanged

## Interpretation

The headline `c_sub ~= 0.004` is mostly a reporting artifact from averaging over all 564 feature dimensions after intentionally zeroing 560 private dimensions.

It is **not** evidence that the active shared directions have near-zero source-target overlap. The active shared directions have large projection compatibility values, roughly `0.19` to `0.87`.

The final `1e-3` mean transfer weight is also mostly an all-dimension average plus post-gate fallback:

- Twelve pre-gate directions are active across the three source regimes.
- Only four directions remain active post-gate, all from the high-shift source.
- The active pre-gate weights are not tiny; they reach `0.434`.
- Mean over all dimensions is small because `4 / 564` directions carry transfer.

## Cause Breakdown

### Legitimate empirical finding: almost no source/target subspace overlap

Not supported by the current reported mean. Active shared directions overlap substantially. The all-dimension mean is not the right statistic for this claim.

### Incorrect principal-angle normalization

Partially supported. The implementation computes squared projection of each target eigenvector onto the source shared subspace, not SVD-based paired principal-angle `cos^2(theta_j)`.

This may be a valid directional membership score, but the paper/report should not call it principal-angle compatibility unless the SVD-based values are computed and logged separately.

### Comparing incompatible bases

Plausible terminology issue, not necessarily a numerical bug. The code compares the full target eigenbasis to the low-rank source subspace, then zeros private target directions. That is compatible with a "target-direction membership in source shared subspace" interpretation, but not with "paired source-target principal directions."

### Dimensionality artifact caused by `N=10`

Supported. The conservative rank cap gives `shared_rank = 4` for `N=10`, so at most 4 of 564 directions can transfer. Any mean over all directions will be near `rank / d`, even with strong active-direction compatibility.

For a random 4-dimensional subspace in 564 dimensions, the expected squared projection of a random unit direction onto that subspace is `4/564 ~= 0.0071`. An all-dimension mean around `0.004` is therefore in the same order as a low-rank-in-high-dimension reporting baseline. This is not, by itself, evidence of pathological incompatibility.

### Multiplicative compatibility terms shrinking too aggressively

Partially supported, but not as a threshold-tuning argument. The product of structural compatibility, variance agreement, and location agreement can be much smaller than any single term. However, in the active high-shift directions the post-product compatibility averages `0.36`, which is not collapsed.

Near/moderate S-RACE rows show final compatibility as zero because the safe gate falls back to TargetOnly after pre-gate evaluation. The current artifacts do not preserve pre-gate compatibility for those fallback rows, so the exact multiplicative shrinkage before gating cannot be reconstructed from CSV alone.

### Safe gate over-conservative

Resolved for the diagnostic rerun, but interpretation remains scientific rather than a license to tune.

`outputs/srace_seed0_diagnostics_v2/srace_mechanism_diagnostics.csv` records:

- near: `safe_gate_margin = -0.042890`, fallback `True`
- moderate: `safe_gate_margin = -0.029024`, fallback `True`
- high: `safe_gate_margin = 0.017760`, fallback `False`

The gate closes near/moderate because their leave-one-out healthy likelihood margins are below the allowed tolerance. The high-shift source passes the gate but still does not improve operational recall/FPR. This argues against blindly loosening the gate: even when it opens, the method remains effectively TargetOnly-equivalent.

### Implementation bug

No evidence of a numeric bug in the observed artifact scale. The tiny means follow directly from:

1. low shared rank (`4`) relative to feature dimension (`564`);
2. reporting means across all directions;
3. safe-gate fallback setting final weights to zero for two regimes.

There is, however, a semantic/API bug risk: `structural_compatibility` is currently named and discussed like principal-angle `cos^2(theta_j)`, but it is not computed by canonical principal-angle SVD.

## Non-Tuning Recommendations

Do **not** loosen thresholds or gates to improve RACE.

Instead, add diagnostics that preserve the scientific interpretation. This has now been implemented for future runs in `src/srace.py` and `experiments/run_srace.py`:

1. Log canonical principal-angle values:
   - `principal_cos2_j = svd(U_t_shared.T @ U_s_shared).singular_values ** 2`
   - mean/min/max over the shared rank.

2. Keep the current projection statistic, but rename/report it as:
   - `target_direction_source_subspace_projection`
   - not `principal_angle_compatibility`.

3. Report compatibility on two scales:
   - active/shared directions only;
   - all feature directions.

4. Log pre-gate and post-gate terms separately:
   - `structural_compatibility_pre_gate`
   - `variance_compatibility_pre_gate`
   - `location_compatibility_pre_gate`
   - `combined_compatibility_pre_gate`
   - `transfer_weight_pre_gate`
   - `transfer_weight_post_gate`

5. Log `safe_gate_margin` in `srace_mechanism_diagnostics.csv`.

6. Add a random-subspace baseline for the observed rank/dimension:
   - expected projection mean `r / d`;
   - optional permutation/null distribution using healthy-only features.

Implemented diagnostic fields include:

- `principal_cos2_*` run-level summaries from canonical SVD principal angles.
- `active_structural_compatibility_*` summaries over shared directions only.
- `variance_compatibility_*` and `location_compatibility_*` summaries.
- `pre_gate_compatibility_*` and `pre_gate_weight_*` summaries.
- `safe_gate_margin`.
- Per-direction `pre_gate_transfer_weight`, `variance_compatibility`, `location_compatibility`, and `pre_gate_compatibility`.

The diagnostic rerun in `outputs/srace_seed0_diagnostics_v2/` now persists these extra fields. It confirms that the compatibility scale explanation is not a numeric bug and that the safe gate, not near-zero active subspace overlap, explains the near/moderate post-gate collapse.

## Bottom Line

The `c_sub ~= 0.004` value should not be interpreted as "source and target shared directions have almost no overlap." It is primarily an all-dimension average after a rank-4 transfer policy in 564 dimensions.

The active shared directions have substantial overlap and active transfer weights are not `1e-3`; they average about `0.24` where transfer is actually active.

The current scientific issue is not "compatibility is truly near zero." The issue is that the method transfers in very few dimensions, falls back to TargetOnly in two regimes with negative healthy-only margins, and remains structurally/operationally equivalent to TargetOnly even in the one regime where the safe gate opens.
