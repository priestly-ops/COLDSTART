# M3 Structural Score Equivalence Audit

Date: 2026-08-12

## Scope

This note freezes the current M3 mechanism-audit evidence for the RACE branch. It is based on the existing frozen-smoke output directory:

`outputs/m3_n10_seeds0_4_equivalence/`

This is not the full commissioning grid. The manifest marks `full_frozen_protocol: false`; treat the result as a mechanism/stop-go audit for the current RACE branch.

## Protocol Evidence

- Manifest: `outputs/m3_n10_seeds0_4_equivalence/m3_manifest.json`
- Dataset hash: `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704`
- Git commit recorded by run: `6842b900869c35918c27c938f70dfefd413309b1`
- Commissioning grid: `N={10}`
- Seeds: `0,1,2,3,4`
- Calibration healthy size: `100`
- Healthy evaluation size: `100`
- Detectors: `TargetOnly`, `RACE`, `SourcePermutation`, `WeightPermutation`
- Frozen partition audit: `15` rows, `0` overlap failures

## Score Equivalence Result

Primary comparison: `TargetOnly` vs `RACE`, `score_split=eval`, across `15` source-regime/seed runs.

- Pearson correlation: mean `0.9999999645`, min `0.9999999221`
- Spearman correlation: mean `0.9999991975`, min `0.9999985793`
- Kendall tau: mean `0.9997929249`, min `0.9996658312`
- Affine fit R2: mean `0.9999999289`, min `0.9999998442`
- Threshold ratio: mean `0.5004968372`, min `0.5002334312`, max `0.5010767256`
- Pairwise order changes: mean `37.8`, min `17`, max `61`

Interpretation: RACE scores are essentially an affine/scalar rescaling of TargetOnly scores in this smoke setting. The approximate factor-of-two conformal threshold shift is matched by an approximate factor-of-two score rescaling.

## Prediction Equivalence Result

Primary comparison: `TargetOnly` vs `RACE`, across `15` paired runs.

- Changed predictions total: `3`
- Changed healthy predictions total: `0`
- Changed anomaly predictions total: `3`
- Maximum changed predictions in any run: `1`

Interpretation: the audit does not prove literal decision identity for every run, but it does prove no meaningful operational decision shift. The changed predictions are rare, anomaly-only, and do not alter the scientific decision.

## Operational Result

Decision file: `outputs/m3_n10_seeds0_4_equivalence/m3_decision.json`

Decision: `NO_MEANINGFUL_TRANSFER_BENEFIT`

Reason: `RACE is essentially tied with TargetOnly across transferability regimes`

Paired recall deltas, RACE minus TargetOnly:

- All regimes: mean `-0.0002649007`, median `0.0`
- High shift: mean `-0.0005298013`, median `0.0`
- Low shift: mean `-0.0002649007`, median `0.0`
- Moderate shift: mean `0.0`, median `0.0`

All paired FPR deltas in the decision table are `0.0`.

## Reviewer-Facing Claim Status

Supported for this smoke audit:

- Source transfer changes internal score scale and conformal threshold magnitude.
- The score transformation is almost perfectly affine relative to TargetOnly.
- Conformal calibration largely cancels the score-scale change at the decision level.
- Nonzero transfer weights are insufficient evidence of useful source transfer.
- The current RACE branch should not be claimed to reduce commissioning sample complexity.

Not supported by this smoke audit:

- RACE outperforms TargetOnly.
- Transferability predicts operational benefit.
- The conclusion generalizes to the full commissioning grid.
- SS-RACE is validated.

## Stop/Go Implication

The current RACE branch is scientifically closed as a positive-transfer mechanism unless a predeclared SS-RACE validation shows a genuine decision-level benefit over TargetOnly and negative controls. The next required step is to freeze the SS-RACE specification and run a small predeclared validation whose manifest and outputs are complete.
