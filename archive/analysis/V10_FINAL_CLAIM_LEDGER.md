# GEODE v10 Final Claim Ledger

**Program:** Safety-Calibrated Higher-Codimension Support  
**Final outcome:** D  
**Finalized:** 28 July 2026

## Registered question

V10 tested whether frozen DINOv2 class features exhibit useful
higher-codimension support that can be represented by calibration-safe bounded
affine tubes or, conditionally, a local atlas. It corrected the score-unit and
open-space failures diagnosed in v9; it did not reopen prior accuracy,
composition, or lifecycle outcomes.

## Executed evidence

### M56: protocol and score-unit lock

M56 implemented dimensionless orthogonal residuals, rank-normalized tangent
overshoot, calibration-only smallest-feasible penalty selection, deterministic
probe families, source-versus-system acceptance, typed lineage records, and
resource accounting. All operands passed and evidence replayed exactly.

### M57: controlled identifiability

Across three seeds, ambient-64 straight tubes recovered ranks 8, 16, and 32
exactly in all nine cells. Pooled independent in-support coverage was 91.35%,
and all 8x tangent probes were rejected. Curved and two-mode supports benefited
from two local patches, while Gaussian-volume, spherical-shell,
random-orientation, and random-label controls behaved as registered.

These results validate the implementation under controlled conditions. They do
not establish higher-codimension support in frozen deep features.

### M58: seed-11 global affine screen

All 18 registered rank-by-extent-by-scale cells were executed. Sixteen were
calibration-infeasible: their system-level 4x tangent acceptance remained above
1% even at the maximum penalty, and several rank-16/32 cells also accepted 8x
probes.

Only rank 8 with 0.99 tangent extents was feasible. The median-overshoot cell
regressed known balanced accuracy by 5.75 points. The interquantile-range cell
improved known balanced accuracy by 0.875 points, below the registered
1.0-point screen gate. Both feasible cells passed all open-space and safety
operands, but predictive utility was co-primary. No cell was retained.

## Final branch dispositions

| Branch | Disposition |
|---|---|
| M56 protocol and score-unit lock | Complete |
| M57 synthetic identifiability | Complete |
| M58 global affine screen | Stopped; 0/18 cells retained |
| M59 three-seed affine confirmation | Blocked by M58 |
| M60 local atlas | Closed; opening condition absent |
| M61 lifecycle utility | Blocked; no retained model |
| M62 artifact-only replay | Complete |

## Final interpretation

V10 has **Outcome D**: a small reproducible residual-support signal remained,
but practical predictive or calibration-safety gates failed. The evidence
supports three narrow conclusions:

1. the dimensionless bounded-tube implementation can identify and safely bound
   controlled higher-codimension supports;
2. most registered affine geometries were not calibration-safe on frozen
   seed-11 features because system-level component masking persisted; and
3. the best safety-feasible global tube did not reach the registered predictive
   improvement required for confirmation.

V10 does not establish the true data manifold, validate shell support, justify
a local atlas on the real features, or support lifecycle deployment. It also
does not universally rule out nonlinear or differently represented manifolds;
those remain untested rather than supported.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m experiments.tier4.verify_v10_final
```

The verifier reads only immutable indexes and evidence JSON. It loads no
training features and opens no final labels.
