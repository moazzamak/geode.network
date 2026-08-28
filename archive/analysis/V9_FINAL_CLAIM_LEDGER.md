# GEODE v9 Final Claim Ledger

**Program:** Surface Support Versus Volumetric Containment  
**Final outcome:** D  
**Finalized:** 28 July 2026

## Registered question

V9 tested whether frozen DINOv2 class support was better represented by
proximity to an existing primitive boundary or to a bounded low-dimensional
manifold tube than by volumetric containment. It did not reopen v6.1 accuracy
parity, v7 composition, or v8 adaptation-utility outcomes.

## Executed evidence

### M51: frozen-component shell occupancy

Across seeds 11, 23, and 37, neither normalized nor metric-corrected fields
produced a meaningful negative own-class interior for any of 48
class-by-seed-by-score diagnostics. The registered equal-mass near-surface and
deep-interior comparison therefore could not be formed. Supporting class
fractions were 0% on every seed, the precision/occupancy practical difference
was 0 points, and 0/9 directional cells passed.

This closes the codimension-one shell hypothesis for the frozen A2 zero level
sets. It does not imply that frozen features fill ambient volumes.

### M53-S1: bounded manifold tubes

The distinct tube screen fit one rank-8, rank-16, or rank-32 bounded affine
patch per known class using geometry data and calibration-only tangent extents.
Against the frozen A2 signed-volume baseline:

| Method | Known balanced accuracy | Unknown recall |
|---|---:|---:|
| Frozen A2 signed volume | 91.750% | 60.5% |
| Rank-8 bounded tube | 91.250% | 87.0% |
| Rank-16 bounded tube | 93.125% | 92.5% |
| Rank-32 bounded tube | 93.375% | 91.5% |
| Rank-32 Gaussian control | 95.125% | 87.0% |
| kNN support control | 91.875% | 76.0% |
| RBF control | 96.250% | 16.0% |

Ranks 16 and 32 passed the one-point predictive screen and improved unknown
recall. Nevertheless, bounded and unbounded residual outputs were identical on
the measured observations. Under synthetic tangent extrapolation, every
bounded rank accepted 100% of probes at 8x its fitted extent. Mean own-component
scores at 8x remained far below the calibrated acceptance thresholds.

The registered open-space kill switch therefore stopped every rank before S2.
No post-hoc penalty rescaling or additional rank search was permitted.

## Final branch dispositions

| Branch | Disposition |
|---|---|
| H1 frozen shell occupancy | Stopped at M51 |
| M52 frozen shell score comparison | Blocked by M51 |
| M53 fitted shell | Blocked by M51/M52 |
| H2 bounded tube S1 | Predictive signal; stopped by open-space safety |
| M53 bounded tube S2 | Blocked |
| M54 lifecycle utility | Blocked |
| M55 artifact-only replay | Complete |

## Final interpretation

V9 has **Outcome D**: a reproducible lower-dimensional residual signal existed,
but the registered practical safety gate failed. The evidence supports two
narrow conclusions:

1. the frozen A2 zero level sets were not calibrated class-support boundaries;
2. low-dimensional residual proximity improved the seed-11 predictive and OOD
   measurements, but the tested tangent bound was too weak relative to the
   calibrated score scale to control open space.

V9 does not establish that deep features lie on a hypersurface, that the tested
tube is a generative manifold, or that a safe tube would improve lifecycle
utility.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m experiments.tier4.verify_v9_final
```

The verifier reads only immutable indexes and evidence JSON. It loads no
training features and opens no final labels.
