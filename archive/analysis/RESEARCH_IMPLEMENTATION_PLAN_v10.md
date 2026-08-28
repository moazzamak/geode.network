# GEODE Research Implementation Plan v10

## Safety-Calibrated Higher-Codimension Support

**Status:** Complete; final Outcome D
**Date:** 28 July 2026  
**Claim ledger:** `analysis/CLAIM_LEDGER_v10.md`  
**Immutable parent:** v9 Outcome D

## 1. Motivation

V9 did not rule out higher-codimension support. Its rank-16 and rank-32
orthogonal-residual models improved seed-11 known balanced accuracy and unknown
recall, but the tangent penalty remained much smaller than the calibrated
acceptance threshold. Bounded and unbounded models therefore behaved
identically and accepted every 8x tangent extrapolation probe.

V10 tests whether that failure came from:

1. the absence of higher-codimension support;
2. incompatible score units and ineffective tangent calibration; or
3. curvature/multimodality that one global affine patch cannot represent.

The program first fixes dimensional consistency, then tests one affine patch,
then conditionally tests a local atlas. It does not revisit shell fitting.

## 2. Design principles

1. **Safety is calibrated before development evaluation.** Tangent penalties
   are selected using calibration-known observations and synthetic probes only.
2. **Residual and tangent terms share dimensionless scale.** This addresses the
   exact v9 failure without tuning to its development result.
3. **Complexity increases conditionally.** A local atlas opens only after a
   global-patch result diagnoses curvature or bridge acceptance.
4. **Open-space behavior is co-primary.** Accuracy cannot compensate for failed
   tangent rejection.
5. **Lifecycle evaluation remains last.** No adaptation claim follows from
   predictive or OOD evidence alone.

## 3. Frozen data and partitions

Reuse the v9 seed-specific identities:

- `geometry_fit`: tangent bases, centers, and residual geometry;
- `score_calibration`: residual scales, tangent extents, penalty feasibility,
  and the 92% known-coverage threshold;
- `development_eval`: known balanced accuracy and class diagnostics;
- `unknown_eval`: classes 8-9 only;
- `episode_validation`: conditional lifecycle evaluation;
- `final_confirmation`: sealed.

M56 must hash-lock every source manifest, split ID, A2 parent, v7 Gaussian
control, v9 evidence/index, score grid, gate, and synthetic-probe generator.
Overlap or lineage mismatch fails closed.

## 4. Model families

### 4.1 Global affine tube

Fit one PCA tangent patch per known class at ranks 8, 16, and 32. Estimate:

- center and basis from `geometry_fit`;
- orthogonal residual scale from `score_calibration`;
- tangent extent and outer scale from calibration quantiles;
- the smallest safety-feasible penalty from the registered grid.

The score is the dimensionless residual plus the rank-normalized tangent
overshoot defined in the claim ledger.

### 4.2 Unbounded residual diagnostic

Set the tangent penalty to zero. This arm can diagnose residual signal but can
never be retained because it is invariant to arbitrary tangent displacement.

### 4.3 Local bounded atlas

Conditionally fit 2 or 4 local patches per class:

1. cluster `geometry_fit` deterministically within class;
2. fit a tangent basis and residual scale per patch;
3. assign calibration observations by nearest geometry-only patch;
4. estimate patch-specific extents without development labels;
5. score by minimum calibrated patch score.

Patch count, rank, and total parameters must remain under the registered A2
budget multiplier. Empty or rank-infeasible patches fail closed.

## 5. Synthetic safety suite

For every patch, generate label-free probes:

1. **axis tangent:** 0.5x, 1x, 2x, 4x, and 8x each tangent extent;
2. **corner tangent:** simultaneous displacement along multiple tangent axes;
3. **normal:** fixed tangent coordinate with increasing orthogonal residual;
4. **mixed:** tangent extrapolation plus in-support normal residual;
5. **bridge:** interpolation and extrapolation between distinct same-class
   patches;
6. **cross-class bridge:** interpolation between nearest competing patches;
7. **random direction:** norm-matched random ambient directions.

Report both own-patch and minimum-system scores. This prevents another model
component from silently accepting a probe rejected by its source patch.

## 6. M56 — Protocol and score-unit lock

**Execution:** unconditional

Implement:

- versioned tube calibration and safety-evidence schemas;
- immutable partition and parent locks;
- dimensionless residual and tangent score utilities;
- deterministic probe generation;
- exact threshold and penalty lineage;
- parameter, fit-work, latency, and temporary-memory accounting.

Required tests:

- invariance to a common rescaling of feature units;
- rank normalization for equal per-axis overshoot;
- zero tangent penalty inside extents;
- monotonic penalty outside extents;
- global minimum score cannot hide source-patch rejection in reported system
  acceptance;
- deterministic PCA signs, clustering, extents, and probes;
- disjoint partitions and sealed-label enforcement;
- invalid/empty/rank-deficient patches fail closed;
- serialization and exact replay.

### 6.1 Result (28 July 2026)

M56 implemented the versioned calibration and safety records, dimensionless
residual/tangent score, deterministic patch assignments and probe families,
smallest-feasible penalty selection, frozen parent/partition locks, and resource
accounting contract. Eight focused tests passed.

The protocol qualification verified six immutable parents and all three v9
partition seeds. On the deterministic contract fixture, the smallest registered
penalty (1) preserved 92.5% calibration-known coverage at threshold 0.7495,
accepted 0% of 4x and 8x tangent probes, and replayed byte-identically. M57 is
open. This fixture verifies protocol mechanics only; it is not real-feature
evidence for the v10 hypothesis.

## 7. M57 — Synthetic identifiability

**Execution:** unconditional

Construct controlled ambient-64 datasets:

- straight rank-8, rank-16, and rank-32 tubes;
- curved arcs requiring a local atlas;
- two-mode same-class support;
- full-dimensional Gaussian volumes;
- shell-only negative control;
- random-label and random-orientation controls.

Measure rank recovery, in-support coverage, tangent rejection, residual
separation, and false manifold detection. Apply the M56-M57 gate. Failure blocks
all real-feature fitting because the implementation cannot distinguish the
registered geometries under controlled conditions.

### 7.1 Result (28 July 2026)

M57 passed all registered protocol and identifiability operands over seeds 11,
23, and 37. Ambient-64 straight tubes recovered true ranks 8, 16, and 32 exactly
in all nine cells, preserved 91.35% pooled independent in-support coverage, and
rejected 100% of 8x tangent probes. Normal-displacement median scores were at
least 43.6x in-support medians, and random-orientation residuals were at least
391.8x fitted residuals.

Two-patch atlases reduced curved-arc residual to 31.8--39.1% of the global
rank-8 residual and two-mode residual to 0.40--0.41%. Full-dimensional Gaussian
volumes and spherical shells retained more than 31% variance beyond rank 32,
while random-label accuracy remained 48.8--51.9% versus 100% for true labels.
Two independent executions produced byte-identical evidence. M58 is open; these
controlled results do not establish useful support in frozen deep features.

## 8. M58 — Seed-11 global affine screen

**Execution:** conditional on M57

For ranks 8, 16, and 32 and each registered extent policy:

1. fit geometry on `geometry_fit`;
2. select the smallest safety-feasible penalty on `score_calibration`;
3. freeze the threshold and all model state;
4. evaluate known, unknown, and synthetic endpoints once;
5. compare with A2, unbounded residual, Gaussian, kNN, and RBF controls.

No development-selected retry is allowed. Apply the M58 gate and retain at most
one cell using the preregistered tie-break.

### 8.1 Result (28 July 2026)

M58 evaluated all 18 registered seed-11 rank-by-extent-by-scale cells. Sixteen
cells were stopped by calibration safety: even at penalty 1024, their 4x
tangent acceptance remained 1.95--87.30%, and four rank-16/32 cells also
accepted 1.56--35.74% at 8x. Only rank 8 with 0.99 tangent extents was feasible
under either outer-scale policy.

The feasible median-overshoot cell regressed known balanced accuracy by 5.75
points. The feasible interquantile-range cell improved it by 0.875 points,
missing the registered 1.0-point gate, although both cells passed every
open-space, unknown-recall, accepted-known, class-coverage, replay, and resource
operand. No cell was retained. M59 is blocked, M60 does not open because the
failure was not specific to bridge or curvature behavior, and M61 is blocked.
M62 finalization is open.

## 9. M59 — Three-seed affine confirmation

**Execution:** conditional on M58

Replay the retained rank, extent, scale, and penalty-selection rule on seeds 11,
23, and 37. The numerical penalty may differ by seed only through the frozen
calibration-feasibility rule.

Report:

- paired per-example bootstrap intervals;
- seed-level effects;
- class-wise accuracy and confusion;
- known coverage/unknown recall curves;
- all synthetic acceptance families;
- Gaussian proximity;
- resource and replay evidence.

Apply the M59 gate. A global affine tube that fails is not retained.

## 10. M60 — Conditional local atlas

**Execution:** conditional

Open when either:

- M59 passes and an atlas may improve practical relevance; or
- M58/M59 shows residual signal but fails specifically on bridge/curvature
  diagnostics while tangent-axis safety is satisfied.

Screen patch counts 2 and 4 and ranks 8, 16, and 32 under a total component
budget no larger than the frozen A2 count. Use seed 11 for the cheap screen,
then all three seeds for one retained atlas.

Add bootstrap patch-stability diagnostics:

- assignment adjusted Rand index;
- principal-angle stability;
- extent variation;
- fraction of calibration observations changing patch.

Apply the M60 gate. Instability or open-space failure closes the atlas.

## 11. M61 — Conditional lifecycle evaluation

**Execution:** only after M59 or M60 passes

Reuse the frozen v8 nine episode cells, 50-label review budget,
coverage-selected review policy, threshold-transfer rule, exhaustive routing,
and transactional adapter. Change only the rejection/support score.

Compare:

1. v8 Gaussian;
2. retained v10 support model;
3. kNN-support diagnostic;
4. no-adaptation control.

Apply the M61 lifecycle gate. Preserve exact rollback, confirmation lineage,
known-class safety, remaining-unknown recall, and fallback behavior.

## 12. M62 — Finalization

Produce:

- `analysis/V10_FINAL_CLAIM_LEDGER.md`;
- immutable indexes for each executed milestone;
- a complete branch-disposition table;
- one artifact-only verifier;
- two byte-identical conclusion replays;
- paper/thesis amendments only after the outcome is frozen.

### 12.1 Result (28 July 2026)

M62 finalized Outcome D. The artifact-only verifier checked three immutable
indexes, nine indexed artifacts, seven branch dispositions, and ten conclusion
operands twice with byte-identical output. It loaded no training features and
opened no final labels. Every v10 branch is complete, stopped, closed, or
blocked.

## 13. Dependency graph

```text
M56 protocol + score units
        |
M57 synthetic identifiability
        |
M58 seed-11 affine screen
   | pass                | residual + curvature only
   v                     v
M59 three-seed affine   M60 local atlas
   |                     |
   +----------+----------+
              |
      eligible safe model?
        no -> M62 D/E
              |
             yes
              v
       M61 lifecycle
              |
              v
       M62 final replay
```

## 14. Required artifacts

1. v10 claim ledger and parent lock;
2. partition and feature manifest;
3. score-unit audit;
4. synthetic identifiability matrix;
5. calibration-feasibility table;
6. exact selected penalty and threshold lineage;
7. global affine rank screen;
8. tangent, normal, mixed, bridge, and random-direction stress evidence;
9. matched A2/Gaussian/kNN/RBF controls;
10. conditional atlas stability and predictive evidence;
11. conditional lifecycle records;
12. immutable final branch ledger and artifact-only replay.

## 15. Interpretation

| Result | Interpretation |
|---|---|
| Residual signal disappears after unit correction | v9 gain was not robust support evidence |
| Safe affine tube passes M59 | Higher-codimension affine support is predictive and bounded |
| Affine fails bridges, safe atlas passes | Support is locally low-dimensional but globally curved/multimodal |
| Accuracy improves but synthetic safety fails | Reproducible signal, unsafe open-space behavior; Outcome D |
| Synthetic suite passes but real OOD fails | Probe suite is insufficient; model is not eligible |
| Safe model improves lifecycle utility | End-to-end support-model result |
| No residual or atlas signal | Higher-codimension support is not the measured bottleneck |

The program tests an operational support model, not a metaphysical claim about
the true shape of the data distribution.
