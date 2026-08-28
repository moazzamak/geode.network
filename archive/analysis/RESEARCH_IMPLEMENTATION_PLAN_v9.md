# GEODE Research Implementation Plan v9

## Surface Support Versus Volumetric Containment

**Status:** complete; final Outcome D
**Date:** 27 July 2026  
**Claim ledger:** `analysis/CLAIM_LEDGER_v9.md`  
**Immutable parents:** v6.1 Outcome D, v7 Outcome C, and v8 Outcome D

## 1. Program decision

The existing GEODE classifier generally interprets negative SDF values as
support: a point deeper inside a primitive is at least as contained as a point
near its boundary. The accumulated evidence suggests this may be the wrong
topological assumption for frozen deep features:

- measured local intrinsic dimension was approximately 15--21 in a
  384-dimensional representation;
- kNN and RBF controls dominated volumetric heads;
- proper Gaussian density outperformed raw affine support;
- boundary- and coverage-aware review sets outperformed density cores;
- low-rank subspace primitives improved over ambient spheres without reaching
  parity.

V9 tests whether support is concentrated near a surface or bounded
lower-dimensional patch rather than throughout an enclosed ambient volume.
The program distinguishes a codimension-one shell from a higher-codimension
manifold tube and does not assume either is true.

## 2. Core experimental principle

The test proceeds in increasing order of flexibility:

1. **diagnose occupancy on frozen components;**
2. **change only score semantics;**
3. **fit shell/tube geometry only if warranted;**
4. **test lifecycle utility only after predictive and open-set qualification.**

This ordering isolates whether any gain comes from treating the same geometry
differently or from adding model capacity.

## 3. Frozen inputs and partitions

### 3.1 Parent artifacts

M51 must hash-lock:

- v6.1 A2 weighted affine components and class order;
- v7 rank-32 Gaussian head and thresholds;
- v7/v8 DINOv2 feature manifests;
- seeds 11, 23, and 37;
- all geometry, calibration, development, OOD, and episode partitions;
- RBF and kNN control predictions where already frozen;
- the v9 score definitions, rank grid, width grid, and statistical policy.

### 3.2 Partition roles

Each seed uses disjoint:

- `geometry_fit`: fit centers, bases, radii, tangent extents, and residual
  scales;
- `score_calibration`: estimate shell widths, score weights, temperatures, and
  known-coverage thresholds;
- `development_eval`: balanced accuracy and known-class diagnostics;
- `unknown_eval`: frozen leave-two-class-out unknown observations;
- `episode_validation`: conditional M54 utility only;
- `final_confirmation`: sealed.

No development or unknown example may influence rank, width, threshold, or
model selection. Partition overlap fails closed.

## 4. Support models

### 4.1 Frozen volume

Reuse the exact v6.1 A2 components, weights, fusion, and class order. This arm
is the score-semantics baseline and must reproduce its frozen predictions.

### 4.2 Frozen shell variants

Compute on the same components:

- absolute normalized field;
- absolute first-order metric-corrected field;
- asymmetric metric shell with separate inside/outside widths.

For multiple components, compare two preregistered fusions:

1. minimum shell distance;
2. normalized soft minimum of shell distance using the frozen A2 temperature.

No component may move in M52.

### 4.3 Bounded manifold tube

Reuse `src/subspace_primitive.py` and the existing rank-16/rank-32
infrastructure. For each component:

1. fit center and tangent basis from `geometry_fit`;
2. estimate perpendicular residual scale;
3. estimate robust tangent extents using calibration-safe quantiles;
4. score perpendicular residual plus a nonnegative outside-extent penalty;
5. prohibit any reward for moving arbitrarily far along the tangent subspace.

Registered ranks are 8, 16, and 32. A rank is ineligible when the geometry
partition has fewer than `rank + 2` independent supports.

### 4.4 Fitted shell

Only if M51 supports H1, allow boundary centers/radii to move under a two-sided
margin loss:

\[
L_{\mathrm{shell}} =
\mathbb{E}_{x\sim k}
\left[\rho\left(|d_k(x)|/\tau_k\right)\right]
+
\gamma\,
\mathbb{E}_{x\not\sim k}
\left[\rho\left((m-|d_k(x)|)/\tau_k\right)\right].
\]

The loss attracts own-class support to a finite-width shell and repels
competing support from that shell. It does not require own-class points to be
inside. Margin \(m\), width \(\tau\), and \(\gamma\) are selected from the
registered calibration grid only.

### 4.5 Controls

Run identical partitions for:

- signed volume;
- low-rank Gaussian;
- kNN support;
- RBF SVM;
- unbounded perpendicular residual;
- random orientation;
- label permutation;
- reversed score direction.

Random and permuted controls must perform near chance or trigger a protocol
audit before any positive interpretation.

## 5. M51 — Protocol lock and occupancy diagnosis

**Execution:** unconditional  
**Purpose:** determine whether existing components exhibit shell-like support
before introducing new fitting.

### 5.1 Implementation

Add a versioned `SurfaceSupportDiagnostic` record containing:

- component and representation hashes;
- score variant and direction;
- class, seed, and partition identity;
- normalized and metric-corrected signed depth quantiles;
- near-surface, deep-interior, and exterior counts;
- own-class precision and competing-class occupancy by stratum;
- unknown occupancy by stratum;
- width-selection provenance;
- exact selected IDs and replay hashes.

### 5.2 Band construction

For each class and seed:

1. estimate a near-surface width from `geometry_fit` only;
2. freeze equal-count near-surface and deep-interior strata;
3. evaluate stratum precision and competing occupancy on `development_eval`;
4. repeat with metric-corrected distance;
5. report classes with no meaningful negative interior separately.

The analysis must not choose the band that maximizes development separation.

### 5.3 Required tests

- sphere metric-shell values equal exact Euclidean radial gaps;
- anisotropic correction matches the existing M1.6 implementation;
- absolute shell is invariant to sign flip;
- signed volume is not invariant to sign flip;
- equal-count strata are deterministic;
- geometry/calibration/evaluation overlap is rejected;
- random-orientation and label-permutation controls replay;
- zero-gradient and center-point cases remain finite;
- fused shell scores preserve component-count normalization.

### 5.4 Gate

Apply the M51 gate frozen in `CLAIM_LEDGER_v9.md`. If H1 fails, do not fit a
learned shell. The bounded-tube diagnostic may continue because codimension-one
failure does not falsify higher-codimension support.

### 5.5 Result (28 July 2026)

M51 hash-locked the A2 weighted rank-32 students, the v7 Gaussian evidence, the
v8 final evidence, and disjoint geometry, calibration, known-development, and
proxy-unknown partitions for seeds 11, 23, and 37. The component-level
normalized and metric-corrected fields produced **0/48** class-by-seed-by-score
cells with enough negative own-class depth to construct the registered
equal-mass near-surface and deep-interior bands. All three gate operands
therefore failed: per-seed supporting class fractions were 0%, the registered
precision/occupancy difference was 0 points, and 0/9 directional cells were
consistent.

The evidence replayed byte-identically. M52 and learned-shell fitting are
closed; M53 remains open only for the distinct bounded-manifold-tube diagnostic.
This result means the A2 zero level sets are not calibrated support boundaries;
it does not test or falsify lower-dimensional tube support.

## 6. M52 — Score-only volume-versus-shell comparison

**Execution:** closed because M51 found no shell signal
**Purpose:** isolate score semantics from geometry fitting.

### 6.1 Factorial

For every seed, compare:

| Components | Score | Fusion | Calibration |
|---|---|---|---|
| frozen A2 | signed volume | frozen weighted softmin | frozen |
| frozen A2 | absolute normalized shell | min and frozen softmin | global temperature |
| frozen A2 | metric-corrected shell | min and frozen softmin | global temperature |
| frozen A2 | asymmetric metric shell | min and frozen softmin | inside/outside widths |

Report parameter-neutral comparisons separately from calibrated comparisons.

### 6.2 Evaluation

- balanced accuracy, NLL, Brier score, and ECE;
- class-wise accuracy and confusion;
- known coverage versus unknown recall curve;
- AUROC and FPR95;
- accepted-known accuracy;
- review precision at the frozen budget;
- score quantiles for correct, incorrect, and unknown examples;
- latency and peak temporary bytes.

### 6.3 Gate and stopping rule

Apply the M52 gate. If no shell score passes, close H1 with Outcome D or E.
Do not optimize shell centers after a score-only failure unless the preregistered
M51 occupancy diagnostic passed and the score-only effect is positive on at
least 2/3 seeds but misses only the 2-point practical threshold.

## 7. M53 — Bounded manifold-tube and fitted-shell study

**Execution:** complete at S1; fitted shell and S2 are closed.

### 7.1 Cheap S1 screen

Use seed 11 only. Compare:

- ranks 8, 16, and 32;
- perpendicular residual alone;
- bounded tangent extents;
- one global tangent penalty;
- Gaussian, kNN, RBF, and signed-volume controls.

Stop a family if it:

- fails to beat signed volume by 1 point;
- loses more than 2 points of unknown recall;
- collapses to one majority class;
- accepts random tangent extrapolations above the known-coverage threshold; or
- exceeds the frozen component/fit-work budget by more than 2x.

### 7.2 S2 confirmation

Retain at most:

- one shell family;
- one bounded-tube rank;
- the unbounded residual diagnostic;
- Gaussian, kNN, RBF, and volume controls.

Run seeds 11, 23, and 37 under identical partitions. Use paired sample-level
bootstrap intervals and report seed-level effects.

### 7.3 Extrapolation stress test

For each affine component, create label-free synthetic probes by:

1. fixing perpendicular residual within the known tube;
2. increasing tangent displacement from 1x to 8x the fitted extent;
3. measuring acceptance under unbounded residual and bounded tube.

The bounded tube must monotonically reduce confidence outside the tangent
extent. The unbounded residual is expected to expose why perpendicular distance
alone is unsafe.

### 7.4 Gate

Apply the M53 gate. If only the unbounded residual improves accuracy, the
program does not advance because that score lacks bounded open-space behavior.

### 7.5 Result (28 July 2026)

The seed-11 screen evaluated ranks 8, 16, and 32 against the frozen A2 signed
volume and matched Gaussian, kNN, and RBF controls. Rank 16 improved known
balanced accuracy from 91.75% to 93.125% and rank 32 to 93.375%; their unknown
recall was 92.5% and 91.5%, respectively, versus 60.5% for A2. Rank 8 lost 0.5
points of accuracy. The bounded and unbounded variants were identical on the
development observations.

All bounded scores increased monotonically under synthetic tangent displacement,
but their mean own-component scores at 8x extent were only 138.10--143.71
against calibrated acceptance thresholds of 492.02--528.70. Consequently every
rank accepted 100% of the 8x tangent probes. The registered open-space kill
switch fired for all ranks, despite the predictive and unknown-recall signal.
M53-S2 and M54 are blocked. The two executions produced byte-identical evidence.

## 8. M54 — Conditional lifecycle test

**Execution:** blocked because M53 did not pass
**Purpose:** determine whether the support model improves adaptation rather
than only closed-set or OOD metrics.

### 8.1 Frozen comparison

Reuse the exact M47:

- nine seed-by-arrival cells;
- 50-label budget;
- coverage-selected review set, which was the strongest eligible deterministic
  M47 arm;
- confirmation oracle;
- production adapter opportunity;
- anchor-quantile transfer;
- exhaustive routing;
- transaction and rollback contracts.

Change only the class support/rejection mechanism. Compare:

1. v8 Gaussian;
2. retained v9 shell or bounded tube;
3. kNN support diagnostic;
4. no-adaptation control.

### 8.2 Endpoint

Report episode and cumulative utility, known regression, remaining-unknown
recall, review count, update work, exact-model evaluations, replay, and
rollback.

### 8.3 Gate

Apply the M54 gate. M54 cannot reopen M50/E12; it is development evidence for a
new v9 support-model claim.

## 9. M55 — Finalization

Produce:

- `analysis/V9_FINAL_CLAIM_LEDGER.md`;
- immutable indexes for every executed milestone;
- a stopped/blocked/failed branch table;
- one artifact-only verifier;
- conclusion replay without training-data or final-label access;
- paper/thesis amendments only after the final outcome is frozen.

### 9.1 Result (28 July 2026)

M55 verified two immutable milestone indexes, seven indexed artifacts, seven
branch dispositions, and eight conclusion operands. Independent executions
produced byte-identical evidence without loading training features or opening
final labels. `analysis/V9_FINAL_CLAIM_LEDGER.md` freezes Outcome D and all
branches are now complete, stopped, closed, or blocked.

## 10. Dependency graph

```text
M51 protocol + occupancy diagnosis
 |\
 | +-- no shell signal --> close fitted-shell branch
 |                         |
 v                         v
M52 frozen shell scores    M53 bounded manifold tube
 |                         |
 +-----------+-------------+
             |
       M53 S2 confirmation
             |
       fail --> terminal D/E
             |
            pass
             v
       M54 lifecycle utility
             |
             v
       M55 final replay
```

## 11. Required artifacts

1. v9 claim ledger and immutable parent lock;
2. partition and feature hash manifest;
3. frozen-component occupancy diagnostics;
4. signed-depth and boundary-distance stratum tables;
5. volume/shell score-only factorial;
6. bounded-tube rank and tangent-extent study;
7. synthetic tangent extrapolation stress test;
8. Gaussian, kNN, RBF, random, permuted, and reversal controls;
9. calibration and threshold lineage;
10. optional fitted-shell loss trajectory;
11. conditional M54 episode records;
12. immutable branch ledger;
13. artifact-only final replay.

## 12. Interpretation guide

The following result patterns have distinct meanings:

| Result | Interpretation |
|---|---|
| Shell beats volume on frozen components | Prior score semantics were wrong for those components |
| Shell fails, bounded tube wins | Support is lower-dimensional but not a codimension-one primitive boundary |
| Perpendicular residual wins accuracy but fails OOD | Local manifold proximity helps, but unbounded tangent support is unsafe |
| Tube beats volume but trails Gaussian by more than 3 points | Geometric hypothesis has signal but insufficient relevance |
| Tube matches Gaussian and improves lifecycle utility | Publishable support-geometry result |
| No shell/tube signal | Volumetric assumption was not the principal measured bottleneck |

The surface hypothesis is falsified by the registered comparisons, not by
visual intuition or post-hoc depth plots.
