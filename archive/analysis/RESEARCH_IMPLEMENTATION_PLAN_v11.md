# GEODE Research Implementation Plan v11

## Directional Conformal Support Envelopes over a Delegated Head

**Status:** Final Outcome E; M68 artifact-only replay complete
**Date:** 28 July 2026  
**Claim ledger:** `analysis/CLAIM_LEDGER_v11.md`  
**Immutable parents:** v6.1 Outcome D, v7 Outcome C, v8 Outcome D, v9 Outcome D,
v10 Outcome D

## 1. Program decision

Three consecutive programs produced the same shaped result: a reproducible
low-rank residual-support signal that fails either the predictive-accuracy
co-primary gate (v10 M58: +0.875 vs a 1.0-point gate) or the open-space safety
gate through system-level component masking (v9 S1; v10: 16/18 cells
calibration-infeasible even at penalty 1024). Meanwhile the same evidence base
contains two positive signals that were never composed:

1. **support quality:** v9 bounded tubes reached 87.0--92.5% unknown recall
   versus 60.5% for the frozen A2 volume head, and the one safety-feasible v10
   cell passed every open-space, unknown-recall, and accepted-known operand;
2. **directional geometry:** M30.2 confirmed +3.97 points for cosine-native
   caps over matched Euclidean spheres on three seeds, with 27.23% kNN-gap
   closure, yet every v9/v10 tube was fit in raw Euclidean coordinates.

V11 therefore changes the registered role of the geometric model. The SDF
model is no longer required to beat the discriminative controls at
classification. It is registered as a **support envelope**: it decides whether
an observation is answerable at all, owns the open-set and lifecycle
contracts, and delegates the which-class decision to the strongest frozen
same-space head. Predictive accuracy becomes a non-inferiority constraint on
the composite, not a superiority endpoint for the geometry.

V11 additionally attacks the one failure mechanism v10 identified precisely:
system-level masking, in which a neighboring class's component accepts a probe
its source patch rejects. It does so with per-class split conformal
thresholds, a registered cross-class contrast condition, and a
negative-guided extent policy. It fits all geometry in the directional metric.

V11 does not reopen shells, CSG, fitter variants, per-class temperatures,
Euclidean accuracy parity, or the true-manifold question. Nonlinear warps
(flows, principal curves, quadrics) are explicitly out of scope and remain
untested.

## 2. Design principles

1. **Role separation is registered up front.** The envelope answers "is this
   supported?"; the delegated head answers "which class?". Neither claim can
   substitute for the other.
2. **Open-space behavior is primary, accuracy is a guardrail.** The gates
   invert the v10 weighting that killed a safety-passing model.
3. **Thresholds carry finite-sample guarantees.** Split conformal calibration
   per class replaces the penalty-grid feasibility search that leaked
   1.95--87.30% at 4x extent in v10.
4. **Masking is measured and gated directly.** Every probe is scored against
   its source class and against the full system, and cross-class acceptance is
   a registered endpoint with its own kill switch.
5. **Geometry lives on the sphere.** Centers are mean directions; tangent
   bases live in the tangent plane at the mean direction; extents and
   residuals are angular. This composes the M30 and v9/v10 signals for the
   first time.
6. **Complexity increases conditionally.** Patch counts above one enter the
   screen grid directly, but only under the frozen A2 budget multiplier, and
   the atlas keeps the v10 bootstrap-stability diagnostics.
7. **Lifecycle evaluation remains last.** No adaptation claim follows from
   predictive or OOD evidence alone.

## 3. Frozen data and partitions

Reuse the v9/v10 seed-specific identities without change:

- `geometry_fit`: mean directions, tangent bases, angular residual geometry,
  patch clustering;
- `score_calibration`: conformal residual quantiles, extents, contrast
  margins, and the 92% known-coverage threshold;
- `development_eval`: known balanced accuracy and class diagnostics;
- `unknown_eval`: classes 8--9 only;
- `episode_validation`: conditional lifecycle evaluation;
- `final_confirmation`: sealed.

M63 must hash-lock every source manifest, split ID, A2 parent, v7 Gaussian
control, v9/v10 evidence indexes, the delegated-head predictions, the
conformal miscoverage level, the contrast-margin grid, and the synthetic-probe
generator. Overlap or lineage mismatch fails closed.

## 4. Model definition

### 4.1 Delegated head

The composite classifier uses a frozen same-space discriminative head fit on
`geometry_fit` and calibrated on `score_calibration`; its predictions are
hash-locked in M63 before any envelope is fit. The registered head is the
frozen RBF control; the frozen logistic head is a registered fallback if RBF
lineage cannot be verified. The head never sees `unknown_eval`,
`episode_validation`, or sealed labels. The envelope may not modify, reweight,
or retrain the head.

### 4.2 Directional bounded tube

For each known class and patch, on L2-normalized features:

1. estimate the mean direction and the tangent plane at that direction from
   `geometry_fit`;
2. fit a rank-\(r\) PCA basis inside the tangent plane
   (\(r \in \{8, 16, 32\}\));
3. define the orthogonal residual as the angular component outside the
   patch subspace, and tangent coordinates as angular components inside it;
4. estimate per-axis tangent extents on `score_calibration` under the
   registered extent policies;
5. form the dimensionless v10 score \(s_k = q_{\perp,k} + \lambda
   q_{\parallel,k}\) with the identical rank normalization.

Patch counts per class are 1, 2, and 4, clustered deterministically within
class on `geometry_fit`. Total components must stay under the frozen A2
budget multiplier. Empty or rank-infeasible patches fail closed.

### 4.3 Extent policies

Three registered policies, selected on calibration data only:

1. **quantile:** per-axis calibration quantiles (0.95, 0.99), as in v10;
2. **negative-guided:** the largest per-axis extent whose induced acceptance
   of other-class calibration observations and 4x synthetic probes is 0%,
   floored at the 0.90 own-class quantile; infeasible axes fail closed;
3. **negative-guided with interquantile outer scale:** policy 2 extents with
   the v10 interquantile-range outer scale.

Policy 2 turns the safety suite from a post-hoc gate into the estimator: the
quantity that failed v9 and v10 is optimized directly, using only calibration
features and label-free probes.

### 4.4 Conformal acceptance and contrast

Per class \(k\), split conformal calibration on the class-\(k\) subset of
`score_calibration` yields threshold \(\tau_k\) as the
\(\lceil (n_k+1)(1-\alpha) \rceil / n_k\) empirical quantile of \(s_k\), with
registered miscoverage \(\alpha = 0.08\) matching the 92% coverage target.

An observation is **accepted** if and only if both:

1. \(s_{k^*}(x) \le \tau_{k^*}\) for \(k^* = \arg\min_k s_k(x)/\tau_k\)
   (normalized minimum, so units are comparable across classes); and
2. the contrast condition
   \(\min_{j \ne k^*} s_j(x)/\tau_j - s_{k^*}(x)/\tau_{k^*} \ge \delta\)
   **or** \(s_{k^*}(x)/\tau_{k^*} \le 1 - \delta\),
   for a contrast margin \(\delta\) selected on calibration data from the
   registered grid \(\{0, 0.05, 0.1, 0.2\}\) as the smallest value whose
   probe acceptance passes the M65 safety operands.

Condition 2 makes near-tie acceptances — the masking signature, in which a
probe leaving one class's support skims another's — fail unless the accepting
class holds the observation deep inside its own calibrated support. All
probes report both own-patch and minimum-system outcomes, as in v10.

## 5. Synthetic safety suite

Reuse the seven v10 probe families (axis tangent at 0.5x--8x, corner tangent,
normal, mixed, same-class bridge, cross-class bridge, random direction),
regenerated on the sphere: displacements follow geodesics, and norm-matched
random directions are drawn in the tangent plane. Add one registered family:

8. **masking probe:** an axis-tangent extrapolation from patch \(k\) directed
   toward the nearest competing class mean, at 2x, 4x, and 8x extent. This
   family exists specifically to measure the v10 failure mechanism and is
   gated separately.

## 6. M63 — Protocol, role, and conformal lock

**Execution:** unconditional

Implement:

- versioned directional-geometry, conformal-calibration, and
  contrast-acceptance schemas;
- immutable partition, parent, and delegated-head prediction locks;
- spherical tangent-plane projection and angular residual utilities;
- per-class split conformal quantile computation with exact tie handling;
- negative-guided extent estimation;
- deterministic geodesic probe generation including the masking family;
- composite-endpoint accounting: envelope decision, head decision, and
  composite outcome are recorded separately for every evaluation observation;
- parameter, fit-work, latency, and temporary-memory accounting.

Required tests:

- invariance of all scores to a common rescaling of pre-normalization units;
- conformal coverage at exactly the registered finite-sample rate on
  synthetic calibration draws;
- contrast condition rejects a constructed near-tie masking case that the
  v10 minimum-score rule accepts;
- negative-guided extents never exceed policy-1 extents and never fall below
  the 0.90 floor;
- zero tangent penalty inside extents; monotonic penalty outside;
- deterministic clustering, PCA signs, extents, probes, and geodesics;
- disjoint partitions, sealed-label enforcement, and head-lineage
  verification;
- invalid, empty, or rank-deficient patches fail closed;
- serialization and exact replay.

### 6.1 Result (28 July 2026)

M63 passed. The implementation added spherical log/exp maps, directional
tangent-plane PCA, the three registered extent policies, exact split-conformal
quantiles, contrast acceptance, all eight geodesic probe families, delegated-head
lineage checks, separate envelope/head/composite endpoint records, resource
accounting, and three versioned runtime schemas.

The protocol fixture verified 10 immutable parent locks and all three frozen
partition seeds. The compliant delegated-head artifacts were generated and
hash-locked separately before M65. Each synthetic class accepted exactly
46/49 calibration observations (93.878%), as required by the finite-sample
rank at \(\alpha=0.08\). The smallest safety-feasible fixture margin was 0;
system acceptance was 0% for 4x and 8x tangent probes and 0% for 4x masking
probes. A separate registered near-tie construction was accepted by the v10
minimum-score rule and rejected at margin 0.1 by the v11 contrast rule.

The fixture's negative-guided extents equaled the policy-1 upper extents on all
16 axes because its 4x negatives were already outside the quantile box;
contraction and 0.90-floor behavior were therefore established by focused
tests, not inferred from this fixture. Nine focused tests, the 485-test
repository suite (plus 27 subtests), schema round trips, and two byte-identical
artifact executions passed. M64 is open. These are protocol and implementation
results only; H1 remains untested until the registered adversarial masking scene.

Before M65, a lineage audit found that the originally referenced M27 RBF
artifact had been trained on all ten CIFAR classes, including proxy-unknown
classes 8--9. It was therefore invalid as the v11 delegated head. No real
envelope had yet been fit. M63 was repaired to fit the RBF only on classes 0--7
in `geometry_fit`, calibrate its frozen estimator on `score_calibration`, and
hash-lock its predictions before M65. The corrected head reached 96.25% known
development balanced accuracy, used 2,426 support vectors, excluded both
proxy-unknown classes from fit and calibration, and replayed exactly. M64 was
re-locked and replayed unchanged against the corrected M63 index.

## 7. M64 — Synthetic masking and directional identifiability

**Execution:** unconditional

Construct controlled datasets on \(S^{63}\):

- straight geodesic tubes at ranks 8, 16, and 32;
- curved arcs and two-mode supports requiring 2 patches;
- **an adversarial masking scene:** two class supports whose tangent
  extrapolations pass near each other, calibrated so the v10 minimum-score
  rule accepts at least 20% of 4x masking probes;
- full-dimensional caps and shell-only negative controls;
- random-label and random-orientation controls.

Gate (all required):

1. rank recovery exact in all straight-tube cells over seeds 11, 23, 37;
2. pooled independent in-support coverage within [90%, 94%] under conformal
   thresholds (finite-sample band);
3. 100% rejection of 8x tangent probes, at most 1% acceptance at 4x;
4. on the masking scene, the contrast rule cuts system-level 4x masking-probe
   acceptance to at most 1% while the replicated v10 rule exceeds 20%;
5. negative-guided extents reduce masking acceptance relative to quantile
   extents at matched coverage;
6. negative controls behave as registered; two executions replay
   byte-identically.

Failure of operand 4 or 5 falsifies the masking mechanism and stops the
program at Outcome E before any real-feature fitting.

### 7.1 Result (28 July 2026)

M64 passed every registered operand over seeds 11, 23, and 37. Directional
rank recovery was exact in all 9/9 straight-tube cells. Pooled independent
in-support coverage was 92.389%, within the registered [90%, 94%] band, and
every 4x and 8x axis-tangent probe was rejected. Individual cell coverages
ranged from 86.750% to 97.500%; only pooled coverage was registered as a gate,
so this spread is an observation to retain for the class-level M65 audit.

In the adversarial masking scene, the replicated v10 normalized-minimum rule
accepted 100% of 100 4x probes. The v11 margin-0.1 contrast rule accepted 0%.
The actual negative-guided estimator contracted the extent from the policy-1
0.95 quantile to its 0.90 floor (a 0.9 ratio), reducing minimum-rule masking
acceptance from 100% to 0% at matched 92.462% known coverage. Both load-bearing
masking operands therefore passed and Outcome E was not triggered.

Two-patch atlases reduced curved-arc median residual to 18.324--50.931% of the
global rank-8 residual and two-mode residual to at most 0.000294%. Full-cap and
shell controls retained at least 30.487% and 30.850% residual variance beyond
rank 32; random-label accuracy was 47.188--52.500%, versus 100% for true
labels. Ten focused tests and the 486-test repository suite (plus 27 subtests)
passed, and two executions produced byte-identical evidence.

The first implementation attempt reused the same observations to fit extents
and calibrate conformal thresholds and obtained only 89.514% independent
coverage. M64 corrected this by using independent extent-fit and conformal
subsets. M65 must preserve that deterministic within-`score_calibration`
separation; otherwise the finite-sample conformal claim does not apply. M65 is
open. M64 remains controlled evidence only and makes no real-feature claim.

## 8. M65 — Seed-11 directional envelope screen

**Execution:** conditional on M64

Grid: ranks \(\{8, 16, 32\}\) x patch counts \(\{1, 2, 4\}\) x the three
extent policies, all under conformal thresholds and calibration-selected
contrast margins. For each cell:

1. fit geometry on `geometry_fit`;
2. calibrate conformal thresholds, extents, and contrast on
   `score_calibration` and label-free probes only;
3. freeze all state;
4. evaluate the composite (envelope + delegated head) once on
   `development_eval`, `unknown_eval`, and the synthetic suite;
5. compare with A2, v7 Gaussian, kNN-support, unbounded-residual, and
   head-only (always-accept) controls, all at matched 92% known coverage.

Gate — a cell is retained only if it:

1. improves unknown recall over the strongest frozen support control (v7
   rank-32 Gaussian) by at least 2.0 points at matched known coverage;
2. keeps composite known balanced accuracy within 1.0 point of the frozen A2
   head (non-inferiority);
3. keeps accepted-known balanced accuracy within 1.0 point of the head-only
   control;
4. accepts 0% of 8x and at most 1% of 4x tangent probes, source and system;
5. accepts at most 1% of 4x masking probes at system level;
6. accepts at most 5% of bridge, mixed, and random-direction probes;
7. covers at least six of eight known classes among accepted observations;
8. stays within 2x the A2 parameter and fit-work budgets, with atlas cells
   additionally passing the v10 bootstrap-stability diagnostics; and
9. replays geometry, calibration, thresholds, and predictions exactly.

At most one cell is retained. Ties prefer fewer patches, then lower rank,
then policy order 2, 3, 1, then smaller contrast margin. No
development-selected retry is allowed.

### 8.1 Result (28 July 2026)

M65 retained 0/27 cells and terminates real-feature advancement. The frozen
1,600-observation `score_calibration` partition was split deterministically and
disjointly into 800 extent-fit and 800 conformal observations (100 per class in
each subset), preserving the M64 correction that extent estimation and
conformal calibration must not reuse observations.

All 18 negative-guided cells failed closed because at least one other-class
calibration observation remained inside the 0.90 own-class extent floor. All
nine quantile cells evaluated every contrast margin and found none safety
feasible. Across those attempts, system acceptance remained 98.828--100% at
4x tangent extent, 92.578--98.828% at 8x, and 96.875--100% for 4x masking
probes. Cross-class bridges were accepted 100%; mixed probes 75--100%; random
directions 87.5--100%. Margins up to 0.2 did not materially repair the
component-masking failure.

Because no cell passed calibration safety, development and unknown labels were
not used to select or retry a cell, atlas bootstrap diagnostics were not
opened, and no predictive, unknown-recall, or budget operand could rescue a
cell. The corrected delegated head itself was 96.25% balanced accurate on
known development observations. The complete 27-cell failure audit replayed
byte-identically; 487 repository tests and 27 subtests passed.

M66 and M67 are blocked. M68 opened to freeze the terminal branch disposition.
The controlled H1 result from M64 does not generalize to frozen real features:
the registered contrast grid cannot overcome system-level acceptance, while
negative-guided estimation is incompatible with the registered 0.90 own-class
floor. No safe directional envelope or H2 result is established.

## 9. M66 — Three-seed confirmation

**Execution:** conditional on M65

Replay the retained cell's rank, patch count, extent policy, miscoverage, and
contrast-selection rule on seeds 11, 23, and 37. Numerical thresholds,
extents, and margins may differ by seed only through the frozen calibration
rules.

Gate:

1. mean unknown-recall improvement over the Gaussian control of at least 2.0
   points with a paired 95% bootstrap lower bound above zero;
2. improvement on at least two of three seeds;
3. all M65 non-inferiority, safety, masking, class-coverage, budget,
   stability, and replay operands on every seed.

## 10. M67 — Conditional lifecycle evaluation

**Execution:** only after M66 passes

Reuse the frozen v8 nine episode cells, 50-label review budget,
coverage-selected review policy, threshold-transfer rule, exhaustive routing,
and transactional adapter. Change only the rejection/support score to the
retained conformal envelope; the delegated head substitutes for the frozen
classifier inside the accepted region.

Compare:

1. v8 Gaussian lifecycle baseline;
2. retained v11 composite;
3. kNN-support diagnostic;
4. no-adaptation control.

Gate: at the frozen 50-label budget, the composite must improve mean episode
utility by at least 2.0 points over the Gaussian baseline with a paired 95%
interval above zero, improve at least 7/9 cells, preserve known-class
balanced accuracy within 1.0 point and remaining-unknown recall within 2.0
points, and preserve exact rollback, confirmation lineage, and fallback
behavior.

## 11. M68 — Finalization

Produce:

- `analysis/V11_FINAL_CLAIM_LEDGER.md`;
- immutable indexes for each executed milestone;
- a complete branch-disposition table;
- one artifact-only verifier;
- two byte-identical conclusion replays;
- paper/thesis amendments only after the outcome is frozen.

### 11.1 Result (28 July 2026)

M68 finalized v11 as Outcome E. The artifact-only verifier checked the three
immutable M63--M65 indexes, 11 indexed artifacts, the final claim ledger, 12
conclusion operands, and all six branch dispositions. Two executions produced
byte-identical evidence and index files while loading no training features and
opening no final labels. Eleven focused v11 tests and the full 487-test,
27-subtest repository suite passed; the suite retained one unrelated pre-existing
runtime warning in `src/sdf_optimizer.py`.

Outcome E is applied through the registered clause that the directional
envelope line closes. M64's controlled masking mechanism did not fail: it passed
all H1 operands. The terminal result is instead that the mechanism did not
transfer to frozen real features. Every M65 cell was calibration-infeasible, so
H2, predictive non-inferiority, three-seed confirmation, and lifecycle utility
remain unsupported. M66 and M67 are permanently blocked.

## 12. Dependency graph

```text
M63 protocol + role + conformal lock
            |
M64 synthetic masking + directional identifiability
    | pass                     | masking mechanism falsified
    v                          v
M65 seed-11 envelope screen   Outcome E
    | retained cell
    v
M66 three-seed confirmation
    | pass
    v
M67 lifecycle evaluation
    |
    v
M68 final replay
```

## 13. Kill switches

- Stop the program at Outcome E if M64 operand 4 or 5 fails: the masking fix
  is the load-bearing mechanism and must work under controlled conditions.
- Stop a cell if negative-guided extents are infeasible on any axis, if
  conformal calibration has fewer than the registered minimum class count, or
  if no contrast margin passes calibration safety.
- Stop the atlas cells on bootstrap instability or budget overrun.
- Block M67 unless M66 passes in full; no partial advancement.
- Any partition leakage, delegated-head retraining, final-label access, or
  replay mismatch produces Outcome F and prohibits a positive claim.

## 14. Required artifacts

1. v11 claim ledger and parent lock;
2. partition, feature, and delegated-head prediction manifests;
3. conformal-calibration and contrast-selection audit;
4. synthetic masking and identifiability matrix;
5. negative-guided extent feasibility table;
6. exact per-class threshold and margin lineage;
7. full screen grid with composite-endpoint decomposition;
8. all eight probe-family stress records, source and system;
9. matched A2/Gaussian/kNN/head-only controls;
10. conditional atlas stability evidence;
11. conditional lifecycle records;
12. immutable final branch ledger and artifact-only replay.

## 15. Interpretation

| Result | Interpretation |
|---|---|
| M64 masking gate fails | Contrast + negative-guided extents do not fix component masking; Outcome E, mechanism falsified cheaply |
| Directional tubes lose the residual signal | The v9/v10 signal was metric-specific to Euclidean coordinates; envelope line closes |
| A cell passes M65 safety but misses unknown-recall gate | Geometry is safe but not better than Gaussian support; envelope adds no value over density |
| M66 passes, M67 fails | A safe, confirmed support envelope exists but does not improve adaptation utility; Outcome B |
| All gates pass | An explicit, editable, conformally calibrated geometric envelope improves open-set safety and lifecycle utility over the Gaussian baseline at non-inferior accuracy; Outcome A |
| Non-inferiority fails everywhere | Rejecting via geometry costs too much accepted-known accuracy; the support role itself is falsified for this geometry |

The program tests an operational support-envelope role, not classification
supremacy and not the true shape of the data distribution. If v11 fails its
own gates, the durable contribution remains the audited lifecycle machinery,
and explicit geometry is falsified for both the classifier and the envelope
role on these features.
