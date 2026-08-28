# Research Analysis: Weak Points and Paths to Competitiveness for an SDF-Based System

**Date:** 26 July 2026

**Sources:** `analysis/RESEARCH_IMPLEMENTATION_PLAN_v5.md`,
`analysis/MILESTONE_RESULTS.md`, `README.md`

**Central question:** How can an SDF-based system become competitive with
state-of-the-art machine learning approaches?

This document is an exploratory analysis, not a claim ledger. Nothing here
overrides a frozen gate, and every proposal below must still enter through the
M16 protocol before any confirmatory statement is made.

---

## 1. Where the evidence actually stands

The measured record is internally consistent and points to one pattern:
**GEODE is a competent density-coverage model being evaluated as a
discriminative classifier, and the gap grows as the representation gets
better.**

| Setting                                 |  GEODE |     Strongest control |       Gap |
| --------------------------------------- | -----: | --------------------: | --------: |
| CIFAR-10 HOG-era protocol (M3, 5 seeds) | 83.11% |       84.02% logistic |  -0.91 pp |
| CIFAR-10 spherical rerun (5 seeds)      | 83.96% |     ~84% logistic/RBF |   ~parity |
| CIFAR-100 superclass (E4, 5 seeds)      | 65.26% |            68.13% RBF |  -2.87 pp |
| WikiText-103 locked window              | 30.36% | 44.50% matched 5-gram | -14.14 pp |
| DINOv2 native 384-d, exact d+2 support  | 80.90% |            96.50% RBF | -15.60 pp |
| DINOv2 native, 1,000/class pilot        | 86.25% |        96.70% kNN/RBF | -10.45 pp |

Three structural facts stand out:

1. **The gap is small on weak features and enormous on strong features.** On
   HOG-era CIFAR features GEODE is within a point of logistic regression. On
   frozen DINOv2 features — precisely the space the v5 plan bets on — it trails
   by 10-15 points while kNN, the least expressive control, reaches 96%. The
   better the representation, the more the coverage-first geometric head throws
   away.
2. **Every added geometric mechanism has returned null or negative.** CSG
   subtraction: null across five-seed reruns for all primitive families.
   Shrinkage/MCD fitters: retained current fitter. Per-class temperatures:
   failed the gate. Adaptive metric policy (M18): negative on both gates.
   Certified routing (M12): exact but never faster. The consistent
   null-results pattern is itself evidence: the bottleneck is not fitting
   quality or fusion detail, it is the objective and the score semantics.
3. **Spheres beating full covariance (83.96% vs 81.03%) is a diagnosis, not a
   victory.** When the most constrained primitive wins, high-dimensional
   covariance estimation is failing, and the model is winning by regularization
   rather than by geometry. This is a classic symptom of sample starvation in
   $O(d^2)$-parameter estimators.

## 2. Weak points, ranked by evidence strength

### W1. Objective mismatch: coverage is not discrimination (critical)

The greedy constructor optimizes class-conditional _capture_ — enclose 95% of
each class's training mass — while every winning control optimizes _boundary
placement_. Logistic regression, RBF SVM, and even weighted kNN spend all of
their capacity where classes meet; GEODE spends nearly all of its capacity
where classes are already unambiguous. The M6 finding that
"geometry/readout extraction is the limiting stage" and the M13/M14 finding
that probabilistic semantics improve NLL but never accuracy both follow from
this: recalibrating a coverage model cannot move its decision boundary.

**Evidence:** M3 (-0.91 pp), E4 (-2.87 pp), DINOv2 native (-15.6 pp), M14
("probability improved raw calibration, not accuracy").

### W2. Sample-complexity wall from the d+2 / full-rank contract (critical)

The spherical seed contract needs $d+2$ points per component, and full
covariance needs $O(d^2)$ samples for stable estimation. In the 384-d DINOv2
space this means 386 points per class _per sphere_, so residual pools starve
after one sphere and per-class coverage stalls at 50-58%. Flowers-102 with 5
examples per class is simply unreachable — while linear probes score 99%. Any
system whose minimum viable component scales with ambient dimension is locked
out of exactly the few-shot, high-dimensional regime where frozen foundation
features are strongest.

**Evidence:** M19 native DINOv2 study (one sphere per class, target unmet),
Flowers-102 support-block, seed-11 pilot (61.5-80.6% coverage at 1,000/class).

### W3. Score semantics are not cross-class comparable (high)

The normalized radial field $f=\sqrt{\sum q_i^2/a_i^2}-1$ is a per-primitive
unitless quantity; softmin fusion then operates in log-space over values that
are neither Euclidean distances nor log-densities. Two consequences are
already measured: (a) accuracy is decided by whichever class field happens to
be scaled favorably, which is why temperature scaling fixes NLL/ECE
dramatically (1.48 to 0.50) without touching accuracy; (b) the geometric field
loses to maximum-softmax-probability on OOD detection (M7, M11.18) — a
distance-based model losing a distance-based task to its own calibrated
readout is a strong signal the raw field is semantically malformed.

**Evidence:** M2 readout table, M7 (max-prob AUROC beats Mahalanobis/kNN but
FPR95 0.43-0.58), M11.18 (0/9 OOD cells passed), M14.

### W4. Geometry-manifold mismatch in modern embedding spaces (high)

The M19 diagnostics show median local intrinsic dimension 15-21, and
radius-to-separation ratios of 1.7-2.0: classes are thin, curved, overlapping
sheets, not unions of compact blobs. Ellipsoids and spheres are volumetric
primitives; covering a 20-d sheet embedded in 384-d with balls either
over-covers (absorbing neighbors) or under-covers (starving coverage).
Moreover DINOv2/SigLIP embeddings are approximately directional (cosine-metric)
spaces; Euclidean/Mahalanobis primitives are the wrong ambient geometry there.
Both tested affine interfaces reduced kNN purity — linear reshaping cannot fix
a topology problem.

**Evidence:** M19 diagnostics table, interface failures (all arms), the
kNN/RBF dominance (methods that respect local manifold structure win).

### W5. The frozen-trunk boundary caps the ceiling (structural)

The plan's own honesty applies: SoTA image systems fine-tune or pretrain
end-to-end. A frozen-representation head can at best tie the best frozen-space
head, which itself trails fine-tuned models by several points on every public
benchmark. GEODE's competitive claim therefore _cannot_ be raw SoTA accuracy;
it must be Pareto competitiveness (Section 5). This is a boundary, not a flaw,
but it must be stated as such or the central question is unanswerable.

### W6. Inference and lifecycle costs scale with explicitness (moderate)

Exhaustive class-field evaluation is authoritative and M12 found no certified
routing break-even in Python. Component counts grow when accuracy improves
(spheres: 139.8 vs 58.6 primitives). Editability, the compensating asset, is
demonstrated only on synthetic/toy models (M8/M9) — the M24 Pareto frontier
that would monetize it is still unrun.

### W7. Temporal track is uncompetitive by a wide margin (moderate, contained)

30.36% vs 44.50% for a matched 5-gram is not a calibration or geometry issue;
the frozen reservoir/window features do not contain the information. This is a
representation failure upstream of GEODE and should not be charged to the
geometric head — but it also means text claims should stay strictly scoped to
the head-effect comparison M21 defines.

## 3. Solutions ordered by expected leverage

### S1. Boundary-seeking construction: fit the discriminant, not the density

The single highest-leverage change. Two concrete versions, both compatible
with the frozen-space contract and M17's machinery:

- **Discriminative placement.** Replace capture-rate seeding with seeding on
  margin-weighted samples (points near the current decision boundary, in the
  spirit of support vectors). The greedy objective becomes reduction in
  development cross-entropy or hinge loss, not class coverage. The 95%
  coverage target is retired as a construction objective and retained only as
  a descriptive metric.
- **Teacher distillation into geometry.** Train the best black-box head (RBF
  or compact MLP) on the frozen features first, then fit GEODE to match the
  _teacher's decision function_ (soft labels / boundary samples) rather than
  the data density. This directly converts the 96.5% RBF into a supervision
  signal. The resulting artifact remains an explicit, editable, rollback-able
  geometric bundle — the auditability claim survives distillation intact. If
  the distilled geometric head lands within 1-2 pp of the teacher, the parity
  question of M19/M17 is effectively answered, and the residual gap becomes a
  measurable geometry-capacity number.

RBF-SVM equivalence is the theoretical anchor: an RBF decision function is
already a soft-min/soft-max over distances to weighted centers. GEODE's field
is expressively adjacent — the difference is entirely _where the centers go
and what loss put them there._

### S2. Low-dimensional primitives: break the d+2 wall

Replace full-ambient primitives with **subspace primitives**: each component
is a local rank-$r$ affine subspace (local PCA / mixture-of-PPCA style) plus
isotropic residual, giving a well-defined SDF (distance to a bounded disk or
capsule in the subspace) with $O(dr)$ parameters and a seed requirement of
$r+2$, not $d+2$. With $r\approx 16$-$32$ matched to the measured intrinsic
dimension (15-21), the 386-points-per-sphere wall becomes ~20-35 points, the
Flowers-102 few-shot regime opens up, and multi-component growth per class is
restored. M18 already built the factorized-precision infrastructure
(`diagonal_low_rank`, shared-subspace families, Woodbury solves); what failed
in M18 was the _selection policy_, not the parameterization — reuse the
families under the S1 objective instead of the coverage objective.

### S3. Match the ambient geometry: directional primitives

For L2-normalized or nearly-normalized embedding spaces, add a cosine-native
primitive family: components defined by a mean direction and angular radius
(a spherical cap SDF on $S^{d-1}$, the geometric analogue of a von
Mises-Fisher component). Weighted kNN's dominance under cosine-friendly
normalization suggests much of the 10-15 pp DINOv2 gap is metric mismatch,
which no amount of ellipsoid fitting in Euclidean coordinates can recover.
This is a contained, testable primitive addition under the existing
primitive-family contract.

### S4. Proper likelihood semantics with shared calibration (M17 variants 2-4)

Adopt log-density scoring — $\log\pi_k + \log\mathcal N(x)$ with log-dets —
as the fused field, so scores are cross-class comparable by construction, and
support/OOD scores become genuine likelihood statements. M14 showed this fixes
calibration; combined with S1 it can also move accuracy because the boundary
is then optimized in a space where the score gradient is meaningful. Keep the
retained global temperature; do not reopen per-class temperatures (already
failed).

### S5. Hybrid support scoring for OOD

Stop asking the raw field to do OOD alone. The M22 design is right; the
missing piece is that the geometric in-support score should be the
_likelihood-semantics_ field of S4 combined with a cached-feature kNN distance
(the strongest measured support signal). An SDF system has a legitimate
structural advantage here — an explicit "outside every component" region —
but only after S4 makes the field a distance-like quantity globally.

### S6. Amortized construction (longer-horizon)

Greedy + RANSAC construction is myopic and slow to place components well. A
small set-transformer that _proposes_ primitive parameters (trained across
many synthetic/dev fitting problems), with the proposals then verified,
accepted, and frozen through the existing transactional machinery, keeps the
artifact fully explicit while removing the constructor's myopia. The learned
proposer never touches inference; it is tooling, not model. This is the one
place where "use a neural network" does not violate the interpretability
boundary, because the output is still an auditable geometric object.

### S7. Concede the trunk, own the head — and say so

Do not chase end-to-end SoTA. The winnable claim, per the plan's own M24
framing, is: **within 0.5-1 pp of the best same-space black-box head, while
being the only non-dominated method on the edit/rollback/audit axes.** That is
a real, publishable competitiveness result. The breakthroughs above (S1+S2+S3)
are what make the "within 0.5-1 pp" half achievable; the lifecycle work
already done (E9-E11) makes the second half nearly banked.

## 4. What a breakthrough would look like, concretely

The compounding bet is S1×S2×S3 in the DINOv2 space, because each addresses an
independent measured failure:

| Failure measured                        | Fix                                | Success signal                                                       |
| --------------------------------------- | ---------------------------------- | -------------------------------------------------------------------- |
| Coverage objective wastes capacity (W1) | S1 boundary/distillation objective | DINOv2 gap shrinks from ~10 pp to <2 pp at fixed component budget    |
| d+2 support starvation (W2)             | S2 rank-r subspace primitives      | ≥3 components/class at 1,000/class; Flowers-102 fit becomes feasible |
| Metric mismatch (W4)                    | S3 cosine-native primitives        | GEODE-kNN gap closes specifically on normalized embeddings           |
| Incomparable scores (W3)                | S4 likelihood field                | OOD: geometric support score beats max-prob on the frozen suite      |

A staged falsification path that fits the existing protocol:

1. **S0/S1 (days):** distill an RBF teacher into a spherical GEODE on the
   existing bounded DINOv2 S1 cache. If distilled GEODE cannot get within
   ~3 pp of the teacher even with a generous component budget, the primitive
   family is expressively insufficient and S2/S3 move to the front.
2. **S1:** subspace-primitive prototype at $r\in\{8,16,32\}$ on the same
   cache; gate on components-per-class ≥ 3 and accuracy over the current
   one-sphere result.
3. **S2 (three seeds):** the winning combination under the M17 gate as
   written (+0.5 pp over strongest non-topology GEODE control, or -2% NLL).
4. Only then M20 topology search — topology search cannot rescue a wrong
   objective, which is exactly why the plan sequences it after M17.

Each step has a cheap kill switch, consistent with the Section 18 stop rules.

## 5. Honest answer to the central question

An SDF-based system becomes competitive with SoTA approaches **not by beating
end-to-end networks at accuracy, but by reaching statistical parity with the
best head on a SoTA frozen representation while dominating every alternative
on lifecycle axes.** The current evidence says parity is blocked by three
correctable choices — a coverage objective, ambient-dimension primitives, and
non-comparable scores — and not by any demonstrated limit of explicit geometry
itself. The plan's diagnosis (representation × semantics × complexity ×
selection × objective) is confirmed by the data; the sharpened version this
analysis adds is a priority ordering: **objective first (S1), sample
complexity second (S2), ambient metric third (S3)**. If, after those three,
GEODE still trails kNN by double digits in strong spaces, the negative result
is clean and important: volumetric geometric heads are structurally unsuited
to high-intrinsic-dimension manifold embeddings, and the project's durable
contribution is the audited-lifecycle machinery, which is already the
strongest part of the evidence base.
