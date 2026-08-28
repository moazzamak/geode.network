# GEODE: Research Report

**Greedy Ellipsoidal Outline Discrimination by Excision — Background, Prior Work, Methodology, and Code Analysis**

_Date: 2026-07-24_

---

## Table of Contents

1. [Background Concepts](#1-background-concepts)
2. [Prior Work](#2-prior-work)
3. [Positioning: What Is (and Is Not) Novel](#3-positioning-what-is-and-is-not-novel)
4. [Methodology](#4-methodology)
5. [Code Analysis](#5-code-analysis)
6. [Identified Issues & Suggested Improvements](#6-identified-issues--suggested-improvements)
7. [References (Quick Index)](#7-references-quick-index)

---

## 1. Background Concepts

### 1.1 Signed Distance Fields (SDFs)

A signed distance field is a function `f: ℝᵈ → ℝ` whose zero level set `{x : f(x)=0}` defines a
surface, with `f < 0` inside and `f > 0` outside. A _metric_ SDF additionally satisfies the Eikonal
property `‖∇f‖ = 1`, so `|f(x)|` equals Euclidean distance to the surface. SDFs are widely used in
computer graphics (ray marching, level-set methods) and, since 2019, as a learned representation of
3D shape (DeepSDF [C1.1], Occupancy Networks [C1.2], SIREN [C1.3], IGR [C1.4]).

**Important nuance for GEODE:** the ellipsoid "SDF" used throughout the codebase,

```
f(x) = √( Σᵢ qᵢ² / aᵢ² ) − 1,   q = Rᵀ(x − c)
```

is a **normalized (Mahalanobis-style) distance, not a metric SDF**. It is exact only for spheres
(all `aᵢ` equal). For anisotropic ellipsoids `‖∇f‖ ≠ 1`: the same numeric value `f = 0.1` means a
much larger Euclidean gap along a long axis than along a short axis. This choice is common
(it is cheap and monotone in true distance for a single ellipsoid) but it has downstream
consequences for softmin fusion, capture thresholds, and ray marching (§6.1).

### 1.2 Constructive Solid Geometry (CSG)

CSG builds complex solids from primitives via Boolean algebra. On SDFs:

- Union: `f_A∪B = min(f_A, f_B)` (smooth version: softmin / log-sum-exp)
- Intersection: `f_A∩B = max(f_A, f_B)`
- Difference: `f_A∖B = max(f_A, −f_B)`

Note that min/max composition of true SDFs yields only a _bound_ on the true distance of the
composite (exact sign, approximate magnitude). GEODE uses normalized softmin for unions (within experts and
across experts) and hard max for set-difference (subtractive ellipsoids).

### 1.3 Mahalanobis Distance and Gaussian Mixtures

For a Gaussian `N(μ, Σ)`, the Mahalanobis distance `√((x−μ)ᵀΣ⁻¹(x−μ))` has ellipsoidal level sets.
GEODE's ellipsoid function with precision `P = R diag(a⁻²) Rᵀ` is exactly a Mahalanobis distance
minus 1. Consequently the model-level normalized softmin

```
D(x) = −(1/α) ln (1/M) Σⱼ exp(−α fⱼ(x))
```

is, up to affine terms, a **log-likelihood of a (heteroscedastic, unnormalized) Gaussian-like mixture**. Classifying by
`argmin_class D_class(x)` is therefore closely analogous to Mixture Discriminant Analysis [C3.1]
and to the Mahalanobis-distance classifier of Lee et al. [C5.4]. This equivalence is central to
positioning the work honestly (§3).

### 1.4 RANSAC and Multi-Model Fitting

RANSAC (Fischler & Bolles, 1981 [C3.2]) fits a model robustly by repeatedly sampling _minimal_
subsets, hypothesizing a model, and counting inliers. Its convergence guarantee depends on the
probability `wᵏ` that all `k` sampled points are inliers of a common model:
`N ≥ log(1−p) / log(1−wᵏ)`. Fitting **multiple** models is the multi-model fitting problem, solved
greedily (sequential RANSAC), globally (PEARL [C3.3]), or progressively (Progressive-X [C3.4]).
GEODE's two-level greedy loop is a sequential/progressive multi-model RANSAC specialized to
ellipsoid quadrics, now augmented with kNN-anchored seeding to improve inlier purity in high dimensions.

**Key caveat:** a general d-dimensional quadric needs `k = d(d+3)/2` points (44 in d=8, 54 in
d=9). Since `wᵏ` decays exponentially in `k`, minimal-sample RANSAC becomes statistically
infeasible above roughly d≈6 — which is exactly why the codebase's covariance fallback dominates
in high dimensions (§5.2).

### 1.5 Mixture of Experts and Distance-Based Routing

Classical MoE (Jacobs et al. 1991 [C4.1]) trains expert networks with a learned softmax gate;
modern sparse MoE (Shazeer [C4.2], Switch [C4.3]) uses learned top-k routing. GEODE replaces the
learned gate with **geometric proximity**: a point "belongs" to whichever expert's SDF is lowest,
and normalized softmin fusion is the soft version of this rule. This is gate-free routing — interpretable and
immune to gate collapse, but not learned jointly with a task loss.

### 1.6 Calibration

Raw SDF scores are not probabilities. Platt scaling [C5.5] fits a logistic map from scores to
probabilities on held-out data — the classical remedy, applied here per model.

### 1.7 Learning Sequences Without Backpropagation Through Time

The Tier-6 "RANSAC-through-time EM" fits next-character predictors on windowed context features
without recurrence or BPTT. The conceptual relatives are reservoir computing (Echo State Networks
[C6.2], Liquid State Machines [C6.3]) — fixed feature dynamics with a cheaply-fitted readout — and
Hinton's Forward-Forward [C6.1] as a modern non-backprop learning proposal. The `sdf` context mode
(feeding the model's own K-dim SDF score vector back as features) is a hand-crafted, non-learned
recurrent state.

---

## 2. Prior Work

Verified references, grouped by theme. **⚠ HIGH OVERLAP** marks work the project must explicitly
differentiate from; **◈ CONTRAST** marks useful framing counterpoints.

### 2.1 Neural Implicit SDF Representations

| Ref  | Work                                                 | Link                                                 | Relevance                                                                                                                                                                             |
| ---- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1.1 | **DeepSDF** — Park et al., CVPR 2019                 | [arXiv:1901.05103](https://arxiv.org/abs/1901.05103) | Established SDFs as a learned representation. GEODE uses closed-form ellipsoid SDFs instead of a neural field — interpretable and CSG-composable, but less expressive per primitive. |
| C1.2 | **Occupancy Networks** — Mescheder et al., CVPR 2019 | [arXiv:1812.03828](https://arxiv.org/abs/1812.03828) | Inside/outside occupancy fields; GEODE's calibrated class-membership score is the continuous analogue in feature space.                                                              |
| C1.3 | **SIREN** — Sitzmann et al., NeurIPS 2020            | [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) | High-fidelity neural SDF level sets; motivates implicit representations generally.                                                                                                    |
| C1.4 | **IGR** — Gropp et al., ICML 2020                    | [arXiv:2002.10099](https://arxiv.org/abs/2002.10099) | Enforces the Eikonal property by regularization. Notably, GEODE's normalized ellipsoid distance does _not_ satisfy Eikonal — see §6.1.                                               |
| C1.5 | **NeuS** — Wang et al., NeurIPS 2021                 | [arXiv:2106.10689](https://arxiv.org/abs/2106.10689) | Maps SDF → probability via a sigmoid, mathematically the same move as GEODE's Platt calibration (different purpose). ◈ CONTRAST                                                      |

### 2.2 Primitive-Based Shape Decomposition

| Ref   | Work                                                                | Link                                                 | Relevance                                                                                                                                                                                                                       |
| ----- | ------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C2.1  | **CSGNet** — Sharma et al., CVPR 2018                               | [arXiv:1712.08290](https://arxiv.org/abs/1712.08290) | Predicts CSG programs (incl. subtraction) for shape reconstruction. ⚠ HIGH OVERLAP on CSG set-difference; GEODE applies it to classification.                                                                                  |
| C2.2  | **UCSG-Net** — Kania et al., NeurIPS 2020                           | [arXiv:2006.09102](https://arxiv.org/abs/2006.09102) | Unsupervised CSG-tree discovery incl. Boolean difference. ⚠ HIGH OVERLAP; GEODE differs via RANSAC fitting + discriminative target.                                                                                            |
| C2.3  | **CAPRI-Net** — Yu et al., CVPR 2022                                | [arXiv:2104.05652](https://arxiv.org/abs/2104.05652) | Adaptive assembly of convex primitives with intersection/subtraction — closest structural mirror of additive+subtractive assembly.                                                                                              |
| C2.4  | **BSP-Net** — Chen et al., CVPR 2020                                | [arXiv:1911.06971](https://arxiv.org/abs/1911.06971) | Shapes as intersections of half-spaces; the piecewise-linear ancestor of ellipsoidal regions.                                                                                                                                   |
| C2.5  | **CvxNet** — Deng et al., CVPR 2020                                 | [arXiv:1909.05736](https://arxiv.org/abs/1909.05736) | Learned convex decomposition. ⚠ HIGH OVERLAP on convex-primitive decomposition; GEODE adds subtraction and non-gradient fitting.                                                                                               |
| C2.6  | **Superquadrics Revisited** — Paschalidou et al., CVPR 2019         | [arXiv:1904.09970](https://arxiv.org/abs/1904.09970) | Unsupervised superquadric (⊃ ellipsoid) fitting. ⚠ HIGH OVERLAP on the primitive family; different objective (geometry vs discrimination).                                                                                      |
| C2.7  | **Volumetric Primitives** — Tulsiani et al., CVPR 2017              | [arXiv:1612.00404](https://arxiv.org/abs/1612.00404) | Seminal cuboid-abstraction work with coverage+parsimony loss — the same coverage/parsimony tension as GEODE's consensus/stagnation rules.                                                                                      |
| C2.8  | **Neural Parts** — Paschalidou et al., CVPR 2021                    | [arXiv:2103.10429](https://arxiv.org/abs/2103.10429) | Expressive learned primitives; the opposite design point to GEODE's deliberately simple analytic primitives. ◈ CONTRAST                                                                                                        |
| C2.9  | **Hierarchical Part Decomposition** — Paschalidou et al., CVPR 2020 | [arXiv:2004.01176](https://arxiv.org/abs/2004.01176) | Hierarchies of superquadric parts; analogue of the two-level expert/ellipsoid hierarchy.                                                                                                                                        |
| C2.10 | **Marching-Primitives** — Liu et al., CVPR 2023                     | [arXiv:2303.13190](https://arxiv.org/abs/2303.13190) | Greedily grows superquadrics directly from an SDF. **⚠ HIGH OVERLAP ★★★ — the single most technically proximate shape work.** GEODE differs in target (feature-space classification), subtractive primitives, and calibration. |
| C2.11 | **EMS Robust Ellipsoid Fitting** — Zhao et al., ICCV 2021           | [arXiv:2110.13337](https://arxiv.org/abs/2110.13337) | EM-based outlier-robust ellipsoid-specific fitting. ⚠ HIGH OVERLAP on the primitive-fitting step; also a candidate drop-in replacement for the SVD quadric fit (§6.4).                                                          |

### 2.3 Classical ML Analogues

| Ref  | Work                                                                 | Link                                                                                         | Relevance                                                                                                                                           |
| ---- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| C3.1 | **Mixture Discriminant Analysis** — Hastie & Tibshirani, JRSS-B 1996 | [DOI:10.1111/j.2517-6161.1996.tb02085.x](https://doi.org/10.1111/j.2517-6161.1996.tb02085.x) | Per-class Gaussian mixtures + Mahalanobis boundaries. **⚠ HIGH OVERLAP ★★★ — GEODE is structurally a RANSAC-fitted, CSG-excision-extended, geometric MDA.** |
| C3.2 | **RANSAC** — Fischler & Bolles, CACM 1981                            | [DOI:10.1145/358669.358692](https://doi.org/10.1145/358669.358692)                           | The foundational robust-fitting loop the constructor extends.                                                                                       |
| C3.3 | **PEARL** — Isack & Boykov, IJCV 2012                                | [DOI:10.1007/s11263-011-0474-7](https://doi.org/10.1007/s11263-011-0474-7)                   | Global (energy-based) multi-model fitting; the principled alternative to greedy sequential extraction. ◈ CONTRAST                                   |
| C3.4 | **Progressive-X** — Barath & Matas, ICCV 2019                        | [arXiv:1906.02290](https://arxiv.org/abs/1906.02290)                                         | Anytime progressive multi-model RANSAC. ⚠ HIGH OVERLAP on algorithmic structure of the two-level loop.                                              |
| C3.5 | **Gradient Boosting** — Friedman, Ann. Stat. 2001                    | [DOI:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)                         | Stagewise greedy additive modeling — GEODE's outer loop with the uncaptured pool as "residual". ◈ CONTRAST                                         |
| C3.6 | **Matching Pursuit** — Mallat & Zhang, IEEE TSP 1993                 | [DOI:10.1109/78.258082](https://doi.org/10.1109/78.258082)                                   | Greedy atom selection from an overcomplete dictionary; ellipsoids as geometric atoms.                                                               |
| C3.7 | **SVDD** — Tax & Duin, ML 2004                                       | [DOI:10.1023/B:MACH.0000008084.60811.49](https://doi.org/10.1023/B:MACH.0000008084.60811.49) | Minimum-volume enclosing hypersphere/ellipsoid one-class classifier; GEODE is the multi-primitive, discriminative generalization.                                  |
| C3.8 | **One-Class SVM** — Schölkopf et al., Neural Comp. 2001              | [DOI:10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965)                 | Kernelized support estimation; implicit-kernel analogue of explicit ellipsoid coverage.                                                             |
| C3.9 | **LVQ / GMLVQ** — Kohonen, Proc. IEEE 1990                           | [DOI:10.1109/5.58325](https://doi.org/10.1109/5.58325)                                       | Prototype-based classification; matrix-LVQ variants learn full-covariance (ellipsoidal) prototypes — very close in spirit to nudged ellipsoids.     |

### 2.4 Mixture of Experts

| Ref  | Work                                                         | Link                                                                     | Relevance                                                                                                                                  |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| C4.1 | **Adaptive Mixtures of Local Experts** — Jacobs et al., 1991 | [DOI:10.1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79) | Foundational MoE with learned gating; GEODE's normalized softmin-of-SDF is gate-free geometric routing. ◈ CONTRAST                                   |
| C4.2 | **Sparsely-Gated MoE** — Shazeer et al., ICLR 2017           | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)                     | Learned top-k sparse routing at scale.                                                                                                     |
| C4.3 | **Switch Transformers** — Fedus et al., JMLR 2022            | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)                     | Top-1 routing + load-balancing loss; the bounding-sphere pruning in `SoftminFusion` is the geometric analogue of sparse expert activation. |

### 2.5 Distance-Based Classifiers, Uncertainty & Calibration

| Ref  | Work                                                   | Link                                                                                 | Relevance                                                                                                                                                                                                                                                                                                 |
| ---- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C5.1 | **Prototypical Networks** — Snell et al., NeurIPS 2017 | [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)                                 | Distance-to-class-prototype classification; GEODE generalizes to multiple anisotropic, oriented prototypes per class with CSG excision.                                                                                                                                                                    |
| C5.2 | **DUQ** — van Amersfoort et al., ICML 2020             | [arXiv:2003.02037](https://arxiv.org/abs/2003.02037)                                 | RBF-centroid distances for deterministic uncertainty. ⚠ HIGH OVERLAP on distance-based class scores; GEODE's SDF+CSG excision is more expressive.                                                                                                                                                                 |
| C5.3 | **SNGP** — Liu et al., NeurIPS 2020                    | [arXiv:2006.10108](https://arxiv.org/abs/2006.10108)                                 | Distance-aware uncertainty via spectral norm + GP head; GEODE's calibrated SDF is a simpler alternative — an OOD/uncertainty evaluation would be a natural added experiment.                                                                                                                             |
| C5.4 | **Mahalanobis OOD** — Lee et al., NeurIPS 2018         | [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)                                 | Class-conditional Gaussians on CNN features + Mahalanobis score + logistic calibration. **⚠ HIGH OVERLAP ★★★ — CNN features → Mahalanobis-type score → logistic calibration is the same pipeline skeleton as Tiers 4/5.** GEODE generalizes to per-class _mixtures_ with excision and RANSAC fitting. |
| C5.5 | **Platt Scaling** — Platt, 1999                        | [PDF](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf) | The calibration method used verbatim.                                                                                                                                                                                                                                                                     |

### 2.6 Non-Backprop / Forward-Only Sequence Learning

| Ref  | Work                                                        | Link                                                                         | Relevance                                                                                                                      |
| ---- | ----------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| C6.1 | **Forward-Forward** — Hinton, 2022                          | [arXiv:2212.13345](https://arxiv.org/abs/2212.13345)                         | Layer-local learning without backprop; kindred motivation to RANSAC-through-time. ◈ CONTRAST (still gradient-based per layer). |
| C6.2 | **Echo State Networks** — Jaeger, GMD Report 148, 2001      | (grey literature; see ResearchGate)                                          | Fixed random recurrence + cheap readout. The `concat`/`sdf` context modes play the role of a hand-built reservoir.             |
| C6.3 | **Liquid State Machines** — Maass et al., Neural Comp. 2002 | [DOI:10.1162/089976602760407955](https://doi.org/10.1162/089976602760407955) | Peer-reviewed companion result to ESNs.                                                                                        |

### 2.7 Set-Difference / Difference-of-Convex Classifiers

| Ref  | Work                                                               | Link                                                                       | Relevance                                                                                                                           |
| ---- | ------------------------------------------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| C7.1 | CSGNet / UCSG-Net (see C2.1/C2.2)                                  | —                                                                          | Boolean difference on learned shapes exists in CAD reconstruction, **not** in discriminative classification.                        |
| C7.2 | **Nearest Convex Hull classifiers** — e.g. Nalbantov et al. 2006   | —                                                                          | Convex class regions with distance tie-breaking rather than explicit carving.                                                       |
| C7.3 | **Multisurface / polyhedral separation** — Mangasarian             | [DOI:10.1007/BF02614780](https://doi.org/10.1007/BF02614780)               | Piecewise-linear enclosing regions; BSP-Net is the neural analogue.                                                                 |
| C7.4 | **DC programming / CCCP** — Yuille & Rangarajan, Neural Comp. 2003 | [DOI:10.1162/08997660360581958](https://doi.org/10.1162/08997660360581958) | "Union of ellipsoids minus union of ellipsoids" is a difference-of-convex set — the formal framework GEODE implicitly operates in. |

> **Literature gap:** no prior work was found that uses Boolean set-difference of ellipsoids as a
> discriminative classifier. Subtractive excision of inter-class overlap (`fit_subtractive_ellipsoids`
> + `_active_repair`) appears to be GEODE's strongest genuine novelty claim.

---

## 3. Positioning: What Is (and Is Not) Novel

### 3.1 Ideas that substantially pre-exist

1. **Per-class ellipsoidal/Mahalanobis scoring on CNN features with logistic calibration** —
   this is essentially Lee et al. 2018 [C5.4] with mixtures. GEODE Tiers 4/5 should be framed as a
   generalization of that pipeline, not an independent invention.
2. **Per-class mixtures of ellipsoidal components as a classifier** — Mixture Discriminant
   Analysis [C3.1] (1996). The softmin-of-Mahalanobis fusion is mathematically a smoothed mixture
   log-likelihood.
3. **Greedy extraction of ellipsoidal primitives from data/SDFs** — Marching-Primitives [C2.10],
   Superquadrics Revisited [C2.6], EMS [C2.11], and the multi-model RANSAC line
   (sequential RANSAC, Progressive-X [C3.4]).
4. **CSG composition (incl. difference) of primitives via min/max of SDFs** — CSGNet [C2.1],
   UCSG-Net [C2.2], CAPRI-Net [C2.3], and decades of graphics literature.
5. **Distance-based gate-free expert routing** — prototype methods [C5.1], RBF networks, DUQ [C5.2].

### 3.2 Combinations that appear genuinely novel

1. **Subtractive (CSG-difference) ellipsoids fitted specifically to _other-class false captures_ to
   excise inter-class overlap** — no discriminative-classification precedent found.
2. **Discriminative RANSAC F_β scoring** inside primitive growth — tunable precision/recall balance
   (`score_beta`) folds boundary awareness into robust geometric fitting (see §6.4).  kNN-anchored
   seeding additionally compensates for the exponential collapse of all-inlier probability in high d.
3. **The full system**: RANSAC-constructed, CSG-excision-sharpened, calibration-topped geometric
   classifier with normalized Softmin fusion, covariance-eigendecomposition nudging, sphere-tracing
   inference, semantic-routed model DAG, and temporal EM extension. The parts are known; the assembly
   is not.

### 3.3 Recommended framing

Present GEODE as: _"a constructive, RANSAC-driven generalization of Mahalanobis/mixture
discriminant classifiers [C3.1, C5.4], borrowing CSG set-difference machinery from shape abstraction
[C2.1, C2.10] to explicitly excise inter-class overlap — a mechanism absent from prior
distance-based classifiers."_ Add baselines accordingly (§6.9): GMM-per-class/MDA, Lee et al.'s
single-Gaussian Mahalanobis, RBF-SVM, and gradient boosting on identical features. The novelty
claims are only credible if GEODE beats or matches the closest classical analogues.

---

## 4. Methodology

### 4.1 Model

- **Primitive** (`EllipsoidExpert`): oriented d-dimensional ellipsoid `(c, a, R, polarity ∈ {±1})`
  with normalized distance `f(x) = ‖diag(a⁻¹) Rᵀ (x−c)‖ − 1`.
- **Expert** (`Expert`): normalized softmin (log-sum-exp / M, sharpness α) union of additive members; hard-max CSG
  excision against the normalized softmin union of subtractive members:
  `f_exp = max(f_add, −f_sub)`. Gradient blends the two branches by smooth-max weights.
- **Class model**: normalized softmin fusion over experts (`SoftminFusion`), accelerated by bounding-sphere
  lower-bound pruning with a certified error < `M·e⁻¹⁰/α`.
- **Classifier**: per-class models; predict `argmin(SDF/scale)` or, when fitted, argmax of
  Platt-calibrated probabilities.

### 4.2 Construction (two-level greedy RANSAC)

- **Outer loop**: grow an Expert from the unexplained pool; lock and remove captures if consensus
  ≥ `consensus_threshold`; repeat until the pool is exhausted or consensus fails.
- **Inner loop**: sample minimal seeds (`k = d(d+3)/2`) from points not yet captured; for `d > 6`
  alternate with kNN-anchored seeds (random anchor + k nearest neighbours, dramatically higher
  inlier purity in high d); fit a quadric by SVD of the design matrix; if the quadric is not an
  ellipsoid, fall back to the seed-covariance ellipsoid with radii `√(λᵢ·d)`; score candidates by
  the F_β discriminative score `(1+β²)·p / ((1+β²)·p + β²·n + 1)` when an exclude pool is
  supplied (default β=1, i.e. F₁), or by raw captures in standard mode; stop on growth stagnation.
- **Refinement**: `NudgeEngine` performs covariance-eigendecomposition center/radii/orientation
  updates (consistent with constructor fallback); `SDFOptimizer` performs analytic-gradient
  cross-entropy refinement of `(μ, P, α)` with heavy-ball momentum and projection back onto the PD cone.
- **Boundary excision**: `fit_subtractive_ellipsoids` fits polarity=−1 ellipsoids to other-class
  points captured inside each expert (signed SDF < threshold), inflates radii by the threshold, and
  nudges via covariance eigendecomposition; `_active_repair` re-runs a focused pass on deeply
  captured false positives (SDF < −2t).

### 4.3 Feature pipelines

- **Images (Tiers 4/5)**: MobileNetV2 embeddings (1280-d) → PCA(128, whiten) → LDA(K−1) →
  StandardScaler, all fitted per CV fold on the training split only.
- **Text (Tier 6)**: char one-hot context windows → PCA → LDA (train-split only), contiguous
  sequential folds; EM loop alternating `TemporalSampler` (E: context construction, incl. the
  self-referential `sdf` mode) with `SDFOptimizer` (M).

### 4.4 Evaluation protocol

Fixed train/test split before CV; k-fold CV on train only; single final test evaluation.
Regression tiers report |SDF| MAE and a geometry-aware R²; classification tiers report CV/test
accuracy, class/expert counts; Tier 6 reports top-1/top-5 accuracy and perplexity against unigram
and n-gram MLE baselines.

---

## 5. Code Analysis

Verified strengths and observations from reading `src/`.

### 5.1 Strengths

- **Leakage discipline** is genuinely careful: per-fold PCA/LDA/scaler fitting, contiguous temporal
  folds, single-use test split. This is better hygiene than many published baselines.
- **`SoftminFusion` pruning** (`sdf_engine.py`): the bounding-sphere lower bound
  `SDF(x) ≥ ‖x−c‖/r_max − 1` is correct (rotation-invariant; max semi-axis bounds the ellipsoid),
  the additive-only bound for CSG experts is correctly argued (subtraction only increases SDF), and
  the pruning error is explicitly bounded (`cutoff = 10/α`). Well documented.
- **`SDFOptimizer`** derivations are correct: `P = R diag(r⁻²) Rᵀ` matches the row-vector
  convention of `compute_sdf`; `∂φ/∂μ = −Pδ/q` and `∂φ/∂P = δδᵀ/(2q)` are the standard Mahalanobis
  gradients; PD-cone projection with eigenvalue clipping is the right stabilization. This is the only
  component that updates **orientation** after construction; `NudgeEngine` now also updates
  orientation via covariance eigendecomposition (§6.5 fix applied).
- **CSG capture logic** `max(f_add, −f_sub)` is exact set-difference semantics for sign purposes,
  and the smooth-max gradient blend is a reasonable subgradient surrogate.
- Modules carry inline tests (`test_sdf_engine`, `test_inference_engine`, `test_nudge_engine`).

### 5.2 Observations (resolved) and remaining issues

1. **Non-metric "SDF"** (§1.4): `f` is a normalized distance. Softmin fusion across ellipsoids with
   very different scales weights them inconsistently in Euclidean terms. A first-order metric
   correction `f_metric ≈ f/‖∇f‖` is implemented in `compute_metric_sdf`. Sphere-tracing in
   `ray_march_depth` now uses this correction (§6.2 fix applied); capture thresholds still use the
   raw normalized value — an acceptable trade-off for unit-variance features after PCA+LDA+scaler.
2. **Ray marching** ✅ **Fixed (§6.2a)**: `InferenceEngine.ray_march_depth` now uses sphere-tracing
   steps `Δ = |f|/‖∇f‖` instead of a fixed step size — faster and scale-consistent.
3. **`NudgeEngine` radii ← std** ✅ **Fixed (§6.5)**: `NudgeEngine.apply_nudge` now uses covariance
   eigendecomposition (center ← mean, radii ← `√(λ·d)`, orientation ← eigenvectors with SVD
   re-orthogonalisation), matching the constructor's fallback convention.
4. **Minimal-sample infeasibility in high d**: the all-inlier probability `wᵏ` collapses (e.g.
   `0.5⁵⁴ ≈ 6·10⁻¹⁷` for d=9). ✅ **Partially fixed (§6.3)**: kNN-anchored seeding now used for
   half the trials when `d > 6`, providing much higher inlier purity per trial. The covariance
   fallback remains the primary fitter for high d; this is honest but worth stating explicitly.
5. **Covariance-fallback radii `√(λᵢ·d)`**: correct in expectation for seed points near the
   ellipsoid surface; remains a heuristic for filled-Gaussian clusters with small `k`.
6. **Softmin union bulge** ✅ **Fixed (§6.2)**: `SoftminFusion` now uses normalized
   `−(1/α)·ln( (1/M)·Σe^{−αf} )`. The `1/M` factor eliminates the artificial halo that grew with
   member count; M coincident ellipsoids now fuse to exactly `f`.
7. **`Expert.center`/`.radii` compatibility shims** ✅ **Fixed (§6.10)**: `.center` now returns the
   centroid of all additive ellipsoid centers; `.radii` returns the element-wise max across all
   additive ellipsoids — both meaningful for multi-member experts.
8. **README duplication** ✅ **Fixed**: duplicate Roadmap section removed.

---

## 6. Identified Issues & Suggested Improvements

Items marked ✅ have been implemented. Open items remain as future work.

### 6.1 Make the field metric (or explicitly embrace the Mahalanobis view)

Two coherent options:

- **(a) Metric SDF** ✅ **Partially applied**: `compute_metric_sdf` provides the `f̂ = f/‖∇f‖`
  correction; sphere-tracing in `ray_march_depth` uses it. Capture thresholds and softmin fusion
  still use the normalized field (acceptable with unit-variance features).
- **(b) Probabilistic view** *(open)*: include `−½ log det Σ` normalization per component and
  classify by true mixture log-likelihood. This would turn GEODE into "RANSAC-initialized MDA +
  CSG excision," connect it cleanly to [C3.1/C5.4], and give principled probabilities (potentially
  replacing Platt).

### 6.2 Normalize the softmin (mixture-consistent fusion)  ✅ Fixed

`SoftminFusion` now uses `−(1/α)·ln( (1/M) Σ e^{−αfᵢ} )` in all code paths (EllipsoidExpert
batch, Expert pruned, and small-M shortcut). Capture volume no longer grows artificially with
member count.

### 6.3 Replace minimal-sample quadric RANSAC in high d  ✅ Partially fixed

kNN-anchored seeding (`knn_seeding=True`, the new default) alternates with random seeds for
`d > 6`, providing dramatically higher inlier purity per trial. Open improvements:
- Adopt **EMS** [C2.11] as the candidate fitter.
- Add an **inlier-driven stopping rule** (Progressive-X-style model-probability test) instead of
  the fixed `max(50, 5·min_seed)` budget.

### 6.4 Principled discriminative score  ✅ Fixed

Replaced `p²/(p+n+1)` with the **F_β score** `(1+β²)·p / ((1+β²)·p + β²·n + 1)`, tunable via
`score_beta` (default 1.0 = F₁). Open extension: implement the
**log-likelihood-ratio** `Σ_captured [log p̂_class − log p̂_other]` as an alternative, which
connects the greedy step to boosting-style stagewise likelihood maximization [C3.5].

### 6.5 Fix `NudgeEngine`  ✅ Fixed

`NudgeEngine.apply_nudge` now updates center, radii, and orientation via covariance
eigendecomposition (center ← mean, radii ← `√(λ·d)`, orientation ← eigenvectors with SVD
re-orthogonalisation), matching the constructor's covariance fallback convention. The same helper
`_nudge_ellipsoid_from_points` is reused in `fit_subtractive_ellipsoids`.

### 6.6 Regularize subtractive excision against overfitting  *(open)*

Subtractive ellipsoids are fitted to _training_ false-captures and then inflated — a recipe for
carving noise. Suggestions: (i) validate each subtractive candidate on a held-out slice (accept
only if it reduces held-out misclassification), (ii) add a parsimony/MDL penalty per primitive
(coverage gain must exceed a per-primitive cost, cf. Tulsiani et al. [C2.7]), and (iii) report
expert/primitive counts alongside accuracy as a complexity metric in all tiers.

### 6.7 Add the missing baselines  *(open)*

For the claims to be credible, every classification tier should include, on the _identical_
feature pipeline: (1) GMM-per-class / MDA [C3.1] with matched component counts, (2) single
Gaussian Mahalanobis (Lee et al. [C5.4]), (3) linear/RBF SVM, (4) gradient boosting, (5) k-NN.
Tier 6 should add an LSTM/Transformer small baseline or at least a logistic-regression readout on
the same PCA/LDA features to isolate what the ellipsoid machinery contributes.

### 6.8 Exploit the natural OOD/uncertainty story  *(open)*

Distance-based classifiers' key modern selling point is _deterministic uncertainty_ (DUQ [C5.2],
SNGP [C5.3], Mahalanobis OOD [C5.4]). GEODE gets an OOD score for free (`min_class SDF`). An OOD
benchmark (e.g. CIFAR-10 in-distribution vs SVHN) would likely be the paper's most compelling
selling point and differentiates it from accuracy-only comparisons it may lose to deep baselines.

### 6.9 Temporal extension: be careful with the `sdf` context mode  *(open)*

Feeding the model's own scores back as features while refitting on data scored by an _earlier_
model is a moving-target (off-policy) EM: convergence is not guaranteed and representations can
drift. Mitigations: freeze the featurizing model per EM round (already partially true), damp
updates, and evaluate whether `sdf` mode actually beats `concat` — if not, drop the claim. Cite
reservoir computing [C6.2/C6.3] as the honest frame: fixed features + cheap readout.

### 6.10 Smaller fixes

- ✅ Removed the duplicated Roadmap section from `README.md`.
- ✅ Fixed `Expert.center`/`.radii` shims — centroid / per-axis max for multi-member experts.
- ✅ `InferenceEngine.ray_march_depth` uses sphere-tracing steps `Δ = |f|/‖∇f‖`.
- ✅ `EllipsoidExpert.__init__` `orientation` now defaults to identity (axis-aligned).
- Document in code that `∂f/∂x` is _not_ unit-norm; done via README note and `compute_metric_sdf`
  docstring.
- *(open)* Unify inline `test_*` functions into a `tests/` suite runnable via `pytest`.
  of fixed steps — faster and accurate.
- The gradient magnitude formula `∂f/∂x = R q/(a²·D)` is exact for the normalized field, but note
  in docs it is _not_ unit-norm; downstream users should not assume Eikonal behavior.
- Consider unifying the inline `test_*` functions into a `tests/` suite runnable via `pytest` so CI
  can gate changes to the geometry code (sign conventions here are easy to silently break).

---

## 7. References (Quick Index)

```
[C1.1] DeepSDF             https://arxiv.org/abs/1901.05103
[C1.2] Occupancy Networks  https://arxiv.org/abs/1812.03828
[C1.3] SIREN               https://arxiv.org/abs/2006.09661
[C1.4] IGR                 https://arxiv.org/abs/2002.10099
[C1.5] NeuS                https://arxiv.org/abs/2106.10689
[C2.1] CSGNet              https://arxiv.org/abs/1712.08290
[C2.2] UCSG-Net            https://arxiv.org/abs/2006.09102
[C2.3] CAPRI-Net           https://arxiv.org/abs/2104.05652
[C2.4] BSP-Net             https://arxiv.org/abs/1911.06971
[C2.5] CvxNet              https://arxiv.org/abs/1909.05736
[C2.6] SQ Revisited        https://arxiv.org/abs/1904.09970
[C2.7] Volumetric Prims    https://arxiv.org/abs/1612.00404
[C2.8] Neural Parts        https://arxiv.org/abs/2103.10429
[C2.9] Hierarchical Parts  https://arxiv.org/abs/2004.01176
[C2.10] Marching-Prims     https://arxiv.org/abs/2303.13190   ★ closest shape-SDF work
[C2.11] EMS Ellipsoid Fit  https://arxiv.org/abs/2110.13337
[C3.1] MDA                 https://doi.org/10.1111/j.2517-6161.1996.tb02085.x  ★ closest ML analogue
[C3.2] RANSAC              https://doi.org/10.1145/358669.358692
[C3.3] PEARL               https://doi.org/10.1007/s11263-011-0474-7
[C3.4] Progressive-X       https://arxiv.org/abs/1906.02290
[C3.5] Gradient Boosting   https://doi.org/10.1214/aos/1013203451
[C3.6] Matching Pursuit    https://doi.org/10.1109/78.258082
[C3.7] SVDD                https://doi.org/10.1023/B:MACH.0000008084.60811.49
[C3.8] One-Class SVM       https://doi.org/10.1162/089976601750264965
[C3.9] LVQ                 https://doi.org/10.1109/5.58325
[C4.1] Jacobs MoE 1991     https://doi.org/10.1162/neco.1991.3.1.79
[C4.2] Sparse MoE          https://arxiv.org/abs/1701.06538
[C4.3] Switch Transformer  https://arxiv.org/abs/2101.03961
[C5.1] Prototypical Nets   https://arxiv.org/abs/1703.05175
[C5.2] DUQ                 https://arxiv.org/abs/2003.02037
[C5.3] SNGP                https://arxiv.org/abs/2006.10108
[C5.4] Mahalanobis OOD     https://arxiv.org/abs/1807.03888   ★ closest ML analogue
[C5.5] Platt Scaling       https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf
[C6.1] Forward-Forward     https://arxiv.org/abs/2212.13345
[C6.2] Echo State Networks Jaeger, GMD Report 148, 2001 (grey literature)
[C6.3] Liquid State Mach.  https://doi.org/10.1162/089976602760407955
[C7.3] Multisurface Sep.   https://doi.org/10.1007/BF02614780
[C7.4] DC Programming/CCCP https://doi.org/10.1162/08997660360581958
```
