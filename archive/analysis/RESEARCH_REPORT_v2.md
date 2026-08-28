# GEODE: Research Report v2

**Greedy Ellipsoidal Outline Discrimination by Excision — Background, Prior Work, Methodology, and Code Analysis**

_Date: 2026-07-25 (revised from v1 to reflect committed improvements in branch HEAD)_

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Background Concepts](#2-background-concepts)
3. [Prior Work and Overlap Analysis](#3-prior-work-and-overlap-analysis)
4. [Methodology](#4-methodology)
5. [Code Analysis](#5-code-analysis)
6. [Remaining Issues and Recommendations](#6-remaining-issues-and-recommendations)
7. [Novelty Assessment](#7-novelty-assessment)
8. [References](#8-references)

---

## 1. Project Summary

**GEODE** (Greedy Ellipsoidal Outline Discrimination by Excision) is a constructive geometric classifier. Each class is represented by a union of anisotropic oriented ellipsoidal experts grown via a two-level greedy RANSAC loop. Class boundaries are sharpened by fitting **subtractive (excised) ellipsoids** that carve inter-class overlap via CSG set-difference, giving each expert a region of the form:

> *interior(E\_add) \ interior(E\_sub1) \ interior(E\_sub2) \ ...*

Distances are encoded as Mahalanobis-style signed distance fields (SDFs); at inference, per-class SDF stacks are fused by a softmin (log-sum-exp) that approximates the log-likelihood of a Gaussian mixture, and a final calibration layer (isotonic or logistic) converts fused SDFs to class probabilities. The result is an interpretable, analytic-gradient model with geometric primitives that can be rendered, edited, and queried for OOD detection without any neural backbone.

---

## 2. Background Concepts

### 2.1 Signed Distance Fields (SDFs) and the Eikonal Property

An SDF `f : ℝᵈ → ℝ` satisfies `f(x) < 0` inside a shape, `f(x) = 0` on the boundary, and `f(x) > 0` outside. A _metric_ SDF additionally satisfies the **Eikonal equation** `‖∇f‖ = 1` everywhere except at ridges, which means `|f(x)|` equals the Euclidean distance to the nearest surface. This property is crucial for sphere-tracing, gradient-based boundary queries, and physically meaningful distance comparisons.

GEODE uses **Mahalanobis-style SDFs** of the form:

```
f(x) = √( Σᵢ qᵢ²/aᵢ² ) − 1
```

where `qᵢ` are coordinates in the ellipsoid's local frame and `aᵢ` are the semi-axes. This is an iso-surface SDF (zero set is the ellipsoid surface), but **‖∇f‖ ≠ 1** for anisotropic ellipsoids, so it is _not_ a metric SDF. This has implications for softmin fusion (see §4.3) and ray marching (see §4.5).

### 2.2 Constructive Solid Geometry (CSG)

CSG represents complex shapes as boolean combinations (union, intersection, set-difference) of primitive solids. For signed distance fields:
- **Union**: `min(f₁, f₂)` (or softmin for smooth blending)
- **Intersection**: `max(f₁, f₂)`
- **Set-difference (A \ B)**: `max(fₐ, −f_b)`

GEODE exploits the set-difference operation to subtract (excise) inter-class regions from a class expert, the core discriminative mechanism.

### 2.3 Mixture Discriminant Analysis (MDA)

MDA [[Hastie & Tibshirani 1996](#ref-mda)] models each class as a Gaussian mixture. Classification assigns a point to the class with the highest mixture density. For class `c` with `K_c` mixture components:

```
p(x | c) = (1/K_c) Σₖ N(x; μ_ck, Σ_ck)
```

The log-sum-exp identity shows that GEODE's softmin fusion approximates `log p(x | c)` up to a constant, with the smoothing parameter `α` controlling how tightly each primitive contributes. GEODE can be viewed as a **RANSAC-fitted, CSG-extended MDA**.

### 2.4 RANSAC for Geometric Model Fitting

RANSAC [[Fischler & Bolles 1981](#ref-ransac)] fits geometric models to data in the presence of outliers. The classic algorithm:
1. Sample a minimal subset (size `s`) to fit the model hypothetically
2. Count inliers (points within tolerance)
3. Repeat until the probability of finding a better model is low

The number of trials needed for confidence `p` with inlier fraction `ε` is:
```
N_trials = log(1 − p) / log(1 − εˢ)
```

For large `s` (which grows quadratically with feature dimension `d`: `s = d(d+3)/2` for an ellipsoid fit), this number grows exponentially, making classical RANSAC impractical for high-dimensional feature spaces.

### 2.5 Log-Sum-Exp Softmin as Mixture Log-Likelihood

The _softmin_ of a set of values `{fₖ}` is:

```
softmin({fₖ}) = −(1/α) log( (1/M) Σₖ exp(−α·fₖ) )
```

The 1/M normalization factor ensures the operation is **mixture-consistent**: for a uniform mixture of `M` components, the softmin of the component densities approximates the mixture log-density, independent of `M`. Without the 1/M factor, adding more components would artificially lower the fused SDF value (the "capture-halo growth" artifact), where a point far from all individual components could still be spuriously captured by the ensemble.

### 2.6 Sphere Tracing

Sphere tracing [[Hart 1996](#ref-sphere-tracing)] is an efficient ray-marching algorithm that, for metric SDFs, takes steps of exactly `|f(x)|` (the safe sphere radius around the current point), guaranteeing no surface is missed. For non-metric SDFs, the step should be `|f(x)| / ‖∇f(x)‖` to recover the safe sphere radius.

---

## 3. Prior Work and Overlap Analysis

### 3.1 Closest Prior Works (Highest Overlap)

| # | Work | Venue | Overlap | Key Difference |
|---|------|-------|---------|----------------|
| 1 | Hastie & Tibshirani (1996) — **MDA** | JRSS-B | GEODE's softmin fusion IS mixture log-likelihood; same Gaussian mixture per class structure | GEODE adds RANSAC fitting, anisotropic orientation updates, and CSG subtraction |
| 2 | Lee et al. (2018) — **Mahalanobis OOD** | NeurIPS | Same pipeline: embedding → per-class Mahalanobis score → calibration layer | GEODE uses RANSAC-fitted mixtures per class, not a single Gaussian; adds CSG subtraction |
| 3 | Liu et al. (2023) — **Marching Primitives** | CVPR | Iterative greedy ellipsoid growth from SDF; same primitive type | Marching Primitives fits to 3D geometry; GEODE applies the same idea to feature-space classification |
| 4 | Barath & Matas (2019) — **Progressive-X** | ICCV | Greedy anytime multi-model RANSAC; outer loop structure mirrors GEODE's two-level loop | Progressive-X handles homography/F-matrix fitting; GEODE targets ellipsoidal class regions |

### 3.2 SDF-Based Machine Learning

| Work | Venue | Relevance |
|------|-------|-----------|
| Park et al. (2019) — DeepSDF [[arXiv:1901.05103]](https://arxiv.org/abs/1901.05103) | CVPR | Implicit neural SDF representation of 3D shapes; GEODE uses analytic (not neural) SDFs |
| Atzmon & Lipman (2020) — SAL [[arXiv:2006.05400]](https://arxiv.org/abs/2006.05400) | NeurIPS | Unsigned-to-signed SDF learning; background for implicit representations |
| Gropp et al. (2020) — IGR [[arXiv:2002.10099]](https://arxiv.org/abs/2002.10099) | ICML | Eikonal regularization for neural SDFs; motivates GEODE's metric SDF correction |
| Mescheder et al. (2019) — Occupancy Networks [[arXiv:1812.03828]](https://arxiv.org/abs/1812.03828) | CVPR | Binary occupancy classifier → SDF equivalent; conceptually adjacent to GEODE's inside/outside framing |
| Chen & Zhang (2019) — IM-NET [[arXiv:1812.02822]](https://arxiv.org/abs/1812.02822) | CVPR | Implicit decoder for shape generation; background for implicit function methods |
| Takikawa et al. (2021) — NGLOD [[arXiv:2101.10994]](https://arxiv.org/abs/2101.10994) | CVPR | Neural Geometric LOD with octree-structured SDFs; demonstrates multi-scale SDF hierarchies |

### 3.3 Gaussian Mixture and Distance-Based Classifiers

| Work | Venue | Relevance |
|------|-------|-----------|
| Hastie & Tibshirani (1996) — **MDA** [[DOI]](https://doi.org/10.1111/j.2517-6161.1996.tb02085.x) | JRSS-B | Direct predecessor; GEODE ≈ RANSAC-fitted MDA with CSG subtraction |
| Reynolds (2009) — GMM Tutorial [[Link]](https://scholar.google.com/scholar?q=Gaussian+Mixture+Models+Reynolds+2009) | MIT Lincoln Laboratory | Standard GMM reference; GEODE uses GMM-like per-class models |
| Lee et al. (2018) — **Mahalanobis OOD** [[arXiv:1807.03888]](https://arxiv.org/abs/1807.03888) | NeurIPS | Closest pipeline match (embed → Mahalanobis distance → calibration); GEODE extends to RANSAC mixtures + CSG |
| Ruff et al. (2018) — Deep SVDD [[arXiv:1803.04903]](https://arxiv.org/abs/1803.04903) | ICML | End-to-end hypersphere classifier; GEODE uses analytic fitting instead of gradient optimization |
| Sun et al. (2022) — KNN OOD [[arXiv:2204.06507]](https://arxiv.org/abs/2204.06507) | NeurIPS | kNN distance for OOD detection; related to GEODE's nearest-class-SDF inference |
| Ming et al. (2023) — SPE [[arXiv:2301.12321]](https://arxiv.org/abs/2301.12321) | ICLR | Spurious OOD feature elimination; motivates GEODE's discriminative excision |
| Chen et al. (2022) — DICE [[arXiv:2203.07341]](https://arxiv.org/abs/2203.07341) | ECCV | Sparsification for OOD; related to GEODE's feature-space boundary approach |

### 3.4 Primitive Fitting and Shape Abstraction

| Work | Venue | Relevance |
|------|-------|-----------|
| Tulsiani et al. (2017) — Volumetric Primitives [[arXiv:1612.00404]](https://arxiv.org/abs/1612.00404) | CVPR | Differentiable primitive fitting; GEODE uses greedy RANSAC instead of gradient fitting |
| Paschalidou et al. (2019) — Superquadrics [[arXiv:1904.09970]](https://arxiv.org/abs/1904.09970) | CVPR | Superquadric primitive decomposition; broader primitive family than GEODE's ellipsoids |
| Deng et al. (2020) — CVX-Net [[arXiv:1909.05736]](https://arxiv.org/abs/1909.05736) | CVPR | Convex decomposition via neural implicits; related but uses gradient-based fitting |
| Liu et al. (2023) — **Marching Primitives** [[arXiv:2303.13190]](https://arxiv.org/abs/2303.13190) | CVPR | Closest structural match: greedy ellipsoid growth from SDF; GEODE applies to feature-space classification |
| Murino et al. (1998) — Ellipsoidal Basis [[DOI]](https://doi.org/10.1016/S0031-3203(97)00059-6) | Pattern Recog. | Ellipsoidal basis functions for classification; GEODE adds RANSAC fitting and CSG subtraction |

### 3.5 RANSAC and Robust Estimation

| Work | Venue | Relevance |
|------|-------|-----------|
| Fischler & Bolles (1981) — **RANSAC** [[DOI]](https://doi.org/10.1145/358669.358692) | CACM | Foundational RANSAC reference; GEODE's construction loop is a two-level RANSAC |
| Raguram et al. (2013) — USAC [[DOI]](https://doi.org/10.1109/TPAMI.2012.257) | TPAMI | Universal RANSAC framework; GEODE could adopt DEGENSAC and SPRT-based termination |
| Barath & Matas (2018) — MAGSAC [[arXiv:1803.07469]](https://arxiv.org/abs/1803.07469) | CVPR | Marginalization-based RANSAC; scores each sample by integration over inlier thresholds |
| Barath & Matas (2019) — **Progressive-X** [[arXiv:1906.02290]](https://arxiv.org/abs/1906.02290) | ICCV | Greedy anytime multi-model RANSAC; same algorithmic structure as GEODE's two-level loop |

### 3.6 CSG and Implicit Representations for Learning

| Work | Venue | Relevance |
|------|-------|-----------|
| Sharma et al. (2018) — CSGNet [[arXiv:1712.08290]](https://arxiv.org/abs/1712.08290) | CVPR | Neural CSG program synthesis; GEODE uses CSG analytically without neural program search |
| Ren et al. (2021) — CAPRI-Net [[arXiv:2104.05652]](https://arxiv.org/abs/2104.05652) | CVPR | Convex hull union for shape reconstruction; intersections not set-differences; different from GEODE's excision |
| Kania et al. (2020) — UCF [[arXiv:2006.09102]](https://arxiv.org/abs/2006.09102) | 3DV | Union of convex functions for SDF; structurally similar but no discriminative excision |
| Jones et al. (2022) — SHRED [[arXiv:2207.09786]](https://arxiv.org/abs/2207.09786) | SIGGRAPH | Explicit CSG tree search for shape editing; different application domain but related representation |

### 3.7 Smooth CSG and Distance Approximations

| Work | Venue | Relevance |
|------|-------|-----------|
| Inigo Quilez (various years) — Smooth SDF Operations [[Link]](https://iquilezles.org/articles/smin/) | Blog | Polynomial smooth-min; related to GEODE's log-sum-exp smooth-min |
| Blinn (1982) — Blobby Molecules [[DOI]](https://doi.org/10.1145/800064.801290) | SIGGRAPH | Exponential density blending; ancestor of GEODE's softmin fusion |
| Muraki (1991) — Volumetric Shape [[DOI]](https://doi.org/10.1145/127719.122743) | SIGGRAPH | Blobby model fitting from range data; related primitive fitting approach |

### 3.8 Calibration and Conformal Prediction

| Work | Venue | Relevance |
|------|-------|-----------|
| Platt (1999) — Platt Scaling [[Link]](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf) | Workshop | Sigmoid calibration of SVM scores; related to GEODE's isotonic/logistic calibration |
| Guo et al. (2017) — Calibration of NNs [[arXiv:1706.04599]](https://arxiv.org/abs/1706.04599) | ICML | Temperature scaling; directly applicable to GEODE's output calibration |
| Angelopoulos et al. (2021) — Conformal Prediction [[arXiv:2107.07511]](https://arxiv.org/abs/2107.07511) | ICLR | Prediction sets with coverage guarantees; applicable to GEODE's geometric decision regions |

---

## 4. Methodology

### 4.1 Primitive SDF: Mahalanobis Ellipsoid

The core primitive is an oriented `d`-dimensional ellipsoid parameterized by:
- Center `μ ∈ ℝᵈ`
- Semi-axis lengths `a ∈ ℝᵈ` (positive)
- Orientation matrix `R ∈ SO(d)` (rotation / frame alignment)

The SDF of a point `x` is computed by transforming to the ellipsoid's local frame:

```
q = R^T (x − μ)
f(x) = √( Σᵢ qᵢ²/aᵢ² ) − 1
```

This is **not a metric SDF** — for anisotropic ellipsoids, `‖∇f‖ ≠ 1`. The gradient is:

```
∇f(x) = (1 / ‖Σᵢ qᵢ²/aᵢ²‖) · R · (q / a²)    [component-wise a²]
```

A metric correction can be obtained as `f_metric = f / ‖∇f‖`, now available via `EllipsoidExpert.compute_metric_sdf()`. This is used by the sphere tracer (§4.5) for proper step sizing.

**Semi-axis fitting** (as of commit 5e9f738): Given inlier points `{xᵢ}`, the NudgeEngine computes the sample covariance `C = (1/n) Σ (xᵢ − μ)(xᵢ − μ)^T`, takes its eigendecomposition `C = V Λ V^T`, and sets:

```
aₖ = √(λₖ · d)     [scaling by √d for d-dimensional ellipsoid volume]
R = V               [eigenvectors form the orientation frame]
```

Orientation is re-orthogonalized via SVD (`U, _, Vt = svd(V); R = U @ Vt`) to handle numerical drift. Previously, the NudgeEngine updated only the center and semi-axes using world-frame standard deviations, ignoring off-diagonal covariance and leaving orientation unchanged.

### 4.2 CSG Expert Model

An **Expert** is a collection of `EllipsoidExpert` primitives with polarities `+1` (additive) or `−1` (subtractive). The expert SDF is:

```
f_expert(x) = max(  softmin({fₖ : polarity = +1}),
                   −softmin({fₖ : polarity = −1})  )
```

The `max` of `f_add` and `−f_sub` is the set-difference (CSG subtraction) of the additive volume by the subtractive volume. This is exact at hard boundaries; the `softmin` provides smooth blending within each polarity group.

### 4.3 Softmin Fusion and the 1/M Normalization

The softmin (log-sum-exp) of `M` values with smoothing `α > 0` is:

```
softmin({fₖ}) = −(1/α) ln( (1/M) Σₖ exp(−α·fₖ) )
```

The 1/M normalization ensures mixture consistency: if all `M` ellipsoids contribute equally, the fused SDF represents a single ellipsoid-equivalent rather than a tighter (lower-valued) blob. Without it, adding more ellipsoids lowers the fused SDF across the captured region (the "capture-halo" artifact), causing a class expert to spuriously capture points near the edges of its members' union.

The CPU implementation in `SoftminFusion` now normalizes all three computation paths:
1. **Standard path**: `-(1/α) * log(sum(exp(-α*f)) / M)`
2. **Fast path** (M < 4): same 1/M applied
3. **Pruned path**: divides by `M_a` (number of _active_ experts after pruning) rather than total M, since pruned experts contribute negligibly (`exp(-10) ≈ 0`)

The class-level fusion in `SoftminFusion._fuse_class_sdf` applies the same 1/M normalization across experts belonging to the same class.

### 4.4 Two-Level Greedy RANSAC Construction

GEODE's construction loop in `GreedyConstructor.fit_subtractive_ellipsoids` has two nested levels:

**Outer loop** (expert growth):
- Maintain a pool of "unexplained" points for each class
- While unexplained pool is non-empty and budget not exhausted:
  - Run inner loop to fit a new expert to unexplained points
  - Remove captured points from the pool
  - Add expert to the class model

**Inner loop** (ellipsoid packing within an expert):
- While the expert can still grow (stagnation check):
  - Sample `T` candidate hypotheses, each via random seed subset of size `min_seed = d(d+3)/2`
  - For `knn_seeding=True` (default when d > 6): every other trial uses an anchor point + its `k` nearest neighbors as the seed set, improving conditioning in high-d
  - Score candidates by **F_β discriminative score**:
    ```
    F_β = (1+β²) · pos / ((1+β²) · pos + β² · neg + 1)
    ```
    where `pos` = points of the current class captured, `neg` = points of other classes captured (leakage), and `β = score_beta` (default 1.0 for balanced F₁)
  - Accept the best candidate; stop if no improvement (stagnation check)

**Subtractive phase**: After the additive expert is grown, a separate pass with `polarity = -1` fits subtractive ellipsoids that target the inter-class overlap region (points where both additive SDF < 0 and any competing class SDF < 0).

### 4.5 Sphere-Tracing Ray March

The `InferenceEngine.ray_march_depth` method traces a ray from `origin` along `direction` to find the first surface crossing. As of commit 5e9f738, it uses **sphere tracing**:

```python
grad = expert.gradient(x)
grad_norm = ‖grad‖
sphere_step = max(|f(x)| / grad_norm, step_size * 0.1)
x ← x − sign(f(x)) * (grad / grad_norm) * sphere_step
```

This replaces the previous fixed-step advance with an adaptive step derived from the local SDF gradient magnitude, which is the correct sphere-tracing formula for non-metric SDFs (`f / ‖∇f‖` gives the safe sphere radius). The `step_size * 0.1` guard ensures a minimum advance per iteration to prevent stalling at flat regions.

### 4.6 Calibration

After construction, the model's raw SDF values are converted to class probabilities by a calibration layer (`_fit_calibrator` in `eval_complex_classification.py`). GEODE supports:
- **Isotonic regression** (non-parametric, monotonic; preferred when calibration data is plentiful)
- **Logistic regression** (Platt scaling; more regularized for small calibration sets)

The calibration input is the vector of class SDF values at each point; the output is a probability simplex.

---

## 5. Code Analysis

### 5.1 File Structure

```
src/
  sdf_engine.py          # Core primitives: EllipsoidExpert, SoftminFusion, Expert
  greedy_constructor.py  # Two-level RANSAC construction loop
  nudge_engine.py        # Post-construction ellipsoid refinement (EM-style)
  inference_engine.py    # Ray marching, SDF queries
  sdf_optimizer.py       # Gradient-based refinement (SDFOptimizer with PD projection)
  gpu_engine.py          # OpenCL GPU kernels for SDF computation

experiments/
  tier4/eval_complex_classification.py   # Main classification pipeline orchestration
  tier5/eval_cifar100_superclass.py      # CIFAR-100 superclass evaluation
analysis/
  RESEARCH_REPORT.md                     # v1 report (superseded by this document)
  RESEARCH_REPORT_v2.md                  # this document
README.md
```

### 5.2 `sdf_engine.py` — Core Primitives

**`EllipsoidExpert`** — implements the Mahalanobis-style SDF, its gradient, and bounding-sphere pruning.

✅ **Resolved (commit 5e9f738)**: Added `compute_metric_sdf(x)` returning `f / ‖∇f‖`, documented as the proper metric SDF for sphere tracing and scale-consistent distance comparisons. The class docstring now explicitly notes the non-Eikonal property.

✅ **Resolved**: `Expert.center` now returns the centroid of all additive ellipsoids (not just the first). `Expert.radii` returns per-axis maximum semi-axis across all additive ellipsoids. Previously both properties returned the values of the first ellipsoid only, making multi-ellipsoid expert bounding boxes incorrect.

✅ **Resolved**: `SoftminFusion` — all three CPU fusion paths now normalize by 1/M (or 1/M_a for pruned path), eliminating the capture-halo growth artifact.

### 5.3 `nudge_engine.py` — Ellipsoid Refinement

✅ **Resolved (commit 5e9f738)**: `_nudge_ellipsoid_from_points()` now performs full covariance eigendecomposition, updating center, semi-axes, AND orientation. Previously, orientation was never updated in this stage (the world-frame `std` was used to set `radii`, which conflates the axes and the orientation). The re-orthogonalization via SVD prevents drift when eigenvectors are near-parallel.

### 5.4 `inference_engine.py` — Ray Marching

✅ **Resolved (commit 5e9f738)**: Fixed-step ray marching replaced with adaptive sphere tracing (`|f| / ‖∇f‖` step sizing). Updated tests in the same file reflect the new behavior.

### 5.5 `greedy_constructor.py` — Construction Loop

✅ **Resolved**: F_β scoring (`_disc_score`) replaces ad-hoc p²/(p+n+1). Tunable `score_beta` parameter (default 1.0).

✅ **Resolved**: kNN seeding (`_knn_seed`) alternates with random sampling for high-d trials when `knn_seeding=True`.

✅ **Resolved (this report)**: F_β comparison baseline mismatch fixed. After the F_β scoring change, the inner loop's acceptance threshold was `float(prev_captured_count)` (a raw count, e.g., 50) while the score was F_β ∈ [0,1). This made it impossible for a second ellipsoid to ever be added to a discriminative-mode expert (`F_β ≤ 1 < 50`). Fixed by introducing `prev_score = 0.0` that tracks the best F_β (or raw count in non-discriminative mode) and is updated after each accepted ellipsoid, ensuring the comparison is always in the same metric as the score.

### 5.6 `gpu_engine.py` — OpenCL Kernels

❌ **Not updated**: The OpenCL kernels `expert_softmin_csg` and `class_softmin` still use unnormalized log-sum-exp:

```c
// Lines 132, 146, 176 — missing 1/cnt normalization:
const float f_add = -1.0f / alpha * log(sumexp_add);   // should be log(sumexp_add / cnt_add)
const float f_sub = -1.0f / alpha * log(sumexp_sub);   // should be log(sumexp_sub / cnt_sub)
class_sdf[n * C + c] = -1.0f / alpha * log(sumexp);    // should be log(sumexp / cnt)
```

The CPU and GPU paths now give different results for multi-component experts. Any experiment comparing CPU vs. GPU inference will see discrepancies that grow with the number of ellipsoids.

### 5.7 `sdf_optimizer.py` — Gradient-Based Refinement

The `SDFOptimizer` class contains its own `_stable_softmin` helper:

```python
def _stable_softmin(self, values, alpha):
    shifted = values - np.min(values)
    return -np.log(np.sum(np.exp(-alpha * shifted))) / alpha + np.min(values)
```

This is the unnormalized version (no 1/M factor), inconsistent with the updated `SoftminFusion`. For the optimizer's loss landscape, the unnormalized form means the loss decreases as more ellipsoids are added even without improving separation quality. However, since `sdf_optimizer.py` operates on a single expert at a time, the inconsistency is milder than in the class-level fuser — it mainly affects the optimizer's objective scale, not inter-expert comparisons.

---

## 6. Remaining Issues and Recommendations

### 6.1 High Priority

#### 6.1.1 Synchronize GPU Kernels with CPU Softmin Normalization
**File**: `src/gpu_engine.py`, lines 132, 146, 176  
**Impact**: High — CPU and GPU inference produce different results for any multi-component expert, silently corrupting any experiment that benchmarks GPU vs CPU paths.  
**Fix**: Add 1/cnt normalization inside `log()` at each of the three kernel sites:

```c
// Line 132:
const float f_add = -1.0f / alpha * log(sumexp_add / (float)cnt_add);
// Line 146:
const float f_sub = -1.0f / alpha * log(sumexp_sub / (float)cnt_sub);
// Line 176:
class_sdf[n * C + c] = -1.0f / alpha * log(sumexp / (float)cnt);
```

### 6.2 Medium Priority

#### 6.2.1 Update `sdf_optimizer.py` Softmin
**File**: `src/sdf_optimizer.py`, `_stable_softmin` method  
**Impact**: Medium — optimizer objective is not scale-consistent with SoftminFusion; optimization may favor configurations that look better by count than by geometric fit quality.  
**Fix**: Add `- log(M) / alpha` to the return value, or pass `M` explicitly.

#### 6.2.2 Normalize GPU Greedy Scoring Path
**File**: `src/greedy_constructor.py`, GPU path softmin identity (around line 333–336)  
**Impact**: Medium — the "softmin of two" combining existing expert SDF with candidate SDF is unnormalized (1/2 factor missing).  
**Fix**: Add the 1/2 normalization: `combined = -(1/α)*log(0.5*(exp(-α*existing) + exp(-α*cand)))`.

#### 6.2.3 Subtractive Carving Held-Out Validation
**Impact**: Medium research value — subtractive ellipsoids may overfit on small training sets, excising correct regions at test time.  
**Fix**: Reserve a held-out validation slice (e.g., 20% of training) and only accept a subtractive ellipsoid candidate if it reduces false captures on the validation slice (not just training). This is analogous to the early stopping typically applied in pruning classifiers.

#### 6.2.4 RANSAC Iteration Budget Formula
**Impact**: Medium — `max(50, 5·min_seed)` is ad-hoc and does not match the classic RANSAC termination formula.  
**Recommendation**: Compute the required number of trials as:
```python
N_trials = ceil(log(1 - confidence) / log(1 - inlier_fraction ** min_seed))
```
with `confidence=0.99` and `inlier_fraction` estimated from the pool occupancy (fraction of unexplained points that are class-positive). The kNN seeding already improves the effective inlier fraction for the seeded trials; this budget formula would complement it.

### 6.3 Lower Priority

#### 6.3.1 Pytest Suite
The inline `test_*` functions are not discovered by pytest. Wrapping them in a `tests/` directory with standard pytest structure would enable CI integration.

#### 6.3.2 Classical Baselines
No classical baselines (GMM-EM, MDA, SVM, Isolation Forest) are compared in the experiments. Adding these in `experiments/` would validate GEODE's claim of superior geometric expressibility and OOD sensitivity.

#### 6.3.3 OOD Evaluation
The architecture is well-suited for OOD detection (a point with min-class SDF > threshold is OOD), but no OOD benchmarks are yet run. The `compute_metric_sdf()` method provides the metric distance for threshold-based OOD detection; combining this with the calibrator output should give competitive results compared to Lee et al. (2018).

---

## 7. Novelty Assessment

### 7.1 Strongest Claim: CSG Set-Difference as a Discriminative Classifier

No prior work found applies **CSG set-difference of ellipsoids as a discriminative mechanism** — i.e., explicitly fitting and subtracting inter-class overlap regions. CSGNet [[Sharma et al. 2018]](https://arxiv.org/abs/1712.08290) uses CSG for 3D shape reconstruction, CAPRI-Net [[Ren et al. 2021]](https://arxiv.org/abs/2104.05652) uses convex hull unions, and MDA/GEODE overlap at the additive mixture level — but none of these uses _subtractive excision_ to improve a classifier's precision. This is GEODE's most distinctive and defensible contribution.

### 7.2 Comparison with Direct Predecessors

| Aspect | MDA [Hastie 1996] | Mahalanobis OOD [Lee 2018] | Marching Primitives [Liu 2023] | **GEODE** |
|--------|:-:|:-:|:-:|:-:|
| Per-class Gaussian mixture | ✅ | ✅ (single) | ✅ (geometric, not classifier) | ✅ |
| Anisotropic ellipsoids | ✅ | ✅ | ✅ | ✅ |
| Orientation updates (EM) | ✅ | ✅ | partial | ✅ |
| RANSAC fitting | ❌ | ❌ | ❌ | ✅ |
| kNN seeding | ❌ | ❌ | ❌ | ✅ |
| CSG subtraction (excision) | ❌ | ❌ | ❌ | ✅ |
| SDF-native inference | ❌ | ❌ | ✅ | ✅ |
| Sphere-tracing ray march | ❌ | ❌ | ❌ | ✅ |
| OOD via geometric distance | ❌ | ✅ | ❌ | ✅ |
| Metric SDF correction | ❌ | ❌ | ❌ | ✅ (compute_metric_sdf) |
| Calibrated probabilities | ❌ | ✅ | ❌ | ✅ |
| Interpretable primitives | partial | ❌ | ✅ | ✅ |

### 7.3 Positioning Statement

> GEODE can be positioned as: *RANSAC-fitted Mixture Discriminant Analysis with CSG set-difference excision, analytic SDF inference, and metric-corrected geometric distance.*

The incremental value over MDA is: (1) RANSAC robustness to outliers in the construction phase, (2) the excision mechanism for discriminative precision, (3) SDF-native inference enabling geometric rendering and OOD detection, and (4) the F_β tunable precision-recall trade-off in construction.

---

## 8. References

<a id="ref-mda"></a>
- **[Hastie & Tibshirani 1996]** Hastie, T., & Tibshirani, R. (1996). Discriminant Analysis by Gaussian Mixtures. *Journal of the Royal Statistical Society: Series B*, 58(1), 155–176. https://doi.org/10.1111/j.2517-6161.1996.tb02085.x

<a id="ref-ransac"></a>
- **[Fischler & Bolles 1981]** Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. *CACM*, 24(6), 381–395. https://doi.org/10.1145/358669.358692

<a id="ref-sphere-tracing"></a>
- **[Hart 1996]** Hart, J. C. (1996). Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces. *The Visual Computer*, 12(10), 527–545. https://doi.org/10.1007/BF02439180

- **[Lee et al. 2018]** Lee, K., Lee, K., Lee, H., & Shin, J. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. *NeurIPS*. https://arxiv.org/abs/1807.03888

- **[Liu et al. 2023]** Liu, Y., et al. (2023). Marching-Primitives: Shape Abstraction from Signed Distance Function. *CVPR*. https://arxiv.org/abs/2303.13190

- **[Barath & Matas 2019]** Barath, D., & Matas, J. (2019). Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm. *ICCV*. https://arxiv.org/abs/1906.02290

- **[Park et al. 2019]** Park, J. J., et al. (2019). DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. *CVPR*. https://arxiv.org/abs/1901.05103

- **[Sharma et al. 2018]** Sharma, G., et al. (2018). CSGNet: Neural Shape Parser for Constructive Solid Geometry. *CVPR*. https://arxiv.org/abs/1712.08290

- **[Ren et al. 2021]** Ren, Z., et al. (2021). CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly. *CVPR*. https://arxiv.org/abs/2104.05652

- **[Tulsiani et al. 2017]** Tulsiani, S., et al. (2017). Learning Shape Abstractions by Assembling Volumetric Primitives. *CVPR*. https://arxiv.org/abs/1612.00404

- **[Ruff et al. 2018]** Ruff, L., et al. (2018). Deep One-Class Classification. *ICML*. https://arxiv.org/abs/1803.04903

- **[Sun et al. 2022]** Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. *NeurIPS*. https://arxiv.org/abs/2204.06507

- **[Guo et al. 2017]** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*. https://arxiv.org/abs/1706.04599

- **[Raguram et al. 2013]** Raguram, R., et al. (2013). USAC: A Universal Framework for Random Sample Consensus. *TPAMI*, 35(8), 2022–2038. https://doi.org/10.1109/TPAMI.2012.257

- **[Barath & Matas 2018]** Barath, D., & Matas, J. (2018). MAGSAC: Marginalizing Sample Consensus. *CVPR*. https://arxiv.org/abs/1803.07469

- **[Paschalidou et al. 2019]** Paschalidou, D., et al. (2019). Superquadrics Revisited: Learning 3D Shape Parsing Beyond Cuboids. *CVPR*. https://arxiv.org/abs/1904.09970

- **[Deng et al. 2020]** Deng, B., et al. (2020). CVX-Net: Learnable Convex Decomposition. *CVPR*. https://arxiv.org/abs/1909.05736

- **[Angelopoulos et al. 2021]** Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. https://arxiv.org/abs/2107.07511

- **[Mescheder et al. 2019]** Mescheder, L., et al. (2019). Occupancy Networks: Learning 3D Reconstruction in Function Space. *CVPR*. https://arxiv.org/abs/1812.03828

- **[Gropp et al. 2020]** Gropp, A., et al. (2020). Implicit Geometric Regularization for Learning Shapes. *ICML*. https://arxiv.org/abs/2002.10099

- **[Atzmon & Lipman 2020]** Atzmon, M., & Lipman, Y. (2020). SAL: Sign Agnostic Learning of Shapes from Raw Data. *CVPR*. https://arxiv.org/abs/2006.05400

---

_Report generated by AI analysis of GEODE source code (branch HEAD, post-commit 5e9f738). All code references are to the committed state._
