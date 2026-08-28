---
description: Project specification and architectural reference for the GEODE codebase. Load when working on any source file in src/, experiments/, or verify_pipeline.py.
applyTo: "**/*.py"
---

# Project Specification: GEODE — Generalized Encoders for Open-Domain Expertise

## 1. Executive Summary

**GEODE** (**G**eneralized **E**ncoders for **O**pen-**D**omain **E**xpertise) is the project's promoted system — the latest frozen system with the best measured combination of choices. The geometric framework below (historically **G**reedy **E**llipsoidal **O**utline **D**iscrimination by **E**xcision) is its legacy core, retained for the `src/` code; it is a framework for learning data manifolds by assembling them from analytically defined geometric primitives rather than learning abstract weights. Experts are **groups of oriented ellipsoids** — both additive and subtractive — whose fused CSG Signed Distance Fields (SDFs) collectively approximate complex surfaces and class boundaries. A RANSAC-inspired greedy assembly process builds each expert incrementally with discriminative boundary awareness, and a multi-stage post-construction pipeline sharpens class boundaries via CSG excision, active repair, and task-appropriate logistic score calibration.

## 2. Core Components

### 2.1 Primitive Unit - `EllipsoidExpert` (`src/sdf_engine.py`)

A d-dimensional oriented ellipsoid with a **polarity** p in {+1, -1}:

- **Additive** (p = +1, default): contributes positive volume.
- **Subtractive** (p = -1): marks a CSG hole -- placed at class-boundary overlap regions.

The SDF uses the eigenvector orientation matrix R from the fitted quadratic form:
q = (x - c)^T R, f(x) = sqrt(sum_i q_i^2 / a_i^2) - 1

### 2.2 Composite Expert - `Expert` (`src/sdf_engine.py`)

An `Expert` groups `EllipsoidExpert` primitives partitioned by polarity. Its SDF is a **CSG set-difference**:
f_add = SoftMin over additive ellipsoids
f_sub = SoftMin over subtractive ellipsoids
f_exp(x) = max(f_add(x), -f_sub(x))

When no subtractive ellipsoids exist this reduces to plain Softmin. The gradient blends both contributions via smooth-max weights:
w_add = exp(a*f_add) / (exp(a*f_add) + exp(-a*f_sub))
grad f_exp = w_add * grad f_add - w_sub \* grad f_sub

### 2.3 Model-level Fusion - `InferenceEngine` (`src/inference_engine.py`)

D*total(x) = -(1/alpha) * ln( (1/M) \_ sum_j exp(-alpha \* f_exp_j(x)) )

The `1/M` normalization ensures M coincident experts fuse to exactly `f` (no artificial capture halo).

### 2.4 Greedy RANSAC Constructor - `GreedyConstructor` (`src/greedy_constructor.py`)

**Outer loop**: grow and lock experts until the unexplained pool is empty.

**Inner loop** - grows one `Expert`:

1. Draw a minimal seed of size min_seed = d\*(d+3)/2 from uncaptured points.
2. Fit a candidate via SVD on the quadratic design matrix. If eigenvalues <= 0 (hyperboloid),
   **fall back to sample covariance**: radii = sqrt(eigenvalues \* d), sorted descending.
3. Score each candidate:
   - **Discriminative mode** (exclude*points provided): score = F*β = (1+β²)·pos / ((1+β²)·pos + β²·neg + 1)
     where pos = class pool captured, neg = exclude pool captured (subsampled to 300).
     β is controlled by `score_beta` (default 1.0 = F₁). Rewards coverage, penalises contamination.
   - **Standard mode**: score = pos (maximise captures only).
4. For `d > 6` with `knn_seeding=True` (default), every other trial uses a kNN-anchored seed
   (random anchor + k nearest neighbours) for much higher inlier purity in high dimensions.
5. Keep the highest-scoring candidate.
6. Stop when growth stalls (< min_growth_fraction \* pool_size).

**Auto iteration budget** (max*iterations=None):
N_iter = max(50, 5 * d\_(d+3)/2)
Always pass `max_iterations=None` for CIFAR/classification unless you have a specific reason to override.

### 2.5 Subtractive Ellipsoid Fitting - `fit_subtractive_ellipsoids` (`src/greedy_constructor.py`)

Post-construction boundary sharpening, applied to every CV fold and the final model:

1. Find other-class points falsely captured inside the expert (SDF < capture_threshold) using
   SIGNED SDF -- NOT abs(SDF). This correctly includes all points inside the additive volume,
   not just near-surface ones.
2. Iteratively RANSAC-fit ellipsoids to those false-capture points.
3. Inflate fitted radii by capture_threshold (ensures points are strictly inside the CSG hole).
4. Nudge center/radii/orientation toward false-capture geometry via covariance eigendecomposition.
5. Attach with polarity = -1.
   Effort tiers: <10% overlap -> 1x; 10-25% -> 2x; >25% -> 3x iteration budget.

### 2.6 Active Misclassification Repair - `_active_repair` (`experiments/tier4/eval_complex_classification.py`)

A second targeted subtractive pass applied after standard boundary sharpening:

- Identifies other-class points with SDF < -2 \* capture_threshold (deep inside, most harmful).
- Runs a focused subtractive RANSAC on only those clusters.
- Complements fit_subtractive_ellipsoids which targets the broader boundary shell.

### 2.7 Nudge Engine - `NudgeEngine` (`src/nudge_engine.py`)

Force-directed refinement of additive ellipsoids after construction. Each point is assigned to the
nearest expert/ellipsoid; centers nudge toward centroid, radii toward std.

### 2.8 Sequential State and SDF Refinement (`src/temporal_sampler.py`, `src/sdf_optimizer.py`)

Sequential tasks keep the core GEODE learning kernel unchanged:

causal representation -> SDF ellipsoid experts -> calibrated SDF-score readout

`TemporalStateEncoder` creates a deterministic, fixed-width causal recurrent state. Its width is
independent of vocabulary size and observation width, allowing the same interface to represent text,
sensors, trajectories, and event streams. Train, calibration, and test data MUST use the same encoder
parameters and seed. The encoder is a representation layer, not a replacement learning kernel, and
Tier 6 does not backpropagate through time.

`SDFOptimizer` performs bounded supervised refinement of existing additive `EllipsoidExpert`
parameters. It uses vectorized minibatch evaluation and analytic gradients of the nested normalized
Softmin loss with respect to center and precision matrix. Precision updates are projected to the
positive-definite cone before recovering radii and orientation. This is supervised refinement, not
an EM M-step: no latent variable is inferred. Subtractive ellipsoids MUST be rejected by this
optimizer until the CSG gradient path is explicitly implemented and tested.

## 3. Classification Pipeline (Tier 4 - CIFAR-10)

Full pipeline in `experiments/tier4/eval_complex_classification.py`.
**All transforms are fitted on training data only per fold -- no data leakage.**

### 3.1 Feature Extraction (fitted once globally)

1. **MobileNetV2** (ImageNet-1k, backbone only) -> 1280-dim embeddings
   `load_cifar_npz` returns raw CNN/HOG features + labels.

### 3.2 Per-fold Transform (refitted for every CV fold and the final model)

2. **PCA** (n_components=128, whiten=True) -- compress + unit-variance
3. **LDA** (n_components = n_classes - 1 = 9) -- maximally discriminative subspace
4. **StandardScaler** -- unit variance per LDA dim (scale-consistent capture_threshold)
   Functions: `_build_transform(X_train, y_train, pca_components, seed)` -> (pca, lda, scaler)
   `_apply_transform(X, pca, lda, scaler)` -> transformed X

### 3.3 Expert Fitting

5. **Discriminative RANSAC** per class: `fit_class_models` passes `exclude_points = X[y != class_id]`
   to `fit_experts` -> `build_model`. Score: pos^2/(pos+neg+1).

### 3.4 Boundary Sharpening

6. `add_subtractive_ellipsoids` -- signed-SDF false-capture carving (SDF < capture_threshold)
7. `_active_repair` -- deep false-positive second pass (SDF < -2 \* capture_threshold)

### 3.5 Calibration and Prediction

8. `compute_score_scales` -- mean(|SDF|) per class on training data
9. `compute_raw_scores` -- (N x n_classes) normalised score matrix
10. `_fit_calibrator` -- LogisticRegression on score matrix -> class labels (Platt scaling)
11. `predict_labels(..., calibrator=calibrator)` -- uses calibrator.predict() when provided

**Prediction criterion**: calibrated argmax. Falls back to argmin(SDF/scale) when no calibrator.
Do NOT use argmin(|SDF|): near-surface points are ambiguous.

## 4. Temporal Text Pipeline (Tier 6)

Full pipeline in `experiments/tier6/eval_temporal_text_prediction.py`.

1. Convert the corpus to versioned printable-ASCII character IDs plus `<unk>`.
2. Build contiguous causal states with `TemporalStateEncoder` (verification default: 16 dimensions).
3. Use forward-chaining splits and purge gaps between geometry, calibration, and evaluation ranges.
4. Fit PCA/LDA/StandardScaler on training history only.
5. Select spherical, diagonal, or full ellipsoid complexity from per-class sample adequacy.
6. Fit additive GEODE experts and compute normalized per-class SDF scores.
7. Fit `StandardScaler` + multinomial `LogisticRegression` on held-out calibration scores.
8. Optionally run bounded `SDFOptimizer` updates, accepting refinement only when it does not regress.

Raw `argmin(SDF)` is diagnostic only because independently fitted class SDFs are not directly
comparable. The score calibrator is the production readout; it does not replace the geometric model.
Report raw and calibrated accuracy separately, probability perplexity over the complete vocabulary,
Top-5 accuracy, sample adequacy, and unigram, n-gram, and linear controls.

The staged 50k-train / 10k-test WikiText-103 GPU run produced 23.91% calibrated accuracy, 54.68%
Top-5 accuracy, and 21.04 perplexity. The unigram baseline was 19.22%; the 3-gram baseline was 48.79%.

## 5. Key Design Rules

1. **Always pass max_iterations=None** for classification fits -- the auto-formula computes the
   right budget from d.
2. **Subtractive ellipsoids and active repair must be applied in every CV fold** (not only the
   final model) so that CV accuracy and test accuracy measure the same complete pipeline.
3. **capture_threshold is scale-sensitive** -- all feature pipelines must produce unit-variance
   dimensions (via PCA whiten=True + StandardScaler after LDA).
4. **polarity=-1 ellipsoids are only fitted by fit_subtractive_ellipsoids and \_active_repair**,
   never by build_model. The greedy constructor only produces additive experts.
5. **Covariance fallback is essential in high-d** -- random seeds in >=8D rarely form valid
   ellipsoids without it.
6. **Per-fold LDA** -- never fit PCA/LDA/StandardScaler on the full dataset before the train/test
   split. Use \_build_transform on training folds only.
7. **Signed SDF for false-capture detection** -- use `sdf < threshold`, NOT `abs(sdf) < threshold`
   in fit_subtractive_ellipsoids. Points deep inside (SDF << 0) are the most important to carve.
8. **Preserve the SDF kernel in sequential tasks** -- temporal encoders produce features for GEODE;
   score calibrators consume GEODE SDF scores. Neither is an alternative classifier kernel.
9. **Use one causal transform across splits** -- keep the temporal encoder seed and parameters fixed,
   and never warm a validation or test state with future observations.
10. **Keep calibration held out** -- fit score calibration after geometry fitting on a purged,
    chronologically later calibration segment.
11. **Refine additive geometry only** -- `SDFOptimizer` must reject subtractive CSG models until
    subtractive derivatives are implemented.

## 6. Evaluation Strategy

All tiers use fixed train/test split + k-fold CV on training data only.

- **Tiers 1-3** (regression): CV/Test MAE of |SDF|, geometry-aware R^2.
- **Tier 4** (classification): CV accuracy +/- std, Test accuracy, experts fitted, classes modeled.
- **Tier 6** (temporal classification): forward-chaining CV, raw/calibrated Top-1, Top-5,
  probability perplexity, pre/post-refinement accuracy, and representation baselines.

Sample adequacy diagnostics (printed by load_cifar_npz):

- min_seed = d\*(d+3)/2 -- minimum quadric seed size (d = n_classes - 1 after LDA)
- minimum = 2 _ min_seed _ n_classes / 0.8 total samples -- RANSAC can run
- recommended = 10 _ min_seed _ n_classes / 0.8 total samples -- good convergence
- max d (minimum tier) and max d (recommended) -- upper bound on safe d for given N

## 7. File Map

```
src/
  sdf_engine.py          -- EllipsoidExpert (polarity), Expert (CSG SDF + gradient)
  greedy_constructor.py  -- GreedyConstructor (discriminative RANSAC), fit_subtractive_ellipsoids
  nudge_engine.py        -- NudgeEngine
  inference_engine.py    -- InferenceEngine (model-level SoftMin, ray marching)
   temporal_sampler.py    -- causal TemporalStateEncoder and sequential pair construction
   sdf_optimizer.py       -- vectorized analytic refinement of additive SDF ellipsoids
experiments/
  common/
    moe_eval.py          -- fit_experts (exclude_points), run_cv_with_fixed_train_test, run_cv_then_test
    dataset_utils.py     -- dataset download helpers
  tier1/                 -- geometry regression (sphere, ellipsoid)
  tier2/                 -- point-cloud reconstruction (ModelNet10)
  tier3/                 -- MNIST manifold fitting
  tier4/
    eval_complex_classification.py  -- CIFAR-10 pipeline:
                                       load_cifar_npz (raw features)
                                       _build_transform / _apply_transform (per-fold)
                                       fit_class_models (discriminative)
                                       add_subtractive_ellipsoids + _active_repair
                                       compute_raw_scores + _fit_calibrator + predict_labels
   tier6/
      eval_temporal_text_prediction.py  -- causal representation, adaptive GEODE fitting,
                                                             held-out score calibration, refinement, and metrics
      test_temporal_text_prediction.py  -- fast temporal and optimizer regression tests
      test_gpu_parity.py                -- CPU/OpenCL inference and construction parity
verify_pipeline.py       -- end-to-end verification, all tiers
```
