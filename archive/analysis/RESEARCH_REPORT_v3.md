# GEODE Research Report v3

**Greedy Ellipsoidal Outline Discrimination by Excision: background, current methodology, contributions, limitations, and research agenda**

Date: 2026-07-25  
Code scope: branch `em-rework`, commit `4354999`, plus the benchmark in
`logs/results/2026-07-25 - verify pipeline benchmark (1).txt`.

---

## Executive Summary

GEODE is a constructive geometric learning system in which class regions and
data manifolds are assembled from oriented ellipsoids. Each primitive provides
a closed-form Mahalanobis-style signed field. Multiple primitives are fused by
normalized softmin, while subtractive primitives can remove incorrectly
captured regions using constructive solid geometry (CSG) set difference.

The project combines five established ideas:

1. ellipsoidal/Mahalanobis class models;
2. mixture discriminant analysis;
3. greedy multi-model RANSAC;
4. CSG composition of implicit primitives; and
5. logistic calibration of distance-derived scores.

Its strongest potentially novel contribution is the **discriminative fitting of
subtractive ellipsoids to other-class false captures**. The ingredients are not
new individually, but this use of CSG excision as a classifier-boundary repair
mechanism appears distinctive relative to the prior work reviewed here.

The implementation is substantially stronger than earlier project reports
suggest. CPU and OpenCL paths now use stable normalized softmin, tests cover
CPU/GPU parity, temporal sampling is causal, class models adapt to sparse
support, and analytic SDF refinement is vectorized. The latest verification run
passes all seven reported tasks. It obtains 84.00% on the CIFAR-10 subset,
64.37% on CIFAR-100 superclasses, and 23.35% character accuracy on the Tier 6
WikiText-103 experiment.

Those results validate software execution, not yet scientific superiority.
There are no matched GMM, Mahalanobis, SVM, k-NN, or boosting baselines on the
same image embeddings. There is no ablation isolating subtractive CSG. Tier 6
beats its unigram and linear controls but is far below the 3-gram baseline
(23.35% versus 48.79%). Its production configuration disables both subtractive
CSG and supervised SDF refinement. The full multiclass logistic score readout
also contributes materially to prediction, so its effect must be isolated from
the geometry.

The approach is viable and improvable. The highest-value next step is not a
larger benchmark. It is a controlled ablation suite that compares identical
features and splits across additive GEODE, GEODE plus excision, calibrated
GEODE, GMM/MDA, single-Gaussian Mahalanobis, and standard classifiers. That
experiment would determine whether the distinctive geometry contributes beyond
the pretrained representation and logistic readout.

---

## 1. Scope and Evidence

This report is based on the current source code, tests, documentation, and the
latest full verification output. It supersedes factual claims in the earlier
reports where the implementation has changed.

Important scope limitations:

- This is a technical literature positioning exercise, not a systematic review.
- A claim that no identical prior method was found is not proof of novelty.
- The verification pipeline uses reduced or specialized task settings and is
  not directly comparable to standard leaderboard protocols.
- Passing a pipeline gate establishes internal consistency, not state-of-the-art
  performance or statistical significance.

The central falsifiable research hypothesis is:

> Fitting subtractive ellipsoids to other-class false captures improves
> generalization at a fixed representation, component budget, calibration
> method, and compute budget.

The present experiments do not isolate this hypothesis. A matched ablation is
therefore required before excision can be claimed as an empirical contribution.

---

## 2. Background and Prior Art

### 2.1 Implicit fields and signed distance functions

An implicit field represents a boundary as the zero level set of a function
$f: R^d -> R$. A signed distance function is negative inside the represented
region, zero at the boundary, and positive outside. A metric SDF additionally
satisfies the Eikonal property $||\nabla f|| = 1$ almost everywhere, making
$|f(x)|$ the Euclidean distance to the surface.

GEODE uses the closed-form ellipsoidal field

$$
f(x) = \sqrt{(x-c)^T P(x-c)} - 1,
\qquad
P = R\,\mathrm{diag}(a^{-2})R^T,
$$

where $c$ is the center, $a$ contains semi-axis lengths, and $R$ is an
orthonormal orientation matrix. This field has the correct sign and zero set,
but it is not a metric Euclidean SDF for anisotropic ellipsoids. It is better
understood as normalized Mahalanobis radius minus one.

Neural implicit methods such as DeepSDF, Occupancy Networks, IGR, and SIREN
learn flexible fields with neural networks. GEODE occupies the opposite design
point: a restricted primitive family with explicit parameters, analytic
gradients, direct CSG composition, and inspectable geometry.

Relevant work:

- Park et al., DeepSDF (CVPR 2019): https://arxiv.org/abs/1901.05103
- Mescheder et al., Occupancy Networks (CVPR 2019):
  https://arxiv.org/abs/1812.03828
- Gropp et al., Implicit Geometric Regularization (ICML 2020):
  https://arxiv.org/abs/2002.10099
- Sitzmann et al., SIREN (NeurIPS 2020):
  https://arxiv.org/abs/2006.09661

### 2.2 Ellipsoids, Mahalanobis distance, and mixture discriminant analysis

Ellipsoidal level sets arise naturally from Gaussian covariance models. A
single class-conditional Gaussian yields a quadratic discriminant boundary;
multiple Gaussians per class yield Mixture Discriminant Analysis (MDA). GEODE's
additive class model is structurally close to an MDA model because each class
contains multiple oriented covariance-like regions and combines their scores
with log-sum-exp.

The analogy is not exact. A normalized Gaussian negative log-likelihood
contains a squared Mahalanobis term, a log-determinant term, mixture weights,
and class priors. GEODE uses Mahalanobis radius minus one and usually uniform
normalized softmin weights. It is therefore a geometric mixture score, not a
proper Gaussian mixture likelihood.

Closest classical references include:

- Hastie and Tibshirani, Discriminant Analysis by Gaussian Mixtures
  (JRSS-B 1996): https://doi.org/10.1111/j.2517-6161.1996.tb02085.x
- Lee et al., Mahalanobis Distance for OOD Detection (NeurIPS 2018):
  https://arxiv.org/abs/1807.03888
- Tax and Duin, Support Vector Data Description (Machine Learning 2004):
  https://doi.org/10.1023/B:MACH.0000008084.60811.49
- Snell et al., Prototypical Networks (NeurIPS 2017):
  https://arxiv.org/abs/1703.05175

Lee et al. is particularly relevant to Tiers 4 and 5: both approaches apply a
pretrained image representation, compute class-conditional Mahalanobis-like
scores, and learn a logistic readout. GEODE generalizes the class model from one
Gaussian region to multiple RANSAC-built regions and adds optional CSG excision.

### 2.3 RANSAC and multi-model fitting

RANSAC repeatedly draws a minimal sample, fits a hypothesis, and evaluates
consensus. For inlier fraction $w$, minimal sample size $s$, and confidence
$p$, the classical trial estimate is

$$
N = \left\lceil
\frac{\log(1-p)}{\log(1-w^s)}
\right\rceil.
$$

GEODE performs sequential multi-model fitting: it grows one expert from the
unexplained pool, removes captured points, and repeats. This is related to
sequential RANSAC and Progressive-X. Its high-dimensional challenge is severe:
the nominal ellipsoid seed size $s=d(d+3)/2$ grows quadratically, while $w^s$
shrinks exponentially. GEODE addresses this with k-nearest-neighbor anchored
seeds, covariance fallback, sample-adequacy checks, and a capped vectorized GPU
candidate path.

Relevant work:

- Fischler and Bolles, RANSAC (CACM 1981):
  https://doi.org/10.1145/358669.358692
- Raguram et al., USAC (TPAMI 2013):
  https://doi.org/10.1109/TPAMI.2012.257
- Barath and Matas, MAGSAC (CVPR 2018):
  https://arxiv.org/abs/1803.07469
- Barath and Matas, Progressive-X (ICCV 2019):
  https://arxiv.org/abs/1906.02290
- Zhao et al., EMS robust ellipsoid fitting (ICCV 2021):
  https://arxiv.org/abs/2110.13337

### 2.4 Primitive decomposition and constructive solid geometry

CSG builds complex regions by unions, intersections, and differences of simple
solids. In sign-preserving field form,

$$
f_{A \cup B} = \min(f_A,f_B),
\qquad
f_{A \cap B} = \max(f_A,f_B),
\qquad
f_{A \setminus B} = \max(f_A,-f_B).
$$

Learned primitive decomposition is well established in 3D reconstruction.
CSGNet, UCSG-Net, CAPRI-Net, BSP-Net, CvxNet, superquadric decomposition, and
Marching-Primitives all overlap with aspects of GEODE's representation. The
main distinction is task and fitting objective: GEODE applies primitive
assembly in feature-space classification and fits subtractive regions from
other-class errors.

Relevant work:

- Tulsiani et al., Learning Shape Abstractions by Assembling Volumetric
  Primitives (CVPR 2017): https://arxiv.org/abs/1612.00404
- Sharma et al., CSGNet (CVPR 2018): https://arxiv.org/abs/1712.08290
- Paschalidou et al., Superquadrics Revisited (CVPR 2019):
  https://arxiv.org/abs/1904.09970
- Chen et al., BSP-Net (CVPR 2020): https://arxiv.org/abs/1911.06971
- Deng et al., CvxNet (CVPR 2020): https://arxiv.org/abs/1909.05736
- Kania et al., UCSG-Net (NeurIPS 2020):
  https://arxiv.org/abs/2006.09102
- Yu et al., CAPRI-Net (CVPR 2022): https://arxiv.org/abs/2104.05652
- Liu et al., Marching-Primitives (CVPR 2023):
  https://arxiv.org/abs/2303.13190

### 2.5 Mixtures of experts and geometric routing

Classical mixture-of-experts systems jointly learn local experts and a gating
network. Sparse modern MoEs use learned top-k routing and load-balancing losses.
GEODE instead uses geometric routing: lower class SDF means stronger membership,
and softmin provides continuous component attribution. This is interpretable
and does not require a neural gate, but the routing is not jointly optimized
with the upstream representation.

- Jacobs et al., Adaptive Mixtures of Local Experts (Neural Computation 1991):
  https://doi.org/10.1162/neco.1991.3.1.79
- Shazeer et al., Sparsely-Gated MoE (ICLR 2017):
  https://arxiv.org/abs/1701.06538
- Fedus et al., Switch Transformers (JMLR 2022):
  https://arxiv.org/abs/2101.03961

### 2.6 Calibration versus learned readout

Distance values from independently fitted class models are rarely comparable
without scale correction. Platt scaling and temperature scaling are standard
ways to map scores to probabilities.

GEODE's Tier 6 readout is more expressive than scalar temperature scaling. It
standardizes the complete class-SDF vector and fits multinomial logistic
regression. For $K$ classes this learns a $K$-input multiclass linear decision
layer. It should therefore be described as a **learned score readout** as well
as a calibrator. It does not replace the ellipsoid representation, but it can
change decision boundaries and may account for a substantial portion of final
accuracy.

- Platt, Probabilistic Outputs for Support Vector Machines (1999)
- Guo et al., On Calibration of Modern Neural Networks (ICML 2017):
  https://arxiv.org/abs/1706.04599

### 2.7 Fixed recurrent features and reservoir computing

Tier 6 uses a deterministic random recurrent state:

$$
h_t = \tanh(W_{in}x_t + \rho h_{t-1}W_r),
$$

where $W_{in}$ is a fixed random projection, $W_r$ is a fixed orthogonal
projection, and $\rho$ is a recurrence coefficient. This is much closer to an
Echo State Network or reservoir computer than to an RNN trained by
backpropagation through time. GEODE learns the geometric readout; it does not
learn the recurrent transition.

- Jaeger, The Echo State Approach to Analysing and Training Recurrent Neural
  Networks (GMD Report 148, 2001)
- Maass et al., Real-Time Computing Without Stable States
  (Neural Computation 2002): https://doi.org/10.1162/089976602760407955

### 2.8 Open-world novelty and unlabeled category discovery

The new automatic-grouping objective has direct precedents. Stream novelty
methods buffer observations that do not fit known concepts and promote
persistent clusters as candidate novelties. Spinosa et al. developed
cluster-based novel-concept detection for streams; Masud et al. studied novel
classes under concept drift and time constraints; and MINAS formalized a
multiclass stream lifecycle for novelty detection. This is the closest prior
lineage to GEODE's rejection buffer and persistent review groups.

Streaming clustering contributes complementary memory mechanisms. DenStream
uses fading density micro-clusters to track evolving structure and noise.
HDBSCAN extracts stable clusters across density levels, while FINCH constructs
a hierarchy from first-neighbor relations without a supplied cluster count or
distance threshold. These methods address persistence, variable density, and
unknown group count, but none can recover semantic groups that the embedding
does not separate.

Visual category-discovery work targets that representation problem. Deep
Transfer Clustering and AutoNovel transfer knowledge from labeled classes and
learn from labeled and unlabeled observations jointly. UNO and Generalized
Category Discovery similarly couple representation learning with discovery;
GCD explicitly permits unlabeled data to contain both known and novel classes
and studies class-count estimation. ORCA addresses bias toward seen classes in
open-world semi-supervised learning. DeepCluster and SCAN further demonstrate
that semantic clustering quality depends on a representation trained for the
partitioning objective rather than on a density algorithm alone.

Human confirmation also has established algorithmic precedents. Interactive
clustering methods such as COBRAS turn sparse must-link and cannot-link answers
into partition refinements. GEODE should reuse that idea after autonomous group
formation instead of treating a user-supplied name as evidence that every
member belongs to a newly publishable class.

Consequently, buffering rejected observations and automatically clustering
them is not a novel contribution. The defensible GEODE-specific research scope
is their integration with versioned support profiles, stable review IDs and
sample provenance, editable ellipsoid graphs, and confirmation-gated,
transactional adaptation with replay, calibration, validation, and rollback.

---

## 3. Current Methodology

### 3.1 Primitive representation

`src/sdf_engine.py` defines `EllipsoidExpert` by center $c$, positive radii
$a$, orientation $R$, and polarity $p \in \{+1,-1\}$. Points are transformed
to local coordinates before evaluating normalized radius. The primitive offers
closed-form gradients and a first-order metric correction $f/||\nabla f||$.

Positive polarity contributes captured volume. Negative polarity defines a CSG
hole. Parameters are directly inspectable and can be edited without retraining
an opaque network.

### 3.2 Expert and class fusion

An `Expert` contains one or more ellipsoids. Positive members are fused using
normalized softmin:

$$
\mathrm{softmin}_\alpha(f_1,\ldots,f_M)
= -\frac{1}{\alpha}
\log\left(\frac{1}{M}\sum_{m=1}^{M}e^{-\alpha f_m}\right).
$$

The $1/M$ factor prevents coincident duplicate components from changing the
fused value. If subtractive members exist, the expert uses

$$
f_{expert}(x)=\max(f_{add}(x),-f_{sub}(x)).
$$

Multiple experts for one class are fused by the same normalized softmin. CPU
code uses stable log-sum-exp and bounding-sphere pruning. OpenCL inference
implements ellipsoid evaluation, expert CSG fusion, and class fusion in float32.
The current kernels include the same component-count normalization as CPU.

### 3.3 Greedy construction

`src/greedy_constructor.py` implements two nested loops.

**Inner loop:**

1. Draw a minimal random or kNN-anchored seed from currently uncaptured points.
2. Fit a general quadric by SVD.
3. Fall back to a covariance ellipsoid if the quadric is not positive definite.
4. Temporarily add the candidate and evaluate newly captured class points.
5. When negative examples are supplied, score purity and coverage with
   $F_\beta$.
6. Accept the best improving candidate and stop when growth stagnates.

**Outer loop:**

1. Lock the grown expert if it captures the required fraction of the current
   unexplained pool.
2. Remove captured points.
3. Start another expert until insufficient support remains.

The GPU training path is not merely a mechanical acceleration of the CPU path
for $d>6$: it generates batches of covariance ellipsoids directly and caps the
candidate budget. This is a practical approximation to the CPU hypothesis
generator and should be treated as a method variant in experiments.

### 3.4 Geometric refinement

Two refinement mechanisms are present.

`NudgeEngine` assigns points to experts and constituent ellipsoids, then moves
centers and covariance eigensystems toward assigned data.

`SDFOptimizer` performs supervised analytic-gradient refinement of additive
ellipsoids. For $\delta=x-c$, $q=\sqrt{\delta^TP\delta}$,

$$
\frac{\partial f}{\partial c}=-\frac{P\delta}{q},
\qquad
\frac{\partial f}{\partial P}=\frac{\delta\delta^T}{2q}.
$$

The optimizer differentiates through nested softmin attribution, applies
heavy-ball momentum, projects updated precision matrices onto the
positive-definite cone, and recovers radii and orientation by eigendecomposition.
Minibatch evaluation and gradient accumulation are vectorized. Subtractive CSG
is explicitly rejected because its optimizer derivative is not implemented.

This is supervised refinement, not expectation-maximization: labels are
observed and no latent variable posterior is inferred.

### 3.5 Discriminative excision

For image classification, `fit_subtractive_ellipsoids` identifies other-class
points captured by an additive expert and fits negative-polarity ellipsoids to
those regions. `_active_repair` performs a second pass on deeply captured false
positives. The resulting sign region is an additive union minus a subtractive
union.

This is the mechanism most specific to GEODE. It is enabled in Tiers 4 and 5,
but disabled in the current Tier 6 verification configuration.

### 3.6 Feature and evaluation pipelines

**Geometry tiers (1-3).** Tiers 1 and 2 reconstruct synthetic or point-cloud
surfaces. Tier 3 fits a manifold to examples of MNIST digit zero after PCA. They
report absolute field error and a project-specific geometry-normalized $R^2$.

**Image tiers (4-5).** Images pass through an ImageNet-pretrained MobileNetV2
backbone. PCA whitening, LDA, and standardization are fitted inside each
training fold. GEODE fits one class model per label, adds subtractive repair,
normalizes class scores, and trains multinomial logistic regression on the SDF
score vector. Tier 4 uses CIFAR-10; Tier 5 maps CIFAR-100 labels to 20
superclasses.

The reported image accuracy is therefore accuracy of the complete
`pretrained backbone -> supervised transform -> GEODE -> logistic readout`
pipeline, not accuracy attributable to GEODE alone.

**Temporal tier (6).** Text is encoded as 96 printable-ASCII-plus-unknown IDs.
A contiguous block is converted to fixed-width causal reservoir states.
Forward-chaining folds and purge gaps separate geometry fitting, calibration,
and validation. PCA, LDA, and standardization are fitted on past data only.
Classes with sufficient support receive full RANSAC geometry; sparse classes
fall back to diagonal or spherical ellipsoids. Per-class field scales are
estimated from in-class samples. A held-out chronological segment trains the
standardized multinomial score readout.

The verification configuration uses 50,000 training pairs, 10,000 test pairs,
a 16-dimensional temporal state, two forward folds, GPU construction/inference,
no subtractive ellipsoids, and zero supervised refinement iterations.

### 3.7 Verification safeguards

Current safeguards include:

- versioned Tier 6 corpus metadata and vocabulary fingerprinting;
- per-class sample-adequacy reporting;
- forward-only temporal splits and purge gaps;
- separate geometry and score-calibration segments;
- stable probability-based perplexity;
- CPU/OpenCL inference and candidate-scoring parity tests;
- GPU cache invalidation after model mutation;
- optimizer rejection of unsupported subtractive geometry; and
- fast synthetic temporal and optimizer regression tests.

---

## 4. Current Empirical Evidence

The latest full verification run reports:

| Tier | Task                                |                            Main result |           Time |
| ---- | ----------------------------------- | -------------------------------------: | -------------: |
| 1    | Sphere regression                   |   Test MAE 0.0221, custom $R^2$ 0.9993 |          1.7 s |
| 1    | Ellipsoid regression                |     Test MAE 0.0264, radii RMSE 0.3178 | included above |
| 2    | ModelNet point-cloud reconstruction |                        Test MAE 0.1202 |        161.9 s |
| 3    | MNIST digit-0 manifold              |                        Test MAE 0.3538 |         23.1 s |
| 4    | CIFAR-10 classification             |                   Test accuracy 84.00% |         54.6 s |
| 5    | CIFAR-100 superclass classification |                   Test accuracy 64.37% |       1193.8 s |
| 6    | WikiText-103 character prediction   | Test accuracy 23.35%, perplexity 20.41 |        114.9 s |

Tier 6 controls were:

| Model/control                             | Accuracy |
| ----------------------------------------- | -------: |
| Unigram                                   |   19.22% |
| Linear readout on temporal-state features |   20.97% |
| GEODE plus calibrated SDF readout         |   23.35% |
| 3-gram lookup                             |   48.79% |

The Tier 6 result establishes that the geometric score representation contains
useful information beyond the tested linear control, but it does not establish
competitive sequence modeling. The 25.44 percentage-point gap to the 3-gram
baseline is the dominant result.

All pipeline tiers passed their current internal thresholds. The thresholds are
primarily regression guards; for example, Tier 6 is required to beat unigram,
remain finite, model at least two classes, and avoid a large refinement
regression. They are not external performance standards.

---

## 5. Contributions

### 5.1 Strongest research contribution: discriminative CSG excision

GEODE fits subtractive geometric primitives from other-class false captures and
uses exact sign-level CSG difference to remove those regions. Prior primitive
decomposition work uses subtraction for shape reconstruction, and prior
distance classifiers use ellipsoidal regions, but this report found no direct
precedent for fitting subtractive ellipsoids specifically as discriminative
classification repair.

This should be presented as a **candidate contribution pending ablation**, not
as an established empirical advantage.

### 5.2 Robust constructive ellipsoidal classifier

The project integrates sequential multi-model construction, discriminative
$F_\beta$ candidate scoring, kNN-anchored seeding, covariance fallback, adaptive
class complexity, and normalized softmin fusion. Each ingredient has precedent,
but their combination provides an interpretable alternative to EM-fitted
mixtures and neural heads.

### 5.3 Analytic, editable geometry with supervised refinement

Centers, axes, orientations, and polarities remain explicit. The same objects
support field queries, gradients, CSG composition, bounding-volume pruning,
and positive-definite analytic refinement. This unifies classification and
geometric reconstruction more directly than a conventional classifier head.

### 5.4 Representation-independent sequential readout

Tier 6 demonstrates that the geometric model can consume a fixed-width causal
state rather than a static image embedding. This supports the architectural
claim that GEODE is a readout over arbitrary feature spaces. The contribution
is system flexibility, not a new recurrent learning rule: the temporal encoder
is a fixed random reservoir.

### 5.5 Engineering contributions

The implementation includes matched CPU/OpenCL semantics, stable normalized
softmin, GPU candidate scoring without downloading full candidate matrices,
cache invalidation, vectorized analytic gradients, causal validation splits,
adaptive sparse-class geometry, and focused regression tests. These are useful
engineering contributions even where the underlying algorithms are established.

### 5.6 Claims that should not be made yet

Current evidence does not support claims that GEODE:

- outperforms MDA, GMMs, SVMs, k-NN, or standard neural heads;
- improves classification because of subtractive CSG specifically;
- provides calibrated probabilities under distribution shift;
- provides validated out-of-distribution detection;
- performs true metric sphere tracing for arbitrary ellipsoid compositions;
- learns temporal dynamics end to end;
- validates supervised SDF refinement on the real Tier 6 corpus; or
- is competitive with even a simple count-based language model.

---

## 6. Issues, Solvability, and Improvements

### 6.1 Critical scientific issue: missing matched baselines and ablations

**Status (2026-07-25): Resolved for Tier 4; Tier 5 replication remains.**

**Problem.** Tiers 4 and 5 report only the complete GEODE pipeline. Because the
pipeline uses pretrained MobileNetV2, supervised LDA, and multinomial logistic
calibration, final accuracy cannot be attributed to the geometric model or CSG
excision.

**Solvability:** High. This is an experimental-design issue, not a fundamental
limitation.

**Required experiment.** On identical cached embeddings and fixed splits,
compare:

1. multinomial logistic regression on transformed features;
2. nearest centroid and single-Gaussian Mahalanobis;
3. class-conditional GMM/MDA with matched component count;
4. k-NN, linear SVM, RBF SVM, and gradient boosting;
5. additive GEODE with raw argmin;
6. additive GEODE with score calibration;
7. additive plus subtractive GEODE with the same calibration; and
8. active repair as a separate final ablation.

Report accuracy, negative log-likelihood, expected calibration error, fit time,
inference time, and primitive count over at least five seeds. Use paired
bootstrap confidence intervals on test prediction differences.

The matched Tier 4 experiment now runs all eight classical baselines, five
GEODE readouts, and the A0/A1/A2 CSG variants on identical transformed features
and untouched test indices for seeds `[11, 23, 37, 53, 71]`. RBF SVM had the
highest mean accuracy at 84.21%, direct logistic regression reached 83.96%,
GEODE feature-logistic reached 84.03%, and the best geometry-native GEODE
readout reached 83.66%. Direct logistic regression had the best mean NLL
(0.456) and ECE (0.019). The artifact also reports fit/inference time and model
complexity. A1 changed mean accuracy by only -0.0125 to +0.0125 percentage
points across readouts, every nonzero paired 95% interval crossed zero, and A2
added no candidates. Thus the missing Tier 4 comparison is repaired, but it
does not establish a CSG benefit or a GEODE advantage over matched classical
methods. The same complete matrix has not yet been rerun on Tier 5.

### 6.2 Critical interpretation issue: the calibrator is a learned classifier

**Status (2026-07-25): Resolved for Tier 4.**

**Problem.** Multinomial logistic regression over the full SDF vector can mix
all class scores. This is more expressive than scalar calibration and may repair
poor raw geometric decisions. Earlier staged Tier 6 diagnostics showed a large
raw-versus-calibrated gap; the consolidated benchmark currently hides raw
accuracy.

**Solvability:** High.

**Improvement.** Always report:

- raw `argmin(SDF/scale)` accuracy;
- diagonal calibration, where each class uses only its own score;
- temperature scaling;
- full multinomial SDF readout; and
- logistic regression directly on the same transformed input features.

This determines whether ellipsoids create useful nonlinear basis features or
whether the final linear layer carries most of the task.

All five readouts are now retained as separate result records. On the matched
five-seed Tier 4 study, raw GEODE reached 83.66% accuracy but had NLL 1.509 and
ECE 0.594. Temperature scaling preserved 83.66% accuracy while improving NLL
to 0.490 and ECE to 0.028. Full multinomial SDF readout reached 83.60%, while
feature-logistic reached 84.03% and direct transformed-feature logistic
regression reached 83.96%. The geometric representation is therefore
competitive on accuracy in this protocol, but the learned probability layer
is essential for calibration and no superiority over the direct linear control
is demonstrated.

### 6.3 Critical Tier 6 issue: weak temporal representation

**Status (2026-07-25): Experimental comparison complete; competitiveness
remains unresolved.**

**Problem.** The fixed random reservoir has finite state width, a single random
seed, and no learned transition. It reaches 23.35%, while a 3-gram lookup reaches
48.79%. The current state therefore discards much of the local symbolic context.

**Solvability:** Medium to high. It does not require replacing GEODE.

**Improvements.** Preserve the SDF readout while comparing representations:

1. concatenate exact last-$n$ character embeddings with the reservoir state;
2. use multiple reservoir seeds and concatenate or average their states;
3. tune recurrence and spectral radius on forward validation only;
4. add leaky-integrator states at several time scales;
5. add count-sketch or random Fourier features for explicit n-grams;
6. use a frozen pretrained sequence encoder as another upstream representation;
7. compare GEODE and linear readouts on every representation.

The most conservative improvement is a hybrid state containing exact recent
characters plus longer-term reservoir memory. It preserves architecture
generality while preventing short-range information loss.

M6 implemented exact windows, single reservoirs, multi-timescale reservoirs,
multi-seed reservoirs, and exact-plus-reservoir hybrid states under causal
alignment tests. Representation and recurrence choices were tuned on forward
validation only, and every representation was compared with a matched linear
control. On the locked 50k/10k WikiText confirmation, the selected exact
five-character representation raised calibrated GEODE accuracy from the old
23.35% pipeline result to 30.36%. This remained below feature-logistic at
34.64%, a matched-data 5-gram at 44.50%, and the practical 5-gram at 47.61%.
The representation work therefore repaired the missing comparison and improved
the benchmark, but it localized the remaining limitation to geometry/readout
extraction rather than establishing competitive temporal learning. Frozen
pretrained sequence encoders and explicit random n-gram features remain useful
future controls, not prerequisites for interpreting the completed M6 result.

### 6.4 Tier 6 methodology is only partially exercised

**Status (2026-07-25): Resolved by a locked mechanism ablation.**

**Problem.** `verify_pipeline.run_tier6()` sets `n_refinement_iters=0` and
`use_subtractive=False`. The benchmark therefore validates initial additive
geometry and calibration, not temporal SDF refinement or excision.

**Solvability:** High for refinement evaluation; medium for subtraction because
rare classes and temporal drift increase overfitting risk.

**Improvement.** Run one controlled ablation matrix on fixed cached samples:

- initial additive model;
- plus one bounded refinement iteration;
- plus two refinement iterations;
- subtractive model without refinement; and
- each result with a calibrator refit on the same held-out segment.

Accept a mechanism only if it improves forward validation and the single final
test result. Rename remaining `em_*` APIs and stale docstrings to `refinement_*`
because no expectation-maximization step exists.

The locked 50k/10k WikiText ablation uses one fixed sample set and ordered,
disjoint geometry, carve-acceptance, calibration, forward-validation, and test
segments. Refinement consumes geometry samples only; every variant refits its
calibrator on the same held-out segment. R0 additive reached 27.21% validation
and 27.05% observational test accuracy. One bounded refinement iteration
reached 28.51% and 27.90%; two cumulative iterations reached 29.00% and 27.90%.
R2 therefore cleared both gates with +1.79 percentage points on validation and
+0.85 points on test, while test NLL improved from 2.776 to 2.754 and ECE from
0.045 to 0.038. The subtractive fork reached 27.47% validation but fell to
26.72% test, added 106 primitives, and increased fit time from 29.3 to 86.5
seconds, so it was rejected. Test metrics remained observational. Active APIs,
result fields, and documentation now use supervised-refinement terminology;
deprecated `--em_*` CLI aliases remain for one release.

### 6.5 Subtractive excision can overfit

**Status (2026-07-25): Resolved in implementation and tested on Tier 4/Tier 6;
no generalization benefit demonstrated.**

**Problem.** Subtractive ellipsoids are selected from training false captures.
There is no separate acceptance set for deciding whether a carve generalizes.
Inflation can remove valid class volume around noisy errors.

**Solvability:** Medium.

**Improvement.** Give geometry construction and carve acceptance separate
training subsets. Accept a negative primitive only if it improves a held-out
objective such as balanced accuracy or class-conditional log loss. Add an MDL
penalty per primitive and minimum validation gain. Record how many points are
recovered and newly damaged by each carve.

The constructor now accepts explicit positive and negative acceptance sets,
scores each proposed carve by held-out balanced-accuracy gain minus an MDL
parameter penalty, enforces a minimum penalized gain, and records recovered
false positives and damaged true positives. When explicit acceptance data are
not supplied, it uses a disjoint exclusion holdout when sample size permits.
Regression tests cover both helpful and damaging carves. The five-seed Tier 4
ablation found no aggregate benefit, and the locked Tier 6 fork improved its
carve-acceptance segment but lost 0.33 percentage points on untouched test while
adding 106 primitives. Subtraction therefore remains opt-in and disabled in
`verify_pipeline`.

### 6.6 High-dimensional RANSAC remains statistically fragile

**Status (2026-07-25): Resolved as a conditional fitter study; current default
retained.**

**Problem.** Minimal seed size grows as $O(d^2)$. Tier 5 reports low sample
adequacy at $d=19$, and its run takes 1,193.8 seconds. The GPU path switches to
batched covariance hypotheses and hard candidate caps, so CPU and GPU training
do not sample the same hypothesis family.

**Solvability:** Medium. The exponential minimal-sample problem cannot be
optimized away, but the fitter can change.

**Improvements.** Compare:

- robust covariance estimators such as Minimum Covariance Determinant;
- EMS-style robust ellipsoid fitting;
- diagonal or low-rank-plus-diagonal precision models;
- class-adaptive dimension selected before LDA;
- incremental split/merge mixture fitting; and
- GMM initialization followed by geometric conversion.

Treat CPU-SVD and GPU-covariance constructors as separate ablation variants.
Match candidate counts and wall-clock budgets when comparing them.

M5 compared quadric SVD, full and diagonal covariance, Ledoit-Wolf shrinkage,
Minimum Covariance Determinant, low-rank-plus-diagonal covariance, and
one-component GMM covariance over dimensions 3-19, then separated clean,
outlier, mislabeled-shell, overlap, and low-support conditions. Candidate-count
and 0.1-second wall-clock controls confirmed a conditional rather than universal
frontier: MCD was uniquely robust to unconstrained outliers, shrinkage led
several clean/low-support cells, and full/GMM covariance led mislabeled-shell
noise. A five-seed Tier 4 screen advanced shrinkage, but the locked Tier 5
confirmation selected the current fitter by 67.67% versus 66.11% validation
accuracy. Test accuracy was tied, while shrinkage used 186 versus 42 primitives,
took 1.91 versus 0.59 seconds, and left three classes empty. The fitter
interface retains robust alternatives for condition-specific use; no universal
replacement or CPU/GPU hypothesis equivalence is claimed.

### 6.7 Potential normalized-softmin pruning inconsistency

**Status: resolved and covered by CPU/GPU regression tests.**

**Problem.** Exact normalized softmin divides by the total component count $M$.
The CPU expert-pruning path evaluates only $M_a$ active experts and divides by
$M_a$. Dropping negligible numerator terms while changing the denominator does
not approximate the original normalized mixture; it changes the score by up to
$\log(M/M_a)/\alpha$. The current stated exponential error bound covers omitted
numerator mass but not denominator replacement. GPU class fusion evaluates the
full count, so models with at least four well-separated experts may diverge even
though existing small-model parity tests pass.

**Solvability:** High.

**Improvement.** Decide the intended semantics:

- For a fixed uniform mixture, retain denominator $M$ after pruning.
- For local renormalization over active experts, document a different model and
  implement the same rule on GPU.

Add a CPU/GPU test with at least four widely separated experts where pruning is
guaranteed to activate.

The CPU pruning path now omits negligible numerator terms while retaining the
total expert count $M$ in the normalized denominator. The OpenCL class-fusion
kernel uses the same total-count rule. Tests compare the pruned CPU result with
the exact four-expert normalized softmin and compare CPU with OpenCL for four
widely separated experts where global pruning activates.

### 6.8 The field is not a true metric SDF

**Status (2026-07-25): Terminology corrected and primitive approximation
quantified; fused/CSG guarantees remain intentionally unsupported.**

**Problem.** $f/||\nabla f||$ is a local first-order distance correction, not
the exact Euclidean distance to a general ellipsoid, union, or CSG boundary.
Using it as a guaranteed safe sphere-tracing step is stronger than the
mathematics supports. Softmin also combines dimensionless normalized radii from
different covariance scales.

**Solvability:** Medium for wording and tests; hard for exact high-dimensional
distance.

**Improvements.** Call the primitive a signed normalized radial field unless
metric distance is explicitly computed. For geometry tasks, compare the
first-order estimate with a numerical closest-point solver and measure missed
surface crossings. For classification, test either:

- locally metric-corrected fusion;
- a proper Gaussian energy including log determinant and priors; or
- learned per-component temperature and mixture weights.

Production documentation now calls the primitive a signed normalized radial
field and describes `compute_metric_sdf` as a first-order correction that is
not a guaranteed conservative sphere-tracing step. A deterministic multistart
SLSQP closest-point reference validates 2D/3D ellipsoids in research tests. On
64 directions, five inside/outside radial factors, and eccentricities 1, 2, 4,
and 8, spheres were exact to numerical tolerance. Worst absolute error rose to
3.13 units for the 2D eccentricity-8 condition and 2.53 units in 3D. No sampled
outside step toward the numerical closest point crossed the primitive boundary.
This supports local scale correction for isolated ellipsoids, not an exact
metric-SDF or safe-step claim for normalized softmin unions or hard CSG.

### 6.9 Hard CSG value and smooth surrogate gradient differ

**Status: resolved and covered by finite-difference tests.**

**Problem.** `Expert.compute_sdf` uses hard `max(f_add,-f_sub)`, while
`compute_gradient` blends branches using smooth-max weights. Near a CSG switch,
the returned gradient is not the derivative or a strict subgradient of the
reported hard-max field.

**Solvability:** High.

**Improvement.** Either return the active hard-max branch gradient with a
defined tie policy, or use the same smooth-max function for both value and
gradient. Add finite-difference tests around carve boundaries.

`Expert.compute_gradient` now follows the active branch of
$\max(f_{add},-f_{sub})$ and returns the average branch gradient only at an
exact tie. Finite-difference tests cover points on both sides of a subtractive
carve boundary.

### 6.10 Reproducibility is incomplete

**Status: resolved for constructor sampling; multi-seed reporting remains an
experimental requirement.**

**Problem.** Dataset splitting and transforms use seeded generators, but the
constructor uses global `np.random` calls. `_knn_seed` accepts an RNG parameter
but ignores it. Repeated runs can therefore vary despite a top-level seed. The
change from an earlier staged 23.91% Tier 6 result to 23.35% in the latest full
run illustrates why complete seed control matters, even though other code
changes may also contribute.

**Solvability:** High.

**Improvement.** Give `GreedyConstructor` an owned
`np.random.Generator`, thread the seed through every constructor, candidate
batch, exclusion subsample, and kNN anchor, and save configuration plus commit
hash beside every result. Run multi-seed summaries rather than one seed.

`GreedyConstructor` now owns a seeded `np.random.Generator`; random candidate
batches, exclusion subsamples, kNN anchors, subtractive seeds, and fallback
validation splits all use it. Same-seed construction has regression coverage,
and experiment manifests record configuration and repository state. Scientific
results must still retain multi-seed summaries rather than relying on one run.

### 6.11 Benchmark interpretation and task coverage

**Status (2026-07-25): Reporting ambiguity resolved; task breadth remains a
declared limitation.**

**Problem.** Several tier metrics can look stronger than their experimental
scope warrants:

- Tier 1 warns that legacy datasets are in use.
- Tier 2 uses only 32 ModelNet shapes in verification.
- Tier 3 fits only digit zero and is not MNIST classification.
- The reported geometry $R^2$ is a project-specific normalization, not standard
  predictive $R^2$.
- Tier 4 uses only 7,500 CIFAR-10 samples and a pretrained backbone.
- Tier 5 uses 15,000 samples, has low RANSAC adequacy, and dominates runtime.
- Tier 6 baseline sample budgets differ: n-gram counts use up to five million
  training characters, while GEODE fits 50,000 sampled pairs.

**Solvability:** High.

**Improvement.** Add a benchmark manifest listing sample count, feature source,
split hash, model budget, and metric definition. Compare methods on matched
training characters as well as their best practical data regime. Rename custom
geometry $R^2$ to avoid confusion or report established surface metrics such as
Chamfer distance and normal consistency.

M0 manifests now record configurations, dataset/feature identities, split
hashes, sample counts, budgets, model structure, and runtime. The geometry
metric is explicitly named `geometry_normalized_residual_score`, and surface
experiments also support symmetric Chamfer distance. M2-M4 use common result
records and matched splits. M6 reports both matched-data and best-practical
n-gram controls, so the data-budget distinction is visible rather than hidden.
These changes repair interpretation and provenance; they do not enlarge the
32-shape Tier 2 verification set, turn Tier 3 into classification, remove the
pretrained Tier 4 backbone, or make Tier 6 competitive.

### 6.12 Calibration and OOD claims are untested

**Status (2026-07-25): Partially resolved on controlled shift, CIFAR-100, and
SVHN; broader natural OOD families remain.**

**Problem.** Perplexity measures in-distribution probability quality only.
There is no expected calibration error, reliability diagram, Brier score,
selective-risk curve, or OOD benchmark. A geometric distance is not
automatically a reliable uncertainty estimate after supervised calibration.

**Solvability:** High experimentally.

**Improvement.** On CIFAR-10, evaluate CIFAR-100, SVHN, LSUN, and textures as
OOD sets. Compare minimum SDF, calibrated maximum probability, energy score,
Mahalanobis baseline, and k-NN distance using AUROC, AUPR, and FPR95. Add
temperature scaling and conformal prediction sets on untouched calibration
data.

M2 added NLL, Brier score, ECE, temperature/diagonal/multinomial readouts, and
paired uncertainty reporting. M7 added controlled radial shifts, selective-risk
and accuracy-rejection curves, conformal sets, and one frozen CIFAR-10 model
evaluated against balanced CIFAR-100 and SVHN families. Maximum-probability
uncertainty led the matched scores with AUROC 0.786/0.831, but FPR95 remained
0.579/0.431; geometry-native SDF and energy scores transferred poorly. Nominal
90% conformal coverage reached 88.61%. The supported claim is therefore
relative ranking on these families, not deployment-grade rejection. LSUN,
textures, semantic/covariate mixtures, and repeated model seeds remain useful
breadth extensions.

### 6.13 Model complexity and resource use need explicit control

**Status (2026-07-25): Resolved for structural size, runtime, throughput, and
scaling; process peak memory remains unmeasured.**

**Problem.** Accuracy alone does not expose how many experts and ellipsoids are
required, how much memory they consume, or how construction scales. Tier 5's
approximately 20-minute runtime is already a practical warning.

**Solvability:** High.

**Improvement.** Report ellipsoids per class, subtractive fraction, parameters,
peak memory, candidate evaluations, fit time, and throughput. Add a primitive
cost to candidate acceptance and produce accuracy-versus-complexity curves.

Shared result records now report experts, additive/subtractive primitives,
fitted parameters, approximate serialized bytes, candidate evaluations, fit
time, inference time, and throughput. Held-out carving supports an MDL-style
parameter penalty. M5 reports accuracy/runtime/primitive tradeoffs for robust
fitters, and M9 independently sweeps classes, dimensions, and primitives while
recording operation counts, serialized growth, and edit/inference latency.
This resolves the main hidden-complexity issue, but approximate model bytes are
not a substitute for measured process peak memory or hardware energy.

---

## 7. Prioritized Research Plan

### Phase A: establish causal evidence

1. Make constructor randomness fully deterministic.
2. Add raw, diagonal-calibrated, temperature-scaled, and full-readout metrics.
3. Add matched classical baselines on cached Tier 4 features.
4. Run additive/excision/active-repair ablations over five seeds.
5. Publish paired confidence intervals and complexity metrics.

**Decision gate:** continue emphasizing CSG excision only if it improves held-out
performance at matched complexity and calibration.

### Phase B: correct semantic and numerical risks

1. Resolve the pruned-softmin denominator semantics.
2. Align hard CSG values and gradients.
3. Test first-order metric correction against numerical ellipsoid distance.
4. Add parity coverage for four-plus separated experts and active pruning.
5. Rename EM-related APIs and logs to supervised refinement.

### Phase C: improve temporal learning without replacing GEODE

1. Add exact recent-character features to the fixed reservoir state.
2. Match GEODE, linear, and n-gram training budgets.
3. Tune multiple reservoir time scales using forward validation.
4. Run bounded refinement and subtraction ablations once each.
5. Compare against a small GRU and a frozen language-model embedding readout.

**Decision gate:** the temporal extension should at least approach the matched
$n$-gram baseline before being presented as a competitive sequence learner.

### Phase D: test the geometry-specific value proposition

1. Add OOD and selective-prediction evaluation.
2. Evaluate robustness to mislabeled data and feature outliers, where RANSAC
   should have a principled advantage.
3. Measure interpretability through primitive stability across seeds and
   human-inspectable carve regions.
4. Evaluate online insertion, deletion, and local editing of primitives.

These experiments align with GEODE's likely strengths better than pursuing raw
accuracy alone.

---

## 8. Recommended Positioning

A defensible current positioning statement is:

> GEODE is a constructive geometric readout that generalizes
> class-conditional Mahalanobis and mixture discriminant models with robust
> multi-ellipsoid fitting and optional CSG excision of inter-class false
> captures. It supports analytic field queries, explicit geometry, calibrated
> score readout, and multiple upstream representations.

The strongest paper should center on one of two stories:

1. **Discriminative CSG:** explicit subtractive primitives improve calibrated
   classification at matched model complexity; or
2. **Robust interpretable geometry:** RANSAC-built ellipsoidal mixtures provide
   editable, uncertainty-aware readouts that remain stable under outliers.

The current evidence is not yet sufficient for either story, but the codebase
is now capable of running the required experiments.

Tier 6 should be framed as an architecture-flexibility demonstration until its
representation closes the gap to simple symbolic baselines. Its current result
does show that GEODE can consume causal state features without changing the core
ellipsoid kernel, which is useful but narrower than competitive language
modeling.

---

## 9. Conclusion

GEODE is technically coherent as an explicit geometric learning framework. Its
core ellipsoid field, normalized fusion, robust construction, CSG composition,
calibrated readout, and analytic refinement form a consistent system. The
latest pipeline demonstrates that the implementation runs across geometry,
images, and sequential text with causal evaluation and GPU support.

The most important unresolved question is scientific attribution: does the
distinctive CSG/RANSAC geometry improve generalization beyond established
mixture models, the pretrained feature extractor, supervised dimensionality
reduction, and the logistic score readout? That question is directly solvable
with matched baselines and ablations.

The approach should therefore be improved by tightening semantics and
experimental controls rather than replacing the SDF ellipsoid kernel. Fixing
pruned-softmin semantics, aligning CSG gradients, controlling randomness,
validating carve acceptance, and strengthening temporal features are all
tractable. High-dimensional minimal-sample RANSAC remains the hardest inherent
constraint and may require robust covariance or EMS-style fitting rather than
larger candidate budgets.

In short: the project contains a plausible novel mechanism and a strong
engineering foundation, but its next milestone should be causal evidence for
that mechanism.

---

## References

1. Fischler, M. A., and Bolles, R. C. (1981). Random Sample Consensus.
   https://doi.org/10.1145/358669.358692
2. Jacobs, R. A., et al. (1991). Adaptive Mixtures of Local Experts.
   https://doi.org/10.1162/neco.1991.3.1.79
3. Hastie, T., and Tibshirani, R. (1996). Discriminant Analysis by Gaussian
   Mixtures. https://doi.org/10.1111/j.2517-6161.1996.tb02085.x
4. Hart, J. C. (1996). Sphere Tracing.
   https://doi.org/10.1007/BF02439180
5. Tulsiani, S., et al. (2017). Learning Shape Abstractions by Assembling
   Volumetric Primitives. https://arxiv.org/abs/1612.00404
6. Lee, K., et al. (2018). A Simple Unified Framework for Detecting
   Out-of-Distribution Samples and Adversarial Attacks.
   https://arxiv.org/abs/1807.03888
7. Sharma, G., et al. (2018). CSGNet. https://arxiv.org/abs/1712.08290
8. Park, J. J., et al. (2019). DeepSDF. https://arxiv.org/abs/1901.05103
9. Paschalidou, D., et al. (2019). Superquadrics Revisited.
   https://arxiv.org/abs/1904.09970
10. Barath, D., and Matas, J. (2019). Progressive-X.
    https://arxiv.org/abs/1906.02290
11. Mescheder, L., et al. (2019). Occupancy Networks.
    https://arxiv.org/abs/1812.02822
12. Chen, Z., et al. (2020). BSP-Net. https://arxiv.org/abs/1911.06971
13. Deng, B., et al. (2020). CvxNet. https://arxiv.org/abs/1909.05736
14. Gropp, A., et al. (2020). Implicit Geometric Regularization.
    https://arxiv.org/abs/2002.10099
15. Kania, K., et al. (2020). UCSG-Net. https://arxiv.org/abs/2006.09102
16. Zhao, Y., et al. (2021). EMS: A Robust Framework for Ellipsoid Modeling.
    https://arxiv.org/abs/2110.13337
17. Yu, F., et al. (2022). CAPRI-Net. https://arxiv.org/abs/2104.05652
18. Liu, Y., et al. (2023). Marching-Primitives.
    https://arxiv.org/abs/2303.13190
19. Spinosa, E. J., de Carvalho, A. C. P. L. F., and Gama, J. (2008).
    Cluster-Based Novel Concept Detection in Data Streams Applied to Intrusion
    Detection in Computer Networks. https://doi.org/10.1145/1363686.1363912
20. Masud, M., et al. (2011). Classification and Novel Class Detection in
    Concept-Drifting Data Streams under Time Constraints.
    https://doi.org/10.1109/TKDE.2010.61
21. de Faria, E. R., de Carvalho, A. C. P. L. F., and Gama, J. (2016).
    MINAS: Multiclass Learning Algorithm for Novelty Detection in Data Streams.
    https://doi.org/10.1007/s10618-015-0433-y
22. Cao, F., Ester, M., Qian, W., and Zhou, A. (2006). Density-Based
    Clustering over an Evolving Data Stream with Noise.
    https://doi.org/10.1137/1.9781611972764.29
23. Campello, R. J. G. B., Moulavi, D., Zimek, A., and Sander, J. (2015).
    Hierarchical Density Estimates for Data Clustering, Visualization, and
    Outlier Detection. https://doi.org/10.1145/2733381
24. Sarfraz, M. S., Sharma, V., and Stiefelhagen, R. (2019). Efficient
    Parameter-Free Clustering Using First Neighbor Relations.
    https://arxiv.org/abs/1902.11266
25. Han, K., Vedaldi, A., and Zisserman, A. (2019). Learning to Discover Novel
    Visual Categories via Deep Transfer Clustering.
    https://arxiv.org/abs/1908.09884
26. Han, K., Rebuffi, S.-A., Ehrhardt, S., Vedaldi, A., and Zisserman, A.
    (2020). Automatically Discovering and Learning New Visual Categories with
    Ranking Statistics. https://arxiv.org/abs/2002.05714
27. Fini, E., et al. (2021). A Unified Objective for Novel Class Discovery.
    https://arxiv.org/abs/2108.08536
28. Vaze, S., Han, K., Vedaldi, A., and Zisserman, A. (2022). Generalized
    Category Discovery. https://arxiv.org/abs/2201.02609
29. Cao, K., Brbic, M., and Leskovec, J. (2022). Open-World Semi-Supervised
    Learning. https://arxiv.org/abs/2102.03526
30. Caron, M., Bojanowski, P., Joulin, A., and Douze, M. (2018). Deep
    Clustering for Unsupervised Learning of Visual Features.
    https://arxiv.org/abs/1807.05520
31. Van Gansbeke, W., Vandenhende, S., Georgoulis, S., Proesmans, M., and Van
    Gool, L. (2020). SCAN: Learning to Classify Images without Labels.
    https://arxiv.org/abs/2005.12320
32. Van Craenendonck, T., Dumancic, S., Van Wolputte, E., and Blockeel, H.
    (2018). COBRAS: Interactive Clustering with Pairwise Queries.
    https://doi.org/10.1007/978-3-030-01768-2_29
