# GEODE Research Analysis v5

## Geometric Expert Learning: Prior Art, Current Contributions, Accuracy Limits, and a Research Agenda

**Literature search refreshed:** 26 July 2026  
**Repository evidence cutoff:** 26 July 2026  
**Repository state:** M0-M15.2b and E0-E6/E8-E11 complete; E7 local-small complete and physical multi-host qualification open  
**Project name:** GEODE, currently expanded as **Geometric Expert Orchestration for Discovery and Evaluation**

---

## Abstract

GEODE learns explicit class fields from geometric primitives, currently oriented
full, diagonal, and spherical covariance regions. It combines additive fields,
optionally subtracts exclusion regions, calibrates the resulting class-score
vector, and wraps the model in deterministic experiment, review, adaptation,
bundle, recovery, and deployment contracts [R1-R8]. This report reassesses that
work against refreshed primary literature and the repository's completed
experiments.

Most individual ideas in GEODE have substantial prior art. Quadratic and mixture
discriminant models, radial-basis and prototype classifiers, robust consensus
fitting, soft minimums, constructive solid geometry, signed and implicit fields,
post-hoc calibration, OOD scoring, density clustering, generalized category
discovery, continual-learning replay, sparse expert routing, latent predictive
self-supervision, two-stage frozen-representation support modeling, and
intrinsically interpretable prototype models were established before this project
[L1-L28,L54-L67]. They must be described as foundations or borrowed mechanisms,
not GEODE inventions.

The defensible repository contribution is narrower: an audited integration of
explicit editable class fields with deterministic construction, matched controls,
immutable evidence, exact recovery, review-first unknown handling, transactional
adaptation, and fail-closed promotion. The completed experiments also contribute
useful negative evidence. Subtractive geometry did not improve aggregate CIFAR-10
accuracy; the geometric text head trailed same-feature linear and n-gram controls;
raw field distance was a weak OOD score; candidate routing failed promotion
against exhaustive inference; and real-model mutation remained gated [R3-R10].

A differentiable boundary is not sufficient for state-of-the-art learning.
Differentiability only makes a chosen finite parameterization locally optimizable.
It does not guarantee that the input representation contains the needed
information, that the parameterization represents the target efficiently, that
the training objective matches the task, that optimization finds the best
parameters, or that finite data identify them. GEODE's current pipeline is only
partly end-to-end differentiable: representation learning, discrete RANSAC
proposal and acceptance, component-count decisions, hard subtraction, and final
calibration are separate stages [R1,R2]. The observed accuracy gap is therefore
better explained by representation, objective, optimization, and statistical
efficiency than by geometry itself.

This report also fixes a design boundary that earlier versions left open. Joint
encoder-head gradient training is **not** a GEODE goal. Every editability,
provenance, and replay contract in this repository is defined over a fixed
embedding: a component means "this region of this space covers these examples",
and an edit is local only because the map from input to feature space does not
move. An encoder that keeps training after components are fitted invalidates
cached components, calibration objects, support profiles, changed-region
measurements, and rollback targets, and eventually makes the geometry
decorative rather than explanatory. The adopted stance is therefore **frozen
trunk, trained interface**: buy a commodity self-supervised backbone, train one
small linear adapter once and freeze it as a hashed artifact, train the
geometric head discriminatively inside that fixed space, and handle any later
representation change as a versioned migration event rather than as gradient
flow (Sections 5.4, 8, and 9). This narrows the research claim to head parity
on a state-of-the-art representation plus a measured editability advantage,
which is both testable and defensible.

The refreshed review identifies a previously underrepresented lineage around
joint-embedding predictive architectures (JEPA), contrastive predictive coding,
self-distillation, and masked latent prediction [L54-L62]. These methods do not
replace GEODE's explicit class-field head: they learn the upstream representation
in which that head operates. Their frozen linear, k-nearest-neighbor, attentive
probe, transfer, and one-class protocols are nevertheless directly reusable.
The revised program therefore adds I-JEPA-class image representations, frozen
video representations, small public transfer datasets, and two-stage support
tests without attempting to reproduce foundation-model pretraining.

---

## 1. Scope and Evidence Policy

This report distinguishes:

1. **Established external ideas**, cited to primary papers or archival surveys as
   `[L#]`;
2. **Repository mechanisms and results**, cited to code, protocols, or frozen
   artifacts as `[R#]`;
3. **Interpretations and proposals**, stated as hypotheses rather than completed
   results.

The refreshed search focused on concepts that directly bear on GEODE:
quadratic/mixture classification, prototype and metric learning, robust
multi-model fitting, implicit fields, universal approximation, geometric deep
learning, sequence and tabular inductive biases, calibration and OOD detection,
category discovery, continual learning, and sparse routing. It is not a formal
systematic review, patent search, or proof of novelty. Absence of an identical
system in this search must not be presented as evidence that the integration is
novel.

“State of the art” is task- and protocol-specific. This report does not compare
numbers across unrelated datasets, budgets, or modalities as if they formed one
ranking.

---

## 2. Background Concepts

### 2.1 Classification as fields and decision regions

A multiclass predictor can assign one scalar score $s_k(x)$ to each class and
predict

$$
\hat y(x)=\arg\min_k s_k(x)
$$

for distance-like scores, or an argmax for logits or posterior probabilities.
The decision boundary between classes $i$ and $j$ is the level set

$$
\mathcal{B}_{ij}=\{x:s_i(x)=s_j(x)\}.
$$

This view includes linear and quadratic discriminant analysis, nearest-prototype
methods, RBF networks, kernel machines, neural classifiers, energy models, and
implicit neural fields. Having a boundary or a differentiable score therefore
does not by itself distinguish GEODE from broad prior art [L1-L8]. The scientific
question is which function family, representation, objective, optimizer, data,
and regularization produce a useful boundary.

### 2.2 GEODE's current primitive field

For transformed input $z=h(x)\in\mathbb{R}^d$, a current oriented primitive has
center $c$, orientation $Q$, and radii $a>0$:

$$
q(z)=\sum_{j=1}^{d}\frac{((z-c)Q)_j^2}{a_j^2},
\qquad
\phi(z)=\sqrt{q(z)}-1.
$$

The sign identifies inside, boundary, and outside. For anisotropic primitives,
$\phi$ is a normalized Mahalanobis radial field, not an exact Euclidean signed
distance. GEODE implements the local correction
$\phi/\lVert\nabla\phi\rVert$, but correctly treats it as a first-order estimate,
not a conservative metric bound [R1].

The quadratic form connects GEODE directly to Mahalanobis distance, Gaussian
class models, quadratic discriminant analysis, and mixture discriminant analysis
[L1-L3]. The difference is not the existence of ellipsoidal contours; it is how
GEODE constructs, composes, edits, and governs those contours.

### 2.3 Composition and subtraction

GEODE combines $M$ additive fields by normalized log-mean-exp:

$$
\Phi(z)=-\frac{1}{\alpha}\log\left(
\frac{1}{M}\sum_{m=1}^{M}e^{-\alpha\phi_m(z)}
\right).
$$

This is a smooth approximation to a minimum. The $1/M$ term makes duplicate
coincident components invariant to component count. Log-sum-exp smoothing and
soft minima are standard mathematical tools; using the normalized form to repair
this repository's duplicate-component offset is an implementation choice, not a
new mathematical operation [R1].

Negative-polarity primitives implement set difference with a hard maximum.
Constructive solid geometry and signed-field set operations long predate GEODE,
and learned CSG/primitive systems already combine explicit parts or infer CSG
programs [L9-L13]. GEODE uses these ideas in a class-feature space rather than
claiming a new CSG algebra.

### 2.4 Continuous implicit fields

DeepSDF, Occupancy Networks, NeRF, and SIREN demonstrate that continuous,
differentiable fields can represent complex shapes, scenes, signals, and
derivatives [L10,L14-L17]. They are important counterexamples to the claim that
“geometry is inherently inaccurate.” Their performance comes from high-capacity
learned parameterizations, coordinate encodings, task-specific rendering or
reconstruction losses, large training sets, and end-to-end optimization.

A field is a representation format. Accuracy depends on the family used to
parameterize the field and how it is learned.

### 2.5 Universal approximation is not a performance guarantee

Universal-approximation results show that sufficiently large neural or
radial-basis function families can approximate broad classes of continuous
functions on compact sets [L4-L7]. Similar intuition applies to unions of small
balls or local primitives: with enough components, many bounded regions can be
approximated arbitrarily closely.

The quantifiers matter. Such results do not say that:

- a small finite model is sufficient;
- the required number of components grows slowly with dimension;
- a greedy or gradient optimizer will find the approximating parameters;
- finite noisy samples identify the correct boundary;
- the learned function generalizes out of sample;
- probabilities or OOD behavior are calibrated;
- the representation preserves task-relevant information.

Expressibility answers “does a parameter setting exist?” Learning answers “can
our data, objective, optimizer, and compute reliably find a good setting?” Those
are different questions.

---

## 3. Prior Art and Attribution Map

| GEODE mechanism                   | Established lineage                                                                      | What must be credited                                                                  | Repository-specific use or difference                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Quadratic covariance regions      | Gaussian/QDA and mixture discriminant analysis [L1-L3]                                   | Quadratic contours, covariance metrics, component mixtures, priors and density scoring | Support-oriented radial fields with explicit editable components; currently omits parts of a full Gaussian discriminant score in raw mode |
| Prototype/distance classification | LVQ, RBF networks, metric learning, prototypical networks [L4-L8]                        | Classification by learned distances to representatives in a feature space              | Multi-primitive class fields plus lifecycle contracts; metric space is mostly fitted before geometry rather than jointly learned          |
| Random consensus construction     | RANSAC and Progressive-X [L18,L19]                                                       | Random minimal subsets, consensus scoring, residual-set multi-model fitting            | Class-conditioned candidate scoring, deterministic fallbacks, frozen acceptance gates, and recovery evidence                              |
| Soft-minimum field fusion         | Smooth min/log-sum-exp methods                                                           | Differentiable approximation of minimum/union                                          | Log-mean normalization prevents duplicate-count score drift in this implementation                                                        |
| Signed fields and CSG subtraction | Classical SDF/CSG; CSGNet, DeepSDF, Occupancy Networks, primitive decomposition [L9-L15] | Zero level sets, union/difference semantics, implicit shape representation             | Applied to class support in transformed feature space; subtraction is validation-gated and empirically disabled by default                |
| Gradient refinement               | Standard discriminative gradient optimization                                            | Backpropagating task loss through differentiable parameters                            | Analytic updates for additive covariance primitives with exact checkpoint/restart; not end-to-end through representation or hard CSG      |
| Score calibration                 | Platt/logistic calibration and temperature scaling [L20,L21]                             | Mapping scores to probabilities on held-out calibration data                           | Multiple frozen class-field readouts and explicit separation of geometry/calibration/test budgets                                         |
| OOD scoring                       | MSP, Mahalanobis, energy, kNN, open-set recognition [L22-L26]                            | Confidence, feature distance, energy, tail/open-space evaluation                       | Matched audit of field-derived and standard scores; no new OOD detector established                                                       |
| Unknown grouping                  | DBSCAN/HDBSCAN/FINCH, stream novelty detection, GCD/SimGCD [L27-L33]                     | Density clustering, unknown buffering, semantic partition metrics, pseudo-label bias   | Stable non-semantic review IDs and a policy boundary that forbids automatic semantic publication                                          |
| Continual adaptation              | Replay and no-forgetting methods such as iCaRL, GEM, and DER [L34-L36]                   | Rehearsal, preservation constraints, incremental classes                               | Transactional structural edits, immutable snapshots, graph migration, delayed confirmation, and rollback                                  |
| Sparse routing                    | Mixture-of-experts, Switch, Expert Choice, hash routing [L37-L40]                        | Conditional computation and learned or fixed routing                                   | Post-hoc geometric candidate omission under exhaustive-oracle checks; all tested candidates remain blocked                                |
| Reproducible lifecycle            | Established MLOps, artifact hashing, checkpointing, canary and rollback practice         | Versioned artifacts, recovery, staged deployment                                       | One fail-closed evidence chain joining experiments, bundles, reviews, recovery, and publication reproduction [R5-R10]                     |
| Latent predictive representation  | CPC, JEPA/I-JEPA/V-JEPA, BYOL, DINO, MAE, and data2vec [L54-L62]                        | Learning transferable representations by predicting or aligning latent targets         | GEODE consumes a released encoder as a frozen artifact; it does not claim the representation objective or reproduce foundation pretraining |
| Two-stage support modeling        | Frozen self-supervised representations followed by one-class statistical heads [L63]     | Separating representation learning from downstream support or anomaly scoring          | Multiclass editable component fields plus calibration, review, replay, and rollback contracts                                              |
| Intrinsically interpretable models | ProtoPNet, ProtoTree, and concept bottleneck models [L64,L65,L67]                       | Prototype evidence, interpretable routing, and concept intervention                     | Feature-space geometric components and transactional structural edits; semantic concepts are not guaranteed                               |
| Low-rank local subspaces          | Mixtures of PPCA/factor analyzers, CLAFIC/local-subspace classification, and k-plane clustering [L68-L72] | Center-plus-low-rank-basis models, isotropic residual variation, and local-subspace decision structure | M29/M31 use axis-scaled affine components with a discriminative radial boundary, not the probabilistic likelihood or EM fit of PPCA/MFA |
| Teacher-to-student fitting        | Model compression, knowledge distillation, TREPAN, and soft decision trees [L73-L76]    | Soft teacher targets, temperature-controlled transfer, and interpretable student extraction | M28 distills an RBF control into explicit geometric components and also matches teacher margins; it does not invent distillation |
| Directional class components      | von Mises-Fisher mixtures, directional statistics, cosine classifiers, weight imprinting, and hyperspherical prototypes [L77-L81] | Unit-sphere data models, angular/cosine scoring, mean directions, and hyperspherical class representatives | M30 uses an explicit angular support radius and audited component lifecycle rather than a vMF likelihood or a learned neural cosine head |
| Tangent-space local models        | Principal geodesic analysis and intrinsic Riemannian statistics [L82,L83]                | Log-map linearization and PCA-like variation analysis in a tangent space                | A1-T adds an explicit angular boundary, rank/support contract, radial score, serialization, edits, and rollback |
| Greedy component selection        | Orthogonal least-squares RBF selection, kernel matching pursuit, and matching pursuit [L84-L86] | Forward atom/center selection by incremental error reduction                            | M28/M31 select explicit class components by direct-label loss reduction under fixed budgets and lifecycle gates |
| Weighted component readout        | Two-stage RBF-network training [L87]                                                     | Separating basis/center construction from supervised output-weight fitting              | A1-W freezes supervised-selected components, then fits nonnegative per-class simplex weights and one global temperature; it is not the unsupervised-center procedure of the classic RBF model |
| RBF support-vector control        | Kernel support-vector machines [L88,L89]                                                 | Maximum-margin kernel classification and support-vector boundary representation         | The RBF SVM is a same-feature control and M28 teacher, not a GEODE mechanism or contribution |
| Conformal uncertainty             | Conformal prediction and inductive confidence machines [L90,L91]                        | Finite-sample coverage under the stated exchangeability assumptions                     | GEODE uses held-out split-conformal sets as an evaluation/control layer; it does not claim the conformal method |
| Reservoir temporal features       | Echo-state networks and liquid-state machines [L92,L93]                                 | Fixed recurrent reservoirs with trained readouts                                       | Tier 6 uses a small deterministic reservoir as one representation control, not a new reservoir architecture |
| Covariance regularization         | Linear shrinkage and minimum-covariance-determinant estimation [L94-L96]                 | Well-conditioned covariance estimation and high-breakdown robust scatter               | M5 compares these established fitters inside the frozen GEODE protocol |
| Local intrinsic dimension         | Maximum-likelihood intrinsic-dimension estimation [L97]                                 | Estimation from log ratios of neighbor distances                                       | M19 reports the median of a fixed-neighborhood Levina-Bickel-style statistic; its code uses a `k` rather than `k-1` numerator convention and should not be described as the exact published estimator |
| Numerical/readout utilities       | L-BFGS/L-BFGS-B and distance-weighted k-nearest neighbors [L98-L100]                     | Limited-memory bounded quasi-Newton optimization and inverse-distance neighbor voting   | A1-W uses deterministic bounded L-BFGS-B; weighted kNN is a declared control |

### 3.1 Ideas that are not GEODE inventions

The project should not claim invention of:

- ellipsoidal or covariance-based classification;
- mixture models or multi-prototype classes;
- RANSAC or greedy residual fitting;
- signed distance or occupancy fields;
- soft-minimum fusion;
- CSG subtraction;
- gradient-based primitive refinement;
- logistic or temperature calibration;
- OOD confidence, Mahalanobis, energy, or kNN scores;
- density clustering or unknown-observation buffering;
- generalized category discovery;
- replay-based continual learning;
- mixture-of-experts routing;
- latent predictive or joint-embedding self-supervised learning;
- frozen linear, kNN, or attentive probing;
- two-stage frozen-representation one-class classification;
- prototype-based or concept-based intrinsic interpretability;
- low-rank local-subspace, PPCA, or factor-analyzer component models;
- teacher distillation or model compression;
- directional statistics, cosine classifiers, or hyperspherical prototypes;
- tangent-space PCA or principal geodesic analysis;
- greedy RBF/atom selection or matching pursuit;
- two-stage RBF basis construction and output-weight fitting;
- support-vector machines, conformal prediction, reservoir computing,
  covariance shrinkage, robust covariance, intrinsic-dimension MLE, L-BFGS, or
  distance-weighted kNN;
- checkpoint, canary, rollback, or content-addressed artifact practice.

### 3.2 Defensible contributions of this repository

The current evidence supports describing the work as:

1. **A concrete explicit-field implementation.** It joins oriented covariance
   primitives, two-level normalized fusion, optional hard subtraction, analytic
   derivatives, CPU/OpenCL parity, and deterministic fitting under one model
   contract [R1,R2].
2. **A conservative experimental methodology.** Frozen splits, matched controls,
   negative results, exact replay, and explicit claim boundaries prevent
   mechanism tests from being reported as benchmark superiority [R3,R4].
3. **A review-first lifecycle integration.** Unknown groups remain non-semantic
   review objects; confirmation permits evaluation but does not bypass replay,
   calibration, graph, rollback, or promotion gates [R5,R6].
4. **A recoverable and auditable model lifecycle.** Stage checkpoints, immutable
   bundles, shadow routing, adaptation transactions, replicated canaries, and
   artifact-only publication reproduction are tested together [R7-R10].
5. **Empirical falsification under the tested representations.** The repository
   shows where subtraction, raw field OOD, temporal geometry, and sparse routing
   did not deliver the expected gains on its frozen or deterministic feature
   pipelines. These results do not establish that the mechanisms fail on
   stronger frozen commodity representations [R3,R4,R6].

Items 2-5 may be distinctive as an integration and evidence discipline. They are
not established as legally or scientifically novel merely because this search
did not find an identical package.

---

## 4. Current Methodology

### 4.1 Learning pipeline

The principal classification path is:

$$
x \xrightarrow{h_{\psi}} z
\xrightarrow{\text{greedy fit}} \{\phi_{km}\}
\xrightarrow{\text{fusion}} \Phi_k(z)
\xrightarrow{g_{\omega}} p(y=k\mid x).
$$

Here:

- $h_{\psi}$ is a pretrained or train-fitted representation such as
  MobileNetV2, HOG, PCA/LDA/scaling, an exact temporal window, or a deterministic
  reservoir;
- $\phi_{km}$ is a primitive field for class $k$ and component $m$;
- fusion creates a class-score vector;
- $g_{\omega}$ is raw argmin, temperature, diagonal logistic, multinomial
  logistic, or another frozen readout [R1-R4].

Construction uses seeded proposals, covariance/SVD fitting, capture and
contamination scoring, acceptance thresholds, residual pools, and optional
held-out subtraction. Additive parameters may later receive supervised analytic
gradient updates. Each stage has deterministic state and recovery contracts
[R1,R7].

### 4.2 How this differs from nearby methods

**Versus QDA/MDA/GMM.** GEODE models support-like radial fields and composes
explicit parts. A Gaussian discriminant includes log determinants, class priors,
and normalized component likelihoods. GEODE's raw score is therefore not simply
“a better GMM”; it optimizes different quantities and discards useful density
terms unless a later readout reconstructs them [L1-L3,R1].

**Versus RBF/prototype networks.** Both classify by distance to local
representatives. Prototypical networks jointly learn the embedding so Euclidean
distance is task-relevant; current GEODE usually freezes or separately fits the
embedding before geometric construction [L6-L8,R2]. That separation is now a
deliberate design boundary rather than an omission (Section 5.4); the open
question is whether a stronger frozen space plus a frozen affine interface
recovers the accuracy that joint embedding learning buys.

**Versus JEPA and latent predictive learning.** CPC predicts future latent states;
I-JEPA predicts representations of masked image regions; V-JEPA predicts masked
video features; BYOL, DINO, and data2vec align or predict teacher representations
[L54-L62]. These methods learn $h_\psi$ from unlabeled structure. GEODE instead
begins after that stage and learns explicit class-conditioned support and decision
fields in a frozen output space. JEPA is therefore a candidate supplier of
$h_\psi$, not a competing geometric head. The reusable comparison is to freeze
the released encoder and fit linear, kNN/prototype, attentive, likelihood, and
GEODE heads on identical features. Reproducing JEPA pretraining is neither
necessary nor feasible at this repository's compute scale.

**Versus two-stage one-class classification.** Prior work already learns or
acquires a self-supervised representation, freezes it, and fits a statistical
one-class head [L63]. That is a direct precedent for GEODE's support-confidence
path. GEODE differs in its multiclass mixture of explicit components and its
review, transactional-edit, replay, and rollback contracts, not in the basic
separation between representation and support model.

**Versus intrinsically interpretable prototype and concept models.** ProtoPNet
and ProtoTree make decisions through learned prototype parts or prototype-guided
routing; concept bottleneck models expose human-named variables that can be
corrected at inference time [L64,L65,L67]. GEODE components are editable but are
not automatically semantic concepts or recognizable input parts. These methods
are therefore required controls for an accuracy-editability claim whenever the
dataset and annotation regime permit a fair comparison.

**Versus neural implicit fields.** Both expose continuous fields and level-set
boundaries. Neural implicit methods use deep coordinate-conditioned networks or
learned latent codes, sharing parameters compositionally across the domain.
GEODE uses a finite explicit primitive set chosen largely by discrete greedy
search [L14-L17,R1].

**Versus deep classifiers.** Deep systems learn representation and boundary
together, usually with many layers that reuse features across classes. GEODE's
class models are mostly independent after a shared fixed transform, improving
editability but reducing shared statistical strength [R1-R4].

**Versus GCD/open-world learning.** GCD methods train representations and
classifiers to recover semantic partitions from labeled and unlabeled data.
GEODE's completed stream path intentionally stops at event surfacing and human
review, so its objective and outputs are not GCD accuracy [L30-L33,R5,R6].

---

## 5. Why a Differentiable Geometric Boundary Is Not Yet SOTA

### 5.1 Short answer

A geometric approach is not inherently less accurate. A sufficiently expressive
field over a sufficiently informative learned representation can approximate an
excellent decision rule. Neural implicit fields and metric-learning systems
already demonstrate this [L7,L14-L17].

The current GEODE learner is less accurate because its **effective learning
system** is constrained in ways that the phrase “differentiable field” hides:

- the fixed representation was never selected or shaped to make local quadratic
  structure meaningful (joint optimization with the head is a declared non-goal;
  see Section 5.4);
- model structure is selected by discrete greedy search;
- the construction objective is not the final predictive loss;
- only additive parameters use the supervised differentiable optimizer;
- class models share little task-trained capacity;
- finite covariance estimates are expensive and unstable in sparse classes;
- smooth radial components may require many pieces for irregular boundaries;
- calibration repairs outputs after fitting rather than shaping representation;
- compute and data budgets are far smaller than modern pretrained systems.

### 5.2 Error decomposition

It is useful to decompose excess risk schematically as

$$
\mathcal{E}
\approx
\mathcal{E}_{\text{representation}}
+\mathcal{E}_{\text{approximation}}
+\mathcal{E}_{\text{estimation}}
+\mathcal{E}_{\text{optimization}}
+\mathcal{E}_{\text{objective}}
+\mathcal{E}_{\text{shift}}.
$$

This is an explanatory decomposition, not an identity. It separates failure
modes that differentiability alone cannot remove.

#### Representation error

A boundary can only use information present in $z=h(x)$. In the image studies,
MobileNetV2 supplies most semantic abstraction. In Tier 6, exact windows or a
small deterministic reservoir do not learn morphology, syntax, long-range
content selection, or semantic context. Transformers, temporal convolutions, and
selective state-space models demonstrate that non-image tasks also benefit from
strong architecture-specific representation learning [L41-L45]. CNNs are not
the only exception.

The repository's own representation comparison is causal evidence: moving from
HOG to MobileNetV2 substantially improved closed-set and neighborhood behavior,
while changing downstream geometry alone did not close the remaining gap [R5].

#### Approximation efficiency

A union of local quadratic regions can approximate complicated sets, but the
number of pieces may grow rapidly when a boundary is twisted, disconnected,
non-convex, thin, or aligned poorly with the coordinates. Deep compositional
models can reuse intermediate features to express such boundaries with fewer
shared parameters. Explicit per-class primitives trade compact compositional
reuse for local interpretability.

Full covariance does not remove this issue. One ellipsoid expresses one convex
quadratic support region. A complex posterior boundary may require many
ellipsoids and carefully coordinated overlaps. The original primitive-family comparison was superseded because its custom
spherical covariance fitter inherited the generic custom-fitter \(2d+1\) seed
instead of the family-correct \(d+2\). A matched five-seed rerun now uses direct
\(d+2\) sphere fitting. Spherical covariance wins selection on all five seeds
and reaches 83.96% mean test accuracy, versus 81.03% for full covariance. The
paired sphere-minus-full difference is 2.94 percentage points with a 95%
t-interval of [2.47, 3.40] percentage points [R12]. This validates an accuracy
advantage under the measured protocol, not an efficiency advantage: spheres use
139.8 primitives on average versus 58.6 for full covariance and take 0.364
versus 0.241 seconds to fit. Separately, the built-in sphere path previously
used a full \(d(d+3)/2\) ellipsoid seed and projected the result; that path is
also corrected to direct \(d+2\) fitting. The constructor and shared fitting
helpers therefore now default to spheres; full and diagonal ellipsoids remain
explicit opt-in families. This is a protocol-backed default, not a claim that
spheres dominate on every dataset or every seed.

The corrected five-seed primitive-by-CSG rerun preserves that conclusion.
Without subtraction, spheres reach 83.94% mean accuracy versus 81.75% for full
covariance and 83.50% for diagonal covariance. Sphere minus full is +2.19
percentage points with a 95% t-interval of [0.93, 3.45]; sphere minus diagonal
is +0.44 points with an interval of [-0.12, 0.99], so superiority over diagonal
covariance is not established. Subtractive A1/A2 changes sphere accuracy by
exactly 0.00 points and accepts no carvings. Full covariance accepts nine
carvings and changes 17 predictions, but its mean accuracy change is 0.00
points with interval [-0.05, 0.05]. Subtraction therefore remains opt-in and
evidence-triggered [R12].

#### Estimation error and dimensionality

A full covariance primitive has $O(d^2)$ parameters, and the constructor's
minimal quadratic seed grows as $d(d+3)/2$. Rare classes and local components may
not have enough independent observations to estimate orientation and radii
reliably. High-dimensional distances also become less discriminative when
irrelevant dimensions dominate [L46,L47].

Diagonal or spherical primitives reduce variance but increase bias. Low-rank
plus diagonal metrics are a more promising middle ground than choosing only
full, diagonal, or spherical covariance.

#### Optimization error

The field is differentiable with respect to additive primitive parameters, but
the complete learner is not one smooth optimization problem. Candidate sampling,
accept/reject decisions, residual removal, component count, topology, hard CSG,
and representation selection are discrete. Early greedy choices change later
residual pools and can be difficult to undo.

The supervised optimizer starts from the constructed topology and cannot add,
split, merge, or remove primitives through gradient descent. It also rejects
subtractive paths. Consequently, differentiability improves parameters inside a
chosen structure; it does not search the whole model family [R1,R2].

#### Objective mismatch

Greedy construction rewards support capture and penalizes contamination. Final
classification is evaluated by balanced accuracy, cross-entropy, calibration,
and sometimes OOD metrics. Those objectives are related but not equivalent.

A generative support estimate can spend capacity modeling regions irrelevant to
the decision boundary. Conversely, direct logistic and RBF controls optimize a
discriminative boundary on the same transformed features. Their small but
consistent advantage on CIFAR is therefore unsurprising [R3,R4].

Raw GEODE radial scores also omit terms that make Gaussian scores comparable
across covariance volumes, such as log determinants and class priors. Calibration
can learn a correction, but the large raw-to-calibrated gap shows that geometry
alone does not determine calibrated posterior evidence [L1-L3,L20,L21,R3].

#### Sharing and compositionality

Current class fields are fitted mostly independently. This is good for local
edits and rollback, but it does not learn features shared across related classes.
Modern neural systems amortize representation learning across classes, examples,
positions, and tasks. Even prototype networks learn a common embedding before
measuring geometric distance [L7].

#### Data and compute scale

“Can represent” comparisons are unfair when one system uses a small frozen
feature budget and another uses large-scale pretraining. BERT, GPT-style models,
Vision Transformers, and modern state-space models derive much of their accuracy
from learned representations, data scale, and optimization infrastructure
[L42-L45]. A geometric head can potentially sit on those representations, but it
does not replace the representation-learning investment.

### 5.3 Why the claim fails outside image analysis too

The proposition that geometry should match SOTA everywhere except images is too
strong.

- **Language and sequences:** order, variable-range dependency, content-based
  selection, and compositional semantics require a representation mechanism.
  Attention, temporal convolution, recurrence, and selective state spaces encode
  these biases [L41-L45]. A geometric output head does not infer them by itself.
- **Audio and time series:** phase, frequency, multiscale temporal structure, and
  nonstationarity similarly require learned or engineered temporal features.
- **Graphs and physical systems:** permutation, locality, symmetry, and
  equivariance matter; geometric deep learning is largely about respecting these
  domain symmetries, not merely drawing a boundary in a flat vector space [L48].
- **Tabular data:** deep learning is not uniformly dominant. Tree ensembles often
  win on medium-sized heterogeneous tables because their axis-aligned,
  discontinuous inductive bias handles irrelevant features and irregular
  functions well [L49,L50]. GEODE must be compared with task-appropriate methods,
  not with one universal neural baseline.

The right hypothesis is narrower:

> Given a fixed task-appropriate representation, an explicit geometric head may
> approach a flexible black-box head on the same features while offering better
> editability, inspectability, local support control, and lifecycle guarantees.

That hypothesis is plausible and testable. It is not yet established by the
repository.

### 5.4 Why joint encoder-head training is a non-goal, not an unfinished task

Sections 5.1-5.3 identify representation error as the largest term. The obvious
remedy — train the encoder together with the geometric head — is rejected here
on principled grounds, not for lack of engineering capacity. Three conflicts are
decisive.

**Edits stop being stable.** An edit is meaningful because a component is a
stable object in a declared representation: deleting it changes predictions only
inside a measurable feature-space region. The map $h_\psi$ from input to feature
space must remain the same map that was in force when the component was fitted.
If the encoder keeps moving, even that feature-space statement loses its
referent. Freezing is necessary for stability, but it does not make the preimage
$h_\psi^{-1}(R)$ of an ellipsoid $R$ simple, connected, semantically coherent, or
human-attributable. Raw-input locality must therefore be estimated empirically on
a frozen evaluation population rather than inferred from feature-space volume.

**Provenance breaks.** The E9-E11 contracts — exact replay, rollback, immutable
bundles, changed-region reporting — are defined over a fixed embedding. A
drifting encoder silently invalidates every cached component, calibration
object, support profile, and review group derived from the old space, without
any signal that they became stale.

**The geometry becomes epiphenomenal.** With sufficient gradient pressure, an
encoder can warp the space so that almost any head separates the classes. The
head then no longer explains the decision; the encoder does. The result is a
black box wearing an interpretable hat, which is exactly the outcome this
project exists to avoid.

The adopted alternative has four parts.

1. **Buy the representation.** Use a frozen self-supervised backbone — DINOv2 or
   SigLIP-class features [L51-L53] — as a commodity artifact. This is standard
   practice for linear-probe and prototype evaluation, and it is philosophically
   aligned with GEODE: the representation becomes an immutable, versioned,
   hash-addressed input, exactly like a model bundle. A lifecycle system is no
   more obliged to invent its own representation learner than a database is
   obliged to invent its own filesystem.
2. **Train a small interface once, then freeze it.** A single linear projection
   or low-rank affine map, trained with the compactness/margin objective under
   the pre-test development protocol, before any geometry is fitted, and then
   frozen and hashed. Restricting the adapter to an affine map preserves geometric
   semantics: ellipsoids remain ellipsoids, Mahalanobis structure is preserved
   up to a fixed change of basis, and representation-space changed-region volume
   remains computable in closed form. Empirical input-space changed sets are
   measured separately. After freezing, every existing GEODE contract applies
   unchanged.
3. **Make the head differentiable, not the pipeline.** Inside the frozen space,
   train component centers, metrics, and temperatures discriminatively with
   cross-entropy, using the deterministic greedy constructor as initialization.
   This is where end-to-end reasoning genuinely pays: it is cheap, stable, and
   preserves editability because components remain explicit objects in a space
   that does not move.
4. **Treat representation change as a lifecycle event.** If the embedding must
   be updated, that is a new frozen embedding plus a geometry refit plus a
   migration report ("component 12 in space v1 corresponds to components 12a and
   12b in v2") plus rollback to the previous bundle. This converts the apparent
   weakness into the project's most differentiated capability: GEODE becomes the
   system that manages representation change auditably, which no end-to-end
   learner can offer.

The resulting claim is sharper and honest:

> Given a frozen commodity representation, an explicit geometric expert head
> retains $X\%$ of the accuracy of the best black-box head on the same features,
> while offering measured edit locality, exact rollback, and audited adaptation
> that the black-box head cannot provide.

If $X$ is near 100 — plausible, because good self-supervised spaces already make
classes compact and close to linearly separable — GEODE wins its comparison
outright: parity in accuracy, plus editability. The project never has to claim
state of the art in general; it claims **head parity on state-of-the-art
representations plus a Pareto advantage on editability**, which is precisely
what the frontier study in Section 9 measures.

---

## 6. Evidence from the Current Repository

| Finding                                                              | Evidence                                                                                                                               | Interpretation                                                                                      |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| CIFAR-10 GEODE is close to, but below, matched logistic/RBF controls | Five-seed study: GEODE about 83.6%, direct logistic about 84.0%, RBF about 84.2% [R3]                                                  | The representation is strong; the geometric head retains much but not all discriminative efficiency |
| Subtraction has no aggregate public-data benefit                     | Corrected \(d+2\) primitive-by-CSG rerun: sphere A1/A2 = 0.00 pp; full-covariance A1/A2 = 0.00 pp [-0.05, 0.05] [R12]                    | More topological flexibility does not help without correctly located, validated exclusion structure |
| CIFAR-100 E4 passes non-inferiority but trails controls              | 65.26% balanced versus 67.33% logistic and 68.13% RBF [R4]                                                                             | The gap persists under a larger class protocol                                                      |
| Tier 6 trails same-feature and sequence controls                     | 30.36% versus 34.64% linear, 44.50% matched 5-gram [R3]                                                                                | Representation and task structure dominate the geometric readout                                    |
| Raw geometry is poorly calibrated                                    | Large raw NLL/ECE improvements after readout calibration [R3]                                                                          | Radial support is not posterior probability                                                         |
| Raw SDF is weak for OOD                                              | Maximum calibrated probability beats raw/corrected field distances; FPR95 remains high [R3,R4]                                         | Bounded geometry in the current feature space is not reliable open-space evidence                   |
| More flexible covariance is not always better                        | Corrected five-seed \(d+2\) sphere rerun beats full covariance by 2.94 pp [2.47, 3.40], while using more primitives and fit time [R12] | Nominal expressivity does not determine accuracy, and accuracy superiority does not imply efficiency |
| Candidate routing does not reduce observed wall-clock latency        | Synthetic M12 and real E5 candidates fail latency/quality gates and remain shadow-only; exhaustive evaluation stays authoritative [R4] | Geometric bounds alone do not guarantee systems speedup                                             |
| Event surfacing is stronger than semantic grouping                   | Event recall is high while ARI/distinct-group recovery remain weak and semantic publication stays human-gated [R5]                     | Representation learning for semantic clustering is missing                                          |
| Frozen probe quality depends strongly on the representation artifact | Corrected bounded M19 S1: identity weighted kNN is 93.00% on DINOv2-small, 87.67% on SigLIP, and 80.33% on I-JEPA ViT-H/16 int8; the strongest identity heads reach 95.00%, 89.67%, and 86.33%, respectively, and neither affine interface improves kNN [R11] | A standard probe is useful representation evidence, but one seed and bounded quantized artifacts cannot establish a head or representation claim |
| DINOv2 transfers strongly to five-shot Flowers-102                    | Bounded identity features: DINOv2 linear 99.35% and weighted kNN 99.02%; SigLIP linear 97.39%; I-JEPA linear 63.73% [R13]              | The CIFAR ordering survives this transfer smoke test, but one bounded split is not a confirmatory representation ranking |

These results are consistent with the error decomposition above. They do not
show that fields are unsuitable; they show that the current field learner and
representation coupling are not yet competitive.

The corrected SigLIP result supersedes the earlier 29.00% weighted-kNN figure.
That figure came from selecting the first ONNX output
(`last_hidden_state[:,0]`) despite a declared `pooler_output` policy. M19 now
binds each pooling policy to a named ONNX output and fails closed when the
required tensor or rank is absent. The I-JEPA arm uses the pinned
`onnx-community/ijepa_vith16_1k` revision
`59ebd911845f639c18e06b1239ac243a30a7d35f`, mean-pools its 784 patch tokens,
and records the upstream CC BY-NC 4.0 restriction. It is suitable for this
non-commercial research evaluation, not unrestricted downstream reuse.

The same S1 replay now reports representation diagnostics on the frozen training
split. Identity 10-NN neighborhood purity is 0.8284 for DINOv2-small, 0.7512 for
SigLIP, and 0.6146 for I-JEPA. Their mean-radius/minimum-centroid-separation
ratios are 1.7060, 1.7332, and 2.0353, respectively. The affine interfaces reduce
raw within-class radius and median local intrinsic dimension, but also reduce
neighborhood purity; they generally worsen the scale-normalized compactness
ratio, except for a small SigLIP linear-interface improvement accompanied by a
material accuracy loss. Thus reduced raw scatter alone is not evidence of
better usable geometry, and no tested interface passes the M19 retention rule.

Component efficiency is preregistered as the number of GEODE primitives in
deterministic constructor order required for each class to reach 95% frozen
training coverage at the declared capture threshold. The identity spaces remain
blocked in the historical CIFAR artifact by the bounded 128-dimensional limit;
independently, its 50 examples per class are below the current spherical
\(d+2\) requirement in every native space. The 64-dimensional affine arms have
50 examples per class, below their 66-point spherical seed, and accept zero
primitives. No component count is imputed when the target is not reached.

The bounded Oxford Flowers-102 transfer run uses the official train,
validation, and test partitions with 5/2/3 deterministic examples per class.
DINOv2-small reaches 99.35% linear, 99.02% weighted-kNN, and 99.02% prototype
test accuracy. SigLIP reaches 97.39%, 94.77%, and 97.39%; I-JEPA reaches
63.73%, 43.14%, and 52.29%. Support-compatible 4-NN training purity is 0.9275,
0.7980, and 0.2544, respectively, with compactness ratios 0.7740, 0.9625, and
1.4188. The calibrated RBF probe is much weaker, especially under five-shot
class support, and is not used to reverse the ordering. Full-covariance GMM is
blocked as under-supported. Native spherical GEODE is also blocked explicitly:
it requires 386, 770, and 1,282 points per class for DINOv2, SigLIP, and I-JEPA,
while this protocol provides five [R13].

A separate native-dimensional CIFAR-10 study then raises DINOv2-small support to
exactly \(d+2=386\) training examples per class, with 100 development and 200
test examples per class. Direct 384-dimensional spherical GEODE now fits
successfully and reaches 80.90% test accuracy. This is well below matched RBF
96.50%, linear 96.15%, weighted-kNN 95.95%, and prototype 94.70% controls
[R14]. Each class fits exactly one sphere; that sphere covers only 50.00% to
58.03% of its training class, and every residual pool is smaller than 386.
Consequently no class reaches the preregistered 95% coverage target.

The result separates feasibility from adequacy. The \(d+2\) contract removes the
quadratic full-ellipsoid support barrier and enables native DINOv2 fitting, but
meeting the algebraic seed minimum does not provide enough support for a
multi-sphere class model or competitive accuracy. The next support study should
increase examples per class enough to leave at least another \(d+2\) residual
points after the first sphere, rather than treating 386 as a recommended sample
size.

A 1,000-training-example-per-class support pilot preserves the same frozen
DINOv2 representation family, one-candidate search budget, 100-example
development partition, and 200-example test partition. The larger support
enables 4--15 spheres per class (70 total) and raises spherical GEODE test
accuracy from 80.90% to 86.25% [R15]. Training coverage rises to 61.50--80.60%,
but no class reaches 95%. Matched weighted-kNN and RBF controls both reach
96.70%, while linear and prototype controls reach 96.15% and 94.55%.

Support scaling therefore helps GEODE materially, but does not close the
frozen-feature accuracy gap or satisfy the component-efficiency target. Because
the pilot demonstrates genuine multi-sphere growth, the preregistered seeds 11,
23, and 37 should now be run at the frozen 1,000-example support level. The
pilot remains development evidence until that independent-seed confirmation is
complete.

---

## 7. Weak Points of the Current Learning Approach

### 7.1 The fixed space was never chosen or shaped for geometric compactness

PCA/LDA/scaling or temporal features are selected before the final geometric
objective is optimized. LDA favors global linear class separation; it need not
make each class a compact union of stable quadratic supports. The weakness is
not that representation and head are separate stages — Section 5.4 argues that
separation is required — but that the fixed space is inherited by accident
rather than selected and shaped, once, for the geometry that will be fitted in
it. The remedy is a stronger frozen backbone plus a one-time frozen affine
interface trained with a compactness/margin objective, not gradient flow from
the head into the encoder.

### 7.2 Greedy structure search is path-dependent

Accepted primitives alter the residual pool. There is no global split/merge,
birth/death, or topology optimization after construction. Determinism makes the
path reproducible, not globally optimal.

### 7.3 Training loss and evaluation loss differ

Capture/contamination objectives, radial refinement, multinomial calibration,
and final risk metrics are separated. Capacity can be spent on support coverage
that does not improve the class boundary.

### 7.4 Primitive complexity is poorly matched to sample support

Full covariance is data hungry; spherical models can be too rigid. Current
adaptive fallbacks help execution but do not provide a principled complexity
prior or uncertainty over covariance estimates.

### 7.5 Class-wise independence wastes shared structure

Separate models improve edit locality but duplicate geometry and fail to exploit
hierarchies or attributes shared across classes.

### 7.6 Hard subtraction is difficult to optimize

CSG difference introduces non-smooth topology and is fitted after additive
construction. The completed data show that this extra flexibility rarely lands
where it improves held-out decisions.

### 7.7 Distance is treated as evidence too readily

A point can be close to a support boundary in a feature space while still being
semantically OOD. Pretrained representations may map unrelated inputs near known
classes. Distance needs density, epistemic, or conformal context and explicit
shift assumptions.

### 7.8 Model selection is broader than the confirmatory evidence

Many exploratory variants were evaluated, but only a subset have repeated,
matched public-data confirmation. The strongest general claim still rests on a
small number of datasets and representations.

### 7.9 Scalability is explicit but not yet efficient

Per-class fields and exhaustive primitive evaluation grow with class and
component counts. Candidate routing has not produced a real latency advantage,
and physical multi-host training remains open.

### 7.10 Stable feature-space geometry is not raw-input semantics

A frozen nonlinear encoder prevents a component's coordinates from drifting, but
does not guarantee that its raw-input preimage is contiguous, perceptually local,
or aligned with a human concept. Analytic ellipsoid volume and overlap are valid
in the declared frozen feature space. Claims about input-space locality,
recognizability, or semantic edit intent require empirical perturbation,
counterfactual, retrieval, and human-review evidence. GEODE is therefore an
interpretable head over a potentially black-box representation, not an
intrinsically interpretable end-to-end model.

---

## 8. Recommended Improvements

### Priority 1: fix a strong frozen space, then freeze a one-time affine interface

Replace the earlier joint-adaptation proposal with a frozen-trunk,
trained-interface design (Section 5.4). Concretely:

1. add frozen self-supervised backbone arms — DINOv2, I-JEPA, and a
   SigLIP/CLIP-class model — alongside the current MobileNetV2 features
   [L51-L55];
2. train one small affine interface $h_\psi$ (linear projection or low-rank
   affine map) **once**, under the pre-test development protocol, before any
   geometry is fitted;
3. freeze and hash $h_\psi$, and treat it thereafter as an immutable input
   artifact with the same provenance status as a model bundle;
4. compare all heads inside that single fixed space.

The interface objective may combine discriminative loss and geometric
regularity:

$$
L = L_{\text{CE}}
+\lambda_{\text{compact}}L_{\text{within}}
+\lambda_{\text{margin}}L_{\text{between}}
+\lambda_{\text{complexity}}\Omega(\Theta).
$$

The comparison matrix is then a head comparison in a fixed space:

1. frozen backbone + current GEODE;
2. frozen backbone + linear/RBF/prototype/mixture controls;
3. frozen backbone + frozen interface + linear head;
4. frozen backbone + frozen interface + GEODE head;
5. frozen backbone + frozen interface + prototype head.

Restricting $h_\psi$ to an affine map is deliberate: it preserves ellipsoidal
semantics in representation space and keeps representation-space changed-region
volume computable.
Prototypical networks and metric learning remain the direct prior-art baselines
[L7,L8], but GEODE's deliverable is a frozen, hash-addressed artifact, not a
jointly training system.

Adopt the standard frozen-representation evaluation ladder from self-supervised
learning: weighted kNN with no learned downstream parameters, linear probing,
then increasingly structured heads. Use released checkpoints only. Full
ImageNet, LVD-142M, WM256, and internet-video pretraining are provenance for the
encoder, not reproducible GEODE experiments [L51,L55-L60].

**Explicit non-goal.** Joint encoder-head gradient training is out of scope
because it invalidates edit-locality and replay contracts. Representation change
is handled as a versioned migration event (Priority 9).

### Priority 2: train the head discriminatively inside the frozen space

All differentiable training happens here, below a frozen encoder and a frozen
interface. Keep the deterministic constructor as initialization, then alternate:

1. soft sample-to-component responsibilities;
2. discriminative cross-entropy updates of centers, metrics, and temperatures;
3. held-out split/merge/birth/death proposals;
4. complexity-penalized acceptance;
5. exact replay and rollback checks.

This borrows from mixture modeling and multi-model fitting rather than claiming a
new optimizer. The key GEODE question is whether these standard refinements can
be made transactional and editable. Because the space does not move, components
remain explicit objects with stable feature-space meaning throughout, so
feature-space edit-locality and rollback guarantees survive gradient training of
the head. Empirical input-domain locality still requires separate measurement.

### Priority 3: add low-rank and shared metrics

Use

$$
P_k = D_k + U_kU_k^\top
$$

with diagonal $D_k$ and low rank $U_k$, or share a global metric plus small
class-specific corrections. This interpolates between spherical/diagonal bias
and full-covariance variance, reduces sample requirements, and shares statistical
strength.

### Priority 4: compare support, likelihood, and discriminative energies fairly

For the same fitted components, evaluate:

- current radial field;
- Gaussian log likelihood including $\log|\Sigma|$ and priors;
- normalized mixture likelihood;
- discriminatively trained field energy;
- logistic/RBF/prototype controls.

This isolates whether underperformance comes from primitive placement or from
throwing away density terms, while recognizing that ordinary discriminative
classifiers can themselves be interpreted as energy-based models [L66].

### Priority 5: use task-native frozen representations outside images

For Tier 6, freeze a downstream budget and compare causal convolution, GRU/LSTM,
small Transformer, and selective state-space representations before changing
geometry [L41-L45]. Each encoder is trained or obtained separately and then
frozen and hashed before any geometry is fitted, exactly as in Priority 1.
Retain matched-data n-gram and same-feature linear controls. The geometric head
should be evaluated as a head, not as a substitute for sequence representation
learning.

Add a separate public video arm following V-JEPA's attentive-probe convention
over frozen features [L56,L57]. Use HMDB-51 for bounded development and UCF-101
for confirmation with a released V-JEPA or another preregistered public video
encoder. Compare a linear probe, an attentive pooling probe, a prototype head,
and GEODE on identical cached clip representations. Something-Something v2 and
Kinetics-400 remain external-validity targets, not core gates, because their
storage, preprocessing, and compute costs are disproportionate to this repository.

For graphs or physical data, use symmetry-aware/equivariant representations
before fitting class fields [L48]. For tabular data, include boosted-tree
baselines and test robustness to irrelevant features [L49,L50].

### Priority 6: replace global subtraction with evidence-triggered local residuals

Do not make CSG default. Activate a subtractive or local residual component only
when an independently verified exclusion region exists and held-out boundary loss
improves. A smooth local neural residual could be compared with an explicit
negative primitive, but any gain must be balanced against editability.

### Priority 7: separate classification confidence from support confidence

Use distinct outputs for:

- relative class posterior;
- in-support evidence;
- epistemic or ensemble disagreement;
- conformal prediction set under stated exchangeability assumptions;
- review priority.

Do not interpret one radial score as all five quantities [L22-L26].

For support confidence, adopt the established two-stage one-class protocol
[L63]: on CIFAR-10, treat each class in turn as the only inlier training class and
the remaining classes as outliers, then report per-class and macro AUROC, AUPR,
and FPR95. Confirm retained support methods on MVTec-AD at image level. This tests
the support head separately from multiclass posterior quality and avoids
inventing a bespoke OOD benchmark.

### Priority 8: benchmark the claimed advantage, not only accuracy

If GEODE's value is editability and lifecycle safety, evaluate that directly:

- changed-region size after an edit;
- number of unaffected predictions preserved exactly;
- rollback latency and success;
- evidence needed per safe adaptation;
- review burden;
- calibration drift;
- model bytes and inference cost;
- accuracy/editability Pareto frontier.

A slightly less accurate model may be scientifically useful if it establishes a
measurable operational advantage. That tradeoff should be explicit rather than
implied.

### Priority 9: handle representation change as an audited migration event

Because the encoder never trains with the head, representation change must have
its own lifecycle path. Define a migration as: new frozen embedding, refit
geometry under the existing deterministic contract, produce a component
correspondence report, and retain rollback to the previous bundle. The study
should measure:

- component correspondence between spaces (which v1 component maps to which v2
  component, including splits and merges, with an unmatched residue reported);
- edit survival: whether previously accepted edits still express the same intent
  after migration, and how much evidence re-review costs;
- calibration and support-profile invalidation, reported explicitly rather than
  silently reused;
- rollback exactness and latency to the pre-migration bundle;
- accuracy and changed-region deltas attributable to the migration alone.

The refreshed search did not identify a directly comparable system-level study.
That absence is not a novelty claim, but it makes this the clearest place where
GEODE can seek differentiation rather than mere predictive competitiveness.

---

## 9. Proposed Experimental Sequence

### Experiment A: head comparison inside frozen spaces

Run five seeds on at least one image and one non-image dataset with the same
frozen splits. Cross:

- representation: current frozen features, frozen DINOv2, frozen I-JEPA,
  frozen SigLIP/CLIP, each optionally composed with the one-time frozen affine
  interface, and a frozen task-native encoder where the dataset requires one;
- head: weighted kNN, linear, RBF, prototype, Gaussian mixture, GEODE;
- budget: matched parameter count and matched fit time.

Every representation arm is frozen and hashed before any head is fitted. This
identifies representation error separately from head approximation and
optimization error while keeping all lifecycle contracts valid, and it directly
estimates the parity ratio $X$ in the claim of Section 5.4.

Use CIFAR-10/100 for continuity and bounded repeated runs, then Oxford
Flowers-102 as the preselected small transfer confirmation because its higher
resolution and many low-support classes better exercise pretrained visual
representations and component sample efficiency. ImageNet-1K linear probing is
the external reference protocol, not a required core run.

### Experiment B: radial field versus full likelihood

Use identical components and compare radial, determinant/prior-corrected
Gaussian, mixture likelihood, and discriminative energy scores. If likelihood
wins, component construction may be adequate while score semantics are not.

### Experiment C: greedy versus alternating topology

Start from identical constructor outputs. Compare parameter-only refinement with
soft reassignment and held-out split/merge. Track predictive gain, component
count, stability, and edit locality.

### Experiment D: intrinsic-dimension and sample-support sweep

Vary ambient dimension, estimated intrinsic dimension, observations per class,
and covariance rank. Measure when full, low-rank, diagonal, and spherical
primitives cross over. This should replace one global primitive-family choice
with a preregistered support-dependent policy.

### Experiment E: task-native temporal representations

On frozen WikiText splits, compare exact window, reservoir, TCN, GRU, small
Transformer, and selective SSM representations with both linear and GEODE heads.
This tests whether GEODE remains behind after representation quality is controlled.

Run an independent video track on HMDB-51, followed by UCF-101 only for retained
methods. Extract and hash clip features once from a released frozen V-JEPA-class
or preregistered public video encoder. Compare linear, attentive, prototype, and
GEODE heads on identical train/validation/test splits. Do not compare against
published V-JEPA numbers unless preprocessing, checkpoint, probe capacity, and
split protocol match.

### Experiment F: accuracy-editability frontier

Apply matched local edits to GEODE, prototype, tree, and compact neural models.
Measure target correction, collateral prediction changes, retraining cost,
rollback exactness, and audit size. This tests the project's most credible
advantage.

Where compatible implementations and annotations exist, include an intrinsically
interpretable prototype model such as ProtoPNet or ProtoTree. Treat concept
bottleneck models as a separate concept-supervised control rather than forcing
them onto datasets without concept labels. Report analytic changed-region volume
only in feature space and empirical changed-example fractions in input datasets.

### Experiment G: representation-migration study

Take a retained GEODE model fitted in frozen space v1 and migrate it to a second
frozen space v2 (for example MobileNetV2 to DINOv2, or DINOv2 to SigLIP, or the
same backbone with a re-trained frozen interface). Refit geometry under the
existing deterministic contract and measure:

- component correspondence, split/merge structure, and unmatched residue;
- survival of previously accepted edits and the review cost of re-confirming
  them;
- calibration and support-profile invalidation and refresh cost;
- exact rollback to the v1 bundle;
- accuracy and changed-region deltas attributable to migration alone.

Include a black-box control — a fine-tuned MLP head on the same two spaces — to
show which of these quantities that control simply cannot report. This is the
most differentiated experiment in the program.

### Experiment H: two-stage support benchmark

Using the same frozen DINOv2 and I-JEPA artifacts as Experiment A, run the ten
CIFAR-10 one-class tasks from prior two-stage representation work [L63]. Compare
kNN distance, Mahalanobis, Gaussian likelihood, raw GEODE field, and calibrated
GEODE support scores. Confirm only retained methods on MVTec-AD using image-level
AUROC, AUPR, and FPR95. Pixel localization is out of scope unless a spatial head
is added and preregistered separately.

### Advancement rule

Do not advance a more complex GEODE learner unless it improves a preregistered
primary endpoint over the current head while preserving:

- matched representation and data budgets;
- a frozen, hashed encoder and interface for the whole comparison;
- exact replay;
- calibration tolerance;
- bounded complexity;
- edit/rollback guarantees;
- independent confirmatory seeds.

---

## 10. Revised Claim Boundary

The evidence supports this statement:

> GEODE is an explicit geometric-head and lifecycle research system. On current
> frozen representations it retains much of the predictive information available
> to classical controls, while providing inspectable components, deterministic
> recovery, transactional edits, and conservative promotion. It has not yet
> demonstrated predictive superiority.

The target claim that the Section 9 program is designed to establish is:

> Given a frozen commodity representation, an explicit geometric expert head
> attains accuracy within a stated margin of the best black-box head on the same
> features, while providing measured edit locality, exact rollback, and audited
> representation migration that the black-box head cannot provide.

This is head parity on a state-of-the-art representation plus a Pareto advantage
on editability. It is deliberately not a claim of state of the art in general.

The evidence does not support:

- that differentiable fields automatically match arbitrary SOTA learners;
- that geometry eliminates the need for task-specific representation learning;
- that ellipsoidal unions are parameter-efficient for every boundary;
- that subtraction generally improves classification;
- that raw distance is calibrated probability or reliable OOD evidence;
- that current review groups discover semantic classes;
- that current routing scales better in wall-clock time;
- that the integration is novel merely because its exact combination was not
  found in this literature search.

The project also declares an explicit non-goal: joint encoder-head gradient
training is out of scope, because it invalidates the edit-locality, provenance,
and replay contracts that constitute the contribution. Representation change is
handled as a versioned migration event, not as gradient flow.

"Measured edit locality" means exact changed regions in the frozen representation
and empirical changed-example sets in the input domain. It does not imply that a
nonlinear encoder's component preimages are analytically simple or intrinsically
semantic. Likewise, JEPA-family results count as representation evidence; they do
not establish a GEODE head contribution unless the same frozen representation,
data, probe protocol, and budget are used for all heads.

---

## 11. Conclusion

The premise behind GEODE remains viable after a stricter literature review, but
it should be reformulated. Explicit differentiable geometry is a useful model
interface, not a complete learning theory. In principle, an expressive field on
a good representation can realize highly accurate decision boundaries. In
practice, representation learning, finite parameter efficiency, sample support,
objective alignment, optimization, regularization, and task structure determine
whether that potential is reached.

Current GEODE underperforms strong controls mostly because it uses a weak fixed
representation, discrete greedy topology, support-oriented objectives, and
limited supervised refinement. Those are engineering and statistical choices, not
unavoidable consequences of geometry. The strongest next test is therefore not
another primitive tweak. It is a controlled head comparison inside strong frozen
spaces, with matched data, parameters, calibration, and compute.

The resolution of the end-to-end question is to refuse it on principled grounds.
GEODE puts a frozen commodity backbone underneath, trains one small affine
interface once and freezes it, trains the geometric head discriminatively inside
that fixed space, and treats any later representation change as an audited
migration event. This keeps every lifecycle guarantee intact and converts the
project's apparent weakness — that it cannot learn its own representation — into
its clearest claim: head parity on state-of-the-art representations, plus
editability, rollback, and migration auditing that an end-to-end learner cannot
offer.

The most defensible contribution today is the combination of explicit editable
models with unusually rigorous lifecycle evidence and honest negative results.
Future work should preserve that strength while importing, with attribution,
standard advances from self-supervised representation learning, metric learning,
mixture optimization, low-rank covariance modeling, task-native encoders,
uncertainty estimation, and category discovery.

---

# References

## Literature

[L1] R. O. Duda, P. E. Hart, and D. G. Stork. _Pattern Classification_, 2nd ed. Wiley, 2001.

[L2] T. Hastie and R. Tibshirani. “Discriminant Analysis by Gaussian Mixtures.” _JRSS B_, 58(1), 1996. https://doi.org/10.1111/j.2517-6161.1996.tb02073.x

[L3] G. J. McLachlan. _Discriminant Analysis and Statistical Pattern Recognition_. Wiley, 1992.

[L4] D. S. Broomhead and D. Lowe. “Multivariable Functional Interpolation and Adaptive Networks.” Royal Signals and Radar Establishment Memorandum 4148, 1988.

[L5] J. Park and I. W. Sandberg. “Universal Approximation Using Radial-Basis-Function Networks.” _Neural Computation_, 3(2), 1991. https://doi.org/10.1162/neco.1991.3.2.246

[L6] T. Kohonen. “Learning Vector Quantization.” In _Self-Organizing Maps_. Springer, 1995.

[L7] J. Snell, K. Swersky, and R. Zemel. “Prototypical Networks for Few-shot Learning.” _NeurIPS_, 2017. https://arxiv.org/abs/1703.05175

[L8] A. Bellet, A. Habrard, and M. Sebban. “A Survey on Metric Learning for Feature Vectors and Structured Data.” 2013. https://arxiv.org/abs/1306.6709

[L9] A. A. G. Requicha. “Representations for Rigid Solids: Theory, Methods, and Systems.” _ACM Computing Surveys_, 12(4), 1980. https://doi.org/10.1145/356827.356833

[L10] J. J. Park et al. “DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.” _CVPR_, 2019. https://arxiv.org/abs/1901.05103

[L11] G. Sharma et al. “CSGNet: Neural Shape Parser for Constructive Solid Geometry.” _CVPR_, 2018. https://arxiv.org/abs/1712.08290

[L12] D. Paschalidou, A. O. Ulusoy, and A. Geiger. “Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids.” _CVPR_, 2019. https://arxiv.org/abs/1904.09970

[L13] W. Liu et al. “Marching-Primitives: Shape Abstraction from Signed Distance Function.” _CVPR_, 2023. https://arxiv.org/abs/2303.13190

[L14] L. Mescheder et al. “Occupancy Networks: Learning 3D Reconstruction in Function Space.” _CVPR_, 2019. https://arxiv.org/abs/1812.03828

[L15] B. Mildenhall et al. “NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.” _ECCV_, 2020. https://arxiv.org/abs/2003.08934

[L16] V. Sitzmann et al. “Implicit Neural Representations with Periodic Activation Functions.” _NeurIPS_, 2020. https://arxiv.org/abs/2006.09661

[L17] M. Tancik et al. “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.” _NeurIPS_, 2020. https://arxiv.org/abs/2006.10739

[L18] M. A. Fischler and R. C. Bolles. “Random Sample Consensus.” _Communications of the ACM_, 24(6), 1981. https://doi.org/10.1145/358669.358692

[L19] D. Barath and J. Matas. “Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm.” _ICCV_, 2019. https://doi.org/10.1109/ICCV.2019.00388

[L20] J. Platt. “Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.” 1999.

[L21] C. Guo et al. “On Calibration of Modern Neural Networks.” _ICML_, 2017. https://arxiv.org/abs/1706.04599

[L22] D. Hendrycks and K. Gimpel. “A Baseline for Detecting Misclassified and Out-of-Distribution Examples.” _ICLR_, 2017. https://arxiv.org/abs/1610.02136

[L23] K. Lee et al. “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.” _NeurIPS_, 2018. https://arxiv.org/abs/1807.03888

[L24] W. Liu et al. “Energy-based Out-of-distribution Detection.” _NeurIPS_, 2020. https://arxiv.org/abs/2010.03759

[L25] Y. Sun et al. “Out-of-Distribution Detection with Deep Nearest Neighbors.” _ICML_, 2022. https://arxiv.org/abs/2204.06507

[L26] A. Bendale and T. Boult. “Towards Open World Recognition.” _CVPR_, 2015. https://arxiv.org/abs/1412.5687

[L27] M. Ester et al. “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.” _KDD_, 1996.

[L28] R. J. G. B. Campello, D. Moulavi, and J. Sander. “Density-Based Clustering Based on Hierarchical Density Estimates.” _PAKDD_, 2013. https://doi.org/10.1007/978-3-642-37456-2_14

[L29] S. Sarfraz, V. Sharma, and R. Stiefelhagen. “Efficient Parameter-free Clustering Using First Neighbor Relations.” _CVPR_, 2019. https://arxiv.org/abs/1902.11266

[L30] S. Vaze et al. “Generalized Category Discovery.” _CVPR_, 2022. https://arxiv.org/abs/2201.02609

[L31] E. Fini et al. “A Unified Objective for Novel Class Discovery.” _ICCV_, 2021. https://arxiv.org/abs/2108.08536

[L32] K. Cao, M. Brbic, and J. Leskovec. “Open-World Semi-Supervised Learning.” _ICLR_, 2022. https://arxiv.org/abs/2102.03526

[L33] X. Wen, B. Zhao, and X. Qi. “Parametric Classification for Generalized Category Discovery: A Baseline Study.” _ICCV_, 2023. https://arxiv.org/abs/2211.11727

[L34] S.-A. Rebuffi et al. “iCaRL: Incremental Classifier and Representation Learning.” _CVPR_, 2017. https://arxiv.org/abs/1611.07725

[L35] D. Lopez-Paz and M. Ranzato. “Gradient Episodic Memory for Continual Learning.” _NeurIPS_, 2017. https://arxiv.org/abs/1706.08840

[L36] P. Buzzega et al. “Dark Experience for General Continual Learning.” _NeurIPS_, 2020. https://arxiv.org/abs/2004.07211

[L37] R. A. Jacobs et al. “Adaptive Mixtures of Local Experts.” _Neural Computation_, 3(1), 1991. https://doi.org/10.1162/neco.1991.3.1.79

[L38] W. Fedus, B. Zoph, and N. Shazeer. “Switch Transformers.” _JMLR_, 23, 2022. https://arxiv.org/abs/2101.03961

[L39] Y. Zhou et al. “Mixture-of-Experts with Expert Choice Routing.” _NeurIPS_, 2022. https://arxiv.org/abs/2202.09368

[L40] S. Roller et al. “Hash Layers for Large Sparse Models.” _NeurIPS_, 2021. https://arxiv.org/abs/2106.04426

[L41] S. Bai, J. Z. Kolter, and V. Koltun. “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.” 2018. https://arxiv.org/abs/1803.01271

[L42] A. Vaswani et al. “Attention Is All You Need.” _NeurIPS_, 2017. https://arxiv.org/abs/1706.03762

[L43] J. Devlin et al. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” _NAACL_, 2019. https://arxiv.org/abs/1810.04805

[L44] T. B. Brown et al. “Language Models are Few-Shot Learners.” _NeurIPS_, 2020. https://arxiv.org/abs/2005.14165

[L45] A. Gu and T. Dao. “Mamba: Linear-Time Sequence Modeling with Selective State Spaces.” 2024. https://arxiv.org/abs/2312.00752

[L46] K. Beyer et al. “When Is Nearest Neighbor Meaningful?” _ICDT_, 1999. https://doi.org/10.1007/3-540-49257-7_15

[L47] C. C. Aggarwal, A. Hinneburg, and D. A. Keim. “On the Surprising Behavior of Distance Metrics in High Dimensional Space.” _ICDT_, 2001. https://doi.org/10.1007/3-540-44503-X_27

[L48] M. M. Bronstein et al. “Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.” 2021. https://arxiv.org/abs/2104.13478

[L49] L. Grinsztajn, E. Oyallon, and G. Varoquaux. “Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?” _NeurIPS_, 2022. https://arxiv.org/abs/2207.08815

[L50] R. Shwartz-Ziv and A. Armon. “Tabular Data: Deep Learning Is Not All You Need.” _Information Fusion_, 81, 2022. https://arxiv.org/abs/2106.03253

[L51] M. Oquab et al. “DINOv2: Learning Robust Visual Features without Supervision.” _TMLR_, 2024. https://arxiv.org/abs/2304.07193

[L52] X. Zhai et al. “Sigmoid Loss for Language Image Pre-Training (SigLIP).” _ICCV_, 2023. https://arxiv.org/abs/2303.15343

[L53] A. Radford et al. “Learning Transferable Visual Models From Natural Language Supervision (CLIP).” _ICML_, 2021. https://arxiv.org/abs/2103.00020

[L54] Y. LeCun. “A Path Towards Autonomous Machine Intelligence.” _OpenReview Technical Report_, 2022. https://openreview.net/forum?id=BZ5a1r-kVsf

[L55] M. Assran et al. “Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.” _ICCV_, 2023. https://arxiv.org/abs/2301.08243

[L56] A. Bardes et al. “Revisiting Feature Prediction for Learning Visual Representations from Video.” 2024. https://arxiv.org/abs/2404.08471

[L57] M. Assran et al. “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.” 2025. https://arxiv.org/abs/2506.09985

[L58] A. van den Oord, Y. Li, and O. Vinyals. “Representation Learning with Contrastive Predictive Coding.” 2018. https://arxiv.org/abs/1807.03748

[L59] J.-B. Grill et al. “Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.” _NeurIPS_, 2020. https://arxiv.org/abs/2006.07733

[L60] M. Caron et al. “Emerging Properties in Self-Supervised Vision Transformers.” _ICCV_, 2021. https://arxiv.org/abs/2104.14294

[L61] K. He et al. “Masked Autoencoders Are Scalable Vision Learners.” _CVPR_, 2022. https://arxiv.org/abs/2111.06377

[L62] A. Baevski et al. “data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.” _ICML_, 2022. https://arxiv.org/abs/2202.03555

[L63] K. Sohn et al. “Learning and Evaluating Representations for Deep One-Class Classification.” _ICLR_, 2021. https://arxiv.org/abs/2011.02578

[L64] C. Chen et al. “This Looks Like That: Deep Learning for Interpretable Image Recognition.” _NeurIPS_, 2019. https://arxiv.org/abs/1806.10574

[L65] P. W. Koh et al. “Concept Bottleneck Models.” _ICML_, 2020. https://arxiv.org/abs/2007.04612

[L66] W. Grathwohl et al. “Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One.” _ICLR_, 2020. https://arxiv.org/abs/1912.03263

[L67] M. Nauta, R. van Bree, and C. Seifert. “Neural Prototype Trees for Interpretable Fine-grained Image Recognition.” _CVPR_, 2021. https://arxiv.org/abs/2012.02046

[L68] M. E. Tipping and C. M. Bishop. “Mixtures of Probabilistic Principal Component Analyzers.” _Neural Computation_, 11(2), 1999. https://doi.org/10.1162/089976699300016728

[L69] Z. Ghahramani and G. E. Hinton. “The EM Algorithm for Mixtures of Factor Analyzers.” Technical Report CRG-TR-96-1, University of Toronto, 1996. https://www.cs.toronto.edu/~hinton/absps/tr96-1.html

[L70] S. Watanabe. “Karhunen-Loève Expansion and Factor Analysis: Theoretical Remarks and Applications.” _Transactions of the Fourth Prague Conference on Information Theory_, 1967.

[L71] J. Laaksonen and E. Oja. “Classification with Learning k-Nearest Neighbors.” _IEEE International Conference on Neural Networks_, 1996.

[L72] P. S. Bradley and O. L. Mangasarian. “k-Plane Clustering.” _Journal of Global Optimization_, 16, 2000. https://doi.org/10.1023/A:1008324625522

[L73] G. Hinton, O. Vinyals, and J. Dean. “Distilling the Knowledge in a Neural Network.” 2015. https://arxiv.org/abs/1503.02531

[L74] C. Buciluă, R. Caruana, and A. Niculescu-Mizil. “Model Compression.” _KDD_, 2006. https://doi.org/10.1145/1150402.1150464

[L75] M. W. Craven and J. W. Shavlik. “Extracting Tree-Structured Representations of Trained Networks.” _NeurIPS_, 1996.

[L76] N. Frosst and G. Hinton. “Distilling a Neural Network Into a Soft Decision Tree.” 2017. https://arxiv.org/abs/1711.09784

[L77] A. Banerjee, I. S. Dhillon, J. Ghosh, and S. Sra. “Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.” _JMLR_, 6, 2005. https://www.jmlr.org/papers/v6/banerjee05a.html

[L78] K. V. Mardia and P. E. Jupp. _Directional Statistics_. Wiley, 2000. https://doi.org/10.1002/9780470316979

[L79] S. Gidaris and N. Komodakis. “Dynamic Few-Shot Visual Learning without Forgetting.” _CVPR_, 2018. https://arxiv.org/abs/1804.09458

[L80] H. Qi, M. Brown, and D. G. Lowe. “Low-Shot Learning with Imprinted Weights.” _CVPR_, 2018. https://arxiv.org/abs/1712.07136

[L81] P. Mettes, E. van der Pol, and C. G. M. Snoek. “Hyperspherical Prototype Networks.” _NeurIPS_, 2019. https://arxiv.org/abs/1901.10514

[L82] P. T. Fletcher, C. Lu, S. M. Pizer, and S. Joshi. “Principal Geodesic Analysis for the Study of Nonlinear Statistics of Shape.” _IEEE Transactions on Medical Imaging_, 23(8), 2004. https://doi.org/10.1109/TMI.2004.831793

[L83] X. Pennec. “Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric Measurements.” _Journal of Mathematical Imaging and Vision_, 25, 2006. https://doi.org/10.1007/s10851-006-6228-4

[L84] S. Chen, C. F. N. Cowan, and P. M. Grant. “Orthogonal Least Squares Learning Algorithm for Radial Basis Function Networks.” _IEEE Transactions on Neural Networks_, 2(2), 1991. https://doi.org/10.1109/72.80202

[L85] P. Vincent and Y. Bengio. “Kernel Matching Pursuit.” _Machine Learning_, 48, 2002. https://doi.org/10.1023/A:1012450327387

[L86] S. G. Mallat and Z. Zhang. “Matching Pursuits with Time-Frequency Dictionaries.” _IEEE Transactions on Signal Processing_, 41(12), 1993. https://doi.org/10.1109/78.258082

[L87] J. Moody and C. J. Darken. “Fast Learning in Networks of Locally-Tuned Processing Units.” _Neural Computation_, 1(2), 1989. https://doi.org/10.1162/neco.1989.1.2.281

[L88] B. E. Boser, I. M. Guyon, and V. N. Vapnik. “A Training Algorithm for Optimal Margin Classifiers.” _COLT_, 1992. https://doi.org/10.1145/130385.130401

[L89] C. Cortes and V. Vapnik. “Support-Vector Networks.” _Machine Learning_, 20, 1995. https://doi.org/10.1007/BF00994018

[L90] V. Vovk, A. Gammerman, and G. Shafer. _Algorithmic Learning in a Random World_. Springer, 2005. https://doi.org/10.1007/b106715

[L91] H. Papadopoulos, K. Proedrou, V. Vovk, and A. Gammerman. “Inductive Confidence Machines for Regression.” _ECML_, 2002. https://doi.org/10.1007/3-540-36755-1_29

[L92] H. Jaeger. “The ‘Echo State’ Approach to Analysing and Training Recurrent Neural Networks.” GMD Report 148, 2001.

[L93] W. Maass, T. Natschläger, and H. Markram. “Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.” _Neural Computation_, 14(11), 2002. https://doi.org/10.1162/089976602760407955

[L94] O. Ledoit and M. Wolf. “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.” _Journal of Multivariate Analysis_, 88(2), 2004. https://doi.org/10.1016/S0047-259X(03)00096-4

[L95] P. J. Rousseeuw. “Multivariate Estimation with High Breakdown Point.” In _Mathematical Statistics and Applications, Volume B_, 1985.

[L96] P. J. Rousseeuw and K. Van Driessen. “A Fast Algorithm for the Minimum Covariance Determinant Estimator.” _Technometrics_, 41(3), 1999. https://doi.org/10.1080/00401706.1999.10485670

[L97] E. Levina and P. J. Bickel. “Maximum Likelihood Estimation of Intrinsic Dimension.” _NeurIPS_, 2004. https://proceedings.neurips.cc/paper/2004/hash/74934548253bcab8490ebd74afed7031-Abstract.html

[L98] D. C. Liu and J. Nocedal. “On the Limited Memory BFGS Method for Large Scale Optimization.” _Mathematical Programming_, 45, 1989. https://doi.org/10.1007/BF01589116

[L99] S. A. Dudani. “The Distance-Weighted k-Nearest-Neighbor Rule.” _IEEE Transactions on Systems, Man, and Cybernetics_, SMC-6(4), 1976. https://doi.org/10.1109/TSMC.1976.5408784

[L100] R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu. “A Limited Memory Algorithm for Bound Constrained Optimization.” _SIAM Journal on Scientific Computing_, 16(5), 1995. https://doi.org/10.1137/0916069

## Repository Sources

[R1] `src/sdf_engine.py`; `src/greedy_constructor.py`; `src/sdf_optimizer.py`; `src/probabilistic_engine.py`.

[R2] `experiments/common/moe_eval.py`; `experiments/common/score_readouts.py`; `experiments/tier6/eval_temporal_text_prediction.py`.

[R3] `analysis/MILESTONE_RESULTS.md`; `logs/results/tier4_csg_ablation_summary.json`; `logs/results/tier6_locked_window5_confirmation.json`; primitive-family and field-ablation artifacts under `logs/results/`.

[R4] `logs/results/e4_cifar_qualification.json`; `logs/results/e5_routing_qualification.json`; `logs/results/e6_transfer_qualification.json`.

[R5] `src/rejection_buffer.py`; `src/discovery_clustering.py`; `src/streaming_discovery.py`; `logs/results/tier4_real_feature_accumulated_groups.json`.

[R6] `src/adaptation_policy.py`; `src/model_editor.py`; `src/replay_constrained_fitter.py`; `logs/results/e9_transactional_adaptation.json`.

[R7] `src/runtime/`; E1-E3 artifacts listed in `analysis/MILESTONE_RESULTS.md`.

[R8] `src/runtime/model_bundle.py`; E3/E4/E6/E8 model registries under `logs/results/`.

[R9] `logs/results/e10_production_rehearsal.json`; `analysis/E10_RECOVERY_RUNBOOK.md`.

[R10] `logs/results/e11_artifact_index.json`; `logs/results/e11_public_study/`; `analysis/END_TO_END_TRAINING_AND_DEPLOYMENT_PLAN.md`.

[R11] `logs/results/v5/m19_frozen_space/m19_s1_evidence.json`; `data/v5/features/m19_s1/extraction_summary.json`.

[R12] `logs/results/tier4_primitive_family_ablation_dplus2_runs.jsonl`;
`logs/results/tier4_primitive_family_ablation_dplus2_summary.json`;
`logs/results/tier4_primitive_csg_ablation_dplus2_runs.jsonl`;
`logs/results/tier4_primitive_csg_ablation_dplus2_summary.json`;
`logs/results/tier4_probabilistic_field_ablation_dplus2_summary.json`;
`logs/results/tier4_hybrid_field_ablation_dplus2_summary.json`;
`logs/results/tier4_global_covariance_temperature_dplus2_summary.json`;
`logs/results/tier4_per_class_covariance_temperature_dplus2_summary.json`.

[R13] `logs/results/v5/m19_flowers102_s1/m19_flowers102_s1_evidence.json`;
`data/v5/features/m19_flowers102_s1/extraction_summary.json`.

[R14] `logs/results/v5/m19_native_dinov2_sphere/m19_native_dinov2_sphere_evidence.json`;
`data/v5/features/m19_native_dinov2_sphere/extraction_summary.json`.

[R15] `logs/results/v5/m19_native_dinov2_sphere_support_pilot/m19_native_dinov2_sphere_support_pilot_evidence.json`;
`data/v5/features/m19_native_dinov2_sphere_support_pilot/extraction_summary.json`.
