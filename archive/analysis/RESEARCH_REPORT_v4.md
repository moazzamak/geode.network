# GEODE Research Report v4

## Explicit Geometric Experts for Classification, Temporal Prediction, and Review-First Open-World Learning

**Literature cutoff:** 24 July 2026

**Repository evidence cutoff:** 26 July 2026

**Repository state:** milestones M0-M15.2b and E0-E6/E8-E11; E7 blocked on a second Ray node

**Scope:** methods, public-dataset evidence, synthetic diagnostics, and literature positioning

---

## Abstract

GEODE represents a class or output as a collection of oriented ellipsoids in a transformed feature space. Each primitive produces a normalized radial field; additive primitives are fused by a normalized soft minimum, and optional subtractive primitives implement explicit set difference. Models are built by a greedy multi-model RANSAC procedure, then may receive supervised field refinement and calibrated score readouts. The same model interface accepts image, point-cloud, and causal temporal representations [R1-R5].

This report makes four bounded findings. First, on a five-seed CIFAR-10 matched-feature study, calibrated GEODE reached 83.62% mean accuracy, while direct logistic regression reached 83.96% and an RBF SVM reached 84.21%; the experiment therefore does not establish a classification advantage over matched classical controls [R6]. Second, subtractive CSG and active repair produced no measurable aggregate CIFAR-10 accuracy gain [R6]. Third, on a locked WikiText-103 character prediction protocol, GEODE reached 30.36%, above a unigram control at 19.22% but below feature-logistic at 34.64%, a matched-data 5-gram at 44.50%, and a practical 5-gram at 47.61% [R7]. Fourth, a held-out-class CIFAR-10 stream study surfaced unknown events with 100% recall but achieved only 61.11% distinct-group recall and 0.059 adjusted Rand index; automatic model mutation consequently remained disabled [R8].

GEODE is not shown to be state of the art on any public benchmark in this repository. Its supported contribution is instead an experimentally audited combination of explicit geometric primitives, calibrated class scoring, review-first open-world bookkeeping, and conservative advancement gates. Negative results are part of that contribution: CSG did not improve aggregate public-data accuracy, text prediction remained behind simple sequence controls, OOD false-positive rates remained high, and synthetic routing proposals did not beat exhaustive inference [R6-R10].

---

## 1. Claim and Evidence Policy

This report separates three kinds of evidence:

1. **External literature claims** cite primary or archival sources as `[L#]`.
2. **Repository claims** cite source, protocol, or machine-readable artifacts as `[R#]`.
3. **Interpretations** are explicitly bounded to the cited protocol and are not presented as universal results.

A result is called **public-dataset evidence** only when the evaluated observations came from a named, publicly available dataset. Synthetic experiments are reported separately and are not used to claim public-dataset accuracy, robustness, or scalability. CIFAR-10 and CIFAR-100 originate from the tiny-images classification collection, SVHN contains cropped street-view digits, WikiText-103 is a long-form language-modeling corpus, MNIST is a handwritten-digit benchmark, and ModelNet is a collection of aligned 3D CAD models [L27-L31].

“State of the art” is task-specific. Closed-set image classification, OOD ranking, generalized category discovery, streaming novelty detection, temporal prediction, and sparse expert routing have different objectives and protocols. This report compares GEODE with representative frontier methods and matched local controls; it does not merge their benchmark numbers into a single ranking.

---

## 2. Research Questions

The completed milestone sequence evaluates six questions [R2,R3]:

- Can explicit unions of oriented ellipsoids form a usable classifier on learned and engineered representations?
- Does subtractive CSG improve held-out public-data performance?
- Can one geometry-and-calibration interface consume non-image, temporal, and point-cloud representations?
- Do geometric scores provide useful OOD or uncertainty signals?
- Can unsupported observations be accumulated into stable, reviewable unlabeled groups without assigning semantic labels automatically?
- Can exact or certified routing reduce inference cost without changing published predictions?

The experiments answer these questions asymmetrically. Representation flexibility is demonstrated across repository tiers, but competitive accuracy is not. Calibration is useful, but raw SDF magnitude is not a reliable OOD score in the tested image setting. Unknown events can be surfaced, but their semantic partition is weak. Candidate counts can grow sublinearly in synthetic routing tests, but measured latency does not improve [R2,R7-R10].

---

## 3. Methodology

### 3.1 Primitive geometry

For input $x\in\mathbb{R}^d$, an oriented ellipsoid has center $c$, semi-axis vector $a>0$, and orientation matrix $Q$. GEODE computes

$$
q(x)=\sum_{j=1}^{d}\frac{((x-c)Q)_j^2}{a_j^2},
\qquad
\phi(x)=\sqrt{q(x)}-1.
$$

The sign of $\phi$ identifies the represented interior, boundary, and exterior. This quantity is a normalized Mahalanobis-style radial field, not an exact Euclidean signed distance for a general anisotropic ellipsoid. The repository also implements the local first-order correction $\phi/\lVert\nabla\phi\rVert$, but documents that correction as approximate and not a conservative sphere-tracing bound [R1].

This representation is related to Gaussian and Mahalanobis classifiers because both use quadratic distance in feature space. Gaussian discriminant methods derive scores from class-conditional densities, including determinant and prior terms; GEODE instead fits bounded support primitives and composes their fields [L1,L2,R1].

### 3.2 Normalized soft-minimum composition

For $M$ additive fields, GEODE uses

$$
\Phi(x)=-\frac{1}{\alpha}\log\left(
\frac{1}{M}\sum_{m=1}^{M}e^{-\alpha\phi_m(x)}
\right).
$$

The $1/M$ normalization prevents duplicate coincident primitives from creating an artificial $\log(M)/\alpha$ offset. As $\alpha$ increases, the composition approaches a hard minimum while retaining a smooth attribution distribution at finite $\alpha$ [R1,R4].

A class may contain several composite experts. The same normalized soft minimum is applied at the primitive-to-expert and expert-to-class levels. Prediction selects the class with the lowest resulting field unless a fitted readout is requested [R4,R5].

### 3.3 Explicit subtraction

A negative-polarity primitive defines a hole. Within a composite expert, the implemented hard CSG rule combines the additive union with the complement of subtractive support. Candidate holes are fitted to exclusion points, checked against reserved hold-out observations, and accepted only when the carve objective passes its gate [R1,R3].

This differs from neural implicit CSG systems that learn a program or decomposition end to end. CSGNet predicts symbolic CSG programs from shape observations; superquadric and primitive-decomposition methods optimize part structure for reconstruction; DeepSDF learns a continuous neural field; Marching-Primitives extracts explicit primitives from learned or observed geometry [L5-L9]. GEODE uses fixed ellipsoid algebra as a classifier in an upstream representation space and does not infer a general symbolic shape program [R1-R5].

### 3.4 Greedy construction

GEODE construction is a greedy multi-model fitting process:

1. transform observations using a train-fitted representation;
2. seed candidate neighborhoods with k-nearest-neighbor structure;
3. estimate center, orientation, and radii with covariance and SVD fallbacks;
4. score support capture and contamination;
5. accept a candidate only if it passes configured gates;
6. remove or down-weight captured observations and repeat;
7. optionally fit held-out subtractive repairs [R3].

RANSAC introduced consensus fitting from random minimal subsets, while later multi-model methods such as Progressive-X jointly reason about several structures and their residual points [L3,L4]. GEODE borrows the robust consensus principle but uses class-conditioned ellipsoidal support, deterministic fallbacks, and explicit stopping and carve gates. It is not a new general-purpose multi-model estimator [R3].

### 3.5 Supervised refinement and readouts

For additive models, the repository can differentiate through both normalized soft-minimum levels and update ellipsoid parameters using cross-entropy. The optimizer rejects subtractive models because the hard set-difference path does not share the same smooth derivative [R4].

Raw class fields can be converted to probabilities by temperature, diagonal logistic, or multinomial readouts fitted only on calibration data. Temperature scaling is a standard post-hoc calibration method; GEODE’s additional readouts learn mappings from its class-field vector rather than modifying the upstream feature extractor [L16,R5].

### 3.6 Upstream representations

GEODE is representation-agnostic in the limited software sense that the geometric learner consumes numeric matrices with a common contract. Repository experiments use pretrained MobileNetV2 features for the main CIFAR studies, and also contain HOG, pooled RGB, ResNet-18, PCA, LDA, exact temporal context, deterministic reservoir state, and point-cloud paths [R2,R5]. MobileNetV2 uses inverted residuals and linear bottlenecks; ResNet uses residual connections; HOG aggregates local gradient orientation statistics [L24-L26].

This does not make GEODE an end-to-end representation learner. In the strongest image experiments, most semantic structure comes from a pretrained backbone and train-fitted dimensionality reduction. Reported GEODE accuracy therefore measures the complete frozen representation-plus-geometry protocol, not ellipsoids operating on raw pixels [R2,R6].

### 3.7 Temporal prediction

Tier 6 converts a causal character history into a fixed-dimensional vector, fits one geometric class model per next-character target, and calibrates the resulting field vector. All representation tuning uses forward-only training folds; the final test sequence is observational and does not select the representation [R2,R7].

The locked experiment uses an exact five-character window, PCA dimension 24, 50,000 training examples, 10,000 test examples, seed 13, no EM refinement, and no subtractive primitives. Its controls include a unigram predictor, logistic regression on the same temporal representation, a 5-gram restricted to matched data, and a practical 5-gram using the available training prefix [R7].

### 3.8 OOD scoring and calibration

The image OOD study compares minimum raw field, metric-corrected field, field energy, maximum calibrated probability, Mahalanobis distance, Gaussian-mixture negative log likelihood, and k-nearest-neighbor distance. Thresholds are selected on an OOD validation split and then reported on disjoint test observations [R9].

Maximum softmax probability is a longstanding OOD baseline. Mahalanobis feature scoring, energy scores, and deep nearest-neighbor methods are established alternatives [L12-L15]. GEODE’s study is a matched local comparison of these score families on one frozen representation and model; it is not a reproduction of each paper’s full training and tuning protocol [R9].

### 3.9 Review-first open-world grouping

Unsupported stream observations are buffered with stable observation identifiers. A frozen policy selects a flag fraction using proxy labels only, embeds flagged observations, clusters them with HDBSCAN, assigns stable non-semantic group identifiers, suppresses redundant reviews, and records delayed feedback. Group IDs are review handles rather than class names [R8,R11].

DBSCAN introduced density-connected clustering with noise; HDBSCAN extends density clustering across a hierarchy; FINCH derives clusters from first-neighbor chains; DenStream maintains density summaries in evolving streams; MINAS combines multiclass learning and novelty detection for data streams [L17-L21,L41]. GEODE does not claim a new clustering algorithm. Its distinctive repository behavior is the transaction boundary around an existing clusterer: grouping may request review, but no semantic class or model mutation is published until independent gates pass [R8,R11].

### 3.10 Routing and bounded growth

Authoritative inference evaluates every class field. M12 tests scalar support-bound pruning, round-batched and class-major schedules, certified nearest-centroid top-$k$ lookup, and calibration-only primitive compression. Exact variants must agree with exhaustive predictions; compressed variants must pass an independently confirmed lower-tail agreement gate [R10].

Sparse neural MoE systems usually learn token-to-expert assignment to reduce activated network computation. Switch Transformers use top-1 routing, Expert Choice reverses assignment so experts select tokens, and hash layers remove a learned router [L32-L34]. GEODE’s routing problem is different: it seeks to avoid evaluating explicit class fields while preserving an already-fitted model’s decision. No tested GEODE alternative was authorized for publication or inference [R10].

---

## 4. Literature Review and State of the Art

### 4.1 Geometric and implicit models

Quadratic discriminant and mixture discriminant analysis provide probabilistic quadratic decision surfaces; RBF SVMs provide kernel decision surfaces without an explicit bounded part decomposition [L1,L2,L23]. Neural implicit methods such as DeepSDF learn continuous shape fields, while CSGNet and primitive-decomposition work target compact, interpretable shape reconstruction [L5-L9]. These are the closest methodological lineages, but their objectives differ from calibrated classification in pretrained feature space.

GEODE’s nearest comparison is therefore not one universal SOTA method. For classification, matched logistic, Gaussian, centroid, GMM, linear-SVM, and RBF-SVM controls are more informative. For explicit shape representation, neural implicit and primitive-decomposition methods are the relevant conceptual comparison. For robust fitting, RANSAC and multi-model consensus are the relevant construction comparison [L1-L9,L23,R6].

### 4.2 OOD detection

Modern OOD work includes confidence baselines, class-conditional feature distance, energy scoring, and deep nearest neighbors [L12-L15]. The repository does not train specialized OOD representations, expose a broad benchmark suite, or compare against recent foundation-model OOD systems. Consequently, its CIFAR-100 and SVHN results characterize the current GEODE score interface only [R9].

### 4.3 Novel and generalized category discovery

UNO jointly transfers representation structure from labeled to unlabeled categories. Generalized Category Discovery (GCD) removes the assumption that every unlabeled observation belongs to a novel class, and ORCA studies open-world semi-supervised learning with simultaneous known and novel categories [L35-L37]. SimGCD supplies a strong parametric baseline, while later teacher-student and conditional self-labeling methods, including Flipped Classroom and OwMatch, continue the learned-representation and pseudo-labeling direction [L38-L40].

These methods optimize semantic partition or classification quality and normally evaluate known/novel accuracy under benchmark-specific matching. GEODE M11 instead optimizes event surfacing, review load, duplicate suppression, stable bookkeeping, and safe mutation gates. Its output is not directly comparable to GCD accuracy, and its low ARI shows that it should not be described as a competitive category-discovery method [R8,R11].

### 4.4 Sparse expert routing

The sparse-MoE frontier couples learned routing with model training and distributed execution [L32-L34]. GEODE’s post-hoc exact-routing experiments neither train a sparse gate nor evaluate a large language model. They address a narrower systems question: whether geometry-derived bounds can reduce evaluations while preserving the exhaustive result. The answer was negative over the tested synthetic range [R10].

### 4.5 Task-by-task comparison

| Task                      | Representative literature frontier                                      | GEODE difference                                                       | Repository conclusion                                                   |
| ------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Closed-set classification | deep representations; kernel and probabilistic controls [L1,L2,L22,L23] | explicit ellipsoid support over frozen features                        | no advantage over matched logistic/RBF controls [R6]                    |
| Implicit geometry         | neural fields, CSG programs, learned primitives [L5-L9]                 | fixed oriented ellipsoids used as class fields                         | explicit and editable, but no matched reconstruction-SOTA study [R1-R3] |
| OOD detection             | confidence, energy, Mahalanobis, deep kNN [L12-L15]                     | reuse calibrated class fields and matched feature controls             | maximum probability ranked best locally; FPR95 remained high [R9]       |
| Category discovery        | UNO, GCD, ORCA, SimGCD, teacher-student methods [L35-L40]               | review queue with stable non-semantic IDs and no automatic publication | events surfaced; semantic partition weak [R8,R11]                       |
| Temporal prediction       | sequence-specific count and neural models [L28]                         | fixed causal representation followed by geometric class models         | below feature-logistic and 5-gram controls [R7]                         |
| Sparse routing            | learned token routing and distributed sparse activation [L32-L34]       | post-hoc exact/certified class-field pruning                           | no measured latency break-even [R10]                                    |

No row supports a state-of-the-art claim. The defensible difference is architectural and procedural: GEODE exposes editable geometric parts and conservative lifecycle gates within one model interface [R1-R5,R8,R10].

---

## 5. Experimental Protocol

### 5.1 Leakage controls

The evaluated pipelines separate geometry fitting, carve acceptance, calibration, validation, and final testing where the task permits. Image transformations are fitted on training observations. Temporal folds preserve chronology, and test metrics do not drive representation selection, early stopping, or model mutation [R2,R3,R7].

The strongest matched CIFAR-10 comparison uses seeds 11, 23, 37, 53, and 71; common transformed features; identical split hashes; and the same training observations for all compared classifiers. Bootstrap or seed-level intervals are retained in the artifact [R6].

### 5.2 Metrics

Closed-set studies report accuracy, balanced accuracy, negative log likelihood (NLL), expected calibration error (ECE), Brier score, and top-5 accuracy. OOD studies report AUROC, area under precision-recall curves, and FPR95. Grouping studies report event recall, distinct-group recall, adjusted Rand index (ARI), purity, duplicate-review rate, and reviews per 1,000 observations [R6,R8,R9].

Accuracy and calibration do not answer the same question: a model may rank the correct class but assign poor probabilities. Likewise, OOD AUROC is threshold-independent, whereas FPR95 reports the false-positive rate required to retain 95% true-positive rate. The report therefore does not convert one metric into another or describe a high AUROC alone as deployment readiness [L16,R9].

---

## 6. Public-Dataset Results

### 6.1 CIFAR-10 matched classification

The five-seed study uses pretrained MobileNetV2 features followed by train-fitted PCA, LDA, and scaling. The table reports means from the frozen artifact [R6].

| Method                                    | Mean accuracy |   NLL |   ECE | Interpretation                                     |
| ----------------------------------------- | ------------: | ----: | ----: | -------------------------------------------------- |
| GEODE A0, diagonal readout                |       83.625% | 0.494 | 0.027 | primary calibrated geometry result                 |
| GEODE A0, multinomial readout             |       83.600% | 0.495 | 0.046 | no gain over diagonal                              |
| GEODE A0, raw fields                      |       83.663% | 1.509 | 0.594 | accuracy retained; probabilities poorly calibrated |
| Logistic regression on GEODE score vector |       84.025% | 0.465 | 0.021 | best calibrated local result by NLL/ECE            |
| Direct feature logistic regression        |       83.963% | 0.456 | 0.019 | matched non-geometric control                      |
| RBF SVM                                   |       84.213% | 0.697 | 0.058 | highest mean accuracy                              |
| Shrinkage Gaussian                        |       83.100% | 0.546 | 0.048 | probabilistic quadratic control                    |
| Matched GMM                               |       80.075% | 0.705 | 0.083 | mixture-density control                            |
| Nearest centroid                          |       82.988% | 0.607 | 0.131 | prototype control                                  |

The differences are small, and the artifact does not establish that GEODE is better than direct logistic regression or RBF SVM. It does show that explicit geometric scores can retain much of the predictive information in the transformed features and can be calibrated to reasonable NLL and ECE [R6].

### 6.2 CIFAR-10 CSG ablation

Across the same five seeds, adding subtractive primitives changed mean accuracy by approximately hundredths of a percentage point, and paired confidence intervals included zero. Active repair added no further candidates and reproduced the subtractive result [R6].

| Geometry               | Diagonal accuracy | Multinomial accuracy | Mean subtractive primitives |
| ---------------------- | ----------------: | -------------------: | --------------------------: |
| A0 additive only       |           83.625% |              83.600% |                         0.0 |
| A1 subtraction enabled |           83.638% |              83.613% |                         0.4 |
| A2 active repair       |           83.638% |              83.613% |                         0.4 |

This public-data ablation falsifies the earlier broad hypothesis that explicit excision necessarily improves generalization. Subtraction remains an opt-in representation capability, not a default accuracy improvement [R6].

### 6.3 CIFAR-100 superclass classification

The Tier 5 task predicts 20 CIFAR-100 superclasses. The latest verification pipeline reports 64.37% test accuracy; a separate fitter confirmation at seed 42 reports 65.50%. The protocols differ, so the values form a documented range rather than independent repetitions of one estimator [R2,R12].

Only one seed is present in the fitter confirmation, and there is no Tier 5 equivalent of the full five-seed Tier 4 matched matrix. This experiment supports pipeline operation on a larger class hierarchy, not a competitive CIFAR-100 or statistical superiority claim [R2,R12].

### 6.4 WikiText-103 character prediction

The locked 50,000/10,000 protocol gives the following results [R7]:

| Method                   | Top-1 accuracy |
| ------------------------ | -------------: |
| GEODE, calibrated        |         30.36% |
| Feature-logistic control |         34.64% |
| Matched-data 5-gram      |         44.50% |
| Best-practical 5-gram    |         47.61% |
| Unigram                  |         19.22% |

GEODE trails the same-feature logistic control by 4.28 percentage points, the matched-data 5-gram by 14.14 points, and the practical 5-gram by 17.25 points. It exceeds the unigram by 11.14 points. Forward-validation accuracy is $27.57\%\pm0.98$ points, calibrated top-5 accuracy is 64.59%, perplexity is 14.25, and raw uncalibrated top-1 accuracy is 10.15% [R7].

The result demonstrates interface flexibility: the same class-field and calibration machinery consumes a causal temporal representation. It does not demonstrate competitive sequence modeling. The strongest simple controls exploit the task structure more effectively [R7].

A separate locked ablation begins at 27.05% additive test accuracy. Two additive refinement variants both reach 27.90%, while subtractive variants do not establish a durable advantage. Those values belong to a different frozen setup and must not be combined with the 30.36% confirmation as repetitions [R2,R13].

### 6.5 OOD detection on CIFAR-100 and SVHN

The frozen ID model reaches 82.28% CIFAR-10 accuracy, 82.57% balanced accuracy, and 0.544 NLL. The same model and score families are tested against CIFAR-100 and SVHN [R9].

| OOD family | Score                          | Test AUROC | Test FPR95 |
| ---------- | ------------------------------ | ---------: | ---------: |
| CIFAR-100  | maximum calibrated probability |      0.786 |      0.579 |
| CIFAR-100  | minimum raw SDF                |      0.552 |      0.899 |
| CIFAR-100  | metric-corrected SDF           |      0.555 |      0.916 |
| CIFAR-100  | Mahalanobis                    |      0.232 |      0.992 |
| CIFAR-100  | kNN distance                   |      0.475 |      0.946 |
| SVHN       | maximum calibrated probability |      0.831 |      0.431 |
| SVHN       | minimum raw SDF                |      0.337 |      0.937 |
| SVHN       | metric-corrected SDF           |      0.343 |      0.955 |
| SVHN       | Mahalanobis                    |      0.082 |      0.998 |
| SVHN       | kNN distance                   |      0.243 |      0.966 |

Maximum calibrated probability is the strongest tested score on both OOD families. Raw and corrected geometric distances do not provide useful ranking in this setup, and even the best FPR95 values, 0.579 and 0.431, are too high to support a deployment claim. These results are specific to this representation, calibration, and split [R9].

### 6.6 Held-out CIFAR-10 stream grouping

The frozen review policy uses L2-normalized MobileNetV2 representations, a 30% flag fraction, and HDBSCAN minimum cluster size and minimum samples of three. On the final held-out-class cells, it reports [R8]:

| Metric                         |   Final mean |
| ------------------------------ | -----------: |
| Event recall                   |      100.00% |
| Distinct-group recall          |       61.11% |
| Unknown-group ARI              |        0.059 |
| Mean cluster purity            |       79.83% |
| Useful-review precision        |       74.07% |
| Duplicate-review rate          |       66.76% |
| Reviews per 1,000 observations |         7.22 |
| Mean time to review            | 1.11 windows |

The system reliably surfaces that unsupported events occurred, but it does not reliably recover their semantic partition. High event recall coexists with low ARI and a high duplicate-review rate. The correct operational interpretation is “review queue generation,” not “automatic class discovery” [R8,R11].

No semantic labels, new classes, or adapted models were published from this experiment. Delayed labels were used for evaluation and policy auditing, not as permission for automatic mutation [R8,R11].

### 6.7 Other public-data tiers

The repository includes MNIST manifold and ModelNet-derived point-cloud evaluations, but the completed evidence ledger does not provide the same five-seed matched-control depth as Tier 4. These tiers support software-path coverage and representation breadth. They are not used here to claim benchmark superiority [R2,R3,L29-L31].

### 6.8 End-to-end qualification

The five-seed E4 CIFAR-100 qualification reached 65.26% mean balanced accuracy,
versus 67.33% for matched logistic regression and 68.13% for the RBF SVM. Its
predeclared five-point non-inferiority gate passed, but the controls remained
better. Mean near/far OOD AUROC was 0.642/0.805 and FPR95 was 0.808/0.608,
preserving the report's weak-near-OOD conclusion [R14,R15].

E6 demonstrated parent-linked CIFAR-100-to-CIFAR-10 transfer with 81.96%
balanced accuracy and zero source forgetting, while again trailing linear and
supervised-adapter controls. E8 packaged WikiText-103 and verified ModelNet40
models under one bundle contract with exact replay. E9 required linked delayed
confirmation before publication and restored the exact parent after rollback.
E10 restored a failed canary and an interrupted promotion in 0.005916 and
0.002797 seconds under a frozen one-second RTO and zero-request RPO [R14,R15].

E7 is not complete. DomainNet is pinned and hash-verified, and Ray runs under
Python 3.12. A 192-image local-small episode replayed exactly across three
logical container nodes, and injected worker-process loss recovered. No
physical multi-host training or physical node-loss recovery claim is made
[R14,R15].

---

## 7. Synthetic Engineering Evidence

Synthetic tests answer mechanism and systems questions under controlled geometry; they do not establish public-dataset performance [R2,R10].

M12 measures approximately linear exhaustive latency growth with class-count slope 1.003 and primitive-count slope 1.008. Exact support-bound routing reduces candidate-count growth to exponent 0.417 and preserves 100% decision agreement, but its overhead prevents latency break-even. Scalar routing reaches only 0.099 times exhaustive throughput, certified top-$k$ reaches 0.564 times, and class-major scheduling reaches 0.870 times. Primitive compression reaches 98.93% confirmed agreement, below the frozen 99.0% gate [R10].

Accordingly, exhaustive exact class-field evaluation remains authoritative. No routing or compression policy advanced to real-feature deployment [R10].

Synthetic geometry audits also validate implementation semantics, deterministic fitting, numerical safeguards, transaction rollback, and edit operations. Those tests support code correctness within their fixtures; they do not substitute for public-data generalization evidence [R2,R3].

---

## 8. What Is Supported

The following claims are supported by current repository evidence:

- GEODE is an explicit oriented-ellipsoid class model with normalized soft-minimum composition and optional hard subtraction [R1,R4].
- The common interface operates on image and causal temporal representations, with additional repository paths for point clouds and engineered features [R2,R5,R7].
- On the frozen CIFAR-10 feature protocol, GEODE is close to but does not beat the strongest matched classical controls [R6].
- Subtractive CSG does not improve aggregate CIFAR-10 accuracy in the completed five-seed ablation [R6].
- Calibration materially improves the usability of raw class fields on Tier 6, but GEODE remains below logistic and 5-gram text controls [R7].
- Maximum calibrated probability is a better local OOD ranker than the tested raw SDF, Mahalanobis, and kNN scores, while FPR95 remains too high for deployment [R9].
- Review-first grouping detects held-out-class events more reliably than it separates their semantic groups [R8].
- Tested exact/certified routing and compression policies do not improve latency enough to replace exhaustive inference [R10].

---

## 9. What Is Not Supported

The current evidence does not support these statements:

- GEODE is state of the art on CIFAR, WikiText, OOD detection, category discovery, or sparse routing [R6-R10].
- Ellipsoidal CSG improves public-data accuracy in general [R6].
- The Tier 6 model is a competitive language model [R7].
- Raw geometric distance is a reliable OOD score for pretrained image features [R9].
- Review groups correspond reliably to semantic classes [R8].
- Automatic open-world mutation is safe or beneficial [R8,R11].
- Sublinear candidate counts imply lower wall-clock latency [R10].
- Synthetic routing experiments establish public-dataset scalability [R10].

A search of the cited literature did not identify the exact combination of normalized ellipsoid fields, explicit review transactions, and the repository’s advancement gates. That absence is not proof of novelty. A formal novelty claim would require a systematic review broader than the sources in this report.

---

## 10. Limitations and Threats to Validity

**Representation dependence.** The strongest image results rely on pretrained MobileNetV2 features. They do not isolate what ellipsoid construction would learn from raw observations or from an end-to-end jointly trained representation [L24,R6].

**Benchmark coverage.** The statistically strongest comparison is one five-seed CIFAR-10 protocol. CIFAR-100 confirmation, OOD, and WikiText confirmation are narrower and mostly single-seed. Broad claims across domains would require repeated matched studies on additional public datasets [R6,R7,R9,R12].

**Protocol non-equivalence.** GEODE’s review-first stream objective differs from GCD and open-world semi-supervised learning benchmarks. Event recall and review load cannot be presented as semantic discovery accuracy [L35-L40,R8].

**Class imbalance.** The WikiText character vocabulary has many low-count classes; the locked artifact reports that most classes are below recommended geometric sample counts. This limits robust ellipsoid estimation and is part of the reason Tier 6 is a flexibility test rather than a competitive result [R7].

**Calibration dependence.** Raw field accuracy and calibrated probability quality diverge sharply in several experiments. Published inference behavior therefore depends on a disjoint, representative calibration split [R6,R7].

**Systems scale.** M12 evaluates controlled synthetic scaling and local hardware behavior. It does not cover distributed execution, accelerator kernels, or the training-time routing regime studied by sparse neural MoEs [L32-L34,R10].

**Literature cutoff.** The review includes verified sources available through 24 July 2026. Fast-moving 2025-2026 preprints are used only to characterize research direction when their identity and task are verified; no unreviewed preprint performance number is used as a benchmark claim.

---

## 11. Reproducibility

The legacy training-and-evaluation entry point is `verify_pipeline.py`.
E11 adds a separate artifact-only reproduction command:

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.e2e.generate_e11_public_study
```

The command reads no training dataset and fits no model. It verifies 30 frozen
configuration, acquisition, and result files against
`logs/results/e11_artifact_index.json`, then regenerates principal-result and
cost tables plus classification and recovery plots. Repeated generation is
byte-identical. Any source-byte change fails against the SHA-256 lock [R14].

Public-data loaders and shared evaluation utilities are under
`experiments/common/`; tier-specific experiments are under
`experiments/tier1/` through `experiments/tier6/`; model code is under `src/`;
and machine-readable outcomes are under `logs/results/` [R2-R5].

The report’s principal numerical claims can be reproduced or inspected from these frozen artifacts:

- CIFAR-10 matched baselines and CSG: [R6]
- WikiText-103 locked confirmation: [R7]
- Held-out CIFAR-10 grouping: [R8]
- CIFAR-100/SVHN OOD families: [R9]
- M12 routing and compression audit: [R10]
- CIFAR-100 fitter confirmation: [R12]
- WikiText refinement ablation: [R13]

Historical figures in earlier reports and README sections are superseded where they conflict with these artifacts. In particular, the locked Tier 6 result is 30.36%, not the earlier 23.35% pipeline figure [R2,R7].

---

## 12. Recommended Next Experiments

1. **Pre-register a multi-dataset matched study.** Repeat the five-seed Tier 4 design on CIFAR-100 superclasses and at least one non-image public dataset, retaining identical representations and calibration budgets across GEODE, logistic, RBF, GMM, and centroid controls [R6,R12].
2. **Treat subtraction as a targeted intervention.** Construct public-data protocols with independently verified holes or exclusion structure; do not enable CSG globally without a validation gain [R6].
3. **Improve Tier 6 representation before geometry.** Compare causal convolutional, recurrent, and transformer representations under a frozen downstream budget; retain n-gram and same-feature linear controls [R7].
4. **Evaluate open-world grouping on standard GCD splits.** Add semantic clustering metrics and benchmark methods while preserving GEODE’s review-only transaction policy [L35-L40,R8].
5. **Broaden OOD protocols.** Use multiple ID datasets, near/far OOD families, repeated seeds, and dedicated modern OOD baselines; retain FPR95 as a blocking metric [L12-L15,R9].
6. **Profile routing on real fitted models.** Reopen M12 only after the class/primitive regime exceeds a preregistered break-even estimate; keep exhaustive inference as the oracle [R10].

---

## 13. Conclusion

GEODE is best understood as an explicit geometric learning and lifecycle-control research system, not as a new universal SOTA learner. Its ellipsoid fields are inspectable and editable, its construction is robustly gated, its probabilities are separately calibrated, and its open-world path can stop at review rather than silently changing a published model [R1-R5,R8,R11].

The public evidence is mixed and informative. CIFAR-10 accuracy is close to matched classical controls but not better; subtractive CSG has no aggregate benefit; WikiText prediction is behind linear and n-gram controls; OOD ranking is usable only in a limited relative sense; and unknown-event surfacing is substantially stronger than semantic group recovery [R6-R9]. Synthetic scaling work likewise closes with a negative result: exhaustive exact inference remains the only authorized path [R10].

The end-to-end study adds recoverable training stages, immutable bundles,
shadow-only candidate evaluation, confirmation-gated adaptation, and local
promotion recovery. These are lifecycle results rather than evidence of a new
predictive advantage. Distributed DomainNet qualification remains blocked and
is reported as such [R14,R15].

These outcomes narrow the credible research claim. GEODE’s current value lies in a unified, explicit representation and unusually conservative empirical workflow. Demonstrating broader scientific advantage now requires repeated public-dataset studies in which the geometric representation, not an unmatched feature or protocol, produces a statistically defensible benefit.

---

# References

## Literature

[L1] T. Hastie and R. Tibshirani. “Discriminant Analysis by Gaussian Mixtures.” _Journal of the Royal Statistical Society: Series B_, 58(1), 1996. https://doi.org/10.1111/j.2517-6161.1996.tb02073.x

[L2] R. O. Duda, P. E. Hart, and D. G. Stork. _Pattern Classification_, 2nd ed. Wiley, 2001.

[L3] M. A. Fischler and R. C. Bolles. “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography.” _Communications of the ACM_, 24(6), 1981. https://doi.org/10.1145/358669.358692

[L4] D. Barath and J. Matas. “Progressive-X: Efficient, Anytime, Multi-Model Fitting Algorithm.” _ICCV_, 2019. https://arxiv.org/abs/1906.02290

[L5] J. J. Park et al. “DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation.” _CVPR_, 2019. https://arxiv.org/abs/1901.05103

[L6] G. Sharma, R. Goyal, D. Liu, E. Kalogerakis, and S. Maji. “CSGNet: Neural Shape Parser for Constructive Solid Geometry.” _CVPR_, 2018. https://arxiv.org/abs/1712.08290

[L7] D. Paschalidou, A. O. Ulusoy, and A. Geiger. “Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids.” _CVPR_, 2019. https://arxiv.org/abs/1904.09970

[L8] S. Tulsiani, H. Su, L. J. Guibas, A. A. Efros, and J. Malik. “Learning Shape Abstractions by Assembling Volumetric Primitives.” _CVPR_, 2017. https://arxiv.org/abs/1612.00404

[L9] M. Liu et al. “Marching-Primitives: Shape Abstraction from Signed Distance Function.” _CVPR_, 2023. https://arxiv.org/abs/2303.13190

[L12] D. Hendrycks and K. Gimpel. “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.” _ICLR_, 2017. https://arxiv.org/abs/1610.02136

[L13] K. Lee et al. “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.” _NeurIPS_, 2018. https://arxiv.org/abs/1807.03888

[L14] W. Liu et al. “Energy-based Out-of-distribution Detection.” _NeurIPS_, 2020. https://arxiv.org/abs/2010.03759

[L15] Y. Sun et al. “Out-of-Distribution Detection with Deep Nearest Neighbors.” _ICML_, 2022. https://arxiv.org/abs/2204.06507

[L16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. “On Calibration of Modern Neural Networks.” _ICML_, 2017. https://arxiv.org/abs/1706.04599

[L17] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu. “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.” _KDD_, 1996. https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf

[L18] R. J. G. B. Campello, D. Moulavi, and J. Sander. “Density-Based Clustering Based on Hierarchical Density Estimates.” _PAKDD_, 2013. https://doi.org/10.1007/978-3-642-37456-2_14

[L19] S. Sarfraz, V. Sharma, and R. Stiefelhagen. “Efficient Parameter-free Clustering Using First Neighbor Relations.” _CVPR_, 2019. https://arxiv.org/abs/1902.11266

[L20] F. Cao, M. Ester, W. Qian, and A. Zhou. “Density-Based Clustering over an Evolving Data Stream with Noise.” _SDM_, 2006. https://doi.org/10.1137/1.9781611972764.29

[L21] E. J. Spinosa, A. C. P. L. F. de Carvalho, and J. Gama. “Novelty Detection with Application to Data Streams.” _Intelligent Data Analysis_, 13(3), 2009. https://doi.org/10.3233/IDA-2009-0373

[L22] A. Krizhevsky, I. Sutskever, and G. E. Hinton. “ImageNet Classification with Deep Convolutional Neural Networks.” _NeurIPS_, 2012. https://doi.org/10.1145/3065386

[L23] C. Cortes and V. Vapnik. “Support-Vector Networks.” _Machine Learning_, 20, 1995. https://doi.org/10.1007/BF00994018

[L24] M. Sandler et al. “MobileNetV2: Inverted Residuals and Linear Bottlenecks.” _CVPR_, 2018. https://arxiv.org/abs/1801.04381

[L25] K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition.” _CVPR_, 2016. https://arxiv.org/abs/1512.03385

[L26] N. Dalal and B. Triggs. “Histograms of Oriented Gradients for Human Detection.” _CVPR_, 2005. https://doi.org/10.1109/CVPR.2005.177

[L27] A. Krizhevsky. “Learning Multiple Layers of Features from Tiny Images.” Technical report, University of Toronto, 2009. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

[L28] S. Merity, C. Xiong, J. Bradbury, and R. Socher. “Pointer Sentinel Mixture Models.” _ICLR_, 2017. Introduces WikiText-2 and WikiText-103. https://arxiv.org/abs/1609.07843

[L29] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. “Gradient-Based Learning Applied to Document Recognition.” _Proceedings of the IEEE_, 86(11), 1998. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

[L30] Y. Netzer et al. “Reading Digits in Natural Images with Unsupervised Feature Learning.” _NeurIPS Workshop_, 2011. http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

[L31] Z. Wu et al. “3D ShapeNets: A Deep Representation for Volumetric Shapes.” _CVPR_, 2015. https://arxiv.org/abs/1406.5670

[L32] W. Fedus, B. Zoph, and N. Shazeer. “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.” _JMLR_, 23, 2022. https://arxiv.org/abs/2101.03961

[L33] Y. Zhou et al. “Mixture-of-Experts with Expert Choice Routing.” _NeurIPS_, 2022. https://arxiv.org/abs/2202.09368

[L34] S. Roller et al. “Hash Layers for Large Sparse Models.” _NeurIPS_, 2021. https://arxiv.org/abs/2106.04426

[L35] E. Fini et al. “A Unified Objective for Novel Class Discovery.” _ICCV_, 2021. https://arxiv.org/abs/2108.08536

[L36] S. Vaze et al. “Generalized Category Discovery.” _CVPR_, 2022. https://arxiv.org/abs/2201.02609

[L37] K. Cao, M. Brbic, and J. Leskovec. “Open-World Semi-Supervised Learning.” _ICLR_, 2022. https://arxiv.org/abs/2102.03526

[L38] X. Wen, B. Zhao, and X. Qi. “Parametric Classification for Generalized Category Discovery: A Baseline Study.” _ICCV_, 2023. https://arxiv.org/abs/2211.11727

[L39] H. Lin et al. “Flipped Classroom: Aligning Teacher Attention with Student in Generalized Category Discovery.” _NeurIPS_, 2024. https://arxiv.org/abs/2409.19659

[L40] S. Niu, L. Lin, J. Huang, and C. Wang. “OwMatch: Conditional Self-Labeling with Consistency for Open-World Semi-Supervised Learning.” _NeurIPS_, 2024. https://arxiv.org/abs/2411.01833

[L41] E. R. de Faria, A. C. P. L. F. de Carvalho, and J. Gama. “MINAS: Multiclass Learning Algorithm for Novelty Detection in Data Streams.” _Data Mining and Knowledge Discovery_, 30(3), 2016. https://doi.org/10.1007/s10618-015-0433-y

## Repository Sources and Evidence

[R1] `src/sdf_engine.py`; `src/greedy_constructor.py`. Primitive field, normalized soft-minimum, subtraction, and support-bound implementation.

[R2] `analysis/MILESTONE_RESULTS.md`. Frozen milestone ledger through M12.7, including reporting rules and superseded-result notes.

[R3] `analysis/RESEARCH_IMPLEMENTATION_PLAN.md`. Experimental gates, split policy, and milestone decision rules.

[R4] `src/sdf_optimizer.py`. Additive supervised field refinement and normalized-softmin gradients.

[R5] `src/model_network.py`; `src/inference_engine.py`; `experiments/common/moe_eval.py`; `experiments/common/dataset_utils.py`. Model interface, inference, calibration, evaluation, and data utilities.

[R6] `logs/results/tier4_csg_ablation_summary.json`. Five-seed CIFAR-10 matched baselines and A0/A1/A2 CSG ablation.

[R7] `logs/results/tier6_locked_window5_confirmation.json`. Locked 50,000/10,000 WikiText-103 confirmation, seed 13.

[R8] `logs/results/tier4_real_feature_accumulated_groups.json`. Final held-out-class CIFAR-10 review-first grouping evaluation.

[R9] `logs/results/tier4_real_ood_families_smoke.json`. Frozen CIFAR-10 ID, CIFAR-100 OOD, and SVHN OOD score comparison.

[R10] `logs/results/tier5_m12_advancement_audit.json`; `logs/results/tier5_confirmed_primitive_compression.json`. Synthetic routing and compression closure audit.

[R11] `src/rejection_buffer.py`; `src/discovery_clustering.py`; `src/streaming_discovery.py`; `src/adaptation_policy.py`; `src/model_editor.py`. Unsupported-observation records, review groups, streaming memory, adaptation gates, and model-edit transactions.

[R12] `logs/results/tier5_fitter_confirmation_runs.jsonl`; `logs/results/2026-07-25 - verify pipeline benchmark (1).txt`. CIFAR-100 superclass confirmation and latest pipeline output.

[R13] `logs/results/tier6_refinement_ablation_locked.json`. Locked WikiText-103 refinement and subtraction ablation.

[R14] `logs/results/e11_artifact_index.json`; `logs/results/e11_public_study/`. SHA-256-locked E0-E10 evidence index and artifact-only E11 publication outputs.

[R15] `analysis/END_TO_END_TRAINING_AND_DEPLOYMENT_PLAN.md`; `analysis/MILESTONE_RESULTS.md`. End-to-end protocols, gate status, bundle identities, and bounded milestone conclusions.
