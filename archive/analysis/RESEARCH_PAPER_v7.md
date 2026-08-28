> **Historical v7 manuscript.** The current v8 arXiv source is
> [`FINAL_RESEARCH_PAPER.tex`](FINAL_RESEARCH_PAPER.tex); build instructions are
> in [`BUILD_PAPER.md`](BUILD_PAPER.md).

# When Strong Open-World Stages Do Not Compose

## A Preregistered Study of Rejection, Discovery, Human Confirmation, Transactional Adaptation, and Model Routing

**Anonymous Authors**  
*Author names, affiliations, and corresponding-author details should be inserted
after venue selection or removal of anonymous-review requirements.*

---

## Abstract

Open-world recognition systems must do more than reject unfamiliar inputs.
They must accumulate rejected evidence, identify persistent candidate groups,
request semantic supervision, update a model safely, route future inputs, and
remain auditable and reversible. Prior work has studied each stage, but evidence
that independently successful stages compose into a useful end-to-end system is
limited. We evaluate this composition under a preregistered, review-gated
protocol using frozen 384-dimensional DINOv2 features on a CIFAR-10
leave-two-class-out proxy. We compare six rejection heads, five discovery
policies, four confirmation-gated update strategies, and four empirical routing
profiles across three frozen seeds.

A rank-32 class-conditional Gaussian was the only rejection head to pass all
registered gates, reaching 92.08% known coverage, 86.17% unknown recall, 0.9561
AUROC, and 98.67% precision at the fixed review budget. HDBSCAN and FINCH then
formed persistent, replay-stable review objects; HDBSCAN reached 100%
distinct-group recall and review precision in the stage-wise evaluation. In
isolation, a human-confirmed rank-16 affine component improved held-out
new-class success by 43.33 percentage points while preserving exact rollback.
No empirical router met the authoritative safety thresholds, although routing
profiles reduced exact-model evaluations by approximately 50% in shadow mode.

Most importantly, the successful stages did not compose. In the integrated
factorial, HDBSCAN and FINCH integrated 0 of 3 confirmable classes. A
reject-everything control integrated all 3 only by reviewing every rejection,
and remaining-unknown recall fell 6.83 points relative to the rejection-stage
baseline. All publication, confirmation, replay, graph, fallback, and rollback
contracts passed. We therefore report a stage-wise lifecycle qualification, not
a closed operational loop. The central result is negative but actionable:
stage-wise metrics can substantially overestimate end-to-end open-world utility,
because clustering optimized for review purity may discard the support diversity
needed for adaptation.

**Keywords:** open-world recognition, open-set recognition, novel-class
discovery, human-in-the-loop learning, continual learning, model routing,
rollback, negative results

---

## 1. Introduction

A deployed classifier eventually encounters inputs outside its original class
inventory. The classical open-world recognition formulation requires a system
to recognize known classes, reject unknowns, and incrementally add newly
supervised classes [1]. In practice, however, a rejected prediction is only the
beginning of a longer process. Rejections must be buffered, recurring structure
must be distinguished from corruption, candidate groups must be reviewed,
semantic labels must be supplied by an authorized actor, model updates must be
validated, and every publication must be reversible.

The literature contains strong mechanisms for individual parts of this process.
OpenMax and the Extreme Value Machine (EVM) calibrate rejection [2,3].
Data-stream systems such as ECSMiner, SAND, ECHO, MINAS, and SENCForest buffer
and organize emerging classes [4--8]. Generalized and continual category
discovery methods identify novel groups in modern representation spaces
[9--13]. Expert Gate and recent modular routing methods select among specialized
models [14--17]. Yet stage-wise success does not establish that these mechanisms
work together under one safety boundary.

This paper asks a deliberately operational question:

> Can calibrated rejection, persistent discovery, human confirmation,
> transactional class adaptation, empirical cross-model routing, and exact
> rollback compose into a review-efficient open-world loop?

We study this question in GEODE, an experimental framework for explicit
geometric models and auditable model lifecycles. Earlier GEODE experiments
found that an explicit weighted affine head reached 91.73% balanced accuracy in
the same frozen DINOv2 space, compared with 96.77% for an RBF SVM. We therefore
do **not** assume that signed-distance geometry is necessary or competitively
optimal. Instead, the acceptance head is substitutable, and geometric support
is treated as an ablation.

The study makes four contributions:

1. **A preregistered stage-to-system protocol.** We separate rejection,
   discovery, confirmation, adaptation, routing, and integrated-loop gates while
   preserving immutable parent artifacts and exact rollback.
2. **A matched stage-wise comparison.** Six rejection heads, five discovery
   policies, four adaptation strategies, and four routing profiles are evaluated
   on identical frozen features and seeds.
3. **A transactional safety boundary.** Unlabeled groups cannot publish semantic
   classes; only linked confirmation events may authorize updates, and rollback
   must restore the exact parent bundle and predictions.
4. **A composition failure result.** Several stages pass independently, but the
   end-to-end loop fails. We show that review purity and adaptation usefulness
   are different objectives and can conflict.

This is not a claim that the detect-buffer-cluster-label-update loop is novel.
Our prior-art audit found no verified system combining all registered stages,
but that search cannot establish universal absence. The contribution is the
controlled composition study and its negative result.

---

## 2. Related Work

### 2.1 Open-set and open-world recognition

Bendale and Boult formalized open-world recognition as known-class recognition
with explicit rejection and incremental class addition [1]. OpenMax calibrates
deep activation vectors using extreme-value theory [2], while EVM models
class-conditional margin tails and supports incremental updates [3]. Frozen
feature-space alternatives include maximum posterior probability [18],
Mahalanobis or class-conditional density scores [19,20], deep nearest neighbors
[21], and distance-aware discriminative models such as DUQ and SNGP [22,23].
Our rejection bakeoff compares posterior, kNN, low-rank Gaussian, EVM-style
margin-tail, weighted affine support, and RBF evidence under one representation
and threshold protocol.

### 2.2 Streaming novel-class detection

Stream-mining systems operationalized reject-buffer-group-update loops before
the modern deep-representation era. ECSMiner detects novel classes under concept
drift [4]; SAND and ECHO address limited labels and evolving concepts [5,6];
MINAS maintains micro-clusters and recurring novelty [7]; and SENCForest targets
emerging new classes [8]. These systems are close to the operational structure
studied here, but do not jointly evaluate modern frozen features, explicit human
publication gates, cross-model routing, and exact transactional rollback.

### 2.3 Novel and generalized category discovery

Generalized category discovery (GCD) clusters unlabeled data containing both
known and unknown categories, often over strong self-supervised features [9].
ORCA jointly addresses seen and novel classes in open-world semi-supervised
learning [10]. Grow and Merge, IGCD, and continual category-discovery methods
extend discovery across time [11--13]. We use frozen-feature clustering controls
rather than representation-training methods because the experiment holds the
DINOv2 representation fixed.

### 2.4 Clustering rejected evidence

DBSCAN introduced density-based clustering with noise [24], HDBSCAN generalized
this idea through hierarchical density estimates [25], and FINCH derives
parameter-light partitions from first-neighbor relations [26]. We compare
streaming micro-clusters, HDBSCAN, FINCH, silhouette-selected k-means, and no
grouping. Cluster identifiers are persistent review identifiers, not semantic
labels.

### 2.5 Expert routing and modular models

Expert Gate routes inputs among task experts using autoencoder reconstruction
error [14]. More recent modular systems compose or route among independently
trained experts and adapters, including DEMix, Branch-Train-Merge, LoraHub, and
post-hoc adapter routers [15--17,27]. Our protocol separates model compatibility
from empirical support: an input/output fingerprint determines whether a model
may be called, while a data-derived routing profile only proposes a shortlist.
Low-confidence or stale profiles must fall back to exhaustive compatible-model
evaluation.

### 2.6 Explicit geometry and low-rank class models

Signed distance fields and constructive solid geometry provide editable,
human-auditable spatial objects [28,29]. In feature space, however, editability
does not guarantee calibrated discrimination. Low-rank class models are closely
related to probabilistic PCA and mixtures of factor analyzers [30,31]. GEODE's
affine components use a center, orthonormal tangent basis, tangent variances,
and isotropic residual variance. The present study evaluates both a
weighted-affine support score and a proper Gaussian likelihood; the latter
wins rejection.

---

## 3. Problem Formulation

Let \(f(x) \in \mathbb{R}^d\) be a frozen representation and
\(\mathcal{K}\) the current set of known classes. An operational open-world
system processes an input through six functions:

1. **Acceptance:** predict \(k \in \mathcal{K}\) or reject.
2. **Buffering and discovery:** store rejected evidence and form persistent
   candidate groups.
3. **Review:** request a semantic judgment for a stable group.
4. **Adaptation:** update an existing class or create a new class only after
   confirmation.
5. **Routing:** select compatible model bundles without hiding unknowns.
6. **Publication and rollback:** publish an immutable child bundle and preserve
   exact recovery of its parent.

We distinguish three labels that are often conflated:

- **rejected:** the acceptance score crosses a threshold;
- **grouped:** rejected samples exhibit persistent structure;
- **confirmed:** an authorized review event assigns semantic meaning.

Only the third state may authorize a semantic publication or model mutation.

### 3.1 Primary endpoint

The integrated endpoint is safe novel-class integration under a fixed review
budget. A system must integrate at least 75% of confirmable classes, reduce
reviewed samples by at least 25% relative to reviewing every rejection, preserve
known accuracy within 1 point, preserve unknown recall within 2 points of the
retained rejection head, create no unconfirmed classes, and pass graph, replay,
rollback, audit, and fallback contracts.

---

## 4. Experimental Protocol

### 4.1 Data and representation

The realized experiment is a bounded CIFAR-10 proxy. We use immutable
384-dimensional CLS-token features from DINOv2-small [32]. Each frozen seed has
10,000 training and 1,000 development examples. Classes 0--7 are known; classes
8 and 9 are withheld from acceptance-head fitting and act as proxy unknowns.
For adaptation, class 8 is confirmable and class 9 remains unknown.

The original preregistration also described CIFAR-100 and DomainNet tracks.
Those broader tracks were not executed in the final v7 chain and are not part of
the evidence reported here. No final-test labels were opened. All reported
model selection and evaluation use frozen development episodes with seeds 11,
23, and 37.

### 4.2 Frozen stages

| Stage | Purpose | Realized use |
|---|---|---|
| S0 | Schema, replay, and leakage tests | Deterministic synthetic fixtures |
| S1 | Cheap falsification | Seed 11 |
| S2 | Stage-wise retention | Seeds 11, 23, 37 |
| S3 | Integrated development loop | Three frozen schedules |
| S4 | Independent confirmation | Not opened because M43 failed |

Every milestone is parent-hash bound. Advancement decisions are made before the
next stage. The final artifact-only verifier reads JSON evidence and indexes
without loading feature arrays or labels.

### 4.3 Acceptance heads

All heads use identical known-class fitting data. A stratified 20% train-only
calibration split selects the novelty threshold at a target 92% known coverage.
We evaluate:

1. multinomial maximum posterior;
2. 1-nearest-support distance with 256 supports per class;
3. rank-32 class-conditional Gaussian likelihood;
4. EVM-style Weibull margin tails with 128 supports per class;
5. two rank-16 weighted affine components per class;
6. sigmoid-calibrated RBF SVM evidence.

The autonomy gate requires, on every seed, at least 90% known coverage, at least
50% unknown recall, no more than 1 point accepted-known accuracy loss, and exact
replay. The review gate requires precision within 2 points of the best
non-geometric control, at least 10 points more unknown recall than the historical
maximum-probability transfer baseline, and passing corruption and resource
checks. The review budget is 50 samples per 1,000.

For a low-rank Gaussian component with center \(\mu\), orthonormal basis \(U\),
tangent variances \(\lambda\), and isotropic residual variance \(\sigma^2\), we
compute

\[
q(x)=\sum_j \frac{(u_j^\top(x-\mu))^2}{\lambda_j}
 + \frac{\lVert (I-UU^\top)(x-\mu)\rVert_2^2}{\sigma^2},
\]

\[
\log p(x \mid k)=-\frac{1}{2}\left(
q(x)+\sum_j\log\lambda_j+(d-r)\log\sigma^2+d\log(2\pi)\right).
\]

Novelty is \(-\max_k \log p(x\mid k)\).

### 4.4 Persistent discovery

Rejected samples enter a bounded FIFO buffer of at most 2,000
384-dimensional records. Five deterministic windows are processed. Candidate
policies are:

- no clustering;
- streaming centroid micro-clusters;
- HDBSCAN;
- FINCH;
- silhouette-selected k-means with at most four groups.

Groups require at least 10 samples and 70% purity for scoring. A stable review
identifier persists when consecutive-window member overlap is at least 50%.
Growth, expiry, split, and merge evidence is recorded explicitly. The stage
gate requires at least 66.1% distinct-group recall, at least 70% review
precision, full recovery in at least 6 of 9 matched cells, exact review-ID
continuity, and no semantic publication before confirmation.

### 4.5 Human confirmation and adaptation

A deterministic oracle simulates human review while preserving the semantic
boundary. Its allowed outcomes are existing class, new class,
corruption/irrelevant, or unresolved. We compare no update, native Gaussian
update, rank-16 affine-component insertion, full class-local refitting, and a
full-retraining control.

For new-class evaluation, 50 confirmed class-8 samples are provided as support.
An adaptation passes only if target success improves by at least 5 points, known
accuracy drops by at most 1 point, unknown recall drops by at most 2 points,
known NLL does not materially regress, graph validation has zero issues,
unconfirmed publication is rejected, and exact replay and rollback hold.

### 4.6 Empirical routing

We construct three independently owned bundles with class orders
\(\{0,1,2,3\}\), \(\{4,5,6,7\}\), and \(\{8\}\). Candidate empirical profiles
are:

- class centroids and radii;
- rank-16 Gaussian summaries;
- 16 prototypes per class;
- rank-16 undercomplete autoencoder reconstruction.

Each profile stores its representation, fit-data, calibration-data, class-order,
dimension, threshold, and update lineage. The shortlist size is two. A profile
is authoritative only if every seed reaches 99% correct-bundle top-1, 99.9%
exhaustive-winner inclusion, 99.9% final agreement, 95% no-confident-route
recall, 25% fewer exact-model evaluations, lower p95 latency, exact update
replay, and 100% stale-profile fallback.

### 4.7 Integrated factorial

Only stage-passing mechanisms may enter the integrated study:

- rank-32 Gaussian rejection;
- HDBSCAN or FINCH discovery, plus no grouping as a burden control;
- delayed confirmation or no confirmation;
- rank-16 confirmed affine insertion or no update;
- exhaustive routing, because empirical routing failed its gate.

This yields 36 seed-by-policy cells. The review budget is fixed at 50 for grouped
arms. The reject-everything baseline reviews the entire rejection stream.

---

## 5. Results

### 5.1 Rejection: proper density outperforms geometric support

Table 1 reports three-seed means. The low-rank Gaussian was the only head to
pass both autonomy and review gates.

**Table 1. Matched acceptance-head results.**

| Head | Known coverage (%) | Unknown recall (%) | AUROC | Review precision (%) | Retained |
|---|---:|---:|---:|---:|:---:|
| Maximum posterior | 91.38 | 60.17 | 0.9053 | 69.33 | No |
| kNN support | 91.83 | 67.00 | 0.9045 | 86.67 | No |
| Low-rank Gaussian | **92.08** | **86.17** | **0.9561** | **98.67** | **Yes** |
| EVM-style Weibull margin | 100.00 | 0.00 | 0.5102 | 22.00 | No |
| Weighted affine support | 90.75 | 53.00 | 0.8763 | 64.67 | No |
| RBF SVM evidence | 93.08 | 12.83 | 0.4010 | 31.33 | No |

The retained Gaussian reached unknown recall of 86.0%, 85.0%, and 87.5% on
seeds 11, 23, and 37, respectively. Review precision was 98%, 100%, and 98%.
Each fitted state occupied approximately 0.78 MiB and fit-plus-score time was
0.50--0.56 s. A registered 1%-scale feature-noise corruption increased false
rejection by at most 0.125 points.

The weighted affine arm produced meaningful rejection signal but did not
justify a primary SDF claim. It trailed Gaussian AUROC by 0.0798 and review
precision by 34 points, and failed a seed-level autonomy operand.

### 5.2 Discovery: multiple groupers pass stage-wise gates

**Table 2. Persistent discovery results.**

| Discovery policy | Distinct-group recall (%) | Review precision (%) | Full-recovery cells | Pass |
|---|---:|---:|---:|:---:|
| No clustering | 0.00 | 0.00 | 6/9 safety cells only | No |
| Streaming micro-clusters | 0.00 | 98.67 | 6/9 safety cells only | No |
| HDBSCAN | **100.00** | **100.00** | **9/9** | **Yes** |
| FINCH | 83.33 | 86.39 | 8/9 | Yes |
| Frozen-feature k-means | 100.00 | 98.67 | 9/9 | Yes |

HDBSCAN was retained as primary and FINCH as a mechanistically distinct
control. Review identifiers replayed exactly, no buffer eviction occurred,
retained-arm memory stayed below 0.75 MiB, and maximum observed window latency
was below 0.10 s. High sample-level precision alone was insufficient: streaming
micro-clusters reviewed mostly unknown samples but failed to separate the two
withheld groups.

### 5.3 Adaptation: isolated new-class insertion passes

Existing-class expansion did not pass. Native Gaussian, class-local refit, and
full-retraining controls improved the targeted rejected-known subset by only
4.17 points on average and passed only seed 23. Affine-component insertion
improved it by 0 points.

For confirmed new-class creation, rank-16 affine insertion was the only passing
arm. Held-out target success rose from 0% to 42%, 52%, and 36% across the three
seeds, a mean gain of 43.33 points. Known accuracy remained 95.63%, 96.63%, and
95.00%; remaining-unknown recall stayed unchanged at 82%, 79%, and 77%.
Known NLL did not change at reported precision. All child bundles replayed
exactly, graph validation found zero issues, and rollback restored the exact
parent hash and predictions.

This is a density-calibrated affine-component result, not evidence that a raw
SDF score is cross-class comparable.

### 5.4 Routing: useful in shadow mode, unsafe as authority

**Table 3. Empirical routing-profile results.**

| Profile | Correct-bundle top-1 (%) | Winner inclusion (%) | Unknown fallback recall (%) | All seeds pass |
|---|---:|---:|---:|:---:|
| Centroid/radius | **93.53** | 99.37 | 31.33 | No |
| Low-rank Gaussian | 91.65 | **99.57** | **71.00** | No |
| Compact prototypes | 89.80 | 98.71 | 24.00 | No |
| Autoencoder reconstruction | 89.14 | 98.75 | 51.33 | No |

Every profile replayed exactly after adaptation and stale profiles fell back
exhaustively. Mean exact-model evaluation reduction ranged from 49.99% to
54.47%. Nevertheless, all profiles missed at least the top-1, winner-inclusion,
unknown-fallback, or latency gate. Routing therefore remained shadow-only, and
the integrated loop used exhaustive compatible-model inference.

### 5.5 Integrated loop: stage-wise success does not compose

**Table 4. Integrated delayed-confirmation and rank-16-insertion cells.**

| Discovery | Integrated classes | Mean reviewed samples | Review reduction vs all rejects (%) | Known accuracy (%) | Remaining-unknown recall (%) |
|---|---:|---:|---:|---:|---:|
| HDBSCAN | 0/3 | 50.00 | 78.78 | 95.75 | 79.33 |
| FINCH | 0/3 | 15.33 | **93.49** | 95.75 | 79.33 |
| No grouping / review all | 3/3 | 235.67 | 0.00 | 95.75 | 79.33 |

FINCH was non-dominated among the eligible grouped arms because it minimized
review burden while preserving known accuracy, but it integrated no class.
HDBSCAN also integrated none. Their per-seed held-out class-8 success was:

- HDBSCAN: 4%, 22%, and 20%;
- FINCH: below the registered 30% integration threshold on every seed;
- review all rejects: 40%, 33.33%, and 50%.

The reject-everything streams contained 223, 231, and 253 records. This control
integrated all three classes only by reviewing all of them, so it failed the
25% burden-reduction requirement. Mean remaining-unknown recall was 79.33%,
6.83 points below the 86.17% M39 baseline and beyond the permitted 2-point
drop.

All safety contracts passed: there were zero unconfirmed semantic
publications, zero unconfirmed mutations, zero false autonomous class
creations, zero graph issues, exact replay, exact rollback, exhaustive fallback,
and complete audit histories. The failure is therefore one of utility and
composition, not transactional correctness.

---

## 6. Why Did Composition Fail?

The results expose a mismatch between the sufficient statistics of adjacent
stages.

### 6.1 Review purity is not adaptation sufficiency

HDBSCAN achieved perfect stage-wise group recall and review precision, yet its
50 reviewed points supported only 4--22% held-out new-class success. FINCH used
still fewer samples and also failed adaptation. In contrast, reviewing all
rejections provided 90--98 confirmed class-8 samples and crossed the integration
threshold on every seed.

The groupers optimized density coherence and review purity. Adaptation required
coverage of the class's within-class geometry. Dense cluster cores can be pure
but unrepresentative of low-density modes or decision-boundary regions. A
review policy should therefore optimize a joint objective such as

\[
\text{utility}(G)=
\alpha\,\text{purity}(G)
+\beta\,\text{coverage}(G)
+\gamma\,\text{boundary diversity}(G)
-\lambda\,\text{review cost}(G),
\]

rather than purity or persistence alone.

### 6.2 Threshold transfer changes downstream recall

The retained rejection head achieved 86.17% unknown recall in isolation, while
the integrated protocol retained 79.33%. The difference illustrates that a
threshold satisfying a local acceptance benchmark does not automatically
preserve end-to-end unknown recall after class creation and schedule
composition.

### 6.3 Routing quality is asymmetric

Winner inclusion above 99% appears strong, but authoritative routing requires
the remaining errors to fail safely. The no-confident-route recall of 24--71%
was inadequate. A router can save computation while preferentially hiding the
inputs for which exhaustive evaluation matters most. Evaluation reduction is
therefore not a safety result.

### 6.4 Editability is not predictive competitiveness

The broader GEODE program repeatedly found that explicit geometric artifacts
can replay and roll back exactly while trailing discriminative controls. In this
study, proper Gaussian likelihood outperformed weighted affine support at
rejection. The compact affine component helped isolated new-class creation, but
that advantage disappeared when support came from retained discovery groups.
Editability remains a systems property, not evidence of predictive superiority.

---

## 7. Reproducibility and Artifact Integrity

All configurations, feature identities, parent indexes, stage decisions, and
branch dispositions are stored in the repository. The final verifier checks:

- 6 immutable milestone indexes;
- 16 indexed artifacts;
- 8 frozen conclusion operands;
- no training-data loading;
- no opening of final labels.

Two final verifier runs produced byte-identical evidence. The command is:

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.tier1.eval_v7_final_replay
```

The final branch ledger is
`analysis/V7_FINAL_CLAIM_LEDGER.md`. Machine-readable evidence is under
`logs/results/v7/`.

### Data and code availability

The implementation, frozen configurations, evidence indexes, and reproduction
commands are maintained in the GEODE project repository. For a non-anonymous
release, the archival repository URL and release DOI should be inserted here
after creating a versioned public release. CIFAR-10 and DINOv2 remain subject to
their respective upstream terms; the frozen representation manifest records the
checkpoint source, Apache-2.0 checkpoint license, preprocessing digest, and
upstream weights digest.

---

## 8. Limitations

**Proxy scale.** The realized study uses CIFAR-10 classes and frozen DINOv2
features. It does not establish performance on large semantic inventories,
long-running natural streams, or domain-level routing.

**Development-only result.** The independent S4 confirmation schedule remained
sealed because M43 failed. The findings are preregistered development evidence,
not an untouched final-test confirmation.

**Simulated review.** A deterministic oracle models human confirmation. Real
reviewers introduce disagreement, latency, expertise variation, and
context-dependent costs.

**Approximate controls.** The EVM-style arm is a local approximation, not a
claim of faithful reproduction of every EVM implementation detail. End-to-end
ORCA, IGCD, or continual-GCD methods were not representation-matched because
they train or adapt representations.

**Class-order specificity.** Classes 8 and 9 serve as proxy unknowns. Different
held-out classes may alter geometric overlap, clusterability, and adaptation
sample complexity.

**No universal novelty claim.** The prior-art audit covered primary literature
available through 26 July 2026 but cannot rule out patents, non-English work,
closed industrial systems, or inaccessible publications.

**No SDF necessity claim.** Weighted affine support failed retention. The paper
does not claim that signed-distance geometry is required for open-world
recognition.

---

## 9. Ethical and Deployment Considerations

Open-world systems can create false confidence by assigning semantic meaning to
coherent but irrelevant or harmful clusters. This study prohibits autonomous
semantic publication and requires human confirmation, immutable provenance,
and exact rollback. These controls reduce but do not eliminate risk. A real
deployment should additionally support reviewer disagreement, appeal and
deletion procedures, privacy-aware retention, subgroup performance audits,
monitoring for feedback loops, and explicit authority boundaries for model
publication.

The experiments use public benchmark representations and simulated review.
They do not involve human subjects or sensitive personal data. The operational
architecture should not be interpreted as authorization for high-stakes use.

---

## 10. Conclusion

We evaluated a preregistered open-world recognition loop that joined calibrated
rejection, persistent grouping, human confirmation, transactional adaptation,
empirical routing, and exact rollback. Several stages succeeded independently:
a low-rank Gaussian produced high-quality rejection; HDBSCAN and FINCH formed
stable review objects; confirmed affine insertion was reversible and useful in
isolation; and routing profiles reduced model evaluations in shadow mode.

The complete loop nevertheless failed. Grouped review sets were pure but did
not provide enough geometric coverage for adaptation, while reviewing every
rejection restored integration only by eliminating the intended human-budget
advantage. The result supports a stage-wise lifecycle qualification, not a
closed operational loop.

The broader lesson is methodological: open-world research should evaluate the
interfaces between stages, not only the stages themselves. Rejection recall,
cluster purity, update accuracy, routing inclusion, and rollback can all look
strong in isolation while the composed system remains unusable. Future work
should train discovery and review selection for downstream adaptation utility,
with coverage and boundary diversity treated as first-class objectives.

---

## References

[1] A. Bendale and T. Boult. “Towards Open World Recognition.” *CVPR*, 2015.
https://arxiv.org/abs/1412.5687

[2] A. Bendale and T. Boult. “Towards Open Set Deep Networks.” *CVPR*, 2016.
https://arxiv.org/abs/1511.06233

[3] E. Rudd, L. Jain, W. Scheirer, and T. Boult. “The Extreme Value Machine.”
*IEEE TPAMI*, 2018. https://arxiv.org/abs/1506.06112

[4] M. Masud et al. “Classification and Novel Class Detection in
Concept-Drifting Data Streams under Time Constraints.” *IEEE TKDE*, 2011.
https://doi.org/10.1109/TKDE.2010.61

[5] A. Haque et al. “SAND: Semi-Supervised Adaptive Novel Class Detection and
Classification over Data Stream.” *AAAI*, 2016.
https://doi.org/10.1609/AAAI.V30I1.10283

[6] A. Haque et al. “ECHO: A Data Stream Classification Method for Detecting
Novel Class Drift.” *ICDE*, 2016. https://doi.org/10.1109/ICDE.2016.7498264

[7] E. de Faria, A. Gama, and A. Carvalho. “Novelty Detection Algorithm for Data
Streams Multi-Class Problems.” *Data Mining and Knowledge Discovery*, 2016.
https://doi.org/10.1007/s10618-015-0433-y

[8] X. Mu, F. Zhu, J. Du, E.-P. Lim, and Z.-H. Zhou. “Classification Under
Streaming Emerging New Classes: A Solution Using Completely Random Trees.”
*IEEE TKDE*, 2017. https://doi.org/10.1109/TKDE.2017.2691702

[9] S. Vaze et al. “Generalized Category Discovery.” *CVPR*, 2022.
https://arxiv.org/abs/2201.02609

[10] K. Cao, M. Brbić, and J. Leskovec. “Open-World Semi-Supervised Learning.”
*ICLR*, 2022. https://arxiv.org/abs/2102.03526

[11] Z. Zhang et al. “Grow and Merge: A Unified Framework for Continuous
Categories Discovery.” *NeurIPS*, 2022. https://arxiv.org/abs/2210.04174

[12] B. Zhao and O. Mac Aodha. “Incremental Generalized Category Discovery.”
*ICCV*, 2023. https://arxiv.org/abs/2304.14310

[13] F. Cendra et al. “Effective Prompt Pool Learning for Continual Category
Discovery.” *ECCV*, 2024. https://arxiv.org/abs/2407.19001

[14] R. Aljundi, P. Chakravarty, and T. Tuytelaars. “Expert Gate: Lifelong
Learning with a Network of Experts.” *CVPR*, 2017.
https://arxiv.org/abs/1611.06194

[15] S. Gururangan et al. “DEMix Layers: Disentangling Domains for Modular
Language Modeling.” 2021. https://arxiv.org/abs/2108.05036

[16] M. Li et al. “Branch-Train-Merge: Embarrassingly Parallel Training of Expert
Language Models.” 2022. https://arxiv.org/abs/2208.03306

[17] C. Huang et al. “LoraHub: Efficient Cross-Task Generalization via Dynamic
LoRA Composition.” *COLM*, 2024. https://arxiv.org/abs/2307.13269

[18] D. Hendrycks and K. Gimpel. “A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks.” *ICLR*, 2017.
https://arxiv.org/abs/1610.02136

[19] K. Lee et al. “A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks.” *NeurIPS*, 2018.
https://arxiv.org/abs/1807.03888

[20] J. Mukhoti et al. “Deep Deterministic Uncertainty: A New Simple Baseline.”
*TMLR*, 2023. https://arxiv.org/abs/2102.11582

[21] Y. Sun et al. “Out-of-Distribution Detection with Deep Nearest Neighbors.”
*ICML*, 2022. https://arxiv.org/abs/2204.06507

[22] J. van Amersfoort et al. “Uncertainty Estimation Using a Single Deep
Deterministic Neural Network.” *ICML*, 2020.
https://arxiv.org/abs/2003.02037

[23] J. Liu et al. “Simple and Principled Uncertainty Estimation with
Deterministic Deep Learning via Distance Awareness.” *NeurIPS*, 2020.
https://arxiv.org/abs/2006.10108

[24] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu. “A Density-Based Algorithm
for Discovering Clusters in Large Spatial Databases with Noise.” *KDD*, 1996.

[25] R. Campello, D. Moulavi, and J. Sander. “Density-Based Clustering Based on
Hierarchical Density Estimates.” *PAKDD*, 2013.
https://doi.org/10.1007/978-3-642-37456-2_14

[26] S. Sarfraz, V. Sharma, and R. Stiefelhagen. “Efficient Parameter-free
Clustering Using First Neighbor Relations.” *CVPR*, 2019.
https://arxiv.org/abs/1902.11266

[27] M. Muqeeth et al. “Learning to Route Among Specialized Experts for
Zero-Shot Generalization.” 2024. https://arxiv.org/abs/2402.05859

[28] A. Requicha. “Representations for Rigid Solids: Theory, Methods, and
Systems.” *ACM Computing Surveys*, 1980.
https://doi.org/10.1145/356827.356833

[29] J. Park et al. “DeepSDF: Learning Continuous Signed Distance Functions for
Shape Representation.” *CVPR*, 2019. https://arxiv.org/abs/1901.05103

[30] M. Tipping and C. Bishop. “Probabilistic Principal Component Analysis.”
*Journal of the Royal Statistical Society: Series B*, 1999.
https://doi.org/10.1111/1467-9868.00196

[31] Z. Ghahramani and G. Hinton. “The EM Algorithm for Mixtures of Factor
Analyzers.” Technical Report CRG-TR-96-1, University of Toronto, 1996.

[32] M. Oquab et al. “DINOv2: Learning Robust Visual Features without
Supervision.” *TMLR*, 2024. https://arxiv.org/abs/2304.07193

---

## Appendix A. Registered Gates

| Stage | Principal advancement condition |
|---|---|
| Rejection | Every seed: >=90% known coverage, >=50% unknown recall, <=1-point known-accuracy loss, exact replay |
| Discovery | >=66.1% group recall or equivalent review savings; >=70% precision; >=6/9 recovery cells; stable IDs |
| Adaptation | >=5-point target gain; <=1-point known drop; <=2-point unknown-recall drop; exact replay/rollback |
| Routing | >=99% top-1; >=99.9% winner inclusion/agreement; >=95% unknown fallback; >=25% evaluation reduction |
| Integrated | >=75% classes integrated; >=25% review reduction; known/unknown preservation; all safety contracts |

## Appendix B. Final Branch Disposition

| Branch | Disposition |
|---|---|
| Low-rank Gaussian rejection | Passed |
| Weighted affine rejection | Failed retention |
| HDBSCAN discovery | Passed stage gate |
| FINCH discovery | Passed stage gate |
| Existing-class expansion | Closed |
| Confirmed new-class insertion | Passed stage gate |
| Authoritative empirical routing | Closed; shadow-only |
| Integrated HDBSCAN/FINCH loop | Failed |
| Independent S4 confirmation | Blocked |
| Final outcome | Stage-wise lifecycle qualification |
