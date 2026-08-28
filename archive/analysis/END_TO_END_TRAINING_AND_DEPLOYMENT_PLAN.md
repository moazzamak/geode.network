# GEODE End-to-End Training, Evaluation, and Deployment Plan

**Date:** 26 July 2026

**Status:** E0-E6, E8-E11, M16, and M18 complete; E7 requires physical
multi-host qualification

**Related evidence:** `analysis/RESEARCH_REPORT_v5.md`,
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v5.md`,
`analysis/MILESTONE_RESULTS.md`

**Preflight review:** amended after a final architecture, statistical, and prior-art audit. Section 17 records the blocking corrections and experimental opportunities.

---

## 1. Objective

Build and evaluate one reproducible GEODE lifecycle that starts with public raw data and ends with a versioned inference service. The lifecycle must exercise:

- loading, fingerprinting, and freezing commodity or task-native representations;
- one-time affine-interface training followed by immutable freezing;
- GEODE construction, refinement, and calibration;
- role fingerprints and empirical support profiles;
- graph composition and compatibility checks;
- exhaustive routing plus shadow candidate routing;
- closed-set, OOD, transfer, open-world, and regression evaluation;
- epoch- and stage-level metric history;
- interruption-safe checkpoint and resume;
- local, multi-process, and multi-machine execution;
- immutable model publication, canary deployment, and rollback.

The objective is not to beat specialized modern systems on every ordinary
subtask. The current research target is head parity with black-box controls on
the same frozen representation, plus a measured Pareto advantage in edit
locality, exact rollback, and audited migration. Joint encoder-head gradient
training is out of scope because encoder drift invalidates component provenance,
calibration and support objects, changed-region measurements, replay, and
rollback.

M16 now supplies the versioned data-stage and seed contract, deterministic
representation/head/readout registry, canonical run and migration schemas,
representation-lineage compatibility guard, paired prediction intervals,
formal Pareto dominance, and SHA-256 artifact indexing. Its ten-cell S0 matrix
replayed byte-identically and rejected a deliberate split mismatch.

M18 supplies reusable spherical, diagonal, full, low-rank, and shared precision
metrics, but its sample-support policy failed both frozen advancement gates.
Therefore these parameterizations remain opt-in research surfaces and the
existing primitive-family default remains authoritative.

---

## 2. Core Decision: Use a Benchmark Federation

No single public dataset naturally tests image classification, domain transfer, temporal prediction, point-cloud learning, OOD detection, open-world class arrival, routing scale, model-graph compatibility, and failure recovery. Dataset choice also cannot test power-loss recovery or cluster scheduling; those are properties of the execution system.

Use four coordinated tracks with one shared run, artifact, checkpoint, metric, and publication protocol:

| Track                    | Public data                                               | Purpose                                                                                                          | Frequency         |
| ------------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------- |
| A. Recovery and CI       | CIFAR-100, CIFAR-10, SVHN                                 | fast end-to-end runs, class splits, OOD, restart and regression tests                                            | every change      |
| B. Flagship transfer     | DomainNet                                                 | source-to-target domain transfer, class growth, representation reuse, distributed feature extraction and fitting | milestone/release |
| C. Temporal flexibility  | WikiText-103                                              | causal splits, epoch history, temporal transfer, non-image model graph                                           | milestone         |
| D. Geometric flexibility | ModelNet40, with synthetic ellipsoids only as diagnostics | point-cloud representation, explicit geometry, dimensional and primitive scaling                                 | milestone         |

### 2.1 Why DomainNet is the flagship

DomainNet provides many classes across visually distinct domains. That supports source-domain representation training, target-domain transfer, held-out classes, domain-specific support profiles, and enough samples to make feature extraction and class fitting meaningful distributed workloads. The study must use the official train/test lists and record the exact downloaded file manifest and license/provenance metadata.

DomainNet does not replace the other tracks. Its image domains cannot establish temporal causality or point-cloud flexibility, and it should not be used for routine fault-injection tests because a full rerun is expensive.

### 2.2 Dataset partitions

Freeze every partition before training:

- **Known-base classes:** available during representation and GEODE training.
- **Known-transfer classes/domains:** labels available only to the transfer stage.
- **Proxy-novel classes:** used to select OOD, review, and clustering policies.
- **Final-novel classes:** labels hidden until final evaluation.
- **Calibration split:** disjoint from geometry, policy selection, and test data.
- **Readout-calibration split:** fits score scaling, temperature, or multinomial readouts only.
- **Risk-control split:** fits abstention, OOD, or conformal thresholds only; it is not reused to choose the score or readout.
- **Validation split:** selects hyperparameters and checkpoints.
- **Final test split:** observational only.
- **Replay set:** frozen examples or feature shards used to measure forgetting after transfer or edits.

For temporal data, replace random partitions with chronological geometry, calibration, validation, and test segments separated by a configurable purge gap.

Every representation must also have a **pretraining provenance record**: source datasets when disclosed, objective, checkpoint hash, license, access date, and any known or plausible overlap with evaluation data. A foundation-model result with undisclosed or un-auditable training data is an external-pretraining stress test, not evidence of learning from the episode's allowed data alone.

---

## 3. End-to-End System Boundary

A run is a durable state machine, not one Python process.

```mermaid
flowchart LR
    A[Acquire and verify data] --> B[Freeze splits]
    B --> C[Load or separately train representation]
    C --> C2[Freeze and hash representation and optional affine interface]
    C2 --> D[Extract content-addressed features]
    D --> E[Fit GEODE classes in parallel]
    E --> F[Refine and calibrate]
    F --> G[Build fingerprints and support profiles]
    G --> H[Validate graph and routing]
    H --> I[Offline evaluation]
    I --> J[Package candidate bundle]
    J --> K[Shadow and canary inference]
    K --> L[Publish or roll back]
    L --> M[Review-only stream and gated adaptation]
```

Each transition must be idempotent. A completed stage is reused only when its input artifact hashes, configuration hash, code compatibility declaration, and output checksums match. A failed or interrupted stage may be retried without mutating a previously committed output.

---

## 4. Durable Run Model

### 4.1 Run identity

Add a `run_id` derived from the canonical experiment configuration and a unique `attempt_id` for each execution attempt. Each run records:

- parent run and parent model bundle, when applicable;
- Git commit and dirty-worktree state;
- normalized configuration and schema version;
- dataset, split, vocabulary, transform, and feature fingerprints;
- random seeds and deterministic-mode settings;
- package, Python, driver, accelerator, and operating-system versions;
- executor type, worker topology, CPU/GPU allocation, and host identifiers;
- lifecycle state and timestamps;
- output artifact names, hashes, sizes, and retention class.

Extend the existing manifest functions in `experiments/common/experiment_manifest.py`; do not create a second incompatible provenance format.

### 4.2 Lifecycle states

Use explicit states:

`CREATED -> DATA_READY -> REPRESENTATION_READY -> FEATURES_READY -> GEOMETRY_READY -> CALIBRATED -> GRAPH_VALIDATED -> EVALUATED -> PACKAGED -> STAGED -> PUBLISHED`

Any stage may enter `FAILED`, `CANCELLED`, or `INTERRUPTED`. A published run is immutable. A resumed attempt continues the same logical run; an incompatible code/config/data change creates a child run.

### 4.3 Atomic stage commit

A stage writes to `runs/<run_id>/attempts/<attempt_id>/<stage>.partial/`. It then:

1. flushes files and closes writers;
2. computes hashes and validates the stage schema;
3. writes `stage_manifest.json` and `SUCCESS` last;
4. atomically renames the local directory, or uploads to an immutable object-store prefix;
5. conditionally updates the `latest_valid` pointer.

Readers ignore directories without `SUCCESS`. Object-storage publication uses immutable objects plus a small generation-numbered pointer; it never relies on renaming an object as an atomic operation.

### 4.4 Checkpoint contents

Every resumable checkpoint contains:

- lifecycle stage, epoch, global step, and data-shard cursor;
- representation weights and optimizer/scheduler/scaler state;
- GEODE centers, radii, orientations, polarities, class modes, and calibrator state;
- refinement optimizer state, when refinement is active;
- Python, NumPy, Torch CPU, and Torch accelerator RNG states;
- distributed sampler epoch and worker-sharding plan;
- best-checkpoint metric and early-stopping state;
- `ModelFingerprint`, `SupportProfile`, transform fingerprint, and class-column order;
- routing-index version and exhaustive-oracle compatibility metadata;
- metric-ledger offset and completed shard/class task IDs;
- checksums for every referenced immutable artifact.

Use explicit JSON plus array/tensor formats rather than pickling the complete process. A checkpoint loader must reject unknown schema versions, missing hashes, incompatible class order, or a mismatched transform fingerprint.

### 4.5 Checkpoint policy

- Representation training: checkpoint every epoch and every configurable number of optimizer steps.
- GEODE class fitting: commit each completed class independently, then assemble the class set deterministically.
- Refinement: checkpoint every epoch.
- Feature extraction: commit fixed-size shards independently.
- Calibration and routing-index construction: checkpoint at stage completion because they are deterministic and comparatively short.
- Publication: retain the best validation checkpoint, the last two valid checkpoints, every published bundle, and checkpoints at milestone boundaries.

The target recovery point objective is at most one epoch for iterative training and one shard or one class fit for map-style stages.

---

## 5. Metrics and Experiment History

### 5.1 Append-only metric ledger

Write one schema-validated event per line to `metrics.jsonl`. Each event includes:

- `run_id`, `attempt_id`, stage, epoch, step, split, and wall-clock time;
- metric name, value, unit, aggregation, and sample count;
- task, dataset, domain, class/group slice, and model component;
- worker rank and host for systems metrics;
- checkpoint ID and model-bundle candidate ID;
- whether the metric was used for selection or is observational only.

A resumed attempt appends a new attempt segment; it does not overwrite history. Duplicate `(attempt_id, stage, epoch, step, metric, slice)` events are rejected or made idempotent by event ID.

Use separate event namespaces for exploratory, selection, and final metrics. Final-test events are write-once and cannot be consumed by a training, calibration, threshold-selection, or early-stopping stage.

### 5.2 Required learning curves

**Representation training**

- train and validation loss;
- top-1/top-5 and balanced accuracy;
- per-domain and per-class accuracy;
- learning rate, gradient norm, and parameter norm;
- source retention and target transfer gain;
- epoch duration, examples/second, GPU utilization, and peak memory.

**GEODE fitting and refinement**

- geometry/refinement cross-entropy by epoch;
- raw, temperature, diagonal, and multinomial accuracy where applicable;
- NLL, Brier score, ECE, and top-k accuracy;
- adaptive/classwise calibration error, risk-coverage curves, selective risk, and coverage at each authorized abstention threshold;
- class capture, contamination, uncovered fraction, and rejected candidates;
- experts, additive/subtractive primitives, parameters, and bytes per class;
- fit time and convergence/fallback reason per class.

**Routing and graph execution**

- exhaustive agreement and top-k candidate recall;
- candidate classes and primitive evaluations per sample;
- exhaustive fallback rate and bound/certificate failures;
- p50/p95/p99 end-to-end and per-node latency;
- throughput, queue time, worker utilization, and load skew;
- fingerprint compatibility failures and support-profile mismatches.

**Transfer and adaptation**

- target gain, source/replay loss, and forgetting;
- calibration drift and OOD AUROC/FPR95;
- OOD AUPR-In/AUPR-Out and open-set classification rate or OSCR-style curves;
- review event recall, ARI, purity, burden, duplicates, and time-to-review;
- proposed, accepted, quarantined, rolled-back, and published mutations;
- changed-region and graph-migration validation outcomes.

### 5.3 Tracking backend

Keep local JSON/JSONL as the portable source of truth. Add an optional MLflow adapter for dashboards, comparison, tags, and artifact browsing. The training and evaluator code must not require a live tracking server; buffered events must sync after a network outage.

ECE remains descriptive rather than a release criterion because binning can hide local or classwise failures. Release decisions use the preregistered proper scoring rules and risk/coverage endpoints. Distribution-free or conformal guarantees may be reported only for the population and exchangeability conditions justified by the calibration protocol; shifted-domain results are empirical unless a shift-aware method and its assumptions are tested.

---

## 6. Training and Transfer Protocol

### 6.1 Representation variants

Compare these explicitly:

1. frozen public pretrained backbone;
2. linear probe on frozen features;
3. source-domain fine-tuning;
4. target-domain last-layer or adapter fine-tuning;
5. full target-domain fine-tuning, where budget permits;
6. frozen representation plus GEODE;
7. transferred representation plus GEODE.

The same transformed features feed matched logistic, nearest-centroid, shrinkage-Gaussian, GMM, linear-SVM, and RBF-SVM controls where computationally feasible. Include a conventional neural classification head to quantify what is lost by replacing the head with GEODE.

Use two representation lanes. The **controlled lane** trains only on the episode's permitted source data or uses a checkpoint with auditable, declared pretraining; it supports causal claims about the episode. The **external lane** may use a strong frozen representation such as DINOv2, but supports only a matched-head comparison under that representation. Never pool the two lanes in one headline result.

### 6.2 Main DomainNet episodes

Pre-register at least three episode types:

- **Domain transfer:** train representation and geometry on source domains; adapt representation or geometry on one target domain; test untouched target examples and source replay.
- **Class transfer:** train on known-base classes; add known-transfer classes through a fingerprinted graph migration; test old/new accuracy and calibration.
- **Open-world stream:** interleave known, shifted-known, proxy-novel, final-novel, and corrupted observations; permit review requests but no automatic semantic publication.

Rotate source and target domains across repeated seeds. Select policies on proxy domains/classes only. Keep final target domains/classes unopened until the policy and resource budget are frozen.

### 6.3 GEODE stage order

For each episode:

1. load one immutable representation checkpoint;
2. extract and hash train/calibration/validation/test feature shards;
3. fit each class independently from geometry data;
4. assemble classes in canonical class-ID order;
5. optionally run additive refinement on geometry data only;
6. fit readouts on calibration data only;
7. create the role fingerprint and empirical support profile separately;
8. build the model graph and validate dimensions and class-column order;
9. build routing metadata without changing the authoritative route;
10. evaluate validation, then final test once;
11. package only if all safety and reproducibility gates pass.

M15 does not change this stage order. Its retained global covariance
temperature is an optional classification readout parameter fit only on the
readout-calibration split. Per-class temperatures failed their advancement
gate, so mixture weights and combined likelihood parameters are not part of
the end-to-end baseline. The feature-only task head remains a required matched
control.

### 6.4 Representation-to-geometry contract

Do not pass a modern embedding directly into a full ellipsoid fitter by default. A full $d$-dimensional ellipsoid has $d(d+3)/2$ free quadratic and linear terms before fusion and calibration; the current constructor uses the same quantity as its default minimal seed size. This becomes statistically and computationally unsuitable for common 384- or 768-dimensional embeddings.

For every class, record $n_c$, $d$, effective rank, condition number, primitive parameter count, and the parameter-to-sample ratio. Compare a nested capacity ladder under one frozen transform and validation budget:

1. sphere;
2. axis-aligned ellipsoid;
3. shrinkage-covariance ellipsoid;
4. low-rank-plus-diagonal ellipsoid;
5. full ellipsoid only when the geometry split supports it and conditioning gates pass.

Select PCA dimension and geometry family without final-test access. Compare raw frozen features, PCA/whitening, and PCA+LDA explicitly; supervised LDA must be fit on geometry-training data only and shared with matched controls. Failed conditioning, insufficient samples, or an excessive complexity ratio causes fallback to a simpler family rather than jittering until a fit succeeds.

The pipeline is end-to-end operationally, not jointly differentiable. A
commodity or task-native encoder is loaded or trained separately, then frozen
and hashed. An optional linear or low-rank affine interface may be trained once
under the pre-test development protocol, but it too is frozen before geometry is
fitted. Discriminative gradients may update only explicit head parameters in
that fixed space. Any later representation replacement creates a new bundle,
invalidates derived geometry/calibration/support artifacts, emits a migration
report, and preserves rollback to the old bundle.

---

## 7. Fingerprinting Test Plan

The current `ModelFingerprint` identifies a role and I/O contract; `SupportProfile` records empirical support and data/transform identity. Preserve that separation.

Every release candidate must pass:

- deterministic signature generation across processes and machines;
- positive swap of models with the same role contract;
- negative swap for input source, output type, task, or class-set mismatch;
- graph validation after class-column expansion;
- rejection of mismatched feature-transform or dataset support profiles;
- class-order invariance of the role signature and class-order strictness of serialized score/calibrator state;
- parent/child provenance for transfer and adaptation;
- migration tests for adding classes, replacing a representation, and rebuilding downstream nodes;
- package hash verification before load and before serving traffic.

Add a separate immutable **artifact identity** hash for exact weights, arrays, calibrators, and graph topology. Two models may share a role fingerprint while having different artifact identities and support profiles.

---

## 8. Routing Test Plan

Exhaustive exact class SDF remains the production oracle until a candidate passes all gates on real fitted models.

### 8.1 Offline routing evaluation

Evaluate exhaustive, bound-based, centroid shortlist, batched, and any learned router on identical query batches. Report:

- exact prediction and score agreement;
- candidate recall for the exhaustive winner;
- latency including lookup, transfer, synchronization, and fallback;
- memory and index-build cost;
- behavior versus classes, primitives, dimension, batch size, and worker count;
- worst-performing classes/domains, not only global means.

### 8.2 Shadow routing in production

Candidate routing runs beside exhaustive inference on sampled requests. It cannot affect returned predictions. Promotion requires:

- 100% winner agreement for exact/certified routing, or a separately approved approximation contract;
- no material calibration or OOD regression;
- p95 latency and resource improvement after all overhead;
- bounded fallback and no persistent class starvation;
- independent confirmation on another dataset/domain and hardware profile.

If no candidate passes, exhaustive routing remains an acceptable research outcome.

Before developing another router, benchmark a packed structure-of-arrays implementation and fused exact class-field kernel against the current exhaustive path. The existing evidence shows routing overhead, not candidate recall, is the limiting factor. Approximate nearest-neighbor libraries may provide a shortlist baseline, but cannot be called exact and cannot control published outputs without exhaustive verification. Reopen sparse routing only when a measured break-even model predicts a gain at the fitted class/primitive scale.

---

## 9. Local and Cluster Execution

### 9.1 Executor abstraction

Define one stage/task API with two initial executors:

- **LocalExecutor:** sequential or process-pool execution for CI and workstation recovery tests.
- **RayExecutor:** multi-process and multi-node execution for feature shards, class fits, seed/domain episodes, and evaluation shards.

Representation fine-tuning should use standard Torch DistributedDataParallel when multiple GPUs cooperate on one model. GEODE class construction is naturally task-parallel and should use one immutable feature input plus independent class tasks rather than distributed shared mutation.

Ray is the recommended first cluster adapter because the workload mixes GPU representation actors, CPU/GPU map tasks, and long-lived inference actors. Slurm or Kubernetes/KubeRay may allocate the machines; experiment code should depend on the executor API, not directly on either scheduler.

### 9.2 Storage and coordination

- Local development: filesystem artifact store and SQLite tracking metadata.
- Cluster/release: S3-compatible versioned object storage such as S3 or MinIO, plus PostgreSQL-backed tracking metadata.
- Workers receive artifact URIs and hashes, never mutable Python model objects as the source of truth.
- Task completion is idempotent and committed by output hash.
- Leases and heartbeats permit abandoned tasks to be retried.
- A single coordinator owns lifecycle-state transitions; workers own only their immutable task outputs.

### 9.3 Distributed partitioning

- Feature extraction: partition by immutable input shard.
- GEODE construction: partition by `(run_id, class_id, seed)`.
- Evaluation: partition by dataset shard, then reduce sufficient statistics and prediction artifacts.
- Hyperparameter studies: partition by frozen configuration and seed.
- Production inference: replicate immutable model bundles; partition requests, not model state.

Never average independently fitted GEODE class models merely because they ran on different workers. Assemble independent class artifacts, or define an explicit ensemble experiment.

---

## 10. Failure-Recovery Qualification

A recovery feature is not complete until fault injection proves it.

### 10.1 Required fault injections

Terminate the coordinator or worker during:

- dataset download and checksum verification;
- split generation;
- feature-shard extraction;
- representation optimizer steps and epoch commit;
- one GEODE class fit;
- model assembly;
- calibration;
- routing-index build;
- artifact upload;
- candidate publication and production-pointer update.

Also test disk-full, corrupted checkpoint, unavailable object store, duplicate task completion, lost worker heartbeat, stale lease, and incompatible resume code.

### 10.2 Recovery acceptance criteria

- no committed artifact is silently corrupted;
- no final-test observation is moved into training after resume;
- resumed and uninterrupted deterministic runs match within declared tolerances;
- no more than one epoch, one feature shard, or one class task is repeated;
- metrics contain an explicit interruption and resume boundary;
- duplicate task outputs collapse to one content identity;
- a failed publication leaves the previous production pointer intact;
- rollback completes without rebuilding the previous model;
- recovery works after coordinator restart on a different machine.

Run a short kill-and-resume test in CI. Run the complete fault matrix before each release milestone.

Declare two reproducibility levels. **Replay identity** requires byte-identical outputs and hashes in a fixed software/hardware deterministic environment. **Scientific equivalence** permits preregistered numeric tolerances across supported accelerators or distributed topologies and compares predictions, metrics, and structural invariants; it does not require equal artifact hashes. Never silently weaken one level into the other.

---

## 11. Production Inference and Rollback

### 11.1 Model bundle

An immutable bundle contains:

- graph topology and node artifact identities;
- all model arrays, representation weights, transforms, and calibrators;
- role fingerprints and support profiles;
- routing metadata and authorized routing mode;
- class-column maps and semantic-router cache version;
- training/evaluation manifest and metric-summary hashes;
- software/environment compatibility range;
- bundle signature and creation provenance.

### 11.2 Registry states

Use `CANDIDATE`, `STAGING`, `CANARY`, `PRODUCTION`, `RETIRED`, and `REVOKED`. Promotion changes a small registry pointer; it never edits a bundle.

### 11.3 Serving gates

Before serving:

- verify bundle and component hashes;
- validate graph fingerprints and dimensions;
- run fixed golden requests and expected-output tolerances;
- load exhaustive routing even when shadow routing is enabled;
- confirm calibration and abstention policy versions;
- warm the model and record startup memory and latency.

Canary rollout begins with shadow traffic, then a small traffic fraction. Monitor latency, errors, abstentions, support drift, candidate-router disagreement, class frequency, and resource saturation. Automatic rollback may react to systems failures and large prespecified regressions; semantic model promotion still requires the research gate and human approval.

---

## 12. Baselines and Success Criteria

### 12.1 Ordinary-task gates

For each public-data episode, compare matched features and splits. A release candidate should:

- remain within a predeclared non-inferiority margin of direct logistic or the strongest feasible classical control for accuracy;
- report NLL and ECE separately from accuracy;
- preserve source/replay performance within a frozen forgetting budget after transfer;
- beat the trivial/unigram/centroid control appropriate to the task;
- disclose when RBF, neural, n-gram, or other specialized controls remain better.

Set numeric margins only after a pilot estimates variance; freeze them before final runs.

The pilot used to choose margins must be separate from confirmatory episodes. Define one primary endpoint per episode, a small ordered set of secondary endpoints, paired comparisons on identical examples, confidence intervals over seeds and domains, and a multiplicity policy before opening final results. Seeds are repeated algorithmic runs, not independent replacement datasets. Domain-transfer model selection must name the allowed validation domains and selection rule; an algorithm without that rule is incomplete.

### 12.2 Unique-capability gates

The showcase succeeds only if one reproducible run demonstrates:

- a representation checkpoint transferred into a new domain or class episode;
- GEODE fitting and calibration resumed after injected interruption;
- compatible graph assembly from fingerprinted components;
- rejection of at least one incompatible or support-mismatched component;
- exhaustive inference plus measured shadow routing;
- one review group with stable provenance across windows;
- a dry-run edit or class migration that either passes all gates or rolls back exactly;
- local and multi-machine runs producing equivalent artifacts/metrics within tolerance;
- canary publication and constant-time rollback to the previous bundle.

A safe rejection or rollback counts as correct behavior. The plan must not reward unsafe mutation merely to demonstrate activity.

---

## 13. Milestones and Advancement Gates

### E0. Freeze contracts

**Deliverables:** run/stage/checkpoint/metric schemas; artifact naming; lifecycle state machine; dataset episode specifications; pretraining provenance; geometry-capacity, calibration-budget, model-selection, and reproducibility-level contracts.

**Gate:** schemas round-trip, reject unknown required fields, and have migration/version rules.

**Implementation status (25 July 2026): COMPLETE.** Versioned frozen run, stage, checkpoint, and metric schemas reject missing, unknown, and future-version fields. The run contract encodes disjoint calibration budgets, pretraining provenance, geometry capacity, model selection, and reproducibility level. The geometry-feasibility probe reports per-class effective rank, conditioning, parameter/sample ratios, eligible families, and fails closed when no authorized family is supportable.

### E1. Durable local runner

**Deliverables:** `LocalExecutor`, atomic stage commits, content hashes, resume planner, CLI status/resume commands.

**Gate:** CIFAR smoke run survives termination during feature extraction and class fitting with no changed final metrics.

**Implementation status (25 July 2026): COMPLETE.** Atomic local stage publication, content hashes, `SUCCESS` markers, idempotent retry, corruption detection, and injected writer-failure behavior are validated. `LocalExecutor` reports pending, partial, committed, and corrupt states, reuses verified predecessors, and backs Tier 4 `--status` and `--resume` commands. On 3,500 public CIFAR-10 HOG examples, failures inside both the feature writer and class-0 fitting stage resumed to all 13 committed stages with byte-identical E1.1 validation artifacts and one metric event. Each failed stage was exposed as partial with downstream stages pending, then safely replaced on retry; the class failure reused committed features and transform with one feature extraction total. The E1 gate has passed for the local Tier 4 smoke path; independently resumable feature shards remain a later scale optimization rather than an E1 correctness blocker.

### E2. Epoch ledger and iterative checkpoints

**Deliverables:** representation and refinement checkpoint adapters; RNG/sampler restoration; JSONL metrics; local dashboard/export.

**Gate:** interrupted and uninterrupted fixed-seed runs match within declared deterministic tolerance and retain complete epoch histories.

**Implementation status (26 July 2026): COMPLETE.** An fsynced append-only metric ledger and atomic explicit JSON/NumPy checkpoint store are validated, including duplicate-event protection, non-pickled arrays, latest-checkpoint discovery, and content-addressed retry rejection. `SDFOptimizer` exports and imports geometry, momentum, hyperparameters, score scales, and class topology through stable keys rather than object IDs. The production Tier 6 refinement and Torch representation loops checkpoint every epoch and restore optimizer, scheduler/scaler, RNG, sampler, counters, and complete history. E2.5 records and validates PyTorch `DistributedSampler` rank, replica count, seed, shuffle/drop policy, epoch, and cursor; two simulated ranks resumed exactly and rejected rank mismatch. E2.6 deterministically projects the authoritative metric ledger into validated JSON, CSV, and a standalone local HTML dashboard. Re-export is idempotent and requires no tracking service. The interrupted/uninterrupted and complete-history gates have passed; E2 is complete.

### E3. Fingerprinted model bundles

**Deliverables:** exact artifact identities, role/support serialization, graph package schema, migration validation.

**Gate:** positive swap, negative compatibility, class expansion, transform mismatch, corruption, and rollback tests pass.

**Implementation status (26 July 2026): COMPLETE.** Content-addressed immutable bundles bind every component path, SHA-256 digest, and byte size to serialized role fingerprints, support profiles, class-column order, graph topology, transform identity, authorized routing mode, semantic-router cache version, training/evaluation/metric evidence hashes, software and environment compatibility, creation provenance, and parent lineage. Publication validates graph dimensions and support compatibility before making a bundle visible. Activation revalidates the complete component set and all hashes before atomically moving `CURRENT`; rollback activates the verified parent without rebuilding it. The standalone qualification passed compatible replacement, task/input/output/class incompatibility, transform mismatch, stale and coordinated class expansion, corruption with unchanged pointer, and exact parent rollback. The complete 192-test regression gate passed.

### E4. End-to-end CIFAR qualification

**Deliverables:** raw-data-to-candidate pipeline over CIFAR-100 with CIFAR-10/SVHN OOD episodes and matched baselines.

**Gate:** five-seed protocol passes leakage checks, restart tests, non-inferiority decision, generalized near/far OOD evaluation, and reproducibility audit.

**Implementation status (26 July 2026): COMPLETE.** The frozen seeds `[11, 23, 37, 53, 71]` use source-index-verified MobileNetV2 features over raw CIFAR-100, preserve the official test rows, and partition development data into disjoint geometry, readout-calibration, risk-control, and validation budgets. CIFAR-10 official-test rows provide near-OOD final evaluation; disjoint train-origin rows provide policy validation and risk control. A fixed disjoint split of SVHN official-test features provides the far-OOD protocol. The 5-point balanced-accuracy non-inferiority margin was frozen from the separate seed-42 pilot before confirmatory execution. Mean GEODE balanced accuracy was 65.26% versus 67.33% logistic and 68.13% RBF SVM. The paired one-sided 95% lower bound versus logistic was -2.43 points, so the preregistered non-inferiority gate passed, but GEODE did not lead either control. Maximum probability was selected on validation for all seeds; final near/far AUROC was 0.642/0.805 and FPR95 was 0.808/0.608, exposing weak near-OOD rejection. A real class-stage interruption resumed without feature reload and reproduced assembly hashes and validation metrics exactly. A fresh deployment-seed replay matched model/prediction hashes, metrics, OOD policy, and threshold exactly. Verified candidate bundle `18ab33416dd20c4c37d0` packages explicit geometry, transform, readout, support, config, and evaluation state. The complete 196-test regression gate passed.

### E5. Routing in the complete pipeline

**Deliverables:** exhaustive instrumentation, shadow router interface, per-stage counters, real-model routing matrix.

**Gate:** oracle agreement is reported on every run; no candidate controls outputs until all real-data quality and latency gates pass.

**Implementation status (26 July 2026): COMPLETE.** A measurement-only shadow
interface now returns exhaustive predictions unconditionally while recording
oracle/candidate class-field and primitive counters, bound work, fallback,
winning-score error, agreement, and p50/p95/p99 latency. The deployed E4 bundle
`18ab33416dd20c4c37d0` is reconstructed from explicit JSON/NumPy state and
reproduces its committed seed-11 final-test prediction hash exactly. The real
20-class, 53-primitive, 19-dimensional matrix measured exact-bound, batched
exact-bound, class-major exact-bound, and certified top-5 candidates at batch
sizes 1, 32, 256, and 1,024. Candidate agreement ranged from 95.3% to 100%,
but no route preserved the deployed per-class-normalized winning scores and no
candidate improved p95 latency. At best, bound routing evaluated 94.0% of the
exhaustive class pairs; certified top-5 evaluated 99.5% and fell back on every
sample. The complete-score-vector multinomial readout also prevents partial
class evaluation from replacing exhaustive inference. All 16 cells reported
oracle agreement and counters, none controlled outputs, and none was promotion
eligible. The E5 safety gate passed with exhaustive routing retained; candidate
promotion remains blocked.

### E6. Transfer-learning qualification

**Deliverables:** frozen, linear-probe, fine-tuned/adapter, and GEODE-head variants; parent-child provenance; replay metrics.

**Gate:** one proxy transfer episode improves target performance without exceeding the frozen source-forgetting budget; controlled and external-pretraining lanes remain separately reported.

**Implementation status (26 July 2026): COMPLETE.** A leakage-safe seed-11
CIFAR-100-to-CIFAR-10 class-transfer proxy uses the E4 controlled public
ImageNet MobileNetV2 checkpoint and disjoint target geometry, readout, risk,
validation, and official-test budgets. The immutable source-only variant, raw
frozen-feature linear probe, supervised PCA/LDA adapter plus probe, and adapter
plus GEODE head are reported separately. The GEODE head reached 81.96% balanced
accuracy, improving 71.96 points over chance and passing the frozen 35-point
target-improvement gate, but remained 0.75 points behind the raw linear probe
and 1.02 points behind the adapter probe. Its NLL was 0.569 versus 0.670 for
the raw probe and 0.511 for the adapter probe. Source balanced behavior was not
mutated: source accuracy remained 65.18%, forgetting was exactly zero, and the
before/after prediction hash matched the E4 committed hash. A complete second
GEODE fit reproduced transform, geometry, readout, and prediction hashes.
Verified child bundle `aaa5a73c0b813240876d` records parent
`18ab33416dd20c4c37d0`; publication did not move the parent registry's current
pointer. No independent externally pretrained checkpoint was present, so that
lane is explicitly `not_run` rather than being conflated with the controlled
lane. The proxy transfer, source-forgetting, replay, provenance, and lane-
separation gates passed; GEODE did not lead the matched target controls.

### E7. DomainNet distributed flagship

**Deliverables:** verified DomainNet manifest, Ray execution, object-store artifacts, multi-domain/class episodes, cluster resource report.

**Gate:** local-small and cluster-small runs meet their declared reproducibility level; worker-loss recovery passes; DomainBed-style model selection is frozen; the full run has complete epoch, class-fit, routing, and transfer histories.

**Implementation status (26 July 2026): PARTIAL; MULTI-HOST GATE BLOCKED.** The repository now has a
lazy optional Ray map executor with retry and cluster-resource reporting, a
strict manifest supporting both legacy per-domain shards and mixed-domain
physical source files, and a frozen model-selection contract with disjoint
clipart validation and sketch final domains. The pinned public Hugging Face
revision `ee20570ae7a29c51571e55a9a17983f7625295d6` is downloaded to the external
`GEODE_CACHE_DIR`: four SHA-256-verified Parquet files totaling 18,521,436,207
bytes, with 586,575 declared samples across 345 classes and all six canonical
domains. The default Python 3.14.6 environment remains unsupported by Ray. A
separate Python 3.12.10 environment starts Ray 2.56.1 successfully and reports
16 CPUs. A Docker Compose rehearsal now starts one head and two worker
containers on the local host. Node-affinity tasks executed on all three logical
nodes, two complete 16-task histories matched SHA-256 exactly, and Ray retried
successfully after a worker process terminated deliberately. The same pinned
DomainNet manifest verified all 18,521,436,207 bytes before submission.
Provider-neutral KubeRay manifests require worker topology spread across
distinct Kubernetes hostnames, and a versioned application image is defined for
later cloud use. The local artifact explicitly records one physical host,
process-loss rather than node-loss recovery, `local_simulation_gate_passed:
true`, and `e7_gate_passed: false`. No cluster-small DomainNet training run,
physical worker-node loss recovery, or full epoch/class-fit/routing/transfer
history is claimed. A frozen local-small episode now performs real distributed
DomainNet image decoding on 192 observations spanning eight classes and all
six domains. Forty-nine row-group tasks executed on all three logical nodes;
two complete feature passes matched SHA-256 exactly. A deterministic
nearest-centroid control fitted only on infograph, painting, quickdraw, and
real reached 18.75% clipart validation balanced accuracy and 28.125%
observational sketch balanced accuracy. Those values are bounded systems
evidence, not a flagship or predictive-performance claim. Final resolution
still requires at least two physical Ray hosts. The fail-closed preflight,
local/final gate separation, legacy compatibility, and manifest tamper tests
pass.
hosts. The fail-closed preflight, local/final gate separation, legacy
compatibility, and manifest tamper tests pass.

### E8. Cross-modal flexibility

**Deliverables:** WikiText-103 and ModelNet40 pipelines using the same lifecycle contracts and registry.

**Gate:** both produce loadable fingerprinted bundles and matched baseline reports without dataset-specific checkpoint formats.

**Implementation status (26 July 2026): COMPLETE.** One shared
content-addressed JSON/NumPy bundle now packages an explicit WikiText-103
character model and an explicit point-cloud reconstruction model, and both
replay their prediction hashes exactly through the same loader/registry
contracts. On bounded real WikiText-103 prefixes, the adaptive 32-character
GEODE head retained 93.6% of test observations and reached 24.47% top-1,
58.33% top-5, and 15.07 perplexity. It trailed the matched linear-context
control (33.92%) and sampled 3-gram control (38.57%). The pinned
`jxie/modelnet40-2048` revision contributes 9,840 train and 2,468 test shapes,
40 labels, and 2,048 points per shape. Both source Parquets and the derived NPZ
are SHA-256 verified under `GEODE_CACHE_DIR`. On the bounded eight-shape
reconstruction episode, 29 GEODE experts reduced test mean absolute SDF
residual from 0.2873 for the matched single-sphere control to 0.2767. Bundle
`d0c93fb69431fb8bab7a` verifies and replays both modalities exactly. The
ModelNet40 artifact is not committed to Git, but its pinned provenance,
conversion manifest, and acquisition evidence are.

### E9. Review and transactional adaptation

**Deliverables:** persistent review IDs, delayed feedback linkage, dry-run adaptation, exact rollback, bundle lineage.

**Gate:** final-novel labels remain hidden until evaluation; no mutation publishes without confirmation and all replay/calibration/graph gates.

**Implementation status (26 July 2026): COMPLETE.** A deterministic review
cluster produces persistent review ID `review-ce1ed001f618`; its unlabeled
records contain no final-novel labels, and delayed confirmation
`confirmation-117f7ae34560` links back to that review before any publication.
The coordinator rejects a candidate whose calibration NLL increase exceeds the
frozen 0.02 limit and leaves the active pointer unchanged. The same candidate
passes as a nonpublishing dry run only after replay, calibration, graph, and
label-isolation gates all pass. Confirmed publication creates child bundle
`a151b8f3c1cfa763f677` with exact parent `b5e3f04cf40c9d321a82`, then the
rollback drill restores the parent pointer without rebuilding either bundle.
The append-only registry retains both artifacts and all transaction phases.

### E10. Production rehearsal

**Deliverables:** replicated inference service, shadow routing, canary promotion, telemetry, rollback drill, recovery runbook.

**Gate:** forced bad canary and coordinator loss both recover to the prior production bundle within the declared recovery objective.

**Implementation status (26 July 2026): COMPLETE.** The frozen rehearsal runs
two replicas from content-addressed bundle `377ad7127e3238eea88d`. Inverted
candidate `24c445b170370858dd36` received measurement-only shadow traffic,
produced 0.0 agreement, never controlled an output, and triggered automatic
rollback to the parent in 0.00592 seconds. A second fault discarded the
coordinator after activating candidate `3e5ba13abf9428f703ea` but before
promotion completion. A fresh coordinator recovered the durable promotion
journal and restored the parent in 0.00280 seconds. Both paths satisfy the
frozen one-second RTO and zero-request RPO. Two replicas reloaded the exact
parent, telemetry remained available, and no partial artifact survived. The
recovery runbook records the corresponding operator procedure.

### E11. Final public study

**Deliverables:** frozen configs, five-seed principal results, negative results, artifact index, plots, cost report, and updated research report.

**Gate:** an independent command or clean machine can reproduce summary tables from immutable artifacts without rerunning training.

**Implementation status (26 July 2026): COMPLETE.** The frozen E11 manifest
indexes 27 configurations, acquisition records, and result artifacts across
E1-E10 by SHA-256; E0 is represented explicitly by contract/test evidence
rather than a nonexistent result artifact. One artifact-only command verifies
the lock and regenerates byte-identical JSON, Markdown, and SVG principal
results, negative results, cost evidence, and recovery plots without loading a
dataset or fitting a model. The five-seed table reproduces 65.26% GEODE,
67.33% logistic, and 68.13% RBF balanced accuracy. The publication retains E5
as a negative routing result and E7 as blocked on a second Ray node.

---

## 14. Recommended Repository Shape

Introduce these boundaries incrementally:

```text
src/runtime/
  schemas.py
  run_state.py
  checkpoint.py
  artifact_store.py
  metrics.py
  executor.py
  local_executor.py
  ray_executor.py
  registry.py

experiments/e2e/
  run_episode.py
  evaluate_bundle.py
  fault_injection.py
  build_report.py

experiments/configs/e2e/
  cifar_smoke.json
  cifar_full.json
  domainnet_proxy.json
  domainnet_full.json
  wikitext103.json
  modelnet40.json
```

Do not move existing research modules merely to match this layout. Add adapters around `experiment_manifest`, `FittedModel`, `ModelNetwork`, `Orchestrator`, `ModelFingerprint`, `SupportProfile`, and `ModelEditor`, then migrate experiments one stage at a time.

---

## 15. First Implementation Slice

Implement only E0-E2 on the existing Tier 4 smoke path before adding DomainNet or cluster dependencies:

1. define versioned run, stage, checkpoint, and metric schemas;
2. wrap Tier 4 feature extraction and per-class fitting as idempotent stages;
3. add atomic filesystem artifacts and `SUCCESS` markers;
4. add per-class completion records and resume planning;
5. emit epoch/stage JSONL metrics;
6. add a controlled `--fail-after` hook for tests;
7. compare uninterrupted and resumed artifacts by hash and metrics;
8. retain existing experiment artifacts and manifests for backward compatibility.

This slice is deliberately small. It proves recoverability and metric history on cheap public data before Ray, MLflow, object storage, DomainNet, or a production service are introduced.

Before E1 implementation, add a geometry-feasibility probe to this slice. It computes class counts, transformed dimension, effective rank, condition numbers, and the capacity ladder's parameter counts without fitting a full run. The probe must fail the configuration early when no authorized geometry family is supportable.

---

## 16. Final Deliverables

The completed program should publish:

- a dataset and split manifest for every track;
- immutable configs and environment records;
- epoch/stage learning curves with selection annotations;
- fitted model, fingerprint, support, graph, and routing artifacts;
- baseline, transfer, OOD, open-world, routing, and systems tables;
- restart and fault-injection evidence;
- local-versus-cluster equivalence results;
- a deployable bundle and registry history;
- a canary/rollback rehearsal log;
- an artifact-only report builder;
- an updated research report that distinguishes public-data claims, systems evidence, and synthetic diagnostics.

---

## 17. Final Preflight Audit

### 17.1 Blocking issues found

| Priority | Issue in the previous plan                              | Failure mode                                                                                                    | Required correction                                                                                            |
| -------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| P0       | No representation-to-geometry capacity contract         | full ellipsoids on high-dimensional embeddings are underdetermined, ill-conditioned, or prohibitively expensive | capacity ladder, effective-rank diagnostics, shrinkage/low-rank variants, and early fallback                   |
| P0       | One generic calibration split served several decisions  | optimistic calibration, OOD, or risk-control claims through repeated reuse                                      | separate readout and risk-control calibration budgets, or preregistered cross-fitting                          |
| P0       | Pretrained feature provenance was not a gate            | benchmark overlap or opaque foundation-model data can be mistaken for GEODE learning                            | controlled and external-pretraining lanes with immutable provenance                                            |
| P0       | Domain transfer lacked an explicit model-selection rule | target-domain information can leak through checkpoint or hyperparameter choice                                  | freeze allowed validation domains and selection rule before final domains                                      |
| P1       | CIFAR-100/SVHN OOD is too narrow                        | far-OOD performance can hide near-OOD and semantic-overlap failures                                             | adopt a generalized near/far OOD protocol and stronger post-hoc controls                                       |
| P1       | Repeated seeds were the main statistical language       | seed variance can be mistaken for population or domain generality                                               | primary endpoints, paired intervals, domain-level variation, and multiplicity control                          |
| P1       | Cross-machine equivalence was underspecified            | valid floating-point variation can break hashes, or loose tolerance can hide drift                              | separate replay identity from scientific equivalence                                                           |
| P1       | "End-to-end training" was ambiguous                     | readers may infer joint optimization although greedy construction is discrete                                   | call the first system an end-to-end lifecycle and isolate joint/alternating research                           |
| P2       | Routing work could repeat a closed negative path        | more shortlist logic adds overhead without wall-clock benefit                                                   | optimize packed exhaustive kernels first and require a break-even forecast                                     |
| P2       | Four full tracks risk scope dilution                    | systems work may consume the budget before a defensible scientific result                                       | gate progression: CIFAR qualification, DomainNet flagship, then one cross-modal confirmation before the second |

These issues do not invalidate the architecture. They change the order of work and narrow what each result may claim.

### 17.2 Prior-art improvements to adopt

- **DomainBed-style selection:** treat model selection as part of the algorithm and retain empirical risk minimization as a mandatory transfer baseline [P1].
- **Natural-shift confirmation:** after DomainNet, add one resource-feasible WILDS episode as external validity rather than another synthetic domain split [P2].
- **Generalized OOD protocol:** use OpenOOD's distinction between generalized OOD settings and compare maximum probability, energy, Mahalanobis/kNN, and ViM where their required features/logits exist [P3,P4].
- **Strong frozen features:** include DINOv2 only in the external-pretraining lane; its value is testing whether GEODE benefits from robust generic features, not proving dataset-clean representation learning [P5].
- **Parameter-efficient transfer:** compare a small visual adapter with linear probing and full fine-tuning. AdaptFormer establishes this as a credible resource baseline for vision transformers; LoRA is relevant to the language track [P6,P7].
- **Risk-controlled abstention:** add split conformal or risk-controlling prediction sets as an optional policy over frozen GEODE scores, with a dedicated holdout and explicit exchangeability scope [P8].
- **Regularized geometry:** compare shrinkage covariance and low-rank structure before full covariance; large-dimensional covariance estimation is a known conditioning problem, not merely an implementation detail [P9].
- **GPU exact-search baseline:** use optimized brute-force/selection implementations as the systems baseline before claiming that an approximate index is needed [P10].

### 17.3 Focused novel research hypotheses

The following are experiments, not claimed breakthroughs:

1. **Capacity-adaptive GEODE:** selecting sphere, diagonal, shrinkage, low-rank, or full geometry per class may improve robustness and reduce fitting/routing cost while preserving explicit editability.
2. **Geometry-aware parameter-efficient transfer:** train only a small representation adapter against a differentiable class-field objective after greedy initialization, with source replay and complexity penalties. Compare against cross-entropy adaptation under identical parameter and compute budgets.
3. **Conformalized GEODE review:** use calibrated field-derived nonconformity to turn abstention into a finite-sample risk/coverage experiment on exchangeable episodes, while treating shifted episodes as empirical stress tests.
4. **Compile before route:** pack heterogeneous primitives into shape buckets and fuse exact evaluation; candidate routing is useful only after arithmetic and memory overhead are no longer dominant.
5. **Structural transfer diagnostic:** measure whether transferred representations reduce the number of primitives and improve condition numbers at matched accuracy. This directly tests whether transfer makes class geometry simpler, rather than only improving a classifier head.

Only hypotheses 1 and 5 belong in the initial scientific path. Hypotheses 2-4 remain gated follow-ups so the core study does not become an unbounded methods project.

### 17.4 Final recommendation

Proceed with E0-E2 only after the four P0 contracts are encoded in schemas and tests. Then run one cheap CIFAR episode to choose a supportable geometry family and validate restart behavior. Do not start full DomainNet training until that episode passes generalized OOD, model-selection, calibration-separation, and reproducibility checks. After DomainNet, run one WILDS or cross-modal confirmation based on which claim remains least supported; do not automatically run every track.

### 17.5 Prior-art sources

[P1] I. Gulrajani and D. Lopez-Paz. “In Search of Lost Domain Generalization.” _ICLR_, 2021. https://arxiv.org/abs/2007.01434

[P2] P. W. Koh et al. “WILDS: A Benchmark of in-the-Wild Distribution Shifts.” _ICML_, 2021. https://arxiv.org/abs/2012.07421

[P3] J. Yang et al. “OpenOOD: Benchmarking Generalized Out-of-Distribution Detection.” _NeurIPS Datasets and Benchmarks_, 2022. https://arxiv.org/abs/2210.07242

[P4] H. Wang et al. “ViM: Out-Of-Distribution with Virtual-logit Matching.” _CVPR_, 2022. https://arxiv.org/abs/2203.10807

[P5] M. Oquab et al. “DINOv2: Learning Robust Visual Features without Supervision.” arXiv:2304.07193, revised 2024. https://arxiv.org/abs/2304.07193

[P6] S. Chen et al. “AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.” _NeurIPS_, 2022. https://arxiv.org/abs/2205.13535

[P7] E. J. Hu et al. “LoRA: Low-Rank Adaptation of Large Language Models.” _ICLR_, 2022. https://arxiv.org/abs/2106.09685

[P8] S. Bates et al. “Distribution-Free, Risk-Controlling Prediction Sets.” arXiv:2101.02703, 2021. https://arxiv.org/abs/2101.02703

[P9] O. Ledoit and M. Wolf. “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices.” _Journal of Multivariate Analysis_, 2004. https://doi.org/10.1016/S0047-259X(03)00096-4

[P10] J. Johnson, M. Douze, and H. Jégou. “Billion-Scale Similarity Search with GPUs.” _IEEE Transactions on Big Data_, 2019. https://arxiv.org/abs/1702.08734
