# GEODE: Geometric Expert Orchestration for Discovery and Evaluation

GEODE is a research system for constructing, evaluating, and operating explicit
geometric expert models. The name no longer assumes that every expert must be
ellipsoidal or that subtraction must be enabled. The current implementation is
built primarily around oriented covariance primitives, including full,
diagonal, and spherical families, composed into class fields with calibrated
readouts. The interfaces are intended to admit other explicit primitive
families as the research evolves.

The research goal is deliberately narrower than end-to-end representation
learning:

> Given a frozen commodity representation, test whether an explicit geometric
> expert head can match the best black-box head on the same features while
> providing measured edit locality, exact rollback, and audited adaptation that
> the black-box head cannot provide.

GEODE therefore follows a **frozen trunk, trained interface** design. A strong
self-supervised backbone such as DINOv2 or SigLIP is treated as an immutable,
versioned, hash-addressed artifact. An optional linear or low-rank affine
interface may be trained once under the pre-test development protocol, then
frozen before any geometry is fitted. All subsequent discriminative training is
restricted to explicit head objects such as component centers, metrics, and
temperatures.

Joint encoder-head gradient training is an explicit non-goal. A moving encoder
would invalidate the stable feature space required by component provenance,
changed-region measurements, cached calibration and support objects, exact
replay, and rollback. When a representation must change, GEODE treats that
change as a versioned migration: fit a new bundle in the new frozen space,
report component correspondence and edit survival, and retain rollback to the
previous bundle.

Earlier reports expand GEODE as **Greedy Ellipsoidal Outline Discrimination by
Excision**. That name describes the original implementation, but the current
expansion reflects the broader primitive, discovery, evaluation, and lifecycle
scope while preserving the established project name.

The repository covers more than a classifier kernel. It includes deterministic
experiment protocols, resumable training stages, immutable model bundles,
review-gated adaptation, routing experiments, distributed rehearsal, and
artifact-only publication reproduction.

## Current Status

The completed evidence supports GEODE as an inspectable geometric modeling and
lifecycle-control system. It does **not** establish state-of-the-art predictive
performance.

| Area                   | Current result                                                                                                                                                               |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CIFAR-10               | GEODE multinomial reached 83.11% across five seeds, versus 84.02% logistic regression and 84.00% RBF SVM.                                                                    |
| Subtractive CSG        | Five-seed ablations found no aggregate accuracy benefit. Subtraction remains optional and validation-gated.                                                                  |
| WikiText-103           | The locked temporal study reached 30.36% top-1, below the 34.64% linear and 44.50% matched 5-gram controls.                                                                  |
| CIFAR-100 superclasses | E4 reached 65.26% balanced accuracy and passed its non-inferiority gate, while logistic and RBF controls reached 67.33% and 68.13%.                                          |
| E5 routing             | Real-bundle and synthetic candidates failed promotion. Exhaustive exact class-field evaluation remains authoritative.                                                        |
| Transfer               | E6 preserved the source model exactly and reached 81.96% on the target proxy, below the 82.71% raw linear and 82.97% adapter controls.                                       |
| Distributed execution  | E7 processed a bounded 192-image DomainNet episode on three logical Ray nodes with exact replay and worker-process recovery. Physical multi-host qualification remains open. |
| Cross-modal bundles    | E8 packages text and point-cloud models under one immutable bundle contract with exact replay.                                                                               |

Head comparisons will use the same frozen features for linear, RBF, prototype,
Gaussian-mixture, compact-MLP, and GEODE heads. DINOv2 and SigLIP are the planned
primary image backbones. The geometric head will be initialized by the
deterministic constructor and may then train explicit component parameters
discriminatively inside the frozen space. Predictive parity, superiority,
editability, and migration are separate gated claims.

The planned representation-migration study will refit geometry under a new
frozen embedding and measure component matches, splits, merges, unmatched mass,
accepted-edit survival, derived-artifact invalidation, and byte-exact rollback.
This capability is planned work and is not yet a supported repository claim.

## Current Limitations

- GEODE has not beaten the strongest matched controls on the principal public
  classification or temporal studies.
- The strongest completed image results depend on pretrained MobileNetV2
  features; the frozen DINOv2/SigLIP comparison has not yet run.
- Accuracy parity with a compact black-box head on strong frozen features has
  not yet been established.
- Frozen affine interfaces, discriminative fixed-space head training, and
  audited representation migration remain planned M17-M26 work, not completed
  capabilities.
- M16 provides representation-lineage and fail-closed compatibility contracts,
  but existing production components, calibration objects, support profiles,
  and feature caches have not all been migrated to them.
- M18's support-dependent metric policy did not beat the best frozen single
  family and achieved no median parameter-byte reduction. The new metric
  families remain opt-in; the existing primitive-family default is unchanged.
- Subtractive CSG has not shown aggregate public-data benefit.
- OOD FPR95 remains too high for a deployment claim.
- Review-first grouping surfaces unknown events more reliably than it recovers
  semantic classes.
- Candidate routing has not improved wall-clock latency over exhaustive
  inference.
- Physical multi-host training and failure recovery remain unqualified.

## Research Roadmap

The immediate program is:

1. **M16 — complete:** preregistered protocol, artifact schemas,
   representation lineage, compatibility guards, migration-report contract,
   paired statistics, Pareto logic, and byte-identical S0 replay.
2. **M18 — complete, stopped:** six metric families and the crossover map are
   available, but the adaptive policy failed predictive and resource gates.
3. **M19 — next:** acquire and fingerprint frozen DINOv2 and SigLIP features; train any
   affine interface once, freeze and hash it, then compare heads in each fixed
   space.
4. **M17:** train centers, metrics, responsibilities, and temperatures
   discriminatively without updating the representation or interface.
5. **M20-M23:** evaluate transactional topology, frozen task-native temporal
   representations, confidence decomposition, and evidence-triggered local
   residuals.
6. **M24:** measure the accuracy-editability Pareto frontier against matched
   prototype, mixture, tree, RBF, and compact-MLP controls.
7. **M26:** run representation migrations with component correspondence,
   accepted-edit survival, stale-artifact invalidation, and exact rollback.
8. **M25:** independently confirm only retained variants and update claims from
   locked artifacts.

No milestone may make the encoder train jointly with the head or weaken an
existing replay, calibration, rollback, or promotion gate. Broader
generalized-category-discovery, OOD, routing, and physical multi-host work
remains valuable, but it does not replace the central fixed-space head-parity
and editability test.
