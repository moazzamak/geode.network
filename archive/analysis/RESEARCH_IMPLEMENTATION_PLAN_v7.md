# GEODE Research Implementation Plan v7

**Status:** preregistration draft after completed v6.1 Outcome D  
**Date:** 26 July 2026  
**Source:** user-supplied prior-art analysis of the six-stage open-world loop  
**Parent evidence:** `analysis/CLAIM_LEDGER_v6_1.md` and
`logs/results/v6_1/final_artifact_replay/`

## 1. Program pivot

v6/v6.1 tested whether an explicit geometric head could reach same-space
predictive parity. It did not: the retained weighted affine head reached 91.73%
development balanced accuracy versus 96.77% for RBF, leaving a 5.03-point gap.
That final Outcome D remains immutable. v7 does not reopen predictive rescue,
retune the failed branch, or use an open-world result to revise v6.1.

v7 instead tests a different claim:

> Can GEODE operate a review-gated open-world recognition loop that joins
> calibrated rejection, persistent unlabeled-group discovery, human semantic
> confirmation, transactional class adaptation, empirical cross-model routing,
> and exact rollback while making the acceptance-head choice substitutable?

The contribution under test is the **operational composition and audit trail**,
not invention of open-world recognition, novelty discovery, expert routing, or
any individual rejection mechanism. SDF geometry is one ablation arm. It is not
the premise, privileged default, or required winner.

## 2. Prior-art and novelty boundary

M38 must verify and freeze the literature through the protocol date before any
novelty wording is permitted. At minimum, the audit covers:

- NNO/open-world recognition, OpenMax, and the Extreme Value Machine;
- ECSMiner, SAND, ECHO, MINAS, and adjacent stream-novelty systems;
- Expert Gate and modern expert/adapter routing;
- ORCA, OpenLDN, NACH, GCD, incremental GCD, and continual GCD;
- ART/ARTMAP, fuzzy ART, SOINN, SENC, and open-world lifelong learning;
- kNN OOD, Mahalanobis/GMM/DDU, DUQ/SNGP, Deep SVDD, and related
  distance-aware uncertainty methods.

Until that audit is complete, the strongest admissible positioning is:

> GEODE evaluates an engineered composition of established open-world stages
> with explicit review, immutable provenance, transactional publication, and
> cross-model routing.

The plan must not claim:

- invention of the detect-buffer-cluster-label-update loop;
- that no 2025-2026 system closes the full loop;
- that an SDF is necessary for open-world recognition;
- semantic novelty from an unlabeled cluster;
- shift-robust conformal guarantees without exchangeability;
- routing safety from model IO fingerprints alone;
- autonomous class creation or live mutation.

## 3. Frozen evidence and reusable machinery

| Existing evidence or mechanism | Frozen result | v7 consequence |
| --- | --- | --- |
| M7 OOD | max probability ranked well, but FPR95 remained 0.579/0.431 | retain posterior controls; do not treat raw SDF sign as novelty |
| M10 open set | no cell passed 90% known coverage and 50% unknown recall across all episodes | retain this as the autonomy gate; add a separate review-only gate |
| M11 discovery | corrected partitions reached 61.1% distinct-group recall; full two-group recovery in 4/9 cells | use as the internal discovery baseline |
| M11 policy | stable review IDs and confirmation-gated actions already exist | extend rather than replace the safety model |
| M12/E5 routing | tested candidate routers did not justify authoritative sparse routing | all new routing begins shadow-only with exhaustive fallback |
| E3/E9/E10 | immutable bundles, confirmation, publication, rollback, and recovery passed | reuse exact transaction contracts |
| v6.1 A3 | rollback passed but 99.9% edit locality failed | do not claim lifecycle locality without a new passing measurement |
| v6.1 Outcome D | explicit head trails RBF by 5.03 points | compare acceptance heads independently of the closed-set classifier |

Reusable code includes `src/open_set.py`, `src/rejection_buffer.py`,
`src/discovery_clustering.py`, `src/streaming_discovery.py`,
`src/adaptation_policy.py`, `src/runtime/adaptation_transaction.py`,
`src/model_fingerprint.py`, `src/shadow_routing.py`, and the immutable bundle
runtime.

Two concepts must remain distinct:

1. **compatibility fingerprint:** the current hash-bound task and IO contract
   used for safe graph wiring and swappability;
2. **empirical routing profile:** the data-derived `SupportProfile` lineage,
   extended with a typed routing summary used to estimate which model should
   inspect a sample.

The second may never replace or weaken the first.

## 4. Primary hypotheses

### H1: rejection is substitutable

An EVM-style margin-tail model, low-rank Gaussian/DDU-style density, or kNN
support score will outperform raw SDF support on rejection quality over the same
frozen representation. An editable SDF may remain useful if its lifecycle value
compensates for a measured quality cost.

### H2: discovery quality depends on the rejection stream

Better calibrated rejection will materially improve persistent-group recall and
review precision even with the same clustering policy. Discovery must therefore
be evaluated jointly with, but causally separated from, the acceptance head.

### H3: human confirmation is a semantic boundary

Persistence and cohesion can create a stable review object, not a class name.
Only a linked human confirmation may authorize evaluation of `UPDATE_EXISTING`
or `CREATE_NEW`; publication remains subject to replay, calibration, OOD,
graph, and rollback gates.

### H4: empirical support profiles can route without hiding unknowns

A data-derived routing profile can reduce model evaluations while preserving
the exhaustive oracle and retaining an explicit no-compatible/unknown path.
Expert Gate reconstruction error is the principal routing baseline.

### H5: composition adds measurable value

The complete review-gated loop will reduce review burden or time-to-safe-class
addition relative to disconnected stage baselines without degrading known-class
accuracy, unknown detection, auditability, or rollback.

## 5. Shared experimental contract

### 5.1 Frozen representation

- Use released, frozen, hash-addressed DINOv2 features as the primary space.
- No encoder fine-tuning, joint representation learning, or test-time feature
  adaptation is allowed in the primary causal matrix.
- ORCA/IGCD-style methods that require representation training must be reported
  as separate end-to-end references, not as representation-matched controls.
- A frozen-feature semi-supervised k-means/GCD control is mandatory.

### 5.2 Data stages

| Stage | Purpose | Data access |
| --- | --- | --- |
| S0 | API, schema, numerical, replay, and leakage tests | deterministic synthetic streams |
| S1 | cheap mechanism falsification | one frozen development stream and seed 11 |
| S2 | retained stage-wise evaluation | frozen seeds 11, 23, 37 |
| S3 | integrated development loop | three frozen class/domain arrival schedules |
| S4 | independent confirmation | untouched schedule and clean run environment |

Primary public-data tracks:

1. **CIFAR-100 class stream:** known classes, held-out novel classes, known-class
   extensions, corruptions, and recurrence in a frozen DINOv2 space;
2. **DomainNet model-routing stream:** separately owned domain/task bundles,
   held-out routing episodes, and explicit no-compatible-expert cases;
3. **synthetic causal stream:** independently controlled novelty, shift,
   corruption, class overlap, recurrence, and label delay.

The exact class identities, episode order, arrival rates, and S4 schedule hashes
must be frozen at M38. Final novel labels remain sealed except at simulated
human-review events and final scoring.

### 5.3 Simulated human policy

Ground-truth labels may enter the system only through a deterministic review
oracle that records:

- review ID and immutable member sample IDs;
- query time and label budget consumed;
- response: existing class, new class, corruption/irrelevant, or unresolved;
- confirmed label, confirmation kind, and timestamp.

The oracle simulates a human for reproducible experiments; it does not make the
system autonomous. Development labels may tune registered policies only on the
development schedule. S4 labels remain sealed until confirmation.

### 5.4 Required controls

**Rejection**

- maximum softmax probability;
- kNN feature distance;
- class-conditional Mahalanobis or low-rank Gaussian density;
- EVM-style margin-tail model;
- current calibrated SDF support;
- retained RBF SVM decision evidence.

OpenMax and DDU are included when their semantics can be reproduced faithfully
under the frozen-representation contract. Otherwise they are external reference
results and the limitation is explicit.

**Discovery**

- no grouping;
- existing streaming micro-clusters;
- HDBSCAN and FINCH;
- frozen-feature semi-supervised k-means/GCD;
- an ORCA/IGCD-family reference using released code when protocol-compatible.

**Routing**

- exhaustive all-compatible-model evaluation;
- compatibility-only centroid/radius shortlist;
- Expert Gate-style undercomplete autoencoder reconstruction error;
- empirical prototype/density fingerprint;
- semantic routing as a descriptive control only, never as novelty evidence.

**Adaptation**

- quarantine/no update;
- full retraining from the frozen support set;
- update existing class;
- create new class and migrate the graph;
- SDF structural update where supported;
- winning non-SDF head's native incremental update.

### 5.5 Metrics

Every stage reports:

- balanced known-class accuracy and coverage;
- unknown recall at 90% known coverage;
- AUROC, AUPR, FPR95, NLL, Brier score, and ECE where defined;
- review precision, review recall, precision at fixed review budget, and labels
  requested per true novel class;
- distinct-group recall, pairwise F1, adjusted Rand index, normalized mutual
  information, fragmentation, merging, recurrence recovery, and review-ID
  stability;
- time/windows to first useful review and to confirmed class maturity;
- routing top-1 accuracy, compatible-expert recall, unknown/no-route recall,
  candidate count, fallback rate, p50/p95/p99 latency, and exhaustive agreement;
- pre/post known accuracy, novel-class accuracy, OOD recall, calibration,
  forgetting, transaction success, rollback exactness, and recovery time;
- stored scalars, serialized bytes, support samples, fit/update work, and wall
  time.

All intervals are seed-paired or per-example paired as appropriate. Report
effect sizes and confidence intervals, not only gate booleans.

## 6. Safety and causal rules

1. Rejection, clustering, semantic confirmation, mutation, and routing are
   separate causal stages and receive separate artifacts.
2. Rejected does not mean novel; clustered does not mean semantic; confirmed
   does not mean safe to publish.
3. Before confirmation, groups are non-semantic `review_id` objects.
4. No development or final label may influence online cluster membership before
   its registered review event.
5. Missing comparator capabilities are `unsupported`, never zero-cost.
6. Every support and routing profile is bound to representation,
   model, class-order, training-data, calibration-data, and policy hashes.
7. A stale profile or fingerprint fails closed to exhaustive evaluation and
   review/quarantine.
8. Sparse routing is shadow-only until its own gate passes; exhaustive inference
   remains authoritative and is the mandatory fallback.
9. Every adaptation is first a dry run against the immutable parent bundle.
10. Publication requires human confirmation, replay, known-class preservation,
    unknown-recall preservation, calibration, graph compatibility, and rollback.
11. No v7 result can revise v6.1 predictive Outcome D.
12. Physical E7 qualification remains independent.

## 7. Milestone map

| Milestone | Question | Mandatory output/gate |
| --- | --- | --- |
| M38 | Is the claim and protocol boundary current and reproducible? | literature lock, parent lock, schedules, schemas, byte-identical S0 |
| M39 | Which acceptance head best serves the loop? | matched rejection-quality/cost/editability matrix |
| M40 | Which rejection-plus-discovery pair produces useful review objects? | budgeted review and group-recovery gate |
| M41 | Can confirmation safely expand or update a model? | transaction, graph migration, replay, and rollback gate |
| M42 | Can empirical support profiles route safely across bundles? | Expert Gate comparison and shadow-routing gate |
| M43 | Does the complete loop outperform disconnected baselines? | end-to-end factorial and ablation ledger |
| M44 | Does the retained loop independently confirm? | untouched schedule, artifact-only reproduction, final claim |

## 8. M38: claim, literature, and protocol lock

### 8.1 Literature lock

Verify primary sources and search work published through July 2026. Produce a
stage-by-stage attribution table recording whether each system includes:

1. explicit known-class rejection;
2. reject buffering;
3. persistent unlabeled grouping;
4. human semantic confirmation;
5. incremental update or class creation;
6. empirical routing across separately owned models;
7. immutable audit and exact rollback.

Absence claims require a documented search query, date, databases, and screening
rule. If a prior system closes all seven stages, remove any composition-rarity
claim and reposition v7 as a lifecycle/audit replication and ablation study.

### 8.2 Protocol lock

Freeze:

- parent evidence hashes and v6.1 Outcome D;
- representation, data, split, schedule, class-order, and seed hashes;
- review budgets and label-delay schedules;
- all rejection, clustering, routing-profile, and transaction configurations;
- comparator support status;
- resource ceilings and stop conditions;
- final-label access audit.

### 8.3 S0 requirements

- canonical schemas for acceptance heads, empirical routing profiles, review
  histories, confirmations, and graph migrations;
- deliberate lineage, class-order, threshold, and profile mismatches fail closed;
- synthetic episodes cover unknown class, known extension, corruption, drift,
  recurrence, and no-compatible-expert cases;
- two complete artifact-only S0 runs are byte-identical;
- no training or final labels are loaded during artifact replay.

M39 cannot start until M38 passes.

### 8.4 Result

**Complete; M39 opened.** The audit found no verified system demonstrating all
seven registered stages, so Outcome E did not fire. The admissible claim remains
an engineered composition of established stages, not invention of the loop or
proof of universal absence. The audit records unresolved patent, non-English,
closed-source industrial, rate-limit, and partial-full-text risks in
`analysis/PRIOR_ART_AUDIT_v7.md`.

The protocol lock verified three immutable v6.1 parent files and froze three
schedule families across S0-S4. Acceptance-head, empirical-routing-profile,
review, confirmation, and graph-migration contracts passed deliberate lineage,
class-order, stale-profile, semantic-boundary, and rollback-parent rejection
tests. Two complete S0 runs produced byte-identical artifacts while reporting
`training_data_loaded: false` and `final_labels_opened: false`. Evidence is under
`logs/results/v7/m38_protocol_lock/`.

## 9. M39: acceptance-head bakeoff

### 9.1 Matched head contract

Fit every head on identical frozen features and labels. Select thresholds only
on the frozen calibration split. Hold the downstream reject buffer and discovery
policy inactive so M39 measures rejection rather than cluster feedback.

Required arms:

1. maximum posterior;
2. kNN distance/support;
3. low-rank per-class Gaussian density with priors and log determinants;
4. EVM-style Weibull margin tails;
5. calibrated weighted-affine SDF support;
6. RBF SVM decision score or calibrated posterior.

For each arm, expose one versioned scalar novelty score, one threshold policy,
one per-class candidate score where meaningful, and an incremental-update
contract or explicit `unsupported` status.

### 9.2 Editability audit

Do not assume SDF editability. Apply the same frozen operations where meaningful:

- add support;
- remove support;
- split/merge a component or local support group;
- delete a class-local object;
- exact rollback.

Report changed predictions, unaffected-prediction preservation, update work,
stored state, support-example dependence, and semantic transparency. Operations
that do not exist natively remain unsupported.

### 9.3 Gates

**Autonomy-qualified rejection** requires every retained S2 seed/episode to
reach at least:

- 90% known coverage;
- 50% unknown recall at that coverage;
- no more than 1.0 percentage point known balanced-accuracy loss;
- calibrated replay and deterministic update/rollback.

This preserves the historical M10 contract. Failure closes autonomous action but
does not close review-only evaluation.

**Review-qualified rejection** requires:

- precision at the frozen review budget no worse than the best non-geometric
  control by more than 2.0 points;
- unknown recall at that budget at least 10 points above the historical M11
  maximum-probability transfer result;
- no material failure on corruption or known-extension controls;
- no resource ceiling violation.

Advance at most:

- the best rejection-quality head;
- the best compact incrementally updatable head if different;
- the SDF arm only if it is within 5 points of the best review precision or
  establishes a measured lifecycle advantage.

If no arm passes the review gate, stop v7 before discovery integration and
publish rejection as the binding failure.

### 9.4 M39 result

**Complete; M40 opened with the low-rank Gaussian only.** On the frozen
leave-two-classes-out DINOv2 proxy across seeds 11, 23, and 37, the rank-32
per-class Gaussian reached 92.08% mean known coverage, 86.17% mean unknown
recall, 0.9561 mean AUROC, and 98.67% mean precision at the 50/1000 review
budget. Every seed passed the 90%/50% autonomy operands, accepted-known
balanced accuracy did not regress, exact refit/replay held, serialized state
remained below 0.8 MiB, and the registered 1% feature-noise corruption control
increased false rejection by at most 0.125 points.

kNN, maximum posterior, and weighted-affine SDF showed useful rejection signal
but failed at least one seed-level autonomy or best-control review-precision
operand. The EVM-style approximation and calibrated RBF evidence failed
rejection outright on this proxy. The SDF arm averaged 90.75% coverage, 53.00%
unknown recall, 0.8763 AUROC, and 64.67% review precision; it therefore does
not advance on either quality or a demonstrated lifecycle advantage. These
results are proxy-unknown evidence only: CIFAR-10 classes 8 and 9 were withheld
from fitting, and no final labels were opened.

## 10. M40: persistent discovery and review utility

### 10.1 Factorial

Cross each retained M39 head with:

1. existing streaming micro-clusters;
2. HDBSCAN;
3. FINCH;
4. frozen-feature semi-supervised k-means/GCD.

The no-clustering arm measures whether grouping adds value. ORCA/IGCD-family
released methods are reported separately when their representation/training
contract differs.

### 10.2 Review-state machine

Every candidate group moves only through:

`emerging -> established -> review_requested -> confirmed/quarantined/expired`

Cluster IDs and review IDs are stable across windows. Split, merge, recurrence,
and expiry events are explicit lineage edges, never silent replacements.

### 10.3 Gate

A pair advances only if, over all S2 schedules:

- distinct-group recall exceeds the frozen M11 61.1% baseline by at least
  5.0 points or matches it with at least 25% fewer reviewed samples;
- review precision at the frozen budget is at least 70%;
- full recovery occurs in at least 6/9 matched cells;
- known extensions and corruptions are not mislabeled as new classes before
  confirmation;
- review-ID continuity is 100% under exact replay;
- memory, latency, and eviction limits pass.

Retain one primary pair and one mechanistically distinct control. If no pair
passes, stop before class adaptation.

### 10.4 M40 result

**Complete; M41 opened with HDBSCAN primary and FINCH control.** The retained
M39 Gaussian generated bounded rejection streams over five deterministic
windows for each of seeds 11, 23, and 37. HDBSCAN recovered both withheld
groups in every schedule, reached 100% review precision, passed all 9/9 matched
recovery/safety cells, and replayed review IDs exactly. FINCH provided the
mechanistically distinct control with 83.33% mean distinct-group recall,
86.39% review precision, and 8/9 matched cells. Frozen-feature k-means also
passed (100% recall, 98.67% precision, 9/9) but was not retained because the
two-arm cap favored FINCH as the distinct control.

The streaming micro-cluster arm produced high-precision reviewed samples but
failed distinct-group recovery; no-clustering failed both utility operands.
No arm published a semantic class before confirmation, no buffer eviction
occurred, retained-arm memory stayed below 0.75 MiB, and maximum observed
window latency was 0.099 seconds. Review and cluster identifiers were stable
under exact replay; lineage growth and expiry remained explicit events.

## 11. M41: human confirmation and transactional adaptation

For each retained review object, the frozen oracle returns one of the four
allowed outcomes. Evaluate both `UPDATE_EXISTING` and `CREATE_NEW` dry runs when
the confirmation permits them.

### 11.1 Existing-class expansion

Compare:

- no update;
- native update of the winning non-SDF head;
- SDF component insertion or boundary edit;
- full class-local refit;
- full-model retraining control.

### 11.2 New-class creation

Class creation must atomically update:

- class order and score width;
- calibration and support profiles;
- compatibility fingerprints and empirical routing profiles;
- graph nodes and downstream dimensions;
- bundle provenance and rollback parent.

### 11.3 Gate

An adaptation type advances only if:

- the confirmed target metric improves by at least 5.0 points;
- known-class balanced accuracy drops no more than 1.0 point;
- unknown recall drops no more than 2.0 points;
- NLL does not materially regress under the frozen policy;
- replay is byte-identical;
- graph validation has zero issues;
- publication without confirmation is rejected;
- rollback restores the exact parent bundle and predictions;
- final novel labels remain hidden outside registered review/scoring events.

Passing rollback alone is reported as rollback qualification, not edit locality
or adaptation advantage.

### 11.4 M41 result

**Complete; M42 opened for confirmed new-class creation only.** A compact
rank-16 affine component created from 50 confirmed review samples improved
held-out new-class success from 0% to 42%, 52%, and 36% on seeds 11, 23, and
37 (43.33 points mean). Known balanced accuracy remained 95.00--96.63%,
remaining-unknown recall was unchanged on every seed, and known NLL did not
move at reported precision. Atomic class-order migration, confirmation linkage,
graph validation, byte-identical replay, and exact parent rollback all passed.
An unconfirmed mutation was rejected.

This is a **density-calibrated affine-component result**, not evidence that a
raw SDF field is cross-class comparable. The registered rank-32 native
Gaussian, class-local refit, and full-retraining controls improved the new-class
target by only 0.67 points on average. Existing-class expansion is closed:
native/class-local/full refits averaged 4.17 points and passed only seed 23,
while component insertion averaged zero. M42 may therefore route the confirmed
new class but may not claim general boundary expansion.

## 12. M42: empirical support-profile routing

### 12.1 Fingerprint contract

Extend the existing versioned `SupportProfile` with a typed empirical-routing
payload, or add a sibling profile only where reconstruction semantics cannot fit
that contract. Do not overload `ModelFingerprint`. Candidate families:

- class/task centroids and radii;
- low-rank Gaussian or density summary;
- compact support prototypes;
- undercomplete autoencoder reconstruction profile (Expert Gate control).

Every routing profile records fit-data lineage, representation hash, output class
order, dimensions, score direction, threshold, serialized size, and update
policy.

### 12.2 Routing protocol

For every sample:

1. enforce IO compatibility;
2. score every compatible empirical routing profile;
3. shortlist or return no-compatible/no-confident-route;
4. run exact model inference on shortlisted bundles;
5. compare against exhaustive compatible-model inference;
6. fall back to exhaustive evaluation whenever confidence or lineage fails.

Semantic descriptors may break ties after empirical compatibility; they may not
override unknown evidence or create a class.

### 12.3 Gate

Authoritative routing remains closed unless all S2 routing episodes pass:

- at least 99.0% correct-bundle top-1 routing;
- at least 99.9% inclusion of the exhaustive winning bundle in the shortlist;
- at least 99.9% final prediction agreement after fallback;
- no-compatible/unknown recall at least 95%;
- stale routing profiles produce 100% fail-closed fallback;
- at least 25% fewer exact model evaluations;
- candidate p95 latency is lower than exhaustive p95 latency;
- empirical routing-profile updates replay exactly after M41 adaptation.

Otherwise routing remains shadow-only and M43 uses exhaustive evaluation.

### 12.4 M42 result

**Complete; routing remains shadow-only and M43 must use exhaustive
evaluation.** No empirical profile passed all three seeds. Centroid/radius was
best on correct-bundle top-1 at 93.53% but reached only 31.33% no-confident
unknown recall. Low-rank Gaussian was best on exhaustive-winner inclusion
(99.57%) and unknown recall (71.00%) but still missed the 99.0%, 99.9%, and
95.0% gates. Compact prototypes and autoencoder reconstruction were weaker.

All profile families replayed their post-M41 update exactly, stale profiles
fell back exhaustively, and mean exact-model evaluation reduction ranged from
49.99% to 54.47%. However, no family lowered candidate p95 latency on every
seed, and final agreement ranged from 98.71% to 99.57%, below the 99.9%
contract. These are useful shadow-routing diagnostics only; no profile may
replace exhaustive compatible-model evaluation.

## 13. M43: integrated loop factorial

Run a minimal identifiable factorial:

- acceptance head: best rejection head vs SDF if retained;
- discovery: retained pair vs no grouping;
- review: delayed confirmation vs no confirmation;
- routing: retained empirical router vs exhaustive;
- adaptation: native incremental update vs no update/full retrain.

Do not combine a mechanism that failed its stage gate. Hold schedules, review
budget, labels, and representation fixed.

### 13.1 Primary endpoint

The primary endpoint is safe novel-class integration under a fixed human budget:

- confirmed novel classes correctly integrated;
- labels/reviews consumed;
- windows to integration;
- post-integration known and novel balanced accuracy;
- unknown recall and false creation count;
- exact rollback and audit completeness.

### 13.2 Full-loop gate

The integrated loop advances to M44 only if it:

- integrates at least 75% of confirmable novel classes within the frozen horizon;
- produces zero unconfirmed semantic publications or model mutations;
- produces zero false autonomous class creations;
- reduces reviewed samples by at least 25% versus reject-everything review;
- preserves known balanced accuracy within 1.0 point of the frozen parent;
- preserves unknown recall within 2.0 points of the retained M39 head;
- passes every publication, graph, replay, rollback, and fallback contract;
- is non-dominated by the disconnected stage baselines on integration success,
  review burden, known accuracy, and update work.

Report the SDF arm as an ablation. If it loses rejection and integrated utility
without a measured lifecycle benefit, remove SDF-specific wording from the
primary open-world claim.

### 13.3 M43 result and M44 disposition

**Complete; M43 failed, M44 is blocked, and v7 ends at Outcome C.** Among the
stage-qualified discovery arms, FINCH was non-dominated on review burden but
integrated 0/3 confirmable classes; HDBSCAN also integrated 0/3. FINCH reduced
reviewed samples by 93.49% versus reviewing every rejection and preserved known
accuracy exactly, but mean remaining-unknown recall was 79.33%, 6.83 points
below the frozen M39 result. All confirmation, publication, graph, exhaustive
fallback, replay, rollback, and audit contracts passed.

The no-grouping reject-everything control integrated all three proxy classes
only by reviewing every rejection (223, 231, and 253 samples), so it failed the
25% review-reduction endpoint. The retained grouping arms supplied purer,
smaller review sets, but those sets did not support the M41 adaptation gain:
HDBSCAN held-out target success was 4%, 22%, and 20%; FINCH never reached the
30% integration threshold. Stage-wise success therefore does not compose into
a closed operational loop.

M44 cannot open because no M43 winner exists to freeze. The final claim is
limited to **Outcome C -- stage-wise lifecycle qualification**. SDF-specific
wording is removed from the primary open-world claim: weighted-affine SDF
failed M39 retention, and the M41 compact affine gain used density-calibrated
scoring without an integrated lifecycle advantage.

## 14. M44: independent confirmation and publication

Freeze the M43 winner before opening S4. Run one clean command in a fresh output
directory on the untouched schedule. No thresholds, cluster policies, review
budgets, routing profiles, or update rules may change.

M44 passes only if:

- the M43 full-loop gate repeats;
- the same qualitative head/routing conclusions hold;
- every artifact verifies against its index;
- a second artifact-only reproduction is byte-identical and loads no training
  data or sealed labels;
- all stopped, unsupported, blocked, and failed branches are published.

**Blocked, not run.** M43 produced no passing winner, so opening an untouched
confirmation schedule would violate the preregistration. The final stage-wise
verifier instead checked six immutable milestone indexes, 16 artifacts, and
eight Outcome-C conclusion operands twice with byte-identical output. It loaded
no training data and opened no final labels. The complete stopped/failed/blocked
ledger is published in `analysis/V7_FINAL_CLAIM_LEDGER.md`.

## 15. Kill switches and outcomes

### Kill switches

1. **M39 failure:** no head passes review-qualified rejection. Stop the program.
2. **M40 failure:** rejection does not yield useful persistent review groups.
   Stop before semantic adaptation.
3. **M41 failure:** confirmation-gated update cannot preserve replay, calibration,
   graph safety, and rollback. Retain review-only operation.
4. **M42 failure:** retain exhaustive routing; do not block M43 if all other
   stages pass.
5. **M43 failure:** publish stage-wise qualifications only; do not claim a closed
   operational loop.
6. **M44 failure:** report exploratory integration without confirmation.

### Outcome taxonomy

- **Outcome A — confirmed operational loop:** M44 passes with a non-dominated,
  review-gated end-to-end system.
- **Outcome B — loop passes without sparse routing:** M44 passes under exhaustive
  model evaluation; cross-model routing remains shadow-only.
- **Outcome C — stage-wise lifecycle qualification:** rejection/discovery/review
  or transactions pass separately, but the integrated gate fails.
- **Outcome D — rejection/discovery bottleneck:** no useful review stream can be
  established on frozen strong features.
- **Outcome E — prior-art displacement:** literature audit finds a system already
  satisfying the claimed composition; contribution becomes replication,
  lifecycle hardening, and controlled ablation.

## 16. Execution order

1. Complete M38 literature and parent-evidence lock.
2. Implement only missing schemas and S0 fixtures.
3. Run M39 one-seed falsification, then three-seed retention.
4. Freeze retained acceptance heads.
5. Run M40 discovery factorial and freeze one pair plus one control.
6. Run M41 confirmation and transaction studies.
7. Run M42 routing independently; retain exhaustive fallback unless it passes.
8. Run M43 integrated factorial with only stage-passing mechanisms.
9. Block M44 when M43 has no passing winner; otherwise freeze and run it once.
10. Generate the final claim ledger and artifact-only reproduction in either
    disposition.
11. Run physical E7 separately when infrastructure is available.

## 17. Required final artifacts

1. verified prior-art coverage matrix and claim boundary;
2. immutable protocol, schedule, and parent-evidence indexes;
3. matched acceptance-head quality/cost/editability table;
4. rejection-by-discovery review-utility matrix;
5. immutable review and confirmation histories;
6. existing-class and new-class transaction reports;
7. empirical-routing-profile and Expert Gate routing comparison;
8. complete end-to-end factorial and ablation ledger;
9. stopped, blocked, unsupported, and failed branch table;
10. artifact-only reproduction command;
11. README and claim-ledger update with Outcome A-E.

## 18. Interpretation boundaries

- A strong EVM, density, or kNN result supports the loop, not GEODE geometry.
- An SDF lifecycle advantage must be measured against native comparator
  operations and stated with its predictive cost.
- Review-only success is not autonomous open-world learning.
- Simulated human confirmation establishes protocol behavior, not usability.
- Frozen-feature results do not establish end-to-end ORCA/IGCD superiority.
- Routing efficiency does not establish rejection quality.
- Exact rollback does not establish edit locality.
- One dataset, schedule, seed, or representation cannot establish generality.
- A negative result is successful if it identifies rejection, discovery,
  adaptation, routing, or composition as the binding layer.
