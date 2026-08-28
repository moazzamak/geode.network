# GEODE Research Implementation Plan v8

## Adaptation Utility as the Registered Endpoint

**Status:** final Outcome D; all milestones dispositioned
**Date:** 27 July 2026  
**Claim ledger:** `analysis/CLAIM_LEDGER_v8.md`  
**Immutable parents:** final v6.1 Outcome D and final v7 Outcome C

## 1. Program decision

V7 showed that individually qualified stages did not compose: Gaussian
rejection, HDBSCAN/FINCH discovery, and transactional rank-16 new-class
insertion passed separate gates, yet the grouped M43 loops integrated 0/3
confirmable classes. The review-everything control integrated 3/3 only by
spending the entire rejection stream.

V8 changes the optimization target rather than reopening any failed branch.
Per-stage metrics become diagnostics. The only performance endpoint that can
advance the program is post-integration balanced-accuracy utility on known plus
confirmed classes at a fixed human-review budget, subject to bounded
known-class regression and immutable transactional safety.

The claim under test is:

> A discovery-to-review-to-adaptation loop selected for downstream utility
> beats the frozen stage-wise-qualified v7 composition on post-integration
> balanced accuracy at equal review budget.

## 2. Registered endpoint and comparison policy

### 2.1 Episode utility

Each episode introduces one previously unknown class into a frozen stream.
Before episode \(e\), the parent recognizes \(K_e\). After review and
confirmation, the candidate transaction may add or update \(D_e\).

\[
U_e =
\operatorname{BA}_{K_e \cup D_e}(\text{child})
-
\operatorname{BA}_{K_e \cup D_e}(\text{parent}).
\]

Rejected or unavailable discovered labels count as incorrect in the parent
term. The cumulative utility uses the final class inventory and the same frozen
evaluation examples. Results report the seed-level mean, standard deviation,
paired episode differences, and a 95% paired bootstrap interval.

### 2.2 Fixed resources

- review budget: \(B=50\) labeled samples per episode;
- seeds: 11, 23, and 37 for development retention;
- known-regression ceiling: \(\epsilon=1.0\) percentage point;
- threshold-transfer unknown-recall band: 2.0 points;
- representation: immutable, hash-addressed frozen features;
- routing authority: exhaustive until M50 explicitly promotes a candidate;
- unspent labels do not transfer between episodes.

Wall-clock time, update work, serialized bytes, review count, and exact-model
evaluations are reported. A resource reduction cannot compensate for lower
utility or a failed safety invariant.

### 2.3 Statistical advancement

M47 passes only if utility-selected review:

1. exceeds density-core selection by at least 5.0 balanced-accuracy points in
   mean \(U_e\);
2. has a paired 95% bootstrap interval whose lower bound is above zero;
3. is positive on at least 7 of 9 seed-by-episode cells;
4. respects \(B\), \(\epsilon\), and every transactional invariant.

Boundary-inclusive selection may advance as the mechanistically distinct
control if it is non-dominated on utility, review work, and update work.

## 3. Frozen episode harness

The single evaluation vehicle is a deterministic episode replay harness built
on `src/runtime/episode_partitions.py`.

### 3.1 Episode contract

Every episode records:

- episode ID, seed, arrival class, and class-order parent;
- immutable stream, anchor, review-candidate, adaptation-support, validation,
  and final-test partition hashes;
- parent bundle and acceptance-policy hashes;
- review budget and consumed labels;
- review IDs, confirmation IDs, selected sample IDs, and selector provenance;
- threshold before and after integration;
- parent, child, and rollback bundle hashes;
- \(U_e\), known regression, remaining-unknown recall, NLL, work, and latency;
- graph, replay, rollback, publication, and routing-fallback results.

The harness must replay any episode from immutable inputs and must reject
partition overlap, parent drift, selector leakage, or class-order mismatch.

### 3.2 Data access

Development episodes may use labels only through:

1. the registered review oracle, up to \(B\);
2. the validation scorer after a candidate review set has been frozen.

The validation partition may evaluate utility but may not contribute support
examples. The final-test/untouched schedule remains sealed until M50. Learned
selection in M48 uses leave-one-episode-out or earlier-episode training only.

## 4. Stage-interface contracts

M45 adds typed, versioned schemas to `src/runtime/schemas.py`. Each handoff
must expose what the consumer requires rather than merely what the producer
already computes.

The implementation map is fixed as follows:

- rejector and buffer evidence: `src/rejection_buffer.py`;
- grouping and candidate partitions: `src/discovery_clustering.py`;
- review-set feasibility and expected gain: `src/candidate_usefulness.py`;
- publication and rollback: `src/runtime/adaptation_transaction.py`;
- candidate-versus-exhaustive routing: `src/shadow_routing.py`;
- durable contracts: `src/runtime/schemas.py`;
- leakage-safe replay partitions: `src/runtime/episode_partitions.py`.

| Interface | Producer currently emits | Consumer needs | Registered mismatch/evidence |
|---|---|---|---|
| Rejector -> buffer | embedding, novelty score, margin, nearest candidates, model/profile IDs | score calibration version, threshold lineage, class-order version, boundary proximity, anchor-set identity | M43 remaining-unknown recall fell 6.83 points after composition; threshold state was not transferred as an updateable contract |
| Buffer -> clusterer | bounded point records and windows | distance/metric lineage, recurrence identity, uncertainty, boundary strata, representativeness weights | Core geometry can be reconstructed, but no contract requires mode or boundary coverage |
| Clusterer -> review selector | cluster membership, persistence, size, density/cohesion | purity estimate, coverage, mode diversity, boundary diversity, expected adapter gain, uncertainty, budget cost | HDBSCAN reached 100% stage-wise recall/precision but only 4--22% integrated target success |
| Review -> adapter | confirmed label and selected support samples | support-distribution summary, omitted-region diagnostics, class-order migration, expected utility, anchor constraints | M41 passed with 50 purpose-built samples; M43 grouped samples failed, showing confirmation alone was insufficient |
| Adapter -> router | child bundle, class order, support profile, transaction lineage | refreshed empirical profile, threshold transfer, exhaustive-winner audit, stale-profile action | E5 candidates preserved neither normalized winning scores nor p95 latency; the deployed multinomial readout required the full score vector |

### 4.1 Contract-violation records

Every interface emits a machine-readable `InterfaceContractAudit` with:

- producer schema and artifact hashes;
- required and supplied statistic names;
- dimensionality, score direction, calibration, and class-order lineage;
- missing, stale, approximated, and unsupported fields;
- downstream utility impact when measurable.

Missing required fields fail closed. Diagnostic fields may be absent only when
the consumer explicitly declares them unsupported.

## 5. M45 — Registration, schemas, and replay harness

**Duration:** 1--2 weeks  
**Execution:** unconditional

### 5.1 Work

1. Freeze `CLAIM_LEDGER_v8.md`, this plan, parent indexes, seeds, episode order,
   \(B\), \(\epsilon\), and statistical policy.
2. Add schemas for:
   - `AdaptationUtilityEndpoint`;
   - `EpisodeReplayContract`;
   - `InterfaceContractAudit`;
   - `ThresholdTransferRecord`;
   - `ReviewSelectionEvidence`;
   - `LocalizedResidualScope`;
   - `IntegratedRoutingDecision`.
3. Extend current records without breaking existing readers:
   - `RejectionRecord`;
   - discovery candidate/review records;
   - adaptation transaction records;
   - support/routing profiles.
4. Implement one episode replay command and S0 synthetic fixtures.
5. Add deliberate mismatch tests for leakage, stale class order, threshold
   lineage, budget overflow, missing confirmation, and rollback-parent drift.

### 5.2 Gate

M45 passes only if two clean S0 runs are byte-identical, all interface
mismatches fail closed, no training/final labels are loaded by the lock
verifier, and parent evidence reproduces v6.1 Outcome D and v7 Outcome C.

### 5.3 Result

**Passed, 27 July 2026.** The immutable M45 lock freezes two parent indexes,
three ordered synthetic episodes, the 50-label budget, the one-point
known-regression ceiling, the paired-bootstrap policy, and all five interface
contracts. Seven typed schemas were added, and rejection, support-profile, and
adaptation records gained optional all-or-none lineage fields without breaking
existing constructors.

The verifier reproduced v6.1 Outcome D and v7 Outcome C from their locked
indexes, materialized three episode contracts and five complete interface
audits, and rejected all six registered mismatch cases. Two clean executions
were byte-identical; neither loaded training data nor opened final labels.
Evidence is frozen under `logs/results/v8/m45_protocol_lock/`. M46 is open.

## 6. M46 — Threshold transfer and sufficient-statistics diagnostics

**Duration:** 2--3 weeks  
**Execution:** complete; unconditional after M45

### 6.1 Threshold-transfer experiment

Compare:

1. frozen pre-integration threshold;
2. class-count heuristic transfer;
3. anchor-set quantile recalibration;
4. per-class anchor recalibration with a global fail-closed threshold.

The anchor set is frozen before the episode and never supplies adaptation
support. Recalibration runs after every class-order change.

**Threshold rule gate:** retain at most one rule that, on every development
episode:

- preserves remaining-unknown recall within 2.0 points of the pre-integration
  value;
- keeps known-class regression within 1.0 point;
- does not consume review labels;
- replays exactly and rolls back with the parent threshold;
- never lowers confidence when anchor lineage is stale; stale state triggers
  exhaustive/frozen-parent fallback.

If no dynamic rule passes, retain the frozen threshold and record threshold
transfer as unsupported. M47 may continue because threshold diagnostics are
unconditional, but M50 cannot claim preserved unknown recall unless one rule
passes there.

### 6.2 Sufficient-statistics audit

Replay v7 M40--M43 and E5 with interface instrumentation. For each failed
consumer, estimate utility loss after adding one missing statistic at a time.
Rank mismatches by paired \(\Delta U_e\), then by failure frequency and
implementation cost.

Required outputs:

- core versus full-rejection coverage by class mode;
- distance-to-parent-boundary distribution;
- selected/support covariance and low-rank subspace coverage;
- omitted-region nearest-neighbor coverage;
- E5 full-score-vector dependency and normalized-winner error;
- A3 changed-region leakage and fusion-path attribution.

The ranked audit freezes the Phase-2 selector features. No feature may be added
after M47 begins.

### 6.3 Result

**Passed, 27 July 2026.** Frozen-threshold, class-count, and global
anchor-quantile transfer all stayed within the registered bands on seeds 11,
23, and 37. The global anchor rule was retained because it directly
recalibrates after the class-order change: mean known-class regression was
0.00 points, mean remaining-unknown recall change was 0.00 points, and mean
remaining-unknown recall was 79.33%. The per-class fail-closed rule was rejected
because its mean known-class regression was 1.875 points. Threshold transfer
consumed no review labels, stale anchor lineage restored the frozen parent
threshold, rollback was exact, and two executions were byte-identical.

The interface audit found that 50 core examples covered only 34.0--39.1% of
the full rejected region under the nearest-neighbor diagnostic, whereas the
boundary-inclusive sets covered 99.37--99.78%. Boundary inclusion increased
proxy adaptation utility by 0.259 points on average, positive on two of three
seeds but far below the M47 advancement threshold. The ranked mismatches were
A3 fusion-scope leakage (15.8-point preservation deficit), E5 full-score-vector
dependence (0.431-point exhaustive-winner omission), and missing boundary
examples (0.259-point proxy utility estimate). M47 is restricted to the six
frozen features serialized in `logs/results/v8/m46_diagnostics/evidence.json`.

## 7. M47 — Joint discovery and utility-driven review

**Duration:** 3--4 weeks  
**Execution:** open; pivotal experiment

### 7.1 Equal-budget arms

All arms receive the same rejection stream and exactly \(B=50\) review labels:

1. **core-selected:** highest-persistence/highest-density cluster cores, matching
   the v7 stage-wise policy;
2. **boundary-inclusive:** stratified selection across core, modes, and
   parent-boundary-distance quantiles;
3. **coverage-selected control:** farthest-first or k-center selection within
   the candidate group;
4. **utility-selected:** enumerate deterministic candidate subsets, fit a cheap
   proxy adapter on the review fold, and select the subset with the highest
   validation \(\Delta U\);
5. **random stratified control:** seeded selection across the same candidate
   group;
6. **review-everything diagnostic:** uncapped support, reported but ineligible
   for equal-budget advancement.

The utility proxy must use a cheaper registered adapter and disjoint validation
examples. It may rank candidate sets but may not alter the production adapter,
threshold, or evaluation partition.

### 7.2 Diagnostics

Report, but do not optimize independently:

- purity and persistence;
- mode and covariance coverage;
- boundary-distance coverage;
- expected and realized utility;
- selector regret;
- reviewed labels and duplicate reviews;
- known regression, remaining-unknown recall, NLL, update work, and latency.

### 7.3 Kill switch

M47 is the pivotal gate. If utility-selected review does not satisfy the four
advancement criteria in Section 2.3, stop the main program with Outcome D.
Do not train a selector and do not run E12. Publish the statistic-mismatch audit
and the negative equal-budget result.

### 7.4 Result

**Failed; final main-program Outcome D, 27 July 2026.** The registered
equal-budget replay evaluated six arms over all nine seed-by-arrival cells.
Utility-selected review achieved mean episode utility of 9.96 points versus
6.36 points for density-core selection, a paired gain of 3.593 points with
95% bootstrap interval [2.794, 4.382]. The gain was positive in 9/9 cells, but
it missed the preregistered 5.0-point mean threshold.

The safety conjunction also failed: every utility-selected transaction used
exactly 50 reviews, linked confirmation, appended one class, and rolled back
exactly, and known-class accuracy did not regress on average; however, six of
nine cells lost more than 2.0 points of remaining-unknown recall. The simple
coverage-selected control reached higher mean utility (10.62 points) than the
utility-selected arm (9.96), while review-everything reached 11.26 points using
300 labels per episode and remained ineligible.

The M47 kill switch is binding. M48 learned selection and M50/E12 are blocked
and must not run. The optional, independently bounded M49 locality branch may
still be dispositioned, but it cannot reopen Outcome D.

## 8. M48 — Learned selection for downstream utility

**Duration:** 4--6 weeks  
**Execution:** blocked by M47 Outcome D

### 8.1 Model

Train a lightweight scorer over candidate examples or candidate subsets using
only features frozen by M46:

- cluster/core distance;
- parent-boundary distance and novelty margin;
- local density;
- mode/medoid identity;
- low-rank coverage contribution;
- candidate redundancy;
- estimated proxy-adapter gain;
- review cost.

Allowed learners are regularized linear/ranking regression, shallow
gradient-boosted trees already available in the environment, or a contextual
bandit with deterministic replay. The scorer is selection tooling, not the
deployed class model.

### 8.2 Leakage and evaluation

Use leave-one-episode-out evaluation. The held-out episode's validation utility
may score the frozen selection only after selection is complete. No target
episode is used to train its own selector.

### 8.3 Gate

The learned selector advances if it:

- retains at least 80% of M47's utility-selected gain over core selection;
- has positive paired gain on at least 7/9 cells;
- respects \(B\) and \(\epsilon\);
- adds no transactional failure;
- replays its scores and selected sample IDs exactly;
- is non-dominated by the deterministic boundary-inclusive policy on utility,
  selection latency, and serialized state.

If it fails, M50 uses the best deterministic M47 policy.

## 9. M49 — Localized residual and discriminative refinement

**Duration:** parallel track  
**Execution:** may begin after M45; optional for M50

### 9.1 Scope

When a confirmed adaptation targets a region, attach a local residual to the
existing rank-32 components rather than globally refitting. Registered residual
families:

- per-component nonnegative weights;
- local temperatures;
- diagonal or low-rank metric scales;
- a bounded class-local affine correction.

Discriminative refinement is allowed only to improve adaptation utility under
the frozen representation. Any reduction of the 5.03-point v6.1 deficit is
descriptive and cannot reopen parity.

### 9.2 Locality definition

Before fitting, freeze the affected region as the union of selected support,
their registered nearest-neighbor radius, and components activated above the
registered responsibility threshold. The evaluation set is split into affected
and unaffected examples by this parent-only rule.

### 9.3 Two-attempt gate

An attempt passes only if:

- \(U_e\) improves by at least 5.0 points over no residual;
- at least 99.9% of unaffected predictions are preserved on every seed;
- known regression is at most 1.0 point;
- remaining-unknown recall drops by at most 2.0 points;
- update work is class/region local;
- replay and rollback are exact.

At most two preregistered residual families may be attempted. If both fail
locality, kill M49 with Outcome E. M50 continues without residuals if M47
passed.

### 9.4 Result

**Closed with no retained residual, 27 July 2026.** Both preregistered attempts
used a parent-only scope with a 0.99 responsibility threshold and bounded
support radius. Local target temperature changed mean utility by 0.000 points;
the bounded class-local affine residual improved it by 0.204 points. Both were
well below the registered 5.0-point gate, and the maximum remaining-unknown
recall drop was 4.333 points.

Both attempts preserved 100% of unaffected predictions on every cell, kept
maximum known regression to 0.125 points, performed only class/region-local
work, and replayed and rolled back exactly. Thus the A3 scope-leakage mechanism
is repaired by explicit fusion gating, but no useful residual is retained.
Outcome E does not fire because locality passed; final Outcome D is unchanged.

## 10. Head substitution ablation

M47 and M50 compare the same selector and adapter under:

1. the retained explicit GEODE/weighted-affine head;
2. low-rank Gaussian or DDU-style density;
3. kNN-support hybrid.

Thresholds are independently calibrated from the same anchor partition; review
budget and episodes are identical. The winning system is selected on utility,
not on whether it uses an SDF. Transactional properties are measured for every
head. If a substitute yields higher utility with equivalent safety, the systems
claim advances with the substitute.

### 10.1 Disposition

The low-rank Gaussian/DDU-style head was the frozen primary in M47. The
same-harness GEODE and kNN-support substitutions are closed without execution
after the binding M47 kill switch: Section 7.3 requires the main program to
stop and prohibits E12 once the primary utility gate fails. Existing M39
evidence remains stage-level only and cannot be relabeled as adaptation
utility. No head-substitution systems claim is made.

## 11. M50 / E12 — Lifecycle qualification

**Execution:** blocked by M47 Outcome D

### 11.1 Frozen loop

Freeze before opening the untouched schedule:

- acceptance head and threshold-transfer rule;
- review selector;
- production adapter and optional residual;
- empirical router candidate and promotion rule;
- episode order, \(B\), \(\epsilon\), and all artifact hashes.

Run a multi-episode:

`reject -> buffer -> select/review -> confirm -> adapt -> recalibrate -> refresh router -> validate -> publish/rollback`

transaction for every episode.

### 11.2 Router promotion inside the loop

Routing is not promoted by a disconnected benchmark. At each child bundle, the
candidate router is compared with exhaustive compatible-model inference on the
episode anchor and evaluation partitions. It controls outputs only if:

- the exhaustive winning bundle is always included;
- final predictions agree exactly after fallback;
- unknown/no-confident cases fall back exhaustively;
- the refreshed profile matches child class order and threshold lineage;
- cumulative utility is unchanged relative to exhaustive routing;
- p95 latency or exact-model evaluations improve by at least 25%.

Otherwise routing remains shadow-only for that and later episodes.

### 11.3 Success gate

M50 passes if the frozen jointly qualified loop:

1. exceeds the v7 Gaussian+HDBSCAN stage-wise composition by at least 5.0
   points in mean \(U_e\) at equal \(B\);
2. has a paired 95% interval for the utility difference above zero;
3. achieves positive cumulative \(U_{1:N}\);
4. keeps per-episode and cumulative known regression within 1.0 point;
5. preserves remaining-unknown recall within 2.0 points using the registered
   threshold-transfer policy;
6. produces no unconfirmed publication, mutation, or false class creation;
7. passes every graph, replay, rollback, audit, and fallback check;
8. is non-dominated by core selection, no adaptation, full class-local refit,
   and review-everything diagnostics on utility, review burden, and update work.

Passing under exhaustive routing yields Outcome B. Authoritative routing is not
required for the utility claim.

## 12. Dependencies and execution order

```text
M45 registration/contracts
 |\
 | +--> M49 residual attempt 1 --> optional attempt 2 --------+
 v                                                           |
M46 threshold/statistic diagnostics                          |
 v                                                           |
M47 equal-budget utility selection --fail--> stop Outcome D  |
 |                                                           |
 +--> M48 learned selector (optional) ------------------------+
 |
 +----------------------------------------------------------> M50 / E12
```

1. Complete and commit M45.
2. Complete and commit M46.
3. Run and commit M47; stop immediately if its kill switch fires.
4. M47 failed: block M48 and M50; finish the independent bounded M49 branch.
5. Close the same-harness head continuation under the binding kill switch.
6. Do not open the untouched M50 schedule.
7. Generate the final claim ledger and artifact-only replay. Complete.

Every milestone updates this plan, `analysis/MILESTONE_RESULTS.md`, the v8 claim
ledger, and README before its commit.

## 13. Required artifacts

1. immutable v8 parent, endpoint, budget, seed, and episode locks;
2. typed interface contracts and mismatch audits;
3. byte-identical episode replay harness;
4. threshold-transfer matrix and anchor lineage;
5. ranked sufficient-statistics mismatch report;
6. equal-budget review-selection factorial;
7. selector leakage audit and learned-selector report when eligible;
8. localized-residual scope and two-attempt ledger;
9. head-substitution utility ablation — blocked by the M47 kill switch;
10. E12 per-episode and cumulative utility report — blocked by M47;
11. immutable review, confirmation, transaction, router-refresh, and rollback
    histories — M47 review/adaptation histories frozen; M50 router histories
    blocked;
12. stopped, blocked, unsupported, and failed branch table;
13. artifact-only reproduction command that loads no training or untouched
    final labels.

### 13.1 Final artifact result

The final verifier checked four immutable milestone indexes, all 11 indexed
artifacts, and six conclusion operands twice with byte-identical output. It
loaded no training data and opened no final labels. The immutable final ledger
is `analysis/V8_FINAL_CLAIM_LEDGER.md`; replay output is under
`logs/results/v8/final_outcome_replay/`.

## 14. Interpretation boundary

The program succeeds only if it improves adaptation utility at equal human
budget. Better purity, AUROC, coverage, routing accuracy, edit locality, or
latency cannot independently establish success. Conversely, a non-SDF head may
win without weakening the systems claim. The explicit artifact remains relevant
only where it contributes measured utility or lifecycle safety.

## 15. Publication packaging

**Complete.** The final v7/v8 technical report is maintained as the
arXiv-compatible source `analysis/FINAL_RESEARCH_PAPER.tex`, with build
instructions in `analysis/BUILD_PAPER.md` and a compiled PDF beside the source.
The report uses adaptation utility as its organizing endpoint, reports M47 as
Outcome D rather than promoting the positive bootstrap interval alone, and
preserves all blocked-branch and untouched-confirmation restrictions. The
earlier Markdown paper remains explicitly labeled as the historical v7
manuscript.

A separate MS thesis-style report is complete at
`analysis/MS_THESIS_REPORT.tex`, with a compiled PDF and
`analysis/BUILD_THESIS.md`. It expands the same frozen evidence into academic
front matter, research questions, background, formalization, system design,
methodology, results, discussion, validity, ethics, reproducibility, and
appendices. It introduces no new experimental claim and preserves the binding
Outcome-D interpretation.

The thesis methodology history was subsequently expanded from the first
recorded milestones rather than beginning at v6.1. The revised 65-page build
now covers the repaired temporal baseline, M0--M15, E0--E11, and v5--v8;
separates software tests from experimental observations; catalogs the
principal tests and factorials; and records how null, negative, blocked, and
infeasible outcomes changed the next protocol.
