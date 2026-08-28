# GEODE Research Implementation Plan v17 — Strategic Iteration

## Where the program stands and what could be new (decision document, 6 August 2026)

**Status:** strategy/decision iteration, written 6 August 2026, **after** the
sealed M109 run. This document records the post-M109 strategic position, a
refreshed literature search on training-cost efficiency, and the candidate new
avenues, so they can be revisited. It is a successor in the v16 line; **no v16
registration, operand, kill switch or prediction is changed or superseded by
this document.** v16 (M107–M112, §5–§6) remains in force. Text here that would
become claimable must first be registered in v16-style with kill switches and a
prior-art sweep, per the audit discipline (§7.1) and the supersede-in-place rule.

---

## 1. What v16 established (sealed evidence, 6 August 2026)

- **M107 crossing reproduced at (t1):** frozen sparse (0.2148) beats the best
  frozen dense at-or-below (0.1971) at sparse 254.6 M MACs — but only in the
  frozen regime.
- **M108:** selection-based dictionary growth (arm (c)) does **not** transfer to
  DomainNet; gap-closing fraction ≤ 0 except arm (c) at 1024 (+0.0081).
- **M109, kill switch 1 fired at (t2)/(t3)/(t4):** once both families train
  their own representation, dense is above sparse at every overlapping MAC
  budget. **The inference-MAC crossing does not survive trunk training.** The
  registered prediction — and the prior art (Ghorbani 2020; Székely 2024) — are
  confirmed. C107.1 narrows to "frozen sparse features beat frozen
  out-of-distribution dense features at two budgets on one corpus".
- **M109, kill switch 3 did not fire, with a twist:** gradient training _moved_
  the sparse curve but **down** (0.2148 → 0.1302). The constructed frozen
  dictionary + closed-form float64 ridge head beats its own gradient-trained
  version. **The program's no-backprop pipeline is better than the backprop
  version of the same model.**
- **M109 §5.2.6 same-data arm:** t4 from-scratch dense 224 (0.1132 @ 6.1 G MACs,
  21.5 M params) sits **below** t4 sparse (0.1302 @ 254.6 M MACs, 4.58 M
  params). The only comparison where both sides saw the same data still favours
  sparse, at 24× fewer MACs and 4.7× fewer parameters.

**What this means:** on the inference-MAC axis, the sparse-dictionary story is
closed (negative, correctly recorded). The program's _measurable_ content now
lives on the other axes: the parameter axis (M110), the training-cost axis
(proposed below), and the low-data crossover map (§6, registered).

---

## 2. Refreshed literature search, 6 August 2026 — training-cost efficiency via sparse growth

**Motive.** The user's question: is the "growing sparse models for compute
efficiency" direction, specifically **training-cost** efficiency, still open?
**Answer from the search: no — it is one of the most crowded areas in ML.** A
refreshed arXiv search (queries below, anchors passed) confirms mature,
decade-old, well-cited lines that already claim the training-efficiency
headline the program might have pursued.

**Queries run (arXiv API, all:phrase, relevance-sorted):**
`"dynamic sparse training"`, `"sparse backpropagation"`, `"Rigging the
Lottery"`, `"lottery ticket hypothesis"`, `"Switch Transformers"`, `"grow and
prune"`, `"growing sparse"`, `"grow sparse" AND "training"`. Anchors (known work
the index must return): RigL (1911.11134), Lottery Ticket (1803.03635 /
1912.05671), Switch Transformers (2101.03961) — all returned, so the index was
sensitive for these queries.

**Families found (with representative work):**

| family                                   | hits | representative work                                                                | training-cost claim already made                          |
| ---------------------------------------- | ---: | ---------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Dynamic Sparse Training                  |   55 | RigL (ICML 2020), SET, SRigL (ICLR 2024), DynaDiag (2025), FedDST                  | sparse-to-sparse training matches dense at constant FLOPs |
| Sparse backpropagation                   |    8 | SparseProp (2023), TinyProp/v2, PockEngine (MICRO 2023), SparseMixer               | 5× / 10× training-cost reductions on-device               |
| Grow-and-prune                           |   30 | NeST, SCANN, GaP (ICLR 2022), **Structured Continuous Sparsification (ICLR 2021)** | 47.4% training-FLOP savings, ResNet-50/ImageNet           |
| Sparse MoE                               |   42 | Switch Transformers (JMLR 2022), ST-MoE, PEER, Million-Expert MoE                  | up to 7× pre-training speedup at matched resources        |
| Lottery tickets                          |  213 | Frankle & Carbin; LMC+LTH (ICML 2020); E-LTH (NeurIPS 2021)                        | sparse subnetworks train to dense accuracy                |
| Bregman / mirror-descent sparse training |    — | LinBreg (JMLR 2022), multilevel mirror descent (2026)                              | 38%→6% of SGD FLOPs, ~50% training-time cut               |

**Reading, stated honestly.** A "training-cost efficiency breakthrough via
sparse growth" is not open; it is claimed by many, at scale, with hardware
speedups. Any new claim in that direction must therefore be **narrower** than
"growing sparse models saves training compute" — and, per §7.1, novelty cannot be
certified from a public unauthenticated search at all (measured recall is a
lottery over query phrasing). The remaining open space is not the general axis
but the program's specific, measured corner of it.

---

## 3. Candidate new avenues (honest assessments; none is a novelty claim)

### A1 — Training-cost efficiency of a closed-form (no-backprop) learner. _(recommended)_

**The idea.** The crowd optimises SGD-based sparse training. The program's
sparse model is unique in that its training is **no-backprop**: dictionary
construction (M108) + whitening + encode + a closed-form float64 ridge solve.
M109's KS3 twist measured that this closed-form pipeline _beats its own
gradient-trained version_ (0.2148 vs 0.1302) — the no-backprop learner is
better than the backprop version of the same architecture.
**The measurement.** Total **training** FLOPs (and, secondarily, wall-clock) to
reach a fixed accuracy, per family, on the same corpus: (a) closed-form sparse
(whiten + encode + gram + ridge, decomposed and summed), (b) SGD-trained
from-scratch dense, (c) SGD-trained pre-trained dense (with the §3.10 LVD-142M
disclosure). Report accuracy-per-training-FLOP.
**Why it is not obviously closed.** M109 closed the _inference-MAC_ axis; the
_total-training-cost_ axis is a different operand that M109 does not contradict.
The DST/MoE/grow-and-prune literature reports training-cost savings for
SGD-based sparse methods; the "closed-form ridge on random features as a
training-cost competitor to SGD to a matched accuracy at 345 classes" figure is
not a standard result we found.
**Why it can still fail (registered before running).** It is a measurement of a
known mechanism (ridge regression is not novel); if the closed-form sparse side
reaches only its ~21% ceiling while dense reaches 53%, the claim narrows to
"cheap, early plateau" — still honest, but modest. Kill switch: if the closed-form
sparse cost to its own ceiling is not an order of magnitude below the dense cost
to the same accuracy, the axis adds nothing beyond M110.

### A2 — Head-vs-representation decomposition (gating for M109/M110; first to run). _(cheap, do first; design registered 6 August 2026)_

**Question.** At M109 (t2), the same frozen sparse codes scored 0.2148 with the
closed-form ridge head but 0.0554 with the 4-epoch SGD head — a 4× collapse
while the dense head _improved_. **Is the sparse side's M109 loss a
head-underfit artefact, or a property of the representation?**

**Design (no trunk training).** Encode the frozen representations once: sparse
codes (arm (a), 3,072 atoms) and dense features (DINOv2-small at registered
resolutions {42, 224}). On each frozen representation, fit heads:

1. **Ridge reference** — closed-form float64 ridge, penalty 1.0 (M107's head;
   the converged linear-head optimum).
2. **SGD sweep** — AdamW linear head over a registered grid of {epochs, lr}
   with early stopping on validation until converged; includes the exact M109
   t2 schedule (4 epochs, lr 3e-4) as a reproduction control.
   Report test accuracy per family and per head; the converged SGD head vs the
   ridge reference; and the residual frozen sparse-vs-dense gap at converged heads.

**Operand.** Test accuracy at the converged head (not the 4-epoch point), plus
the epochs/learning-rate budget required to converge.

**Registered prediction.** The converged SGD head on frozen sparse codes
approaches the ridge reference (0.2148); the t2 collapse is a schedule artefact,
and M109's (t3)/(t4) "sparse degrades more" verdict stands on representation
training, not on the head.

**Kill switch.** If the converged SGD head on frozen sparse codes stays far
below the ridge reference (beyond a registered margin), the sparse codes are
not linearly separable by SGD at any practical budget — an optimisation finding
that changes A5's head choice for Arm P (ridge-only) and re-opens the t2
reading.

**Why it is first.** It is cheap (head-only training on frozen codes; no trunk
gradients), it decides whether M109's negative is a head artefact, it confirms
M110's split reporting (KS2: the head is 92.5% of the sparse count — the axis
measures the head), and it fixes A5's per-expert head choice for Arm P (ridge vs
trained-at-convergence).

**Config (registered before measurement).** Frozen representations {sparse 3072,
dense 42, dense 224}; head grid epochs {4, 16, 64}, lr {3e-4, 1e-3, 3e-3}, AdamW,
cosine, early stopping patience 2, validation fraction 0.05; ridge penalty 1.0;
seed 109; same shared subsample.

**[recorded after execution] Result, 6 August 2026.** A2 ran sealed on the 9070 XT
(evidence `logs/results/v16/a2_head/evidence.json`, admissible). **Reproduction
gates passed**: the ridge reference reproduces M109 t1 (max delta 0.00078 <=
0.002) and the 4-epoch SGD cell reproduces M109 t2 (max delta 0.00499 <= 0.01),
so the head measurements are pinned to the sealed M109 evidence. **The
registered prediction is refuted; the kill switch fired.** On frozen sparse
codes, the closed-form ridge head (0.2148) beats the **best** SGD head over the
whole {4,16,64} x {3e-4,1e-3,3e-3} grid (0.1490 @ e16/lr3e-3) by 6.6 pp — the
sparse codes are not linearly separable by SGD at any practical budget. On
frozen DINOv2 features, SGD **beats** ridge (r42 0.2466 vs 0.1971; r224 0.6459
vs 0.5368). So the M109 t2 sparse collapse is **not** (only) a 4-epoch
under-fit artefact: SGD is the wrong head optimiser for sparse codes, ridge is
right, and the head choice is family-dependent. **Consequence for A5: Arm P uses
the closed-form ridge head per expert, never a trained linear head.** This also
re-inforces the closed-form-beats-gradient reading of M109 KS3 at the head level.

### A3 — Dynamic sparse-dictionary growth (DST applied to atoms). _(expected negative)_

Grow/prune _atoms_ during training (reconstruction-error growth, dead-atom
pruning) vs M108 static construction vs M109 plain gradient. Uses the program's
rig; the DST literature predicts it will at best match dense — and the program's
dictionary already lost to dense even gradient-trained — so expected negative.
Its value is closing "did we try growth the way the field grows it?" honestly.

### A4 — Low-data crossover map (already registered, §6). _(fallback)_

Map where the frozen+ridge sparse win lives as a function of train size
(12.5/25/50/100%). §8's "honest place the window can be shown to live". Not new;
listed here for completeness and as the consolidation target.

### A5 — Routed specialists: two expert families (SDF/geometric vs patch-dictionary) under one routing protocol. _(new, user-suggested, 6 August 2026; design amended 6 August 2026)_

**The idea (amended).** The independent variable is the **expert representation
family**: SDF/geometric (the GEODE SDF/CSG machinery, v7–v14) vs the
**patch dictionary** (v16). "Trunk and exact primitive can be whatever works
best" per family — but the per-family configuration is registered **before**
measurement, never tuned after. Both are dispatched as **per-domain specialists**
by a **fingerprint router** under one protocol, so the comparison is
single-factor (family) at fixed task, router, opponent and cost accounting.

**Protocol (fixed for both families and the dense opponent).** Closed-set
345-way classification on DomainNet's shared subsample; per-domain specialists
(one per domain / per registered cluster); a fingerprint router (interpretable,
not a learned black-box gate — `src/model_fingerprint.py`,
`src/candidate_routing.py`); each input pays router + one specialist; total
inference cost is that sum, never the sum of all experts.

**Three arms, judged on two operands.**

1. **Arm D** — the global dense opponent (DINOv2-small, one model, no routing).
2. **Arm P** — per-domain patch-dictionary specialists (v16 pipeline).
3. **Arm S** — per-domain SDF/GEODE specialists.

- **Operand 1:** the two families (P vs S) at a **fixed per-expert budget** —
  which representation family is the better sparse expert under the router?
- **Operand 2:** the better family vs **Arm D at matched total cost** — does the
  routed specialist beat the global dense model on the non-natural domains?

**The one thing that is NOT "whatever works best": the basis.** The two families
sit at different stack positions — GEODE was a head on frozen DINOv2 features;
the patch dictionary is raw-pixel. If Arm S uses DINOv2 features it inherits the
dense opponent and cannot win an efficiency comparison by construction. **Both
families therefore use a registered non-dense basis derived from raw pixels**
(clarified 6 August 2026: Arm P keeps the v16 whitened-patch basis — that is its
identity and the M108/M109 reproduction anchor; Arm S uses a registered
PCA-of-raw-pixels basis sized to keep the RANSAC fit dimension within the
fittable range). Neither family uses DINOv2 features; a DINOv2-trunk SDF arm may
be carried only as a separate, disclosed, efficiency-excluded arm.

**Registered per-family config (before measurement).** SDF family: **spherical
primitive by default** — the `spherical_covariance` fitter with the corrected
`d+2` sphere seed, which is the measured best primitive under the program's own
protocol (v5 §R12: five-seed sphere beats full covariance by 2.94 pp
[2.47, 3.40]; without subtraction +2.19 pp [0.93, 3.45]; the old spherical
fitter's `2d+1` seed was an unfair artefact and was superseded). Full and
diagonal ellipsoids remain registered alternatives in case the per-domain,
345-class setting changes the primitive ranking. Also registered for the SDF
family: PCA dimension, ray steps, CSG on/off. Patch-dictionary family: atoms,
patch size. All fixed at registration; a change is a re-registration.

**Why this is the strongest new avenue.** (1) It attacks M109's actual failure
mode (global capacity vs specialisation) instead of scaling the same global
object. (2) It reuses the program's existing, previously-measured machinery
(routing, fingerprinting, SDF, dictionary) rather than building new. (3) It is
"growing sparse models" as the field does it — grow _experts_, not atoms —
consistent with the program's original goal. (4) It converts **both** prior
negatives into one controlled question: v13/v14's blocked OOD dominance for SDF
(M85: geometry AUROC 0.585 vs kNN 0.575, dominance blocked on the rejection
task) and M109's MAC loss for the patch dictionary — neither family has been
measured on this protocol (closed-set classification, per-domain specialists,
matched total cost). If SDF also loses under A5's protocol, the v13/v14 negative
was representation-family-wide, not task-specific; if it wins, the negative was
task-specific and representation matters.

**Honest prior-art caveats.** Every ingredient is published: learned MoE routing
(Switch, ST-MoE), routing with no learned gate (Roller et al. 2021, "Hash
Layers" — returned in the 6 Aug search), per-domain experts, sparse-dictionary
features, and SDF/CSG classifiers. A public unauthenticated search cannot
certify the exact combination is unclaimed; §7.1's audit must run before any
claim, and novelty is never claimed. The contribution, if any, is a
**measurement under the program's protocol** of which expert family, if either,
beats a global dense model at matched total cost.

**Instrument preconditions (before any arm runs).**

1. **SDF softmin normalisation — verified satisfied, 6 August 2026.** The audit
   (25 July) flagged a missing 1/M in the GPU kernels. Direct reading confirms it
   is already fixed in both paths: CPU `SoftminFusion.fuse` uses
   `log( Σ exp(...) / M )` (src/sdf_engine.py) and the GPU kernels
   `expert_softmin_csg` / `class_softmin` use `log(sumexp / (float)cnt)`
   (src/gpu_engine.py). Arm S may use either path; a CPU/GPU parity check on the
   fused SDF still runs at startup, and any divergence beyond a registered bound
   voids the SDF arm.
2. SDF inference is O(K·d²·ray_steps) ray-marched and may lose the cost operand
   before accuracy is judged — that is a finding, not a bug, but it is why
   Operand 2 is "matched total cost", never accuracy alone.
3. SDF construction is O(d³) RANSAC and is infeasible above roughly d=256 — the
   registered non-dense basis must keep the fit dimension inside the fittable
   range, and the basis cost is counted in the operand.

**Floor-feasibility finding and scope decision, 6 August 2026.** Checking the
shared subsample's per-domain structure against the §3.5 floor (11.23 rows per
fitted dimension) before building: per-domain train rows are {11,224, 11,587,
16,217, 42,457, 40,773, 15,742}; per-(domain,class) cells average ~67 rows (min
1, max 222). **Arm P is feasible but budget-capped** (per-domain dictionary
atoms <= D_d/44.92 = {249, 258, 361, 945, 907, 350}); **Arm S (per-class
spherical SDF experts) is floor-VOID at any useful dimension** (d=8 needs ~101
rows/cell, most cells are below). **Decision: A5 runs in phases. Phase 1 (now):
Arm P-only on the shared subsample** — per-domain patch-dictionary specialists
(atoms floor-capped above) with the closed-form ridge head (per A2), oracle
router primary + a registered cheap non-dense fingerprint router, Arm D
re-measured, per-domain eval and matched-cost accounting. Arm S is deferred to a
full-split corpus (the only regime where it is floor-feasible) and its
floor-voidness on the shared subsample is itself reported as a finding.
**Option 1 (the full-split, two-family A5 — Arm P + Arm S on 409,832 rows, the
only regime where Arm S is floor-feasible) is explicitly deferred, not dropped:
it is re-considered after Phase 1's verdict, because Phase 1 decides whether
routing lifts the sparse ceiling at all before paying the larger compute.**

**A5-p1 first sealed run VOID + registered repair, 6 August 2026.** The first
sealed run of A5 Phase 1 self-voided on its own instrument precondition, which
is the gate doing its job. Two defects found, both registered as a repair and
applied to every arm before any re-run:

1. **Registration arithmetic error on d1's atom budget.** The floor cap is
   atoms_d = floor(D_d/44.92); for d1, D_d = 11,587, 11587/44.92 = 257.947, so
   the cap is **257**, but the config was registered with **258** — one atom
   over the floor. The floor gate caught it (`ok: false`) and the runner wrote
   a clean void evidence (no results). All other domains verified: {249, 361,
   945, 907, 350} are each <= floor(D_d/44.92). **Fix:** 258 -> 257 in
   `a5_routed.json` (recorded there as `_floor_repair`).
2. **The floor gate and the fitted object disagreed by the validation split.**
   The gate was computed on D_d total rows while the ridge heads fitted on a
   vestigial 5% validation carve-out (train_fit). A5 has no SGD and selects no
   hyperparameter, so a validation split has nothing to do; with it, the
   recorded rows-per-fitted-dimension would read ~10.7 < 11.23 next to a passed
   floor gate. M107's dense arm ("a dense arm has no validation split to carve
   off") and its per-domain mixture arm both fit on ALL train rows. **Fix:** A5
   now fits every head on all train rows; rows_d = D_d, so the recorded
   rows-per-fitted-dimension agrees with the floor gate (>= 11.23 at the cap).

Both fixes are pure instrument-consistency repairs: no registered accuracy,
hypothesis, or comparison rule changed. The re-run uses the repaired
`a5_routed.json` and the same sealed output path.

**A5 Phase-1 result (SEALED, admissible, 6 August 2026).** Evidence:
`logs/results/v16/a5_routed/evidence.json` (payload-sha256'd). Instrument
gates all pass; both registered kill switches fired.

- **t1 reproduction: exact.** Arm D re-measured delta 0.00000 at both r28
  (0.15986) and r42 (0.19716) vs M107 — the full-train fit repair was correct
  (M107 also fits on all train rows).
- **Floor gate: passed.** All six specialists at their caps, recorded
  rows-per-fitted-dimension {11.269, 11.271, 11.231, 11.232, 11.238, 11.244}
  > = 11.23 everywhere.
- **Fingerprint router (secondary, caveated): 0.479 routing accuracy** —
  per-image mean/std nearest-centroid over-routes to quickdraw (12,952 vs
  10,497 true) and under-routes real (2,519 vs 10,224) and infograph (1,686 vs
  2,945). Reported as a gate; verdicts below are read off the ORACLE-routed
  arms only (design gap #1's rule).

Oracle-routed Arm P (per-domain specialists at floor-capped atoms, closed-form
ridge head per A2): overall **0.2092**. Per domain — clipart 0.1700 (249),
infograph 0.0567 (257), painting 0.1092 (361), quickdraw 0.3352 (945), real
0.2121 (907), sketch 0.1122 (350).

- **KS1 (ceiling lift) FIRED.** Oracle-routed Arm P 0.2092 vs global frozen
  sparse 0.2148; the registered margin was +0.01, the measured margin is
  **-0.0056**. Routing does NOT lift the sparse ceiling: splitting the same
  total atom budget (~3,069 across six per-domain dictionaries vs 3,072 in one
  global dictionary) is slightly WORSE overall. **Registered consequence: the
  program consolidates.** The "routing lifts the frozen sparse ceiling" thesis
  is closed negative; M109's verdict stands and routing adds nothing overall.
- **KS2 (vs dense at matched cost) FIRED.** Oracle-routed Arm P beats
  re-measured Arm D (r28 per-domain) on **5 of 6 domains** while costing
  1.4-7.2x less compute per image:

  | domain    | Arm P  | Arm D r28 | delta   | routed MACs | dense r28 MACs |
  | --------- | ------ | --------- | ------- | ----------- | -------------- |
  | clipart   | 0.1700 | 0.1575    | +0.0125 | 19.95M      | 107.6M         |
  | infograph | 0.0567 | 0.0431    | +0.0136 | 20.59M      | 107.6M         |
  | painting  | 0.1092 | 0.1014    | +0.0078 | 28.92M      | 107.6M         |
  | quickdraw | 0.3352 | 0.1968    | +0.1384 | 75.71M      | 107.6M         |
  | real      | 0.2121 | 0.2137    | -0.0017 | 72.66M      | 107.6M         |
  | sketch    | 0.1122 | 0.0711    | +0.0412 | 28.04M      | 107.6M         |

  Cost-matching rule satisfied: every routed total (router 6,144 + specialist
  19.95-75.71M MACs) is BELOW dense r28 (107.6M), so dense is compared at a
  resolution whose per-image MACs are at-or-above the routed cost and the
  sparse side does not win by being cheaper. This is the ONE registered regime
  where the sparse family wins on accuracy (KS2's registered consequence), and
  it is consistent with the prior-art position (Ghorbani 2020): random-feature
  models cannot beat trained transformers on overall image accuracy; the
  program's measurable win is per-domain at matched-or-lower cost.

**Consequence for Option 1 (deferred full-split, two-family A5 with Arm S).**
Phase 1's job was to decide "whether routing lifts the sparse ceiling at all
before paying the larger compute" — it does NOT (KS1). Per the registered
decision rule, Option 1's expensive premise failed and the program
consolidates; the larger full-split compute is not justified by a
ceiling-lift question. Option 1 remains available ONLY if a representation-
family question (spherical SDF vs patch-dictionary at fixed budget) is pursued
explicitly — that question is genuinely unanswered by Phase 1, but it is no
longer motivated by ceiling-lift and should be weighed against its cost.

**M113 — Learned (fitted) dictionary vs random dictionary, 6 August 2026.**
Every sealed sparse figure (M107/M108/M109/M110/A2/A5) uses a dictionary of
RANDOM whitened patches — or at best a _selection_ from that random pool.
M108's measured "learning" (arms c select*discriminative, e ridge-leverage)
was importance sampling, not fitting: at 3,072 atoms it scored 0.2138 / 0.2120
vs random 0.2153 — **selection from a random pool does not beat the pool**.
The atoms have never been \_fitted* to the whitened-patch manifold. So the
~0.21 frozen ceiling is the ceiling of a _random_ basis, and whether a
_learned_ basis lifts it is untested.

**Question.** At matched atoms (3,072) and matched encode MACs, does a fitted
(k-means/VQ) dictionary lift the frozen sparse ceiling above the random
dictionary? Is the ceiling a property of the sparse family or of the random
basis? (Ghorbani 2020's limit is on _random_ features; a fitted basis is the
untested regime.)

**Arms** (shared subsample, global M108 whitener, triangle encode, 2x2 pool,
closed-form ridge head penalty 1.0 = M108's chosen constant, fit on all
138,000 train rows, per-domain eval):

- **(a) random-3072** — M108 arm (a) exact construction, re-measured, gated to
  reproduce M108's sealed a_random_3072 (0.2153) within 0.002 or the run is
  VOID. Reference ceiling.
- **(b) learned-3072** — mini-batch k-means (VQ) centroids fitted on a
  registered 2,000,000-whitened-patch pool (3,000 train images, seed 22), GPU.
  Same atoms count, same 108-dim space -> identical encode MACs to (a).
  **Primary arm.**
- **(c) learned-topk-64** — arm (b)'s dictionary, ridge on the full-width code
  with only the top-64 nearest-atom activations per patch nonzero (zero-padded
  to the full atom dimension, so per-image MACs are the SAME as (b) — sparse
  code, dense storage). Reported, not a verdict: does top-k sparsity help
  accuracy at matched cost? The genuine head/encode cost cuts (compact sparse
  code + sparse ridge; approximate neighbor search) are deferred.

**Kill switches (registered before measurement):**

- **KS1 (learned lifts the ceiling):** if (b) does not beat (a) overall by
  > = +0.01 at matched atoms/MACs, the random basis is not the binding
  > constraint and the learned-dictionary thesis fails at this budget.
  > Registered prediction: k-means centroids remove a random draw's Poisson
  > sampling noise, so (b) should beat (a) by a modest amount; whether it clears
  > +0.01 is the open question.
- **KS2 (vs dense at-or-below cost):** (b) at 254.6M total MACs vs best M107
  dense point at-or-below that cost (dense r42: 0.1972 at 215.6M). If (b) <
  0.1972 + 0.01, no global accuracy win is licensed. (Honest note: (b) pays
  ~18% more MACs than r42, so this is accuracy-at-cost, NOT an efficiency
  claim; the efficiency regimes are per-domain — A5 KS2 — and the deferred
  approximate-search top-k.)
- **Per-domain (reported):** (b) vs (a) and vs M107 dense per-domain,
  extending A5's per-domain story to a learned global basis.

**Cost.** One-time k-means fit = O(iters . batch . atoms . dim), reported as a
disclosed training-time cost (analogous to the dense pretraining disclosure).
Encode/head accounted per section 2.9.3(iv) as M108. Run cost ~= 3 x
3072-atom encode passes ~= 1-1.5 h.

**M113 result (SEALED, admissible, 6 August 2026).** Evidence:
`logs/results/v16/m113_learned/evidence.json` (payload-sha256'd). Instrument
clean: arm (a) random-3072 reproduces M108's sealed 0.21528 with delta
+0.00000 (t1 passed); device gfx1201 verified.

- random-3072: **0.2153** (reference ceiling, re-measured).
- learned-3072 (k-means/VQ, 2,000,000-whitened-patch pool, 60 iters): **0.2101**.
- learned-topk64: **0.1795**.

**KS1 FIRED (decisive negative).** Learned - random = **-0.0052** (needed
+0.01). Fitting the dictionary does NOT lift the ceiling — it slightly hurts
on EVERY domain (per-domain deltas all in [-0.008, -0.002]). The ~0.21 frozen
ceiling is the **sparse family's**, not the random basis's. The
learned-dictionary (VQ) thesis is closed at 3,072 atoms. VQ fit quality
disclosed: 311/3,072 dead units reinit (10%); even a converged VQ would need
+0.01 and the measured direction is uniformly negative, so the conclusion is
robust to fit quality.

**Top-k sparse codes HURT** (-0.031 overall, negative on every domain,
-0.039..-0.013 per domain): the soft triangle code over ALL atoms carries the
signal; hard top-64 truncation destroys it. Arm (c) also does not cut cost
(zero-padded to the full atom dimension). Dead end as designed; any
sparsification/quantisation that wants to win the MAC axis must PRESERVE the
soft geometry rather than truncate it.

**KS2 did NOT fire — the licensed win is pre-existing and non-specific.**
learned-3072 0.2101 vs dense r42 0.1972 = +0.0129 at 254.6M vs 215.6M MACs
(accuracy-at-cost, +18% MACs, not efficiency). Random-3072 already beats dense
r42 by +0.018 (M108 sealed curve), so this gap is the pre-existing
frozen-sparse-vs-small-dense difference, NOT a learned effect.

**Consequence.** The learned-basis line is closed for the VQ family. The
program's only measured win remains A5 KS2 (per-domain routed specialists at
1.4-7.2x lower cost). Remaining breakthrough bets, by evidence support:
(1) a hardware-native **binary soft-code encode** that preserves the triangle
geometry in Hamming space — the MAC axis, never won, and arm (c)'s failure
explicitly warns that the soft signal must be preserved; (2) the **full-split
Arm S** representation question (the only non-patch basis, genuinely
unmeasured).

**M114 — Binary soft-code encode: can Hamming (XOR/popcount) preserve the
frozen sparse accuracy at a fraction of the encode cost? 6 August 2026.**
M113 arm (c) showed hard top-k truncation destroys the soft triangle code
(-0.031). But the signal lives in the SOFT assignment over all atoms — so
quantise the DISTANCES (learned binary hash -> Hamming), never truncate the
assignments. ITQ-style learning (Gong et al., CVPR 2011) fits a rotation so
sign codes preserve Euclidean geometry; Hamming distance is XOR + POPCNT, each
a single CPU instruction, so the binary encode needs NO GPU cdist GEMM. The
sparse family has never won the MAC axis; this attacks it while preserving the
soft signal.

**Arms** (shared subsample, M108-exact whitener, the SAME random-3072
dictionary as M113, closed-form ridge head penalty 1.0, per-domain eval):

- **(a) float cdist** — QUOTED from M113 sealed random-3072 = 0.2153 @ 254.6M
  MACs (M113 re-measured M108's with delta +0.00000 on 6 Aug; the dictionary
  is rebuilt by the identical deterministic construction). Not re-run.
- **(b) binary RANDOM-108** — seeded Gaussian projection -> 108 bits, Hamming
  triangle, ridge. Control: does hash LEARNING matter?
- **(c) binary ITQ-108** — learned hash fitted on a registered 100,000-
  whitened-patch pool (seed 33, 50 iters) -> 108 bits (the 108-dim whitened
  patch's linear-projection limit), Hamming triangle, ridge. **PRIMARY.**
- **(d) binary ITQ-64** — bit-width sensitivity (reported, not a verdict).

**Kill switches (registered before measurement):**

- **KS1 (hash learning matters):** if (c) does not beat (b) by >= +0.01, the
  learned hash adds nothing over random bits.
- **KS2 (the MAC-axis breakthrough):** if (c) >= float(0.2153) - 0.01 AND the
  binary total ops <= float total MACs / 3, the sparse family wins the MAC
  axis at preserved accuracy — the first measured time.

**Cost (registered).** Float = whitening (P*108^2) + encode (P*108*A) + head
(A*4*classes). Binary = whitening + projection MACs (P*108*B) + Hamming ops
(P*A*words XOR + P*A\*words popcount, words = ceil(B/64), each 1 op — a 64-bit
XOR and a 64-bit POPCNT are each single CPU instructions, so counting them at
1 op each is conservative-to-fair against a 108-dim MAC dot) + head. B is
capped at 108 = the whitened patch's dimensionality (a linear projection has
at most 108 informative bits). The binary encode runs on CPU (no GPU cdist);
disclosed as part of the hardware story. At B=108: binary total ~30.2M ops vs
float 254.6M MACs (~8.4x), B=64 ~22.3M (~11.4x).

**Design gaps found and fixed, 6 August 2026 (before implementation).** A
pre-implementation audit surfaced five gaps that would confound the run; each is
fixed here:

1. **The router is undefined and untested on DomainNet.**
   `src/model_fingerprint.py` is model-identity/swappability hashing, not an
   image→domain dispatcher; `src/candidate_routing.py` was built for the GEODE
   orchestrator, not for 6-style-domain dispatch. Routing error would be
   indistinguishable from specialist quality. **Fix:** the **oracle router
   (true domain labels) is the primary control** — it isolates specialist
   quality; a **registered fingerprint router** (non-dense feature and cost
   defined below) is a secondary arm whose routing accuracy is measured and
   reported as a gate beside every routed figure. Family and vs-dense verdicts
   (KS1–KS3) are read off the **oracle-routed arms**; a routed arm below a
   registered routing-accuracy floor is reported with that caveat, never as a
   family verdict.
2. **Per-domain sample floor not checked.** Per-domain specialists see ~1/6 of
   the shared subsample (~23k rows, ~67 rows/class/domain); the §3.5 floor
   (11.23 rows per fitted dimension) gates each specialist's budget. **Fix:**
   per-domain row counts and the floor are registered and checked as a gate; a
   domain that cannot meet its floor is reported under-supported, never as a
   negative.
3. **Arm D not pinned.** **Fix:** Arm D is **re-measured** DINOv2-small under
   the same protocol (never quoted from M107), at registered resolution(s)
   chosen so per-image MACs ≥ the routed total cost (cost-matching rule), with
   per-domain evaluation.
4. **The non-dense rule must cover the router.** If the router's fingerprint
   were a DINOv2 feature it would pay the dense opponent's trunk. **Fix:** the
   registered fingerprint is non-dense (cheap raw-pixel statistic / small fixed
   transform); a DINOv2-feature router is a separate disclosed,
   efficiency-excluded arm only.
5. **Per-domain reporting + cost matching not specified.** **Fix:** total cost
   over the test set = router + one specialist per input, reported per domain;
   all arms report per-domain accuracy (the hypothesis is domain-specific).

**Kill switches (registered if pursued).**

- **KS1 (family):** if Arm P and Arm S do not differ beyond a registered margin
  at the fixed per-expert budget on the **oracle-routed** arms, the
  representation family is not the deciding variable and no family claim is
  made.
- **KS2 (vs dense):** if neither family beats Arm D at matched total cost on at
  least the non-natural domains (quickdraw, sketch, clipart), the routed-
  specialist idea adds nothing beyond M109 and the program consolidates.
- **KS3 (prior transfer):** if Arm S under this protocol reproduces v13/v14's
  failure (no dominance), the v13/v14 negative transfers to classification and
  the SDF family is closed for the efficiency claim.

---

## 4. Recommendation

1. **Do A2 now** (cheap; makes M109 trustworthy; feeds M110's kill switch 2).
2. **M110 is done** (6 August 2026; parameter axis; both kill switches fired —
   see v16 §5.3 and ledger C110.1).
3. **A5 next, ahead of A1**: the two-family (SDF vs patch-dictionary) routed-
   specialist comparison. Register it in v16 style before measurement: question,
   the two operands, the non-dense basis rule, the per-family configs, the
   SDF-kernel-fix precondition, kill switches KS1–KS3, and the prior-art sweep
   for the fingerprint-router + hash-layer overlap. The router/fingerprint cost
   is an operand, never excluded.
4. **Then A1** (closed-form training-cost efficiency of a no-backprop learner) —
   measurement framed, never a novelty claim.
5. **Defer A3 and M111/M112** (all test the closed inference-MAC crossing);
   keep A4 as the consolidation/fallback.
6. **M113 now (6 August 2026)**: the learned (fitted) dictionary vs random
   dictionary ceiling test — every sealed figure so far used a RANDOM (or
   selected-from-random) basis; the atoms have never been fitted. M113 asks
   whether a k-means/VQ dictionary lifts the 0.21 ceiling at matched atoms and
   MACs. It is the cheapest test of whether the ceiling is the family's or the
   random basis's; its KS1 decides whether the learned-basis line (then +
   per-domain routing from A5, + approximate top-k) is worth pursuing as the
   efficiency breakthrough.

## 5. What is not changed

v16 registrations, operands, kill switches and predictions stand. Any avenue
above that proceeds must be registered in v16 style (question, operand, kill
switches, prior-art sweep, config before measurement) before it produces
claimable figures. This document is the revisitable record of the decision
space, not a claim.
