# RESEARCH IMPLEMENTATION PLAN v22

## Re-exploring the additive approach: staged architecture, construction search, integration, and growth

Date: 14 August 2026. Status: registered plan; execution underway (see the
execution log at the end).

**How to read this document.** This is a fresh start, written in plain English so
a software engineer can follow it without milestone archaeology. Every finding
from the past is restated here as: _what we asked, what we did, what happened,
what it implies, and what it does NOT imply._ Milestone IDs appear only as
pointers to sealed evidence, never as substitutes for explanation. All future
results are recorded in this same style.

---

## 1. The goal

We want to know whether a system built from **additive, non-trained
constructions** (fixed feature encoders plus closed-form fits) can learn and
adapt across tasks — static image tasks, and temporal tasks that unfold over
time via a **delay+feedback mechanism** (reservoir computing) — and whether it
can be **more efficient than trained deep networks**, in particular more
efficient than the industry default of training a large network and pruning it
down.

Our working hypothesis: **building up additively may beat training-big-then-pruning
in the low-cost regime**, because additive components are fitted in closed form
(no optimizer), are deterministic and auditable, and can be grown one piece at a
time targeted at where the system currently fails.

Everything below is prior art as a mechanism. The contribution is **sealed
measurement of which combination works**, on a fixed corpus, at matched cost.

---

## 2. What we already know — plain-English evidence

Each row states the question, the answer, and the scope. The scope column is
binding (see section 3): a result may only be used inside its scope.

| #   | Question we tested                                                                      | What happened (plain English)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | What it implies                                                                                                     | What it does NOT imply (scope)                                                                                                                                                                                                  |
| --- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E1  | Can a fixed, never-trained image dictionary compete with trained networks?              | We built a dictionary of random whitened image patches; each image is coded by how strongly each patch matches each dictionary atom (a soft "triangle" code), pooled into one vector, and classified by a closed-form linear fit. Full-corpus result: 26.1% top-1 on 345-class DomainNet (409,832 rows) with the 6,144-atom construction, ~500.7M MACs/image by the sealed ledger. Accuracy-wise this beats DINOv2-small at low resolutions up to r56 (24.5% @ 367.5M) but at 1.36x the per-image cost, NOT fewer. The cost-efficiency claim that does hold is the smaller 3,072-atom construction (254.6M MACs): it beats the matched dense arm (r42, 19.7%) by 1.8 points at ~2x fewer MACs at 138k rows (M113/M121 sealed). | The approach is real in the low-cost regime: a never-trained system holds the cost-accuracy frontier up to a point. | That it can reach trained-network accuracy at high compute — its best (26.1%) is about half the best dense accuracy (53.8%) at ~12x fewer operations; the cheapest sealed point is ~24x cheaper.                                |
| E2  | Does adding more dictionary atoms keep helping?                                         | No: beyond ~6,144 atoms, accuracy stops rising. The code's information lives in roughly 8 effective dimensions regardless of atom count.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | For THIS construction, width is not the lever; the code compresses hard.                                            | That every additive construction is low-rank: we never changed the pooling or the coding rule (see the v22 factorial, section 6).                                                                                               |
| E3  | Does more training data help?                                                           | Yes, strongly, all the way to the full corpus: 22.5% at 138k rows → 26.1% at 410k rows, with the gain accelerating at the end.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Data is the measured lever for this family.                                                                         | That the absolute level transfers to other corpora; the steepness may, the level probably won't.                                                                                                                                |
| E4  | Do deeper (multi-layer) closed-form heads help?                                         | No: stacking 2–3 closed-form layers on the code did not lift accuracy.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | For this code, a single linear readout is enough.                                                                   | That depth never helps any additive code.                                                                                                                                                                                       |
| E5  | Do trained heads help on these codes?                                                   | A trained linear head did worse than the closed-form one (about 15% vs 21.5%). On dense DINOv2 features the same head family did BETTER than the closed-form fit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Heads and features must match: trained heads exploit rich features; closed-form fits suit this sparse code.         | That trained heads always fail on additive codes — the opposite held on dense features.                                                                                                                                         |
| E6  | Does the head objective matter (regularization strength, smoothed labels, margin loss)? | Regularization is flat across two orders of magnitude; label smoothing is provably just a re-scaling of regularization for this head (verified exactly); a margin objective did not converge at the registered settings.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | For this code, the bottleneck is the code itself, not the head.                                                     | That head choices never matter — it is specific to this code and those schedules.                                                                                                                                               |
| E7  | Do different random draws of the dictionary help?                                       | No: two independent draws, averaged at the score level, reached 22.1% vs one wider pool's 22.5%, and the effective rank stayed ~8.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | The low-rank structure belongs to the patch-coding RECIPE, not to any particular draw.                              | That different recipes (different pooling, scales, coding) cannot help — they were never varied.                                                                                                                                |
| E8  | Does learning the dictionary help (k-means, discriminative selection)?                  | At this scale, no. On a smaller corpus (CIFAR-10) greedy selection halved the atom count, but that did not transfer here.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Dictionary learning, as we tried it, does not fix the recipe.                                                       | That growth is dead: we never grew a component targeted at the system's actual errors (residual-targeted growth, section 7).                                                                                                    |
| E9  | Can we compress the codes to binary?                                                    | Not for accuracy (~3 points lost, and it does not close with more data or width). Yes for a cost-only route.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Binary codes are a cost tool, not a quality tool.                                                                   | That no compression of additive codes can ever work.                                                                                                                                                                            |
| E10 | Do per-domain specialists beat one global model?                                        | Per domain, yes: each specialist beats the global model on its own domain at ~5.6× less compute. But assembled into one system with a domain router, the system loses: hard routing scored 18.8%, and even with PERFECT routing the ceiling was 20.5% — below the global model's 22.5%. The reason is data: each specialist sees only its domain's rows (~1/6 of the data).                                                                                                                                                                                                                                                                                                                                                    | Specialization wins locally but loses pooled, for this split.                                                       | That the specialist idea is dead: mixing specialists' scores (fusion) instead of hard-picking one was never tried, and it is mathematically guaranteed to be at least as good as any single arm (section 6, integration layer). |
| E11 | Can a cheap linear router identify the task from the codes?                             | At coarse grain, yes: domains are identified at 75.6% (with style-adjacent confusions: paintings misread as photos ~47% of the time). At fine grain, no: classes are identified at only 22.5%.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Task identity is cheaply inferable at coarse granularity only.                                                      | That fine-grained routing is impossible — it was never attempted with fusion or competence signals.                                                                                                                             |
| E12 | Engineering pieces                                                                      | (a) A contract gate rejects out-of-contract inputs with ZERO learned compute while preserving in-contract accuracy exactly — the one clean integration success, and its recipe matters (exact contract + preservation test). (b) A programmatic count memory works at KB scale with an optimum window of 4 on a small language task. (c) On that same task, tiny trained transformers beat every fixed construction by ~10× — so learned components are used wherever the measured price of learning pays.                                                                                                                                                                                                                     | The system's shell (contracts, gates, memory) is built and measured; learned fallbacks have a measured role.        | That fixed constructions can compete on sequences — they measurably cannot.                                                                                                                                                     |
| E13 | Can we predict, from the code's spectrum, when the sparse system beats the dense one?   | No: spectral/margin models predict the crossing wrongly in both directions. Kept as diagnostics only.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | The win is a thin-margin, argmax phenomenon that theory does not yet track.                                         | Anything about new constructions — this was about prediction tools for one code.                                                                                                                                                |

Sealed evidence for all rows lives under `logs/results/`; the scope-annotated
archive is `analysis/LESSONS_ARCHIVE_v22.md`.

---

## 3. The new epistemic rules (binding)

The past habit of treating construction-specific results as general verdicts is
retired. Three rules now bind every future milestone:

1. **Scope-bound citation.** Any document citing an old negative against a new
   design must state the negative's measured scope and show the new design is
   inside it. Otherwise the citation is void.
2. **Re-test on axis change.** If a new design varies the axis a negative was
   measured on, the negative becomes a registered PRIOR (a predicted outcome),
   never a pre-emptive rejection. We measure anyway; either outcome is recorded
   as progress (prior confirmed → scope extended; prior refuted → scope
   contracted).
3. **The archive grows, it never re-decides.** New measurements extend
   `LESSONS_ARCHIVE_v22.md`; old verdicts are not rewritten, only re-scoped.

---

## 4. The staged system architecture

A system that learns and adapts, in eight stages. The central invariant, stated
once: **everything the system can know exactly must be checked before anything
it knows fuzzily** (exact contract → cheap construction → fuzzy router →
learned fallback only where it pays).

```
input
  → S1 fingerprint & contract check (exact, zero-cost)
  → S2 additive embedding (fixed construction; the "code")
  → S3 task identifier (fuzzy, cheap linear router)
  → S4 dispatcher (contract + identity + state → chooses model group)
  → S5 task models (global model + specialist registry, closed-form fits)
  → S6 output contract (typed validation + confidence → accept/reject)
  → output
  S7 state & memory (task history, reject log, performance ledger, windowed
     memory; for temporal tasks it also holds the reservoir's echo state)
  S8 adaptation loop (new task → fingerprint update → closed-form fit/extension
     → acceptance gate → promotion; growth lives here)

temporal branch (tasks that unfold over time):
  S2 code at each time step
    → memory: fixed random reservoir (delay + feedback; echo state held in S7)
       or additive tap-delay line (no feedback; concatenate the last k codes)
       or programmatic primitives (Python ring buffers, counters, FSM states)
    → S5 temporal readout (closed-form fit on the memory state: next-step or label)
    → S6 output contract (same gates as the static branch)
```

- **S1 fingerprint/contract:** typed metadata (shape, dtype, domain, provenance).
  Exact, not inferred. This is what makes out-of-contract rejection free (E12a).
- **S2 embedding:** the additive code. The design space of this stage is the
  v22 factorial (section 6). For temporal tasks, the same code is fed one time
  step at a time into the delay+feedback branch below.
- **S3 identifier:** infers coarse task identity from the code (E11). Fine
  identity is not assumed possible.
- **S4 dispatcher:** chooses which model group runs, from contract + identity +
  state. Rule-based today; a planner hook is reserved.
- **S5 models:** one global model plus a registry of specialists. Fitted in
  closed form; data-elastic (E3).
- **S6 output contract:** validates the output's type and confidence; low
  confidence routes to rejection or fallback.
- **S7 state/memory:** everything the system remembers, feeding S3/S4/S8. For
  temporal tasks, S7 also holds the memory state — the reservoir's echo state,
  the delay-line window, or the programmatic primitives' state — that carries
  information across time steps.
- **S8 adaptation:** how the system changes over time — registering tasks,
  extending data, adding components. Growth is a first-class operation here,
  and so is its inverse: **group splitting** (split-and-rebuild, section 7).

**The temporal branch — delay + feedback (reservoir computing).** A fixed
random reservoir is the additive family applied to time: the input code is
mixed into a pool of randomly connected units (the delay line), each unit's
new value is a combination of its previous value and the input (the feedback),
and the pool's state — the echo state — is what a closed-form readout fits.
Nothing in the reservoir is trained; only the readout is solved in closed form
(ridge, exactly like the static branch). Two properties make it a natural fit
for this programme: the **echo state property** (the pool must fade old inputs,
which is guaranteed by keeping the spectral radius of the connection matrix
below 1) is a contract we can CHECK exactly before trusting any readout; and
the readout stays a plain closed-form fit, so every piece of the static
branch's machinery (data lever E3, growth, gates, fusion) transfers unchanged.
Two clarifications to keep the family straight: nothing in the reservoir is
PRUNED — the recurrent weights are drawn once at random and frozen forever,
and neither training nor pruning ever touches them; and feedback is NOT a
necessity — the same memory can be built purely additively, with no recurrence
at all (a tap-delay line that concatenates the last k codes into one vector
for the readout, or decayed running sums). And where the task is programmable,
the **exact-first** option is plain Python state — ring buffers, counters,
running statistics, finite-state machines (the programme's programmatic
primitives line, E12b and its literature review) — feeding the same ridge
readout. Phase T registers all three arms, so the feedback version must beat
the plain-sum version AND the hand-written-state version to justify its cost.
The screen for this branch is Phase T (section 6) and milestone M147 (section 9).

---

## 5. Options per stage (what we need to explore)

| Stage | Options to explore                                                                                                                                                                                                                                               | Notes                                                                                                                       |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| S1    | Typed specs (shape/dtype/domain/range); row-wise guards; provenance tags; cost-class tags; learned/derived contracts                                                                                                                                             | Typed specs measured (E12a); learned contracts open                                                                         |
| S2    | (a) sealed baseline (6×6 patches, triangle code, one global pool); (b) spatial-pyramid pooling (1×1+2×2+4×4); (c) power-normalisation after pooling; (d) multi-scale patches (3/5/7); (e) cosine patch coding; (f) k-means dictionary; combinations of the above | Each option is prior art; each is aimed at a measured defect: pooling fixes spatial loss, power-norm fixes rank correlation |
| S3    | Linear identity router (measured 75.6% domains); k-NN; competence router (routes to whichever model is likely correct — section 6); confidence-gated routing; tiny learned router; HTN planner                                                                   | Identity routing is a proxy; competence routing is the real target                                                          |
| S4    | Rule-based cheapest-primitive; budget optimizer (buy cheapest learning that clears the contract); cascade/reject thresholds                                                                                                                                      | Rule-based built; optimizer open                                                                                            |
| S5    | Global closed-form model; per-domain specialists; **late fusion** over all models' scores; sparse mixtures; tiny trained fallbacks; trained/MLP heads as co-adaptation reads                                                                                     | Fusion is mathematically ≥ any single arm it contains (E10's fix)                                                           |
| S6    | Typed output validation; confidence → reject; calibration; per-domain operating points                                                                                                                                                                           | Thresholds tuned at system level, never per stage                                                                           |
| S7    | None; windowed count memory (w≈4); per-task ledgers; competence history                                                                                                                                                                                          | Feeds growth decisions                                                                                                      |
| S8    | Closed-form refit on extended data; task registration; head-only updates; **residual-targeted growth**; replay buffers; drift monitor                                                                                                                            | Growth is section 7                                                                                                         |

**Temporal-branch options (extending the matrix above):** S2 — fixed random
reservoir (echo-state property: spectral radius < 1; weights drawn once, never
trained or pruned), leak rate, sparsity; **non-recurrent additive variants:**
tap-delay line (concatenate the last k codes), decayed running sums,
multi-window sums; **programmatic primitives (Python):** ring buffers,
counters, running statistics, finite-state machines — deterministic,
auditable, zero learning (the E12b line; see
`PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`); S5 — ridge readout on the echo
state, the concatenated delay-line vector, or the primitive states (next-step
prediction or label); S7 — the echo-state buffer, the delay-line window, or
the primitive state itself. All variants run on CPU and reuse sealed codes
(Phase T, section 6).

**Adaptation-axis addition (S8):** group splitting — when a registered task
group is later discovered to be two sub-tasks, split it into two closed-form
specialists and promote only if the fused pair beats the incumbent
(section 7, M149).

---

## 6. The search strategy (keeping the combinatorics under control)

**Budget in encodes, not cells.** Only S2 options (and new specialist
dictionaries) cost GPU hours (~1 hour each). Everything downstream is a fit on
cached codes — minutes. Rule: never re-encode for something that can be fitted.

**Phase A — construction screen (few encodes).** Fix everything else at the
sealed winners (closed-form head, full data, linear router) and screen 4 S2
cells: spatial-pyramid pooling, power-normalisation, both combined, multi-scale.
Gate: beat the current frontier (Q(6144, 409832) = 0.261362 at the sealed
6144-atom construction's ~500.7M MACs/image) at matched cost. Every cell
carries three cheap reads to protect against the co-adaptation blind spot:

- **ridge read** (the standard closed-form head);
- **trained-head read** (one small trained head on the same frozen code);
- **closed-form joint-tuning read** (tiny joint grid over the construction's
  scalar knobs + the head).

A construction is only closed as a negative after the ridge AND trained-head
reads fail. Near-misses get a rescue round with the trained head.

**Phase B — combination sweep (free).** On the winning constructions' cached
codes, sweep the cheap axes: heads, routers, fusion, calibration, memory rules.
Dozens of cells, minutes each.

**Phase C — interactions and escalation.** Test the few places the evidence says
interactions live (construction × data, construction × head, router ×
specialists), then escalate winners up the data ladder.

**Phase T — the temporal track (delay + feedback).** Reservoir computing is
cheap to screen: standard sequence benchmarks (one-step-ahead prediction on a
chaotic series, a discrete sequence task) run on CPU in minutes, and a
DomainNet variant feeds the SEALED image codes as a stream, so nothing new is
encoded. Controls, registered in advance: the echo-state property is verified
(spectral radius < 1, warm-up discarded) BEFORE any readout is trusted; the
no-memory baseline is ridge on the current step alone; the **additive arm** is
a tap-delay line (concatenated last-k codes + ridge, no feedback); the
**programmatic arm** is a small Python state machine (ring buffers, counters,
running statistics) feeding the same ridge readout — the reservoir must beat
all of them to justify its recurrence; the trained baseline is a small GRU
plus the E12c prior (tiny transformer on the language task). Scope note:
the reservoir family WAS measured on the DSL next-token task in M134
(r128 ppl 29.72, r512 27.54 — both losing to count-memory w4 3.32 and the
transformer 2.80), so on THAT axis a reservoir loss is a REGISTERED PRIOR
(E12c/M134), not an open question; M147 re-tests the reservoir on a new axis
(one-step-ahead on a chaotic series — the ESN's home turf) where the prior
does not bind, and adds the two arms M134 never ran: the additive tap-delay
line and the programmatic primitives (epistemic rule 2; M134 is a prior, not
a verdict).

**The separability assumption is registered and tested, not trusted.** Phases
A/B assume stages combine independently; we test that assumption at the
interaction cells and correct it where it fails.

**Integration layer (why local wins stopped translating, and the fixes).**
E10 and the v7-era integration failures each have a specific fix:

1. **Route on competence, not identity:** train the router to predict which
   model will be correct (using cached per-model predictions), not which label
   the row has. Identity accuracy is a diagnostic; outcome accuracy is the
   metric.
2. **Fuse instead of dispatch:** one closed-form fit over the concatenated
   scores of all specialists plus the global model. Guaranteed ≥ any single
   arm; hard dispatch stays only where cost demands it.
3. **Interface sufficiency:** every interface is defined by the downstream
   stage's acceptance test, not the upstream stage's metric.
4. **Operating-point co-design:** all thresholds tuned jointly at system level
   on held-out rows; per-domain operating characteristics reported.
5. **Joint-budget allocation:** the final integration cell sweeps the winners'
   budgets together (construction width × data × head capacity), because
   measured interactions mean one-at-a-time tuning can sit at a jointly bad
   point.
6. **Preservation contracts:** every integration step must first reproduce its
   sealed parts exactly, then be judged on its delta (the E12a recipe).
7. **Negative-control the integration:** fused system vs best single arm;
   competence router vs identity router vs random router. Integration that
   does not beat its best part is recorded as a finding about the interface.

**One end-to-end arbiter (M146, section 9):** a single cell with gradients
through the additive code, to MEASURE what full co-adaptation would buy. It is
a measuring stick, not the destination. Its outcome decides what we ship:
frozen system (if it ties), hybrid (if tiny learned parts win), or
small-trained-network-with-contract-gate (if end-to-end wins everywhere).

---

## 7. Growth: building up instead of pruning down

**Why growth is hard for deep networks (and why that pushed the industry to
train-big-then-prune):** growing a unit mid-training breaks credit assignment
(the gradient has to learn what the new unit is for while everything moves);
grown/sparse structures map badly to GPU layouts; and big-blob training follows
predictable scale laws, so companies can forecast it.

**Why our family does not have those problems:** components fit in closed form
(a growth step is an exact solve, not an optimizer walk); arithmetic stays
deterministic; and the data lever is measured (E3), so grown systems can be
budgeted against a measured curve.

**How growth plugs into the stages:**

- **Grow where:** S7 tracks per-row competence; growth targets the rows where
  the fused system currently fails — the new component is fitted on the
  system's RESIDUAL errors. This is gradient-boosting over closed-form base
  learners.
- **Grow what:** an atom, a specialist, a new-scale dictionary, or reservoir
  neurons — appended to the registry, never retraining what exists. (Reservoir
  growth keeps old connection weights fixed; only new units' weights are drawn,
  so the echo-state property can be re-verified after each append.)
- **Grow which:** the dispatcher picks the cheapest component that clears the
  contract (the measured "price of learning" rule).
- **Grow when:** the adaptation loop triggers on contract failures or drift;
  growth is auditable and reversible via the transactional machinery.

**Splitting is growth's sibling — split-and-rebuild.** The inverse case: a
task was registered under one group A, and later the competence ledger shows
A's errors have split into two subpopulations X and Y — A is now more
complicated than it should be. The recipe is the same transaction:

1. **Trigger from S7:** bimodal residuals, drift, or a new task fingerprint
   landing inside A with distinct statistics.
2. **Propose:** cluster A's rows into X and Y (e.g., 2-means on the codes or
   on residual directions) and fit two closed-form specialists.
3. **Test:** fused {X, Y} vs incumbent A on held-out rows; promote only with
   a registered margin; otherwise keep A.
4. **Promote transactionally:** atomic registry swap, rollback kept.
5. **Repeatable indefinitely — but the gate enforces the measured limits:**
   each child must keep enough rows (E3's data lever), and the fused split
   must beat A at matched total cost (E10's data-starvation lesson).
   Splitting is "like pruning" only in direction: pruning removes units to
   shrink a big model; splitting redistributes rows among finer models,
   starting from a small one — capacity is spent only where the data shows
   structure, and the promotion gate is what makes "without losing accuracy"
   an enforced property rather than a hope.

Trained variant: when the group model is a trained component, split by
branching two children from A's weights and fine-tuning under the section
11.2 lock/unlock protocol; the promotion gate is identical.

**Honest caveat:** "additive is more efficient than pruning" is currently a
hypothesis with partial support (E1, and 4.58M additive parameters beating a
22.3M same-data trained model). No pruned-dense comparison has ever been run.
The registered control (M144, section 9) settles it on this corpus.

---

## 8. Prior art map — what is concluded, and the gap we explore on top

| Area                                         | Concluded in prior art (we will NOT redo)                                                                              | The gap we explore on top                                                                                                                         |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Learning curves of fixed features            | Bordelon–Canatar–Pehlevan et al.: the accuracy-vs-data theory for random features + ridge is complete.                 | Measure the curves for OUR constructions; never claim the theory.                                                                                 |
| Joint model×data scaling                     | Kaplan, Chinchilla, Muennighoff, et al.: compute-optimal scaling is established for trained networks.                  | Measure whether the same shape holds for a non-learned system (E3 is our instance).                                                               |
| Pre-deep image coding                        | Fisher Vectors, VLAD, spatial pyramids, power-normalisation: concluded since ~2012–2016.                               | Re-measure their COMBINATIONS inside our staged system at matched cost — nobody did.                                                              |
| Scattering / fixed filter banks              | Mallat et al.: concluded.                                                                                              | Not on our critical path; an option in S2 only if a screen suggests it.                                                                           |
| Bilinear/covariance pooling                  | Lin et al. 2015 onward: concluded.                                                                                     | Same as above.                                                                                                                                    |
| Domain-routed MoE specialists                | Med-MoE, DA-MoE, AnchorMoE, et al.: an active concluded line, all trained end-to-end.                                  | Closed-form specialists + late fusion + competence routing, measured for footprint/energy.                                                        |
| Sparse growth training for DNNs              | DST/RigL, grow-and-prune, Structured Continuous Sparsification, MoE growth, lottery tickets: an active concluded area. | Growth inside a closed-form family, residual-targeted, with a pruned-dense baseline — not done.                                                   |
| Pruning                                      | Established (magnitude, structured, lottery tickets).                                                                  | A pruned-dense baseline at OUR compute budgets, so "additive vs pruning" is a measurement, not an opinion.                                        |
| Tool use / contracts / reject-option routing | Toolformer onward; selective classification (Geifman–El-Yaniv); typed tool schemas.                                    | Our no-LLM-at-inference contract-gated variant is built (E12a); nothing novel claimed.                                                            |
| HTN + LLM planning for routing               | ChatHTN and neighbours.                                                                                                | The exact combination (HTN dispatch over non-LLM specialists, symbolic at inference) is unresolved-not-novel; registered as a future option (S4). |

Temporal additions to the map (for the delay+feedback branch):

- **Reservoir computing / echo state networks — concluded:** Jaeger's echo
  state networks (2001, 2007) and the liquid-state-machine line (Maass et al. 2002) established the echo-state property, memory capacity, and closed-form
  ridge readouts. We do NOT redo that theory. **Our gap:** measure a fixed
  random reservoir inside this staged system at matched cost, with
  residual-targeted reservoir growth, contract-gated streaming, and trained
  sequence baselines measured alongside.
- **Sequence baselines — concluded:** trained RNN/GRU/transformer results on
  standard sequence benchmarks. **Our gap:** use them only as measuring sticks
  for the reservoir, never as claims of our own.
- **Programmatic primitives — concluded:** hand-written state machines,
  counters, and buffers are the classical control baseline, and the programme
  has its own literature review
  (`PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`). We do NOT redo it. **Our
  gap:** run primitives as an arm of the temporal screen, feeding the same
  ridge readout, so the reservoir is measured against the exact-first
  alternative rather than only against learned and random ones.

**Everything we measure is prior art as a mechanism.** Our contribution is the
sealed measurement of combinations, at matched cost, on a fixed corpus — the
thing the field's separate-stage studies never assembled.

---

## 9. Registered milestones (hypotheses and gates in plain English)

| Milestone                                               | Hypothesis, in one sentence                                                                                         | Gate (kill switch)                                                                                                                                                                                                                                          | Cost class                                 |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **M142** — construction factorial screen                | A better construction (spatial pooling / power-norm / multi-scale) beats the sealed frontier at matched cost.       | Any screened cell beats Q(6144, 409832) = 0.261362 at matched cost (the sealed 6144-atom construction's ~500.7M MACs/image, the registered M142 atom-solver target); a cell failing both its ridge and trained-head reads is archived as a scoped negative. | 4 encodes (~4 h GPU) + cheap fits          |
| **M143** — integration layer on cached sealed artifacts | Late fusion and competence routing recover the specialist gains that hard routing lost (E10).                       | Fused ≥ best single arm (verified, not assumed); competence router ≥ identity router ≥ random router; system-level thresholds on held-out rows.                                                                                                             | Fits only (minutes)                        |
| **M144** — pruned-dense baseline                        | A structured-pruned DINOv2-small at our compute budgets sets the true bar for "additive vs pruning".                | Reported as a baseline curve; no win/loss gate, it is a measuring stick.                                                                                                                                                                                    | 1–2 pruning passes + fits                  |
| **M145** — residual-targeted growth                     | Growing new closed-form components on the fused system's residual errors beats static fusion at matched total cost. | Growth curve beats static fusion by a registered margin at matched total ops; control = blind greedy selection (the E8 prior).                                                                                                                              | Fits + a few encodes of grown dictionaries |
| **M146** — end-to-end arbiter                           | One cell with gradients through the additive code measures the price of freezing.                                   | No win/loss gate; its outcome selects what we ship (frozen / hybrid / trained-small-with-gate).                                                                                                                                                             | One trained run                            |
| **M137** (carried) — HTN dispatcher                     | Plan-structured dispatch over specialists beats rule-based dispatch on robustness/cost.                             | Registered separately later; not on the critical path.                                                                                                                                                                                                      | Fits                                       |

**M147 — temporal-memory screen (reservoir vs additive vs programmatic).**
Hypothesis: a fixed random reservoir — or its purely additive tap-delay-line
cousin, or plain Python programmatic primitives — plus a closed-form ridge
readout predicts sequences in the low-cost regime at matched cost. Gate: beats
the no-memory baseline (ridge on the current step alone) by a registered
margin on the registered sequence benchmarks; the reservoir arm is compared
against the tap-delay-line arm (concatenated last-k codes + ridge) and, where
the task is programmable, against the programmatic arm (ring buffers,
counters, FSM states + ridge), so feedback must earn its keep over plain sums
AND hand-written state; the echo-state property is verified before any
reservoir readout is trusted; multi-seed reservoirs; the trained baseline
(small GRU + the E12c prior) is measured alongside. Scope: the reservoir's
loss on the DSL token task is already sealed (M134 r128 29.72 / r512 27.54
vs count w4 3.32 / transformer 2.80) and is carried as a registered prior;
this screen re-tests on the chaotic-series axis where that prior does not
bind, and on the DSL only as an anchor reproduction. Cost class: CPU fits;
reuses sealed codes (no new encodes).

**M149 — group splitting (split-and-rebuild).** Hypothesis: when a registered
task group shows two subpopulations, a transactional split into two
closed-form specialists plus a fused readout beats the incumbent group without
accuracy loss, repeatably. Gate: fused {X, Y} ≥ incumbent A + a registered
margin on held-out rows; per-child row floor enforced (data starvation, E10);
promotion transactional with rollback; controls: a random split and a blind
2-means split, so the split signal must be real rather than an artefact of the
proposal step. Cost class: fits only (minutes), plus one cheap specialist
encode where a new dictionary is required.

Every milestone carries: t1-style anchor reproduction of its sealed inputs before
any new number is trusted; a premise check that every cell is feasible BEFORE
compute is dispatched; smoke runs that refuse the sealed output directory; and
results recorded in the plain-English style of section 2, with scope labels.

---

## 10. Execution order and budget

1. **M142** (construction screen) — the only expensive phase; ~4 encodes, one
   day of the same compute the programme has been running.
2. **M143** (integration layer) — free; runs on cached artifacts while M142
   encodes.
3. **M144** (pruned-dense baseline) — small; in parallel with M142.
4. **M147** (reservoir screen) — CPU-cheap; runs in parallel with everything
   above.
5. **M146** (arbiter) — after M142, on its winners and the sealed baseline.
6. **M145** (growth) — after M143 gives us the fused system to grow.
7. **M149** (group splitting) — after M145; splitting is growth's inverse and
   reuses the same registry, fusion, and transactional gates.
8. **M137** (HTN dispatcher) — whenever the dispatcher becomes the bottleneck.

Total committed compute before the first re-decision point: roughly one week of
the same throughput the programme has used for every milestone so far (the
reservoir track adds minutes, not days).

**Re-decision points:** after M142+M144 we know whether better constructions
exist and where pruning actually sits. After M143 we know whether integration
recovers the specialist gains. After M147 we know whether delay+feedback earns
its place on the temporal path or whether plain sums or programmatic
primitives hold it. After M146 we know the price of freezing. After M149 we
know whether split-and-rebuild recovers structure the original registration
missed. Each point re-sizes the rest of the plan; nothing after a
re-decision point runs blind.

---

## 11. The operational cycle — how a winning combination is trained and validated

The answer changes the moment gradients couple, so this section is written in
two regimes. Every rule below is part of the plan, not an afterthought.

### 11.1 The frozen system — one pass, no epochs

1. **Seal splits once** — train / validation / held-out test, plus per-group
   folds. Registered before anything is fit.
2. **Fit pass (single pass, no epochs)** — stream the train rows once; each
   component accumulates its Gram matrix and label cross-terms and solves
   exactly (ridge). Global model, each specialist, each group-membership
   model, and the router are all separate exact solves on the same cached
   codes.
3. **Fusion pass** — one more exact solve over the concatenated scores
   (stacking weights). This is the only global pass, and it is one solve, not
   a loop.
4. **Calibrate pass** — per-group confidence thresholds on validation rows
   (reject-option tuning).
5. **Validate** — one scoring pass; record accuracy per group, MACs, and
   reject rate. Done.

Components never lock/unlock here: fitting one component does not disturb the
others, because a ridge solve on top of a fixed code has no gradient to leak.

### 11.2 The trained scenarios (hybrid / trained-small-with-gate) — the full cycle

If M146's arbiter says gradients pay, the loop returns — including the two
failure modes seen before: overlapping-group boundaries and missing
rejections. Both are handled by protocol, not by hope:

1. **Epoch loop with a locked inner schedule:** heads fit on frozen codes →
   a few encoder steps with heads frozen → heads refit → next epoch. This is
   the standard linear-probe-then-fine-tune pattern, applied stage by stage
   instead of blindly end-to-end.
2. **Grouped structure, soft dispatch:** a coarse group decision in front,
   then a within-group decision; the group decision is always soft (weights,
   never hard picks — E10's lesson). Ambiguity between adjacent groups stays
   visible to the fusion layer instead of being resolved by whoever's
   gradients are bigger.
3. **Rejection is never an emergent property:** the exact S1 contract gate
   stays in front of learned compute in every scenario (E12a). In-contract
   uncertainty gets a rejector tuned on validation — a separate instrument,
   not a softmax byproduct. If a rejector is trained, it is trained with an
   explicit abstention target.
4. **Per-group metrics every epoch:** pooled accuracy plus per-domain floors,
   because pooled numbers hid E10's local losses before. Early stopping on
   pooled AND per-group floors together — a group may not be sacrificed to
   lift the average.
5. **Granularity is capped by evidence:** coarse identity is learnable
   (75.6% domains), fine identity is not (22.5% classes, E11). Grouping
   belongs at the coarse tier; below it, soft membership only.

### 11.3 Where iteration genuinely lives (both regimes)

- The **design search**: change one cell → refit cheaply → validate → keep or
  archive. This is a research loop, not a training loop.
- The **joint-tuning grids**: tiny grids over construction scalars + head —
  exact solves per cell, minutes each.
- The **growth loop**: fit on residuals → append → re-solve fusion → validate,
  bounded and registered. Growth stays **lock-by-construction**: old weights
  are never touched, the new component is the only free thing, and the fusion
  is re-solved, not retrained. This is what keeps growth from re-introducing
  the credit-assignment mess the frozen family avoids.
- **The split cycle** (M149): propose a split from S7's ledger → test the
  fused pair against the incumbent → promote or roll back. One transaction
  per split, no epochs.
- **Deployment refits**: new rows arrive → re-solve only the affected
  components → promote only if the newcomer beats the incumbent on held-out
  rows, else roll back (transactional).

### 11.4 When lock/unlock is legitimate

Lock/unlock cycles are needed only where gradients couple: inside the hybrid's
trained parts, and in the M146 arbiter itself. The frozen core, the exact
gate, and grown components never unlock. The research controls (trained-head
read, closed-form joint-tuning read) exist precisely to MEASURE what that
locking buys before it is ever trusted.

---

## 12. Execution log (live; results recorded in the section 2 style)

### 14 Aug 2026 — M147 temporal-memory screen: PASSED (sealed)

Question: on one-step-ahead Mackey-Glass prediction (tau=17, RK4 dt=0.1,
sampled every 1.0 time unit), does the reservoir earn its recurrence?
Result (NRMSFE): no-memory ridge 0.1459; tap-delay k=8 0.0181; programmatic
primitives 0.0032; reservoir best per seed 0.0027 / 0.0022 / 0.0025
(u=1024, rho 0.9-0.99), beating the best non-recurrent arm by >=5% relative
on all three seeds; echo-state property verified per run; the M134 DSL anchor
reproduced exactly (delta 0.000000); the tiny MLP baseline scored 0.1504 as
configured (unconverged Adam — a measuring stick, not part of the gate).
Implication: feedback earns its keep on the chaotic-series axis.
Scope: the M134 prior stands (reservoir LOSES the DSL token task); the two
axes disagree, so the reservoir's value is axis-dependent. Programmatic
primitives remain strong (0.0032); the reservoir edge is real but modest.
Evidence: `logs/results/v16/m147_temporal_memory/evidence.json`.

### 14 Aug 2026 — registered corrections (applied before M142 dispatch)

- E1's frontier claim conflated two constructions. Corrected: 26.1% is at the
  6144-atom construction (~500.7M MACs/image), beating dense r56 on accuracy
  at 1.36x cost; the 31%-fewer-MACs claim belonged to the sealed 3072-atom
  construction vs dense r42 at 138k rows. Phase A and M142 gates are now
  registered against Q(6144, 409832) = 0.261362 at matched 500.7M cost.
- Phase T's "a reservoir was never measured" was wrong: M134 measured
  reservoirs on the DSL (r128 ppl 29.72, r512 27.54, both losing). That loss
  is now carried as a registered prior; M147 re-tested the new axis.
- M148 survey sealed (claim ledger: `analysis/PRIOR_ART_V22_FINAL.md`).

### 14 Aug 2026 — M143 integration layer: SEALED NEGATIVE (scoped), rescue registered

Full run: all six specialist anchors bit-exact (delta +0.000000), class head
0.2246, router 0.7559. Phase 2 (fusion + competence router fit on a 50/50
split of the sealed test rows): fused 0.1463, competence 0.0557, identity
0.1861, random 0.0772, global 0.2251 -> gate fired.
Implication: on the held-out-fit protocol the integration layer does not
recover the specialist gains.
Scope: a diagnostic on the cached scores showed the fitter is sound
(global-only stacking 0.159 vs global 0.2251 is a rows-per-class effect: the
fusion saw 40 rows/class vs the arms' 400). The negative binds only the
held-out-fit protocol.
Rescue registered (rule 2): **M143b** fits the same fusion and competence
router on the arms' own TRAIN scores (phase 1 caches specialist/global score
matrices for the full train split, ~2.5h GPU) and evaluates on the sealed
test scores.
**M143b SEALED (14 Aug):** fused 0.2243 vs global 0.2246 — the train-score
fit fully recovers the global arm (M143's 0.1463 was the held-out-fit
artefact, confirmed) — but does not exceed it; competence routing 0.1826
loses to identity routing 0.1877. Gate fired (competence clause).
Implication: on this corpus the integration layer adds no measured value
over the global arm: fusion ties it, competence routing loses to the simple
domain router. The interface itself is the finding.
Evidence: `logs/results/v16/m143_integration_layer/evidence.json` and
`logs/results/v16/m143b_train_fusion/evidence.json`.

### 14 Aug 2026 — M149 group splitting: SEALED NEGATIVE (scoped)

Full run on the M143 test-score cache: zero domains passed (d0 incumbent
0.0414 / fused 0.0650 / random 0.0836; d2 0.0160/0.0412/0.0519; d3
0.1657/0.1431/0.1313; d4 0.1043/0.1002/0.0761; d1, d5 skipped on the
cluster-degeneracy floor).
Implication: on this protocol the split operation has no measured room.
Scope: the held-out incumbent re-fit (1400 rows) weakens the baseline; the
proper test (split children fit on the group's TRAIN rows vs the real
specialist incumbent) is **M149b**. Evidence: `logs/results/v16/m149_group_split/evidence.json`.
**M149b SEALED (14 Aug):** domain 2 PASSES — fused(2-means) 0.1276 vs the
real incumbent 0.1111 AND vs fused(random) 0.1242. Domains 0/3/4: the split
beats the incumbent (+4.1/+1.8/+5.0 points) but NOT the random-split control
— the gain there is capacity, not structure; domain 1 neither; domain 5
skipped (cluster degeneracy). Gate passes (>= 1 domain).
Implication: split-and-rebuild is a measured, promotable registry operation
where the data shows real two-subpopulation structure (one domain here);
elsewhere splitting adds capacity without structure. The controls did their
job: the random split caught capacity-only gains.
Evidence: `logs/results/v16/m149b_group_split/evidence.json`.

### 14 Aug 2026 — M142 factorial: C1 power-norm re-adjudicated within instrument

C1 sealed run exposed two registration defects, disclosed in the evidence:
the p=1.0 anchor's invariance claim was false (per-row scaling is not
argmax-invariant once an intercept exists) and the C1 fitter scores the raw
codes 0.2195 rather than the sealed fit path's 0.22487. Substantive numbers:
p=0.5 ridge cells 0.2328/0.2332/0.2363 vs the raw same-fitter reference
0.2195 = +1.68 points at matched cost; ext600 (207k) 0.2483.
**C1b SEALED, PASSED:** raw same-fitter anchor exact (p=1.0+L2 vs raw delta
+0.000000); best cell 0.2363 >= 0.2195 + 0.005 -> the power-norm lifts the
sealed codes within instrument. Trained-head reads collapsed again (0.003)
even at the a2 converged schedule — measured, E5-consistent, disclosed.
Implication: construction CAN be varied to beat the sealed recipe at matched
cost; the trained-head blind spot persists on these codes.
Evidence: `logs/results/v16/m142_factorial/evidence.json` and
`logs/results/v16/m142_factorial_c1b/evidence.json`.

### 14 Aug 2026 — M143/M149/M142 build and smoke history (superseded by the entries above)

M143's first smoke exposed the stacking/router overfit at penalty 1.0
(fit 1.0 / eval 0.026); the penalty ladder on a validation slice fixed it.
A second smoke exposed the score-derived competence target's scale
dependence; amended to the label-derived correct-arm target (N97.3).
M142 C1's original anchor ("p=1.0 + L2 must reproduce 0.22487 exactly")
was misregistered: per-row scaling is not argmax-invariant once an intercept
exists, and the C1 fitter scores the raw codes 0.2195 rather than the sealed
fit path's 0.22487 — see the re-adjudication entry above.
cells C2 (spatial-pyramid), C3 (multi-scale) and C4 (SPM + power-norm) are
registered but not yet dispatched: the full-data codes were never cached and
a fresh encode is ~2h per cell (M141 took 2.25h).

### 14 Aug 2026 — M142 C2 build: registered corrections and amendments (before dispatch)

The C2 runner, configs and tests are written
(`eval_v16_m142_c2.py`, `m142_c2{,_smoke}.json`,
`test_v16_m142_c2.py`, 11 tests green). Three things were corrected at
registration, all recorded BEFORE any C2 accuracy was measured:

1. **Matched-MAC atom count.** The handoff's "~5,383" does not satisfy its
   own equation (5,383 -> 475.2M, not 500.7M). The registered equation is
   operative: (500,711,184 - 8,503,056) / (729*108 + 729 + 21*345) =
   5,676.75 -> **atoms = 5,677**; C2 total = 500,733,018 (+0.0044% of the
   sealed total). The runner gates on this arithmetic at startup.
2. **Encoder anchor (mean-coupling).** The triangle activation subtracts the
   mean distance over the WHOLE atom set, so 5,677-atom codes are NOT
   column-comparable to the sealed 6,144-atom codes (the dictionaries nest
   exactly; the columns do not — measured during the smoke, max delta 0.55).
   t1 therefore runs the SPM encoder at the sealed 6,144 atoms and requires
   the 2x2 level to reproduce the sealed f6144 memmap bitwise. Smoke result:
   delta 0.000e+00.
3. **Fit amendment — REFUTED before dispatch by the pre-dispatch validation,
   superseded.** The first fit amendment (reduced-rank ridge, top-r SVD of
   the standardised code) was tested on the sealed f6144 codes at 138k rows
   before any C2 measurement: the direct ridge reproduces the sealed
   reference exactly (0.224609 vs 0.22487), but the reduced-rank ridge
   scores 0.0441 (r=256), 0.0264 (r=1024), 0.0302 (r=2048) — delta ~-0.19 at
   every registered r. Cause: the ridge solution's energy is concentrated in
   the DENSE middle of the spectrum (where sigma ~ sqrt(penalty) ~ 1), not
   in the top-r singular directions; the sealed code's spectrum decays so
   slowly (tail 2e-4 at r=1024) that truncation discards the solution. The
   reduced-rank read is void.
4. **Second exact-subspace route — also REFUTED before dispatch.** A
   Lanczos-Galerkin solve ((G+lambda I) W = C solved in the k-dimensional
   Krylov subspace of the Gram, one basis shared by all 345 label columns,
   2k data passes) was probed on the same sealed codes: k=25 -> 0.032,
   k=50 -> 0.049, k=100 -> 0.067, k=200 -> 0.086 vs direct 0.2246. The
   plateau is thousands of dimensions away; at the C2 width each k costs two
   full-data passes (~6 min), so the exact ridge solve at width 119,217 is
   NOT executable on this machine. Per-class LSQR is ruled out by the 345x
   RHS factor. Verdict: the registered matched-cost cell (5,677 atoms,
   width 119,217) cannot be fitted with the sealed ridge head here.
5. **Re-scope registered BEFORE measurement (the pre-announced fallback).**
   The C2 question — does spatial-pyramid pooling beat a single 2x2 pool at
   matched per-image cost — is answered at a width-feasible cost point. The
   atom pair is solved from the same ledger equation as before, with pool
   adds counted on BOTH arms (both sum the same 729-activation map) and the
   widths capped for the sealed Gram fit (peak = 3 x width^2 x 8 bytes):
   b*(729*108 + 729 + 4*345) = a*(729*108 + 729 + 21*345), i.e.
   b*80,841 = a*86,706. Registered pair: **b = 2,062 (single pool),
   a = 1,923 (SPM, 21 bins)**, both ~175.2M MACs/image (delta 0.02%), SPM
   width 40,383 (peak ~39 GB RAM), pool width 8,248. Everything else is
   unchanged: same whitener/dictionary prefixes, same row schedule, same
   sealed direct-fit path (no surrogate solver anywhere — t3 is dropped),
   same ridge ladder, per-level diagnostics, trained-head read. The baseline
   encode is premise-pinned by the sealed atom ladder at 138k: Q(2062, 138000) must land within +-0.002 of the monotone envelope
   [Q(1536,138000), Q(3072,138000)]. Gate: Q_SPM(1923, 409832) >=
   Q_pool(2062, 409832) + 0.005 at matched cost. The sealed
   Q(6144, 409832) = 0.261362 is reported as context, not the gate.
   NO C2 accuracy figure is measured until this re-scope is registered.

Smoke (4,000/1,000 rows, gates skipped) ran end-to-end through the
re-scoped pipeline: encodes, anchors, direct ladder fits, per-level reads,
scoring; tiny-cell accuracies ~1/345 are the expected smoke artifact.
Four smoke defects were found and fixed before dispatch (solve-key types,
config truncation, Standardiser leak into the evidence, cache-dir creation
order) plus one full-run defect found at dispatch (schedule part 1 must
index the subsampled corpus POSITIONALLY, not with raw-corpus indices —
fixed and re-dispatched; no accuracy was produced before the fix).

### 14 Aug 2026 — M142 C2 spatial-pyramid vs single pool: SEALED PASSED (re-scoped)

Question: does spatial-pyramid pooling (21 bins) beat a single 2x2 pool at
matched per-image cost (~175.2M MACs, the width-feasible re-scope pair
pool b=2,062 / SPM a=1,923)?
Result: **PASSED.** Q_SPM(1923, 409832) = 0.260493 vs Q_pool(2062, 409832)
= 0.227536 at penalty 1.0 -> gain +0.03296 (>= +0.005) at matched cost
(175,238,694 vs 175,197,198 MACs, delta 0.024%). All anchors held: t1 both
encoders bit-exact (0.000e+00); t2 sealed full-data reproduction exact
(delta -2.8e-16); t4 premise Q(2062, 138000) = 0.2064 inside the sealed
atom-ladder envelope [0.1970, 0.2153]. The SPM ridge ladder is flat
(0.2598/0.2605/0.2598 at lambda 0.1/1/10); the pool arm's best is 0.2286
(lambda 0.1). At 138k rows: SPM 0.2145 vs pool 0.2064. Per-level
diagnostics: 1x1 -> 0.1539, 2x2 -> 0.2240, 4x4 -> 0.2601 — the fine level
carries the pyramid (the 4x4 level alone nearly equals the full pyramid at
full data; the coarse levels add ~nothing there). The trained-head read on
the SPM codes collapsed (0.0089, E5-consistent) but the closure rule needs
BOTH reads to fail, so the cell stands positive on the ridge read.
Implication: E2's scope note was the right one — the sealed recipe's 2x2
pool was discarding measurable spatial information, and finer pooling is
the measured fix. The SPM construction reaches the sealed full-data
frontier (0.2614 at 6144 atoms / 500.7M MACs) at ~2.9x fewer MACs, and
beats the dense ladder's r56 (0.2450 @ 367.5M) on BOTH accuracy and cost.
Scope: one construction axis (pooling granularity), one corpus, one cost
point, the closed-form head. It does NOT say the sealed recipe's spatial
loss transfers to other corpora, nor that trained heads can use the SPM
code (they measurably cannot, E5).
Evidence: `logs/results/v16/m142_c2/evidence.json`.

### 14 Aug 2026 — M142 C4 power-normalisation on the SPM codes: SEALED PASSED

Question: does the Fisher-vector post-processing (signed power + per-row
L2) lift the SPM construction the way it lifted the sealed codes (the C1b
prior: +1.68 at 138k)?
Result: **PASSED.** The t1 anchor reproduced the sealed C2 read exactly
(raw refit 0.260493, delta +3.9e-16). The best cell (p=0.5, lambda=0.1)
scores 0.278551 at full data — gain +0.0181 over the raw same-fitter
reference (0.260493), gate cleared. The gain is entirely the square root:
p=1.0 (L2 alone) HURTS slightly (0.2544-0.2551 at full data); p=0.5 lifts
every penalty (0.2767-0.2786). At 138k: p=1.0 -> 0.2106 (below the raw
0.2145), p=0.5 -> 0.2274 (+1.3 over raw). The trained-head read collapsed
again (0.0029, E5-consistent); the closure rule needs BOTH reads to fail,
so the cell stands positive on the ridge read.
Implication: the C1b transform transfers to the SPM codes and stacks with
the pooling win — the promoted SPM-family recipe is 21-bin SPM + signed
square root + per-row L2, ridge (lambda 0.1), scoring 0.2786 at ~175.2M
MACs/image: +1.7 points over the sealed full-data frontier (0.2614 at
500.7M) at ~2.9x fewer MACs, and +3.4 points over dense r56 (0.2450 at
367.5M) at half the cost. The dense ladder still leads above ~0.31
(r70 0.3118 at 564.2M).
Scope: one construction axis (power exponent), one corpus, one cost point,
the closed-form head. The trained-head blind spot persists on these codes.
Evidence: `logs/results/v16/m142_c4/evidence.json`.

### 14 Aug 2026 — M142 C3 multi-scale 3/5/7 vs single 6x6 pool: SEALED PASSED

Question: does a three-scale patch construction (3x3 + 5x5 + 7x7, one 2x2
pool per scale, concatenated) beat the single-scale 6x6 pool at matched
per-image cost (~175.2M MACs)?
Result: **PASSED.** Q_MS(409832) = 0.242145 vs Q_pool(2062, 409832) =
0.227536 at penalty 1.0 -> gain +0.0146 (>= +0.005) at matched cost
(175,153,892 vs 175,197,198, delta 0.025%). Anchors exact: t1 bitwise
determinism 0.0; t2 sealed full-data reproduction delta -2.8e-16; t4 pool
refit delta -2.8e-17. MS ladder 0.2431/0.2421/0.2383 (lambda 0.1/1/10).
At 138k: 0.2157 (pool 0.2064). Per-scale diagnostics: 3x3 alone 0.1823,
5x5 alone 0.1921, 7x7 alone 0.1769 — EVERY scale alone loses to the 6x6
pool (0.2275); the concatenation is additive (the sum beats every part).
Trained head 0.0096 (E5-consistent); closure needs BOTH reads to fail, so
the cell stands positive on the ridge read.
Implication: the construction axis keeps paying. At matched ~175.2M the
factorial now reads: single 6x6 pool 0.2275 < multi-scale 0.2421 <
21-bin SPM 0.2605 < SPM + signed sqrt + L2 0.2786. The SPM family stays
the winner; multi-scale is second. The registered follow-up (multi-scale

- power-norm) is the next free cell.
  Scope: one corpus, one cost point, the closed-form head, three fixed
  scales with equal MAC shares. It does NOT claim the scale combination
  transfers, nor that SPM+MS interactions (unmeasured) are additive.
  Evidence: `logs/results/v16/m142_c3/evidence.json`.

### 14 Aug 2026 — M142 C3b power-normalisation on the multi-scale codes: SEALED PASSED

Question: does the Fisher-vector post-processing lift the multi-scale
construction the way it lifted the SPM codes (the C4 prior: +1.8)?
Result: **PASSED.** The t1 anchor reproduced the sealed C3 read exactly
(raw refit 0.242145, delta -1.4e-16). Best cell p=0.5, lambda=0.1 ->
0.250667, gain +0.0085 over the raw reference, gate cleared. As with C4,
the gain is entirely the square root: p=1.0 (L2 alone) HURTS
(0.2329-0.2370); p=0.5 lifts every penalty (0.2490-0.2507). At 138k:
p=1.0 -> 0.2100 (below raw 0.2157), p=0.5 -> 0.2239 (+0.8). Trained-head
read collapsed (0.0032, E5-consistent); closure needs BOTH reads to fail.
Implication: the power-norm transform transfers to the multi-scale codes
too (smaller gain than on SPM: +0.9 vs +1.8). The factorial at ~175.2M
now reads: pool 0.2275 < MS 0.2421 < MS+sqrt+L2 0.2507 < SPM 0.2605 <
SPM+sqrt+L2 0.2786. The SPM family remains the Phase-A winner.
Scope: one corpus, one cost point, closed-form head; trained-head blind
spot persists.
Evidence: `logs/results/v16/m142_c3b/evidence.json`.

### 15 Aug 2026 — M144 pruned-dense baseline: first dispatch VOID (pixel pipeline), fix registered, re-dispatch

First dispatch (14-15 Aug, overnight): runner, configs, unit tests green
(head-row scoring, exact-subnetwork invariant), smoke passed (parity
1.79e-06; keep=0.5 -> 0.518 nonzero fraction, 185.0M MACs). Full run
completed and VOIDED ITSELF on the t2 anchor: unpruned r56 read 0.196696
vs the sealed M107 r56 0.245014492753623 (delta -0.048319 > 0.002).
Void is recorded in `logs/results/v16/m144_pruned_dense/evidence.json`
(void=true) — the run is void, not negative, and no M144 figure may be
quoted from it.

Root cause (found by diffing against the sealed M107 runner): a pixel-
pipeline mismatch, invisible to the t1 parity guard (which pins
ONNX-vs-torch on synthetic fixed inputs only). The sealed r56 arm
(M107 `d4c_small_56`) fed the ONNX encoder the ORIGINAL-resolution
images decoded from the parquet stream and resized original->56 with
PIL bilinear (`_materialise_original`, digest-tagged cache). The M144
runner instead upsampled the 32x32 decoded cache to 56 with torch
bilinear — a two-step resize with a different kernel, hence different
features, hence a different ridge fit. The registered recipe's
"resolution 56 (the sealed r56 arm)" always meant M107's pixels; the
implementation did not honour it.

Amendment registered BEFORE any re-measurement (15 Aug 2026):

1. The M107 original-resolution r56 pixels are re-materialised with the
   exact M107 function and environment (`_materialise_original` under
   `.venv`, PIL 12.3.0 — the M107 interpreter), digest-tagged
   `domainnet_m107/63f590097008f749/`.
2. The M144 runner now reads those digest-tagged r56 memmaps instead of
   upsampling the 32x32 cache; the evidence records the pixel source and
   tag. The t2 anchor (reproduce 0.245014492753623 within 0.002) is the
   pixel-fidelity gate for any re-materialisation.
3. No accuracy figure from the void run is quoted anywhere; the new run
   writes fresh evidence over the voided one.
   The M144 recipe (measuring stick, no win/loss gate) is otherwise
   unchanged.

### 15 Aug 2026 — M144 pruned-dense baseline: SEALED (measuring stick, corrected pixel path)

Re-dispatch after the amendment. t1 parity 1.79e-06 (bound 1e-4). t2
unpruned r56 = 0.2450144927536232 vs sealed 0.245014492753623, delta
+1.9e-16 — exact reproduction, the void's cause confirmed and removed.
Result (ridge readout, penalty 1.0, the sealed r56 arm's exact pixels):
keep=1.0 -> 0.245014 @ 367,513,344 MACs; keep=0.5 -> 0.107623 @
185,029,632 MACs (0.518 of params nonzero); keep=0.25 -> 0.069478 @
104,443,648 MACs (0.304 of params nonzero). Structured magnitude pruning
collapses the dense arm hard: -13.7 points at half the channels, -17.6 at
a quarter, while the additive SPM+sqrt recipe holds 0.278551 at ~175.2M
MACs — the promoted sparse recipe beats UNPRUNED dense r56 by +3.4 points
at ~2.1x fewer MACs, and the pruned dense arms at matched-or-more MACs by
+13.7 to +17.6. The additive-vs-pruning comparison on this corpus now has
a measured dense side: pruning down to additive-scale parameter counts is
not a path to this frontier.
Scope: one corpus, one model family (DINOv2-small), one resolution (r56,
the sealed ladder level), channel-magnitude pruning of attention+MLP only,
no fine-tuning after pruning (structured prune + retrain is unmeasured).
Evidence: `logs/results/v16/m144_pruned_dense/evidence.json` (voided
first-dispatch figures are NOT quoted anywhere).

### 15 Aug 2026 — M145 residual-targeted growth: SCOPED NEGATIVE (sealed)

Question: does growing one new closed-form specialist on the fused
system's residual errors beat static fusion by the registered margin,
with blind greedy selection unable to explain the gain?
Result: **NEGATIVE.** All anchors held exactly: a1 static reproduction
delta 0.0 (fused 0.1462608695652174, global 0.22510144927536233,
penalty 10000.0); a2 d0 specialist path delta 0.0 (0.19357142857142856);
a3 prefix property ok. Premise: 2,760 error rows; budgets {32, 64}
clear the floor (22 and 11 rows/dim). Gate fired at BOTH budgets:
growth_fused 0.145275 (g=32) / 0.145101 (g=64) vs static_fused
0.146261 — deltas -0.0010 / -0.0012, below the +0.005 margin — and the
blind-greedy control MATCHES the growth arm (0.146725 / 0.146899,
slightly above growth at both budgets). A floor-sized specialist fitted
on the static fusion's 2,760 error rows adds ~nothing to the fused
system, and blind dictionary selection explains the (null) effect as
well as residual targeting does. Growth neither beat static fusion nor
its control, and the fused system stays below the global arm (0.2251).
Implication: residual-targeted growth, as registered, does not pay on
this corpus at the floor-feasible budgets. The scoped finding is about
the growth INTERFACE on M143's fused stack (itself a sealed negative);
it transfers no claim to other corpora or to growth on a healthier base
system.
Evidence: `logs/results/v16/m145_growth/evidence.json`.
Control note: the M108 order-parity check failed on the GPU port; as
registered, the numpy reference ran the full selection (backend cpu,
parity_checked false, recorded).

### 15 Aug 2026 — M146 arbiter: first dispatch VOID (anchor protocol), amendment registered, re-dispatch

First dispatch voided itself on the t1 anchor: r1 0.229623 (best of the
{0.1, 1.0, 10.0} ladder, winning at 10.0) vs the sealed C4 138k read
0.2273623188405797 — delta +2.261e-03 > tol 1e-6. Void recorded in
`logs/results/v16/m146_arbiter/evidence.json`; no M146 figure is quoted
from it.

Root cause: the REGISTRATION misdescribed the sealed read. C4's
`cells_138k` protocol fits penalties **[1.0] only** — the sealed
0.2273623188405797 is the penalty-1.0 read, not the best of a ladder.
The runner implemented best-of-ladder selection, which picks a different
fit (penalty 10.0). Amendment registered BEFORE any re-measurement:
the t1 anchor is the PENALTY-1.0 read (C4's exact protocol); the
{0.1, 10.0} rungs are reported as diagnostics and are never selected
for the anchor. No trained rung (r2/r3) ran in the void dispatch.

### 15 Aug 2026 — M146 end-to-end arbiter: SEALED — freezing holds (the frozen system ships)

Re-dispatch after the amendment. t1 anchor delta +0.000e+00 (0.2273623188405797
reproduced exactly, penalty 1.0). Result (138k level, the shared M109
schedule):

- r1 frozen codes + closed-form ridge: **0.227362** (0 trainable params).
- r2 frozen transformed codes + trained linear head: 0.042609 (val
  0.049130, 4 epochs, 13.9M params) — the E5 trained-head collapse,
  reproduced on the PROMOTED codes.
- r3 trainable dictionary + trained head through the differentiable
  SPM+sqrt encoder (whitener frozen): 0.106029 (val 0.110145, 8 epochs,
  14.1M params).
  Shipping map: r3 − r1 = **−0.1213** — gradients through the additive
  code do NOT pay on the winner construction; co-adapting the dictionary
  lifts the collapse (0.043 → 0.106) but stays 12.1 points below the
  frozen closed-form read. M109's sealed answer on the sealed construction
  (frozen 0.2148 > trained-dict 0.1588 > trained-dict+whitener 0.1302)
  extends to the promoted SPM+sqrt winner: THE FROZEN SYSTEM SHIPS.
  Implication: the price of freezing is NEGATIVE on both the sealed and
  the promoted construction — the closed-form ridge readout over the
  frozen code is not a stand-in for a trained head; it is the better
  system on this corpus. This selects the shipping mode for the v22
  program: frozen system, closed-form heads, with the contract gate in
  front (E12a).
  Scope: one corpus, one construction (the C4 winner), the 138k level,
  one seeded schedule, the M109 t3 structure (whitener frozen). Trained
  whitener rungs and other constructions are unmeasured.
  Evidence: `logs/results/v16/m146_arbiter/evidence.json` (voided
  first-dispatch figures are NOT quoted anywhere).

### 15 Aug 2026 — Whitepaper written and number-verified

`analysis/WHITEPAPER_ADDITIVE_FROZEN_SYSTEM_v22.md` (373 lines) writes the
overall approach up for an outside reader: the three research questions,
the eight-stage architecture + temporal branch, the promoted frozen
construction, the method discipline (anchors, premise gates, smoke
refusal, void-not-negative, scope-bound citation), all sealed results
(M142 factorial, E1-E13, dense + pruned-dense ladders, M143/M143b,
M145, M146, M147, M149/M149b, M103 CIFAR-10), the prior-art claim ledger
with named references and their claims (M148), the shipping mode, scope,
open questions, and a numbered reference list. Every cited number was
re-checked against the sealed evidence on disk in a verification pass;
one typo was found and fixed (M103 per-arm spread is 0.0131, not 0.013),
and the E1 comparison is stated in the corrected form (3,072-atom vs
r42 is a 1.18x-cost win, not fewer MACs).

### 16 Aug 2026 — v23 investigation plan REGISTERED (nothing measured)

`analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md` registers the questions
v22 opened and the cheapest admissible path to them: M150–M164 across
four waves (fits-only -> one-encode cells -> trained runs -> corpus
decision), each cell with its anchors, premise gates, controls, and gate
pre-registered. The v22 epistemic rules carry unchanged. No accuracy has
been measured under v23; M155 gates M156's budgets and runs first.

### Remaining milestones — precise recipes for the next dispatch

- **M142 C2 (SPM 1+4+16 bins) — DONE, sealed PASSED (verdict above).** The
  cached artifacts for the next cells: `v16/m142_c2/spm1923_fulltrain.npy`
  (409,832 x 40,383), `pool2062_fulltrain.npy` (409,832 x 8,248),
  `m142_c2_fulltrain_labels.npz` on the cache drive.
- **M142 C4 (SPM + power-norm) — DONE, sealed PASSED (verdict above).**
  The promoted SPM-family recipe: 21-bin SPM (1,923 atoms) + signed square
  root + per-row L2, ridge lambda 0.1 -> 0.278551 at ~175.2M MACs.
- **M142 C3 (multi-scale 3/5/7) — DONE, sealed PASSED (verdict above).**
  Cached artifacts: `v16/m142_c3/ms357_fulltrain.npy` (409,832 x 13,244)
  and `v16/m142_c2/pool2062_fulltest.npy` (persisted by the C3 run).
- **M142 C3b (multi-scale + power-norm) — DONE, sealed PASSED (verdict
  above).** Promoted MS-family recipe: ms357 + signed sqrt + L2, ridge
  lambda 0.1 -> 0.250667 at ~175.2M MACs.
- **M144 (pruned-dense baseline).** DONE, sealed (verdict above). The
  pruned-dense curve at the sealed dense ladder level: keep 1.0 -> 0.245014
  @ 367.5M; keep 0.5 -> 0.107623 @ 185.0M; keep 0.25 -> 0.069478 @
  104.4M. Pixel path: M107 `_materialise_original` originals, digest-tagged.
- **M145 (residual growth).** DONE, sealed SCOPED NEGATIVE (verdict
  above). Needs M143's fused system: compute fused
  residual errors on the fit half, fit one new closed-form specialist on the
  error rows (dictionary from the global pool, seeded), append, re-solve
  fusion, evaluate on the eval half; control = blind greedy selection (the
  M108/E8 prior). Gate: growth curve beats static fusion by the registered
  margin at matched total ops.
  REGISTERED CELL DESIGN (15 Aug 2026, before any measurement):
  - Base system = the M143 sealed stack on the cached score matrices
    (`v16/m143/scores.npz`): 6 x 512-atom A5 specialists + global f6144
    head, stacking over 7x345 scores, M143 split (seed 33: fit 17,250 /
    eval 17,250 of the sealed test rows), penalty selected on the M143
    valid slice (valid_seed 55, frac 0.8) over {1,10,100,1000,10000}.
    Static-fusion anchor (a1): fused eval must reproduce M143's sealed
    0.1462608695652174 and global 0.22510144927536233 (tol 1e-9, same
    cached matrices, same code).
  - Error rows = fit-half rows the static fusion mispredicts at its
    selected penalty. PREMISE GATE: ceil(n_error / (4\*g)) >= 10 (the
    section 5.3 floor) at every budget, checked in-run before any fit.
    AMENDED 15 Aug 2026 BEFORE any growth measurement (a1 premise read
    only): the registered expectation of ~13.4k error rows was wrong.
    The stacking head fits its own half nearly perfectly (the stacking
    guarantee + fit-half overfit), so the residual population is 2,760
    of the 17,250 fit-half rows (16.0%), measured by the a1 static
    reproduction (bitwise exact vs the sealed M143 read). The original
    budgets {128, 256} are therefore below the floor (6 and 3 rows per
    fitted dimension) and are VOID, not skipped. The budgets are amended
    to the floor-feasible rungs **{32, 64}** (22 and 11 rows/dim; g=64
    is the largest power of two that clears the floor at the measured
    population). No growth accuracy had been measured when this
    amendment was registered; only the static premise had been read.
  - Growth arm per budget g: dictionary = first g atoms of the GLOBAL
    pool in the [11,100]-seeded permutation (= a prefix of the f6144
    dictionary, asserted in-run); encode all 34,500 test rows; ridge head
    (lambda 1.0) fit on the ERROR ROWS ONLY; append as arm 8; re-solve
    stacking (same valid-slice penalty protocol); fused read on the eval
    half.
  - Control arm per budget g: dictionary = first g atoms of the M108
    blind-greedy order (`select_discriminative`, group-OMP vs centred
    one-hot, on the fit-half rows — the E8 prior; GPU port with the M108
    order-parity check); head fit on the SAME error rows; identical
    append + re-solve. Growth vs control differ ONLY in how the
    dictionary is chosen (fixed seeded prefix vs blind greedy); all else
    matched, same encode cost per arm.
  - Anchors: a1 static reproduction (above); a2 specialist encode path —
    rebuild the M108 whitener + domain-0 512-atom dictionary and
    reproduce M143's d0 own-domain anchor 0.19357142857142856 (tol 0.002)
    before trusting any new dictionary's codes; a3 prefix property
    (growth dictionaries nested, g128 subset g256).
  - GATE (per budget, eval half): growth_fused >= static_fused + 0.005
    AND growth_fused > control_fused. Control >= growth at any budget
    means the gain is NOT residual targeting's, and that budget's growth
    claim fails (the E8 prior explains it).
  - Ops ledger disclosed per arm: specialist encode MACs (34,500 rows),
    control additionally its pool encode + OMP selection MACs; growth's
    extra ops vs static fusion are disclosed, not matched away (a grown
    system adds a component by construction; the growth-vs-control
    comparison is the matched-ops one).
  - Smoke declares inadmissibility and must refuse the sealed output
    directory; smoke skips no gate except inadmissibility.
- **M146 (end-to-end arbiter).** DONE, sealed (verdict above): r1 frozen
  ridge 0.227362 > r3 trained dict+head 0.106029 > r2 trained head
  0.042609 — freezing holds on the winner construction; the frozen
  system ships. One cell with gradients through the additive
  code (soft patch encoder + ridge readout trained jointly, seeded schedule);
  measures the price of freezing; selects the shipping mode
  (frozen / hybrid / trained-small-with-gate).
  REGISTERED CELL DESIGN (15 Aug 2026, before any measurement): the
  promoted C4 winner construction (21-bin SPM, 1,923 atoms, signed sqrt
  - per-row L2) at the sealed 138k level. Rungs: r1 frozen codes +
    closed-form ridge (the t1 anchor: reproduce the sealed C4 138k read
    0.2273623188405797 at PENALTY 1.0 — C4's cells_138k protocol fits
    penalties [1.0] only — tol 1e-6, from the cached C2/C4 codes; the
    {0.1, 10.0} rungs are diagnostics, never selected for the anchor);
    r2 frozen
    transformed codes + TRAINED linear head (the E5 read on the promoted
    codes); r3 trainable dictionary + trained head through the
    differentiable SPM+sqrt encoder, whitener FROZEN (the M109 t3
    structure). Shared M109 schedule (AdamW batch 64, cosine 3e-4,
    wd 1e-4, patience 2; r2 4 epochs, r3 8; val frac 0.05, val_seed 66).
    No win/loss gate — a measuring stick; r3 vs r1 selects the shipping
    mode (gradients pay -> hybrid/trained; freezing holds -> the frozen
    system ships, extending M109's sealed answer to the winner
    construction). The trained rungs' codes/features use float32 torch
    arithmetic (the differentiable path); the frozen anchor stays on the
    exact numpy path. Smoke declares inadmissibility and refuses the
    sealed output directory.
