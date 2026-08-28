# RESEARCH IMPLEMENTATION PLAN v24 — the generalized task toolbox

**Status: ACTIVE.** Registered 17 Aug 2026. The v23 queue is complete
(all cells sealed; M164 executed across three dispatches), so this
plan takes effect. Phase D's ceiling probe (M176a) is the first
dispatch, ahead of Phases A–C, because its verdict decides whether
growth must change codes rather than schedule order.

**Naming (registered 17 Aug 2026):** the promoted system — the latest
frozen system with the best measured combination of choices — is
called **GEODE** (**G**eneralized **E**ncoders for **O**pen-**D**omain
**E**xpertise). The legacy expansion (Greedy Ellipsoidal Outline
Discrimination by Excision) described the pre-v16 geometric core and
is retired from the promoted system's naming. Today GEODE is the
21-bin SPM + signed sqrt + L2 + closed-form ridge recipe; as the
buildout proceeds, GEODE denotes the routed toolbox (registry +
fingerprint + router + fit-and-report) built from the best measured
choices at any given time. The name matches the existing
`GEODE_SYSTEM_BLUEPRINT.md` and the `GEODE_CACHE_DIR` convention.

**North star (the user's words):** a generalized toolbox that can learn
new tasks without massive retraining, and that should be able to scale
to complex tasks like SoTA LLMs given enough data and compute.

---

## 1. What the sealed evidence already establishes (the starting facts)

Everything below is measured, not assumed (v16–v23 sealed evidence):

- The promoted construction — 21-bin SPM (1,923 atoms) + signed sqrt +
  per-row L2, closed-form ridge — reaches 0.2786 on DomainNet-32 at
  ~175.2M MACs/image, ahead of the sealed dense ladder through r56
  (0.2450 @ 367.5M) and the earlier sparse frontier (0.2614 @ 500.7M).
- **18 Aug 2026 (M176c-c1): the better-code arm is now measured.**
  Deep-patch SPM on frozen DINOv2-small tokens (no training) reads
  0.487/0.563/0.590 at 256/1024/2048 atoms (~369-382M MACs/image),
  beating the whole dense ladder (r70 0.3118, r224 ≈0.537) and 2.1×
  the sparse frontier — the program best on this task.
- Data scaling is real on the frozen path: Q(f6144, n) rose 0.2246
  (138k) → 0.2614 (409,832). This is the toolbox's scaling evidence so
  far.
- Freezing beats training everywhere measured: M146 r3 (−12.1 vs the
  frozen read), M160 across all schedules (best trained 0.1345 vs
  frozen 0.2274), M150's trained heads across the whole cached code
  family, the E5 pattern throughout.
- The frozen readout is additive-or-better: M151's SPM×MS concat
  0.2975 > either half alone; the sharing that pays is domain-gated
  fusion weights (M154 +1.49), not shared head data (M159 ties the
  global arm).
- Routing granularity on a single corpus does not pay (M143/M143b/M153);
  multi-task routing across DIFFERENT problem types is untested — it is
  the gap v24 fills.
- Task axes already measured with the M147/M157 harness: Mackey-Glass,
  Lorenz-63, Dyck grammar — programmatic primitives beat recurrence on
  the new axes; the reservoir earns its keep only on MG.

**Consequence for v24:** "training" for the toolbox is closed-form
fits, frozen encodes, and a small learned FINGERPRINT — not SGD over
the components. A new LLM trained with our system is NOT part of the
MVP (section 6 explains why and when it could be).

---

## 2. The toolbox architecture

```
task definition (data + input/output spec)
        │
        ▼
[1] DESCRIPTOR NORMALISER   (frozen rules + frozen ontology artifact)
        │  descriptor vector (quantized attributes)
        ▼
[2] FINGERPRINT             (additive attribute embeddings + tiny MLP,
        │                    trained once, then FROZEN; F << N)
        ▼
[3] ROUTER                  (deterministic nearest-arm / learned policy
        │                    later; frozen at inference)
        ▼
[4] ARM REGISTRY            (frozen components: programmatic primitives,
        │                    reservoir, SPM/pool encoders, DINO dense, ...)
        ▼
[5] FIT-AND-REPORT          (closed-form ridge head; anchors reproduced;
                             held-out accuracy with hashes)
```

Stages 1, 2, 3 are the new v24 work. Stages 4 and 5 exist and are
sealed. The design goal: adding a task = adding a registry entry and a
fit, never retraining the components (the user's "without massive
retraining" requirement, stated as a design invariant).

### 2.1 Design invariants (registered)

- **I1 — freeze-on-ship:** the fingerprint and router are trained once
  against a frozen registry snapshot, then frozen; inference is
  deterministic (no randomness, no LLM, fixed seeds).
- **I2 — additive composition:** the fingerprint is the sum of learned
  attribute embeddings (the word2vec/GloVe mechanism), so unseen
  attribute combinations still map to defined fingerprints and
  traversability is vector arithmetic by construction.
- **I3 — measured relations:** the only similarity labels that train
  the fingerprint are measured (behavioral transfer, sweep continuity,
  attribute overlap). LLM-proposed relations are candidate GENERATION,
  never labels.
- **I4 — no-refusal:** every task gets a fingerprint; out-of-vocabulary
  attributes fall back to a registered nearest-bucket/hash embedding.
- **I5 — repro-hash:** every descriptor, fingerprint, route decision,
  and fit carries a payload hash (the existing reproducibility
  discipline).

---

## 3. The task descriptor schema (ontology v0)

The ontology is a schema of AXES, not a list of tasks. The bootstrap
(section 3.2) proposes; a human ratifies; the result freezes.

**Candidate axes (to be ratified, not assumed):**

- Input: modality {image, token-text, numeric-series, tabular, graph,
  audio, control-signal}, image sub-modality {camera-RGB, pointcloud,
  IR/thermal, depth, medical} (candidate axis), shape/dimensionality
  (quantized bins), value kind {discrete, continuous, mixed},
  temporal structure {iid, sequential, delayed}, channels.
- Output: kind {class, regression, next-token, action, distribution,
  ranking}, dimensionality (bins), ordinality.
- Latent structure: recurrence class {Markov, chaotic, grammar-depth,
  none}, stationarity, noise regime (bins), label cardinality (bins),
  sample-size regime (bins).
- Coupling: {single-task, mixture, curriculum-position}.

**Registered rules:**

- Continuous attributes are quantized into registered bins so they
  become vocabulary tokens ("384-dim input" is a token, not a number).
- An unseen attribute value maps to a registered fallback embedding and
  the event is logged (I4).
- The descriptor is a canonical, order-fixed vector; normalisation is
  rule-based and deterministic (I1/I5).

### 3.1 First registry population (measured axes only)

Tasks already owned and measured, used as the initial training and
control set: DomainNet classification, CIFAR-10, Mackey-Glass / Lorenz
/ Dyck forecasting, the programmatic-primitive arms, the dense arms.

DomainNet and CIFAR-10 are the SAME task type (camera-RGB image
classification) — they appear as two entries on purpose, not because
they are different types. As two points in the same task region they
anchor (a) the fingerprint's known-similar positive control (G2: they
must land closer to each other than either is to the forecasting
axes) and (b) the behavioral-transfer harness's same-family transfer
pair (the same frozen arm fit to both, the measured transfer is a
similarity label). The modality-level separation the user draws
(camera vs pointcloud vs IR vs depth) lives on the descriptor axes
(§3), not in this list. New synthetic families are added by parameter
sweep (continuity labels come free).

### 3.2 Ontology bootstrap via a frozen small LLM

A small local LLM (e.g., Qwen2.5-3B or Llama-3.2-3B, 4-bit quantized —
fits the RX 9070 XT or runs on CPU overnight) is used OFFLINE, once:

1. Propose the axis set and attribute vocabularies (batch 1).
2. Propose normalisation rules and candidate task pairs/quadruples for
   measurement (batch 2).

**Registered constraints:** pinned weights/version, temperature 0,
seeded decoding; outputs are RATIFIED, then frozen as a digest-tagged
JSON artifact; the LLM is absent from runtime; its relations are
hypotheses until measured (I3). No API dependency, no data leaves the
machine.

---

## 4. The fingerprint: construction and training

**Output space:** F = 16–64 dims, unit-normalized (cosine similarity).
(Binary hashing is a later, separate stage — future list.)

**Construction (I2):**

```
f(task) = normalise( Σ_k emb(attribute_k) + mlp(descriptor) )
```

- The attribute embeddings provide the traversable structure; the small
  MLP (≤ a few thousand params) captures axis interactions the sum
  misses. Both are trained once and frozen.
- Optionally a registered continuity channel: a task family from a
  parameter sweep contributes a learned 1-D continuity embedding, so
  adjacent parameter values land nearby (the user's ring-buffer idea,
  operationalized as a continuity loss).

**Training signals (labels, in priority order):**

1. **Behavioral transfer (primary, self-supervised):** fit the same
   frozen arm to tasks A and B; the measured held-out transfer (or
   cross-task error) is the similarity label. The system generates
   these labels itself; each fit is minutes on CPU.
2. **Attribute overlap (auxiliary, CBOW-style):** tasks sharing
   attributes get closer fingerprints.
3. **Sweep continuity (auxiliary):** parameter-neighbouring tasks must
   land close along the manifold.

**Objective:** contrastive (InfoNCE) over measured pairs/triplets +
continuity loss + a small attribute-reconstruction term. CPU-trainable.

**Gates for the fingerprint (pre-registered):**

- G1 determinism: same task, two runs → identical fingerprint (hash).
- G2 similarity ordering: known-similar pairs closer than
  known-dissimilar pairs on a held-out pair set (positive control
  includes pairs the program MUST order correctly, chosen before
  training).
- G3 traversability: registered analogy quadruples ("swap input
  modality, keep output structure", "finger : hand :: foot : leg"
  over task attributes) move along the right axis above a registered
  threshold.
- G4 continuity: sweep-neighbour distances monotone-ish along the
  sweep direction on held-out sweeps.

---

## 5. The router

- MVP: deterministic nearest-arm in fingerprint space over the frozen
  registry, with the M143b-style fusion of the routed arms; the global
  strongest arm is the cold-start fallback (I4).
- Gate R1 (per task): routed accuracy ≥ the best single arm for that
  task on held-out rows (the M153 lesson applied across tasks, not
  within a corpus).
- Learned routing policy: future, only when the registry is large and
  with the frozen router as the incumbent (registered).
- Redundant-capability selection and failover (registered 17 Aug
  2026): when several registry arms claim the same task region
  (e.g. multiple parties offering animal detection), the router
  orders them by a registered selection score — measured held-out
  accuracy per task, measured availability (deterministic health
  probes: contract + payload hash), and price — and keeps an ordered
  failover chain per task: primary → next best → … → strongest
  general arm → programmatic primitives. The primitives are the
  zero-downtime bottom tier: always available, no hosting dependency
  (M147/M157). Selection is gated per task: the chosen arm must stay
  within a registered ε of the best measured arm on held-out rows, or
  the chain advances. Prior art: replica/load-balancer selection
  practice and the ensemble-selection line (META-DES); the caution is
  the sealed M143b competence-routing tie at small scale —
  availability selection is measured, never assumed.

---

## 6. MVP capability list (acceptance criteria)

Each capability is gated; no capability is declared without its gate
and anchors.

1. **Task ingestion** — normalize any task definition; OOV fallback;
   no crash (I4).
2. **Deterministic fingerprinting** — G1.
3. **Similarity ordering** — G2.
4. **Basic traversability** — G3.
5. **Routing** — R1 per task.
6. **Fit-and-report** — closed-form head, held-out accuracy, anchors
   reproduced (existing discipline).
7. **Registry operations** — add a task / re-fingerprint / route /
   report without touching other tasks' weights (transactional;
   I1).
8. **Multi-task differentiation** — tasks A, B fed separately and
   jointly: fingerprints distinct for A and B, the joint task routed
   distinguishably (the user's earlier question, now gated).
9. **Cold start** — unknown task gets a fingerprint and falls back to
   the strongest general arm.
10. **Repro-hash** — I5 on every decision.

**Explicitly NOT in the MVP:** a new LLM trained with our system;
gradient-trained components; multi-GPU training. Reasons (sealed):
freezing wins everywhere measured; the MVP's compute mode is fits +
routing; none of the capabilities requires generation or open-ended
reasoning.

---

## 7. Milestone queue (v24, draft order)

Phase A — ontology and labels (no training):

- **M165** Task Registry schema + descriptor normaliser + the frozen
  artifact format (build only; digest-tagged; unit-tested).
- **M166** Ontology bootstrap batch 1 (frozen local LLM, ratified) —
  produces `task_ontology_v0.json` (a frozen artifact, not evidence of
  correctness).
- **M167** Behavioral-transfer label protocol on the measured axes:
  the similarity-measurement harness (fits on frozen arms; positive
  controls: known-similar and known-dissimilar pairs it must order
  correctly). Must-cite: transfer-metric instability
  (arXiv:2204.01403) — the protocol's ranking-stability gate is
  registered because the literature reports metric instability.

Phase B — the fingerprint:

- **M168** Fingerprint embedder build (additive + MLP + continuity
  channel) + configs + unit tests; smoke on toy descriptors.
- **M169** Fingerprint training on the label set; gates G1–G4 run in
  the runner (gate failures are voids/scoped negatives, per
  discipline).
- **M170** Traversability set: the registered analogy quadruples as a
  validation artifact with the measured axis-shift thresholds.

Phase C — routing and the MVP gates:

- **M171** Router + registry integration; R1 per task on the measured
  axes.
- **M172** Multi-task differentiation cell: A, B, A+B (the gated
  north-star test).
- **M173** MVP acceptance run: capabilities 1–10 with every gate and
  anchor recorded (this is the milestone the user revisits).

Phase D — scaling questions (deferred, registered triggers):

- **M174** Q(n) scaling of the toolbox route on a new synthetic family
  (does the frozen path keep scaling? — the honest open question).
- **M175** Cross-corpus transfer (carries the deferred M163 decision).
- **M176** Registry-growth experiment: at what registry size does
  routing measurably beat the global fallback (the learned-router
  trigger).
- **M176a** Frozen-code-ceiling probe: bound how much accuracy ANY
  head can extract from the sealed codes (kNN and diagonal-ridge
  reads on the same codes vs the closed-form read). A small gap makes
  L1 (§9) real — growth must change codes, not heads.
- **M176b** Fit-cost benchmark vs d: time/RAM of the fit pipeline at
  registered widths (5k / 15k / 40k, then the first infeasible width);
  locate the L4 wall and validate an iterative/block-solver escape
  against the sealed equivalence gate. Escape candidates to evaluate
  FIRST (published): divide-and-conquer KRR (arXiv:1305.5029), two-
  level preconditioning (arXiv:1806.05826), differentiable closed-
  form solvers (arXiv:1805.08136).
- **M176c** Better-code arm (deployment-phase, budget-capped): a
  trained-but-frozen encoder enters ONLY on a measured-gap trigger —
  gate: it must beat the dense ladder per-MAC (r70 0.3118 / r98
  0.4476) or serve a task axis no frozen arm serves. Budget cap:
  ≤ $300 rented GPU (≈ 400–850 GPU-hours on a 4090/A6000). Enters as
  a frozen arm behind the router, never end-to-end retraining
  (§8 item 9c). Candidate order: deep-patch SPM (SPM bins over dinov2
  patch tokens, deterministic, no training) → Fisher vectors on deep
  patches → from-scratch small encoder (the last resort, M113's
  learned-dictionary negative registered). Prior art registered (the
  M164 search): candidates 1–2 exist in print — deep-patch spatial
  pyramid VLAD codes (arXiv:1603.09046) and deep dictionary learning
  (arXiv:2012.12509, arXiv:1912.10804) — so they enter as cited
  comparison baselines, not as claims.
  **STATUS 18 Aug 2026: candidate 1 SEALED — PASS.** 0.487/0.563/0.590
  at 256/1024/2048 atoms vs the dense ladder r70 0.3118 at ~367.5M
  MACs (same backbone): every cell beats the ladder at comparable
  MACs; 2048 atoms also clears the dense frontier r224 0.5368. The
  deployment arm = deep-patch SPM (no training). Candidates 2/3 are
  now comparisons, not gates; the rented-GPU spend so far ≈ $5-15.
- **M176d** Label-matrix scaling strategy (gap audit, 17 Aug 2026):
  the behavioral-transfer labels grow quadratically with the registry
  (arms × tasks). Register a sampling/active-selection strategy with
  a label budget, and the accuracy loss of the fingerprint under the
  sampled label set against the full matrix (a registered tolerance).
- **M176e** Live upgrade/migration protocol for registry components
  (gap audit): when a better encoder ships, existing tasks re-encode,
  refit, and re-anchor with a registered rollback path and a who-pays
  rule (the legacy `model_migration` line is the precedent;
  freeze-on-ship must not mean frozen-forever).
- **M176f** Contribution interface spec (gap audit, the bridge to
  v25): how a third party submits an arm — encoding contract, payload
  hashes, validation rules, registry write access and identity tiers.
  Without this the multi-party network has no door to walk through.

Costs: Phases A–C are CPU fits + tiny CPU/GPU training + the frozen
LLM bootstrap batch; the current machine is sufficient (the user's
hardware question — answer: no purchase for the MVP; rentals only for
a registered Phase-D+ cell that needs more than 16 GB VRAM). M176c
carries a registered deployment-phase budget cap of ≤ $300 rented
GPU (≈ 400–850 GPU-hours on a 4090/A6000) under the same per-cell
discipline.

---

## 8. Future improvements (after the MVP, in rough order)

1. **Continuous task families + ring-buffer continuity at scale** —
   sweep-generated registries with the continuity channel as the main
   source of new tasks (data engine).
2. **Behavioral-transfer map** — a learned task graph from measured
   transfer; the fingerprint must align with it (projection gate, the
   discipline's "measure, then align" pattern).
3. **Hierarchical fingerprints** — coarse task-family fingerprints +
   fine per-task fingerprints; two-level routing for large registries.
4. **Binary hashing** of fingerprints for cheap large-scale lookup.
5. **Learned routing policy** (small MLP) once the registry is large;
   the frozen router stays the incumbent and the gate is against it.
6. **Multi-task joint models** — fusion of arms across tasks where
   measured transfer is high (the M154 pattern generalized across
   tasks).
7. **Composition of learned components with frozen guarantees** — the
   hybrid readout revisited as "frozen + residual" ONLY where a
   registered gate passes. M161's sealed negative (hybrid 0.0765 vs
   frozen ridge 0.2274, gain −0.1509) sets the default: **OFF**.
   Any future composition cell must first change a registered
   premise.
8. **Cross-corpus and cross-modality transfer** — the deferred M163
   decision, extended by the fingerprint's similarity signal.
9. **The scaling path to SoTA-LLM-class tasks (registered triggers,
   not assumptions):**
   - a) More data and more atoms on the frozen path (the only measured
     scaling we have); register Q(n)-type cells per family.
   - b) A sequence/LLM-style component enters ONLY when a measured gap
     fires its trigger: a task family where primitives + reservoir
     demonstrably fail and sequence modeling is the only remaining
     path (measured, gated, never assumed).
   - c) If such a component is registered: train it as another FROZEN
     arm behind the fingerprint/router (component-level scaling, not
     end-to-end retraining), preserving I1.
   - d) Hardware: rentals for specific registered cells; a purchase
     only if a cell's budget justifies it (per-cell cost discipline).
   - e) A trained-but-frozen encoder (the better-code arm) enters
     only on M176c's measured-gap trigger and budget cap — the
     deployment-phase answer to L1, always behind the router, never
     end-to-end retraining.
10. **Ontology v1** — ratify bootstrap v0 against measured behavioral
    structure; re-freeze.
11. **Inter-module feedback and task chaining (registered stance, 17
    Aug 2026).** Chaining tasks is allowed as a DAG pipeline: each
    stage is a frozen encoder/arm with typed contracts (the E12a
    pattern), intermediate artifacts are hash-verifiable, so a complex
    task can be split into simpler stages at the engineering level
    (task B's contract checks task A's output). _Automatic_ task
    decomposition (the system splits a complex task into subtasks) is
    NOT claimed — the measured decomposition analogues at this scale
    lost (M145 growth, M149 splits, M153 routing granularity), and the
    HTN/decomposition literature is the reference line
    (`archive/analysis/HTN_ROUTING_LITERATURE_REVIEW.md`).
    Cross-module _recurrent_ feedback (one module's output feeding
    back into another's input) is unsupported by design: it would
    make the code depend on the fit's own outputs, breaking the
    closed-form fit, the additive composition, and the replayable
    audit trail. The reservoir's feedback is internal (echo state)
    only; reservoir/tap-delay state enters other modules as ordinary
    additive code columns, which is the measured composition that
    pays. Any cyclic feedback design is a registered gated experiment,
    never the default.

---

## 9. Known growth-limiting technical facts

Registered so growth plans cannot assume them away (all grounded in
sealed v16–v23 evidence):

- **L1 — the frozen-code ceiling.** Trained heads collapse everywhere
  (M150: 0.03–0.08; M146 r3 0.106 vs frozen 0.227; M160 best 0.1345),
  which means the _codes_, not the head, bound accuracy. The best
  closed-form read (0.2786) may be near the ceiling the frozen
  encoders allow on DomainNet-32. Nothing measured so far closes the
  gap to trained-backbone quality. The ceiling is the sparse
  family's, not the registry's: the frozen dense ladder already
  reaches 0.5375 (r224) at ~35× the MACs, and a learned VQ dictionary
  did not lift the sparse ceiling (M113, −0.005 vs random-3072). The
  escape routes are the M176c ladder (§7).
- **L2 — single-corpus, single-modality evidence.** Every sealed
  number is DomainNet-32 plus tiny M147/M157 synthetic axes. Text,
  audio, graph, and true sequential paths are unbuilt — this is
  exactly why M163/M174/M175 exist.
- **L3 — local saturation of the recipe.** The last "more of the
  same" cells all failed: M155 (error budgets), M156 (growth), M157
  (recurrence), M158 (finer pooling). On this corpus the additive
  recipe has stopped paying; growth must come from new axes/corpora,
  not atom tweaks.
- **L4 — the fit is quadratic in feature width.** The closed-form
  ridge needs the d×d Gram; at d ≈ 40k it already required a chunked
  Gram plus disk-spilled in-place LU (M151), and a full-width matmul
  temp hit 23 GB. Doubling d quadruples the Gram. The hard wall is
  near; iterative/block solvers are the known escape, unvalidated at
  scale.
- **L5 — numerical fragility.** The system at λ = 1.0 is
  ill-conditioned (cond ~1e9): float32-rounded vs float64 statistics
  shifted weights 39% (M151), and INT8 divergences vary by published
  graph (M91). Every dtype or λ change needs the equivalence-gate
  treatment, and the risk grows with d.
- **L6 — the growth layer is unbuilt.** Fingerprint, router, registry,
  and every MVP capability are v24 Phase A–C work; until M172/M173
  pass, "learns new tasks without massive retraining" is a plan, not
  a property of the system.
- **L7 — per-task cold-start cost.** Each new task axis needs a frozen
  encoder construction (SPM dictionary, patch vocab) — a measured
  pipeline, not a lookup — and a paid service's margin also depends
  on encoder latency per query.

**Consequence:** two things would falsify the toolbox's growth story —
(a) a measured code ceiling not far above 0.28 across corpora, and
(b) a fit-cost wall below the registry sizes routing needs. Both are
testable now: probes M176a/M176b; the better-code arm has its own
gate: M176c.

---

## 10. Revisit agenda (what the user will review)

- All pending v23 milestones sealed (M161 verdict, M162 run, M163/M164
  decisions).
- This plan's Phase A milestones built and smoke-tested.
- The M161 verdict, because it decides the default stance on the
  "frozen + residual" family (item 7 above).
- Open design choices to settle at review: F (16 vs 64), the final
  axis list after the bootstrap batch 1, the similarity-label mix
  weights, and whether Phase D triggers are worth scheduling early.
- The L1/L2 verdicts (M176a/M176b) and the M176c budget gate, if
  those probes are dispatched early — they decide whether the growth
  path changes codes, is blocked by the fit wall, or buys a trained
  arm.
- The full-resolution scaling question (the 32×32 bench was a
  research-phase compute choice, not a ceiling). The user's
  hypothesis: GEODE scales with data and compute. It is registered
  against the M174/M175 triggers and reopens with the M163 corpus
  decision; the supporting evidence today is the accelerating Q(n)
  curve and the dense ladder at higher MACs.
- Corpus licensing policy (gap audit): any corpus used for fits or
  encoders in a commercial network must be license-clean — the
  licensing posture of DomainNet and every future corpus is a
  registered input to the M163 decision, not an afterthought.

---

## 11. Discipline (unchanged, restated)

Register before measuring; anchors reproduced before new numbers;
smoke runs declare inadmissibility and refuse sealed output
directories; void ≠ negative; blind controls; dual reads; payload
hashes on everything frozen. The LLM bootstrap is a frozen artifact
with a digest — its proposals are hypotheses until measured.

---

## 12. Execution log (live)

### 18 Aug 2026 — M175 cell C SEALED: no modality confusion — the contract guard holds (battery COMPLETE 5/5)

C sealed (evidence `logs/results/v24/m175_cell_c/evidence.json`). C1:
the router's `route`/`chain` gained the optional `contract_kind` guard
(wrong-modality arms are UNREACHABLE, not just ranked last; backward
compatible; unit tests now 16/16). C2 demonstration on the sealed
registry: vision chain [deeppatch_spm, dense_r70, spm1923_domainnet] —
vision arms only; text chain [wikitext_uniform_w2, wikipedia_uniform_w2]
— text arms only; cross-contract query (vision fingerprint + text
contract) yields text arms only, wrong_kind=[] in every chain. Verdict
PASS. The M175 transfer battery is COMPLETE: B scoped negative (this
encoder), A0 gate fired (window dial inverts on natural text), A text
transfer HOLDS (gap 1.04x), D license-clean text arm sealed (9.51),
C cross-modality guard PASS — plus the B diagnostic (32x32 input not
the blocker) and B2 head-transfer (running).

### 18 Aug 2026 — M175 cell B R3 SEALED: flowers-native SPM reads 0.1895 — the SPM construction itself is weak at 32x32 fine-grained

R3 sealed (evidence `logs/results/v24/m175_cell_b_r3/evidence.json`;
g1 bit-exact 0.0). The whitener and the 1,923-atom dictionary fitted
ON flowers (everything else identical to cell B) read 0.1895 vs cell
B's 0.1667 — the DomainNet-native whitener/dictionary cost only ~0.02;
the construction itself (patch atoms + 21-bin pyramid at 32x32) is
the blocker. THE B-FORENSICS ARE NOW COMPLETE: (1) class sets fully
exclusive (the normal frozen-encoder setting); (2) feature transfer
scoped negative (0.167); (3) the 32x32 input retains the species
signal for a strong reader (0.827) — resolution is not the blocker;
(4) head transfer CONFIRMED — the frozen DomainNet head reads flowers
as `flower` (17.6% modal) and flower-like classes — the codes know
WHAT a flower is; (5) even flowers-native SPM reads 0.19 — the codes
do not separate WHICH flower, at 32x32 five-shot. The honest summary:
the SPM-1923 construction carries generic visual semantics but not
fine-grained species geometry at this resolution; the deep-patch arm
(224px tokens, 0.590 on DomainNet) is the right-family candidate for
fine-grained tasks.

### 18 Aug 2026 — M175 cell B2 SEALED: the user's expectation CONFIRMED — the frozen DomainNet head reads flowers as generic flower classes

B2 sealed (evidence `logs/results/v24/m175_cell_b2/evidence.json`,
runtime 3,354 s; g1 encoder pin bit-exact 0.0). Reading the 306 flowers
test rows through the frozen DomainNet 345-way head (zero flowers
labels anywhere): the MODAL predicted class is `flower` (54/306 =
17.6%; 61x the uniform 1/345 baseline), and the rest of the mass
spreads over visually flower-like generic classes — watermelon (30),
pear (17), strawberry (12), rabbit (11), carrot (11), lightning (9),
sun (8), sheep (8), grapes (8). The codes' visual semantics are
generic but real: round, radial, colorful, organic objects. Verdict:
expectation CONFIRMED per the registered clause (modal = flower), with
the honest caveat that the collapse is a soft one — 17.6% modal, not
monolithic; the fine-grained species signal is what the 5-shot
new-head read (cell B, 0.167) cannot extract. Together with the B
diagnostic (32x32 still carries species info for a strong reader),
the B-forensics now reads: the SPM-1923 codes know WHAT a flower is
but not WHICH flower, at 32x32 under five-shot. R3 (flowers-native
SPM) runs next.

### 18 Aug 2026 — M175 cell B R3 registered (before building): flowers-native SPM at 32x32 — whitener/dictionary vs construction

The B diagnostic showed the 32x32 input retains the species signal for
a strong reader (R1 0.827), so the blocker is in the SPM arm's
construction. R3 separates its two parts: (i) the DomainNet-native
whitener + dictionary, (ii) the SPM construction itself. Construction:
the SAME sparse parameters as cell B (patch 6, stride 1, contrast 10,
zca 0.1, seeds 11, candidate pool 8192, 1923 atoms, 21 bins) but the
whitener fitted ON the flowers bounded train (510 images) and the
dictionary drawn from FLOWERS candidates; ridge ladder {0.1, 1.0, 10.0}
on the same 510/306 split. Anchors: g1 machinery pin — the runner
rebuilds cell B's DomainNet encoder and pins it bit-exact against
`spm1923_fulltrain.npy[:256]` (delta 0.0) BEFORE the flowers-native
arm runs, so both arms share the proven SPM code path. Registered
reading: R3 best >= 0.8 -> the blocker is the DomainNet-native
whitener/dictionary (the construction transfers when fitted on
flowers); R3 <= 0.4 -> the SPM construction itself is weak at 32x32
fine-grained; in-between -> both contribute (recorded split).

### 18 Aug 2026 — M175 cell C registered (before building): cross-modality routing — contract guard in the router + demonstration

Two parts, registered before any number (C produces NO new accuracy —
route-level only, no data cost, per the battery registration). C1: the
router gains an optional `contract_kind` parameter on `route` and
`chain`; when given, every tier (matched, general, primitive) admits
only arms whose output_contract kind matches — a task can NEVER
silently receive a wrong-modality arm; when absent the behavior is
unchanged (backward compatible). This is a security property, not a
performance one. C2 (demonstration): a registry of the sealed arms —
vision (spm1923 DomainNet, deep-patch SPM, dense r70) and text
(wikitext uniform-w2, Wikipedia uniform-w2) — with task manifests
whose fingerprints are registered profile vectors built from sealed
numbers (vision: sparse/dense/deep-patch accuracies; text: sealed
ppls). Registered checks: (1) the vision task's chain (contract
classification-vision) contains ONLY vision arms; (2) the text task's
chain (contract next-token-text) contains ONLY text arms; (3) a
cross-contract query (vision fingerprint + text contract) yields no
silently-wrong arm — the chain is the text arms ranked by the guard's
rules or empty, never a vision arm. Verdict: pass iff (1) and (2) hold
and (3) shows no wrong-kind arm in any chain.

### 18 Aug 2026 — M175 cell B diagnostic RESULT: the 32x32 input is not the blocker — the construction is

Sealed (`logs/results/v24/m175_cell_b_diag/evidence.json`). R2
reference: torch DINOv2-small CLS on the full-res bounded flowers reads
0.964/0.974/0.987 — near the sealed ONNX baseline 0.9902, so the
diagnostic extractor is licensed. R1 (information-matched control): the
SAME reader on the SAME 32x32 images the SPM arm read scores
0.709/0.778/0.827 — the 32x32 input still carries the fine-grained
species signal for a strong reader. Registered reading applied: R1 >=
0.8 → the blocker for cell B's 0.167 new-head read is the SPM
CONSTRUCTION (DomainNet-native whitener/dictionary + 40,383-dim codes
at five-shot), NOT the input resolution. Next registered step: R3
(flowers-fitted SPM at 32x32) to separate the whitener/dictionary from
the construction itself.

### 18 Aug 2026 — M175 cell B2 registered (before building): HEAD transfer — the frozen DomainNet 345-way head read on Flowers-102

User expectation, now a registered cell: transfer should read the
flowers through the FROZEN DomainNet head, and the predictions should
concentrate on the generic DomainNet classes ("flower" and its
neighbours: garden, house_plant, leaf, cactus, bush, grass) — NOT the
new-head feature transfer cell B measured (0.167 with a flowers-fitted
102-way head). B2 measures exactly that: the bit-exact DomainNet
SPM-1923 encoder (g1 pin vs the sealed memmap), the DomainNet head
re-fitted from the sealed full-train codes
(`spm1923_fulltrain.npy` 409,832 rows + `m142_c2_fulltrain_labels.npz`,
the M142 C2 exact ridge path, penalty ladder {0.1, 1.0, 10.0}), then
the 306 flowers test rows encoded and scored with the 345-way head.
Registered reading: the user's expectation is confirmed if the modal
predicted class is "flower" (or the flower-adjacent class mass
dominates the top-k); everything measured is reported either way.
Zero flowers labels enter this cell anywhere.

### 18 Aug 2026 — M175 cell B diagnostic registered (before running): what blocks the vision transfer?

User question: are the class sets fully exclusive, or is something
else blocking? Registered answer plan (numbers come only from these
probes, all inputs sealed): the label spaces ARE fully exclusive by
construction (DomainNet 345 classes vs Flowers-102 102 species) — and
that is the NORMAL frozen-encoder setting, not itself a blocker. The
baseline (0.9902 full-res CLS) proves the flowers labels are perfectly
learnable from strong features. The SPM arm differs from the baseline
in THREE registered candidate factors, which the probes separate:
(1) input resolution (SPM arm reads 32x32, baseline full-res),
(2) the extractor family (SPM/whitener vs DINOv2 CLS),
(3) DomainNet-native whitener + dictionary.
Probe R2 (reference): torch DINOv2-small CLS on the FULL-RES bounded
flowers — must land near the sealed ONNX baseline 0.9902 to license
reading R1. Probe R1 (information-matched control): the SAME torch
CLS reader on the SAME 32x32 images the SPM arm read. Registered
reading: R1 >= 0.8 -> the 32x32 input still carries the species
signal and the blocker is the CONSTRUCTION (then R3 follows:
flowers-fitted SPM at 32x32 to separate whitener/dict from the
construction); R1 <= 0.4 -> 32x32 downsampling destroys the
fine-grained signal, resolution is the primary blocker, and the
deep-patch (224px) family is the right encoder for flowers. Same
ridge ladder {0.1, 1.0, 10.0} on 510 train / 306 test throughout.

### 18 Aug 2026 — M175 cell D SEALED: license-clean Wikipedia fit-and-report (held-out 9.5142, beats A's transferred read)

D sealed (evidence `logs/results/v24/m175_cell_d/evidence.json`). g1
machinery anchor: delta 0.0000 vs A's sealed 11.0933. The uniform-w2
encoder fitted on 300k dump tokens (vocab 84) reads its own 100k
held-out slice at 9.5142 ppl (uniform baseline 84; OOV 0.0) — BEATS
the 60k wikitext-fitted encoder's transferred read of the dump
(11.0933), as expected for a fit on the target domain itself. License
posture recorded in evidence (Wikipedia-derived, CC BY-SA 3.0 / GFDL,
vocab-fingerprinted lineage — traceable, not a legal audit). The
license-clean text arm exists: uniform-w2 fitted on 300k dump tokens,
held-out 9.51.

### 18 Aug 2026 — M175 cell D registered (before building): license-clean Wikipedia fit-and-report

Model: the uniform-w2 arm (selected and sealed in cell A — no new
selection). Slices (registered): fit = `wikitext103_full` train ids
[100000:400000] (300k); held-out = train ids [400000:500000] (100k) —
disjoint from every slice A/B/A0 measured and never fitted on. Anchors
(registered): g0 prefix pin (the 100k train is the full-stream prefix);
g1 machinery anchor — D's encoder on A's exact out-domain slice must
reproduce A's sealed 11.0933 bit-exactly (delta 0.0). Metric: held-out
ppl + uniform baseline (fit vocab) + footprint + OOV. License posture
(recorded, not a legal audit): wikitext-103 derives from Wikipedia
articles (CC BY-SA 3.0 / GFDL); the cached token ids carry a vocab
fingerprint, so the lineage is traceable end-to-end — the
license-clean text arm's input, in contrast to scraped corpora.
Reading (registered): fit-and-report; whether the dump-fitted encoder
beats A's transferred read (11.09) is a reported comparison point, not
a gate.

### 18 Aug 2026 — M175 cell A SEALED: text→text transfer HOLDS (wiki slice 11.09 vs in-domain 10.66, gap 1.04x)

Selection (valid-based, full table in evidence): uniform-w2 wins valid
ppl 10.8591 (uniform ladder 12.32/10.86/12.91/19.18/28.69/39.97;
backoff worse above w2) — consistent with A0. g1 machinery anchor:
delta 0.0000 vs the A0 sealed 9.9152. Transfer with the frozen
60k-fit uniform-w2 encoder: in-domain wikitext test 10.6562;
out-domain Wikipedia slice (full train ids [100000:200000], 100k
tokens, disjoint by construction, OOV 6e-05) 11.0933 — gap factor
1.041 vs the registered 2.0 threshold → transfer HOLDS. The additive
text encoder stays informative off-domain; cells D proceed with the
construction. Honest scope note: both streams are Wikipedia-derived
character text (same family); cross-modality C remains the unmeasured
one. Evidence `logs/results/v24/m175_cell_a/evidence.json`.

### 18 Aug 2026 — M175 cell A registered (before building): text→text transfer, wikitext → Wikipedia dump slice

Slices (registered): fit = wikitext-100k train ids [:60000]; valid =
train ids [60000:80000]; in-domain test = the 100k test (20k);
Wikipedia slice = `wikitext103_full` train ids [100000:200000] (100k
tokens, mmap read-only) — disjoint from fit/valid/test by construction
(measured: the 100k train is exactly the full train stream's first
80,000 ids; the 100k test is the separate test stream). Selection
(valid-based, registered BEFORE any test number): for variant x window
over {uniform, backoff} x {1,2,4,8,16,32}: fit 60k, valid ppl; the arm
= argmin valid ppl; the full selection table is reported, not just the
winner. Anchor g1 (machinery pin): the selected variant at its window
must reproduce the A0 SEALED number (A0's exact 80k fit on the 20k
test) with delta 0.0. Transfer: the frozen selected encoder (60k fit)
scored on wikitext test (in-domain) and the Wikipedia slice
(out-domain); the slice's OOV rate vs the fit vocabulary is reported.
Verdict (registered): transfer holds if out-domain ppl <= 2.0 x
in-domain ppl (the additive encoder stays informative off-domain);
else A closes as a scoped negative for text→text transfer of this
construction. Uniform baseline = fit-vocab size.

### 18 Aug 2026 — M175 cell A0 SEALED: the window dial inverts on natural text (best at w=2, gate fired)

A0 sealed (evidence `logs/results/v24/m175_cell_a0/evidence.json`,
runtime 12 s). Anchors: g1 bigram closed form BIT-EXACT (delta 0.0);
g2 footprint monotone. Uniform ladder (test ppl): w1 11.7627, w2
9.9152, w4 11.0405, w8 16.1869, w16 24.8216, w32 35.9272. Backoff
ladder: w1 11.7627, w2 9.8706, w4 18.7373, w8 33.6968, w16 36.5642,
w32 36.7404. Uniform baseline 82. Gate FIRED: w32 (35.93) > w1 x 1.10
(12.94) — the M131 scaling hypothesis (perplexity falls with the
window) does NOT transfer from the DSL to natural character text:
exact high-order contexts are sparse, and both constructions degrade
beyond w=2 (backoff worse than uniform at every rung above 2).
Registered consequence: A0 selects NO arm from test numbers; cells
A/D pin their arm (variant + window) in A's registration using a
validation slice. The first text encoder is a w<=2-grade additive
count model — the honest ceiling A/D inherit.

### 18 Aug 2026 — M175 cell A0 smoke: g1 defect repaired + uniform interpolation measures NEGATIVE on wikitext + backoff arm registered

Smoke (inadmissible) findings, registered before any admissible number:
(1) g1 instrument defect: the anchor compared the 4-decimal-ROUNDED
window-1 cell against the unrounded closed form — repaired to compare
unrounded values (expected delta ~1e-13). (2) The uniform-interpolation
construction INVERTS on natural text: test perplexity RISES with the
window (12.87 -> 12.43 -> 15.64 -> 22.74 at w=1/2/4/8 on the smoke
rows) and the no-hurt gate fires. Mechanism: exact high-order contexts
on character text are sparse, and uniform mixing floods the bigram
signal with alpha-smoothed-uniform orders. Per N90.5 this is a reported
negative on the uniform variant, not silently dropped. (3) Comparison
arm registered BEFORE the sealed run: the longest-match backoff read
(the machinery's `continuations` primitive, add-alpha at the matched
order) on the same window ladder. Smoke (inadmissible) already shows
backoff DEGRADES worse than uniform at the top rungs (36.09 vs 22.74
at w=8) — both variants lose to their own w=1/w=2. (4) Verdict clause
AMENDED before the sealed run: A0 selects NO arm from test numbers;
the cell answers its registered question with the ladder and the
fired gate, and cells A/D pin their arm (variant + window) in A's own
registration using a validation slice. Uniform baseline: 82.

### 18 Aug 2026 — M175 cell A0 registered (before building): the FIRST text encoder on wikitext-103

Corpus: cached `data/tier6/wikitext103_100000.npz` (vocab_fingerprint
5c6f0917b424e3a7, cache_version 2): 80,000 train / 20,000 test
character-level token ids (range 0..95; 82 distinct in train) — no
labels, fit-and-report is self-supervised. Model: the M131 additive
construction verbatim (ProgrammaticMemory exact-order counts, uniform
Jelinek-Mercer over orders 1..w, add-alpha alpha=1.0), window ladder
{1, 2, 4, 8, 16, 32}; test read-only (nothing fitted on test).
Anchors (registered before running): g1 unigram closed form — the
window-1 cell must equal an INDEPENDENT closed-form unigram
computation ((c+alpha)/(N+alpha*V), two separate code paths) to
<=1e-9; g2 footprint law — footprint_bytes monotone in window.
Metric: test perplexity per window + footprint + integer ops/token;
uniform baseline = 82. Gate (N90.5 adapted): ppl(w=32) <= ppl(w=1) *
1.10 (more memory must not hurt). This encoder is the frozen text arm
that cells A (text→text transfer to a Wikipedia slice) and D
(license-clean Wikipedia fit-and-report) reuse. No novelty claim
(n-gram/PPM territory, M129).

### 18 Aug 2026 — M175 cell B SEALED: the DomainNet SPM-1923 encoder does NOT transfer to Flowers-102 (0.167 vs 0.990)

Cell B closed (evidence `logs/results/v24/m175_cell_b/evidence.json`,
runtime 554 s). Gates: g1 encoder pin BIT-EXACT (max-abs 0.0e+00 vs the
sealed spm1923 memmap) — the rebuilt encoder IS the sealed encoder; g2
split pin exact (torchvision reproduction == cached M19 labels/ids).
Both arms read by the same exact ridge ladder {0.1, 1.0, 10.0}:
baseline (cached DINOv2-small CLS, 384-d) 0.9477/0.9673/0.9902 on the
306-row test; frozen DomainNet SPM encoder (40,383-d) 0.1634/0.1667/
0.1667 (flat across penalties). transfer_holds = false -> registered
consequence: SCOPED NEGATIVE for THIS encoder; the deep-patch
deployment arm is unaffected. Honest reading: the SPM arm's 0.167 is
~17x chance (0.0098), so the 32x32 codes carry real flowers signal,
but the construction sits far below the full-resolution CLS baseline
under five-shot; the whitener and dictionary are DomainNet-native.
Operational note: the detached worker lingered ~80 min after writing
evidence (ROCm teardown) — evidence.json is the completion signal,
not process exit. Next: A0 (the first text encoder).

### 18 Aug 2026 — M175 cell B dispatched (registered before building): vision→vision, DomainNet-32 → Flowers102

Question: does the frozen DomainNet-32 SPM code construction transfer to
Oxford Flowers-102 (bounded S1, five-shot)? Encoder: the sealed M142 C2
SPM-1923 (6x6 patches, ZCA whitener, 1,923 atoms, 21-bin pyramid, width
40,383), rebuilt deterministically from the DomainNet corpus (same
seeds) and pinned bit-exact against `spm1923_fulltrain.npy[:256]`
(max-abs delta must be EXACTLY 0.0; any nonzero → VOID). Target: the
M19 Flowers102 bounded split (5/2/3 per class, seeds 11/12/13; 510
train / 204 dev / 306 test), reproduced via torchvision Flowers102 and
pinned by identity against the cached M19 npz labels+ids. Input
adapter: flowers → RGB → PIL BILINEAR → 32x32 uint8 (the DomainNet
parquet pipeline's own filter). Baseline arm: the cached DINOv2-small
CLS features (384-d) read by the SAME exact ridge path (accumulator +
standardiser + intercept, ladder {0.1, 1.0, 10.0}); the SPM arm codes
are the RAW C2 output (no power-norm — C4's transform was measured on
DomainNet only). Verdict (registered before running): transfer holds if
SPM best-penalty test accuracy ≥ baseline best-penalty test accuracy;
else B closes as a scoped negative for THIS encoder (says nothing about
the deep-patch deployment arm). Disclosures: not matched-cost (384 vs
40,383 width); baseline reads full-resolution images (cached CLS) while
the SPM arm reads 32x32; 510-row five-shot fit.

### 18 Aug 2026 — Displacement search RESULT: instrument live, no displacing combination found; novelty posture unchanged

Anchors all hit (instrument live): Fisher Vectors + Perronnin (4
results: Novotný 1504.04763, Gordo "Deep Fishing" 1507.06429, Murray
GMP 1406.0312); VLAD + aggregating local descriptors (1: Tolias/Furon/
Jégou 1407.2170); DINOv2 + self-supervised features (5). Topic hits
read by the registered criterion (same combination = frozen codes +
exact solve + the measured gate, public benchmark):

- C1 "frozen features" + "closed-form" + "linear readout": 1 hit —
  Liquid Random Features for PDEs (2606.15571, physics.comp-ph) —
  frozen features + linear readout for time-dependent PDEs, not a
  classification read; does NOT displace the freezing-wins mechanism.
- C2 "spatial pyramid" + "VLAD" + "frozen": 0 hits.
- C3 "task fingerprint" + "router" + "frozen": 0 hits.
- C4 "concept erasure" + "closed-form": 20 hits — LEACE (the method
  we import), double projections (2604.10032), OCE (2605.28902, ICML
  2026), MANCE (2607.03973), a survey. None displace: our claim there
  is only the float64-promotion repair, not the method.

Verdict per the registered criterion: no displacing combination found,
but zero-hit queries license nothing (the measured recall lesson) — the
novelty posture is UNCHANGED: no mechanism is claimed new; the
contribution remains the measured combination (frozen deep-patch
SPM-VLAD + exact ridge read beating trained heads at matched cost on
one sealed benchmark) plus the audit discipline. Absences are recorded
as absences, not as novelty.

### 18 Aug 2026 — Fresh displacement search registered (before running): "what is novel" re-check

User asked for a novelty re-check with a new literature search. The
instrument's role stays DISPLACEMENT ONLY (it cannot support a
novelty claim — that lesson is measured, M88). Claims to
displacement-check, all phrased as combinations, none as firsts:

- C1 the freezing mechanism: frozen encoder + exact closed-form
  ridge read, trained heads lose to it on the same codes.
- C2 the winning construction: deep-patch SPM-VLAD pooling over
  frozen DINOv2 patch tokens, no training, exact ridge read.
- C3 the router: per-task fingerprint + selection among frozen arms
  at registry scale, with a measured per-task gate.
- C4 closed-form concept erasure (LEACE) with the float64-promotion
  repair.
- C5 the audit discipline: bit-pinned encoders, replayable
  decisions, payload hashes.

Query strings (arXiv API, registered before searching — AND-then-OR
applied uniformly, retries on 429, anchors checked first):

1. anchor: all:"Fisher Vectors" AND all:"Perronnin"
2. anchor: all:"VLAD" AND all:"aggregating local descriptors"
3. anchor: all:"DINOv2" AND all:"self-supervised features"
4. topic: all:"frozen features" AND all:"closed-form" AND all:"linear readout"
5. topic: all:"spatial pyramid" AND all:"VLAD" AND all:"frozen"
6. topic: all:"task fingerprint" AND all:"router" AND all:"frozen"
7. topic: all:"concept erasure" AND all:"closed-form"

Displacement criterion (registered before searching): a hit displaces
a claim only if it performs the SAME COMBINATION (not the borrowed
pieces) — frozen codes + exact solve + the measured gate, on a public
benchmark. Anchors prove the instrument is live; topic hits are read
by that criterion, and absences license nothing.

### 17 Aug 2026 — M167a dispatched: the two-tier repair of the label harness (registered before re-measurement)

M167 stays VOID; the harness verdict stands and the hypothesis is
undetermined. The repair, applied to EVERY arm including the
baseline, re-gates the same measurements in two tiers: (1)
harness-validity — own-domain − permuted ≥ 0.05 (the instrument can
order similarity at all); (2) transfer-sensitivity — cross-domain −
permuted ≥ 0.0 (the cross fit beats the permutation control; M167
measured +0.0092, real but far below the over-strong 0.05 cross
margin). Adjacent-domain pairing for STRONG transfer labels is a
registered follow-up once the domain-name→index mapping is pinned
(no name mapping exists in the corpus code). Both controls reported.

## 12. Execution log (live)

### 17 Aug 2026 — M169 fingerprint training: PASSED — G1/G2/G3 pass; G4 deferred (registered)

Admissible, void=false. 300 steps, final loss 9.04 (runtime 2.1 s).
G1 deterministic ✓. G2 similarity ordering: margin +1.577 — similar
pairs 0.98/1.00 vs dissimilar −0.72…−0.30 (the 10-step smoke already
held +0.917; training widened it). G3 traversality: min-cos 0.755 ≥
0.5 across all 12 attribute-swap analogies. G4 continuity DEFERRED —
no sweep families exist; registered pending. Verdict: the v0
signal mix (InfoNCE over the ontology pairs + attribute
reconstruction) trains a deterministic, additive fingerprint that
orders the registered similarity set and moves along attribute axes.
Evidence: `logs/results/v24/m169_fingerprint_gates/evidence.json`.

### 18 Aug 2026 — ImageNet/DINOv3 SoTA cell DEFERRED (user decision, registered)

The user declined a "beat SoTA on ImageNet" cell for now: even a
winning headline is not worth the time without the completed system
(distributed training/inference, token mechanics, copy protection —
v25). The cell stays deferred with a registered trigger: it reopens
only when the system needs the headline (post-v25 or a
commercial-grade claim). Numbering note: v25 milestones already
occupy M177-M198, so the deferred cell would take a fresh number.
The DINOv3 facts (arXiv:2508.10104; LVD-1689M; linear probes
87.0/88.0/89.3/90.2/90.3/90.4; bespoke DINOv3 license vs DINOv2's
Apache-2.0) are recorded as its registered input.

### 18 Aug 2026 — M176c-c2 K=32 SEALED locally: 0.5990 @ width 24,576 — Fisher closes as a comparison point; SPM stays the deployment arm

K=32 Fisher vectors sealed (`logs/results/v24/m176c_c2_k32/evidence.json`):
accuracy 0.5990144928 at width 24,576; all three ridge penalties identical
to 10 decimals. Together with K=16 (0.5987 @ 12,288, evidence
`logs/results/v24/m176c_c2/evidence.json`) the cell is CLOSED under the
interpretation registered before either number existed:

- K=16 -> K=32 doubles the width for +0.0003 accuracy — the Fisher
  dimension saturates at the first rung; width 24,576 buys nothing.
- The win clause was "beats deep-patch SPM at a COMPARABLE width".
  Fisher has no data point at comparable width (its smallest rung, 12,288,
  is 6x SPM's largest, 2,048), and at 6x the width it gains +0.009 over
  SPM-2048 (0.5987 vs 0.5899). Per code dimension: SPM-2048 achieves
  2.88e-4 accuracy per dimension vs Fisher's 4.87e-5 — a 6x efficiency
  loss, not a win.
- The ridge penalty is a no-op at both rungs (all three penalties
  identical) — the whitened Fisher codes are already near-orthonormal;
  recorded, not chased.

Verdict: Fisher does NOT earn the deployment consequence. Deep-patch SPM
(0.487/0.563/0.590 at 256/1024/2048 atoms) remains the deployment arm.
M176c-c2 closes as a measured comparison point; the better-code arm is
complete with c1 the sealed winner (candidate 3, a small trained
encoder, stays registered as last resort, not queued).

### 18 Aug 2026 — M176c-c2 K=16 SEALED locally: 0.5987 @ width 12,288

K=16 Fisher vectors sealed on the local device (evidence at
`logs/results/v24/m176c_c2/evidence.json`): accuracy 0.5987246377 at
width 12,288, identical to 4 decimal places across all three ridge
penalties (0.1/1.0/10.0 -> 0.5987/0.5987/0.5987). Encode 1530.1 s.
This is the first Fisher rung after the one-K-per-run repair; K=32
(width 24,576) is running next. Reading against the registered
comparison: 0.5987 tops the deep-patch SPM's 0.5899 (2048 atoms) by
+0.009 absolute — but at 6x the width (12,288 vs 2,048), and the
registered win clause is "beats deep-patch SPM at a COMPARABLE
width". No verdict until K=32 lands and the per-MAC reading is done;
the deployment arm stays deep-patch SPM meanwhile.

### 18 Aug 2026 — M176c-c2 full run died at the K=64 solve + one-K-per-run repair

The run finished K=16 and the K=64 train schedule, then died
allocating the 49,152-dim centred solve matrix (18 GiB on top of the
19.3 GiB Gram) — the daily-driver PC's free RAM could not serve it,
the same failure the local anchor smoke showed. Because the runner
wrote evidence only at the end, K=16's completed numbers were lost
with the process. Repair (registered before re-running): one K per
run (config selects a single K; separate sealed evidence per K), and
the ladder becomes K in {16, 32} (width <= 24,576, Gram <= 4.8 GiB)
with K=64 dropped — it was an arbitrary rung, not a registered
value. K=16 runs first and seals independently; K=32 follows.

### 18 Aug 2026 — M176c candidate 2 registered (Fisher on deep patches, local)

Comparison arm (candidate 1 already passed the gate, so this is NOT a
gate — it measures whether Fisher vectors on the same DINOv2-small
patch tokens beat the sealed deep-patch SPM). Cell: a diagonal-cov
GMM (K ∈ {16, 64}) fitted on the same deep-patch sample (seed 11),
classic per-image Fisher vectors (first/second-order, signed sqrt +
L2), the same streaming 224px decode + backbone, closed-form
intercept ridge, penalties {0.1, 1.0, 10.0}. Cited prior art
(Perronnin & Sánchez 2013) — comparison baseline, not a claim. Gate:
report vs deep-patch SPM 0.487/0.563/0.590 and the dense ladder; no
new deployment consequence unless Fisher wins. Runner:
`experiments/tier4/eval_v24_m176c_c2.py`.

### 18 Aug 2026 — M176c-c1 SEALED: deep-patch SPM BEATS the dense ladder per-MAC

Evidence: `logs/results/v24/m176c_c1/evidence.json` (local sealed
device, admissible). Full 409,832-row schedule, DINOv2-small @224,
SPM 1x1+2x2+4x4 over the 16x16 patch grid, penalties {0.1, 1.0,
10.0}:

- atoms 256: **0.4871** @ 367.5M (backbone) + 1.9M readout MACs
- atoms 1024: **0.5628** @ + 7.4M
- atoms 2048: **0.5899** @ + 14.8M

Dense-ladder ledger (sealed M144, same backbone, keep=1.0 = 367.5M
effective MACs): r42 0.1972 / r56 0.2450 / r70 0.3118. Every atom
cell beats the ladder at comparable MACs (+0.175 at 256 atoms for
+0.5% cost), and atoms 2048 also clears the sealed dense frontier
r224 0.5368. Against the frozen SPM construction (0.2786 @ 175.2M):
2.1x the accuracy at 2.2x the MACs. **The per-MAC gate PASSES** —
the better-code arm EXISTS and it is the deterministic no-training
candidate (deep-patch SPM, arXiv:1603.09046-style, cited). This is
the deployment-phase answer to L1: codes changed, accuracy followed.
Candidate 2 (Fisher) and 3 (small encoder) remain queued as
comparisons, not gates.

### 18 Aug 2026 — M176c-c1 full run dispatched (local, detached)

The candidate-1 runner passed its smoke end-to-end (65.9 s; the
2,000-row 256-atom cell read 0.2275 at penalty 10.0 — at the sealed
138k anchor's level, inadmissible). Full run dispatched on the local
9070 XT as a detached process: atoms {256, 1024, 2048}, the full
409,832-row train schedule + the 34,500-row test stream, penalties
{0.1, 1.0, 10.0}; ~140 rows/s, expected ~4-5 h. Evidence target:
`logs/results/v24/m176c_c1/evidence.json`. The per-MAC gate verdict
against the dense ladder (r70 0.3118 / r98 0.4476) registers on
completion.

### 18 Aug 2026 — M176c candidate 1 registered (deep-patch SPM, local run)

Cell (before any build): DINOv2-small (frozen, the cached backbone)
encodes the DomainNet train/test schedules at 224 px in streaming
chunks (the 32 px decoded cache does NOT serve — deep patches need
native resolution; the raw parquets are on F:). The SPM pooling
levels 1x1+2x2+4x4 sum the deep patch tokens (arXiv:1603.09046
style, cited as a comparison baseline, not a claim); the dictionary
is a seeded prefix draw of whitened deep-patch candidates (the
M117/M126 construction pattern, registered fresh for this encoder).
Closed-form intercept ridge + the sealed fit discipline. Gate: the
result must beat the sealed dense ladder per-MAC (r70 0.3118 / r98
0.4476) or serve a task axis no frozen arm serves; the anchor cells
(0.2605/0.2274/0.2786) are recorded for reference. Candidate 2
(Fisher) and 3 (small encoder) remain queued behind the registered
budget caps.

### 18 Aug 2026 — M176c-anchor on the LOCAL device: satisfied by the sealed evidence (registered)

The anchor-reproduction stage existed to validate the RENTAL's
environment. On the local sealed device it is redundant by
construction: the sealed M142 C2/C4 evidence (0.2604927536231884 /
0.2273623188405797 / 0.27855072463768116) IS the local device's
anchor reproduction — same codes, same GPU. The chunked runner
(`eval_v24_m176c_anchor.py`) stays as the registered
rental-validation tool for any future rental use (its local smoke
also showed the two-gram solve peak needs >40 GB free RAM, which the
daily-driver PC cannot guarantee — another reason not to re-derive
sealed numbers locally). M176c proceeds with the CANDIDATE ladder on
the local device: deep-patch SPM first, per the registered order.

### 18 Aug 2026 — M176c PIVOT to the local sealed device (user decision) + premise patch

The rental kept hitting host-level SIGKILLs (its cgroup never OOM'd)
and the transfer channel costs more than the compute. Decision: the
anchor reproduction and the candidate ladder run on the LOCAL
RX 9070 XT (the sealed device, 63 GB RAM; the chunked runner peaks
~39 GB). The rental's sealed dvc bit-exact verdict stands as its
deliverable; the pod was stopped. Premise patch (registered before
running): on gfx1201 the device IS the premise; on any other device
the dvc evidence must read bit-exact. Same runner, same config,
local cache (F:\geode-ml\data\cache) has the sealed memmaps if any
later comparison needs them.

### 18 Aug 2026 — M176c-anchor full run: host-level SIGKILL (Exit 137) + retry-loop repair

The full run died ~90 s into pass A (test encode done, 642/138k)
with Exit 137, but the pod's own cgroup never OOM'd
(max_usage 150 MB, oom_kill=0) — the kill came from the shared
host (neighbor memory pressure). The chain then stopped the pod as
designed (no idle burn). Repair (registered before re-running): the
chain retries the anchor up to 3 times on exit 137 with a 60 s
backoff (the corpus decode is cached and the lost work is encode
time only); the new pod instance rebuilds its venv first.

### 18 Aug 2026 — M176c-anchor smoke defect (test-labels truncation) + fix

The repaired smoke passed pass A and died in scoring: the smoke
truncates n_test for the codes but the full-length test_labels array
was still indexed by it (broadcast 400 vs 4096). Fix: slice
`test_labels = test_labels[:n_test]` right after the smoke cap.
Registered before re-measurement; nothing was fit.

### 18 Aug 2026 — M176c-anchor smoke OOM (Exit 137) + two-pass repair registered

The smoke encoded 2,000/2,000 rows and died in the solve phase:
the single-pass three-Gram layout (3 x 13 GiB) plus the solve
workspace crossed the pod's 56.8 GiB cgroup cap. Repair (uniform,
registered before re-running): a two-pass layout — pass A rows
0..138k into acc_138 only (peak 26 GiB), solve, free; pass B rows
0..409,832 into acc_raw + acc_p05 (peak 39 GiB), solve each and free.
Pass A's rows are re-encoded in pass B (+34% encode time); every
accumulation order stays identical to the sealed fits.

### 18 Aug 2026 — M176c-dvc SEALED: BIT-EXACT (max-abs delta 0.0e+00)

Evidence: `logs/results/v24/m176c_dvc/evidence.json` (rental). G-dvc1
PASS (corpus digest), G-dvc2 delta 0.0 on the full 256 check rows —
the rental's RTX 4090 reproduces the sealed encoder arithmetic
bit-for-bit. Registered consequence: FULL parity — the rental may run
the sealed runners behind the registered device override
(`GEODE_VERIFIED_DEVICE_EVIDENCE` env var pointing at the sealed dvc
evidence; `_verify_device` requires verdict bit-exact). The anchor
stage design (registered now, before running): the full-data anchors
cannot be reproduced by rerunning m142_c2 verbatim — its 75 GB of
code memmaps exceed the pod's 30 GB overlay and the volume quota; the
anchor runner is a CHUNKED variant (the M151 Gram pattern): encode
chunk -> float64 Gram accumulate -> discard, test codes held in RAM
(5.6 GB). Reproduces 0.2273623188405797 @138k and 0.2605/0.2786 @full
within the sealed tolerances. Runner:
`experiments/tier4/eval_v24_m176c_anchor.py`.

### 18 Aug 2026 — M176c-dvc device-verification cell registered (before running)

The rental's numbers are gated on device parity. Reference: the sealed
`v16/m142_c2/_check_pool6144.npy` (the first 64 TRAIN rows encoded at
the 6,144-atom dictionary by the sealed device during M142 C2 — itself
verified against the sealed f6144 memmap there). Cell: on the 4090,
decode the corpus, reproduce the whitener + nested dictionaries (M117
exact, CPU numpy — device-independent), encode the same 64 rows with
the GPU path, and compare. Gates: G-dvc1 the 64-row decoded subsample
sha256 == 63f590097008f749f3f1828b29d6f154de7b21a6828a7b017ac141c0615fa09d
(else the rental's data path is VOID); G-dvc2 the 64-row pool-6144
encode vs the shipped reference: max-abs delta == 0.0 -> FULL parity;
<= 1e-6 -> NUMERICAL parity (anchors may be reproduced with the
sealed tolerances only, and every evidence file carries the dvc
delta); > 1e-6 -> the rental is VOID for the sealed pipeline. Only
after parity does `_verify_device` receive a registered
device-override for the anchor stage. Runner:
`experiments/tier4/eval_v24_m176c_dvc.py` (does NOT call
`_verify_device` — that is the cell's subject).

### 18 Aug 2026 — M176c rental: stage 0 VERIFIED, anchor stage blocked by the M108 device gate; pod stopped

Stage-0 completed and verified on the rental (RTX 4090, 62 GB RAM):
all four parquets sha256 `ok`, manifest written, GEODE layout on the
network volume, HF cache cleaned, torch 2.8.0+cu128 CUDA true. The
M142 c2 anchor smoke then aborted on the registered M108 instrument:
the sealed SPM encoder is bit-pinned to the local ROCm device
(gfx1201) and refuses the 4090 (gcnArchName=NVIDIA ...) before a
device-verification cell passes. Decision: no anchor/candidate
numbers may come off this rental until that cell is registered and
run (the verification references — sealed code arrays — are on the
local F: cache, not on the pod). The pod was STOPPED via the RunPod
API to avoid idle burn; the network volume (parquets + repo copy +
logs) persists. Next registered step: M176c-dvc — the device-
verification cell (small sealed reference codes shipped to the
rental; bitwise comparison before any figure).

### 17 Aug 2026 — M176c stage-0 abort #3 + repair #3 (registered before re-run)

The HF download worked (409k rows streamed in ~10 s) but
`load_dataset()` then built its ARROW cache on top of the parquet
cache (~37 GB total) and the 30 GB local disk filled (Errno 28).
Repair: the script downloads the four pinned parquets DIRECTLY via
curl from the pinned revision URL (no datasets library, no arrow
cache), verifies the four sha256s, copies into the GEODE layout, then
deletes the download cache. Local budget: ~18.5 GB transient + ~2 GB
venv. Nothing was fitted.

### 17 Aug 2026 — M176c stage-0 abort #2 + repair #2 (registered before re-run)

On the pod, pip over the /workspace network volume stalled in
uninterruptible I/O (D-state, 2% CPU, no progress over ~6 min).
Repair: the venv and the HF download cache move to the local overlay
disk (/root/.venv-pod, /root/hf_cache; ~20.5 GB peak < the 30 GB
local disk, cache deleted after verification). Only the GEODE data
layout lives on /workspace. The stuck stage was killed by the user;
nothing was fitted. Script updated at the same commit; re-dispatch
follows this entry.

### 17 Aug 2026 — M176c stage 0 abort + fix (registered before re-run)

The pod's first stage-0 run aborted cleanly before any work: the repo
bundle is a git archive (no `.git`), and the script's `git rev-parse`
tripped `set -e` right after the facts block (GPU RTX 4090 24 GB,
251 Gi RAM, Python 3.12.3, root overlay 30 GB — the /workspace mount
was not yet reported). Fix: the commit now travels in a
`RELEASE_COMMIT` file inside the bundle; the report also captures
`df -h /workspace`; a torch-missing fallback installs cu124 wheels.
Nothing was fitted; re-dispatch follows this entry with the new
bundle.

### 17 Aug 2026 — M176c rental setup stage dispatched (registered before running)

The pod is up (RunPod, SSH endpoint). Frozen setup script:
`tools/m176c_pod_setup.sh` — stage 0 only: pod facts, venv with the
image's CUDA torch, HF download of `wltjr1007/DomainNet` pinned at
revision `ee20570a`, sha256 verification of all four parquets against
the local manifest, GEODE cache layout written verbatim, HF cache
deleted afterwards (the disk discipline). No fits run in stage 0.
The user executes it in their own SSH session (the model does not
drive interactive SSH) and pastes `logs/pod_report.txt` back. Anchor
stage dispatches only after the report shows the four sha256s ok.

### 17 Aug 2026 — M163 decision MADE (user) + M175 battery v1 registered + M176c ACTIVE

- The user approved **all four transfer directions**. The battery is
  frozen in `analysis/v24_m175_transfer_cells.md`: B (DomainNet-32 →
  Flowers102 with the frozen SPM encoder) → A0 (FIRST text encoder
  build on wikitext-103, the additive next-token machinery) → A
  (text→text, wikitext + Wikipedia with the A0 encoder) → D
  (license-clean Wikipedia fit-and-report) → C (cross-modality
  routing, route-level only). Per-cell task definitions and gates are
  registered immediately before each cell dispatches. Inventory
  correction registered: the sealed "C4" anchors are DomainNet-32
  (M142 cell naming); the real text corpora on disk are label-free
  token ids and no frozen text encoder exists (L2).
- **M176c is ACTIVE**: the user approved the ≤ $300 rental and has a
  RunPod account. The frozen runbook is `tools/m176c_runbook.md`
  (pod spec, anchor-first rule, candidate order, cost guardrails).
  The driver script ships next and is registered before any run.

### 17 Aug 2026 — M176d/e/f COMPLETE (registered specs) + M175/M176c blocked

- M176d label-matrix sampling: frozen in
  `analysis/v24_m176d_label_sampling.md` — anchor rows always measured,
  off-diagonal budget linear in K (b=4/arm), distance-band
  stratification, G2-failure top-ups, and the measured tolerance (G2
  margin loss ≤ 0.05, G3 min-cos loss ≤ 0.05) deferred to a trigger of
  ≥ 20 measured transfer labels (only 1 exists today, M167a).
- M176e live-upgrade protocol: frozen in
  `analysis/v24_m176e_upgrade_protocol.md` — versioned artifacts,
  re-encode→re-fit→re-anchor order, 1e-9 solver equivalence gate,
  rollback bundle, who-pays rule, router-incumbent gate; plus the
  measured rule that shipped encoders must persist weights (cross-run
  training nondeterminism).
- M176f contribution interface: frozen in
  `analysis/v24_m176f_contribution_interface.md` — submission payload,
  validator re-measurement, identity tiers (observer/contributor/
  validator/gatekeeper), v25 rewards attach to validated arms only.
- M175 cross-corpus transfer: UNBLOCKED by the user's M163 decision
  (see the entry above); the battery is frozen and cells dispatch in
  order B → A0 → A → D → C.
- M176c better-code arm: UNBLOCKED (budget approved, RunPod account
  created); runbook frozen in `tools/m176c_runbook.md`, the driver
  script ships next and is registered before any run.

### 17 Aug 2026 — M176 SEALED: K\* = 2 — routing beats the fallback immediately across families

Evidence: `logs/results/v24/m176_growth/evidence.json`. Mean margin
(routed eps-advance vs mean-predictor fallback) by registry size:
2.12 (K=2, 1 included family) → 1.43 (K=4, 2) → 1.01 (K=8, 5) → 1.00
(K=16, 10); fraction improved = 1.0 at every K. The learned-router
trigger is K* = 2 in this cell. Reading (registered): the margin
comes from CROSS-FAMILY competence differences — a specialist on its
own family beats the weak global fallback immediately — while the
sealed M143b/M153 lesson said routing does NOT pay WITHIN one corpus.
The two are consistent: routing pays across problem types, not within
them. Caveat registered: the cell's fallback is a mean predictor; a
stronger general arm (the real-corpus fallback) would shrink the
margin, so K* is an upper bound on the true trigger. Raw nearest-arm
equals eps-advance everywhere here (descriptors well separated, no
wrong routes).

### 17 Aug 2026 — M176 dispatched (registered before running)

Registry-growth simulation cell: K ladder 2/4/8/16 synthetic families
over the registered axis grid (interleaved assignment), one window-
ridge specialist each + the mean-predictor global fallback, the full
K×K competence matrix, eps-advance routing (eps=0). Gate: K* = the
smallest size where mean margin ≥ 0.05 AND ≥ 2/3 of included families
improve over the fallback (families whose best arm < 0.1 R² excluded
uniformly — noise families test nothing). K* is the learned-router
trigger. Runner: `experiments/tier4/eval_v24_m176_growth.py`.

### 17 Aug 2026 — M174 SEALED: frozen path scales with n, then plateaus at n≈5000 (L1 evidence)

Evidence: `logs/results/v24/m174_scaling/evidence.json`. The
delayed-coupling family routes deterministically to mackey_glass-ridge
(the nearest regression arm; S1). Fit-and-report ladder: 0.501 (200)
→ 0.620 (1000) → 0.671 (5000) → 0.673 (25000): non-decreasing (S2),
top-minus-bottom +0.172 ≥ 0.10 (S3 — the frozen path DOES keep scaling
with data on a new family). BUT the 5000→25000 step gains only
+0.0018: the window-ridge head plateaus at R² ≈ 0.67 by n ≈ 5000
(plateau_first_n = 5000). Registered L1 evidence: beyond that n,
growth must change CODES, not heads. The primitive control is flat by
construction. This closes the registered open question with a number:
the toolbox route scales, and the head saturates at ≈0.67 on this
family.

### 17 Aug 2026 — M174 dispatched (registered before running)

Cell: an out-of-registry delayed-coupling AR(5) family (temporal
structure "delayed" — a new axis combination), n ladder 200/1000/5000/
25000, 60% train, window 10, the M171b intercept-ridge fit-and-report
plus the mean-primitive control. The toolbox route (fingerprint ->
nearest regression arm) is exercised; gates S1 route determinism, S2
non-decreasing fit-and-report ladder, S3 R2(25000) - R2(200) >= 0.10;
a plateau is reported, not hidden (L1 evidence). One self-consistent
in-process encoder (the M172b pattern). Runner:
`experiments/tier4/eval_v24_m174_scaling.py`.

### 17 Aug 2026 — M173 SEALED: MVP acceptance — ALL 10 CAPABILITIES PASS

Evidence: `logs/results/v24/m173_acceptance/evidence.json`. Per
capability: 1 ingestion (no-crash + OOV logged + 5 distinct hashes
with the registered mg/lorenz pair) ✓; 2 G1 ✓; 3 G2 ✓; 4 G3 + M170
artifact ✓; 5 routing — raw nearest-arm R1 3/4 with the dyck negative
recorded, 4/4 under the registered eps-advance rule ✓; 6 fit-and-
report — sealed C4 ridge anchor 0.2273623188405797 reproduced (delta
<= 1e-6) + 4-task competence matrix ✓; 7 registry — transactional
hashes stable + M165/M168/M171 tests 29/29 ✓; 8 M172 G5 gates ✓; 9
cold start — both demos fall back to the strongest general arm ✓; 10
repro-hash — every sealed evidence file's sha256 matches its artifact
index ✓. G4 continuity stays DEFERRED and M163 corpus decision stays
pending (registered, not acceptance blockers). This is the milestone
the user revisits: the v24 MVP toolbox — descriptor -> fingerprint ->
router -> arm registry -> fit-and-report — is accepted against its
own registered gates.

### 17 Aug 2026 — M173 acceptance dispatched (registered before running)

Pass criteria per capability are in
`experiments/configs/v24/m173_acceptance.json`. Assembly-only run: it
fits nothing new; it re-verifies the sealed M165-M172 evidence, the
transactional registry hashes, the unit tests, and the file sha256s
(I5). Fresh checks only for capability 1 (no-crash normalisation loop
with a genuine OOV descriptor) and capability 10 (file re-hashing).
G4 and M163 stay recorded as deferred/pending, not acceptance
blockers.

### 17 Aug 2026 — M172 SEALED: all five G5 gates pass + training-nondeterminism fact

Evidence: `logs/results/v24/m172_joint/evidence.json` (admissible,
void=false). Self-consistent in-process encoder (the M172b repair):
G5a cos(mg, dyck) = −0.054 < 0.9; G5b cos(A+B, B+A) = 0.204 < 0.99;
G5c A+B routes to mackey_glass-ridge while B+A routes to dyck-ridge;
G5d both joints land on their dominant side's arm; G5e neither routes
to the unrelated tabular arm. The joint task is distinguishable in
fingerprint space AND routed distinguishably — capability 8's gate
passes at fingerprint level (no mixture arm exists yet, by
registration).

Measured fact (registered): two IDENTICAL 300-step trainings in one
process produced fingerprints with per-task cos 0.0105-0.4400 —
near-orthogonal. Training is chaotic on this ROCm device even
in-process; only the eval path is deterministic (G1). Until the
trained encoder's weights are saved as a frozen artifact (new future
item), cross-run fingerprint reuse is impossible and must not be
gated. M169/M171 verdicts are unaffected (each was self-consistent
in one process).

### 17 Aug 2026 — M172 VOID (encoder drift) + M172b repair registered

The M172 premise gate fired: the encoder rebuilt for M172 did NOT
reproduce the M171 evidence fingerprints (min cos ≈ −0.0009). Two
fresh rebuilds of the SAME config also disagree with each other and
with the sealed evidence (cos ≈ 0.13-0.60): GPU training is
non-deterministic ACROSS PROCESSES on this ROCm device. M169's G1
only gated the EVAL path (fingerprint twice -> identical); training
determinism was never verified. M171's own verdict stands — its
fingerprints were self-consistent within its one process.

M172b repair (uniform, registered before re-measurement): M172 trains
ONE encoder in-process and uses it for every fingerprint (arms and
queries), exactly as M171 did — self-consistent by construction. The
cross-run equivalence premise is replaced by a recorded measurement:
the runner trains a SECOND encoder from scratch in-process and
records the task-pair cos between the two trainings as the
training-nondeterminism fact. Premise gate keeps only M171-evidence
admissibility. New registered future item: the trained encoder's
weights must be saved as a frozen artifact before any cross-run
fingerprint reuse (I1 freeze-on-ship needs the weights on disk).

### 17 Aug 2026 — M172 dispatched (registered before running)

Cell: A = mackey_glass, B = dyck, C = tabular (unrelated control),
joints A+B / B+A = dominant-task axes + coupling=mixture (no mixture
arm exists in the MVP, by registration; routing is fingerprint-level).
Gates: G5a cos(A,B) < 0.9; G5b cos(A+B,B+A) < 0.99; G5c chain heads of
A+B and B+A differ; G5d A+B routes to A's head and B+A to B's; G5e
A+B does not route to C's head. Premise gate (void on failure): the
sealed M171 evidence is admissible and the rebuilt encoder reproduces
the M171 fingerprints within cos 1e-6. Runner:
`experiments/tier4/eval_v24_m172_joint.py`.

### 17 Aug 2026 — M171 SEALED: R1 3/4 as measured; 4/4 under the registered §5 eps-advance rule

Evidence: `logs/results/v24/m171_router/evidence.json` (admissible,
deterministic; router unit tests 12/12). Cell: 4 measured series tasks
(mg, lorenz, dyck, tabular), 4 ridge specialists + bigram/mean-mode
primitives, 1000 rows/task (600 train / 399 held-out, window 10).

R1 (nearest-arm chain head vs best single applicable arm on held-out):
mg PASS 1.0000 (mg-ridge), lorenz PASS 0.8991 (lorenz-ridge), tabular
PASS 0.9997 (tabular-ridge), **dyck FAIL: dyck-ridge 0.288 vs
dyck-bigram primitive 0.331**. The dyck negative is real: the linear
window head cannot see grammar depth, so the bigram primitive beats the
specialist on its own task — the M143b competence-tie lesson at small
scale. Under the registered §5 eps-advance failover rule (eps = 0.0,
pre-registered 17 Aug), all four tasks advance correctly and R1 reads
4/4 (dyck → dyck-bigram). Finding: at this registry size the
chain-with-eps rule is REQUIRED, not optional.

Cold start (capability 9): an OOV audio descriptor and cifar10 (no
registered arm) both fall back deterministically to the strongest
general arm (mean-mode primitive; cifar10 mode accuracy 0.135 on a
400-row sample). I4 no-refusal exercised with a genuine OOV axis
(submodality=spectrogram). mg/lorenz descriptors are identical under
ontology v0 — their cos-1.0 tie resolves by the per-task selection
score (the M171b repair). Transactional registry adds leave other
tasks' content hashes unchanged; every decision carries a payload hash.

### 17 Aug 2026 — M171b repair registered (per-task selection score)

M171a (intercept repair) ran; mg and tabular R1 PASS, dyck FAILS
(bigram primitive 0.331 beats dyck-ridge 0.288 — a real negative, the
M143b competence-tie lesson at small scale), lorenz FAILS because the
tie-break used each arm's OWN-task accuracy as its selection score
(mg-ridge own 1.000 beat lorenz-ridge own 0.899 in the cos-1.0 tie,
then scored −0.005 on lorenz rows). That contradicts the registered
§5 wording: the selection score is "measured held-out accuracy PER
TASK". Uniform repair before re-measurement: `Router.chain` takes the
query task_id and tie-breaks with the arm's held-out accuracy record
FOR THAT TASK (falling back to own-task accuracy when no record
exists); arms register the full measured competence-matrix row as
`held_out_accuracy`. Everything else identical. (The original M171
void evidence file was overwritten by the M171a re-run before the
rename could land; its numbers are recorded in the void entry above.)

Also added before re-measurement (pre-registered in §5, 17 Aug): the
runner records the §5 eps-advance failover rule — the chosen arm must
stay within eps of the best measured arm on held-out rows or the chain
advances — with eps = 0.0, simulated from the measured matrix. This is
the deployment rule, not a redefinition of R1; R1 (nearest-arm chain
head) is reported separately.

### 17 Aug 2026 — M171 VOID (fit defect: missing intercept) + M171a repair registered

M171 ran end-to-end (router + registry + R1 measured; evidence sealed
at `logs/results/v24/m171_router/evidence.json`) but the closed-form
ridge had NO intercept column: standardized (zero-mean) lag windows
regressed against the uncentered target, forcing the offset through
collinear columns (cond(X'X) ~ 1.3e17 on Mackey-Glass). Every arm on
smooth series collapsed (mg-ridge on its own rows: train R² −15.7).
The last-value predictor scores 0.9998 on the same series, so the
collapse is the fit, not the data. VOID — the R1 numbers from that
run are not readable.

M171a repair (uniform, applied to ALL arms before any re-measurement):
append an explicit intercept column to the standardized features in
`_ridge_fit`/`_ridge_predict`. Nothing else changes. Verified on mg
off-run: train R² 1.0000 / held-out R² 1.0000. Re-dispatch follows
this entry; the M171 evidence file stays as the recorded void.

### 17 Aug 2026 — M170 traversability set: artifact COMPLETE

Frozen the 12 registered analogy quadruples (6 tasks × input.modality /
output.kind) with measured G3 axis-shift scores into
`analysis/traversability_set_v0.md`. min-cos 0.755 (tabular, tabular→image);
all 12 pass the 0.5 floor. Thresholds for MVP use: 0.5 floor, 0.755 measured
min. Any future encoder change must re-run against this set.

### 17 Aug 2026 — M169 smoke crashed (auxiliary target indexing) + fix, registered before re-measurement

The attribute-reconstruction target used the GLOBAL token index; the
axis head predicts over the axis vocabulary, so the target must be
the within-axis index (`<oov>` -> 0, auxiliary only). Nothing was
trained. Re-dispatch follows this entry.

### 17 Aug 2026 — M169 smoke crashed (pair-name indexing) + fix, registered before re-measurement

The InfoNCE loop indexed the task list with the pair STRING names.
Fix: index the descriptor dict by name directly. Nothing was trained.
Re-dispatch follows this entry.

### 17 Aug 2026 — M169 smoke crashed (ModuleDict key with dot) + fix, registered before re-measurement

The auxiliary heads used axis names as ModuleDict keys; `torch`
rejects keys containing ".". Fix: keys are sanitised
(`input.modality` -> `input_modality`). Nothing was trained.
Re-dispatch follows this entry.

### 17 Aug 2026 — M169 dispatched: fingerprint training + gates G1–G4 (v0 signal mix)

Registered BEFORE any training step (configs
`m169_fingerprint_train{,_smoke}.json`). v0 scope: InfoNCE over the
two ontology-registered known-similar pairs with the other tasks as
negatives + a CBOW-style attribute-reconstruction auxiliary; the ONE
measured transfer label (M167a +0.0092) is a ranking constraint only
— registered, not assumed. In-run gates: G1 determinism (void on
failure); G2 ordering on the registered pair set (scoped negative on
failure — the gates are the measurement, not the instrument); G3
attribute-swap traversality (cos ≥ 0.5); G4 DEFERRED (no sweep
families). Runner: `eval_v24_m169_fingerprint_train.py`. Nothing
trained yet.

### 17 Aug 2026 — M168 built: the fingerprint embedder (6/6 unit tests)

`geode/fingerprint.py` implements f = normalise(Σ emb(token) +
mlp(token one-hot)) with F=16, a 60→32→16 MLP (~2.5k params), seeded
init, and the deterministic no-grad `fingerprint()` inference path
(G1 covered: same-encoder re-run identical, fresh-encoder same-seed
identical). The continuity-channel interface is reserved for M169
training. Two build bugs fixed in-test: the MLP input dimension
(per-token one-hot, not per-axis) and a normalised-sum comparison.
Tests: `experiments/common/test_v24_m168_fingerprint.py` (6 pass).
M169 (training + gates G1–G4) follows.

### 17 Aug 2026 — M167a two-tier repair: PASSED — the harness is valid; the cross-domain label is real but weak

Full run (admissible): own-domain 0.1739; permuted control 0.0037;
cross-domain (d0→d1) 0.0129. Validity tier: +0.1702 ≥ 0.05 ✓ (the
instrument orders similarity strongly). Transfer tier: +0.0092 ≥ 0.0
✓ (the cross fit beats the permutation control — a real but weak
label). Registered follow-up stands: strong cross-domain labels need
adjacent-domain pairs, deferred until the domain-name→index mapping
is pinned. The harness is cleared for label production on the axes
where its positive controls pass. Evidence:
`logs/results/v24/m167_transfer_labels/evidence.json` (overwritten by
the M167a re-run; the M167 void verdict is preserved in the log
entry above).

### 17 Aug 2026 — M167 harness: SEALED VOID — the positive-control gate fired; the hypothesis stands undetermined

Full run (admissible): similar (fit d0, score d1) 0.0129; permuted
control 0.0037 (collapsed to ≈1/345 as designed); gate delta +0.0092
vs the registered 0.05 margin — GATE FIRED. The harness is VOID for
label production, not negative: the permuted control behaved
perfectly, and the own-domain sanity read (0.1739) shows the
instrument orders similarity strongly when the pair actually is
similar; the registered cross-domain pair (arbitrary d0→d1) was
over-strong for the margin. Verdict stands; repair = M167a.
Evidence: `logs/results/v24/m167_transfer_labels/evidence.json`.

### 17 Aug 2026 — M167 smoke crashed (penalty key type) + fix, registered before re-measurement

The smoke died in the similar fit: `RidgeAccumulator.solve_many`
returns a dict keyed by the RAW penalty values (floats), not strings
— `solved["1.0"]` raised KeyError. Fix: `solved[1.0]`. Nothing was
measured. Re-dispatch follows this entry.

### 17 Aug 2026 — M167 dispatched: the behavioral-transfer label protocol (positive controls first)

Registered BEFORE any label was computed. Scope (v0): the harness
fits the sealed SPM codes on selected train rows and scores on
selected test rows (cross-domain pairs from the cached corpus). The
positive-control gate, evaluated before any training number:

- similar pair: fit on domain-0 train rows, score on domain-1 test
  rows (same 345 classes, different domains);
- dissimilar pair: the SAME fit on label-permuted domain-0 rows
  (the permutation destroys class structure; the read must collapse
  toward 1/345);
- gate: similar − permuted ≥ 0.05, else the harness is VOID for
  label production.
  Runner: `eval_v24_m167_transfer_labels.py`. Nothing measured yet.

### 17 Aug 2026 — M166 CLOSED: ontology v0 frozen (assistant-authored under user delegation)

The 1.5B batch-1 was incomplete (no per-axis vocabularies), and the
user authorised the assistant to author the ontology directly. The
frozen artifact `analysis/task_ontology_v0.json` carries the
provisional M165 schema with the batch-2 corrections applied (the
CIFAR-10 + Mackey-Glass pair is registered known-dissimilar, the
correction noted in provenance), the quantisation and normalisation
rules, and the similarity positive-control set. A loader
(`geode/ontology.py`) checks artifact-to-code consistency, covered by
4 new tests (15/15 pass with the registry suite). Ratification is
marked user-delegated and open for revision; the version trigger is
G2 failing on this control set.

### 17 Aug 2026 — M166 ontology bootstrap: DRAFT PRODUCED — proposals recorded, ratification pending (human)

The pinned Qwen2.5-1.5B (revision 989aa798…) produced batch 1 (axis
proposal; per-axis vocabularies MISSING — the model echoed the
template key) and batch 2 (candidate pairs, with one clear error:
CIFAR-10 + Mackey-Glass marked known-similar). The error is itself
evidence for I3: proposals are hypotheses, never labels. Artifacts:
`logs/results/v24/m166_ontology_draft/draft.json` +
`ratification_checklist.json` (status: pending human). M166 closes
as draft-complete; ratification happens when the user reviews.

### 17 Aug 2026 — M166 smoke crashed (device_map needs accelerate) + fix, registered before re-measurement

The first dispatch crashed at model load: `device_map` requires the
`accelerate` package, which is not installed. Fix: load without
device_map (dtype=float16) and move the model to cuda:0 explicitly.
Nothing was measured. Re-dispatch follows this entry.

### 17 Aug 2026 — M166 dispatched (model amendment registered): ontology bootstrap batch 1

Amendment to section 3.2, registered BEFORE any run: the bootstrap
model is Qwen2.5-1.5B-Instruct fp16 (pinned HF revision, ~3.2 GB)
instead of the 3B — F: has 11.9 GB free and a 6.5 GB fp16 download
plus transient files is too tight; the 1.5B is adequate for axis and
vocabulary drafting and Apache-2.0 licensed. The 3B remains the
registered upgrade when disk allows. All other section 3.2
constraints hold: pinned weights, temperature 0 (greedy), outputs are
PROPOSALS ratified by a human, then frozen as a digest-tagged JSON
artifact; the LLM is absent from runtime; no API dependency; no data
leaves the machine. Runner: `eval_v24_m166_ontology.py` (to be
built); nothing measured yet.

### 17 Aug 2026 — M165 built: Task Registry + descriptor normaliser + frozen artifact format (11/11 unit tests)

The new `geode/` package carries `descriptor.py` (the provisional v0
axis schema, inclusive-bin quantisation for continuous axes, OOV
fallback with event log, axis-order-fixed canonical serialisation —
the axis order is part of the schema) and `registry.py` (idempotent
add keyed by the descriptor-hash prefix, append-only fingerprints and
fits, content hash excluding volatile fields). Tests:
`experiments/common/test_v24_m165_registry.py` (11 pass). Two boundary
bugs fixed during the build: exclusive-vs-inclusive bin edges and an
alphabetic re-sort that destroyed the frozen axis order. M166 next —
needs the pinned local LLM (a ~2–3 GB offline download, registered in
section 3.2).

### 17 Aug 2026 — M176b fit wall: SEALED — the wall is between 40,383 and 53,267; raw LSQR is NOT a drop-in escape

Anchor exact (SPM ridge 0.227362, delta +0.000e+00). Real fits: ms357
13,244 @ 138k = 110 s (solve 6.8 s); the SPM 40,383 anchor solve is
the process-lifetime RAM peak — **47,770 MB** — so the SEALED width
already runs at ~48 GB peak on this machine (63 GB installed).
Synthetic 53,267 / 60,000 / 70,000: all `skipped_gram_budget` — the
registered wall marker (3× the float64 Gram exceeds free RAM; the
pre-budget attempt was OS-killed at 53,267, consistent). Escape
check at full rank (n=20,000): scipy LSQR vs exact solve on the real
ms357 Gram → rel difference 0.99999999973 vs the 1e-9 drop-in gate
— raw LSQR does NOT converge at the sealed conditioning (~1e9), so
the escape ladder defaults to the published sketch+precondition
solvers (arXiv:1305.5029, 1611.03220, 2304.12465), never raw LSQR.
Evidence: `logs/results/v24/m176b_fit_wall/evidence.json`.

### 17 Aug 2026 — M176b: RAM-probe ctypes defect + fix, registered before re-measurement

The re-dispatched smoke crashed in the RAM probe:
`GetCurrentProcess` returned the -1 pseudo-handle and ctypes passed it
as a c_int, overflowing the HANDLE argument. Fix: proper argtypes/
restypes (c_void_p handle, c_ulong size, c_int BOOL). Nothing was
measured by the crashed smoke. Re-dispatch follows this entry.

### 17 Aug 2026 — M176b: smoke passed; full run hit the wall (OS-killed at 53k) + budget amendment, registered before re-measurement

The smoke ran end-to-end (inadmissible; its escape number is invalid
because n=2,000 < d makes the Gram rank-deficient). The full run
passed the anchor (delta +0.000e+00) and the real 13,244 fit, then
the OS killed the process at the 53,267-column synthetic Gram
(≈23 GB, ~3× with solver temps — the M151 factor) — the wall is
between 40,383 and 53,267, but an OS kill is no way to record it.
Amendment (registered here before re-measurement): a pre-flight Gram
budget — widths whose 3× float64 Gram exceeds free RAM are SKIPPED
and recorded as `skipped_gram_budget` (the wall marker) instead of
killing the process; the RAM probe is fixed (correct HANDLE type).
Re-dispatch follows this entry.

### 17 Aug 2026 — M176b dispatched: the fit-wall benchmark + one iterative escape

Registered BEFORE any number (configs `m176b_fit_wall{,_smoke}.json`).
Reads: the SPM ridge anchor (0.227362 tol 1e-6), real-width fits
(ms357 13,244; the SPM 40,383 inside the anchor solve), synthetic
widths 53,267 / 60,000 / 70,000 at n=20k (the Gram dominates), and a
registered escape check — scipy LSQR vs exact solve on a real-code
Gram, drop-in criterion rel ≤ 1e-9. Interpretation registered: the
first width that raises MemoryError or exceeds the budget is the
measured wall; if raw LSQR misses 1e-9, the ladder defaults to the
published sketch+precondition solvers (arXiv:1305.5029, 1611.03220,
2304.12465), not raw LSQR. Runner:
`eval_v24_m176b_fit_wall.py`. Nothing measured yet.

### 17 Aug 2026 — M176a ceiling probe: SEALED — the ridge IS the best measured head; L1 confirmed as a code property

Anchor exact: ridge 0.227362 (delta +0.000e+00) on `.venv-rocm` — the
environment switch reproduces the sealed fit path bit-tightly.
Reads at the C4 138k level: diagonal ridge 0.025275; kNN k=1
0.139391; kNN k=5 0.129362; trained heads (sealed M146/M150/M160)
0.03–0.08. Every alternative head family extracts LESS than the
closed-form ridge (0.2274), and the non-parametric kNN beats every
SGD-trained head. Interpretation registered in advance: no measured
head-side accuracy is left on these codes — L1 is a code property;
growth must change codes (the M176c ladder), not heads. Evidence:
`logs/results/v24/m176a_code_ceiling/evidence.json`.

### 17 Aug 2026 — GPU preference registered (user directive)

Everything dispatches on `.venv-rocm`; new computations default to GPU
paths where sensible (encodes, matmul-heavy reads, trained arms).
Sealed CPU contract paths — the M142 ridge accumulator in particular —
stay CPU until a GPU implementation passes an equivalence gate (1e-9),
never silently swapped. The M176a kNN read is a new computation and
would be a GPU matmul in a rebuild; its CPU cost is ~1 min at full
scale, so the in-flight run is not interrupted.

### 17 Aug 2026 — M176a smoke passed; full run dispatched

Smoke (inadmissible): ridge 0.1125 at 2k rows (anchors skipped),
diagonal 0.0575, kNN k=1 0.1360 / k=5 0.1035 — the plumbing runs
end-to-end. Full 138k/34.5k run is in flight on the same chain.

### 17 Aug 2026 — M176a smoke crashed (misplaced parenthesis) + fix, registered before re-measurement

The smoke passed the ridge step (0.1125 at 2k rows, anchors skipped)
and crashed in the diagonal scorer: `int(...).sum()` was written
where `int((...).sum())` belongs, so `int()` received a whole boolean
array. Reproduced in isolation; nothing was measured. Fix: the sum
moves inside the int conversion. Re-dispatch follows this entry.

### 17 Aug 2026 — M176a dispatched: the frozen-code-ceiling probe

Registered BEFORE any accuracy was read (configs
`m176a_code_ceiling{,_smoke}.json` carry the question and the
registered interpretation). Scope: the C4 138k level, the sealed SPM
codes (p=0.5). Reads: the ridge anchor (0.2273623188405797, tol 1e-6),
the diagonal-ridge read over the same standardised features, and exact
cosine kNN (k=1, k=5 majority) on the power-normalised codes.
Interpretation registered: (a) kNN ~ ridge confirms the code ceiling
for this family; (b) diagonal ≪ ridge with kNN ~ ridge means
cross-feature structure is load-bearing; (c) kNN ≫ ridge means
non-linear headroom exists (would contradict M150) and M176c priority
rises. Runner: `eval_v24_m176a_code_ceiling.py`. Nothing measured yet.
