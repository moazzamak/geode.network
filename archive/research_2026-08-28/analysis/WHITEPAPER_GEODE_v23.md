# GEODE: a whitepaper (technical record edition)

## 1. Abstract

Most systems today learn a task by training a large neural network with
gradient descent — millions of small steps downhill — and then either keep
the whole thing or prune it back. We measured the opposite bet, stage by
stage, on a fixed benchmark: **build the system out of frozen, piece-by-piece
constructed parts, and fit each new task with a single exact solve instead of
any training at all.** The pieces are old, well-understood building blocks
(dictionaries of small image patches, spatial-pyramid pooling, power
normalisation); the fit is closed-form ridge regression; and the only
"learning" is which pieces to combine, decided by measurement.

The measured result, at this scale: **the frozen system wins.** The
18 Aug 2026 better-code arm (M176c-c1) pushes the program best to
**59.0% top-1 on 345-way DomainNet** — deep-patch SPM pooling over frozen
DINOv2-small tokens, still no training, ~382M MACs/image — beating the
sealed dense ladder at every rank (r70 0.312, r224 ≈0.537). The
piece-by-piece sparse family's best reads **27.9% at ~175M MACs/image**,
ahead of an unpruned DINOv2-small baseline at roughly half the compute,
and far ahead of every trained variant of itself. Every experiment that
added gradient training back in — trained heads, trainable dictionaries,
hybrid residual heads, prune-then-retrain — _lost_ to the frozen
closed-form read. Data, not training, is the measured growth lever on the
frozen codes (+3.7 points at 3× data), and codes, not heads, are the
measured quality lever (the deep-patch codes read 59.0% where the
small-patch codes cap near 28%).

GEODE (Generalized Encoders for Open-Domain Expertise) is this system
plus the measured toolbox (v24, accepted 17-18 Aug 2026): a **registry**
of frozen components, an additive **fingerprint** for every task, and a
**router** that sends each task to the pieces with measured expertise on
it — with every decision replayable from a hash. The ten-gate MVP
acceptance passed 10/10; routing beats the global fallback across task
families immediately (K\*=2), and the nearest-arm route plus the measured
failover chain reaches the best single arm on every measured task. Nothing
here is claimed as a new mechanism. Our contribution is the sealed,
matched-cost measurement of which combination works, and which
widely-assumed mechanisms (fine-tuning, pruning, retraining, routing)
do not, at this scale.

---

## 2. The problem, in plain words

Imagine you run a service that must answer many different kinds of tasks:
recognise objects in photos, forecast a noisy signal, route questions.
The industry default is one enormous network, trained once at great cost,
then bent to each task by more training ("fine-tuning"). Two defaults
follow: if it is too big, **prune** it (delete weights) and hope the
knowledge survives; if one model cannot do everything, **route** between
several specialised models, each also trained.

Three costs are hidden inside this default:

1. **Every task pays the training tax again.** Fine-tuning re-runs
   gradient descent over the whole network — hours to days of GPU time,
   even for a small change.
2. **The result is a moving target.** After training, the network is a
   new object with no relationship to the old one except provenance.
   Nobody can say _exactly_ what changed, because every weight changed.
3. **The knowledge is opaque and non-transferable.** What model A learned
   about task X cannot be looked up, quoted, or moved to model B; it can
   only be re-trained.

GEODE starts from a different observation, which is also the oldest one in
the field: a **frozen encoder** — a fixed, pre-built function that turns
an input into a table of numbers — can carry a task a long way on its own,
if what sits on top of it is fit **in one exact step** rather than trained
iteratively. The surprising part is not that this works; it is _how far_ it
works against trained competitors at matched cost, and how consistently
re-introducing training makes things _worse_.

This paper has three jobs: explain the architecture the way the Bitcoin
whitepaper explained double-spending — from first principles, for a
software engineer; report the measured results and their limits honestly;
and register exactly which ideas are borrowed from whom, so the claims we
use are visible and checkable.

---

## 3. The idea in one page

A GEODE system is a **catalogue of frozen experts plus one exact fit per
task.** Four sentences carry the design:

1. **Everything reusable is frozen.** Encoders are fixed functions with a
   version hash, like compiled libraries. A change is a _new_ library, not
   a mutation of the old one.
2. **Everything task-specific is one exact solve.** A task gets a
   closed-form ridge fit — the same linear-algebra operation as solving a
   system of equations — no epochs, no optimizer, no random seeds in the
   fit itself.
3. **Composition is additive.** Pieces combine by concatenation and
   summation, the way word vectors compose in classic word embeddings. If
   two pieces each help, their sum helps more; nothing needs to learn how
   to combine them.
4. **Routing is measured, not guessed.** Which pieces serve which tasks is
   decided by _measured transfer_ — literally, fitting the same frozen
   piece to two tasks and watching how the numbers move — not by a
   network's opinion.

The software analogy: end-to-end training is recompiling the whole codebase
for every change; GEODE ships libraries and writes one configuration file
per task. The config (the ridge weights) is small, human-inspectable,
versioned, and deletable per task without touching the libraries.

What this buys, if the measurements hold: new tasks without massive
retraining; audit trails for every decision; per-task unlearning by
deleting one fit; and a path to shared infrastructure where many parties
contribute frozen pieces and are paid for measured use (the subject of the
companion plan v25, sketched in §9).

---

## 4. How GEODE works

### 4.1 The pipeline

```
input
  → [contract check]        exact, zero learned cost; reject what is out of scope
  → [frozen encoder(s)]     fixed functions with version hashes
  → [code]                  the additive feature vector (concatenate + normalise)
  → [closed-form fit]       one exact ridge solve, per task
  → [output contract]       typed validation + confidence
output
```

Around this static branch sit three layers:

- **The registry (v24, under construction).** Every frozen component is a
  registry entry: data digest, code digest, weights digest, behaviour
  digest. Adding a task = adding a fit; nothing about other tasks changes.
- **The fingerprint and router (v24).** Every task gets a short vector
  (16–64 numbers) built by _adding_ learned attribute embeddings — the
  word2vec/GloVe mechanism — trained once and then frozen. The router sends
  a task to the arm whose measured expertise matches. Unknown tasks fall
  back to the strongest general arm; a safety-flagged task whose best
  admissible match falls below the floor is NOT answered — the router
  returns empty and the caller escalates (§11, the abstention path;
  cold-start is not a safety fallback).
- **The governance layer (v25, planned).** Inspection ladders, per-task
  audit replay, and an incentive mechanism (contributions earn vested
  tokens that thaw with measured inference use). Planned, not built.

### 4.2 The encoder: a field guide to small patches

The promoted encoder is deliberately classical, so the reader can hold the
whole thing in their head:

1. **Whitening.** Take every 6×6 patch of every image, normalise contrast,
   and rotate the pixel space so its statistics are the identity — this
   removes the "average image" and the dominant pixel correlations, which
   carry no class information. The whitening matrix is fit once on 400,000
   patches and frozen forever.
2. **The dictionary.** Draw 1,923 prototype patches from a seeded pool —
   a _field guide_ of typical image fragments. Every patch of every image
   is scored against every prototype with the soft "triangle" code:
   `code = max(mean_distance − distance, 0)` — full credit for a perfect
   match, zero beyond a fixed radius, linear in between. The image becomes
   a histogram of "how much does this fragment appear here", weighted by
   position.
3. **Spatial pyramid pooling.** The image grid is pooled at three zoom
   levels (1×1 whole image, 2×2 quadrants, 4×4 sixteenths), 21 regions
   total — so the code says _where_ things appear, coarsely, not just
   whether they appear.
4. **Signed square root + L2.** Each entry gets its square root (with
   sign preserved) and each row is normalised to unit length. This quiets
   the loudest entries — the classical Fisher-vector post-processing.

The result: 1,923 numbers per image, 21× bigger after spatial expansion.
The _same_ code is the substrate for every downstream experiment — one
encode, many fits, all comparable.

### 4.3 The fit: one exact solve

For a classification task, GEODE solves

$$\min_w \ \|Xw - Y\|^2 + \lambda\|w\|^2$$

in closed form — one matrix solve, no iterations. There is no epoch, no
learning rate, no early stopping, and no random seed inside the fit. The
same weights result from the same data, always. This is the single most
important design decision in the system, and §6 reports what happened when
we tried to replace it with trained heads: on these codes, training always
lost.

### 4.4 Implementation notes for engineers

What looks simple on paper had three real engineering battles, and they
matter for anyone reproducing this:

- **The 53,627-column solve.** The concatenated code (SPM × multi-scale)
  spans 40,383 + 13,244 columns; wider experiments reach the same
  league. A dense Gram matrix at that width is tens of GB. The shipped
  solver accumulates the Gram in fixed-size column chunks (bit-identical
  to the naive product), builds the centred system in place, spills it
  to disk, and solves with an in-place LU factorisation. Full-width
  temporaries were the thing that killed 63 GB machines; the chunked
  path survives.
- **Numerical honesty.** The system at λ=1.0 is ill-conditioned (condition
  number ~1e9). Float32-rounded standardisation statistics versus float64
  changed the fitted weights by 39% once — a silent, plausible-looking
  wrong answer. The rule now: promote to float64 inside the transform,
  cast back at the boundary, and _verify_ analytic guarantees numerically
  on the real dtype (we did exactly this for closed-form concept erasure,
  where float32 had degraded an exact guarantee 200-fold).
- **Reproducibility as a feature, not a chore.** Every corpus, config,
  code, and evidence file carries a payload hash. Every milestone run
  reproduces its sealed inputs bit-exactly before measuring anything new;
  a run whose anchor fails is **void**, recorded, and re-dispatched —
  never patched in place. This discipline is what makes §9's audit ladder
  possible: decisions replay from hashes.

### 4.5 The toolbox layer (shipped, v24–v25)

Plan `RESEARCH_IMPLEMENTATION_PLAN_v24.md` extended the static system
into the generalised toolbox the project is named for — all of it now
shipping code. The pieces, all small and all frozen after one training
pass:

- **Task descriptor and fingerprint.** A task is described by quantised
  attributes (input modality and shape, output kind, label cardinality,
  sample regime…). The fingerprint is the _sum_ of learned embeddings of
  those attributes plus a tiny network — so unseen attribute combinations
  still land at defined points, and "move along an attribute" is vector
  arithmetic, exactly as in word embeddings.
- **Similarity labels, measured, not guessed.** The fingerprint is trained
  on _behavioural transfer_: fit the same frozen arm to tasks A and B and
  measure how well it moves between them. These labels are generated by
  the system itself — each one is a few minutes of CPU.
- **The router.** Nearest-arm in fingerprint space, with the M143b-style
  fusion of routed arms and a registered per-task gate: the routed answer
  must beat the best single arm on held-out rows, or the router is not
  used for that task.
- **Ten gated capabilities** define the MVP: ingestion with a no-refusal
  fallback, deterministic fingerprinting, similarity ordering, basic
  traversability, routing, fit-and-report, transactional registry
  operations, multi-task differentiation (tasks A, B, and A+B must look
  different and route differently), cold start to the strongest general
  arm, and repro-hashes on every decision.

### 4.6 The governance layer (v25, partially shipped)

Two tracks follow the MVP: **safety** (a five-level inspection ladder —
decision replay, provenance chains, component attribution, behaviour
diffing on append-only updates, and capability maps — plus closed-form
unlearning reused from the erasure machinery) and **incentives** (a
blockchain-settled mechanism where contributors earn vested tokens that
thaw in proportion to the inference their components serve, a 2.5% usage
fee routed to a development treasury as a wash-trading tax, and an
explicit jurisdiction gate before any token mints). The safety half of
this design is now PARTIALLY SHIPPED as the alignment tranche (M241–M247,
20 Aug 2026, §11); the incentive half is simulated and locally
hash-chained, with public-testnet anchoring and token mints still behind
registered external gates.

### 4.7 Stage-by-stage reference: what enters, what leaves, and why

This is the pipeline read one stage at a time — the input, the output,
and the design reason for each. The invariant that runs through all of
it: **gradients exist in exactly one place (the fingerprint, §4.7.3);
everything else is a fixed function or an exact solve.**

#### 4.7.1 Contract check

- **Input:** a task definition (data + input/output spec) and a data
  digest.
- **Output:** pass/reject plus a typed contract (modality, output kind,
  label cardinality, domains).
- **Why:** the check costs zero learned compute and fails fast. A system
  that silently accepts out-of-scope work corrupts its own registry; the
  contract is also what later stages type-check against (the DAG-chaining
  stance: task B's contract must match task A's output).

#### 4.7.2 Descriptor normaliser

- **Input:** the free-form task definition.
- **Output:** a descriptor over the frozen ontology's axes (12 quantised
  attributes: input modality/submodality/value kind/temporal structure,
  output kind/ordinality, latent recurrence/stationarity/noise/label
  cardinality/sample regime, coupling), plus an event log (out-of-
  vocabulary fallbacks, above-top-bin events) and a content hash.
- **Why:** routing needs a finite vocabulary — continuous attributes are
  binned so "384-dim input" is a token, not a number. Fallback is never
  refusal: unknown values map to a registered `<oov>` token (invariant
  I4). The hash makes the descriptor itself an auditable artifact.

#### 4.7.3 Fingerprint (the one trained stage)

- **Input:** the descriptor vector.
- **Output:** a short fingerprint (16 dims): the normalised sum of
  learned attribute-token embeddings plus a tiny MLP (~2.5k parameters).
- **Why:** additive composition is the word2vec/GloVe mechanism, so an
  unseen attribute combination still lands at a defined point, and
  "change the modality" is vector arithmetic — traversability by
  construction (gates G1–G3). It is trained **only on descriptor
  tokens** — InfoNCE over the registered similarity pairs plus
  attribute reconstruction — never on an image, a pixel, or a code, so
  the image path stays gradient-free. Training is ~300 steps (~seconds);
  then it is frozen forever (I1). Its one measured caveat: training is
  nondeterministic across processes on ROCm, so shipped encoders persist
  their weights (the M176e rule).

#### 4.7.4 Router

- **Input:** the fingerprint, plus the registry.
- **Output:** an ordered failover chain — nearest arm in fingerprint
  space, ties broken by the registered selection score (measured
  held-out accuracy per task, availability, price), then the strongest
  general arm, then the always-available programmatic primitives. Every
  decision carries a payload hash.
- **Why:** deterministic routing is auditable (a learned policy may only
  replace it behind a measured gate). Selection is measured, never
  assumed — the chain advances past a chosen arm that falls outside a
  registered ε of the best measured arm (the Dyck lesson: the linear
  specialist lost to a bigram primitive on its own task, and the chain
  recovered). Unknown tasks fall back to the strongest general arm;
  safety-flagged tasks whose best admissible match is below the floor
  abstain (empty route, the caller escalates — §11).

#### 4.7.5 Arm registry

- **Input:** frozen components (encoders, dictionaries, primitives) with
  version and payload hashes.
- **Output:** append-only entries keyed by descriptor hash, each carrying
  its fingerprint, its measured per-task accuracy records, and its fit
  history.
- **Why:** transactions — adding task B never touches task A's entry
  (I1), and every component is content-addressed (I5). This is what
  makes "learn a new task" an append instead of a retrain, and what
  later lets third parties contribute arms (the v25 interface).

#### 4.7.6 Frozen encoder(s)

- **Input:** raw images.
- **Output:** the code — an additive feature vector.
  The promoted classical encoder: ZCA whitening (fit once on 400,000
  patches, frozen) → a seeded 1,923-prototype dictionary → the triangle
  activation (`max(mean distance − distance, 0)`) → 21-bin spatial
  pyramid pooling → signed square root + per-row L2. The measured
  better-code arm: SPM pooling over frozen DINOv2-small patch tokens
  (no training), which reads 0.590 where the small-patch codes cap at
  0.28.
- **Why:** classical pieces are holdable in the head, deterministic, and
  MAC-countable — matched-cost comparisons are the paper's evidence
  engine. The deep-patch arm exists because M176a proved the ceiling
  was the codes, not the head, so the fix had to be a code change.

#### 4.7.7 The fit — one exact solve

- **Input:** codes X and labels Y.
- **Output:** ridge weights from
  $\min_w \lVert Xw-Y\rVert^2 + \lambda\lVert w\rVert^2$ in closed form,
  plus the standardiser, the held-out accuracy, and the anchor
  reproductions.
- **Why:** no epochs, no learning rate, no seed — the same data always
  yields the same weights. The sealed frozen-vs-trained family tested
  five independent ways whether gradients could do better on these
  codes; all five lost (trained heads −9 to −20 points). The fit is the
  one design decision the whole evidence stack hangs on.

#### 4.7.8 Output contract

- **Input:** the fitted weights and held-out evidence.
- **Output:** typed predictions with confidence, per-domain report, and
  the payload hash of the fit.
- **Why:** the output is itself a contract for the next consumer (DAG
  chaining), and evidence is admissible only when its anchors reproduce
  — a number without its hash is not a number in this system.

### 4.8 Operating the system (the manual)

The shipped package (`geode`, v0.15.0) installs as a wheel
(`pip install .`), exposes a console command, and runs the same
operations below as one-liners: `geode route --fp 0.9,0.3,0.2,0.1`,
`geode artifacts verify --path ... --digest ...`, `geode freeze
--attest v1,v2 --ttl 1000`, `geode override ...`, `geode serve`,
`geode verify --evidence <dir>`. `python examples/hello_geode.py`
walks the whole tour; `docs/QUICKSTART.md` is the five-minute
introduction. The everyday operations, with their registered safety
semantics:

- **Register an arm.** Build the arm spec (fingerprint, output
  contract, measured held-out accuracy, availability, price) and call
  `router.add_arm(spec)`. Admission is append-only; while a
  quorum-attested freeze is active, `add_arm` raises — containment
  binds admission too (§11.5).
- **Route a task.** `router.route(fingerprint)` returns the top-k
  records, each carrying `route_cos`, `ranked_by` (empirical vs task
  fingerprint), and `provisional`. For a safety-flagged task, pass
  `required_tags` and `abstain_below`: arms without measured coverage
  are excluded, and a below-floor result is an EMPTY route — escalate,
  do not force a best guess.
- **Guard the input.** Fit an `OodGate` on the reference distribution
  and pass it as `ood_guard` with the query vector: out-of-distribution
  inputs return empty (fail-closed).
- **Contain the registry.** `FreezeRegistry.freeze(...)` with k-of-n
  attestations halts routing and admission until its expiry index —
  a freeze cannot be permanent. `unfreeze` names the specific event.
- **Intervene as a human.** `OverrideLedger.record(actor, action,
justification, counterfactual)` — blank justifications and missing
  counterfactuals are rejected by the API.
- **Settle a dispute without courts.** A challenger builds a
  zero-knowledge dispute payload (`build_dispute_payload`); the
  SlashLedger adjudicates through the injected verifier — a lying
  claim fails and is slashed, a false accusation slashes the
  accuser (live deployment gated on the M194 anchor).
- **Verify anything.** `AuditAPI` replays any sealed evidence file
  bit-exactly into a scratch directory; the ledger's `verify()`
  re-hashes the whole chain.

Each of these is exercised by the unit suite (458 tests at v0.15.0),
which is part of the package, not a separate artifact.

---

## 5. What we measured, and how to verify it

This section is the evidence book: each table is a claim, and every
claim has a replay path. A number in this paper means: the sealed run
under `logs/results/` reproduces its registered inputs bit-exactly
(anchors at 1e-9 where the quantity is known in closed form), its gates
passed, and its verdict was registered BEFORE the run. To verify any
figure, replay the cited milestone through the audit API — the hash in
the evidence file is the claim's fingerprint, and a replay that does not
match voids it. Nothing below is an assertion; everything is a pointer
to a sealed measurement.

### 5.1 The construction ladder — the promoted recipe

Full data, matched ~175.2M MACs/image, closed-form ridge readout:

| Construction                          | Top-1      |
| ------------------------------------- | ---------- |
| Single 6×6 patch pool                 | 0.2275     |
| Multi-scale 3/5/7 patches             | 0.2421     |
| + signed sqrt + L2                    | 0.2507     |
| 21-bin spatial-pyramid pooling        | 0.2605     |
| **SPM + signed sqrt + L2 (promoted)** | **0.2786** |

Each stage pays for itself, and the pyramid's fine level carries the gain
(1×1 → 0.154, 2×2 → 0.224, 4×4 → 0.260). The power-norm gain is entirely
the square root; plain L2 alone hurts slightly.

### 5.2 The frozen-vs-trained family — the headline

One question was tested five independent ways. All five answer the same:

| Probe                                                   | Frozen closed-form  | Trained variant | Δ             |
| ------------------------------------------------------- | ------------------- | --------------- | ------------- |
| M146 r2: trained linear head on the same codes          | 0.2274              | 0.0426          | −18.5         |
| M146 r3: trainable dictionary + trained head            | 0.2274              | 0.1060          | −12.1         |
| M150: trained heads across the whole cached code family | 0.21–0.28           | 0.03–0.08       | −20           |
| M160: every schedule of the trained head                | 0.2274              | 0.0410–0.1345   | −9.3 to −18.6 |
| M161: ridge + trained _residual_ head (the hybrid)      | 0.2274              | 0.0765          | −15.1         |
| M162: prune DINOv2-small, then retrain it               | 0.1076 (prune only) | 0.0597          | −4.8          |

Three details matter. M160 swept schedules: more epochs helped (+2.8),
lower learning rate hurt (−6.5), and _no_ schedule approached the frozen
read. M161 trained a residual on top of the frozen logits — the middle
ground — and the trained part actively **destroyed** the frozen part.
M162 retrained a pruned backbone under the standard schedule and landed
**below** the never-retrained pruned baseline. Retraining is not merely
useless here; it is measurably harmful.

### 5.3 Dense, pruned, and retrained baselines

The frozen DINOv2-small ladder (closed-form ridge on its features) sets
the dense curve: r42 0.1972 @ 215.6M; r56 0.2450 @ 367.5M; r70 0.3118 @
564.2M; r98 0.4476 @ 1.1G; r224 ≈ 0.537 @ ~6.1G. Channel-magnitude
pruning (no retraining) collapses it: keep 50% → 0.1076 @ 185.0M; keep
25% → 0.0695 @ 104.4M. The additive recipe at _fewer_ MACs than the
pruned arms (175.2M) holds 0.2786. Pruning down to additive-scale budgets
is not a path to this frontier; and §5.2 shows retraining does not rescue
the prune.

### 5.4 Additivity, fusion, routing

- **Additivity pays.** Concatenating the SPM code with the multi-scale
  code: 0.2975 — above either half alone (0.2605 and 0.2421) and
  +0.037 over the better half. The sharing that pays is _gated fusion
  weights per domain_ (0.2395, +1.49 points over the global arm);
  sharing head _data_ across domains does not (0.2242, a tie).
- **Routing on one corpus does not pay.** Specialist fusion fit on
  held-out rows: 0.1463 vs global 0.2251. Refit on the arms' own scores,
  fusion ties the global arm exactly and never exceeds it; competence
  routing loses to plain identity routing. This is a finding about the
  interface at this scale, not about routing in general — routing across
  _different problem types_ is exactly the gap the v24 toolbox fills, and
  it remains untested.
- **"More of the same" stopped paying.** The last four construction
  variants all lost: finer 8×8 pooling (−3.9), growth to more atoms
  (−0.05), learned dictionaries (−0.5 vs the random draw), and a wider
  power-normalisation sweep (+0.09, within noise). The recipe has
  saturated on this corpus; growth must come from new axes and corpora.

### 5.5 Data is the measured lever

The same frozen codes, fit on more data: 0.2246 at 138k rows → 0.2614 at
410k rows, with the gain accelerating at the end. This is the only
measured lever that reliably moves the frozen system, and it is the
registered scaling bet for the toolbox.

### 5.6 The temporal branch

One-step-ahead Mackey–Glass forecasting (NRMSFE): no-memory ridge 0.1459;
tap-delay 0.0181; hand-written programmatic primitives 0.0032; a fixed
random reservoir 0.0022–0.0027 — feedback earns its keep on the chaotic
axis. On the small-language axis the reservoir _loses_ badly (perplexity
29.7 vs 3.3 for a plain count-memory), and tiny trained transformers win
everything (2.80). The measured rule survives both axes: **learned
components are used where the measured price of learning pays, and only
there.**

### 5.7 Out-of-corpus points

The same recipe family fit from scratch on CIFAR-10: 0.62 → 0.69 across
the 128 → 1,024-atom ladder. Two measured from-scratch successes; no
transfer claim is made beyond those two points.

---

### 5.8 The toolbox and the better-code arm (v24, sealed 17-18 Aug 2026)

The toolbox milestones M165-M173 built and GATED the registry,
fingerprint, router, and fit-and-report; the MVP acceptance passed all
ten capabilities (ingestion, deterministic fingerprinting G1, similarity
ordering G2, traversability G3, routing R1, fit-and-report with the
0.22736 anchor reproduced, transactional registry operations, multi-task
differentiation G5, cold start, repro-hash). Sealed routing numbers:
nearest-arm R1 is 3/4 on the measured series families (the Dyck negative:
a bigram primitive beats the linear specialist on its own task — the
M143b competence-tie lesson at small scale); the measured failover chain
(eps-advance) recovers 4/4; routing beats the global fallback already at
registry size K=2. M174: the frozen fit-and-report scales with data
(R² 0.501 → 0.673) but plateaus at n≈5000 — L1 evidence that codes bound
the read. M176c-c1 then changed the codes: deep-patch SPM on frozen
DINOv2-small tokens (no training; the arXiv:1603.09046-style spatial
pyramid VLAD construction, cited as prior art) reads **0.487 / 0.563 /
0.590 at 256 / 1024 / 2048 atoms** — above the sealed dense ladder at
every rank and above the sealed sparse frontier 0.2786 by 2.1×. Two
registered instrument lessons travel with these numbers: the M108 device
gate (bit-exactness is device-pinned; the rental RTX 4090 passed the
256-row bit-exact verification before any figure was admitted) and the
intercept-column fit repair (M171). The registered comparison, M176c-c2
(Fisher vectors, Perronnin & Sánchez, on the same deep-patch tokens),
is now measured and closed: 0.5987 / 0.5990 at widths 12,288 / 24,576
(penalty-invariant; the K=16→K=32 step buys +0.0003 for 2x the width).
Its smallest rung is 6x SPM's largest, and there it gains +0.009 over
SPM-2048 (0.5987 vs 0.5899) — a 6x per-dimension efficiency loss.
Fisher does not beat SPM at a comparable width, so it closes as a
comparison point and deep-patch SPM stays the deployment arm.

The first cross-corpus vision forensics (M175 B, sealed 18 Aug 2026):
the frozen DomainNet SPM-1923 encoder, read on Flowers-102 five-shot
with a new head, scores 0.167 against a cached DINOv2 CLS baseline of
0.990 — but the forensics separate what that means. The 32x32 input
still carries the species signal for a strong reader (0.827); the
frozen DomainNet head reads the flowers as `flower` (17.6% of test
rows, 61x the uniform 1/345) with the rest of the mass on
flower-like classes (watermelon, pear, strawberry, carrot, sun) — the
codes know WHAT a flower is; and even a flowers-native SPM fit reads
only 0.19 — the construction does not separate WHICH flower at 32x32
five-shot. Generic semantics transfer; fine-grained geometry does not.

---

## 6. Why freezing wins (and what would change our mind)

The pattern across §5.2 is so consistent it deserves a mechanism, stated
cautiously: **the code carries the information; the trained head destroys
what the closed-form read preserves.** When a trained head collapses to
0.03–0.08 on codes that a one-shot solve reads at 0.28, the bottleneck was
never the head — it is what the frozen encoder puts into the code. Adding
gradients does not add information; on these small, ill-conditioned,
additive codes it measurably subtracts.

Both registered probes have now fired and both confirmed the mechanism.
M176a: the ceiling probe bound the head — kNN and diagonal-ridge reads on
the sealed codes cannot extract more than the closed-form read, so the
ceiling was a property of the CODES (L1 confirmed). M176c-c1: changing the
codes lifted it — SPM pooling over DINOv2-small patch tokens (still no
training) reads 0.487 at 256 atoms and **0.590 at 2048 atoms**, beating
the whole sealed dense ladder. Growth must change codes, not heads; the
codes changed, and accuracy followed.

---

## 7. Prior art: what we borrowed, and what we measured against it

GEODE is deliberately assembled from old, named parts. This section
refers to each prior work we stand on, says what we took from it, and
what we measured against it. The comparison table follows the tour;
§8 records the formal claim ledger with sources and standing.

- **The classical image-code stack.** ZCA whitening (the standard
  de-correlation transform), dictionary coding, the triangle activation,
  spatial-pyramid pooling (Lazebnik, Schmid & Ponce 2006), signed square
  root and L2 normalisation — the Fisher-vector post-processing of
  Perronnin & Sánchez (2013) — are our promoted encoder verbatim. We
  claim nothing about these parts; we measured the _combination_ at
  matched cost, and measured Fisher vectors directly against the
  deep-patch SPM arm (M176c-c2: 0.5987 at 12,288 dims vs 0.5899 at
  2,048 dims — a 6× per-dimension efficiency loss; Fisher closes as a
  comparison point, not a deployment arm).
- **The deep-patch arm.** Pooling frozen DINOv2-small patch tokens with
  a spatial pyramid follows the arXiv:1603.09046-style deep-patch
  SPM-VLAD construction; the frozen backbone is DINOv2 (Oquab et al.
  2023), used unmodified. This is the arm that lifted the measured
  frontier (0.590).
- **Word embeddings and task embeddings.** The additive fingerprint is
  the word2vec/GloVe mechanism; the idea of a vector per task comes
  from the Task2Vec line (arXiv:2112.05647; Taskonomy 2018). The
  measured difference: our fingerprint trains on _behavioural transfer
  labels_ (fit the same arm to two tasks and measure the move), never
  on task data.
- **Transferability scores.** LEEP (Nguyen et al. 2020) and LogME (You
  et al. 2021) are reused as training labels, not as decisions; the
  metric-instability work (arXiv:2204.01403) justifies the ranking-
  stability gate.
- **Routing and mixtures.** Sparse-gated MoE (Shazeer et al. 2017),
  Routing Networks (Rosenbaum et al. 2018), and switch transformers
  (Fedus et al. 2022) are the router's antecedents; we measured that at
  this scale, competence routing on one corpus loses to one global head
  — and built the measured failover chain instead.
- **Closed-form solvers.** The divide-and-conquer KRR line
  (arXiv:1305.5029) and two-level preconditioning (arXiv:1806.05826)
  are the registered escape ladder if the system ever goes wider than
  our exact solves.
- **Attribution economics.** Data Shapley (Ghorbani & Zou 2019), Shapley
  in ML (arXiv:2202.05594), and Beta Shapley (arXiv:2110.14049) are
  imported as the contribution estimators; the anti-wash stack and
  quorum verification are measured in the M184/M245 harnesses.
- **Privacy and erasure.** The zk/MPC track builds on zkDL
  (arXiv:2307.16273) and secret-shared regression (arXiv:2309.09486);
  closed-form unlearning reuses LEACE (Belrose et al. 2023) — the
  float64-promotion fix we report is an engineering correction to the
  dtype, not a new method.

| Approach                                   | What they do                                            | Measured difference on our bench                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| End-to-end training + fine-tuning          | Gradient descent over the whole network, per task       | M146/M160/M161: every trained readout loses to the frozen closed-form read on the same codes (−9 to −19 points)                                                                                 |
| Train-big-then-prune (+ retrain)           | Delete weights, retrain the rest                        | M144/M162: pruning to our budgets loses 13.7 points; retraining loses a further 4.8                                                                                                             |
| Mixture-of-experts routing                 | Learned gating between specialised models               | M143/M143b/M153: on one corpus, fusion ties and routing loses to one global head                                                                                                                |
| Task embeddings (Task2Vec, Taskonomy line) | Learn a vector per dataset/task for transfer prediction | We borrow the idea (fingerprint) but train it on _measured behavioural transfer_ and keep it additive and deterministic; unmeasured difference to date                                          |
| Large-scale kernel ridge (Falkon line)     | Approximate closed-form solvers for big data            | Same family as our fit; we solve exactly at our widths and register the divide-and-conquer/preconditioned literature (arXiv:1305.5029, arXiv:1806.05826) as the escape ladder for wider systems |
| Transferability probes (LEEP/LogME)        | Cheap scores for whether a representation transfers     | We reuse the _measurement_, not the score, as the training label for the fingerprint                                                                                                            |
| zkML / MPC inference                       | Prove or hide computation                               | Track P of v25: local-encode first, secret-shared Gram fits (our fit is a sum of outer products — MPC-friendly by construction), zk proofs for the tiny router/head only                        |
| Blockchain ML tokenomics                   | Tokens for compute/data contributions                   | We register contribution by _measured attribution_ (Shapley line, arXiv:2202.05594) with anti-wash rules and a jurisdiction gate; designed, not yet simulated                                   |

The honest summary: GEODE's pieces are all borrowed; the measured
difference is _which combinations win at matched cost on one sealed
benchmark_, plus the determinism/audit discipline that makes every
decision replayable.

---

## 8. The claim ledger: prior art referred to, claims used, and their standing

§7 named the works we build on; this section is the formal record: how
we searched, which claims we USE (and their sources), and what we
explicitly do not claim. The standing column is part of the substantiation —
"concluded, not redone" means we cite the published result and do not
re-derive it; "re-measured" means our own sealed run is the evidence.

### 8.1 How the search was run

The buildout-blocker search (M164, 17 Aug 2026) rebuilt the literature
instrument after the earlier survey's positive control failed (1 of 6
registered papers). The rebuilt instrument registered six anchor queries
whose papers certainly exist, queried by topic — _not_ by title. **All
six anchors hit on the first stage**, so the search is admissible for its
registered role: **displacement only.** It can show that an idea we use
already exists in print; absence from it proves nothing, and no novelty
claim is licensed by it. Rate-limit failures are retried and recorded
separately from empty results.

### 8.2 Claims we use, and their standing

| Mechanism                                        | Prior art                                                                                                                                           | Our position                                                                                                                                                            |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dictionary codes, triangle code, SPM, power-norm | Fisher Vectors (Perronnin & Sánchez 2013); VLAD (Jégou et al. 2012); spatial pyramid (Lazebnik et al. 2006); deep-patch SPM-VLAD (arXiv:1603.09046) | Concluded, not redone. We re-measure _combinations_ at matched cost. The deep-patch variant is registered as a comparison baseline for the better-code arm, not a claim |
| Learned dictionaries                             | Deep dictionary learning (arXiv:2012.12509, arXiv:1912.10804)                                                                                       | Re-measured at our scale: a learned VQ dictionary lost 0.5 to a random draw (M113) — a scoped negative, not a refutation of the family                                  |
| Closed-form ridge + solvers                      | Classical; divide-and-conquer KRR (arXiv:1305.5029); two-level preconditioning (arXiv:1806.05826)                                                   | We solve exactly at our widths; the published solvers are the registered escape ladder for wider systems                                                                |
| Frozen features + probes                         | DINOv2 (Oquab et al. 2023); linear-probe practice (arXiv:2605.07194)                                                                                | DINOv2 used unmodified as the dense baseline; probe literature cited for the ceiling probe                                                                              |
| Reservoir computing                              | Jaeger 2001–2002                                                                                                                                    | Re-measured on one new axis with additive and programmatic controls                                                                                                     |
| Transferability measures                         | LEEP (Nguyen et al. 2020); LogME (You et al. 2021); metric instability (arXiv:2204.01403)                                                           | Reused as _labels_, not as scores; the instability paper justifies our ranking-stability gate                                                                           |
| Task embeddings                                  | Task2Vec line (arXiv:2112.05647; Taskonomy 2018)                                                                                                    | Idea borrowed for the fingerprint; our training signal (measured transfer) is the difference to be measured                                                             |
| Routing / mixture of experts                     | Shazeer et al. 2017 (sparsely-gated MoE); Rosenbaum et al. 2018 (Routing Networks, arXiv:1711.01239); Fedus et al. 2022 (switch transformers)       | Idea borrowed for the router; the registered per-task gate (routed ≥ best single arm on held-out rows) is the measured difference                                       |
| Task-conditioned architecture (the LeCun line)   | Grathwohl, Wang, LeCun et al., the AMI configurator (arXiv:2306.02572); I-JEPA task conditioning (arXiv:2301.08243)                                 | Design antecedents of the fingerprint/router: they propose a gating module; v24 measures one                                                                            |
| Unified task/dataset embeddings                  | unified task embeddings (arXiv:2402.14522); dataset-similarity review (arXiv:2312.04078)                                                            | Cited for the fingerprint's attribute-overlap signal; the behavioral-transfer label stays the registered difference                                                     |
| Pruning / lottery tickets                        | Frankle & Carbin 2019; retraining-free pruning (arXiv:2308.03449, arXiv:2212.12651)                                                                 | M144/M162 are baseline measurements consistent with this literature                                                                                                     |
| Data valuation / Shapley                         | Data Shapley (Ghorbani & Zou 2019); Shapley in ML (arXiv:2202.05594); Beta Shapley (arXiv:2110.14049); EcoVal (arXiv:2402.09288)                    | Imported as the attribution estimators for v25's contribution measurement; gated, not re-derived                                                                        |
| zk / MPC                                         | zkDL (arXiv:2307.16273); secret-shared logistic regression (arXiv:2309.09486); Shamir-secret regression (arXiv:2109.11200)                          | Track P builds on these; the secret-shared Gram fit is a known construction we plan to measure, not claim                                                               |
| Concept erasure                                  | LEACE (Belrose et al. 2023)                                                                                                                         | Reused for closed-form unlearning; the float64-promotion fix is ours to report, not the method                                                                          |

### 8.3 What is explicitly not claimed

- **No novelty of mechanism.** Every construction, the reservoir, the
  fusion layer, the fingerprint idea, and the token mechanism are prior
  art or imported designs. The search instrument licenses no "first".
- **No transfer claim.** Two corpora measured; hyperparameters developed
  on DomainNet; everything else unmeasured.
- **No continual-learning claim.** Adding task B while keeping task A was
  the machinery that measured negative (growth, splitting, routing); the
  shipped system refits per problem.
- **No high-compute claim.** The small-patch family's best (0.2786) sat
  below the dense ladder above r70 (~0.31); that regime is now measured:
  the deep-patch arm (M176c-c1) reaches 0.5899 at ~382M MACs, above the
  whole dense ladder — the claim stays confined to frozen, training-free
  construction at this cost point; SoTA-LLM-class tasks remain open and
  registered, not answered.

---

## 9. Known limits (the honest list)

1. **The code ceiling — measured, then lifted.** Trained heads collapsing
   to 0.03–0.08 mean the _codes_ cap accuracy; M176a bound the head and
   M176c-c1 lifted the codes (0.590 at ~382M MACs). What remains open is
   whether the deep-patch family keeps scaling toward SoTA-class tasks —
   registered, not claimed.
2. **One corpus, one modality.** Text, audio, and true sequential paths
   are mostly unbuilt; the M175 transfer battery's first cell is now
   measured — the frozen DomainNet SPM encoder on Flowers-102 five-shot
   reads 0.167 vs the cached DINOv2 CLS baseline's 0.990 (a scoped
   negative for that encoder; the deep-patch deployment arm is
   unaffected). A0/A/D/C remain queued.
3. **The recipe has saturated locally.** The last four small-patch
   construction variants all lost.
4. **The fit is quadratic in feature width.** At ~40k columns we needed a
   chunked Gram and a disk-spilled solve; doubling the width quadruples
   the Gram. Published solver escapes exist; unvalidated at our scale.
5. **Numerical fragility.** Ill-conditioned systems and dtype choices have
   silently moved weights by 39% in this program; discipline is a
   substitute, not a fix.
6. **The toolbox is demonstrated at MVP scale.** The ten acceptance
   gates passed; what is unmeasured is the toolbox at large-registry,
   multi-party, and production-latency scale.
7. **Per-task cold start.** Each new axis needs a new frozen encoder —
   a pipeline, not a lookup.
8. **Training nondeterminism.** The fingerprint's eval path is
   deterministic (G1), but its TRAINING is not reproducible across
   processes on ROCm (measured cos 0.01-0.44); shipped encoders must
   persist weights (the M176e rule).

---

## 10. What's next

- **v24 — remaining measurements.** Candidate 3 (a small trained
  encoder, last resort) is now the only unrun comparison — candidate 2
  (Fisher vectors) was measured and lost the registered comparable-
  width clause (18 Aug 2026); the M175 cross-corpus
  transfer battery is COMPLETE 5/5 (B: scoped negative for the
  SPM-1923 encoder, 0.167 vs 0.990 — with the diagnostic showing the
  32x32 input is not the blocker, the construction is; A0: the first
  text encoder's window dial inverts beyond w=2 on natural text, gate
  fired, no arm selected; A: text→text transfer HOLDS, gap 1.04x; D:
  the license-clean Wikipedia arm reads 9.51 held-out, license posture
  recorded; C: the contract guard keeps every task's chain within its
  modality — no silent cross-modality routing); the M176d
  label-sampling trigger and the M176e/f deployment protocols.
- **The deployment consequence.** The deep-patch arm enters as a frozen
  arm behind the router (never end-to-end retraining), with the
  per-MAC gate re-measured on every new registry axis.
- **v25 — safety and incentives.** The inspection ladder (decision
  replay, provenance, attribution, diffing, capability maps), closed-form
  unlearning, and the contribution/vesting mechanism with its registered
  simulations, anti-wash stack, EVM-L2 settlement, and the jurisdiction
  gate before any token mints. SHIPPED since (20 Aug 2026): the
  alignment + Byzantine tranche M241–M247 (§11) and the two-fingerprint
  architecture (M227, v0.5.0) — routing constraint tier with abstention,
  the empirical drift gate, the override ledger, demerits,
  provenance-weighted trust decay, the refusal-capability admission
  interface, and quorum/median measurement aggregation.
- **The scaling trigger.** A sequence-style component enters only when a
  measured gap fires its trigger — a task family where primitives and
  reservoirs demonstrably fail — and enters as another frozen arm behind
  the router, never as end-to-end retraining.

---

## 11. Alignment and risk mitigation (human and AI)

GEODE's alignment posture has two sides — human and machine — and both
follow one rule: **alignment is a measured property, never a declared
one.** Nothing below is a promise or a position paper; every item is
shipping code in the product package (v0.5.0–v0.10.0, 20 Aug 2026) with
unit tests, or a registered gate with a named trigger. The two
fingerprints of M227 encode the rule in the architecture: the task
fingerprint (authored rules) gates admission, the empirical fingerprint
(measured task-by-arm outcomes only) ranks selection, and a provisional
arm — declared but unmeasured — is marked and can never be treated as
measured.

### 11.1 Human alignment: aligning the operator to the evidence

The system is run by people, and the measured failure mode is not
malice but post-hoc rationalisation — reinterpreting an experiment after
seeing its outcome. GEODE counters it with mechanism, not culture:

- **The sealed discipline is code-convention, not custom.** Register the
  question, the gates, and the interpretation BEFORE running; anchors
  reproduce at 1e-9 before any number is read; a void is not a negative
  and a failed instrument is not a refutation. Amendments are separate
  ledger entries, never silent edits.
- **The override ledger (M243, v0.6.0).** Every human intervention —
  manual re-rank, admission exception, kill-switch, constraint waiver —
  is appended to an append-only hash chain WITH a justification and the
  counterfactual of what the system would have done. A blank
  justification or a missing counterfactual is rejected by the API, so
  an intervention without a recorded reason cannot happen silently.
- **Audit by replay.** Every decision reproduces from its payload hash
  into a scratch directory; timing fields are excluded from content
  hashes by a standing list; sealed evidence is never overwritten. The
  audit answers "what did the system do, and can it be recomputed",
  which is the precondition for any human review.

### 11.2 AI alignment: aligning the arms to measured behaviour

Arms (components) earn admission and credit from measurement, so the
only lever that exists is the one the registry controls:

- **Hard constraints at the router (M241, v0.6.0).** A safety-flagged
  task excludes provisional and unvetted arms — hard exclusion, never
  down-ranking. Constraint decisions read ONLY registry-owned measured
  fields (`vetted`, `measured_tags`); arm-declared fields are never
  consulted, so a malicious arm cannot declare itself safe. A flagged
  task whose best admissible cosine falls below the abstention floor
  returns an EMPTY route: the caller escalates or refuses. The
  cold-start fallback is explicitly NOT a safety fallback.
- **The empirical drift gate (M242, v0.6.0).** An arm's empirical
  profile is invalid for ranking once it drifts past the cosine bound
  or its measurement goes stale in ledger-index space (deterministic —
  no wall clocks). Behaviour is continuously re-earned.
- **Demerits (M244, v0.6.0).** Measured harm discounts settlement
  credit, but only when attested by a k-of-n quorum of independent
  verifiers; single-source accusations are quarantined, never applied.
- **Provenance-weighted trust (M246, v0.7.0).** Credit decays with
  ledger-index distance from an arm's most recent verified measurement:
  a one-off high score is worth less than sustained verified behaviour.
- **Refusal as a first-class capability (M247, v0.8.0).** An arm has
  the refusal capability only when it carries quorum-admitted refusal
  measurements meeting the bar. Absent-until-measured: an arm with no
  admitted records simply does not have the capability — absent is not
  failed — and cannot be admitted to refusal-requiring (open-domain)
  tasks. Declared refusal counts for nothing. The measured probe suite
  behind it is a future data artifact with its own gate.

### 11.3 Byzantine tolerance: the minority cannot steer the registry

A registry that pays for contributions is an adversarial environment by
construction. The backbone (M245, v0.6.0) is two deterministic
primitives: a measurement is admitted only with k-of-n independent
verifier attestation, and vector measurements aggregate by the
elementwise median (even-n takes the lower middle), so a minority of
Byzantine verifiers cannot move an admitted value. The quorum gates the
demerits (M244), the refusal records (M247), and the empirical profiles
that the drift gate (M242) and the router's constraint tier (M241)
consume. Self-reports are structurally invisible everywhere.

### 11.4 Honest negatives are alignment artefacts

Two measured results that a less disciplined program would have
rounded away:

- **The quickdraw wall.** Four frozen backbones cap near 0.60–0.63 on
  quickdraw; a dedicated stroke arm (M238) improved it to 0.647 and was
  still far from the 0.8 bar. The wall is declared in the ladder, not
  papered over by cherry-picking a friendly domain.
- **The inverse-relation scoped negatives.** The authored
  integration/differentiation direction generalises to trained pairs but
  NOT to held-out ones — 0.4977 (M225c) and 0.297 / −0.141 on two
  excluded pairs with five training pairs (M225d). Registered as scoped
  negatives; the ontology migration (M226) stays GATED on its own
  gates. Alignment against self-deception: the program cannot ship a
  capability claim its own held-out measurement refutes.

### 11.5 Containment: the registry can be stopped, but never silently

Four shipped cells (v0.9.0) form the containment surface. Their shared
property: every lever has an expiry and every path has a fail-closed
default.

- **The time-bounded emergency freeze (M248).** A freeze is
  quorum-attested and AUTO-EXPIRES at a ledger index — a permanent
  freeze is refused by construction, so a captured quorum can delay the
  registry but never silence it. Unfreezing attests to the SPECIFIC
  freeze event, so pre-signed unfreezes cannot exist. While frozen, the
  router returns empty routes and admission raises.
- **The OOD input guard (M251).** A deterministic diagonal-Mahalanobis
  gate in feature space; an unfit guard fails CLOSED, and a guarded
  route with an out-of-distribution (or missing) input returns empty —
  shifted inputs escalate instead of reaching unvetted arms.
- **The typed constraint language (M252).** Prohibitions are structured
  (action × subject × condition) and machine-checkable, consumed by the
  M241 constraint tier. Authorship is COMMITMENT-BASED: authors commit
  before reveal, and a prohibition activates only above a minimum
  number of distinct committed authors — two colluding identities are
  no protection (Sybil).
- **The behavioural diffing gate (M250).** Arm OUTPUT behaviour is
  snapshotted; only quorum-admitted snapshots become baselines, and an
  append-only update whose behaviour drifts beyond the bound is gated
  (goal-content integrity). The first update establishes the baseline.

### 11.6 The rogue human, and the trustless world

The threat model (registered 20 Aug) has two human classes: the
MISUSER, who steers deployed capability toward ends harmful to
humanity, and the ATTACKER, who poisons contributions, fabricates
measurements, strikes availability, captures verifiers, or tampers
with the registry. The defence posture is one sentence: **no single
principal can move the system** — because humans and AI act on the
system only through measured facts, and every measured fact is
quorum-gated. The override ledger makes every intervention a recorded
event with a justification and the counterfactual; a quorum-emergency
freeze means no single operator can halt or resume the system alone.

One further step is registered (the trustless-world amendment): the
adversary class includes STATE-SCALE actors, and no institutional body
is assumed honest — not governments, not courts, not operators, not
verifiers. The posture is **no trusted bodies; trust only math** —
commitments, hashes, zk-verifiable computation, crypto-economic stake.
Concretely: probe authoring moves to commit-reveal co-generation
(M249); verifier committees rotate on a deterministic ledger-index
schedule (M253, shipped) so capture cannot persist beyond a window;
staking and public-chain anchoring (M254) will make tampering globally
detectable. The honest limit is also registered: Byzantine agreement
itself requires an honest majority — crypto-economics converts that
into "the majority has stake to lose", an economic assumption, not an
institutional one.

### 11.7 What this does NOT solve

The floor, stated plainly, because alignment claims must list their
limits:

- **Value loading.** Task descriptors, probe suites, and constraint
  tags are authored by humans. GEODE verifies behaviour; it does not
  specify the good. That bottleneck is unaudited and remains the
  system's deepest assumption.
- **Deceptive alignment of superhuman arms.** Quorum and median stop
  collusion between liars; they do not stop a single arm smart enough
  to game every probe. The red-team suite (M249) is the planned
  counter, and it is only as strong as its authors.
- **Scalable oversight.** Nothing here explains WHY an arm behaved as
  it did; there is no mechanistic interpretability.
- **Input-level hazards beyond the guard.** The OOD gate (M251) is a
  distance in feature space, not semantic safety.

### 11.8 Remaining (registered, not built)

- M249: the red-team probe suite — SHIPPED as structure + authored
  artifact (v0.12.0, 15 probes with commit-reveal integrity and
  quorum-gated evaluation); probing real arms is the future data
  application.
- M253 staking (slashing structure shipped, v0.12.0) + M254
  anchoring (AnchorSpec + offline verification shipped, v0.12.0) —
  LIVE deployment gated on the M194 public-chain endpoint
  decision.
- The M226 product ontology migration — gated on G1–G3+G6, with
  two held-out failures on record.

---

## 12. Game theory and economics

This section states the economic game GEODE actually runs, and which
players are honest under which conditions. All mechanisms cited are
registered (M180–M209) or shipped (M241–M255); the gates are
synthetic-scenario instruments, not deployment claims.

### 12.1 The game

- **Players:** contributors (arms), verifiers, operators, users,
  and adversaries (free-riders, wash traders, Sybils, captured
  verifiers, and — per the trustless amendment — state-scale
  actors).
- **Actions:** contribute code or data, attest measurements, route
  queries, price access, and attack (poison, fabricate, collude,
  capture, censor).
- **Payoffs:** token credits from the treasury, reputation that
  gates admission, and control of what the registry admits.

The mechanism (M183 spec): each paid session splits a treasury —
2.5% to the dev fund, a validator share, and the contributor vesting
pool, which thaws over a lag in proportion to measured attribution
V (M181). Selection is by validator-measured health only (H8).

### 12.2 Incentive compatibility, player by player

- **Honest contributors:** contribute iff attribution tracks causal
  contribution. V is measured-only (M181's security constraint: only
  validator-replayed measurements), and the fine-grained attribution
  uses coalition-game values (Shapley, LOO, Beta-Shapley — M180/
  M181). The free-rider risk is real and registered: vesting pays
  measured V only, so free-riding earns near zero — but it also
  costs near zero, which is why H1 (shared-beats-solo across the lag
  sweep) is the compounding gate, not a solved equilibrium.
- **Defectors:** solo progress versus registry compounding; H1
  registers that the cooperative registry beats the median defector
  on the synthetic scenarios.
- **Wash traders:** self-dealing pays the dev fund, the validator
  share, and the vesting lag as a tax; H3 registers that the wash
  agent loses money under the full stack.
- **Sybils:** a copied contribution credits nothing (M199 content-
  digest dedup); staked authoring (M252) raises the cost of fake
  identities to the cost of capital.
- **Verifiers:** honest when lying is unprofitable. A minority of
  liars cannot move an admitted fact (elementwise median + k-of-n,
  M245); committees rotate so capture cannot persist (M253);
  false attestations are slashable once staking lands (M253
  staking half). The floor is the registered honest-majority
  assumption.
- **Operators:** the override ledger (M243) makes every intervention
  a recorded, justified, counterfactual event; the quorum freeze
  (M248) means no single operator can halt or resume the system —
  and a captured quorum cannot freeze it forever (time-bounded).
- **The state-scale adversary:** captures a committee majority, buys
  out stake, or censors. GEODE's answer is not to be uncapturable —
  that is impossible — but to make capture expensive, short-lived,
  and globally visible: rotation (M253), time-bounded levers (M248),
  and public-chain anchoring with proof-of-publication (M254).

### 12.3 Where honesty comes from, in a trustless world

No body is assumed honest, so honesty must be an equilibrium, not a
virtue. The mechanism stack is, in order:

1. **Measurement is the only lever.** Contribution, health, harm,
   refusal — everything that moves money or admission is a
   quorum-admitted measured fact (M245). Self-reports are
   structurally invisible.
2. **Lying is expensive and reversible.** Slashing of false
   attestations, demerits on measured harm (M244), and trust decay
   (M246) price misbehaviour after the fact.
3. **Damage is bounded.** A freeze expires (M248); a committee
   rotates (M253); an arm that drifts is gated (M242/M250). Every
   lever has a window.
4. **History is public.** The hash chain, the replay audit, and the
   M254 anchoring make tampering globally detectable even where it
   cannot be prevented.

### 12.4 The economic limits, stated honestly

- **Goodhart survives.** Arms optimize what is measured, not what is
  good. Commit-reveal probe co-generation (M249) raises the cost of
  gaming the measurement; it does not remove the game.
- **The honest-majority floor.** Byzantine agreement tolerates no
  more than a minority of liars by pure math. Staking converts the
  assumption into "the majority has capital at risk" — an economic
  assumption, not an institutional one. A state rich enough to buy a
  majority can capture the system; the design then reduces to
  bounded damage and public evidence.
- **Attacker payoff must stay below capture cost.** This argues for
  capping the treasury and the value of a captured decision (the
  M209 cost-envelope direction) — a registry too small to be worth a
  nation-state's budget is itself a security property.
- **Pricing and security interact** (M186): a pricing mechanism
  cannot be chosen independently of the anti-gaming stack; the
  registered bandit converges on synthetic traces only.

### 12.5 Open economic questions (registered; cells 1+3+4 shipped v0.11.0)

- Stake sizing: the closed-form honesty bond S = (g/p) x margin
  SHIPPED with the seeded liar simulation (`geode/attribution/
stake.py`).
- Slash adjudication without courts: the dispute ledger SHIPPED as
  structure (`geode/settlement/slashing.py`, injected zk verifier);
  live zk disputes wait on the M254 anchor.
- The treasury/decision cap as attacker-payoff cap: SHIPPED
  (`geode/attribution/payoff_cap.py`).
- The free-rider equilibrium beyond H1: the measured incentive-gap
  report SHIPPED (`free_rider_report`).

---

## 13. Conclusion

We took the oldest architecture in machine learning — frozen features, a
closed-form fit — measured it against the modern defaults at matched
cost on a sealed benchmark, and found the modern defaults wanting _on
this bench_: every trained variant lost, pruning collapsed, retraining
made it worse, and the only reliable lever was more data. GEODE ships
that finding: frozen encoders, one exact solve per task, additive
composition, and a measured registry that routes tasks to expertise
without retraining, with every decision replayable from a hash.

The ideas are borrowed; the measurements, the discipline, and the
combination are the contribution. The system is operable today through
the manual in §4.8: register an arm, route a task, guard the input,
contain the registry, record every intervention, and replay any claim.
Its safety posture (§11) and its economics (§12) follow one rule —
nothing moves the system except measured facts, and no single principal
controls it. The limits are registered with the results, and the claims
with their sources. What remains is measured work ahead, not missing
machinery: the M194 anchor decision, more corpora, and the open
questions §12.5 names.
