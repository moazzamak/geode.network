# GEODE/CG-MoE Engineering Track — Plan (registered 12 Aug 2026)

Status: registered before building. The research programme (v19) is fully executed
and sealed (M125/M122/M123/M126 fired; M117/M124 passed; M128 diagnostics executed;
M127 + M129 literature surveys registered). This plan is the ENGINEERING track the
user asked to start: making the GEODE system competitive on **footprint and energy**
at a **modest accuracy cost** — not a research claim.

---

## What was discussed (task inventory)

1. **HTN-style routing** — surveyed (M127, `analysis/HTN_ROUTING_LITERATURE_REVIEW.md`),
   registered as v19 §10 future work. Not on any active plan; referenced below.
2. **Programmatic primitives + hybrid router** — surveyed (M129, evidence
   `logs/results/v16/m129_programmatic_primitives_litsearch/evidence.json`; review
   `analysis/PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`). This is the active
   engineering direction.
3. **Primitive memory (programmatic lookback)** — the "fetch memory from how far
   back" idea; maps to cache/n-gram-style memory. Future phase of the engineering
   track (B4 below).
4. **LLM-of-source / next-token prediction from additive primitives** — the
   engineering-competitive framing (constrained domain, small footprint, low energy).
   Future phase (B4 below). Not now.
5. **v19 closure + paper/thesis build** — the remaining research deliverable (Plan C).

## Guiding constraints (registered)

- **Measurement-only, never novelty.** Everything built here is measured against the
  sealed DomainNet corpus or a defined engineering benchmark; the literature reviews
  (M127, M129) license no novelty claim.
- **Footprint/energy story is the goal**; a modest accuracy cost is acceptable.
- **Learned model is not the default**: it is used only where no programmatic
  primitive exists (fingerprint classification, open-goal routing, fallback).
- **Reuse**: `src/model_fingerprint.py` (InputSpec/OutputSpec), `src/candidate_routing.py`
  (CertifiedTopKRouter), `src/inference_engine.py` (Expert primitives).

---

## Plan A — v19 closure (DONE)

- [x] M128 executed + committed (`75883e9`): §5.6 verdict written (spectrum flat vs
      atoms, eff-rank ~7.8; margins: correct class loses argmax for ~78% of samples).
- [x] M129 survey evidence read → review doc written (`PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`)
      → v19 §9 references + §11 future-work entry added (alongside the HTN note).
- [x] Seal v19: fully resolved (M125/M122/M123/M126 fired; M117/M124 passed).
- [x] Self-contained findings report written (`analysis/RESEARCH_REPORT_v19.md`, `7669b92`).

## Plan B — Engineering track: programmatic primitives + hybrid router (ACTIVE)

Goal: a GEODE-system component library where well-defined computations are
**programmatic primitives** sharing the fingerprint interface with learned
primitives, and a **contract-gated router** dispatches to the cheapest correct
primitive, with a fallback path. Small footprint + low energy; modest accuracy cost.

### B1 — ProgrammaticPrimitive abstraction (`src/programmatic_primitive.py`) ✅ BUILT + TESTED

- A `ProgrammaticPrimitive` class implementing the same surface as learned primitives:
  a fingerprint (`InputSpec`/`OutputSpec` from `model_fingerprint.py`), a `route`
  contract (what input shapes/domains it accepts), and a `score`/`predict` entry point
  wrapping a plain Python/C function.
- Built-in programmatic primitives (pure numpy, no learned parameters):
  - contract checks (shape, dtype, value range, domain membership),
  - math primitives (norm, cosine similarity, arithmetic),
  - transform primitives (the M108 whitener/encoder are already programmatic — expose
    them as primitives with fingerprints).
- Cost model: each primitive reports its cost class (e.g., O(d), O(d²), constant) so
  the router can pick the cheapest that satisfies the contract.

### B2 — ContractGatedRouter (`src/contract_router.py`) ✅ BUILT + TESTED

- Routes an input by fingerprint: match input spec → select the cheapest primitive
  whose contract accepts it → dispatch → validate output spec.
- Out-of-contract inputs: no learned forward pass; route to reject/fallback
  (reuse the `fallback_mask` pattern from `CertifiedTopKRouter`).
- Planner hook: the router exposes a decision log so a symbolic planner (HTN, v19 §10
  future work) can drive it later; today the router is rule-based (closed goals).

### B3 — Registered measurement (protocol-gated) ✅ SEALED (M130, 12 Aug 2026)

Registered config `experiments/configs/v16/m130_contract_gate.json` (N89 notes,
hypothesis before measurement); evidence `logs/results/v16/m130_contract_gate/evidence.json`.

- **H (measured, sealed corpus):** contract-gated routing rejects out-of-contract
  inputs with ZERO learned forward passes while preserving in-contract accuracy.
  **RESULT: PASSED.**
  - Known-value gate: ridge re-fit reproduced the sealed M126 8192 accuracy
    0.2228 exactly (0.222783, |delta| 1.7e-5, t1 0.002) — the learned path through
    the router is the same object as the sealed one.
  - In-contract: all 34500 sealed test rows reach the learned fallback (0 gate
    dispatches); in-contract accuracy through the router = 0.222783 (unchanged).
  - Out-of-contract: 5 corruption classes (nan, inf, wrong_width, out_of_range,
    dtype_int) — 100% dispatched to the programmatic reject gate at CONSTANT cost,
    ZERO learned forward passes (vs 390 G MACs for the learned path over the same
    row count). Reject gate footprint 0 bytes vs 45.2 MB learned weights.
  - Costs reported separately (guard FLOPs vs learned MACs), never wall-clock.
- Registered displacing neighbours: M129 D5 (reject-option / selective
  classification), D6 (energy-aware routing), D1/D4 (rule/retrieval tool
  selection). This measures the same quantities the field measures; no novelty
  claim (N89.5).
- Smoke caught + fixed: the in-contract value envelope must be the FULL streaming
  extent of the sealed codes, not a head block (a head block missed code tails and
  wrongly gate-dispatched real rows).

### B4 — Primitive memory + additive next-token model (ACTIVE 12 Aug 2026)

Two registered engineering milestones, both measurement-only (M129 verdict).

**B4a — ProgrammaticMemory (`src/programmatic_memory.py`)** ✅ BUILT + TESTED — the "fetch memory from
how far back" idea: a fully programmatic, count-based memory with a **window
parameter**, no trained weights, no backprop. API: `register` (ingest a token/event
stream), `counts(context, window)` / `predict_next(context, window)` (return the
observed continuation distribution for the longest matching context up to `window`).
This is the cache/n-gram/variable-order-Markov primitive (M129 D7, classic PPM/cache
LM design). Built + unit tested (9 tests in
`experiments/common/test_v20_programmatic_memory.py`), then measured in B4b.

**B4b — additive next-token model (registered measurement)** ✅ SEALED (M131, 12 Aug 2026)

Registered config `experiments/configs/v16/m131_additive_next_token.json` (N90 notes);
evidence `logs/results/v16/m131_additive_next_token/evidence.json`. Deterministic
seeded DSL corpus (2.31M train / 220k valid / 238k test tokens, vocab 46, split by
seed — zero network, byte-reproducible). Additive model = uniform mixture over orders
1..K of add-alpha-smoothed exact-order counts from `ProgrammaticMemory`.

**H (measured):** test perplexity 5.782 (w1) → 3.971 (w2) → 3.525 (w3) → **3.324 (w4,
best)** → 3.447 (w8); uniform baseline 46. **RESULT: PASSED** — sanity gate not fired
(window8 3.447 ≤ window1×1.10 = 6.36). The honest shape is a **U-curve in the window**:
lookback helps steeply to an optimum (w4), then dilute (w8) as sparse high-order counts
push the mixture toward uniform — but never worse than the unigram baseline.

- **The "how far back" dial has a measured optimum at window 4** for this DSL at
  add-one smoothing. Footprint vs window: 1.8KB (w1) → 241KB (w4) → **27.4MB (w8)**;
  ops/token 18.5 → 71.4 (integer lookups/adds only, no multiply-accumulates).
- Backoff matched-length histogram (test, window 8): 97% of lookups match the FULL
  window (231,378/238,412) — the DSL is highly structured; the memory almost never
  backs off.
- Transformer arm DISCLOSED as not run (display-GPU TDR risk); reported vs uniform
  baseline + window sweep only (N90.6).
- Learnings: (1) on real counts, add-one smoothing makes high-order unseen contexts
  dilute the mixture → the optimum lookback is where marginal signal ≈ smoothing
  noise, not the max window; (2) the marginal cost of orders 5–8 is enormous
  (241KB→27.4MB, 114×) for a _regression_ in ppl (3.32→3.45) — the cost-benefit
  optimum is well below the max window.

### B4b follow-up — matched-footprint frontier (M133, 12 Aug 2026) ✅ SEALED

Registered config `experiments/configs/v16/m133_matched_footprint.json` (N92 notes);
evidence `logs/results/v16/m133_matched_footprint/evidence.json`. Completes the
M131-disclosed transformer arm: six tiny transformers (12k → 7.1M params,
48KB → 28.6MB fp32) trained on the same DSL train split (fixed seed, CPU) and
matched to the five count-model points by nearest footprint.

**RESULT: the trained transformer wins at EVERY matched footprint. No crossover.**

| count point | count ppl | vs transformer | transformer ppl |
| ----------- | --------- | -------------- | --------------- |
| w1 (1.8KB)  | 5.782     | t12k (48KB)    | 2.799           |
| w2 (12KB)   | 3.971     | t12k (48KB)    | 2.799           |
| w3 (57KB)   | 3.525     | t12k (48KB)    | 2.799           |
| w4 (241KB)  | 3.324     | t74k (296KB)   | 2.580           |
| w8 (27.4MB) | 3.447     | t28m (28.6MB)  | 2.730           |

- Even the SMALLEST 48KB transformer (2.799) beats the count model's best (3.324
  at 241KB). The "tiny regime where count models win" does not exist on this DSL.
- Transformer perplexity is U-shaped in size at fixed steps: best at 235k params
  (2.559); larger models (2.1M, 7.1M) are slightly WORSE (2.647, 2.730) — the
  fixed 6000-step budget (~1.3 epochs) undertrains bigger models. A joint
  (params × steps) effect, mirroring the v19 (atoms × data) finding in training
  time rather than footprint.
- Honest conclusion: the additive count model is NOT competitive in quality at
  matched footprint. Its value is the zero-training / integer-only / programmatic /
  KB-scale property — not perplexity. The sequence-capability gap is quantified,
  and the count-model "breakthrough" framing is closed by measurement.

### B4b follow-up — fixed-construction sequence families (M134, 13 Aug 2026) ✅ SEALED

Registered config `experiments/configs/v16/m134_fixed_sequence.json` (N93 notes);
evidence `logs/results/v16/m134_fixed_sequence/evidence.json`. Two no-backprop
families (reservoir + fixed-attention, each with a closed-form ridge readout) added
to the matched-footprint frontier; count + transformer points reused from sealed
M131/M133. A bug in the fixed-attention construction was found and fixed (spurious
broadcast axis in the attention mix → outer product over the batch, 32GB OOM) and
ALL cells re-run.

**RESULT: neither fixed-construction family narrows the gap — they are an order of
magnitude WORSE than even the additive count model at matched footprint.**

| family          | points (ppl @ fp)        | vs count best (3.32) | vs transformer best (2.56) |
| --------------- | ------------------------ | -------------------- | -------------------------- |
| transformer     | 2.56–2.80 @ 48KB–28.6MB  | —                    | —                          |
| count           | 3.32–5.78 @ 1.8KB–27.4MB | —                    | —                          |
| reservoir       | 26.6–29.7 @ 112KB–17.5MB | ~8–9× worse          | ~10× worse                 |
| fixed-attention | 34.8–35.1 @ 91KB–1.2MB   | ~10× worse           | ~13× worse                 |

- The "price of learning" for sequence models on this DSL is quantified: learned
  attention/recurrence/embeddings are worth roughly an order of magnitude in
  perplexity over fixed random constructions.
- Notable: EXPLICIT structure (the count model's exact n-gram counts) beats RANDOM
  structure (reservoir/fixed-attention) by ~10× on a regular grammar — the count
  model is the second-best family, far ahead of both fixed constructions.
- This closes the last open capability question: the in-identity toolkit (fixed
  construction + ridge) cannot approach transformer-level sequence modelling, and
  the gap is not recoverable by swapping the fixed construction for a reservoir or
  fixed attention. The sequence-modelling frontier is now fully measured across
  four families.
- M134 also surfaced a genuine measurement-hygiene bug: a wrong-broadcast matmul
  silently produced plausible-looking numbers (71.8/43.3 in smoke) that a shape
  audit caught only after the 32GB OOM on fa256. Fixed + re-run ALL.

## Plan C — Paper/thesis build ✅ DONE (12 Aug 2026)

- `MS_THESIS_REPORT.tex`: Phase XIV (joint-budget scaling frontier) added to the
  methodology-evolution chapter; `FINAL_RESEARCH_PAPER.tex`: "Recent extension"
  section appended; `BUILD_PAPER.md`: v19 report build path documented.
  Committed `a818b10`.

## v20 status (12 Aug 2026): COMPLETE

All registered v20 milestones are sealed and pushed to `origin/master`:
B1/B2 (programmatic primitives + contract router, 19 tests), B3 (M130
contract-gate measurement PASSED), B4a (programmatic memory, 10 tests), B4b
(M131 additive next-token measurement PASSED), Plan C (paper/thesis build).
The engineering track's footprint/energy story is measured end-to-end: reject
gates cost zero learned forward passes, the memory's "how far back" dial has
a measured optimum, and the write-ups (RESEARCH_REPORT_v19, both literature
reviews) are in place.

---

## Execution order

1. Read M129 evidence → write `analysis/PROGRAMMATIC_PRIMITIVES_LITERATURE_REVIEW.md`.
2. Extend v19 §10 with the programmatic-primitives idea (reference M129 review).
3. Commit + push the plan, review, config, and instrument fix to `origin/master`.
4. Start B1 (ProgrammaticPrimitive) and B2 (ContractGatedRouter) — pure library code.
   [x] B1/B2 built + unit tested (`experiments/common/test_v20_programmatic_router.py`, 19 tests).
5. Register + run B3 measurement on the sealed corpus.
   [x] M130 sealed: contract-gated routing PASSED (zero learned forward passes on
   out-of-contract, in-contract accuracy preserved exactly).
6. B4a: ProgrammaticMemory library + tests + docs.
   [x] Built + tested (`d12c5c6`), pushed.
7. B4b: M131 registered measurement (additive next-token on a constrained DSL).
   [x] Sealed + PASSED (evidence `logs/results/v16/m131_additive_next_token`):
   U-curve in the window, optimum at w4, gate not fired, transformer arm
   disclosed as not run.
8. Plan C: paper/thesis build updates.
   [x] Phase XIV in thesis + recent-extension section in paper + BUILD_PAPER
   v19 path (`a818b10`). v20 COMPLETE.
