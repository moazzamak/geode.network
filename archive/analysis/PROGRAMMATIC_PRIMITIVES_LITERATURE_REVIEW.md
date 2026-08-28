# Programmatic primitives + hybrid router — literature review (12 Aug 2026)

Registered search: `experiments/configs/v16/m129_programmatic_primitives_litsearch.json`,
evidence at `logs/results/v16/m129_programmatic_primitives_litsearch/evidence.json`
(arXiv + Semantic Scholar, dated snapshot, NOT replayable; 35 S2 queries rate-limited
and recorded as failures — never as empty).

## The idea under review (registered before search)

A GEODE-style system where well-defined computations (math, transforms, shape/range
checks) are **programmatic primitives** sharing the same fingerprint interface as
learned primitives; a **contract-gated router** (fingerprint match → cheapest correct
primitive) dispatches inputs, with a fallback path; a learned model is used only where
no programmatic primitive exists. Goal: win on footprint and energy at a modest
accuracy cost. This is an ENGINEERING build, not a research claim.

## What the search found

**Instrument check:** anchors retrieved (145 hits across sources; all 6 anchors found
on at least one source) — search surface live. 714 hits across 8 families (48–138 per
family, arXiv-dominant).

### 1. Tool-augmented models (D1) — established, "model decides when to invoke"

Toolformer-line and beyond: **ART** (automatic multi-step reasoning + tool use),
**MuMath-Code**, **Large Language Models as Tool Makers**, **PORTS** (preference-
optimized retrieval for tool selection), **RaTA-Tool** (retrieval-based tool
selection with multimodal LLMs), "Integrating External Tools with LLMs to Improve
Accuracy". The "give the model a calculator/code interpreter for what it does badly"
principle is fully established — every major platform does it.

### 2. Neuro-symbolic hybrids (D2) — established, heterogeneous

NeSyCoCo (neuro-symbolic concept composer), Gram-Space (codebook compression for
memory-efficient neuro-symbolic AI), neuro-symbolic generative diffusion, generalizable
neuro-symbolic QA. Composition of learned + symbolic modules is a well-populated field;
none of these is an additive-sparse GEODE-style specialist system.

### 3. Interface contracts (D3) — actively standardized, mostly for LLM function calling

**Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling**,
OmniStruct (universal text-to-structure across schemas), GhostShell, NTILC (neural tool
invocation via learned compression), function-calling data synthesis/auditing. The
field's answer to the interface-contract problem is **typed schemas** (JSON-schema-style
tool specs). Directly applicable: the GEODE `InputSpec`/`OutputSpec` fingerprint _is_
such a schema; the "protocol-agnostic" lesson says the programmatic primitive should
expose the same protocol as the learned primitive.

### 4. Planner/controller selecting modules (D4) — established, LLM-at-inference

**Chameleon** (plug-and-play compositional reasoning: LLM picks and composes tools/
modules per input), **HYDRA** (hyper agent for dynamic compositional visual reasoning),
Describe-Explain-Plan-Select, MapAgent (hierarchical agent with dynamic tool
integration), CORAL. The SOTA router for open goals is an LLM acting as controller at
inference — which costs exactly what the footprint story forbids. The HTN/M127 finding
repeats: keep the LLM OFFLINE (build the method/vocabulary library once), execute
symbolically at inference.

### 5. Fallback / abstention / cascade (D5) — mature, directly usable

Reject-option / selective classification (Geifman–El-Yaniv **SelectiveNet**, budgeted
classification with rejection, AUC-based selective classification), **CascadeDebate**
(cost-aware LLM cascades), uncertainty-driven rejection surveys. The "what happens when
the selected module is wrong or out of contract" problem has a 20-year mature answer:
**a confidence/reject threshold + a cascade/fallback path**. For GEODE this maps onto
the existing `fallback_mask` pattern with a threshold on the fused SDF score.

### 6. Energy/footprint-aware routing (D6) — an active measured goal

Stable-MoE / SpaceMoE / SiftMoE (similarity-aware energy-efficient expert selection),
"Efficient MoE LLM Inference with Apple Silicon NPUs", a "Survey on Inference
Optimization for MoE", StructuredDNA (energy-aware transformer routing), fixed/
biologically-inspired routing ("More Experts Than Galaxies"). The field measures
energy-aware dispatch as a first-class objective — consistent with GEODE's matched-cost
discipline.

### 7. Skill/primitive libraries + task planner (D7) — established robotics pattern

Uni-Skill (self-evolving skill repository), Logic-Skill Programming (optimization-based
sequential skill planning), Plan-Seq-Learn, Being-0 (VLM + modular skills), task-and-
motion planning with skill libraries. "A library of reusable primitives composed by a
planner" is the standard robotics architecture; the GEODE angle (perception specialists
as primitives) is where M127's HTN survey already landed.

### 8. Program synthesis / DSL / code-as-primitive (D8) — established

MathDSL, "Synthesis of Mathematical Programs from Natural Language", "Natural Language
Commanding via Program Synthesis", BRIDGE (domain-guided synthesis), multi-modal
program inference (pre-trained LMs + component-based synthesis). Turning a spec into an
executable programmatic module is a mature field — relevant if primitives are ever
_synthesized_ rather than hand-written.

## Gap analysis (vs the specific idea)

Every component exists separately and the field is actively converging on hybrids:

- tool use (D1) ✓, neuro-symbolic composition (D2) ✓, typed tool schemas (D3) ✓,
  LLM-as-controller (D4) ✓, reject/cascade fallback (D5) ✓, energy-aware routing (D6) ✓,
  skill libraries + planners (D7) ✓, program synthesis (D8) ✓.

What the search did NOT find is the exact combination: **a non-LLM specialist system
(GEODE-style sparse primitives) where programmatic primitives share the fingerprint
interface with learned primitives, a RULE-BASED contract-gated router (no LLM at
inference) dispatches by contract to the cheapest correct primitive, and a reject/
cascade fallback handles out-of-contract inputs — measured for footprint and energy.**
The D4 work routes with an LLM at inference (the opposite of the footprint goal); the
D6 work is about MoE token routing, not programmatic-vs-learned dispatch; the D3 work
is schemas for LLM function calling, not a unified protocol for learned + programmatic
perception primitives.

CAVEAT (per programme rules): absence from arXiv/S2 is UNRESOLVED, not "first". The
concept ("hybrid neuro-symbolic system with tools") is clearly not novel. Only the
narrow packaging — a GEODE-specific, no-LLM-at-inference, contract-gated dispatch with
matched-cost measurement — is distinctive, and that distinctiveness is an engineering
choice, not a novelty claim.

## Verdict: worth building as ENGINEERING + MEASUREMENT (never novelty)

The literature supplies the techniques for each issue raised in the plan:

| Issue                | Literature technique (source family)                                         | GEODE application                                                                                                 |
| -------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Interface contract   | Typed schemas, protocol-agnostic tool integration (D3)                       | `InputSpec`/`OutputSpec` as the contract; `ProgrammaticPrimitive` exposes the same protocol as learned primitives |
| Router, closed goals | Rule/retrieval-based dispatch; LLM OFFLINE for method maintenance (D4, M127) | `ContractGatedRouter`: fingerprint match → cheapest contract-accepting primitive                                  |
| Router, open goals   | LLM-as-controller exists but costs an LLM at inference (D4)                  | Registered boundary: only if a cheap-enough controller exists; otherwise keep out                                 |
| Fallback             | Reject option / selective classification / cascades (D5)                     | Threshold on fused SDF score + reuse `fallback_mask`; programmatic reject = zero forward passes                   |
| Selection function   | Retrieval-based tool selection (PORTS, RaTA-Tool, D1)                        | Router can be retrieval/rule-based, not a learned gate                                                            |
| Energy measurement   | Energy-aware dispatch as measured goal (D6)                                  | Matched-cost measurement on the sealed corpus (per-input MACs, forward-pass count)                                |

## Search evidence summary

- Anchors: 145 hits, all 6 retrieved, instrument live.
- Family hits: D1 60, D2 108, D3 102, D4 48, D5 62, D6 86, D7 138, D8 110.
- S2 rate-limited 35 times; all recorded as failures (failed_queries), none as empty.
