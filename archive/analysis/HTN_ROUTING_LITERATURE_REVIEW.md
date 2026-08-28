# HTN-style hierarchical routing — literature review (12 Aug 2026)

Registered search: `experiments/configs/v16/m127_htn_routing_litsearch.json`, evidence at
`logs/results/v16/m127_htn_routing_litsearch/evidence.json` (arXiv + Semantic Scholar,
dated snapshot, NOT replayable; 25 S2 queries rate-limited and recorded as failures).

## The idea under review (registered before search)

Could a lightweight LLM plus an HTN-style task vocabulary and methods (tasks with input
requirements and effects that decompose into subtasks, chained into a plan) drive
hierarchical expert routing in the GEODE/CG-MoE router — i.e., the routing decision is an
HTN plan whose tasks are specialist models, and the LLM maintains the vocabulary/methods?

## What the search found

**Instrument check:** all 5 anchors retrieved (classic HTN, SHOP2, LLM routing, ReAct,
hierarchical MoE) — the search surface is live. 496 unique titles across 6 query families
(80–104 hits per family).

### 1. Classic HTN planning (the formalism the idea names) — fully established

SHOP2 (Nau et al.), HTN Acting (2018), HDDL 2.1, the IPC HTN track, HATP (robotics),
HTN-to-STRIPS encodings, HTN with preferences/temporal planning. A task vocabulary with
methods that decompose tasks into subtasks is a 50-year-old formalism. Nothing new there.

### 2. LLM × HTN hybrids — actively being built (2025–2026), closest neighbours

- **ChatHTN** (arXiv:2505.11814, 2025, Muñoz-Avila et al.): interleaves symbolic HTN
  planning with ChatGPT-generated task decompositions; provably sound.
- **Online Learning of HTN Methods for integrated LLM-HTN Planning** (arXiv:2511.12901,
  2025): learns _generalized HTN methods_ from LLM decompositions (beyond memoization),
  reducing LLM calls. This is exactly "the LLM maintains the method library" — but for
  action planning domains, not perception-model dispatch.
- **HTN Planning with LLM-Generated Heuristics** (arXiv:2605.07707, 2026, Meneguzzi et
  al.): LLMs generate search heuristics for HTN planners (extends Corrêa-Pereira-Seipp
  2025 from classical to HTN).
- **Automatically Learning HTN Methods from Landmarks** (arXiv:2404.06325, 2024):
  symbolic (non-LLM) method learning.

### 3. LLM routing / model selection — established, mostly flat, LLM-to-LLM

RouteLLM family, FrugalGPT/cascades, **Neural Bandit Optimal LLM Selection for a Pipeline
of Subtasks** (arXiv:2508.09958, 2025 — selects a _sequence_ of LLMs, one per subtask,
with outputs feeding downstream: the "chain of tasks" intuition, learned bandit not HTN),
HAPS (hierarchical LLM routing), LogRouter (two-level LLM routing), Select-then-Solve
(paradigm routing for agents), Routing with Generated Data (LLM skill estimation +
expert selection).

### 4. Hierarchical / task-guided MoE routing — established, neural gates

THOR-MoE (hierarchical task-guided routing for NMT), HI-MoE, HDMoLE, HAMoBE, CBDES MoE,
hierarchical CoT+MoE, Multi-level goal decomposition (neuro-symbolic planners: ADaPT,
Fast and Accurate Task Planning). These use _learned_ gates or LLM planners for _agent
action_, not HTN-structured dispatch over specialist perception models.

## Gap analysis (vs the specific idea)

Every component exists separately; the field is converging on the intersection fast.
What this search did NOT find is the exact combination: **an HTN-structured inference
router for a non-LLM specialist/expert perception system** — tasks = specialist models
with declared input/output contracts (preconditions/effects), the LLM maintaining the
vocabulary/methods **offline**, and the plan executed **symbolically and cheaply** at
inference. The LLM×HTN work targets action planning; the routing work is flat/learned
and LLM-to-LLM; the hierarchical-MoE work is neural.

CAVEAT (per programme rules): absence from arXiv/S2 is UNRESOLVED, not "first". The
general concept ("LLM hierarchical planning for routing") is clearly not novel; the
distinctive angle is narrow and the field moves fast. No novelty claim may be made.

## Worth pursuing? — Yes, as a MEASUREMENT milestone (never novelty)

The programme's niche is sealed measurement. The idea yields a testable hypothesis on
the sealed DomainNet corpus:

- **H:** HTN-structured dispatch (a plan over specialist models, preconditions/effects,
  executed symbolically) beats the flat router (candidate top-k / per-domain specialists)
  on robustness and/or cost for multi-domain, multi-class, and out-of-contract inputs.
- **Design constraints to preserve the programme's cost story:** the lightweight LLM must
  work OFFLINE (build/maintain the method library, as ChatHTN's method learning and the
  GEODE semantic_router already do with cached descriptors) — never per-input. Inference
  is a symbolic plan lookup over a small method library.
- **Honest scope:** HTN routing can improve allocation/robustness, NOT the per-expert
  accuracy ceiling (sealed: sparse ~0.22–0.32/domain vs dense 0.54). Expected payoff is
  routing efficiency and graceful degradation, modest in absolute terms.
- **Registered displacing neighbours for the experiment:** ChatHTN, online LLM-HTN method
  learning, LLM-generated HTN heuristics, neural bandit subtask selection, RouteLLM/
  cascades, THOR-MoE, HAPS, LogRouter.

## Search evidence summary

- Anchors: 180 hits across sources, instrument live.
- Family hits: D1 104, D2 86, D3 94, D4 80, D5 88, D6 96 (arXiv-dominant; S2 heavily
  rate-limited, 25 failures recorded, not counted as empty).
