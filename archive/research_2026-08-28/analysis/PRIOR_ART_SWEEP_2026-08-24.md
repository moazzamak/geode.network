# Prior-Art Sweep — similar work (24 Aug 2026)

Registered in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` ("PRIOR-ART SEARCH
registered") BEFORE the queries ran. Raw instrument output:
`logs/results/prior_art_search_2026-08-24/arxiv_sweep.json` (run 1)
and `arxiv_sweep_run2.json` (run 2, validated). Script:
`tools/prior_art_search.py`.

## 1. Instrument registration and validation

- **Claim under test:** GEODE's assembly — composable registrations
  of frozen artifacts + deterministic replayable verification +
  payment by measured held-out use, settled in native ETH with
  epoch vesting and burn slashing — has no known direct prior.
- **Displacement criteria:** a prior system displaces the assembly
  claim if it has (a) payment by MEASURED held-out utility of
  FROZEN artifacts, (b) deterministic, replayable decisions, and
  (c) native-ETH epoch-vested settlement with burn slashing.
- **Anchors:** Bittensor title query (liveness) and a topic query
  that must surface Bittensor WITHOUT its title (sensitivity).
- **Run 1:** liveness anchor passed (11 hits); the sensitivity
  anchor `all:"decentralized machine learning network"` returned 0
  — the instrument was insensitive to topic phrasing, so every
  empty result was uninterpretable. Repair (registered): two-stage
  phrasing with shorter ANDs, applied UNIFORMLY to all queries
  (never only the empty ones).
- **Run 2:** all 11 queries returned hits; zero residual failures
  (no 429s); the repaired sensitivity anchor
  `all:"blockchain" AND all:"machine learning" AND all:"network"`
  returned 15 hits including "BitTensor: A Peer-to-Peer
  Intelligence Market" (2020). **The instrument is validated
  sensitive.**

## 2. What the sweep found

Nothing found combines all three displacement criteria. The
neighbors cluster on three separate axes:

**Axis 1 — the incentive/marketplace layer (closest overall):**

- BitTensor / Bittensor (2020, whitepaper; plus 2025-2026 empirical
  critiques and subnet studies). Staking-for-emission, mutable
  models, no replay discipline, token emission rather than ETH
  vesting. Already named in the paper as the closest neighbor.
- Golden Grain (2020) — secure decentralized model marketplace
  (MLaaS), Byzantine-fault tolerant via trusted hardware.
- Dropbear (2022) — ML marketplaces with Byzantine model agreement.
- FL-Market (2021) — trading private models in federated learning.
- SAKSHI (2023) — decentralized AI platforms survey/proposal.
- PredictChain (2023) — decentralized blockchain AI collaboration.
- IOTA (2025) — incentivized orchestrated training architecture
  (fine-tuning incentives).
- Incentivizing Permissionless Distributed Learning of LLMs (2025).

**Axis 2 — the verification layer (zkML / verifiable inference):**

- opML (2024), opp/ai (2024), ezDPS (2022), SVIP (2024), TOPLOC
  (2025), TensorCommitments (2026), NanoZK (2026), Jolt Atlas
  (2026), Artemis (2024), Slalom (2018, TEE), surveys (2025).
  These prove an inference was performed; none attaches an
  attribution/market economy to frozen registered artifacts.
- HadAgent (2026) — decentralized agentic serving with
  proof-of-inference blockchain consensus (closest on the
  verification+serving axis; economics not the GEODE form).

**Axis 3 — decentralized serving/infrastructure:**

- POKT Network decentralized LLM inference (2024), Parallax (2025),
  DeServe (2025), DGrid-style networks — cost/latency/permissionless
  serving, not measured-utility attribution.
- "Token Inflation: How Dishonest Providers Can Overcharge for LLM
  Usage" (2026) — metering-fraud critique; directly relevant to
  GEODE's metering claims (recommend reading).

**Live projects (web pass, secondary sources, not audited):**
Bittensor; Gensyn; Cuckoo AI; Swan Inference (Swan Chain);
DEPINfer; DGrid.AI; a GitHub zkSNARK-based decentralized-ML-
inference marketplace prototype (anandpr19); and the known
ecosystem (Ritual, Sentient, Sahara, Morpheus, Prime Intellect,
Lilypad, OpenGradient, Bagel, Nous). All are compute/serving
markets, inference APIs, or token-emission networks. None found
describes frozen-artifact registration with measured held-out
utility, deterministic replay, and ETH epoch-vested burn-slashed
settlement.

## 3. Verdict (per the registered discipline)

| Criterion                                                    | Found anywhere?                                                                                                       |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| (a) payment by measured held-out utility of FROZEN artifacts | No — markets pay for serving/uptime/stake; evaluation-based attribution of frozen artifacts was not found             |
| (b) deterministic, replayable decisions as the audit basis   | Partial — zkML proves single inferences; no system found that makes routing + answers + payments one replayable chain |
| (c) native-ETH epoch vesting + burn slashing                 | No                                                                                                                    |

**Registered consequence:** public unauthenticated search can only
find displacing work; absence from it means nothing. The sweep
found no displacer of the assembly claim, but that is recorded as
"no displacer found by this instrument", NOT as novelty. The
whitepaper's wording ("claims no new algorithm... value is in the
assembly and the discipline") stands; "first" is not claimed
anywhere.

## 4. Follow-ups

- Fold the strongest neighbors into `WHITEPAPER_GEODE.tex` Prior
  Art (candidates: opML, SVIP/TOPLOC, HadAgent, Dropbear, Golden
  Grain) — requires bibitems.
- Read the Bittensor empirical critique (2025) and the
  token-inflation metering paper (2026) against GEODE's metering
  and emission claims.

## 5. Lessons and improvements (applied)

- **Bittensor + its empirical critique (2025) + subnet risk-factor
  study (2026):** concentration, emission-driven gaming, and
  per-subnet risk factors are the documented failure modes of
  staking-for-emission. GEODE's no-stake, no-emission, measured-
  utility rules were already the answer; the critique is now cited
  as the recorded evidence for those choices.
- **Token Inflation (2026):** per-token billing is hard to audit
  when the provider hides model, tokenizer, and execution. GEODE's
  answer is metering from the typed answer plus replay. IMPROVEMENT
  APPLIED: the whitepaper's Metering rule now states that an
  inflated count is a replay-visible deviation on the slash path
  (previously implied only in the design doc).
- **Dropbear (2022) / Golden Grain (2020):** two ways to vouch for
  hosted-model results while keeping weights private — Byzantine
  model agreement, and TEE attestation. GEODE uses neither:
  frozen hashes + replay give the same guarantee without a
  trusted-hardware assumption. Cited as the alternatives GEODE
  deliberately does not need.
- **opML (2024) / SVIP (2024) / HadAgent (2026):** the state of the
  art proves that a computation happened; none attaches the proof
  to an attribution economy over frozen registered artifacts —
  which is the axis GEODE occupies. Cited.
- **POKT / Parallax / DeServe (2024-2025):** decentralized serving
  optimizes cost and latency, not attribution. No change; GEODE's
  host economics remain the distinct axis.
- **Gauntlet/IOTA (2025):** incentivized _training_ of foundation
  models is feasible but pays pseudo-gradients, not measured
  held-out utility. Validates GEODE's narrower choice: pay frozen
  artifacts for measured use, never training contribution.

## 6. Whitepaper changes (this date)

- Prior Art: new bibitems + citations — `bittensor2025critique`,
  `goldengrain2020`, `dropbear2022`, `opml2024`, `svip2024`,
  `hadagent2026`, `tokeninflation2026`.
- Metering rule: inflated-count → slash-path clause added.
