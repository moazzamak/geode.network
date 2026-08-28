# PRE-PUBLICATION FEASIBILITY & THREAT RE-REVIEW — 28 Aug 2026

Status: written as part of the public-release due diligence. It
re-adjudicates the three-layer verdict of
`FEASIBILITY_THREAT_REVIEW_2026-08-28.md` (this folder) against the
final sealed evidence (M300b, M341, M344–M346, M305a, M332, M335,
M336, M334) and against the prior-art findings of 28 Aug (COMET
ICLR 2025, CoMET arXiv 2605.20674, Hash Layers). It does not replace
the earlier review; it sharpens it for what a reader of the public
whitepaper can now verify.

---

## 1. Verdicts, re-adjudicated

### 1.1 Learning/composition thesis — FEASIBLE AS A PROGRAM, NOT PROVEN AS A CLAIM

- **Concatenation of frozen codes carries signal (M341, sealed):**
  on the clean native-resolution cell, the fused design reads
  0.5479 vs the single encoder's 0.2421, and CCA alignment loses to
  raw concatenation (0.5106). Fusion works; bridges are optional.
  This is the load-bearing measured premise of the composed-codes
  architecture.
- **The linearity ceiling is a readout ceiling, not a feature
  ceiling (M300b, sealed):** the hash-seeded random-feature map
  lifts the quickdraw wall from 0.6335 to 0.6753, while lifting
  none of text (M344), audio (M345), or number-series (M346).
  The repair is axis-specific, not universal — reported as such.
- **Breadth is real but bounded (M344–M346, sealed):** the
  frozen-encoder-plus-closed-form-head recipe is measured on four
  modalities with honest negatives (text nulls under the map; the
  Gramian series-to-image bridge fails at 4.144 NRMSFE against
  0.00317). The wrapper-axis problem is bounded, not solved.
- **Prior-art position after today's sweep:** fixed routing was
  already established (Hash Layers, NeurIPS 2021; COMET, ICLR
  2025) and the frozen-concatenation recipe was independently
  published with a tabular-foundation-model head (CoMET, arXiv
  2605.20674, May 2026). The whitepaper now attributes both. What
  remains unclaimed by any located neighbor: the registry of
  independent frozen artifacts with versioned code-bus manifests
  and payment by measured downstream use. Verdict: the claim is
  narrow, honest, and survives contact with the neighbors found
  to date; it remains an absence claim, recorded as absence.
- **Open cells that could still falsify:** M342 federation
  reframe, M343 adapter registration, M347 scoped-serving product
  cell, M344-bar/M345-bar deployment bars, and the newly
  registered M348 (task-relation arithmetic) — all pre-registered
  with gates; none is load-bearing for the mechanism layer.

### 1.2 Network as a whole — PLAUSIBLE AT TESTNET SCALE, UNPROVEN ECONOMICALLY

- **Shipped and tested:** 963/963 Python tests and 59/59 EVM
  harness tests green (28 Aug). The registry, router, ledger,
  settlement, admission, probes, and governance rules are code
  with sealed tests.
- **Measured attack economics:** the bucketed-confidence oracle
  prices head extraction at 55.2× expected lifetime revenue
  (M332); the sequential mismatch test convicts a 99.5%-agreeing
  substitute at 0.958 within a median 2383 sessions while falsely
  convicting an honest host at 0.004 (M305a); validator Sybil
  recovery is 0.4 of registration cost with a 2.5× admissible
  margin (M335); settlement gas is measured (M336); the
  adversarial scenario sweep closes the copycat race with a
  measured −865 profit for the copycat (M334).
- **Unproven:** real-demand pricing, validator economics under a
  live cost trace (M335's live re-run), the demand pilot (M339),
  and everything behind the M188 counsel gate. The economic
  mechanism is a conjecture under test; the paper says so.

### 1.3 The wager — UNPROVEN, AS STATED

The wager (collaboration beats secrecy) has no evidence yet.
Nothing in the public material claims otherwise; the paper states
the wager as a wager.

---

## 2. Learning/composition feasibility in detail

| Question | Evidence | Status |
| --- | --- | --- |
| Does concatenating frozen codes beat single encoders? | M341: 0.5479 vs 0.2421 | MEASURED YES (one clean cell) |
| Do alignment bridges beat concatenation? | M341: CCA 0.5106 < 0.5479 | MEASURED NO — bridges optional |
| Is the linearity ceiling repairable? | M300b: 0.6335 → 0.6753 | MEASURED YES on vision only |
| Does the repair generalize? | M344 text null, M345 −0.039, M346 bridge 4.144 | MEASURED NO — bounded |
| Does the recipe cover modalities? | text 0.857/0.828, audio 0.879, vision scoped 0.901, series 0.0032 | MEASURED, with negatives |
| Is the frozen-concat recipe novel? | CoMET arXiv 2605.20674 (TFM head) | NO — attributed |
| Is fixed routing novel? | Hash Layers; COMET ICLR 2025 | NO — attributed |
| Is the registry+bus+payment combination claimed elsewhere? | two registered sweeps; none found | ABSENCE RECORDED AS ABSENCE |

**Feasibility conclusion for the learning/composing approach:**
the composition mechanism works at single-corpus scale and the
paper's claim is now correctly scoped to it. The main honesty risk
was overclaiming novelty in subcomponents (fixed routing,
frozen-concat recipe); both are now attributed in the paper. The
remaining risk is scale: multi-corpus, multi-party composition is
unmeasured, and the paper says so.

---

## 3. Network feasibility in detail

**What a launch must still close (unchanged from the queue):**

- M339 demand pilot (real demand before real money at scale).
- M335 live validator-cost trace (the Sybil window is
  parameter-dependent; a live trace above the ceiling closes it).
- M188 counsel (token classification; privacy obligations).
- M318 Pedersen-vs-publish key decision.
- N=9 validator recruitment and the librarian key ceremony.
- R-A7b temperature/ECE measurement (registered pending).
- M344-bar/M345-bar deployment bars (pending stronger frozen
  encoders; the bars are open, not failed).

**Feasibility conclusion:** the mechanism layer is buildable and
tested in the small; the economic layer is untested in the wild.
Nothing in the whitepaper claims otherwise. The whitepaper is
launch-safe on this axis.

---

## 4. Threat vector register (fresh pass)

| Vector | Defense (shipped) | Residual (stated in paper) |
| --- | --- | --- |
| Copycat / re-registration | dedup by artifact hash + behavioural signature (0.95) | fresh-address copy is an identity-bound limit |
| Copycat pricing race | per-axis price floor, timelock, M334 sweep (−865) | predatory cycles slowed, not eliminated |
| Wash trading | 2.5% dock twice per loop, held-out scoring | none claimed beyond cost |
| Sybil validators | activation window, activity floor, tenure, M335 0.4 recovery | economic barrier, not a proof |
| Probing the sealed corpus | aggregate-only verdicts, 4-digit precision, fees, rotation | cost barrier, not a proof |
| Head extraction | bucketed confidence, budgets, lottery router (M332 55.2×) | payer indifferent to cost can still extract |
| Substituted serving (crude) | shadow probe, commit-before-reveal, 1/(ρδ) horizon | near-copy case is the real residual |
| Substituted serving (near-copy) | sequential mismatch test (M305a 0.958 / 0.004) | measured, not perfect |
| Executor–host collusion | sampled judges, commit-before-compare, both slashed | corrupt fraction of the pool raised to k_e |
| Validator collusion / wrong labels | audits, wrong-label exclusion, pre-reveal collusion proof | corrupt majority outside mechanism reach |
| Librarian rewrite | anchored-prefix immutability, fork rule, replacement | cadence is the rewrite window |
| Librarian withholding | force-inclusion queue (on-chain inbox) | liveness statistics make silence visible |
| Governance capture | earned-weight-only votes, 20% cap, no premine, d distinct identities | genesis is concentrated; path out is automatic |
| Coercion / censorship demands | no selection surface, artifact-scoped fixed-effect freezes | frontend operators remain compellable |
| Forged authority orders | multi-channel pinning, nexus gate | forgery not impossible; effect capped at freeze |
| Input/code inversion | no-retention contract covers codes; FHE tier | contract breach and inversion risk named |
| Malicious sealed trunk | sameness certified, artifact itself not audited | largest blast radius; stated, not hidden |
| Privacy-vs-capability moat gap | separate claims; dependency chain registered | moat and capability disjoint today |
| Front-running price changes | timelock, epoch-boundary effect, session lock | re-lock path bounded |
| Budget exhaustion / spamming | per-payer per-axis per-epoch query budgets | used-over-cap rate is the visible surface |
| Griefing by silence | quorum over responders, resample, demerits | victim pays no second fee by design |
| False disputes | bounded deposit, loser pays replay, burn | false claim has no upside |

**Threat conclusion:** every vector has a shipped defense and a
named residual; the largest residual is the malicious-sealed-trunk
case, which the paper already calls out. No new unhandled vector
was found in this pass.

---

## 5. Findings from today's audit (N-series)

- **N1 (citation, HIGH, FIXED):** the CoMET citation had no
  authors or ID and cited the acronym expansion as a title. The
  real paper is Bergström, Mehrotra, Krishnan, "Modular Multimodal
  Classification Without Fine-Tuning: A Simple Compositional
  Approach", arXiv 2605.20674 (2026). Fixed in the whitepaper
  bibliography and prose; the frozen-concat recipe is now
  attributed in the measured-fusion paragraph.
- **N2 (attribution, HIGH, FIXED):** the router section implied
  the learned-gate-to-fixed-rule replacement was GEODE's. Fixed:
  Hash Layers (Roller et al., NeurIPS 2021) and COMET (Shaier et
  al., ICLR 2025) are cited; the paper now states what GEODE adds
  (public, replayable, quality-per-charge ranking of independent
  frozen artifacts).
- **N3 (bibliography, MEDIUM, FIXED):** AdapterHub had a
  mismatched author list and Bittensor had wrong authors and year.
  Both corrected against arXiv records (2007.07779, 2003.03917).
- **N4 (process, LOW, REGISTERED):** several neighbor citations
  (CoMET 2026, HadAgent, Token Inflation, opML, SVIP, Golden
  Grain, Dropbear) are absent from the litsearch cache — they were
  verified directly today via arXiv. CoMET was published 20 May
  2026 — before the M341 clean cell (28 Aug 2026) — and was
  located only during this publication audit; the paper no longer
  describes the frozen-concatenation recipe as independently
  discovered. The two registered sweeps'
  conclusions are unaffected (their claims concern the
  combination, which none of these papers has), but the cache gap
  is recorded so future sweeps re-run through the instrument.
- **N5 (registration, INFO):** M348 (task-relation arithmetic,
  code-space v2) registered as an unsequenced later milestone with
  pre-registered constraints; the whitepaper claims no
  task-relation property and should not until M348 measures one.

---

## 6. Whitepaper claim audit (what a hostile reader will check)

| Claim in the paper | Backing | Risk |
| --- | --- | --- |
| "No new learning algorithm; parts old and named" | true and now fully attributed | low |
| "Claim: the composed-codes architecture" | M341 + bus design + two sweeps | medium — absence claim, hedged |
| "Payment follows measured work" | settlement code + tests | low (code), high (live economics) |
| "Frozen, replayable, deterministic" | sealed tests, replay oracle | low |
| "Fusion works, bridges optional" | M341 0.5479/0.5106 | low |
| "Repair is not universal" | M344–M346 negatives published | low |
| "Extraction costs 55× lifetime revenue" | M332 registered simulation | medium — model-dependent |
| "Probe convicts 99.5% substitute at 0.958" | M305a sealed | low |
| "No system combines the three" | two sweeps | medium — absence claim, hedged |
| "The mechanism is a conjecture under test" | stated repeatedly | low |

---

## 7. Honest boundaries of this review

- All measured numbers cited here are the sealed values in the
  archived evidence; none was re-measured for this review.
- The prior-art position is bounded by two registered sweeps and
  today's direct verifications; it is an absence claim, never a
  first claim.
- Feasibility for the mechanism layer is code-level; feasibility
  for the economic layer is simulation-level; neither is
  live-network evidence, and none is claimed to be.
