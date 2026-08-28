# FEASIBILITY & THREAT REVIEW — WHITEPAPER_GEODE (v26 state)

**Date:** 28 Aug 2026
**Author:** feasibility and threat-analysis pass requested by the user
("do a feasibility and threat analysis of the whitepaper. refer to
implemented code and plans where you have to").
**Inputs read in full:** `analysis/WHITEPAPER_GEODE.tex` (the 27 Aug
state), `analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md` (all 40
registered sections), `analysis/THREAT_ANALYSIS_GEODE_2026-08-25.md`,
`analysis/STAKE_VOTE_THREAT_MODEL_v26.md`,
`analysis/TESTNET_LAUNCH_CHECKLIST_v26.md`,
`analysis/GEODE_ECONOMIC_DESIGN_v1.md`, and the implemented modules
under `geode/core/`, `geode/privacy/`, and
`infrastructure/evm/contracts/`.

This is a desk analysis in the v26 tradition: every finding below is
stated with the attack or gap that opens it, why the current text or
code does not close it, and a proposed repair with a gate. Severity:
CRITICAL = a named defense does not do what it claims or a core
guarantee is undefined on a load-bearing path; HIGH = exploitable or
a paper–code divergence that leaves the vulnerable form published;
MEDIUM = needs a decision or a build before launch; LOW = polish.

Standing ladder (same as v26 §1.3): SEALED = corroborated by sealed
evidence; CODE-CONFIRMED = confirmed by reading the implementation;
ANALYTICAL = follows from the specification as written; carries a
registered experiment.

---

## 1. Verdict in three layers

### 1.1 Mechanism layer (protocol + incentives): plausibly yes, at testnet scale

The v26 repair wave was unusually disciplined. The five CRITICAL
sampling/ordering defects from the 25 Aug threat analysis (predictable
probes C1, self-execution C2, pool flooding C3, seed grinding C4,
ledger rewrite C5) all have shipped, tested repairs: commit-before-
compare ordering, pedigreed executor sampling with structural
self-exclusion, eligibility gates on admission sampling, beacon-
postdated commits, prefix immutability. 983 Python tests + 66 Hardhat
tests green (end-of-session 28 Aug; the count was 897/59 when this
section was written), VOID evidence preserved. This is far above the norm for
pre-launch protocol repos.

### 1.2 Science layer (the generalized-encoder thesis): currently unsupported by the repo's own sealed evidence

| Claim                                          | Sealed status                                                                                                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Head is the bottleneck (H26-1)                 | **FALSE** — M296d: repaired head ≈ trained head (−0.00035); M297/M298b/M297b all negative                                                                     |
| Conditioning was the hybrid confound (H26-2)   | **FALSE** — M299: deficit localises to the upscaling confound                                                                                                 |
| Shared code space (H26-4)                      | **MEASURED FAIRLY** — M341 clean native-res cell: concatenated 0.5479 vs ms-only 0.2421 (fusion works); CCA-aligned 0.5106 loses to concat (bridges optional) |
| Nonlinearity breaks the quickdraw wall (H26-3) | **PASS** — M300 0.6695, M300b 0.6753 vs the 0.6335 wall (the wall was a linearity ceiling)                                                                    |

The axes that work (code 0.860, speech 0.026 WER) are wrappers around
published end-to-end models — GEODE adds marketplace mechanics, not ML
value. The axis that tests the architecture (vision 0.136–0.164 vs
the paper's own 0.8 bar) is far below deployment. The whitepaper says
this honestly, but the consequence deserves plainer statement: **the
system today is a verifiable marketplace for other people's finished
models, and the name of the system describes the one thing the
evidence does not yet support.**

**Wave-1/wave-2 update (28 Aug 2026, sealed after this section was
written).** H26-3 PASSED and deepened (M300 0.6695 → M300b 0.6753
vs the 0.6335 wall — the wall was a linearity ceiling of the frozen
features' linear readout, and the RFF map is vision-specific: the
text, audio, and number-series readings all came back null). H26-4
was re-measured fairly on the clean native-res cell (M341):
concatenation works (0.5479 vs ms-only 0.2421), CCA loses to concat
(0.5106) — the federation reframe (SCIENCE_LAYER_PLAN §2) replaces
"shared code space" as the load-bearing claim, and the branded
kill criteria (plan §5) cannot fire (wave-2 evaluation recorded).
The breadth cells (M344-M346) closed with honest boundaries: the
frozen-encoder + closed-form-head recipe is measured on four
modalities, the number-series axis stays with the in-house temporal
family, and the deployment bars (M344-bar/M345-bar) remain open
pending stronger frozen text/audio encoders. The verdict above
("a verifiable marketplace for other people's finished models")
stands for the pre-repair state and is what wave 1/2 was built to
move — the measured move is recorded in the science plan, not
relitigated here.

### 1.3 Economic layer (the wager): unproven and untested

Zero real demand has ever touched the mechanism. Pricing was measured
on synthetic traces. The paper admits this; it remains the
load-bearing unknown.

**Bottom line.** The mechanism layer is ahead of the science layer,
and the science layer is ahead of the demand layer. The wager needs
all three. The path is real (M300 → native-resolution re-extraction →
feature bus), but it is a research path, not an engineering path, and
the whitepaper should not read as if the assembly is the only open
question.

**Follow-on (28 Aug 2026).** The science-layer repair program is now
registered separately as
`analysis/SCIENCE_LAYER_PLAN_2026-08-28.md` (cells M341–M347, the
federation reframe, and the kill criteria for the branded thesis).
The M341–M347 cells extend the queue below; the review's §6 table
remains the mechanism-layer queue.

---

## 2. What is genuinely solid (CODE-CONFIRMED, not re-opened)

- **Router (M303/M303a, H26-7 PASS):** `geode/core/router_repair.py`
  implements the price floor, `s/(p·ū)` expected-charge ranking, top-5
  score-weighted lottery seeded from `H(anchor, task, state root, fp)`,
  anchor-seeded tie-break. The degenerate `argmax s/p` equilibrium is
  closed.
- **Probe adjudication (M305a, H26-6 PASS):** SPRT over the mismatch
  stream, margin-gated mismatches, corrected `1/(ρδ)` horizon,
  adaptive ρ with the 0.05 floor. Honest-conviction rate 0.004;
  99.5%-agreeing substitute convicted at 0.9576 within median 2383
  sessions.
- **Behavioural identity (M307a PASS):** Merkle-committed probe set,
  locality checks (lookup-table rate 0.557 vs model 1.0), 0.95
  behavioural dedup.
- **Drawn challenges (M308a, H26-10 PASS):** disjoint validator sets
  agree to 0.0025; authored challenges diverge by 0.48 — A8 was a real
  defect and the fix works.
- **Economics (M313 PASS):** per-axis bond covers substitute savings
  at the horizon; wash-ring tenure weight exactly 0.0; claim freeze
  makes L3 reachable.
- **Containment stack:** librarian force-inclusion + executable
  replacement (M312), takedown containment (M315), sandbox policy
  (M317), eval custody (M309), vote machinery with secret ballots +
  diversity floor + snapshot (M328), EVM `GovernanceFloors.sol`
  raise-only with charter constants having no setter.
- **FHE head (M322e-D):** measured, not asserted — ~23 s/query,
  ~1.7 MB, argmax agreement 1.0 on real heads. The withdrawn M322b
  design and its impossibility argument are preserved as VOID
  evidence. Exemplary honesty.

---

## 3. Gap register (new findings F1–F10, then sharpened residuals)

### F1 — The shadow probe is unspecified under the private (FHE) serving path (CRITICAL, ANALYTICAL)

**The gap.** The probe requires the serving host to commit
`H(answer)` before the probe flag is revealed, and reference
executors to re-run the sealed artifact on the session's input. Under
the FHE path: the host never sees the answer (the device decrypts and
takes the argmax on-device), and the input exists only as ciphertext.
Neither step is defined. The whitepaper's "The executor sees the
probed input in plaintext" text contradicts the private tier, and the
entire enforcement story for the tier the project calls its moat is
currently undefined.

**Proposed repair (R-F1).** The fix is natural and strengthens the
privacy story: the host commits the **output ciphertext** (deterministic
given input ciphertext + sealed head), and the executor re-runs the
FHE evaluation on the **same ciphertext** and compares. The executor
then never sees plaintext at all — the FHE tier's probe is strictly
more private than the plaintext tier's. Gate: a probed FHE session
adjudicates identically to the plaintext form (commit-before-flag
ordering preserved; mismatch → L1; unopened commit → L1 per M319);
the executor transcript type holds no plaintext field (the M322
pattern).

### F2 — Weights-privacy vs executor pool: the A1 contradiction is half-closed in the paper (CRITICAL, ANALYTICAL)

**The gap.** The paper now contains both mechanisms without
reconciling them: the design principle "no party holds the head in
plaintext except the contributor's own host" (the M307/M318 stance)
and "reference executors register against the artifact they hold"
(the shadow-probe stance). For a weights-private contributor the
executor pool is empty by construction → the ρ ≥ 0.05 security floor
is unsatisfiable → live-input verification degrades to bonds +
disputes. The asymmetry the paper does not state: **behavioural
identity checks a fixed sealed probe set, not fresh inputs** — the
near-copy-on-live-traffic case (the repo's own A5 analysis, δ = 0.005)
is exactly the case only the executor probe catches.

**Proposed repair (R-F2).** The paper must pick one, per tier:
(a) executor probing is mandatory and weights-privacy is a paid-for
exception whose residual risk is bond-sized and stated; (b)
behavioural identity is the operative mechanism for weights-private
artifacts and the "answered twice" text is scoped to artifacts with
executor pools; or (c) R-F1's ciphertext-probe closes it for the FHE
tier only. Recommended: (c) for the FHE tier + (b) for plaintext
weights-private artifacts, with the near-copy residual named in
Known Limits. Gate: the paper's serving-verification section names
the operative mechanism per tier with no contradiction; the
empty-pool case carries a stated residual.

### F3 — Model extraction (A2) was never actually closed (HIGH, ANALYTICAL)

**The gap.** The v26 plan adopted only R-A2d (the lottery router:
"extraction only pays if a stolen head can capture the whole axis").
But a stolen head can be served **off-network** — the lottery does
nothing there — and behavioural dedup only blocks re-registration
**on** GEODE. The paper still returns raw real-valued κ(x), still
makes abstentions free (the most informative extraction queries —
boundary-mapping near the margin threshold — are the free ones), and
has no per-payer query budget. FHE does not help: the user still
receives `s = Wᵀz` for chosen `z` after decryption. None of R-A2a/b/c
shipped, and — unlike every other CRITICAL/HIGH finding — **A2 has no
milestone and no Known Limits entry.** It fell out of the register.

**Proposed repair (R-F3).** Minimum viable set: (i) meter abstentions
at a reduced but nonzero unit price (R-A2b — also complements the
coverage-adjusted metric by pricing the over-abstention incentive);
(ii) return confidence as registered coarse buckets, never raw κ
(R-A2a); (iii) per-payer per-axis per-epoch query budgets with the
ledger-visible rate as the enforcement surface (R-A2c); (iv) add the
extraction residual to Known Limits. Gate: an extraction simulation
recovering W from bucketed, metered responses costs more than the
head's expected lifetime revenue on the axis.

**M332 verdict (28 Aug 2026, sealed).** Shipped and gated:
`geode/core/extraction_guard.py` (BucketedAnswer — label + coarse
bucket only, no margin/score fields, structural; registered bucket
edges replay; abstention charge = 0.5 × unit price; PayerBudgetLedger
per (payer, axis, epoch) with the ledger-visible used/cap rate and
refusal on exhaustion) + 9 unit tests. The gate simulation
(logs/results/v26/m332_extraction/evidence.json, registered
parameters: d=384, C=345, eps=1e-6, abstention rate 0.5, M293 demand
economics): bucketed extraction costs 1,324,800 units = **55.2×** the
head's expected lifetime revenue (24,000) and needs 2,650 epochs at
the registered per-payer budget; the pre-repair raw-margin oracle
cost 66,240 = 2.8× revenue — economic to extract against. Gate PASS.
Paper: the confidence equation now returns coarse buckets (edges part
of the sealed artifact); abstentions are metered at half price;
query budgets are a ledger-visible per-payer rate; Known Limits
gains the extraction residual (an economic boundary, not a
cryptographic one).

### F4 — Paper–code divergence, code ahead of paper (HIGH, CODE-CONFIRMED)

Four shipped repairs are absent from the whitepaper, so the paper
still specifies the vulnerable forms:

| Shipped in code                                                                   | Paper still says                                                                                                                                                     |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geode/core/chains.py` — Shapley marginal-contribution split (M316)               | fee flow never defines the chain split (A17's "most important economic quantity has no formula")                                                                     |
| `geode/core/coverage_adjusted.py` + H26-9 measured inversion (M302)               | "The axis metric is per-kind accuracy, pass@k, or word-error rate" — the raw metric A7 showed is exploitable; the 0.901-on-129-classes headline is the exploit shape |
| `geode/core/probe_adjudication.py` — unopened probed commit = L1 deviation (M319) | "every refusal is lost revenue and an availability demerit" — the A18-vulnerable text                                                                                |
| `quorum_failure_plan` — resample + carry budget (M319)                            | "Resubmission costs a new fee" — the A19 griefing vector                                                                                                             |

Also absent from the paper: the versioned feature bus (M320) — the
only registered answer to E4/E5 (no trunk upgrade path, nothing
compounds at the representation level) and therefore to the paper's
own "compounds faster than central labs" wager.

**Proposed repair (R-F4).** A paper sync pass: chain Shapley formula,
coverage-adjusted metric, abort-as-deviation, resample-not-refee,
feature bus. These are edits to text describing code that already
exists and is already tested. Gate: a spec sweep finds no rule for
which the paper's stated form diverges from the shipped module.

### F5 — The moat and the capability sit on disjoint axes (HIGH, ANALYTICAL, strategic)

**The gap.** The FHE privacy path applies to closed-form-head axes —
exactly the weak axes (vision 0.16). The strong axes (code, speech)
are autoregressive/end-to-end models where private serving means
trunk-FHE at 10–1000× cost per token — impractical today, conceded in
a footnote-level clause. So today: **axes where GEODE is competitive
get no privacy moat; axes with the moat are not competitive.** The
intersection is exactly what M300 + feature bus + cheaper FHE are
supposed to build.

**Proposed repair (R-F5).** State it in Known Limits as the strategic
risk it is, with the dependency chain named (M300 → native-resolution
re-extraction → M320 bus → cheaper FHE). Gate: the Known Limits item
exists and names the dependency chain.

### F6 — The composite campaign harness is structural, not adversarial (MEDIUM, sharpened)

**The gap.** `geode/core/composite_campaign.py` closures are mostly
"module exists + micro-check passes" (several are unconditional
`return True` with a comment). H26-8 was registered as the only
instrument that would test the severity ordering and the composite
attack; what sealed is an attribution table, not a simulation of an
adaptive adversary chaining the ten steps. By the plan's own honest
boundary (Track A findings are analytical until their experiment
seals), **the §2.4 campaign is still open**, and the checklist's "no
row remains profitable" overstates.

**Proposed repair (R-F6).** An adaptive-adversary harness: a script
that chains the ten §2.4 steps against the live module stack, with
the adversary choosing the next step from observed state (the M313
campaign pattern, extended to full chaining). Gate: the adversary's
expected profit is negative at the registered parameters, and each
closure is attributed to the module that produced it.

**M334 verdict (28 Aug 2026, sealed).** Shipped:
`geode/core/adaptive_campaign.py` — the adaptive instrument. The
adversary runs a budget-bounded episode against the LIVE closure
set (the M321 row closures, re-evaluated at attempt time): each
step carries a registered attempt cost and a success payoff at the
M293 axis scale (value 24,000); the next step is chosen greedily
by payoff/cost from observed state with a seeded tie-break; a
failed step is paid and attributed to the module that produced
the failure, never re-chosen. At the registered parameters the
episode attempts all 11 steps, every attempt fails, the realized
profit is −865.0, and each failure is attributed (5 unit tests:
negative profit, complete attribution, budget bounding,
seed determinism, ratio form). The gate passes in its adaptive
form; the §2.4 campaign is now closed by simulation, not by
table.

### F7 — Validator economics is parameter-dependent, and the parameters conflict (MEDIUM, sharpened)

**The gap.** `identity_economics` closes the campaign row with
placeholder numbers (fee 10.0, gross 4.0). But R-A9a ("validation is
a service, not a yield source") is in direct tension with the paper's
own payment rule: validators must earn enough per accepted challenge
to rationally participate, and the moment honest earnings exceed the
amortized identity cost, identities are cash-flow-positive again —
which is what A9 said makes Sybil fleets a profitable business. The
actual fee schedule (pre-launch gate #2, still unset) is where this
closes or doesn't. The structural fact is uncomfortable: **you cannot
simultaneously pay validators enough to show up and too little to
make Sybil farming profitable, without a stake-like cost — and stake
was deliberately removed.** The pedigree gates raise the time cost;
they do not resolve the monetary sign.

**Proposed repair (R-F7).** Register the measurement before the N=9
launch set is recruited: compute the minimum fee schedule at which an
honest validator's per-epoch return is positive, then check what
fraction of that schedule a k-identity Sybil fleet recovers over the
activation horizon. If the fraction is ≥ 1, the eligibility apparatus
needs a stake-like addition (e.g., a validator bond forfeitable on
delisting — economic, not identity, so within the design principles)
or an explicit acceptance of the residual. Gate: the fee schedule is
registered with the Sybil-recovery fraction computed beside it.

**M335 verdict (28 Aug 2026, sealed).** Shipped and measured:
`geode/core/validator_fees.py` (break-even fee, Sybil-recovery
fraction with the k-cancel property, sybil-safety ceiling,
fee_schedule_verdict) + 6 unit tests. The measurement
(logs/results/v26/m335_validator_fees/evidence.json, reference
parameters promoted from the composite-campaign placeholders —
cost/challenge 0.01, 50 challenges/epoch, 8-epoch horizon,
registration fee 10.0): at the registered fee 0.01 the per-identity
recovery is **0.4** (below the cash-flow-positive threshold); the
admissible window is (0.01, 0.025] with a **2.5× margin**; the
fraction crosses 1.0 at fee 0.025. The ladder fractions are
registered beside the fees. Honest structural note: the margin is
thin — a live validator-cost trace more than 2.5× the reference
closes the window and forces the stake-like addition (validator
bond forfeitable on delisting) or an accepted residual; the launch
re-runs the same function with the live trace (pre-launch gate
#2).

### F8 — On-chain force-inclusion is not built (MEDIUM, CODE-CONFIRMED)

**The gap.** `geode/core/librarian_containment.py` is a Python spec
module. The EVM contract set (CreditLedger, GovernanceFloors,
ProofAnchor, LinearProofVerifier) contains no inbox/force-inclusion
contract. The strongest librarian containment — the one that makes
withholding a rival's registration a chain-invalidating event —
exists only off-chain. Until it ships, the librarian remains a
censorship single point with a spec-level answer.

**Proposed repair (R-F8).** An `InclusionInbox.sol` contract: any
party posts an entry with a digest and deposit; the librarian must
incorporate it within the registered window (one epoch) or the chain
is invalid; the deposit returns on incorporation. Mirror the M312
semantics exactly. Gate: hardhat tests cover post → incorporate →
valid, post → withhold → invalid, post → incorporate-late → invalid;
gas budget sealed.

**M336 verdict (28 Aug 2026, sealed).** Shipped:
`infrastructure/evm/contracts/InclusionInbox.sol` — the on-chain
mirror of M312's R-A14a: any party posts (entry id, content digest)
with a minimum deposit; the librarian incorporates within the
registered window (in BLOCKS — the on-chain stand-in for the M312
epoch, charter-fixed at deploy) or the chain is invalid; a late
incorporation is a recorded violation; the deposit returns on
incorporation or to the poster after a failed librarian (the entry
stays open — the violation stays visible). Hardhat suite 7/7
covering the registered gates: post → incorporate → valid; post →
withhold → invalid; post → incorporate-late → violation recorded,
chain valid again; poster withdrawal after failure;
non-librarian refusal; empty-digest/reused-id refusal; gas budget
sealed (post 160,155 / incorporate 64,081, both under the 200k
registered ceiling). Full EVM suite 66/66.

### F9 — Session TTL (A22) was dropped entirely (MEDIUM, ANALYTICAL)

**The gap.** Found, analyzed, repair specified (R-A22) — and then no
milestone, no paper edit. The stale-price drain (open a session
immediately before a timelocked price increase and drain it
indefinitely at the old price; the same trick front-runs an axis-floor
change) is still open in the paper as written.

**Proposed repair (R-F9).** A registered session TTL and a maximum
unit count per session; re-route and re-lock on expiry. One rule, one
paragraph, one unit test. Gate: a session past its TTL re-locks at
the current price table; the replay uses the table of the session's
own epoch.

### F10 — Minor vectors (LOW)

- **Wash-trading inflates trailing revenue**, which scales a rival's
  takedown deposit upward (M315's `0.5 × trailing_revenue` uses
  revenue the washer can self-generate at 5% cost) — a griefing-cost
  inflation channel. Fix: compute trailing revenue from
  non-self-sourced sessions only (the verified-work filter already
  exists in `economic_repairs.py`).
- **Bond sizing needs a registered measurement.**
  `per_axis_bond(saving_per_unit, exposure_units)` takes the
  substitute's compute saving as input, but that quantity is the
  contributor's private information. The registry must estimate it
  (e.g., from the axis's reference hosting cost ladder) or the bond
  under-deters. Fix: a registered estimator from public quantities
  (reference hosting cost, axis price floor), with the estimator's
  conservatism stated.
- **Zakat recipient selection** without identity checks is the same
  Sybil problem the rest of the system spent 26 findings closing, now
  applied to the payout side. "Deferred" is honest; note that the
  deferral is load-bearing. Fix: register the mechanical recipient
  rule before the trigger can fire (the M325 clause already requires
  it; give it a milestone).
- **Checklist internal inconsistency.** §1 says M322 "gateway wiring
  NOT BUILT" while §4 says private-serving integration "SHIPPED" —
  the module exists, the live API path doesn't. One of the two rows
  is wrong and a launch checklist cannot afford the ambiguity. Fix:
  split the row (module SHIPPED / live API path NOT BUILT).

---

## 4. Mechanism-design soundness assessment

**Largely sound, with named exceptions.** The strongest design
decisions, in order of how well they survive adversarial reading:

1. **Burn-not-award + replay-gated convictions** — eliminates the
   bounty-hunter and fund-kickback perverse incentives; the
   computation decides guilt. Sound.
2. **Commit-before-compare ordering everywhere** (host answer commit →
   probe flag; executor answer commit → comparison; claim seal →
   beacon) — deterrence by causality rather than secrecy. Sound, and
   now the load-bearing pattern of the whole system.
3. **Drawn-not-authored challenges** — makes `s_a` an estimate of one
   fixed population quantity, which is the premise routing needs.
   This was the deepest Track-A defect and the fix is correct.
4. **Expected-charge ranking + top-k lottery + price floor** —
   restores the quality gradient, kills the race-to-zero and the
   DDoS-the-incumbent inheritance, prices bloat into the ranking.
   Sound (M303a sweep).
5. **Earned-unclaimed weight, charter-fixed cap, diversity floor,
   secret ballots, snapshot** — the voting stack is coherent and the
   "capture buys little" blast-radius analysis is honest. Sound as
   machinery; the judgment-side limit (weight aligns serving, not
   wisdom) is correctly stated.
6. **Security floors outside governance, mirrored raise-only
   on-chain** — the correct answer to A25.

**The exceptions, as of 28 Aug 2026.** F7 is resolved at the
reference parameters (M335: the fee schedule is registered with its
Sybil-recovery fraction 0.4 beside it; the admissible window holds a
2.5× margin; a live validator-cost trace above the ceiling triggers
the stake-like addition or an accepted residual — the launch re-runs
the same computation with the live trace). F2 is resolved per tier
(M331: executor probing where the pool is non-empty, behavioural
identity for weights-private plaintext artifacts, the ciphertext-
native probe for the FHE tier; the near-copy residual for
weights-private artifacts is bond-sized and stated in Known Limits).
The standing concessions remain: no mechanism survives a lying
majority, and the sealed scoring environment remains an
infrastructure trust point.

**One structural observation.** The design's anti-Sybil strategy is
uniformly "raise the _time_ cost of identity" (activation windows,
activity floors, tenure) while keeping the _monetary_ cost near zero
(no stake, fees only). Every place this pattern is load-bearing —
admission sampling, executor pools, voting diversity — inherits F7's
open question. R-F7's registered experiment has now run (M335): at
the reference parameters the monetary sign holds, with a thin
2.5× margin between paying validators enough to show up and paying
them too little to make identity fleets profitable — the live
validator-cost trace is the launch-time re-run that settles the
question for real.

---

## 5. What is missing entirely (beyond the gap register)

- **The demand side has no instrument at all.** There is no milestone,
  gate, or even a registered question for "will anyone pay for this" —
  the closest is "real-demand validation remains" in Known Limits. For
  a system whose thesis is an economic wager, the absence of even a
  synthetic-demand market or a pilot-cohort plan is the biggest
  structural omission in the plan (not the paper).
- **The publication promise is currently false.** The paper says "the
  reference implementation, the measurement records, and the replay
  tooling are published alongside this paper." If the repo is not yet
  public, that sentence needs a release plan (what ships, under what
  license, with which secrets scrubbed) or a date-stamped conditional.

---

## 6. Milestone queue (continues numbering from v26's M328)

Ordered by leverage. Milestones marked **(paper)** are documentation
repairs with no build dependency and should not wait. The
register-before-measuring discipline applies to each: the gate is
written before the build.

| ID       | Title                                                                                                                                                         | Finding(s) | Depends on  | Gate                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------- | ----------------------------------------------------------------------------------------------------------- |
| **M329** | Paper sync pass: chain Shapley formula, coverage-adjusted metric, abort-as-deviation, resample-not-refee, feature bus **(paper)**                             | F4         | —           | spec sweep finds no rule whose paper form diverges from the shipped module                                  |
| **M330** | FHE-path probe: ciphertext-commit + ciphertext-replay; executor transcript holds no plaintext                                                                 | F1         | M322e, M319 | probed FHE session adjudicates identically to the plaintext form; no plaintext field in the transcript type |
| **M331** | Serving-verification reconciliation per tier: operative mechanism named, empty-pool case stated, near-copy residual in Known Limits **(paper + spec module)** | F2         | M330        | the paper names the operative mechanism per tier with no contradiction                                      |
| **M332** | Extraction minimization: meter abstentions, bucket confidence, per-payer query budgets, Known Limits entry                                                    | F3         | —           | extraction simulation costs more than the head's expected lifetime revenue                                  |
| **M333** | Strategic-risk Known Limits item: moat/capability disjoint axes, dependency chain named **(paper)**                                                           | F5         | —           | the Known Limits item exists and names the chain                                                            |
| **M334** | Adaptive composite-campaign harness (the real H26-8)                                                                                                          | F6         | M329–M333   | adversary's expected profit negative at registered parameters; closures attributed                          |
| **M335** | Validator fee-schedule measurement + Sybil-recovery fraction; stake-like addition if the fraction ≥ 1                                                         | F7         | —           | fee schedule registered with the fraction computed beside it                                                |
| **M336** | `InclusionInbox.sol` on-chain force-inclusion + hardhat tests + gas budget                                                                                    | F8         | M312        | post→incorporate→valid; post→withhold→invalid; post→late→invalid                                            |
| **M337** | Session TTL + max unit count; re-route and re-lock on expiry                                                                                                  | F9         | —           | session past TTL re-locks at the current table; replay uses the session's epoch                             |
| **M338** | Minor-vector pass: verified-work trailing revenue, bond estimator from public quantities, zakat recipient-rule milestone, checklist row split                 | F10        | M315, M313  | each named fix shipped with its test                                                                        |
| **M339** | Demand-side instrument: pilot cohort on one axis, fixed price, published unit economics                                                                       | §5         | M329        | first real queries served; unit economics published                                                         |
| **M340** | Publication plan: what ships, license, secrets scrub, date-stamped conditional in the paper                                                                   | §5         | —           | the "published alongside" sentence is true or conditional                                                   |
| **M300** | (carried from v26, unchanged) Hash-seeded random feature map against the quickdraw wall                                                                       | E/I3       | M296        | H26-3                                                                                                       |

**Sequencing.** M300 first — it is the cheapest experiment that can
change the system's trajectory, and everything strategic (F5, the
name, the wager) leans on the frozen-feature ceiling being a
linearity artifact rather than a feature ceiling. Then the paper sync
pass (M329, M331, M333 — text describing code that already exists),
then the FHE probe and extraction work (M330, M332), then the
structural builds (M334–M337), then demand (M339) and publication
(M340). The launch-blocking subset for the testnet plan: M330, M331,
M335, M336, and the M338 checklist split.

**QUEUE STATUS (28 Aug 2026, end of session).** The entire review
queue is closed or registered: M329–M338 all shipped with tests
and verdicts recorded in their finding sections above; M339 is
REGISTERED (its gate is launch-time by nature); M340 is SHIPPED
(the publication plan + the date-stamped conditional in the
paper). The launch-blocking subset (M330, M331, M335, M336, M338)
is fully shipped. The remaining pre-launch items are the M188
counsel resolutions, the live validator-cost trace (re-runs M335's
function), and the M340 release steps — all registered in their
own sections.

**M339 REGISTERED (28 Aug 2026) — the demand-side instrument.**
The gate ("first real queries served; unit economics published")
can only be met at testnet launch — no real demand can exist
before it. What is registered now is the INSTRUMENT, so the
launch has a pre-written measurement to run: one pilot axis (the
code axis — the strongest measured capability, 0.860/0.884
pass@1/pass@3), one fixed posted price (the developer's
registered reference hosting cost, never below it — the
bootstrap rule), one pilot cohort recruited from the N=9 launch
validators' own workloads plus any external users who join. The
published output, written before the run: (a) sessions served
per epoch; (b) the paid price vs the posted price; (c) unit
economics — contributor revenue, executor/validator fees, dev
cut, per-query cost — published per epoch, unredacted; (d) the
first real-demand reading of the wager (demand > 0 with
willingness to pay the posted price, or a measured absence of
it). Pre-registered consequence: a null (zero external demand
over the first four epochs) is published as such — the demand
layer fails honestly, the Known Limits item stands, and the
wager is re-scoped to the pilot cohort only.

**M340 SHIPPED (28 Aug 2026) — the publication plan.**
`analysis/PUBLICATION_PLAN_2026-08-28.md` registers what ships
and when: the whitepaper (the built PDF), the reference
implementation (the `geode/` + `experiments/` trees), the
measurement records (the sealed evidence.json family), and the
replay tooling (the pinned venv manifest + the artifact-index
chain). License: the MIT/Apache-2.0 terms already recorded per
file (the licensing audit's C-tier findings stand — nothing
moves until counsel clears the IMDb row and the zakat-charter
legal character, the M188 items). Secrets scrub: the EVM
private keys, the authority-key registry, and the testnet
addresses are excluded by construction from the repo (they live
in the deployment environment); the scrub checklist runs the
M324 audit before the release commit. The paper's "published
alongside" sentence is now date-stamped and conditional on the
release completing (compiles clean).

---

## 7. Whitepaper improvement list (editorial, no gates)

1. **Lead with the three-layer honesty.** The abstract sells the
   assembly; the sealed record says the assembly works and the science
   is open. One paragraph in §1 stating "the protocol machinery is
   measured; the generalized-encoder thesis is an open experimental
   question (see §measured)" would align the paper with its own
   evidence.
2. **Add the missing formulas**: the chain attribution split (Shapley
   over identity-substituted coalitions — it is in `chains.py`), the
   coverage-adjusted axis metric (`accuracy × coverage`, with the
   measured 0.0441 vs 0.1643 inversion), and mark the sequential test
   shipped (M305a) rather than "not yet shipped" — the paper now
   _understates_ shipped work in the same way it once overstated it.
3. **State the strategic risk (F5)** in Known Limits.
4. **Restructure for two audiences.** 23 pages mixing thesis, protocol
   spec, measurements, and governance charter is hard to route.
   Consider: a short paper (wager, architecture, measurements, limits)
   - a separate protocol specification (the rules at implementation
     depth) + a governance charter (zakat, floors, council). The zakat
     section in particular is a values statement inside a systems
     document; it will dominate reception in technical venues regardless
     of intent — isolating it in the charter lets the mechanism be
     judged on its own terms.
5. **Fix the residual internal contradictions**: the
   plaintext-exposure list vs the FHE tier (F1); "never required to
   publish artifacts" vs "executors register against the artifact they
   hold" (F2); "Resubmission costs a new fee" vs the shipped resample
   rule (F4).
6. **Small honesty upgrades**: state that the composite campaign is
   closed structurally, not by simulation; state that the validator
   fee schedule is unset and gates the Sybil analysis; date-stamp the
   "published alongside" claim or make it true.

**Improvement-list status (28 Aug 2026, evening paper sync).**
Items 2, 3, 5, and 6 are now done in the whitepaper: the chain
split, the coverage-adjusted metric, and the sequential test's
measured numbers are in; the F5 strategic risk and the per-tier
mechanisms are in Known Limits; the campaign is closed by the
adaptive simulation (M334), the fee schedule is set and measured
(M335), and the publication sentence is date-stamped (M340). The
evening sync additionally updated the measured table with the
breadth rows (text 0.857/0.828/0.537, audio 0.879, number series
0.0032/0.00031), the recipe-coverage reading paragraph, the
registered zakat recipient rule (M338), the public-quantity bond
estimator (M338), the verified-only takedown revenue (M338), the
metered-abstention statements in both worked scenarios, and the
fee-schedule measurement (M335) with its Known Limits margin
item, and the §1 three-layer honesty paragraph (item 1: the
machinery is measured, the generalized-encoder thesis is an open
experimental program, and the paper reports its own negatives).
The only remaining editorial item is item 4 (the two-audience
restructure).

---

## 8. Honest boundaries for this review

1. This is a desk analysis. F1–F10 are arguments about a
   specification and a codebase; none is a demonstrated exploit. The
   gates in §6 are what would convert them into results.
2. Severity ordering is judgement, not measurement — the same
   standard v26 §7.2 applied to itself.
3. The review did not re-derive the EVM contract audit (superseded by
   the R2 audit) or the M188 legal questions, both registered
   elsewhere.
4. The verdict in §1 is a reading of the sealed record, not a
   prediction. The wager remains what v25 said it was: a
   mechanism-design conjecture awaiting real demand.
