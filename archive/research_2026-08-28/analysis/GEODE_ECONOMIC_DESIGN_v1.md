# GEODE Economic Design — consolidated decisions (v1, 24 Aug 2026)

Consolidates the 24 Aug design discussion into one register. The
living source of truth remains `RESEARCH_IMPLEMENTATION_PLAN_v25.md`
(execution log); this file is the readable design statement. Nothing
here authorizes deployment: M194/M188/M190 remain user-gated, and the
contract rework must re-pass the harness + ethskills audit before any
chain exposure.

## 0. State of the world

- No token exists; nothing has ever minted off the local Hardhat
  harness; nothing is deployed to any testnet or mainnet.
- The system is tokenless-first: settlement in **native ETH at
  market rate** (the chain decision is FINALIZED, 24 Aug — no WETH;
  ETH is native on both the settlement chain and the anchor chain).
- A GEODE token may launch later ONLY if governance requires it
  (registered C9), gated on M188 counsel classification.

## 1. Settlement currency

- **Asset:** native ETH. `deposit()` payable with `msg.value`;
  payouts are guarded `call{value:}` pulls. Forced-ETH is excluded
  from solvency via an internal `ethHeld` counter (never read raw
  `address(this).balance`). No WETH: ETH is native on the
  settlement chain (Arbitrum One) and the anchor chain (Ethereum),
  so payment and claim stay on one asset with no wrapper.
- **Pricing (user decision, 24 Aug) — market-driven:** each
  contributor sets their own price, in ETH, per unit of work
  (query / audio-second / token / execution attempt), at
  registration, posted next to the measured scores — the same
  price field for an arm (an inference rate) and a primitive (a
  per-execution rate). **The unit of work is derived from the
  task descriptor (24 Aug, updated):** the unit is a function of
  the (input type, output type) PAIR — not either alone, not a
  free field. Registered rows: (image | text, class label) and
  (number series, number) → query; (audio, transcript) → audio
  second (input duration); (text, transcript) → token; (text,
  audio) → audio second (output duration); primitives →
  execution attempt. A pair without a row is not an admissible
  task until the table is extended (a registered rule change). No
  registration chooses its unit, so every price on an axis is in
  one unit (router efficiency comparisons are within one unit),
  and a different descriptor — unit included — is a different
  task (content-hashed). No central band — hosts
  price their
  own energy, hardware, and running costs; the market aggregates
  local knowledge no central cost model can. **Price changes are
  timelocked with a notice period** (the royalty pattern) — kills
  bait-and-switch and bounds router churn. A USD equivalent is
  display-only in the UI; no oracle in the settlement path. The
  M186 study's scope narrows to margin-discovery alternatives
  (auction/bandit) on top of this default.
  - **Update mechanics:** a price change is announced at least one
    epoch ahead and takes effect at an epoch boundary. The next
    epoch's price table is a hash-chained ledger entry, so replay
    uses the exact table of its day.
  - **Session lock:** a session pays the price posted when it was
    routed; no mid-session re-pricing. Arm revenue = units served ×
    locked price × 97.5% (the dev dock applied).
  - **Metering:** units are derived from the typed answer (tokens
    generated, audio seconds, attempts used) — deterministic and
    replay-verifiable; an inflated meter is a replay-visible
    deviation (the slash path).
  - **Router cadence:** the router re-sorts at epoch boundaries
    only; within an epoch routing is stable.
  - **Headroom rule, price side:** the dev prices bootstrap arms at
    registered reference hosting cost, never below — no subsidized
    price to defend the crown.
  - **No price floor:** the anti-sybil floor is the registration
    fee plus the notice period, not a minimum price.
  - **Max unit price (task field, 24 Aug):** the task descriptor
    carries an optional per-unit price ceiling; routing filters
    candidates by posted price ≤ ceiling before applying the mode
    rule; no qualified capability → abstention. The ceiling is
    part of the hashed descriptor, so routing stays
    deterministic.
  - **Max spend (session field, 24 Aug):** the declaration may
    carry an optional TOTAL charge cap, in ETH (a USD figure is
    display-only). Serving stops when the remaining cap is less
    than one unit; a unit that would exceed it is not served (for
    generation, the answer is bounded to the last affordable
    token). The user pays only for metered units — the cap is a
    limit, not a pre-payment, and nothing runs past it. The cap
    is part of the hashed declaration, so the stop replays.
  - **Answer guarantee (25 Aug):** payment is for the measured
    computation, never for correctness on live data (a live sample
    carries no label); confidence is a measurement, not a
    warranty; refunds follow only from measured contract
    violations, decided by replay.
  - **Registration proofs (24 Aug):** contributors submit a compact
    Bulletproofs-style contract proof for the sealed HEAD at
    registration (the M193 machinery); the network verifies it, and
    the fingerprint becomes a cryptographic statement about the
    artifact. The encoder stays measured, not proved (M195).
    Validator honesty is audit-enforced — not provable at
    registration; per-point challenge commitments are their
    cryptographic anchor. **Settlement proof verification (25
    Aug):** proofs are batch-verified by a sampled, paid verifier
    step before their hashes anchor — an invalid proof is a
    ledger-disputable L1 deviation (replay-gated burn).

- **Pull only:** every payout is claimed by the beneficiary, never
  pushed. The claimer pays gas.
- **Abstentions are free (default registered 24 Aug, flagged for
  user override):** an abstention is recorded and costs nothing;
  it is unpaid to the arm.
- **Batches skip-and-emit:** a bad entry (delay, over-amount, cap,
  self-payment) is skipped with an event; it never reverts the batch.

## 3. Attribution and vesting

- **Epoch-vested credits:** a credited amount vests linearly over
  **N = 4 epochs** (28 days, 7-day epochs), first tranche after the
  first epoch. Vested credits are claimable any time after (pull).
- **Why N=4 (derivation):** with earnings rate R and detection at
  epoch T, a cheater's net is (T−2N)·R — profitable only if
  undetected for > 2× the window. At N=4 the evasion horizon is
  > 56 days against per-epoch replay-gated detection; an honest
  > contributor's worst wait is 28 days. The promise N·R scales with
  > the cheater's rate (self-scaling deterrence for large attackers).
  > The detection horizon for subtle attribution gaming is unmeasured:
  > H-series simulation sets N by evidence. N is timelock-adjustable.
- **Non-transferable claims:** unvested AND vested-but-unclaimed
  balances are account-bound mappings with no transfer/assign API,
  ever. Realized only by pull-claim to the credited account. Post-
  claim ETH is ordinary ETH.

## 4. Slashing: burn, graded, replay-gated

- **Slash = burn.** Slashed amounts move to a `burnedTotal` bucket
  excluded from every claimable bucket forever. Nobody gains from a
  slash: contributors don't harvest rivals (no restore-to-pool), and
  the dev doesn't profit (no dev-fund routing). Burn is
  unclaimable-by-design, not dead funds.
- **Graded ladder:**
  - Level 0: underperformance/downtime — no slash; the market
    penalizes via reduced attribution.
  - Level 1: provably fraudulent output / bit-inexact deviation —
    burn the unvested promise remainder.
  - Level 2: provable adversarial attack — full promise burn +
    registry delisting (kills future income).
  - Level 3: coordinated attack — delist + replay-gated burn of
    vested credits.
- **Provability gate:** convictions are replay-gated — a slash
  requires a mechanical re-run of the sealed artifact on sealed data
  showing the deviation (M214/M193b machinery). Any party files the
  dispute with a deposit; the librarian files the verdict's evidence
  hash; the math decides guilt. There is NO adjudicator role.
  P(false conviction) ≈ 0 for honest contributors.
- **Refund-to-payer rejected:** a cheater paying through
  sock-puppet payers would recover slashed funds via refunds.

## 5. Registration (stake retired)

- **No stake, no principal lockup.** Stake's three jobs are
  replaced: (1) anti-spam/DoS → flat registration fee paid directly
  to the dev fund (timelock-adjustable); (2) slash collateral → the
  slashable unvested promise; (3) self-payment exclusion key → the
  registration's PAYOUT address (a payment from the beneficiary
  address cannot thaw its own arm — structural, address-based,
  C1-compliant); operator key and payout address are separate
  registration fields (cold-key hygiene, 24 Aug).
- **One registration form for both kinds (24 Aug):** arms and
  primitives share the same registration — operator key, payout
  address, price per unit of work, sealed claim — and the same
  admission: one validator challenge session (a primitive's
  challenges are reference executions). The kinds differ only in
  who executes them.
- **Deduplication (25 Aug):** the artifact hash is the registry
  key — an identical hash is the same artifact and cannot register
  twice; a copycat registers a different artifact and earns
  admission on its own measurement.
- Consequence: no opportunity cost, so the bootstrap problem is
  "are fees attractive", not "is lockup worth the yield".
- Residual: the LIBRARIAN admits registrations (setAdmitted),
  executes the deterministic task-map extensions (M276: new task
  descriptors + fingerprints, clerk-only — it files, the rule
  decides), and records attribution batches — bounded by
  measured-quality gates; quorum/governance later.
- **Evaluation-data attack surface (24 Aug):** sealed evaluation
  corpora invite probing (oracle queries, per-row influence from
  score deltas, steganographic outputs, validator leaks). Defenses:
  aggregate-only verdicts (one score per axis, never per-row
  outputs or per-class breakdowns); scores reported to bounded
  precision (default: four significant digits) so single-row
  influence (~1/n) is below resolution — the exact value stays in
  the ledger; every submission pays the registration fee (the
  oracle is priced); splits rotate; validators hold the split under
  minimal access with the replay-gated slash path for leaks; arms
  are frozen at scoring time (no code beyond the sealed forward
  pass). Rubric posture: the rubric is minimal and aligned with the
  held-out metric itself — the metric is the goal. **Validators
  are sharded:** each holds one shard of the split and receives
  only aggregates (the sandbox pattern); a leaked row identifies
  its holder (slash path); shards reshuffle each rotation and
  retired rows are replaced. Residual: probing resistance is
  SNR-limited, not absolute; a corrupt majority of validators
  remains outside the mechanism's reach. **Challenge layer
  (user-proposed, 24 Aug):** admission confirmation is a public
  challenge session — each validator commits hashes of (input,
  expected output), poses inputs, the frozen artifact answers, and
  outputs are revealed and scored; a correct answer landing before
  its reveal is structural proof of collusion (slash path);
  challenges use disposable data (revealed points are public);
  quorum default = two-thirds of the sampled validator set,
  verdict = quorum-weighted challenge score against the published
  floor. **Validator mechanics (24 Aug):** permissionless per-axis
  registration with a fee; a validator joins a sample only after an
  activation window (2 epochs) and only while active (≥ half of
  sampled rounds responded in the trailing 2 epochs — the M294
  gates applied to admission); the sampled set = hash(anchor-after-
  commit, axis, admission commit) — no one chooses their judges
  and the seed postdates the commit (no grinding); challenge
  audits re-label a sample of revealed points, paid from the
  challenge budget at the registry-set rate (unpaid audit work
  would be shirked); dishonesty burns the unvested validator
  promise (replay-gated) and delists; validators only push ledger
  entries, so a silent validator is skipped (quorum over
  responders) — DDoS costs the attacker, not the admission;
  validators are paid per accepted challenge from the contributor's
  submission budget at the registry-set per-challenge fee (never
  contributor-set, so the fee cannot double as a bribe),
  epoch-vested, pull-claimed, 2.5% dock. Nobody is paid for
  silence. **Serving substitution (24 Aug):** a
  registered fraction ρ of queries (default 0.05, the M209
  redundant-sampling line) is shadow-answered by a reference
  execution of the sealed artifact on that session's own input
  (live, never the registration challenge set); answers
  must match exactly (the artifact is deterministic); detection
  horizon ~1/(ρ·rate) sits inside the vesting promise, so caught
  gains burn (L1 deviation, repeat → L2 delist); every active axis
  also carries a per-epoch minimum probe (default one) so quiet
  axes stay inside the horizon; residual probabilistic — ρ is
  timelock-adjustable upward. **(25 Aug, hardened):** the serving
  host commits H(answer) before the probe flag is revealed
  (commitment ordering replaces secrecy); executors are sampled
  per probed session from the artifact's pool under validator-style
  eligibility (activation 2 epochs, activity floor, tenure ramp
  T=4) — a fresh key is cheap, a pedigree is not; k_e=2 executors
  (default) sampled by H(anchor, session, host-commit), revealed
  only after the host's commit, so no single executor judges a
  session and a judge cannot be pre-bribed. **Probe funding
  (25 Aug, user):** the probed contributor pays — the session's
  credit is docked the registered reference-run price, which pays
  the reference executor (a replaceable, ledger-measured role);
  honesty monitoring is a contributor cost like admission, so the
  dev fund carries no per-session monitoring line (the zakat end
  state stays whole) and a wash ring cannot drain the fund through
  forced probes. Executors hold the artifact and register into its
  pool; the docked price is divided among the sampled set (the
  contributor's dock stays one registered price). The executor
  sees the probed input in plaintext — a second plaintext point
  alongside the gateway and the serving host, under the same
  no-retention, no-training data contract. **Probe semantics (25 Aug):** the probe
  certifies artifact identity (repeatability), not live accuracy —
  serving-time samples carry no labels, so identity is the only
  claim measurable there; accuracy is an admission/re-registration
  instrument. On a deterministic artifact an honest contributor
  matches every probe (the bar is sameness, never a correctness
  percentage); a cheaper model that answers identically everywhere
  is the same function, no violation; any match band for benign
  nondeterminism is calibrated to measured reference-vs-reference
  divergence at registration — never to 1 − accuracy. **Probe
  disputes (25 Aug):** the reference answer is hash-committed
  before the serving answer is compared (no fabricating a mismatch
  after seeing it, no rubber-stamping a substitution); the
  disputed input is hash-pinned at session time and a mismatch
  claim is settled by deterministic replay of the sealed artifact —
  deposit the replay cost, loser pays it and burns, under the same
  slashable promise both roles carry; the executor's fee is the
  flat registered reference-run price either way and deviation
  burns are destroyed, never awarded to it (a false claim has no
  upside); the disputed credit escrows until the replay settles;
  probe assignment stays gateway-seeded, so nobody can force
  arbitrary extra computation on a victim. **Failure
  handling (24 Aug):** no-shows → responder quorum (minimum 3),
  fail-closed, unspent budget returned; void challenges (input
  contract, type check, reuse, quota) score/earn nothing; weight
  capped at m per round (no volume domination); wrong labels →
  exclusion, score recomputed, fee burns, repeat delist; disputes
  → deposit (the registered reference-run price — bounded,
  affordable) + evidence → deterministic replay in a sealed replay
  environment (the disputed input is reproduced only there, never
  the ledger): upheld returns the deposit and slashes the wrong
  party, rejected burns the deposit, loser pays the full replay
  cost;
  one replay appeal per contributor (replay that reproduces closes
  it, appellant pays the replay cost; otherwise verdict corrected
  and budget refunded); max R=3 rounds; forks resolve by the
  latest Ethereum anchor over UNANCHORED suffixes only — a chain
  whose anchored prefix disagrees with any earlier anchor is
  invalid outright (prefix immutability), and such a divergence is
  a registered incident for librarian replacement.

## 6. Roles and governance

- **Librarian (public name; `recorder` in code):** the role files
  records and executes deterministic registry extensions with no
  discretion — the user's naming decision (24 Aug): recorder is
  the librarian. Single key initially; timelocked `setRecorder`
  (2-day, matching other parameters); `renounceRecorder()` as the
  keyless end-state (freezes attribution — documented); the mature
  path upgrades the role to a governance contract with no human
  key. Open item: rename the contract identifiers
  (`setLibrarian` / `renounceLibrarian`) in the pending contract
  rework.
- **Dev fund:** spent at admin discretion initially; the timelocked
  devFund-change path moves it to a multisig later. **END STATE
  (user decision, 24 Aug):** once GEODE is mature, the fund's
  purpose converts permanently into a **zakat rule** — 2.5%
  (one-fortieth) of every fee goes to those who need it most:
  resource-poor contributors, educational programs, incentive
  programs; in the best case (poverty eliminated) a generic UBI.
  Belief basis (author's own words, labeled as a belief): a mature
  economy's ~2–2.5%/yr growth is produced by everyone in it, so
  everyone involved has a right to a share of it, regardless of
  whether their role was considered important. The rule is fixed
  in advance so the mature-phase purpose is not a later discretion
  item. **FIXED POINT (user decision, 24 Aug):** the end state is
  outside ordinary governance — no quorum can re-purpose or dilute
  it (M189 veto path 1 and the M196 fixed-points list extended).
  User rationale, registered privately: the hypercapitalist system
  is weakened precisely by ignoring wealth taxes of this kind.
- **Founder compensation:** dev-fund-only. No token to mint.
- **Dev non-competition:** the dev/operator entity registers NO
  arms and earns NO attribution, in bootstrap or after — with ONE
  exception below (bootstrap arms). Enforcement: public ordered
  registry + legal commitments (on-chain enforcement would need
  identity — C1).

## 7. Bootstrap: MVP utility proof by dev arms (user decision)

- The dev ships a **minimum viable product with an initial patch of
  bootstrap arms**: the sealed arms already measured (code 0.8598,
  TTS WER 0.0519, vision 0.901 scoped, ASR) — limited in scope,
  cheap to serve (frozen checkpoints + closed-form heads).
- **Bootstrap arms DO earn usage fees (corrected 24 Aug):** the dev
  runs rented inference servers and the fees pay for them. This is
  the registered **inference-host role** (paid market rate for
  serving) — not contributor compensation by stealth. The dev
  claims no coverage-novelty bonus on bootstrap arms.
- **Anti-competition mechanism — the headroom rule (user, 24 Aug):**
  the dev ships each bootstrap arm at roughly **80–90% of its
  available capability** — deliberately sub-maximal, REAL and
  measured (e.g., the 1.5B coder at 0.5976, not the 7B at
  0.8598), sealed like everything else. ROUTING RULE: a
  contributor arm that is STRICTLY BETTER on the same task axis
  gets routing priority — the bootstrap arm keeps only residual/
  fallback traffic (e.g., contributor-arm downtime). PEDIGREE IS
  IGNORED: the dev arm's history, momentum and coverage-novelty
  bonus count for nothing against a measured improvement; the
  crown passes by measurement alone. DEV SELF-RESTRAINT
  registered: the dev never re-registers improved versions of its
  own bootstrap arms to defend its position — the headroom is
  permanent, and any violation is visible in the public ordered
  registry (same address re-registering an upgrade). THE BAR IS
  PUBLISHED: the MVP ships with the measured bootstrap quality per
  axis as an explicit, legible target ("beat X on code and the
  axis is yours"). Equal-but-not-better contributor arms keep the
  bootstrap arm in priority (the bar must be BEATEN — the design
  intent is invitation to improvement, not near-tie splitting).
  Formal delisting of a bootstrap arm follows once a contributor
  arm holds the axis.
- **Deprecation rule:** a dev bootstrap arm is deprecated when any
  contributor arm covers its task axis at strictly better measured
  quality — first by routing priority, then by registry delisting.
- Rationale: a whitepaper-only launch reads as vaporware and never
  gets momentum; a working MVP proves utility. The bootstrap arms
  exist to demonstrate — they lose traffic and retire the moment
  real contributors serve the same axes.

## 8. Free-rider (copycat) policy

- The system is deliberately effort-blind: it rewards served value
  (accuracy/efficiency/novelty), not training provenance — proving
  provenance on-chain would need identity (C1).
- The publisher's structural head start: they hold the artifact
  before publication and can register at release; first-registration
  - deterministic routing = momentum; the coverage-novelty bonus
    goes to the publisher, never a copycat. (25 Aug: the bonus's
    MEASUREMENT is unregistered — measured from usage it would be
    a washable subsidy. It ships NOTHING in the MVP until a
    measurement rule is registered; the whitepaper's earn table
    does not include it.)
- Residual: a slow or absent publisher can lose the axis to a
  copycat — correct; the network's goal is that the capability is
  served.
- The momentum claim is an H-series simulation question (copycat
  race: publisher t=0 vs copycat t=k).

## 9. Conformance posture (for counsel, M188)

- Non-transferable unrealized claims (no secondary market for
  unrealized dues; weakens investment-instrument characteristics).
- Bearer-key residual documented as an inherent limit of the
  permissionless Byzantine design (cash-analogous).
- Identity layer: per-address KYC deliberately not performed — the
  exchanges' on/off-ramp KYC is the identity chokepoint, and the
  public anchored ledger is the trace. Side benefit: the project
  holds no personal data.
- Questions for counsel: classification of the (currently
  non-existent) token; whether the service itself crosses
  GwG/MiCA-CASP thresholds regardless of third-party KYC; liability
  (M198).

## 10. Contract implementation standards (ethskills)

The contract rework implements all of the above and is gated by the
ethskills checklists fetched for the 24 Aug audit:
general + access-control + reentrancy + precision-math + DoS +
proxies (+ ERC20 where relevant) checklists, with the audit finding
format and severity scale from the audit skill. The rework
invalidates the sealed evidence; after re-harnessing it must
re-pass: 70+ tests, the coverage commit gate (100% line+branch on
every authored contract; LinearProofVerifier excluded as library
code — see `EVM_CONTRACT_AUDIT_2026-08-24.md` §3), gas budgets
re-measured, and a fresh audit section with findings in
`[X-N] Title / Severity / Category / Location / Description / PoC /
Recommendation` format. Medium+ findings block deployment.

## 11. Standard primitive library (free-with-network layer, user 24 Aug)

- **Shipped free, from day one — the launch catalog by category
  (25 Aug, user):**
  - **Memory:** count-based variable-order memory (register,
    backoff lookup, predict-next) — BUILT
    (`src/programmatic_memory.py`; packaging as a standard
    Primitive pending).
  - **Math:** scale, clip, affine, L2 normalize, reductions,
    softmax, log/exp — core eight BUILT (`src/primitive.py`);
    reductions/softmax pending.
  - **Symbolic math:** simplify, expand, factor, substitute,
    differentiate, integrate, linear/polynomial solve — PENDING
    (pinned computer-algebra engine).
  - **Logic:** threshold, comparison, and, or, not, where — core
    BUILT.
  - **Signal:** delay BUILT; FFT/mel/STFT/resample PENDING as
    packaged primitives (the mel stage exists in the audio
    experiments).
  - **Text:** versioned tokenization, Unicode normalization, case
    folding, character n-grams, edit distance — PENDING.
  - **Image:** grayscale, colorspace conversion, flip, rotate,
    resize, crop, normalize — PENDING; codec pinned.
  - **Code execution:** sandboxed engine for any registered pure
    function — BUILT (`src/programmatic_primitive.py`).
- **Entry rule (25 Aug):** a primitive enters the standard library
  only if it is pure, deterministic, resource-bounded, and pinned
  to an exact dependency version. No randomness, no network, no
  wall clock.
- **Placement policy (25 Aug, user overrule):** the standard
  library is BROAD on purpose. A small library is a demo, not
  infrastructure. Every Tier-A candidate in
  `analysis/SLP_CANDIDATE_CATALOG_v1.md` is packaged for launch;
  Tier-B candidates admit as their pins and certificates land.
  The full possibility space — 36 domains × 18 operation
  families, ~720 named primitives, with the tool-call equivalence
  map — lives in
  `analysis/SLP_POSSIBILITY_SPACE_v1.md`.
- **Documentation duty (25 Aug, user):** the SLP lives at
  `src/slp/`. Every primitive ships with its `PrimitiveSpec` and
  a factory docstring in STE style. The findability index is
  `src/slp/CATALOG.md`, generated from `src/slp/_catalog.py` and
  checked in — never edited by hand. `src/slp/README.md` is the
  developer guide: determinism contract, naming conventions, and
  the add-a-primitive checklist (register the PENDING entry
  first, build the factory, add the determinism test, flip to
  BUILT, regenerate the catalog).
  The dev fund carries the maintenance duty, including
  re-certification when a pinned dependency upgrades. The SLP
  holds code-defined transforms ONLY — never learned models —
  so a broad free layer cannot compete with contributor arms.
  Third-party primitives remain for what the network does not
  maintain and for what fails the entry rule.
- **Contributor-local, not dev-hosted:** each contributor runs the
  standard stages on their own machine; the dev earns NOTHING from
  them (no server costs to recover — unlike the dev-hosted
  bootstrap arms). Determinism is the trust story: the sealed code
  hash means a stage run by contributor A bit-matches the
  validator's replay, so the dev need not host anything.
- **Fee attribution:** when a session's chain uses standard stages
  plus a contributor's arm, fees attribute ONLY to the
  contributor's arm(s). Standard stages are infrastructure, not
  competitors — consistent with the no-central-planner and
  dev-doesn't-compete rules.
- **Maintenance:** the standard library is a public good maintained
  from the dev fund (audits/tooling), like the rest of the
  treasury's registered purposes.
- **THE ONE CARVE-OUT — the code-execution engine runs USER code:**
  unlike memory/math/transforms (which only run sealed inputs),
  a programming engine executes arbitrary end-user programs on the
  CONTRIBUTOR's machine. Registered: (a) the engine is opt-in per
  arm, declared in the arm's spec; (b) it is sandboxed with the
  M289/M290 machinery (resource ceilings, no network/filesystem,
  disposable execution) — the contributor's exposure is bounded
  and declared, never hidden. The engine itself is dev-shipped
  trusted code; what runs INSIDE it is untrusted.
- **THIRD-PARTY PRIMITIVES ARE PAID (user clarification, 24 Aug):**
  only the dev-provided STANDARD library is free. Any other
  registered primitive is a fee-bearing artifact: when a host runs
  it, the usage fees split between the primitive's PAYOUT ADDRESS
  (a royalty) and the HOST (the contributor running it). The
  contributor sets the rate (the user's working intuition:
  ~20–30%); the rate is TIMELOCKED with a notice period — a
  change does not take effect immediately, and users see it coming
  so they can migrate away. OPEN micro-decisions: (a) notice
  period length (proposal: one epoch, 7 days — users need time to
  migrate workloads; the 2-day CHANGE_DELAY is the consistency
  alternative); (b) royalty ceiling (proposal: none — market
  discipline; a 99% royalty finds no hosts; the anti-wash stack
  covers author-host collusion). Primitive contributors earn by
  usage exactly like arm contributors — the same registration
  form, the same validator vetting, the same rate field — no
  central planner sets rates.
  ALL primitive usage fees — contributor royalty and host share
  alike — are subject to the same 2.5% dev-fund dock as every
  other fee in the system (user clarification, 24 Aug).
- **Anti-wash note:** author royalties do not bypass the
  self-payment exclusion (a payer owning the served components
  cannot thaw from its own payments); author-hosting-own-primitive
  is legitimate service provision, and inflated usage through
  colluding payers is the existing wash vector the stack already
  blocks. A host that silently modifies a third-party primitive is
  serving an artifact other than the sealed one: replay-visible,
  same slash path (25 Aug).
- **ROLE CLARIFICATION (24 Aug):** the primitive author is the
  registration's royalty address --- a payout field, not a role; the
  host is the executing address. For arms there is no separate host
  role (serving is the contributor's obligation; availability
  measured; hired hardware is a private arrangement). There is NO
  adjudicator role --- disputes are open, deposited, and decided by
  replay.

## 12. Still open

- ~~B6 recorder mechanics confirmation (timelock + renounce)~~ —
  DONE in the whitepaper-aligned rework (24 Aug): `setLibrarian` /
  `renounceLibrarian`, owner-only; no human key remains after
  renunciation.
- **Re-audit DONE (24 Aug):** the reworked contracts re-passed the
  harness (46 tests, 100% coverage on every authored contract,
  re-sealed gas evidence) and a fresh ethskills conformance review:
  `analysis/EVM_CONTRACT_AUDIT_2026-08-24_R2.md`. Two Medium
  findings were found and fixed in-session (credits now require
  admission; level-2 slashes bind the artifact to its payout
  address). One governance trust point carried ([C1-M]: the
  single-key librarian becomes the M189 quorum contract before
  mainnet). The 24-Aug original audit is superseded.
- H-series simulation set **DONE (25 Aug):** `logs/results/v25/m293_hseries/evidence.json` (M293 sealed). Copycat race, detection-horizon sweep, and bootstrap dynamics all PASS on the registered scenarios (see the plan's M293 SEALED entry). The vesting window N=4 clears half the p90 detection horizon of every cheat class at the registered detection capabilities (binding class: attribution gaming, p90 4 epochs); real per-epoch detection rates remain a deployment question.
- M188 counsel engagement (user action). M194 Sepolia approval
  (user action; deployment on hold).

## 13. Quorum takedown (M294, user decision 25 Aug)

Socially destructive content must be removable by the network
majority before it becomes a legal problem for the network and its
hosts. The mechanism (spec `analysis/v25_m294_quorum_takedown_spec.md`):

- **Proposal + deposit:** anyone files a takedown with evidence
  references (challenge reveals are the natural substrate); the
  deposit is set at the registered cost of one vote round —
  proposal spam cannot tire the voter set for free, and a real
  report is never priced out; the deposit burns on a rejected
  verdict.
- **Voters:** the first k=9 validators of the axis pool ordered by
  `hash(epoch, artifactId)` — no one chooses their judges.
- **Eligibility (anti-flood, 25 Aug):** sampleable only after an
  activation window (2 epochs from registration), only while above
  the activity floor (responded in ≥ half of sampled rounds), and
  only with RECENT work — a responded round within the trailing
  W=2 epochs; a silent veteran loses the vote entirely. Votes carry
  tenure weight `min(1, tenure/T)`, T=4 epochs — fresh registrations
  carry ~zero quorum weight.
- **Verdict:** support weight ≥ 2/3 of sampled weight, minimum three
  responders, fail-closed below 1.0 total weight (cold start); the
  librarian FILES the deterministic count (never decides).
- **Effect:** permanent delist — no burn, no retroactive credit
  destruction (distinct from the slash ladder, which is
  replay-gated and burns). `CreditLedger` skips credits for
  delisted artifacts (`"delisted"` skip reason); the flag carries
  the quorum-record hash.
- **Emergency stage:** the M248 quorum freeze is the time-bounded
  containment; a per-artifact serve-freeze is a registered
  extension.
- **Honest boundaries:** takedown is report-driven (the network
  cannot scan content); it stops payments and registry listing,
  not off-path serving; a validator supermajority is a standing
  griefing surface, bounded by sampling, deposits, demerits, and
  the replay-gated slash path for false evidence. The definition of
  "socially destructive" is deliberately unformalized — a
  governance + counsel question (M188 brief Q5).
