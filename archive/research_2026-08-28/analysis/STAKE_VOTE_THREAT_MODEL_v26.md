# Stake-Vote Threat Model (v26) — Hybrid voting weight under Byzantine attack

Date: 27 Aug 2026
Status: ANALYSIS (registered before any build of M325/M326/M327 machinery)
Scope: the hybrid voting-weight system registered in plan §8.26–§8.28
(M325 dev-fund governance, M326 unified voting weight, M327 bootstrap),
including its interactions with M303 (router), M307 (behavioural
identity), M313 (verified-work accrual), M314 (security floors), M315
(takedown), M319 (selective abort), M322 (private serving), M323
(content orders).

Marking convention: **[registered]** = the protection is already in
the design; **[proposed]** = a gap found by this analysis, flagged for
decision, NOT yet part of the design. Nothing marked `[proposed]` is
in force until registered.

---

## 1. The system under analysis

One rule decides governance votes. Whether a vote counts is
pedigree: an activation window (two epochs), an activity floor
(responded rounds), and a verified-work-only record. How much it
weighs is earnings: the voter's thawed-but-unclaimed balance, counted
per behavioural identity, forfeitable on replay-gated conviction.
No pre-mine, no airdrop, no mint. Account-bound and non-transferable.
A charter-fixed 20% cap clips any single identity; excess counts at
zero. Votes are public ledger records. A genesis council runs votes
during bootstrap and sunsets by timelock.

The burn is gated on technical correspondence only: a replay of the
sealed artifact on sealed data. Votes are judgments. No replay can
settle a vote, so a wrong vote is not slashable. That single fact
divides the attack surface in two: the weight's backing is protected
by computation, the votes it buys are protected only by the vote's
own machinery.

## 2. Threat actors

- **Hoarding operator.** One party serving heavily, never claiming.
- **Cartel.** Several colluding operators, possibly one owner with
  many distinct artifacts.
- **Competitor.** Wants a rival delisted, frozen, or burned.
- **Well-capitalized outsider.** Buys influence where influence is
  for sale.
- **Corrupt validator set.** Up to a majority of the pool.
- **State actor.** Uses the tier-1 order path, or compels operators
  directly.
- **Developer.** Self-dealing, if any capability exists to do it.
- **Griefer.** Wants to lock, stall, or annoy at negative value.
- **The Sybil.** Cheap fresh identities, if they earn anything.

## 3. Attack catalog

### A. Weight acquisition attacks

**A1. Accumulate one identity past the cap.** The hoarder serves and
never claims, hoping to hold a supermajority alone.

- Closed: the 20% cap is charter-fixed, outside governance; excess
  counts at zero **[registered]**. The takedown quorum is 2/3 of
  SAMPLED weight, so one capped identity is always short.
- Residual: the cap is per identity, not per party (A2).

**A2. Multi-identity accumulation.** One party registers many
genuinely distinct artifacts, each earning its own capped weight.

- Bounded: every identity must pass measured admission, earn through
  the lottery-spread traffic, and survive dedup (copies collapse)
  **[registered]**.
- Cost: each identity is a real serving business — compute,
  availability, admission fees, bonds. A Byzantine actor with a
  compute budget can still run several.
- Residual: party-level identity is deliberately absent (identity
  checks are banned). The bound is cost, not a hard rule. This is
  the principal open flank of the whole system and is carried into
  §6 as proposal P1 (adopted 27 Aug as M328).

**A3. Buy weight.** Purchase a key whose account holds unclaimed
credits.

- Closed: weight is account-bound and non-transferable; a claim
  removes the weight **[registered]**.
- Residual: the wallet is the identity boundary. A key can be sold
  whole, weight included. Inherent to permissionless systems; the
  paper states it. Vote-buying that pays per outcome is attacked
  separately (C4).

**A4. Mint or inflate weight.** Pre-mine, airdrop, or fake verified
work.

- Closed: no pre-mine, no airdrop **[registered]**; accrual counts
  only sampled, verified work — self-generated volume under any
  address accrues zero **[registered]** (wash rings lose money every
  loop); the developer, the council, and the bootstrap operator
  cannot create or receive weight outside the earning path
  **[registered]**.

**A5. Park weight forever at zero marginal cost.** Defer claims
indefinitely to keep influence.

- Bounded: parked balance is the burn pool — it is forfeitable
  capital, not free influence **[registered]**; eligibility requires
  recent activity, so parked weight without serving work goes inert
  **[registered]**.
- Residual: parking costs more for the poor than the rich; the cap
  clips the resulting skew per identity.

**A6. Identity churn to dodge the cap.** Retire capped identities,
re-register fresh ones, repeat.

- Closed: weight dies with a claim or conviction — churn abandons
  the accumulated balance and starts at zero; fresh identities must
  survive the activation window and the activity floor before any
  vote counts **[registered]**. Churn is a treadmill that pays the
  cap each cycle.

### B. Vote-capture attacks

**B1. Stack the sampled voter set.** Place own identities in the pool
so the hash sample lands on them.

- Closed: judges are sampled by hash from the pool; nobody chooses
  their judges **[registered]**; the seed comes from an external
  randomness beacon, so no party can grind it **[registered]**;
  pool-flooding fails the pedigree gate — activation window,
  activity floor, recent work **[registered]**.
- Residual: an adversary that legitimately operates a large share
  of the pool samples proportionally — that is A2 again, bounded by
  cost.

**B2. Self-interested votes.** Vote to protect one's own axis from
takedown, or to pace fund spending toward one's own demand.

- Inherent: votes are judgments, not computations; economic
  self-interest cannot be slashed away **[registered]**.
- Bounded: the cap limits any one voter; the quorum needs 2/3 of
  sampled weight; effects are fixed (suspend, never burn; pace,
  never redirect) **[registered]**; every vote is public
  **[registered]**, so sustained self-dealing is visible and
  politically costly.

**B3. Captured-quorum power grab.** A quorum that wants to rewrite
the rules for itself.

- Closed: the cap, the security floors, and the zakat end-state sit
  outside ordinary governance **[registered]**; no path exists to
  route fund money to any quorum member (inexpressibility)
  **[registered]**; the developer holds no escalation capability
  **[registered]**; compliance-policy changes need the
  cross-jurisdiction nexus quorum **[registered]**.
- Residual: the remaining adjustable knobs (registry-set fees, the
  reference-run price, timelocks upward) are real but small; a
  captured quorum's changes are public and reversible by the next
  quorum.

**B4. Abstention and stall.** A Byzantine minority never wins a
vote; can it freeze governance forever by refusing quorum?

- Closed: quorums fail closed — no action happens, and the
  mechanical paths (zakat trigger, floors, timelocks, freezes with
  expiry) do not wait for votes **[registered]**. Stalling a vote
  stalls only the pacing dial, never the charter.

### C. Forfeiture and framing attacks

**C1. Frame a rival to burn their weight.** Fabricate a deviation to
trigger the slash.

- Closed: burns are replay-gated — the sealed artifact is re-run on
  sealed data; a false conviction needs a false replay, whose
  probability is effectively zero for an honest contributor
  **[registered]**; disputes carry a bounded deposit so the right
  is affordable, and the loser pays **[registered]**; appeals cite
  registered evidence classes **[registered]**.
- Residual: if the rival's own host is compromised, the replay
  convicts them "correctly." Host security is the contributor's
  burden — stated, not hidden.

**C2. Escrow-lock a rival.** Freeze their credits with repeated
community reports.

- Closed: every tier-2 reporter posts a deposit; an unconfirmed
  freeze burns the deposits and releases the escrow
  **[registered]**; freezes expire **[registered]**; abuse feeds
  public notice-abuse statistics **[registered]**.
- Residual: a state with a valid tier-1 order freezes without a
  deposit — but that path requires an authenticated, in-nexus
  order; forgery is closed at the signature check **[registered]**.

**C3. Grief the accrual path.** Selective aborts and admission
gaming to deny rivals verified work.

- Closed: M319 selective-abort adjudication; admission resampling
  instead of re-fee; griefing costs the griefer **[registered]**.

**C4. Vote-buying with verification.** Pay a voter for a specific
vote. Public ballots make the payment checkable: the buyer can
verify the vote was cast as agreed.

- CLOSED by M328 (registered 27 Aug 2026): ballots are Pedersen
  commitments opened only as weighted sums by a threshold tally
  committee. No individual ballot is ever published, so no bribe
  can be checked against a ballot and no voter can prove its
  vote. The ledger still carries who voted, the commitments, the
  proofs, and the opened sums — the anti-coercion record
  survives.
- Residual: a corrupt majority of the tally committee could
  deanonymize individual ballots — the standing
  corrupt-validator boundary. Plain commit-reveal was analysed
  and REJECTED for this slot: ex-post bribery stays verifiable
  against opened ballots, so it closes nothing.

**C5. Corrupt the validators that certify accrual.** Inflate one's
own verified work via a corrupt auditor set.

- Bounded: validators are hash-sampled, their challenges are
  capped per round, and auditors are paid and audited in turn
  **[registered]**.
- Residual: a corrupt majority of the validator pool is outside the
  mechanism's reach — the paper's standing limit.

### D. Process attacks (Byzantine liveness/safety)

**D1. Seed grinding.** Influence the judge sample by choosing the
anchor.

- Closed: sampling randomness comes from an external beacon the
  librarian does not produce **[registered]**; the lottery seed
  binds the anchor, the task, the registry state, and the
  fingerprint **[registered]**.
- Residual: a party appending ledger entries can delay a sample's
  clock but never choose the sample — stated in the paper.

**D2. Ledger-level vote forgery.** Rewrite the vote record.

- Closed: prefix immutability; anchors; the force-inclusion queue;
  executable librarian replacement **[registered]**.

**D3. Compel the voters directly.** Threaten individuals who hold
weight.

- Bounded: votes are one of many governance inputs; the compliance
  path (tier-1 orders) already gives a state its lawful channel, so
  compulsion buys little extra **[registered]**; public records
  convert pressure into visible bargaining **[registered]**.
- Residual: a state can compel an operator within its borders to
  claim, to vote, or to stop serving. The defense is the
  jurisdiction posture, not cryptography.

### E. Bootstrap attacks

**E1. Council capture.** The genesis council colludes before any
stake exists.

- Bounded: the council is multi-party, never developer-only
  **[registered]**; its weight is not stake **[registered]**; it
  cannot route fund money to itself **[registered]**; it cannot
  mint or receive weight **[registered]**; it sunsets by timelock
  **[registered]**.
- Residual: early networks are concentrated. The design makes the
  path out of concentration automatic; it does not pretend
  otherwise — stated in the paper.

**E2. The developer plants weight.** No pre-mine covers direct
minting **[registered]**; the developer could serve its own
bootstrap arms and earn weight like any operator — that is the
honest, visible path, and the headroom rule prices its arms at
reference hosting cost **[registered]**.

## 4. What holds unconditionally in a Byzantine world

These properties do not depend on honesty of any party:

1. **No one controls the weight.** Weight is earned, capped,
   account-bound, and dies with a claim or a conviction. No key —
   developer, council, or state — can create, transfer, or unilaterally
   raise it.
2. **No one controls the destination of money.** Fund money has no
   path to any named party; disbursement at the end state is
   mechanical with no pause path.
3. **No one controls the floor.** The cap, the security floors, and
   the zakat rule sit outside governance; a captured quorum cannot
   lower any of them, including its own cap.
4. **Burns are computations, not judgments.** A weight can only be
   destroyed by replay; a vote can only suspend, freeze, or pace.
   The most powerful voter cannot burn a rival's weight with a vote.
5. **Every act is recorded.** Votes, orders, freezes, and pacing
   decisions are public ledger entries. A Byzantine actor's play is
   visible before it lands and auditable after.
6. **Abstention is safe.** Every quorum fails closed. Capturing the
   system requires winning sampled, capped, public votes — never
   merely preventing them.

## 5. What degrades (the honest boundary)

- **Judgment quality is not protected.** The weight aligns honest
  technical behavior (everything a replay can check). It does not
  align wise voting. A Byzantine quorum that is technically honest
  can still vote destructively within the fixed effects — the
  containment layer, not the weight, is what bounds that damage.
- **Party-level concentration is cost-bounded, not impossible.**
  One owner may hold several capped identities. (P1 below, adopted
  as M328.)
- **Key sales transfer influence.** Account-bound credits cannot be
  transferred, but a key can be sold whole.
- **A lying majority of the validator pool is outside reach.** The
  design reduces that risk to economics and visibility; it does not
  remove it.
- **Ballot secrecy rests on the tally committee.** M328 closes
  verifiable bribery; a corrupt majority of the sampled committee
  could deanonymize individual ballots --- the standing
  corrupt-validator boundary, moved but not removed.

## 6. Adopted protections (registered 27 Aug 2026 as M328, plan §8.29)

**P1. Quorum diversity floor.** A vote ratifies only when its
supporting weight comes from at least $d$ distinct behavioural
identities, with $d = \max(3, \lceil 0.2 \cdot
n_{\text{responders}}\rceil)$. One owner of many artifacts must
hold sampled weight across several genuinely distinct serving
businesses to ratify anything. No party identity is needed — the
floor is over identities, not owners.

**P2. Secret-ballot tally.** Each ballot is a Pedersen commitment
$C_v = g^{o_v} h^{r_v}$ with a membership proof for $o_v \in
\{0,1\}$, signed and public. After the window, a tally committee
of $k_t$ sampled validators (disjoint from the voter set where
possible) opens only the weighted sums $\sum w_v o_v$ and $\sum
w_v$ by threshold ($t = \lceil k_t/2 \rceil + 1$). Individual
ballots are never published. Unopened weight above one third of
the sampled weight fails the vote closed. This kills verifiable
bribery and vote-proving to coercers; plain commit-reveal was
rejected because ex-post bribery stays verifiable there.

**P3. Weight snapshot at vote opening.** Each vote freezes every
sampled voter's weight at the epoch-boundary snapshot that opens
it. Claims and accruals during the vote do not move the tally.
The snapshot is a ledger entry bound to the opening anchor; the
diversity floor and the forfeiture backing are computed on it.

## 7. Conclusion of the analysis

The system is engineered so that in a fully Byzantine world the
attacker can hurt themselves and burn money, but cannot mint,
transfer, or unilaterally raise weight; cannot redirect money;
cannot lower floors; cannot burn a rival by vote; cannot buy a
verifiable ballot; and cannot act unrecorded. The two flanks this
analysis found — multi-identity accumulation and verifiable
vote-buying — are now closed by M328 (P1, P2) with stated
residuals: the diversity floor is cost-bounded against
party-level concentration, and ballot secrecy rests on the tally
committee's majority. The standing limits remain: key sales, and
a lying majority of the validator pool.
