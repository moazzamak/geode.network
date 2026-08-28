# M294 — Quorum Takedown (registered 25 Aug 2026)

Proposed by the user ("network-majority blacklist for socially
destructive artifacts"); spec registered BEFORE the build, per the
house rule.

## Why this exists

The slash ladder delists artifacts for REPLAY-GATED offenses —
provably fraudulent measurement, decided by recomputation. Socially
destructive content is a JUDGMENT, not a computation: it has no
replay oracle. This mechanism is therefore the system's FIRST and
ONLY discretionary power, and every property below exists to keep it
contained: quorum-gated, recorded, distinct from slashing, and
identity-free (C1 — an artifact is judged, never a person).

## Voter eligibility (Sybil defense — AMENDMENT, 25 Aug, registered before building)

User concern: someone who can register as a validator (or contributor)
cheaply could flood the pool when a decision is due and steer the
sampled set. Fee alone fails (wealth-sensitive, linear) and a time
lock alone fails (pre-registered dormant accounts). The registered
fix combines three gates, applied to ALL validator sampling
(admission verdicts, audits, and takedowns):

1. **Activation window:** a validator is not sampleable until
   A = 2 epochs after registration (working default).
2. **Activity floor:** sampleable only if it responded in at least
   half of its sampled rounds over the trailing window (zero rounds
   counts as active; silence once sampled counts against it).
3. **Recency (AMENDMENT 2, 25 Aug — registered before building):**
   tenure alone is not enough — a validator must have PERFORMED the
   role recently. It must have responded to at least one sampled
   round within the trailing W = 2 epochs; otherwise it carries
   zero weight, full tenure or not. Voting power is earned by
   recent work, not by old work. (Validators vote; arms do not —
   a contributor who registered an arm but stopped serving it was
   never a voter.)
4. **Tenure-weighted votes:** vote weight = min(1, tenure / T) with
   T = 4 epochs — a linear ramp from zero at activation to full
   weight at 4 epochs of tenure. Fresh floods contribute ~zero
   quorum weight; established validators dominate.

Fee remains the anti-spam floor, not the defense. Mass registrations
are public ledger entries and read as anomaly patterns (M199).

## The mechanism

1. **Proposal.** Any party files a takedown proposal: artifact hash,
   evidence references (ledger entries — challenge reveals are the
   natural substrate), and a deposit. A proposal does NOT by itself
   freeze or delist anything.
2. **Sampling.** The voter set is the first k entries of the axis's
   validator pool ordered by `hash(epoch, artifactId)` — no one
   chooses their judges. Defaults: k = 9, same as admission.
3. **Voting.** Sampled validators vote `support`/`oppose` on the
   proposal id; one vote per validator per proposal; votes are
   ledger entries.
4. **Verdict.** Ratified iff support ≥ two-thirds of the sampled set
   AND at least three responders (the admission rule). The check is
   deterministic — counting recorded votes — so the librarian FILES
   the verdict, never decides it. Refusing to file a ratified
   verdict is the registered divergence → librarian-replacement
   path.
5. **Effect.** A ratified takedown marks the artifact DELISTED —
   permanently (frozen artifacts cannot be rehabilitated; a fixed
   version is a new artifact with a new admission). Settlement
   enforcement: `CreditLedger.recordCredits` skips delisted
   artifacts with a `"delisted"` skip reason; the flag is filed by
   the librarian with the quorum-record hash.
6. **Deposit.** Returned on a ratified verdict; burned on a rejected
   one. Repeated rejected proposals from one party are recorded
   demerits. Voters citing provably false evidence stay exposed to
   the replay-gated slash path.
7. **Distinct from slashing.** Takedown delists and stops future
   income; it BURNS nothing (no retroactive destruction of vested
   credits). Mixing the two would turn a content vote into a
   financial weapon.
8. **Emergency half.** The M248 quorum freeze (already built,
   `geode/core/freeze.py`) is the time-bounded containment stage;
   takedown is the permanent end. A per-artifact serve-freeze is a
   registered extension, not built here.

## Registered gates (before building)

- **G1 verdict form (amended):** ratified exactly when
  support_weight ≥ 2/3 · total sampled weight AND responders ≥ 3
  AND total sampled weight ≥ 1.0 (fail-closed on a cold start or a
  weightless pool); below any bound it fails closed.
- **G2 no self-dealing:** votes outside the sampled set are ignored;
  duplicate votes count once.
- **G2b eligibility:** a validator with tenure below the activation
  window, below the activity floor, or WITHOUT a responded round in
  the trailing W epochs carries zero weight and is never in the
  sampled set.
- **G3 permanence:** once delisted, an artifact stays delisted for
  every later index; no un-delist path exists.
- **G4 distinctness:** takedown never moves credits — burnedTotal
  and creditsOf are untouched by the takedown path (it is not a
  slash).
- **G5 enforcement:** the ledger contract skips credits for a
  delisted artifact with the `"delisted"` reason, and reverts
  `setDelisted` on unknown artifacts and non-librarian callers.
- **G6 honest boundary (not a gate):** takedown is report-driven;
  the network cannot scan content (inputs are not retained, arms
  are black boxes). Takedown cannot prevent serving outside the
  settlement path — it stops the PAYMENTS and the registry listing.

## Honest boundaries

- A validator supermajority can always take down an artifact — this
  is the intended power, and it is also the new griefing surface
  (mitigated by sampling, deposits, demerits, and the slash path for
  false evidence).
- "Socially destructive" has no formal definition here. The verdict
  record (evidence refs + votes) is public, so any takedown is
  auditable after the fact; the definition itself is a governance
  question for the M189 quorum and counsel (see the M188 brief).
