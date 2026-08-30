# THREAT ANALYSIS — GEODE SETTLEMENT PROTOCOL AND EVM CODE (30 Aug 2026)

**Scope.** A fresh, complete threat analysis of the settlement
protocol and its implemented EVM contracts, driven by the
settlement-DoS concern raised during the publishability review: "if
there is a repeated attempt at a denial of service, seven days per
attempt is a bad number; we need to be robust against it, and we need
a very thorough threat analysis again of the entire protocol,
including the implemented code."

Method: (1) read every contract in `infrastructure/evm` in full;
(2) map every surface where one party can block, delay, or grief
another, or the network as a whole; (3) classify each finding as
fixed, mitigated-by-cost, or registered-residual; (4) where a surface
was open and fixable, implement and gate the fix before writing this
document. Nothing in this document is a finding that was not either
already registered or fixed and tested in the same session.

Baselines before this session: EVM suite 168 passing; CreditLedger
deployed bytecode 24,487 bytes (EIP-170 limit 24,576). After: 178
passing; 24,498 bytes.

## 1. Threat model

Actors (from the paper, plus the outside inputs the contract carries):

| Actor                | Trust   | What it can block/delay                                           |
| -------------------- | ------- | ----------------------------------------------------------------- |
| User / payer         | none    | nothing (already bounded)                                         |
| Contributor          | none    | registration spam (priced), credit-claim honesty (attested)       |
| Host                 | none    | serving substitution (probabilistically detected)                 |
| Validator            | none    | verdict capture (weight-capped)                                   |
| Reference executor   | none    | probe collusion (needs every sampled executor)                    |
| Librarian            | partial | **the settlement key: recording credits, fast-path root posting** |
| Developer / dev fund | none    | parameter capture (timelocked), fee schedule                      |
| Any funded party     | none    | **challenge campaigns (the DoS that opened this pass)**           |

The relevant property: **no single party, and no cheaply-funded
party, may halt or push settlement indefinitely.** Settlement is the
per-epoch conversion of recorded credits and evidence into on-chain
payables, via the attribution root. Everything else in the protocol
serves it.

## 2. Surface map: every blocking / delay / griefing path

| #   | Surface                                                                           | Who                                  | Was it a live gap?             | Verdict                                                                                                                        |
| --- | --------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| S1  | Challenge every filing to push settlement out by one attestation window           | any funded party                     | **YES**                        | **FIXED this session** (escalating global fee + 3-day window)                                                                  |
| S2  | Poster contract that reverts on ETH receive jams the inbox queue head forever     | any party that can deploy a contract | **YES**                        | **FIXED this session** (bond return is now pull, M383 lesson applied to the bond)                                              |
| S3  | Librarian posts nothing: no attribution root for an epoch                         | librarian                            | YES (registered M385 residual) | MITIGATED: any party files the root under a bond (F1 closure, 29 Aug); fast-path race remains (registered)                     |
| S4  | Librarian stops recording credits                                                 | librarian                            | YES (registered)               | MITIGATED: failure is visible; keyless replacement is quorum-gated; new credits halt but verdicts do not (registered residual) |
| S5  | Spam the inbox to bloat the ledger / force chain-invalidity                       | any funded party                     | YES (registered M365/G24)      | MITIGATED: superlinear fee (O(N^3)), capped per-epoch obligation, recorded (29 Aug)                                            |
| S6  | Capture the attestation quorum by amassing vested weight                          | funded party                         | YES (registered)               | MITIGATED: per-identity 20% cap, participation floor, min 3 identities; single key cannot carry a verdict                      |
| S7  | Repeat a slash/registry/root challenge at flat cost to drain challengers or delay | any funded party                     | subset of S1                   | **FIXED this session** (same global heat as S1)                                                                                |
| S8  | Governance replacement blocked by a hostile quorum                                | any funded party                     | no (quorum-gated by design)    | REGISTERED: a hostile majority is outside the mechanism (paper known-limits)                                                   |
| S9  | Dev-fund / operations-line claim paths reverted by a hostile receiver             | any party                            | no                             | FIXED earlier (M383 pull); all claim paths are pull with revert handling                                                       |
| S10 | Bond stuck forever when a filing is never resolved                                | filers/challengers                   | no                             | REGISTERED: bonds are pullable after resolution; an unresolved filing's bond is released on finalize/execute (permissionless)  |

## 3. S1: repeated-challenge settlement DoS — the fix

**Original state.** Any party filed an epoch's attribution root under
a 1 ETH bond. Any party could challenge it under the same flat bond.
A challenged filing parks settlement until the attestation window
closes; a challenge that reaches no verdict proceeds as if
unchallenged and the challenger's bond burns. An attacker who simply
does not care about the 1 ETH per attempt could challenge every
epoch: each settlement delayed by one attestation window, **7 days
per attempt, flat cost**. That was the user-identified gap.

**The fix (implemented and gated this session, registered before
dispatch):**

1. **ATTEST_WINDOW is 3 days** (was 7). Cuts the per-attempt delay
   by more than half.
2. **A global, decaying challenge counter ("heat") drives an
   escalating fee.** The Nth live challenge within a 21-day half-life
   pays a non-refundable fee of `SLASH_BOND * 2^(heat-1)` (capped at
   `SLASH_BOND * 2^9` = 512 ETH) on top of the refundable base bond.
   First challenge in a quiet period: free. Second: 1 ETH fee. Third: 2. Fourth: 4. Fifth: 8. The fee is credited to the operations
   pool, which funds the settlement bounties — it never goes to a
   party.
3. **Heat is global, not per-address.** Rotating addresses cannot
   reset it.
4. **Heat decays only on 21 days of silence**, halving per elapsed
   half-life (10 or more elapsed half-lives clear it). A weekly
   campaign (the natural cadence, one challenge per settlement) never
   decays and compounds without bound.
5. **Dead challenges revert at the state guard, before the fee
   logic.** A challenge to a resolved, already-challenged, or
   window-closed filing costs nothing and does not heat the counter.

**Residual after the fix (registered in the paper known-limits, this
session).** A party with enough funds can still push a single epoch's
settlement out by one attestation window (3 days) once. A sustained
denial of service is priced exponentially, not removed; the cost of a
campaign of N challenges within one half-life is
`(2^N - 1) * SLASH_BOND` in fees alone, on top of N burned bonds, and
the window is now 3 days per attempt. This is a cost barrier, not a
proof — the paper says so.

**Gates (all new, `test/challenge_griefing.test.js`, 8 tests):**
first challenge free; second pays bond\*2 and the fee reaches the
operations pool; a sustained campaign compounds (doubling per live
challenge); heat is global across addresses; heat decays only on 21
days of silence (20 days is not silence); a weekly campaign never
decays; a dead challenge reverts cheaply (AlreadyResolved) without
heating the counter; a double challenge on one filing is refused
cheaply (AlreadyChallenged) without heating the counter; ATTEST_WINDOW
is 3 days.

## 4. S2: the reverting-poster inbox jam — the fix

**Original state.** `InclusionInbox.incorporate` returned the poster's
bond by push (`payable(e.poster).call{value: amount}`). A poster that
is a contract reverting on `receive()` would make `incorporate` revert
forever for the head entry. Because incorporation is FIFO and the
chain is invalid while the head sits past its deadline, **one
reverting poster permanently jammed the queue for everyone**: the
chain could never become valid again, and every entry behind the head
was stuck. This was the same class as the M383 pull lesson (a
reverting recipient blocking the queue) applied to the fee earner, but
the bond was still pushed.

**The fix (implemented and gated this session).** The bond is now
credited to the poster's `claimable` balance and pulled via `claim()`
— in `incorporate` and in `withdrawBond`. A reverting poster cannot
jam the queue; it simply cannot pull its own bond. The queue head
advances, the chain stays valid, and a foreign caller can clear
entries behind it.

**Gate (`test/inclusion_inbox.test.js`, 1 new test).** A
`RejectingPoster` (a contract that reverts on ETH receive) posts; the
librarian incorporates; the entry is closed, the bond is zero, the
chain is valid, the bond sits in `claimable`, and a foreign caller
clears a second entry behind it.

**Remaining note.** `claim()` and `claimOperations()` are pull paths
with revert handling; a hostile receiver's own claim fails without
affecting anyone else. The dev-fund and ledger claim paths were
already converted to pull (`_pull`) in this session for the same
reason.

## 5. S3: the librarian fast-path race (registered, not a live gap)

The librarian can still post the root directly (`postAttributionRoot`,
write-once) before any filing executes. A correct fast-path root lands
and a concurrent permissionless filing for the same epoch resolves
with a skip (the filer's bond is returnable). This is a race in favor
of the honest, prompt librarian, which is the intended fast path. The
one named residual: a librarian could post a wrong root on the fast
path and it would land unchallenged if nobody files/challenges within
the window — but that is exactly S1's neighbor, the challenge-gated
residual already registered in the paper (§7.23, "The settlement root
is challenge-gated, not verification-gated"), and the filer's bond and
the payees' standing bound it. Not a new finding.

## 6. Fresh findings from the code read (this session)

F1. **The challenge fee was charging on dead challenges** (the
reorder fix in §3.5). In the first implementation of `_openChallenge`
the fee logic ran before the `AlreadyResolved`/`AlreadyChallenged`/
`WindowClosed` guards, so a caller could pay a fee and heat the
counter for a challenge that could never succeed. Reordered: guards
first, fee last. This also keeps the "dead challenge costs nothing"
property that makes the escalation fair to honest challengers.

F2. **Storage layout was broken by the first placement of the new
slots.** `challengeHeat`/`challengeHeatAt` were initially declared
mid-layout (after `slashBondsBurned`), which shifts every subsequent
slot and would corrupt a proxy upgrade from the committed v1.0.0
layout. Relocated to the end of the storage declarations (after
`rootBountyClaimable`), with a comment marking them as appended.

F3. **`onlyOwnerOrGovernance` was dead code** (removed) and
`challengeFeesBurned` no longer exists (fee routes to the operations
pool); the balance-accounting NatSpec and the operations-pool NatSpec
were brought back in line.

## 7. Registered residuals (unchanged, restated for completeness)

- **Credit recording is a single-key act** (paper known-limits). A
  stopped or captured key halts new credits; it cannot change a
  verdict. The coupling: the same key writes credits and, over
  vesting, shapes quorum weight; the 20% cap is what keeps any single
  identity from carrying a verdict.
- **The settlement root is challenge-gated, not verification-gated**
  (paper known-limits). An early, unchallenged wrong root can land and
  is then write-once permanent; bounded by what the undrawn pool holds
  and by the standing of the parties it would mis-pay.
- **The 20% weight cap / participation floor**: a wrong root that
  would mis-pay less than one-third of eligible vested weight cannot
  be voided by the harmed parties alone and lands by default unless
  the unaffected majority participates.
- **Inbox spam is priced, not removed** (M365/G24): superlinear fee,
  capped per-epoch obligation; buying N slots of delay costs O(N^3);
  a spammer who posts first can push an honest poster's deadline out.
- **A hostile majority is outside every mechanism** (paper
  known-limits): validators, quorum, governance.

## 8. Follow-ups registered (do not gate this session's work)

- F4. The `claim()` paths in CreditLedger and InclusionInbox could
  share one canonical `_pull` implementation; they are already
  functionally identical (revert-on-failure, CEI). Mechanical, no
  behavior change; deferred.
- F5. A `challengeHeat`/`challengeHeatAt` view could be added to the
  tooling's exported ABI surface for monitoring dashboards (heat
  level is public already; no contract change needed).
- F6. The escalation constants (21-day half-life, cap at 512 ETH,
  3-day attestation window) are hardcoded `public constant`. They are
  deliberately NOT parameterized: making them mutable reopens the
  timelock-governance surface for a griefing parameter. If live
  operation ever needs to change them, that is a deployment decision,
  not a runtime one. Recorded here so nobody "fixes" the constants
  into parameters later without a registered reason.

## 9. Verification

- `npx hardhat compile`: clean; CreditLedger deployed bytecode
  24,498 bytes (< 24,576 EIP-170 limit, 78-byte margin).
- `npx hardhat test`: **178 passing** (168 baseline + 8 new
  challenge-griefing gates + 1 reverting-poster gate + 1 earlier new
  gate from the pull conversion).
- Dead files removed: `ChallengePricing.sol`, `AttestationQuorum.sol`
  (both superseded by the compact inline escalation; the library
  variant measured WORSE, 25,123 bytes, and was reverted).
- All claim paths audited for push-vs-pull; no remaining push to a
  caller-controlled address outside the guarded `_pull` helper.
