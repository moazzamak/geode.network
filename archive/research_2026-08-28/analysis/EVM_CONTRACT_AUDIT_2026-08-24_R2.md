# GEODE Smart-Contract Re-Audit — R2 (post-alignment)

Date: 24 Aug 2026. Scope: `infrastructure/evm/contracts/` AFTER the
whitepaper-alignment rework (`CreditLedger.sol` rewritten to native
ETH; `ProofAnchor.sol` renamed; `LinearProofVerifier.sol` unchanged;
`VestingVault.sol`, `GeodeToken.sol`, and the stake machinery
retired). Methodology: the ethskills checklists used in the original
`EVM_CONTRACT_AUDIT_2026-08-24.md` (which is SUPERSEDED for the old
contract set and is the baseline this review resolves against).

Verdict up front: **the reworked harness conforms to all seven
registered rules (as re-interpreted for the new design); no Critical,
High, or remaining Medium findings.** Two Medium-class issues found
during this review were fixed in-session ([R2-1-M], [R2-2-M]); one
governance trust point is carried explicitly ([C1]). Nothing is
deployed onchain and nothing should be until the M194/M188/M190
bucket-1 gates pass (user approvals).

---

## 1. The registered requirements (restated, M202 + alignment)

1. **Local EVM first.** No contract code merges without passing the
   full harness suite on the local EVM.
2. **100% measured coverage as a commit gate.** Statements, branches,
   functions, and lines for every authored contract (library code
   excluded per the registered `.solcover.js` note).
3. **Upgradeability.** UUPS; upgrades owner-authorized; the harness
   contains an upgrade rehearsal that proves STATE PRESERVATION.
4. **Admin release.** A tested path to renounce admin control;
   post-release invariants proven in the harness.
5. **Exploit checklist, each with ≥1 test:** reentrancy; CEI
   ordering; overflow-safe arithmetic; no unbounded loops in
   settlement; batched off-chain settlement; pull-over-push;
   timelocked parameter changes; storage-minimal design.
6. **Efficiency.** Gas budgets for hot paths measured and registered
   in sealed evidence.
7. **Anti-wash mechanics (C1, economic-only):** 2.5% dev-fund dock;
   self-payment exclusion keyed on the PAYOUT address (no stake, no
   whitelists); registration fee; N=4 epoch vesting; graded burn
   ladder. Identity-based mechanisms forbidden.

## 2. Conformance matrix

| #   | Rule               | Status   | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --- | ------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Local EVM first    | **PASS** | 46/46 tests pass (`npx hardhat test`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 2   | 100% coverage gate | **PASS** | Fresh run: `CreditLedger` and `ProofAnchor` at 100% stmts/branch/fn/lines. `LinearProofVerifier` excluded from instrumentation per the registered OOM note. Sealed `harness_evidence.json` gate: true.                                                                                                                                                                                                                                                                                                                                 |
| 3   | UUPS + rehearsal   | **PASS** | `_disableInitializers`, `_authorizeUpgrade` owner-only; `upgrade rehearsal preserves state` test (registration + credits + pool intact after `upgradeToAndCall`).                                                                                                                                                                                                                                                                                                                                                                      |
| 4   | Admin release      | **PASS** | `renounceOwnership` then owner==0; test proves `setLibrarian`, `renounceLibrarian`, `pause`, and upgrades all revert.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 5   | Exploit checklist  | **PASS** | Reentrancy: transient guard + two reentrant mocks (claim, claimDevFund). CEI: `deposit` has no external calls (native ETH); claims mutate state before `call{value:}`. Overflow: Solidity 0.8 checked; vesting math uses 1e18-scaled fractions. Bounded loops: `MAX_BATCH=64`; vesting windows capped at N=4 epochs; burn loop bounded by the live window. Pull-over-push: claims and dev-fund pulls. Timelocks: 2-day dev-fund/registration-fee; 7-day price changes. Storage-minimal: per-address N-window buckets collapsed lazily. |
| 6   | Efficiency         | **PASS** | Re-sealed budgets: deposit 97,602 / register 116,260 / record10 143,770 / claimDevFund 40,135 / claim 98,247 / slash 83,368 / anchor 46,016.                                                                                                                                                                                                                                                                                                                                                                                           |
| 7   | Anti-wash          | **PASS** | `DEV_FUND_BPS=25`; self-payment skip keyed on the payout address (tests assert `CreditSkipped` "self-payment"); flat registration fee to the fund; N=4 vesting; burn ladder L0-L3; no identity anywhere.                                                                                                                                                                                                                                                                                                                               |

## 3. Old findings resolution table

| Old finding                                    | Disposition                                                                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [1-M] vault thaw never delivers principal      | **MOOT** — vault retired; `CreditLedger.claim()` delivers vested principal.                                                 |
| [2-M] one bad entry reverts the batch          | **RESOLVED** — all malformed credits skip-and-emit (unregistered, not admitted, self-payment, over-pool); tests cover each. |
| [3-M] single-address recorder, instant changes | **CARRIED as [C1]** — see below.                                                                                            |
| [4-M] CEI inversion in stake/deposit           | **MOOT/RESOLVED** — stake retired; `deposit()` has no external calls.                                                       |
| [5-L] fee-on-transfer token                    | **MOOT** — native ETH.                                                                                                      |
| [6-L] slash strands tokens                     | **RESOLVED by design** — slashes move to `burnedTotal`, a bucket no claim path touches (the registered burn rule).          |
| [7-L] ProofAnchor documented as recorder-gated | **RESOLVED** — NatSpec now states permissionless anchoring; front-run caveat carried as Info.                               |
| [8-L] instant setMinter                        | **MOOT** — token retired.                                                                                                   |
| [9-I] transient guard in upgradeable contract  | **Carried Info** — verified safe (per-transaction storage, no layout impact).                                               |
| [10-I] re-vest resets the delay anchor         | **MOOT** — bucket-based N=4 vesting replaces the delay anchor.                                                              |
| [11-I] `_modExp` mulmod vs 0x05 precompile     | **Carried Info** — unchanged code; re-measure on the target chain.                                                          |
| [12-I] `_generator` h==0 guard                 | **Carried Info** — probability ≈ 1/P; optional hardening.                                                                   |
| [13-I] `anchoredAt[h]==0` sentinel             | **Carried Info** — verified safe (genesis executes no txs).                                                                 |

## 4. R2 findings (found this review)

### [R2-1-M] Credits flowed to unadmitted registrations — **Medium, FIXED**

**Location:** `CreditLedger.recordCredits`.
**Description:** The reworked ledger credited any REGISTERED artifact,
admitted or not. The whitepaper's rule is that only admitted
artifacts earn usage fees. A librarian bug or a premature batch could
pay a failed admission.
**Fix:** `recordCredits` now skips with event `CreditSkipped(...
"not admitted")`. `setAdmitted` remains the librarian's file-execute
step. Tests: `skips credits for an unadmitted registration` (skip,
then admission unblocks the same batch).

### [R2-2-M] Level-2 slash delist was not bound to its victim — **Medium, FIXED**

**Location:** `CreditLedger.slash`.
**Description:** A level-2/3 slash accepted any `artifactId`: a wrong
id silently skipped the delist, and a mismatched victim/artifact pair
could delist an innocent registration.
**Fix:** level ≥ 2 now reverts `NotRegistered` for an unknown
artifact and `WrongTarget` unless the artifact's payout address is the
slashed party. Tests: both reverts + the admitted flag untouched on
revert.

### [C1-M] Single-key librarian is the payout authority — **Medium, carried as a registered trust point**

**Location:** `CreditLedger` librarian role.
**Description:** As in old [3-M]: the librarian can credit the whole
pool; `setLibrarian` is instant owner-only. This is the bootstrap
trust model the whitepaper states ("an operator key during bootstrap,
a governance contract with no human key at maturity").
**Status:** accepted as a PRE-DEPLOYMENT GOVERNANCE TASK — the
librarian becomes the M189 quorum contract before mainnet; the
renounce path is tested. Not a code bug under the registered design.

### [R2-3-I] Price-change timelock is a 7-day floor, not epoch-boundary aligned — **Info, registered**

**Location:** `CreditLedger.schedulePriceChange` /
`PRICE_CHANGE_DELAY = 7 days`.
**Description:** The whitepaper says price changes take effect at
epoch boundaries. The contract enforces a one-epoch NOTICE floor;
the boundary alignment itself lives in the off-chain price table
(sessions lock the price at routing; the router re-sorts at epoch
boundaries). The on-chain field is the floor, not the authority.
**Status:** registered as the intended split.

### [R2-4-I] A payout address that cannot receive ETH strands its credits — **Info**

**Location:** `register`/`claim`.
**Description:** A payout address that rejects ETH (e.g., a contract
with a reverting receive) can never claim. Same class as registering
a burn address; the contributor chooses the payout address.
**Status:** accepted; the self-payment exclusion and SendFailed
paths are tested.

### [R2-5-I] Forced ETH — **Info, now property-tested**

**Location:** solvency accounting.
**Description:** The ledger never reads raw `address(this).balance`
(the `ethHeld` counter excludes forced donations). New test `forced
ETH cannot be claimed` proves a 1-ETH forced gift cannot be claimed.

## 5. Test and evidence state (this review)

- `npx hardhat test`: **46 passing** (41 under coverage + 5
  verifier-only, per `.solcover.js`).
- Coverage gate: **true** (100% on both authored contracts).
- `harness_evidence.json` re-sealed with the new budgets and count.
- M212 cross-language gate re-run: **green** (registration +
  admission + credits + proof-anchor checks, `anchored == batches`).

## 6. Remaining before deployment (all registered)

- Librarian → governance contract (M189) before mainnet ([C1-M]).
- M194/M188/M190 user approvals; deployment ON HOLD.
- Optional Info hardening: [12-I] `h == 0` guard in the verifier.
