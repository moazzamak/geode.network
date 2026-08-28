# M202 — Local-EVM harness spec (registered 19 Aug 2026; AMENDED 25 Aug 2026)

Scope: every piece of blockchain-related code for GEODE is developed and
tested against a local EVM before any testnet or mainnet exposure.
This spec registers the harness topology and the rules before building
(M204).

**Amendment (25 Aug 2026, after the whitepaper-aligned rework of
24 Aug):** the original contract set (`GeodeToken` ERC-20 + `VestingVault`)
is retired; the description below reads the amended set. The rule
numbering is unchanged; the R2 audit
(`EVM_CONTRACT_AUDIT_2026-08-24_R2.md`) is the conformance matrix
against these rules.

## Topology

- **Runner:** Hardhat (MIT) with its in-process local EVM network; no
  external node needed for tests.
- **Contracts under test:** `CreditLedger` (native-ETH settlement
  ledger: deposit/registration/N-epoch vesting/slash-burn) and
  `ProofAnchor` (permissionless hash anchoring). OpenZeppelin
  Contracts (MIT) as the base library.
- **Location:** `infrastructure/evm/`.

## Registered rules

1. **Local EVM first.** No contract code merges without passing the
   full harness suite on the local EVM.
2. **100% measured coverage as a commit gate.** `solidity-coverage`
   reports line AND branch coverage for every contract we author
   (library code under `node_modules` is excluded); a commit that
   drops any of our contracts below 100% is rejected.
3. **Upgradeability.** UUPS proxy pattern; upgrades are two-step and
   themselves ledger events; the harness must contain an upgrade
   rehearsal test (deploy v1 → upgrade v2 → state preserved).
4. **Admin release.** A tested path to renounce admin / transfer
   control to governance (M189) or burn the admin key. The harness
   must prove post-release invariants hold (no further upgrades, no
   parameter changes).
5. **Exploit checklist, each with at least one test:**
   - reentrancy (the claim path must be reentrancy-safe),
   - checks-effects-interactions ordering,
   - overflow-safe arithmetic (Solidity ≥ 0.8),
   - no unbounded loops in settlement transactions,
   - batched off-chain settlement (`recordCredits` processes batches,
     never per-session on-chain),
   - pull-over-push payments,
   - timelocked parameter changes,
   - storage-minimal design.
6. **Efficiency.** Gas budgets for the ledger's hot paths (deposit,
   register, recordCredits batch, claim, slash) are measured in the
   harness and registered in the evidence.
7. **Anti-wash mechanics in the ledger (C1, economic-only):**
   - `DEV_FUND_BPS = 25` (2.5%) of deposits routed to the dev fund;
   - self-payment exclusion keyed on the registration's PAYOUT
     address (a payment from the beneficiary cannot thaw its own
     credits — no address whitelists anywhere);
   - N=4 epoch-vested credits as the standing slashable promise;
   - a flat registration fee to the dev fund;
   - the graded burn ladder (slashed amounts move to a bucket no
     claim path can touch).
     Identity-based mechanisms are forbidden (C1 rule).

## Evidence

The harness writes `infrastructure/evm/evidence/harness_evidence.json`
(hash of sources + coverage summary + gas budgets + test counts).
Sealed 25 Aug 2026: 46 tests, coverage gate true, budgets registered.
