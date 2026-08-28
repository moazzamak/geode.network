# GEODE local-EVM harness (whitepaper-aligned, 24 Aug 2026)

Local-EVM development and test harness for GEODE's blockchain code.
Registered rules: see `analysis/v25_m202_evm_harness_spec.md` and
`analysis/GEODE_ECONOMIC_DESIGN_v1.md`.

## Layout

- `contracts/CreditLedger.sol` — UUPS-upgradeable native-ETH
  settlement ledger: deposit with the 2.5% dev-fund split; the
  unified registration form (operator key + payout address + price
  per unit + sealed claim) with a timelocked registration fee;
  librarian-gated attribution batches with skip-and-emit
  (self-payment exclusion keys on the PAYOUT address); linear N=4
  epoch vesting with pull-only account-bound claims; the graded
  burn slash ladder (L0-L3, replay-gated, evidence hash recorded in
  the Burned event); timelocked dev-fund and per-registration price
  changes; pause; transient reentrancy guard.
- `contracts/ProofAnchor.sol` — the per-query proof-hash anchor
  (append-only, permissionless).
- `contracts/LinearProofVerifier.sol` — the direct on-chain port of
  the M193b verifier (bit-exact cross-language verification).
- `contracts/mocks/ReceiverMocks.sol` — test-only receivers
  (rejecting + reentrant; excluded from the coverage gate).
- `test/` — 41 tests; 100% statements/branches/functions/lines on
  every authored contract (the verifier is coverage-excluded by
  `.solcover.js`, registered).
- `scripts/harness_evidence.js` — gas budgets + coverage summary +
  source hashes → `evidence/harness_evidence.json`.
- `hardhat.config.js` — local harness plus the registered chain
  targets (Sepolia anchors; Arbitrum Sepolia rehearsal; Arbitrum One
  settlement); keys come from the environment, nothing is
  committed.

Retired with the whitepaper alignment: `GeodeToken.sol` (no token),
`VestingVault.sol` (vesting folded into `CreditLedger`), and the
stake machinery.

## Commands

```bash
npm install          # once
npx hardhat compile
npx hardhat test     # the suite
npx hardhat coverage # 100% gate (statements/branches/functions/lines)
npx hardhat run scripts/harness_evidence.js
```

## The coverage gate

`hardhat coverage` must report 100% statements, branches, functions,
and lines for every contract under `contracts/` (mocks excluded).
Commits that drop any of our contracts below 100% are rejected.

## Notes

- Node >= 20 required (Hardhat does not support Node 25 officially;
  CI should pin Node 22).
- Solidity 0.8.28, EVM target `cancun` (the transient reentrancy
  guard uses EIP-1153).
- OpenZeppelin 5.6 APIs differ from older tutorials: UUPS is
  stateless (no `__UUPSUpgradeable_init`), `__Ownable2Step_init()`
  is empty and must be paired with `__Ownable_init(msg.sender)`.
