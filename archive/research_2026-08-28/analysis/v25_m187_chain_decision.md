# M187 — chain/platform decision (FINALIZED 24 Aug 2026)

Registered 18 Aug 2026 in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` §6.
Decision form: the DEFAULT STANCE is registered here; alternatives
carry an explicit burden of proof in the cost model.

**FINALIZED (24 Aug 2026, user directive to pin the network before
publishing the whitepaper):** (a) **Ethereum L1** for the anchors —
Sepolia testnet (11155111) first, Ethereum mainnet thereafter;
(b) **Arbitrum One** for settlement / vesting / commit-reveal;
(c) testnet rehearsal on Sepolia (anchors) + Arbitrum Sepolia
(settlement). WETH premise checked: ETH is Arbitrum One's native
asset, so no wrapper is required; the registered WETH fallback rule
(economic design, currency section) remains for ERC20-shaped
integration demand only. Deployment timing remains gated on M188
(counsel); the network choice itself is decided.

**Default stance (23 Aug 2026, refined after the user's
network-decision review; superseded by the finalization above):** (a) **Ethereum Layer 1** for the anchors
(calldata hash anchoring of the ledger tip) — the anchor's value is
trust, the frequency is low, and $1-5 per anchor is trivial at this
profile; (b) **Arbitrum One** for the token + vesting + settlement
batch + commit-reveal contracts — stage-1 rollup, live fraud proofs,
permissionless validation, EVM tooling identical to the
`infrastructure/evm` harness. Base is removed from the default
(single trusted sequencer is the wrong trust posture for an evidence
network); OP Mainnet remains the registered fallback. Optional
complement: OpenTimestamps (Bitcoin) notarization for milestone
seals. The M185 `anchor_spec` fields (tip, record_count,
last_record_hash) are the anchor payload.

**18 Aug 2026 original default (superseded):** EVM Layer 2 (Base /
Arbitrum / OP, in that preference order) for the token + vesting
contracts, with Layer 1 anchors. Kept for the record.

**Why registered now:** the choice interacts with M185's anchor cost
and with H6's audit completeness; recording the default BEFORE any
token exists keeps the cost model honest.

**Burden of proof (registered):** any alternative (pure off-chain
ledger, sidechain, alt-L1) must show, in the cost model:

1. anchor cost per record ≤ 2× the L2-calldata baseline, and
2. replay/audit equivalence with the L1-anchor default (M177 L0), and
3. an exit/rollup-security story at least as strong as the default's.

Nothing here mints anything; M188 remains the hard gate.
