# M208 — Bittensor subnet feasibility study

Registered 19 Aug 2026 (§4.14). Facts sourced 19 Aug 2026 from the
official Bittensor docs (`bittensor.com/docs`; network, emissions,
burned-register pages). All figures are as-documented; anything marked
[verify] must be re-checked before a spend decision.

## 1. Documented mechanics (the facts the study rests on)

- Emission: 1 TAO/block base, first halving Dec 2025 → **0.5 TAO/block
  ≈ 3,600 TAO/day** network-wide, cap 21M TAO.
- Each subnet has its own **alpha** token (dTAO, since Feb 2025),
  traded in a protocol-owned Balancer-style pool vs TAO.
- **Per-tempo split** of subnet emission: **18% subnet owner**, 41%
  miners, 41% validators + stakers. (This is a chain-level rule, not a
  custom one — transparent by construction, which fits our dev-fund
  disclosure requirement.)
- Subnet emission share = price-EMA demand share, burn-adjusted,
  filtered by an emission gate (theta); **new subnets start near zero
  EMA and climb slowly** (launch-pump blunting).
- Registration: a floating TAO cost, burned/recycled; a collateral
  lock share `p` stakes part of it as miner collateral. Full subnets
  evict the lowest-emission neuron.
- Slots: ≤256 UIDs per subnet, ≤128 validator permits; validator
  permits by stake weight (alpha stake + 0.18×root TAO stake);
  consensus = stake-weighted median of validator weights (kappa ≈ 0.5).
- Chain: Substrate, 12 s blocks; an EVM layer exists (EVM fees recycle
  into issuance).

## 2. The mapping (what a GEODE subnet is)

- **Miners** = GEODE inference hosts (frozen encoders + heads, later
  DNN components per §4.13).
- **Validators** = GEODE's measurement layer: the router, M180
  coalition attribution, anti-wash gates, and H8/H9 machinery, packaged
  as the subnet's scoring logic.
- **The 2.5% dev fund** = a share of the on-chain 18% owner cut routed
  to the treasury — disclosed, never stealthy (§4.14 trade-off 3).
- **Attribution inside the subnet**: validator weights follow measured
  marginal contribution; the self-payment exclusion and poisoning
  conviction (M201) run validator-side with evidence anchored on-chain.

## 3. Startup capital — the honest arithmetic

- What you get: **operating income for participants** — miners and
  validators earn alpha from day one of consensus, redeemable to TAO
  through the subnet pool. No fiat war chest needed to pay
  contributors; no token to issue (M188 shrinks to holding/selling
  TAO).
- What you do NOT get: upfront capital. A young subnet's price-EMA
  starts near zero and rises slowly, so early alpha is nearly
  worthless; the team's owner-cut income materializes only as the
  subnet's price/demand grows. Registration costs a floating TAO burn
  up front [verify current amount].
- The team's realistic early position: run the first miners and
  validators (earn alpha), hold a share of the owner cut as the dev
  fund, and optionally stake. This is a slow, demand-gated revenue
  ramp — not a raise.

## 4. Risks, honestly

1. **Launch-pump blunting**: the price-EMA smoothing means emissions
   track sustained demand; without real users, alpha stays cheap.
2. **Alpha market volatility**: alpha exposure is a trading risk; the
   treasury holding alpha must decide sell/stake policy.
3. **Weight-gaming culture**: stake-weighted median consensus is
   gameable in known ways; our validator-measured attribution is the
   defense, and it must be validated in that adversarial environment
   (H8/H9 become load-bearing).
4. **Owner-cut sensitivity**: 18% owner take is community-sensitive;
   our dev-fund share must be transparent and capped.
5. **Regulatory**: earning and selling TAO/alpha is a different
   exposure than issuing — M188-lite review required, but the
   crypto-asset-issuer liability disappears.

## 4.5 UX status (checked 19 Aug 2026 against the official docs)

The user's remembered pain points — unclear how to buy tokens, no
commonly trusted wallets — are substantially resolved on the platform
level as documented today:

- **MetaMask is a first-class path.** The Bittensor EVM (mainnet chain
  ID 964, listed on ChainList) supports standard `eth_*` JSON-RPC,
  MetaMask, Hardhat, and Remix; `btcli evm config --format metamask`
  prints paste-ready network settings and `btcli evm key export`
  imports the keystore directly.
- **Hardware and trusted wallets:** Ledger (Polkadot generic app),
  Polkadot Vault (QR), Talisman/Polkadot.js browser-extension
  signing (the private key never touches the CLI machine), and M-of-N
  multisig coldkeys are all documented flows.
- **Buying TAO:** TAO itself is a major-CEX-listed asset [verify
  current listings]; hosted bridges (tao.app/bridge) move TAO between
  substrate and EVM wallets; vTAO is liquid-staked TAO as an ERC-20
  on the Bittensor EVM, bridgeable to Base.
- **Tooling overhaul:** a unified v11 SDK + `btcli` with
  intent-based `plan`/`execute`, `--dry-run` on every submit,
  machine-readable error codes with remediation hints, `btcli evm
doctor` diagnostics, address books, scoped proxies, MEV-shielded
  submission, and docs published as a machine-readable catalog.
- **Remaining rough edges (honest):** the dual address domain
  (ss58 `5…` vs h160 `0x…` with mirror addresses and the 9-vs-18
  decimals trap) is still conceptually confusing; hotkey files are
  stored unencrypted by design (documented security tier); lost
  mnemonics are unrecoverable; alpha remains per-subnet (unit-tagged
  `Balance` types prevent silent mixing, but users must understand
  subnet-specific tokens).

Verdict: the platform-level UX is no longer a blocker. The residual
UX burden is on OUR subnet (making alpha, attribution, and payouts
legible to end users) — which the tokenless credit-ledger design
(M207) already addresses on the settlement side.

## 5. Go / no-go

- **Go** if the standalone registry reaches a working measurement
  layer, the M188-lite review clears holding/selling TAO, and the
  team accepts the slow revenue ramp + alpha exposure.
- **No-go** (standalone stays) if registration cost or alpha
  exposure is unacceptable — the tokenless registry path is
  unaffected either way.

**Recommendation: prepare the subnet as the scale-up vehicle, gate the
actual registration on (a) a working validator package, (b) M188-lite,
(c) a live quote of the registration burn.** No spend now.
