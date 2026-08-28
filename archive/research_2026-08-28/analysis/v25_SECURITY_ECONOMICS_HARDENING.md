# v25 — Security & Economics Hardening: the five considerations

Registered 19 Aug 2026 in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` §4.11 /
§6 (M199–M204), BEFORE any build. This document records the corner-case
exploration and the registered solutions. Nothing below is assumed to
work: every mechanism item is gated (H3/H8/H9 or its own gate) before
anything mints or deploys.

---

## C1. Truly anonymous anti-spam — no whitelists, no identity

Requirement: the app must be truly anonymous, so spam/wash-trade
prevention cannot rely on whitelisting addresses or identity
verification. The registered tool: 2.5% of a user's inference spend
goes to the development fund instead of to the contributor/trainer/
inference runner (§4.5), plus the full anti-wash stack.

### Corner cases explored (each with a registered countermeasure)

1. **Self-payment wash.** A contributor buys inference from their own
   servers to thaw vested tokens. Countermeasures (already registered,
   §4.5): (a) the 2.5% dev-fund cut taxes every round-trip — self-
   payment returns ≤ 97.5¢ per $1; (b) **self-payment exclusion**: thaw
   only from payments by wallets holding no ownership stake in the
   components served; (c) minimum thaw delay after minting kills
   instant round-trips; (d) per-epoch attribution anti-concentration
   caps. Honest arithmetic: the tax alone does NOT break self-payment
   if the thaw rate exceeds 1.025× — the exclusion + delay + caps do
   the work, and H3 (wash must lose vs the no-defenses baseline) is
   the judge, not the parameter values.

2. **Collusion rings.** A↔B mutual payment with no common ownership.
   Countermeasures: the 2.5% cut applies to every ring transaction, so
   a k-party ring loses 2.5% per hop per epoch — the ring's aggregate
   cost grows linearly with ring size while the thaw benefit is
   bounded by each party's attribution share; per-epoch caps bound the
   benefit; ledger anomaly tests (H3) flag ring structure (payment
   graph cycles) as evidence for the audit, not as auto-bans.

3. **Inference farms / self-rental.** Running a farm to serve
   plausible-looking demand. Countermeasures: thaw keys on sessions
   selected by validator-measured quality (H8), so fake demand on a
   bad arm thaws nothing; the dev-fund cut still applies; demand
   traces (M186) price against measured accuracy, so farmed traffic
   must outbid real demand to be selected — a cost the farm pays to
   the network.

4. **Sybil contributors.** Many identities submitting the same or
   near-duplicate contributions to multiply attribution. Counter-
   measures: content-addressed contribution digests (duplicates
   collapse to one ledger entry — the append-only ledger already
   hashes payloads); per-epoch attribution caps; marginal-contribution
   measurement (M180 bake-off) only credits what improves the coalition
   value — a copy adds zero.

5. **Dust storms.** Cheap tiny sessions to look active or to probe
   routing. Countermeasures: per-session minimum settlement and
   off-chain batching (never one transaction per session); probes are
   cheap for attackers, so liveness credit is probe-independent
   (§4.10).

6. **Selection front-running / griefing.** Watching pending sessions
   to steal or disrupt selection. Countermeasures: sealed selection
   (commit-reveal), deterministic failover chains, and slashing
   validators that leak selection (§4.10).

7. **Payment laundering through the dev fund.** The 2.5% pool itself
   is not a wash channel: it is spent by treasury governance
   (§M189/M189 spec) on audits and public goods; a washer cannot
   direct it.

**Registered rule (C1):** no anti-abuse mechanism may condition on
identity, address reputation, or any whitelist. Every countermeasure
above is economic or structural. If a simulation (H3/H8) shows the
stack insufficient, the fallback is raising the cut and lengthening
delays — never introducing identity.

---

## C2. Copy protection viable for 10+ years, quantum-resistant

Requirement: if something is encrypted today, it must not be breakable
within 10 years, including by quantum computers, to stop competitors
stealing model data.

### What can and cannot be protected (honest boundary, continuing §4.8)

- A competitor can always reimplement the _method_ once published —
  cryptography cannot hide an algorithm.
- A contributor _server_ necessarily holds the weights it serves —
  no cryptographic scheme prevents a malicious host from copying what
  it computes on. Deterrence there is economic and legal (attribution
  ledger, contracts), not cryptographic.
- What encryption CAN protect: model artifacts at rest and in transit,
  and the accumulated measured-transfer dataset and registry.

### Registered cryptographic stack (10-year, post-quantum)

| Asset                       | Scheme today                      | 10-year posture                                                                                                                        |
| --------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Weights / data at rest      | AES-256-GCM                       | AES-256-GCM (Grover ⇒ effective 128-bit, adequate)                                                                                     |
| Session payloads in transit | TLS 1.3 + app-layer AES-256-GCM   | same; enforce hybrid KEX                                                                                                               |
| Key agreement               | X25519                            | hybrid X25519 + ML-KEM-1024 (FIPS 203) from day one of the registry                                                                    |
| Signatures                  | Ed25519                           | hybrid Ed25519 + ML-DSA-87 (FIPS 204)                                                                                                  |
| Hash anchors                | SHA-256 (existing payload hashes) | add SHA3-256/SHAKE256 anchors for new artifacts; existing SHA-256 anchors remain valid as-integrity, new long-horizon anchors are SHA3 |
| Randomness                  | CSPRNG                            | unchanged                                                                                                                              |

Rules: only standardized PQ primitives (FIPS 203/204/205); no custom
crypto; hybrid classic+PQ during the transition so a break of either
family still leaves the other; artifact keys live in HSMs/key-vaults,
never in the ledger.

### Honest limit

If the frozen encoder is published in the thesis, its weights are
public by choice and nothing is breakable to protect. The
trade-secret window (§4.8 item 4) is the actual lever: what stays
encrypted for 10 years is the _measured data and registry_, not
published math. The PQ stack above protects that window.

---

## C3. Server liveness, poisoning resistance, encrypted routing

Requirement: contributors run their own inference servers; the router
may send them encrypted data; the system must ensure liveness, penalize
unavailable servers, downgrade poisoners; returned data must be
encrypted and unbreakable/leakable neither by collusion nor quantum
systems; the model must never become copyable by the user — computation
stays on the servers. User asks whether zk/FHE/MPC apply, considering
copy protection.

### Registered protocol (extends §4.7 Track P and §4.10)

**Encrypted flow per session:**

1. User holds the session key; payload is encrypted to the selected
   server's public key (hybrid X25519+ML-KEM per C2). Only the server
   can open it. Raw inputs never touch the registry.
2. The server computes the frozen encoder + head locally (model never
   leaves the server) and encrypts the result **to the user's key**.
   Colluding third parties observing the channel learn nothing;
   quantum-safe per C2.
3. Optional P1 stage 1 (MPC over the linear head path) removes even
   the single-server plaintext view; the encoder itself stays
   client-side at stage 0 (the user can run it locally and submit only
   features). FHE for the full encoder stays behind the registered
   demand/cost trigger (M195) — never a default.
4. zk (M193) proves correctness of the tiny components (router
   decision, head computation) without revealing anything else.

**Liveness:** deterministic health probes (contract + payload hash,
validator-measured only, H8); ordered failover chain scored by
measured accuracy × measured availability × price; thaw keys only on
actually served sessions — downtime prices itself; no self-reporting.

**Poisoning (a server returning bad/random data):**

1. Redundant sampling: a small registered fraction of sessions are
   double-routed; a mismatching second opinion flags the pair.
2. Held-out probe tasks: validators submit tasks with known-reference
   outputs; a poisoner's responses mismatch.
3. Conviction needs cryptographic evidence (output commitment +
   mismatch proof) recorded on the ledger; punishment is exclusion
   from selection + slash of the stake backing the server, scaled to
   the attested damage. Downgrade is automatic and evidence-bound —
   no validator discretion, no identity needed.
4. Gate H9 (new): on the registered scenario set, the mechanism
   catches a registered poisoner within the registered session budget
   without excluding any honest server.

**Copy protection on the server:** the client receives outputs only;
the model cannot be reconstructed from a small number of outputs
(honest limit: enough queries can distill a student — deterrence is
economic/contractual; the frozen-head closed form is the asset that
must not ship, and it never does — only scores ship).

---

## C4. Blockchain code rigor — local EVM, 100% coverage

Requirement: every piece of blockchain-related code must be verified
to work, be secure, efficient, and not exploitable; contracts stay
upgradeable and can later be released from admin control. Start with
a local EVM; keep 100% test coverage.

### Registered engineering rules

- **Local EVM first:** all contracts are developed and tested against
  a local EVM (M204 harness). No testnet/prod deployment before the
  harness passes.
- **100% branch coverage required:** the coverage gate blocks any
  commit of contract code below 100% measured coverage. (The repo's
  Python models — `geode/incentives.py`, `ledger.py`, `pricing.py` —
  already carry their own 100%-style suites; the same bar applies to
  Solidity.)
- **Upgradeability:** standard proxy patterns (UUPS or transparent
  proxy, OpenZeppelin-derived, MIT license) with a two-step admin;
  upgrades are themselves ledger events (hash-anchored) and replay-
  audited (H6).
- **Tokenless-first (decided 19 Aug; reworked 24 Aug):** settlement
  launches in **native ETH** at market rate (Arbitrum One) via the
  `CreditLedger` — deposit/registration/N=4 epoch-vested credits/
  graded burn ladder. Stake is retired (registration fee + slashable
  unvested promise replace it); `VestingVault`/`GeodeToken` are
  retired. The M188 gate stays the only path to ever minting a
  token (C9). Harness re-passed 25 Aug: 46 tests, 100% coverage —
  see `EVM_CONTRACT_AUDIT_2026-08-24_R2.md`.
- **Admin release:** a registered, gated path to renounce admin /
  hand control to governance (M189) or burn the admin key once the
  parameters are frozen; "releasable from admin's control" is a
  first-class state transition with a test.
- **Exploit checklist (each with a test):** reentrancy guards,
  checks-effects-interactions ordering, overflow-safe arithmetic
  (Solidity ≥0.8 or explicit), no unbounded loops in settled
  transactions, batched off-chain settlement, timelocked parameter
  changes, storage-minimal design, pull-over-push payments.
- **Efficiency:** per-session events are batched; one settlement
  transaction per batch, never per session; gas budgets are measured
  in the harness and registered.
- **Prior art:** OpenZeppelin Contracts (MIT), Foundry/Anvil tooling
  (MIT/Apache), Solidity compiler — all permissive licenses. No
  patented constructions reimplemented (C5 rule).

---

## C5. Prior art and patent avoidance

Rule: reuse permissive-licensed implementations (MIT/Apache/BSD);
standardized algorithms only (NIST FIPS 203/204/205 for PQ crypto,
SHA3, AES-GCM); never reimplement patented constructions.

Named prior art to cite (not copy):

- MPC inference: ezPC/Athos, SecretFlow-SPU (Apache-2), MP-SPDZ
  (BSD-3), Iron (ICML 2024), BOLT (HPCA 2024).
- FHE: Zama TFHE-rs / Concrete-ML (BSD-3-Clear), OpenFHE (BSD-2).
- Anti-Sybil / incentive: proof-of-work economics literature; staking
  and slashing designs from public PoS networks (research-grade
  papers, not code).
- PQ crypto: NIST FIPS 203 (ML-KEM), 204 (ML-DSA), 205 (SLH-DSA).
- Smart contracts: OpenZeppelin (MIT), ERC-20/ERC-1155 standards
  (EIPs — public specifications).

The literature-search discipline from M88 applies: no "first" claims
from unauthenticated public search; the audit records queries +
registered anchors.
