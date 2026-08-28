# GEODE Architecture

GEODE (Generalized Encoders for Open-Domain Expertise) is a product:
frozen learning models (fixed programs — see
`docs/DEPLOYMENT.md` §1 for the plain-language explanation) plus
closed-form ridge heads, routed through a deterministic orchestrator,
settled on an append-only ledger that maps to an EVM `CreditLedger`,
with optional cryptographic privacy (secret sharing) and
zero-knowledge proofs of head computation.

This document describes the package architecture for contributors. It
assumes the project's internal terms ("sealed evidence", "milestone")
from the research narrative, which lives in
`analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`.

## Layering

```
experiments/            research runners, configs, sealed evidence
   |  (may import geode; NEVER the reverse)
   v
geode/                  the product package
   ├── audit/           replay, provenance, erasure        (lowest layer)
   ├── hashing.py       canonical JSON + payload hash
   ├── core/            domain model, admission, routing, orchestration
   ├── attribution/     coalition games, incentives, pricing
   ├── settlement/      the CreditLedger attribution wire
   ├── privacy/         secret sharing + zero-knowledge arguments
   └── api/             the application layer (FastAPI service + console)
```

Dependency rules (enforced by review; see `docs/TESTING.md` for how
the suite validates them):

1. `geode` never imports `experiments.*`. The hashing primitives are
   defined in `geode.hashing`; `experiments.common.v5_artifacts` and
   `experiments.common.experiment_manifest` re-export them.
2. Within `geode`, dependencies point downward only:
   - `audit` imports only `hashing` and the standard library;
   - `core` imports `audit`, `hashing`, and `core` itself;
   - `attribution`, `settlement`, `privacy` import `core`, `audit`,
     `hashing` — never each other.
   - `api` is the application layer: it may import any geode layer,
     never `experiments`.
3. The public API is `geode/__init__.py` (`__all__`); each subpackage
   has its own `__init__.py` with a curated `__all__`. Deep imports
   (`geode.core.arm`) are allowed; flat names are only those the
   public API exports.

## Modules

### geode.audit

- `audit.py` — `AuditAPI`: L0 deterministic replay + L1 provenance;
  `TIMING_FIELDS` (the registered wall-clock exclusion set);
  `ReplayReport`, `ProvenanceReport`.
- `erasure.py` — `AffineMap` / `leace_eraser` with an `erasure_certificate`.

### geode.core

- `hashing.py` (top-level module) — `canonical_json` and `payload_hash`.
- `descriptor.py` — `NormalisedDescriptor` and the `AXES` schema.
- `fingerprint.py` — `FingerprintEncoder` (per-domain fingerprints).
- `ontology.py` — the frozen task ontology loader + consistency check.
- `registry.py` — `TaskRegistry`.
- `capability.py` — the R-capability rules (`rule_r_*`).
- `dnn_admission.py` — `DNNSubmission` / `AdmissionRegistry`: admission
  verifies, never trains; duplicate collapse.
- `arm.py` — arm adapters: any admitted DNN or sealed head becomes a
  validated, routable arm spec. Size-agnostic by design.
- `router.py` — deterministic nearest-arm routing with the registered
  failover chain (fingerprint match → general arms → primitives).
- `ledger.py` — `AppendOnlyLedger`: append-only, hash-chained registry.
- `orchestrator.py` — the serve loop: register → route → attribute →
  record, over any heterogeneous arm set.

### geode.attribution

- `attribution.py` — coalition-game attribution (Shapley, beta-Shapley,
  leave-one-out, ranking stability).
- `incentives.py` — the anti-wash incentive round simulations.
- `pricing.py` — posted-price / auction / bandit demand studies.

### geode.settlement

- `settlement.py` — `build_credit_batches`: deterministic,
  chain-anchored `recordCredits` payloads from ledger route records,
  with the registered mask bits, fee split, and staked-payer exclusion
  mirroring the contract exactly. `verify_batch_rules` is the
  conformance gate.

### geode.privacy

- `secret_sharing.py` — additive / replicated Gram shares and Shamir
  sharing over the registered prime.
- `zk_linear.py` — the single-move argument (measured infeasible at
  the real width; sealed as such).
- `zk_bulletproofs.py` — the log-sized argument (1,024-byte proofs at
  the real width; the incumbent protocol).
- `zk_onchain.py` — the proof serialization bridge matching the
  on-chain layout, without modifying the sealed protocol module.

### geode.api (the application layer, M217/M223)

- `service.py` — the FastAPI service: POST /arms (409 on duplicates,
  422 on bad specs), POST /route, GET /ledger (chain + verification),
  POST /settlement/batches, POST /snapshot, POST /demo/seed, GET
  /health, and the console frontend at `/`.
- `persistence.py` — snapshot save/load: arm specs + route requests;
  loading replays through the public orchestrator API and reproduces
  the chain tip bit-exactly.
- `static/index.html` — the no-build-step console.

## The EVM side (`infrastructure/evm`)

Solidity contracts mirror the whitepaper's registered rules
(aligned 24 Aug 2026):

- `CreditLedger` — the native-ETH settlement ledger: the unified
  registration form (operator key + payout address + price per unit
  - sealed claim), librarian-gated attribution batches with
    skip-and-emit (self-payment exclusion keys on the PAYOUT
    address), N=4 epoch vesting with pull-only claims, the graded
    burn slash ladder (L0-L3, replay-gated), and timelocked dev-fund/
    registration-fee changes.
- `LinearProofVerifier` — the direct on-chain port of the M193b
  verifier (bit-exact cross-language verification).
- `ProofAnchor` — the per-query proof-hash anchor (append-only,
  permissionless).
- `scripts/post_batch.js`, `scripts/verify_onchain.js`,
  `scripts/anchor_proof.js` — the cross-language gates.

Retired with the whitepaper alignment: `GeodeToken.sol` (no token),
`VestingVault.sol` (vesting folded into `CreditLedger`), and the
stake machinery.

## Invariants (the sealed discipline)

- Deterministic serialization; `payload_hash` over the canonical JSON
  shape; no wall-clock fields inside any content hash
  (`TIMING_FIELDS`).
- Register before measuring; gates before numbers; void ≠ negative;
  failure records preserved.
- Sealed anchors reproduce bit-exactly before any measurement is read.

## Versioning

`geode.__version__` follows Semantic Versioning. Bump MINOR for
backwards-compatible public API additions, MAJOR for breaking changes,
PATCH for bug fixes.
