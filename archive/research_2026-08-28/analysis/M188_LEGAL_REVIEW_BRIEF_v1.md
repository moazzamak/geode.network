# M188 — Legal Review Brief (v1, 23 Aug 2026)

Prepared for external counsel. This document frames the questions;
it makes no legal claims and no conclusions are pre-committed.

## 1. The instrument being reviewed

A network (GEODE) where model arms (frozen publisher checkpoints +
closed-form ridge heads) are admitted by measured, quorum-attested
capability claims. **Tokenless-first (user decision, 19 Aug;
re-confirmed 24 Aug):** no token is planned for launch.

- settlement: native ETH at market rate on Arbitrum One (EVM L2);
  the ledger hash is anchored to Ethereum L1 periodically (M187
  decision, registered). No stablecoin, no wrapper on the chosen
  configuration.
- contributions: a flat registration fee (paid to the dev fund)
  replaces staking — there is NO stake and no principal lockup;
  admission cost is a challenge budget that pays sampled validators.
- earnings: usage credits vest linearly over N=4 epochs and are
  account-bound (no transfer/assign API) until vested AND claimed;
  post-claim ETH is ordinary ETH.
- penalties: graded burn ladder, replay-gated convictions; slashed
  amounts move to a bucket no claim path can touch (nobody gains).
- governance: m-of-n multisig treasury; measured-rationale voting;
  three veto paths (M189).
- the TOKEN remains a REGISTERED LATER OPTION (C9), introduced only
  if governance requires it and only after this brief's
  classification answer. Counsel is asked to answer the
  classification question so the option can be priced; nothing
  schedules it.

## 2. Questions for counsel

1. **Classification.** Under the relevant securities regimes (US
   federal + the EU MiCA framework), is the optional later token
   (C9) — should it ever mint, given the revenue-linked vesting
   design and the governance structure —
   a security / e-money token / utility token / other? Which
   features would change the classification, and what design
   changes (if any) are required before minting?
2. **Jurisdiction and coercion.** The registered threat model
   includes legal and military coercion of operators (the
   "$5 wrench" class). What is the jurisdiction posture: where
   should the registry operator be incorporated, and what
   structural separation (operator vs validators vs token issuer)
   materially changes the coercion surface? This is the M188
   coercion review registered with the release-blocker plan.
   Extension (27 Aug 2026, M324): counsel is also asked whether the
   inexpressibility posture — the protocol contains no mechanism
   to select, block, throttle, or price users by jurisdiction,
   address, or device, and the developer holds no capability to
   implement one — is a defensible answer to a state demanding
   user/IP/region-level controls, and how the multi-jurisdictional
   nexus quorum and jurisdiction mobility interact with the
   coercion analysis.
3. **Regulatory surface (M197).** AML obligations (GwG) for a
   payment-adjacent token; MiCA service authorization for
   EU-facing services; GDPR/data-protection posture for user
   inference data (features-not-raw-data design; per-arm
   tiered-privacy policy pending).
4. **Liability (M198).** Who is liable when a routed arm errs —
   the contributor, the validator, the gateway operator? The
   per-arm indemnity design and an errors-and-omissions posture
   need counsel review before the MVP (M190).
5. **Takedown and intermediary liability (M294).** The network adds
   a quorum takedown: a supermajority of sampled validators can
   permanently delist an artifact on recorded evidence, with the
   librarian filing the deterministic vote count. Questions for
   counsel: does a working quorum-takedown posture support an
   intermediary/notice-and-action framing (DSA-style) for the
   network and its hosts, and what are the residual liabilities
   when an artifact serves illegal content before a takedown
   completes? The verdict record is public; the definition of
   "socially destructive" is deliberately unformalized.

## 3. Facts counsel needs (all registered in the project plan)

- Deployment topology: hosted public gateway (default), tiered
  self-hosting; arms/validators operated by suppliers; users run
  nothing (registered 23 Aug).
- Chain plan: Ethereum L1 anchors + Arbitrum One settlement
  (M187 refined default, 23 Aug).
- Settlement currency: ETH at market rate (user decision, 24 Aug) —
  native ETH is the working default (WETH as fallback); the previous
  stablecoin assumption is superseded.
- Non-transferable claims (user decision, 24 Aug): earned-but-
  unvested balances are account-bound with no transfer/assign API —
  no secondary market for unrealized dues; post-claim ETH is
  ordinary ETH.
- Conformance posture (user, 24 Aug): controls cover every
  coverable path; the bearer-key (wallet sale) residual is an
  accepted limit of the permissionless Byzantine design, documented
  deliberately — analogous to cash/bearer-instrument limits that
  regulators treat as account-level control endpoints. Counsel is
  asked to confirm this posture reads as good-faith completeness.
- Identity layer (user, 24 Aug): per-address KYC is deliberately NOT
  performed because the exchange on/off-ramps (Coinbase, Binance,
  etc.) already link wallets to identities; an authority with an
  address can trace it through those chokepoints without project
  cooperation. Side benefit: the project holds no personal data.
  Counsel is asked whether this argument is persuasive to BaFin/
  regulators and whether any EU-facing service requires identity
  collection regardless.
- No token minted; no mainnet deployment; all measurements sealed
  with evidence under `logs/results/v25/`.
- The measurement discipline (register-before-measuring, hash-
  chained ledger, commit-reveal submissions) is the audit
  substrate — product and research are the same process.

## 4. Scope of this brief

This is the registered M188 input. Counsel's answers gate M190
(MVP deployment) and the funded-key decision (M194 remains a
separate user decision). Until counsel responds, nothing mints
and nothing anchors to a public chain.

## 5. v2 addendum (26 Aug 2026)

Three additional questions, added after an external review of the
design's legal narrative. The review produced no conclusions to
adopt; it surfaced three gaps that belong in counsel's scope.
This addendum frames the questions only and pre-commits nothing.

6. **Payment-services classification (ZAG / PSD2) of the
   settlement path.** The registered settlement design is
   librarian-aggregated batches on Arbitrum One in native ETH,
   with the ledger hash anchored to Ethereum L1. A peer-to-peer
   reading of the escrow does not resolve whether the librarian's
   aggregation step, or the escrow itself, constitutes executing
   payment transactions within the meaning of ZAG / PSD2. Counsel
   is asked: does the settlement path fall inside payment
   services, and which design change (if any) moves it clearly
   outside?

7. **DSA Art. 7 fit for the probe-and-takedown posture.** The
   shadow probe, the quorum takedown, and the planned
   notice-and-action flows are voluntary own-initiative
   investigations. Counsel is asked whether Art. 7 DSA preserves
   an intermediary posture for the network and its hosts, whether
   the marketplace qualifies as a relevant intermediary under the
   DSA at all, and what the residual liability is for the window
   between an artifact going live and a takedown completing.

8. **Developer liability for content-adjacent judgments.** A
   validator quorum votes takedown and slashing on report-driven
   evidence; the definition of "socially destructive" is
   deliberately unformalized, and slashed amounts are burned (no
   party gains). Counsel is asked what liability attaches to the
   developers who designed the voting rules when a quorum makes a
   content-adjacent judgment, and whether the burn-not-award
   slashing design materially changes that exposure.

## 6. v3 addendum (27 Aug 2026)

One additional question, added after the private-serving design
(M322) was registered for the MVP. The design's invariant is that
the developer/network operator processes request content in no
tier. This addendum frames the question only and pre-commits
nothing.

9. **Encrypted-processing posture for the premium private tier
   (M322d).** In the premium tier the user's device FHE-encrypts
   the input; the _contributor's_ hardware (never the developer's)
   evaluates the frozen trunk over the ciphertext; the device
   decrypts its own features and completes the head protocol
   (owner-anchored masking, M322b), under which the contributor
   receives only a uniform mask. Counsel is asked: (a) does
   ciphertext-only evaluation of the trunk by the contributor
   attract reduced data-protection obligations relative to
   plaintext serving (GDPR processor/controller posture, DPA
   scope), and is the encrypted-processing argument persuasive to
   regulators; (b) does the developer's metadata-only role (no
   ciphertext custody, no evaluation, no plaintext) hold across
   all three tiers as designed, and what residual exposure (if
   any) attaches to hosting the registry, the ledger commitments,
   and the FHE parameter plumbing; (c) does the tiered design —
   default on-device, edge quantized, premium FHE — change any
   answer to questions 3 and 4 above.

## 7. v4 addendum (27 Aug 2026)

One further question, added after the content-report design (M323)
was registered. This addendum frames the question only and
pre-commits nothing.

10. **Ministerial freeze, non-compliance, and sensitive-category
    evidence (M323).** The registered design: (a) a legal notice in
    registered format triggers a ministerial, code-enforced freeze
    of the artifact's credits and serving entries — no vote, no
    discretion; (b) validators confirm only technical
    correspondence (ledger commitments, replay equality) and never
    judge legality; (c) funds cannot move during a freeze even if
    validators contest a valid order; (d) for sensitive categories
    the ledger and public record carry only commitments and notice
    references — the underlying content is reproduced only inside
    the sealed replay environment and shown only to the authority;
    (e) under M323a, the ministerial freeze is jurisdiction-scoped
    (nexus gate): out-of-nexus orders are downgraded to reports and
    the validator quorum determines nexus as a procedural finding,
    never a legality judgment. Counsel is asked: who is held
    responsible if validators contest a valid order or attempt to
    release funds, and does the code-ministerial freeze (zero
    operator discretion) satisfy safe-harbor / notice-and-action
    expectations, or must a human compliance operator exist; does
    the commitment-only evidence posture for sensitive categories
    (hash correspondence + authority notice, no content ever
    public) meet the relevant reporting obligations, or are there
    jurisdictions requiring the developer to handle such material
    directly; and does the jurisdiction-scoped quorum nexus gate
    materially change the coercion surface (question 2) — in
    particular, whether declining out-of-nexus state orders is
    defensible, and whether in-jurisdiction orders must always be
    complied with to preserve the intermediary posture.
