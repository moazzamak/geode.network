# TESTNET LAUNCH PLAN (v26, 27 Aug 2026)

Registered before any launch work. Covers the four decisions the
user named: the minimum validator set, the librarian key ceremony,
the privacy invariants that must hold at launch and stay held, and
the key-disposal procedure. Every number here is derived from
registered floors, not asserted. The launch venue set and the
authority-key nexus list remain the user's decisions (checklist
§6); this plan binds them wherever they appear as variables.

## 1. Minimum validator set at launch

**Registered answer: N = 9 validators for the first epoch.**

The mechanical floor is 3 (admission sample k ≥ 3; takedown
responders ≥ max(3, ceil(0.1·pool))). Launching AT the floor is
meaningless for the registered security posture, because every
sample would be the whole pool and one corrupt validator would sit
in every judgment. The launch minimum is derived from the
registered sampling structure:

- **N = 9** makes the admission sample (k = 3) one third of the
  pool. With a registered launch corruption budget of ≤ 1 corrupt
  validator, every 3-sample is honest-majority; with 2 corrupt,
  a 3-sample is honest-majority with probability ~0.92 — the
  registered honest boundary, stated not hidden.
- The diversity floor (d = max(3, ceil(0.2·n))) binds on a
  3-responder vote: all three responders must be distinct
  behavioural identities — satisfiable at N = 9, vacuous at N = 3.
- Audit independence: the audit rule re-labels by two validators
  outside the session's sampled set — possible at N = 9.
- Reference executors (k_e ≥ 2) and the probe shadow overlap the
  same pool without exhausting it.

**Launch constraints on the nine (registered):**

1. Multi-party, never developer-only (M327): the bootstrap arm
   operator, recruited operators, and at least two independent
   validator operators not affiliated with the developer.
2. Distinct behavioural identities (M307): the nine must not
   collapse under dedup — each operates a genuinely distinct
   serving/validating artifact.
3. The registered corruption budget at launch is ≤ 1 validator;
   if two or more of the nine are ever suspected of collusion,
   the launch checklist's validator-set gate fails and the set is
   re-recruited BEFORE the next epoch.
4. The bootstrap council (M327) runs governance votes during the
   bootstrap epoch and sunsets by timelock; the nine validators
   are its membership core.

**Honest boundary:** nine validators is a bootstrap set, not a
decentralized network. The design makes the path out of
concentration automatic (lottery-spread revenue, the cap, the
diversity floor); it does not pretend otherwise. This is stated in
the whitepaper and restated here.

## 2. Librarian key generation

The librarian is ONE operator key during bootstrap (registered);
at maturity it becomes a governance contract with no human key.
The launch procedure:

1. **Ceremony machine.** An air-gapped machine, freshly installed,
   never networked before or after. No cloud, no VM, no remote
   access. OS media checksummed against the publisher's digest.
2. **Entropy.** Physical dice (≥ 4 throws per byte of the 32-byte
   seed, von-Neumann debiased) mixed with the OS CSPRNG in a
   registered combiner (hash of dice-string ‖ CSPRNG bytes).
3. **Key material.** secp256k1 (EVM-compatible — the librarian
   signs Ethereum transactions against the settlement contracts).
   Derived from the seed with a registered derivation (the launch
   tool pins a standard library; the derivation path is a
   registry artifact).
4. **The split.** Shamir 2-of-3 of the seed across three
   registered custodians: the bootstrap arm operator, a second
   operator, and a registered third party not affiliated with
   either. No single custodian can reconstruct the key; any two
   can.
5. **Verification.** The derived address is registered as the
   librarian in the settlement contract at deploy; a test
   transaction on Sepolia is sent and recorded BEFORE any real
   state exists. The ceremony transcript (public parts only:
   addresses, hashes of the share-check values, timestamps) is a
   ledger entry. Share material never enters any digital system
   connected to a network; the paper copies are stored in sealed,
   tamper-evident envelopes at the custodians' premises.
6. **The deputy path is the continuity mechanism.** The registered
   executable-replacement rule (≥ half of the validators endorse
   a divergence reason; a deterministic deputy takes over at the
   next epoch) is the recovery AND disposal mechanism — the key
   is never a single point of failure without a registered exit.

## 3. Privacy invariants at launch and after

Each is a LAUNCH GATE — it must hold before the first epoch and
every upgrade must re-pass it (the M324 inexpressibility audit):

1. **Serving-tier auditability.** Every session records WHICH tier
   served it: on-device encoder, FHE private path (measured
   ~23 s/query, premium-priced), or the plaintext tier (disclosed
   as such). The tier mix is a public statistic; a plaintext
   session that was sold as private is a ledger-visible contract
   violation.
2. **The FHE path is ciphertext-only.** The contributor's
   transcript contains ciphertexts only (QG2/QG4); the device
   decrypts locally. Scheme parameters are registry artifacts
   pinned to the 128-bit security table; any library upgrade
   re-runs the noise gates before adoption (parameter drift is
   registered, never silent).
3. **No plaintext model anywhere.** The registry stores the
   content hash and the M318 Pedersen commitments of the
   quantized head — never the head. The head stays on the
   contributor's host; the proof layer binds to the commitment.
4. **The ledger carries commitments only.** Answer entries are
   H(answer‖nonce); content orders carry hashes and references
   (M323-G3); the sensitive-category path is authority-only and
   reproduces evidence only into the sealed replay environment.
5. **The gateway's data contract.** The plaintext-tier gateway
   retains nothing, trains on nothing, and logs no request
   content — enforced by configuration at launch and audited
   (the no-retention clause is a data contract, not a policy
   choice).
6. **Economic-only incentives hold.** No identity collection in
   any launch component; the behavioural-identity mechanism
   fingerprints artifacts, not people.
7. **Upgrade gate.** Any change that adds a processing surface
   (a new field, a new role, a new endpoint) must pass the M324
   schema/capability audit BEFORE merge: no user/IP/region
   selection surface may ever exist. This is the "no future
   privacy problems" clause — future code is admitted by the
   same inexpressibility rule, not by review goodwill.
8. **Residuals restated at launch** (they do not block launch;
   they are displayed): feature-inversion against public encoders
   (codes are not anonymous); the plaintext tier's contributor
   exposure; ballot secrecy rests on the tally committee's
   majority; a key can be sold whole. All four are in the
   whitepaper's known limits.

## 4. Key disposal

Disposal is a registered event, not an improvisation:

1. **Trigger.** Any of: a registered divergence reason collecting
   ≥ half of validator endorsements; a suspected compromise; a
   planned rotation.
2. **Procedure (registered order):**
   a. The custodians destroy their Shamir shares: paper burned in
   the presence of a second custodian; any digital copies on
   the air-gapped machine zeroed with a registered wipe tool,
   then the machine's storage is physically destroyed or
   reformatted per the ceremony's destruction clause.
   b. The executable-replacement flow fires: the deputy key takes
   over at the next epoch; the settlement contract's librarian
   address rotates through the registered mechanism.
   c. The old public key is published as REVOKED in the ledger —
   the same watched-revocation pattern as the authority-key
   registry. Clients reject entries signed by the revoked key
   after the rotation epoch.
   d. A disposal record (public parts only) is a ledger entry:
   date, old key hash, new key hash, endorsement count.
3. **The anchor continuity.** The Ethereum anchor cadence makes
   the disposal window visible: the old key's last valid entry
   and the deputy's first are adjacent in the chain, and prefix
   immutability makes any post-disposal forgery by the old key
   invalid by construction.
4. **What disposal cannot do.** Destroying the key does not
   destroy the ledger (anchors + prefix immutability), the
   registry, or the escrow states (contract-level, keyed by the
   librarian address — the deputy inherits them). Disposal is a
   rotation, never a reset.

## 5. Binding to the checklist

The launch checklist gains these gates:

- [ ] N = 9 validators recruited, multi-party, distinct behavioural identities (gate 1).
- [ ] Key ceremony executed per §2; librarian address registered on the target testnet; test transaction recorded.
- [ ] Privacy launch gates 1–8 (§3) audited and recorded.
- [ ] Disposal procedure (§4) rehearsed once on the testnet before the first real epoch.
