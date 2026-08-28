# M189 — treasury governance spec (multisig, voting, veto paths)

Registered 18 Aug 2026 in `RESEARCH_IMPLEMENTATION_PLAN_v25.md` §6.
Spec only; no deployment. The treasury is the M183 flow's on-ledger
pool (dev fund, validator pool, contributor vesting pool).

**Registered structure:**

- **Multisig:** treasury parameter changes require m-of-n signatures
  over the registered signer set (validators + gatekeepers); the
  default is a 3-of-5 with rotating members — the concrete numbers are
  set with the M184 simulation outputs, not here.
- **Voting:** parameter-change proposals are versioned artifacts with
  payload hashes (the M177 discipline); a proposal is valid only if it
  carries (a) the proposed diff, (b) the measured rationale citing
  sealed evidence, (c) a replay reference. Voting is hash-anchored.
- **Veto paths:** (1) the dev fund's 2.5% route and its end-state
  purpose — the zakat rule fixed on 24 Aug 2026 — cannot be changed
  by vote at all; changing either is a protocol-level event (M196);
  (2) the jurisdiction gate (M188) can halt any treasury function
  unilaterally; (3) the audit API can freeze a proposal whose
  referenced evidence fails replay.
- **Anti-concentration:** no signer set may contain a majority from one
  actor class; the M184 free-rider/wash agents model the failure modes
  this rule exists to prevent.

**Not in this spec:** legal personality of the treasury, tax treatment,
and the jurisdiction-specific wrappers (M188/M197).
