# M183 — token flow + vesting contract spec (paper spec v0, no deployment)

Registered 18 Aug 2026 in
`RESEARCH_IMPLEMENTATION_PLAN_v25.md` section 6. This is a PAPER SPEC:
registered invariants and flow definitions only. No token exists, no
numbers are chosen (supply, inflation, cliff lengths wait on the M184
simulations), and the jurisdiction gate (M188) blocks any mint before
external counsel.

## Actors (from the v25 plan section 4)

- **Contributors** — data, arms/encoders, fingerprints, ontology work,
  audits. Earn vested tokens.
- **Inference hosts** — serve the frozen system; a market-rate service
  role, paid for hosting, not for minting.
- **Validators / auditors** — run the replay and attribution checks;
  paid from the treasury.

## Token flow (one paid inference session)

```
payer
  |-- host fee (market rate, negotiated off-ledger)
  +-- treasury (on-ledger)
        |-- 2.5% development fund        (fixed, anti-wash)
        |-- validator/auditor pool       (measured work)
        +-- contributor vesting pool     (measured attribution)
```

The contributor vesting pool is split by the measured value function V
(M181): each component's share = V(component) / sum of V over the
components that fired in the session's replayed decision chain. The
split is computed ONLY from validator-replayed measurements (M177 L0),
never from self-report.

## Vesting

Vested tokens thaw proportionally to the inference payments their
components attract. Thaw events reference ledger-anchored measurement
hashes; a thaw is a function of (attribution record, session replay
record), not of any actor's claim. A registered cliff + linear thaw
schedule is the default form; its numbers are set by the M184
simulations, not here.

## Registered invariants (the spec's core)

- **I1 — measurement-only thaw.** Every thaw event must reference a
  ledger-anchored hash of a validator-replayed measurement (M177 L0 /
  M180 / M181 outputs). A thaw keyed on anything else is invalid by
  construction.
- **I2 — no self-reported contribution.** Uptime, accuracy, data value:
  none of it enters the reward function from the contributor's mouth.
  (H8's availability honesty extends this to hosts.)
- **I3 — append-only attribution.** The registry and the attribution
  records are append-only. Unlearning (M179) is route-exclusion plus an
  auditable record, never deletion of history.
- **I4 — the 2.5% dev-fund route is fixed.** Changing it is a protocol
  upgrade (M196 governance), not a parameter tweak.
- **I5 — wash must lose.** The dev-fund siphon plus the vesting time
  lag are the registered anti-wash mechanisms; H3 (adversarial wash
  traders lose money) is the gate that validates them in simulation
  BEFORE any token exists.
- **I6 — jurisdiction gate.** No mint, no testnet payment, before the
  M188 securities analysis by external counsel. Hard decision point,
  not a footnote.

## Explicitly NOT in this spec

Supply/inflation numbers, cliff/veto specifics, chain choice (M187),
smart-contract code, deployment, and all legal surface (M188, M197,
M198). The spec is a conjecture about mechanism design; the M184
simulations are the instrument that tests it.
