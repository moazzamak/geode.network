# M176f — Contribution interface spec (registered, v0; the v25 bridge)

Frozen 2026-08-17. How a third party submits an arm to the GEODE
registry. Without this, the multi-party network has no door.

## Submission payload (all fields mandatory)

1. `encoding_contract` — the frozen provider spec (name, version, input
   contract, determinism statement) that produced the codes.
2. `arm_payload` — the frozen component (weights / primitive table),
   with `payload_hash` computed over the payload only (no wall clocks).
3. `validation_evidence` — held-out accuracy on a registered task,
   measured against the registered gates: fit-and-report with anchors
   (ridge anchor reproduced, tol 1e-6), G1 determinism on re-run, and a
   declared task descriptor with its fingerprint.
4. `selection_metadata` — per-task held-out accuracy records, health
   probe contract, price.
5. `provenance` — identity of the submitter + the budget epoch.

## Validation rules (the registry runs these, not the submitter)

- Payload hash recomputed from the payload itself; mismatch = reject.
- The arm's own-task held-out accuracy is RE-MEASURED by a validator
  tier against the registered protocol before it may enter any route
  (measured selection is never assumed — the section-5 rule).
- Redundant-capability check: an arm that claims an existing task
  region enters the failover chain by the registered selection score,
  never in front of a better-measured arm.

## Identity tiers (frozen)

- `observer` — read reports and hashes.
- `contributor` — submit arms; submissions enter the validation queue.
- `validator` — re-measure submitted arms; two-of-three agreement
  advances a submission.
- `gatekeeper` — registry write access; append-only, every write
  carries a payload hash and is recorded in the append log.

## The v25 link

Rewards (v25: stakes, burns, the 2.5% dev fund) attach to VALIDATED
arms and re-measurements, never to submissions alone — the incentive
must pay for evidence, not for claims.
